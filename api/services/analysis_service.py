import math
import re
from functools import lru_cache

import numpy as np
from django.db.models import Q, Sum

from api.models import (
    CarbonTaxRate,
    DEAReferenceData,
    Edge,
    EdgePipeSizing,
    LoadProfile,
    Network,
    NetworkDesignConstants,
    Node,
    PipeSizing,
)


class NetworkAnalysisService:
    EUR_TO_CAD = 1.45
    MATERIAL_PREMIUM = 1.15
    CIVIL_LABOR_PREMIUM = 2.0
    ONTARIO_GRID_KGCO2_PER_KWH = 0.040
    STEEL_DENSITY_KG_M3 = 7850.0
    STEEL_WALL_RATIO = 0.045
    STEEL_MIN_WALL_M = 0.0026
    STEEL_MAX_WALL_M = 0.020
    STEEL_A1_A3_KGCO2E_PER_KG = 2.49119095715841
    PIPE_COMPONENT_ALLOWANCE = 0.15
    START_YEAR = 2026
    HORIZON_YEARS = 20
    DEA_YEAR = 2030
    _carbon_rates_cache = None
    DATA_CENTER_SOURCE_TOKENS = {
        "data_center",
        "data_centre",
        "data center",
        "data centre",
        "datacenter",
        "datacentre",
    }
    BOILER_FUEL_KGCO2_PER_KWH = {
        "natural_gas": 0.185,
        "gas": 0.185,
        "oil": 0.268,
        "fuel_oil": 0.268,
        "biomass": 0.0,
        "electric": 0.0,
    }

    PIPE_TECH_ALIASES = {
        "twin pipe, buried": "Energy Transport District Heating Transmission",
        "buried twin pipe": "Energy Transport District Heating Transmission",
        "transmission": "Energy Transport District Heating Transmission",
        "distribution city": "Energy Transport District Heating Distribution, City",
        "distribution suburban": "Energy Transport District Heating Distribution, Suburban",
        "distribution new area": "Energy Transport District Heating Distribution, New Area",
    }

    @classmethod
    def generate_report(cls, network_id):
        return cls(network_id).report()

    def __init__(self, network_id):
        self.network = Network.objects.select_related("project").get(pk=network_id)
        self._assert_analysis_scope()
        self.constants, _ = NetworkDesignConstants.objects.get_or_create(network=self.network)
        self._building_annual_heat_cache = None

    def report(self):
        annual_heat_kwh = self._annual_heat_kwh()
        node_capex, _node_catalog_opex, embedded_capex, _embedded_catalog_opex = self._node_capex_and_opex(annual_heat_kwh)
        pipe_capex, _pipe_catalog_opex = self._pipe_capex_and_opex(annual_heat_kwh)

        total_capex = pipe_capex + node_capex + embedded_capex
        fixed_om_rate = self._fixed_om_rate()
        discount_rate = self._discount_rate()
        source_cop = self._source_cop()
        annual_fixed_om = total_capex * fixed_om_rate
        annual_source_input_kwh = self._source_input_energy_kwh(annual_heat_kwh, source_cop)
        annual_variable_energy_opex = annual_source_input_kwh * self._energy_cost_cad_per_kwh()
        annual_opex = annual_fixed_om + annual_variable_energy_opex
        boiler_annual_carbon_t = self._explicit_boiler_annual_carbon_t()
        data_center_annual_carbon_t = self._data_center_grid_annual_carbon_t(annual_heat_kwh, source_cop)
        annual_operational_carbon_t = boiler_annual_carbon_t + data_center_annual_carbon_t
        embodied_carbon_t = self._embodied_carbon_tco2e()
        embodied_carbon_cost = embodied_carbon_t * self._carbon_price_for_year(self.START_YEAR)

        capex_components = self._component_list({
            "Distribution piping": pipe_capex,
            "Explicit source/equipment nodes": node_capex,
        })
        opex_components = self._component_list({
            "Fixed O&M": annual_fixed_om,
            "Variable source energy": annual_variable_energy_opex,
        })
        emission_components = self._component_list({
            "Data center electricity (Ontario grid)": data_center_annual_carbon_t,
            "Explicit boiler fuel": boiler_annual_carbon_t,
            "Embodied infrastructure carbon": embodied_carbon_t,
        })

        projection = []
        component_projection = []
        cumulative_pv_opex = 0.0
        cumulative_pv_operational_carbon = 0.0
        cumulative_pv_total = total_capex + embodied_carbon_cost
        cumulative_opex = 0.0
        cumulative_carbon = embodied_carbon_cost
        cumulative_opex_by_category = {item["category"]: 0.0 for item in opex_components}
        cumulative_carbon_by_category = {
            "Embodied infrastructure carbon": embodied_carbon_cost,
        } if embodied_carbon_cost else {}

        for index, year in enumerate(range(self.START_YEAR, self.START_YEAR + self.HORIZON_YEARS), start=1):
            carbon_price = self._carbon_price_for_year(year)
            operational_carbon_cost_components = [
                {
                    "category": item["category"],
                    "value": item["value"] * carbon_price,
                }
                for item in emission_components
                if item["category"] != "Embodied infrastructure carbon"
            ]
            nominal_operational_carbon_cost = sum(item["value"] for item in operational_carbon_cost_components)
            discount_factor = self._discount_factor(index, discount_rate)
            discounted_opex = annual_opex * discount_factor
            discounted_operational_carbon = nominal_operational_carbon_cost * discount_factor
            cumulative_opex += annual_opex
            cumulative_pv_opex += discounted_opex
            cumulative_pv_operational_carbon += discounted_operational_carbon
            cumulative_carbon += discounted_operational_carbon
            cumulative_pv_total = total_capex + embodied_carbon_cost + cumulative_pv_opex + cumulative_pv_operational_carbon
            for item in opex_components:
                cumulative_opex_by_category[item["category"]] += item["value"]
            for item in operational_carbon_cost_components:
                cumulative_carbon_by_category[item["category"]] = (
                    cumulative_carbon_by_category.get(item["category"], 0.0) +
                    item["value"] * discount_factor
                )
            projection.append({
                "year": year,
                "cumulative_capex": round(total_capex, 2),
                "cumulative_opex": round(cumulative_pv_opex, 2),
                "cumulative_carbon": round(cumulative_carbon, 2),
                "cumulative_total": round(cumulative_pv_total, 2),
                "annual_opex": round(annual_opex, 2),
                "discounted_opex": round(discounted_opex, 2),
                "annual_operational_carbon_cost": round(nominal_operational_carbon_cost, 2),
                "discounted_operational_carbon_cost": round(discounted_operational_carbon, 2),
            })
            component_projection.append({
                "year": year,
                "carbon_price_cad_per_tonne": round(carbon_price, 2),
                "discount_factor": round(discount_factor, 6),
                "capex": self._rounded_components(capex_components),
                "annual_opex": self._rounded_components(opex_components),
                "cumulative_opex": self._rounded_components([
                    {"category": category, "value": value}
                    for category, value in cumulative_opex_by_category.items()
                ]),
                "annual_emissions_tco2e": self._rounded_components(emission_components),
                "annual_carbon_cost": self._rounded_components(
                    operational_carbon_cost_components +
                    ([{
                        "category": "Embodied infrastructure carbon",
                        "value": embodied_carbon_cost,
                    }] if index == 1 and embodied_carbon_cost else [])
                ),
                "discounted_opex": self._rounded_components([
                    {"category": item["category"], "value": item["value"] * discount_factor}
                    for item in opex_components
                ]),
                "discounted_carbon_cost": self._rounded_components(
                    [
                        {"category": item["category"], "value": item["value"] * discount_factor}
                        for item in operational_carbon_cost_components
                    ] +
                    ([{
                        "category": "Embodied infrastructure carbon",
                        "value": embodied_carbon_cost,
                    }] if index == 1 and embodied_carbon_cost else [])
                ),
                "cumulative_carbon_cost": self._rounded_components([
                    {"category": category, "value": value}
                    for category, value in cumulative_carbon_by_category.items()
                ]),
            })

        total_carbon_debt = projection[-1]["cumulative_carbon"] if projection else embodied_carbon_cost
        total_lifecycle = projection[-1]["cumulative_total"] if projection else total_capex + embodied_carbon_cost
        discounted_heat_kwh = self._discounted_heat_kwh(annual_heat_kwh, discount_rate)

        capex_breakdown = self._rounded_components(capex_components)

        return {
            "kpis": {
                "total_capex_cad": round(total_capex, 2),
                "total_lifecycle_cost_cad": round(total_lifecycle, 2),
                "lcoh_cad_per_kwh": round(total_lifecycle / discounted_heat_kwh, 4) if discounted_heat_kwh else 0,
                "total_carbon_debt_cad": round(total_carbon_debt, 2),
                "embodied_carbon_tco2e": round(embodied_carbon_t, 2),
                "embodied_carbon_cost_cad": round(embodied_carbon_cost, 2),
                "annual_operational_emissions_tco2e": round(annual_operational_carbon_t, 2),
                "discounted_heat_kwh": round(discounted_heat_kwh, 2),
            },
            "capex_breakdown": capex_breakdown,
            "component_breakdown": {
                "capex": capex_breakdown,
                "annual_opex": self._rounded_components(opex_components),
                "annual_emissions_tco2e": self._rounded_components(emission_components),
                "annual_carbon_cost": (
                    component_projection[0]["annual_carbon_cost"]
                    if component_projection
                    else []
                ),
            },
            "analysis_assumptions": {
                "method": "Total Unified Lifecycle Cost (TULC)",
                "start_year": self.START_YEAR,
                "horizon_years": self.HORIZON_YEARS,
                "discount_rate": discount_rate,
                "fixed_om_rate": fixed_om_rate,
                "source_cop": source_cop,
                "energy_cost_cad_per_kwh": self._energy_cost_cad_per_kwh(),
                "ontario_grid_kgco2_per_kwh": self.ONTARIO_GRID_KGCO2_PER_KWH,
                "carbon_price_year_0_cad_per_tonne": self._carbon_price_for_year(self.START_YEAR),
                "capex_localization": {
                    "eur_to_cad": self.EUR_TO_CAD,
                    "material_premium": self.MATERIAL_PREMIUM,
                    "civil_labor_premium": self.CIVIL_LABOR_PREMIUM,
                },
                "boundary": "Cradle-to-operation over 20 years; end-of-life financial and carbon costs are excluded.",
            },
            "lifecycle_projection_20yr": projection,
            "component_lifecycle_20yr": component_projection,
        }

    def _component_list(self, values):
        return [
            {"category": category, "value": float(value or 0.0)}
            for category, value in values.items()
            if abs(float(value or 0.0)) >= 0.005
        ]

    def _rounded_components(self, components):
        return [
            {
                "category": item["category"],
                "value": round(float(item["value"] or 0.0), 2),
            }
            for item in components
        ]

    def _node_capex_and_opex(self, annual_heat_kwh):
        central_capex = 0.0
        central_annual_opex = 0.0
        embedded_capex = 0.0
        embedded_annual_opex = 0.0
        nodes = (
            Node.objects
            .filter(network=self.network)
            .select_related("building", "boiler", "geothermal", "thermal_storage")
        )

        for node in nodes:
            technology = (node.dea_technology_name or self._explicit_node_technology(node) or "").strip()
            if not technology:
                continue

            capacity_kw = self._node_capacity_kw(node)
            if capacity_kw <= 0:
                continue

            rows = self._dea_rows(technology)
            if not rows.exists():
                continue

            capacity_mw = capacity_kw / 1000
            base_capacity_mw = self._dea_capacity_mw(rows) or max(capacity_mw, 1.0)
            alpha = self._dea_scaling_alpha(rows)
            investment_row = self._first_row(
                rows,
                Q(parameter__icontains="Nominal investment"),
                Q(parameter__icontains="total") | Q(parameter__icontains="heat pump, installation incl."),
            )
            if not investment_row:
                investment_row = self._first_row(rows, Q(parameter__icontains="investment"))
            if not investment_row:
                continue

            base_cost_cad = self._dea_investment_to_cad(investment_row, base_capacity_mw)
            capex = (base_cost_cad * self.MATERIAL_PREMIUM) * ((capacity_mw / base_capacity_mw) ** alpha)
            central_capex += capex
            central_annual_opex += self._node_fixed_om_cad(rows, capacity_mw, capex)
            central_annual_opex += self._node_variable_om_cad(rows, annual_heat_kwh)

        return central_capex, central_annual_opex, embedded_capex, embedded_annual_opex

    def _explicit_boiler_annual_carbon_t(self):
        return self._linked_boiler_annual_carbon_t()

    def _linked_boiler_annual_carbon_t(self):
        emissions_t = 0.0
        nodes = Node.objects.filter(network=self.network, boiler__isnull=False).select_related("boiler")
        for node in nodes:
            efficiency = self._normalise_efficiency(getattr(node.boiler, "efficiency", None), default=0.9)
            fuel_type = (
                getattr(node.boiler, "fuel_type", None) or
                getattr(node.boiler, "boiler_type", None) or
                ""
            ).strip().lower().replace(" ", "_")
            factor = self.BOILER_FUEL_KGCO2_PER_KWH.get(fuel_type, 0.0)
            fuel_kwh = self._node_annual_heat_kwh(node) / efficiency
            emissions_t += (fuel_kwh * factor) / 1000
        return emissions_t

    def _data_center_grid_annual_carbon_t(self, annual_heat_kwh, source_cop):
        source_nodes = self._data_center_source_nodes()
        if not source_nodes:
            return 0.0

        source_input_kwh = self._source_input_energy_kwh(annual_heat_kwh, source_cop)
        return (source_input_kwh * self.ONTARIO_GRID_KGCO2_PER_KWH) / 1000

    def _source_input_energy_kwh(self, annual_heat_kwh, source_cop):
        if not self._data_center_source_nodes():
            return 0.0
        return annual_heat_kwh / max(source_cop, 0.01)

    def _embodied_carbon_tco2e(self):
        if not Edge.objects.filter(network=self.network).exists():
            return 0.0
        total_kgco2e = 0.0
        for segment in self._pipe_segments_for_analysis():
            length_m = max(self._safe_float(segment.get("length_m")), 0.0)
            diameter_m = max(self._safe_float(segment.get("diameter_m")) or 0.08, 0.02)
            mass_kg = self._steel_pipe_mass_per_meter(diameter_m) * length_m
            total_kgco2e += mass_kg * self.STEEL_A1_A3_KGCO2E_PER_KG * (1 + self.PIPE_COMPONENT_ALLOWANCE)
        return total_kgco2e / 1000.0

    def _pipe_segments_for_analysis(self):
        latest_sizing = (
            PipeSizing.objects
            .filter(network=self.network)
            .order_by("-created_at", "-id")
            .first()
        )
        if latest_sizing:
            rows = list(
                EdgePipeSizing.objects
                .filter(pipe_sizing=latest_sizing)
                .select_related("edge", "route")
            )
            if rows:
                return [
                    {
                        "length_m": self._sizing_length(row),
                        "diameter_m": row.diameter_m,
                    }
                    for row in rows
                ]
        return [
            {
                "length_m": edge.length,
                "diameter_m": 0.08,
            }
            for edge in Edge.objects.filter(network=self.network).only("id", "length")
        ]

    def _steel_pipe_mass_per_meter(self, internal_diameter_m):
        wall_m = min(
            max(internal_diameter_m * self.STEEL_WALL_RATIO, self.STEEL_MIN_WALL_M),
            self.STEEL_MAX_WALL_M,
        )
        outer_diameter_m = internal_diameter_m + (2 * wall_m)
        steel_area_m2 = math.pi * ((outer_diameter_m ** 2) - (internal_diameter_m ** 2)) / 4
        return steel_area_m2 * self.STEEL_DENSITY_KG_M3

    def _fixed_om_rate(self):
        value = self._safe_float(getattr(self.constants, "fixed_om_rate", 0.02))
        return value if value >= 0 else 0.02

    def _discount_rate(self):
        value = self._safe_float(getattr(self.constants, "discount_rate", 0.04))
        return value if value >= 0 else 0.04

    def _source_cop(self):
        value = self._safe_float(getattr(self.constants, "source_cop", 1.0))
        return value if value > 0 else 1.0

    def _energy_cost_cad_per_kwh(self):
        value = self._safe_float(getattr(self.constants, "electricity_cost", 0.15))
        return value if value >= 0 else 0.15

    def _discount_factor(self, year_index, discount_rate):
        return 1 / ((1 + max(discount_rate, 0.0)) ** year_index)

    def _discounted_heat_kwh(self, annual_heat_kwh, discount_rate):
        return sum(
            annual_heat_kwh * self._discount_factor(index, discount_rate)
            for index in range(1, self.HORIZON_YEARS + 1)
        )

    def _data_center_source_nodes(self):
        nodes = (
            Node.objects
            .filter(network=self.network)
            .select_related("building")
        )
        return [
            node for node in nodes
            if self._is_data_center_source_node(node)
        ]

    def _is_data_center_source_node(self, node):
        building = node.building
        source_type = str(getattr(building, "waste_heat_source_type", "") or "").lower().replace("-", "_")
        role_text = " ".join([
            str(node.node_type or ""),
            str(node.name or ""),
            str(source_type or ""),
            str(getattr(building, "building_type", "") or ""),
        ]).lower()
        is_data_center = any(token in role_text for token in self.DATA_CENTER_SOURCE_TOKENS)
        is_source = (
            node.node_type == "producer" or
            bool(source_type and any(token in source_type for token in self.DATA_CENTER_SOURCE_TOKENS))
        )
        return is_source and is_data_center

    def _node_annual_heat_kwh(self, node):
        if node.building_id:
            if self._building_annual_heat_cache is None:
                building_ids = list(
                    Node.objects
                    .filter(network=self.network, building_id__isnull=False)
                    .values_list("building_id", flat=True)
                    .distinct()
                )
                self._building_annual_heat_cache = {
                    row["building_id"]: self._safe_float(row["annual_heat_kwh"])
                    for row in LoadProfile.objects
                    .filter(building_id__in=building_ids, profile_type="heating")
                    .values("building_id")
                    .annotate(annual_heat_kwh=Sum("value"))
                }
            profile_energy = self._safe_float(self._building_annual_heat_cache.get(node.building_id))
            if profile_energy > 0:
                return profile_energy
        return max(self._node_capacity_kw(node), 0.0) * float(self.constants.hours_per_year or 4000)

    def _pipe_capex_and_opex(self, annual_heat_kwh):
        latest_sizing = (
            PipeSizing.objects
            .filter(network=self.network)
            .order_by("-created_at", "-id")
            .first()
        )
        sizing_rows = list(
            EdgePipeSizing.objects
            .filter(pipe_sizing=latest_sizing)
            .select_related("edge", "route")
        ) if latest_sizing else []

        if sizing_rows:
            segments = [
                {
                    "length_m": self._sizing_length(row),
                    "diameter_m": float(row.diameter_m or 0.1),
                    "peak_load_kw": float(row.peak_load_kw or 0.0),
                    "delta_t": float(row.delta_t or 20.0),
                    "dea_pipe_category": row.dea_pipe_category or "Twin pipe, buried",
                }
                for row in sizing_rows
            ]
        else:
            segments = [
                {
                    "length_m": float(edge.length or 0.0),
                    "diameter_m": 0.1,
                    "peak_load_kw": 0.0,
                    "delta_t": 20.0,
                    "dea_pipe_category": "Twin pipe, buried",
                }
                for edge in Edge.objects.filter(network=self.network)
            ]

        total_capex = 0.0
        total_length_km = 0.0
        peak_mw_sum = 0.0
        for segment in segments:
            length_m = max(segment["length_m"], 0.0)
            diameter_m = max(segment["diameter_m"], 0.02)
            curve = self._pipe_cost_curve(segment["dea_pipe_category"], segment["delta_t"])
            unit_cost = self._pipe_unit_cost_cad(curve, diameter_m)
            total_capex += length_m * unit_cost
            total_length_km += length_m / 1000
            peak_mw_sum += max(segment["peak_load_kw"], 0.0) / 1000

        pipe_annual_opex = self._pipe_annual_opex(annual_heat_kwh, total_length_km, peak_mw_sum)
        return total_capex, pipe_annual_opex

    def _annual_heat_kwh(self):
        network_profiles = LoadProfile.objects.filter(network=self.network, profile_type="heating")
        energy = self._profile_energy_kwh(network_profiles)
        if energy > 0:
            return energy

        building_ids = list(
            Node.objects
            .filter(network=self.network, building_id__isnull=False)
            .values_list("building_id", flat=True)
            .distinct()
        )
        if building_ids:
            energy = self._profile_energy_kwh(
                LoadProfile.objects.filter(building_id__in=building_ids, profile_type="heating")
            )
            if energy > 0:
                return energy

        fallback_kw = 0.0
        for node in Node.objects.filter(network=self.network).select_related("building"):
            fallback_kw += max(self._node_capacity_kw(node), 0.0)
        return fallback_kw * self.constants.hours_per_year

    def _profile_energy_kwh(self, queryset):
        return self._safe_float(queryset.aggregate(total=Sum("value")).get("total"))

    def _node_capacity_kw(self, node):
        info = node.other_information or {}
        if isinstance(info, dict):
            for key in [
                "capacity_kw",
                "peak_load_kw",
                "heating_demand_kw",
                "waste_heat_kw",
                "available_heat_kw",
                "thermal_capacity_kw",
                "capacity",
            ]:
                value = self._safe_float(info.get(key))
                if value > 0:
                    return value

        if node.building:
            for value in [node.building.heating_demand, node.building.cooling_demand]:
                value = self._safe_float(value)
                if value > 0:
                    return value
        if node.boiler:
            return self._safe_float(getattr(node.boiler, "capacity", 0.0))
        if node.thermal_storage:
            return self._safe_float(getattr(node.thermal_storage, "capacity", 0.0))
        return 0.0

    def _explicit_node_technology(self, node):
        if node.boiler_id:
            return "Gas boiler - boiler - natural gas - medium"
        if node.geothermal_id:
            return "Geothermal plant, electric heat pump, 2000m. DH temperature 80/40 degree - renewable heat - geothermal - medium"
        return None

    def _normalise_efficiency(self, value, default=0.9):
        efficiency = self._safe_float(value)
        if efficiency <= 0:
            efficiency = default
        if efficiency > 1:
            efficiency /= 100
        return max(efficiency, 0.01)

    def _dea_rows(self, technology):
        return DEAReferenceData.objects.filter(
            technology__iexact=technology,
            target_year=self.DEA_YEAR,
        ).filter(Q(estimate_type__iexact="ctrl") | Q(estimate_type__iexact="est"))

    def _first_row(self, queryset, *conditions):
        query = queryset
        for condition in conditions:
            query = query.filter(condition)
        return query.order_by("id").first()

    def _dea_capacity_mw(self, rows):
        row = self._first_row(rows, Q(parameter__icontains="Generating capacity for one unit"))
        return self._safe_float(row.value if row else None)

    def _dea_scaling_alpha(self, rows):
        row = self._first_row(
            rows,
            Q(parameter__icontains="scaling") | Q(parameter__icontains="exponent") | Q(parameter__icontains="alpha"),
        )
        value = self._safe_float(row.value if row else None)
        return value if value > 0 else 1.0

    def _dea_investment_to_cad(self, row, base_capacity_mw):
        value = self._safe_float(row.value)
        unit = (row.unit or row.parameter or "").lower()
        if "meur" in unit and "/mw" in unit:
            return value * base_capacity_mw * 1_000_000 * self.EUR_TO_CAD
        if "meur" in unit:
            return value * 1_000_000 * self.EUR_TO_CAD
        if "/mw" in unit:
            return value * base_capacity_mw * self.EUR_TO_CAD
        return value * self.EUR_TO_CAD

    def _node_fixed_om_cad(self, rows, capacity_mw, capex_cad):
        row = self._first_row(rows, Q(parameter__icontains="Fixed O&M"))
        if not row:
            return 0.0
        unit = (row.unit or row.parameter or "").lower()
        value = self._safe_float(row.value)
        if "%" in unit or "pct" in unit or "investment" in unit:
            return capex_cad * value
        if "/mw" in unit:
            return value * capacity_mw * self.EUR_TO_CAD
        return value * self.EUR_TO_CAD

    def _node_variable_om_cad(self, rows, annual_heat_kwh):
        row = self._first_row(rows, Q(parameter__icontains="Variable O&M"))
        if not row:
            return 0.0
        unit = (row.unit or row.parameter or "").lower()
        value = self._safe_float(row.value)
        annual_mwh = annual_heat_kwh / 1000
        if "/mwh" in unit:
            return value * annual_mwh * self.EUR_TO_CAD
        return 0.0

    def _sizing_length(self, row):
        if row.length_m:
            return float(row.length_m)
        if row.route and row.route.distance_meters:
            return float(row.route.distance_meters)
        return float(row.edge.length or 0.0)

    @lru_cache(maxsize=24)
    def _pipe_cost_curve(self, pipe_category, delta_t):
        technology = self.PIPE_TECH_ALIASES.get((pipe_category or "").lower(), pipe_category)
        rows = self._dea_rows(technology).filter(parameter__icontains="Investment costs; single line")
        diameters = []
        costs = []
        for row in rows:
            capacity_kw = self._representative_capacity_kw(row.parameter)
            if capacity_kw <= 0:
                continue
            diameter_m = self._diameter_for_capacity_kw(capacity_kw, delta_t or 20.0)
            cost_eur_per_m = self._pipe_row_cost_eur_per_m(row, capacity_kw / 1000)
            if diameter_m > 0 and cost_eur_per_m > 0:
                diameters.append(diameter_m)
                costs.append(cost_eur_per_m)

        if len(diameters) >= 3:
            quad, linear, intercept = np.polyfit(diameters, costs, 2)
            return {
                "source": "dea",
                "intercept": float(intercept),
                "linear": float(linear),
                "quadratic": float(quad),
            }

        catalog = self.constants.pipe_catalog if isinstance(self.constants.pipe_catalog, list) else []
        catalog_points = [
            (self._safe_float(item.get("di")), self._safe_float(item.get("cost")))
            for item in catalog
            if isinstance(item, dict) and self._safe_float(item.get("di")) > 0 and self._safe_float(item.get("cost")) > 0
        ]
        if len(catalog_points) >= 3:
            diameters, costs = zip(*catalog_points)
            quad, linear, intercept = np.polyfit(diameters, costs, 2)
            return {
                "source": "catalog",
                "intercept": float(intercept),
                "linear": float(linear),
                "quadratic": float(quad),
            }
        return {
            "source": "fallback",
            "intercept": float(self.constants.fallback_pipe_cost_per_meter or 150.0),
            "linear": 0.0,
            "quadratic": 0.0,
        }

    def _pipe_unit_cost_cad(self, curve, diameter_m):
        if curve["source"] == "dea":
            cost_eur = (
                curve["intercept"] * self.CIVIL_LABOR_PREMIUM +
                curve["linear"] * self.MATERIAL_PREMIUM * diameter_m +
                curve["quadratic"] * (diameter_m ** 2)
            )
            return max(cost_eur * self.EUR_TO_CAD, 0.0)
        cost_cad = curve["intercept"] + curve["linear"] * diameter_m + curve["quadratic"] * (diameter_m ** 2)
        return max(cost_cad, 0.0)

    def _pipe_annual_opex(self, annual_heat_kwh, total_length_km, peak_mw_sum):
        technology = "Energy Transport District Heating Transmission"
        rows = self._dea_rows(technology)
        fixed_row = self._first_row(rows, Q(parameter__icontains="Fixed O&M"))
        variable_row = self._first_row(rows, Q(parameter__icontains="Variable O&M"))

        fixed = 0.0
        if fixed_row and peak_mw_sum > 0 and total_length_km > 0:
            fixed = self._safe_float(fixed_row.value) * peak_mw_sum * total_length_km * self.EUR_TO_CAD

        variable = 0.0
        if variable_row and total_length_km > 0:
            variable = self._safe_float(variable_row.value) * (annual_heat_kwh / 1000) * total_length_km * self.EUR_TO_CAD

        return fixed + variable

    def _representative_capacity_kw(self, parameter):
        text = (parameter or "").lower().replace(",", "")
        above_match = re.search(r"above\s+(\d+(?:\.\d+)?)\s*(kw|mw)", text)
        if above_match:
            value = float(above_match.group(1))
            return value * (1000 if above_match.group(2) == "mw" else 1)

        range_match = re.search(r"(\d+(?:\.\d+)?)\s*(kw|mw)?\s*[-–]\s*(\d+(?:\.\d+)?)\s*(kw|mw)", text)
        if range_match:
            start = float(range_match.group(1))
            start_unit = range_match.group(2) or range_match.group(4)
            end = float(range_match.group(3))
            end_unit = range_match.group(4)
            start_kw = start * (1000 if start_unit == "mw" else 1)
            end_kw = end * (1000 if end_unit == "mw" else 1)
            return (start_kw + end_kw) / 2

        single_match = re.search(r"(\d+(?:\.\d+)?)\s*(kw|mw)", text)
        if single_match:
            value = float(single_match.group(1))
            return value * (1000 if single_match.group(2) == "mw" else 1)
        return 0.0

    def _diameter_for_capacity_kw(self, capacity_kw, delta_t):
        rho = float(self.constants.rho or 971.8)
        cp = float(self.constants.cp or 4.19)
        velocity = min(max(float(self.constants.max_velocity or 2.0) * 0.7, 0.8), 2.0)
        mass_flow_kg_s = capacity_kw / max(cp * delta_t, 0.001)
        volume_flow_m3_s = mass_flow_kg_s / rho
        area_m2 = volume_flow_m3_s / velocity
        return math.sqrt(max(4 * area_m2 / math.pi, 0.0001))

    def _pipe_row_cost_eur_per_m(self, row, representative_mw):
        value = self._safe_float(row.value)
        text = f"{row.unit or ''} {row.parameter or ''}".lower()
        if "meur/km" in text:
            return value * 1_000_000 / 1000
        if "/mw/m" in text:
            return value * representative_mw
        return value

    def _carbon_price_for_year(self, year):
        rates = self._carbon_rates()
        if year in rates:
            return rates[year]
        previous_years = [candidate for candidate in rates if candidate < year]
        if not previous_years:
            return 0.0
        price = rates[max(previous_years)]
        for _ in range(max(previous_years) + 1, year + 1):
            price *= 1.015
        return price

    @classmethod
    def _carbon_rates(cls):
        if cls._carbon_rates_cache is None:
            cls._carbon_rates_cache = {
                item.year: item.price_cad_per_tonne
                for item in CarbonTaxRate.objects.all().order_by("year")
            }
        return cls._carbon_rates_cache

    def _assert_analysis_scope(self):
        project_id = self.network.project_id
        if Node.objects.filter(network=self.network, building_id__isnull=False).exclude(building__project_id=project_id).exists():
            raise ValueError("Network contains a building node from another project.")
        if Node.objects.filter(network=self.network, boiler_id__isnull=False).exclude(boiler__project_id=project_id).exists():
            raise ValueError("Network contains a boiler node from another project.")
        if Edge.objects.filter(network=self.network).exclude(start_node__network=self.network).exists():
            raise ValueError("Network contains an edge whose start node belongs to another network.")
        if Edge.objects.filter(network=self.network).exclude(end_node__network=self.network).exists():
            raise ValueError("Network contains an edge whose end node belongs to another network.")

    def _safe_float(self, value):
        try:
            if value is None:
                return 0.0
            return float(value)
        except (TypeError, ValueError):
            return 0.0
