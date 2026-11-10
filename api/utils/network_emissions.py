import math
import xml.etree.ElementTree as ET
from collections import defaultdict

import requests
from django.core.cache import cache

from api.models import (
    Edge,
    EdgePipeSizing,
    LoadProfile,
    Network,
    Node,
    PipeSizing,
    Route,
    assert_network_project_integrity,
)
from api.utils.network_costing import (
    _catalog_pipe_for_diameter,
    _network_length_entries,
    _normal_catalog,
    get_or_create_design_constants,
)


OEKOB_BASE_URL = "https://www.oekobaudat.de/OEKOBAU.DAT/resource"
OEKOB_DATASTOCK_UUID = "ca70a7e6-0ea4-4e90-a947-d44585783626"
OEKOB_DATASTOCK_NAME = "OBD_2024_I"

LIFECYCLE_STAGE_GROUPS = {
    "production": ["A1", "A2", "A3", "A1-A3"],
    "construction": ["A4", "A5"],
    "use": ["B1", "B2", "B3", "B4", "B5", "B6", "B7"],
    "end_of_life": ["C1", "C2", "C3", "C4"],
    "beyond_boundary": ["D"],
}

MATERIALS = {
    "steel": {
        "label": "Steel pipe",
        "query": "Stahlrohr",
        "density_kg_m3": 7850.0,
        "wall_ratio": 0.045,
        "min_wall_m": 0.0026,
        "max_wall_m": 0.020,
        "fallback_modules": {
            "A1-A3": 2.49119095715841,
            "C2": 0.0100772533213305,
            "C4": 0.00164686494353777,
            "D": -1.32349983793772,
        },
        "fallback_dataset": "ÖKOBAUDAT OBD_2024_I Steel pipe",
    },
    "pe_hd": {
        "label": "PE-HD pipe",
        "query": "PE-HD pipe",
        "density_kg_m3": 950.0,
        "wall_ratio": 0.080,
        "min_wall_m": 0.0030,
        "max_wall_m": 0.035,
        "fallback_modules": {
            "A1-A3": 2.32488641181328,
            "C2": 0.00340594400973465,
            "C3": 3.45187493039417,
            "D": -1.52435389181727,
        },
        "fallback_dataset": "ÖKOBAUDAT OBD_2024_I Sewer pipe PE-HD",
    },
    "pvc": {
        "label": "PVC pipe",
        "query": "PVC pipe",
        "density_kg_m3": 1400.0,
        "wall_ratio": 0.065,
        "min_wall_m": 0.0030,
        "max_wall_m": 0.030,
        "fallback_modules": {
            "A1-A3": 2.75985645738677,
            "C2": 0.00426009080011634,
            "C3": 2.25350607781064,
            "D": -0.596308502265591,
        },
        "fallback_dataset": "ÖKOBAUDAT OBD_2024_I Sewer pipe PVC",
    },
}

LOCAL_BOILER_BASE_CASE = {
    "fuel_type": "natural_gas",
    "thermal_efficiency": 0.90,
    "fuel_emission_factor_kgco2e_per_kwh": 0.185,
    "capacity_safety_factor": 1.15,
}

FALLBACK_HEAT_EUI_KWH_M2_YR = {
    "residential": 120.0,
    "commercial": 150.0,
    "industrial": 180.0,
    "public": 140.0,
    "other": 140.0,
}


def build_network_emissions_summary(
    network_id,
    *,
    material="steel",
    refresh_epd=False,
    component_allowance_pct=15.0,
):
    network = Network.objects.get(pk=network_id)
    assert_network_project_integrity(network)
    constants = get_or_create_design_constants(network)
    nodes = Node.objects.filter(network=network)
    edge_count = Edge.objects.filter(network=network).count()

    if edge_count == 0:
        return _build_local_boiler_base_case_summary(
            network,
            constants,
            nodes,
        )

    material_key = material if material in MATERIALS else "steel"
    material_meta = MATERIALS[material_key]
    factor = get_material_gwp_factor(material_key, refresh=refresh_epd)
    allowance_fraction = max(0.0, _float_or_default(component_allowance_pct, 15.0)) / 100.0

    latest_sizing = (
        PipeSizing.objects.filter(network=network)
        .order_by("-created_at", "-id")
        .first()
    )
    sizing_rows = []
    sizing_by_edge = {}
    if latest_sizing:
        sizing_rows = list(
            EdgePipeSizing.objects.filter(pipe_sizing=latest_sizing)
            .select_related("edge", "route")
        )
        sizing_by_edge = {item.edge_id: item for item in sizing_rows}

    entries, length_basis, length_stats = _network_length_entries(
        network,
        sizing_by_edge=sizing_by_edge,
        sizing_rows=sizing_rows,
    )
    pipe_catalog = _normal_catalog(constants.pipe_catalog)

    segments = []
    by_diameter = defaultdict(lambda: {
        "dn": None,
        "diameter_m": None,
        "segment_count": 0,
        "length_m": 0.0,
        "mass_kg": 0.0,
        "pipe_emissions_kgco2e": 0.0,
        "allowance_emissions_kgco2e": 0.0,
        "total_emissions_kgco2e": 0.0,
    })
    lifecycle_totals = defaultdict(float)

    total_length_m = 0.0
    total_pipe_mass_kg = 0.0
    total_pipe_emissions = 0.0
    total_allowance_emissions = 0.0
    total_beyond_boundary = 0.0

    for entry in entries:
        edge = entry["edge"]
        sizing = entry.get("sizing") or sizing_by_edge.get(edge.id)
        diameter_m = _resolve_diameter_m(sizing, pipe_catalog)
        catalog_pipe = _catalog_pipe_for_diameter(pipe_catalog, diameter_m)
        dn = int(catalog_pipe["dn"]) if catalog_pipe and "dn" in catalog_pipe else _diameter_to_dn(diameter_m)
        length_m = float(entry.get("length_m") or 0.0)
        mass_per_m = pipe_mass_per_meter(diameter_m, material_meta)
        mass_kg = mass_per_m * length_m
        pipe_emissions = mass_kg * factor["cradle_to_grave_kgco2e_per_kg"]
        allowance_emissions = pipe_emissions * allowance_fraction
        total_segment = pipe_emissions + allowance_emissions
        beyond_boundary = mass_kg * factor["module_groups"]["beyond_boundary"]

        for group, value in factor["module_groups"].items():
            if group == "beyond_boundary":
                continue
            lifecycle_totals[group] += mass_kg * value * (1.0 + allowance_fraction)

        total_length_m += length_m
        total_pipe_mass_kg += mass_kg
        total_pipe_emissions += pipe_emissions
        total_allowance_emissions += allowance_emissions
        total_beyond_boundary += beyond_boundary

        key = str(dn or "unsized")
        bucket = by_diameter[key]
        bucket["dn"] = dn
        bucket["diameter_m"] = diameter_m
        bucket["segment_count"] += 1
        bucket["length_m"] += length_m
        bucket["mass_kg"] += mass_kg
        bucket["pipe_emissions_kgco2e"] += pipe_emissions
        bucket["allowance_emissions_kgco2e"] += allowance_emissions
        bucket["total_emissions_kgco2e"] += total_segment

        segments.append({
            "edge_id": edge.id,
            "edge_name": edge.name,
            "route_id": entry.get("route_id"),
            "source_route_ids": entry.get("source_route_ids", []),
            "source_edge_ids": entry.get("source_edge_ids", []),
            "overlap_count": entry.get("overlap_count", 1),
            "is_shared": entry.get("is_shared", False),
            "length_m": length_m,
            "diameter_m": diameter_m,
            "dn": dn,
            "material": material_key,
            "mass_kg": mass_kg,
            "mass_per_m_kg": mass_per_m,
            "pipe_emissions_kgco2e": pipe_emissions,
            "allowance_emissions_kgco2e": allowance_emissions,
            "total_emissions_kgco2e": total_segment,
            "beyond_boundary_kgco2e": beyond_boundary,
            "factor_kgco2e_per_kg": factor["cradle_to_grave_kgco2e_per_kg"],
            "factor_source": factor["source"],
        })

    component_breakdown = [
        {
            "component": "Pipe material",
            "basis": f"{material_meta['label']} mass from diameter x length",
            "emissions_kgco2e": total_pipe_emissions,
        },
        {
            "component": "Fittings, valves and casing allowance",
            "basis": f"{allowance_fraction * 100:.1f}% of pipe material emissions",
            "emissions_kgco2e": total_allowance_emissions,
        },
    ]

    total_emissions = total_pipe_emissions + total_allowance_emissions
    diameter_breakdown = sorted(
        by_diameter.values(),
        key=lambda item: (item["dn"] is None, item["dn"] or 0),
    )

    node_counts = {
        "consumer": nodes.filter(node_type__in=["consumer", "building"]).count(),
        "producer": nodes.filter(node_type__in=["producer", "boiler"]).count(),
        "junction": nodes.filter(node_type="junction").count(),
        "total": nodes.count(),
    }

    return {
        "network": {
            "id": network.id,
            "name": network.name,
            "project_id": network.project_id,
            "project_name": network.project.name if network.project else None,
            "network_type": network.network_type,
            "derivation_stage": network.derivation_stage,
            "derivation_method": network.derivation_method,
        },
        "emissions_basis": {
            "public_api": "ÖKOBAUDAT REST API",
            "api_url": f"{OEKOB_BASE_URL}/datastocks/{OEKOB_DATASTOCK_UUID}/processes/",
            "datastock": OEKOB_DATASTOCK_NAME,
            "indicator": "GWP-total / Climate change",
            "boundary": "cradle-to-grave modules A, B and C; module D reported separately",
            "unit": "kg CO2e",
            "length_basis": length_basis,
            "route_length_method": length_stats.get("method"),
            "pipe_sizing_id": latest_sizing.id if latest_sizing else None,
            "pipe_sizing_name": latest_sizing.name if latest_sizing else None,
            "material": material_key,
            "material_label": material_meta["label"],
            "density_kg_m3": material_meta["density_kg_m3"],
            "component_allowance_pct": allowance_fraction * 100,
            "diameter_basis": "latest pipe sizing diameter; falls back to nearest catalog pipe or DN80",
            "mass_basis": "hollow cylinder using internal diameter, estimated wall thickness, and material density",
            "factor": factor,
        },
        "metrics": {
            "node_count": node_counts["total"],
            "consumer_count": node_counts["consumer"],
            "producer_count": node_counts["producer"],
            "junction_count": node_counts["junction"],
            "edge_count": edge_count,
            "route_count": Route.objects.filter(edge__network=network).count(),
            "segment_count": len(segments),
            "total_length_m": total_length_m,
            "individual_route_length_m": length_stats.get("individual_route_length_m", total_length_m),
            "overlap_deduped_length_m": length_stats.get("overlap_deduped_length_m", 0.0),
            "overlap_ratio": length_stats.get("overlap_ratio", 0.0),
            "total_pipe_mass_kg": total_pipe_mass_kg,
            "total_emissions_kgco2e": total_emissions,
            "total_emissions_tco2e": total_emissions / 1000.0,
            "pipe_material_emissions_kgco2e": total_pipe_emissions,
            "component_allowance_emissions_kgco2e": total_allowance_emissions,
            "beyond_boundary_module_d_kgco2e": total_beyond_boundary,
            "average_kgco2e_per_m": total_emissions / total_length_m if total_length_m else 0.0,
            "average_pipe_mass_kg_per_m": total_pipe_mass_kg / total_length_m if total_length_m else 0.0,
        },
        "lifecycle_modules": [
            {"stage": _stage_label(key), "key": key, "emissions_kgco2e": lifecycle_totals.get(key, 0.0)}
            for key in ["production", "construction", "use", "end_of_life"]
        ],
        "component_breakdown": component_breakdown,
        "breakdown_by_diameter": diameter_breakdown,
        "segments": sorted(segments, key=lambda item: item["total_emissions_kgco2e"], reverse=True),
    }


def build_network_emissions_compare(
    network_ids,
    *,
    material="steel",
    refresh_epd=False,
    component_allowance_pct=15.0,
):
    ids = []
    for network_id in network_ids:
        try:
            ids.append(int(network_id))
        except (TypeError, ValueError):
            continue
    if not ids:
        return []

    networks = list(Network.objects.filter(id__in=ids).only("id", "project_id"))
    found_ids = {network.id for network in networks}
    missing_ids = [network_id for network_id in ids if network_id not in found_ids]
    if missing_ids:
        raise ValueError(f"Networks not found: {missing_ids}")

    project_ids = {network.project_id for network in networks}
    if len(project_ids) > 1:
        raise ValueError("Network emissions comparison is project-scoped. Select networks from one project only.")

    return [
        build_network_emissions_summary(
            network_id,
            material=material,
            refresh_epd=refresh_epd,
            component_allowance_pct=component_allowance_pct,
        )
        for network_id in ids
    ]


def _build_local_boiler_base_case_summary(network, constants, nodes):
    boiler_assumptions = dict(LOCAL_BOILER_BASE_CASE)
    lifespan_years = int(constants.lifespan_years or 30)
    building_nodes = list(
        nodes
        .filter(building__isnull=False)
        .select_related("building")
        .order_by("id")
    )
    consumer_nodes = [
        node for node in building_nodes
        if node.node_type in ["consumer", "building"]
    ]
    selected_nodes = consumer_nodes or building_nodes
    unique_buildings = {}
    for node in selected_nodes:
        if node.building_id and node.building_id not in unique_buildings:
            unique_buildings[node.building_id] = node.building

    buildings = list(unique_buildings.values())
    building_ids = [building.id for building in buildings]
    demand_stats = _building_heat_demand_stats(
        buildings,
        constants,
    )

    boiler_rows = []
    total_annual_heat_kwh = 0.0
    total_peak_heat_kw = 0.0
    total_installed_capacity_kw = 0.0
    total_annual_fuel_kwh = 0.0
    total_annual_emissions_kgco2e = 0.0

    for building in buildings:
        stats = demand_stats.get(building.id, {})
        annual_heat_kwh = float(stats.get("annual_heat_kwh") or 0.0)
        peak_heat_kw = float(stats.get("peak_heat_kw") or 0.0)
        installed_capacity_kw = peak_heat_kw * boiler_assumptions["capacity_safety_factor"]
        annual_fuel_kwh = (
            annual_heat_kwh / boiler_assumptions["thermal_efficiency"]
            if boiler_assumptions["thermal_efficiency"] > 0
            else 0.0
        )
        annual_emissions = annual_fuel_kwh * boiler_assumptions["fuel_emission_factor_kgco2e_per_kwh"]
        study_period_emissions = annual_emissions * lifespan_years

        total_annual_heat_kwh += annual_heat_kwh
        total_peak_heat_kw += peak_heat_kw
        total_installed_capacity_kw += installed_capacity_kw
        total_annual_fuel_kwh += annual_fuel_kwh
        total_annual_emissions_kgco2e += annual_emissions

        boiler_rows.append({
            "building_id": building.id,
            "building_name": building.name or f"Building {building.id}",
            "building_type": building.building_type,
            "demand_source": stats.get("source", "none"),
            "profile_count": int(stats.get("profile_count") or 0),
            "annual_heat_kwh": annual_heat_kwh,
            "annual_heat_mwh": annual_heat_kwh / 1000.0,
            "peak_heat_kw": peak_heat_kw,
            "installed_boiler_capacity_kw": installed_capacity_kw,
            "thermal_efficiency": boiler_assumptions["thermal_efficiency"],
            "annual_fuel_kwh": annual_fuel_kwh,
            "fuel_type": boiler_assumptions["fuel_type"],
            "fuel_emission_factor_kgco2e_per_kwh": boiler_assumptions["fuel_emission_factor_kgco2e_per_kwh"],
            "annual_emissions_kgco2e": annual_emissions,
            "total_emissions_kgco2e": study_period_emissions,
            "edge_id": None,
            "edge_name": building.name or f"Building {building.id}",
            "route_id": None,
            "source_route_ids": [],
            "source_edge_ids": [],
            "overlap_count": 1,
            "is_shared": False,
            "length_m": 0.0,
            "diameter_m": None,
            "dn": None,
            "material": "local_boiler",
            "mass_kg": 0.0,
            "mass_per_m_kg": 0.0,
            "pipe_emissions_kgco2e": 0.0,
            "allowance_emissions_kgco2e": 0.0,
            "beyond_boundary_kgco2e": 0.0,
            "factor_kgco2e_per_kg": None,
            "factor_source": "local_boiler_base_case_assumption",
        })

    total_study_period_emissions = total_annual_emissions_kgco2e * lifespan_years
    lifecycle_totals = {
        "production": 0.0,
        "construction": 0.0,
        "use": total_study_period_emissions,
        "end_of_life": 0.0,
    }

    node_counts = {
        "consumer": nodes.filter(node_type__in=["consumer", "building"]).count(),
        "producer": nodes.filter(node_type__in=["producer", "boiler"]).count(),
        "junction": nodes.filter(node_type="junction").count(),
        "total": nodes.count(),
    }

    return {
        "network": {
            "id": network.id,
            "name": network.name,
            "project_id": network.project_id,
            "project_name": network.project.name if network.project else None,
            "network_type": network.network_type,
            "derivation_stage": network.derivation_stage,
            "derivation_method": network.derivation_method,
        },
        "emissions_basis": {
            "mode": "local_boiler_base_case",
            "mode_label": "Local boiler base case",
            "public_api": "Local boiler engineering assumption",
            "api_url": None,
            "datastock": "No network edges: one boiler per consumer building",
            "indicator": "Operational GWP from boiler fuel use",
            "boundary": f"Base case boiler fuel emissions over {lifespan_years} years; distribution network embodied emissions are zero because no edges exist",
            "unit": "kg CO2e",
            "length_basis": "no edges; local boilers at buildings",
            "route_length_method": "not_applicable",
            "pipe_sizing_id": None,
            "pipe_sizing_name": None,
            "material": "local_boiler",
            "material_label": "Local building boilers",
            "density_kg_m3": None,
            "component_allowance_pct": 0.0,
            "diameter_basis": "not applicable",
            "mass_basis": "not applicable",
            "boiler_assumptions": boiler_assumptions,
            "demand_basis": "Hourly heating load profiles integrated as kWh; falls back to building heating demand times full-load hours, then floor-area EUI if needed",
            "factor": {
                "material": "natural_gas_boiler",
                "material_label": "Natural gas boiler fuel use",
                "source": "local_engineering_assumption",
                "dataset_name": "Natural gas boiler base case",
                "dataset_uuid": None,
                "dataset_version": None,
                "dataset_url": None,
                "modules": {"B6": boiler_assumptions["fuel_emission_factor_kgco2e_per_kwh"]},
                "module_groups": {
                    "production": 0.0,
                    "construction": 0.0,
                    "use": boiler_assumptions["fuel_emission_factor_kgco2e_per_kwh"],
                    "end_of_life": 0.0,
                    "beyond_boundary": 0.0,
                },
                "cradle_to_grave_kgco2e_per_kg": None,
                "fuel_emission_factor_kgco2e_per_kwh": boiler_assumptions["fuel_emission_factor_kgco2e_per_kwh"],
                "declared_unit_assumption": "kWh fuel input",
            },
        },
        "metrics": {
            "node_count": node_counts["total"],
            "consumer_count": node_counts["consumer"],
            "producer_count": node_counts["producer"],
            "junction_count": node_counts["junction"],
            "edge_count": 0,
            "route_count": 0,
            "segment_count": len(boiler_rows),
            "total_length_m": 0.0,
            "individual_route_length_m": 0.0,
            "overlap_deduped_length_m": 0.0,
            "overlap_ratio": 0.0,
            "total_pipe_mass_kg": 0.0,
            "total_emissions_kgco2e": total_study_period_emissions,
            "total_emissions_tco2e": total_study_period_emissions / 1000.0,
            "pipe_material_emissions_kgco2e": 0.0,
            "component_allowance_emissions_kgco2e": 0.0,
            "beyond_boundary_module_d_kgco2e": 0.0,
            "average_kgco2e_per_m": 0.0,
            "average_pipe_mass_kg_per_m": 0.0,
            "base_case_building_count": len(boiler_rows),
            "annual_heat_demand_kwh": total_annual_heat_kwh,
            "annual_heat_demand_mwh": total_annual_heat_kwh / 1000.0,
            "peak_heat_demand_kw": total_peak_heat_kw,
            "installed_boiler_capacity_kw": total_installed_capacity_kw,
            "annual_boiler_fuel_kwh": total_annual_fuel_kwh,
            "annual_boiler_emissions_kgco2e": total_annual_emissions_kgco2e,
            "annual_boiler_emissions_tco2e": total_annual_emissions_kgco2e / 1000.0,
            "boiler_efficiency": boiler_assumptions["thermal_efficiency"],
            "boiler_fuel_emission_factor_kgco2e_per_kwh": boiler_assumptions["fuel_emission_factor_kgco2e_per_kwh"],
            "study_period_years": lifespan_years,
            "average_kgco2e_per_mwh_heat": (
                total_annual_emissions_kgco2e / (total_annual_heat_kwh / 1000.0)
                if total_annual_heat_kwh
                else 0.0
            ),
        },
        "lifecycle_modules": [
            {"stage": _stage_label(key), "key": key, "emissions_kgco2e": lifecycle_totals.get(key, 0.0)}
            for key in ["production", "construction", "use", "end_of_life"]
        ],
        "component_breakdown": [
            {
                "component": "Local boiler fuel use",
                "basis": (
                    f"Annual heat demand / {boiler_assumptions['thermal_efficiency']:.0%} "
                    f"efficiency x {boiler_assumptions['fuel_emission_factor_kgco2e_per_kwh']:.3f} kg CO2e/kWh fuel "
                    f"x {lifespan_years} years"
                ),
                "emissions_kgco2e": total_study_period_emissions,
            },
        ],
        "breakdown_by_diameter": [],
        "building_boilers": sorted(
            boiler_rows,
            key=lambda item: item["total_emissions_kgco2e"],
            reverse=True,
        ),
        "segments": sorted(
            boiler_rows,
            key=lambda item: item["total_emissions_kgco2e"],
            reverse=True,
        ),
    }


def _building_heat_demand_stats(buildings, constants):
    building_ids = [building.id for building in buildings]
    stats = defaultdict(lambda: {
        "annual_heat_kwh": 0.0,
        "peak_heat_kw": 0.0,
        "profile_count": 0,
        "source": "load_profiles",
    })

    profiles = (
        LoadProfile.objects
        .filter(building_id__in=building_ids, profile_type="heating")
        .values("building_id", "value", "start_date", "end_date")
    )
    for profile in profiles:
        building_id = profile["building_id"]
        value_kw = max(_float_or_default(profile.get("value"), 0.0), 0.0)
        duration_h = _profile_duration_hours(profile.get("start_date"), profile.get("end_date"))
        stats[building_id]["annual_heat_kwh"] += value_kw * duration_h
        stats[building_id]["peak_heat_kw"] = max(stats[building_id]["peak_heat_kw"], value_kw)
        stats[building_id]["profile_count"] += 1

    full_load_hours = max(float(constants.hours_per_year or 4000), 0.0)
    for building in buildings:
        existing = stats.get(building.id)
        if existing and existing["profile_count"] > 0:
            continue

        peak_kw = max(_float_or_default(building.heating_demand, 0.0), 0.0)
        if peak_kw > 0:
            stats[building.id] = {
                "annual_heat_kwh": peak_kw * full_load_hours,
                "peak_heat_kw": peak_kw,
                "profile_count": 0,
                "source": "building_heating_demand_full_load_hours",
            }
            continue

        floor_area = max(_float_or_default(building.floor_area, 0.0), 0.0)
        if floor_area > 0:
            eui = FALLBACK_HEAT_EUI_KWH_M2_YR.get(
                building.building_type,
                FALLBACK_HEAT_EUI_KWH_M2_YR["other"],
            )
            annual_heat = floor_area * eui
            stats[building.id] = {
                "annual_heat_kwh": annual_heat,
                "peak_heat_kw": annual_heat / full_load_hours if full_load_hours else 0.0,
                "profile_count": 0,
                "source": "floor_area_heat_eui_fallback",
            }
            continue

        stats[building.id] = {
            "annual_heat_kwh": 0.0,
            "peak_heat_kw": 0.0,
            "profile_count": 0,
            "source": "no_heat_demand_available",
        }

    return stats


def _profile_duration_hours(start_date, end_date):
    if not start_date or not end_date:
        return 1.0
    try:
        hours = (end_date - start_date).total_seconds() / 3600.0
    except (TypeError, AttributeError):
        return 1.0
    if not math.isfinite(hours) or hours <= 0:
        return 1.0
    return min(hours, 24.0 * 366.0)


def get_material_gwp_factor(material_key, refresh=False):
    cache_key = f"oekobaudat:gwp:{OEKOB_DATASTOCK_UUID}:{material_key}:v2"
    if not refresh:
        cached = cache.get(cache_key)
        if cached:
            return cached

    try:
        factor = _fetch_oekobaudat_factor(material_key)
        cache.set(cache_key, factor, timeout=60 * 60 * 24 * 30)
        return factor
    except Exception:
        return _fallback_factor(material_key)


def _fetch_oekobaudat_factor(material_key):
    material = MATERIALS[material_key]
    response = requests.get(
        f"{OEKOB_BASE_URL}/datastocks/{OEKOB_DATASTOCK_UUID}/processes/",
        params={"search": "true", "name": material["query"]},
        timeout=12,
    )
    response.raise_for_status()
    process = _select_process(response.content, material["query"])
    if not process:
        raise ValueError("No matching ÖKOBAUDAT process found.")

    detail_url = (
        f"{OEKOB_BASE_URL}/datastocks/{OEKOB_DATASTOCK_UUID}/processes/"
        f"{process['uuid']}?version={process['version']}&format=xml"
    )
    detail = requests.get(detail_url, timeout=12)
    detail.raise_for_status()
    modules = _parse_gwp_modules(detail.content)
    if not modules:
        raise ValueError("Matching ÖKOBAUDAT process did not include GWP-total modules.")

    return _factor_payload(
        material_key,
        modules,
        source="oekobaudat_api",
        dataset_name=process["name"],
        dataset_uuid=process["uuid"],
        dataset_version=process["version"],
        dataset_url=detail_url,
    )


def _select_process(xml_bytes, query):
    root = ET.fromstring(xml_bytes)
    processes = []
    for process in root.findall("{http://www.ilcd-network.org/ILCD/ServiceAPI/Process}process"):
        uuid = process.findtext("{http://www.ilcd-network.org/ILCD/ServiceAPI}uuid")
        version = process.findtext("{http://www.ilcd-network.org/ILCD/ServiceAPI}dataSetVersion")
        names = [
            item.text or ""
            for item in process.findall("{http://www.ilcd-network.org/ILCD/ServiceAPI}name")
        ]
        if uuid and version:
            processes.append({
                "uuid": uuid,
                "version": version,
                "name": next((name for name in names if name), query),
                "names": names,
            })
    if not processes:
        return None

    query_terms = [part.lower() for part in query.replace("-", " ").split() if part]

    def score(process):
        names = " ".join(process["names"]).lower()
        exact = query.lower() in names
        term_score = sum(1 for term in query_terms if term in names)
        return (1 if exact else 0, term_score, -len(names))

    return sorted(processes, key=score, reverse=True)[0]


def _parse_gwp_modules(xml_bytes):
    root = ET.fromstring(xml_bytes)
    for result in root.findall(".//{http://lca.jrc.it/ILCD/Process}LCIAResult"):
        descriptions = " ".join(
            (item.text or "")
            for item in result.findall(".//{http://lca.jrc.it/ILCD/Common}shortDescription")
        ).lower()
        if "gwp-total" not in descriptions and "climate change" not in descriptions:
            continue
        if any(token in descriptions for token in ["fossil", "biogenic", "luluc"]):
            continue
        modules = {}
        for amount in result.findall(".//{http://www.iai.kit.edu/EPD/2013}amount"):
            module = amount.attrib.get("{http://www.iai.kit.edu/EPD/2013}module")
            if not module:
                continue
            try:
                modules[module] = float(amount.text or 0.0)
            except (TypeError, ValueError):
                continue
        if modules:
            return modules
    return {}


def _fallback_factor(material_key):
    material = MATERIALS[material_key]
    return _factor_payload(
        material_key,
        material["fallback_modules"],
        source="fallback_oekobaudat_reference",
        dataset_name=material["fallback_dataset"],
        dataset_uuid=None,
        dataset_version=None,
        dataset_url=None,
    )


def _factor_payload(material_key, modules, *, source, dataset_name, dataset_uuid, dataset_version, dataset_url):
    module_groups = {
        group: sum(float(modules.get(module, 0.0)) for module in module_names)
        for group, module_names in LIFECYCLE_STAGE_GROUPS.items()
    }
    cradle_to_grave = sum(
        value for group, value in module_groups.items()
        if group != "beyond_boundary"
    )
    return {
        "material": material_key,
        "material_label": MATERIALS[material_key]["label"],
        "source": source,
        "dataset_name": dataset_name,
        "dataset_uuid": dataset_uuid,
        "dataset_version": dataset_version,
        "dataset_url": dataset_url,
        "modules": modules,
        "module_groups": module_groups,
        "cradle_to_grave_kgco2e_per_kg": cradle_to_grave,
        "beyond_boundary_kgco2e_per_kg": module_groups["beyond_boundary"],
        "declared_unit_assumption": "kg material",
    }


def pipe_mass_per_meter(diameter_m, material_meta):
    di = max(float(diameter_m or 0.08), 0.005)
    wall = min(
        material_meta["max_wall_m"],
        max(material_meta["min_wall_m"], di * material_meta["wall_ratio"]),
    )
    outer = di + (2.0 * wall)
    area = math.pi * ((outer ** 2) - (di ** 2)) / 4.0
    return area * material_meta["density_kg_m3"]


def _resolve_diameter_m(sizing, pipe_catalog):
    if sizing and sizing.diameter_m:
        return float(sizing.diameter_m)
    if pipe_catalog:
        dn80 = next((pipe for pipe in pipe_catalog if int(pipe.get("dn") or 0) == 80), None)
        return float((dn80 or pipe_catalog[0])["di"])
    return 0.08


def _diameter_to_dn(diameter_m):
    if not diameter_m:
        return None
    return int(round(float(diameter_m) * 1000))


def _float_or_default(value, default):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _stage_label(key):
    return {
        "production": "Production A1-A3",
        "construction": "Construction A4-A5",
        "use": "Use B1-B7",
        "end_of_life": "End of life C1-C4",
        "beyond_boundary": "Beyond boundary D",
    }.get(key, key)
