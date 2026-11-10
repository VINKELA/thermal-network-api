from collections import defaultdict

from django.db.models import Avg, Max, Sum

from api.models import (
    Edge,
    EdgePipeSizing,
    ImportedBuildingHeating,
    ImportedHeatingShape,
    Network,
    NetworkDesignConstants,
    PipeSizing,
    Route,
    assert_network_project_integrity,
)
from api.utils.route_geometry import route_physical_segments


def get_or_create_design_constants(network):
    constants, _ = NetworkDesignConstants.objects.get_or_create(network=network)
    return constants


def build_network_cost_summary(network_id, cost_per_meter=None):
    network = Network.objects.get(pk=network_id)
    assert_network_project_integrity(network)
    constants = get_or_create_design_constants(network)
    has_cost_override = cost_per_meter is not None and cost_per_meter != ""
    fallback_cost = _float_or_default(cost_per_meter, constants.fallback_pipe_cost_per_meter)
    pipe_catalog = _normal_catalog(constants.pipe_catalog)
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
            .select_related("edge")
        )
        sizing_by_edge = {item.edge_id: item for item in sizing_rows}
    if not latest_sizing or not sizing_rows:
        raise ValueError("Network costing requires a completed pipe-sizing run.")

    entries, length_basis, length_stats = _network_length_entries(
        network,
        sizing_by_edge=sizing_by_edge,
        sizing_rows=sizing_rows,
    )
    hourly_profile = _hourly_flow_profile(network)
    breakdown_by_diameter = defaultdict(lambda: {
        "diameter_m": None,
        "dn": None,
        "segment_count": 0,
        "length_m": 0.0,
        "construction_cost": 0.0,
        "pumping_lifecycle_cost": 0.0,
        "thermal_loss_lifecycle_cost": 0.0,
        "thermal_loss_kw": 0.0,
        "total_cost": 0.0,
    })

    segment_breakdown = []
    total_length = 0.0
    construction_cost = 0.0
    pumping_lifecycle_cost = 0.0
    thermal_loss_lifecycle_cost = 0.0
    total_thermal_loss_kw = 0.0
    annual_pumping_energy_kwh = 0.0
    total_pressure_drop_pa = 0.0

    for entry in entries:
        edge = entry["edge"]
        length = entry["length_m"] or 0.0
        sizing = entry.get("sizing") or sizing_by_edge.get(edge.id)
        diameter_m = sizing.diameter_m if sizing else None
        catalog_pipe = _catalog_pipe_for_diameter(pipe_catalog, diameter_m)
        if has_cost_override:
            unit_cost = fallback_cost
            cost_source = "cost_per_meter_override"
        elif catalog_pipe:
            unit_cost = float(catalog_pipe.get("cost"))
            cost_source = "pipe_catalog"
        else:
            unit_cost = fallback_cost
            cost_source = "fallback"
        dn = int(catalog_pipe["dn"]) if catalog_pipe and "dn" in catalog_pipe else None
        source_length = float(entry.get("source_length_m") or edge.length or length or 0.0)
        pressure_drop_pa = float(sizing.pressure_drop_pa or 0.0) if sizing else 0.0
        if source_length > 0 and length > 0 and length < source_length:
            pressure_drop_pa *= length / source_length
        volume_flow_m3_s = float(sizing.volume_flow_m3_s or 0.0) if sizing else 0.0

        pipe_cost = length * unit_cost
        pumping = _pumping_lifecycle_metrics(
            volume_flow_m3_s,
            pressure_drop_pa,
            constants,
            hourly_profile.get("flow_factors"),
        )
        pump_cost = pumping["lifecycle_cost"]
        thermal_loss_kw = _thermal_loss_kw(length, constants)
        thermal_loss_cost = _thermal_loss_lifecycle_cost(thermal_loss_kw, constants)
        total_segment_cost = pipe_cost + pump_cost + thermal_loss_cost

        total_length += length
        construction_cost += pipe_cost
        pumping_lifecycle_cost += pump_cost
        thermal_loss_lifecycle_cost += thermal_loss_cost
        total_thermal_loss_kw += thermal_loss_kw
        annual_pumping_energy_kwh += pumping["annual_energy_kwh"]
        total_pressure_drop_pa += pressure_drop_pa

        key = str(dn or "unsized")
        bucket = breakdown_by_diameter[key]
        bucket["diameter_m"] = diameter_m
        bucket["dn"] = dn
        bucket["segment_count"] += 1
        bucket["length_m"] += length
        bucket["construction_cost"] += pipe_cost
        bucket["pumping_lifecycle_cost"] += pump_cost
        bucket["thermal_loss_lifecycle_cost"] += thermal_loss_cost
        bucket["thermal_loss_kw"] += thermal_loss_kw
        bucket["total_cost"] += total_segment_cost

        segment_breakdown.append({
            "edge_id": edge.id,
            "edge_name": edge.name,
            "route_id": entry.get("route_id"),
            "source_route_ids": entry.get("source_route_ids", []),
            "source_edge_ids": entry.get("source_edge_ids", []),
            "overlap_count": entry.get("overlap_count", 1),
            "is_shared": entry.get("is_shared", False),
            "length_m": length,
            "source_length_m": entry.get("source_length_m"),
            "diameter_m": diameter_m,
            "dn": dn,
            "unit_cost_per_m": unit_cost,
            "cost_source": cost_source,
            "construction_cost": pipe_cost,
            "pumping_lifecycle_cost": pump_cost,
            "annual_pumping_energy_kwh": pumping["annual_energy_kwh"],
            "thermal_loss_lifecycle_cost": thermal_loss_cost,
            "thermal_loss_kw": thermal_loss_kw,
            "total_cost": total_segment_cost,
            "velocity_m_s": sizing.velocity_m_s if sizing else None,
            "pressure_drop_pa": pressure_drop_pa if sizing else None,
        })

    sizing_stats = EdgePipeSizing.objects.filter(pipe_sizing=latest_sizing) if latest_sizing else EdgePipeSizing.objects.filter(edge__network=network)
    sizing_count = sizing_stats.count()
    diameter_stats = sizing_stats.aggregate(
        avg_diameter_m=Avg("diameter_m"),
        max_diameter_m=Max("diameter_m"),
        max_velocity_m_s=Max("velocity_m_s"),
        total_pressure_drop_pa=Sum("pressure_drop_pa"),
    )

    total_lifecycle_cost = construction_cost + pumping_lifecycle_cost + thermal_loss_lifecycle_cost
    diameter_breakdown = sorted(
        breakdown_by_diameter.values(),
        key=lambda item: (item["dn"] is None, item["dn"] or 0),
    )

    return {
        "network": {
            "id": network.id,
            "name": network.name,
            "derivation_stage": network.derivation_stage,
            "derivation_method": network.derivation_method,
        },
        "cost_basis": {
            "length_basis": length_basis,
            "route_length_method": length_stats.get("method"),
            "route_length_precision_m": length_stats.get("precision_m"),
            "pricing_mode": "cost_per_meter_override" if has_cost_override else "pipe_catalog_with_fallback",
            "fallback_cost_per_meter": fallback_cost,
            "pipe_catalog": pipe_catalog,
            "pipe_sizing_id": latest_sizing.id if latest_sizing else None,
            "pipe_sizing_name": latest_sizing.name if latest_sizing else None,
            "thermal_loss_model": "configured_linear_twin_pipe_screen",
            "thermal_loss_energy_value_cad_per_kwh": constants.electricity_cost,
            "pumping_model": hourly_profile["model"],
            "pumping_profile_year": hourly_profile.get("year"),
            "pumping_profile_source": hourly_profile.get("source"),
            "pumping_profile_hours": hourly_profile.get("hours", 0),
        },
        "metrics": {
            "node_count": network.node_set.count(),
            "edge_count": Edge.objects.filter(network=network).count(),
            "route_count": Route.objects.filter(edge__network=network).count(),
            "segment_count": len(entries),
            "sized_segment_count": sizing_count,
            "total_length_m": total_length,
            "individual_route_length_m": length_stats.get("individual_route_length_m", total_length),
            "overlap_deduped_length_m": length_stats.get("overlap_deduped_length_m", 0.0),
            "overlap_ratio": length_stats.get("overlap_ratio", 0.0),
            "construction_cost": construction_cost,
            "pumping_lifecycle_cost": pumping_lifecycle_cost,
            "annual_pumping_energy_kwh": annual_pumping_energy_kwh,
            "thermal_loss_lifecycle_cost": thermal_loss_lifecycle_cost,
            "total_lifecycle_cost": total_lifecycle_cost,
            "average_cost_per_meter": construction_cost / total_length if total_length else 0.0,
            "average_lifecycle_cost_per_meter": total_lifecycle_cost / total_length if total_length else 0.0,
            "total_thermal_loss_kw": total_thermal_loss_kw,
            "average_thermal_loss_w_per_m": total_thermal_loss_kw * 1000 / total_length if total_length else 0.0,
            "operating_temperature_c": (constants.supply_temperature_c + constants.return_temperature_c) / 2,
            "ambient_temperature_c": constants.ambient_temperature_c,
            "temperature_delta_c": (
                (constants.supply_temperature_c + constants.return_temperature_c) / 2
                - constants.ambient_temperature_c
            ),
            "average_diameter_m": diameter_stats["avg_diameter_m"] or 0.0,
            "max_diameter_m": diameter_stats["max_diameter_m"] or 0.0,
            "max_velocity_m_s": diameter_stats["max_velocity_m_s"] or 0.0,
            "total_pressure_drop_pa": total_pressure_drop_pa,
        },
        "breakdown_by_diameter": diameter_breakdown,
        "segments": segment_breakdown,
        "design_constants": _constants_payload(constants),
    }


def build_network_compare_metrics(network_ids, cost_per_meter=None):
    networks = list(Network.objects.filter(id__in=network_ids).only("id", "project_id"))
    found_ids = {network.id for network in networks}
    missing_ids = [network_id for network_id in network_ids if int(network_id) not in found_ids]
    if missing_ids:
        raise ValueError(f"Networks not found: {missing_ids}")
    project_ids = {network.project_id for network in networks}
    if len(project_ids) > 1:
        raise ValueError("Network comparison is project-scoped. Select networks from one project only.")

    summaries = []
    for network_id in network_ids:
        summaries.append(build_network_cost_summary(network_id, cost_per_meter=cost_per_meter))
    return summaries


def _network_length_entries(network, sizing_by_edge=None, sizing_rows=None):
    sizing_rows = list(sizing_rows or [])
    routes = list(
        Route.objects.filter(edge__network=network)
        .select_related("edge", "edge__start_node", "edge__end_node")
        .order_by("id")
    )
    sized_physical_segments = [
        sizing for sizing in sizing_rows
        if sizing.length_m is not None and (sizing.segment_path or sizing.source_route_ids or sizing.source_edge_ids)
    ]
    if sized_physical_segments:
        stats = route_physical_segments(routes, sizing_by_edge=sizing_by_edge or {})["stats"] if routes else {
            "method": "sized_physical_segments",
            "precision_m": None,
            "individual_route_length_m": 0.0,
            "unique_length_m": sum(float(sizing.length_m or 0.0) for sizing in sized_physical_segments),
            "overlap_deduped_length_m": 0.0,
            "overlap_ratio": 0.0,
        }
        return [
            {
                "edge": sizing.edge,
                "route_id": sizing.route_id,
                "length_m": sizing.length_m or 0.0,
                "source_length_m": sizing.route.distance_meters if sizing.route else sizing.edge.length,
                "source_route_ids": sizing.source_route_ids or ([sizing.route_id] if sizing.route_id else []),
                "source_edge_ids": sizing.source_edge_ids or [sizing.edge_id],
                "overlap_count": sizing.overlap_count or 1,
                "is_shared": sizing.is_shared_segment,
                "sizing": sizing,
            }
            for sizing in sized_physical_segments
        ], "sized_physical_route_segments", stats

    if routes:
        route_geometry = route_physical_segments(routes, sizing_by_edge=sizing_by_edge or {})
        return route_geometry["entries"], "physical_route_union", route_geometry["stats"]

    edges = list(
        Edge.objects.filter(network=network)
        .select_related("start_node", "end_node")
        .order_by("id")
    )
    entries = [
        {
            "edge": edge,
            "route_id": None,
            "length_m": edge.length,
            "source_length_m": edge.length,
            "source_route_ids": [],
            "source_edge_ids": [edge.id],
            "overlap_count": 1,
            "is_shared": False,
        }
        for edge in edges
    ]
    total_edge_length = sum(float(edge.length or 0.0) for edge in edges)
    return entries, "edges", {
        "method": "edge_length_sum",
        "precision_m": None,
        "individual_route_length_m": 0.0,
        "unique_length_m": total_edge_length,
        "overlap_deduped_length_m": 0.0,
        "overlap_ratio": 0.0,
    }


def _normal_catalog(pipe_catalog):
    if not isinstance(pipe_catalog, list):
        return []
    normal = []
    for item in pipe_catalog:
        if not isinstance(item, dict):
            continue
        try:
            normal.append({
                "dn": int(item["dn"]),
                "di": float(item["di"]),
                "cost": float(item["cost"]),
                "roughness": float(item.get("roughness", 0.000045)),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(normal, key=lambda pipe: pipe["di"])


def _catalog_pipe_for_diameter(pipe_catalog, diameter_m):
    if not pipe_catalog or not diameter_m:
        return None
    return min(pipe_catalog, key=lambda pipe: abs(pipe["di"] - float(diameter_m)))


def _pumping_lifecycle_cost(volume_flow_m3_s, pressure_drop_pa, constants):
    return _pumping_lifecycle_metrics(volume_flow_m3_s, pressure_drop_pa, constants)["lifecycle_cost"]


def _pumping_lifecycle_metrics(volume_flow_m3_s, pressure_drop_pa, constants, hourly_factors=None):
    if volume_flow_m3_s <= 0 or pressure_drop_pa <= 0 or constants.pump_efficiency <= 0:
        return {"annual_energy_kwh": 0.0, "lifecycle_cost": 0.0}
    pump_power_kw = (volume_flow_m3_s * pressure_drop_pa) / (constants.pump_efficiency * 1000)
    if hourly_factors:
        # Darcy-Weisbach pressure loss is approximately proportional to flow
        # squared, so hydraulic power follows the cube of the flow fraction.
        annual_energy_kwh = pump_power_kw * sum(float(factor) ** 3 for factor in hourly_factors)
    else:
        annual_energy_kwh = pump_power_kw * constants.hours_per_year
    annual_cost = annual_energy_kwh * constants.electricity_cost
    return {
        "annual_energy_kwh": annual_energy_kwh,
        "lifecycle_cost": annual_cost * _present_value_factor(constants.discount_rate, constants.lifespan_years),
    }


def _hourly_flow_profile(network):
    building_rows = list(
        ImportedBuildingHeating.objects.filter(building__node__network=network)
        .filter(building__heating_demand__gt=0)
        .select_related("building")
        .distinct()
    )
    demand_building_ids = set(
        network.node_set.filter(building__heating_demand__gt=0).values_list("building_id", flat=True)
    )
    if not demand_building_ids or {row.building_id for row in building_rows} != demand_building_ids:
        return {"model": "equivalent_full_load_hours_fallback", "hours": 0}

    shapes = list(ImportedHeatingShape.objects.filter(project_id=network.project_id).order_by("year"))
    candidates = []
    for shape in shapes:
        year_key = str(shape.year)
        if len(shape.hourly_weights or []) not in (8760, 8784):
            continue
        if any(year_key not in (row.annual_peak_kw or {}) for row in building_rows):
            continue
        error = sum(
            abs(float(row.building.heating_demand or 0.0) - float(row.annual_peak_kw[year_key] or 0.0))
            for row in building_rows
        )
        candidates.append((error, shape))
    if not candidates:
        return {"model": "equivalent_full_load_hours_fallback", "hours": 0}

    _, shape = min(candidates, key=lambda item: item[0])
    weights = [max(float(value), 0.0) for value in shape.hourly_weights]
    peak_weight = max(weights or [0.0])
    if peak_weight <= 0:
        return {"model": "equivalent_full_load_hours_fallback", "hours": 0}
    return {
        "model": "hourly_profile_flow_cube",
        "year": shape.year,
        "source": shape.provenance,
        "hours": len(weights),
        "flow_factors": [value / peak_weight for value in weights],
    }


def _thermal_loss_kw(length_m, constants):
    if length_m <= 0:
        return 0.0
    coefficient = max(float(constants.thermal_loss_coefficient_w_m_k or 0.0), 0.0)
    supply_delta = max(float(constants.supply_temperature_c) - float(constants.ambient_temperature_c), 0.0)
    return_delta = max(float(constants.return_temperature_c) - float(constants.ambient_temperature_c), 0.0)
    return coefficient * length_m * (supply_delta + return_delta) / 1000.0


def _thermal_loss_lifecycle_cost(thermal_loss_kw, constants):
    if thermal_loss_kw <= 0:
        return 0.0
    annual_cost = thermal_loss_kw * 8760 * max(float(constants.electricity_cost or 0.0), 0.0)
    return annual_cost * _present_value_factor(constants.discount_rate, constants.lifespan_years)


def _present_value_factor(discount_rate, years):
    years = max(int(years or 0), 0)
    rate = max(float(discount_rate or 0.0), 0.0)
    if years == 0:
        return 0.0
    return years if rate == 0 else (1 - (1 + rate) ** -years) / rate


def _float_or_default(value, default):
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _constants_payload(constants):
    return {
        "id": constants.id,
        "network": constants.network_id,
        "rho": constants.rho,
        "cp": constants.cp,
        "mu": constants.mu,
        "max_velocity": constants.max_velocity,
        "min_velocity": constants.min_velocity,
        "target_pressure_drop": constants.target_pressure_drop,
        "electricity_cost": constants.electricity_cost,
        "pump_efficiency": constants.pump_efficiency,
        "discount_rate": constants.discount_rate,
        "lifespan_years": constants.lifespan_years,
        "hours_per_year": constants.hours_per_year,
        "supply_temperature_c": constants.supply_temperature_c,
        "return_temperature_c": constants.return_temperature_c,
        "ambient_temperature_c": constants.ambient_temperature_c,
        "thermal_loss_coefficient_w_m_k": constants.thermal_loss_coefficient_w_m_k,
        "thermal_loss_coefficient_w_m_k": constants.thermal_loss_coefficient_w_m_k,
        "fallback_pipe_cost_per_meter": constants.fallback_pipe_cost_per_meter,
        "pipe_catalog": constants.pipe_catalog,
    }
