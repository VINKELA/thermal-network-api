import math

from ..models import EdgePipeSizing
from .pipe_sizing import HydraulicFormulas, RuntimeDesignCriteria

GRAVITY_M_S2 = 9.80665


def _to_float(value, default=0.0):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_round(value, digits=3):
    if value is None or not math.isfinite(value):
        return None
    return round(value, digits)


def _catalog_for(criteria):
    pipes = criteria.STD_PIPES or []
    normalized = []
    for pipe in pipes:
        diameter = _to_float(pipe.get("di"))
        if diameter <= 0:
            continue
        normalized.append({
            "dn": int(_to_float(pipe.get("dn"), diameter * 1000)),
            "diameter_m": diameter,
            "roughness_m": _to_float(pipe.get("roughness"), 0.000045),
            "cost_per_m": _to_float(pipe.get("cost")),
        })
    return sorted(normalized, key=lambda item: item["diameter_m"])


def _nearest_catalog_pipe(catalog, diameter_m):
    if not catalog:
        return None
    return min(catalog, key=lambda pipe: abs(pipe["diameter_m"] - diameter_m))


def _status_for_velocity(velocity, criteria):
    if velocity > criteria.MAX_VELOCITY:
        return "high"
    if velocity < criteria.MIN_VELOCITY and velocity > 0:
        return "low"
    return "target"


def _status_for_pressure_drop(pressure_drop_pa_per_m, criteria):
    if pressure_drop_pa_per_m > criteria.TARGET_PRESSURE_DROP:
        return "high"
    return "target"


def _evaluate_section(
    *,
    label,
    criteria,
    length_m,
    diameter_m,
    peak_load_kw,
    delta_t_c,
    mass_flow_kg_s=None,
    stored_pressure_drop_pa=None,
):
    diameter_m = max(_to_float(diameter_m), 0.0)
    length_m = max(_to_float(length_m), 0.0)
    peak_load_kw = max(_to_float(peak_load_kw), 0.0)
    delta_t_c = max(_to_float(delta_t_c), 0.0)

    if mass_flow_kg_s is None:
        mass_flow_kg_s = HydraulicFormulas.calculate_mass_flow(peak_load_kw, delta_t_c, criteria)
    mass_flow_kg_s = max(_to_float(mass_flow_kg_s), 0.0)

    volume_flow_m3_s = mass_flow_kg_s / criteria.RHO if criteria.RHO else 0.0
    area_m2 = (math.pi * diameter_m * diameter_m / 4) if diameter_m else 0.0
    velocity_m_s = (volume_flow_m3_s / area_m2) if area_m2 else 0.0
    reynolds_number = HydraulicFormulas.calculate_reynolds(velocity_m_s, diameter_m, criteria) if velocity_m_s and diameter_m else 0.0
    friction_factor = HydraulicFormulas.calculate_friction_factor(reynolds_number, diameter_m) if reynolds_number and diameter_m else 0.0
    pressure_drop_pa = HydraulicFormulas.calculate_head_loss_pa(velocity_m_s, diameter_m, length_m, criteria)
    if stored_pressure_drop_pa is not None:
        pressure_drop_pa = max(_to_float(stored_pressure_drop_pa), 0.0)

    pressure_drop_pa_per_m = pressure_drop_pa / length_m if length_m else 0.0
    head_loss_m = pressure_drop_pa / (criteria.RHO * GRAVITY_M_S2) if criteria.RHO else 0.0
    pump_efficiency = max(_to_float(criteria.PUMP_EFFICIENCY), 0.0)
    pumping_power_kw = (volume_flow_m3_s * pressure_drop_pa) / (pump_efficiency * 1000) if pump_efficiency else 0.0
    required_delta_t_c = peak_load_kw / (mass_flow_kg_s * criteria.CP) if mass_flow_kg_s and criteria.CP else delta_t_c
    required_supply_temperature_c = criteria.RETURN_TEMPERATURE_C + required_delta_t_c
    served_load_kw = mass_flow_kg_s * criteria.CP * delta_t_c

    return {
        "label": label,
        "peak_load_kw": _safe_round(peak_load_kw, 3),
        "served_load_kw": _safe_round(served_load_kw, 3),
        "mass_flow_kg_s": _safe_round(mass_flow_kg_s, 5),
        "volume_flow_m3_s": _safe_round(volume_flow_m3_s, 7),
        "volume_flow_m3_h": _safe_round(volume_flow_m3_s * 3600, 4),
        "diameter_m": _safe_round(diameter_m, 5),
        "diameter_mm": _safe_round(diameter_m * 1000, 2),
        "velocity_m_s": _safe_round(velocity_m_s, 5),
        "pressure_drop_pa": _safe_round(pressure_drop_pa, 3),
        "pressure_drop_kpa": _safe_round(pressure_drop_pa / 1000, 5),
        "pressure_drop_pa_per_m": _safe_round(pressure_drop_pa_per_m, 5),
        "head_loss_m": _safe_round(head_loss_m, 5),
        "pumping_power_kw": _safe_round(pumping_power_kw, 5),
        "delta_t_c": _safe_round(delta_t_c, 3),
        "return_temperature_c": _safe_round(criteria.RETURN_TEMPERATURE_C, 3),
        "required_delta_t_c": _safe_round(required_delta_t_c, 3),
        "required_supply_temperature_c": _safe_round(required_supply_temperature_c, 3),
        "configured_supply_temperature_c": _safe_round(criteria.SUPPLY_TEMPERATURE_C, 3),
        "reynolds_number": _safe_round(reynolds_number, 0),
        "friction_factor": _safe_round(friction_factor, 6),
        "velocity_status": _status_for_velocity(velocity_m_s, criteria),
        "pressure_drop_status": _status_for_pressure_drop(pressure_drop_pa_per_m, criteria),
        "supply_temperature_status": "shortfall" if required_supply_temperature_c > criteria.SUPPLY_TEMPERATURE_C else "met",
    }


def _temperature_steps(current_delta_t):
    base = [10, 15, 20, 25, 30, 40, 50, 60]
    current = round(max(_to_float(current_delta_t), 1), 1)
    return sorted({step for step in base if step > 0} | {current})


def build_edge_pipe_hydraulic_analysis(edge_pipe_sizing_id):
    sizing = EdgePipeSizing.objects.select_related(
        "pipe_sizing",
        "pipe_sizing__network",
        "edge",
        "edge__start_node",
        "edge__end_node",
        "route",
    ).get(id=edge_pipe_sizing_id)

    network_id = sizing.pipe_sizing.network_id or sizing.edge.network_id
    criteria = RuntimeDesignCriteria(network_id)
    catalog = _catalog_for(criteria)
    diameter_m = max(_to_float(sizing.diameter_m), 0.02)
    catalog_pipe = _nearest_catalog_pipe(catalog, diameter_m)
    nominal_dn = catalog_pipe["dn"] if catalog_pipe else round(diameter_m * 1000)
    length_m = max(
        _to_float(sizing.length_m)
        or _to_float(getattr(sizing.route, "distance_meters", None))
        or _to_float(getattr(sizing.edge, "length", None)),
        0.0,
    )
    peak_load_kw = max(_to_float(sizing.peak_load_kw), 0.0)
    mass_flow_kg_s = max(_to_float(sizing.mass_flow_kg_s), 0.0)
    delta_t_c = max(_to_float(sizing.delta_t), criteria.SUPPLY_TEMPERATURE_C - criteria.RETURN_TEMPERATURE_C)

    baseline = _evaluate_section(
        label="Design",
        criteria=criteria,
        length_m=length_m,
        diameter_m=diameter_m,
        peak_load_kw=peak_load_kw,
        delta_t_c=delta_t_c,
        mass_flow_kg_s=mass_flow_kg_s,
        stored_pressure_drop_pa=sizing.pressure_drop_pa,
    )
    baseline["nominal_dn"] = nominal_dn

    flow_variation = []
    for factor in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        flow = mass_flow_kg_s * factor
        flow_variation.append(_evaluate_section(
            label=f"{int(factor * 100)}% flow",
            criteria=criteria,
            length_m=length_m,
            diameter_m=diameter_m,
            peak_load_kw=flow * criteria.CP * delta_t_c,
            delta_t_c=delta_t_c,
            mass_flow_kg_s=flow,
        ))

    diameter_variation = []
    for pipe in catalog:
        diameter_variation.append(_evaluate_section(
            label=f"DN{pipe['dn']}",
            criteria=criteria,
            length_m=length_m,
            diameter_m=pipe["diameter_m"],
            peak_load_kw=peak_load_kw,
            delta_t_c=delta_t_c,
            mass_flow_kg_s=mass_flow_kg_s,
        ))

    temperature_variation = []
    for delta_t in _temperature_steps(delta_t_c):
        temperature_variation.append(_evaluate_section(
            label=f"{delta_t:g} C delta T",
            criteria=criteria,
            length_m=length_m,
            diameter_m=diameter_m,
            peak_load_kw=peak_load_kw,
            delta_t_c=delta_t,
        ))

    current_index = next(
        (index for index, pipe in enumerate(catalog) if pipe["dn"] == nominal_dn),
        max(0, len(catalog) // 2),
    )
    next_pipe = catalog[min(current_index + 1, len(catalog) - 1)] if catalog else {"diameter_m": diameter_m, "dn": nominal_dn}
    smaller_pipe = catalog[max(current_index - 1, 0)] if catalog else {"diameter_m": diameter_m, "dn": nominal_dn}
    combined_variation = [
        _evaluate_section(
            label="Low load",
            criteria=criteria,
            length_m=length_m,
            diameter_m=diameter_m,
            peak_load_kw=peak_load_kw * 0.75,
            delta_t_c=delta_t_c,
        ),
        baseline,
        _evaluate_section(
            label="High load",
            criteria=criteria,
            length_m=length_m,
            diameter_m=diameter_m,
            peak_load_kw=peak_load_kw * 1.25,
            delta_t_c=delta_t_c,
        ),
        _evaluate_section(
            label="High load, low delta T",
            criteria=criteria,
            length_m=length_m,
            diameter_m=diameter_m,
            peak_load_kw=peak_load_kw * 1.25,
            delta_t_c=max(delta_t_c * 0.75, 1),
        ),
        _evaluate_section(
            label=f"High load, DN{next_pipe['dn']}",
            criteria=criteria,
            length_m=length_m,
            diameter_m=next_pipe["diameter_m"],
            peak_load_kw=peak_load_kw * 1.25,
            delta_t_c=max(delta_t_c * 0.75, 1),
        ),
        _evaluate_section(
            label=f"Tight pipe DN{smaller_pipe['dn']}",
            criteria=criteria,
            length_m=length_m,
            diameter_m=smaller_pipe["diameter_m"],
            peak_load_kw=peak_load_kw,
            delta_t_c=delta_t_c,
        ),
    ]

    return {
        "id": sizing.id,
        "segment_name": sizing.edge.name or f"Pipe section {sizing.id}",
        "pipe_sizing_id": sizing.pipe_sizing_id,
        "network_id": network_id,
        "length_m": _safe_round(length_m, 3),
        "is_shared_segment": sizing.is_shared_segment,
        "overlap_count": sizing.overlap_count,
        "source_route_ids": sizing.source_route_ids,
        "source_edge_ids": sizing.source_edge_ids,
        "constants": {
            "rho_kg_m3": _safe_round(criteria.RHO, 3),
            "cp_kj_kg_k": _safe_round(criteria.CP, 4),
            "mu_pa_s": _safe_round(criteria.MU, 8),
            "min_velocity_m_s": _safe_round(criteria.MIN_VELOCITY, 3),
            "max_velocity_m_s": _safe_round(criteria.MAX_VELOCITY, 3),
            "target_pressure_drop_pa_m": _safe_round(criteria.TARGET_PRESSURE_DROP, 3),
            "pump_efficiency": _safe_round(criteria.PUMP_EFFICIENCY, 4),
            "supply_temperature_c": _safe_round(criteria.SUPPLY_TEMPERATURE_C, 3),
            "return_temperature_c": _safe_round(criteria.RETURN_TEMPERATURE_C, 3),
        },
        "baseline": baseline,
        "variations": {
            "flow": flow_variation,
            "diameter": diameter_variation,
            "temperature": temperature_variation,
            "combined": combined_variation,
        },
    }
