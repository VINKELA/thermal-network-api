import copy
import json
import math
from collections import defaultdict
from typing import Iterable

from django.db import transaction
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform as shapely_transform

from api.models import (
    Building,
    Edge,
    EdgeTranslation,
    Network,
    NetworkExpansionAttachment,
    NetworkExpansionRun,
    Node,
    Route,
    assert_network_project_integrity,
)
from api.utils.network_versioning import (
    build_route_data,
    clone_network,
    decode_polyline,
    encode_polyline,
)
from api.utils.route_geometry import unique_route_length


DEFAULT_CANDIDATE_SCORE_WEIGHTS = {
    "linear_heat_density": 0.35,
    "heat_demand": 0.18,
    "distance_to_route": 0.17,
    "connection_cost": 0.12,
    "available_heat_fit": 0.12,
    "load_diversity": 0.06,
}


def normalize_route_infill_params(raw_params):
    params = dict(raw_params or {})
    params["max_distance_m"] = _float_value(params.get("max_distance_m"), 100.0)
    params["min_heat_kw"] = _float_value(params.get("min_heat_kw"), None)
    params["limit"] = _int_value(params.get("limit"), 500)
    params["building_ids"] = _list_value(params.get("building_ids"))
    params["tags"] = _list_value(params.get("tags") or params.get("building_tags") or params.get("tag"))
    params["fsas"] = _list_value(params.get("fsas") or params.get("building_fsas") or params.get("fsa"))
    params["building_types"] = _list_value(params.get("building_types") or params.get("building_type"))
    params["source_edge_translation_id"] = _int_value(
        params.get("source_edge_translation_id") or params.get("edge_translation_id"),
        None,
    )
    params["score_weights"] = _score_weights(params.get("score_weights"))
    params["annual_full_load_hours"] = _float_value(params.get("annual_full_load_hours"), None)
    params["pipe_cost_per_m"] = _float_value(params.get("pipe_cost_per_m"), None)
    return params


def preview_route_infill(source_network, raw_params=None, *, include_rejected=True):
    params = normalize_route_infill_params(raw_params)
    source_network = Network.objects.select_related("project").get(pk=source_network.pk)
    assert_network_project_integrity(source_network)

    routes = _source_routes(source_network, params.get("source_edge_translation_id"))
    if not routes:
        raise ValueError("The selected network has no route geometry to expand from.")

    route_index = _build_route_index(routes)
    score_context = _candidate_score_context(source_network, params)
    candidates = []
    for building in _candidate_buildings(source_network, params):
        nearest = _nearest_route(building, route_index)
        if not nearest:
            continue

        accepted = nearest["distance_to_route_m"] <= params["max_distance_m"]
        if accepted or include_rejected:
            candidates.append(_candidate_payload(building, nearest, accepted, params["max_distance_m"]))

    score_summary = _attach_candidate_scores(candidates, score_context, params)
    candidates.sort(
        key=lambda item: (
            not item["accepted"],
            -(item.get("suitability_score") or 0),
            item["distance_to_route_m"],
            item["building"]["id"],
        )
    )
    limit = params.get("limit")
    if limit and include_rejected:
        candidates = candidates[:limit]

    accepted_candidates = [candidate for candidate in candidates if candidate["accepted"]]
    return {
        "source_network": _network_payload(source_network),
        "source_edge_translation_id": params.get("source_edge_translation_id"),
        "max_distance_m": params["max_distance_m"],
        "routes": [_route_payload(route) for route in routes],
        "candidates": candidates,
        "score_summary": score_summary,
        "summary": {
            "candidate_count": len(candidates),
            "accepted_count": len(accepted_candidates),
            "rejected_count": len(candidates) - len(accepted_candidates),
            "added_heat_kw": sum(_float_value(candidate["building"].get("heating_demand"), 0.0) for candidate in accepted_candidates),
            "added_service_length_m": sum(candidate["distance_to_route_m"] for candidate in accepted_candidates),
        },
    }


def create_route_infill_network(source_network, raw_params=None):
    params = normalize_route_infill_params(raw_params)
    preview = preview_route_infill(source_network, params, include_rejected=False)
    accepted = [candidate for candidate in preview["candidates"] if candidate["accepted"]]
    if not accepted:
        raise ValueError("No buildings are close enough to the existing routes for the selected distance.")

    source_network = Network.objects.select_related("project").get(pk=source_network.pk)
    source_routes = _source_routes(source_network, params.get("source_edge_translation_id"))
    source_translation = _single_translation(source_routes)
    output_name = params.get("name") or f"{source_network.name} - Route Infill"

    clone_result = None
    run = None
    try:
        clone_result = clone_network(
            source_network,
            name=output_name,
            stage=Network.DerivationStage.ROUTE_INFILL,
            method="route_infill",
            parameters={
                "source_network_id": source_network.id,
                "source_edge_translation_id": params.get("source_edge_translation_id"),
                "max_distance_m": params["max_distance_m"],
                "tags": params["tags"],
                "fsas": params["fsas"],
                "building_types": params["building_types"],
                "building_ids": params["building_ids"],
                "min_heat_kw": params["min_heat_kw"],
                "score_weights": params["score_weights"],
                "candidate_scoring": {
                    "method": "pre_optimization_suitability_score",
                    "annual_full_load_hours": preview.get("score_summary", {}).get("assumptions", {}).get("annual_full_load_hours"),
                    "pipe_cost_per_m": preview.get("score_summary", {}).get("assumptions", {}).get("pipe_cost_per_m"),
                },
            },
            include_edges=True,
            include_connections=False,
            include_routes=False,
            include_pipe_sizing=False,
        )

        route_map = _clone_source_routes(source_routes, clone_result.edge_map)
        run = NetworkExpansionRun.objects.create(
            source_network=source_network,
            output_network=clone_result.network,
            source_edge_translation=source_translation,
            name=output_name,
            max_distance_m=params["max_distance_m"],
            parameters=copy.deepcopy(clone_result.network.derivation_parameters or {}),
            candidate_count=len(preview["candidates"]),
            attached_count=len(accepted),
            rejected_count=max(0, len(preview["candidates"]) - len(accepted)),
            added_heat_kw=preview["summary"]["added_heat_kw"],
            added_service_length_m=preview["summary"]["added_service_length_m"],
        )

        source_route_lookup = {route.id: route for route in source_routes}
        tap_items_by_source_route = {}
        actual_added_service_length = 0.0
        for candidate in accepted:
            building = Building.objects.get(pk=candidate["building"]["id"])
            source_route = source_route_lookup.get(candidate["source_route_id"])
            if not source_route:
                continue
            cloned_route = route_map.get(source_route.id)
            cloned_edge = cloned_route.edge if cloned_route else clone_result.edge_map.get(source_route.edge_id)
            if not cloned_edge:
                continue

            service_length = float(candidate["distance_to_route_m"] or 0.0)
            route_mode = _route_travel_mode(source_route)
            tap_coords = (candidate["snap_point"]["lat"], candidate["snap_point"]["lng"])
            building_coords = (building.latitude, building.longitude)
            route_data = build_route_data(
                tap_coords,
                building_coords,
                route_mode,
                use_google=True,
            )
            service_length = float(route_data.get("distance") or service_length)

            with transaction.atomic():
                tap_node = Node.objects.create(
                    network=clone_result.network,
                    name=f"Tap for {building.name or building.id}",
                    latitude=candidate["snap_point"]["lat"],
                    longitude=candidate["snap_point"]["lng"],
                    node_type="junction",
                    other_information={
                        "is_route_infill_tap": True,
                        "source_route_id": source_route.id,
                        "source_edge_id": source_route.edge_id,
                        "building_id": building.id,
                    },
                )
                building_node = Node.objects.create(
                    network=clone_result.network,
                    name=building.name or f"Building {building.id}",
                    latitude=building.latitude,
                    longitude=building.longitude,
                    node_type=_node_type_for_building(building),
                    building=building,
                    other_information={
                        "added_by_route_infill": True,
                        "network_expansion_run_id": run.id,
                        "distance_to_route_m": candidate["distance_to_route_m"],
                        "source_route_id": source_route.id,
                    },
                )

                service_edge = Edge.objects.create(
                    network=clone_result.network,
                    edge_algorithm=None,
                    name=f"{building.name or building.id} service connection",
                    start_node=tap_node,
                    end_node=building_node,
                    length=service_length,
                    is_default=False,
                    other_information={
                        "route_infill_service": True,
                        "network_expansion_run_id": run.id,
                        "source_route_id": source_route.id,
                        "source_edge_id": source_route.edge_id,
                        "route_travel_mode": route_mode,
                        "route_requested_mode": route_mode,
                        "route_resolved_mode": route_data.get("mode") or route_mode,
                        "straight_distance_to_route_m": candidate["distance_to_route_m"],
                        "routed_distance_m": service_length,
                    },
                )
                service_route = Route.objects.create(
                    edge=service_edge,
                    translation=None,
                    route_type="service",
                    polyline=route_data["polyline"],
                    distance_meters=route_data["distance"],
                    duration_seconds=route_data.get("duration") or 0,
                )
                NetworkExpansionAttachment.objects.create(
                    run=run,
                    building=building,
                    new_node=building_node,
                    tap_node=tap_node,
                    source_route=source_route,
                    source_edge=source_route.edge,
                    service_edge=service_edge,
                    service_route=service_route,
                    status="attached",
                    reason="Within selected route distance",
                    distance_to_route_m=service_length,
                    snap_latitude=tap_node.latitude,
                    snap_longitude=tap_node.longitude,
                )

            actual_added_service_length += service_length
            tap_items_by_source_route.setdefault(source_route.id, []).append({
                "tap_node": tap_node,
                "source_route": source_route,
                "along_m": candidate.get("along_m"),
            })

        _split_routes_at_infill_taps(route_map, tap_items_by_source_route)
        run.added_service_length_m = actual_added_service_length
        run.save(update_fields=["added_service_length_m"])

        routes_for_length = Route.objects.filter(edge__network=clone_result.network).select_related("edge")
        clone_result.network.total_length = unique_route_length(routes_for_length)
        clone_result.network.save(update_fields=["total_length", "updated_at"])
    except Exception:
        if clone_result and clone_result.network and clone_result.network.id:
            Network.objects.filter(pk=clone_result.network.id).delete()
        raise

    preview = copy.deepcopy(preview)
    preview["summary"]["added_service_length_m"] = run.added_service_length_m
    preview["summary"]["routed_added_service_length_m"] = run.added_service_length_m
    return {
        **preview,
        "status": "success",
        "message": f"Created expanded network '{clone_result.network.name}' with {len(accepted)} added buildings.",
        "network_id": clone_result.network.id,
        "expansion_run_id": run.id,
        "output_network": _network_payload(clone_result.network),
    }


def _source_routes(network, source_edge_translation_id=None):
    queryset = Route.objects.filter(edge__network=network).select_related(
        "edge",
        "edge__start_node",
        "edge__end_node",
        "translation",
    )
    if source_edge_translation_id:
        queryset = queryset.filter(translation_id=source_edge_translation_id)
    return list(queryset.order_by("id"))


def _candidate_buildings(network, params):
    existing_building_ids = Node.objects.filter(
        network=network,
        building__isnull=False,
    ).values_list("building_id", flat=True)
    queryset = Building.objects.filter(project=network.project).exclude(id__in=existing_building_ids)
    queryset = queryset.exclude(latitude__isnull=True).exclude(longitude__isnull=True)

    if params["building_ids"]:
        queryset = queryset.filter(id__in=params["building_ids"])
    if params["tags"]:
        queryset = queryset.filter(tag__in=params["tags"])
    if params["fsas"]:
        queryset = queryset.filter(fsa__in=params["fsas"])
    if params["building_types"]:
        queryset = queryset.filter(building_type__in=params["building_types"])
    if params["min_heat_kw"] is not None:
        queryset = queryset.filter(heating_demand__gte=params["min_heat_kw"])

    return queryset.order_by("id")


def _build_route_index(routes):
    all_points = []
    route_points = []
    for route in routes:
        points = _anchored_route_points(route)
        if len(points) < 2:
            continue
        route_points.append((route, points))
        all_points.extend(points)

    if not all_points:
        raise ValueError("The selected network has route records, but none have usable geometry.")

    centroid_lat = sum(point[0] for point in all_points) / len(all_points)
    centroid_lng = sum(point[1] for point in all_points) / len(all_points)
    epsg = _utm_epsg(centroid_lng, centroid_lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inverse_transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    items = []
    for route, points in route_points:
        line = LineString([(lng, lat) for lat, lng in points])
        projected = shapely_transform(transformer.transform, line)
        if projected.is_empty or projected.length <= 0:
            continue
        items.append({"route": route, "line": projected, "points": points})
    if not items:
        raise ValueError("The selected network has route records, but none have usable projected geometry.")
    return {
        "items": items,
        "transformer": transformer,
        "inverse_transformer": inverse_transformer,
    }


def _nearest_route(building, route_index):
    point = Point(float(building.longitude), float(building.latitude))
    projected_point = shapely_transform(route_index["transformer"].transform, point)
    best = None
    for item in route_index["items"]:
        distance = float(item["line"].distance(projected_point))
        along = item["line"].project(projected_point)
        snap = item["line"].interpolate(along)
        if best is None or distance < best["distance_to_route_m"]:
            lng, lat = route_index["inverse_transformer"].transform(snap.x, snap.y)
            best = {
                "route": item["route"],
                "distance_to_route_m": distance,
                "along_m": float(along),
                "snap_point": {"lat": float(lat), "lng": float(lng)},
            }
    return best


def _candidate_payload(building, nearest, accepted, max_distance_m):
    route = nearest["route"]
    return {
        "building": {
            "id": building.id,
            "name": building.name,
            "tag": building.tag,
            "building_type": building.building_type,
            "fsa": building.fsa,
            "latitude": building.latitude,
            "longitude": building.longitude,
            "heating_demand": building.heating_demand,
            "cooling_demand": building.cooling_demand,
        },
        "accepted": bool(accepted),
        "status": "accepted" if accepted else "rejected",
        "reason": "Within selected route distance" if accepted else f"Farther than {max_distance_m:.0f} m from route",
        "distance_to_route_m": nearest["distance_to_route_m"],
        "along_m": nearest.get("along_m"),
        "snap_point": nearest["snap_point"],
        "source_route_id": route.id,
        "source_edge_id": route.edge_id,
        "source_route_type": route.route_type,
    }


def _candidate_score_context(network, params):
    constants = _network_design_constants(network)
    annual_full_load_hours = (
        params.get("annual_full_load_hours")
        or getattr(constants, "hours_per_year", None)
        or 2000.0
    )
    pipe_cost_per_m = (
        params.get("pipe_cost_per_m")
        or getattr(constants, "fallback_pipe_cost_per_meter", None)
        or 150.0
    )

    inventory = _network_heat_inventory(network)
    source_heat_kw = _float_value(getattr(inventory, "total_waste_heat_kw", None), None)
    served_heat_kw = _float_value(getattr(inventory, "total_heating_load_kw", None), None)
    building_type_counts = defaultdict(int)
    existing_building_count = 0

    inferred_source_kw = 0.0
    inferred_served_kw = 0.0
    for node in Node.objects.filter(network=network).select_related("building"):
        building = node.building
        if building:
            existing_building_count += 1
            building_type_counts[building.building_type or "other"] += 1

        node_kind = (node.node_type or "").lower()
        is_source = node_kind == "producer" or bool(getattr(building, "waste_heat_source_type", None))
        heat_kw = _float_value(getattr(building, "heating_demand", None), 0.0) if building else 0.0
        cooling_kw = _float_value(getattr(building, "cooling_demand", None), 0.0) if building else 0.0
        if is_source:
            inferred_source_kw += max(cooling_kw, heat_kw, 0.0)
        elif building and node_kind in {"consumer", "building", "cluster"}:
            inferred_served_kw += heat_kw

    if source_heat_kw is None:
        source_heat_kw = inferred_source_kw
    if served_heat_kw is None:
        served_heat_kw = inferred_served_kw

    source_heat_kw = max(_float_value(source_heat_kw, 0.0), 0.0)
    served_heat_kw = max(_float_value(served_heat_kw, 0.0), 0.0)
    available_heat_surplus_kw = max(source_heat_kw - served_heat_kw, 0.0)
    has_heat_inventory = source_heat_kw > 0 or served_heat_kw > 0

    return {
        "annual_full_load_hours": max(_float_value(annual_full_load_hours, 2000.0), 1.0),
        "pipe_cost_per_m": max(_float_value(pipe_cost_per_m, 150.0), 0.0),
        "source_heat_kw": source_heat_kw,
        "served_heat_kw": served_heat_kw,
        "available_heat_surplus_kw": available_heat_surplus_kw,
        "has_heat_inventory": has_heat_inventory,
        "building_type_counts": dict(building_type_counts),
        "existing_building_count": existing_building_count,
    }


def _attach_candidate_scores(candidates, context, params):
    if not candidates:
        return _score_summary_payload([], context, params)

    for candidate in candidates:
        building = candidate.get("building") or {}
        heat_kw = max(_float_value(building.get("heating_demand"), 0.0), 0.0)
        distance_m = max(_float_value(candidate.get("distance_to_route_m"), 0.0), 0.0)
        scoring_length_m = max(distance_m, 1.0)
        annual_heat_mwh = heat_kw * context["annual_full_load_hours"] / 1000.0
        linear_heat_density = annual_heat_mwh / scoring_length_m if scoring_length_m else 0.0
        estimated_cost = distance_m * context["pipe_cost_per_m"]
        availability_fit = _heat_availability_fit(heat_kw, context)
        diversity = _load_diversity_factor(building.get("building_type"), context)
        candidate["candidate_metrics"] = {
            "annual_heat_mwh": annual_heat_mwh,
            "linear_heat_density_mwh_per_m_year": linear_heat_density,
            "estimated_connection_cost_cad": estimated_cost,
            "available_heat_surplus_kw": context["available_heat_surplus_kw"],
            "available_heat_fit": availability_fit,
            "load_diversity_factor": diversity,
        }
        candidate["annual_heat_mwh"] = annual_heat_mwh
        candidate["linear_heat_density_mwh_per_m_year"] = linear_heat_density
        candidate["estimated_connection_cost_cad"] = estimated_cost
        candidate["available_heat_fit"] = availability_fit
        candidate["load_diversity_factor"] = diversity

    demand_scores = _normalized_map(candidates, lambda item: item["building"].get("heating_demand"), higher_is_better=True, log_scale=True)
    density_scores = _normalized_map(candidates, lambda item: item.get("linear_heat_density_mwh_per_m_year"), higher_is_better=True, log_scale=True)
    cost_scores = _normalized_map(candidates, lambda item: item.get("estimated_connection_cost_cad"), higher_is_better=False, log_scale=True)
    weights = params["score_weights"]
    max_distance_m = max(_float_value(params.get("max_distance_m"), 100.0), 1.0)

    for index, candidate in enumerate(candidates):
        distance_m = max(_float_value(candidate.get("distance_to_route_m"), 0.0), 0.0)
        distance_score = max(0.0, min(1.0, 1.0 - (distance_m / max_distance_m)))
        components = {
            "linear_heat_density": density_scores[index],
            "heat_demand": demand_scores[index],
            "distance_to_route": distance_score,
            "connection_cost": cost_scores[index],
            "available_heat_fit": _clamp01(candidate.get("available_heat_fit")),
            "load_diversity": _clamp01(candidate.get("load_diversity_factor")),
        }
        weighted_score = sum(components[key] * weights.get(key, 0.0) for key in DEFAULT_CANDIDATE_SCORE_WEIGHTS)
        score = round(weighted_score * 100.0, 1)
        candidate["suitability_score"] = score
        candidate["suitability_label"] = _score_label(score)
        candidate["score_breakdown"] = [
            {
                "key": key,
                "label": _score_component_label(key),
                "weight": round(weights.get(key, 0.0), 3),
                "score": round(value * 100.0, 1),
            }
            for key, value in components.items()
        ]
        candidate["score_rationale"] = _score_rationale(candidate, context)

    ranked = sorted(candidates, key=lambda item: (-(item.get("suitability_score") or 0), item["distance_to_route_m"], item["building"]["id"]))
    for rank, candidate in enumerate(ranked, start=1):
        candidate["suitability_rank"] = rank

    return _score_summary_payload(candidates, context, params)


def _score_summary_payload(candidates, context, params):
    scores = [candidate.get("suitability_score") for candidate in candidates if candidate.get("suitability_score") is not None]
    densities = [
        candidate.get("linear_heat_density_mwh_per_m_year")
        for candidate in candidates
        if candidate.get("linear_heat_density_mwh_per_m_year") is not None
    ]
    top_candidate = None
    if candidates:
        top_candidate = max(candidates, key=lambda item: item.get("suitability_score") or 0)

    return {
        "method": "pre_optimization_suitability_score",
        "description": "Analysis-only ranking used to explain which candidates look promising before optimization.",
        "weights": params.get("score_weights") or DEFAULT_CANDIDATE_SCORE_WEIGHTS,
        "assumptions": {
            "annual_full_load_hours": context.get("annual_full_load_hours"),
            "pipe_cost_per_m": context.get("pipe_cost_per_m"),
            "source_heat_kw": context.get("source_heat_kw"),
            "served_heat_kw": context.get("served_heat_kw"),
            "available_heat_surplus_kw": context.get("available_heat_surplus_kw"),
            "has_heat_inventory": context.get("has_heat_inventory"),
        },
        "average_score": round(sum(scores) / len(scores), 1) if scores else 0.0,
        "average_linear_heat_density_mwh_per_m_year": round(sum(densities) / len(densities), 4) if densities else 0.0,
        "top_candidate": {
            "building_id": top_candidate["building"]["id"],
            "building_name": top_candidate["building"].get("name"),
            "score": top_candidate.get("suitability_score"),
            "linear_heat_density_mwh_per_m_year": top_candidate.get("linear_heat_density_mwh_per_m_year"),
        } if top_candidate else None,
    }


def _heat_availability_fit(heat_kw, context):
    heat_kw = max(_float_value(heat_kw, 0.0), 0.0)
    if heat_kw <= 0:
        return 0.0
    if not context.get("has_heat_inventory"):
        return 0.5
    surplus_kw = max(_float_value(context.get("available_heat_surplus_kw"), 0.0), 0.0)
    return max(0.0, min(1.0, surplus_kw / heat_kw))


def _load_diversity_factor(building_type, context):
    total = context.get("existing_building_count") or 0
    if total <= 0:
        return 0.5
    counts = context.get("building_type_counts") or {}
    share = counts.get(building_type or "other", 0) / total
    return max(0.0, min(1.0, 1.0 - share))


def _score_rationale(candidate, context):
    demand = _float_value(candidate["building"].get("heating_demand"), 0.0)
    density = _float_value(candidate.get("linear_heat_density_mwh_per_m_year"), 0.0)
    distance = _float_value(candidate.get("distance_to_route_m"), 0.0)
    availability = candidate.get("available_heat_fit")
    rationale = []
    if density >= 1.0:
        rationale.append("High heat density for the added service length.")
    elif density > 0:
        rationale.append("Lower heat density; connection length is large relative to demand.")
    if demand > 0:
        rationale.append(f"Adds {demand:.0f} kW of heat demand.")
    if distance <= 30:
        rationale.append("Very close to an existing route.")
    elif distance <= 100:
        rationale.append("Moderate distance from an existing route.")
    if context.get("has_heat_inventory"):
        if availability >= 1:
            rationale.append("Current heat surplus can cover this candidate in isolation.")
        elif availability > 0:
            rationale.append("Current heat surplus covers only part of this candidate in isolation.")
        else:
            rationale.append("No spare heat capacity is visible in the selected network data.")
    else:
        rationale.append("Heat availability is neutral because source capacity data is incomplete.")
    return rationale


def _normalized_map(items, value_getter, *, higher_is_better=True, log_scale=False):
    values = []
    for item in items:
        value = _float_value(value_getter(item), 0.0)
        value = max(value, 0.0)
        values.append(math.log1p(value) if log_scale else value)

    if not values:
        return []
    min_value = min(values)
    max_value = max(values)
    if math.isclose(max_value, min_value):
        constant_score = 1.0 if max_value > 0 else 0.0
        return [constant_score for _ in values]

    normalized = [(value - min_value) / (max_value - min_value) for value in values]
    if not higher_is_better:
        normalized = [1.0 - value for value in normalized]
    return [max(0.0, min(1.0, value)) for value in normalized]


def _score_label(score):
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Strong"
    if score >= 45:
        return "Moderate"
    return "Weak"


def _score_component_label(key):
    return {
        "linear_heat_density": "Linear heat density",
        "heat_demand": "Heat demand",
        "distance_to_route": "Distance to route",
        "connection_cost": "Connection cost",
        "available_heat_fit": "Available heat fit",
        "load_diversity": "Load diversity",
    }.get(key, key)


def _network_design_constants(network):
    try:
        return network.design_constants
    except Exception:
        return None


def _network_heat_inventory(network):
    try:
        return network.heat_inventory
    except Exception:
        return None


def _route_payload(route):
    return {
        "id": route.id,
        "route_type": route.route_type,
        "distance_meters": route.distance_meters,
        "source_edge_id": route.edge_id,
        "path": [{"lat": lat, "lng": lng} for lat, lng in _anchored_route_points(route)],
    }


def _anchored_route_points(route):
    start = (route.edge.start_node.latitude, route.edge.start_node.longitude)
    end = (route.edge.end_node.latitude, route.edge.end_node.longitude)
    existing = {
        "polyline": route.polyline,
        "distance": route.distance_meters,
        "duration": route.duration_seconds,
    }
    route_data = build_route_data(start, end, _route_travel_mode(route), use_google=False, existing_route_data=existing)
    return decode_polyline(route_data["polyline"])


def _route_travel_mode(route):
    if route.translation and route.translation.translation:
        return route.translation.translation

    edge_info = copy.deepcopy(getattr(route.edge, "other_information", None)) or {}
    for key in ("route_requested_mode", "route_travel_mode", "route_resolved_mode", "travel_mode"):
        mode = edge_info.get(key)
        if mode and mode != "direct_fallback":
            return mode

    source_route_id = edge_info.get("source_route_id")
    try:
        source_route_id = int(source_route_id) if source_route_id else None
    except (TypeError, ValueError):
        source_route_id = None
    if source_route_id and source_route_id != int(route.id):
        source_route = (
            Route.objects
            .filter(id=source_route_id)
            .select_related("translation", "edge")
            .first()
        )
        if source_route:
            return _route_travel_mode(source_route)

    network_method = getattr(getattr(route.edge, "network", None), "derivation_method", None)
    if network_method in {"driving", "walking", "bicycling", "transit"}:
        return network_method

    return "driving"


def _clone_source_routes(source_routes, edge_map):
    route_map = {}
    source_route_count = len(source_routes)
    for source_route in source_routes:
        cloned_edge = edge_map.get(source_route.edge_id)
        if not cloned_edge:
            continue
        is_source_primary = _source_route_should_be_primary(source_route, source_route_count)
        route_data = build_route_data(
            (cloned_edge.start_node.latitude, cloned_edge.start_node.longitude),
            (cloned_edge.end_node.latitude, cloned_edge.end_node.longitude),
            _route_travel_mode(source_route),
            use_google=False,
            existing_route_data={
                "polyline": source_route.polyline,
                "distance": source_route.distance_meters,
                "duration": source_route.duration_seconds,
            },
        )
        edge_info = copy.deepcopy(cloned_edge.other_information) or {}
        edge_info["source_route_id"] = source_route.id
        edge_info["route_initial_trunk"] = is_source_primary
        edge_info["route_infill_source_primary"] = is_source_primary
        edge_info["route_travel_mode"] = _route_travel_mode(source_route)
        edge_info["route_resolved_mode"] = route_data.get("mode") or edge_info["route_travel_mode"]
        cloned_edge.other_information = edge_info
        cloned_edge.save(update_fields=["other_information", "updated_at"])
        route_map[source_route.id] = Route.objects.create(
            edge=cloned_edge,
            translation=None,
            route_type="supply" if is_source_primary else source_route.route_type,
            polyline=route_data["polyline"],
            distance_meters=route_data["distance"],
            duration_seconds=route_data["duration"],
        )
    return route_map


def _source_route_should_be_primary(source_route, source_route_count):
    if source_route_count == 1:
        return True
    edge_info = getattr(source_route.edge, "other_information", None) or {}
    return (
        source_route.route_type == "supply" or
        bool(edge_info.get("route_initial_trunk")) or
        bool(edge_info.get("route_infill_source_primary"))
    )


def _split_routes_at_infill_taps(route_map, tap_items_by_source_route):
    for source_route_id, tap_items in tap_items_by_source_route.items():
        cloned_route = route_map.get(source_route_id)
        if not cloned_route or not tap_items:
            continue

        cloned_edge = cloned_route.edge
        points = decode_polyline(cloned_route.polyline)
        if len(points) < 2:
            continue

        projected_points, cumulative, route_epsg = _project_route_points(points)
        if not cumulative or cumulative[-1] <= 0:
            continue

        split_items = [{
            "node": cloned_edge.start_node,
            "along_m": 0.0,
        }]
        for item in tap_items:
            tap_node = item["tap_node"]
            along_m = _project_node_along_route(tap_node, projected_points, route_epsg)
            split_items.append({
                "node": tap_node,
                "along_m": max(0.0, min(float(along_m), cumulative[-1])),
            })
        split_items.append({
            "node": cloned_edge.end_node,
            "along_m": cumulative[-1],
        })
        split_items.sort(key=lambda item: item["along_m"])

        route_type = cloned_route.route_type
        route_mode = _route_travel_mode(cloned_route)
        translation = cloned_route.translation
        base_duration = float(cloned_route.duration_seconds or 0.0)
        base_name = cloned_edge.name or f"Route {source_route_id}"
        edge_algorithm = cloned_edge.edge_algorithm
        is_default = cloned_edge.is_default
        other_information = copy.deepcopy(cloned_edge.other_information) or {}
        source_edge_id = other_information.get("source_edge_id", cloned_edge.id)
        is_source_primary = bool(other_information.get("route_infill_source_primary")) or route_type == "supply"

        cloned_route.delete()
        cloned_edge.delete()

        total_length = cumulative[-1]
        for index, (start_item, end_item) in enumerate(zip(split_items, split_items[1:]), start=1):
            start_node = start_item["node"]
            end_node = end_item["node"]
            start_along = float(start_item["along_m"])
            end_along = float(end_item["along_m"])
            if end_along < start_along:
                start_along, end_along = end_along, start_along
            segment_points = _route_segment_points(
                points,
                cumulative,
                start_along,
                end_along,
                (start_node.latitude, start_node.longitude),
                (end_node.latitude, end_node.longitude),
            )
            segment_length = max(end_along - start_along, _path_length_m(segment_points))
            segment_info = copy.deepcopy(other_information)
            segment_info.update({
                "route_initial_trunk": is_source_primary,
                "route_infill_trunk_segment": is_source_primary,
                "route_infill_source_primary": is_source_primary,
                "source_route_id": source_route_id,
                "source_edge_id": source_edge_id,
                "route_travel_mode": route_mode,
                "route_resolved_mode": route_mode,
                "split_segment_index": index,
            })
            segment_edge = Edge.objects.create(
                network=start_node.network,
                edge_algorithm=edge_algorithm,
                name=f"{base_name} segment {index}",
                start_node=start_node,
                end_node=end_node,
                length=segment_length,
                is_default=is_default,
                other_information=segment_info,
            )
            Route.objects.create(
                edge=segment_edge,
                translation=translation,
                route_type=route_type,
                polyline=encode_polyline(segment_points),
                distance_meters=segment_length,
                duration_seconds=(base_duration * (segment_length / total_length)) if total_length else 0,
            )


def _project_route_points(points):
    centroid_lat = sum(point[0] for point in points) / len(points)
    centroid_lng = sum(point[1] for point in points) / len(points)
    epsg = _utm_epsg(centroid_lng, centroid_lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    projected = [transformer.transform(float(lng), float(lat)) for lat, lng in points]
    cumulative = [0.0]
    for start, end in zip(projected, projected[1:]):
        cumulative.append(cumulative[-1] + math.hypot(end[0] - start[0], end[1] - start[1]))
    return projected, cumulative, epsg


def _project_node_along_route(node, projected_points, epsg):
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    point = Point(transformer.transform(float(node.longitude), float(node.latitude)))
    line = LineString(projected_points)
    return float(line.project(point))


def _route_segment_points(points, cumulative, start_along, end_along, start_coords, end_coords):
    segment = [
        (float(start_coords[0]), float(start_coords[1])),
    ]
    epsilon = 0.05
    for point, along in zip(points[1:-1], cumulative[1:-1]):
        if start_along + epsilon < along < end_along - epsilon:
            segment.append(point)
    segment.append((float(end_coords[0]), float(end_coords[1])))
    if len(segment) == 2 and _distance_m(segment[0], segment[1]) < 0.05:
        return [segment[0], segment[1]]
    return segment


def _path_length_m(points):
    return sum(_distance_m(start, end) for start, end in zip(points, points[1:]))


def _distance_m(start, end):
    earth_radius = 6371000.0
    start_lat, start_lng = math.radians(float(start[0])), math.radians(float(start[1]))
    end_lat, end_lng = math.radians(float(end[0])), math.radians(float(end[1]))
    d_lat = end_lat - start_lat
    d_lng = end_lng - start_lng
    h = math.sin(d_lat / 2) ** 2 + math.cos(start_lat) * math.cos(end_lat) * math.sin(d_lng / 2) ** 2
    return 2 * earth_radius * math.atan2(math.sqrt(h), math.sqrt(1 - h))


def _node_type_for_building(building):
    text = " ".join([
        str(building.building_type or ""),
        str(building.waste_heat_source_type or ""),
        str(building.name or ""),
    ]).lower()
    if any(token in text for token in ["data center", "datacenter", "industrial", "producer", "waste heat"]):
        return "producer"
    return "consumer"


def _single_translation(routes: Iterable[Route]):
    translation_ids = {route.translation_id for route in routes if route.translation_id}
    if len(translation_ids) != 1:
        return None
    return EdgeTranslation.objects.filter(id=next(iter(translation_ids))).first()


def _network_payload(network):
    return {
        "id": network.id,
        "name": network.name,
        "project_id": network.project_id,
        "project_name": network.project.name if network.project_id else None,
        "network_type": network.network_type,
        "total_length": network.total_length,
        "derivation_stage": network.derivation_stage,
    }


def _utm_epsg(lng, lat):
    zone = int((float(lng) + 180) // 6) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def _list_value(raw):
    if raw in [None, "", "all", "null"]:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            raw = parsed
        except (TypeError, ValueError):
            raw = raw.split(",")
    if not isinstance(raw, (list, tuple, set)):
        raw = [raw]
    return [item for item in raw if item not in [None, ""]]


def _score_weights(raw):
    weights = dict(DEFAULT_CANDIDATE_SCORE_WEIGHTS)
    if raw not in [None, ""]:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except (TypeError, ValueError):
                raw = {}
        if isinstance(raw, dict):
            for key in DEFAULT_CANDIDATE_SCORE_WEIGHTS:
                if key in raw:
                    weights[key] = max(_float_value(raw.get(key), weights[key]), 0.0)

    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_CANDIDATE_SCORE_WEIGHTS)
    return {key: value / total for key, value in weights.items()}


def _clamp01(raw):
    return max(0.0, min(1.0, _float_value(raw, 0.0)))


def _float_value(raw, default):
    if raw in [None, ""]:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _int_value(raw, default):
    if raw in [None, ""]:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default
