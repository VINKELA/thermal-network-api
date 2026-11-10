import math
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_ROUTE_PRECISION_M = 2.0
MIN_SEGMENT_LENGTH_M = 0.05


def decode_polyline(polyline: str) -> List[Tuple[float, float]]:
    if not polyline:
        return []

    points = []
    index = 0
    lat = 0
    lng = 0

    while index < len(polyline):
        lat_delta, index = _decode_polyline_value(polyline, index)
        lng_delta, index = _decode_polyline_value(polyline, index)
        lat += lat_delta
        lng += lng_delta
        points.append((lat / 1e5, lng / 1e5))

    return points


def unique_route_length(routes: Iterable, precision_m: float = DEFAULT_ROUTE_PRECISION_M) -> float:
    return route_physical_segments(routes, precision_m=precision_m)["stats"]["unique_length_m"]


def route_physical_segments(
    routes: Iterable,
    *,
    precision_m: float = DEFAULT_ROUTE_PRECISION_M,
    sizing_by_edge: Optional[dict] = None,
    prefer_shapely: bool = True,
) -> dict:
    route_list = [route for route in routes if route is not None]
    individual_length = sum(_safe_float(getattr(route, "distance_meters", 0.0)) for route in route_list)

    if not route_list:
        return _empty_result(individual_length, precision_m, "no_routes")

    if not prefer_shapely:
        return _route_physical_segments_by_snapped_edges(
            route_list,
            precision_m=precision_m,
            sizing_by_edge=sizing_by_edge or {},
            individual_length=individual_length,
        )

    try:
        return _route_physical_segments_shapely(
            route_list,
            precision_m=precision_m,
            sizing_by_edge=sizing_by_edge or {},
            individual_length=individual_length,
        )
    except Exception:
        return _route_physical_segments_by_snapped_edges(
            route_list,
            precision_m=precision_m,
            sizing_by_edge=sizing_by_edge or {},
            individual_length=individual_length,
        )


def _route_physical_segments_shapely(route_list, *, precision_m, sizing_by_edge, individual_length):
    from pyproj import Transformer
    from shapely import set_precision
    from shapely.geometry import LineString
    from shapely.ops import transform as shapely_transform
    from shapely.ops import unary_union

    decoded_items = []
    all_points = []
    fallback_entries = []

    for route in route_list:
        try:
            points = decode_polyline(route.polyline)
        except (TypeError, ValueError):
            points = []

        if len(points) < 2:
            fallback_entries.append(_entry_from_route(route, _safe_float(getattr(route, "distance_meters", 0.0))))
            continue

        decoded_items.append({"route": route, "points": points})
        all_points.extend(points)

    if not decoded_items:
        return _fallback_result(fallback_entries, individual_length, precision_m, "route_distance_sum_no_geometry")

    centroid_lat = sum(point[0] for point in all_points) / len(all_points)
    centroid_lng = sum(point[1] for point in all_points) / len(all_points)
    epsg = _utm_epsg(centroid_lng, centroid_lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inverse_transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    projected_items = []
    for item in decoded_items:
        line = LineString([(lng, lat) for lat, lng in item["points"]])
        if line.is_empty or line.length == 0:
            fallback_entries.append(_entry_from_route(item["route"], _safe_float(getattr(item["route"], "distance_meters", 0.0))))
            continue

        projected_line = shapely_transform(transformer.transform, line)
        if precision_m and precision_m > 0:
            projected_line = set_precision(projected_line, precision_m)

        if projected_line.is_empty or projected_line.length <= MIN_SEGMENT_LENGTH_M:
            fallback_entries.append(_entry_from_route(item["route"], _safe_float(getattr(item["route"], "distance_meters", 0.0))))
            continue

        route_distance = _safe_float(getattr(item["route"], "distance_meters", 0.0))
        distance_scale = route_distance / projected_line.length if route_distance > 0 and projected_line.length > 0 else 1.0
        projected_items.append({"route": item["route"], "line": projected_line, "distance_scale": distance_scale})

    if not projected_items:
        return _fallback_result(fallback_entries, individual_length, precision_m, "route_distance_sum_no_geometry")

    unioned = unary_union([item["line"] for item in projected_items])
    entries = []

    for segment in _line_parts(unioned):
        segment_length = float(segment.length)
        if segment_length <= MIN_SEGMENT_LENGTH_M:
            continue

        candidates = _covering_route_items(segment, projected_items)
        if not candidates:
            candidates = [_nearest_route_item(segment, projected_items)]
        representative = _representative_route_item(candidates, sizing_by_edge)
        entries.append(_entry_from_route(
            representative["route"],
            segment_length * representative.get("distance_scale", 1.0),
            source_routes=[candidate["route"] for candidate in candidates],
            path_points=_projected_segment_path(segment, inverse_transformer),
        ))

    entries.extend(fallback_entries)
    unique_length = sum(entry["length_m"] for entry in entries)
    return {
        "entries": entries,
        "stats": _stats_payload(
            individual_length,
            unique_length,
            precision_m,
            "projected_polyline_union",
        ),
    }


def _route_physical_segments_by_snapped_edges(route_list, *, precision_m, sizing_by_edge, individual_length):
    meters_per_degree = 111_320.0
    degree_precision = max(precision_m, DEFAULT_ROUTE_PRECISION_M) / meters_per_degree
    segment_map = {}

    for route in route_list:
        try:
            points = decode_polyline(route.polyline)
        except (TypeError, ValueError):
            points = []

        if len(points) < 2:
            key = ("fallback", getattr(route, "id", id(route)))
            segment_map[key] = {
                "route": route,
                "length_m": _safe_float(getattr(route, "distance_meters", 0.0)),
                "source_routes": [route],
                "path_points": _route_path_points(route),
            }
            continue

        route_geometry_length = sum(_haversine_distance(start, end) for start, end in zip(points, points[1:]))
        route_distance = _safe_float(getattr(route, "distance_meters", 0.0))
        distance_scale = route_distance / route_geometry_length if route_distance > 0 and route_geometry_length > 0 else 1.0

        for start, end in zip(points, points[1:]):
            snapped_start = _snap_lat_lng(start, degree_precision)
            snapped_end = _snap_lat_lng(end, degree_precision)
            if snapped_start == snapped_end:
                continue
            key = tuple(sorted([snapped_start, snapped_end]))
            length = _haversine_distance(start, end) * distance_scale
            existing = segment_map.get(key)
            if not existing:
                segment_map[key] = {
                    "route": route,
                    "length_m": length,
                    "source_routes": [route],
                    "path_points": [_lat_lng_payload(start), _lat_lng_payload(end)],
                }
                continue
            existing["source_routes"].append(route)
            existing["route"] = _representative_route([existing["route"], route], sizing_by_edge)
            existing["length_m"] = max(existing["length_m"], length)

    entries = [
        _entry_from_route(
            item["route"],
            item["length_m"],
            source_routes=item["source_routes"],
            path_points=item.get("path_points"),
        )
        for item in segment_map.values()
    ]
    unique_length = sum(entry["length_m"] for entry in entries)
    return {
        "entries": entries,
        "stats": _stats_payload(individual_length, unique_length, precision_m, "snapped_polyline_segments"),
    }


def _covering_route_items(segment, projected_items):
    candidates = []
    min_overlap = max(MIN_SEGMENT_LENGTH_M, min(1.0, segment.length * 0.1))
    for item in projected_items:
        try:
            overlap = segment.intersection(item["line"]).length
        except Exception:
            overlap = 0.0
        if overlap >= min_overlap:
            candidates.append(item)
    return candidates


def _nearest_route_item(segment, projected_items):
    return min(projected_items, key=lambda item: segment.distance(item["line"]))


def _representative_route_item(candidates, sizing_by_edge):
    return max(candidates, key=lambda item: _route_score(item["route"], sizing_by_edge))


def _representative_route(routes, sizing_by_edge):
    return max(routes, key=lambda route: _route_score(route, sizing_by_edge))


def _route_score(route, sizing_by_edge):
    edge_id = getattr(route, "edge_id", None)
    sizing = sizing_by_edge.get(edge_id) if sizing_by_edge else None
    diameter = _safe_float(getattr(sizing, "diameter_m", 0.0)) if sizing else 0.0
    pressure_drop = _safe_float(getattr(sizing, "pressure_drop_pa", 0.0)) if sizing else 0.0
    return (
        diameter,
        pressure_drop,
        _safe_float(getattr(route, "distance_meters", 0.0)),
        getattr(route, "id", 0) or 0,
    )


def _entry_from_route(route, length_m, *, source_routes: Optional[Sequence] = None, path_points: Optional[Sequence] = None):
    source_routes = list(source_routes or [route])
    source_route_ids = sorted({getattr(source_route, "id", None) for source_route in source_routes if getattr(source_route, "id", None) is not None})
    source_edge_ids = sorted({getattr(source_route, "edge_id", None) for source_route in source_routes if getattr(source_route, "edge_id", None) is not None})
    source_route_types = sorted({
        getattr(source_route, "route_type", None)
        for source_route in source_routes
        if getattr(source_route, "route_type", None)
    })
    route_infos = [
        getattr(getattr(source_route, "edge", None), "other_information", None) or {}
        for source_route in source_routes
    ]
    has_initial_trunk_segment = any(
        bool(info.get("route_initial_trunk")) or bool(info.get("route_infill_source_primary"))
        for info in route_infos
    )
    has_trunk_segment = any(
        bool(info.get("route_infill_trunk_segment"))
        for info in route_infos
    )
    source_node_ids = sorted({
        node_id
        for source_route in source_routes
        for node_id in (
            getattr(getattr(source_route, "edge", None), "start_node_id", None),
            getattr(getattr(source_route, "edge", None), "end_node_id", None),
        )
        if node_id is not None
    })
    return {
        "edge": route.edge,
        "route": route,
        "route_id": getattr(route, "id", None),
        "length_m": float(length_m or 0.0),
        "source_length_m": _safe_float(getattr(route, "distance_meters", 0.0)),
        "source_route_ids": source_route_ids,
        "source_route_types": source_route_types,
        "has_supply": "supply" in source_route_types,
        "has_initial_trunk_segment": bool(has_initial_trunk_segment),
        "has_confirmed_trunk_segment": bool(has_initial_trunk_segment),
        "has_trunk_segment": bool(has_trunk_segment),
        "source_edge_ids": source_edge_ids,
        "source_node_ids": source_node_ids,
        "overlap_count": len(source_route_ids) or 1,
        "is_shared": len(source_route_ids) > 1,
        "path": list(path_points) if path_points else _route_path_points(route),
    }


def _line_parts(geometry):
    if geometry.is_empty:
        return []
    if geometry.geom_type == "LineString":
        return [geometry]
    if geometry.geom_type == "MultiLineString":
        return list(geometry.geoms)
    if geometry.geom_type == "GeometryCollection":
        parts = []
        for geom in geometry.geoms:
            parts.extend(_line_parts(geom))
        return parts
    return []


def _projected_segment_path(segment, inverse_transformer):
    points = []
    for x, y in segment.coords:
        lng, lat = inverse_transformer.transform(x, y)
        points.append({"lat": float(lat), "lng": float(lng)})
    return points


def _route_path_points(route):
    try:
        points = decode_polyline(route.polyline)
    except (TypeError, ValueError):
        points = []
    return [_lat_lng_payload(point) for point in points]


def _lat_lng_payload(point):
    lat, lng = point
    return {"lat": float(lat), "lng": float(lng)}


def _stats_payload(individual_length, unique_length, precision_m, method):
    overlap_removed = max(0.0, individual_length - unique_length)
    return {
        "method": method,
        "precision_m": precision_m,
        "individual_route_length_m": individual_length,
        "unique_length_m": unique_length,
        "overlap_deduped_length_m": overlap_removed,
        "overlap_ratio": overlap_removed / individual_length if individual_length else 0.0,
    }


def _empty_result(individual_length, precision_m, method):
    return {
        "entries": [],
        "stats": _stats_payload(individual_length, 0.0, precision_m, method),
    }


def _fallback_result(entries, individual_length, precision_m, method):
    unique_length = sum(entry["length_m"] for entry in entries)
    return {
        "entries": entries,
        "stats": _stats_payload(individual_length, unique_length, precision_m, method),
    }


def _decode_polyline_value(polyline, index):
    result = 0
    shift = 0
    while True:
        if index >= len(polyline):
            raise ValueError("Invalid encoded polyline")
        chunk = ord(polyline[index]) - 63
        index += 1
        result |= (chunk & 0x1F) << shift
        shift += 5
        if chunk < 0x20:
            break
    delta = ~(result >> 1) if result & 1 else result >> 1
    return delta, index


def _utm_epsg(lng, lat):
    zone = int((float(lng) + 180) // 6) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def _snap_lat_lng(point, degree_precision):
    lat, lng = point
    return (
        round(float(lat) / degree_precision) * degree_precision,
        round(float(lng) / degree_precision) * degree_precision,
    )


def _haversine_distance(start, end):
    radius = 6_371_000.0
    lat1, lng1 = map(math.radians, start)
    lat2, lng2 = map(math.radians, end)
    d_lat = lat2 - lat1
    d_lng = lng2 - lng1
    a = math.sin(d_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(d_lng / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _safe_float(value):
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
