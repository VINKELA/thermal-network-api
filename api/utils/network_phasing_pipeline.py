from copy import deepcopy

from api.models import (
    ClusterConfiguration,
    Edge,
    EdgeAlgorithm,
    EdgeConnectionAlgorithm,
    EdgeTranslation,
    Network,
    PipeSizing,
    Route,
    assert_network_project_integrity,
)
from api.utils.complementary_clusters import create_complementary_cluster_configuration_from_network
from api.utils.network_versioning import (
    build_route_data,
    clone_edges_to_network,
    clone_network,
    create_edge_connection_version,
    create_edge_generation_version,
    create_route_tracing_version,
)
from api.utils.pipe_sizing import NetworkSizer
from api.utils.route_geometry import unique_route_length


def prepare_network_for_phasing(base_network, raw_params=None, *, progress_callback=None):
    """
    Build the normal persisted design chain before phasing:

    base nodes -> edge generation -> edge connection -> route tracing ->
    pipe sizing -> optional complementary clusters -> phasing input network.
    """
    params = deepcopy(raw_params or {})
    pipeline_params = _pipeline_params(params)
    pipeline_enabled = _bool_value(pipeline_params.get("prepare_pipeline"), False)

    base_network = Network.objects.select_related("project").get(pk=base_network.pk)
    assert_network_project_integrity(base_network)

    if not pipeline_enabled:
        return {
            "network": base_network,
            "params": params,
            "pipeline": {
                "enabled": False,
                "input_network_id": base_network.id,
                "phasing_network_id": base_network.id,
                "steps": [],
            },
        }

    if _bool_value(pipeline_params.get("reuse_sized_network"), True):
        existing = _existing_sized_network(base_network)
        if existing:
            sized_network, sizing = existing
            params = _prepare_cluster_params_if_needed(
                sized_network,
                params,
                pipeline_params,
                progress_callback=progress_callback,
            )
            steps = [{
                "stage": "pipe_sizing",
                "network_id": sized_network.id,
                "pipe_sizing_id": sizing.id if sizing else None,
                "reused": True,
            }]
            if params.get("preparation_cluster_step"):
                steps.append(params.pop("preparation_cluster_step"))
            pipeline = {
                "enabled": True,
                "reused_existing_sized_network": True,
                "input_network_id": base_network.id,
                "phasing_network_id": sized_network.id,
                "pipe_sizing_network_id": sized_network.id,
                "pipe_sizing_id": sizing.id if sizing else None,
                "steps": steps,
            }
            params["preparation_pipeline"] = pipeline
            params["input_network_id"] = base_network.id
            return {"network": sized_network, "params": params, "pipeline": pipeline}

    steps = []
    prefix = pipeline_params.get("name_prefix") or params.get("name") or base_network.name

    _notify(progress_callback, "edge_generation", "Generating candidate edges from network nodes.")
    generated_network, edge_algorithm = create_edge_generation_version(
        base_network,
        pipeline_params["edge_generation_method"],
        name=f"{prefix} - Generated Edges",
        parameters=pipeline_params["edge_generation_parameters"],
    )
    steps.append({
        "stage": "edge_generation",
        "network_id": generated_network.id,
        "edge_algorithm_id": edge_algorithm.id,
        "method": edge_algorithm.algorithm,
        "total_length_m": edge_algorithm.total_edge_length,
    })

    _notify(progress_callback, "edge_connection", "Optimizing generated edges into a connected network.")
    connected_network, connection_algorithm = create_edge_connection_version(
        edge_algorithm,
        pipeline_params["edge_connection_method"],
        name=f"{prefix} - Optimized Edges",
        parameters=pipeline_params["edge_connection_parameters"],
    )
    steps.append({
        "stage": "edge_connection",
        "network_id": connected_network.id,
        "edge_connection_algorithm_id": connection_algorithm.id,
        "method": connection_algorithm.algorithm,
        "total_length_m": connection_algorithm.total_edge_length,
    })

    _notify(progress_callback, "route_tracing", "Tracing optimized edges onto route geometry.")
    routed_network, edge_translation, routes = create_route_tracing_version(
        connection_algorithm,
        pipeline_params["route_mode"],
        name=f"{prefix} - Routed Network",
        parameters=pipeline_params["route_parameters"],
    )
    steps.append({
        "stage": "route_tracing",
        "network_id": routed_network.id,
        "edge_translation_id": edge_translation.id,
        "method": edge_translation.translation,
        "route_count": len(routes),
        "total_length_m": edge_translation.total_edge_length,
    })

    _notify(progress_callback, "pipe_sizing", "Sizing routed pipe sections.")
    sizing_result = create_pipe_sizing_version_from_route(
        edge_translation,
        {
            "method": pipeline_params["pipe_sizing_method"],
            "year": pipeline_params["pipe_sizing_year"],
            "name_prefix": f"{prefix} - Pipe Sizing",
            "network_name": f"{prefix} - Sized Network",
        },
    )
    sized_network = sizing_result["network"]
    pipe_sizing = sizing_result["pipe_sizing"]
    steps.append({
        "stage": "pipe_sizing",
        "network_id": sized_network.id,
        "pipe_sizing_id": pipe_sizing.id,
        "method": pipe_sizing.method,
        "edges_sized": sizing_result["edges_sized"],
        "route_length_method": sizing_result.get("route_length_method"),
        "overlap_deduped_length_m": sizing_result.get("overlap_deduped_length_m"),
    })

    params = _prepare_cluster_params_if_needed(
        sized_network,
        params,
        pipeline_params,
        progress_callback=progress_callback,
    )
    if params.get("preparation_cluster_step"):
        steps.append(params.pop("preparation_cluster_step"))

    pipeline = {
        "enabled": True,
        "reused_existing_sized_network": False,
        "input_network_id": base_network.id,
        "phasing_network_id": sized_network.id,
        "edge_generation_network_id": generated_network.id,
        "edge_algorithm_id": edge_algorithm.id,
        "edge_connection_network_id": connected_network.id,
        "edge_connection_algorithm_id": connection_algorithm.id,
        "route_tracing_network_id": routed_network.id,
        "edge_translation_id": edge_translation.id,
        "pipe_sizing_network_id": sized_network.id,
        "pipe_sizing_id": pipe_sizing.id,
        "steps": steps,
    }
    params["preparation_pipeline"] = pipeline
    params["input_network_id"] = base_network.id
    return {"network": sized_network, "params": params, "pipeline": pipeline}


def create_pipe_sizing_version_from_route(edge_translation, raw_params=None):
    params = deepcopy(raw_params or {})
    method = params.get("method") or "peak_sizing"
    year = params.get("year") or 2024
    name_prefix = params.get("name_prefix") or method.replace("_", " ").title()

    edge_translation = EdgeTranslation.objects.select_related("network", "source_network").get(pk=edge_translation.pk)
    source_network = edge_translation.network
    if not source_network:
        raise ValueError("Route tracing is not attached to a network.")
    assert_network_project_integrity(source_network)

    source_routes = list(
        Route.objects.filter(translation=edge_translation)
        .select_related("edge", "edge__start_node", "edge__end_node")
        .order_by("id")
    )
    if not source_routes:
        raise ValueError("The selected route tracing has no routes to size.")

    clone_result = clone_network(
        source_network,
        name=params.get("network_name") or f"{source_network.name} - {name_prefix}",
        stage=Network.DerivationStage.PIPE_SIZING,
        method=method,
        parameters={
            "source_network_id": source_network.id,
            "source_edge_translation_id": edge_translation.id,
            "topology_source": "route",
            "year": year,
            "method": method,
        },
        include_edges=False,
        include_connections=False,
        include_routes=False,
        include_pipe_sizing=False,
    )

    unique_edges = []
    seen_edge_ids = set()
    for route in source_routes:
        if route.edge_id not in seen_edge_ids:
            unique_edges.append(route.edge)
            seen_edge_ids.add(route.edge_id)

    clone_result.edge_map = clone_edges_to_network(
        unique_edges,
        clone_result.network,
        clone_result.node_map,
    )

    cloned_routes_for_sizing = []
    edge_lengths_to_update = {}
    route_params = edge_translation.parameters or {}
    for route in source_routes:
        cloned_edge = clone_result.edge_map.get(route.edge_id)
        if not cloned_edge:
            continue
        start_coords = (cloned_edge.start_node.latitude, cloned_edge.start_node.longitude)
        end_coords = (cloned_edge.end_node.latitude, cloned_edge.end_node.longitude)
        route_data = build_route_data(
            start_coords,
            end_coords,
            edge_translation.translation or "driving",
            use_google=route_params.get("use_google", True),
            existing_route_data={
                "polyline": route.polyline,
                "distance": route.distance_meters,
                "duration": route.duration_seconds,
            },
        )
        cloned_route = Route.objects.create(
            edge=cloned_edge,
            translation=None,
            route_type=route.route_type,
            polyline=route_data["polyline"],
            distance_meters=route_data["distance"],
            duration_seconds=route_data["duration"],
        )
        cloned_routes_for_sizing.append(cloned_route)
        if route_data["distance"]:
            edge_lengths_to_update[cloned_edge.id] = route_data["distance"]

    if edge_lengths_to_update:
        cloned_edges_to_update = list(Edge.objects.filter(id__in=edge_lengths_to_update.keys()))
        for cloned_edge in cloned_edges_to_update:
            cloned_edge.length = edge_lengths_to_update[cloned_edge.id]
        Edge.objects.bulk_update(cloned_edges_to_update, ["length"])

    pipe_sizing = PipeSizing.objects.create(
        network=clone_result.network,
        source_network=source_network,
        name=f"{name_prefix} - {clone_result.network.name}",
        method=method,
        cost_per_meter=0,
        source=f"Phasing pipeline sizing from route tracing ({year})",
    )

    sizer = NetworkSizer(clone_result.network.id, pipe_sizing.id)
    sizer.build_topology(conn_algorithm_id=None, year=year)
    _run_sizing_method(sizer, method)
    route_sizing_stats = sizer.collapse_route_results_to_physical_segments(cloned_routes_for_sizing) or {}
    sizer.save_results()

    clone_result.network.total_length = unique_route_length(
        Route.objects.filter(edge__network=clone_result.network).select_related("edge")
    )
    clone_result.network.save(update_fields=["total_length", "updated_at"])

    return {
        "network": clone_result.network,
        "pipe_sizing": pipe_sizing,
        "edges_sized": len(sizer.results),
        "route_length_method": route_sizing_stats.get("method"),
        "overlap_deduped_length_m": route_sizing_stats.get("overlap_deduped_length_m", 0.0),
    }


def _run_sizing_method(sizer, method):
    if method == "peak_sizing":
        sizer.sizing_peak_load()
    elif method == "diversity":
        sizer.sizing_diversity()
    elif method == "pressure_optimized":
        sizer.sizing_pressure_optimized()
    elif method == "lifecycle_cost":
        sizer.sizing_lifecycle_economic()
    elif method == "return_temp_opt":
        sizer.sizing_variable_delta_t()
    else:
        raise ValueError(f"Unknown pipe sizing method: {method}")


def _prepare_cluster_params_if_needed(network, params, pipeline_params, *, progress_callback=None):
    candidate_source_mode = _candidate_source_mode(params)
    include_clusters = _bool_value(params.get("include_clusters"), False) or candidate_source_mode == "clusters_only"
    selected_config_id = params.get("complementary_cluster_configuration_id") or params.get("cluster_configuration_id")
    ensure_clusters = bool(selected_config_id) or _bool_value(
        pipeline_params.get("ensure_complementary_clusters"),
        include_clusters,
    )
    if not ensure_clusters:
        return params

    existing_config = None
    if selected_config_id:
        existing_config = ClusterConfiguration.objects.filter(pk=selected_config_id).first()
        if existing_config and existing_config.base_network_id == network.id:
            params["include_clusters"] = True
            params["complementary_cluster_configuration_id"] = existing_config.id
            return params

    _notify(progress_callback, "complementary_clusters", "Creating complementary clusters for the sized network.")
    result = create_complementary_cluster_configuration_from_network(
        network,
        max_radius_km=pipeline_params.get("complementary_cluster_radius_km") or 0.1,
        name=f"Complementary clusters for {network.name}",
        description="Created automatically by the phasing preparation pipeline.",
    )
    params["include_clusters"] = True
    params["complementary_cluster_configuration_id"] = result["config_id"]
    params["cluster_ids"] = []
    params["preparation_cluster_step"] = {
        "stage": "complementary_clusters",
        "cluster_configuration_id": result["config_id"],
        "clusters_created": result["clusters_created"],
        "buildings_included": result["buildings_included"],
    }
    return params


def _candidate_source_mode(params):
    value = str(
        params.get("candidate_source_mode") or
        params.get("candidate_asset_scope") or
        params.get("candidate_scope") or
        ""
    ).strip().lower().replace("-", "_")
    if value in {"cluster", "clusters", "clusters_only"}:
        return "clusters_only"
    if value in {"building", "buildings", "buildings_only"}:
        return "buildings_only"
    return "buildings_and_clusters"


def _existing_sized_network(base_network):
    if base_network.derivation_stage == Network.DerivationStage.PIPE_SIZING:
        sizing = PipeSizing.objects.filter(network=base_network).order_by("-created_at").first()
        if sizing:
            return base_network, sizing

    child = (
        Network.objects
        .filter(parent_network=base_network, derivation_stage=Network.DerivationStage.PIPE_SIZING)
        .order_by("-created_at")
        .first()
    )
    if child:
        sizing = PipeSizing.objects.filter(network=child).order_by("-created_at").first()
        if sizing:
            return child, sizing
    return None


def _pipeline_params(raw_params):
    nested = raw_params.get("pipeline") if isinstance(raw_params.get("pipeline"), dict) else {}
    params = {**raw_params, **nested}
    return {
        "prepare_pipeline": params.get("prepare_pipeline"),
        "reuse_sized_network": params.get("reuse_sized_network"),
        "edge_generation_method": params.get("edge_generation_method") or EdgeAlgorithm.EdgeAlgorithmType.DELAUNAY,
        "edge_generation_parameters": params.get("edge_generation_parameters") or params.get("edge_params") or {},
        "edge_connection_method": params.get("edge_connection_method") or EdgeConnectionAlgorithm.EdgeConnectionAlgorithmType.MST,
        "edge_connection_parameters": params.get("edge_connection_parameters") or params.get("connection_params") or {},
        "route_mode": params.get("route_tracing_mode") or params.get("route_mode") or "driving",
        "route_parameters": params.get("route_parameters") or {"use_google": True},
        "pipe_sizing_method": params.get("pipe_sizing_method") or "peak_sizing",
        "pipe_sizing_year": params.get("pipe_sizing_year") or params.get("year") or 2024,
        "ensure_complementary_clusters": params.get("ensure_complementary_clusters"),
        "complementary_cluster_radius_km": _float_value(params.get("complementary_cluster_radius_km"), 0.1),
        "name_prefix": params.get("pipeline_name_prefix") or params.get("name_prefix"),
    }


def _notify(callback, stage, message, extra=None):
    if callback:
        callback(stage, message, extra or {})


def _bool_value(value, default=False):
    if value in [None, ""]:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _float_value(value, default):
    try:
        if value in [None, ""]:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default
