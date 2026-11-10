import copy
import json
import uuid
from dataclasses import dataclass

import networkx as nx
import numpy as np
from django.db import transaction
from django.db.models import Q, Sum
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors

from api.models import (
    BuildingCluster,
    Cluster,
    Edge,
    EdgeAlgorithm,
    EdgeConnectionAlgorithm,
    EdgeConnectionAlgorithmEdge,
    EdgePipeSizing,
    EdgeTranslation,
    Network,
    NetworkDesignConstants,
    Node,
    PipeSizing,
    Route,
    assert_network_project_integrity,
    validate_project_isolation,
)
from api.utils.distance_calculations import haversine_distance
from api.utils.google_routing import get_google_maps_route
from api.utils.route_geometry import unique_route_length


ROUTE_ENDPOINT_ANCHOR_TOLERANCE_M = 2.0
_ROUTE_DATA_CACHE = {}


@dataclass
class NetworkCloneResult:
    network: Network
    node_map: dict[int, Node]
    edge_map: dict[int, Edge]
    edge_algorithm_map: dict[int, EdgeAlgorithm]
    edge_connection_algorithm_map: dict[int, EdgeConnectionAlgorithm]
    edge_translation_map: dict[int, EdgeTranslation]
    pipe_sizing_map: dict[int, PipeSizing]


def normalize_parameters(parameters):
    if parameters is None:
        return {}
    if isinstance(parameters, dict):
        return copy.deepcopy(parameters)
    if isinstance(parameters, str):
        try:
            parsed = json.loads(parameters)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"raw": parameters}
    return {"value": parameters}


def clone_network(
    source_network,
    *,
    name=None,
    stage=Network.DerivationStage.OTHER,
    method=None,
    parameters=None,
    include_edges=True,
    include_connections=True,
    include_routes=True,
    include_pipe_sizing=False,
):
    """
    Duplicate a network into a child version and remap direct network components.

    Only direct network state is copied. Operation records such as edge
    generations, connection runs, route tracings, and pipe sizing runs are not
    cloned; each derivation creates its own operation record on the child.
    """
    source_network = Network.objects.get(pk=source_network.pk)
    assert_network_project_integrity(source_network)
    params = normalize_parameters(parameters)

    with transaction.atomic():
        target = Network.objects.create(
            project=source_network.project,
            name=name or f"{source_network.name} - {stage.replace('_', ' ').title()}",
            description=source_network.description,
            network_type=source_network.network_type,
            parent_network=source_network,
            total_length=source_network.total_length,
            radius=source_network.radius,
            cluster_algorithm=source_network.cluster_algorithm,
            latitude=source_network.latitude,
            longitude=source_network.longitude,
            derivation_stage=stage,
            derivation_method=method,
            derivation_parameters=params,
        )

        cluster_map = _clone_clusters(source_network, target)
        node_map = _clone_nodes(source_network, target, cluster_map)
        _clone_network_design_constants(source_network, target)
        edge_algorithm_map = {}
        edge_map = {}
        edge_connection_algorithm_map = {}
        edge_translation_map = {}
        pipe_sizing_map = {}

        if include_edges:
            edge_map = clone_edges_to_network(
                Edge.objects.filter(network=source_network),
                target,
                node_map,
            )

        target.total_length = (
            Edge.objects.filter(network=target).aggregate(total=Sum("length"))["total"] or 0.0
        )
        target.save(update_fields=["total_length", "updated_at"])

        return NetworkCloneResult(
            network=target,
            node_map=node_map,
            edge_map=edge_map,
            edge_algorithm_map=edge_algorithm_map,
            edge_connection_algorithm_map=edge_connection_algorithm_map,
            edge_translation_map=edge_translation_map,
            pipe_sizing_map=pipe_sizing_map,
        )


def create_edge_generation_version(source_network, method, *, name=None, parameters=None):
    params = normalize_parameters(parameters)
    output_name = name or f"{source_network.name} - {method.replace('_', ' ').title()} Edges"
    result = clone_network(
        source_network,
        name=output_name,
        stage=Network.DerivationStage.EDGE_GENERATION,
        method=method,
        parameters=params,
        include_edges=False,
        include_connections=False,
        include_routes=False,
        include_pipe_sizing=False,
    )
    edge_algorithm = generate_edges_for_network(
        result.network,
        method,
        params,
        name=output_name,
        source_network=source_network,
    )
    _update_network_length(result.network)
    return result.network, edge_algorithm


def create_edge_connection_version(source_edge_algorithm, method="mst", *, name=None, parameters=None):
    validate_project_isolation(source_edge_algorithm)
    source_network = source_edge_algorithm.network
    assert_network_project_integrity(source_network)
    params = normalize_parameters(parameters)
    output_name = name or f"{source_network.name} - {method.replace('_', ' ').title()} Connected"
    selected_source_edges = select_connection_edges(source_edge_algorithm, method, params)
    if not selected_source_edges:
        raise ValueError("No candidate edges were selected for the connection topology.")
    diagnostics = _connection_graph_diagnostics(selected_source_edges)

    result = clone_network(
        source_network,
        name=output_name,
        stage=Network.DerivationStage.EDGE_CONNECTION,
        method=method,
        parameters={**params, "source_edge_algorithm_id": source_edge_algorithm.id},
        include_edges=False,
        include_connections=False,
        include_routes=False,
        include_pipe_sizing=False,
    )

    edge_map = {}
    total_length = 0.0
    for source_edge in selected_source_edges:
        start_node = result.node_map.get(source_edge.start_node_id)
        end_node = result.node_map.get(source_edge.end_node_id)
        if not start_node or not end_node:
            continue

        cloned_edge = Edge.objects.create(
            network=result.network,
            edge_algorithm=None,
            name=source_edge.name,
            start_node=start_node,
            end_node=end_node,
            length=source_edge.length,
            is_default=source_edge.is_default,
            other_information={
                **(source_edge.other_information or {}),
                "source_edge_id": source_edge.id,
                "source_edge_algorithm_id": source_edge.edge_algorithm_id,
            },
        )
        edge_map[source_edge.id] = cloned_edge
        total_length += source_edge.length or 0.0

    connection_algorithm = EdgeConnectionAlgorithm.objects.create(
        network=result.network,
        source_network=source_network,
        edge_algorithm=None,
        algorithm=method,
        name=output_name,
        description=f"{method.upper()} connection derived from {source_edge_algorithm.name}",
        parameters={
            **params,
            "source_network_id": source_network.id,
            "source_edge_algorithm_id": source_edge_algorithm.id,
            "diagnostics": diagnostics,
        },
        total_edge_length=total_length,
    )

    EdgeConnectionAlgorithmEdge.objects.bulk_create([
        EdgeConnectionAlgorithmEdge(edge_connection_algorithm=connection_algorithm, edge=edge)
        for edge in edge_map.values()
    ])

    result.network.total_length = total_length
    result.network.save(update_fields=["total_length", "updated_at"])
    return result.network, connection_algorithm


def create_route_tracing_version(source_connection_algorithm, translation_type="driving", *, name=None, parameters=None):
    validate_project_isolation(source_connection_algorithm)
    source_network = source_connection_algorithm.network
    assert_network_project_integrity(source_network)
    params = normalize_parameters(parameters)
    if translation_type not in EdgeTranslation.routing_values():
        raise ValueError(f"{translation_type!r} is not an active routing strategy.")
    output_name = name or f"{source_network.name} - {translation_type.replace('_', ' ').title()} Routes"

    result = None
    try:
        result = clone_network(
            source_network,
            name=output_name,
            stage=Network.DerivationStage.ROUTE_TRACING,
            method=translation_type,
            parameters={
                **params,
                "source_network_id": source_network.id,
                "source_edge_connection_algorithm_id": source_connection_algorithm.id,
            },
            include_edges=False,
            include_connections=False,
            include_routes=False,
            include_pipe_sizing=False,
        )

        source_links = list(
            EdgeConnectionAlgorithmEdge.objects.filter(
                edge_connection_algorithm=source_connection_algorithm
            )
            .select_related("edge", "edge__start_node", "edge__end_node")
            .order_by("id")
        )
        if not source_links:
            raise ValueError("The connection run has no selected edges to route.")
        result.edge_map = clone_edges_to_network(
            [link.edge for link in source_links],
            result.network,
            result.node_map,
        )

        translation = EdgeTranslation.objects.create(
            network=result.network,
            source_network=source_network,
            translation=translation_type,
            name=output_name,
            description=f"{translation_type.title()} route tracing derived from {source_connection_algorithm.name}",
            is_default=False,
            parameters={
                **params,
                "source_network_id": source_network.id,
                "source_edge_connection_algorithm_id": source_connection_algorithm.id,
            },
            algorithm=None,
            edge_connection_algorithm=None,
        )

        total_route_length = 0.0
        routes = []
        use_google = bool(params.get("use_google", translation_type != EdgeTranslation.EdgeTranslationType.DIRECT))
        fallback_policy = params.get("fallback_policy", "error")
        source_counts = {}
        for link in source_links:
            edge = result.edge_map.get(link.edge_id)
            if not edge:
                continue
            start_coords = (edge.start_node.latitude, edge.start_node.longitude)
            end_coords = (edge.end_node.latitude, edge.end_node.longitude)
            route_data = build_route_data(
                start_coords,
                end_coords,
                translation_type,
                use_google=use_google,
                fallback_policy=fallback_policy,
            )
            source = route_data.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1

            is_primary = _edge_touches_heat_producer(edge)
            edge_info = copy.deepcopy(edge.other_information) or {}
            edge_info["route_initial_trunk"] = bool(is_primary)
            edge_info["route_source"] = source
            edge_info["route_mode"] = route_data.get("mode", translation_type)
            edge_info["route_geometry_has_loop"] = bool(route_data.get("geometry_has_loop", False))
            edge.other_information = edge_info
            edge.save(update_fields=["other_information", "updated_at"])
            route = Route.objects.create(
                edge=edge,
                translation=translation,
                polyline=route_data["polyline"],
                distance_meters=route_data["distance"],
                duration_seconds=route_data["duration"],
                route_type="supply" if is_primary else "return",
            )
            total_route_length += route.distance_meters
            routes.append(route)

        physical_route_length = unique_route_length(routes)
        translation.total_edge_length = physical_route_length
        translation.parameters = {
            **(translation.parameters or {}),
            "fallback_policy": fallback_policy,
            "route_source_counts": source_counts,
            "route_count": len(routes),
            "summed_route_length_m": total_route_length,
            "unique_physical_length_m": physical_route_length,
        }
        translation.save(update_fields=["total_edge_length", "parameters"])
        result.network.total_length = physical_route_length
        result.network.save(update_fields=["total_length", "updated_at"])
        return result.network, translation, routes
    except Exception:
        if result and result.network and result.network.id:
            Network.objects.filter(pk=result.network.id).delete()
        raise


def _edge_touches_heat_producer(edge):
    return _node_is_heat_producer(edge.start_node) or _node_is_heat_producer(edge.end_node)


def _node_is_heat_producer(node):
    if not node:
        return False

    node_type = (node.node_type or "").lower()
    building = getattr(node, "building", None)
    name = " ".join([
        str(node.name or ""),
        str(getattr(building, "name", "") or ""),
        str(getattr(building, "waste_heat_source_type", "") or ""),
    ]).lower()

    return (
        node_type in {"producer", "boiler"} or
        "producer" in name or
        "data center" in name or
        "data_center" in name or
        "data centre" in name or
        "datacenter" in name or
        "waste heat" in name or
        "waste_heat" in name or
        "treatment plant" in name
    )


def select_connection_edges(edge_algorithm, method="mst", parameters=None):
    params = normalize_parameters(parameters)
    source_edges = list(
        Edge.objects.filter(edge_algorithm=edge_algorithm)
        .select_related("start_node", "end_node")
        .order_by("length", "id")
    )
    if method not in EdgeConnectionAlgorithm.selection_values():
        raise ValueError(f"{method!r} is not an active edge-connection method.")
    if not source_edges:
        raise ValueError("The source edge-generation run has no candidate edges.")

    graph = _connection_candidate_graph(edge_algorithm, source_edges)
    if not nx.is_connected(graph):
        component_sizes = sorted((len(component) for component in nx.connected_components(graph)), reverse=True)
        raise ValueError(
            "Candidate graph is disconnected and cannot produce a connected thermal network. "
            f"Found {len(component_sizes)} components with sizes {component_sizes}."
        )

    if method == EdgeConnectionAlgorithm.EdgeConnectionAlgorithmType.MST:
        selected_graph = nx.minimum_spanning_tree(graph, weight="weight", algorithm="kruskal")

    elif method == EdgeConnectionAlgorithm.EdgeConnectionAlgorithmType.SHORTEST_PATH_TREE:
        root_id = _resolve_connection_root(graph, params)
        paths = nx.single_source_dijkstra_path(graph, root_id, weight="weight")
        selected_pairs = set()
        for path in paths.values():
            selected_pairs.update(_ordered_pair(a, b) for a, b in zip(path, path[1:]))
        selected_graph = graph.edge_subgraph(selected_pairs).copy()
        selected_graph.add_nodes_from(graph.nodes)
        if not nx.is_tree(selected_graph):
            raise ValueError("Shortest-path selection did not produce a spanning tree.")

    elif method == EdgeConnectionAlgorithm.EdgeConnectionAlgorithmType.STEINER_TREE:
        terminal_ids = _resolve_steiner_terminals(graph, params)
        steiner_method = params.get("steiner_method", "mehlhorn")
        if steiner_method not in {"kou", "mehlhorn"}:
            raise ValueError("steiner_method must be 'kou' or 'mehlhorn'.")
        selected_graph = nx.algorithms.approximation.steiner_tree(
            graph,
            terminal_ids,
            weight="weight",
            method=steiner_method,
        )

    elif method == EdgeConnectionAlgorithm.EdgeConnectionAlgorithmType.REDUNDANT_MST:
        selected_graph = nx.minimum_spanning_tree(graph, weight="weight", algorithm="kruskal")
        tree_pairs = {_ordered_pair(a, b) for a, b in selected_graph.edges}
        available = {
            _ordered_pair(a, b): data["weight"]
            for a, b, data in graph.edges(data=True)
            if _ordered_pair(a, b) not in tree_pairs
        }
        try:
            augmentation = list(nx.k_edge_augmentation(
                selected_graph,
                k=2,
                avail=available,
                partial=bool(params.get("allow_partial", False)),
            ))
        except nx.NetworkXUnfeasible as exc:
            raise ValueError(
                "Candidate edges cannot provide two-edge reliability. "
                "Use a denser edge generator or set allow_partial=true for an explicitly partial result."
            ) from exc
        selected_graph.add_edges_from((a, b, graph[a][b]) for a, b in augmentation)

    else:
        raise ValueError(f"Unsupported edge connection method: {method}")

    return [graph[a][b]["edge"] for a, b in selected_graph.edges]


def _connection_candidate_graph(edge_algorithm, source_edges):
    graph = nx.Graph()
    nodes = Node.objects.filter(
        network=edge_algorithm.network,
        latitude__isnull=False,
        longitude__isnull=False,
    )
    graph.add_nodes_from((node.id, {"node": node}) for node in nodes)
    for edge in source_edges:
        if edge.start_node_id == edge.end_node_id:
            raise ValueError(f"Candidate edge {edge.id} is a self-loop.")
        pair = _ordered_pair(edge.start_node_id, edge.end_node_id)
        weight = float(edge.length or 0.0)
        if weight < 0:
            raise ValueError(f"Candidate edge {edge.id} has a negative length.")
        existing = graph.get_edge_data(*pair)
        if existing is None or (weight, edge.id) < (existing["weight"], existing["edge"].id):
            graph.add_edge(*pair, weight=weight, edge=edge)
    return graph


def _resolve_connection_root(graph, params):
    source_node_id = params.get("source_node_id") or params.get("root_node_id")
    if source_node_id is not None:
        source_node_id = int(source_node_id)
        if source_node_id not in graph:
            raise ValueError(f"source_node_id {source_node_id} is not in the candidate graph.")
        return source_node_id

    producer_ids = [node_id for node_id, data in graph.nodes(data=True) if _node_is_heat_producer(data.get("node"))]
    if len(producer_ids) == 1:
        return producer_ids[0]
    raise ValueError(
        "Shortest-path tree requires source_node_id unless the candidate graph contains exactly one recognized heat producer."
    )


def _resolve_steiner_terminals(graph, params):
    supplied = params.get("terminal_node_ids")
    if supplied is not None:
        if not isinstance(supplied, (list, tuple, set)):
            raise ValueError("terminal_node_ids must be a list of node IDs.")
        terminal_ids = {int(value) for value in supplied}
    else:
        terminal_ids = {
            node_id
            for node_id, data in graph.nodes(data=True)
            if (data["node"].node_type or "").lower() not in {"junction", "connector"}
        }
    missing = terminal_ids.difference(graph.nodes)
    if missing:
        raise ValueError(f"Terminal nodes are not in the candidate graph: {sorted(missing)}")
    if len(terminal_ids) < 2:
        raise ValueError("Steiner selection requires at least two terminal nodes.")
    return sorted(terminal_ids)


def _connection_graph_diagnostics(edges):
    graph = nx.Graph()
    graph.add_weighted_edges_from(
        (edge.start_node_id, edge.end_node_id, float(edge.length or 0.0))
        for edge in edges
    )
    return {
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "is_connected": nx.is_connected(graph),
        "is_tree": nx.is_tree(graph),
        "cycle_count": len(nx.cycle_basis(graph)),
        "bridge_count": len(list(nx.bridges(graph))) if nx.is_connected(graph) else None,
        "edge_connectivity": nx.edge_connectivity(graph) if graph.number_of_nodes() > 1 and nx.is_connected(graph) else 0,
    }


def _ordered_pair(a, b):
    return (a, b) if a < b else (b, a)


def generate_edges_for_network(network, method, parameters=None, *, name=None, source_network=None):
    params = normalize_parameters(parameters)
    nodes = list(
        Node.objects.filter(network=network)
        .exclude(latitude__isnull=True)
        .exclude(longitude__isnull=True)
        .order_by("id")
    )
    if len(nodes) < 2:
        raise ValueError("Network needs at least two nodes with coordinates.")

    geographic_coords = np.array([(float(node.latitude), float(node.longitude)) for node in nodes])
    metric_coords = _project_coordinates_to_local_meters(geographic_coords)
    method = method or EdgeAlgorithm.EdgeAlgorithmType.DELAUNAY
    if method not in EdgeAlgorithm.generation_values():
        raise ValueError(
            f"{method!r} is not an edge-generation method. "
            "Generate candidate edges first, then use a connection algorithm such as MST."
        )
    unique_suffix = str(uuid.uuid4())[:8]
    pairs = _edge_pairs_for_method(method, nodes, metric_coords, params)
    diagnostics = _candidate_graph_diagnostics(len(nodes), pairs)
    stored_params = {**params, "diagnostics": diagnostics, "coordinate_space": "local_equirectangular_m"}
    algorithm = EdgeAlgorithm.objects.create(
        network=network,
        source_network=source_network,
        algorithm=method,
        parameters=stored_params,
        name=name or f"{network.name} - {method.replace('_', ' ').title()} - {unique_suffix}",
    )

    total_length = _create_edges_from_pairs(network, algorithm, nodes, geographic_coords, pairs)
    algorithm.total_edge_length = total_length
    algorithm.save(update_fields=["total_edge_length"])
    return algorithm


def _edge_pairs_for_method(method, nodes, coords, params):
    if method == EdgeAlgorithm.EdgeAlgorithmType.DISTANCE_THRESHOLD:
        d_max = _positive_float(params.get("d_max") or params.get("distance_threshold") or 1000, "d_max")
        return [
            (i, j)
            for i in range(len(nodes))
            for j in range(i + 1, len(nodes))
            if np.linalg.norm(coords[i] - coords[j]) <= d_max
        ]

    if method == EdgeAlgorithm.EdgeAlgorithmType.COMPLETE:
        max_nodes = int(params.get("max_nodes", 200))
        if len(nodes) > max_nodes:
            raise ValueError(
                f"Complete graph is limited to {max_nodes} nodes; this network has {len(nodes)}. "
                "Increase max_nodes explicitly only for a controlled benchmark."
            )
        return [(i, j) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]

    if method == EdgeAlgorithm.EdgeAlgorithmType.DELAUNAY:
        if len(nodes) < 3:
            return [(0, 1)]
        try:
            tri = Delaunay(coords)
        except Exception:
            return _gabriel_pairs(coords)
        pairs = set()
        for simplex in tri.simplices:
            for idx in range(3):
                a, b = sorted((simplex[idx], simplex[(idx + 1) % 3]))
                if a != b:
                    pairs.add((a, b))
        return sorted(pairs)

    if method in (EdgeAlgorithm.EdgeAlgorithmType.KNN, EdgeAlgorithm.EdgeAlgorithmType.MUTUAL_KNN):
        requested_k = int(params.get("k", 3))
        if requested_k < 1:
            raise ValueError("k must be at least 1.")
        k = min(requested_k, len(nodes) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="ball_tree").fit(coords)
        _, indices = nbrs.kneighbors(coords)
        neighborhoods = [set(int(value) for value in row[1:]) for row in indices]
        pairs = set()
        for i, neighbors in enumerate(neighborhoods):
            for neighbor_idx in neighbors:
                if method == EdgeAlgorithm.EdgeAlgorithmType.MUTUAL_KNN and i not in neighborhoods[neighbor_idx]:
                    continue
                pairs.add(tuple(sorted((i, neighbor_idx))))
        return sorted(pairs)

    if method == EdgeAlgorithm.EdgeAlgorithmType.GABRIEL:
        return _gabriel_pairs(coords)

    if method == EdgeAlgorithm.EdgeAlgorithmType.RNG:
        return _rng_pairs(coords)

    if method == EdgeAlgorithm.EdgeAlgorithmType.BETA_SKELETON:
        beta = float(params.get("beta", 1.5))
        if not 1.0 <= beta <= 2.0:
            raise ValueError("beta must be between 1 and 2 for the connected lune-based candidate graph family.")
        return _beta_skeleton_pairs(coords, beta)

    raise ValueError(f"Unsupported edge generation method: {method}")


def _project_coordinates_to_local_meters(coords):
    """Project latitude/longitude to a local metric plane for geometric predicates."""
    lat0 = np.radians(np.mean(coords[:, 0]))
    lon0 = np.radians(np.mean(coords[:, 1]))
    lat = np.radians(coords[:, 0])
    lon = np.radians(coords[:, 1])
    earth_radius_m = 6371008.8
    return np.column_stack((
        earth_radius_m * (lon - lon0) * np.cos(lat0),
        earth_radius_m * (lat - lat0),
    ))


def _positive_float(value, name):
    number = float(value)
    if number <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return number


def _candidate_graph_diagnostics(node_count, pairs):
    graph = nx.Graph()
    graph.add_nodes_from(range(node_count))
    graph.add_edges_from(pairs)
    component_count = nx.number_connected_components(graph)
    return {
        "node_count": node_count,
        "edge_count": len(pairs),
        "is_connected": component_count == 1,
        "component_count": component_count,
        "density": nx.density(graph),
    }


def _create_edges_from_pairs(network, algorithm, nodes, coords, pairs):
    seen = set()
    total_length = 0.0
    edges = []
    for i, j in pairs:
        if i == j:
            continue
        a, b = sorted((int(i), int(j)))
        if (a, b) in seen:
            continue
        seen.add((a, b))
        distance = haversine_distance(coords[a][0], coords[a][1], coords[b][0], coords[b][1])
        total_length += distance
        edges.append(Edge(
            network=network,
            edge_algorithm=algorithm,
            start_node=nodes[a],
            end_node=nodes[b],
            length=distance,
            name=f"{nodes[a].name or nodes[a].id}-{nodes[b].name or nodes[b].id}",
            other_information={
                "generated_by": algorithm.algorithm,
                "algorithm_id": algorithm.id,
            },
        ))

    Edge.objects.bulk_create(edges)
    return total_length


def clone_edges_to_network(source_edges, target_network, node_map, edge_algorithm=None):
    edge_map = {}
    for edge in source_edges:
        validate_project_isolation(edge)
        if edge.network.project_id != target_network.project_id:
            raise ValueError(
                f"Edge {edge.id} belongs to project {edge.network.project_id}, "
                f"but target network {target_network.id} belongs to project {target_network.project_id}."
            )
        start_node = node_map.get(edge.start_node_id)
        end_node = node_map.get(edge.end_node_id)
        if not start_node or not end_node:
            continue

        other_information = copy.deepcopy(edge.other_information) or {}
        other_information["source_edge_id"] = edge.id
        if edge.edge_algorithm_id:
            other_information["source_edge_algorithm_id"] = edge.edge_algorithm_id

        cloned = Edge.objects.create(
            network=target_network,
            edge_algorithm=edge_algorithm,
            name=edge.name,
            start_node=start_node,
            end_node=end_node,
            length=edge.length,
            is_default=edge.is_default,
            other_information=other_information,
        )
        edge_map[edge.id] = cloned
    return edge_map


def encode_polyline(points):
    result = []
    prev_lat = 0
    prev_lng = 0

    for lat, lng in points:
        lat_i = int(round(float(lat) * 1e5))
        lng_i = int(round(float(lng) * 1e5))
        result.append(_encode_polyline_value(lat_i - prev_lat))
        result.append(_encode_polyline_value(lng_i - prev_lng))
        prev_lat = lat_i
        prev_lng = lng_i

    return "".join(result)


def decode_polyline(polyline):
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


def _decode_polyline_value(polyline, index):
    result = 0
    shift = 0
    while True:
        if index >= len(polyline):
            raise ValueError("Invalid encoded polyline")
        chunk = ord(polyline[index]) - 63
        index += 1
        result |= (chunk & 0x1f) << shift
        shift += 5
        if chunk < 0x20:
            break
    delta = ~(result >> 1) if result & 1 else result >> 1
    return delta, index


def _encode_polyline_value(value):
    value = ~(value << 1) if value < 0 else value << 1
    chunks = []
    while value >= 0x20:
        chunks.append(chr((0x20 | (value & 0x1f)) + 63))
        value >>= 5
    chunks.append(chr(value + 63))
    return "".join(chunks)


def route_distance(points):
    total = 0.0
    for start, end in zip(points, points[1:]):
        total += haversine_distance(start[0], start[1], end[0], end[1])
    return total


def orient_and_anchor_route_points(points, start_coords, end_coords):
    if len(points) < 2:
        return make_direct_route(start_coords, end_coords)

    start = (float(start_coords[0]), float(start_coords[1]))
    end = (float(end_coords[0]), float(end_coords[1]))
    oriented = list(points)
    forward_distance = (
        haversine_distance(start[0], start[1], oriented[0][0], oriented[0][1]) +
        haversine_distance(end[0], end[1], oriented[-1][0], oriented[-1][1])
    )
    reverse_distance = (
        haversine_distance(start[0], start[1], oriented[-1][0], oriented[-1][1]) +
        haversine_distance(end[0], end[1], oriented[0][0], oriented[0][1])
    )
    if reverse_distance < forward_distance:
        oriented.reverse()

    anchored = list(oriented)
    if haversine_distance(start[0], start[1], anchored[0][0], anchored[0][1]) > ROUTE_ENDPOINT_ANCHOR_TOLERANCE_M:
        anchored.insert(0, start)
    else:
        anchored[0] = start

    if haversine_distance(end[0], end[1], anchored[-1][0], anchored[-1][1]) > ROUTE_ENDPOINT_ANCHOR_TOLERANCE_M:
        anchored.append(end)
    else:
        anchored[-1] = end

    deduped = []
    for point in anchored:
        current = (float(point[0]), float(point[1]))
        if not deduped or _round_point(deduped[-1]) != _round_point(current):
            deduped.append(current)
    return deduped


def build_route_data(
    start_coords,
    end_coords,
    travel_mode="driving",
    *,
    use_google=True,
    existing_route_data=None,
    fallback_policy="error",
):
    """
    Return route geometry that is usable for pipe/network visualization.

    Google can occasionally return extreme detours for very nearby buildings, and
    older overlap-avoidance code left synthetic 4-point routes in existing data.
    This keeps provider routes when they are sane, retries common modes when not,
    and falls back to a direct segment only when no route source produces a
    physically plausible path.
    """
    if existing_route_data:
        normalized = normalize_route_data(existing_route_data, start_coords, end_coords)
        if normalized:
            normalized["source"] = "existing"
            normalized["mode"] = travel_mode
            return normalized

    if travel_mode == EdgeTranslation.EdgeTranslationType.DIRECT or not use_google:
        direct = direct_route_data(start_coords, end_coords)
        direct.update({"source": "direct", "mode": "direct"})
        return direct

    if fallback_policy not in {"error", "direct"}:
        raise ValueError("fallback_policy must be 'error' or 'direct'.")

    cache_key = _route_cache_key(start_coords, end_coords, travel_mode)
    cached = _ROUTE_DATA_CACHE.get(cache_key)
    if cached:
        normalized_cached = normalize_route_data(cached, start_coords, end_coords)
        if normalized_cached:
            normalized_cached.update({"source": "google_cache", "mode": travel_mode})
            return normalized_cached

    route_data = get_google_maps_route(start_coords, end_coords, travel_mode)
    normalized = normalize_route_data(route_data, start_coords, end_coords)
    if normalized:
        normalized.update({"source": "google", "mode": travel_mode})
        _ROUTE_DATA_CACHE[cache_key] = copy.deepcopy(normalized)
        return normalized

    if fallback_policy == "direct":
        direct = direct_route_data(start_coords, end_coords)
        direct.update({"source": "direct_fallback", "mode": "direct"})
        return direct
    raise ValueError(f"No valid {travel_mode} corridor route was returned for {start_coords} -> {end_coords}.")


def _route_cache_key(start_coords, end_coords, travel_mode):
    start = tuple(round(float(value), 6) for value in start_coords)
    end = tuple(round(float(value), 6) for value in end_coords)
    a, b = sorted((start, end))
    return (a, b, travel_mode)


def normalize_route_data(route_data, start_coords, end_coords):
    if not route_data or not route_data.get("polyline"):
        return None

    try:
        points = decode_polyline(route_data["polyline"])
    except (TypeError, ValueError):
        return None

    if len(points) < 2:
        return None
    geometry_has_loop = route_has_loop(points)

    direct_distance = haversine_distance(
        start_coords[0],
        start_coords[1],
        end_coords[0],
        end_coords[1],
    )
    path_distance = route_distance(points)
    route_distance_m = float(route_data.get("distance") or path_distance or 0.0)

    if not route_distance_is_plausible(route_distance_m, direct_distance):
        return None

    anchored_points = orient_and_anchor_route_points(points, start_coords, end_coords)
    anchored_distance = route_distance(anchored_points)

    return {
        "polyline": encode_polyline(anchored_points),
        "distance": max(route_distance_m, anchored_distance),
        "duration": float(route_data.get("duration") or 0.0),
        "geometry_has_loop": geometry_has_loop,
    }


def route_distance_is_plausible(route_distance_m, direct_distance_m):
    if route_distance_m <= 0:
        return False
    if direct_distance_m <= 0:
        return True

    # A road path may be much longer than the building-to-building chord for very
    # short edges, but the old synthetic routes were often tens of kilometres for
    # local segments. These limits remove those without rejecting normal road
    # access around blocks.
    if direct_distance_m < 25:
        max_distance = 300
    elif direct_distance_m < 100:
        max_distance = 600
    elif direct_distance_m < 500:
        max_distance = max(1200, direct_distance_m * 6)
    else:
        max_distance = max(2500, direct_distance_m * 4)

    min_distance = direct_distance_m * 0.4
    return min_distance <= route_distance_m <= max_distance


def direct_route_data(start_coords, end_coords):
    points = make_direct_route(start_coords, end_coords)
    return {
        "polyline": encode_polyline(points),
        "distance": route_distance(points),
        "duration": 0,
        "mode": "direct_fallback",
    }


def route_has_loop(points):
    rounded_points = [_round_point(point) for point in points]
    interior_points = rounded_points[1:-1]
    if len(interior_points) != len(set(interior_points)):
        return True

    segments = list(zip(points, points[1:]))
    for i, segment_a in enumerate(segments):
        for j, segment_b in enumerate(segments):
            if j <= i + 1:
                continue
            if i == 0 and j == len(segments) - 1:
                continue
            if segments_intersect(segment_a[0], segment_a[1], segment_b[0], segment_b[1]):
                return True
    return False


def make_direct_route(start_coords, end_coords):
    return [
        (float(start_coords[0]), float(start_coords[1])),
        (float(end_coords[0]), float(end_coords[1])),
    ]


def segments_intersect(a, b, c, d):
    ax, ay = a[1], a[0]
    bx, by = b[1], b[0]
    cx, cy = c[1], c[0]
    dx, dy = d[1], d[0]

    def orient(px, py, qx, qy, rx, ry):
        value = (qy - py) * (rx - qx) - (qx - px) * (ry - qy)
        if abs(value) < 1e-12:
            return 0
        return 1 if value > 0 else 2

    def on_segment(px, py, qx, qy, rx, ry):
        return (
            min(px, rx) <= qx <= max(px, rx) and
            min(py, ry) <= qy <= max(py, ry)
        )

    o1 = orient(ax, ay, bx, by, cx, cy)
    o2 = orient(ax, ay, bx, by, dx, dy)
    o3 = orient(cx, cy, dx, dy, ax, ay)
    o4 = orient(cx, cy, dx, dy, bx, by)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and on_segment(ax, ay, cx, cy, bx, by):
        return True
    if o2 == 0 and on_segment(ax, ay, dx, dy, bx, by):
        return True
    if o3 == 0 and on_segment(cx, cy, ax, ay, dx, dy):
        return True
    if o4 == 0 and on_segment(cx, cy, bx, by, dx, dy):
        return True
    return False


def _round_point(point):
    return (round(float(point[0]), 5), round(float(point[1]), 5))


def _mst_pairs(nodes, coords):
    candidate_edges = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            dist = haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            candidate_edges.append((dist, i, j))
    candidate_edges.sort()

    parent = list(range(len(nodes)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    pairs = []
    for _, i, j in candidate_edges:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
            pairs.append((i, j))
    return pairs


def _gabriel_pairs(coords):
    pairs = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            midpoint = (coords[i] + coords[j]) / 2
            radius_sq = np.sum((coords[i] - coords[j]) ** 2) / 4
            if all(
                k in (i, j) or np.sum((coords[k] - midpoint) ** 2) > radius_sq
                for k in range(len(coords))
            ):
                pairs.append((i, j))
    return pairs


def _rng_pairs(coords):
    pairs = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist_ij = np.linalg.norm(coords[i] - coords[j])
            valid = True
            for k in range(len(coords)):
                if k in (i, j):
                    continue
                if max(np.linalg.norm(coords[i] - coords[k]), np.linalg.norm(coords[j] - coords[k])) < dist_ij:
                    valid = False
                    break
            if valid:
                pairs.append((i, j))
    return pairs


def _beta_skeleton_pairs(coords, beta):
    if np.isclose(beta, 1.0):
        return _gabriel_pairs(coords)
    if np.isclose(beta, 2.0):
        return _rng_pairs(coords)

    pairs = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            p = coords[i]
            q = coords[j]
            distance = np.linalg.norm(p - q)
            if distance == 0:
                continue
            center_a = (1 - beta / 2) * p + (beta / 2) * q
            center_b = (beta / 2) * p + (1 - beta / 2) * q
            radius = beta * distance / 2
            lune_is_empty = all(
                k in (i, j)
                or np.linalg.norm(coords[k] - center_a) >= radius
                or np.linalg.norm(coords[k] - center_b) >= radius
                for k in range(len(coords))
            )
            if lune_is_empty:
                pairs.append((i, j))
    return pairs


def _tsp_pairs(nodes, coords):
    graph = nx.Graph()
    for i in range(len(nodes)):
        graph.add_node(i)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            graph.add_edge(i, j, weight=haversine_distance(coords[i][0], coords[i][1], coords[j][0], coords[j][1]))
    tour = nx.approximation.traveling_salesman_problem(graph, weight="weight", cycle=False)
    return [tuple(sorted((tour[i], tour[i + 1]))) for i in range(len(tour) - 1)]


def _clone_clusters(source_network, target_network):
    cluster_map = {}
    for cluster in Cluster.objects.filter(network=source_network):
        cloned = Cluster.objects.create(
            network=target_network,
            name=cluster.name,
            description=cluster.description,
            type=cluster.type,
            energy_balance_status=cluster.energy_balance_status,
            total_source_capacity=cluster.total_source_capacity,
            total_sink_capacity=cluster.total_sink_capacity,
            heating_demand=cluster.heating_demand,
            cooling_demand=cluster.cooling_demand,
            latitude=cluster.latitude,
            longitude=cluster.longitude,
            no_of_buildings=cluster.no_of_buildings,
            total_edge_length=cluster.total_edge_length,
            cluster_algorithm=cluster.cluster_algorithm,
            effective_radius=cluster.effective_radius,
            cluster_configuration=cluster.cluster_configuration,
        )
        cluster_map[cluster.id] = cloned
        BuildingCluster.objects.bulk_create([
            BuildingCluster(building=link.building, cluster=cloned)
            for link in BuildingCluster.objects.filter(cluster=cluster).select_related("building")
        ])
    return cluster_map


def _clone_nodes(source_network, target_network, cluster_map):
    node_map = {}
    for node in Node.objects.filter(network=source_network).select_related(
        "building",
        "pv",
        "geothermal",
        "boiler",
        "thermal_storage",
        "cluster",
        "network_node",
    ):
        cloned = Node.objects.create(
            network=target_network,
            name=node.name,
            latitude=node.latitude,
            longitude=node.longitude,
            node_type=node.node_type,
            building=node.building,
            pv=node.pv,
            geothermal=node.geothermal,
            boiler=node.boiler,
            thermal_storage=node.thermal_storage,
            cluster=cluster_map.get(node.cluster_id),
            network_node=target_network if node.network_node_id == source_network.id else node.network_node,
            dea_technology_name=node.dea_technology_name,
            other_information=copy.deepcopy(node.other_information),
            is_default=node.is_default,
        )
        node_map[node.id] = cloned
    return node_map


def _clone_network_design_constants(source_network, target_network):
    source_constants = NetworkDesignConstants.objects.filter(network=source_network).first()
    if not source_constants:
        NetworkDesignConstants.objects.create(network=target_network)
        return

    NetworkDesignConstants.objects.create(
        network=target_network,
        rho=source_constants.rho,
        cp=source_constants.cp,
        mu=source_constants.mu,
        max_velocity=source_constants.max_velocity,
        min_velocity=source_constants.min_velocity,
        target_pressure_drop=source_constants.target_pressure_drop,
        electricity_cost=source_constants.electricity_cost,
        pump_efficiency=source_constants.pump_efficiency,
        fixed_om_rate=source_constants.fixed_om_rate,
        discount_rate=source_constants.discount_rate,
        source_cop=source_constants.source_cop,
        lifespan_years=source_constants.lifespan_years,
        hours_per_year=source_constants.hours_per_year,
        supply_temperature_c=source_constants.supply_temperature_c,
        return_temperature_c=source_constants.return_temperature_c,
        ambient_temperature_c=source_constants.ambient_temperature_c,
        thermal_loss_coefficient_w_m_k=source_constants.thermal_loss_coefficient_w_m_k,
        fallback_pipe_cost_per_meter=source_constants.fallback_pipe_cost_per_meter,
        pipe_catalog=copy.deepcopy(source_constants.pipe_catalog),
    )


def _clone_edge_algorithms(source_network, target_network):
    algorithm_map = {}
    for algorithm in EdgeAlgorithm.objects.filter(network=source_network):
        cloned = EdgeAlgorithm.objects.create(
            network=target_network,
            source_network=algorithm.source_network or source_network,
            algorithm=algorithm.algorithm,
            name=algorithm.name,
            distance_threshold=algorithm.distance_threshold,
            description=algorithm.description,
            is_default=algorithm.is_default,
            parameters=copy.deepcopy(algorithm.parameters),
            total_edge_length=algorithm.total_edge_length,
        )
        algorithm_map[algorithm.id] = cloned
    return algorithm_map


def _clone_edges(source_network, target_network, node_map, edge_algorithm_map=None):
    return clone_edges_to_network(
        Edge.objects.filter(network=source_network).select_related("start_node", "end_node", "edge_algorithm"),
        target_network,
        node_map,
    )


def _clone_edge_connections(source_network, target_network, edge_algorithm_map, edge_map):
    connection_map = {}
    for connection in EdgeConnectionAlgorithm.objects.filter(network=source_network).select_related("edge_algorithm"):
        cloned = EdgeConnectionAlgorithm.objects.create(
            network=target_network,
            source_network=connection.source_network or source_network,
            edge_algorithm=edge_algorithm_map.get(connection.edge_algorithm_id),
            algorithm=connection.algorithm,
            name=connection.name,
            distance_threshold=connection.distance_threshold,
            description=connection.description,
            is_default=connection.is_default,
            parameters=copy.deepcopy(connection.parameters),
            total_edge_length=connection.total_edge_length,
        )
        connection_map[connection.id] = cloned

    EdgeConnectionAlgorithmEdge.objects.bulk_create([
        EdgeConnectionAlgorithmEdge(
            edge_connection_algorithm=connection_map[link.edge_connection_algorithm_id],
            edge=edge_map[link.edge_id],
        )
        for link in EdgeConnectionAlgorithmEdge.objects.filter(
            edge_connection_algorithm__network=source_network,
            edge_id__in=edge_map.keys(),
        )
        if link.edge_connection_algorithm_id in connection_map and link.edge_id in edge_map
    ])
    return connection_map


def _clone_edge_translations(source_network, edge_connection_algorithm_map):
    translation_map = {}
    translations = EdgeTranslation.objects.filter(
        Q(edge_connection_algorithm__network=source_network) |
        Q(algorithm__network=source_network)
    ).distinct()
    for translation in translations:
        cloned = EdgeTranslation.objects.create(
            network=edge_connection_algorithm_map.get(translation.edge_connection_algorithm_id).network if edge_connection_algorithm_map.get(translation.edge_connection_algorithm_id) else None,
            source_network=translation.source_network or source_network,
            translation=translation.translation,
            name=translation.name,
            description=translation.description,
            is_default=translation.is_default,
            parameters=copy.deepcopy(translation.parameters),
            total_edge_length=translation.total_edge_length,
            algorithm=edge_connection_algorithm_map.get(translation.algorithm_id),
            edge_connection_algorithm=edge_connection_algorithm_map.get(translation.edge_connection_algorithm_id),
        )
        translation_map[translation.id] = cloned
    return translation_map


def _clone_routes(source_network, edge_map, edge_translation_map):
    route_map = {}
    for route in Route.objects.filter(edge__network=source_network).select_related("edge", "translation", "sub_route"):
        if route.edge_id not in edge_map:
            continue
        cloned = Route.objects.create(
            route_type=route.route_type,
            edge=edge_map[route.edge_id],
            translation=edge_translation_map.get(route.translation_id),
            sub_route=None,
            polyline=route.polyline,
            distance_meters=route.distance_meters,
            duration_seconds=route.duration_seconds,
        )
        route_map[route.id] = cloned

    for route in Route.objects.filter(id__in=route_map.keys()).exclude(sub_route__isnull=True):
        cloned = route_map.get(route.id)
        cloned_sub_route = route_map.get(route.sub_route_id)
        if cloned and cloned_sub_route:
            cloned.sub_route = cloned_sub_route
            cloned.save(update_fields=["sub_route"])


def _clone_pipe_sizing(source_network, target_network, edge_map):
    sizing_map = {}
    sizing_qs = PipeSizing.objects.filter(
        Q(network=source_network) | Q(edge_pipe_sizings__edge__network=source_network)
    ).distinct()
    for sizing in sizing_qs:
        cloned = PipeSizing.objects.create(
            network=target_network,
            source_network=sizing.source_network or source_network,
            method=sizing.method,
            name=sizing.name,
            description=sizing.description,
            cost_per_meter=sizing.cost_per_meter,
            source=sizing.source,
        )
        sizing_map[sizing.id] = cloned
        EdgePipeSizing.objects.bulk_create([
            EdgePipeSizing(
                edge=edge_map[item.edge_id],
                pipe_sizing=cloned,
                peak_load_kw=item.peak_load_kw,
                mass_flow_kg_s=item.mass_flow_kg_s,
                volume_flow_m3_s=item.volume_flow_m3_s,
                diameter_m=item.diameter_m,
                velocity_m_s=item.velocity_m_s,
                delta_t=item.delta_t,
                pressure_drop_pa=item.pressure_drop_pa,
                route=None,
            )
            for item in EdgePipeSizing.objects.filter(pipe_sizing=sizing, edge_id__in=edge_map.keys())
            if item.edge_id in edge_map
        ])
    return sizing_map


def _update_network_length(network):
    network.total_length = Edge.objects.filter(network=network).aggregate(total=Sum("length"))["total"] or 0.0
    network.save(update_fields=["total_length", "updated_at"])
