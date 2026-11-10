from api.models import Network, Node, Edge, EdgeAlgorithm, EdgeConnectionAlgorithm, EdgeConnectionAlgorithmEdge
import numpy as np
import json
import uuid
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from api.utils.distance_calculations import haversine_distance

def connect_by_distance_threshold(network, d_max):
    network = Network.objects.get(pk=network_id)
    nodes = list(Node.objects.filter(network=network))
    edges_created = 0
    total_length = 0.0
    unique_suffix = str(uuid.uuid4())[:8]
    edge_algorithm = EdgeAlgorithm.objects.create(
        network=network,
        algorithm=EdgeAlgorithm.EdgeAlgorithmType.DISTANCE_THRESHOLD,
        parameters=json.dumps({'d_max': d_max}),
        name=f"{network.name} - Distance Threshold - {unique_suffix}"
    )
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            if distance <= d_max:
                unique_name = f"{node1.name or node1.id}-{node2.name or node2.id}-{i}-{j}"
                # check if edge between these nodes already exists
                if Edge.objects.filter(network=network, start_node=node1, end_node=node2, edge_algorithm=edge_algorithm).exists():
                    # check if edge is shorter than existing edge
                    existing_edge = Edge.objects.get(network=network, start_node=node1, end_node=node2, edge_algorithm=edge_algorithm)
                    if existing_edge.length > distance:
                        #replace existing edge with shorter one
                        edges_created -= 1  # decrement since we're replacing
                        total_length -= existing_edge.length
                        existing_edge.length = distance
                        existing_edge.name = unique_name
                        existing_edge.save(update_fields=['length', 'name'])
            else:
                Edge.objects.create(
                    network=network,
                    start_node=node1,
                    end_node=node2,
                    length=distance,
                    name=unique_name, # type: ignore
                    edge_algorithm=edge_algorithm
                )
            edges_created += 1
            total_length += distance
    edge_algorithm.total_edge_length = total_length
    edge_algorithm.save(update_fields=["total_edge_length"])
    return edge_algorithm
def connect_by_gabriel(network):
    """
    Connect nodes following Gabriel Graph rules:
    An edge exists between A and B if no other node C is in the circle 
    with diameter AB.
    """
    network = Network.objects.get(pk=network_id)
    nodes = list(Node.objects.filter(network=network))

    edges_created = 0
    total_length = 0.0
    unique_suffix = str(uuid.uuid4())[:8]
    
    # Create algorithm metadata record
    edge_algorithm = EdgeAlgorithm.objects.create(
        network=network,
        algorithm=EdgeAlgorithm.EdgeAlgorithmType.GABRIEL,
        parameters=json.dumps({}),
        name=f"{network.name} - Gabriel Graph - {unique_suffix}"
    )
    
    n = len(nodes)
    
    # Pre-compute all coordinates as numpy array for vector operations
    coord_array = np.array(coords)
    
    for i in range(n):
        for j in range(i+1, n):
            node1 = nodes[i]
            node2 = nodes[j]
            a = coord_array[i]
            b = coord_array[j]
            
            # Calculate midpoint and radius
            midpoint = (a + b) / 2
            radius_sq = np.sum((a - b)**2) / 4  # Squared for comparison
            
            # Check Gabriel condition
            valid_edge = True
            for k in range(n):
                if k == i or k == j:
                    continue
                c = coord_array[k]
                if np.sum((c - midpoint)**2) <= radius_sq:
                    valid_edge = False
                    break
            
            if valid_edge:
                distance = haversine_distance(a[0], a[1], b[0], b[1])
                unique_name = f"{node1.name or node1.id}-{node2.name or node2.id}-{i}-{j}"
                
                # Check for existing edge
                existing_edge = Edge.objects.filter(
                    network=network,
                    start_node=node1,
                    end_node=node2,
                    edge_algorithm=edge_algorithm
                ).first()
                
                if existing_edge:
                    if existing_edge.length > distance:
                        # Update if shorter connection found
                        edges_created -= 1
                        total_length -= existing_edge.length
                        existing_edge.length = distance
                        existing_edge.name = unique_name
                        existing_edge.save(update_fields=['length', 'name'])
                else:
                    Edge.objects.create(
                        network=network,
                        start_node=node1,
                        end_node=node2,
                        length=distance,
                        name=unique_name,
                        edge_algorithm=edge_algorithm
                    )
                
                edges_created += 1
                total_length += distance
    
    # Update algorithm metadata
    edge_algorithm.total_edge_length = total_length
    edge_algorithm.save(update_fields=["total_edge_length"])
    
    return edge_algorithm
def connect_by_delaunay(network):
    network = Network.objects.get(pk=network.id)
    nodes = list(Node.objects.filter(network=network))
        # Get coordinates for all nodes
    coords = np.array([(node.latitude, node.longitude) for node in nodes])
    edges_created = 0
    total_length = 0.0
    unique_suffix = str(uuid.uuid4())[:8]
    edge_algorithm = EdgeAlgorithm.objects.create(
        network=network,
        algorithm=EdgeAlgorithm.EdgeAlgorithmType.DELAUNAY,
        parameters=json.dumps({}),
        name=f"{network.name} - Delaunay - {unique_suffix}"
    )
    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i+1)%3]
            if a != b:
                edges.add(frozenset({a, b}))
    for edge in edges:
        a, b = sorted(edge)
        node1 = nodes[a]
        node2 = nodes[b]
        lat1, lon1 = coords[a]
        lat2, lon2 = coords[b]
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        unique_name = f"{node1.name or node1.id}-{node2.name or node2.id}-{a}-{b}"
        if Edge.objects.filter(network=network, start_node=node1, end_node=node2, edge_algorithm=edge_algorithm).exists():
            # check if edge is shorter than existing edge
            existing_edge = Edge.objects.get(network=network, start_node=node1, end_node=node2, edge_algorithm=edge_algorithm)
            if existing_edge.length > distance:
                #replace existing edge with shorter one
                edges_created -= 1  # decrement since we're replacing
                total_length -= existing_edge.length
                existing_edge.length = distance
                existing_edge.name = unique_name
                existing_edge.save(update_fields=['length', 'name'])
        else:
            Edge.objects.create(
                network=network,
                start_node=node1,
                end_node=node2,
                length=distance,
                name=unique_name,
                edge_algorithm=edge_algorithm
            )        
        edges_created += 1
        total_length += distance
    edge_algorithm.total_edge_length = total_length
    edge_algorithm.save(update_fields=["total_edge_length"])
    return edge_algorithm

def connect_by_mst(network):
    """Connect nodes using Minimum Spanning Tree (MST) with similar patterns to Delaunay implementation"""
    network = Network.objects.get(pk=network.id)
    nodes = list(Node.objects.filter(network=network))
    edges_created = 0
    total_length = 0.0
    unique_suffix = str(uuid.uuid4())[:8]
    
    # Create edge algorithm record (consistent with Delaunay)
    edge_algorithm = EdgeAlgorithm.objects.create(
        network=network,
        algorithm=EdgeAlgorithm.EdgeAlgorithmType.MST,
        parameters=json.dumps({}),
        name=f"{network.name} - Steiner MST - {unique_suffix}"
    )
    coords = np.array([(node.latitude, node.longitude) for node in nodes])
    # Create a list of all possible edges with their distances
    edges = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            edges.append((distance, i, j))
    
    # Sort edges by distance (Kruskal's algorithm)
    edges.sort()
    
    # Union-Find data structure for Kruskal's
    parent = list(range(len(nodes)))
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u
    
    # Kruskal's algorithm to find MST
    mst_edges = []
    for dist, i, j in edges:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
            mst_edges.append((i, j, dist))
    
    # Create edges in database following Delaunay pattern
    for i, j, dist in mst_edges:
        node1 = nodes[i]
        node2 = nodes[j]
        unique_name = f"{node1.name or node1.id}-{node2.name or node2.id}-{i}-{j}"
        
        # Check for existing edge (consistent with Delaunay implementation)
        existing_edge = Edge.objects.filter(
            network=network,
            start_node=node1,
            end_node=node2,
            edge_algorithm=edge_algorithm
        ).first()
        
        if existing_edge:
            if existing_edge.length > dist:
                # Replace existing edge with shorter one
                edges_created -= 1
                total_length -= existing_edge.length
                existing_edge.length = dist
                existing_edge.name = unique_name
                existing_edge.save(update_fields=['length', 'name'])
        else:
            Edge.objects.create(
                network=network,
                start_node=node1,
                end_node=node2,
                length=dist,
                name=unique_name,
                edge_algorithm=edge_algorithm
            )
        
        edges_created += 1
        total_length += dist
    
    # Update algorithm metadata (consistent with Delaunay)
    edge_algorithm.total_edge_length = total_length
    edge_algorithm.save(update_fields=["total_edge_length"])
    
    return edge_algorithm
import uuid
import json
import uuid
import json

def connect_by_mst3(edge_algorithm):
    """
    Select the Minimum Spanning Tree (MST) from an existing EdgeAlgorithm's edges, 
    and map the selected edges to a new EdgeConnectionAlgorithm.
    """
    network = edge_algorithm.network
    
    # Fetch existing edges for this specific algorithm, sorted by length
    # Sorting at the database level replaces the manual in-memory sort
    edges = list(Edge.objects.filter(edge_algorithm=edge_algorithm).order_by('length'))
    
    if not edges:
        return None  # Or raise an exception if preferred
        
    unique_suffix = str(uuid.uuid4())[:8]
    
    # Extract unique node IDs directly from the fetched edges 
    # to initialize the Union-Find structure
    node_ids = set()
    for edge in edges:
        node_ids.add(edge.start_node_id)
        node_ids.add(edge.end_node_id)
        
    # Dictionary-based Union-Find since database IDs are not contiguous
    parent = {node_id: node_id for node_id in node_ids}
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u
        
    mst_edges = []
    total_length = 0.0
    
    # Kruskal's algorithm to find MST from EXISTING edges
    for edge in edges:
        root_u = find(edge.start_node_id)
        root_v = find(edge.end_node_id)
        
        # If they are not in the same set, adding this edge doesn't create a cycle
        if root_u != root_v:
            parent[root_v] = root_u
            mst_edges.append(edge)
            total_length += edge.length
            
    # Create the EdgeConnectionAlgorithm record
    edge_connection_algorithm = EdgeConnectionAlgorithm.objects.create(
        network=network,
        edge_algorithm=edge_algorithm,
        algorithm=EdgeConnectionAlgorithm.EdgeConnectionAlgorithmType.MST,
        name=f"{edge_algorithm.name} - MST Connection - {unique_suffix}",
        description=f"MST derived from base algorithm: {edge_algorithm.name}",
        parameters=json.dumps({"source_edge_algorithm_id": edge_algorithm.id}),
        total_edge_length=total_length
    )
    
    # Map the selected edges using EdgeConnectionAlgorithmEdge
    # Using bulk_create is significantly faster than saving them one-by-one in a loop
    connection_edges = [
        EdgeConnectionAlgorithmEdge(
            edge_connection_algorithm=edge_connection_algorithm,
            edge=edge
        )
        for edge in mst_edges
    ]
    EdgeConnectionAlgorithmEdge.objects.bulk_create(connection_edges)
    
    return edge_connection_algorithm
def connect_by_knn(network, k):
    network = Network.objects.get(pk=network_id)
    nodes = list(Node.objects.filter(network=network))
    edges_created = 0
    total_length = 0.0
    unique_suffix = str(uuid.uuid4())[:8]
    edge_algorithm = EdgeAlgorithm.objects.create(
        network=network,
        algorithm=EdgeAlgorithm.EdgeAlgorithmType.KNN,
        parameters=json.dumps({'k': k}),
        name=f"{network.name} - KNN - {unique_suffix}"
    )
    if len(nodes) <= k:
        k = len(nodes) - 1
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    for i, node in enumerate(nodes):
        for neighbor_idx in indices[i][1:]:
            neighbor_node = nodes[neighbor_idx]
            unique_name = f"{node.name or neighbor_node.id}-{neighbor_node.name or node.id}"
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[neighbor_idx]
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            if Edge.objects.filter(network=network, start_node=node, end_node=neighbor_node, edge_algorithm=edge_algorithm).exists():
                    # check if edge is shorter than existing edge
                    existing_edge = Edge.objects.get(network=network, start_node=node, end_node=neighbor_node, edge_algorithm=edge_algorithm)
                    if existing_edge.length > distance:
                        #replace existing edge with shorter one
                        edges_created -= 1  # decrement since we're replacing
                        total_length -= existing_edge.length
                        existing_edge.length = distance
                        existing_edge.name = unique_name
                        existing_edge.save(update_fields=['length', 'name'])
            else:
                Edge.objects.create(
                    network=network,
                    start_node=node,
                    end_node=neighbor_node,
                    length=distance,
                    name=unique_name,
                    edge_algorithm=edge_algorithm
                )
            edges_created += 1
            total_length += distance
    edge_algorithm.total_edge_length = total_length
    edge_algorithm.save(update_fields=["total_edge_length"])
    return edge_algorithm
def update_algorithm_metrics(algorithm, edge_count, total_length):
    algorithm.total_edge_length = total_length
    algorithm.save(update_fields=['total_edge_length'])
def connect_by_beta_skeleton(network, beta=1.5):
    network = Network.objects.get(pk=network_id)
    nodes = list(Node.objects.filter(network=network))
    algorithm = create_algorithm_record(network, EdgeAlgorithm.EdgeAlgorithmType.BETA_SKELETON, {'beta': beta})
    edges_created = 0
    total_length = 0.0
    n = len(nodes)
    
    for i in range(n):
        for j in range(i+1, n):
            a = np.array(coords[i])
            b = np.array(coords[j])
            dist_ab = np.linalg.norm(a - b)
            valid = True
            
            for k in range(n):
                if k == i or k == j:
                    continue
                c = np.array(coords[k])
                if beta == 1:  # Gabriel
                    midpoint = (a + b) / 2
                    if np.linalg.norm(c - midpoint) <= dist_ab / 2:
                        valid = False
                        break
                else:
                    # β-skeleton lune test
                    angle = np.arccos(np.dot(c-a, b-a)/(np.linalg.norm(c-a)*np.linalg.norm(b-a)))
                    if angle < np.pi/2:
                        radius = dist_ab / (2 * beta)
                        center = a + (b-a)/2 + np.sqrt(radius**2 - (dist_ab/2)**2) * np.array([-(b-a)[1], (b-a)[0]])/dist_ab
                        if np.linalg.norm(c - center) <= radius:
                            valid = False
                            break
            
            if valid:
                dist = haversine_distance(*coords[i], *coords[j])
                if create_edge(network, nodes[i], nodes[j], dist, algorithm):
                    edges_created += 1
                    total_length += dist
    
    update_algorithm_metrics(algorithm, edges_created, total_length)
    return algorithm 
def connect_by_rng(network):
    network = Network.objects.get(pk=network_id)
    nodes = list(Node.objects.filter(network=network))

    algorithm = create_algorithm_record(network, EdgeAlgorithm.EdgeAlgorithmType.RNG)
    edges_created = 0
    total_length = 0.0
    n = len(nodes)
    
    for i in range(n):
        for j in range(i+1, n):
            a = coords[i]
            b = coords[j]
            dist_ab = haversine_distance(*a, *b)
            valid = True
            
            for k in range(n):
                if k == i or k == j:
                    continue
                dist_ak = haversine_distance(*a, *coords[k])
                dist_bk = haversine_distance(*b, *coords[k])
                if dist_ak < dist_ab and dist_bk < dist_ab:
                    valid = False
                    break
            
            if valid:
                if create_edge(network, nodes[i], nodes[j], dist_ab, algorithm):
                    edges_created += 1
                    total_length += dist_ab
    
    update_algorithm_metrics(algorithm, edges_created, total_length)
    return algorithm
def connect_by_delaunay2(network):
    """
    Step 1: Build ONLY the Delaunay mesh edges for this network.
    - Uses node coordinates.
    - Saves edges with algorithm = DELAUNAY.
    - Does NOT try to compute any MST or single path.
    """
    nodes = network.nodeList

    if len(nodes) < 3:
        return None  # not enough nodes to connect

    # Get coordinates for all nodes (lat, lon)
    coords = np.array([(node.latitude, node.longitude) for node in nodes])

    edges_created = 0
    total_length = 0.0
    unique_suffix = str(uuid.uuid4())[:8]

    edge_algorithm = EdgeAlgorithm(
        network=network,
        algorithm=EdgeAlgorithm.EdgeAlgorithmType.DELAUNAY,
        parameters=json.dumps({}),
        name=f"{network.name} - Delaunay - {unique_suffix}",
    )
    # Run Delaunay triangulation
    tri = Delaunay(coords)

    # Collect unique undirected edges from triangles
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            if a != b:
                edges.add(frozenset({a, b}))
    network.edges = []
    # Create edges in DB
    for edge in edges:
        a, b = sorted(edge)
        node1 = nodes[a]
        node2 = nodes[b]
        lat1, lon1 = coords[a]
        lat2, lon2 = coords[b]

        distance = haversine_distance(lat1, lon1, lat2, lon2)
        unique_name = f"{node1.name or node1.id}-{node2.name or node2.id}-{a}-{b}"

        # For a clean “mesh only” step, we just create the edge for this algorithm
        e = Edge(
            network=network,
            start_node=node1,
            end_node=node2,
            length=distance,
            name=unique_name,
            edge_algorithm=edge_algorithm,
        )
        network.edges.append(e)

        edges_created += 1
        total_length += distance
    edge_algorithm.total_edge_length = total_length
    network.edgeAlgorithm = edge_algorithm
    return network
def connect_by_mst2(network):
    """
    Step 2: Use existing Delaunay edges as candidates,
    run MST on them, and save only the MST edges
    (single continuous tree through all nodes).
    """
    nodes = network.nodeList

    if len(nodes) < 2:
        return None

    # Map node id -> index and index -> node
    node_id_to_index = {node.id: idx for idx, node in enumerate(nodes)}
    index_to_node = {idx: node for idx, node in enumerate(nodes)}

    # Get all Delaunay edges for this network
    # (you can filter further if you only want edges from a specific run)
    delaunay_edges = network.edges

    # Build candidate edge list: (distance, u_idx, v_idx)
    edges = []
    for e in delaunay_edges:
        u_idx = node_id_to_index[e.start_node_id]
        v_idx = node_id_to_index[e.end_node_id]
        dist = e.length
        edges.append((dist, u_idx, v_idx))

    if not edges:
        return None  # no candidate edges to build an MST from

    # Create MST EdgeAlgorithm record
    unique_suffix = str(uuid.uuid4())[:8]
    edge_algorithm = EdgeConnectionAlgorithm(
        network=network,
        algorithm=EdgeAlgorithm.EdgeAlgorithmType.MST,
        parameters=json.dumps({
            "source": "delaunay_edges"
        })
    )

    # Sort edges by distance (Kruskal's algorithm)
    edges.sort()

    # Union-Find (Disjoint Set) for Kruskal
    parent = list(range(len(nodes)))

    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # path compression
            u = parent[u]
        return u

    mst_edges = []
    k = 0
    for dist, i, j in edges:
        currentEdge = edges[k]
        k += 1
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
            mst_edges.append((i, j, dist, True))
        mst_edges.append((i, j, dist, False))

    # Save MST edges in DB
    edges_created = 0
    total_length = 0.0
    network.edges = []
    for i, j, dist, selected  in mst_edges:
        node1 = index_to_node[i]
        node2 = index_to_node[j]
        unique_name = f"{node1.name or node1.id}-{node2.name or node2.id}-{i}-{j}"

        edge = Edge(
            network_id=network.id,
            start_node=node1,
            end_node=node2,
            length=dist,
            name=unique_name
        )
        if selected:
            edge.edge_connection_algorithm = edge_algorithm
        network.edges.append(edge)
        edges_created += 1
        total_length += dist

    edge_algorithm.total_edge_length = total_length
    network.edge_connection_algorithm = edge_algorithm
    return network

def connect_by_tsp(network):
    network = Network.objects.get(pk=network_id)
    nodes = list(Node.objects.filter(network=network))
    algorithm = create_algorithm_record(network, EdgeAlgorithm.EdgeAlgorithmType.TSP)
    G = nx.Graph()
    
    # Build complete graph
    for i, node in enumerate(nodes):
        G.add_node(i, pos=coords[i])
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            dist = haversine_distance(*coords[i], *coords[j])
            G.add_edge(i, j, weight=dist)
    
    # Christofides approximation
    try:
        tour = nx.approximation.christofides(G)
        edges_created = 0
        total_length = 0.0
        
        for i in range(len(tour)-1): # type: ignore
            dist = G[tour[i]][tour[i+1]]['weight'] # type: ignore
            if create_edge(network, nodes[tour[i]], nodes[tour[i+1]], dist, algorithm): # type: ignore
                edges_created += 1
                total_length += dist
        
        update_algorithm_metrics(algorithm, edges_created, total_length)
        return algorithm
    except:
        return connect_by_mst(nodes, coords, network)  # Fallback

