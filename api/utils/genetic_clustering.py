import numpy as np
import random
import pandas as pd
import time
import logging
import json
from itertools import combinations, product

from django.db.models import Sum, Q
from django.apps import apps
from scipy.spatial.distance import pdist, cdist
from django.db import transaction
from scipy.spatial.distance import pdist
# --- LOCAL IMPORTS ---
# Ensure these match your actual file structure
from api.utils.node_connection_algorithms import connect_by_delaunay2 
from api.utils.google_routing import translate2
from api.models import (
    Network, Cluster, BuildingCluster, Building, Node, 
    EdgeTranslation, Route, ClusterAlgorithm, EdgeAlgorithm, Edge, 
    ClusterConfiguration, ClusterConfigurationNetwork, 
    EdgeConnectionAlgorithm, EdgeConnectionAlgorithmEdge, LoadProfile
)
from api.utils.distance_calculations import haversine_distance

# --- LOGGING ---
logger = logging.getLogger('GeneticClustering')
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# --- CONFIG ---
POPULATION_SIZE = 10
GENERATIONS = 50
MUTATION_RATE = 0.1
ELITISM_COUNT = 2
MIN_VAL = 0.0
MAX_VAL = 100.0
MAX_MERGE_DISTANCE_KM = 1.0 

# =============================================================================
# 1. BULK DATA FETCHING & METRICS
# =============================================================================

def prefetch_data(network_instance):
    t_start = time.time()
    initial_clusters = network_instance.clusterList
    if not initial_clusters: return None, None, None, []

    cluster_map = {}
    all_building_ids = set()
    cluster_meta = [] 

    for c in initial_clusters:
        b_ids = c.buildingList
        cluster_map[c.id] = b_ids
        all_building_ids.update(b_ids)
        cluster_meta.append((c.id, c.latitude, c.longitude))

    buildings_qs = Building.objects.filter(id__in=all_building_ids).values(
        'id', 'latitude', 'longitude', 'building_type'
    )
    buildings_df = pd.DataFrame(list(buildings_qs))
    if not buildings_df.empty:
        buildings_df.set_index('id', inplace=True)

    profiles_qs = LoadProfile.objects.filter(
        building_id__in=all_building_ids,
        profile_type__in=['heating', 'waste_heat']
    ).values('building_id', 'start_date', 'value', 'profile_type')
    
    loads_df = pd.DataFrame(list(profiles_qs))
    
    return buildings_df, loads_df, cluster_map, cluster_meta


def calc_metrics_fast(building_ids, buildings_df, loads_df):
    """
    Calculates cluster metrics aligned with Marquant et al. (2017).
    
    Returns:
        DY (float): Mean Distance (km). Lower = Denser. 
        HI (float): Heterogeneity (1 - Homogeneity Index). Higher = More Diverse.
        MAG (float): Magnitude (Total Heating Demand in kWh/year).
        WH (float): Waste Heat Potential (Total Waste Heat in kWh/year).
    """
    if not building_ids: 
        return 0.0, 0.0, 0.0, 0.0

    # Filter for valid buildings in this cluster
    valid_ids = buildings_df.index.intersection(building_ids)
    if len(valid_ids) == 0: 
        return 0.0, 0.0, 0.0, 0.0
    
    b_subset = buildings_df.loc[valid_ids]
    
    # Initialize metrics
    MAG = 0.0
    WH = 0.0
    HI = 0.0
    
    if not loads_df.empty:
        # --- 1. MAGNITUDE [MAG] ---
        # Filter strictly for heating loads
        heating_loads = loads_df[
            (loads_df['building_id'].isin(building_ids)) & 
            (loads_df['profile_type'] == 'heating')
        ]
        
        if not heating_loads.empty:
            MAG = float(heating_loads['value'].sum())
            
            # --- 3. HOMOGENEITY INDEX [HI] (Heating Only) ---
            if len(building_ids) >= 2:
                # Pivot to shape: (Time_Steps x Buildings)
                pivot = heating_loads.pivot_table(index='start_date', columns='building_id', values='value')
                
                if pivot.shape[1] >= 2:
                    # Calculate correlation matrix
                    corr_matrix = pivot.corr().values
                    
                    # Extract upper triangle (excluding diagonal self-correlation)
                    upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                    
                    if len(upper_tri) > 0:
                        HI = float(np.nanmean(upper_tri))
                        # Ensure HI is non-negative
                        HI = max(0.0, HI)

        # --- 1b. WASTE HEAT [WH] ---
        wh_loads = loads_df[
            (loads_df['building_id'].isin(building_ids)) & 
            (loads_df['profile_type'] == 'waste_heat')
        ]
        if not wh_loads.empty:
            WH = float(wh_loads['value'].sum())

    # --- 2. SPATIAL DENSITY [DY] (Mean Distance) ---
    # The paper defines DY explicitly as "mean distance".
    # Calculation uses Minkowski Euclidean distance[cite: 195].
    coords = b_subset[['latitude', 'longitude']].dropna().values
    
    DY = 0.0
    if len(coords) < 2:
        # A single building has a mean internal distance of 0.
        DY = 0.0 
    else:
        # Calculate pairwise Euclidean distances
        dists = pdist(coords, metric='euclidean')
        
        # Convert degrees to km (Approx 111km per degree)
        avg_dist_deg = np.mean(dists)
        avg_dist_km = float(avg_dist_deg) * 111.0
        
        # Return Mean Distance directly (Lower value = Denser cluster)
        DY = avg_dist_km 

    # Invert HI to return Heterogeneity (1 - HI) for the scoring function
    # Note: The paper uses (1 - HI) in the score formula [cite: 273]
    HI = 1.0 - HI 

    return DY, HI, MAG, WH
def _local_calculate_mst_edges(network):
    """
    Selects MST edges from the network's existing Delaunay edges.
    Returns list of (u_idx, v_idx, distance, edge_obj)
    """
    nodes = network.nodeList
    delaunay_edges = network.edges
    
    node_id_map = {n.id: i for i, n in enumerate(nodes)}
    candidates = []
    
    for e in delaunay_edges:
        try:
            u = node_id_map[e.start_node.id]
            v = node_id_map[e.end_node.id]
            candidates.append((e.length, u, v, e))
        except KeyError: continue
            
    candidates.sort(key=lambda x: x[0])
    
    parent = list(range(len(nodes)))
    def find(i):
        if parent[i] != i: parent[i] = find(parent[i])
        return parent[i]
    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_j] = root_i
            return True
        return False
        
    mst_selection = []
    for dist, u, v, edge_obj in candidates:
        if union(u, v):
            mst_selection.append((u, v, dist, edge_obj))
            
    return mst_selection

def get_cluster_physical_length(building_ids):
    try:
        temp_network = Network(project_id=1, name="CalcTemp")
        temp_network.nodeList = []
        buildings = Building.objects.filter(id__in=building_ids)
        for i, b in enumerate(buildings):
            temp_network.nodeList.append(Node(latitude=b.latitude, longitude=b.longitude, id=i+1, building=b))
        if len(temp_network.nodeList) < 2: return 0.0
        
        # 1. Delaunay
        connect_by_delaunay2(temp_network)
        # 2. MST
        mst_edges = _local_calculate_mst_edges(temp_network)
        total_len = sum(dist for _, _, dist, _ in mst_edges)
        
        return total_len * 1.3 # 1.3 tortuosity factor
    except Exception:
        return 0.0

# =============================================================================
# 3. GENETIC ALGORITHM CORE
# =============================================================================
import numpy as np
from scipy.spatial.distance import cdist

def precompute_candidates(cluster_map, cluster_meta, buildings_df, loads_df):
    """
    Generates candidate clusters and normalizes their metrics for the Genetic Algorithm.
    """
    cache = []
    
    # --- PHASE 1: Generate Raw Candidates ---
    
    # 1. Singletons
    for c_info in cluster_meta:
        c_id, b_ids = c_info[0], cluster_map[c_info[0]]
        # metrics = (DY, HI, MAG, WH)
        metrics = calc_metrics_fast(b_ids, buildings_df, loads_df)
        length = get_cluster_physical_length(b_ids)
        cache.append({'type': 'singleton', 'c_ids': [c_id], 'b_ids': b_ids, 'metrics': metrics, 'length': length})

    # 2. Pairs
    coords = np.array([(c[1], c[2]) for c in cluster_meta])
    # Calculate distances between cluster centroids
    dist_matrix_km = cdist(coords, coords, metric='euclidean') * 111.0 if len(coords) > 1 else np.zeros((1,1))
    
    N = len(cluster_meta)
    for i in range(N):
        for j in range(i + 1, N):
            # Only consider merging if centroids are close enough
            if dist_matrix_km[i, j] < MAX_MERGE_DISTANCE_KM:
                c_ids = [cluster_meta[i][0], cluster_meta[j][0]]
                b_ids = []
                for cid in c_ids: b_ids.extend(cluster_map[cid])
                
                metrics = calc_metrics_fast(b_ids, buildings_df, loads_df)
                length = get_cluster_physical_length(b_ids)
                cache.append({'type': 'merge_2', 'c_ids': c_ids, 'b_ids': b_ids, 'metrics': metrics, 'length': length})

    # --- PHASE 2: Global Normalization ---
    # We must normalize relative to the entire population of candidates to make scores comparable.
    
    if not cache:
        return []

    # Extract all metrics into a numpy array for vectorized min/max calculation
    # Shape: (N_candidates, 4) -> Columns: DY, HI, MAG, WH
    all_metrics = np.array([c['metrics'] for c in cache])

    # Calculate Min and Max for each column
    mins = all_metrics.min(axis=0)
    maxs = all_metrics.max(axis=0)
    
    # Avoid division by zero (if max == min, range is 0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0 

    # Apply Normalization
    # 1. DY (Mean Distance): Lower is Better. Invert: (Max - Val) / Range
    norm_DY = (maxs[0] - all_metrics[:, 0]) / ranges[0]
    
    # 2. HI (Heterogeneity): Higher is Better. (Val - Min) / Range
    norm_HI = (all_metrics[:, 1] - mins[1]) / ranges[1]
    
    # 3. MAG (Magnitude): Higher is Better. (Val - Min) / Range
    norm_MAG = (all_metrics[:, 2] - mins[2]) / ranges[2]
    
    # 4. WH (Waste Heat): Higher is Better. (Val - Min) / Range
    norm_WH = (all_metrics[:, 3] - mins[3]) / ranges[3]

    # --- PHASE 3: Inject Normalized Values ---
    # Update the cache dictionaries with the new normalized tuple
    for i, candidate in enumerate(cache):
        candidate['norm_metrics'] = (
            float(norm_DY[i]), 
            float(norm_HI[i]), 
            float(norm_MAG[i]), 
            float(norm_WH[i])
        )

    return cache
def solve_partition_fast(weights, candidates_cache, singleton_map):
    """
    Greedy Iterative Solver.
    Finds ALL non-overlapping merges that improve score.
    """
    w_a, w_b, w_c, w_d = weights
    
    # 1. Calculate scores
    for cand in candidates_cache:
        D, HI, MAG, WH = cand['metrics']
        cand['_score'] = (w_a * D) + (w_b * (1.0 - HI)) + (w_c * MAG) + (w_d * WH)

    # 2. Setup Baseline
    available_cids = set(singleton_map.keys())
    current_length = sum(singleton_map[cid]['length'] for cid in available_cids)
    
    selected_merges = []
    
    # 3. Sort candidates by score (Highest First)
    sorted_candidates = sorted(
        [c for c in candidates_cache if c['type'] != 'singleton'], 
        key=lambda x: x['_score'], 
        reverse=True
    )

    # 4. Iterate and Merge
    for cand in sorted_candidates:
        cand_cids = set(cand['c_ids'])
        
        # Check if parts are still available
        if not cand_cids.issubset(available_cids):
            continue

        # Check Improvement
        parts_score = sum(singleton_map[cid]['_score'] for cid in cand['c_ids'])
        
        if cand['_score'] > parts_score:
            # Apply Merge
            available_cids.difference_update(cand_cids)
            selected_merges.append(cand)
            
            # Update Length
            parts_len = sum(singleton_map[cid]['length'] for cid in cand['c_ids'])
            current_length = current_length - parts_len + cand['length']

    # 5. Result Package
    # We pack all selections into a "combined candidate" object
    combined_cand = {
        'c_ids': [cid for m in selected_merges for cid in m['c_ids']],
        'b_ids': [bid for m in selected_merges for bid in m['b_ids']],
        'sub_groups': selected_merges # Critical for reconstruction
    }
    
    if not selected_merges:
        return current_length, None
        
    return current_length, combined_cand

def optimize_weights_fast(candidates_cache, active_mask):
    singleton_map = {c['c_ids'][0]: c for c in candidates_cache if c['type'] == 'singleton'}
    population = [[random.uniform(MIN_VAL, MAX_VAL) for _ in range(4)] for _ in range(POPULATION_SIZE)]
    best_sol = {"length": float('inf'), "weights": [], "cand": None}

    for gen in range(GENERATIONS):
        scored = []
        for ind in population:
            w = [val * mask for val, mask in zip(ind, active_mask)]
            length, cand = solve_partition_fast(w, candidates_cache, singleton_map)
            scored.append((length, ind, cand))
            if length < best_sol['length']: best_sol = {"length": length, "weights": w, "cand": cand}

        scored.sort(key=lambda x: x[0])
        next_pop = [s[1] for s in scored[:ELITISM_COUNT]]
        while len(next_pop) < POPULATION_SIZE:
            p1 = sorted(random.sample(scored, 3), key=lambda x: x[0])[0][1]
            p2 = sorted(random.sample(scored, 3), key=lambda x: x[0])[0][1]
            idx = random.randint(0, 3)
            child = p1[:random.randint(1, 3)] + p2[random.randint(1, 3):]
            if len(child) != 4: child = p1 # Fallback
            
            if random.random() < MUTATION_RATE:
                child[idx] = max(MIN_VAL, min(child[idx] + random.uniform(-5, 5), MAX_VAL))
            next_pop.append(child)
        population = next_pop
    return best_sol

def run_experiment_combinations(network_instance):
    logger.info("=== STARTING GENETIC CLUSTERING EXPERIMENT ===")
    b_df, l_df, c_map, c_meta = prefetch_data(network_instance)
    if b_df is None or b_df.empty: return []

    candidates = precompute_candidates(c_map, c_meta, b_df, l_df)
    results = []
    metric_names = ["Density", "Heterogeneity", "Magnitude", "WasteHeat"]
    all_masks = list(product([0, 1], repeat=4))

    for mask in all_masks:
        if sum(mask) == 0: continue
        active_names = [metric_names[i] for i, m in enumerate(mask) if m]
        exp_name = "GA_" + "_".join(active_names)
        
        logger.info(f"--- Running {exp_name} | Mask: {mask} ---")
        best = optimize_weights_fast(candidates, list(mask))
        
        # --- RECONSTRUCT PARTITION (HANDLES ITERATIVE MERGES) ---
        final_groups = []
        merged_cids = set()
        
        if best['cand']:
            # Handle multiple merges if present
            merges = best['cand'].get('sub_groups', [best['cand']])
            for m in merges:
                final_groups.append({'c_ids': m['c_ids'], 'b_ids': m['b_ids']})
                merged_cids.update(m['c_ids'])
        
        # Add remaining singletons
        for c in c_meta:
            if c[0] not in merged_cids:
                final_groups.append({'c_ids': [c[0]], 'b_ids': c_map[c[0]]})

        # Weights Display
        raw_w = best['weights']
        total_w = sum(raw_w)
        final_w = [round(w/total_w, 4) if total_w > 0 else 0.0 for w in raw_w]

        logger.info(f"   > Best Length: {best['length']:.2f}")
        logger.info(f"   > Optimized Weights: {final_w}")

        save_optimal_configuration(network_instance.project_id, final_groups, exp_name, f"Weights: {final_w}")
        results.append({"experiment": exp_name, "length": best['length'], "weights": final_w})

    return results

# =============================================================================
# 4. SAVING LOGIC (DELAUNAY -> MST -> CLONE -> ROUTE)
# =============================================================================
from django.db import transaction #
# ... other imports

def save_optimal_configuration(project_id, partition_data, network_name, description):
    """
    Saves the final network to DB atomically.
    Arguments:
      partition_data: A LIST of dicts [{'c_ids': [...], 'b_ids': [...]}, ...]
    """
    if not partition_data:
        print(f"Skipping save for {network_name}: No data")
        return None

    try:
        # Start a database transaction
        with transaction.atomic():
            # 1. Create Base Network Objects
            algo = ClusterAlgorithm.objects.create(
                project_id=project_id, name=f"Algo-{network_name}", 
                description=description, algorithm_type='optics', methodology='genetic'
            )
            config = ClusterConfiguration.objects.create(
                cluster_algorithm=algo, name=f"Config-{network_name}", description=description
            )
            net = Network.objects.create(
                project_id=project_id, name=network_name, network_type='district_heating'
            )
            ClusterConfigurationNetwork.objects.create(
                cluster_configuration=config, network=net
            )

            # 2. Reconstruct Nodes/Clusters
            net.nodeList = []
            
            # Flatten all building IDs to fetch them in one query
            all_b_ids = [bid for group in partition_data for bid in group['b_ids']]
            b_objs = {b.id: b for b in Building.objects.filter(id__in=all_b_ids)}

            for i, group in enumerate(partition_data):
                # group is {'c_ids': [...], 'b_ids': [...]}
                buildings_ids = group['b_ids']
                
                # Get building objects for this cluster
                b_list = [b_objs[bid] for bid in buildings_ids if bid in b_objs]
                if not b_list: continue
                
                # Calculate Centroid
                avg_lat = sum(b.latitude for b in b_list) / len(b_list)
                avg_lon = sum(b.longitude for b in b_list) / len(b_list)

                # Create Cluster
                cluster = Cluster.objects.create(
                    network=net, name=f"Cluster {i+1}", latitude=avg_lat, longitude=avg_lon,
                    no_of_buildings=len(b_list), cluster_configuration=config
                )
                
                # Create Node
                node = Node.objects.create(
                    network=net, cluster=cluster, latitude=avg_lat, longitude=avg_lon,
                    name=f"Node {i+1}"
                )
                net.nodeList.append(node)

                # Link Buildings
                for b in b_list:
                    BuildingCluster.objects.create(building_id=b.id, cluster=cluster)

            # 3. EDGE GENERATION & CONNECTION (Standard Workflow)
            
            # A. Generate Mesh (Delaunay)
            connect_by_delaunay2(net) # Assumes this populates net.edges in-memory
            
            mst_algo = EdgeAlgorithm.objects.create(
                network=net, 
                name=f"MST-{network_name}", 
                algorithm=EdgeAlgorithm.EdgeAlgorithmType.DELAUNAY,
                is_default=True,
                total_edge_length = getattr(net.edgeAlgorithm, 'total_edge_length', 0)
            )
            
            created_edges = {}
            for edge in net.edges:
                e = Edge.objects.create(
                    network=net,
                    start_node=edge.start_node,
                    end_node=edge.end_node,
                    length=edge.length,
                    name=f"MST-{edge.start_node.id}-{edge.end_node.id}",
                    edge_algorithm=mst_algo 
                )
                created_edges[e.name] = e

            mst_conn = EdgeConnectionAlgorithm.objects.create(
                network=net,
                edge_algorithm=mst_algo,
                name=f"Conn-{network_name}",
                algorithm=EdgeConnectionAlgorithm.EdgeConnectionAlgorithmType.MST,
                is_default=True
            )

            # B. Select Tree (MST)
            mst_selection = _local_calculate_mst_edges(net)
            final_edges = []
            
            for u, v, dist, original_edge in mst_selection:
                lookup_key = f"MST-{original_edge.start_node.id}-{original_edge.end_node.id}"
                equiv_edge = created_edges[lookup_key]
                
                EdgeConnectionAlgorithmEdge.objects.create(
                    edge_connection_algorithm=mst_conn,
                    edge=equiv_edge
                )
                final_edges.append(equiv_edge)

            # Temporarily point network to MST edges for translation
            original_edges_ref = net.edges
            net.edges = final_edges 
            net.edge_connection_algorithm = mst_conn

            # 4. ROUTING / TRANSLATION
            trace = translate2(net, 'driving') # External call (OSRM/Google) happens here
            
            trans_obj = EdgeTranslation.objects.create(
                translation='driving', name=f"Trans-{network_name}", is_default=True,
                edge_connection_algorithm=mst_conn,
                total_edge_length = trace.get('total_edge_length')
            )
            
            net.edges = original_edges_ref # Restore original edges reference

            for r_data in trace.get('routes', []):
                matching_edge = next((e for e in final_edges if e.id == r_data.edge.id), None)
                
                if matching_edge:
                    Route.objects.create(
                        edge=matching_edge, 
                        translation=trans_obj,
                        polyline=r_data.polyline,
                        distance_meters=r_data.distance_meters,
                        duration_seconds=r_data.duration_seconds,
                        route_type="supply" if r_data.route_type == 'supply' else "return"
                    )

            logger.info(f"[Save] {network_name}: Saved {len(final_edges)} edges.")

    except Exception as e:
        logger.error(f"[Save] Error saving {network_name}: {e}")
        # Re-raise to ensure the transaction rolls back
        raise e
def save_optimal_configuration2(project_id, partition_data, network_name, description):
    """
    Saves the final network to DB.
    Arguments:
      partition_data: A LIST of dicts [{'c_ids': [...], 'b_ids': [...]}, ...]
    """
    if not partition_data:
        print(f"Skipping save for {network_name}: No data")
        return None

    try:
        # 1. Create Base Network Objects
        algo = ClusterAlgorithm.objects.create(
            project_id=project_id, name=f"Algo-{network_name}", 
            description=description, algorithm_type='optics', methodology='genetic'
        )
        config = ClusterConfiguration.objects.create(
            cluster_algorithm=algo, name=f"Config-{network_name}", description=description
        )
        net = Network.objects.create(
            project_id=project_id, name=network_name, network_type='district_heating'
        )
        ClusterConfigurationNetwork.objects.create(
            cluster_configuration=config, network=net
        )

        # 2. Reconstruct Nodes/Clusters
        net.nodeList = []
        
        # Flatten all building IDs to fetch them in one query
        all_b_ids = [bid for group in partition_data for bid in group['b_ids']]
        b_objs = {b.id: b for b in Building.objects.filter(id__in=all_b_ids)}

        # --- FIX: Iterate directly over the list ---
        for i, group in enumerate(partition_data):
            # group is {'c_ids': [...], 'b_ids': [...]}
            buildings_ids = group['b_ids']
            
            # Get building objects for this cluster
            b_list = [b_objs[bid] for bid in buildings_ids if bid in b_objs]
            if not b_list: continue
            
            # Calculate Centroid
            avg_lat = sum(b.latitude for b in b_list) / len(b_list)
            avg_lon = sum(b.longitude for b in b_list) / len(b_list)

            # Create Cluster
            cluster = Cluster.objects.create(
                network=net, name=f"Cluster {i+1}", latitude=avg_lat, longitude=avg_lon,
                no_of_buildings=len(b_list), cluster_configuration=config
            )
            
            # Create Node
            node = Node.objects.create(
                network=net, cluster=cluster, latitude=avg_lat, longitude=avg_lon,
                name=f"Node {i+1}"
            )
            net.nodeList.append(node)

            # Link Buildings
            for b in b_list:
                BuildingCluster.objects.create(building_id=b.id, cluster=cluster)

        # 3. EDGE GENERATION & CONNECTION (Standard Workflow)
        
        
      
        # A. Generate Mesh (Delaunay)
        connect_by_delaunay2(net)
        mst_algo = EdgeAlgorithm.objects.create(
            network=net, 
            name=f"MST-{network_name}", 
            algorithm=EdgeAlgorithm.EdgeAlgorithmType.DELAUNAY,
            is_default=True,
            total_edge_length = net.edgeAlgorithm.total_edge_length
        )
        created_edges = {}
        for edge in net.edges:
            e = Edge.objects.create(
                network=net,
                start_node=edge.start_node,
                end_node=edge.end_node,
                length=edge.length,
                name=f"MST-{edge.start_node.id}-{edge.end_node.id}",
                edge_algorithm=mst_algo 
            )
            created_edges[e.name] = e

        mst_conn = EdgeConnectionAlgorithm.objects.create(
            network=net,
            edge_algorithm=mst_algo,
            name=f"Conn-{network_name}",
            algorithm=EdgeConnectionAlgorithm.EdgeConnectionAlgorithmType.MST,
            is_default=True
        )
        # B. Select Tree (MST)
        mst_selection = _local_calculate_mst_edges(net)
        final_edges = []
        for u, v, dist, original_edge in mst_selection:
            equiv_edge = created_edges[original_edge.name]
            #new_edge = Edge.objects.create(
                #network=net,
                #start_node=original_edge.start_node,
                #end_node=original_edge.end_node,
                #length=dist,
                #name=f"MST-{original_edge.start_node.id}-{original_edge.end_node.id}"
            #)
            EdgeConnectionAlgorithmEdge.objects.create(
                edge_connection_algorithm=mst_conn,
                edge=equiv_edge
            )
            final_edges.append(equiv_edge)

      

        # Temporarily point network to MST edges for translation
        original_edges_ref = net.edges
        net.edges = final_edges 
        net.edge_connection_algorithm = mst_conn

        trace = translate2(net, 'driving')
          # 4. ROUTING
        trans_obj = EdgeTranslation.objects.create(
            translation='driving', name=f"Trans-{network_name}", is_default=True,
            edge_connection_algorithm=mst_conn,
            total_edge_length = trace.get('total_edge_length')
        )
        net.edges = original_edges_ref # Restore

        for r_data in trace.get('routes', []):
            matching_edge = next((e for e in final_edges if e.id == r_data.edge.id), None)
            
            if matching_edge:
                Route.objects.create(
                    edge=matching_edge, 
                    translation=trans_obj,
                    polyline=r_data.polyline,
                    distance_meters=r_data.distance_meters,
                    duration_seconds=r_data.duration_seconds,
                    route_type="supply" if r_data.route_type == 'supply' else "return"
                )

        logger.info(f"[Save] {network_name}: Saved {len(final_edges)} edges.")

    except Exception as e:
        logger.error(f"[Save] Error saving {network_name}: {e}")
        # Re-raise to see trace in logs if needed
        raise e