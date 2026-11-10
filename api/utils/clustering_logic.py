

from api.utils.clustering_algorithms import *
from api.models import (
    Network, Cluster, BuildingCluster, Building, Node, Project, 
    EdgeTranslation, Route, ClusterAlgorithm, EdgeAlgorithm, Edge, 
    ClusterConfiguration, ClusterConfigurationNetwork, 
    EdgeConnectionAlgorithm, EdgeConnectionAlgorithmEdge, LoadProfile
)
 
def run_cluster_algorithm(project_id, algo_type, params):
    """
    Run one clustering experiment for a given project, algorithm type, and parameters.
    
    Args:
        request: Django request object
        project_id: ID of the project
        algo_type: ClusterAlgorithm.ClusterAlgorithmType (enum)
        params: dict of parameters relevant to the algorithm
    """
    result = None


    project = Project.objects.get(pk=project_id)
    buildings = list(Building.objects.filter(project=project))
    # remove buildings with zero heating_demand
    buildings = [b for b in buildings if (b.heating_demand or 0) > 0]

    if len(buildings) < 2:
        return JsonResponse({'status': 'error', 'message': 'Network needs at least 2 nodes'})

    # Dispatch based on algorithm type
    if algo_type == ClusterAlgorithm.ClusterAlgorithmType.DBSCAN:
        result = cluster_buildings_with_dbscan(buildings,
                                eps=params.get('epsilon', 0.5)
                                )
    elif algo_type == ClusterAlgorithm.ClusterAlgorithmType.RADIUS_BASED:
        result = cluster_buildings_by_radius(buildings,"heating_demand", "max",
                                radius_km=params.get('radius_km', 0.5)
                                )

    elif algo_type == ClusterAlgorithm.ClusterAlgorithmType.OPTICS:
        result = cluster_optics(buildings,
                                max_radius_km=params.get('radius_km', 0.5),
                                min_samples=params.get('min_samples', 2))

    elif algo_type == ClusterAlgorithm.ClusterAlgorithmType.KMEANS:
        result = cluster_kmeans(buildings,
                                n_clusters=params.get('n_clusters', 5))

    elif algo_type == ClusterAlgorithm.ClusterAlgorithmType.HIERARCHICAL:
        result = cluster_hierarchical(buildings,
                                        n_clusters=params.get('n_clusters', 5),
                                        linkage=params.get('linkage', 'average'))

    elif algo_type == ClusterAlgorithm.ClusterAlgorithmType.GMM:
        result = cluster_gmm(buildings,
                                n_clusters=params.get('n_clusters', 5))

    elif algo_type == ClusterAlgorithm.ClusterAlgorithmType.SPECTRAL:
        result = cluster_spectral(buildings,
                                    n_clusters=params.get('n_clusters', 5))

    else:
        result = cluster_mst(buildings,
                                n_clusters=params.get('n_clusters', 5))

    return result

def create_network_from_clusters2(project, clusters, algorithmType=None, parameters=None):
    """
    Create a Network with Clusters and Nodes from building clusters.

    Args:
        project (Project): The project this network belongs to.
        clusters (list[list[Building]]): 2D array of building clusters.
        network_name (str): Name of the network.

    Returns:
        Network: The created network instance.
    """

    #create network algorithm record
    algorithm = ClusterAlgorithm(
        name=f"{algorithmType}  based Clustering for project {project.name}, id {project.id}",
        description=f"Auto-generated cluster from project {project.name}, id {project.id}",
        parameters= parameters,
        algorithm_type=algorithmType
    )

    # Create the network
    network = Network(
        project=project,
        name=f"network from clustered buildings with algorithm "+ str(algorithm.id),
        description=f"Auto-generated network from project {project.name}, id {project.id} and algorithm {algorithm} {algorithm.name} and id {algorithm.id}",
        network_type="district_heating",  # could refine later
        total_length=0.0,
        cluster_algorithm=algorithm,
        id = 1
    )
    i = 0
    network.clusterList = []
    network.nodeList = []
    clusterIdx = 1
    nodeIdx = 1

    for item in enumerate(clusters):

        building_group = item[1]
        idx = item[0] + 1
        if not building_group:
            continue
        # Connect buildings in this cluster using mst
        #total_edge_length = mst_total_length(idx, building_group)
        number = idx
        # Aggregate demands
        total_heating = sum(b.heating_demand or 0 for b in building_group)
        total_cooling = sum(b.cooling_demand or 0 for b in building_group)

        # Midpoint of cluster
        latitudes = [b.latitude for b in building_group if b.latitude is not None]
        longitudes = [b.longitude for b in building_group if b.longitude is not None]
        cluster_lat = mean(latitudes) if latitudes else None
        cluster_lon = mean(longitudes) if longitudes else None
        # Create cluster
        cluster = Cluster(
            network=network,
            name=f"Cluster {number}",
            description=f"Cluster {number} with {len(building_group) + 1} buildings",
            type="distance_based",
            heating_demand=total_heating,
            cooling_demand=total_cooling,
            latitude=cluster_lat,
            longitude=cluster_lon,
            total_edge_length=0,
            effective_radius = 0,
            cluster_algorithm_id=algorithm.id,
            no_of_buildings=len(building_group),
            id = clusterIdx
        )
       
        clusterIdx += 1
        # Link buildings to cluster
        cluster.buildingList = []
        for b in building_group:
            cluster.buildingList.append(b.id)
        network.clusterList.append(cluster)

        # Determine node type
        if total_cooling > total_heating:
            node_type = "producer"   # cooling-dominant cluster
        else:
            node_type = "consumer"   # heating-dominant cluster

        # Create node for this cluster
        node = Node(
            network=network,
            name=f"Node {nodeIdx}",
            node_type=node_type,
            cluster=cluster,
            latitude=cluster_lat,
            longitude=cluster_lon,
            is_default=False,
            id = nodeIdx
        )
        nodeIdx += 1
        network.nodeList.append(node)

    return network
