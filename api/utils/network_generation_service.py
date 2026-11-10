from django.db import transaction
from api.models import (
    Network, Node, Cluster, ClusterConfiguration, 
    BuildingCluster, Project
)

class NetworkGenerationService:

    def create_network_from_configuration(self, config_id, network_name=None):
        """
        MACRO NETWORK: Creates a network where each NODE is a CLUSTER.
        Used for designing the backbone piping between cluster centers.
        """
        try:
            config = ClusterConfiguration.objects.get(id=config_id)
        except ClusterConfiguration.DoesNotExist:
            return {"status": "error", "message": "Configuration not found"}

        # Trace Project from the algorithm used
        project = config.cluster_algorithm.project
        
        if not network_name:
            network_name = f"Macro Network: {config.name}"

        with transaction.atomic():
            # 1. Create the Macro Network
            network = Network.objects.create(
                project=project,
                name=network_name,
                description=f"Inter-cluster network generated from Config: {config.name}",
                network_type='combined' 
            )

            # 2. Iterate over Clusters (NOT Buildings)
            clusters = Cluster.objects.filter(cluster_configuration=config)
            nodes_created = []

            for cluster in clusters:
                # Determine Node Type based on Balance
                # If the cluster has excess heat, it acts as a Producer for the macro grid
                # If it has a deficit, it acts as a Consumer
                if cluster.energy_balance_status == 'excess_heat':
                    node_role = 'producer'
                elif cluster.energy_balance_status == 'heat_deficit':
                    node_role = 'consumer'
                else:
                    node_role = 'cluster' # Default generic type if balanced or unknown

                # Create the Node representing the Cluster
                node = Node.objects.create(
                    network=network,
                    name=f"Node: {cluster.name}",
                    node_type=node_role, 
                    cluster=cluster, # Link the node to the cluster object
                    latitude=cluster.latitude, # Centroid Lat
                    longitude=cluster.longitude, # Centroid Lon
                    is_default=False
                )
                nodes_created.append(node)

            return {
                "status": "success",
                "network_id": network.id,
                "network_name": network.name,
                "type": "macro_inter_cluster",
                "nodes_created": len(nodes_created)
            }

    def create_network_from_single_cluster(self, cluster_id, network_name=None):
        """
        MICRO NETWORK: Creates a network where nodes are BUILDINGS within one cluster.
        Used for designing the local piping inside a specific cluster.
        """
        try:
            cluster = Cluster.objects.get(id=cluster_id)
        except Cluster.DoesNotExist:
            return {"status": "error", "message": "Cluster not found"}

        project = cluster.cluster_configuration.cluster_algorithm.project

        if not network_name:
            network_name = f"Micro Network: {cluster.name}"

        with transaction.atomic():
            # 1. Create the Micro Network
            network = Network.objects.create(
                project=project,
                name=network_name,
                description=f"Intra-cluster network for Cluster {cluster.id}",
                network_type='combined'
            )

            # 2. Iterate over Buildings in this Cluster
            building_links = BuildingCluster.objects.filter(cluster=cluster).select_related('building')
            nodes_created = []

            for link in building_links:
                b = link.building
                
                # Determine Node Type based on Building attributes
                is_producer = (
                    b.waste_heat_source_type is not None or 
                    b.load_type == 'cooling_dominant'
                )
                node_role = 'producer' if is_producer else 'consumer'

                # Create Node representing the Building
                node = Node.objects.create(
                    network=network,
                    name=f"Node: {b.name}",
                    node_type=node_role,
                    building=b,      # Link to the building
                    cluster=cluster, # Keep reference to parent cluster
                    latitude=b.latitude,
                    longitude=b.longitude,
                    is_default=False
                )
                nodes_created.append(node)

            return {
                "status": "success",
                "network_id": network.id,
                "network_name": network.name,
                "type": "micro_intra_cluster",
                "nodes_created": len(nodes_created)
            }