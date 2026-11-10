import numpy as np
from django.db import transaction
from django.db.models import F
from api.models import (
    Cluster, ClusterConfiguration, BuildingCluster, 
    Building, Network, Node, Edge, Boiler  # Assuming you have an Edge model
)

class ClusterBalancingService:
    """
    Algorithms to balance 'Deficit' clusters by modifying the configuration
    or adding new infrastructure.
    """

    def __init__(self, config_id):
        self.config_id = config_id
        try:
            self.config = ClusterConfiguration.objects.get(id=config_id)
        except ClusterConfiguration.DoesNotExist:
            raise ValueError("Configuration not found")

    # =========================================================================
    # STRATEGY 1: REARRANGE NODES (Load Shifting)
    # =========================================================================
    def balance_by_shifting_loads(self):
        """
        MOVES buildings from a 'Deficit' cluster to the nearest 'Excess' cluster.
        This physically changes the cluster composition.
        """
        with transaction.atomic():
            # 1. Identify Players
            deficit_clusters = list(Cluster.objects.filter(
                cluster_configuration=self.config, 
                energy_balance_status='heat_deficit'
            ))
            excess_clusters = list(Cluster.objects.filter(
                cluster_configuration=self.config, 
                energy_balance_status='excess_heat'
            ))

            if not deficit_clusters or not excess_clusters:
                return {"status": "info", "message": "No balancing opportunities found."}

            moves_log = []

            # 2. Iterate through Deficit Clusters
            for d_cluster in deficit_clusters:
                # Calculate how much load we need to shed (Deficit Amount)
                current_deficit = d_cluster.total_sink_capacity - d_cluster.total_source_capacity
                
                if current_deficit <= 0: continue # Already fixed

                # 3. Find Nearest Excess Cluster
                nearest_excess = self._find_nearest_cluster(d_cluster, excess_clusters)
                if not nearest_excess: continue

                # Check excess capacity
                available_excess = nearest_excess.total_source_capacity - nearest_excess.total_sink_capacity
                if available_excess <= 0: continue

                # 4. Find Candidate Buildings to Move
                # We want buildings in the Deficit Cluster that are CLOSEST to the Excess Cluster
                # (Ideally, on the border)
                candidates = self._get_border_buildings(d_cluster, nearest_excess)

                for b_link in candidates:
                    building_load = b_link.building.heating_demand or 0
                    
                    # CONDITIONS TO MOVE:
                    # 1. Deficit cluster still has deficit
                    # 2. Excess cluster has room to take this load
                    if current_deficit > 0 and available_excess >= building_load:
                        
                        # EXECUTE MOVE
                        # Unlink from Deficit
                        b_link.delete()
                        # Link to Excess
                        BuildingCluster.objects.create(building=b_link.building, cluster=nearest_excess)
                        
                        # Update Counters
                        current_deficit -= building_load
                        available_excess -= building_load
                        
                        moves_log.append(f"Moved {b_link.building.name} from {d_cluster.name} to {nearest_excess.name}")
                    
                    if current_deficit <= 0:
                        break # Deficit Cluster is now Balanced!

                # 5. Recalculate Balance for both clusters after moves
                self._recalculate_cluster_balance(d_cluster)
                self._recalculate_cluster_balance(nearest_excess)

            return {"status": "success", "moves": moves_log}

    # =========================================================================
    # STRATEGY 2: INTER-CLUSTER CONNECTION (Macro Edges)
    # =========================================================================
    def balance_by_interconnection(self):
        """
        STRATEGY: ECONOMICALLY VIABLE INTERCONNECTION (Greedy Descending)
        1. Sorts Deficit Clusters in DESCENDING order (largest needs first).
        2. Evaluates connections to nearest Surplus Clusters.
        3. Only builds a pipe if Pipe Cost <= Boiler Cost.
        4. Deploys local Gas Boilers for any remaining deficit.
        """
        from django.db import transaction

        # --- ECONOMIC COST ASSUMPTIONS ---
        # Adjust these based on real-world market prices for your research
        c_pipe = 500.0   # Cost to trench and lay 1 meter of pipe ($/m)
        c_boiler = 50.0  # Cost to install 1 kW of local gas boiler capacity ($/kW)

        with transaction.atomic():
            project = self.config.cluster_algorithm.project
            network = Network.objects.create(
                project=project,
                name=f"Viable Interconnections ({self.config.name})",
                network_type='combined',
                description="Greedy descending allocation with economic viability threshold."
            )

            # 1. Create Nodes for all Clusters
            clusters = list(Cluster.objects.filter(cluster_configuration=self.config))
            cluster_node_map = {}

            for c in clusters:
                cooling = c.cooling_demand or 0
                heating = c.heating_demand or 0
                node = Node.objects.create(
                    network=network, 
                    name=f"Cluster {c.id}", 
                    cluster=c, 
                    latitude=c.latitude, 
                    longitude=c.longitude,
                    node_type='producer' if cooling > heating else 'consumer'
                )
                cluster_node_map[c.id] = node

            # 2. Separate into Deficit and Excess
            deficit_clusters = [c for c in clusters if (c.cooling_demand or 0) < (c.heating_demand or 0)]
            excess_clusters = [c for c in clusters if (c.cooling_demand or 0) > (c.heating_demand or 0)]
            
            # --- THE "BIG ROCKS FIRST" SORTING STEP ---
            # Sort descending: Largest net heating demand first
            deficit_clusters.sort(
                key=lambda c: (c.heating_demand or 0) - (c.cooling_demand or 0), 
                reverse=True
            )
            
            # Master availability map for surplus clusters
            available_capacities = {
                c.id: (c.cooling_demand or 0) - (c.heating_demand or 0) 
                for c in excess_clusters
            }
            
            edges_created = []
            boilers_deployed = []

            # 3. Process Connections with Viability Check
            for sink_c in deficit_clusters:
                needed = (sink_c.heating_demand or 0) - (sink_c.cooling_demand or 0)
                
                # Find nearest sources sorted by distance
                sorted_sources = self._sort_clusters_by_distance(sink_c, excess_clusters)

                for source_c in sorted_sources:
                    if needed <= 0: 
                        break

                    available = available_capacities[source_c.id]
                    if available <= 0: 
                        continue

                    # Calculate potential transfer and distance
                    transfer_amount = min(needed, available)
                    distance_m = self._haversine_dist(sink_c, source_c) * 1000
                    safe_distance = max(distance_m, 1.0) # Prevent division by zero

                    # --- EXPLICIT VIABILITY CHECK ---
                    total_pipe_cost = c_pipe * safe_distance
                    total_boiler_cost = c_boiler * transfer_amount
                    
                    if total_pipe_cost > total_boiler_cost:
                        # Skip this connection. It costs too much to pipe this energy this far.
                        continue

                    # If viable, create the Edge (Pipe)
                    Edge.objects.create(
                        network=network,
                        start_node=cluster_node_map[source_c.id],
                        end_node=cluster_node_map[sink_c.id],
                        length=distance_m,
                        other_information={
                            "capacity_kW": transfer_amount,
                            "estimated_pipe_cost": total_pipe_cost,
                            "avoided_boiler_cost": total_boiler_cost
                        },
                        name=f"Pipe: C{source_c.id} -> C{sink_c.id} ({transfer_amount}kW)"
                    )
                    
                    needed -= transfer_amount
                    available_capacities[source_c.id] -= transfer_amount 
                    edges_created.append(f"Connected C{source_c.id} to C{sink_c.id} ({transfer_amount}kW over {round(distance_m, 1)}m)")

                # 4. Process Leftovers (Deploy Aux Boilers)
                if needed > 0:
                    gas_boiler = Boiler.objects.create(
                        project=project,
                        name=f"Gas Boiler - Cluster {sink_c.id}",
                        boiler_type='gas',
                        capacity=needed,
                        efficiency=0.9,
                        fuel_type='natural_gas',
                        status='operational',
                    )
                    
                    Node.objects.create(
                        network=network, 
                        name=f"Local Gas Plant (Cluster {sink_c.id})",
                        node_type='producer', 
                        latitude=sink_c.latitude, 
                        longitude=sink_c.longitude,
                        boiler=gas_boiler, 
                        cluster=sink_c,
                        other_information={
                            "capacity_kW": needed, 
                            "fuel": "natural_gas", 
                            "status": "active"
                        }
                    )
                    boilers_deployed.append(f"Boiler at C{sink_c.id} ({needed}kW)")

            return {
                "status": "success", 
                "network_id": network.id, 
                "edges": edges_created,
                "boilers": boilers_deployed
            }
    # =========================================================================
    # STRATEGY 3: AUXILIARY BOILER (New Node)
    # =========================================================================

    def balance_by_auxiliary_source(self):
        """
        STRATEGY: DECENTRALIZED GAS BOILERS
        1. Creates a Network to hold the simulation.
        2. Creates Nodes representing each existing Cluster.
        3. Identifies Deficit Clusters (Heating Demand > Cooling Demand).
        4. Adds a specific 'Local Gas Heating Plant' Node at the center of EACH deficit cluster.
        """
        from django.db import transaction # Ensure this is imported at the top of your file

        with transaction.atomic():
            project = self.config.cluster_algorithm.project
            # 1. Create the new simulation Network
            network = Network.objects.create(
                project=project,
                name=f"Decentralized Gas Boilers ({self.config.name})",
                network_type='combined',
                description="Each deficit cluster is supported by its own local gas-powered heating plant."
            )
            
            clusters = list(Cluster.objects.filter(cluster_configuration=self.config))
            cluster_node_map = {}
            boilers_created = 0
            total_deficit_covered = 0
            
            for c in clusters:
                heating = c.heating_demand or 0
                cooling = c.cooling_demand or 0
                
                # Determine the base role of the cluster
                if cooling > heating: 
                    role = 'producer'
                elif heating > cooling: 
                    role = 'consumer'
                else: 
                    role = 'prosumer'

                # 2. Create the Node representing the Cluster itself
                cluster_node = Node.objects.create(
                    network=network, 
                    name=f"Cluster {c.id}", 
                    cluster=c, 
                    latitude=c.latitude, 
                    longitude=c.longitude, 
                    node_type=role
                )
                cluster_node_map[c.id] = cluster_node

                # 3. Check for a Heating Deficit
                deficit = heating - cooling
                
                if deficit > 0:
                    # Create the physical Gas Boiler asset
                    # Note: Adjust fields like 'fuel_type' or 'capacity' to match your Boiler model
                    gas_boiler = Boiler.objects.create(
                        name=f"Gas Boiler - Cluster {c.id}",
                        project=project,
                        boiler_type='gas',
                        fuel_type='natural_gas',
                        capacity=deficit,
                        efficiency=0.9, # Example efficiency
                        status='operational',

                    )
                    
                    # 4. Create a specific Node for the Boiler at the cluster's center
                    Node.objects.create(
                        network=network,
                        name=f"Local Gas Plant (Cluster {c.id})",
                        node_type='producer',
                        latitude=c.latitude,   # Center of the cluster
                        longitude=c.longitude, # Center of the cluster
                        boiler=gas_boiler,
                        cluster=c,
                        other_information={
                            "capacity_kW": deficit, 
                            "fuel": "natural_gas",
                            "status": "active"
                        }
                    )
                    
                    boilers_created += 1
                    total_deficit_covered += deficit

            return {
                "status": "success", 
                "network_id": network.id, 
                "boilers_deployed": boilers_created,
                "total_capacity_added": total_deficit_covered
            }

    # STRATEGY 4: MOBILE THERMAL STORAGE (MTS)
    # =========================================================================
    def balance_by_mobile_storage(self):
        """
        STRATEGY: MOBILE THERMAL STORAGE (MTS)
        1. Creates a Network with ALL clusters as Nodes.
        2. Identifies Truck Routes between Excess (Source) and Deficit (Sink) clusters.
        3. Creates 'mobile_route' Edges (dashed lines) connecting the existing nodes.
        """
        with transaction.atomic():
            project = self.config.cluster_algorithm.project
            network = Network.objects.create(
                project=project,
                name=f"MTS Logistics Network ({self.config.name})",
                network_type='combined',
                description="Logistics network using truck-based thermal storage routes."
            )
            
            clusters = list(Cluster.objects.filter(cluster_configuration=self.config))
            cluster_node_map = {}
            nodes_created_count = 0

            # ---------------------------------------------------------
            # 1. Create Nodes for ALL Clusters (Deficit, Excess, Balanced)
            # ---------------------------------------------------------
            for c in clusters:
                heating = c.heating_demand or 0
                cooling = c.cooling_demand or 0
                
                if cooling > heating: 
                    role = 'producer'
                elif heating > cooling: 
                    role = 'consumer'
                else: 
                    role = 'prosumer'

                node = Node.objects.create(
                    network=network, 
                    name=f"Cluster {c.id}", 
                    cluster=c, 
                    latitude=c.latitude, 
                    longitude=c.longitude, 
                    node_type=role
                )
                cluster_node_map[c.id] = node
                nodes_created_count += 1

            # ---------------------------------------------------------
            # 2. Create Truck Routes (Edges)
            # ---------------------------------------------------------
            deficit_clusters = [c for c in clusters if (c.heating_demand or 0) > (c.cooling_demand or 0)]
            excess_clusters = [c for c in clusters if (c.cooling_demand or 0) > (c.heating_demand or 0)]
            
            routes_log = []

            for sink_c in deficit_clusters:
                if not excess_clusters: break
                
                # Find nearest Excess Cluster to supply this sink
                nearest = self._find_nearest_cluster(sink_c, excess_clusters)
                if not nearest: continue

                dist_km = self._haversine_dist(sink_c, nearest)
                
                # Logic check: In real engineering, MTS is usually for medium distances (e.g., 2km - 10km)
                # We will create the route regardless for visualization purposes.

                # Create MTS Route (Edge) linking the EXISTING nodes from Step 1
                Edge.objects.create(
                    network=network,
                    start_node=cluster_node_map[nearest.id],
                    end_node=cluster_node_map[sink_c.id],
                    length=dist_km * 1000,
                    other_information={
                        "capacity": 2500, # Example standard: 2.5 MWh Truck
                        "estimated_trips_per_day": 4,
                        "distance_km": dist_km
                    },
                    name=f"Truck Route: {nearest.name} -> {sink_c.name}"
                )
                routes_log.append(f"Route from {nearest.name} to {sink_c.name} ({dist_km:.2f}km)")

            return {
                "status": "success", 
                "network_id": network.id, 
                "nodes_count": nodes_created_count,
                "routes_count": len(routes_log),
                "routes": routes_log
            }
    # =========================================================================
    # HELPERS
    # =========================================================================
    def _find_nearest_cluster(self, target, candidates):
        if not candidates: return None
        distances = [self._haversine_dist(target, c) for c in candidates]
        return candidates[np.argmin(distances)]

    def _sort_clusters_by_distance(self, target, candidates):
        if not candidates: return []
        # Return list of candidates sorted by distance to target
        return sorted(candidates, key=lambda c: self._haversine_dist(target, c))

    def _haversine_dist(self, c1, c2):
        # ... (Reuse your Haversine logic here) ...
        # Simplified:
        R = 6371
        dlat = np.radians(c2.latitude - c1.latitude)
        dlon = np.radians(c2.longitude - c1.longitude)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(c1.latitude)) * np.cos(np.radians(c2.latitude)) * np.sin(dlon/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    def _get_border_buildings(self, d_cluster, e_cluster):
        # Return buildings in d_cluster sorted by proximity to e_cluster
        links = BuildingCluster.objects.filter(cluster=d_cluster).select_related('building')
        
        # Calculate distance of each building to the Excess Cluster Centroid
        # This is a bit computationally heavy, but effective
        candidates = []
        for link in links:
            dist = self._haversine_point_to_cluster(link.building, e_cluster)
            candidates.append((dist, link))
        
        # Sort by shortest distance (Ascending)
        candidates.sort(key=lambda x: x[0])
        return [x[1] for x in candidates]

    def _haversine_point_to_cluster(self, building, cluster):
        # Calc distance between building lat/lon and cluster centroid
        # ... implementation ...
        return 0.0 # Placeholder

    def _recalculate_cluster_balance(self, cluster):
        # Re-run your aggregation logic (Source vs Sink)
        # Update cluster.energy_balance_status
        pass
