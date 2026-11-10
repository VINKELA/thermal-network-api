import math
import networkx as nx
from django.db import transaction
from django.db.models import Max, Q, Sum, Count
from ..models import (
    Project, Network, Node, Edge, PipeSizing, EdgePipeSizing, 
    Building, Route, Cluster, EdgeConnectionAlgorithmEdge, LoadProfile,
    NetworkDesignConstants, assert_network_project_integrity, validate_project_isolation
) #
from .route_geometry import route_physical_segments


# ==============================================================================
# 1. PHYSICS & CONSTANTS
# ==============================================================================
class DesignCriteria:
    """
    Physical and Economic constants for the sizing engine.
    """
    # Fluid Properties (Water @ 80°C)
    RHO = 971.8  # Density [kg/m³]
    CP = 4.19    # Specific Heat [kJ/kg·K]
    MU = 0.000355 # Dynamic Viscosity [Pa·s]

    # Constraints
    MAX_VELOCITY = 2.5      # [m/s] Erosion limit
    MIN_VELOCITY = 0.5      # [m/s] Air entrainment limit
    TARGET_PRESSURE_DROP = 100 # [Pa/m] Standard target for pressure optimization
    
    # Economics (for Lifecycle Costing)
    ELECTRICITY_COST = 0.15 # [$/kWh]
    PUMP_EFFICIENCY = 0.75  # [%]
    LIFESPAN_YEARS = 30     # [years]
    HOURS_PER_YEAR = 4000   # [hours] (Equivalent full-load hours)
    SUPPLY_TEMPERATURE_C = 70.0
    RETURN_TEMPERATURE_C = 50.0
    AMBIENT_TEMPERATURE_C = 10.0

    # Standard Pipe Catalog (Internal Diameter [m], Roughness [m], Cost [$/m])
    STD_PIPES = [
        {'dn': 20,  'di': 0.020, 'cost': 15,  'roughness': 0.000045},
        {'dn': 25,  'di': 0.025, 'cost': 20,  'roughness': 0.000045},
        {'dn': 32,  'di': 0.032, 'cost': 30,  'roughness': 0.000045},
        {'dn': 40,  'di': 0.040, 'cost': 40,  'roughness': 0.000045},
        {'dn': 50,  'di': 0.050, 'cost': 55,  'roughness': 0.000045},
        {'dn': 65,  'di': 0.065, 'cost': 70,  'roughness': 0.000045},
        {'dn': 80,  'di': 0.080, 'cost': 90,  'roughness': 0.000045},
        {'dn': 100, 'di': 0.100, 'cost': 120, 'roughness': 0.000045},
        {'dn': 125, 'di': 0.125, 'cost': 160, 'roughness': 0.000045},
        {'dn': 150, 'di': 0.150, 'cost': 200, 'roughness': 0.000045},
        {'dn': 200, 'di': 0.200, 'cost': 300, 'roughness': 0.000045},
        {'dn': 250, 'di': 0.250, 'cost': 450, 'roughness': 0.000045},
        {'dn': 300, 'di': 0.300, 'cost': 600, 'roughness': 0.000045},
        {'dn': 350, 'di': 0.350, 'cost': 800, 'roughness': 0.000045},
        {'dn': 400, 'di': 0.400, 'cost': 1100, 'roughness': 0.000045},
    ]


class RuntimeDesignCriteria:
    def __init__(self, network_id):
        self.RHO = DesignCriteria.RHO
        self.CP = DesignCriteria.CP
        self.MU = DesignCriteria.MU
        self.MAX_VELOCITY = DesignCriteria.MAX_VELOCITY
        self.MIN_VELOCITY = DesignCriteria.MIN_VELOCITY
        self.TARGET_PRESSURE_DROP = DesignCriteria.TARGET_PRESSURE_DROP
        self.ELECTRICITY_COST = DesignCriteria.ELECTRICITY_COST
        self.PUMP_EFFICIENCY = DesignCriteria.PUMP_EFFICIENCY
        self.LIFESPAN_YEARS = DesignCriteria.LIFESPAN_YEARS
        self.HOURS_PER_YEAR = DesignCriteria.HOURS_PER_YEAR
        self.DISCOUNT_RATE = 0.04
        self.SUPPLY_TEMPERATURE_C = DesignCriteria.SUPPLY_TEMPERATURE_C
        self.RETURN_TEMPERATURE_C = DesignCriteria.RETURN_TEMPERATURE_C
        self.AMBIENT_TEMPERATURE_C = DesignCriteria.AMBIENT_TEMPERATURE_C
        self.STD_PIPES = DesignCriteria.STD_PIPES

        constants = NetworkDesignConstants.objects.filter(network_id=network_id).first()
        if constants:
            self.RHO = constants.rho
            self.CP = constants.cp
            self.MU = constants.mu
            self.MAX_VELOCITY = constants.max_velocity
            self.MIN_VELOCITY = constants.min_velocity
            self.TARGET_PRESSURE_DROP = constants.target_pressure_drop
            self.ELECTRICITY_COST = constants.electricity_cost
            self.PUMP_EFFICIENCY = constants.pump_efficiency
            self.LIFESPAN_YEARS = constants.lifespan_years
            self.HOURS_PER_YEAR = constants.hours_per_year
            self.DISCOUNT_RATE = constants.discount_rate
            self.SUPPLY_TEMPERATURE_C = constants.supply_temperature_c
            self.RETURN_TEMPERATURE_C = constants.return_temperature_c
            self.AMBIENT_TEMPERATURE_C = constants.ambient_temperature_c
            self.STD_PIPES = constants.pipe_catalog or DesignCriteria.STD_PIPES

class HydraulicFormulas:
    @staticmethod
    def calculate_mass_flow(load_kw, delta_t, criteria=DesignCriteria):
        """ Q = m * cp * dT  =>  m = Q / (cp * dT) """
        if load_kw < 0:
            raise ValueError("Design load cannot be negative.")
        if delta_t <= 0 or criteria.CP <= 0:
            raise ValueError("Specific heat and design temperature difference must be positive.")
        return load_kw / (criteria.CP * delta_t) # [kg/s]

    @staticmethod
    def calculate_reynolds(v, d, criteria=DesignCriteria):
        """ Re = (rho * v * d) / mu """
        if v < 0 or d <= 0 or criteria.RHO <= 0 or criteria.MU <= 0:
            raise ValueError("Velocity must be non-negative and fluid properties and diameter must be positive.")
        return (criteria.RHO * v * d) / criteria.MU

    @staticmethod
    def calculate_friction_factor(re, d, roughness=0.000045):
        """ Swamee-Jain equation (Approx of Colebrook-White) """
        if re <= 0:
            return 0.0
        if d <= 0 or roughness < 0:
            raise ValueError("Diameter must be positive and roughness cannot be negative.")
        if re < 2000: return 64 / re # Laminar
        if re < 4000: return 0.04 # Transition zone approximation
        
        # Turbulent flow
        term1 = roughness / (3.7 * d)
        term2 = 5.74 / (re ** 0.9)
        return 0.25 / (math.log10(term1 + term2) ** 2)

    @staticmethod
    def calculate_head_loss_pa(v, d, length, criteria=DesignCriteria, roughness=0.000045):
        """ Darcy-Weisbach: dP = f * (L/D) * (rho * v^2)/2 """
        if length < 0:
            raise ValueError("Pipe length cannot be negative.")
        if v == 0:
            return 0.0
        re = HydraulicFormulas.calculate_reynolds(v, d, criteria)
        f = HydraulicFormulas.calculate_friction_factor(re, d, roughness)
        return f * (length / d) * (criteria.RHO * (v**2) / 2)


# ==============================================================================
# 2. MAIN CALCULATION ENGINE
# ==============================================================================
class NetworkSizer:
    def __init__(self, network_id, sizing_config_id):
        self.network_id = network_id
        self.sizing_config = PipeSizing.objects.get(id=sizing_config_id)
        if self.sizing_config.network_id != self.network_id:
            raise ValueError("Pipe sizing configuration must belong to the network being sized.")
        assert_network_project_integrity(self.sizing_config.network)
        self.criteria = RuntimeDesignCriteria(network_id)
        self.graph = None
        self.root_node_id = None
        
        # Graph Metadata
        self.node_loads = {}      # kW
        self.node_consumers = {}  # Count of buildings
        self.results = []         # Final EdgePipeSizing objects
        
        # Geometry / Topological Linking
        self.conn_algorithm_id = None
        self.catalog = self._validated_catalog()

    def _validated_catalog(self):
        catalog = []
        for item in self.criteria.STD_PIPES or []:
            try:
                pipe = {
                    **item,
                    'dn': int(item['dn']),
                    'di': float(item['di']),
                    'cost': float(item['cost']),
                    'roughness': float(item.get('roughness', 0.000045)),
                }
            except (KeyError, TypeError, ValueError):
                raise ValueError("Every pipe catalogue row requires numeric dn, di and cost values.")
            if pipe['di'] <= 0 or pipe['cost'] < 0 or pipe['roughness'] < 0:
                raise ValueError("Pipe diameter must be positive; cost and roughness cannot be negative.")
            catalog.append(pipe)
        if not catalog:
            raise ValueError("The pipe catalogue is empty.")
        return sorted(catalog, key=lambda pipe: pipe['di'])

    def get_peak_load_values(self, building, year=None):
        # Get max heating load
        heating_profiles = LoadProfile.objects.filter(
            building=building, 
            profile_type='heating'
        )
        cooling_profiles = LoadProfile.objects.filter(
            building=building, 
            profile_type='cooling'
        )
        if year:
            heating_profiles = heating_profiles.filter(start_date__year=year)
            cooling_profiles = cooling_profiles.filter(start_date__year=year)

        peak_heating = heating_profiles.aggregate(max_load=Max('value'))['max_load']

        # Get max cooling load
        peak_cooling = cooling_profiles.aggregate(max_load=Max('value'))['max_load']

        return {
            'peak_heating_kw': peak_heating or max(building.heating_demand or 0.0, 0.0),
            'peak_cooling_kw': peak_cooling or max(building.cooling_demand or 0.0, 0.0)
        }

    def build_topology(self, conn_algorithm_id=None, year=None):
        """
        Maps Django Edge/Node models to NetworkX DiGraph.
        Resolves flow direction starting from Producer nodes.
        """
        self.conn_algorithm_id = conn_algorithm_id
        
        # 1. Fetch Edges
        edges = Edge.objects.filter(network_id=self.network_id).select_related('start_node', 'end_node')
        routes_by_edge = {
            route.edge_id: route
            for route in Route.objects.filter(edge__network_id=self.network_id).order_by('id')
        }
        valid_edge_ids = None
        if conn_algorithm_id:
            edge_connection_algorithm_edges = EdgeConnectionAlgorithmEdge.objects.filter(
                edge_connection_algorithm_id=conn_algorithm_id)
            valid_edge_ids = {ecae.edge_id for ecae in edge_connection_algorithm_edges}

        # 2. Build an UNDIRECTED graph first
        U = nx.Graph()
        for e in edges:
            if valid_edge_ids is None or e.id in valid_edge_ids:
                route = routes_by_edge.get(e.id)
                length = route.distance_meters if route and route.distance_meters else e.length
                U.add_edge(e.start_node.id, e.end_node.id, db_obj=e, length=length, route=route)

        # 3. Identify Root (Producer)
        producers = list(Node.objects.filter(network_id=self.network_id, node_type='producer').order_by('id'))
        if not producers:
            # Legacy generated networks represented the source as the sole
            # zero-demand building rather than preserving node_type=producer.
            producers = list(
                Node.objects.filter(
                    network_id=self.network_id,
                    building__isnull=False,
                    building__heating_demand__lte=0,
                ).order_by('id')
            )
        if len(producers) != 1:
            raise ValueError(
                "Pipe sizing requires exactly one explicit producer or one unambiguous "
                f"zero-demand source candidate; found {len(producers)}."
            )
        self.root_node_id = producers[0].id
        if self.root_node_id not in U:
            raise ValueError("The producer is not connected to the selected topology.")
        if not nx.is_connected(U):
            raise ValueError("Pipe sizing requires a connected topology.")
        if not nx.is_tree(U):
            raise ValueError("Meshed topology detected. A hydraulic flow solver is required before pipe sizing.")
        
        # 4. The Magic: Force a directed outward tree from the producer using BFS
        G = nx.DiGraph()
        # This guarantees edges only flow AWAY from the producer
        directed_edges = list(nx.bfs_edges(U, source=self.root_node_id))
        
        for u, v in directed_edges:
            # Transfer the edge data from the undirected graph to the directed one
            edge_data = U.get_edge_data(u, v)
            G.add_edge(u, v, **edge_data)

        self.graph = G
        
        # 5. Base Load Aggregation (Bottom-Up)
        self._accumulate_loads(year=year)
    def _accumulate_loads(self, year=None):
        """
        Traverses leaves -> root to sum up loads.
        
        LOGIC:
        1. Node is CLUSTER: 
           - Fetch all buildings via 'BuildingCluster'.
           - Calculate peak load for EACH building (using profiles).
           - Sum these peaks to get the Cluster Total.
           - Fallback to static Cluster.heating_demand if profiles missing.
           
        2. Node is BUILDING: 
           - Use LoadProfile peak OR fallback to Building.heating_demand.
        """
        # 1. Guaranteed Bottom-Up Traversal
        # Because build_topology now forces a perfect tree flowing from the producer,
        # this will safely return a list starting with the furthest leaves and ending at the root.
        traversal = list(nx.dfs_postorder_nodes(self.graph, self.root_node_id))

        # 2. Define the Profile Filter (Heating + Optional Year)
        profile_filter = Q(load_profiles__profile_type='heating')
        if year:
            profile_filter &= Q(load_profiles__start_date__year=year)

        # 3. Fetch Nodes
        nodes = Node.objects.filter(network_id=self.network_id).select_related('building', 'cluster')

        # 4. Assign Base Loads (Leaf Nodes)
        for n in nodes:
            load = 0.0
            consumers = 0

            # CASE A: Node is a CLUSTER (Bottom-Up Aggregation)
            if n.cluster:
                # Find all buildings in this cluster
                cluster_buildings = Building.objects.filter(
                    buildingcluster__cluster=n.cluster
                ).annotate(
                    # Calculate Peak for EACH building individually
                    b_peak=Max('load_profiles__value', filter=profile_filter)
                )

                # Sum the peaks
                agg_stats = cluster_buildings.aggregate(
                    total_profile_peak=Sum('b_peak'),
                    total_static_demand=Sum('heating_demand'),
                    count=Count('id')
                )

                # Priority Logic
                if agg_stats['total_profile_peak'] and agg_stats['total_profile_peak'] > 0:
                    load = agg_stats['total_profile_peak']
                elif agg_stats['total_static_demand'] and agg_stats['total_static_demand'] > 0:
                    load = agg_stats['total_static_demand']
                elif n.cluster.heating_demand and n.cluster.heating_demand > 0:
                    load = n.cluster.heating_demand

                consumers = agg_stats['count'] if agg_stats['count'] > 0 else (n.cluster.no_of_buildings or 1)

            # CASE B: Node is a SINGLE BUILDING
            elif n.building:
                # Get peak from yearly load profiles
                peak = self.get_peak_load_values(n.building, year=year)
                is_producer = (
                    n.node_type == 'producer' or
                    bool(getattr(n.building, 'waste_heat_source_type', None))
                )
                consumers = 0 if is_producer else 1
                load = 0.0 if is_producer else peak['peak_heating_kw']
                
                print(f"Node {n.id} - Consumers: {consumers}, Peak Heating: {peak['peak_heating_kw']}, Peak Cooling: {peak['peak_cooling_kw']}")

            # Store the base (individual) load for this node before we start summing up the tree
            self.node_loads[n.id] = load
            self.node_consumers[n.id] = consumers

     # 5. Accumulate up the tree (Children -> Parents)
        for node_id in traversal:
            total_load = self.node_loads.get(node_id, 0.0)
            total_consumers = self.node_consumers.get(node_id, 0)
            
            for child_id in self.graph.successors(node_id):
                total_load += self.node_loads.get(child_id, 0.0)
                total_consumers += self.node_consumers.get(child_id, 0)
            
            self.node_loads[node_id] = total_load
            self.node_consumers[node_id] = total_consumers

        # --- ADD THIS TO PROVE IT WORKS ---
        print("\n=== TOPOLOGY ACCUMULATION COMPLETE ===")
        print(f"Producer (Node {self.root_node_id}) must supply:")
        print(f"Total Load: {self.node_loads[self.root_node_id]:.2f} kW")
        print(f"Total Consumers: {self.node_consumers[self.root_node_id]}")
        print("======================================\n")

    def sizing_peak_load(self, design_delta_t=None):
        design_delta_t = design_delta_t or self._design_delta_t()
        for u, v, data in self.graph.edges(data=True):
            load_kw = self.node_loads[v]
            print(f"Edge ({u} -> {v}): Peak Load = {load_kw:.2f} kW")
            m_dot = HydraulicFormulas.calculate_mass_flow(load_kw, design_delta_t, self.criteria)
            m_dot = max(m_dot, 0.0) # Ensure no negative mass flow
            print(f"Edge ({u} -> {v}): Mass Flow = {m_dot:.2f} kg/s")
            best_pipe = self._select_pipe_by_velocity(m_dot, self.criteria.MAX_VELOCITY)
            print(f"Edge ({u} -> {v}): Selected Pipe DN{best_pipe['dn']} with Velocity = {best_pipe['velocity']:.2f} m/s")
            self._store_result(data['db_obj'], m_dot, best_pipe, load_kw, design_delta_t)

    def sizing_pressure_optimized(self, design_delta_t=None):
        design_delta_t = design_delta_t or self._design_delta_t()
        for u, v, data in self.graph.edges(data=True):
            load_kw = self.node_loads[v]
            m_dot = HydraulicFormulas.calculate_mass_flow(load_kw, design_delta_t, self.criteria)
            
            best_pipe = None
            for pipe in self.catalog:
                area = math.pi * (pipe['di'] ** 2) / 4
                velocity = (m_dot / self.criteria.RHO) / area
                if velocity > self.criteria.MAX_VELOCITY: continue
                
                loss_pa = HydraulicFormulas.calculate_head_loss_pa(velocity, pipe['di'], 1.0, self.criteria, pipe['roughness'])
                if loss_pa <= self.criteria.TARGET_PRESSURE_DROP:
                    best_pipe = {**pipe, 'velocity': velocity}
                    break 
            
            if not best_pipe:
                raise ValueError("No catalogue diameter satisfies both velocity and pressure-gradient limits.")

            self._store_result(data['db_obj'], m_dot, best_pipe, load_kw, design_delta_t)

    def sizing_lifecycle_economic(self, design_delta_t=None):
        design_delta_t = design_delta_t or self._design_delta_t()
        for u, v, data in self.graph.edges(data=True):
            load_kw = self.node_loads[v]
            m_dot = HydraulicFormulas.calculate_mass_flow(load_kw, design_delta_t, self.criteria)
            length = data['length']
            
            best_pipe = None
            min_total_cost = float('inf')

            for pipe in self.catalog:
                area = math.pi * (pipe['di'] ** 2) / 4
                velocity = (m_dot / self.criteria.RHO) / area
                if velocity > self.criteria.MAX_VELOCITY: continue

                capex = pipe['cost'] * length
                loss_pa = HydraulicFormulas.calculate_head_loss_pa(velocity, pipe['di'], length, self.criteria, pipe['roughness'])
                vol_flow = m_dot / self.criteria.RHO
                pump_power_kw = (vol_flow * loss_pa) / (self.criteria.PUMP_EFFICIENCY * 1000)
                
                annual_opex = pump_power_kw * self.criteria.HOURS_PER_YEAR * self.criteria.ELECTRICITY_COST
                discount = max(float(self.criteria.DISCOUNT_RATE), 0.0)
                years = max(int(self.criteria.LIFESPAN_YEARS), 1)
                pv_factor = years if discount == 0 else (1 - (1 + discount) ** -years) / discount
                opex = annual_opex * pv_factor
                total = capex + opex

                if total < min_total_cost:
                    min_total_cost = total
                    best_pipe = {**pipe, 'velocity': velocity}

            if not best_pipe:
                raise ValueError("No catalogue diameter satisfies the velocity limit.")
                 
            self._store_result(data['db_obj'], m_dot, best_pipe, load_kw, design_delta_t)

    # --- HELPERS ---
    def _design_delta_t(self):
        delta_t = self.criteria.SUPPLY_TEMPERATURE_C - self.criteria.RETURN_TEMPERATURE_C
        if delta_t <= 0:
            raise ValueError("Supply temperature must exceed return temperature.")
        return delta_t

    def _select_pipe_by_velocity(self, mass_flow, max_v):
        vol_flow = mass_flow / self.criteria.RHO
        for pipe in self.catalog:
            area = math.pi * (pipe['di'] ** 2) / 4
            v = vol_flow / area
            if v <= max_v:
                return {**pipe, 'velocity': v}
        raise ValueError("No catalogue diameter satisfies the velocity limit.")
    def _store_result(self, edge_obj, m_dot, pipe_data, peak_kw, dt):
        """
        Saves result and calculates final pressure drop for the UI.
        """
        velocity = pipe_data['velocity']
        diameter = pipe_data['di']
        route = Route.objects.filter(edge=edge_obj).first()
        length = route.distance_meters if route and route.distance_meters else edge_obj.length
        
        # Calculate the actual pressure drop for this specific segment
        pressure_drop = HydraulicFormulas.calculate_head_loss_pa(velocity, diameter, length, self.criteria, pipe_data.get('roughness', 0.000045))

        self.results.append(EdgePipeSizing(
            edge=edge_obj,
            pipe_sizing=self.sizing_config,
            peak_load_kw=peak_kw,
            mass_flow_kg_s=m_dot,
            volume_flow_m3_s=m_dot / self.criteria.RHO,
            diameter_m=diameter,
            velocity_m_s=velocity, 
            delta_t=dt,
            pressure_drop_pa=pressure_drop, # <--- SAVE IT HERE
            route=route,
            length_m=length,
            source_route_ids=[route.id] if route else [],
            source_edge_ids=[edge_obj.id],
            overlap_count=1,
            is_shared_segment=False,
        ))

    def collapse_route_results_to_physical_segments(self, routes):
        """
        Replace per-logical-route sizing rows with per-physical-pipe sizing rows.

        Route tracing can produce several logical routes over the same street
        segment. For construction and hydraulics that overlap is one pipe, so
        the shared segment must be counted once and sized for the combined flow
        that passes through it.
        """
        route_list = list(routes or [])
        if not route_list or not self.results:
            return None

        result_by_edge = {result.edge_id: result for result in self.results}
        route_by_id = {route.id: route for route in route_list}
        route_geometry = route_physical_segments(route_list, sizing_by_edge=result_by_edge)
        segment_results = []

        for segment in route_geometry["entries"]:
            source_results = [
                result_by_edge[edge_id]
                for edge_id in segment.get("source_edge_ids", [])
                if edge_id in result_by_edge
            ]
            if not source_results:
                representative_edge = segment.get("edge")
                source_results = [result for result in self.results if result.edge_id == representative_edge.id] if representative_edge else []
            if not source_results:
                continue

            peak_kw = sum(max(result.peak_load_kw or 0.0, 0.0) for result in source_results)
            mass_flow = sum(max(result.mass_flow_kg_s or 0.0, 0.0) for result in source_results)
            length = float(segment.get("length_m") or 0.0)
            delta_t = self._combined_delta_t(peak_kw, mass_flow, source_results)
            pipe_data = self._select_pipe_for_segment(mass_flow, length)
            pressure_drop = HydraulicFormulas.calculate_head_loss_pa(
                pipe_data['velocity'],
                pipe_data['di'],
                length,
                self.criteria,
                pipe_data.get('roughness', 0.000045),
            )
            representative_route = route_by_id.get(segment.get("route_id"))

            segment_results.append(EdgePipeSizing(
                edge=segment["edge"],
                pipe_sizing=self.sizing_config,
                peak_load_kw=peak_kw,
                mass_flow_kg_s=mass_flow,
                volume_flow_m3_s=mass_flow / self.criteria.RHO if self.criteria.RHO else 0.0,
                diameter_m=pipe_data['di'],
                velocity_m_s=pipe_data['velocity'],
                delta_t=delta_t,
                pressure_drop_pa=pressure_drop,
                route=representative_route,
                length_m=length,
                source_route_ids=segment.get("source_route_ids", []),
                source_edge_ids=segment.get("source_edge_ids", []),
                overlap_count=segment.get("overlap_count", 1),
                is_shared_segment=segment.get("is_shared", False),
                segment_path=segment.get("path", []),
            ))

        if segment_results:
            self.results = segment_results
        return route_geometry["stats"]

    def _combined_delta_t(self, peak_kw, mass_flow, source_results):
        if peak_kw > 0 and mass_flow > 0 and self.criteria.CP > 0:
            return peak_kw / (mass_flow * self.criteria.CP)
        if source_results:
            return sum(result.delta_t or 0.0 for result in source_results) / len(source_results)
        return self.criteria.SUPPLY_TEMPERATURE_C - self.criteria.RETURN_TEMPERATURE_C

    def _select_pipe_for_segment(self, mass_flow, length):
        method = self.sizing_config.method
        if method == 'pressure_gradient':
            selected = self._select_pressure_optimized_pipe(mass_flow)
            if selected:
                return selected
        if method == 'lifecycle_cost':
            selected = self._select_lifecycle_pipe(mass_flow, length)
            if selected:
                return selected
        return self._select_pipe_by_velocity(mass_flow, self.criteria.MAX_VELOCITY)

    def _select_pressure_optimized_pipe(self, mass_flow):
        for pipe in self.catalog:
            area = math.pi * (pipe['di'] ** 2) / 4
            velocity = (mass_flow / self.criteria.RHO) / area if area else 0.0
            if velocity > self.criteria.MAX_VELOCITY:
                continue
            loss_pa_per_m = HydraulicFormulas.calculate_head_loss_pa(velocity, pipe['di'], 1.0, self.criteria, pipe['roughness'])
            if loss_pa_per_m <= self.criteria.TARGET_PRESSURE_DROP:
                return {**pipe, 'velocity': velocity}
        return None

    def _select_lifecycle_pipe(self, mass_flow, length):
        best_pipe = None
        min_total_cost = float('inf')
        for pipe in self.catalog:
            area = math.pi * (pipe['di'] ** 2) / 4
            velocity = (mass_flow / self.criteria.RHO) / area if area else 0.0
            if velocity > self.criteria.MAX_VELOCITY:
                continue
            capex = pipe['cost'] * length
            loss_pa = HydraulicFormulas.calculate_head_loss_pa(velocity, pipe['di'], length, self.criteria, pipe['roughness'])
            vol_flow = mass_flow / self.criteria.RHO if self.criteria.RHO else 0.0
            pump_power_kw = (vol_flow * loss_pa) / (self.criteria.PUMP_EFFICIENCY * 1000) if self.criteria.PUMP_EFFICIENCY else 0.0
            annual_opex = pump_power_kw * self.criteria.HOURS_PER_YEAR * self.criteria.ELECTRICITY_COST
            discount = max(float(self.criteria.DISCOUNT_RATE), 0.0)
            years = max(int(self.criteria.LIFESPAN_YEARS), 1)
            pv_factor = years if discount == 0 else (1 - (1 + discount) ** -years) / discount
            opex = annual_opex * pv_factor
            total = capex + opex
            if total < min_total_cost:
                min_total_cost = total
                best_pipe = {**pipe, 'velocity': velocity}
        return best_pipe
    
    def save_results(self):
        with transaction.atomic():
            for result in self.results:
                validate_project_isolation(result)
            EdgePipeSizing.objects.filter(pipe_sizing=self.sizing_config).delete()
            EdgePipeSizing.objects.bulk_create(self.results)
