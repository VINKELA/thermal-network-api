import logging
import networkx as nx
import pyomo.environ as pyo
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional

# Import your Django models (adjust the import path based on your app name)
from myapp.models import Network, Node, Edge

logger = logging.getLogger(__name__)

class ThermalNetworkOptimizer:
    """
    Two-Stage Pareto Filter Framework for Thermal Network Decarbonization.
    Stage 1: MILP Optimization (Cost vs. Carbon) using Pyomo.
    Stage 2: Reduced Composite Sustainability Index (CSI) evaluation.
    """

    def __init__(self, network_id: int):
        self.network_id = network_id
        self.network_name = ""
        self.graph = nx.Graph()
        
    # ==========================================
    # STAGE 0: DATA EXTRACTION & ROUTING
    # ==========================================
    def build_spatial_graph(self) -> nx.Graph:
        """Extracts the Django relational data into a mathematical NetworkX graph."""
        try:
            network_obj = Network.objects.get(id=self.network_id)
            self.network_name = network_obj.name
            self.graph.graph['name'] = self.network_name
        except Network.DoesNotExist:
            raise ValueError(f"Network with ID {self.network_id} not found.")

        # 1. Load Nodes
        nodes = Node.objects.filter(network=network_obj).select_related(
            'building', 'boiler', 'geothermal', 'pv', 'thermal_storage'
        )
        
        for node in nodes:
            demand, capacity, opex, emission_factor = 0.0, 0.0, 0.0, 0.0
            
            if node.node_type == 'consumer' and node.building:
                demand = float(node.building.heating_demand or 0.0)
                
            elif node.node_type == 'producer':
                if node.boiler:
                    capacity = float(node.boiler.capacity)
                    opex = float(getattr(node.boiler, 'opex_per_kwh', 0.05))
                    emission_factor = float(getattr(node.boiler, 'emission_factor_kg_per_kwh', 0.2))
                elif node.geothermal:
                    capacity = float(node.geothermal.total_heating_capacity)
                    opex = float(getattr(node.geothermal, 'opex_per_kwh', 0.01))
                    emission_factor = 0.0 # Zero carbon

            self.graph.add_node(
                node.id, 
                node_type=node.node_type,
                demand=demand, 
                max_capacity=capacity,
                opex=opex,
                carbon_factor=emission_factor
            )

        # 2. Load Edges (Spatial connections)
        edges = Edge.objects.filter(network=network_obj).select_related('start_node', 'end_node')
        for edge in edges:
            self.graph.add_edge(edge.start_node.id, edge.end_node.id, length=float(edge.length))
            
        logger.info(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        return self.graph

    def apply_greedy_source_centric_routing(self):
        """
        PLACEHOLDER: Inject your custom routing logic here.
        This modifies self.graph in-place by pruning mathematically unviable 
        edges before handing the topology to the Pyomo solver, saving matrix computations.
        """
        # Example: Remove edges that are longer than a specific viability threshold
        # long_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d['length'] > 1500]
        # self.graph.remove_edges_from(long_edges)
        pass

    # ==========================================
    # STAGE 1: MILP OPTIMIZATION (THE PARETO GENERATOR)
    # ==========================================
    def solve_milp(self, max_carbon_allowance: float) -> Tuple[Optional[Dict], Optional[float], Optional[float]]:
        """
        Solves the network dispatch for a given carbon constraint.
        Minimizes purely for OPEX.
        """
        model = pyo.ConcreteModel()
        
        # Partition sets to prevent matrix bloat
        producers = [n for n in self.graph.nodes() if self.graph.nodes[n].get('node_type') == 'producer']
        consumers = [n for n in self.graph.nodes() if self.graph.nodes[n].get('node_type') == 'consumer']
        
        model.Producers = pyo.Set(initialize=producers)
        model.Consumers = pyo.Set(initialize=consumers)
        
        # Decision Variable: Hourly dispatch (simplified here to peak capacity)
        model.Dispatch = pyo.Var(model.Producers, domain=pyo.NonNegativeReals)
        
        # Constraints
        def capacity_rule(m, p):
            return m.Dispatch[p] <= self.graph.nodes[p]['max_capacity']
        model.capacity_limit = pyo.Constraint(model.Producers, rule=capacity_rule)
        
        total_demand = sum(self.graph.nodes[c]['demand'] for c in model.Consumers)
        def demand_rule(m):
            return sum(m.Dispatch[p] for p in m.Producers) >= total_demand
        model.meet_demand = pyo.Constraint(rule=demand_rule)
        
        def carbon_cap_rule(m):
            total_carbon = sum(m.Dispatch[p] * self.graph.nodes[p]['carbon_factor'] for p in m.Producers)
            return total_carbon <= max_carbon_allowance
        model.carbon_cap = pyo.Constraint(rule=carbon_cap_rule)
        
        # Objective Function
        def cost_objective(m):
            return sum(m.Dispatch[p] * self.graph.nodes[p]['opex'] for p in m.Producers)
        model.obj = pyo.Objective(rule=cost_objective, sense=pyo.minimize)
        
        # Execution
        solver = pyo.SolverFactory('cbc') # Ensure CBC is installed on the host machine
        result = solver.solve(model, tee=False)
        
        if result.solver.termination_condition == pyo.TerminationCondition.optimal:
            dispatch_config = {p: pyo.value(model.Dispatch[p]) for p in model.Producers}
            optimal_cost = pyo.value(model.obj)
            actual_carbon = sum(dispatch_config[p] * self.graph.nodes[p]['carbon_factor'] for p in model.Producers)
            return dispatch_config, optimal_cost, actual_carbon
            
        return None, None, None

    # ==========================================
    # STAGE 2: REDUCED CSI EVALUATION
    # ==========================================
    def calculate_csi(self, dispatch_config: Dict[int, float]) -> float:
        """
        Evaluates the optimal configuration based strictly on Technical and Social metrics.
        Assumes dispatch_config contains {node_id: dispatched_kw}.
        """
        if not dispatch_config:
            return 0.0
            
        total_dispatch = sum(dispatch_config.values())
        if total_dispatch == 0:
            return 0.0

        # Technical Metric: Decentralization (Penalty for relying heavily on a single node)
        max_single_source = max(dispatch_config.values())
        decentralization_score = 1.0 - (max_single_source / total_dispatch)
        
        # Social Metric: Carbon-Free Local Resource Utilization
        zero_carbon_dispatch = sum(
            disp for node_id, disp in dispatch_config.items() 
            if self.graph.nodes[node_id]['carbon_factor'] == 0
        )
        local_resource_score = zero_carbon_dispatch / total_dispatch
        
        # Apply Analytic Hierarchy Process (AHP) Weights
        w_tech = 0.5
        w_social = 0.5
        
        return (w_tech * decentralization_score) + (w_social * local_resource_score)

    # ==========================================
    # EXECUTION PIPELINE
    # ==========================================
    def run_analysis(self, carbon_steps: int = 10) -> pd.DataFrame:
        """
        Main execution method. Sweeps the Epsilon constraint to generate the Pareto front,
        then calculates the CSI for each viable point.
        """
        self.build_spatial_graph()
        self.apply_greedy_source_centric_routing()
        
        # Determine the maximum possible carbon baseline (to set the upper bound)
        # This is essentially the emissions if we met ALL demand using the dirtiest source
        total_demand = sum(self.graph.nodes[c]['demand'] for c in [n for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'consumer'])
        dirtiest_factor = max([d['carbon_factor'] for n, d in self.graph.nodes(data=True) if d.get('node_type') == 'producer'], default=0)
        max_carbon_baseline = total_demand * dirtiest_factor

        # Create the Epsilon bounds
        carbon_limits = np.linspace(max_carbon_baseline, 0, num=carbon_steps)
        
        results = []
        
        logger.info(f"Starting Pareto generation over {carbon_steps} steps...")
        
        for cap in carbon_limits:
            dispatch, cost, actual_carbon = self.solve_milp(cap)
            
            if dispatch is not None:
                csi = self.calculate_csi(dispatch)
                results.append({
                    'Carbon_Cap_Constraint': cap,
                    'Actual_Emissions_kg': actual_carbon,
                    'Total_System_OPEX': cost,
                    'Reduced_CSI_Score': csi,
                    'Feasible': True,
                    'Configuration': dispatch
                })
            else:
                # The model could not solve (e.g., carbon cap is too strict to meet demand)
                results.append({
                    'Carbon_Cap_Constraint': cap,
                    'Feasible': False
                })

        df_results = pd.DataFrame(results)
        return df_results

# ==========================================
# USAGE EXAMPLE
# ==========================================
if __name__ == "__main__":
    # In a real Django environment, you would call this from a view or task runner
    # optimizer = ThermalNetworkOptimizer(network_id=1)
    # pareto_dataframe = optimizer.run_analysis(carbon_steps=15)
    # print(pareto_dataframe[pareto_dataframe['Feasible'] == True])
    pass