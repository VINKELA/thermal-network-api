import os
import logging
import math
import bisect
import networkx as nx
import pyomo.environ as pyo
import numpy as np
import pandas as pd

# Visualization Libraries
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple, Optional

# Adjust these imports to match your Django app's structure
from api.models import Network, Node, Edge, LoadProfile
from django.db.models import Sum

# Restored Verbose Logging
logging.getLogger('pyomo.core').setLevel(logging.WARNING) # Only hide Pyomo internal math spam
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==========================================
# CANADIAN ENGINEERING CONSTANTS
# ==========================================
BOILER_CAPEX_PER_KW = 824.0          
TRENCH_CAPEX_PER_METER = 5313.0      
COMMERCIAL_BOILERS_KW = [88, 117, 147, 220, 293, 440, 586, 733, 879, 1172]
PROJECT_LIFESPAN_YEARS = 25
MAX_CONNECTION_DISTANCE_M = 1000.0  # 1km radius limit

# ==========================================
# PART 1: OPTIMIZER ENGINE
# ==========================================
class MacroNetworkOptimizer:
    """Hybrid Bottom-Up Macro Optimizer. Solves hourly dispatch and sizes equipment."""
    
    def __init__(self, deficit_network_id: int, excess_ledger: dict = None, time_steps: int = 8760):
        self.network_id = deficit_network_id
        self.excess_ledger = excess_ledger if excess_ledger is not None else {}
        self.network_name = ""
        self.graph = nx.Graph()
        self.time_steps = time_steps 
        self.total_trench_capex = 0.0
        self.derived_deficit_profile = np.zeros(time_steps)
        self.derived_source_profiles = {} 

    def _size_commercial_boiler(self, required_kw: float) -> float:
        if required_kw <= 0: return 0.0
        max_unit = COMMERCIAL_BOILERS_KW[-1]
        if required_kw > max_unit:
            num_max_units = math.floor(required_kw / max_unit)
            remainder = required_kw % max_unit
            total = num_max_units * max_unit
            if remainder > 0:
                idx = bisect.bisect_left(COMMERCIAL_BOILERS_KW, remainder)
                total += COMMERCIAL_BOILERS_KW[idx]
            return float(total)
        else:
            idx = bisect.bisect_left(COMMERCIAL_BOILERS_KW, required_kw)
            return float(COMMERCIAL_BOILERS_KW[idx])

    def _get_network_centroid(self, target_network_id: int) -> Tuple[Optional[float], Optional[float]]:
        network = Network.objects.get(id=target_network_id)
        if getattr(network, 'latitude', None) and getattr(network, 'longitude', None):
            return float(network.latitude), float(network.longitude)
            
        buildings = Node.objects.filter(network=network, building__isnull=False).select_related('building')
        lats, lons = [], []
        for n in buildings:
            if getattr(n.building, 'latitude', None) and getattr(n.building, 'longitude', None):
                lats.append(float(n.building.latitude))
                lons.append(float(n.building.longitude))
                
        if lats and lons:
            return sum(lats) / len(lats), sum(lons) / len(lons)
            
        logger.warning(f"Could not determine coordinates for Network {target_network_id}.")
        return None, None

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        R = 6371000  # Earth radius in meters
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)
        a = math.sin(delta_phi / 2.0)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2
        return R * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

    def build_spatial_graph(self) -> nx.Graph:
        try:
            deficit_network = Network.objects.get(id=self.network_id)
            self.network_name = deficit_network.name
            self.graph.graph['name'] = self.network_name
        except Network.DoesNotExist:
            raise ValueError(f"Deficit Network {self.network_id} not found.")

        # 1. DEFICIT NETWORK PROFILE
        deficit_buildings = Node.objects.filter(network=deficit_network, building__isnull=False).select_related('building')
        for node in deficit_buildings:
            if node.building:
                profiles = list(LoadProfile.objects.filter(building=node.building, profile_type='heating').order_by('start_date').values_list('value', flat=True))
                if len(profiles) >= self.time_steps: 
                    self.derived_deficit_profile += np.array(profiles[:self.time_steps])
                elif len(profiles) > 0:
                    padded = np.zeros(self.time_steps)
                    padded[:len(profiles)] = profiles
                    self.derived_deficit_profile += padded

        self.graph.add_node(
            'central_deficit_network', 
            node_type='network_consumer', math_role='consumer',
            demand_profile=self.derived_deficit_profile.tolist(), is_dynamic=False, needs_sizing=False
        )

        # 2. EXCESS LEDGER & 1KM MACRO-ROUTING
        deficit_coords = self._get_network_centroid(self.network_id)
        
        for source_net_id, available_supply_array in self.excess_ledger.items():
            if np.sum(available_supply_array) > 0:
                source_coords = self._get_network_centroid(source_net_id)
                
                if deficit_coords[0] is not None and source_coords[0] is not None:
                    distance_m = self._haversine_distance(
                        deficit_coords[0], deficit_coords[1], 
                        source_coords[0], source_coords[1]
                    )
                    
                    # THE 1KM GEOFENCE CONSTRAINT
                    if distance_m > MAX_CONNECTION_DISTANCE_M:
                        logger.info(f"   [SKIPPED] Source Network {source_net_id} is {distance_m:,.0f}m away (Exceeds 1km limit).")
                        continue
                        
                    self.total_trench_capex += (distance_m * TRENCH_CAPEX_PER_METER)
                    logger.info(f"   [ROUTED] Trunk Line: {distance_m:,.0f}m connected to Source Network {source_net_id}.")

                # Add to graph only if it passed the distance check
                self.derived_source_profiles[source_net_id] = available_supply_array
                self.graph.add_node(
                    f'source_network_{source_net_id}', 
                    node_type='network_producer', math_role='producer',
                    is_dynamic=True, needs_sizing=False, 
                    availability_profile=available_supply_array.tolist(), 
                    opex=0.01, carbon_factor=0.0     
                )

        # 3. VIRTUAL BOILER
        self.graph.add_node(
            'virtual_boiler', 
            node_type='boiler', math_role='producer', is_dynamic=False, needs_sizing=True, 
            max_capacity=0.0, opex=0.044, carbon_factor=0.202, is_virtual=True
        )

        return self.graph

    def evaluate_business_as_usual(self) -> dict:
        peak_demand_kw = np.max(self.derived_deficit_profile)
        total_annual_demand_kwh = np.sum(self.derived_deficit_profile)
        
        installed_boiler_kw = self._size_commercial_boiler(peak_demand_kw)
        capex = installed_boiler_kw * BOILER_CAPEX_PER_KW
        annual_opex = total_annual_demand_kwh * 0.044
        
        return {
            'Scenario': 'Business As Usual (100% Gas)',
            'Carbon_Cap_Constraint_kg': float('inf'), 
            'Actual_Emissions_kg': total_annual_demand_kwh * 0.202,
            'Solver_Raw_Boiler_Need_kW': peak_demand_kw,
            'Commercial_Installed_Boiler_kW': installed_boiler_kw,
            'Total_System_CAPEX_$': capex,
            'Total_Annual_OPEX_$': annual_opex,
            'Real_World_25_Year_Lifecycle_Cost_$': capex + (annual_opex * PROJECT_LIFESPAN_YEARS),
            'Reduced_CSI_Score': 0.0,
            'Feasible': True,
            'Raw_Timeseries_Dispatch': None 
        }

    def solve_milp(self, max_carbon_allowance: float) -> Optional[Dict]:
        model = pyo.ConcreteModel()
        model.Time = pyo.RangeSet(0, self.time_steps - 1)
        
        producers = [n for n in self.graph.nodes() if self.graph.nodes[n].get('math_role') == 'producer']
        boilers = [n for n in producers if self.graph.nodes[n].get('needs_sizing')]
        
        model.Producers = pyo.Set(initialize=producers)
        model.Boilers = pyo.Set(initialize=boilers)
        model.Dispatch = pyo.Var(model.Producers, model.Time, domain=pyo.NonNegativeReals)
        model.BoilerCapacity = pyo.Var(model.Boilers, domain=pyo.NonNegativeReals)
        
        def capacity_rule(m, p, t):
            if self.graph.nodes[p]['needs_sizing']: return m.Dispatch[p, t] <= m.BoilerCapacity[p]
            elif self.graph.nodes[p]['is_dynamic']: return m.Dispatch[p, t] <= self.graph.nodes[p]['availability_profile'][t]
        model.capacity_limit = pyo.Constraint(model.Producers, model.Time, rule=capacity_rule)
        
        def demand_rule(m, t):
            return sum(m.Dispatch[p, t] for p in m.Producers) >= self.graph.nodes['central_deficit_network']['demand_profile'][t]
        model.meet_demand = pyo.Constraint(model.Time, rule=demand_rule)
        
        def carbon_cap_rule(m):
            return sum(m.Dispatch[p, t] * self.graph.nodes[p]['carbon_factor'] for p in m.Producers for t in m.Time) <= max_carbon_allowance
        model.carbon_cap = pyo.Constraint(rule=carbon_cap_rule)
        
        def lifecycle_cost_objective(m):
            total_boiler_capex = sum(m.BoilerCapacity[b] * BOILER_CAPEX_PER_KW for b in m.Boilers)
            annual_opex = sum(m.Dispatch[p, t] * self.graph.nodes[p]['opex'] for p in m.Producers for t in m.Time)
            return total_boiler_capex + self.total_trench_capex + (annual_opex * PROJECT_LIFESPAN_YEARS)
            
        model.obj = pyo.Objective(rule=lifecycle_cost_objective, sense=pyo.minimize)
        
        result = pyo.SolverFactory('cbc').solve(model, tee=False)
        if result.solver.termination_condition == pyo.TerminationCondition.optimal:
            dispatch_config = {p: [pyo.value(model.Dispatch[p, t]) for t in model.Time] for p in model.Producers}
            return {
                'dispatch': dispatch_config,
                'raw_boiler_sizes': {b: pyo.value(model.BoilerCapacity[b]) for b in model.Boilers},
                'lifecycle_cost': pyo.value(model.obj),
                'actual_carbon': sum(sum(dispatch_config[p]) * self.graph.nodes[p]['carbon_factor'] for p in model.Producers)
            }
        return None

    def calculate_csi(self, dispatch_config: Dict[str, list]) -> float:
        if not dispatch_config: return 0.0
        annual_production = {p: sum(arr) for p, arr in dispatch_config.items()}
        total_dispatch = sum(annual_production.values())
        if total_dispatch == 0: return 0.0
        decentralization_score = 1.0 - (max(annual_production.values()) / total_dispatch)
        zero_carbon_dispatch = sum(prod for p, prod in annual_production.items() if self.graph.nodes[p]['carbon_factor'] == 0)
        return (0.5 * decentralization_score) + (0.5 * (zero_carbon_dispatch / total_dispatch))

    def run_pareto_analysis(self, carbon_steps: int = 10) -> pd.DataFrame:
        self.build_spatial_graph()
        results = [self.evaluate_business_as_usual()]
        
        total_annual_demand = np.sum(self.derived_deficit_profile)
        max_carbon_baseline = total_annual_demand * 0.202

        for cap in np.linspace(max_carbon_baseline, 0, num=carbon_steps):
            milp = self.solve_milp(cap)
            if milp:
                commercial_boiler_kw = sum(self._size_commercial_boiler(s) for s in milp['raw_boiler_sizes'].values())
                actual_capex = (commercial_boiler_kw * BOILER_CAPEX_PER_KW) + self.total_trench_capex
                annual_opex = sum(sum(milp['dispatch'][p]) * self.graph.nodes[p]['opex'] for p in milp['dispatch'])
                
                results.append({
                    'Scenario': f'Optimized Cap {cap:,.0f} kg',
                    'Carbon_Cap_Constraint_kg': cap,
                    'Actual_Emissions_kg': milp['actual_carbon'],
                    'Solver_Raw_Boiler_Need_kW': sum(milp['raw_boiler_sizes'].values()),
                    'Commercial_Installed_Boiler_kW': commercial_boiler_kw,
                    'Total_System_CAPEX_$': actual_capex,
                    'Total_Annual_OPEX_$': annual_opex,
                    'Real_World_25_Year_Lifecycle_Cost_$': actual_capex + (annual_opex * 25),
                    'Reduced_CSI_Score': self.calculate_csi(milp['dispatch']),
                    'Feasible': True,
                    'Raw_Timeseries_Dispatch': milp['dispatch']
                })
            else:
                results.append({'Scenario': f'Infeasible Cap {cap:,.0f} kg', 'Feasible': False})

        return pd.DataFrame(results)

# ==========================================
# PART 2: REPORTING (STATIC MATPLOTLIB + INTERACTIVE PLOTLY)
# ==========================================
def generate_comprehensive_report(pareto_df: pd.DataFrame, optimizer: MacroNetworkOptimizer, output_dir: str = "results"):
    """Generates both static PNGs (Matplotlib) and Interactive Dashboards (Plotly)."""
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    feasible_df = pareto_df[pareto_df['Feasible'] == True].copy()
    if feasible_df.empty: return

    PLOT_HOURS =8760
    time_index = range(PLOT_HOURS)
    
    # ---------------------------------------------------------
    # RESTORED MATPLOTLIB EXPORTS (For Thesis Document)
    # ---------------------------------------------------------
    # 1. MACRO PROFILES
    plt.figure(figsize=(14, 6))
    plt.plot(time_index, optimizer.derived_deficit_profile[:PLOT_HOURS], color='black', linewidth=2.5, label='Aggregated Deficit Demand')
    total_supply = np.zeros(PLOT_HOURS)
    for profile in optimizer.derived_source_profiles.values(): total_supply += profile[:PLOT_HOURS]
    plt.fill_between(time_index, 0, total_supply, color='#2ca02c', alpha=0.5, label='Aggregated Waste Heat Supply')
    plt.title('Derived Macro-Profiles: Demand vs. Available Waste Heat', fontsize=14, fontweight='bold')
    plt.xlabel('Hour of the Year', fontsize=12); plt.ylabel('Thermal Power (kW)', fontsize=12)
    plt.legend(loc='upper right', fontsize=11); plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "STATIC_1_derived_macro_profiles.png"), dpi=300)
    plt.close()

    optimized_df = feasible_df.dropna(subset=['Raw_Timeseries_Dispatch'])
    if len(optimized_df) >= 2:
        scenarios = pd.DataFrame([feasible_df.iloc[0], optimized_df.iloc[len(optimized_df) // 2], optimized_df.iloc[-1]])
        labels = ['BAU (100% Gas)', 'Optimized Balanced Mix', 'Strict Cap (Waste Heat Heavy)']
        
        # 2. ABSOLUTE BAU COMPARISON (Costs & Emissions)
        fig, (ax_cost, ax_env) = plt.subplots(1, 2, figsize=(14, 6))
        costs_millions = scenarios['Real_World_25_Year_Lifecycle_Cost_$'].values / 1_000_000
        bars_cost = ax_cost.bar(labels, costs_millions, color='#e41a1c', width=0.5)
        ax_cost.set_title('25-Year Lifecycle Cost', fontsize=13, fontweight='bold')
        ax_cost.set_ylabel('Cost (Million $CAD)', fontsize=11)
        ax_cost.grid(axis='y', linestyle='--', alpha=0.5)
        for bar in bars_cost: ax_cost.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'${bar.get_height():.2f}M', ha='center', va='bottom', fontweight='bold')

        emissions_tonnes = scenarios['Actual_Emissions_kg'].values / 1000
        bars_env = ax_env.bar(labels, emissions_tonnes, color='#4daf4a', width=0.5)
        ax_env.set_title('Annual Carbon Emissions', fontsize=13, fontweight='bold')
        ax_env.set_ylabel('Emissions (Tonnes CO2/yr)', fontsize=11)
        ax_env.grid(axis='y', linestyle='--', alpha=0.5)
        for bar in bars_env: ax_env.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{bar.get_height():,.0f} t', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "STATIC_2_absolute_bau_comparison.png"), dpi=300)
        plt.close()

    # 3. PEAK WEEK DISPATCH
    if not optimized_df.empty:
        best_config = optimized_df.loc[optimized_df['Reduced_CSI_Score'].idxmax()]
        boiler_dispatch, waste_dispatch = np.zeros(PLOT_HOURS), np.zeros(PLOT_HOURS)
        for node_id, dispatch_array in best_config['Raw_Timeseries_Dispatch'].items():
            if optimizer.graph.nodes[node_id]['node_type'] == 'boiler': boiler_dispatch += np.array(dispatch_array[:PLOT_HOURS])
            elif 'source_network' in str(node_id): waste_dispatch += np.array(dispatch_array[:PLOT_HOURS])
                
        plt.figure(figsize=(14, 6))
        plt.stackplot(time_index, waste_dispatch, boiler_dispatch, labels=['Aggregated Waste Heat', 'Virtual Boiler'], colors=['#2ca02c', '#d62728'], alpha=0.7)
        plt.plot(time_index, boiler_dispatch + waste_dispatch, color='black', linewidth=2, linestyle='--', label='Aggregated Demand')
        plt.title(f'Hourly Thermal Dispatch - Peak Week (CSI Score: {best_config["Reduced_CSI_Score"]:.3f})', fontsize=14, fontweight='bold')
        plt.xlabel('Hour of the Year', fontsize=12); plt.ylabel('Thermal Power (kW)', fontsize=12)
        plt.legend(loc='upper right', fontsize=11); plt.grid(True, linestyle='--', alpha=0.4)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "STATIC_3_dispatch_peak_week.png"), dpi=300)
        plt.close()

    # ---------------------------------------------------------
    # NEW PLOTLY EXPORTS (For Web Dashboard / Presentation)
    # ---------------------------------------------------------
    fig_pareto = px.scatter(
        feasible_df, x="Actual_Emissions_kg", y="Real_World_25_Year_Lifecycle_Cost_$",
        color="Reduced_CSI_Score", size="Commercial_Installed_Boiler_kW",
        hover_data=["Scenario", "Total_System_CAPEX_$"], color_continuous_scale="Viridis",
        title=f"Pareto Frontier: Cost vs. Emissions ({optimizer.network_name})",
        labels={"Actual_Emissions_kg": "Annual Carbon (kg CO2)", "Real_World_25_Year_Lifecycle_Cost_$": "25-Year Cost ($CAD)"}
    )
    fig_pareto.update_traces(mode='lines+markers', marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig_pareto.update_layout(template="plotly_white", hovermode="closest")
    fig_pareto.write_html(os.path.join(output_dir, "INTERACTIVE_1_pareto_frontier.html"))

    if not optimized_df.empty:
        dispatch_data = best_config['Raw_Timeseries_Dispatch']
        labels = ["Virtual Gas Boiler"]
        colors = ["#ef553b"]
        source_totals, source_indices = [], []
        
        boiler_total = sum([sum(arr) for nid, arr in dispatch_data.items() if optimizer.graph.nodes[nid]['node_type'] == 'boiler'])
        source_totals.append(boiler_total)
        
        current_idx = 1
        for node_id, dispatch_array in dispatch_data.items():
            if 'source_network' in str(node_id):
                labels.append(f"Waste Heat: {node_id}")
                colors.append("#00cc96")
                source_totals.append(sum(dispatch_array))
                source_indices.append(current_idx)
                current_idx += 1
                
        sink_idx = current_idx
        labels.append(f"Deficit Network: {optimizer.network_name}")
        colors.append("#636efa")

        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(pad=20, thickness=30, line=dict(color="black", width=0.5), label=labels, color=colors),
            link=dict(source=[0] + source_indices, target=[sink_idx] * len([0] + source_indices), value=source_totals, color="rgba(200, 200, 200, 0.4)")
        )])
        fig_sankey.update_layout(title_text=f"Annual Energy Dispatch Flow - {best_config['Scenario']}", template="plotly_white")
        fig_sankey.write_html(os.path.join(output_dir, "INTERACTIVE_2_annual_energy_sankey.html"))

def generate_portfolio_master_report(portfolio_data: list, output_dir: str):
    """Generates EXPLICIT side-by-side cost & emission comparisons across the entire project."""
    if not portfolio_data: return
    df = pd.DataFrame(portfolio_data).sort_values('Network')

    total_bau_cost, total_opt_cost = df['BAU_Cost'].sum(), df['Opt_Cost'].sum()
    total_bau_emissions, total_opt_emissions = df['BAU_Emissions'].sum(), df['Opt_Emissions'].sum()
    
    # ---------------------------------------------------------
    # PORTFOLIO MATPLOTLIB (Static PNG)
    # ---------------------------------------------------------
    x = np.arange(len(df['Network']))
    width = 0.35
    
    # 1. Static Cost Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, df['BAU_Cost'] / 1e6, width, label='BAU Cost', color='#e41a1c')
    ax.bar(x + width/2, df['Opt_Cost'] / 1e6, width, label='Optimized Cost', color='#4daf4a')
    ax.set_ylabel('Lifecycle Cost (Million CAD)')
    ax.set_title('Portfolio Cost Comparison (BAU vs Optimized)')
    ax.set_xticks(x); ax.set_xticklabels(df['Network'], rotation=45, ha='right')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "PORTFOLIO_STATIC_Cost_Comparison.png"), dpi=300)
    plt.close()

    # 2. Static Emissions Comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, df['BAU_Emissions'] / 1000, width, label='BAU Emissions', color='#e41a1c')
    ax.bar(x + width/2, df['Opt_Emissions'] / 1000, width, label='Optimized Emissions', color='#4daf4a')
    ax.set_ylabel('Annual Emissions (Tonnes CO2)')
    ax.set_title('Portfolio Emissions Comparison (BAU vs Optimized)')
    ax.set_xticks(x); ax.set_xticklabels(df['Network'], rotation=45, ha='right')
    ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "PORTFOLIO_STATIC_Emissions_Comparison.png"), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # PORTFOLIO PLOTLY (Interactive HTML)
    # ---------------------------------------------------------
    fig_cost = go.Figure(data=[
        go.Bar(name='Business As Usual (Gas)', x=df['Network'], y=df['BAU_Cost'], marker_color='#ef553b'),
        go.Bar(name='Optimized (Waste Heat)', x=df['Network'], y=df['Opt_Cost'], marker_color='#00cc96')
    ])
    fig_cost.update_layout(title=f"Portfolio Lifecycle Cost Comparison<br><sup>Total Savings: ${(total_bau_cost - total_opt_cost):,.2f}</sup>", barmode='group', template="plotly_white")
    fig_cost.write_html(os.path.join(output_dir, "PORTFOLIO_INTERACTIVE_1_Cost_Comparison.html"))

    fig_emissions = go.Figure(data=[
        go.Bar(name='Business As Usual (Gas)', x=df['Network'], y=df['BAU_Emissions'], marker_color='#ef553b'),
        go.Bar(name='Optimized (Waste Heat)', x=df['Network'], y=df['Opt_Emissions'], marker_color='#00cc96')
    ])
    fig_emissions.update_layout(title=f"Portfolio Carbon Emissions Comparison<br><sup>Carbon Avoided: {(total_bau_emissions - total_opt_emissions):,.0f} kg CO2</sup>", barmode='group', template="plotly_white")
    fig_emissions.write_html(os.path.join(output_dir, "PORTFOLIO_INTERACTIVE_2_Emissions_Comparison.html"))

    df.to_csv(os.path.join(output_dir, "PORTFOLIO_MASTER_DATA.csv"), index=False)

# ==========================================
# PART 3: PORTFOLIO ORCHESTRATOR
# ==========================================
class ProjectThermalOrchestrator:
    """Scans a Project, classifies networks by energy balance, and optimizes Deficit Networks."""
    def __init__(self, project_id: int, time_steps: int = 8760):
        self.project_id = project_id
        self.time_steps = time_steps
        self.networks_summary = {'deficit': [], 'excess': [], 'balanced': []}
        
    def _classify_project_networks(self):
        logger.info(f"--- Classifying Networks for Project ID: {self.project_id} ---")
        networks = Network.objects.filter(project_id=self.project_id)
        if not networks.exists(): 
            logger.warning("No networks found.")
            return
        
        for net in networks:
            buildings = Node.objects.filter(network=net, building__isnull=False).select_related('building')
            building_ids = [n.building.id for n in buildings if n.building]
            
            if not building_ids:
                logger.warning(f"Network '{net.name}' has no buildings attached. Defaulting to Balanced.")
                self.networks_summary['balanced'].append(net)
                continue

            total_heating_kwh = LoadProfile.objects.filter(building_id__in=building_ids, profile_type='heating').aggregate(Sum('value'))['value__sum'] or 0.0
            total_supply_kwh = LoadProfile.objects.filter(building_id__in=building_ids, profile_type='waste_heat').aggregate(Sum('value'))['value__sum'] or 0.0
            
            max_load = max(total_heating_kwh, total_supply_kwh)
            tolerance = max_load * 0.05 if max_load > 0 else 0
            
            if total_supply_kwh > (total_heating_kwh + tolerance): 
                self.networks_summary['excess'].append(net)
                logger.info(f"[EXCESS] '{net.name}': Surplus {total_supply_kwh - total_heating_kwh:,.0f} kWh")
            elif total_supply_kwh < (total_heating_kwh - tolerance): 
                self.networks_summary['deficit'].append(net)
                logger.info(f"[DEFICIT] '{net.name}': Shortfall {total_heating_kwh - total_supply_kwh:,.0f} kWh")
            else: 
                self.networks_summary['balanced'].append(net)
                logger.info(f"[BALANCED] '{net.name}'")
                
    def run_portfolio_optimization(self, carbon_steps: int = 10, base_output_dir: str = "project_results"):
        self._classify_project_networks()
        if not self.networks_summary['deficit']: return {}

        project_results = {}
        if not os.path.exists(base_output_dir): os.makedirs(base_output_dir)
        portfolio_master_data = []
        
        # 1. INITIALIZE GLOBAL LEDGER
        logger.info("--- Initializing Global Excess Ledger ---")
        global_excess_ledger = {}
        for excess_net in self.networks_summary['excess']:
            aggregated_supply = np.zeros(self.time_steps)
            buildings = Node.objects.filter(network=excess_net, building__isnull=False).select_related('building')
            for b_node in buildings:
                if b_node.building:
                    s_profs = list(LoadProfile.objects.filter(building=b_node.building, profile_type='waste_heat').order_by('start_date').values_list('value', flat=True))
                    if len(s_profs) >= self.time_steps: aggregated_supply += np.array(s_profs[:self.time_steps])
                    elif len(s_profs) > 0:
                        padded = np.zeros(self.time_steps)
                        padded[:len(s_profs)] = s_profs
                        aggregated_supply += padded
            if np.sum(aggregated_supply) > 0: 
                global_excess_ledger[excess_net.id] = aggregated_supply
                logger.info(f"Ledger added {np.sum(aggregated_supply):,.0f} kWh from Network {excess_net.id}")

        # 2. OPTIMIZATION LOOP
        for deficit_net in self.networks_summary['deficit']:
            logger.info(f"========== OPTIMIZING DEFICIT NETWORK: {deficit_net.name} ==========")
            try:
                optimizer = MacroNetworkOptimizer(deficit_network_id=deficit_net.id, excess_ledger=global_excess_ledger, time_steps=self.time_steps)
                pareto_df = optimizer.run_pareto_analysis(carbon_steps=carbon_steps)
                
                # 3. STATE UPDATE (DEDUCT ASSIGNED HEAT)
                optimized_df = pareto_df.dropna(subset=['Raw_Timeseries_Dispatch'])
                if not optimized_df.empty:
                    chosen_row = optimized_df.iloc[len(optimized_df) // 2]
                    for node_id, dispatch_array in chosen_row['Raw_Timeseries_Dispatch'].items():
                        if 'source_network_' in str(node_id):
                            net_id = int(node_id.replace('source_network_', ''))
                            heat_used = np.sum(dispatch_array)
                            global_excess_ledger[net_id] -= np.array(dispatch_array)
                            global_excess_ledger[net_id] = np.clip(global_excess_ledger[net_id], 0, None)
                            logger.info(f"   [LEDGER DEDUCT] {heat_used:,.0f} kWh claimed from Source Network {net_id}")

                    # MASTER REPORT TRACKING
                    bau_row = pareto_df.iloc[0]
                    portfolio_master_data.append({
                        'Network': deficit_net.name,
                        'BAU_Cost': bau_row['Real_World_25_Year_Lifecycle_Cost_$'],
                        'Opt_Cost': chosen_row['Real_World_25_Year_Lifecycle_Cost_$'],
                        'BAU_Emissions': bau_row['Actual_Emissions_kg'],
                        'Opt_Emissions': chosen_row['Actual_Emissions_kg']
                    })

                net_safe_name = deficit_net.name.replace(" ", "_").lower()
                net_output_dir = os.path.join(base_output_dir, f"network_{deficit_net.id}_{net_safe_name}")
                generate_comprehensive_report(pareto_df, optimizer, output_dir=net_output_dir)
                
                csv_df = pareto_df.drop(columns=['Raw_Timeseries_Dispatch'])
                csv_df.to_csv(os.path.join(net_output_dir, "pareto_data.csv"), index=False)
                project_results[deficit_net.name] = "Success"
                logger.info(f"   -> Successfully completed {deficit_net.name}")
            except Exception as e:
                project_results[deficit_net.name] = f"Failed: {str(e)}"
                logger.error(f"   -> FAILED {deficit_net.name}: {str(e)}")
                
        # GENERATE MASTER REPORT
        if portfolio_master_data:
            logger.info("--- Generating Final Portfolio Master Reports ---")
            generate_portfolio_master_report(portfolio_master_data, base_output_dir)
                
        return project_results