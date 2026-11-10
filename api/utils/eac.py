import pandas as pd
import json
from django.utils import timezone
from oemof import solph

# Import your models
from api.models import Network, Node, LoadProfile, NetworkSimulationResult

# --- HELPER: Financial Math ---
def calculate_annuity(capex, n, wacc):
    if wacc == 0 or n == 0:
        return capex / n if n > 0 else 0
    return capex * (wacc * (1 + wacc) ** n) / ((1 + wacc) ** n - 1)

def run_oemof_simulation(network_id, hours=24, supply_temp=80, return_temp=60):
    try:
        # 1. SETUP
        network = Network.objects.get(id=network_id)
        nodes = Node.objects.filter(network=network)
        
        start_time = timezone.now().replace(minute=0, second=0, microsecond=0)
        time_index = pd.date_range(start_time, periods=hours, freq="h")
        es = solph.EnergySystem(timeindex=time_index)

        # 2. BUSES
        heat_bus = solph.Bus(label="heat_bus")
        electricity_bus = solph.Bus(label="electricity_bus") # Changed from gas to electricity
        es.add(heat_bus, electricity_bus)

        # --- DEFAULTS & PRICES ---
        ELEC_PRICE = 0.15        # $/kWh (Higher than gas)
        
        # Electric Boiler Parameters
        BOILER_CAPEX = 80        # $/kW (Electric boilers are usually cheaper to buy than gas ones)
        BOILER_LIFETIME = 20     # Years
        BOILER_EFFICIENCY = 0.99 # Very efficient
        
        # Thermal Storage Parameters
        STORAGE_CAPEX = 20       # $/kWh (Cost per unit of energy capacity)
        STORAGE_LIFETIME = 25    # Years
        STORAGE_LOSS_RATE = 0.01 # 1% heat loss per time step
        
        WACC = 0.05              # 5% Interest

        # Calculate Annualized Costs
        boiler_annuity = calculate_annuity(BOILER_CAPEX, BOILER_LIFETIME, WACC)
        storage_annuity = calculate_annuity(STORAGE_CAPEX, STORAGE_LIFETIME, WACC)

        # 3. ADD SOURCES
        
        # A. Electricity Grid Connection
        es.add(
            solph.components.Source(
                label="grid_electricity",
                outputs={electricity_bus: solph.flows.Flow(variable_costs=ELEC_PRICE)}
            )
        )

        # B. ELECTRIC BOILER (Optimized)
        es.add(
            solph.components.Converter(
                label="central_electric_boiler",
                inputs={electricity_bus: solph.flows.Flow()},
                outputs={
                    heat_bus: solph.flows.Flow(
                        nominal_value=solph.Investment(
                            ep_costs=boiler_annuity,
                            minimum=0,
                            maximum=100000
                        ),
                        variable_costs=0.01
                    )
                },
                conversion_factors={heat_bus: BOILER_EFFICIENCY}
            )
        )

        # C. THERMAL STORAGE (Optimized)
        # The solver decides the tank volume (nominal_storage_capacity)
        es.add(
            solph.components.GenericStorage(
                label="thermal_storage",
                inputs={heat_bus: solph.flows.Flow(
                    # We assume charge rate scales freely with capacity or is cheap to size
                    nominal_value=solph.Investment(ep_costs=0) 
                )},
                outputs={heat_bus: solph.flows.Flow(
                    # We assume discharge rate scales freely with capacity or is cheap to size
                    nominal_value=solph.Investment(ep_costs=0)
                )},
                loss_rate=STORAGE_LOSS_RATE,
                initial_storage_level=0.5, # Start half full
                balanced=False,            # Don't force end level = start level (for short runs)
                # INVESTMENT: Optimize the Volume (kWh)
                nominal_storage_capacity=solph.Investment(
                    ep_costs=storage_annuity,
                    minimum=0,
                    maximum=500000
                )
            )
        )

        # 4. NODES LOOP (DEMAND ONLY)
        for node in nodes:
            if hasattr(node, 'building') and node.building:
                building = node.building
                loadprofiles_qs = LoadProfile.objects.filter(building=building)
                
                if loadprofiles_qs.exists():
                    loadprofile_df = pd.DataFrame.from_records(loadprofiles_qs.values("value"))
                    if len(loadprofile_df) >= hours:
                        demand_values = loadprofile_df["value"].iloc[:hours]
                        max_demand = demand_values.max()
                        fix_values = demand_values / max_demand if max_demand > 0 else [0]*hours
                        
                        es.add(solph.components.Sink(
                            label=f"building_{building.id}",
                            inputs={heat_bus: solph.flows.Flow(nominal_value=max_demand, fix=fix_values)}
                        ))

        # 5. SOLVE
        model = solph.Model(es)
        model.solve(solver="cbc", solve_kwargs={"tee": True})

        # 6. CALCULATE COSTS & RESULTS
        results = solph.processing.results(model)
        
        total_capex = 0.0
        total_opex = 0.0
        optimal_boiler_size = 0.0
        optimal_storage_size = 0.0
        json_results = {}

        # Iterate results
        for (node_from, node_to), value in results.items():
            flow_label = f"{node_from.label}->{node_to.label}"
            
            # A. Flow Data
            if value.get('sequences') is not None:
                flow_values = value['sequences']['flow'].tolist()
                json_results[flow_label] = flow_values
                
                # OPEX: Electricity Cost
                if "grid_electricity" in node_from.label:
                    total_elec_kwh = sum(flow_values)
                    total_opex += total_elec_kwh * ELEC_PRICE

            # B. Investment Data (Boiler)
            if node_from.label == "central_electric_boiler" and "heat_bus" in node_to.label:
                size = value['scalars'].get('invest', 0.0)
                optimal_boiler_size = size
                total_capex += size * BOILER_CAPEX
                json_results["optimal_boiler_size_kw"] = size

        # C. Investment Data (Storage)
        # Storage investment is stored in the node itself, not a flow
        storage_node = [n for n in results.keys() if n[0].label == "thermal_storage" and n[1] is None]
        if storage_node:
            # Access the generic storage results
            storage_result = results[storage_node[0]]
            if 'scalars' in storage_result:
                 storage_cap = storage_result['scalars'].get('invest', 0.0)
                 optimal_storage_size = storage_cap
                 total_capex += storage_cap * STORAGE_CAPEX
                 json_results["optimal_storage_size_kwh"] = storage_cap

        # 7. SUMMARY
        final_summary = {
            "optimal_boiler_size_kw": round(optimal_boiler_size, 2),
            "optimal_storage_size_kwh": round(optimal_storage_size, 2),
            "total_upfront_capex_usd": round(total_capex, 2),
            "simulation_opex_usd": round(total_opex, 2),
            "simulation_hours": hours,
            "details": json_results
        }

        print(f"--- RESULTS ---")
        print(f"Boiler:  {final_summary['optimal_boiler_size_kw']} kW")
        print(f"Storage: {final_summary['optimal_storage_size_kwh']} kWh")
        print(f"CAPEX:   ${final_summary['total_upfront_capex_usd']}")
        print(f"OPEX:    ${final_summary['simulation_opex_usd']}")

        save_simulation_result(
            network=network,
            hours=hours,
            solver="cbc",
            supply_temp=supply_temp,
            return_temp=return_temp,
            ambient_temp=0,
            pump_efficiency=0.9,
            results_json=final_summary,
            status="success",
            solver_message="Optimal"
        )

        return final_summary

    except Exception as e:
        print(f"Error: {e}")
        return None

def save_simulation_result(network, hours, solver, supply_temp, return_temp, ambient_temp, pump_efficiency, results_json, status, solver_message):
    NetworkSimulationResult.objects.create(
        network=network,
        hours=hours,
        solver=solver,
        supply_temperature=supply_temp,
        return_temperature=return_temp,
        ambient_temperature=ambient_temp,
        pump_efficiency=pump_efficiency,
        results_json=json.dumps(results_json),
        status=status,
        solver_message=solver_message,
        created_at=timezone.now()
    )