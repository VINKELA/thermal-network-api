from datetime import datetime
import pandas as pd
from oemof import solph
from api.models import Network, Node, Edge, LoadProfile, Building, Boiler  # <-- Added Boiler
from api.utils.pipe_thermal_hydraulic import compute_branch_requirements

from django.utils.timezone import now
from api.models import NetworkSimulationResult
def run_oemof_simulation(network_id, hours=24, supply_temp=80, return_temp=60, ambient_temp=0):
    try:
        # Get network from DB
        network = Network.objects.get(id=network_id)
        nodes = Node.objects.filter(network=network)
        edges = Edge.objects.filter(network=network)

        # Create time index for simulation
        time_index = pd.date_range(datetime.now(), periods=hours, freq="h")
        es = solph.EnergySystem(timeindex=time_index)

        # Define buses
        heat_bus = solph.Bus(label="heat_bus")
        waste_bus = solph.Bus(label="waste_bus")
        gas_bus = solph.Bus(label="gas_bus")
        electricity_bus = solph.Bus(label="electricity_bus")
        es.add(heat_bus, waste_bus, gas_bus, electricity_bus)

        # Map Django nodes to oemof buses
        buses = {}
        es.add(
    solph.components.Source(
        label="natural_gas_supply",
        outputs={gas_bus: solph.flows.Flow(variable_costs=10)}))
        # Add components based on node type
        for node in nodes:
            # --- BUILDING ---
            if node.building :
                building = Building.objects.get(id=node.building.id)
                loadprofiles_qs = LoadProfile.objects.filter(building=building)

                if not loadprofiles_qs.exists():
                    continue
                loadprofile_df = pd.DataFrame.from_records(
                    loadprofiles_qs.values("value", "start_date", "end_date")
                )
                details = compute_branch_requirements(
                    laod_profile=loadprofile_df["value"]    ,
                    dt=supply_temp - return_temp,
                    v=1.0,
                    supply_temp_c=supply_temp,
                )
                fix_values = loadprofile_df["value"] / loadprofile_df["value"].max()

                if building.load_type == 'heating_dominant':
                    es.add(
                        solph.components.Sink(
                            label=f"heat_sink_building_{building.id}",
                            inputs={
                                heat_bus: solph.flows.Flow(
                                    nominal_value=loadprofile_df["value"].max(),
                                    fix=fix_values
                                )
                            }
                        )
                    )
                elif building.demand_type == 'cooling_dominant':
                    es.add(
                        solph.components.Source(
                            label=f"waste_heat_source_building_{building.id}",
                            outputs={
                                heat_bus: solph.flows.Flow(
                                    nominal_value=loadprofile_df["value"].max(),
                                    variable_costs=2,
                                    fix = fix_values
                                ),
                            }
                        )
                    )
 
            # --- BOILER ---
            elif node.boiler:
                boiler = Boiler.objects.get(id=node.boiler.id)

                # Example: boiler takes gas and produces heat
                es.add(
                    solph.components.Converter(
                        label=f"boiler_{boiler.id}",
                        inputs={gas_bus: solph.flows.Flow()},
                        outputs={
                                heat_bus: solph.flows.Flow(
                                nominal_capacity=10000000000,  # Assuming Boiler model has capacity
                                variable_costs=boiler.variable_cost if hasattr(boiler, 'variable_cost') else 0
                            )
                        },
                        conversion_factors={gas_bus: boiler.efficiency if hasattr(boiler, 'efficiency') else 0.9}
                    )
                )
        #model pipe line and the losses
        model = solph.Model(es)
        model.solve(solver="cbc", solve_kwargs={"tee": True})

        # Collect results
        results = solph.processing.results(model)
        for (from_obj, to_obj), flow in results.items():
            print(f"Flow from {from_obj} to {to_obj}:")
            print(flow)
        NetworkSimulationResult.objects.create(
            network=network,
            hours=hours,
            solver="cbc",
            supply_temperature=supply_temp,
            return_temperature=return_temp,
            ambient_temperature=ambient_temp,
            results_json={
                "results": str(results)  # Convert to string for JSONField
            },
            status="success",
            termination_condition="model.Solver.termination_condition",
            solver_message="OK"
        )

        return results

    except Exception as e:
        print(f"Error running oemof simulation: {e}")
        return None


def save_simulation_result(
    network,
    hours,
    solver,
    supply_temp,
    return_temp,
    ambient_temp,
    pump_efficiency,
    results_json,
    total_heat_demand_kwh=None,
    total_heat_supplied_kwh=None,
    total_losses_kwh=None,
    total_pump_power_kwh=None,
    status="success",
    termination_condition=None,
    solver_message=None
):
    """
    Save a simulation run to NetworkSimulationResult.
    """
    sim_result = NetworkSimulationResult.objects.create(
        network=network,
        hours=hours,
        solver=solver,
        supply_temperature=supply_temp,
        return_temperature=return_temp,
        ambient_temperature=ambient_temp,
        pump_efficiency=pump_efficiency,
        total_heat_demand_kwh=total_heat_demand_kwh,
        total_heat_supplied_kwh=total_heat_supplied_kwh,
        total_losses_kwh=total_losses_kwh,
        total_pump_power_kwh=total_pump_power_kwh,
        results_json=results_json,
        status=status,
        termination_condition=termination_condition,
        solver_message=solver_message
    )
    return sim_result


if __name__ == "__main__":
    network_id = 1  # Replace with your network ID
    results = run_oemof_simulation(network_id)
    print("Simulation results:", results)