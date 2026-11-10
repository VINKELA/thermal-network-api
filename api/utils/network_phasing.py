"""
Network phasing / evolution engine.

This module answers the planning question:

    Given an already-routed base network, which extra buildings or
    complementary clusters should be connected, in which planning year,
    under which demand / fuel-price scenario?

Important terms used throughout the file:

- candidate row:
    A plain Python dictionary representing one possible future connection.
    A candidate can be one building or one complementary cluster. It stores
    heat demand, connection route length, distance to trunk, estimated CAPEX,
    linear heat density, suitability score, and the service route geometry.

- period:
    A planning phase, usually represented by a year such as 2028, 2030, 2035.
    A period can optionally carry phase gates: available heat, CAPEX budget,
    and maximum new connections.

- scenario:
    A demand / economics case, such as low, reference, or high demand.
    Scenarios carry probability, demand multiplier, discount rate, and gas
    price multiplier.

- expected-value decision mode:
    The default planning style. One shared build-out plan is optimized using
    the probability-weighted value across all scenarios. The same selected
    candidates appear in every scenario, but reported heat and value differ.

- scenario-specific decision mode:
    Each scenario is optimized independently. Low, reference, and high demand
    can therefore select different candidates or connect the same candidates
    in different years.

- selected_by_key / selected_by_year:
    Compact solver outputs. selected_by_key maps a candidate key to its first
    selected year. selected_by_year maps a year to the candidate rows newly
    connected in that year.

- PhasingResult rows:
    Database rows that materialize the paper-style indexed decisions:
    N_i,t,m, Nbar_i,t,m, E_i,j,t,m, Ebar_i,j,t,m, P_i,t,m, and Q_i,t,m.
"""

import copy
import datetime as dt
import math
from collections import defaultdict

import requests
from django.db import transaction
from django.db.models import Count, Q, Sum
from pyproj import Transformer
from shapely.geometry import LineString, Point
from shapely.ops import transform as shapely_transform

from api.models import (
    Building,
    BuildingCluster,
    CandidateConnection,
    CarbonTaxRate,
    Cluster,
    ClusterConfiguration,
    Edge,
    EdgeTranslation,
    Network,
    OntarioNaturalGasRate,
    PhasingResult,
    PhasingStudy,
    PlanningPeriod,
    PlanningScenario,
    Route,
    LoadProfile,
    Node,
    assert_network_project_integrity,
)
from api.utils.network_versioning import (
    build_route_data,
    clone_network,
    decode_polyline,
    encode_polyline,
    normalize_route_data,
    route_distance,
)
from api.utils.network_expansion import _clone_source_routes, _route_travel_mode, _split_routes_at_infill_taps
from api.utils.route_geometry import unique_route_length


DEFAULT_PERIODS = [2028, 2030, 2035]
DEFAULT_SCENARIOS = [
    {"name": "Low demand", "probability": 0.25, "discount_rate": 0.045, "demand_multiplier": 0.85, "fuel_price_multiplier": 0.85},
    {"name": "Reference demand", "probability": 0.50, "discount_rate": 0.045, "demand_multiplier": 1.00, "fuel_price_multiplier": 1.00},
    {"name": "High demand", "probability": 0.25, "discount_rate": 0.045, "demand_multiplier": 1.20, "fuel_price_multiplier": 1.25},
]
DEFAULT_SCORE_WEIGHTS = {
    "linear_heat_density": 0.35,
    "heat_demand": 0.20,
    "connection_cost": 0.18,
    "distance_to_trunk": 0.15,
    "available_heat_fit": 0.08,
    "diversity": 0.04,
}
DEFAULT_PHASING_ASSUMPTIONS = {
    "project_lifespan_years": 30,
    "electricity_cost_cad_per_kwh": 0.15,
    "grid_emission_factor_kgco2e_per_kwh": 0.040,
    "gas_emission_factor_kgco2e_per_kwh": 0.185,
    "base_boiler_efficiency": 0.90,
    "pump_efficiency": 0.75,
    "target_pressure_drop_pa_per_m": 100.0,
    "design_delta_t_c": 30.0,
    "max_velocity_m_s": 2.5,
    "min_velocity_m_s": 0.5,
}
UNLIMITED_AVAILABLE_HEAT_KW = 1_000_000_000_000.0
GREEDY_SOLVER_NAME = "deterministic_greedy_expected_value"
MILP_SOLVER_NAME = "milp_expected_value_highs"
FULL_CHRONOLOGY_SOLVER_MODE = "full_chronology_milp"
FULL_CHRONOLOGY_SOLVER_NAME = "milp_full_chronology_highs"
EXPECTED_VALUE_DECISION_MODE = "expected_value"
SCENARIO_SPECIFIC_DECISION_MODE = "scenario_specific"


def normalize_phasing_params(raw_params):
    """Normalize API/UI input into one predictable parameter dictionary.

    The frontend sends a mix of user-facing controls and optional advanced
    fields. This function converts aliases, defaults, strings, booleans, and
    blank values into the canonical keys used by the rest of the engine.
    Keeping that cleanup here prevents every downstream function from having
    to handle multiple spellings of the same idea.
    """
    params = dict(raw_params or {})
    params["name"] = params.get("name") or params.get("study_name")
    params["description"] = params.get("description") or ""

    # Planning periods are the time axis of the phasing model. Each period can
    # be just a year, or a richer object with heat/budget/connection limits.
    params["periods"] = _normalize_periods(params.get("periods") or params.get("years"))

    # Scenarios are the uncertainty axis. They do not create buildings or
    # routes; they scale demand/value and, in scenario-specific mode, can cause
    # different selections.
    params["scenarios"] = _normalize_scenarios(params.get("scenarios"))

    # Solver mode controls the optimization algorithm. Scenario decision mode
    # controls whether there is one shared plan or one plan per scenario.
    params["solver_mode"] = _normalize_solver_mode(
        params.get("solver_mode") or params.get("solver") or params.get("optimizer_mode")
    )
    params["scenario_decision_mode"] = _normalize_scenario_decision_mode(
        params.get("scenario_decision_mode") or
        params.get("decision_mode") or
        params.get("scenario_mode")
    )
    params["max_distance_m"] = _float_value(params.get("max_distance_m"), 250.0)
    raw_min_lhd = params.get("min_linear_heat_density_mwh_per_m_year")
    if raw_min_lhd in [None, ""]:
        raw_min_lhd = params.get("min_lhd")
    params["min_linear_heat_density_mwh_per_m_year"] = _float_value(raw_min_lhd, 0.05)
    params["available_heat_kw"] = _float_value(params.get("available_heat_kw"), None)
    params["demand_year"] = _int_value(
        params.get("demand_year") or
        params.get("profile_year") or
        params.get("heat_profile_year"),
        None,
    )
    params["annual_full_load_hours"] = _float_value(params.get("annual_full_load_hours"), None)
    params["pipe_cost_per_m"] = _float_value(params.get("pipe_cost_per_m"), None)
    params["heat_price_cad_per_mwh"] = _float_value(params.get("heat_price_cad_per_mwh"), 90.0)
    params["gas_price_cad_per_mwh"] = _float_value(params.get("gas_price_cad_per_mwh"), None)
    params["gas_price_region"] = params.get("gas_price_region") or "Union South Rate Zone"
    params["project_lifespan_years"] = _int_value(params.get("project_lifespan_years") or params.get("lifespan_years"), None)
    params["electricity_cost_cad_per_kwh"] = _float_value(
        params.get("electricity_cost_cad_per_kwh") or params.get("electricity_cost"),
        None,
    )
    params["grid_emission_factor_kgco2e_per_kwh"] = _float_value(
        params.get("grid_emission_factor_kgco2e_per_kwh") or params.get("grid_emission_factor"),
        None,
    )
    params["gas_emission_factor_kgco2e_per_kwh"] = _float_value(
        params.get("gas_emission_factor_kgco2e_per_kwh") or params.get("gas_emission_factor"),
        None,
    )
    params["base_boiler_efficiency"] = _float_value(
        params.get("base_boiler_efficiency") or params.get("boiler_efficiency"),
        None,
    )
    params["pump_efficiency"] = _float_value(params.get("pump_efficiency"), None)
    params["target_pressure_drop_pa_per_m"] = _float_value(
        params.get("target_pressure_drop_pa_per_m") or params.get("target_pressure_drop"),
        None,
    )
    params["design_delta_t_c"] = _float_value(params.get("design_delta_t_c") or params.get("design_delta_t"), None)
    params["max_velocity_m_s"] = _float_value(params.get("max_velocity_m_s") or params.get("max_velocity"), None)
    params["min_velocity_m_s"] = _float_value(params.get("min_velocity_m_s") or params.get("min_velocity"), None)
    params["carbon_price_cad_per_tonne"] = _float_value(params.get("carbon_price_cad_per_tonne"), None)
    params["assume_unlimited_available_heat"] = _bool_value(
        params.get("assume_unlimited_available_heat") or
        params.get("ignore_available_heat_constraint") or
        params.get("unlimited_available_heat"),
        False,
    )
    params["candidate_source_mode"] = _normalize_candidate_source_mode(
        params.get("candidate_source_mode") or
        params.get("candidate_asset_scope") or
        params.get("candidate_scope")
    )
    params["include_buildings"] = params["candidate_source_mode"] != "clusters_only"
    params["include_clusters"] = _bool_value(params.get("include_clusters"), False)
    params["create_networks"] = _bool_value(params.get("create_networks"), True)
    params["route_mode"] = params.get("route_mode") or "driving"
    params["route_candidate_preview"] = _bool_value(params.get("route_candidate_preview"), False)
    params["candidate_limit"] = _int_value(params.get("candidate_limit"), 1000)
    params["candidate_selection_mode"] = params.get("candidate_selection_mode") or params.get("candidate_mode") or "auto"
    params["candidate_ids"] = _list_value(params.get("candidate_ids") or params.get("building_ids"))
    params["cluster_ids"] = _list_value(params.get("cluster_ids"))
    params["complementary_cluster_configuration_id"] = _int_value(
        params.get("complementary_cluster_configuration_id") or
        params.get("cluster_configuration_id"),
        None,
    )
    params["tags"] = _list_value(params.get("tags") or params.get("tag"))
    if params["candidate_source_mode"] == "clusters_only":
        params["include_buildings"] = False
        params["include_clusters"] = True
        params["candidate_ids"] = []
        params["tags"] = []
    elif params["candidate_source_mode"] == "buildings_only":
        params["include_buildings"] = True
        params["include_clusters"] = False
        params["cluster_ids"] = []
        params["complementary_cluster_configuration_id"] = None
    elif params["cluster_ids"] or params["complementary_cluster_configuration_id"]:
        params["include_clusters"] = True
    params["weights"] = _normalize_weights(params.get("score_weights") or params.get("weights"))
    return params


def preview_network_phasing(base_network, raw_params=None):
    """Run the phasing engine without writing a PhasingStudy to the database."""
    params = normalize_phasing_params(raw_params)
    base_network = Network.objects.select_related("project").get(pk=base_network.pk)
    assert_network_project_integrity(base_network)
    context = _build_context(base_network, params)
    candidate_rows = _candidate_rows(base_network, params, context)
    plan = _solve_phasing_plan(candidate_rows, params, context)
    return _preview_payload(base_network, params, context, candidate_rows, plan)


def create_network_phasing_study(base_network, raw_params=None, *, task_id=None):
    """Run phasing and persist the complete study, decisions, and outputs.

    This is the durable version of preview_network_phasing. It creates:
    - PhasingStudy: the study header and input assumptions.
    - PlanningPeriod / PlanningScenario rows: the t and m dimensions.
    - CandidateConnection rows: the candidate i dimension.
    - Optional child networks: physical phase outputs.
    - PhasingResult rows: the decision matrix for every candidate-period-
      scenario combination.
    """
    params = normalize_phasing_params(raw_params)
    base_network = Network.objects.select_related("project").get(pk=base_network.pk)
    assert_network_project_integrity(base_network)
    context = _build_context(base_network, params)
    candidate_rows = _candidate_rows(base_network, params, context)
    plan = _solve_phasing_plan(candidate_rows, params, context)

    with transaction.atomic():
        # Store the parameter payload alongside the study so the UI can reopen
        # old studies exactly as they were run, including phase gates.
        study = PhasingStudy.objects.create(
            project=base_network.project,
            base_network=base_network,
            name=params["name"] or f"{base_network.name} - Network Phasing",
            description=params["description"],
            status=PhasingStudy.Status.RUNNING,
            parameters={
                **_phasing_parameter_payload(params, context),
                "input_network_id": params.get("input_network_id") or base_network.id,
                "preparation_pipeline": params.get("preparation_pipeline"),
                "solver_metadata": plan.get("solver_metadata", {}),
            },
            solver=plan.get("solver") or GREEDY_SOLVER_NAME,
            task_id=task_id,
        )

        period_map = {}
        for sequence, period_data in enumerate(params["periods"], start=1):
            # A blank period heat field means "fall back to the base network
            # heat availability"; unlimited heat is stored as NULL in the row
            # and explicitly documented in study.parameters.
            period_available_heat_kw = None if params["assume_unlimited_available_heat"] else (
                period_data.get("available_heat_kw")
                if period_data.get("available_heat_kw") is not None
                else context["base_available_heat_kw"]
            )
            period = PlanningPeriod.objects.create(
                study=study,
                sequence=sequence,
                year=period_data["year"],
                label=period_data.get("label") or f"Phase {sequence} ({period_data['year']})",
                available_heat_kw=period_available_heat_kw,
                capex_budget_cad=period_data.get("capex_budget_cad"),
            )
            period_map[period.year] = period

        scenario_map = {}
        for scenario_data in params["scenarios"]:
            scenario = PlanningScenario.objects.create(
                study=study,
                name=scenario_data["name"],
                probability=scenario_data["probability"],
                discount_rate=scenario_data["discount_rate"],
                demand_multiplier=scenario_data["demand_multiplier"],
                fuel_price_multiplier=scenario_data.get("fuel_price_multiplier", 1.0),
                uncertainty_revealed_year=scenario_data.get("uncertainty_revealed_year"),
            )
            scenario_map[scenario.name] = scenario

        candidate_map = {}
        for row in candidate_rows:
            candidate = CandidateConnection.objects.create(
                study=study,
                candidate_type=row["candidate_type"],
                building_id=row.get("building_id"),
                cluster_id=row.get("cluster_id"),
                name=row["name"],
                candidate_role=row["candidate_role"],
                available_from_year=row.get("available_from_year"),
                peak_heat_kw=row["peak_heat_kw"],
                annual_heat_mwh=row["annual_heat_mwh"],
                available_heat_kw=row["available_heat_kw"],
                distance_to_trunk_m=row["distance_to_trunk_m"],
                connection_length_m=row["connection_length_m"],
                estimated_connection_cost_cad=row["estimated_connection_cost_cad"],
                linear_heat_density_mwh_per_m_year=row["linear_heat_density_mwh_per_m_year"],
                suitability_score=row["suitability_score"],
                status=row["status"],
                rejection_reason=row.get("rejection_reason"),
                snap_latitude=row["snap_point"]["lat"] if row.get("snap_point") else None,
                snap_longitude=row["snap_point"]["lng"] if row.get("snap_point") else None,
                source_route_id=row.get("source_route_id"),
                source_edge_id=row.get("source_edge_id"),
                metadata={
                    "score_breakdown": row.get("score_breakdown", []),
                    "service_path": row.get("service_path", []),
                    "reference_hydraulic_economics": row.get("reference_hydraulic_economics", {}),
                    "building_count": row.get("building_count", 0),
                },
            )
            candidate_map[row["key"]] = candidate

        output_networks_by_year = {}
        if params["create_networks"]:
            # Expected-value mode creates one child network per phase. Scenario-
            # specific mode creates one child network per scenario/phase because
            # the selected candidate set can differ by scenario.
            output_networks_by_year = (
                _create_scenario_phase_networks(base_network, study, plan, candidate_map, params)
                if params.get("scenario_decision_mode") == SCENARIO_SPECIFIC_DECISION_MODE
                else _create_phase_networks(base_network, study, plan, candidate_map, params)
            )

        _persist_results(study, plan, candidate_map, period_map, scenario_map, output_networks_by_year, params)
        objective = sum(result["discounted_net_benefit_cad"] for result in plan["results"] if result["node_newly_connected"])
        study.objective_value = objective
        study.status = PhasingStudy.Status.COMPLETED
        study.save(update_fields=["objective_value", "status", "updated_at"])

    return phasing_study_report(study)


def phasing_study_report(study):
    study = PhasingStudy.objects.select_related("project", "base_network").get(pk=study.pk)
    periods = list(study.periods.all().order_by("sequence", "year"))
    scenarios = list(study.scenarios.all())
    candidates = list(study.candidates.select_related("building", "cluster", "source_route", "source_edge"))
    results = list(study.results.select_related("period", "scenario", "candidate", "output_network"))
    service_route_map = _service_route_payload_map(results)
    report_context = _context_from_parameter_payload(study.parameters or {})
    period_limits_by_year = {
        _int_value(period.get("year"), None): period
        for period in (study.parameters or {}).get("periods", [])
        if isinstance(period, dict)
    }

    phase_rows = []
    for period in periods:
        period_rows = [result for result in results if result.period_id == period.id]
        rows = [result for result in period_rows if result.node_newly_connected]
        unique_rows = _unique_candidate_results(rows)
        output = next((result.output_network for result in period_rows if result.output_network_id), None)
        phase_rows.append({
            "year": period.year,
            "label": period.label,
            "output_network_id": output.id if output else None,
            "output_network_name": output.name if output else None,
            "new_connections": len(unique_rows),
            "added_heat_kw": sum(result.candidate.peak_heat_kw for result in unique_rows if result.candidate.candidate_role == CandidateConnection.CandidateRole.CONSUMER),
            "added_source_kw": sum(result.candidate.available_heat_kw for result in unique_rows if result.candidate.candidate_role == CandidateConnection.CandidateRole.SOURCE),
            "capex_cad": sum(result.capex_cad for result in unique_rows),
            "cumulative_capex_cad": max([result.cumulative_capex_cad for result in period_rows] or [0.0]),
            "cumulative_heat_kw": max([result.cumulative_heat_kw for result in period_rows] or [0.0]),
            "candidates": [_candidate_result_payload(result, service_route_map) for result in unique_rows],
        })

    result_payloads = [_result_payload(result, service_route_map, report_context) for result in results]
    return {
        "id": study.id,
        "name": study.name,
        "status": study.status,
        "base_network": _network_payload(study.base_network),
        "parameters": study.parameters or {},
        "solver": study.solver,
        "objective_value": study.objective_value,
        "error": study.error,
        "periods": [_period_payload(period, period_limits_by_year.get(period.year)) for period in periods],
        "scenarios": [_scenario_payload(scenario) for scenario in scenarios],
        "candidates": [_candidate_payload(candidate) for candidate in candidates],
        "results": result_payloads,
        "timeline": phase_rows,
        "charts": _chart_payload(phase_rows),
        "map": _map_payload(study.base_network, result_payloads),
        "created_at": study.created_at,
        "updated_at": study.updated_at,
    }


def _build_context(network, params):
    """Collect fixed network, route, heat, economic, and hydraulic assumptions.

    The solver receives this one context dictionary instead of reaching back
    into Django models repeatedly. That keeps the optimization code focused on
    decisions while this function handles data discovery: source routes, main
    trunk, network constants, available heat, carbon price, and gas price basis.
    """
    routes = _source_routes(network)
    edges = list(Edge.objects.filter(network=network).select_related("start_node", "end_node").order_by("id"))
    trunk = _main_trunk(routes, edges)
    # The simplified thesis assumption currently used here is:
    # main trunk = longest existing routed line/edge in the base network.
    route_index = _build_route_index([trunk["route"]]) if trunk.get("route") else _build_edge_index([trunk["edge"]])
    constants = _network_design_constants(network)
    inventory = _network_heat_inventory(network)
    gas_rate = _gas_rate_basis(params)
    first_period_year = (params["periods"][0] or {}).get("year") if params.get("periods") else dt.date.today().year
    source_heat_kw = _float_value(getattr(inventory, "total_waste_heat_kw", None), None)
    served_heat_kw = _float_value(getattr(inventory, "total_heating_load_kw", None), None)
    inferred_source_kw, inferred_served_kw = _infer_network_heat(network, params["demand_year"])
    if source_heat_kw is None:
        source_heat_kw = inferred_source_kw
    if served_heat_kw is None:
        served_heat_kw = inferred_served_kw
    # Available heat is the unused producer capacity already present in the
    # base network. Users can override it or mark heat as effectively unlimited
    # when they want linear heat density / economics to drive the selection.
    inferred_available_heat_kw = max(_float_value(source_heat_kw, 0.0) - _float_value(served_heat_kw, 0.0), 0.0)
    configured_available_heat_kw = params["available_heat_kw"]
    if configured_available_heat_kw is None:
        configured_available_heat_kw = inferred_available_heat_kw
    base_available_heat_kw = (
        UNLIMITED_AVAILABLE_HEAT_KW
        if params["assume_unlimited_available_heat"]
        else configured_available_heat_kw
    )

    cp_value = _context_value(params, constants, "cp", None, 4.19)
    cp_j_per_kg_k = cp_value * 1000.0 if cp_value and cp_value < 100.0 else cp_value
    project_lifespan_years = max(_int_value(
        params.get("project_lifespan_years"),
        getattr(constants, "lifespan_years", None) or DEFAULT_PHASING_ASSUMPTIONS["project_lifespan_years"],
    ), 1)
    carbon_price = params["carbon_price_cad_per_tonne"]
    carbon_price_source = "manual"
    if carbon_price is None:
        carbon_basis = _carbon_tax_basis_for_year(first_period_year)
        carbon_price = carbon_basis["price_cad_per_tonne"]
        carbon_price_source = carbon_basis["source"]

    return {
        "routes": routes,
        "edges": edges,
        "trunk": trunk,
        "route_index": route_index,
        "rho": max(_context_value(params, constants, "rho", None, 971.8), 1.0),
        "cp_j_per_kg_k": max(_float_value(cp_j_per_kg_k, 4190.0), 1.0),
        "mu": max(_context_value(params, constants, "mu", None, 0.000355), 0.000001),
        "annual_full_load_hours": params["annual_full_load_hours"] or getattr(constants, "hours_per_year", None) or 4000.0,
        "pipe_cost_per_m": params["pipe_cost_per_m"] or getattr(constants, "fallback_pipe_cost_per_meter", None) or 150.0,
        "project_lifespan_years": project_lifespan_years,
        "electricity_cost_cad_per_kwh": max(_context_value(
            params,
            constants,
            "electricity_cost",
            "electricity_cost_cad_per_kwh",
            DEFAULT_PHASING_ASSUMPTIONS["electricity_cost_cad_per_kwh"],
        ), 0.0),
        "grid_emission_factor_kgco2e_per_kwh": max(_context_value(
            params,
            constants,
            None,
            "grid_emission_factor_kgco2e_per_kwh",
            DEFAULT_PHASING_ASSUMPTIONS["grid_emission_factor_kgco2e_per_kwh"],
        ), 0.0),
        "gas_emission_factor_kgco2e_per_kwh": max(_context_value(
            params,
            constants,
            None,
            "gas_emission_factor_kgco2e_per_kwh",
            DEFAULT_PHASING_ASSUMPTIONS["gas_emission_factor_kgco2e_per_kwh"],
        ), 0.0),
        "base_boiler_efficiency": _bounded_float(
            params.get("base_boiler_efficiency"),
            getattr(gas_rate, "assumed_boiler_efficiency", None) or DEFAULT_PHASING_ASSUMPTIONS["base_boiler_efficiency"],
            0.01,
            1.0,
        ),
        "pump_efficiency": _bounded_float(
            params.get("pump_efficiency"),
            getattr(constants, "pump_efficiency", None) or DEFAULT_PHASING_ASSUMPTIONS["pump_efficiency"],
            0.01,
            1.0,
        ),
        "target_pressure_drop_pa_per_m": max(_context_value(
            params,
            constants,
            "target_pressure_drop",
            "target_pressure_drop_pa_per_m",
            DEFAULT_PHASING_ASSUMPTIONS["target_pressure_drop_pa_per_m"],
        ), 1.0),
        "design_delta_t_c": max(_context_value(
            params,
            constants,
            None,
            "design_delta_t_c",
            DEFAULT_PHASING_ASSUMPTIONS["design_delta_t_c"],
        ), 1.0),
        "max_velocity_m_s": max(_context_value(
            params,
            constants,
            "max_velocity",
            "max_velocity_m_s",
            DEFAULT_PHASING_ASSUMPTIONS["max_velocity_m_s"],
        ), 0.01),
        "min_velocity_m_s": max(_context_value(
            params,
            constants,
            "min_velocity",
            "min_velocity_m_s",
            DEFAULT_PHASING_ASSUMPTIONS["min_velocity_m_s"],
        ), 0.0),
        "carbon_price_cad_per_tonne": max(_float_value(carbon_price, 0.0), 0.0),
        "carbon_price_source": carbon_price_source,
        "carbon_price_year": first_period_year,
        "base_source_heat_kw": _float_value(source_heat_kw, 0.0),
        "base_served_heat_kw": _float_value(served_heat_kw, 0.0),
        "base_available_heat_kw": _float_value(base_available_heat_kw, 0.0),
        "inferred_available_heat_kw": _float_value(inferred_available_heat_kw, 0.0),
        "configured_available_heat_kw": _float_value(configured_available_heat_kw, 0.0),
        "available_heat_is_unlimited": params["assume_unlimited_available_heat"],
        "demand_year": params["demand_year"],
        "existing_type_counts": _existing_building_type_counts(network),
        "gas_price_cad_per_mwh": (
            params["gas_price_cad_per_mwh"]
            if params["gas_price_cad_per_mwh"] is not None
            else _float_value(getattr(gas_rate, "cad_per_mwh_useful_heat", None), 0.0)
        ),
        "gas_price_region": getattr(gas_rate, "region", params.get("gas_price_region")),
        "gas_price_source": getattr(gas_rate, "source_name", None),
        "gas_price_source_url": getattr(gas_rate, "source_url", None),
        "gas_price_effective_date": str(gas_rate.effective_date) if gas_rate else None,
        "gas_price_cents_per_m3": _float_value(getattr(gas_rate, "effective_price_cents_per_m3", None), None),
    }


def _gas_rate_basis(params):
    region = params.get("gas_price_region") or "Union South Rate Zone"
    queryset = OntarioNaturalGasRate.objects.all().order_by("-effective_date")
    if region:
        regional = queryset.filter(region=region)
        if regional.exists():
            return regional.first()
    return queryset.first()


def _carbon_tax_basis_for_year(year):
    year = _int_value(year, dt.date.today().year)
    exact = CarbonTaxRate.objects.filter(year=year).first()
    if exact:
        return {"price_cad_per_tonne": exact.price_cad_per_tonne, "source": f"carbon_tax_rate:{exact.year}"}

    prior = CarbonTaxRate.objects.filter(year__lte=year).order_by("-year").first()
    if prior:
        escalation_years = max(year - prior.year, 0)
        return {
            "price_cad_per_tonne": prior.price_cad_per_tonne * ((1.015) ** escalation_years),
            "source": f"carbon_tax_rate:{prior.year}+1.5pct_escalation",
        }

    future = CarbonTaxRate.objects.filter(year__gte=year).order_by("year").first()
    if future:
        return {"price_cad_per_tonne": future.price_cad_per_tonne, "source": f"carbon_tax_rate:{future.year}"}

    return {"price_cad_per_tonne": 0.0, "source": "not_configured"}


def _context_value(params, constants, constant_attr, param_key, default):
    if param_key and params.get(param_key) is not None:
        return _float_value(params.get(param_key), default)
    if constant_attr and constants is not None:
        value = getattr(constants, constant_attr, None)
        if value is not None:
            return _float_value(value, default)
    return _float_value(default, default)


def _bounded_float(raw, default, minimum=None, maximum=None):
    value = _float_value(raw, default)
    if minimum is not None:
        value = max(value, minimum)
    if maximum is not None:
        value = min(value, maximum)
    return value


def _phasing_parameter_payload(params, context):
    return {
        "solver_mode": params["solver_mode"],
        "scenario_decision_mode": params["scenario_decision_mode"],
        "periods": params["periods"],
        "max_distance_m": params["max_distance_m"],
        "min_linear_heat_density_mwh_per_m_year": params["min_linear_heat_density_mwh_per_m_year"],
        "available_heat_kw": context["configured_available_heat_kw"],
        "annual_full_load_hours": context["annual_full_load_hours"],
        "pipe_cost_per_m": context["pipe_cost_per_m"],
        "heat_price_cad_per_mwh": params["heat_price_cad_per_mwh"],
        "gas_price_cad_per_mwh": context["gas_price_cad_per_mwh"],
        "gas_price_region": context.get("gas_price_region"),
        "gas_price_source": context.get("gas_price_source"),
        "gas_price_source_url": context.get("gas_price_source_url"),
        "gas_price_effective_date": context.get("gas_price_effective_date"),
        "gas_price_cents_per_m3": context.get("gas_price_cents_per_m3"),
        "available_heat_is_unlimited": context["available_heat_is_unlimited"],
        "inferred_available_heat_kw": context["inferred_available_heat_kw"],
        "effective_available_heat_kw": context["base_available_heat_kw"],
        "demand_year": params["demand_year"],
        "candidate_source_mode": params["candidate_source_mode"],
        "include_buildings": params["include_buildings"],
        "include_clusters": params["include_clusters"],
        "assume_unlimited_available_heat": params["assume_unlimited_available_heat"],
        "candidate_selection_mode": params["candidate_selection_mode"],
        "complementary_cluster_configuration_id": params["complementary_cluster_configuration_id"],
        "create_networks": params["create_networks"],
        "route_candidate_preview": params["route_candidate_preview"],
        "route_mode": params["route_mode"],
        "weights": params["weights"],
        "rho": context["rho"],
        "cp_j_per_kg_k": context["cp_j_per_kg_k"],
        "mu": context["mu"],
        "project_lifespan_years": context["project_lifespan_years"],
        "electricity_cost_cad_per_kwh": context["electricity_cost_cad_per_kwh"],
        "grid_emission_factor_kgco2e_per_kwh": context["grid_emission_factor_kgco2e_per_kwh"],
        "gas_emission_factor_kgco2e_per_kwh": context["gas_emission_factor_kgco2e_per_kwh"],
        "base_boiler_efficiency": context["base_boiler_efficiency"],
        "pump_efficiency": context["pump_efficiency"],
        "target_pressure_drop_pa_per_m": context["target_pressure_drop_pa_per_m"],
        "design_delta_t_c": context["design_delta_t_c"],
        "max_velocity_m_s": context["max_velocity_m_s"],
        "min_velocity_m_s": context["min_velocity_m_s"],
        "carbon_price_cad_per_tonne": context["carbon_price_cad_per_tonne"],
        "carbon_price_source": context["carbon_price_source"],
        "carbon_price_year": context["carbon_price_year"],
    }


def _context_from_parameter_payload(parameters):
    parameters = dict(parameters or {})
    defaults = DEFAULT_PHASING_ASSUMPTIONS
    return {
        "rho": _float_value(parameters.get("rho"), 971.8),
        "cp_j_per_kg_k": _float_value(parameters.get("cp_j_per_kg_k"), 4190.0),
        "mu": _float_value(parameters.get("mu"), 0.000355),
        "annual_full_load_hours": _float_value(parameters.get("annual_full_load_hours"), 4000.0),
        "pipe_cost_per_m": _float_value(parameters.get("pipe_cost_per_m"), 150.0),
        "project_lifespan_years": _int_value(parameters.get("project_lifespan_years"), defaults["project_lifespan_years"]),
        "electricity_cost_cad_per_kwh": _float_value(parameters.get("electricity_cost_cad_per_kwh"), defaults["electricity_cost_cad_per_kwh"]),
        "grid_emission_factor_kgco2e_per_kwh": _float_value(parameters.get("grid_emission_factor_kgco2e_per_kwh"), defaults["grid_emission_factor_kgco2e_per_kwh"]),
        "gas_emission_factor_kgco2e_per_kwh": _float_value(parameters.get("gas_emission_factor_kgco2e_per_kwh"), defaults["gas_emission_factor_kgco2e_per_kwh"]),
        "base_boiler_efficiency": _bounded_float(parameters.get("base_boiler_efficiency"), defaults["base_boiler_efficiency"], 0.01, 1.0),
        "pump_efficiency": _bounded_float(parameters.get("pump_efficiency"), defaults["pump_efficiency"], 0.01, 1.0),
        "target_pressure_drop_pa_per_m": _float_value(parameters.get("target_pressure_drop_pa_per_m"), defaults["target_pressure_drop_pa_per_m"]),
        "design_delta_t_c": _float_value(parameters.get("design_delta_t_c"), defaults["design_delta_t_c"]),
        "max_velocity_m_s": _float_value(parameters.get("max_velocity_m_s"), defaults["max_velocity_m_s"]),
        "min_velocity_m_s": _float_value(parameters.get("min_velocity_m_s"), defaults["min_velocity_m_s"]),
        "carbon_price_cad_per_tonne": _float_value(parameters.get("carbon_price_cad_per_tonne"), 0.0),
        "gas_price_cad_per_mwh": _float_value(parameters.get("gas_price_cad_per_mwh"), 0.0),
        "heat_price_cad_per_mwh": _float_value(parameters.get("heat_price_cad_per_mwh"), 0.0),
    }


def _candidate_rows(network, params, context):
    """Build the candidate pool for this study.

    Candidates are same-project assets that are not already in the base
    network. The selected UI scope determines whether we include buildings,
    complementary clusters, or both. After rows are built, scoring is attached
    so the UI can explain suitability even before optimization.
    """
    rows = []
    include_buildings = params["include_buildings"] and params["candidate_source_mode"] != "clusters_only"
    include_clusters = params["include_clusters"] and params["candidate_source_mode"] != "buildings_only"

    if include_buildings:
        rows.extend(_building_candidate_rows(network, params, context))
    if include_clusters:
        rows.extend(_cluster_candidate_rows(network, params, context))

    if params["candidate_source_mode"] == "clusters_only":
        rows = [
            row for row in rows
            if row["candidate_type"] == CandidateConnection.CandidateType.CLUSTER
        ]
    elif params["candidate_source_mode"] == "buildings_only":
        rows = [
            row for row in rows
            if row["candidate_type"] == CandidateConnection.CandidateType.BUILDING
        ]

    _attach_scores(rows, params, context)
    rows.sort(key=lambda row: (row["status"] != CandidateConnection.Status.ELIGIBLE, -row["suitability_score"], row["distance_to_trunk_m"], row["name"]))
    limit = params["candidate_limit"]
    return rows[:limit] if limit else rows


def _building_candidate_rows(network, params, context):
    if params["candidate_selection_mode"] == "selected" and not params["candidate_ids"]:
        return []
    existing_building_ids = Node.objects.filter(network=network, building__isnull=False).values_list("building_id", flat=True)
    queryset = Building.objects.filter(project=network.project).exclude(id__in=existing_building_ids)
    queryset = queryset.exclude(latitude__isnull=True).exclude(longitude__isnull=True)
    if params["candidate_ids"]:
        queryset = queryset.filter(id__in=params["candidate_ids"])
    if params["tags"]:
        queryset = queryset.filter(tag__in=params["tags"])

    rows = []
    for building in queryset.order_by("id"):
        nearest = _nearest_reference(building.latitude, building.longitude, context["route_index"])
        if not nearest:
            continue
        role = _candidate_role_for_building(building)
        heat_metrics = _building_heat_metrics(building, params.get("demand_year"), context)
        peak_heat_kw = 0.0 if role == CandidateConnection.CandidateRole.SOURCE else heat_metrics["peak_heat_kw"]
        available_heat_kw = heat_metrics["source_heat_kw"] if role == CandidateConnection.CandidateRole.SOURCE else 0.0
        annual_heat_mwh = heat_metrics["source_annual_mwh"] if role == CandidateConnection.CandidateRole.SOURCE else heat_metrics["annual_heat_mwh"]
        rows.append(_candidate_row(
            key=f"building:{building.id}",
            candidate_type=CandidateConnection.CandidateType.BUILDING,
            name=building.name or f"Building {building.id}",
            role=role,
            latitude=building.latitude,
            longitude=building.longitude,
            peak_heat_kw=peak_heat_kw,
            available_heat_kw=available_heat_kw,
            annual_heat_mwh=annual_heat_mwh,
            building_id=building.id,
            cluster_id=None,
            building_type=building.building_type or "other",
            building_count=1,
            available_from_year=None,
            nearest=nearest,
            params=params,
            context=context,
        ))
    return rows


def _cluster_candidate_rows(network, params, context):
    if params["candidate_selection_mode"] == "selected" and not params["cluster_ids"]:
        return []
    config_id = params.get("complementary_cluster_configuration_id")
    queryset = Cluster.objects.filter(
        cluster_configuration__cluster_type=ClusterConfiguration.ClusterConfigurationType.COMPLEMENTARY,
        cluster_configuration__base_network=network,
        cluster_configuration__cluster_algorithm__project=network.project,
    )
    if config_id:
        config = ClusterConfiguration.objects.select_related("cluster_algorithm", "base_network").filter(pk=config_id).first()
        if not config:
            raise ValueError("The selected complementary cluster configuration was not found.")
        if config.cluster_type != ClusterConfiguration.ClusterConfigurationType.COMPLEMENTARY:
            raise ValueError("Phasing can only use complementary cluster configurations.")
        if config.base_network_id != network.id:
            raise ValueError("The complementary cluster configuration must belong to the selected base network.")
        if config.cluster_algorithm.project_id != network.project_id:
            raise ValueError("The complementary cluster configuration belongs to a different project.")
        queryset = queryset.filter(cluster_configuration_id=config_id)
    elif not params["cluster_ids"]:
        return []

    queryset = queryset.exclude(latitude__isnull=True).exclude(longitude__isnull=True)
    if params["cluster_ids"]:
        queryset = queryset.filter(id__in=params["cluster_ids"])

    rows = []
    for cluster in queryset.order_by("id"):
        nearest = _nearest_reference(cluster.latitude, cluster.longitude, context["route_index"])
        if not nearest:
            continue
        heat_metrics = _cluster_heat_metrics(cluster, params.get("demand_year"), context)
        peak_heat_kw = heat_metrics["peak_heat_kw"]
        source_heat_kw = heat_metrics["source_heat_kw"]
        available_heat_kw = max(source_heat_kw - peak_heat_kw, 0.0)
        role = CandidateConnection.CandidateRole.SOURCE if available_heat_kw > peak_heat_kw else CandidateConnection.CandidateRole.CONSUMER
        annual_heat_mwh = heat_metrics["source_annual_mwh"] if role == CandidateConnection.CandidateRole.SOURCE else heat_metrics["annual_heat_mwh"]
        building_count = cluster.no_of_buildings or BuildingCluster.objects.filter(cluster=cluster).count()
        rows.append(_candidate_row(
            key=f"cluster:{cluster.id}",
            candidate_type=CandidateConnection.CandidateType.CLUSTER,
            name=cluster.name or f"Cluster {cluster.id}",
            role=role,
            latitude=cluster.latitude,
            longitude=cluster.longitude,
            peak_heat_kw=0.0 if role == CandidateConnection.CandidateRole.SOURCE else peak_heat_kw,
            available_heat_kw=available_heat_kw if role == CandidateConnection.CandidateRole.SOURCE else 0.0,
            annual_heat_mwh=annual_heat_mwh,
            building_id=None,
            cluster_id=cluster.id,
            building_type="cluster",
            building_count=building_count,
            available_from_year=None,
            nearest=nearest,
            params=params,
            context=context,
        ))
    return rows


def _candidate_row(**kwargs):
    params = kwargs["params"]
    context = kwargs["context"]
    nearest = kwargs["nearest"]
    route_data = _candidate_route_data(kwargs, nearest, params, context)
    service_path = _route_data_path(route_data)
    length_m = max(_float_value(route_data.get("distance"), nearest.get("distance_to_trunk_m")), 1.0)
    annual_mwh = kwargs.get("annual_heat_mwh")
    if annual_mwh is None:
        annual_mwh = kwargs["peak_heat_kw"] * context["annual_full_load_hours"] / 1000.0
        if kwargs["role"] == CandidateConnection.CandidateRole.SOURCE:
            annual_mwh = kwargs["available_heat_kw"] * context["annual_full_load_hours"] / 1000.0
    lhd = annual_mwh / length_m if length_m else 0.0
    cost = length_m * context["pipe_cost_per_m"]
    reference_hydraulics = _candidate_hydraulic_economics(
        kwargs["peak_heat_kw"],
        annual_mwh,
        length_m,
        context,
    )
    status = CandidateConnection.Status.ELIGIBLE
    reason = ""
    if kwargs["role"] == CandidateConnection.CandidateRole.CONSUMER and kwargs["peak_heat_kw"] <= 0:
        status = CandidateConnection.Status.REJECTED
        reason = "No heat demand is available for this candidate"
    elif kwargs["role"] == CandidateConnection.CandidateRole.SOURCE and kwargs["available_heat_kw"] <= 0:
        status = CandidateConnection.Status.REJECTED
        reason = "No recoverable heat source capacity is available for this candidate"
    elif nearest["distance_to_trunk_m"] > params["max_distance_m"]:
        status = CandidateConnection.Status.REJECTED
        reason = f"Farther than {params['max_distance_m']:.0f} m from the main trunk"
    elif lhd < params["min_linear_heat_density_mwh_per_m_year"] and kwargs["role"] != CandidateConnection.CandidateRole.SOURCE:
        status = CandidateConnection.Status.REJECTED
        reason = f"Linear heat density below {params['min_linear_heat_density_mwh_per_m_year']:.2f} MWh/m/year"

    return {
        "key": kwargs["key"],
        "candidate_type": kwargs["candidate_type"],
        "building_id": kwargs["building_id"],
        "cluster_id": kwargs["cluster_id"],
        "name": kwargs["name"],
        "candidate_role": kwargs["role"],
        "building_type": kwargs["building_type"],
        "building_count": kwargs.get("building_count") or 0,
        "available_from_year": kwargs["available_from_year"],
        "peak_heat_kw": kwargs["peak_heat_kw"],
        "annual_heat_mwh": annual_mwh,
        "available_heat_kw": kwargs["available_heat_kw"],
        "distance_to_trunk_m": length_m,
        "direct_distance_to_trunk_m": nearest["distance_to_trunk_m"],
        "connection_length_m": length_m,
        "estimated_connection_cost_cad": cost,
        "linear_heat_density_mwh_per_m_year": lhd,
        "snap_point": nearest["snap_point"],
        "source_route_id": nearest.get("source_route_id"),
        "source_edge_id": nearest.get("source_edge_id"),
        "service_path": service_path,
        "route_travel_mode": route_data.get("mode") or _candidate_route_mode(nearest, params, context),
        "service_path_is_routed": bool(route_data.get("routed")) or (
            len(service_path) > 2 and route_data.get("mode") != "direct_fallback"
        ),
        "reference_hydraulic_economics": reference_hydraulics,
        "mass_flow_kg_s": reference_hydraulics["mass_flow_kg_s"],
        "estimated_diameter_m": reference_hydraulics["diameter_m"],
        "pressure_drop_pa": reference_hydraulics["pressure_drop_pa"],
        "pump_power_kw": reference_hydraulics["pump_power_kw"],
        "annual_pumping_cost_cad": reference_hydraulics["annual_pumping_cost_cad"],
        "annual_pumping_emissions_tco2e": reference_hydraulics["annual_pumping_emissions_tco2e"],
        "status": status,
        "rejection_reason": reason,
    }


def _candidate_hydraulic_economics(peak_heat_kw, annual_heat_mwh, length_m, context):
    peak_heat_kw = max(_float_value(peak_heat_kw, 0.0), 0.0)
    annual_heat_mwh = max(_float_value(annual_heat_mwh, 0.0), 0.0)
    length_m = max(_float_value(length_m, 0.0), 0.0)
    if peak_heat_kw <= 0 or annual_heat_mwh <= 0 or length_m <= 0:
        return {
            "mass_flow_kg_s": 0.0,
            "volume_flow_m3_s": 0.0,
            "diameter_m": 0.0,
            "design_velocity_m_s": 0.0,
            "pressure_drop_pa": 0.0,
            "pressure_drop_pa_per_m": context["target_pressure_drop_pa_per_m"],
            "pump_power_kw": 0.0,
            "equivalent_pump_hours": 0.0,
            "annual_pumping_kwh": 0.0,
            "annual_pumping_cost_cad": 0.0,
            "annual_pumping_emissions_tco2e": 0.0,
        }

    delta_t = max(context["design_delta_t_c"], 1.0)
    cp = max(context["cp_j_per_kg_k"], 1.0)
    rho = max(context["rho"], 1.0)
    min_velocity = max(context["min_velocity_m_s"], 0.0)
    max_velocity = max(context["max_velocity_m_s"], min_velocity, 0.01)
    design_velocity = min(max(1.5, min_velocity), max_velocity)

    mass_flow_kg_s = (peak_heat_kw * 1000.0) / (cp * delta_t)
    volume_flow_m3_s = mass_flow_kg_s / rho
    flow_area_m2 = volume_flow_m3_s / design_velocity if design_velocity > 0 else 0.0
    diameter_m = math.sqrt((4.0 * flow_area_m2) / math.pi) if flow_area_m2 > 0 else 0.0
    pressure_drop_pa = context["target_pressure_drop_pa_per_m"] * length_m
    pump_power_kw = (
        (volume_flow_m3_s * pressure_drop_pa) / (context["pump_efficiency"] * 1000.0)
        if context["pump_efficiency"] > 0 else 0.0
    )
    equivalent_hours = (annual_heat_mwh * 1000.0 / peak_heat_kw) if peak_heat_kw > 0 else context["annual_full_load_hours"]
    equivalent_hours = max(0.0, min(equivalent_hours, 8760.0))
    annual_pumping_kwh = pump_power_kw * equivalent_hours
    annual_pumping_cost_cad = annual_pumping_kwh * context["electricity_cost_cad_per_kwh"]
    annual_pumping_emissions_tco2e = (
        annual_pumping_kwh * context["grid_emission_factor_kgco2e_per_kwh"] / 1000.0
    )
    return {
        "mass_flow_kg_s": mass_flow_kg_s,
        "volume_flow_m3_s": volume_flow_m3_s,
        "diameter_m": diameter_m,
        "design_velocity_m_s": design_velocity,
        "pressure_drop_pa": pressure_drop_pa,
        "pressure_drop_pa_per_m": context["target_pressure_drop_pa_per_m"],
        "pump_power_kw": pump_power_kw,
        "equivalent_pump_hours": equivalent_hours,
        "annual_pumping_kwh": annual_pumping_kwh,
        "annual_pumping_cost_cad": annual_pumping_cost_cad,
        "annual_pumping_emissions_tco2e": annual_pumping_emissions_tco2e,
    }


def _candidate_route_data(candidate_kwargs, nearest, params, context):
    snap = nearest.get("snap_point") or {}
    start = (snap.get("lat"), snap.get("lng"))
    end = (candidate_kwargs.get("latitude"), candidate_kwargs.get("longitude"))
    if any(value is None for value in [*start, *end]):
        return {"distance": nearest.get("distance_to_trunk_m") or 0.0, "polyline": ""}
    mode = _candidate_route_mode(nearest, params, context)
    try:
        return build_route_data(
            (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1])),
            mode,
            use_google=params.get("route_candidate_preview", False),
        )
    except Exception:
        return build_route_data(
            (float(start[0]), float(start[1])),
            (float(end[0]), float(end[1])),
            mode,
            use_google=False,
        )


def _candidate_route_mode(nearest, params, context):
    source_route_id = nearest.get("source_route_id")
    if source_route_id:
        for route in context.get("routes", []):
            if route.id == source_route_id:
                return _route_travel_mode(route) or params["route_mode"]
    return params["route_mode"]


def _route_data_path(route_data):
    points = []
    try:
        points = decode_polyline(route_data.get("polyline"))
    except Exception:
        points = []
    return [{"lat": float(lat), "lng": float(lng)} for lat, lng in points]


def _attach_scores(rows, params, context):
    if not rows:
        return
    heat_scores = _normalized(rows, lambda row: row["peak_heat_kw"] or row["available_heat_kw"], higher=True, log_scale=True)
    density_scores = _normalized(rows, lambda row: row["linear_heat_density_mwh_per_m_year"], higher=True, log_scale=True)
    cost_scores = _normalized(rows, lambda row: row["estimated_connection_cost_cad"], higher=False, log_scale=True)
    distance_scores = _normalized(rows, lambda row: row["distance_to_trunk_m"], higher=False, log_scale=True)
    diversity_scores = [_diversity_score(row, context) for row in rows]
    weights = params["weights"]

    for index, row in enumerate(rows):
        available_fit = 1.0
        if row["candidate_role"] == CandidateConnection.CandidateRole.CONSUMER and not params["assume_unlimited_available_heat"]:
            demand = max(row["peak_heat_kw"], 1.0)
            available_fit = min(1.0, context["base_available_heat_kw"] / demand) if context["base_available_heat_kw"] > 0 else 0.35
        components = {
            "linear_heat_density": density_scores[index],
            "heat_demand": heat_scores[index],
            "connection_cost": cost_scores[index],
            "distance_to_trunk": distance_scores[index],
            "available_heat_fit": available_fit,
            "diversity": diversity_scores[index],
        }
        score = sum(components[key] * weights[key] for key in weights)
        if row["status"] == CandidateConnection.Status.REJECTED:
            score *= 0.35
        row["suitability_score"] = round(score * 100.0, 1)
        row["score_breakdown"] = [
            {"key": key, "label": key.replace("_", " ").title(), "score": round(value * 100.0, 1), "weight": round(weights[key], 3)}
            for key, value in components.items()
        ]


def _solve_phasing_plan(candidate_rows, params, context):
    """Choose the scenario treatment first, then the optimization algorithm."""
    if params.get("scenario_decision_mode") == SCENARIO_SPECIFIC_DECISION_MODE:
        return _solve_scenario_specific_plan(candidate_rows, params, context)
    if params.get("solver_mode") == FULL_CHRONOLOGY_SOLVER_MODE:
        return _solve_full_chronology_milp_plan(candidate_rows, params, context)
    if params.get("solver_mode") == "milp":
        return _solve_milp_plan(candidate_rows, params, context)
    return _solve_expected_plan(candidate_rows, params, context)


def _solve_scenario_specific_plan(candidate_rows, params, context):
    """Optimize a separate build-out plan for each scenario.

    Expected-value mode optimizes one shared plan across all scenarios. That is
    useful when the goal is one robust planning schedule. Scenario-specific mode
    asks a different question: "If this scenario happens, what should the phase
    schedule look like?" To answer that, this wrapper runs the chosen base
    solver once per scenario and combines the scenario-specific result rows.
    """
    combined_results = []
    union_selected_by_key = {}
    union_selected_by_year = defaultdict(list)
    union_year_keys = defaultdict(set)
    selected_by_scenario = {}
    scenario_metadata = {}

    for scenario in params["scenarios"]:
        # Reuse the normal greedy/MILP solver with a one-scenario parameter
        # set. This makes that scenario's demand multiplier, fuel multiplier,
        # and discount rate directly affect feasibility and objective value.
        scenario_params = {**params, "scenarios": [scenario], "scenario_decision_mode": EXPECTED_VALUE_DECISION_MODE}
        scenario_plan = _solve_single_mode_plan(candidate_rows, scenario_params, context)
        scenario_name = scenario["name"]
        selected_by_scenario[scenario_name] = {
            "selected_by_key": scenario_plan.get("selected_by_key", {}),
            "selected_by_year": scenario_plan.get("selected_by_year", {}),
        }
        scenario_metadata[scenario_name] = scenario_plan.get("solver_metadata", {})
        combined_results.extend(scenario_plan.get("results", []))

        # These union maps are not the scenario answer. They are compatibility
        # summaries for candidate tables and preview payloads that expect a
        # single selected_by_key/selected_by_year structure.
        for key, year in (scenario_plan.get("selected_by_key") or {}).items():
            previous_year = union_selected_by_key.get(key)
            union_selected_by_key[key] = min(previous_year, year) if previous_year is not None else year
        for year, rows in (scenario_plan.get("selected_by_year") or {}).items():
            for row in rows:
                if row["key"] in union_year_keys[year]:
                    continue
                union_year_keys[year].add(row["key"])
                union_selected_by_year[year].append(row)

    base_solver = _solver_name_for_mode(params.get("solver_mode"))
    return {
        "selected_by_key": union_selected_by_key,
        "selected_by_year": dict(union_selected_by_year),
        "selected_by_scenario": selected_by_scenario,
        "results": combined_results,
        "solver": f"scenario_specific_{base_solver}",
        "solver_metadata": {
            "mode": "scenario_specific",
            "base_solver": base_solver,
            "scenario_count": len(params["scenarios"]),
            "scenarios": scenario_metadata,
            "description": "Each scenario is optimized independently, so candidate selections and phase timing can differ by scenario.",
        },
    }


def _solve_single_mode_plan(candidate_rows, params, context):
    """Run one solver mode without applying the scenario wrapper again."""
    solver_mode = params.get("solver_mode")
    if solver_mode == FULL_CHRONOLOGY_SOLVER_MODE:
        return _solve_full_chronology_milp_plan(candidate_rows, params, context)
    if solver_mode == "milp":
        return _solve_milp_plan(candidate_rows, params, context)
    return _solve_expected_plan(candidate_rows, params, context)


def _solver_name_for_mode(solver_mode):
    if solver_mode == FULL_CHRONOLOGY_SOLVER_MODE:
        return FULL_CHRONOLOGY_SOLVER_NAME
    if solver_mode == "milp":
        return MILP_SOLVER_NAME
    return GREEDY_SOLVER_NAME


def _solve_expected_plan(candidate_rows, params, context):
    """Greedy screening baseline using expected candidate value.

    The greedy engine is intentionally transparent:
    - connect eligible sources first because they increase available heat;
    - rank consumers by expected net value and suitability;
    - accept a candidate only when it fits the current phase gates.

    It is good for fast screening and explanation. It is not guaranteed to be
    globally optimal when candidates compete for scarce phase budget or heat.
    """
    periods = params["periods"]
    scenarios = params["scenarios"]
    eligible = [row for row in candidate_rows if row["status"] == CandidateConnection.Status.ELIGIBLE]
    selected_by_key = {}
    selected_by_year = defaultdict(list)
    selected_source_capacity_kw = 0.0
    selected_consumer_peak_kw = 0.0

    for period in periods:
        year = period["year"]
        capex_budget = period.get("capex_budget_cad")
        connection_cap = period.get("max_new_connections")
        period_capex = 0.0
        period_connection_count = 0
        # Start each phase with the configured heat for that period, then add
        # sources and subtract consumers already selected in earlier phases.
        if params["assume_unlimited_available_heat"]:
            available_heat_kw = UNLIMITED_AVAILABLE_HEAT_KW
        else:
            available_heat_kw = (
                _period_available_heat_kw(period, context, params)
                + selected_source_capacity_kw
                - selected_consumer_peak_kw
            )

        source_options = [
            row for row in eligible
            if row["key"] not in selected_by_key
            and row["candidate_role"] == CandidateConnection.CandidateRole.SOURCE
            and _candidate_available(row, year)
        ]
        for row in sorted(source_options, key=lambda item: (-item["suitability_score"], item["estimated_connection_cost_cad"])):
            # max_new_connections acts like an annual construction capacity.
            if connection_cap is not None and period_connection_count >= connection_cap:
                break
            if capex_budget is not None and period_capex + row["estimated_connection_cost_cad"] > capex_budget:
                continue
            selected_by_key[row["key"]] = year
            selected_by_year[year].append(row)
            selected_source_capacity_kw += row["available_heat_kw"]
            available_heat_kw += row["available_heat_kw"]
            period_capex += row["estimated_connection_cost_cad"]
            period_connection_count += 1

        consumer_options = [
            row for row in eligible
            if row["key"] not in selected_by_key
            and row["candidate_role"] == CandidateConnection.CandidateRole.CONSUMER
            and _candidate_available(row, year)
        ]
        consumer_options.sort(key=lambda row: (-_expected_candidate_value(row, year, scenarios, params, context), -row["suitability_score"]))
        for row in consumer_options:
            if connection_cap is not None and period_connection_count >= connection_cap:
                break
            expected_peak = row["peak_heat_kw"] * _expected_multiplier(scenarios)
            # In expected-value mode this is probability-weighted demand. In
            # scenario-specific mode this function is called with one scenario,
            # so it becomes that scenario's demand.
            if not params["assume_unlimited_available_heat"] and expected_peak > available_heat_kw:
                continue
            if capex_budget is not None and period_capex + row["estimated_connection_cost_cad"] > capex_budget:
                continue
            selected_by_key[row["key"]] = year
            selected_by_year[year].append(row)
            if not params["assume_unlimited_available_heat"]:
                selected_consumer_peak_kw += expected_peak
                available_heat_kw -= expected_peak
            period_capex += row["estimated_connection_cost_cad"]
            period_connection_count += 1

    return _finalize_plan(
        candidate_rows,
        selected_by_key,
        selected_by_year,
        params,
        context,
        GREEDY_SOLVER_NAME,
        {"mode": "greedy", "description": "Ranked screening baseline using expected candidate value."},
    )


def _solve_milp_plan(candidate_rows, params, context):
    """Mixed-integer optimization for candidate-year selection.

    Decision variable:
        x[i,t] = 1 if candidate i is newly connected in planning period t.

    Objective:
        maximize present value of scenario-weighted benefit minus connection
        CAPEX. scipy.optimize.milp minimizes, so objective coefficients are
        stored as negative values.

    Constraints:
    - a candidate can be newly connected at most once;
    - candidates cannot connect before available_from_year;
    - phase CAPEX cannot exceed capex_budget_cad when provided;
    - phase count cannot exceed max_new_connections when provided;
    - cumulative demand cannot exceed available heat plus selected sources.
    """
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except Exception as exc:
        raise ValueError("MILP optimizer requires scipy.optimize.milp to be available in the backend environment.") from exc

    periods = params["periods"]
    scenarios = params["scenarios"]
    eligible = [row for row in candidate_rows if row["status"] == CandidateConnection.Status.ELIGIBLE]
    if not eligible or not periods:
        return _finalize_plan(
            candidate_rows,
            {},
            defaultdict(list),
            params,
            context,
            MILP_SOLVER_NAME,
            {"mode": "milp", "description": "No eligible candidate-period decisions were available."},
        )

    period_count = len(periods)
    variable_count = len(eligible) * period_count

    def variable_index(candidate_index, period_index):
        return candidate_index * period_count + period_index

    objective = np.zeros(variable_count, dtype=float)
    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)

    for candidate_index, row in enumerate(eligible):
        for period_index, period in enumerate(periods):
            index = variable_index(candidate_index, period_index)
            if not _candidate_available(row, period["year"]):
                upper_bounds[index] = 0.0
            # Negative sign converts maximize value into minimize cost.
            objective[index] = -_expected_candidate_value(row, period["year"], scenarios, params, context)

    constraints = []
    # Candidate uniqueness: sum_t x[i,t] <= 1.
    for candidate_index, _row in enumerate(eligible):
        coefficients = np.zeros(variable_count, dtype=float)
        for period_index in range(period_count):
            coefficients[variable_index(candidate_index, period_index)] = 1.0
        constraints.append(LinearConstraint(coefficients, -np.inf, 1.0))

    for period_index, period in enumerate(periods):
        capex_budget = period.get("capex_budget_cad")
        if capex_budget is None:
            pass
        else:
            # Phase budget: sum_i capex_i * x[i,t] <= budget_t.
            coefficients = np.zeros(variable_count, dtype=float)
            for candidate_index, row in enumerate(eligible):
                coefficients[variable_index(candidate_index, period_index)] = max(
                    _float_value(row.get("estimated_connection_cost_cad"), 0.0),
                    0.0,
                )
            constraints.append(LinearConstraint(coefficients, -np.inf, max(_float_value(capex_budget, 0.0), 0.0)))

        connection_cap = period.get("max_new_connections")
        if connection_cap is not None:
            # Phase construction capacity: sum_i x[i,t] <= cap_t.
            coefficients = np.zeros(variable_count, dtype=float)
            for candidate_index, _row in enumerate(eligible):
                coefficients[variable_index(candidate_index, period_index)] = 1.0
            constraints.append(LinearConstraint(coefficients, -np.inf, max(_int_value(connection_cap, 0), 0)))

    if not params["assume_unlimited_available_heat"]:
        expected_demand_multiplier = _expected_multiplier(scenarios)
        for period_index, period in enumerate(periods):
            # Cumulative heat balance through this phase:
            # base_heat_t + selected_sources - selected_consumers >= 0.
            coefficients = np.zeros(variable_count, dtype=float)
            for candidate_index, row in enumerate(eligible):
                for previous_period_index in range(period_index + 1):
                    index = variable_index(candidate_index, previous_period_index)
                    if row["candidate_role"] == CandidateConnection.CandidateRole.SOURCE:
                        coefficients[index] += max(_float_value(row.get("available_heat_kw"), 0.0), 0.0)
                    else:
                        coefficients[index] -= max(_float_value(row.get("peak_heat_kw"), 0.0), 0.0) * expected_demand_multiplier
            base_available_heat_kw = _period_available_heat_kw(period, context, params)
            constraints.append(LinearConstraint(coefficients, -base_available_heat_kw, np.inf))

    options = {
        "time_limit": max(_float_value(params.get("milp_time_limit_seconds"), 60.0), 1.0),
        "mip_rel_gap": max(_float_value(params.get("milp_relative_gap"), 0.01), 0.0),
    }
    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=int),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options=options,
    )
    if result.x is None:
        raise ValueError(f"MILP optimizer could not find a feasible phasing plan: {result.message}")

    selected_by_key = {}
    selected_by_year = defaultdict(list)
    for candidate_index, row in enumerate(eligible):
        for period_index, period in enumerate(periods):
            if result.x[variable_index(candidate_index, period_index)] > 0.5:
                selected_by_key[row["key"]] = period["year"]
                selected_by_year[period["year"]].append(row)
                break

    objective_value = -float(result.fun) if result.fun is not None else 0.0
    metadata = {
        "mode": "milp",
        "solver": "scipy.optimize.milp_highs",
        "status": int(getattr(result, "status", -1)),
        "message": str(getattr(result, "message", "")),
        "objective_value_cad": objective_value,
        "relative_gap": _float_value(getattr(result, "mip_gap", None), None),
        "variable_count": variable_count,
        "constraint_count": len(constraints),
    }
    return _finalize_plan(
        candidate_rows,
        selected_by_key,
        selected_by_year,
        params,
        context,
        MILP_SOLVER_NAME,
        metadata,
    )


def _solve_full_chronology_milp_plan(candidate_rows, params, context):
    """Paper-style chronology MILP with explicit state and build variables.

    The simpler MILP decides only x[i,t] = "candidate i is newly connected in
    period t". This full mode expands that compact decision into the variables
    used in the phasing paper:

    - N[i,t,m]: candidate node exists / is connected by period t in scenario m
    - Nbar[i,t,m]: candidate node is newly connected in period t
    - E[i,t,m]: candidate service edge/path exists by period t
    - Ebar[i,t,m]: candidate service edge/path is newly built in period t
    - P[i,t,m]: source/plant candidate is active by period t

    In this app a "candidate edge" is the routed service connection from the
    existing network corridor to the building/cluster candidate. We do not yet
    optimize a separate city-wide candidate-edge graph between future
    candidates; that would be a larger topology expansion model.
    """
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except Exception as exc:
        raise ValueError("Full chronology MILP requires scipy.optimize.milp to be available in the backend environment.") from exc

    periods = params["periods"]
    scenarios = params["scenarios"]
    eligible = [row for row in candidate_rows if row["status"] == CandidateConnection.Status.ELIGIBLE]
    if not eligible or not periods:
        return _finalize_plan(
            candidate_rows,
            {},
            defaultdict(list),
            params,
            context,
            FULL_CHRONOLOGY_SOLVER_NAME,
            {"mode": FULL_CHRONOLOGY_SOLVER_MODE, "description": "No eligible chronology decisions were available."},
        )

    candidate_count = len(eligible)
    period_count = len(periods)
    scenario_count = len(scenarios)
    state_size = candidate_count * period_count * scenario_count
    blocks = {
        "node": 0,
        "node_build": state_size,
        "edge": state_size * 2,
        "edge_build": state_size * 3,
        "plant": state_size * 4,
    }
    variable_count = state_size * len(blocks)

    def variable_index(block, candidate_index, period_index, scenario_index):
        return blocks[block] + ((candidate_index * period_count + period_index) * scenario_count + scenario_index)

    objective = np.zeros(variable_count, dtype=float)
    lower_bounds = np.zeros(variable_count, dtype=float)
    upper_bounds = np.ones(variable_count, dtype=float)
    integrality = np.ones(variable_count, dtype=int)

    for candidate_index, row in enumerate(eligible):
        is_source = row["candidate_role"] == CandidateConnection.CandidateRole.SOURCE
        capex = max(_float_value(row.get("estimated_connection_cost_cad"), 0.0), 0.0)
        for period_index, period in enumerate(periods):
            available = _candidate_available(row, period["year"])
            for scenario_index, scenario in enumerate(scenarios):
                if not available:
                    for block in blocks:
                        upper_bounds[variable_index(block, candidate_index, period_index, scenario_index)] = 0.0
                    continue
                if not is_source:
                    upper_bounds[variable_index("plant", candidate_index, period_index, scenario_index)] = 0.0

                years_from_start = max(period["year"] - periods[0]["year"], 0)
                annual_value = _annual_candidate_value_cad(row, scenario, params, context)
                present_value = _present_value(
                    annual_value,
                    scenario["discount_rate"],
                    context["project_lifespan_years"],
                    years_from_start,
                )
                probability = max(_float_value(scenario.get("probability"), 1.0), 0.0)
                # Minimize negative expected benefit plus expected build cost.
                # Because node_build and edge_build are constrained together,
                # this is equivalent to maximizing expected net present value.
                objective[variable_index("node_build", candidate_index, period_index, scenario_index)] -= probability * present_value
                objective[variable_index("edge_build", candidate_index, period_index, scenario_index)] += probability * capex

    constraint_rows = []
    constraint_cols = []
    constraint_values = []
    lower_limits = []
    upper_limits = []

    def add_constraint(entries, lower, upper):
        row_index = len(lower_limits)
        for column_index, value in entries:
            if value:
                constraint_rows.append(row_index)
                constraint_cols.append(column_index)
                constraint_values.append(float(value))
        lower_limits.append(lower)
        upper_limits.append(upper)

    for candidate_index, row in enumerate(eligible):
        is_source = row["candidate_role"] == CandidateConnection.CandidateRole.SOURCE
        for scenario_index, _scenario in enumerate(scenarios):
            node_build_once_entries = []
            edge_build_once_entries = []
            for period_index, _period in enumerate(periods):
                node_var = variable_index("node", candidate_index, period_index, scenario_index)
                node_build_var = variable_index("node_build", candidate_index, period_index, scenario_index)
                edge_var = variable_index("edge", candidate_index, period_index, scenario_index)
                edge_build_var = variable_index("edge_build", candidate_index, period_index, scenario_index)
                plant_var = variable_index("plant", candidate_index, period_index, scenario_index)

                node_build_once_entries.append((node_build_var, 1.0))
                edge_build_once_entries.append((edge_build_var, 1.0))

                if period_index == 0:
                    add_constraint([(node_var, 1.0), (node_build_var, -1.0)], 0.0, 0.0)
                    add_constraint([(edge_var, 1.0), (edge_build_var, -1.0)], 0.0, 0.0)
                else:
                    previous_node_var = variable_index("node", candidate_index, period_index - 1, scenario_index)
                    previous_edge_var = variable_index("edge", candidate_index, period_index - 1, scenario_index)
                    add_constraint([(node_var, 1.0), (previous_node_var, -1.0), (node_build_var, -1.0)], 0.0, 0.0)
                    add_constraint([(edge_var, 1.0), (previous_edge_var, -1.0), (edge_build_var, -1.0)], 0.0, 0.0)

                # A future node cannot be counted as connected unless its
                # service route exists, and the service route has no meaning
                # unless the candidate node exists. This represents the
                # endpoint/path precedence part of the chronology formulation.
                add_constraint([(node_var, 1.0), (edge_var, -1.0)], 0.0, 0.0)
                add_constraint([(node_build_var, 1.0), (edge_build_var, -1.0)], 0.0, 0.0)

                if is_source:
                    add_constraint([(plant_var, 1.0), (node_var, -1.0)], 0.0, 0.0)

            add_constraint(node_build_once_entries, -np.inf, 1.0)
            add_constraint(edge_build_once_entries, -np.inf, 1.0)

    # Expected-value mode is a shared investment schedule. The scenario axis is
    # still present, but all build/existence variables are forced to agree
    # across scenarios. Scenario-specific mode calls this solver one scenario
    # at a time, so these equalities naturally disappear.
    if params.get("scenario_decision_mode") != SCENARIO_SPECIFIC_DECISION_MODE and scenario_count > 1:
        for candidate_index, _row in enumerate(eligible):
            for period_index, _period in enumerate(periods):
                for scenario_index in range(1, scenario_count):
                    for block in blocks:
                        add_constraint([
                            (variable_index(block, candidate_index, period_index, scenario_index), 1.0),
                            (variable_index(block, candidate_index, period_index, 0), -1.0),
                        ], 0.0, 0.0)

    for period_index, period in enumerate(periods):
        capex_budget = period.get("capex_budget_cad")
        connection_cap = period.get("max_new_connections")
        for scenario_index, scenario in enumerate(scenarios):
            if capex_budget is not None:
                entries = [
                    (
                        variable_index("edge_build", candidate_index, period_index, scenario_index),
                        max(_float_value(row.get("estimated_connection_cost_cad"), 0.0), 0.0),
                    )
                    for candidate_index, row in enumerate(eligible)
                ]
                add_constraint(entries, -np.inf, max(_float_value(capex_budget, 0.0), 0.0))

            if connection_cap is not None:
                entries = [
                    (variable_index("node_build", candidate_index, period_index, scenario_index), 1.0)
                    for candidate_index, _row in enumerate(eligible)
                ]
                add_constraint(entries, -np.inf, max(_int_value(connection_cap, 0), 0))

            if not params["assume_unlimited_available_heat"]:
                demand_multiplier = _float_value(scenario.get("demand_multiplier"), 1.0)
                entries = []
                for candidate_index, row in enumerate(eligible):
                    if row["candidate_role"] == CandidateConnection.CandidateRole.SOURCE:
                        entries.append((
                            variable_index("plant", candidate_index, period_index, scenario_index),
                            max(_float_value(row.get("available_heat_kw"), 0.0), 0.0),
                        ))
                    else:
                        entries.append((
                            variable_index("node", candidate_index, period_index, scenario_index),
                            -max(_float_value(row.get("peak_heat_kw"), 0.0), 0.0) * demand_multiplier,
                        ))
                base_available_heat_kw = _period_available_heat_kw(period, context, params)
                add_constraint(entries, -base_available_heat_kw, np.inf)

    constraint_matrix = coo_matrix(
        (constraint_values, (constraint_rows, constraint_cols)),
        shape=(len(lower_limits), variable_count),
    ).tocsr()
    constraints = LinearConstraint(
        constraint_matrix,
        np.array(lower_limits, dtype=float),
        np.array(upper_limits, dtype=float),
    )

    options = {
        "time_limit": max(_float_value(params.get("milp_time_limit_seconds"), 90.0), 1.0),
        "mip_rel_gap": max(_float_value(params.get("milp_relative_gap"), 0.01), 0.0),
    }
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=constraints,
        options=options,
    )
    if result.x is None:
        raise ValueError(f"Full chronology MILP could not find a feasible phasing plan: {result.message}")

    decisions = {}
    for candidate_index, row in enumerate(eligible):
        for period_index, period in enumerate(periods):
            for scenario_index, scenario in enumerate(scenarios):
                key = (row["key"], period["year"], scenario["name"])
                decisions[key] = {
                    "node_connected": result.x[variable_index("node", candidate_index, period_index, scenario_index)] > 0.5,
                    "node_newly_connected": result.x[variable_index("node_build", candidate_index, period_index, scenario_index)] > 0.5,
                    "edge_exists": result.x[variable_index("edge", candidate_index, period_index, scenario_index)] > 0.5,
                    "edge_newly_built": result.x[variable_index("edge_build", candidate_index, period_index, scenario_index)] > 0.5,
                    "plant_exists": result.x[variable_index("plant", candidate_index, period_index, scenario_index)] > 0.5,
                }

    objective_value = -float(result.fun) if result.fun is not None else 0.0
    metadata = {
        "mode": FULL_CHRONOLOGY_SOLVER_MODE,
        "solver": "scipy.optimize.milp_highs",
        "status": int(getattr(result, "status", -1)),
        "message": str(getattr(result, "message", "")),
        "objective_value_cad": objective_value,
        "relative_gap": _float_value(getattr(result, "mip_gap", None), None),
        "candidate_count": candidate_count,
        "period_count": period_count,
        "scenario_count": scenario_count,
        "variable_count": variable_count,
        "constraint_count": len(lower_limits),
        "decision_variables": {
            "N_i_t_m": "candidate node connected by period t in scenario m",
            "Nbar_i_t_m": "candidate node newly connected in period t",
            "E_i_t_m": "candidate service route exists by period t in scenario m",
            "Ebar_i_t_m": "candidate service route newly built in period t",
            "P_i_t_m": "source candidate active by period t in scenario m",
        },
        "constraint_families": [
            "N_i_t_m = N_i,t-1,m + Nbar_i_t_m",
            "E_i_t_m = E_i,t-1,m + Ebar_i_t_m",
            "N_i_t_m = E_i_t_m for candidate service connections",
            "P_i_t_m = N_i_t_m for source candidates",
            "each node/service edge can be newly built at most once",
            "candidate availability blocks early construction",
            "period CAPEX and construction-count gates",
            "period/scenario heat availability balance",
            "shared-plan non-anticipativity across scenarios when expected-value mode is active",
        ],
    }
    return _finalize_chronology_plan(candidate_rows, decisions, params, context, metadata)


def _finalize_plan(candidate_rows, selected_by_key, selected_by_year, params, context, solver, solver_metadata=None):
    """Expand compact solver decisions into period/scenario result rows.

    Solvers return only "candidate X is first connected in year Y". The UI and
    database need the full indexed view: for every candidate, every period, and
    every scenario, whether the node is already connected, newly connected, or
    not connected. This function creates that full matrix.
    """
    periods = params["periods"]
    scenarios = params["scenarios"]
    eligible = [row for row in candidate_rows if row["status"] == CandidateConnection.Status.ELIGIBLE]
    selected_by_year = defaultdict(list, selected_by_year or {})
    cumulative_capex = 0.0
    cumulative_heat = 0.0
    results = []

    for period in periods:
        year = period["year"]
        new_rows = selected_by_year.get(year, [])
        cumulative_capex += sum(row["estimated_connection_cost_cad"] for row in new_rows)
        cumulative_heat += sum(
            row["peak_heat_kw"]
            for row in new_rows
            if row["candidate_role"] == CandidateConnection.CandidateRole.CONSUMER
        )

        for row in eligible:
            selected_year = selected_by_key.get(row["key"])
            # connected = N_i,t,m, newly = Nbar_i,t,m. In expected-value mode
            # these booleans are shared across scenarios. In scenario-specific
            # mode this function is called once per scenario, so they can differ
            # across scenario result rows.
            connected = selected_year is not None and selected_year <= year
            newly = selected_year == year
            for scenario in scenarios:
                results.append(
                    _plan_result_row(row, period, scenario, connected, newly, cumulative_heat, cumulative_capex, params, context)
                )

    rejected = [row for row in candidate_rows if row["status"] == CandidateConnection.Status.REJECTED]
    for row in rejected:
        for period in periods:
            for scenario in scenarios:
                results.append(
                    _plan_result_row(
                        row,
                        period,
                        scenario,
                        False,
                        False,
                        cumulative_heat,
                        cumulative_capex,
                        params,
                        context,
                        reason=row.get("rejection_reason"),
                    )
                )

    return {
        "selected_by_key": selected_by_key,
        "selected_by_year": dict(selected_by_year),
        "results": results,
        "solver": solver,
        "solver_metadata": solver_metadata or {},
    }


def _finalize_chronology_plan(candidate_rows, decisions, params, context, solver_metadata=None):
    """Convert explicit chronology variables into the normal plan payload.

    The full chronology solver already decides every candidate-period-scenario
    state. This function keeps those exact states instead of reconstructing
    them from a single selected year. The top-level selected_by_key and
    selected_by_year maps remain as compatibility summaries for maps, tables,
    and physical child-network creation.
    """
    periods = params["periods"]
    scenarios = params["scenarios"]
    eligible = [row for row in candidate_rows if row["status"] == CandidateConnection.Status.ELIGIBLE]
    rejected = [row for row in candidate_rows if row["status"] == CandidateConnection.Status.REJECTED]
    selected_by_key = {}
    selected_by_year = defaultdict(list)
    selected_year_keys = defaultdict(set)
    cumulative_by_scenario = {
        scenario["name"]: {"heat_kw": 0.0, "capex_cad": 0.0}
        for scenario in scenarios
    }
    results = []

    for period in periods:
        year = period["year"]
        for scenario in scenarios:
            scenario_name = scenario["name"]
            cumulative = cumulative_by_scenario[scenario_name]
            newly_selected_rows = []
            for row in eligible:
                state = decisions.get((row["key"], year, scenario_name), {})
                if state.get("node_newly_connected"):
                    newly_selected_rows.append(row)
                    previous_year = selected_by_key.get(row["key"])
                    selected_by_key[row["key"]] = min(previous_year, year) if previous_year is not None else year
                    if row["key"] not in selected_year_keys[year]:
                        selected_year_keys[year].add(row["key"])
                        selected_by_year[year].append(row)

            cumulative["capex_cad"] += sum(
                row["estimated_connection_cost_cad"]
                for row in newly_selected_rows
                if decisions.get((row["key"], year, scenario_name), {}).get("edge_newly_built")
            )
            cumulative["heat_kw"] += sum(
                row["peak_heat_kw"]
                for row in newly_selected_rows
                if row["candidate_role"] == CandidateConnection.CandidateRole.CONSUMER
            )

            for row in eligible:
                state = decisions.get((row["key"], year, scenario_name), {})
                results.append(_plan_result_row_with_state(
                    row,
                    period,
                    scenario,
                    state,
                    cumulative["heat_kw"],
                    cumulative["capex_cad"],
                    params,
                    context,
                ))

            for row in rejected:
                results.append(_plan_result_row_with_state(
                    row,
                    period,
                    scenario,
                    {
                        "node_connected": False,
                        "node_newly_connected": False,
                        "edge_exists": False,
                        "edge_newly_built": False,
                        "plant_exists": False,
                    },
                    cumulative["heat_kw"],
                    cumulative["capex_cad"],
                    params,
                    context,
                    reason=row.get("rejection_reason"),
                ))

    return {
        "selected_by_key": selected_by_key,
        "selected_by_year": dict(selected_by_year),
        "results": results,
        "solver": FULL_CHRONOLOGY_SOLVER_NAME,
        "solver_metadata": solver_metadata or {},
    }


def _plan_result_row_with_state(row, period, scenario, state, cumulative_heat, cumulative_capex, params, context, reason=None):
    """Build one result row while preserving explicit chronology decisions."""
    connected = bool(state.get("node_connected"))
    newly = bool(state.get("node_newly_connected"))
    if reason is None:
        if newly:
            reason = "Built in this phase by the chronology optimizer"
        elif connected:
            reason = "Inherited from an earlier phase by the chronology constraint"
        else:
            reason = "Not selected by the chronology optimizer"

    payload = _plan_result_row(
        row,
        period,
        scenario,
        connected,
        newly,
        cumulative_heat,
        cumulative_capex,
        params,
        context,
        reason=reason,
    )
    payload.update({
        "edge_exists": bool(state.get("edge_exists")),
        "edge_newly_built": bool(state.get("edge_newly_built")),
        "plant_exists": bool(state.get("plant_exists")),
    })
    return payload


def _create_phase_networks(base_network, study, plan, candidate_map, params):
    """Create physical child networks for a phase schedule.

    Each phase network starts as a clone of the base network. Then all
    candidates selected up to that phase are attached with routed service
    connections. This preserves the initial routed trunk and lets the output
    network show the cumulative build-out at that planning year.
    """
    source_routes = _source_routes(base_network)
    output_networks = {}
    previous_network = base_network
    cumulative_rows = []
    name_suffix = params.get("phase_network_name_suffix")
    scenario_name = params.get("scenario_name")
    for period in params["periods"]:
        new_rows = plan["selected_by_year"].get(period["year"], [])
        if not new_rows and not cumulative_rows:
            continue
        cumulative_rows.extend(new_rows)
        phase_name = f"{study.name} - {period.get('label') or period['year']}"
        if name_suffix:
            phase_name = f"{phase_name} - {name_suffix}"
        clone_result = clone_network(
            base_network,
            name=phase_name,
            stage=Network.DerivationStage.ROUTE_INFILL,
            method="network_phasing",
            parameters={
                "phasing_study_id": study.id,
                "period_year": period["year"],
                "scenario_name": scenario_name,
                "selected_candidate_keys": [row["key"] for row in cumulative_rows],
            },
            include_edges=True,
            include_connections=False,
            include_routes=False,
            include_pipe_sizing=False,
        )
        if previous_network and previous_network.id != base_network.id:
            clone_result.network.parent_network = previous_network
            clone_result.network.save(update_fields=["parent_network", "updated_at"])

        phase_translation = EdgeTranslation.objects.create(
            network=clone_result.network,
            source_network=base_network,
            translation=params["route_mode"],
            name=f"{clone_result.network.name} - Phasing Routes",
            description="Route tracing for network phasing output, including original routes and phased service connections.",
            is_default=False,
            parameters={
                "phasing_study_id": study.id,
                "period_year": period["year"],
                "scenario_name": scenario_name,
                "route_mode": params["route_mode"],
                "fallback": "provider_route_or_trunk_aligned",
            },
            algorithm=None,
            edge_connection_algorithm=None,
        )
        route_map = _clone_source_routes(source_routes, clone_result.edge_map)
        cloned_route_ids = [route.id for route in route_map.values()]
        if cloned_route_ids:
            Route.objects.filter(id__in=cloned_route_ids).update(translation=phase_translation)
        tap_items_by_source_route = defaultdict(list)
        for row in cumulative_rows:
            candidate = candidate_map[row["key"]]
            # The candidate service line is routed using the same route mode as
            # the parent route where possible, so phased additions visually
            # match the original routed network.
            _add_candidate_to_network(
                clone_result.network,
                candidate,
                row,
                route_map,
                clone_result.edge_map,
                tap_items_by_source_route,
                params,
                phase_translation,
            )

        _split_routes_at_infill_taps(route_map, tap_items_by_source_route)
        routes_for_length = Route.objects.filter(edge__network=clone_result.network).select_related("edge")
        physical_length = unique_route_length(routes_for_length)
        phase_translation.total_edge_length = physical_length
        phase_translation.save(update_fields=["total_edge_length"])
        clone_result.network.total_length = physical_length
        clone_result.network.save(update_fields=["total_length", "updated_at"])
        output_networks[period["year"]] = clone_result.network
        previous_network = clone_result.network
    return output_networks


def _create_scenario_phase_networks(base_network, study, plan, candidate_map, params):
    """Create physical child networks for scenario-specific schedules."""
    output_networks = {}
    for scenario in params["scenarios"]:
        scenario_name = scenario["name"]
        scenario_plan = (plan.get("selected_by_scenario") or {}).get(scenario_name, {})
        selected_by_year = scenario_plan.get("selected_by_year") or {}
        if not selected_by_year:
            continue
        scenario_networks = _create_phase_networks(
            base_network,
            study,
            {
                **plan,
                "selected_by_year": selected_by_year,
            },
            candidate_map,
            {
                **params,
                "scenario_name": scenario_name,
                "phase_network_name_suffix": f"{scenario_name} Scenario",
            },
        )
        for year, network in scenario_networks.items():
            output_networks[(scenario_name, year)] = network
    return output_networks


def _add_candidate_to_network(network, candidate, row, route_map, edge_map, tap_items_by_source_route, params, phase_translation=None):
    source_route = route_map.get(row.get("source_route_id"))
    source_edge = edge_map.get(row.get("source_edge_id"))
    snap = row.get("snap_point") or {}
    if not snap or (not source_route and not source_edge):
        return

    node_lat, node_lng = _candidate_coordinates(candidate)
    if node_lat is None or node_lng is None:
        return

    route_mode = (_route_travel_mode(source_route) if source_route else None) or params["route_mode"]
    start_coords = (snap["lat"], snap["lng"])
    end_coords = (node_lat, node_lng)
    route_data = (
        _stored_candidate_service_route_data(row, start_coords, end_coords)
        or _phasing_service_route_data(source_route, source_edge, start_coords, end_coords, route_mode)
    )
    tap_node = Node.objects.create(
        network=network,
        name=f"Phase tap for {candidate.name}",
        latitude=snap["lat"],
        longitude=snap["lng"],
        node_type="junction",
        other_information={
            "phasing_study_id": candidate.study_id,
            "candidate_id": candidate.id,
            "is_phasing_tap": True,
            "source_route_id": row.get("source_route_id"),
        },
    )
    node = Node.objects.create(
        network=network,
        name=candidate.name,
        latitude=node_lat,
        longitude=node_lng,
        node_type=_node_type_for_candidate(candidate),
        building=candidate.building if candidate.building_id else None,
        cluster=candidate.cluster if candidate.cluster_id else None,
        other_information={
            "phasing_study_id": candidate.study_id,
            "candidate_id": candidate.id,
            "candidate_role": candidate.candidate_role,
        },
    )
    edge = Edge.objects.create(
        network=network,
        edge_algorithm=None,
        name=f"{candidate.name} phased service connection",
        start_node=tap_node,
        end_node=node,
        length=route_data["distance"],
        is_default=False,
        other_information={
            "network_phasing_service": True,
            "phasing_study_id": candidate.study_id,
            "candidate_id": candidate.id,
            "route_travel_mode": route_mode,
            "route_resolved_mode": route_data.get("mode") or route_mode,
        },
    )
    Route.objects.create(
        edge=edge,
        translation=phase_translation,
        route_type="service",
        polyline=route_data["polyline"],
        distance_meters=route_data["distance"],
        duration_seconds=route_data.get("duration") or 0,
    )
    service_path = _route_data_path(route_data)
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    metadata.update({
        "service_path": service_path,
        "service_path_is_routed": len(service_path) > 2,
        "service_route_distance_m": route_data["distance"],
        "route_travel_mode": route_mode,
        "route_resolved_mode": route_data.get("mode") or route_mode,
    })
    candidate.metadata = metadata
    candidate.save(update_fields=["metadata"])
    if source_route:
        tap_items_by_source_route[row.get("source_route_id")].append({
            "tap_node": tap_node,
            "source_route": source_route,
            "along_m": row.get("along_m"),
        })


def _stored_candidate_service_route_data(row, start_coords, end_coords):
    route_mode = str(row.get("route_travel_mode") or "").lower()
    if route_mode == "direct_fallback" or not row.get("service_path_is_routed"):
        return None

    points = []
    for point in row.get("service_path") or []:
        lat = _float_value(point.get("lat") if isinstance(point, dict) else None, None)
        lng = _float_value(point.get("lng") if isinstance(point, dict) else None, None)
        if lat is not None and lng is not None:
            points.append((lat, lng))
    if len(points) < 3:
        return None

    existing_route_data = {
        "polyline": encode_polyline(points),
        "distance": row.get("connection_length_m") or route_distance(points),
        "duration": 0,
    }
    normalized = normalize_route_data(existing_route_data, start_coords, end_coords)
    if not normalized:
        return None
    normalized["mode"] = row.get("route_travel_mode") or "stored_candidate_route"
    normalized["routed"] = len(points) > 2
    return normalized


def _phasing_service_route_data(source_route, source_edge, start_coords, end_coords, route_mode):
    route_data = build_route_data(start_coords, end_coords, route_mode, use_google=True)
    if route_data.get("mode") != "direct_fallback":
        return route_data

    provider_fallback = _osrm_service_route_data(start_coords, end_coords, route_mode)
    if provider_fallback:
        return provider_fallback

    routed_fallback = _trunk_aligned_service_route_data(source_route, start_coords, end_coords) if source_route else None
    if not routed_fallback and source_edge:
        routed_fallback = _edge_aligned_service_route_data(source_edge, start_coords, end_coords)
    return routed_fallback or route_data


def _osrm_service_route_data(start_coords, end_coords, route_mode):
    profile = {
        "driving": "driving",
        "walking": "walking",
        "bicycling": "cycling",
        "cycling": "cycling",
    }.get(route_mode, "driving")
    url = (
        f"https://router.project-osrm.org/route/v1/{profile}/"
        f"{float(start_coords[1])},{float(start_coords[0])};"
        f"{float(end_coords[1])},{float(end_coords[0])}"
    )
    try:
        response = requests.get(
            url,
            params={"overview": "full", "geometries": "polyline", "steps": "false"},
            timeout=15,
        )
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None

    if data.get("code") != "Ok" or not data.get("routes"):
        return None
    route = data["routes"][0]
    normalized = normalize_route_data(
        {
            "polyline": route.get("geometry"),
            "distance": route.get("distance"),
            "duration": route.get("duration"),
        },
        start_coords,
        end_coords,
    )
    if not normalized:
        return None
    if float(normalized.get("distance") or 0) <= 0.5:
        return None
    normalized["mode"] = f"osrm_{profile}"
    return normalized


def _trunk_aligned_service_route_data(source_route, start_coords, end_coords):
    points = _route_points(source_route)
    return _aligned_service_route_data(points, start_coords, end_coords, "trunk_aligned_fallback")


def _edge_aligned_service_route_data(source_edge, start_coords, end_coords):
    points = [
        (source_edge.start_node.latitude, source_edge.start_node.longitude),
        (source_edge.end_node.latitude, source_edge.end_node.longitude),
    ]
    return _aligned_service_route_data(points, start_coords, end_coords, "edge_aligned_fallback")


def _aligned_service_route_data(points, start_coords, end_coords, mode):
    if len(points) < 2:
        return None

    projected = _project_service_geometry(points, start_coords, end_coords)
    if not projected:
        return None

    start_xy = projected["start"]
    end_xy = projected["end"]
    unit = projected["unit"]
    vector_to_end = (end_xy[0] - start_xy[0], end_xy[1] - start_xy[1])
    direct_m = math.hypot(vector_to_end[0], vector_to_end[1])
    if direct_m <= 1.0:
        return None

    along_m = vector_to_end[0] * unit[0] + vector_to_end[1] * unit[1]
    if direct_m < 12.0:
        minimum_along_m = direct_m * 0.6
    else:
        minimum_along_m = min(max(12.0, direct_m * 0.35), 80.0)
    if abs(along_m) < minimum_along_m:
        along_m = minimum_along_m if along_m >= 0 else -minimum_along_m
    maximum_along_m = max(minimum_along_m, direct_m * 1.25)
    along_m = max(-maximum_along_m, min(maximum_along_m, along_m))

    waypoint_xy = (
        start_xy[0] + unit[0] * along_m,
        start_xy[1] + unit[1] * along_m,
    )
    route_points = _unproject_service_points(
        projected["inverse_transformer"],
        [start_xy, waypoint_xy, end_xy],
    )
    route_points = _dedupe_service_points(route_points)
    if len(route_points) < 3:
        return None

    return {
        "polyline": encode_polyline(route_points),
        "distance": route_distance(route_points),
        "duration": 0,
        "mode": mode,
    }


def _project_service_geometry(route_points, start_coords, end_coords):
    all_points = list(route_points) + [start_coords, end_coords]
    centroid_lat = sum(float(point[0]) for point in all_points) / len(all_points)
    centroid_lng = sum(float(point[1]) for point in all_points) / len(all_points)
    epsg = _utm_epsg(centroid_lng, centroid_lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inverse_transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)

    projected_route = [
        transformer.transform(float(lng), float(lat))
        for lat, lng in route_points
        if lat is not None and lng is not None
    ]
    if len(projected_route) < 2:
        return None

    start_xy = transformer.transform(float(start_coords[1]), float(start_coords[0]))
    end_xy = transformer.transform(float(end_coords[1]), float(end_coords[0]))
    best_segment = None
    best_distance = None
    for first, second in zip(projected_route, projected_route[1:]):
        segment = (second[0] - first[0], second[1] - first[1])
        length = math.hypot(segment[0], segment[1])
        if length <= 0:
            continue
        t = (
            ((start_xy[0] - first[0]) * segment[0] + (start_xy[1] - first[1]) * segment[1])
            / (length * length)
        )
        t = max(0.0, min(1.0, t))
        snap = (first[0] + segment[0] * t, first[1] + segment[1] * t)
        distance = math.hypot(start_xy[0] - snap[0], start_xy[1] - snap[1])
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_segment = segment

    if not best_segment:
        return None
    segment_length = math.hypot(best_segment[0], best_segment[1])
    if segment_length <= 0:
        return None

    return {
        "start": start_xy,
        "end": end_xy,
        "unit": (best_segment[0] / segment_length, best_segment[1] / segment_length),
        "inverse_transformer": inverse_transformer,
    }


def _unproject_service_points(inverse_transformer, projected_points):
    points = []
    for x, y in projected_points:
        lng, lat = inverse_transformer.transform(x, y)
        points.append((float(lat), float(lng)))
    return points


def _dedupe_service_points(points):
    deduped = []
    for point in points:
        if not deduped or route_distance([deduped[-1], point]) > 1.0:
            deduped.append(point)
    return deduped


def _safe_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _persist_results(study, plan, candidate_map, period_map, scenario_map, output_networks_by_year, params):
    """Write PhasingResult rows for every candidate-period-scenario decision.

    The solver works with dictionaries for speed. Persistence converts each
    result dictionary into the normalized database model:
    - node_connected      -> N_i,t,m
    - node_newly_connected -> Nbar_i,t,m
    - edge_exists         -> E_i,j,t,m, simplified as service edge exists
    - edge_newly_built    -> Ebar_i,j,t,m
    - plant_exists        -> P_i,t,m for source candidates
    - heat_supplied_kw    -> Q_i,t,m
    """
    rows = []
    for result in plan["results"]:
        candidate = candidate_map[result["candidate_key"]]
        period = period_map[result["year"]]
        scenario = scenario_map[result["scenario_name"]]
        rows.append(PhasingResult(
            study=study,
            scenario=scenario,
            period=period,
            candidate=candidate,
            output_network=_output_network_for_result(output_networks_by_year, scenario.name, period.year),
            node_connected=result["node_connected"],
            node_newly_connected=result["node_newly_connected"],
            edge_exists=result["edge_exists"],
            edge_newly_built=result["edge_newly_built"],
            plant_exists=result["plant_exists"],
            heat_supplied_kw=result["heat_supplied_kw"],
            capex_cad=result["capex_cad"],
            discounted_net_benefit_cad=result["discounted_net_benefit_cad"],
            cumulative_heat_kw=result["cumulative_heat_kw"],
            cumulative_capex_cad=result["cumulative_capex_cad"],
            reason=result["reason"],
        ))
    PhasingResult.objects.bulk_create(rows, batch_size=500)
    # CandidateConnection.status is a study-level summary. In scenario-specific
    # mode this means "selected in at least one scenario"; the exact scenario
    # differences remain in PhasingResult.
    selected_keys = {result["candidate_key"] for result in plan["results"] if result["node_newly_connected"]}
    for key in selected_keys:
        CandidateConnection.objects.filter(pk=candidate_map[key].pk).update(status=CandidateConnection.Status.SELECTED)


def _output_network_for_result(output_networks_by_year, scenario_name, year):
    return (
        output_networks_by_year.get((scenario_name, year)) or
        output_networks_by_year.get(year)
    )


def _plan_result_row(row, period, scenario, connected, newly, cumulative_heat, cumulative_capex, params, context, reason=None):
    """Build one result row for one candidate-period-scenario combination."""
    demand_kw = row["peak_heat_kw"] * scenario["demand_multiplier"] if row["candidate_role"] == CandidateConnection.CandidateRole.CONSUMER else 0.0
    source_kw = row["available_heat_kw"] if row["candidate_role"] == CandidateConnection.CandidateRole.SOURCE else 0.0
    heat_kw = demand_kw if row["candidate_role"] == CandidateConnection.CandidateRole.CONSUMER else source_kw
    capex = row["estimated_connection_cost_cad"] if newly else 0.0
    value_breakdown = _annual_candidate_value_breakdown(row, scenario, params, context)
    annual_revenue = value_breakdown["net_annual_value_cad"]
    years_from_start = max(period["year"] - params["periods"][0]["year"], 0)
    discounted = (
        # Probability weighting is appropriate only when multiple scenarios are
        # active in expected-value mode. Scenario-specific mode calls the solver
        # with one scenario at a time, so the probability simply scales the
        # displayed scenario objective consistently with the input scenario.
        scenario["probability"] *
        (_present_value(annual_revenue, scenario["discount_rate"], context["project_lifespan_years"], years_from_start) - capex)
        if newly else 0.0
    )
    return {
        "candidate_key": row["key"],
        "year": period["year"],
        "scenario_name": scenario["name"],
        "node_connected": connected,
        "node_newly_connected": newly,
        "edge_exists": connected,
        "edge_newly_built": newly,
        "plant_exists": connected and row["candidate_role"] == CandidateConnection.CandidateRole.SOURCE,
        "heat_supplied_kw": heat_kw if connected else 0.0,
        "capex_cad": capex,
        "annual_value_cad": annual_revenue if connected else 0.0,
        "annual_value_breakdown": value_breakdown if connected or newly else {},
        "discounted_net_benefit_cad": discounted,
        "cumulative_heat_kw": cumulative_heat,
        "cumulative_capex_cad": cumulative_capex,
        "reason": reason or ("Selected in this phase" if newly else ("Already connected from earlier phase" if connected else "Not selected in this phase")),
    }


def _expected_candidate_value(row, year, scenarios, params, context):
    """Present-value objective contribution for connecting one candidate."""
    years_from_start = max(year - params["periods"][0]["year"], 0)
    expected_value = 0.0
    for scenario in scenarios:
        annual_revenue = _annual_candidate_value_cad(row, scenario, params, context)
        pv_revenue = _present_value(
            annual_revenue,
            scenario["discount_rate"],
            context["project_lifespan_years"],
            years_from_start,
        )
        expected_value += scenario["probability"] * (pv_revenue - row["estimated_connection_cost_cad"])
    return expected_value


def _annual_candidate_value_cad(row, scenario, params, context):
    return _annual_candidate_value_breakdown(row, scenario, params, context)["net_annual_value_cad"]


def _annual_candidate_value_breakdown(row, scenario, params, context):
    """Annual benefit/cost components for one candidate under one scenario.

    Consumers create value from heat sales plus avoided gas-boiler operating
    cost and avoided carbon cost, minus service-pipe pumping cost. Sources do
    not create consumer heat revenue here; they mainly add available heat for
    the network and can be selected by the source-first logic.
    """
    if row["candidate_role"] != CandidateConnection.CandidateRole.CONSUMER:
        return {
            "heat_sales_cad": 0.0,
            "avoided_gas_cost_cad": 0.0,
            "avoided_carbon_cost_cad": 0.0,
            "pumping_cost_cad": 0.0,
            "boiler_emissions_tco2e": 0.0,
            "pumping_emissions_tco2e": 0.0,
            "net_carbon_savings_tco2e": 0.0,
            "net_annual_value_cad": 0.0,
        }

    demand_multiplier = _float_value(scenario.get("demand_multiplier"), 1.0)
    fuel_multiplier = _float_value(scenario.get("fuel_price_multiplier"), 1.0)
    annual_heat_mwh = max(row["annual_heat_mwh"] * demand_multiplier, 0.0)
    peak_heat_kw = max(row["peak_heat_kw"] * demand_multiplier, 0.0)
    heat_sales_rate = _float_value(params.get("heat_price_cad_per_mwh"), 0.0)
    gas_rate = _float_value(context.get("gas_price_cad_per_mwh"), 0.0)
    hydraulics = _candidate_hydraulic_economics(
        peak_heat_kw,
        annual_heat_mwh,
        row["connection_length_m"],
        context,
    )
    boiler_fuel_kwh = (
        annual_heat_mwh * 1000.0 / context["base_boiler_efficiency"]
        if context["base_boiler_efficiency"] > 0 else 0.0
    )
    boiler_emissions_tco2e = boiler_fuel_kwh * context["gas_emission_factor_kgco2e_per_kwh"] / 1000.0
    pumping_emissions_tco2e = hydraulics["annual_pumping_emissions_tco2e"]
    net_carbon_savings_tco2e = boiler_emissions_tco2e - pumping_emissions_tco2e
    heat_sales_cad = annual_heat_mwh * heat_sales_rate
    avoided_gas_cost_cad = annual_heat_mwh * gas_rate * fuel_multiplier
    avoided_carbon_cost_cad = net_carbon_savings_tco2e * context["carbon_price_cad_per_tonne"]
    pumping_cost_cad = hydraulics["annual_pumping_cost_cad"]
    return {
        "heat_sales_cad": heat_sales_cad,
        "avoided_gas_cost_cad": avoided_gas_cost_cad,
        "avoided_carbon_cost_cad": avoided_carbon_cost_cad,
        "pumping_cost_cad": pumping_cost_cad,
        "boiler_emissions_tco2e": boiler_emissions_tco2e,
        "pumping_emissions_tco2e": pumping_emissions_tco2e,
        "net_carbon_savings_tco2e": net_carbon_savings_tco2e,
        "net_annual_value_cad": heat_sales_cad + avoided_gas_cost_cad + avoided_carbon_cost_cad - pumping_cost_cad,
    }


def _present_value(annual_value, discount_rate, horizon_years, years_from_start):
    annual_value = _float_value(annual_value, 0.0)
    discount_rate = _float_value(discount_rate, 0.0)
    horizon_years = max(_int_value(horizon_years, 1), 1)
    years_from_start = max(_float_value(years_from_start, 0.0), 0.0)
    if discount_rate <= 0:
        stream_value = annual_value * horizon_years
    else:
        stream_value = annual_value * ((1 - ((1 + discount_rate) ** -horizon_years)) / discount_rate)
    return stream_value / ((1 + max(discount_rate, 0.0)) ** years_from_start)


def _annual_candidate_value_cad_legacy(row, scenario, params, context):
    heat_sales_rate = _float_value(params.get("heat_price_cad_per_mwh"), 0.0)
    gas_rate = _float_value(context.get("gas_price_cad_per_mwh"), 0.0)
    fuel_multiplier = _float_value(scenario.get("fuel_price_multiplier"), 1.0)
    effective_rate = heat_sales_rate + (gas_rate * fuel_multiplier)
    return row["annual_heat_mwh"] * scenario["demand_multiplier"] * effective_rate


def _expected_multiplier(scenarios):
    return sum(s["probability"] * s["demand_multiplier"] for s in scenarios) or 1.0


def _candidate_available(row, year):
    return not row.get("available_from_year") or row["available_from_year"] <= year


def _period_available_heat_kw(period, context, params):
    """Return the heat capacity available at a specific planning period.

    Priority order:
    1. unlimited heat switch;
    2. period-specific available_heat_kw from the phase limits UI;
    3. base network inferred/configured available heat.

    This is what makes staged heat availability possible: later periods can
    have larger heat limits, so candidates that do not fit early can become
    feasible later.
    """
    if params.get("assume_unlimited_available_heat"):
        return UNLIMITED_AVAILABLE_HEAT_KW
    configured = _float_value(period.get("available_heat_kw") if isinstance(period, dict) else None, None)
    if configured is not None:
        return max(configured, 0.0)
    return max(_float_value(context.get("base_available_heat_kw"), 0.0), 0.0)


def _source_routes(network):
    return list(Route.objects.filter(edge__network=network).select_related(
        "edge",
        "edge__start_node",
        "edge__end_node",
        "translation",
    ).order_by("id"))


def _main_trunk(routes, edges):
    if routes:
        route = max(routes, key=lambda item: _float_value(item.distance_meters, 0.0))
        return {"route": route, "edge": route.edge, "length_m": _float_value(route.distance_meters, 0.0)}
    if edges:
        edge = max(edges, key=lambda item: _float_value(item.length, 0.0))
        return {"route": None, "edge": edge, "length_m": _float_value(edge.length, 0.0)}
    raise ValueError("The selected network needs at least one route or edge before phasing can be evaluated.")


def _build_route_index(routes):
    route_points = []
    all_points = []
    for route in routes:
        points = _route_points(route)
        if len(points) >= 2:
            route_points.append((route, points))
            all_points.extend(points)
    if not route_points:
        raise ValueError("The selected trunk route has no usable geometry.")
    return _projected_index([
        {"route": route, "edge": route.edge, "points": points}
        for route, points in route_points
    ], all_points)


def _build_edge_index(edges):
    rows = []
    all_points = []
    for edge in edges:
        points = [
            (edge.start_node.latitude, edge.start_node.longitude),
            (edge.end_node.latitude, edge.end_node.longitude),
        ]
        points = [(float(lat), float(lng)) for lat, lng in points if lat is not None and lng is not None]
        if len(points) >= 2:
            rows.append({"route": None, "edge": edge, "points": points})
            all_points.extend(points)
    if not rows:
        raise ValueError("The selected trunk edge has no usable geometry.")
    return _projected_index(rows, all_points)


def _projected_index(rows, all_points):
    centroid_lat = sum(point[0] for point in all_points) / len(all_points)
    centroid_lng = sum(point[1] for point in all_points) / len(all_points)
    epsg = _utm_epsg(centroid_lng, centroid_lat)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    inverse_transformer = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    items = []
    for row in rows:
        line = LineString([(lng, lat) for lat, lng in row["points"]])
        projected = shapely_transform(transformer.transform, line)
        if projected.length > 0:
            items.append({**row, "line": projected})
    return {"items": items, "transformer": transformer, "inverse_transformer": inverse_transformer}


def _nearest_reference(latitude, longitude, route_index):
    point = Point(float(longitude), float(latitude))
    projected_point = shapely_transform(route_index["transformer"].transform, point)
    best = None
    for item in route_index["items"]:
        distance = float(item["line"].distance(projected_point))
        along = float(item["line"].project(projected_point))
        snap = item["line"].interpolate(along)
        if best is None or distance < best["distance_to_trunk_m"]:
            lng, lat = route_index["inverse_transformer"].transform(snap.x, snap.y)
            route = item.get("route")
            edge = item.get("edge")
            best = {
                "distance_to_trunk_m": distance,
                "along_m": along,
                "snap_point": {"lat": float(lat), "lng": float(lng)},
                "source_route_id": route.id if route else None,
                "source_edge_id": edge.id if edge else None,
            }
    return best


def _route_points(route):
    from api.utils.network_versioning import decode_polyline
    try:
        points = decode_polyline(route.polyline)
    except Exception:
        points = []
    if len(points) < 2:
        return [
            (route.edge.start_node.latitude, route.edge.start_node.longitude),
            (route.edge.end_node.latitude, route.edge.end_node.longitude),
        ]
    return points


def _candidate_role_for_building(building):
    text = " ".join([
        str(building.waste_heat_source_type or ""),
        str(building.building_type or ""),
        str(building.name or ""),
    ]).lower()
    if any(token in text for token in ["data", "centre", "center", "wastewater", "industrial", "waste heat", "arena", "restaurant"]):
        return CandidateConnection.CandidateRole.SOURCE if building.waste_heat_source_type else CandidateConnection.CandidateRole.CONSUMER
    return CandidateConnection.CandidateRole.CONSUMER


def _candidate_coordinates(candidate):
    obj = candidate.building if candidate.building_id else candidate.cluster
    if not obj:
        return None, None
    return obj.latitude, obj.longitude


def _node_type_for_candidate(candidate):
    if candidate.candidate_role == CandidateConnection.CandidateRole.SOURCE:
        return "producer"
    if candidate.candidate_type == CandidateConnection.CandidateType.CLUSTER:
        return "cluster"
    return "consumer"


def _building_heat_metrics(building, demand_year=None, context=None):
    static_heat_kw = max(_float_value(getattr(building, "heating_demand", None), 0.0), 0.0)
    static_source_kw = max(
        _float_value(getattr(building, "cooling_demand", None), 0.0),
        static_heat_kw if _candidate_role_for_building(building) == CandidateConnection.CandidateRole.SOURCE else 0.0,
        0.0,
    )

    heating = _profile_peak_and_annual([building.id], "heating", demand_year)
    cooling = _profile_peak_and_annual([building.id], "cooling", demand_year)
    waste_heat = _profile_peak_and_annual([building.id], "waste_heat", demand_year)
    source_profile = waste_heat if waste_heat["has_data"] else cooling

    full_load_hours = _float_value((context or {}).get("annual_full_load_hours"), 4000.0)
    peak_heat_kw = heating["peak_kw"] if heating["has_data"] else static_heat_kw
    annual_heat_mwh = heating["annual_mwh"] if heating["has_data"] else peak_heat_kw * full_load_hours / 1000.0
    source_heat_kw = source_profile["peak_kw"] if source_profile["has_data"] else static_source_kw
    source_annual_mwh = source_profile["annual_mwh"] if source_profile["has_data"] else source_heat_kw * full_load_hours / 1000.0
    return {
        "peak_heat_kw": max(peak_heat_kw, 0.0),
        "annual_heat_mwh": max(annual_heat_mwh, 0.0),
        "source_heat_kw": max(source_heat_kw, 0.0),
        "source_annual_mwh": max(source_annual_mwh, 0.0),
    }


def _cluster_heat_metrics(cluster, demand_year=None, context=None):
    building_ids = list(
        BuildingCluster.objects
        .filter(cluster=cluster)
        .values_list("building_id", flat=True)
    )
    full_load_hours = _float_value((context or {}).get("annual_full_load_hours"), 4000.0)
    static_heat_kw = max(_float_value(getattr(cluster, "heating_demand", None), 0.0), 0.0)
    static_source_kw = max(_float_value(getattr(cluster, "cooling_demand", None), 0.0), 0.0)
    if not building_ids:
        return {
            "peak_heat_kw": static_heat_kw,
            "annual_heat_mwh": static_heat_kw * full_load_hours / 1000.0,
            "source_heat_kw": static_source_kw,
            "source_annual_mwh": static_source_kw * full_load_hours / 1000.0,
        }

    heating = _profile_peak_and_annual(building_ids, "heating", demand_year)
    cooling = _profile_peak_and_annual(building_ids, "cooling", demand_year)
    waste_heat = _profile_peak_and_annual(building_ids, "waste_heat", demand_year)
    source_profile = waste_heat if waste_heat["has_data"] else cooling
    peak_heat_kw = heating["peak_kw"] if heating["has_data"] else static_heat_kw
    annual_heat_mwh = heating["annual_mwh"] if heating["has_data"] else peak_heat_kw * full_load_hours / 1000.0
    source_heat_kw = source_profile["peak_kw"] if source_profile["has_data"] else static_source_kw
    source_annual_mwh = source_profile["annual_mwh"] if source_profile["has_data"] else source_heat_kw * full_load_hours / 1000.0
    return {
        "peak_heat_kw": max(peak_heat_kw, 0.0),
        "annual_heat_mwh": max(annual_heat_mwh, 0.0),
        "source_heat_kw": max(source_heat_kw, 0.0),
        "source_annual_mwh": max(source_annual_mwh, 0.0),
    }


def _profile_peak_and_annual(building_ids, profile_type, demand_year=None):
    if not building_ids or not demand_year:
        return {"has_data": False, "peak_kw": 0.0, "annual_mwh": 0.0}
    queryset = LoadProfile.objects.filter(
        building_id__in=building_ids,
        profile_type=profile_type,
        start_date__gte=dt.datetime(int(demand_year), 1, 1, tzinfo=dt.timezone.utc),
        start_date__lt=dt.datetime(int(demand_year) + 1, 1, 1, tzinfo=dt.timezone.utc),
    )
    aggregate = queryset.aggregate(count=Count("id"), annual_kwh=Sum("value"))
    if not aggregate.get("count"):
        return {"has_data": False, "peak_kw": 0.0, "annual_mwh": 0.0}
    peak = queryset.values("start_date").annotate(total_kw=Sum("value")).order_by("-total_kw").first()
    return {
        "has_data": True,
        "peak_kw": _float_value((peak or {}).get("total_kw"), 0.0),
        "annual_mwh": _float_value(aggregate.get("annual_kwh"), 0.0) / 1000.0,
    }


def _profile_coincident_peak_kw(building_ids, profile_type, demand_year=None):
    metrics = _profile_peak_and_annual(building_ids, profile_type, demand_year)
    return metrics["peak_kw"] if metrics["has_data"] else None


def _infer_network_heat(network, demand_year=None):
    source_kw = 0.0
    consumer_building_ids = []
    for node in Node.objects.filter(network=network).select_related("building"):
        building = node.building
        if not building:
            continue
        role = _candidate_role_for_building(building)
        if node.node_type == "producer" or role == CandidateConnection.CandidateRole.SOURCE:
            source_kw += _building_heat_metrics(building, demand_year, None)["source_heat_kw"]
        else:
            consumer_building_ids.append(building.id)

    served_kw = _profile_coincident_peak_kw(consumer_building_ids, "heating", demand_year)
    if served_kw is None:
        served_kw = sum(
            max(_float_value(value, 0.0), 0.0)
            for value in Building.objects.filter(id__in=consumer_building_ids).values_list("heating_demand", flat=True)
        )
    return source_kw, served_kw


def _existing_building_type_counts(network):
    counts = defaultdict(int)
    for building_type in Node.objects.filter(network=network, building__isnull=False).values_list("building__building_type", flat=True):
        counts[building_type or "other"] += 1
    return dict(counts)


def _diversity_score(row, context):
    counts = context.get("existing_type_counts") or {}
    total = sum(counts.values())
    if total <= 0:
        return 0.5
    share = counts.get(row.get("building_type") or "other", 0) / total
    return max(0.0, min(1.0, 1.0 - share))


def _network_design_constants(network):
    try:
        return network.design_constants
    except Exception:
        return None


def _network_heat_inventory(network):
    try:
        return network.heat_inventory
    except Exception:
        return None


def _preview_payload(base_network, params, context, candidate_rows, plan):
    """Build the non-persistent API response used by Preview.

    The payload mirrors a saved study report closely enough that the frontend
    can render the same maps, candidate tables, and charts for previews and
    completed runs. In scenario-specific mode, the raw results contain the
    scenario-specific decisions; the top-level timeline is kept as a union/
    compatibility summary while the frontend recomputes the active timeline
    from result rows when the user changes scenario.
    """
    selected_keys = plan["selected_by_key"]
    candidates_by_key = {row["key"]: row for row in candidate_rows}
    timeline = []
    cumulative_capex = 0.0
    cumulative_heat = 0.0
    for period in params["periods"]:
        rows = plan["selected_by_year"].get(period["year"], [])
        cumulative_capex += sum(row["estimated_connection_cost_cad"] for row in rows)
        cumulative_heat += sum(row["peak_heat_kw"] for row in rows if row["candidate_role"] == CandidateConnection.CandidateRole.CONSUMER)
        timeline.append({
            "year": period["year"],
            "label": period.get("label") or str(period["year"]),
            "new_connections": len(rows),
            "added_heat_kw": sum(row["peak_heat_kw"] for row in rows),
            "added_source_kw": sum(row["available_heat_kw"] for row in rows),
            "capex_cad": sum(row["estimated_connection_cost_cad"] for row in rows),
            "cumulative_heat_kw": cumulative_heat,
            "cumulative_capex_cad": cumulative_capex,
            "candidates": [_candidate_row_payload(row, selected_keys) for row in rows],
        })
    return {
        "base_network": _network_payload(base_network),
        "parameters": _phasing_parameter_payload(params, context),
        "solver": plan.get("solver") or GREEDY_SOLVER_NAME,
        "solver_metadata": plan.get("solver_metadata", {}),
        "main_trunk": {
            "route_id": context["trunk"].get("route").id if context["trunk"].get("route") else None,
            "edge_id": context["trunk"].get("edge").id if context["trunk"].get("edge") else None,
            "length_m": context["trunk"].get("length_m"),
        },
        "periods": params["periods"],
        "scenarios": params["scenarios"],
        "candidates": [_candidate_row_payload(row, selected_keys) for row in candidate_rows],
        "results": [_preview_result_payload(result, candidates_by_key) for result in plan["results"]],
        "timeline": timeline,
        "charts": _chart_payload(timeline),
        "map": _preview_map_payload(base_network, candidate_rows, selected_keys),
    }


def _candidate_row_payload(row, selected_keys):
    return {
        **row,
        "candidate_point": _candidate_point_from_path(row.get("service_path")),
        "selected_year": selected_keys.get(row["key"]),
        "selected": row["key"] in selected_keys,
    }


def _candidate_payload(candidate):
    metadata = candidate.metadata if isinstance(candidate.metadata, dict) else {}
    service_path = metadata.get("service_path", [])
    reference_hydraulics = metadata.get("reference_hydraulic_economics", {})
    candidate_point = _candidate_point_from_path(service_path)
    if not candidate_point:
        lat, lng = _candidate_coordinates(candidate)
        if lat is not None and lng is not None:
            candidate_point = {"lat": float(lat), "lng": float(lng)}
    return {
        "id": candidate.id,
        "candidate_type": candidate.candidate_type,
        "building": candidate.building_id,
        "cluster": candidate.cluster_id,
        "name": candidate.name,
        "candidate_role": candidate.candidate_role,
        "available_from_year": candidate.available_from_year,
        "peak_heat_kw": candidate.peak_heat_kw,
        "annual_heat_mwh": candidate.annual_heat_mwh,
        "available_heat_kw": candidate.available_heat_kw,
        "building_count": candidate.cluster.no_of_buildings if candidate.cluster_id and candidate.cluster and candidate.cluster.no_of_buildings is not None else metadata.get("building_count", 1 if candidate.candidate_type == CandidateConnection.CandidateType.BUILDING else 0),
        "distance_to_trunk_m": candidate.distance_to_trunk_m,
        "connection_length_m": candidate.connection_length_m,
        "estimated_connection_cost_cad": candidate.estimated_connection_cost_cad,
        "linear_heat_density_mwh_per_m_year": candidate.linear_heat_density_mwh_per_m_year,
        "suitability_score": candidate.suitability_score,
        "status": candidate.status,
        "rejection_reason": candidate.rejection_reason,
        "snap_point": {"lat": candidate.snap_latitude, "lng": candidate.snap_longitude} if candidate.snap_latitude is not None else None,
        "service_path": service_path,
        "candidate_point": candidate_point,
        "reference_hydraulic_economics": reference_hydraulics,
        "mass_flow_kg_s": reference_hydraulics.get("mass_flow_kg_s"),
        "estimated_diameter_m": reference_hydraulics.get("diameter_m"),
        "pressure_drop_pa": reference_hydraulics.get("pressure_drop_pa"),
        "pump_power_kw": reference_hydraulics.get("pump_power_kw"),
        "annual_pumping_cost_cad": reference_hydraulics.get("annual_pumping_cost_cad"),
        "annual_pumping_emissions_tco2e": reference_hydraulics.get("annual_pumping_emissions_tco2e"),
        "metadata": metadata,
    }


def _candidate_result_payload(result, service_route_map=None):
    payload = _candidate_payload(result.candidate)
    route_payload = _service_route_payload_for_result(result, service_route_map)
    if route_payload:
        payload.update(route_payload)
        payload["candidate_point"] = _candidate_point_from_path(route_payload.get("service_path"))
    payload.update({
        "result_id": result.id,
        "year": result.period.year,
        "scenario": result.scenario.name,
        "capex_cad": result.capex_cad,
        "heat_supplied_kw": result.heat_supplied_kw,
        "reason": result.reason,
    })
    return payload


def _result_payload(result, service_route_map=None, context=None):
    metadata = result.candidate.metadata if isinstance(result.candidate.metadata, dict) else {}
    route_payload = _service_route_payload_for_result(result, service_route_map)
    service_path = (route_payload or {}).get("service_path") or metadata.get("service_path", [])
    candidate_point = _candidate_point_from_path(service_path)
    if not candidate_point:
        lat, lng = _candidate_coordinates(result.candidate)
        if lat is not None and lng is not None:
            candidate_point = {"lat": float(lat), "lng": float(lng)}
    route_metadata = {
        key: value
        for key, value in (route_payload or {}).items()
        if key != "service_path"
    }
    value_breakdown = {}
    annual_value_cad = 0.0
    if context:
        row = {
            "candidate_role": result.candidate.candidate_role,
            "annual_heat_mwh": result.candidate.annual_heat_mwh,
            "peak_heat_kw": result.candidate.peak_heat_kw,
            "connection_length_m": result.candidate.connection_length_m,
        }
        scenario = _scenario_payload(result.scenario)
        params = {"heat_price_cad_per_mwh": context.get("heat_price_cad_per_mwh", 0.0)}
        value_breakdown = _annual_candidate_value_breakdown(row, scenario, params, context)
        annual_value_cad = value_breakdown["net_annual_value_cad"] if result.node_connected else 0.0
    return {
        "id": result.id,
        "year": result.period.year,
        "period_label": result.period.label,
        "scenario": result.scenario.name,
        "candidate_id": result.candidate_id,
        "candidate_type": result.candidate.candidate_type,
        "candidate_name": result.candidate.name,
        "candidate_role": result.candidate.candidate_role,
        "node_connected": result.node_connected,
        "node_newly_connected": result.node_newly_connected,
        "edge_exists": result.edge_exists,
        "edge_newly_built": result.edge_newly_built,
        "plant_exists": result.plant_exists,
        "heat_supplied_kw": result.heat_supplied_kw,
        "capex_cad": result.capex_cad,
        "annual_value_cad": annual_value_cad,
        "annual_value_breakdown": value_breakdown if result.node_connected or result.node_newly_connected else {},
        "discounted_net_benefit_cad": result.discounted_net_benefit_cad,
        "cumulative_heat_kw": result.cumulative_heat_kw,
        "cumulative_capex_cad": result.cumulative_capex_cad,
        "output_network_id": result.output_network_id,
        "reason": result.reason,
        "service_path": service_path,
        "candidate_point": candidate_point,
        "metadata": {**metadata, **route_metadata},
        "connection_length_m": result.candidate.connection_length_m,
        "linear_heat_density_mwh_per_m_year": result.candidate.linear_heat_density_mwh_per_m_year,
        "estimated_connection_cost_cad": result.candidate.estimated_connection_cost_cad,
        "suitability_score": result.candidate.suitability_score,
    }


def _service_route_payload_map(results):
    network_ids = {result.output_network_id for result in results if result.output_network_id}
    candidate_ids = {result.candidate_id for result in results if result.candidate_id}
    if not network_ids or not candidate_ids:
        return {}

    payloads = {}
    routes = (
        Route.objects
        .filter(edge__network_id__in=network_ids, edge__other_information__network_phasing_service=True)
        .select_related("edge")
        .order_by("id")
    )
    for route in routes:
        info = route.edge.other_information if isinstance(route.edge.other_information, dict) else {}
        candidate_id = _safe_int(info.get("candidate_id"))
        if candidate_id not in candidate_ids:
            continue
        service_path = _route_data_path({"polyline": route.polyline})
        if len(service_path) < 2:
            continue
        key = (route.edge.network_id, candidate_id)
        payload = {
            "service_path": service_path,
            "service_path_is_routed": len(service_path) > 2,
            "service_route_distance_m": float(route.distance_meters or route.edge.length or 0.0),
            "route_travel_mode": info.get("route_travel_mode"),
            "route_resolved_mode": info.get("route_resolved_mode") or route.route_type,
            "output_service_route_id": route.id,
            "output_service_edge_id": route.edge_id,
        }
        existing = payloads.get(key)
        if not existing or len(payload["service_path"]) > len(existing.get("service_path", [])):
            payloads[key] = payload
    return payloads


def _service_route_payload_for_result(result, service_route_map=None):
    if not service_route_map or not result.output_network_id:
        return None
    return service_route_map.get((result.output_network_id, result.candidate_id))


def _preview_result_payload(result, candidates_by_key):
    row = candidates_by_key.get(result["candidate_key"], {})
    return {
        "id": f"{result['candidate_key']}:{result['year']}:{result['scenario_name']}",
        "candidate_key": result["candidate_key"],
        "year": result["year"],
        "period_label": str(result["year"]),
        "scenario": result["scenario_name"],
        "candidate_id": row.get("building_id") or row.get("cluster_id"),
        "candidate_type": row.get("candidate_type"),
        "candidate_name": row.get("name"),
        "candidate_role": row.get("candidate_role"),
        "building_count": row.get("building_count", 1 if row.get("candidate_type") == CandidateConnection.CandidateType.BUILDING else 0),
        "node_connected": result["node_connected"],
        "node_newly_connected": result["node_newly_connected"],
        "edge_exists": result["edge_exists"],
        "edge_newly_built": result["edge_newly_built"],
        "plant_exists": result["plant_exists"],
        "heat_supplied_kw": result["heat_supplied_kw"],
        "capex_cad": result["capex_cad"],
        "annual_value_cad": result.get("annual_value_cad", 0.0),
        "annual_value_breakdown": result.get("annual_value_breakdown", {}),
        "discounted_net_benefit_cad": result["discounted_net_benefit_cad"],
        "cumulative_heat_kw": result["cumulative_heat_kw"],
        "cumulative_capex_cad": result["cumulative_capex_cad"],
        "output_network_id": None,
        "reason": result["reason"],
        "service_path": row.get("service_path", []),
        "candidate_point": _candidate_point_from_path(row.get("service_path")),
        "connection_length_m": row.get("connection_length_m"),
        "linear_heat_density_mwh_per_m_year": row.get("linear_heat_density_mwh_per_m_year"),
        "estimated_connection_cost_cad": row.get("estimated_connection_cost_cad"),
        "suitability_score": row.get("suitability_score"),
    }


def _period_payload(period, limits=None):
    limits = limits if isinstance(limits, dict) else {}
    return {
        "id": period.id,
        "sequence": period.sequence,
        "year": period.year,
        "label": period.label,
        "available_heat_kw": period.available_heat_kw,
        "capex_budget_cad": period.capex_budget_cad,
        "max_new_connections": _int_value(limits.get("max_new_connections"), None),
    }


def _scenario_payload(scenario):
    return {
        "id": scenario.id,
        "name": scenario.name,
        "probability": scenario.probability,
        "discount_rate": scenario.discount_rate,
        "demand_multiplier": scenario.demand_multiplier,
        "fuel_price_multiplier": scenario.fuel_price_multiplier,
        "uncertainty_revealed_year": scenario.uncertainty_revealed_year,
    }


def _chart_payload(timeline):
    return {
        "years": [row["year"] for row in timeline],
        "new_connections": [row["new_connections"] for row in timeline],
        "added_heat_kw": [row["added_heat_kw"] for row in timeline],
        "capex_cad": [row["capex_cad"] for row in timeline],
        "cumulative_capex_cad": [row["cumulative_capex_cad"] for row in timeline],
        "cumulative_heat_kw": [row.get("cumulative_heat_kw", 0.0) for row in timeline],
    }


def _map_payload(base_network, results):
    selected = [result for result in results if result.get("node_newly_connected")]
    return _preview_map_payload(base_network, [], {}, selected_results=selected)


def _preview_map_payload(base_network, candidate_rows, selected_keys, selected_results=None):
    routes = []
    from api.utils.network_versioning import decode_polyline
    for route in _source_routes(base_network):
        try:
            points = decode_polyline(route.polyline)
        except Exception:
            points = []
        routes.append({
            "id": route.id,
            "route_type": route.route_type,
            "path": [{"lat": lat, "lng": lng} for lat, lng in points],
        })
    candidate_paths = []
    if selected_results is not None:
        for result in selected_results:
            candidate_paths.append({
                "key": f"result:{result.get('id')}",
                "name": result.get("candidate_name"),
                "year": result.get("year"),
                "scenario": result.get("scenario"),
                "selected": True,
                "role": result.get("candidate_role"),
                "candidate_type": result.get("candidate_type"),
                "candidate_point": result.get("candidate_point"),
                "service_path": result.get("service_path") or [],
                "score": result.get("suitability_score"),
            })
    else:
        for row in candidate_rows:
            candidate_paths.append({
                "key": row["key"],
                "name": row["name"],
                "year": selected_keys.get(row["key"]),
                "selected": row["key"] in selected_keys,
                "role": row["candidate_role"],
                "candidate_type": row.get("candidate_type"),
                "candidate_point": _candidate_point_from_path(row.get("service_path")),
                "service_path": row.get("service_path", []),
                "score": row.get("suitability_score"),
                "status": row.get("status"),
            })
    return {"routes": routes, "candidate_paths": candidate_paths}


def _candidate_point_from_path(service_path):
    if not service_path:
        return None
    first = service_path[-1]
    try:
        lat = float(first.get("lat"))
        lng = float(first.get("lng"))
    except (AttributeError, TypeError, ValueError):
        return None
    return {"lat": lat, "lng": lng}


def _unique_candidate_results(results):
    seen = set()
    unique = []
    for result in results:
        if result.candidate_id in seen:
            continue
        seen.add(result.candidate_id)
        unique.append(result)
    return unique


def _network_payload(network):
    return {
        "id": network.id,
        "name": network.name,
        "project_id": network.project_id,
        "network_type": network.network_type,
        "total_length": network.total_length,
    }


def _normalize_periods(raw):
    if raw in [None, "", []]:
        raw = DEFAULT_PERIODS
    if isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    periods = []
    for index, item in enumerate(raw, start=1):
        if isinstance(item, dict):
            year = _int_value(item.get("year"), None)
            if year is None:
                continue
            periods.append({
                "year": year,
                "label": item.get("label") or f"Phase {index} ({year})",
                "available_heat_kw": _float_value(item.get("available_heat_kw"), None),
                "capex_budget_cad": _float_value(item.get("capex_budget_cad"), None),
                "max_new_connections": _positive_int_or_none(
                    item.get("max_new_connections") or
                    item.get("max_connections") or
                    item.get("connection_cap")
                ),
            })
        else:
            year = _int_value(item, None)
            if year is not None:
                periods.append({"year": year, "label": f"Phase {index} ({year})"})
    unique = {period["year"]: period for period in periods}
    return [unique[year] for year in sorted(unique)]


def _normalize_scenarios(raw):
    if raw in [None, "", []]:
        raw = DEFAULT_SCENARIOS
    scenarios = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        scenarios.append({
            "name": item.get("name") or f"Scenario {len(scenarios) + 1}",
            "probability": max(_float_value(item.get("probability"), 0.0), 0.0),
            "discount_rate": _float_value(item.get("discount_rate"), 0.045),
            "demand_multiplier": _float_value(item.get("demand_multiplier"), 1.0),
            "fuel_price_multiplier": _float_value(item.get("fuel_price_multiplier"), 1.0),
            "uncertainty_revealed_year": _int_value(item.get("uncertainty_revealed_year"), None),
        })
    total = sum(item["probability"] for item in scenarios)
    if total <= 0:
        return copy.deepcopy(DEFAULT_SCENARIOS)
    for item in scenarios:
        item["probability"] = item["probability"] / total
    return scenarios


def _normalize_weights(raw):
    weights = dict(DEFAULT_SCORE_WEIGHTS)
    if isinstance(raw, dict):
        for key in weights:
            if key in raw:
                weights[key] = max(_float_value(raw.get(key), weights[key]), 0.0)
    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_SCORE_WEIGHTS)
    return {key: value / total for key, value in weights.items()}


def _normalize_candidate_source_mode(raw):
    value = str(raw or "buildings_and_clusters").strip().lower().replace("-", "_")
    aliases = {
        "all": "buildings_and_clusters",
        "auto": "buildings_and_clusters",
        "both": "buildings_and_clusters",
        "building": "buildings_only",
        "buildings": "buildings_only",
        "buildings_only": "buildings_only",
        "cluster": "clusters_only",
        "clusters": "clusters_only",
        "clusters_only": "clusters_only",
    }
    return aliases.get(value, "buildings_and_clusters")


def _normalize_solver_mode(raw):
    value = str(raw or "greedy").strip().lower().replace("-", "_")
    aliases = {
        "baseline": "greedy",
        "screening": "greedy",
        "screening_baseline": "greedy",
        "greedy": "greedy",
        "deterministic_greedy": "greedy",
        "deterministic_greedy_expected_value": "greedy",
        "optimizer": "milp",
        "optimiser": "milp",
        "milp": "milp",
        "milp_optimizer": "milp",
        "mixed_integer": "milp",
        "mixed_integer_linear_program": "milp",
        "chronology": FULL_CHRONOLOGY_SOLVER_MODE,
        "full_chronology": FULL_CHRONOLOGY_SOLVER_MODE,
        "chronology_milp": FULL_CHRONOLOGY_SOLVER_MODE,
        "full_chronology_milp": FULL_CHRONOLOGY_SOLVER_MODE,
        "paper_chronology": FULL_CHRONOLOGY_SOLVER_MODE,
        "paper_chronology_milp": FULL_CHRONOLOGY_SOLVER_MODE,
    }
    return aliases.get(value, "greedy")


def _normalize_scenario_decision_mode(raw):
    value = str(raw or EXPECTED_VALUE_DECISION_MODE).strip().lower().replace("-", "_")
    aliases = {
        "expected": EXPECTED_VALUE_DECISION_MODE,
        "expected_value": EXPECTED_VALUE_DECISION_MODE,
        "shared": EXPECTED_VALUE_DECISION_MODE,
        "shared_plan": EXPECTED_VALUE_DECISION_MODE,
        "stochastic_expected": EXPECTED_VALUE_DECISION_MODE,
        "scenario": SCENARIO_SPECIFIC_DECISION_MODE,
        "scenario_specific": SCENARIO_SPECIFIC_DECISION_MODE,
        "per_scenario": SCENARIO_SPECIFIC_DECISION_MODE,
        "scenario_by_scenario": SCENARIO_SPECIFIC_DECISION_MODE,
    }
    return aliases.get(value, EXPECTED_VALUE_DECISION_MODE)


def _normalized(rows, getter, *, higher=True, log_scale=False):
    values = []
    for row in rows:
        value = max(_float_value(getter(row), 0.0), 0.0)
        values.append(math.log1p(value) if log_scale else value)
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        score = 1.0 if high > 0 else 0.0
        return [score for _ in values]
    scores = [(value - low) / (high - low) for value in values]
    if not higher:
        scores = [1.0 - value for value in scores]
    return [max(0.0, min(1.0, value)) for value in scores]


def _list_value(raw):
    if raw in [None, "", "all", "null"]:
        return []
    if isinstance(raw, str):
        return [item.strip() for item in raw.split(",") if item.strip()]
    if not isinstance(raw, (list, tuple, set)):
        raw = [raw]
    return [item for item in raw if item not in [None, ""]]


def _bool_value(raw, default=False):
    if raw in [None, ""]:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).lower() in {"1", "true", "yes", "on"}


def _float_value(raw, default):
    if raw in [None, ""]:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _int_value(raw, default):
    if raw in [None, ""]:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _positive_int_or_none(raw):
    value = _int_value(raw, None)
    if value is None or value < 1:
        return None
    return value


def _utm_epsg(lng, lat):
    zone = int((float(lng) + 180) // 6) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone
