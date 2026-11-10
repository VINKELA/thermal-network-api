from celery import shared_task
from django.db import transaction
from django.db.models import Q
from .models import Building, Project, LoadProfile, Network, Node
from .intelligence.predictors import BuildingEstimator
from .intelligence.waste_heat_estimator import WasteHeatEstimator
import calendar
from datetime import datetime, timedelta
# DELETE or CHANGE this line if you have it: from datetime import timezone 
from .utils.clustering_algorithms import *
from .utils.genetic_clustering import *
# USE THIS INSTEAD:
from django.utils import timezone 
import traceback
from .utils.clustering_logic import run_cluster_algorithm, create_network_from_clusters2
from api.utils.clustering_algorithms import *
from .utils.genetic_clustering import run_experiment_combinations

def _resolve_generation_scope(project_id, network_id=None):
    project = Project.objects.get(id=project_id)
    network = None
    buildings = project.building_set.all()

    if network_id:
        network = Network.objects.select_related('project').get(id=network_id, project_id=project_id)
        building_ids = Node.objects.filter(
            network=network,
            building__isnull=False,
        ).values_list('building_id', flat=True).distinct()
        buildings = Building.objects.filter(project=project, id__in=building_ids)

    return project, network, buildings


@shared_task
def generate_waste_heat_profiles(project_id, year=2024, network_id=None):
    year = int(year)
    project, network, scoped_buildings = _resolve_generation_scope(project_id, network_id)
    sources = scoped_buildings.exclude(waste_heat_source_type__isnull=True).exclude(waste_heat_source_type='')
    
    # 1. Single Bulk Delete (Highly Optimized)
    waste_heat_delete_query = LoadProfile.objects.filter(
        building__in=sources, 
        profile_type='waste_heat', # Corrected to match database schema
        start_date__year=year
    )
    if network:
        waste_heat_delete_query = waste_heat_delete_query.filter(Q(network=network) | Q(network__isnull=True))
    else:
        waste_heat_delete_query = waste_heat_delete_query.filter(network__isnull=True)
    waste_heat_delete_query.delete()

    total_created = 0

    for building in sources:
        batch_objects = []
        monthly_expected_heat_kwh = {
    1: 14880000,  # Jan: 31 days (744 hours) * 20,000 kW
    2: 13440000,  # Feb: 28 days (672 hours) * 20,000 kW
    3: 14880000,  # Mar: 31 days (744 hours) * 20,000 kW
    4: 14400000,  # Apr: 30 days (720 hours) * 20,000 kW
    5: 14880000,  # May: 31 days (744 hours) * 20,000 kW
    6: 14400000,  # Jun: 30 days (720 hours) * 20,000 kW
    7: 14880000,  # Jul: 31 days (744 hours) * 20,000 kW
    8: 14880000,  # Aug: 31 days (744 hours) * 20,000 kW
    9: 14400000,  # Sep: 30 days (720 hours) * 20,000 kW
    10: 14880000, # Oct: 31 days (744 hours) * 20,000 kW
    11: 14400000, # Nov: 30 days (720 hours) * 20,000 kW
    12: 14880000  # Dec: 31 days (744 hours) * 20,000 kW
}
        for month in range(1, 13):
            monthly_total_kwh = monthly_expected_heat_kwh.get(month)
            if monthly_total_kwh <= 0:
                continue

            _, days_in_month = calendar.monthrange(year, month)
            hours_in_month = days_in_month * 24
            avg_load_kw = monthly_total_kwh / hours_in_month
            
            # Start time generation
            curr = timezone.make_aware(datetime(year, month, 1, 0, 0, 0))
            
            while curr.month == month:
                batch_objects.append(LoadProfile(
                    building=building,
                    network=network,
                    profile_type='waste_heat', # Corrected
                    value=round(avg_load_kw, 4),
                    unit='kW',
                    start_date=curr,
                    end_date=curr + timedelta(hours=1),
                    source_model='albert_thesis_v1'
                ))
                curr += timedelta(hours=1)
        
        # 2. Bulk Create per building using native batch_size
        if batch_objects:
            with transaction.atomic():
                LoadProfile.objects.bulk_create(batch_objects, batch_size=5000)
            total_created += len(batch_objects)

    scope_name = f"network {network.id}" if network else f"project {project.id}"
    return f"Generated {total_created} hourly waste heat records for {sources.count()} sources in {scope_name}."

@shared_task
def generate_profiles_for_project(project_id, model_type='hdd_daily_ca_v1', year=2024, network_id=None, model_options=None):
    """Generate the selected model, preserving previous active profiles in an archive."""
    from .intelligence.heating_studies import HeatingStudyRequest, run_heating_study, persist_selected_profile
    from .intelligence.heating_models import MODEL_IDS
    import numpy as np

    available = {m['id'] for m in BuildingEstimator.get_available_models()}
    if model_type not in available:
        raise ValueError(f'Unknown model: {model_type}')
    selected = model_type if model_type in MODEL_IDS else 'hdd_daily_ca_v1'
    payload = dict(model_options or {})
    payload.update(project_id=project_id, network_id=network_id, year=year,
                   models=[selected], reference=selected)
    serializer = HeatingStudyRequest(data=payload)
    serializer.is_valid(raise_exception=True)
    from .intelligence.heating_studies import consumer_building_queryset
    from .intelligence.heating_weather import fetch_canadian_weather

    options = serializer.validated_data
    project, network, buildings = consumer_building_queryset(options)
    if options.get('building_ids'):
        buildings = buildings.filter(id__in=options['building_ids'])
        if buildings.count() != len(set(options['building_ids'])):
            raise ValueError('Selected buildings are outside the consumer scope.')
    building_ids = list(buildings.order_by('id').values_list('id', flat=True))
    if not building_ids:
        raise ValueError('No consumer buildings in this selection.')
    weather = fetch_canadian_weather(options['year'], options['location'])
    created, archives = 0, []
    # Bound numerical memory for large projects. A single transaction prevents partial activation.
    with transaction.atomic():
        Project.objects.select_for_update().get(pk=project.pk)
        for offset in range(0, len(building_ids), 25):
            batch_options = {**options, 'building_ids': building_ids[offset:offset + 25]}
            response, matrices = run_heating_study(batch_options, weather=weather)
            if model_type in MODEL_IDS:
                matrix = matrices[model_type]
            else:
                matrix = np.array([[row['value'] for row in BuildingEstimator.generate_heating_profile(
                    b['annual_kwh'], model_type, int(year), b['building_type'])]
                    for b in response['building_inputs']])
                if matrix.shape[1] != len(response['timestamps']):
                    raise ValueError('Legacy model returned an incomplete year.')
                response['legacy_model'] = model_type
                response['results'] = []
            result = persist_selected_profile(response, matrix, model_type)
            created += result['created']
            archives.append(result['archive_id'])
    return f"Generated {created} hourly records with {model_type}; archives {', '.join(archives)}."

@shared_task(bind=True)
def run_genetic_clustering_task(self, project_id):
    """
    Celery task to run the full genetic clustering experiment asynchronously.
    """
    try:
        # 1. Lazy load models to avoid import cycles
        Project = apps.get_model('api', 'Project')
        ClusterAlgorithm = apps.get_model('api', 'ClusterAlgorithm')
        
        print(f"[{self.request.id}] Starting Genetic Experiment for Project {project_id}")

        # 2. Fetch Project
        project = Project.objects.get(pk=project_id)

        # 3. Initial Clustering (OPTICS)
        # Note: Depending on your implementation, this might return raw data or objects.
        # Ensure create_network_from_clusters2 returns the 'Network' instance needed for the experiment.
        clusters = run_cluster_algorithm(project.id, ClusterAlgorithm.ClusterAlgorithmType.OPTICS, {})
        
        # This creates the 'initial_clusters' network instance used as the base
        network_instance = create_network_from_clusters2(project, clusters, ClusterAlgorithm.ClusterAlgorithmType.OPTICS, {})

        # 4. Run the Exhaustive Experiment
        # This calls the function we defined in the previous step
        results = run_experiment_combinations(network_instance)

        print(f"[{self.request.id}] Experiment Completed. {len(results)} configurations saved.")
        
        return {
            'status': 'completed',
            'project_id': project_id,
            'configurations_created': len(results)
        }

    except Exception as e:
        # Log the full error to the console/file so you can debug async failures
        error_details = traceback.format_exc()
        print(f"[{self.request.id}] Task Failed:\n{error_details}")
        
        # Re-raise the exception so Celery marks the task as FAILURE
        raise e

import uuid
from celery import shared_task
from django.db import transaction
from .models import Project, Building, LoadProfile

@shared_task
def import_buildings_task(data_list, project_id):
    """
    Celery task to process building data asynchronously.
    """
    try:
        project = Project.objects.get(pk=project_id)
        created_count = 0
        
        # Helper to safely get values or None (handling None from JSON serialization)
        def get_val(row, col, default=None):
            val = row.get(col)
            return val if val is not None else default

        with transaction.atomic():
            for row in data_list:
                # Handle Boolean conversion
                raw_bool = str(get_val(row, 'is_year_round_operation', 'True')).lower()
                is_year_round = raw_bool in ['true', '1', 'yes', 'y']

                # Generate Name if missing
                name = get_val(row, 'name')
                if not name or str(name).strip() == '': 
                    name = f"Bldg_{uuid.uuid4().hex[:8]}"

                Building.objects.create(
                    project=project,
                    name=str(name),
                    latitude=row['latitude'],
                    longitude=row['longitude'],
                    
                    # Demographics
                    building_type=get_val(row, 'building_type', 'other'),
                    fsa=get_val(row, 'fsa'),
                    
                    # Loads
                    load_type=get_val(row, 'load_type', 'heating_dominant'),
                    floor_area=get_val(row, 'floor_area'),
                    heating_demand=get_val(row, 'heating_demand'),
                    cooling_demand=get_val(row, 'cooling_demand'),
                    height=get_val(row, 'height'),
                    number_of_floors=get_val(row, 'number_of_floors'),
                    
                    # Waste Heat
                    waste_heat_source_type=get_val(row, 'waste_heat_source_type'),
                    is_year_round_operation=is_year_round
                )
                created_count += 1
        
        return f"Successfully imported {created_count} buildings."

    except Project.DoesNotExist:
        return "Project not found."
    except Exception as e:
        return f"Error importing buildings: {str(e)}"

@shared_task
def import_profiles_task(data_list, project_id):
    """
    Celery task to process load profiles asynchronously.
    """
    created_count = 0
    errors = []
    
    for index, row in enumerate(data_list):
        building_name = row.get('building_name')
        try:
            # Case-insensitive match scoped to the selected project.
            building = Building.objects.get(project_id=project_id, name__iexact=str(building_name))
        except Building.DoesNotExist:
            errors.append(f"Row {index+1}: Building '{building_name}' not found.")
            continue
        except Building.MultipleObjectsReturned:
            errors.append(f"Row {index+1}: Building '{building_name}' is duplicated in this project.")
            continue
        
        try:
            # Create Heating Profile
            if row.get('heating') is not None and row['heating'] > 0:
                LoadProfile.objects.create(
                    building=building, 
                    profile_type='heating', 
                    unit='kW', 
                    value=row['heating'], 
                    start_date=row['start_date'], 
                    end_date=row['end_date']
                )
                created_count += 1
            
            # Create Cooling Profile
            if row.get('cooling') is not None and row['cooling'] > 0:
                LoadProfile.objects.create(
                    building=building, 
                    profile_type='cooling', 
                    unit='kW', 
                    value=row['cooling'], 
                    start_date=row['start_date'], 
                    end_date=row['end_date']
                )
                created_count += 1
        except Exception as e:
            errors.append(f"Row {index+1}: Error saving profile - {str(e)}")
            continue

    return f"Created {created_count} profiles. Errors: {len(errors)}"

    # api/tasks.py
import logging
from celery import shared_task
from .utils.thermal_optimization_service import ProjectThermalOrchestrator
from django.db.models import Sum
logger = logging.getLogger(__name__)


@shared_task(bind=True)
def run_network_route_expansion_task(self, network_id: int, params: dict, project_id=None):
    """
    Create a route-infill/evolved network in the background.

    Route tracing can call Google routing for many service connections, so it
    should not run inside the HTTP request/response cycle.
    """
    from .models import Network
    from .utils.network_expansion import create_route_infill_network

    try:
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "starting",
                "message": "Preparing network evolution.",
                "network_id": network_id,
            },
        )

        network = Network.objects.select_related("project").get(pk=network_id)
        if project_id is not None and int(network.project_id) != int(project_id):
            raise ValueError("Network does not belong to the selected project.")

        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "routing",
                "message": "Creating route infill network and routed service connections.",
                "network_id": network_id,
            },
        )

        result = create_route_infill_network(network, params or {})
        return {
            "status": "success",
            "task_id": self.request.id,
            "message": result.get("message"),
            "network_id": result.get("network_id"),
            "expansion_run_id": result.get("expansion_run_id"),
            "output_network": result.get("output_network"),
            "summary": result.get("summary"),
        }
    except Exception as exc:
        logger.exception("Network route expansion failed for network %s", network_id)
        return {
            "status": "failed",
            "network_id": network_id,
            "task_id": self.request.id,
            "error": str(exc),
        }


@shared_task(bind=True)
def run_network_phasing_task(self, network_id: int, params: dict, project_id=None):
    """
    Run a network phasing study in the background and create phase networks.
    """
    from .models import Network, PhasingStudy
    from .utils.network_phasing import create_network_phasing_study
    from .utils.network_phasing_pipeline import prepare_network_for_phasing

    study = None
    try:
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "starting",
                "message": "Preparing network phasing study.",
                "network_id": network_id,
            },
        )

        network = Network.objects.select_related("project").get(pk=network_id)
        if project_id is not None and int(network.project_id) != int(project_id):
            raise ValueError("Network does not belong to the selected project.")

        def notify(stage, message, extra=None):
            self.update_state(
                state="PROGRESS",
                meta={
                    "stage": stage,
                    "message": message,
                    "network_id": network.id,
                    **(extra or {}),
                },
            )

        prepared = prepare_network_for_phasing(
            network,
            params or {},
            progress_callback=notify,
        )
        phasing_network = prepared["network"]
        phasing_params = prepared["params"]

        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "solving",
                "message": "Scoring candidates, solving phase schedule, and creating phase networks.",
                "network_id": phasing_network.id,
                "input_network_id": network.id,
                "preparation_pipeline": prepared.get("pipeline"),
            },
        )

        result = create_network_phasing_study(phasing_network, phasing_params, task_id=self.request.id)
        return {
            "status": "success",
            "task_id": self.request.id,
            "message": "Network phasing study completed.",
            "study_id": result.get("id"),
            "base_network": result.get("base_network"),
            "timeline": result.get("timeline"),
            "charts": result.get("charts"),
            "preparation_pipeline": prepared.get("pipeline"),
        }
    except Exception as exc:
        logger.exception("Network phasing failed for network %s", network_id)
        if study is not None:
            study.status = PhasingStudy.Status.FAILED
            study.error = str(exc)
            study.save(update_fields=["status", "error", "updated_at"])
        return {
            "status": "failed",
            "network_id": network_id,
            "task_id": self.request.id,
            "error": str(exc),
        }


@shared_task(bind=True)
def run_route_tracing_task(self, edge_connection_algorithm_id: int, translation_type="driving", name=None, parameters=None, project_id=None):
    """
    Trace route geometry in the background and create a derived routed network.
    """
    from .models import EdgeConnectionAlgorithm
    from .utils.network_versioning import create_route_tracing_version

    try:
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "starting",
                "message": "Preparing route tracing.",
                "edge_connection_algorithm_id": edge_connection_algorithm_id,
            },
        )

        source_connection = (
            EdgeConnectionAlgorithm.objects
            .select_related("network", "network__project")
            .get(pk=edge_connection_algorithm_id)
        )
        if project_id is not None and int(source_connection.network.project_id) != int(project_id):
            raise ValueError("Connection algorithm does not belong to the selected project.")

        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "routing",
                "message": f"Tracing {translation_type or 'driving'} routes in the background.",
                "edge_connection_algorithm_id": edge_connection_algorithm_id,
            },
        )

        output_network, translation, routes = create_route_tracing_version(
            source_connection,
            translation_type or "driving",
            name=name,
            parameters=parameters or {},
        )
        return {
            "status": "success",
            "task_id": self.request.id,
            "message": f"Created route-traced network '{output_network.name}' using {translation_type or 'driving'}.",
            "source_network_id": source_connection.network_id,
            "network_id": output_network.id,
            "edge_connection_algorithm_id": source_connection.id,
            "edge_translation_id": translation.id,
            "routes_created": len(routes),
        }
    except Exception as exc:
        logger.exception("Route tracing failed for connection algorithm %s", edge_connection_algorithm_id)
        return {
            "status": "failed",
            "task_id": self.request.id,
            "edge_connection_algorithm_id": edge_connection_algorithm_id,
            "error": str(exc),
        }


@shared_task(bind=True)
def run_portfolio_optimization_task(self, project_id: int, carbon_steps: int = 10, time_steps: int = 168):
    """
    Background task to run the full thermal portfolio optimization.
    """
    logger.info(f"Starting async optimization for Project {project_id}")
    
    try:
        # Initialize the orchestrator
        orchestrator = ProjectThermalOrchestrator(project_id=project_id, time_steps=time_steps)
        
        # Run the math and generate the charts
        summary_report = orchestrator.run_portfolio_optimization(
            carbon_steps=carbon_steps, 
            base_output_dir=f"media/project_{project_id}_results" # Saving to Django's media folder
        )
        
        # Return the summary so it gets saved to the Celery backend (e.g., Redis/Database)
        return {
            "status": "success",
            "project_id": project_id,
            "summary": summary_report
        }
        
    except Exception as e:
        logger.error(f"Task failed for Project {project_id}: {str(e)}")
        return {
            "status": "failed",
            "project_id": project_id,
            "error": str(e)
        }
