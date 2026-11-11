from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .views import *
from .heating_profile_views import HeatingModelCatalogView, HeatingProfileComparisonView, ApplyHeatingProfileView, HeatingStudyBuildingsView
from .heating_sensitivity_views import PipeSensitivityStartView, PipeSensitivityStatusView
from .break_even_views import BreakEvenStartView, BreakEvenStatusView
router = DefaultRouter()
router.register(r'countries', views.CountryViewSet)
router.register(r'provinces', views.ProvinceViewSet)
router.register(r'cities', views.CityViewSet)
router.register(r'projects', views.ProjectViewSet)
router.register(r'project-memberships', views.ProjectMembershipViewSet, basename='project-memberships')
router.register(r'networks', views.NetworkViewSet)
router.register(r'clusters', ClusterViewSet, basename='clusters')
router.register(r'buildings', views.BuildingViewSet)
router.register(r'nodes', views.NodeViewSet)
router.register(r'edges', views.EdgeViewSet)


router.register(r'pumps', views.PumpViewSet)
router.register(r'valves', views.ValveViewSet)
router.register(r'heatpumps', views.HeatPumpViewSet)
router.register(r'heatexchangers', views.HeatExchangerViewSet)
router.register(r'pipes', views.PipeViewSet)
router.register(r'junctions', views.JunctionViewSet)
router.register(r'pvs', views.PVSystemViewSet)
router.register(r'geothermal', views.GeothermalSystemViewSet)
router.register(r'boilers', views.BoilerViewSet)
router.register(r'thermalstorages', views.ThermalStorageViewSet)
router.register(r'sensors', views.SensorViewSet)
router.register(r'meters', views.MeterViewSet)
router.register(r'routes', views.RouteViewSet)
router.register(r'edge_algorithms', views.EdgeAlgorithmViewSet)
router.register(r'edge_connection_algorithms', views.EdgeConnectionAlgorithmViewSet)
router.register(r'edge_translations', views.EdgeTranslationViewSet)
router.register(r'cluster_configurations', views.ClusterConfigurationViewSet)
router.register(r'cluster_algorithms', views.ClusterAlgorithmViewSet)
router.register(r'pipe_sizings', views.PipeSizingViewSet)
router.register(r'edge_pipe_sizings', views.EdgePipeSizingViewSet, basename='edge_pipe_sizing')
router.register(r'network_design_constants', views.NetworkDesignConstantsViewSet, basename='network_design_constants')
router.register(r'carbon-tax-rates', views.CarbonTaxRateViewSet, basename='carbon_tax_rates')
router.register(r'ontario-natural-gas-rates', views.OntarioNaturalGasRateViewSet, basename='ontario_natural_gas_rates')
router.register(r'phasing-studies', views.PhasingStudyViewSet, basename='phasing-studies')

from .imported_heating_views import ImportedHeatingView

urlpatterns = [
    path('break-even/', BreakEvenStartView.as_view(), name='break-even-start'),
    path('break-even/<uuid:run_id>/', BreakEvenStatusView.as_view(), name='break-even-status'),
    path('imported-heating/', ImportedHeatingView.as_view()),
    path("pipe-sensitivity/", PipeSensitivityStartView.as_view()),
    path("pipe-sensitivity/<uuid:run_id>/", PipeSensitivityStatusView.as_view()),
    path('heating-models/', HeatingModelCatalogView.as_view(), name='heating-models'),
    path('heating-study-buildings/', HeatingStudyBuildingsView.as_view(), name='heating-study-buildings'),
    path('heating-profile-comparison/', HeatingProfileComparisonView.as_view(), name='heating-profile-comparison'),
    path('heating-profile-apply/', ApplyHeatingProfileView.as_view(), name='heating-profile-apply'),
    path('auth/register/', RegisterView.as_view(), name='auth-register'),
    path('auth/login/', LoginView.as_view(), name='auth-login'),
    path('auth/logout/', LogoutView.as_view(), name='auth-logout'),
    path('auth/me/', MeView.as_view(), name='auth-me'),
    path('auth/password-reset/', PasswordResetRequestView.as_view(), name='auth-password-reset'),
    path('auth/password-reset-confirm/', PasswordResetConfirmView.as_view(), name='auth-password-reset-confirm'),
    path('auth/change-password/', ChangePasswordView.as_view(), name='auth-change-password'),
    path('projects/generate_profiles/', views.trigger_profile_generation),
    path('', include(router.urls)),
    path('projects/<int:project_id>/create-network/', CreateNetworkView.as_view(), name='create-network'),
    path('networks/nodes/<int:pk>/', NetworkDetailView.as_view(), name='network-detail'),
    path('networks/<int:network_id>/connect_edges/', views.connect_network_edges, name='connect_network_edges'),
    path('networks/<int:network_id>/cost-summary/', NetworkCostSummaryView.as_view(), name='network-cost-summary'),
    path('networks/<int:network_id>/emissions-summary/', NetworkEmissionsSummaryView.as_view(), name='network-emissions-summary'),
    path('networks/<int:network_id>/analysis/', NetworkAnalysisView.as_view(), name='network-analysis'),
    path('networks/<int:network_id>/demand-map/', NetworkDemandMapView.as_view(), name='network-demand-map'),
    path('networks/<int:network_id>/expansion-candidates/', NetworkExpansionCandidatesView.as_view(), name='network-expansion-candidates'),
    path('networks/<int:network_id>/expand-from-routes/', NetworkRouteExpansionView.as_view(), name='network-expand-from-routes'),
    path('networks/<int:network_id>/phasing-preview/', NetworkPhasingPreviewView.as_view(), name='network-phasing-preview'),
    path('networks/<int:network_id>/phasing-run/', NetworkPhasingRunView.as_view(), name='network-phasing-run'),
    path('clusters/<int:cluster_id>/get_cluster_networks/', views.get_cluster_data, name='get_cluster_data'),
    path('projects/<int:project_id>/cluster_project_buildings/', views.cluster_project_buildings, name='cluster_project_buildings'), # type: ignore
 
    path('networks/<int:network_id>/cluster_nodes/', views.cluster_network_nodes, name='cluster_network_nodes'), # type: ignore
    path('projects/<int:project_id>/run_genetic_clustering/', views.run_genetic_clustering, name='run_genetic_clustering'), # type: ignore
    path('upload/<str:upload_type>/', BulkUploadView.as_view(), name='bulk-upload'),
    path('projects/<int:conn_id>/sizing/', CalculatePipeSizingView.as_view(), name='calculate-pipe-sizing'), # backwards-compatible legacy path
    path('edge_connection_algorithms/<int:conn_id>/sizing/', CalculatePipeSizingView.as_view(), name='calculate-pipe-sizing-from-connection'),
    path('edge_translations/<int:translation_id>/sizing/', CalculatePipeSizingFromRouteView.as_view(), name='calculate-pipe-sizing-from-route'),
    path('project/cluster/', SourceAnchoredClustering.as_view(), name='project-clustering'),
    path('network/create-from-cluster/', CreateNetworkFromClusterView.as_view(), name='create-network-from-cluster'),
    path('config/balance/', BalanceClusterView.as_view(), name='balance-clusters'),
    path('tasks/<str:task_id>/status/', views.check_task_status, name='task-status'),
    path('optimize/start/', views.trigger_optimization, name='trigger_optimization'),
    path('optimize/sapi/edge-translationstatus/<str:task_id>/', views.check_optimization_status, name='check_optimization_status'),
    path('edge-translation/<int:algorithm_connection_id>/', views.edge_translation, name='edge-translation'),
    # Sidebar Directory
    path('network-entities/', NetworkEntityDirectoryView.as_view(), name='network-entities'),
    path('load-profiles/', IndividualLoadProfileView.as_view(), name='individual-load-profiles'),
    path('edge-algorithms/', EdgeAlgorithmChoicesView.as_view(), name='edge-algorithms'),
    path('algorithms/choices/', EdgeAlgorithmChoicesView.as_view(), name='edge-algorithm-choices'),
    # Notice this is no longer tied to a specific network ID in the URL path
    path('aggregate-profile/', SystemAggregateProfileView.as_view(), name='system-aggregate-profile'),
    path('heat-utilization/', AnnualHeatUtilizationView.as_view(), name='annual-heat-utilization'),
    ]
