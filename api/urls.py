from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .views import AlgorithmByNetworkDetailView, CreateNetworkView, EdgeTranslationAPIView, NetworkByTranslationView, NetworkDetailView, NetworkTranslationDetailView, TranslationsByAlgorithmDetailView

router = DefaultRouter()
router.register(r'countries', views.CountryViewSet)
router.register(r'provinces', views.ProvinceViewSet)
router.register(r'cities', views.CityViewSet)
router.register(r'projects', views.ProjectViewSet)
router.register(r'networks', views.NetworkViewSet)
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
router.register(r'edge_translations', views.EdgeTranslationViewSet)


urlpatterns = [
    path('', include(router.urls)),
    path('projects/<int:project_id>/create-network/', CreateNetworkView.as_view(), name='create-network'),
    path('networks/nodes/<int:pk>/', NetworkDetailView.as_view(), name='network-detail'),
    path('networks/<int:network_id>/connect_edges/', views.connect_network_edges, name='connect_network_edges'),
    path('edge-translations/', EdgeTranslationAPIView.as_view(), name='edge-translations'),
    path(
        'edge_algorithms/<int:edge_algorithm_id>/translations/<int:translation_id>/',
        NetworkTranslationDetailView.as_view(),
        name='network-by-translation'
    ),
     path(
        'getAlgorithmByNetworkID/<int:network_id>/',
        AlgorithmByNetworkDetailView.as_view(),
        name='getAlgorithmByNetworkID'
    ),
    path(
        'getTranslationByAlgorithmID/<int:algorithm_id>/',
        TranslationsByAlgorithmDetailView.as_view(),
        name='getTranslationByAlgorithmID'
    ),
]