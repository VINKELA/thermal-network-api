from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views
from .views import CreateNetworkView, NetworkDetailView

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

urlpatterns = [
    path('', include(router.urls)),
 path('projects/<int:project_id>/create-network/', CreateNetworkView.as_view(), name='create-network'),
     path('networks/nodes/<int:pk>/', NetworkDetailView.as_view(), name='network-detail'),

]