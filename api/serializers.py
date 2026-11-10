from rest_framework import serializers
from statistics import mean
from django.contrib.auth import get_user_model
from .models import *
from .access_control import project_role_for_user
from .utils.route_geometry import route_physical_segments, unique_route_length

User = get_user_model()

# ==============================================================================
# 1. BASE & GEOGRAPHIC SERIALIZERS
# ==============================================================================

class LocationSerializer(serializers.ModelSerializer):    
    class Meta:
        model = Location
        fields = '__all__'
        abstract = True

class CountrySerializer(serializers.ModelSerializer):
    class Meta: model = Country; fields = '__all__'

class ProvinceSerializer(serializers.ModelSerializer):
    country = CountrySerializer()
    class Meta: model = Province; fields = '__all__'

class CitySerializer(serializers.ModelSerializer):
    province = ProvinceSerializer(required=False)
    country = CountrySerializer(required=False)
    class Meta: model = City; fields = '__all__'

class CarbonTaxRateSerializer(serializers.ModelSerializer):
    class Meta:
        model = CarbonTaxRate
        fields = ['id', 'year', 'price_cad_per_tonne']


class OntarioNaturalGasRateSerializer(serializers.ModelSerializer):
    class Meta:
        model = OntarioNaturalGasRate
        fields = [
            'id',
            'utility',
            'region',
            'effective_date',
            'year',
            'commodity_price_cents_per_m3',
            'gas_cost_adjustment_cents_per_m3',
            'effective_price_cents_per_m3',
            'cad_per_m3',
            'cad_per_mwh',
            'cad_per_mwh_useful_heat',
            'kwh_per_m3',
            'assumed_boiler_efficiency',
            'source_name',
            'source_url',
            'retrieved_date',
        ]


class UserSummarySerializer(serializers.ModelSerializer):
    display_name = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ['id', 'username', 'email', 'first_name', 'last_name', 'display_name', 'is_staff']

    def get_display_name(self, obj):
        full_name = obj.get_full_name()
        return full_name or obj.email or obj.username


class ProjectMembershipSerializer(serializers.ModelSerializer):
    user = UserSummarySerializer(read_only=True)
    invited_by = UserSummarySerializer(read_only=True)
    project_name = serializers.CharField(source='project.name', read_only=True)

    class Meta:
        model = ProjectMembership
        fields = ['id', 'project', 'project_name', 'user', 'role', 'invited_by', 'created_at', 'updated_at']
        read_only_fields = ['id', 'invited_by', 'created_at', 'updated_at']


class ProjectInviteSerializer(serializers.Serializer):
    project_id = serializers.IntegerField()
    email = serializers.EmailField()
    username = serializers.CharField(required=False, allow_blank=True)
    first_name = serializers.CharField(required=False, allow_blank=True)
    last_name = serializers.CharField(required=False, allow_blank=True)
    role = serializers.ChoiceField(choices=ProjectMembership.Role.choices, default=ProjectMembership.Role.VIEWER)


class RegisterSerializer(serializers.Serializer):
    username = serializers.CharField(required=False, allow_blank=True)
    email = serializers.EmailField()
    first_name = serializers.CharField(required=False, allow_blank=True)
    last_name = serializers.CharField(required=False, allow_blank=True)
    password = serializers.CharField(write_only=True, min_length=8)


class LoginSerializer(serializers.Serializer):
    username = serializers.CharField()
    password = serializers.CharField(write_only=True)


class PasswordResetRequestSerializer(serializers.Serializer):
    email = serializers.EmailField()


class PasswordResetConfirmSerializer(serializers.Serializer):
    uid = serializers.CharField()
    token = serializers.CharField()
    new_password = serializers.CharField(write_only=True, min_length=8)


class ChangePasswordSerializer(serializers.Serializer):
    old_password = serializers.CharField(write_only=True)
    new_password = serializers.CharField(write_only=True, min_length=8)


# ==============================================================================
# 2. CORE HELPERS (MUST BE DEFINED FIRST)
# ==============================================================================

class NodeMapSerializer(serializers.ModelSerializer):
    """
    Lightweight node info (ID + Coords).
    Defined at the top because it is used by Edges, Algorithms, and Networks.
    """
    class Meta:
        model = Node
        fields = ['id', 'name', 'latitude', 'longitude', 'node_type']

class NodeSerializer(LocationSerializer):
    """Full Node Info"""
    network = serializers.PrimaryKeyRelatedField(queryset=Network.objects.all())
    class Meta:
        model = Node
        fields = '__all__'

class EdgeSerializer(serializers.ModelSerializer):
    """
    Full Edge Info.
    MUST be defined before EdgeConnectionAlgorithmEdgeSerializer.
    """
    start_node = NodeMapSerializer(read_only=True)
    end_node = NodeMapSerializer(read_only=True)
    network = serializers.PrimaryKeyRelatedField(queryset=Network.objects.all())

    class Meta:
        model = Edge
        fields = '__all__'


def physical_route_payloads(routes):
    route_geometry = route_physical_segments(routes)
    payloads = []
    for index, entry in enumerate(route_geometry["entries"], start=1):
        route = entry.get("route")
        edge = entry.get("edge")
        payloads.append({
            "id": f"physical-segment-{index}",
            "representative_route_id": entry.get("route_id"),
            "representative_edge_id": edge.id if edge else None,
            "representative_edge_name": edge.name if edge else None,
            "route_type": getattr(route, "route_type", None),
            "distance_meters": entry.get("length_m", 0.0),
            "source_distance_meters": entry.get("source_length_m", 0.0),
            "source_route_ids": entry.get("source_route_ids", []),
            "source_edge_ids": entry.get("source_edge_ids", []),
            "source_node_ids": entry.get("source_node_ids", []),
            "overlap_count": entry.get("overlap_count", 1),
            "is_shared": entry.get("is_shared", False),
            "path": entry.get("path", []),
            "edge": route_edge_payload(edge),
        })
    return payloads


def route_display_payloads(routes):
    payloads = []
    for route in routes:
        edge = getattr(route, "edge", None)
        payloads.append({
            "id": route.id,
            "representative_route_id": route.id,
            "representative_edge_id": edge.id if edge else None,
            "representative_edge_name": edge.name if edge else None,
            "route_type": route.route_type,
            "polyline": route.polyline,
            "distance_meters": route.distance_meters,
            "duration_seconds": route.duration_seconds,
            "translation": route.translation_id,
            "translation_name": route.translation.name if route.translation else None,
            "source_route_ids": [route.id],
            "source_edge_ids": [route.edge_id] if route.edge_id else [],
            "source_node_ids": route_node_ids(route),
            "overlap_count": 1,
            "is_shared": False,
            "edge": route_edge_payload(edge),
        })
    return payloads


def route_node_ids(route):
    edge = getattr(route, "edge", None)
    return [
        node_id for node_id in [
            getattr(edge, "start_node_id", None),
            getattr(edge, "end_node_id", None),
        ]
        if node_id is not None
    ]


def route_edge_payload(edge):
    if not edge:
        return None
    return {
        "id": edge.id,
        "name": edge.name,
        "length": edge.length,
        "start_node": NodeMapSerializer(edge.start_node).data if edge.start_node else None,
        "end_node": NodeMapSerializer(edge.end_node).data if edge.end_node else None,
    }

class EdgeVisualizerSerializer(serializers.ModelSerializer):
    """Formatted specifically for Google Maps Polyline"""
    path = serializers.SerializerMethodField()
    class Meta:
        model = Edge
        fields = ['id', 'length', 'path']
    def get_path(self, obj):
        return [
            {'lat': obj.start_node.latitude, 'lng': obj.start_node.longitude},
            {'lat': obj.end_node.latitude, 'lng': obj.end_node.longitude}
        ]


# ==============================================================================
# 3. PROJECT & PROFILE
# ==============================================================================

class ProjectListSerializer(serializers.ModelSerializer):
    has_imported_hdd = serializers.SerializerMethodField()

    def get_has_imported_hdd(self, obj):
        return obj.imported_heating_shapes.exists()

    current_user_role = serializers.SerializerMethodField()
    member_count = serializers.IntegerField(source='memberships.count', read_only=True)

    class Meta:
        model = Project
        fields = ['id', 'name', 'city', 'country', 'created_at', 'current_user_role', 'member_count', 'has_imported_hdd']

    def get_current_user_role(self, obj):
        request = self.context.get('request')
        return project_role_for_user(getattr(request, 'user', None), obj.id) if request else None

class ProjectDetailSerializer(serializers.ModelSerializer):
    current_user_role = serializers.SerializerMethodField()
    member_count = serializers.IntegerField(source='memberships.count', read_only=True)

    class Meta:
        model = Project
        fields = '__all__'

    def get_current_user_role(self, obj):
        request = self.context.get('request')
        return project_role_for_user(getattr(request, 'user', None), obj.id) if request else None
class ProjectSerializer(serializers.ModelSerializer): # Legacy support
    class Meta: model = Project; fields = '__all__'

# api/serializers.py

class LoadProfileListSerializer(serializers.ModelSerializer):
    class Meta:
        model = LoadProfile
        # CRITICAL FIX: Added 'value' so the chart has Y-axis data
        fields = ['id', 'profile_type', 'unit', 'value', 'start_date', 'end_date']
class LoadProfileDetailSerializer(serializers.ModelSerializer):
    class Meta: model = LoadProfile; fields = '__all__'
class LoadProfileSerializer(serializers.ModelSerializer): # Legacy support
    class Meta: model = LoadProfile; fields = '__all__'


# ==============================================================================
# 4. NETWORK (DEPENDS ON NODES/EDGES)
# ==============================================================================

class NodeListSerializer(serializers.ModelSerializer):
    name = serializers.SerializerMethodField()
    building_type = serializers.SerializerMethodField()
    network_project = serializers.IntegerField(source='network.project_id', read_only=True)

    class Meta:
        model = Node
        fields = [
            'id', 'name', 'node_type', 'latitude', 'longitude',
            'network', 'network_project', 'building', 'boiler', 'building_type',
            'dea_technology_name',
        ]

    def get_name(self, obj):
        if obj.building:
            return obj.building.name or f"Building {obj.building_id}"
        if obj.boiler:
            return obj.boiler.name or f"Boiler {obj.boiler_id}"
        return obj.name or f"Node {obj.id}"

    def get_building_type(self, obj):
        return obj.building.building_type if obj.building else None
class NodeDetailSerializer(NodeSerializer): pass # Alias

class EdgeListSerializer(serializers.ModelSerializer):
    class Meta: model = Edge; fields = ['id', 'name', 'length', 'start_node', 'end_node']
class EdgeDetailSerializer(EdgeSerializer): pass # Alias

class NetworkListSerializer(serializers.ModelSerializer):
    project = ProjectListSerializer(read_only=True)
    project_id = serializers.IntegerField(read_only=True)
    parent_network_name = serializers.CharField(source='parent_network.name', read_only=True)
    route_count = serializers.SerializerMethodField()
    class Meta:
        model = Network
        fields = ['id', 'name', 'description', 'network_type', 
        'total_length', 'project', 'created_at', 'project_id',
        'parent_network', 'parent_network_name', 'derivation_stage',
        'derivation_method', 'route_count']

    def get_route_count(self, obj):
        return Route.objects.filter(edge__network=obj).count()

class NetworkDesignConstantsSerializer(serializers.ModelSerializer):
    class Meta:
        model = NetworkDesignConstants
        fields = '__all__'

class NetworkDetailSerializer(serializers.ModelSerializer):
    project = ProjectListSerializer(read_only=True)
    
    # Simple aggregations can often be done via source, but careful with N+1 issues
    node_count = serializers.IntegerField(source='node_set.count', read_only=True)
    edge_count = serializers.IntegerField(source='edge_set.count', read_only=True)
    route_count = serializers.SerializerMethodField()
    
    nodes = NodeSerializer(many=True, read_only=True, source='node_set')    
    edges = EdgeSerializer(many=True, read_only=True, source='edge_set')
    routes = serializers.SerializerMethodField()

    # Use SerializerMethodField for calculated values
    latitude = serializers.SerializerMethodField()
    longitude = serializers.SerializerMethodField()
    total_length = serializers.SerializerMethodField()

    class Meta: 
        model = Network 
        fields = '__all__'

    def get_latitude(self, obj):
        nodes = obj.node_set.all()
        # 1. Isolate only the valid numbers
        valid_lats = [node.latitude for node in nodes if node.latitude is not None]
        
        # 2. Return None if the list is completely empty
        if not valid_lats:
            return None 
            
        # 3. Safely calculate
        return mean(valid_lats)

    def get_longitude(self, obj):
        nodes = obj.node_set.all()
        valid_longs = [node.longitude for node in nodes if node.longitude is not None]
        
        if not valid_longs:
            return None
            
        return mean(valid_longs)

    def get_total_length(self, obj):
        routes = Route.objects.filter(edge__network=obj).select_related('edge')
        if routes.exists():
            return unique_route_length(routes)

        edges = obj.edge_set.all()
        # Added safety filter so sum() doesn't crash on a NoneType length
        return sum([edge.length for edge in edges if edge.length is not None])

    def get_route_count(self, obj):
        return Route.objects.filter(edge__network=obj).count()

    def get_routes(self, obj):
        routes = list(Route.objects.filter(edge__network=obj).select_related(
            'edge',
            'edge__start_node',
            'edge__end_node',
            'translation',
        ).order_by('id'))

        return route_display_payloads(routes)
class NetworkSerializer(serializers.ModelSerializer): # Legacy support
    project = ProjectSerializer(read_only=True)
    class Meta: model = Network; fields = '__all__'


class PlanningPeriodSerializer(serializers.ModelSerializer):
    class Meta:
        model = PlanningPeriod
        fields = ['id', 'study', 'sequence', 'year', 'label', 'available_heat_kw', 'capex_budget_cad']


class PlanningScenarioSerializer(serializers.ModelSerializer):
    class Meta:
        model = PlanningScenario
        fields = [
            'id', 'study', 'name', 'probability', 'discount_rate',
            'demand_multiplier', 'fuel_price_multiplier', 'uncertainty_revealed_year',
        ]


class CandidateConnectionSerializer(serializers.ModelSerializer):
    building_count = serializers.SerializerMethodField()

    class Meta:
        model = CandidateConnection
        fields = [
            'id', 'study', 'candidate_type', 'building', 'cluster', 'name',
            'candidate_role', 'available_from_year', 'peak_heat_kw', 'annual_heat_mwh',
            'available_heat_kw', 'building_count', 'distance_to_trunk_m', 'connection_length_m',
            'estimated_connection_cost_cad', 'linear_heat_density_mwh_per_m_year',
            'suitability_score', 'status', 'rejection_reason', 'snap_latitude',
            'snap_longitude', 'source_route', 'source_edge', 'metadata',
        ]

    def get_building_count(self, obj):
        metadata = obj.metadata if isinstance(obj.metadata, dict) else {}
        if obj.candidate_type == CandidateConnection.CandidateType.BUILDING:
            return 1
        if obj.cluster_id and obj.cluster and obj.cluster.no_of_buildings is not None:
            return obj.cluster.no_of_buildings
        return metadata.get('building_count', 0)


class PhasingResultSerializer(serializers.ModelSerializer):
    candidate_name = serializers.CharField(source='candidate.name', read_only=True)
    period_year = serializers.IntegerField(source='period.year', read_only=True)
    scenario_name = serializers.CharField(source='scenario.name', read_only=True)
    output_network_name = serializers.CharField(source='output_network.name', read_only=True)

    class Meta:
        model = PhasingResult
        fields = [
            'id', 'study', 'scenario', 'scenario_name', 'period', 'period_year',
            'candidate', 'candidate_name', 'output_network', 'output_network_name',
            'node_connected', 'node_newly_connected', 'edge_exists',
            'edge_newly_built', 'plant_exists', 'heat_supplied_kw',
            'capex_cad', 'discounted_net_benefit_cad', 'cumulative_heat_kw',
            'cumulative_capex_cad', 'reason', 'created_at',
        ]


class PhasingStudyListSerializer(serializers.ModelSerializer):
    base_network_name = serializers.CharField(source='base_network.name', read_only=True)
    project_name = serializers.CharField(source='project.name', read_only=True)
    period_count = serializers.IntegerField(source='periods.count', read_only=True)
    scenario_count = serializers.IntegerField(source='scenarios.count', read_only=True)
    candidate_count = serializers.IntegerField(source='candidates.count', read_only=True)

    class Meta:
        model = PhasingStudy
        fields = [
            'id', 'name', 'description', 'status', 'project', 'project_name',
            'base_network', 'base_network_name', 'solver', 'objective_value',
            'task_id', 'error', 'period_count', 'scenario_count',
            'candidate_count', 'created_at', 'updated_at',
        ]


class PhasingStudyDetailSerializer(PhasingStudyListSerializer):
    periods = PlanningPeriodSerializer(many=True, read_only=True)
    scenarios = PlanningScenarioSerializer(many=True, read_only=True)
    candidates = CandidateConnectionSerializer(many=True, read_only=True)

    class Meta(PhasingStudyListSerializer.Meta):
        fields = PhasingStudyListSerializer.Meta.fields + ['parameters', 'periods', 'scenarios', 'candidates']


# ==============================================================================
# 5. PHYSICAL COMPONENTS
# ==============================================================================

class BuildingListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Building
        fields = [
            'id', 'name', 'building_type', 'tag', 'fsa', 'latitude', 'longitude',
            'heating_demand', 'cooling_demand', 'floor_area', 'waste_heat_source_type',
        ]
class BuildingDetailSerializer(LocationSerializer):
    class Meta: model = Building; fields = '__all__'
class BuildingSerializer(LocationSerializer): # Legacy
    class Meta: model = Building; fields = '__all__'

class PumpSerializer(LocationSerializer):
    class Meta: model = Pump; fields = '__all__'
class ValveSerializer(LocationSerializer):
    class Meta: model = Valve; fields = '__all__'
class HeatPumpSerializer(LocationSerializer):
    class Meta: model = HeatPump; fields = '__all__'
class HeatExchangerSerializer(LocationSerializer):
    class Meta: model = HeatExchanger; fields = '__all__'
class PipeSerializer(LocationSerializer):
    class Meta: model = Pipe; fields = '__all__'
class JunctionSerializer(LocationSerializer):
    class Meta: model = Junction; fields = '__all__'
class PVSystemSerializer(LocationSerializer):
    class Meta: model = PVSystem; fields = '__all__'
class GeothermalSystemSerializer(LocationSerializer):
    class Meta: model = GeothermalSystem; fields = '__all__'
class BoilerSerializer(LocationSerializer):
    class Meta: model = Boiler; fields = '__all__'
class ThermalStorageSerializer(LocationSerializer):
    class Meta: model = ThermalStorage; fields = '__all__'
class SensorSerializer(LocationSerializer):
    class Meta: model = Sensor; fields = '__all__'
class MeterSerializer(LocationSerializer):
    class Meta: model = Meter; fields = '__all__'


# ==============================================================================
# 6. CLUSTERING ALGORITHMS
# ==============================================================================

class ClusterAlgorithmSummarySerializer(serializers.ModelSerializer):
    class Meta: model = ClusterAlgorithm; fields = ['id', 'name', 'algorithm', 'parameters']

class ClusterMapBuildingSerializer(serializers.ModelSerializer):
    class Meta: model = Building; fields = ['id', 'name', 'building_type', 'latitude', 'longitude']

class ConfigBuildingSerializer(serializers.ModelSerializer):
    class Meta:
        model = Building
        fields = ['id', 'name', 'latitude', 'longitude', 'building_type', 'load_type', 'waste_heat_source_type']

class ClusterListSerializer(serializers.ModelSerializer):
    network_name = serializers.CharField(source='network.name', read_only=True)
    cluster_configuration_name = serializers.CharField(source='cluster_configuration.name', read_only=True)
    project_id = serializers.SerializerMethodField()
    class Meta:
        model = Cluster
        fields = [
            'id', 'name', 'type', 'no_of_buildings', 'heating_demand', 'cooling_demand',
            'total_source_capacity', 'total_sink_capacity', 'energy_balance_status',
            'latitude', 'longitude', 'effective_radius', 'network', 'network_name',
            'cluster_algorithm', 'cluster_configuration', 'cluster_configuration_name',
            'project_id',
        ]

    def get_project_id(self, obj):
        if obj.network_id:
            return obj.network.project_id
        if obj.cluster_configuration_id and obj.cluster_configuration.cluster_algorithm_id:
            return obj.cluster_configuration.cluster_algorithm.project_id
        if obj.cluster_algorithm_id:
            return obj.cluster_algorithm.project_id
        return None

class ClusterDetailSerializer(serializers.ModelSerializer):
    buildings = serializers.SerializerMethodField()
    network = serializers.StringRelatedField(read_only=True)
    cluster_algorithm = ClusterAlgorithmSummarySerializer(read_only=True)
    class Meta: model = Cluster; fields = '__all__'
    def get_buildings(self, obj):
        cluster_links = BuildingCluster.objects.filter(cluster=obj).select_related('building')
        building_instances = [link.building for link in cluster_links]
        return ClusterMapBuildingSerializer(building_instances, many=True).data

class ClusterAlgorithmListSerializer(serializers.ModelSerializer):
    class Meta: model = ClusterAlgorithm; fields = ['id', 'name', 'algorithm', 'parameters', 'project']
class ClusterAlgorithmDetailSerializer(serializers.ModelSerializer):
    clusters = ClusterListSerializer(many=True, read_only=True, source='cluster_algorithm')
    class Meta: model = ClusterAlgorithm; fields = '__all__'
class ClusterAlgorithmSerializer(ClusterAlgorithmDetailSerializer): pass # Legacy

class ConfigClusterSerializer(serializers.ModelSerializer):
    buildings = serializers.SerializerMethodField()
    class Meta:
        model = Cluster
        fields = [
            'id', 'name', 'latitude', 'longitude', 'effective_radius',
            'no_of_buildings', 'heating_demand', 'cooling_demand',
            'total_source_capacity', 'total_sink_capacity',
            'buildings', 'energy_balance_status',
        ]
    def get_buildings(self, obj):
        links = BuildingCluster.objects.filter(cluster=obj).select_related('building')
        return ConfigBuildingSerializer([l.building for l in links], many=True).data
class ClusterConfigurationListSerializer(serializers.ModelSerializer):
    cluster_count = serializers.IntegerField(read_only=True)
    total_buildings = serializers.IntegerField(read_only=True)
    algorithm = serializers.CharField(source='cluster_algorithm.algorithm', read_only=True)
    project = serializers.IntegerField(source='cluster_algorithm.project_id', read_only=True)
    base_network_name = serializers.CharField(source='base_network.name', read_only=True)
    source_cluster_configuration_name = serializers.CharField(source='source_cluster_configuration.name', read_only=True)
    class Meta:
        model = ClusterConfiguration
        fields = [
            'id', 'name', 'description', 'parameters', 'cluster_algorithm', 'algorithm',
            'project', 'cluster_type', 'base_network', 'base_network_name',
            'source_cluster_configuration', 'source_cluster_configuration_name',
            'cluster_count', 'total_buildings',
        ]
class ClusterConfigurationDetailSerializer(serializers.ModelSerializer):
    clusters = ConfigClusterSerializer(many=True, read_only=True, source='cluster_configuration')
    base_network_name = serializers.CharField(source='base_network.name', read_only=True)
    source_cluster_configuration_name = serializers.CharField(source='source_cluster_configuration.name', read_only=True)
    class Meta: model = ClusterConfiguration; fields = '__all__'
class ClusterConfigurationSerializer(ClusterConfigurationDetailSerializer): pass # Legacy
class ConfigSerializer(serializers.ModelSerializer):
    class Meta: model = ClusterConfiguration; fields = '__all__'


# ==============================================================================
# 7. EDGE & CONNECTION ALGORITHMS (DEPENDS ON EDGE SERIALIZER)
# ==============================================================================

class EdgeAlgorithmListSerializer(serializers.ModelSerializer):
    
    network = NetworkListSerializer(read_only=True)
    source_network = NetworkListSerializer(read_only=True)
    
    class Meta: model = EdgeAlgorithm; fields = ['id', 'name', 'algorithm', 'total_edge_length', 'network', 'source_network']
    
class EdgeAlgorithmDetailSerializer(serializers.ModelSerializer):
    edges = EdgeVisualizerSerializer(many=True, read_only=True)
    nodes = serializers.SerializerMethodField()
    network = NetworkListSerializer(read_only=True)
    source_network = NetworkListSerializer(read_only=True)
    class Meta: model = EdgeAlgorithm; fields = ['id', 'name', 'algorithm', 'total_edge_length', 'network', 'source_network', 'parameters', 'edges', 'nodes']
    def get_nodes(self, obj):
        nodes = Node.objects.filter(network=obj.network)
        return NodeMapSerializer(nodes, many=True).data
class EdgeAlgorithmSerializer(serializers.ModelSerializer): # Legacy
    class Meta: model = EdgeAlgorithm; fields = '__all__'

# --- Connection Algorithms ---

class EdgeConnectionAlgorithmEdgeSerializer(serializers.ModelSerializer):
    """
    Serializes the intermediate table. 
    Requires EdgeSerializer to be defined above.
    """
    edge = EdgeSerializer(read_only=True)
    class Meta: model = EdgeConnectionAlgorithmEdge; fields = ['id', 'edge']

class ConnectionEdgeSerializer(serializers.ModelSerializer):
    id = serializers.ReadOnlyField(source='edge.id')
    length = serializers.ReadOnlyField(source='edge.length')
    path = serializers.SerializerMethodField()
    class Meta: model = EdgeConnectionAlgorithmEdge; fields = ['id', 'length', 'path']
    def get_path(self, obj):
        start = obj.edge.start_node
        end = obj.edge.end_node
        return [{'lat': start.latitude, 'lng': start.longitude}, {'lat': end.latitude, 'lng': end.longitude}]

class EdgeConnectionAlgorithmListSerializer(serializers.ModelSerializer):
    calculated_total_length = serializers.FloatField(read_only=True)
    network = NetworkListSerializer(read_only=True)
    source_network = NetworkListSerializer(read_only=True)
    edge_algorithm_name = serializers.SerializerMethodField()
    class Meta:
        model = EdgeConnectionAlgorithm
        fields = ['id', 'name', 'algorithm', 'calculated_total_length', 'total_edge_length', 'network', 'source_network', 'edge_algorithm', 'edge_algorithm_name']
    def get_edge_algorithm_name(self, obj):
        if obj.edge_algorithm:
            return obj.edge_algorithm.name
        source_id = (obj.parameters or {}).get('source_edge_algorithm_id')
        return f"Source edge algorithm {source_id}" if source_id else None

class EdgeConnectionAlgorithmDetailSerializer(serializers.ModelSerializer):
    calculated_total_length = serializers.FloatField(read_only=True)
    edges = ConnectionEdgeSerializer(source='edges_connection', many=True, read_only=True)
    nodes = serializers.SerializerMethodField()
    network = NetworkListSerializer(read_only=True)
    source_network = NetworkListSerializer(read_only=True)
    edge_algorithm_name = serializers.SerializerMethodField()
    class Meta:
        model = EdgeConnectionAlgorithm
        fields = ['id', 'name', 'algorithm', 'description', 'parameters', 'calculated_total_length', 'total_edge_length', 'network', 'source_network', 'edge_algorithm', 'edge_algorithm_name', 'edges', 'nodes']
    def get_nodes(self, obj):
        nodes = Node.objects.filter(network=obj.network)
        return NodeMapSerializer(nodes, many=True).data
    def get_edge_algorithm_name(self, obj):
        if obj.edge_algorithm:
            return obj.edge_algorithm.name
        source_id = (obj.parameters or {}).get('source_edge_algorithm_id')
        return f"Source edge algorithm {source_id}" if source_id else None


# ==============================================================================
# 8. ROUTING & TRANSLATION
# ==============================================================================

class RouteSerializer(serializers.ModelSerializer):
    class Meta: model = Route; fields = '__all__'

class RouteMapSerializer(serializers.ModelSerializer):
    class Meta: model = Route; fields = ['id', 'route_type', 'polyline', 'distance_meters', 'duration_seconds']

class EdgeTranslationLengthMixin:
    total_edge_length = serializers.SerializerMethodField()

    def get_total_edge_length(self, obj):
        routes = list(obj.routes.select_related('edge').all())
        if routes:
            return unique_route_length(routes)
        return obj.total_edge_length

class EdgeTranslationListSerializer(EdgeTranslationLengthMixin, serializers.ModelSerializer):
    network = NetworkListSerializer(read_only=True)
    source_network = NetworkListSerializer(read_only=True)
    edge_connection_algorithm_name = serializers.SerializerMethodField()
    class Meta:
        model = EdgeTranslation
        fields = [
            'id', 'name', 'translation', 'total_edge_length',
            'network', 'source_network', 'edge_connection_algorithm',
            'edge_connection_algorithm_name',
        ]
    def get_edge_connection_algorithm_name(self, obj):
        if obj.edge_connection_algorithm:
            return obj.edge_connection_algorithm.name
        source_id = (obj.parameters or {}).get('source_edge_connection_algorithm_id')
        return f"Source connection {source_id}" if source_id else None

class EdgeTranslationDetailSerializer(EdgeTranslationLengthMixin, serializers.ModelSerializer):
    routes = RouteMapSerializer(many=True, read_only=True)
    nodes = serializers.SerializerMethodField()
    network = NetworkListSerializer(read_only=True)
    source_network = NetworkListSerializer(read_only=True)
    edge_connection_algorithm_name = serializers.SerializerMethodField()
    class Meta:
        model = EdgeTranslation
        fields = [
            'id', 'name', 'translation', 'description', 'total_edge_length',
            'parameters', 'network', 'source_network', 'edge_connection_algorithm',
            'edge_connection_algorithm_name', 'routes', 'nodes',
        ]
    def get_nodes(self, obj):
        edge_ids = obj.routes.values_list('edge_id', flat=True)
        edges = Edge.objects.filter(id__in=edge_ids)
        start_ids = edges.values_list('start_node_id', flat=True)
        end_ids = edges.values_list('end_node_id', flat=True)
        nodes = Node.objects.filter(id__in=set(start_ids) | set(end_ids))
        return NodeMapSerializer(nodes, many=True).data
    def get_edge_connection_algorithm_name(self, obj):
        if obj.edge_connection_algorithm:
            return obj.edge_connection_algorithm.name
        source_id = (obj.parameters or {}).get('source_edge_connection_algorithm_id')
        return f"Source connection {source_id}" if source_id else None
class EdgeTranslationSerializer(serializers.ModelSerializer): # Legacy
    class Meta: model = EdgeTranslation; fields = '__all__'

class TranslationWithAlgorithmSerializer(EdgeTranslationLengthMixin, serializers.ModelSerializer):
    algorithm = EdgeAlgorithmSerializer()
    class Meta: model = EdgeTranslation; fields = ['id', 'name', 'translation', 'parameters', 'total_edge_length', 'algorithm']

class EdgeWithRoutesSerializer(serializers.ModelSerializer):
    start_node = NodeSerializer()
    end_node = NodeSerializer()
    routes = serializers.SerializerMethodField()
    class Meta: model = Edge; fields = ['id', 'name', 'start_node', 'end_node', 'length', 'routes']
    def get_routes(self, obj):
        tid = self.context.get('translation_id')
        if tid: return RouteSerializer(obj.routes.filter(translation_id=tid), many=True).data
        return []

class NetworkByTranslationSerializer(serializers.ModelSerializer):
    nodes = serializers.SerializerMethodField()
    edges = serializers.SerializerMethodField()
    translation = serializers.SerializerMethodField()
    algorithms = serializers.SerializerMethodField()
    class Meta:
        model = Network
        fields = ['id', 'name', 'description', 'network_type', 'total_length', 'nodes', 'edges', 'translation', 'algorithms']
    def get_nodes(self, obj):
        tid = self.context.get('translation_id')
        if not tid: return []
        edges = Edge.objects.filter(network=obj, route__translation_id=tid).select_related('start_node', 'end_node')
        nids = set()
        for e in edges: nids.add(e.start_node_id); nids.add(e.end_node_id)
        return NodeSerializer(Node.objects.filter(id__in=nids), many=True).data
    def get_edges(self, obj):
        tid = self.context.get('translation_id')
        if not tid: return []
        edges = Edge.objects.filter(network=obj, route__translation_id=tid).distinct().select_related('start_node', 'end_node').prefetch_related('route_set')
        return EdgeWithRoutesSerializer(edges, many=True, context={'translation_id': tid}).data
    def get_translation(self, obj):
        tid = self.context.get('translation_id')
        if not tid: return None
        try: return EdgeTranslationSerializer(EdgeTranslation.objects.get(id=tid, network=obj)).data
        except: return None
    def get_algorithms(self, obj):
        tid = self.context.get('translation_id')
        if not tid: return []
        aids = Edge.objects.filter(network=obj, route__translation_id=tid).values_list('edge_algorithm_id', flat=True).distinct()
        return EdgeAlgorithmSerializer(EdgeAlgorithm.objects.filter(id__in=aids), many=True).data


class PipeSizingSerializer(serializers.ModelSerializer):
    network_name = serializers.CharField(source='network.name', read_only=True)
    source_network_name = serializers.CharField(source='source_network.name', read_only=True)
    topology_source = serializers.SerializerMethodField()
    topology_source_name = serializers.SerializerMethodField()
    source_edge_connection_algorithm_id = serializers.SerializerMethodField()
    source_edge_translation_id = serializers.SerializerMethodField()

    class Meta:
        model = PipeSizing
        fields = '__all__'

    def _parameters(self, obj):
        return obj.network.derivation_parameters if obj.network and obj.network.derivation_parameters else {}

    def get_topology_source(self, obj):
        params = self._parameters(obj)
        if params.get('source_edge_translation_id'):
            return 'route'
        if params.get('source_edge_connection_algorithm_id'):
            return 'connection'
        return None

    def get_source_edge_connection_algorithm_id(self, obj):
        return self._parameters(obj).get('source_edge_connection_algorithm_id')

    def get_source_edge_translation_id(self, obj):
        return self._parameters(obj).get('source_edge_translation_id')

    def get_topology_source_name(self, obj):
        params = self._parameters(obj)
        translation_id = params.get('source_edge_translation_id')
        if translation_id:
            translation = EdgeTranslation.objects.filter(id=translation_id).first()
            return translation.name if translation else f"Route tracing {translation_id}"

        connection_id = params.get('source_edge_connection_algorithm_id')
        if connection_id:
            connection = EdgeConnectionAlgorithm.objects.filter(id=connection_id).first()
            return connection.name if connection else f"Connection {connection_id}"

        return None

# ==============================================================================
# 9. PIPE SIZING & ANALYSIS
# ==============================================================================

class EdgePipeSizingDetailSerializer(serializers.ModelSerializer):
    """
    Optimized for Visualization: Includes Geometry (Edge Nodes + Route Polyline).
    Uses existing NodeMapSerializer and RouteMapSerializer to keep payload light.
    """
    # Reuse existing EdgeSerializer to get start/end node coordinates
    edge = EdgeSerializer(read_only=True)
    # Reuse existing RouteMapSerializer to get the decoded polyline
    route = RouteMapSerializer(read_only=True)

    class Meta:
        model = EdgePipeSizing
        fields = [
            'id', 
            'pipe_sizing', 
            'edge', 
            'route', 
            'dea_pipe_category',
            'peak_load_kw', 
            'mass_flow_kg_s', 
            'volume_flow_m3_s', 
            'diameter_m', 
              'velocity_m_s', 
              'delta_t',
              'pressure_drop_pa',
              'length_m',
              'source_route_ids',
              'source_edge_ids',
              'overlap_count',
              'is_shared_segment',
              'segment_path',
          ]

class EdgePipeSizingListSerializer(serializers.ModelSerializer):
    """
    Lighter serializer for table views (No nested geometry).
    """
    edge_name = serializers.CharField(source='edge.name', read_only=True)
    
    class Meta:
        model = EdgePipeSizing
        fields = [
            'id', 
            'edge_name', 
            'peak_load_kw', 
              'mass_flow_kg_s', 
              'diameter_m', 
              'dea_pipe_category',
              'velocity_m_s',
              'delta_t',
              'pressure_drop_pa',
              'length_m',
              'overlap_count',
              'is_shared_segment',
          ]
class ProjectClusteringSerializer(serializers.Serializer):
    project_id = serializers.IntegerField(required=True, help_text="ID of the Project to cluster")
    w1 = serializers.FloatField(required=False, default=0.5, min_value=0.0, max_value=1.0, help_text="Weight for Spatial Distance (0.0 - 1.0)")
    w2 = serializers.FloatField(required=False, default=0.5, min_value=0.0, max_value=1.0, help_text="Weight for Energy Balance (0.0 - 1.0)")
    epsilon = serializers.FloatField(required=False, default=0.2, min_value=0.001, help_text="DBSCAN Search Radius (Normalized 0-1)")
    min_samples = serializers.IntegerField(required=False, default=3, min_value=2, help_text="Minimum buildings per cluster")
from rest_framework import serializers

class NetworkFromClusterSerializer(serializers.Serializer):
    source_type = serializers.ChoiceField(choices=['configuration', 'cluster'], help_text="configuration = Macro Network (Nodes are Clusters). cluster = Micro Network (Nodes are Buildings).")
    source_id = serializers.IntegerField(help_text="ID of the Configuration or Cluster")
    network_name = serializers.CharField(required=False, allow_blank=True)
    project_id = serializers.IntegerField(required=False, help_text="Required if source_type is 'cluster' to identify the project context for the cluster.")

class ClusterBalancingSerializer(serializers.Serializer):
    config_id = serializers.IntegerField(
        help_text="ID of the Cluster Configuration to balance."
    )
    
    STRATEGY_CHOICES = [
        ('load_shifting', 'Load Shifting (Move Buildings)'),
        ('interconnection', 'Interconnection (Pipe Network)'),
        ('auxiliary', 'Auxiliary Source (Add Boiler Node)'),
        ('mobile_storage', 'Mobile Thermal Storage (Truck Routes)')
    ]
    
    strategy = serializers.ChoiceField(
        choices=STRATEGY_CHOICES, 
        help_text="The algorithm to use for balancing deficit clusters."
    )

    def validate_config_id(self, value):
        if not ClusterConfiguration.objects.filter(id=value).exists():
            raise serializers.ValidationError("Cluster Configuration not found.")
        return value
class SourceAnchoredClusteringSerializer(serializers.Serializer):
    project_id = serializers.IntegerField(required=True, help_text="ID of the Project to cluster")
    max_radius_km = serializers.FloatField(required=False, default=0.5, min_value=0.1, help_text="Maximum distance (in KM) for a sink to be assigned to a source")
    name = serializers.CharField(required=False, allow_blank=True, max_length=100, help_text="Optional display name for the saved cluster configuration")

from rest_framework import serializers
from .models import Network, Node, Building, Boiler, LoadProfile



class NetworkEntitySerializer(serializers.ModelSerializer):
    """
    A unified serializer to extract Building or Boiler details 
    from a Node so the frontend list renders consistently.
    """
    entity_id = serializers.SerializerMethodField()
    name = serializers.SerializerMethodField()
    entity_type = serializers.CharField(source='node_type')
    building_type = serializers.SerializerMethodField()

    class Meta:
        model = Node
        fields = ['id', 'entity_id', 'name', 'entity_type', 'building_type']

    def get_entity_id(self, obj):
        if obj.node_type == 'building' and obj.building:
            return obj.building.id
        elif obj.node_type == 'boiler' and obj.boiler:
            return obj.boiler.id
        return None

    def get_name(self, obj):
        if obj.node_type == 'building' and obj.building:
            return obj.building.name or f"Building {obj.building.id}"
        elif obj.node_type == 'boiler' and obj.boiler:
            return obj.boiler.name or f"Boiler {obj.boiler.id}"
        return obj.name

    def get_building_type(self, obj):
        if obj.node_type == 'building' and obj.building:
            return obj.building.building_type
        return None

class LoadProfileSerializer(serializers.ModelSerializer):
    class Meta:
        model = LoadProfile
        fields = ['start_date', 'value', 'unit', 'profile_type']
