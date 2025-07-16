from rest_framework import serializers
from .models import *

class CountrySerializer(serializers.ModelSerializer):
    class Meta:
        model = Country
        fields = '__all__'

class ProvinceSerializer(serializers.ModelSerializer):
    country = CountrySerializer()
    
    class Meta:
        model = Province
        fields = '__all__'

class CitySerializer(serializers.ModelSerializer):
    province = ProvinceSerializer(required=False)
    country = CountrySerializer(required=False)
    
    class Meta:
        model = City
        fields = '__all__'

class ProjectSerializer(serializers.ModelSerializer):
    class Meta:
        model = Project
        fields = '__all__'

class NetworkSerializer(serializers.ModelSerializer):
    class Meta:
        model = Network
        fields = '__all__'

class LocationSerializer(serializers.ModelSerializer):    
    class Meta:
        model = Location
        fields = '__all__'
        abstract = True

class BuildingSerializer(LocationSerializer):
    class Meta:
        model = Building
        fields = '__all__'

# Network and Nodes Serializers
class NodeSerializer(LocationSerializer):
    network = serializers.PrimaryKeyRelatedField(queryset=Network.objects.all())
    
    class Meta:
        model = Node
        fields = '__all__'

class EdgeSerializer(serializers.ModelSerializer):
    network = serializers.PrimaryKeyRelatedField(queryset=Network.objects.all())
    start_node = serializers.PrimaryKeyRelatedField(queryset=Node.objects.all())
    end_node = serializers.PrimaryKeyRelatedField(queryset=Node.objects.all())
    
    class Meta:
        model = Edge
        fields = '__all__'


class RouteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Route
        fields = '__all__'
# Equipment and Components Serializers
class PumpSerializer(LocationSerializer):
    class Meta:
        model = Pump
        fields = '__all__'

class ValveSerializer(LocationSerializer):
    class Meta:
        model = Valve
        fields = '__all__'

class HeatPumpSerializer(LocationSerializer):
    class Meta:
        model = HeatPump
        fields = '__all__'

class HeatExchangerSerializer(LocationSerializer):
    class Meta:
        model = HeatExchanger
        fields = '__all__'

class PipeSerializer(LocationSerializer):
    class Meta:
        model = Pipe
        fields = '__all__'

class JunctionSerializer(LocationSerializer):
    class Meta:
        model = Junction
        fields = '__all__'

class PVSystemSerializer(LocationSerializer):
    class Meta:
        model = PVSystem
        fields = '__all__'

class GeothermalSystemSerializer(LocationSerializer):
    class Meta:
        model = GeothermalSystem
        fields = '__all__'

class BoilerSerializer(LocationSerializer):
    class Meta:
        model = Boiler
        fields = '__all__'

class ThermalStorageSerializer(LocationSerializer):
    class Meta:
        model = ThermalStorage
        fields = '__all__'

class SensorSerializer(LocationSerializer):
    class Meta:
        model = Sensor
        fields = '__all__'

class MeterSerializer(LocationSerializer):
    class Meta:
        model = Meter
        fields = '__all__'
class EdgeAlgorithmSerializer(serializers.ModelSerializer):
    class Meta:
        model = EdgeAlgorithm
        fields = '__all__'

class EdgeTranslationSerializer(serializers.ModelSerializer):
    class Meta:
        model = EdgeTranslation
        fields = '__all__'

class EdgeWithRoutesSerializer(serializers.ModelSerializer):
    start_node = NodeSerializer()
    end_node = NodeSerializer()
    routes = serializers.SerializerMethodField()
    
    class Meta:
        model = Edge
        fields = ['id', 'name', 'start_node', 'end_node', 'length', 'routes']
    
    def get_routes(self, obj):
        translation_id = self.context.get('translation_id')
        if translation_id:
            routes = obj.routes.filter(translation_id=translation_id)  # Changed from route to routes
            return RouteSerializer(routes, many=True).data
        return []

class RouteSerializer(serializers.ModelSerializer):
    class Meta:
        model = Route
        fields = ['id', 'polyline', 'distance_meters', 'duration_seconds']

class NetworkByTranslationSerializer(serializers.ModelSerializer):
    nodes = serializers.SerializerMethodField()
    edges = serializers.SerializerMethodField()
    translation = serializers.SerializerMethodField()
    algorithms = serializers.SerializerMethodField()
    
    class Meta:
        model = Network
        fields = ['id', 'name', 'description', 'network_type', 'total_length',
                 'nodes', 'edges', 'translation', 'algorithms']
    
    def get_nodes(self, obj):
        translation_id = self.context.get('translation_id')
        if translation_id:
            # Get all nodes that are part of edges with routes for this translation
            edges = Edge.objects.filter(
                network=obj,
                route__translation_id=translation_id
            ).select_related('start_node', 'end_node')
            
            node_ids = set()
            for edge in edges:
                node_ids.add(edge.start_node_id)
                node_ids.add(edge.end_node_id)
            
            nodes = Node.objects.filter(id__in=node_ids)
            return NodeSerializer(nodes, many=True).data
        return []
    
    def get_edges(self, obj):
        translation_id = self.context.get('translation_id')
        if translation_id:
            edges = Edge.objects.filter(
                network=obj,
                route__translation_id=translation_id
            ).distinct().select_related('start_node', 'end_node').prefetch_related('route_set')
            
            return EdgeWithRoutesSerializer(edges, many=True, context={
                'translation_id': translation_id
            }).data
        return []
    
    def get_translation(self, obj):
        translation_id = self.context.get('translation_id')
        if translation_id:
            try:
                translation = EdgeTranslation.objects.get(id=translation_id, network=obj)
                return EdgeTranslationSerializer(translation).data
            except EdgeTranslation.DoesNotExist:
                return None
        return None
    
    def get_algorithms(self, obj):
        translation_id = self.context.get('translation_id')
        if translation_id:
            algorithm_ids = Edge.objects.filter(
                network=obj,
                route__translation_id=translation_id
            ).values_list('edge_algorithm_id', flat=True).distinct()
            
            algorithms = EdgeAlgorithm.objects.filter(id__in=algorithm_ids)
            return EdgeAlgorithmSerializer(algorithms, many=True).data
        return []
    nodes = serializers.SerializerMethodField()
    edges = serializers.SerializerMethodField()
    translation = serializers.SerializerMethodField()
    algorithms = serializers.SerializerMethodField()
    
    class Meta:
        model = Network
        fields = ['id', 'name', 'description', 'network_type', 'total_length',
                 'nodes', 'edges', 'translation', 'algorithms']
    
    def get_nodes(self, obj):
        # Get all nodes that are part of edges with routes for this translation
        translation_id = self.context.get('translation_id')
        if translation_id:
            node_ids = Edge.objects.filter(
                network=obj,
                routes__translation_id=translation_id  # Changed from route to routes
            ).values_list('start_node_id', 'end_node_id').distinct()
            # Flatten the list of tuples
            node_ids = set([id for pair in node_ids for id in pair])
            nodes = Node.objects.filter(id__in=node_ids)
            return NodeSerializer(nodes, many=True).data
        return []
    
    def get_edges(self, obj):
        translation_id = self.context.get('translation_id')
        if translation_id:
            edges = Edge.objects.filter(
                network=obj,
                routes__translation_id=translation_id  # Changed from route to routes
            ).distinct()
            return EdgeWithRoutesSerializer(edges, many=True, context={
                'translation_id': translation_id
            }).data
        return []
    
    def get_translation(self, obj):
        translation_id = self.context.get('translation_id')
        if translation_id:
            try:
                translation = EdgeTranslation.objects.get(id=translation_id)
                return EdgeTranslationSerializer(translation).data
            except EdgeTranslation.DoesNotExist:
                return None
        return None
    
    def get_algorithms(self, obj):
        translation_id = self.context.get('translation_id')
        if translation_id:
            # Get algorithms used by edges that have routes with this translation
            algorithm_ids = Edge.objects.filter(
                network=obj,
                routes__translation_id=translation_id  # Changed from route to routes
            ).values_list('edge_algorithm_id', flat=True).distinct()
            algorithms = EdgeAlgorithm.objects.filter(id__in=algorithm_ids)
            return EdgeAlgorithmSerializer(algorithms, many=True).data
        return []
    
class NetworkByTranslationSerializer(serializers.ModelSerializer):
    nodes = serializers.SerializerMethodField()
    edges = serializers.SerializerMethodField()
    translation = serializers.SerializerMethodField()
    algorithms = serializers.SerializerMethodField()
    
    class Meta:
        model = Network
        fields = ['id', 'name', 'description', 'network_type', 'total_length',
                 'nodes', 'edges', 'translation', 'algorithms']
    
    def get_nodes(self, obj):
        translation_id = self.context.get('translation_id')
        if translation_id:
            # Get all nodes that are part of edges with routes for this translation
            edges = Edge.objects.filter(
                network=obj,
                route__translation_id=translation_id
            ).select_related('start_node', 'end_node')
            
            node_ids = set()
            for edge in edges:
                node_ids.add(edge.start_node_id)
                node_ids.add(edge.end_node_id)
            
            nodes = Node.objects.filter(id__in=node_ids)
            return NodeSerializer(nodes, many=True).data
        return []
    
    def get_edges(self, obj):
        translation_id = self.context.get('translation_id')
        if translation_id:
            edges = Edge.objects.filter(
                network=obj,
                route__translation_id=translation_id
            ).distinct().select_related('start_node', 'end_node').prefetch_related('route_set')
            
            return EdgeWithRoutesSerializer(edges, many=True, context={
                'translation_id': translation_id
            }).data
        return []
    
    def get_translation(self, obj):
        translation_id = self.context.get('translation_id')
        if translation_id:
            try:
                translation = EdgeTranslation.objects.get(id=translation_id, network=obj)
                return EdgeTranslationSerializer(translation).data
            except EdgeTranslation.DoesNotExist:
                return None
        return None
    
    def get_algorithms(self, obj):
        translation_id = self.context.get('translation_id')
        if translation_id:
            algorithm_ids = Edge.objects.filter(
                network=obj,
                route__translation_id=translation_id
            ).values_list('edge_algorithm_id', flat=True).distinct()
            
            algorithms = EdgeAlgorithm.objects.filter(id__in=algorithm_ids)
            return EdgeAlgorithmSerializer(algorithms, many=True).data
        return [] 
class TranslationWithAlgorithmSerializer(serializers.ModelSerializer):
    algorithm = EdgeAlgorithmSerializer()
    
    class Meta:
        model = EdgeTranslation
        fields = ['id', 'name', 'translation', 'parameters', 'total_edge_length', 'algorithm']
