import json
import time
from django.forms import model_to_dict
import requests
from rest_framework import viewsets, status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import NotFound
from .models import *
from .serializers import *
import networkx as nx
import numpy as np
from django.db import transaction
from django.db.models import Prefetch
from .models import Network, Node, Edge, Junction
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from .models import Network, Node, Edge
from scipy.spatial import Delaunay
from sklearn.neighbors import NearestNeighbors
import numpy as np
import math
import uuid
from rest_framework import generics
import numpy as np
from django.db import transaction
import time
import requests
from googlemaps.convert import decode_polyline, encode_polyline # type: ignore
from shapely.geometry import LineString
import polyline
from shapely.geometry import LineString

class CountryViewSet(viewsets.ModelViewSet):
    queryset = Country.objects.all()
    serializer_class = CountrySerializer

class ProvinceViewSet(viewsets.ModelViewSet):
    queryset = Province.objects.all()
    serializer_class = ProvinceSerializer

class CityViewSet(viewsets.ModelViewSet):
    queryset = City.objects.all()
    serializer_class = CitySerializer

class ProjectViewSet(viewsets.ModelViewSet):
    queryset = Project.objects.all()
    serializer_class = ProjectSerializer

class NetworkViewSet(viewsets.ModelViewSet):
    queryset = Network.objects.all()
    serializer_class = NetworkSerializer
class ClusterViewSet(viewsets.ModelViewSet):
    queryset = Cluster.objects.all()
    serializer_class = ClusterSerializer
class RouteViewSet(viewsets.ModelViewSet):
    queryset = Route.objects.all()
    serializer_class = RouteSerializer
class BuildingViewSet(viewsets.ModelViewSet):
    queryset = Building.objects.all()
    serializer_class = BuildingSerializer

class NodeViewSet(viewsets.ModelViewSet):
    queryset = Node.objects.all()
    serializer_class = NodeSerializer

class EdgeViewSet(viewsets.ModelViewSet):
    queryset = Edge.objects.all()
    serializer_class = EdgeSerializer

class PumpViewSet(viewsets.ModelViewSet):
    queryset = Pump.objects.all()
    serializer_class = PumpSerializer

class ValveViewSet(viewsets.ModelViewSet):
    queryset = Valve.objects.all()
    serializer_class = ValveSerializer

class HeatPumpViewSet(viewsets.ModelViewSet):
    queryset = HeatPump.objects.all()
    serializer_class = HeatPumpSerializer

class HeatExchangerViewSet(viewsets.ModelViewSet):
    queryset = HeatExchanger.objects.all()
    serializer_class = HeatExchangerSerializer

class PipeViewSet(viewsets.ModelViewSet):
    queryset = Pipe.objects.all()
    serializer_class = PipeSerializer

class JunctionViewSet(viewsets.ModelViewSet):
    queryset = Junction.objects.all()
    serializer_class = JunctionSerializer

class PVSystemViewSet(viewsets.ModelViewSet):
    queryset = PVSystem.objects.all()
    serializer_class = PVSystemSerializer

class GeothermalSystemViewSet(viewsets.ModelViewSet):
    queryset = GeothermalSystem.objects.all()
    serializer_class = GeothermalSystemSerializer

class BoilerViewSet(viewsets.ModelViewSet):
    queryset = Boiler.objects.all()
    serializer_class = BoilerSerializer

class ThermalStorageViewSet(viewsets.ModelViewSet):
    queryset = ThermalStorage.objects.all()
    serializer_class = ThermalStorageSerializer

class SensorViewSet(viewsets.ModelViewSet):
    queryset = Sensor.objects.all()
    serializer_class = SensorSerializer

class MeterViewSet(viewsets.ModelViewSet):
    queryset = Meter.objects.all()
    serializer_class = MeterSerializer

class NetworkByTranslationView(generics.GenericAPIView):
    serializer_class = NetworkByTranslationSerializer
    
    def get(self, request, *args, **kwargs):
        network_id = kwargs.get('network_id')
        translation_id = kwargs.get('translation_id')
        
        try:
            network = Network.objects.get(pk=network_id)
        except Network.DoesNotExist:
            raise NotFound("Network not found")
        
        # Verify the translation exists
        try:
            EdgeTranslation.objects.get(pk=translation_id)
        except EdgeTranslation.DoesNotExist:
            raise NotFound("Translation not found")
        
        serializer = self.serializer_class(
            network,
            context={
                'translation_id': translation_id,
                'request': request
            }
        )
        
        return Response(serializer.data)
class EdgeAlgorithmViewSet(viewsets.ModelViewSet):
    queryset = EdgeAlgorithm.objects.all()
    serializer_class = EdgeAlgorithmSerializer
class EdgeTranslationViewSet(viewsets.ModelViewSet):
    queryset = EdgeTranslation.objects.all()
    serializer_class = EdgeTranslationSerializer


class NetworkDetailView(APIView):
    def get(self, request, pk):
        try:
            # Get the network with all related nodes, edges, and junctions
            network = Network.objects.prefetch_related(
                Prefetch('node_set', queryset=Node.objects.all()),
                Prefetch('edge_set', queryset=Edge.objects.select_related('start_node', 'end_node')),
            ).get(pk=pk)
            
            # Serialize the data
            network_data = NetworkSerializer(network).data
            nodes_data = NodeSerializer(network.node_set.all(), many=True).data # type: ignore
            edges_data = EdgeSerializer(network.edge_set.all(), many=True).data # type: ignore
            
            response_data = {
                'network': network_data,
                'nodes': nodes_data,
                'edges': edges_data,
                'stats': {
                    'node_count': len(nodes_data),
                    'edge_count': len(edges_data),
                    'total_length': network.total_length
                }
            }
            
            return Response(response_data)
            
        except Network.DoesNotExist:
            return Response(
                {"error": "Network not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class CreateNetworkView(APIView):
    @transaction.atomic
    def post(self, request, project_id):
        try:
            project = Project.objects.get(pk=project_id)
            
            # Create a new network for this project
            network = Network.objects.create(
                project=project,
                name=f"{project.name} {project.city} Thermal Network",
                network_type="district_heating",
                description="Automatically generated network"
            )
            
            # Get all infrastructure points for this project
            buildings = Building.objects.filter(project=project)
            pumps = Pump.objects.filter(project=project)
            valves = Valve.objects.filter(project=project)
            heat_pumps = HeatPump.objects.filter(project=project)
            
            # Combine all infrastructure points
            infrastructure_points = []
            infrastructure_points.extend([(b.id, b.latitude, b.longitude, 'building', b.name, b.building_type) for b in buildings]) # type: ignore
            infrastructure_points.extend([(p.id, p.latitude, p.longitude, 'pump', p.name,"") for p in pumps]) # type: ignore
            infrastructure_points.extend([(v.id, v.latitude, v.longitude, 'valve', v.name, "") for v in valves]) # type: ignore
            infrastructure_points.extend([(hp.id, hp.latitude, hp.longitude, 'heat_pump', hp.name, "") for hp in heat_pumps]) # type: ignore
            
            if len(infrastructure_points) < 2:
                return Response(
                    {"error": "Not enough infrastructure points to create a network (minimum 2 required)"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Create nodes for all infrastructure points
            nodes = []
            for point_id, lat, lng, point_type, infra_name, type in infrastructure_points:
                node_name = infra_name if infra_name else f"{point_type.capitalize()} {point_id}"
                node = Node.objects.create(
                    network=network,
                    name=node_name,
                    latitude=lat,
                    longitude=lng,
                    node_type='consumer' if type not in ['commercial', 'industrial'] else 'producer'
                )
                nodes.append((node.id, lat, lng)) # type: ignore
            
            # Create a complete graph with all possible edges
            G = nx.Graph()
            
            # Add nodes to graph
            for node_id, lat, lng in nodes:
                G.add_node(node_id, pos=(lat, lng))
            
            network.save()
            
            return Response({
                "message": f"Network created successfully with {len(nodes)} nodes",
                "network_id": network.id, # type: ignore
            }, status=status.HTTP_201_CREATED)
            
        except Project.DoesNotExist:
            return Response(
                {"error": "Project not found"},
                status=status.HTTP_404_NOT_FOUND
            )
        except Exception as e:
            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )

class EdgeTranslationAPIView(APIView):
    """
    API endpoint that creates edge translations and routes from an edge algorithm using Google Maps API.
    """
    
    def post(self, request, *args, **kwargs):
        # Extract parameters from request
        edge_algorithm_id = request.data.get('edge_algorithm_id')
        translation_type = request.data.get('translation_type')
        name = request.data.get('name', '')
        description = request.data.get('description', '')
        is_default = request.data.get('is_default', False)
        parameters = request.data.get('parameters', {})
        if not edge_algorithm_id or not translation_type:
            return Response(
                {'error': 'edge_algorithm_id and translation_type are required'},
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # Get the edge algorithm
            edge_algorithm = EdgeAlgorithm.objects.get(pk=edge_algorithm_id)
        except EdgeAlgorithm.DoesNotExist:
            return Response(
                {'error': 'EdgeAlgorithm not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        # Validate translation type against Google Maps options
        valid_translation_types = {
            #'driving': 'driving',
            'walking': 'walking',
            #'bicycling': 'bicycling',
            'transit': 'transit'
        }

        if translation_type not in valid_translation_types:
            return Response(
                {'error': f'Invalid translation type. Valid types are: {", ".join(valid_translation_types.keys())}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        return translate(edge_algorithm, translation_type,parameters, description, name)




    def _merge_segments_to_polyline(self, segments):
        """Merge multiple LineString segments into a single polyline"""
        coords = []
        for seg in segments:
            if seg.geom_type == 'LineString':
                coords.extend(seg.coords)
            elif seg.geom_type == 'MultiLineString':
                for part in seg.geoms:
                    coords.extend(part.coords)
        
        points = [{'lat': y, 'lng': x} for x, y in coords]
        return encode_polyline(points)

    def _line_to_polyline(self, line):
        """Convert LineString to encoded polyline"""
        if line.is_empty:
            return ""
        
        if line.geom_type == 'LineString':
            coords = list(line.coords)
        elif line.geom_type == 'MultiLineString':
            coords = []
            for part in line.geoms:
                coords.extend(part.coords)
        else:
            return ""
        
        points = [{'lat': y, 'lng': x} for x, y in coords]
        return encode_polyline(points)
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r * 1000  # Convert to meters
class NetworkTranslationDetailView(generics.GenericAPIView):
    def get(self, request, edge_algorithm_id, translation_id):
        # Verify network exists
        try:
            algorithm = EdgeAlgorithm.objects.get(pk=edge_algorithm_id)
        except EdgeAlgorithm.DoesNotExist:
            raise NotFound("Algorithm not found")
        try:
            network = Network.objects.get(pk=algorithm.network.id) # type: ignore
        except Network.DoesNotExist:
            raise NotFound("Network not found")
        # Get translation with prefetched algorithm and verify it belongs to this network
        try:
            translation = EdgeTranslation.objects.select_related('algorithm').get(
                pk=translation_id,
                algorithm=algorithm

            )
        except EdgeTranslation.DoesNotExist:
            raise NotFound("Translation not found for this network")

        # Get all routes for this translation with prefetched edge and nodes
        routes = Route.objects.filter(
            translation=translation,
        ).select_related(
            'edge__start_node',
            'edge__end_node'
        )

        # Get unique nodes from edges
        nodes = set()
        for route in routes:
            nodes.add(route.edge.start_node)
            nodes.add(route.edge.end_node)

        return Response({
            'network': NetworkSerializer(network).data,
            'translation': TranslationWithAlgorithmSerializer(translation).data,
            'routes': RouteSerializer(routes, many=True).data,
            'nodes': NodeSerializer(nodes, many=True).data,
            'statistics': {
                'total_routes': routes.count(),
                'total_nodes': len(nodes),
                'total_distance': sum(route.distance_meters for route in routes),
                'total_duration': sum(route.duration_seconds for route in routes)
            }
        })

class AlgorithmByNetworkDetailView(generics.GenericAPIView):
    def get(self, request, network_id):

        # Verify network exists
        try:
            network = Network.objects.get(pk=network_id)
        except Network.DoesNotExist:
            raise NotFound("Network not found")
        try:
            algorithms = EdgeAlgorithm.objects.filter(network=network)
        except EdgeAlgorithm.DoesNotExist:
            raise NotFound("Algorithm not found")
        return Response({
            'algorithms': EdgeAlgorithmSerializer(algorithms, many=True).data,
        })
class TranslationsByAlgorithmDetailView(generics.GenericAPIView):
    def get(self, request, algorithm_id):

        # Verify network exists
        try:
            algorithm = EdgeAlgorithm.objects.get(pk=algorithm_id)
        except EdgeAlgorithm.DoesNotExist:
            raise NotFound("Algorithm not found")
        try:
            translations = EdgeTranslation.objects.filter(algorithm=algorithm)
        except EdgeTranslation.DoesNotExist:
            raise NotFound("Translation not found")
        return Response({
            'translations': EdgeTranslationSerializer(translations, many=True).data,
        })

@csrf_exempt  # Temporarily disable CSRF for testing
@require_http_methods(["POST"])
def cluster_network_nodes(request, network_id):
    try:
        network = Network.objects.get(pk=network_id)
        nodes = list(Node.objects.filter(network=network))
        
        if len(nodes) < 2:
            return JsonResponse({'status': 'error', 'message': 'Network needs at least 2 nodes'})        
       # Parse JSON data from request body
        result = cluster_nodes(network_id)
        cluster = result['meta']   # type: ignore

        return JsonResponse( {        
            'status': 'success',
            'message': f'Successfully created clusters for network',
            'cluster': cluster.name
            
        })
    except Network.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Network not found'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)

@csrf_exempt  # Temporarily disable CSRF for testing
@require_http_methods(["GET"])
@csrf_exempt
@require_http_methods(["GET"])
def get_cluster_data(request, cluster_id):
    data = get_cluster(cluster_id)
    if not data:
        return JsonResponse({
            'status': 'error',
            'message': f'Cluster with id {cluster_id} not found'
        }, status=404)

    return JsonResponse({
        'status': 'success',
        'message': f'Cluster network data for cluster {cluster_id}',
        'data': data
    })
       
@csrf_exempt  # Temporarily disable CSRF for testing
@require_http_methods(["POST"])
def connect_network_edges(request, network_id):
    try:
        network = Network.objects.get(pk=network_id)
        nodes = list(Node.objects.filter(network=network))
        
        if len(nodes) < 2:
            return JsonResponse({'status': 'error', 'message': 'Network needs at least 2 nodes'})
        
        # Get coordinates for all nodes
        coords = np.array([(node.latitude, node.longitude) for node in nodes])
        
       # Parse JSON data from request body
        data = json.loads(request.body)
        method = data.get('method')
        params = data.get('params', {})
        if method == EdgeAlgorithm.EdgeAlgorithmType.DISTANCE_THRESHOLD:
            # Distance Threshold method
            d_max = float(params.get('d_max', 1000))  # default 1km
            edges_created = connect_by_distance_threshold(nodes, coords, network, d_max)
            
        elif method == EdgeAlgorithm.EdgeAlgorithmType.DELAUNAY:
            # Delaunay Triangulation method
            edges_created = connect_by_delaunay(nodes, coords, network)
            
        elif method == EdgeAlgorithm.EdgeAlgorithmType.KNN:
            # k-Nearest Neighbors method
            k = int(params.get('k', 3))  # default 3 neighbors
            edges_created = connect_by_knn(nodes, coords, network, k)
        elif method == EdgeAlgorithm.EdgeAlgorithmType.MST:
            # Steiner Tree approximation method
            edges_created = model_to_dict(connect_by_mst(nodes, coords, network))
        elif method == EdgeAlgorithm.EdgeAlgorithmType.GABRIEL:
            # Steiner Tree approximation method
            edges_created = connect_by_gabriel(nodes, coords, network)
        elif method == EdgeAlgorithm.EdgeAlgorithmType.BETA_SKELETON:
            # Steiner Tree approximation method
            edges_created = connect_by_beta_skeleton(nodes, coords, network)
        elif method == EdgeAlgorithm.EdgeAlgorithmType.RNG:
            # Steiner Tree approximation method
            edges_created = connect_by_rng(nodes, coords, network)
        elif method == EdgeAlgorithm.EdgeAlgorithmType.TSP:
            # Steiner Tree approximation method
            edges_created = connect_by_tsp(nodes, coords, network)


        else:
            return JsonResponse({'status': 'error', 'message': 'Invalid method specified'})
        
        return JsonResponse({
            'status': 'success',
            'message': f'Successfully created {edges_created} edges using {method}',
            'edges_created': edges_created
        })
        
    except Network.DoesNotExist:
        return JsonResponse({'status': 'error', 'message': 'Network not found'}, status=404)
    except Exception as e:
        return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
#cluster -> means create subnetworks based on distance
def cluster_nodes(network_id: int):
    try:
        network = Network.objects.get(pk=network_id)
    except Network.DoesNotExist:
        print(f"Network with id {network_id} does not exist")
        return 0

    # Create cluster
    cluster = Cluster.objects.create(
        network=network,
        name=f"{network.name} - Cluster",
        description="Cluster created"
    )

    nodes = Node.objects.filter(network=network)
    if not nodes.exists():
        return 0

    # Select all producer nodes
    producers = nodes.filter(node_type='producer')
    if not producers.exists():
        return 0

    lat, lng = calculate_centroid(producers)

    # Create network for producers
    prod_network = Network.objects.create(
        name=f"{network.name} - Producers Cluster",
        network_type=network.network_type,
        project=network.project,
        total_length=0.0,
        parent_network=network,
        cluster=cluster,
        latitude=lat,
        longitude=lng
    )

    clusters = {}
    sub_networks = {}

    producer_nodes = []
    for producer in producers:
        p_node = Node.objects.create(
            network=prod_network,
            name=producer.name,
            latitude=producer.latitude,
            longitude=producer.longitude,
            node_type='producer'
        )
        producer_nodes.append(p_node)
        clusters[producer.name] = []
        sub_networks[producer.name] = {}

    # Connect producer nodes using MST
    edge_algorithm = connect_by_mst(
        producer_nodes,
        [(p.latitude, p.longitude) for p in producer_nodes],
        prod_network
    )
    translate(edge_algorithm, 'transit')

    # Select all consumer nodes
    consumers = nodes.filter(node_type='consumer')
    for consumer in consumers:
        closest_producer = min(
            producers,
            key=lambda p: haversine_distance(
                consumer.latitude, consumer.longitude,
                p.latitude, p.longitude
            ),
            default=None
        )
        if closest_producer:
            clusters[closest_producer.name].append({
                "consumer": consumer,
                "producer": closest_producer
            })

    # Create sub-networks for each producer cluster
    for producer_name, data in clusters.items():
        if not data:
            continue

        consumer_nodes = [entry["consumer"] for entry in data]
        producer = data[0]["producer"]

        lat, lng = calculate_centroid(consumer_nodes)

        sub_network = Network.objects.create(
            name=f"{network.name} - {producer_name} Consumers Cluster",
            network_type=network.network_type,
            project=network.project,
            total_length=0.0,
            parent_network=network,
            cluster=cluster,
            latitude=lat,
            longitude=lng
        )

        sub_networks[producer_name] = {
            "producer": producer,
            "network": sub_network
        }

        c_nodes = []

        # Add producer node to sub-network
        Node.objects.create(
            network=sub_network,
            name=producer.name,
            latitude=producer.latitude,
            longitude=producer.longitude,
            node_type='producer'
        )

        # Add consumer nodes to sub-network
        for consumer in consumer_nodes:
            c_node = Node.objects.create(
                network=sub_network,
                name=consumer.name,
                latitude=consumer.latitude,
                longitude=consumer.longitude,
                node_type='consumer'
            )
            c_nodes.append(c_node)

        # Connect sub-network nodes using MST
        alg = connect_by_mst(
            c_nodes,
            [(n.latitude, n.longitude) for n in c_nodes],
            sub_network
        )
        translate(alg, 'transit')

    # Create main cluster network
    main_cluster = Network.objects.create(
        name=f"{network.name} - Cluster - Network",
        network_type=network.network_type,
        project=network.project,
        total_length=0.0,
        parent_network=network,
        cluster=cluster
    )

    # Add sub-network nodes to main cluster
    for producer_name, item in sub_networks.items():
        if item == {}:
            continue
        sub_net = item["network"]
        producer = item["producer"]

        nodes = []

        # Add sub-network node
        n = Node.objects.create(
            network=main_cluster,
            name=sub_net.name,
            latitude=sub_net.latitude,
            longitude=sub_net.longitude,
            node_type="consumer"
        )
        nodes.append(n)

        # Add producer node
        p_node = Node.objects.create(
            network=main_cluster,
            name=producer.name,
            latitude=producer.latitude,
            longitude=producer.longitude,
            node_type="producer"
        )
        nodes.append(p_node)

        edge_algorithm = connect_by_mst(
            nodes,
            [(n.latitude, n.longitude) for n in nodes],
            main_cluster
        )
        translate(edge_algorithm, 'transit')

    return {"clusters": sub_networks, "meta": cluster}
def translate(edge_algorithm, translation_type = 'transit',
               parameters = {}, description = '', name = '', is_default=False):
        # Validate required fields


    # Get all edges for this algorithm
    edges = Edge.objects.filter(edge_algorithm=edge_algorithm)

    total_edge_length = 0
    created_routes = []
        # Create EdgeTranslation
    edge_translation = EdgeTranslation.objects.create(
        translation=translation_type,
        name=f"{name or translation_type}",
        description=description,
        is_default=is_default,
        parameters=parameters,
        algorithm=edge_algorithm
    )
    r = [];
    for edge in edges:
        # Get coordinates for start and end nodes
        start_coords = (edge.start_node.latitude, edge.start_node.longitude)
        end_coords = (edge.end_node.latitude, edge.end_node.longitude)
        
        isPrimary = edge.start_node.node_type == "producer"
        # Call Google Maps Directions API
        route_data = get_google_maps_route(
            start_coords, 
            end_coords, 
            get_valid_translation_type(translation_type)             
        )
        

        # Create Route
        route = Route.objects.create(
            edge=edge,
            translation=edge_translation,
            polyline=route_data['polyline'], # type: ignore
            distance_meters=route_data['distance'], # type: ignore
            duration_seconds=route_data['duration'], # type: ignore
            route_type = "supply" if isPrimary else "return"
            )
        r.append(route.polyline)
        total_edge_length += route.distance_meters
        created_routes.append(route)
        
        # Add delay to avoid hitting rate limits
        time.sleep(0.1)

    # Update total_edge_length for the translation
    edge_translation.total_edge_length = total_edge_length
    edge_translation.save()
    # Serialize response
    translation_serializer = EdgeTranslationSerializer(edge_translation, many=False)
    route_serializer = RouteSerializer(created_routes, many=True)

    return Response({
        'translations': translation_serializer.data,
        'routes': route_serializer.data,
        'total_edge_length': total_edge_length,
        'message': f'Successfully created 1 translations and {len(created_routes)} routes'
    }, status=status.HTTP_201_CREATED)

def get_google_maps_route(start_coords, end_coords, travel_mode):
    """Fetch route from Google Maps API"""
    try:
        base_url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            'origin': f"{start_coords[0]},{start_coords[1]}",
            'destination': f"{end_coords[0]},{end_coords[1]}",
            'mode': travel_mode,
            'key': "AIzaSyCaXWRoAPR8EvJ8FhqpZRTWMo8aXmPE13g",
            'units': 'metric'
        }
        
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'OK' and data.get('routes'):
            route = data['routes'][0]
            leg = route['legs'][0]
            return {
                'polyline': route['overview_polyline']['points'],
                'distance': leg['distance']['value'],
                'duration': leg['duration']['value']
            }
        return None
    except Exception as e:
        print(f"Error fetching route: {str(e)}")
        return None   
def get_valid_translation_type(translationType):
     # Validate translation type against Google Maps options
    valid_translation_types = {
        #'driving': 'driving',
        'walking': 'walking',
        #'bicycling': 'bicycling',
        'transit': 'transit'
    }
    return valid_translation_types[translationType]
    

def connect_by_distance_threshold(nodes, coords, network, d_max):
    edges_created = 0
    total_length = 0.0
    unique_suffix = str(uuid.uuid4())[:8]
    edge_algorithm = EdgeAlgorithm.objects.create(
        network=network,
        algorithm=EdgeAlgorithm.EdgeAlgorithmType.DISTANCE_THRESHOLD,
        parameters=json.dumps({'d_max': d_max}),
        name=f"{network.name} - Distance Threshold - {unique_suffix}"
    )
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            if distance <= d_max:
                unique_name = f"{node1.name or node1.id}-{node2.name or node2.id}-{i}-{j}"
                # check if edge between these nodes already exists
                if Edge.objects.filter(network=network, start_node=node1, end_node=node2, edge_algorithm=edge_algorithm).exists():
                    # check if edge is shorter than existing edge
                    existing_edge = Edge.objects.get(network=network, start_node=node1, end_node=node2, edge_algorithm=edge_algorithm)
                    if existing_edge.length > distance:
                        #replace existing edge with shorter one
                        edges_created -= 1  # decrement since we're replacing
                        total_length -= existing_edge.length
                        existing_edge.length = distance
                        existing_edge.name = unique_name
                        existing_edge.save(update_fields=['length', 'name'])
            else:
                Edge.objects.create(
                    network=network,
                    start_node=node1,
                    end_node=node2,
                    length=distance,
                    name=unique_name, # type: ignore
                    edge_algorithm=edge_algorithm
                )
            edges_created += 1
            total_length += distance
    edge_algorithm.total_edge_length = total_length
    edge_algorithm.save(update_fields=["total_edge_length"])
    return edge_algorithm
def connect_by_gabriel(nodes, coords, network):
    """
    Connect nodes following Gabriel Graph rules:
    An edge exists between A and B if no other node C is in the circle 
    with diameter AB.
    """
    edges_created = 0
    total_length = 0.0
    unique_suffix = str(uuid.uuid4())[:8]
    
    # Create algorithm metadata record
    edge_algorithm = EdgeAlgorithm.objects.create(
        network=network,
        algorithm=EdgeAlgorithm.EdgeAlgorithmType.GABRIEL,
        parameters=json.dumps({}),
        name=f"{network.name} - Gabriel Graph - {unique_suffix}"
    )
    
    n = len(nodes)
    
    # Pre-compute all coordinates as numpy array for vector operations
    coord_array = np.array(coords)
    
    for i in range(n):
        for j in range(i+1, n):
            node1 = nodes[i]
            node2 = nodes[j]
            a = coord_array[i]
            b = coord_array[j]
            
            # Calculate midpoint and radius
            midpoint = (a + b) / 2
            radius_sq = np.sum((a - b)**2) / 4  # Squared for comparison
            
            # Check Gabriel condition
            valid_edge = True
            for k in range(n):
                if k == i or k == j:
                    continue
                c = coord_array[k]
                if np.sum((c - midpoint)**2) <= radius_sq:
                    valid_edge = False
                    break
            
            if valid_edge:
                distance = haversine_distance(a[0], a[1], b[0], b[1])
                unique_name = f"{node1.name or node1.id}-{node2.name or node2.id}-{i}-{j}"
                
                # Check for existing edge
                existing_edge = Edge.objects.filter(
                    network=network,
                    start_node=node1,
                    end_node=node2,
                    edge_algorithm=edge_algorithm
                ).first()
                
                if existing_edge:
                    if existing_edge.length > distance:
                        # Update if shorter connection found
                        edges_created -= 1
                        total_length -= existing_edge.length
                        existing_edge.length = distance
                        existing_edge.name = unique_name
                        existing_edge.save(update_fields=['length', 'name'])
                else:
                    Edge.objects.create(
                        network=network,
                        start_node=node1,
                        end_node=node2,
                        length=distance,
                        name=unique_name,
                        edge_algorithm=edge_algorithm
                    )
                
                edges_created += 1
                total_length += distance
    
    # Update algorithm metadata
    edge_algorithm.total_edge_length = total_length
    edge_algorithm.save(update_fields=["total_edge_length"])
    
    return edge_algorithm
def connect_by_delaunay(nodes, coords, network):
    edges_created = 0
    total_length = 0.0
    unique_suffix = str(uuid.uuid4())[:8]
    edge_algorithm = EdgeAlgorithm.objects.create(
        network=network,
        algorithm=EdgeAlgorithm.EdgeAlgorithmType.DELAUNAY,
        parameters=json.dumps({}),
        name=f"{network.name} - Delaunay - {unique_suffix}"
    )
    tri = Delaunay(coords)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i+1)%3]
            if a != b:
                edges.add(frozenset({a, b}))
    for edge in edges:
        a, b = sorted(edge)
        node1 = nodes[a]
        node2 = nodes[b]
        lat1, lon1 = coords[a]
        lat2, lon2 = coords[b]
        distance = haversine_distance(lat1, lon1, lat2, lon2)
        unique_name = f"{node1.name or node1.id}-{node2.name or node2.id}-{a}-{b}"
        if Edge.objects.filter(network=network, start_node=node1, end_node=node2, edge_algorithm=edge_algorithm).exists():
            # check if edge is shorter than existing edge
            existing_edge = Edge.objects.get(network=network, start_node=node1, end_node=node2, edge_algorithm=edge_algorithm)
            if existing_edge.length > distance:
                #replace existing edge with shorter one
                edges_created -= 1  # decrement since we're replacing
                total_length -= existing_edge.length
                existing_edge.length = distance
                existing_edge.name = unique_name
                existing_edge.save(update_fields=['length', 'name'])
        else:
            Edge.objects.create(
                network=network,
                start_node=node1,
                end_node=node2,
                length=distance,
                name=unique_name,
                edge_algorithm=edge_algorithm
            )        
        edges_created += 1
        total_length += distance
    edge_algorithm.total_edge_length = total_length
    edge_algorithm.save(update_fields=["total_edge_length"])
    return edge_algorithm
def connect_by_mst(nodes, coords, network):
    """Connect nodes using Minimum Spanning Tree (MST) with similar patterns to Delaunay implementation"""
    edges_created = 0
    total_length = 0.0
    unique_suffix = str(uuid.uuid4())[:8]
    
    # Create edge algorithm record (consistent with Delaunay)
    edge_algorithm = EdgeAlgorithm.objects.create(
        network=network,
        algorithm=EdgeAlgorithm.EdgeAlgorithmType.MST,
        parameters=json.dumps({}),
        name=f"{network.name} - Steiner MST - {unique_suffix}"
    )
    
    # Create a list of all possible edges with their distances
    edges = []
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[j]
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            edges.append((distance, i, j))
    
    # Sort edges by distance (Kruskal's algorithm)
    edges.sort()
    
    # Union-Find data structure for Kruskal's
    parent = list(range(len(nodes)))
    
    def find(u):
        while parent[u] != u:
            parent[u] = parent[parent[u]]  # Path compression
            u = parent[u]
        return u
    
    # Kruskal's algorithm to find MST
    mst_edges = []
    for dist, i, j in edges:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i
            mst_edges.append((i, j, dist))
    
    # Create edges in database following Delaunay pattern
    for i, j, dist in mst_edges:
        node1 = nodes[i]
        node2 = nodes[j]
        unique_name = f"{node1.name or node1.id}-{node2.name or node2.id}-{i}-{j}"
        
        # Check for existing edge (consistent with Delaunay implementation)
        existing_edge = Edge.objects.filter(
            network=network,
            start_node=node1,
            end_node=node2,
            edge_algorithm=edge_algorithm
        ).first()
        
        if existing_edge:
            if existing_edge.length > dist:
                # Replace existing edge with shorter one
                edges_created -= 1
                total_length -= existing_edge.length
                existing_edge.length = dist
                existing_edge.name = unique_name
                existing_edge.save(update_fields=['length', 'name'])
        else:
            Edge.objects.create(
                network=network,
                start_node=node1,
                end_node=node2,
                length=dist,
                name=unique_name,
                edge_algorithm=edge_algorithm
            )
        
        edges_created += 1
        total_length += dist
    
    # Update algorithm metadata (consistent with Delaunay)
    edge_algorithm.total_edge_length = total_length
    edge_algorithm.save(update_fields=["total_edge_length"])
    
    return edge_algorithm

def connect_by_knn(nodes, coords, network, k):
    edges_created = 0
    total_length = 0.0
    unique_suffix = str(uuid.uuid4())[:8]
    edge_algorithm = EdgeAlgorithm.objects.create(
        network=network,
        algorithm=EdgeAlgorithm.EdgeAlgorithmType.KNN,
        parameters=json.dumps({'k': k}),
        name=f"{network.name} - KNN - {unique_suffix}"
    )
    if len(nodes) <= k:
        k = len(nodes) - 1
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    for i, node in enumerate(nodes):
        for neighbor_idx in indices[i][1:]:
            neighbor_node = nodes[neighbor_idx]
            unique_name = f"{node.name or neighbor_node.id}-{neighbor_node.name or node.id}"
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[neighbor_idx]
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            if Edge.objects.filter(network=network, start_node=node, end_node=neighbor_node, edge_algorithm=edge_algorithm).exists():
                    # check if edge is shorter than existing edge
                    existing_edge = Edge.objects.get(network=network, start_node=node, end_node=neighbor_node, edge_algorithm=edge_algorithm)
                    if existing_edge.length > distance:
                        #replace existing edge with shorter one
                        edges_created -= 1  # decrement since we're replacing
                        total_length -= existing_edge.length
                        existing_edge.length = distance
                        existing_edge.name = unique_name
                        existing_edge.save(update_fields=['length', 'name'])
            else:
                Edge.objects.create(
                    network=network,
                    start_node=node,
                    end_node=neighbor_node,
                    length=distance,
                    name=unique_name,
                    edge_algorithm=edge_algorithm
                )
            edges_created += 1
            total_length += distance
    edge_algorithm.total_edge_length = total_length
    edge_algorithm.save(update_fields=["total_edge_length"])
    return edge_algorithm
def create_algorithm_record(network, algorithm_type, params=None):
    return EdgeAlgorithm.objects.create(
        network=network,
        algorithm=algorithm_type,
        parameters=json.dumps(params or {}),
        name=f"{network.name} - {algorithm_type.label} - {uuid.uuid4().hex[:8]}"
    )

def update_algorithm_metrics(algorithm, edge_count, total_length):
    algorithm.total_edge_length = total_length
    algorithm.save(update_fields=['total_edge_length'])
def connect_by_beta_skeleton(nodes, coords, network, beta=1.5):
    algorithm = create_algorithm_record(network, EdgeAlgorithm.EdgeAlgorithmType.BETA_SKELETON, {'beta': beta})
    edges_created = 0
    total_length = 0.0
    n = len(nodes)
    
    for i in range(n):
        for j in range(i+1, n):
            a = np.array(coords[i])
            b = np.array(coords[j])
            dist_ab = np.linalg.norm(a - b)
            valid = True
            
            for k in range(n):
                if k == i or k == j:
                    continue
                c = np.array(coords[k])
                if beta == 1:  # Gabriel
                    midpoint = (a + b) / 2
                    if np.linalg.norm(c - midpoint) <= dist_ab / 2:
                        valid = False
                        break
                else:
                    # β-skeleton lune test
                    angle = np.arccos(np.dot(c-a, b-a)/(np.linalg.norm(c-a)*np.linalg.norm(b-a)))
                    if angle < np.pi/2:
                        radius = dist_ab / (2 * beta)
                        center = a + (b-a)/2 + np.sqrt(radius**2 - (dist_ab/2)**2) * np.array([-(b-a)[1], (b-a)[0]])/dist_ab
                        if np.linalg.norm(c - center) <= radius:
                            valid = False
                            break
            
            if valid:
                dist = haversine_distance(*coords[i], *coords[j])
                if create_edge(network, nodes[i], nodes[j], dist, algorithm):
                    edges_created += 1
                    total_length += dist
    
    update_algorithm_metrics(algorithm, edges_created, total_length)
    return algorithm 
def connect_by_rng(nodes, coords, network):
    algorithm = create_algorithm_record(network, EdgeAlgorithm.EdgeAlgorithmType.RNG)
    edges_created = 0
    total_length = 0.0
    n = len(nodes)
    
    for i in range(n):
        for j in range(i+1, n):
            a = coords[i]
            b = coords[j]
            dist_ab = haversine_distance(*a, *b)
            valid = True
            
            for k in range(n):
                if k == i or k == j:
                    continue
                dist_ak = haversine_distance(*a, *coords[k])
                dist_bk = haversine_distance(*b, *coords[k])
                if dist_ak < dist_ab and dist_bk < dist_ab:
                    valid = False
                    break
            
            if valid:
                if create_edge(network, nodes[i], nodes[j], dist_ab, algorithm):
                    edges_created += 1
                    total_length += dist_ab
    
    update_algorithm_metrics(algorithm, edges_created, total_length)
    return algorithm
def create_edge(network, node1, node2, distance, algorithm, unique_suffix=""):
    unique_name = f"{node1.name or node1.id}-{node2.name or node2.id}-{unique_suffix}"
    edge, created = Edge.objects.update_or_create(
        network=network,
        start_node=node1,
        end_node=node2,
        edge_algorithm=algorithm,
        defaults={
            'length': distance,
            'name': unique_name
        }
    )
    return created
def connect_by_tsp(nodes, coords, network):
    algorithm = create_algorithm_record(network, EdgeAlgorithm.EdgeAlgorithmType.TSP)
    G = nx.Graph()
    
    # Build complete graph
    for i, node in enumerate(nodes):
        G.add_node(i, pos=coords[i])
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            dist = haversine_distance(*coords[i], *coords[j])
            G.add_edge(i, j, weight=dist)
    
    # Christofides approximation
    try:
        tour = nx.approximation.christofides(G)
        edges_created = 0
        total_length = 0.0
        
        for i in range(len(tour)-1): # type: ignore
            dist = G[tour[i]][tour[i+1]]['weight'] # type: ignore
            if create_edge(network, nodes[tour[i]], nodes[tour[i+1]], dist, algorithm): # type: ignore
                edges_created += 1
                total_length += dist
        
        update_algorithm_metrics(algorithm, edges_created, total_length)
        return algorithm
    except:
        return connect_by_mst(nodes, coords, network)  # Fallback
def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # Convert decimal degrees to radians 
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371  # Radius of earth in kilometers
    return c * r * 1000  # Return distance in meters

def calculate_centroid(locations) -> tuple:
    if not locations:
        raise ValueError("Location list is empty")

    total_lat = sum(loc.latitude for loc in locations)
    total_lon = sum(loc.longitude for loc in locations)
    count = len(locations)

    centroid_lat = total_lat / count
    centroid_lon = total_lon / count

    return (centroid_lat, centroid_lon)

# Assume these are correctly defined elsewhere, e.g., as wrappers for polyline.decode/encode
# IMPORTANT: Ensure your decode_polyline returns a list of (lat, lon) tuples, not dictionaries.
# If it returns dictionaries, you'll need to adapt it.
# Let's define a robust one here for clarity:
def decode_polyline(encoded_string):
    try:
        # polyline.decode() from the 'polyline' library correctly returns list of (lat, lon) tuples
        return polyline.decode(encoded_string)
    except Exception as e:
        print(f"DEBUG: Error decoding polyline: {encoded_string[:50]}... Error: {e}")
        return None

def encode_polyline(decoded_coords):
    try:
        # polyline.encode expects list of (lat, lon) tuples
        return polyline.encode(decoded_coords)
    except Exception as e:
        print(f"DEBUG: Error encoding polyline: {e}")
        return None

def combine(routes: list[Route], tolerance: float = 0.00005) -> list[Route] | None:
    """
    Combines polylines from a list of route objects into a single continuous
    polyline and then simplifies it. It modifies the polyline attribute of the
    first route in the list and clears others (or returns a new list).

    Args:
        routes (list[Route]): A list of Route objects, each containing an encoded polyline string.
        tolerance (float): The tolerance for the Douglas-Peucker simplification algorithm.
                           A higher value results in more aggressive simplification.

    Returns:
        list[Route] | None: A new list containing a single Route object with the combined
                            and simplified polyline, or None if input is invalid or an error occurs.
    """
    if not routes:
        print("DEBUG (combine_and_simplify_routes): Input list of routes is empty.")
        return None

    routes.sort(key=lambda x: len(x.polyline), reverse=True)

    all_coords = []
    # Decode each polyline string and collect all coordinates
    for i, route_obj in enumerate(routes):
        decoded_segment_raw = decode_polyline(route_obj.polyline)

        if decoded_segment_raw is None:
            print(f"DEBUG (combine_and_simplify_routes): Warning: Skipping route {i} due to decoding error.")
            continue

        if not decoded_segment_raw:
            print(f"DEBUG (combine_and_simplify_routes): Warning: Route {i} decoded to an empty list of coordinates. Skipping.")
            continue

        # --- FIX STARTS HERE ---
        # Ensure coordinates are (lat, lon) tuples, not dictionaries.
        # This loop transforms them if they are dictionaries.
        decoded_segment_tuples = []
        for point in decoded_segment_raw:
            if isinstance(point, dict) and 'lat' in point and 'lng' in point:
                decoded_segment_tuples.append((point['lat'], point['lng'])) # type: ignore
            elif isinstance(point, (list, tuple)) and len(point) == 2:
                # Assuming it's already (lat, lon) tuple/list
                decoded_segment_tuples.append(tuple(point))
            else:
                print(f"DEBUG (combine_and_simplify_routes): Warning: Unexpected coordinate format in segment {i}: {point}. Skipping point.")
                # You might want to raise an error or handle this more strictly
                continue
        # --- FIX ENDS HERE ---

        # Append coordinates, handling potential duplicate at segment junctions
        if all_coords and all_coords[-1] == decoded_segment_tuples[0]:
            all_coords.extend(decoded_segment_tuples[1:])
        else:
            all_coords.extend(decoded_segment_tuples)

    if not all_coords:
        print("DEBUG (combine_and_simplify_routes): No valid coordinates were collected from the input routes.")
        return None

    print(f"DEBUG (combine_and_simplify_routes): Total combined raw coordinates: {len(all_coords)} points.")
    # Print a few raw coordinates to verify their format
    print(f"DEBUG (combine_and_simplify_routes): Sample raw coords: {all_coords[:5]}")


    try:
        if len(all_coords) < 2:
            print(f"DEBUG (combine_and_simplify_routes): Combined coordinates too few ({len(all_coords)}). Cannot form a LineString for simplification.")
            if len(all_coords) == 1:
                combined_encoded_polyline = encode_polyline(all_coords)
                if combined_encoded_polyline:
                    routes[0].polyline = combined_encoded_polyline
                    return [routes[0]]
                return None
            return None

        # LineString expects (lon, lat) internally, but it can take (lat, lon) if consistently applied.
        # However, the standard way is (x, y) which for geographic is (lon, lat).
        # Let's ensure we convert to (lon, lat) for shapely's constructor for clarity and robustness.
        coords_for_shapely = [(lon, lat) for lat, lon in all_coords]
        line = LineString(coords_for_shapely) # Now this should work!

        print(f"DEBUG (combine_and_simplify_routes): Created LineString. Is it valid? {line.is_valid}. Original points: {len(line.coords)}")

        if not line.is_valid:
            print("DEBUG (combine_and_simplify_routes): Warning: The combined LineString is not valid. This might impact simplification.")

        simplified_line = line.simplify(tolerance, preserve_topology=True)
        print(f"DEBUG (combine_and_simplify_routes): Simplified LineString type: {simplified_line.geom_type}. Simplified points: {len(simplified_line.coords)}")

        if simplified_line.is_empty:
            print("DEBUG (combine_and_simplify_routes): Simplified LineString is empty. Tolerance might be too high.")
            return None
        elif simplified_line.geom_type == 'Point':
            # shapely Point coords are (x, y) which is (lon, lat)
            simplified_coords_for_encoding = [(simplified_line.y, simplified_line.x)] # type: ignore # Convert back to (lat, lon)
            print("DEBUG (combine_and_simplify_routes): Simplified to a single Point.")
        else: # Should be a LineString
            # Convert shapely's (lon, lat) .coords back to polyline's (lat, lon)
            simplified_coords_for_encoding = [(lat, lon) for lon, lat in simplified_line.coords]

        if not simplified_coords_for_encoding:
            print("DEBUG (combine_and_simplify_routes): Simplified coordinates list is empty after processing. Cannot encode.")
            return None
        
        print(f"DEBUG (combine_and_simplify_routes): Final simplified coords for encoding (first 5): {simplified_coords_for_encoding[:5]}")

        combined_encoded_polyline = encode_polyline(simplified_coords_for_encoding)

        if combined_encoded_polyline:
            routes[0].polyline = combined_encoded_polyline
            print(f"DEBUG (combine_and_simplify_routes): Successfully encoded final polyline. Length: {len(combined_encoded_polyline)}")
            return [routes[0]]
        else:
            print("DEBUG (combine_and_simplify_routes): Failed to encode the final simplified polyline.")
            return None

    except Exception as e:
        print(f"DEBUG (combine_and_simplify_routes): An unhandled error occurred during polyline combination/simplification: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_cluster(cluster_id: int, algorithm=1):
    try:
        cluster = Cluster.objects.get(id=cluster_id)
    except Cluster.DoesNotExist:
        return None

    networks = Network.objects.filter(cluster=cluster)
    network_list = []
    first = networks[0]
    last = networks[len(networks) - 1]
    list = [first, last]
    for i in range(len(list)):
        network = list[i]
        nodes_list = []
        routes_list = []
        algorithms = EdgeAlgorithm.objects.filter(network=network)
        for algorithm in algorithms:
            translations = EdgeTranslation.objects.filter(algorithm=algorithm)
            for translation in translations:
                nodes = Node.objects.filter(network=network)
                nodes_list.extend(nodes)  # ✅ flattening
                routes = Route.objects.filter(translation=translation)
                routes_list.extend(routes)  # ✅ flattening

        net = {
            "id": network.id, # type: ignore
            "name": network.name,
            "nodes": [model_to_dict(n) for n in nodes_list],
            "routes": [model_to_dict(r) for r in routes_list]
        }
        network_list.append(net)

    return {
        "cluster": {
            "name": cluster.name
                    },
        "networks": network_list
        }