import json
import time
from django.conf import settings
import requests
from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.exceptions import NotFound
from .models import *
from .serializers import *
from .models import Project, Network, Node, Edge, Building, Pump, Valve, HeatPump
from math import sqrt
import networkx as nx
from scipy.spatial import distance
import numpy as np
from django.db import transaction
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.db.models import Prefetch
from .models import Network, Node, Edge, Junction
from .serializers import (
    NetworkSerializer,
    NodeSerializer,
    EdgeSerializer,
    JunctionSerializer,
    TranslationWithAlgorithmSerializer  # Added import for missing serializer
)
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
class RouteViewSet(viewsets.ModelViewSet):
    queryset = Route.objects.all()
    serializer_class = RouteSerializer


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
            nodes_data = NodeSerializer(network.node_set.all(), many=True).data
            edges_data = EdgeSerializer(network.edge_set.all(), many=True).data
            
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
            infrastructure_points.extend([(b.id, b.latitude, b.longitude, 'building', b.name, b.building_type) for b in buildings])
            infrastructure_points.extend([(p.id, p.latitude, p.longitude, 'pump', p.name,"") for p in pumps])
            infrastructure_points.extend([(v.id, v.latitude, v.longitude, 'valve', v.name, "") for v in valves])
            infrastructure_points.extend([(hp.id, hp.latitude, hp.longitude, 'heat_pump', hp.name, "") for hp in heat_pumps])
            
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
                nodes.append((node.id, lat, lng))
            
            # Create a complete graph with all possible edges
            G = nx.Graph()
            
            # Add nodes to graph
            for node_id, lat, lng in nodes:
                G.add_node(node_id, pos=(lat, lng))
            
            network.save()
            
            return Response({
                "message": f"Network created successfully with {len(nodes)} nodes",
                "network_id": network.id,
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
        
        # Validate required fields
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
            'driving': 'driving',
            'walking': 'walking',
            'bicycling': 'bicycling',
            'transit': 'transit'
        }
        
        if translation_type not in valid_translation_types:
            return Response(
                {'error': f'Invalid translation type. Valid types are: {", ".join(valid_translation_types.keys())}'},
                status=status.HTTP_400_BAD_REQUEST
            )
        
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
        
        for edge in edges:
            # Get coordinates for start and end nodes
            start_coords = (edge.start_node.latitude, edge.start_node.longitude)
            end_coords = (edge.end_node.latitude, edge.end_node.longitude)
            
            # Call Google Maps Directions API
            route_data = self._get_google_maps_route(
                start_coords, 
                end_coords, 
                valid_translation_types[translation_type]
            )
            

            # Create Route
            route = Route.objects.create(
                edge=edge,
                translation=edge_translation,
                polyline=route_data['polyline'],
                distance_meters=route_data['distance'],
                duration_seconds=route_data['duration']
            )
            
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
    
    def _get_google_maps_route(self, start_coords, end_coords, travel_mode):
        """
        Get route data from Google Maps Directions API.
        
        Args:
            start_coords: Tuple of (latitude, longitude)
            end_coords: Tuple of (latitude, longitude)
            travel_mode: One of 'driving', 'walking', 'bicycling', or 'transit'
            
        Returns:
            Dictionary with polyline, distance (meters), and duration (seconds)
            or None if API call fails
        """
        try:
            base_url = "https://maps.googleapis.com/maps/api/directions/json"
            
            params = {
                'origin': f"{start_coords[0]},{start_coords[1]}",
                'destination': f"{end_coords[0]},{end_coords[1]}",
                'mode': travel_mode,
                'key': "AIzaSyBO5jrMOXJIgYHKgb1v0Et6azV_dieaX1I",  # Replace with your actual API key
                'units': 'metric'
            }
            
            # Add transit-specific parameters if needed
            if travel_mode == 'transit':
                params.update({
                    'transit_mode': 'bus|train|subway|tram',
                    'alternatives': 'false'
                })
            
            response = requests.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if data['status'] == 'OK' and data.get('routes'):
                route = data['routes'][0]
                leg = route['legs'][0]
                
                return {
                    'polyline': route['overview_polyline']['points'],
                    'distance': leg['distance']['value'],  # in meters
                    'duration': leg['duration']['value']   # in seconds
                }
            
            print(f"Google Maps API error: {data.get('status')}, {data.get('error_message', 'No error message')}")
            return None
            
        except Exception as e:
            print(f"Error getting route from Google Maps API: {str(e)}")
            return None
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
    def get(self, request, network_id, translation_id):
        # Verify network exists
        try:
            network = Network.objects.get(pk=network_id)
        except Network.DoesNotExist:
            raise NotFound("Network not found")

        # Get translation with prefetched algorithm and verify it belongs to this network
        try:
            translation = EdgeTranslation.objects.select_related('algorithm').get(
                pk=translation_id,
                algorithm__network=network  # Ensure translation's algorithm belongs to network
            )
        except EdgeTranslation.DoesNotExist:
            raise NotFound("Translation not found for this network")

        # Get all routes for this translation with prefetched edge and nodes
        routes = Route.objects.filter(
            translation=translation,
            edge__network=network  # Ensure edge belongs to this network
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
    return edges_created

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
    return edges_created

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
            lat1, lon1 = coords[i]
            lat2, lon2 = coords[neighbor_idx]
            distance = haversine_distance(lat1, lon1, lat2, lon2)
            Edge.objects.create(
                network=network,
                start_node=node,
                end_node=neighbor_node,
                length=distance,
                name=f"{node.name or node.id}-{neighbor_node.name or neighbor_node.id}",
                edge_algorithm=edge_algorithm
            )
            edges_created += 1
            total_length += distance
    edge_algorithm.total_edge_length = total_length
    edge_algorithm.save(update_fields=["total_edge_length"])
    return edges_created
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