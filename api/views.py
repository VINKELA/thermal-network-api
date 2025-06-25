from rest_framework import viewsets
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
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
    JunctionSerializer
)

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
                description="Automatically generated network using Steiner tree approximation"
            )
            
            # Get all infrastructure points for this project
            buildings = Building.objects.filter(project=project)
            pumps = Pump.objects.filter(project=project)
            valves = Valve.objects.filter(project=project)
            heat_pumps = HeatPump.objects.filter(project=project)
            
            # Combine all infrastructure points
            infrastructure_points = []
            infrastructure_points.extend([(b.id, b.latitude, b.longitude, 'building') for b in buildings])
            infrastructure_points.extend([(p.id, p.latitude, p.longitude, 'pump') for p in pumps])
            infrastructure_points.extend([(v.id, v.latitude, v.longitude, 'valve') for v in valves])
            infrastructure_points.extend([(hp.id, hp.latitude, hp.longitude, 'heat_pump') for hp in heat_pumps])
            
            if len(infrastructure_points) < 2:
                return Response(
                    {"error": "Not enough infrastructure points to create a network (minimum 2 required)"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            
            # Create nodes for all infrastructure points
            nodes = []
            for point_id, lat, lng, point_type in infrastructure_points:
                node = Node.objects.create(
                    network=network,
                    name=f"{point_type.capitalize()} {point_id}",
                    latitude=lat,
                    longitude=lng,
                    node_type='consumer' if point_type == 'building' else 'junction'
                )
                nodes.append((node.id, lat, lng))
            
            # Create a complete graph with all possible edges
            G = nx.Graph()
            
            # Add nodes to graph
            for node_id, lat, lng in nodes:
                G.add_node(node_id, pos=(lat, lng))
            
            # Add edges with weights (distance between nodes)
            for i, (node1_id, lat1, lng1) in enumerate(nodes):
                for j, (node2_id, lat2, lng2) in enumerate(nodes):
                    if i < j:  # Avoid duplicate edges
                        # Calculate haversine distance between points
                        dist = haversine(lat1, lng1, lat2, lng2)
                        G.add_edge(node1_id, node2_id, weight=dist)
            
            # Find approximate Steiner tree (minimum spanning tree as approximation)
            mst = nx.minimum_spanning_tree(G)
            
            # Create edges in database
            total_length = 0
            for edge in mst.edges():
                node1 = Node.objects.get(pk=edge[0])
                node2 = Node.objects.get(pk=edge[1])
                
                # Calculate actual edge length
                edge_length = haversine(
                    node1.latitude, node1.longitude,
                    node2.latitude, node2.longitude
                )
                total_length += edge_length
                
                Edge.objects.create(
                    network=network,
                    start_node=node1,
                    end_node=node2,
                    length=edge_length,
                    name=f"Edge {node1.name} to {node2.name}"
                )
            
            # Update network total length
            network.total_length = total_length
            network.save()
            
            return Response({
                "message": f"Network created successfully with {len(nodes)} nodes and {len(mst.edges())} edges",
                "network_id": network.id,
                "total_length": total_length
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