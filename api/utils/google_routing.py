import requests
import time
import os
from rest_framework.response import Response
from rest_framework import status
from api.models import Edge, EdgeTranslation, Route, EdgeConnectionAlgorithmEdge
from api.serializers import EdgeTranslationSerializer, RouteSerializer
from api.utils.route_geometry import unique_route_length


def _physical_total_length(routes, fallback_total):
    return unique_route_length(routes) if routes else fallback_total


def get_valid_translation_type(translationType):
     # Validate translation type against Google Maps options
    valid_translation_types = {
        'driving': 'driving',
        'walking': 'walking',
        'bicycling': 'bicycling'
    }
    return valid_translation_types[translationType]
def translate(edge_algorithm, translation_type = 'transit',
               parameters = {}, description = '', name = '', is_default=False):
        # Validate required fields


    # Get all edges for this algorithm
    edges = Edge.objects.filter(edge_algorithm=edge_algorithm)

    total_edge_length = 0
    created_routes = []
        # Create EdgeTranslation
    edge_translation = EdgeTranslation.objects.create(
        network=edge_algorithm.network,
        translation=translation_type,
        name=f"{name or translation_type}",
        description=description,
        is_default=is_default,
        parameters=parameters,
        algorithm=edge_algorithm
    )
    r = []
    for edge in edges:
        # Get coordinates for start and end nodes
        start_coords = (edge.start_node.latitude, edge.start_node.longitude)
        end_coords = (edge.end_node.latitude, edge.end_node.longitude)
        
        isPrimary = edge.start_node.node_type == "producer"
        # Call Google Maps Directions API
        route_data = get_google_maps_route(
            start_coords, 
            end_coords, 
            translation_type            
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
    total_edge_length = _physical_total_length(created_routes, total_edge_length)
    edge_translation.total_edge_length = total_edge_length
    edge_translation.save()
    # Serialize response
    translation_serializer = EdgeTranslationSerializer(edge_translation, many=False)
    route_serializer = RouteSerializer(created_routes, many=True)

    return {
        'translations': translation_serializer.data,
        'routes': route_serializer.data,
        'total_edge_length': total_edge_length,
        'message': f'Successfully created 1 translations and {len(created_routes)} routes'
    }

def translate2(network, translation_type = 'transit',
               parameters = {}, description = '', name = '', is_default=False):
        # Validate required fields


    # Get all edges for this algorithm
    edges = network.edges

    total_edge_length = 0
    created_routes = []
        # Create EdgeTranslation
    edge_translation = EdgeTranslation(
        translation=translation_type,
        name=f"{name or translation_type}",
        description=description,
        is_default=is_default,
        parameters=parameters,
        edge_connection_algorithm= network.edge_connection_algorithm

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
            translation_type            
        )
        

        # Create Route
        route = Route(
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
    total_edge_length = _physical_total_length(created_routes, total_edge_length)
    edge_translation.total_edge_length = total_edge_length
    network.translation = edge_translation
    # Serialize response
    translation_serializer = EdgeTranslationSerializer(edge_translation, many=False)
    route_serializer = RouteSerializer(created_routes, many=True)

    return {
        'translations': edge_translation,
        'routes': created_routes,
        'total_edge_length': total_edge_length,
        'message': f'Successfully created 1 translations and {len(created_routes)} routes'
    }

def translate3(edges, translation_type = 'transit', edge_algorithmType = None,
               parameters = {}, description = '', name = '', is_default=False):
        # Validate required fields


    total_edge_length = 0
    created_routes = []
    #create ed
    edge_algorithm = EdgeAlgorithm.objects.get(name=edge_algorithmType)
        # Create EdgeTranslation
    edge_translation = EdgeTranslation.objects.create(
        network=edge_connection_algorithm.network,
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
            translation_type            
        )

        if not route_data:
            continue


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
    total_edge_length = _physical_total_length(created_routes, total_edge_length)
    edge_translation.total_edge_length = total_edge_length
    edge_translation.save()
    # Serialize response
    translation_serializer = EdgeTranslationSerializer(edge_translation, many=False)
    route_serializer = RouteSerializer(created_routes, many=True)

    return {
        'translations': translation_serializer.data,
        'routes': route_serializer.data,
        'total_edge_length': total_edge_length,
        'message': f'Successfully created 1 translations and {len(created_routes)} routes'
    }
import time

def translate4(edge_connection_algorithm, translation_type='transit',
              parameters=None, description='', name='', is_default=False):
    
    if parameters is None:
        parameters = {}

    # 1. Fetch edges via the EdgeConnectionAlgorithmEdge mapping table.
    # Using select_related fetches the Edge and its Nodes in a single query,
    # preventing hundreds of separate database hits inside the loop.
    connection_edges = EdgeConnectionAlgorithmEdge.objects.filter(
        edge_connection_algorithm=edge_connection_algorithm
    ).select_related('edge', 'edge__start_node', 'edge__end_node')

    total_edge_length = 0
    created_routes = []
    
    # 2. Create EdgeTranslation
    # Note: Ensure your EdgeTranslation model is updated to link to edge_connection_algorithm 
    # instead of edge_algorithm (or update the kwarg below to match your actual field name)
    edge_translation = EdgeTranslation.objects.create(
        network=edge_connection_algorithm.network,
        translation=translation_type,
        name=f"{name or translation_type}",
        description=description,
        is_default=is_default,
        parameters=parameters,
        edge_connection_algorithm=edge_connection_algorithm 
    )
    
    r = []
    for connection_edge in connection_edges:
        # Extract the actual Edge instance from the mapping model
        edge = connection_edge.edge
        
        # Get coordinates for start and end nodes
        start_coords = (edge.start_node.latitude, edge.start_node.longitude)
        end_coords = (edge.end_node.latitude, edge.end_node.longitude)
        
        isPrimary = edge.start_node.node_type == "producer"
        
        # Call Google Maps Directions API
        route_data = get_google_maps_route(
            start_coords, 
            end_coords, 
            translation_type            
        )

        # Create Route
        route = Route.objects.create(
            edge=edge,
            translation=edge_translation,
            polyline=route_data['polyline'], # type: ignore
            distance_meters=route_data['distance'], # type: ignore
            duration_seconds=route_data['duration'], # type: ignore
            route_type="supply" if isPrimary else "return"
        )
        
        r.append(route.polyline)
        total_edge_length += route.distance_meters
        created_routes.append(route)
        
        # Add delay to avoid hitting rate limits
        time.sleep(0.1)

    # Update total_edge_length for the translation
    total_edge_length = _physical_total_length(created_routes, total_edge_length)
    edge_translation.total_edge_length = total_edge_length
    edge_translation.save(update_fields=['total_edge_length'])
    
    # Serialize response
    translation_serializer = EdgeTranslationSerializer(edge_translation, many=False)
    route_serializer = RouteSerializer(created_routes, many=True)

    return {
        'translations': translation_serializer.data,
        'routes': route_serializer.data,
        'total_edge_length': total_edge_length,
        'message': f'Successfully created 1 translation and {len(created_routes)} routes'
    }
def map_google_type_to_waste_source(google_types, place_name=""):
    """
    Maps Google Place types to F. Albert (2025) waste heat source categories.
    
    :param google_types: list of strings (e.g., ['supermarket', 'store'])
    :param place_name: string (Optional, helps catch Arenas/Data Centres hidden as 'generic')
    :return: string matching Building.WASTE_HEAT_SOURCE_CHOICES or None
    """
    
    # Normalize inputs
    types_set = set(t.lower() for t in google_types)
    name_lower = place_name.lower() if place_name else ""

    # --- 1. DATA CENTRES [Section 4.2.2] ---
    # Google often categorizes these as 'electronics_store' or generic 'point_of_interest'.
    # We must look for keywords in the name or specific rare types.
    if 'data center' in name_lower or 'data centre' in name_lower:
        return 'data_centre'
    
    # --- 2. GROCERY STORES [Section 4.2.3] ---
    # High confidence mapping
    if 'supermarket' in types_set or 'grocery_or_supermarket' in types_set:
        return 'grocery_store'

    # --- 3. COLD STORAGE [Section 4.2.4] ---
    # Often 'storage', 'warehouse', or 'distribution'
    if 'storage' in types_set:
        # Refine by name if possible to avoid self-storage
        if 'cold' in name_lower or 'frozen' in name_lower or 'logistics' in name_lower:
            return 'cold_storage'
        # If ambiguous, might return None or assume generic storage isn't enough.
    
    # --- 4. ICE ARENAS [Section 4.2.5] ---
    # 'stadium' is a strong indicator, but local rinks are often just 'point_of_interest'.
    if 'stadium' in types_set:
        return 'ice_arena'
    
    # Fallback: Check name for keywords
    arena_keywords = ['arena', 'rink', 'skating', 'ice complex', 'community centre', 'community center']
    if any(k in name_lower for k in arena_keywords):
        return 'ice_arena'

    # --- 5. SPACE COOLING [Section 4.2.6] ---
    # Large commercial entities that reject heat via chillers in summer.
    # We prioritize the specific sources above first.
    cooling_sources = [
        'shopping_mall', 
        'department_store', 
        'hospital', 
        'lodging',         # Hotels
        'office',          # Large office buildings
        'university', 
        'school',
        'casino',
        'airport'
    ]
    
    if any(t in types_set for t in cooling_sources):
        return 'space_cooling'

    return None
def get_google_maps_route(start_coords, end_coords, travel_mode, api_key=None):
    """Fetch a route from Google Maps without embedding credentials in source."""
    key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        return None
    try:
        base_url = "https://maps.googleapis.com/maps/api/directions/json"
        params = {
            'origin': f"{start_coords[0]},{start_coords[1]}",
            'destination': f"{end_coords[0]},{end_coords[1]}",
            'mode': travel_mode,
            'key': key,
            'units': 'metric'
        }
        
        response = requests.get(base_url, params=params, timeout=15)
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
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return None   
