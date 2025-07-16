import googlemaps
from django.conf import settings
from datetime import datetime

gmaps = googlemaps.Client(key="AIzaSyBO5jrMOXJIgYHKgb1v0Et6azV_dieaX1I")

def get_road_path(start_point, end_point, mode='driving'):
    """
    Get road path between two points using Google Maps Directions API
    Returns polyline, distance (meters), and duration (seconds)
    """
    try:
        now = datetime.now()
        directions_result = gmaps.directions(
            start_point,
            end_point,
            mode=mode,
            departure_time=now
        )
        
        if not directions_result:
            return None, None, None
            
        route = directions_result[0]['legs'][0]
        polyline = directions_result[0]['overview_polyline']['points']
        distance = route['distance']['value']  # in meters
        duration = route['duration']['value']  # in seconds
        
        return polyline, distance, duration
        
    except Exception as e:
        print(f"Error getting directions: {e}")
        return None, None, None