from django.core.management.base import BaseCommand
from api.models import Node, Edge, EdgeTranslation, Route, Network
from api.utils.gmaps import get_road_path

class Command(BaseCommand):
    help = 'Translate GIS coordinates to road paths using Google Maps API'

    def handle(self, *args, **options):
        # Example usage - you would typically get these from your database
        network = Network.objects.first()
        if not network:
            self.stdout.write(self.style.ERROR('No networks found. Create a network first.'))
            return

        # Create or get nodes
        start_point = (40.7128, -74.0060)  # New York coordinates
        end_point = (34.0522, -118.2437)  # Los Angeles coordinates

        start_node = Node.objects.create(
            network=network,
            latitude=start_point[0],
            longitude=start_point[1],
            node_type='supply'
        )

        end_node = Node.objects.create(
            network=network,
            latitude=end_point[0],
            longitude=end_point[1],
            node_type='consumer'
        )

        # Create edge between nodes
        edge = Edge.objects.create(
            network=network,
            start_node=start_node,
            end_node=end_node,
            length=0,  # Will be updated
            is_default=True
        )

        # Get road path
        polyline, distance, duration = get_road_path(start_point, end_point)
        
        if polyline:
            # Update edge length
            edge.length = distance
            edge.save()

            # Create edge translation - NOW CORRECTLY REFERENCING THE EDGE
            edge_translation = EdgeTranslation.objects.create(
                edge=edge,  # Fixed - now referencing the Edge instance
                translation='driving',
                name='Driving Route',
                is_default=True
            )

            # Create route
            Route.objects.create(
                edge=edge,
                translation=edge_translation,
                polyline=polyline,
                distance_meters=distance,
                duration_seconds=duration
            )

            self.stdout.write(self.style.SUCCESS(f'Successfully created route with distance: {distance}m and duration: {duration}s'))
        else:
            self.stdout.write(self.style.ERROR('Failed to get route from Google Maps API'))