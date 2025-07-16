# myapp/management/commands/create_mcmaster_buildings.py
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime, timedelta
from api.models import (
    Country, Province, City, Project, Building
)

class Command(BaseCommand):
    help = 'Creates McMaster thermal network project with buildings only'

    def handle(self, *args, **options):
        # 1. Create geographic hierarchy
        canada = self.create_country()
        ontario = self.create_province(canada)
        hamilton = self.create_city(ontario, canada)
        
        # 2. Create project
        project = self.create_project(canada, ontario, hamilton)
        
        # 3. Create buildings with accurate GIS coordinates
        self.create_buildings(project)
        
        self.stdout.write(self.style.SUCCESS('\nSuccessfully created all buildings with accurate GIS coordinates'))

    def create_country(self):
        country, created = Country.objects.get_or_create(
            name="Canada",
            code="CAN",
            latitude=56.1304,
            longitude=-106.3468
        )
        if created:
            self.stdout.write(self.style.SUCCESS('Created Canada'))
        return country

    def create_province(self, country):
        province, created = Province.objects.get_or_create(
            name="Ontario",
            country=country,
            latitude=51.2538,
            longitude=-85.3232
        )
        if created:
            self.stdout.write(self.style.SUCCESS('Created Ontario'))
        return province

    def create_city(self, province, country):
        city, created = City.objects.get_or_create(
            name="Hamilton",
            province=province,
            country=country,
            latitude=43.2557,
            longitude=-79.8711,
            population=579200
        )
        if created:
            self.stdout.write(self.style.SUCCESS('Created Hamilton'))
        return city

    def create_project(self, country, province, city):
        project, created = Project.objects.get_or_create(
            name="McMaster University Thermal Network",
            description="District heating system for McMaster University campus",
            country=country,
            province=province,
            city=city,
            start_date=datetime.now().date(),
            end_date=datetime.now().date() + timedelta(days=365*5),
            budget=25000000.00,
            status="in_progress"
        )
        if created:
            self.stdout.write(self.style.SUCCESS('Created Project'))
        return project

    def create_buildings(self, project):
        buildings_data = [
            # Nuclear Reactor (heat source) - Actual location
            {
                "name": "McMaster Nuclear Reactor (MNR)",
                "building_type": "industrial",
                "latitude": 43.261251236183504,
                "longitude":  -79.92157852735507,
                "floor_area": 5000,
                "heating_demand": 0,  # Producer
                "cooling_demand": 0
            },
            # Restaurants
            {
                "name": "Phoenix Craft House and Grill",
                "building_type": "commercial",
                "latitude": 43.2629541330204, 
                "longitude": -79.92068808183345,
                "floor_area": 2500,
                "heating_demand": 350,
                "cooling_demand": 200
            },
            {
                "name": "Bistro in Mary Keyes Residence",
                "building_type": "commercial",
                "latitude": 43.26231520883189, 
                "longitude": -79.9222195588532,
                "floor_area": 1800,
                "heating_demand": 280,
                "cooling_demand": 150
            },
            # Health Sciences Complex
            {
                "name": "Health Sciences Centre (HSC)",
                "building_type": "public",
                "latitude": 43.260336751767895, 
                "longitude": -79.91728704080425,
                "floor_area": 45000,
                "heating_demand": 2800,
                "cooling_demand": 3200
            },
            # Engineering Buildings
            {
                "name": "John Hodgins Engineering Building (JHE)",
                "building_type": "public",
                "latitude": 43.260657094019884, 
                "longitude": -79.92023747071741,
                "floor_area": 32000,
                "heating_demand": 2100,
                "cooling_demand": 2400
            },
            {
                "name": "McMaster Museum of Art",
                "building_type": "public",
                "latitude": 43.262824359990425, 
                "longitude": -79.91727077197686,
                "floor_area": 18000,
                "heating_demand": 1400,
                "cooling_demand": 1600
            },
          
            # Athletic Facilities
            {
                "name": "David Braley Athletic Centre (DBAC)",
                "building_type": "public",
                "latitude":43.26508984613477, 
                "longitude": -79.91659951615942,
                "floor_area": 28000,
                "heating_demand": 1900,
                "cooling_demand": 1800
            },
            # Science Buildings
            {
                "name": "Burke Science Building (BSB)",
                "building_type": "public",
                "latitude": 43.262234653258815, 
                "longitude": -79.9202102900079,
                "floor_area": 35000,
                "heating_demand": 2200,
                "cooling_demand": 2500
            },
            {
                "name": "Arthur Bourns Building (ABB)",
                "building_type": "public",
                "latitude": 43.260346100438895, 
                "longitude": -79.92232437540929,
                "floor_area": 25000,
                "heating_demand": 1800,
                "cooling_demand": 2000
            }
        ]

        for data in buildings_data:
            building, created = Building.objects.get_or_create(
                project=project,
                **data
            )
            if created:
                self.stdout.write(self.style.SUCCESS(
                    f'Created Building: {building.name} at {building.latitude}, {building.longitude}'
                ))
