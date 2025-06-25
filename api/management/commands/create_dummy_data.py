# myapp/management/commands/create_mcmaster_infrastructure.py
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime, timedelta
from api.models import (
    Country, Province, City, Project, Network,
    Building, Pump, Valve, HeatPump, ThermalStorage
)

class Command(BaseCommand):
    help = 'Creates McMaster thermal network project with buildings and infrastructure (excluding nodes/edges)'

    def handle(self, *args, **options):
        # 1. Create geographic hierarchy
        canada = self.create_country()
        ontario = self.create_province(canada)
        hamilton = self.create_city(ontario, canada)
        
        # 2. Create project
        project = self.create_project(canada, ontario, hamilton)
        
        # 3. Create network
        network = self.create_network(project)
        
        # 4. Create buildings
        self.create_buildings(project)
        
        # 5. Create network infrastructure
        self.create_pumps(project)
        self.create_valves(project)
        self.create_heat_pumps(project)
        self.create_thermal_storage(project)
        
        self.stdout.write(self.style.SUCCESS('\nSuccessfully created all infrastructure components'))

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

    def create_network(self, project):
        network, created = Network.objects.get_or_create(
            project=project,
            name="McMaster Central Heating Network",
            description="Primary district heating network connecting key buildings on campus",
            network_type="district_heating"
        )
        if created:
            self.stdout.write(self.style.SUCCESS('Created Network'))
        return network

    def create_buildings(self, project):
        buildings_data = [
            # Rector's building (source)
            {
                "name": "University Hall (Rector's Office)",
                "building_type": "public",
                "latitude": 43.2627,
                "longitude": -79.9190,
                "floor_area": 15000,
                "heating_demand": 1200,
                "cooling_demand": 800
            },
            # Other buildings
            {
                "name": "John Hodgins Engineering Building",
                "building_type": "public",
                "latitude": 43.2609,
                "longitude": -79.9194,
                "floor_area": 25000,
                "heating_demand": 1800,
                "cooling_demand": 1200
            },
            {
                "name": "Health Sciences Centre",
                "building_type": "public",
                "latitude": 43.2603,
                "longitude": -79.9168,
                "floor_area": 45000,
                "heating_demand": 3000,
                "cooling_demand": 2000
            },
            {
                "name": "Mills Memorial Library",
                "building_type": "public",
                "latitude": 43.2635,
                "longitude": -79.9174,
                "floor_area": 30000,
                "heating_demand": 2200,
                "cooling_demand": 1500
            },
            {
                "name": "Burke Science Building",
                "building_type": "public",
                "latitude": 43.2619,
                "longitude": -79.9202,
                "floor_area": 18000,
                "heating_demand": 1400,
                "cooling_demand": 900
            }
        ]

        for data in buildings_data:
            building, created = Building.objects.get_or_create(
                project=project,
                **data
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created Building: {building.name}'))

    def create_pumps(self, project):
        pumps_data = [
            {
                "name": "Main Circulation Pump",
                "pump_type": "circulation",
                "latitude": 43.2616,
                "longitude": -79.9186,
                "flow_rate": 500,
                "head": 30,
                "power": 75,
                "efficiency": 85,
                "status": "operational"
            },
            {
                "name": "Engineering Branch Pump",
                "pump_type": "booster",
                "latitude": 43.2611,
                "longitude": -79.9197,
                "flow_rate": 200,
                "head": 20,
                "power": 30,
                "efficiency": 80,
                "status": "operational"
            }
        ]

        for data in pumps_data:
            pump, created = Pump.objects.get_or_create(
                project=project,
                **data
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created Pump: {pump.name}'))

    def create_valves(self, project):
        valves_data = [
            {
                "name": "Main Isolation Valve",
                "valve_type": "gate",
                "latitude": 43.2615,
                "longitude": -79.9184,
                "diameter": 300,
                "status": "open"
            },
            {
                "name": "Engineering Zone Valve",
                "valve_type": "ball",
                "latitude": 43.2610,
                "longitude": -79.9196,
                "diameter": 200,
                "status": "open"
            }
        ]

        for data in valves_data:
            valve, created = Valve.objects.get_or_create(
                project=project,
                **data
            )
            if created:
                self.stdout.write(self.style.SUCCESS(f'Created Valve: {valve.name}'))

    def create_heat_pumps(self, project):
        heat_pump, created = HeatPump.objects.get_or_create(
            project=project,
            name="Central Heat Pump Station",
            heat_pump_type="water_source",
            latitude=43.2620,
            longitude=-79.9180,
            heating_capacity=5000,
            cooling_capacity=3000,
            cop=3.5,
            status="operational"
        )
        if created:
            self.stdout.write(self.style.SUCCESS(f'Created Heat Pump: {heat_pump.name}'))

    def create_thermal_storage(self, project):
        storage, created = ThermalStorage.objects.get_or_create(
            project=project,
            name="Central Thermal Storage Tank",
            storage_type="hot_water",
            latitude=43.2618,
            longitude=-79.9182,
            capacity=20000,  # kWh
            volume=500,      # m³
            max_temperature=90,
            min_temperature=40,
            insulation_type="polyurethane",
            status="operational"
        )
        if created:
            self.stdout.write(self.style.SUCCESS(f'Created Thermal Storage: {storage.name}'))