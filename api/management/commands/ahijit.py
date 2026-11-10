# myapp/management/commands/create_burlington_network.py

import pandas as pd
from django.core.management.base import BaseCommand
from datetime import datetime, timedelta
from api.models import Country, Province, City, Project, Building, Boiler
from api.views import estimate_peak_load

class Command(BaseCommand):
    help = 'Creates Burlington thermal network project with buildings and a boiler'

    def handle(self, *args, **options):
        # 1. Create geographic hierarchy
        canada = self.create_country()
        ontario = self.create_province(canada)
        burlington = self.create_city(ontario, canada)
        # 2. Create project
        project = self.create_project(canada, ontario, burlington)

        # 3. Create buildings from CSV (with FSA)
        #self.create_buildings(project, f"C:\\Users\\orjik\\source\\thermal-network-api\\individualBuildings\\500.csv")

        self.stdout.write(self.style.SUCCESS(
            '\n✅ Successfully created Burlington Thermal Network with buildings and boiler'
        ))

    def create_country(self):
        country, _ = Country.objects.get_or_create(
            name="Canada",
            code="CAN",
            latitude=56.1304,
            longitude=-106.3468
        )
        return country

    def create_province(self, country):
        province, _ = Province.objects.get_or_create(
            name="Ontario",
            country=country,
            latitude=51.2538,
            longitude=-85.3232
        )
        return province

    def create_city(self, province, country):
        city, _ = City.objects.get_or_create(
            name="Burlington",
            province=province,
            country=country,
            latitude=43.3255,
            longitude=-79.7990,
            population=186948  # approx
        )
        return city

    def create_project(self, country, province, city):
        project, _ = Project.objects.get_or_create(
            name="Burlington Thermal Network",
            description="District heating system for Burlington buildings",
            country=country,
            province=province,
            city=city,
            start_date=datetime.now().date(),
            end_date=datetime.now().date() + timedelta(days=365*5),
            budget=25000000.00,
            status="in_progress"
        )
        return project

    def create_buildings(self, project, csv_path):
        """
        Load buildings from CSV (with FSA) and create Building objects.
        Expected columns: BUILDINGID, BUILDINGTY, Latitude, Longitude, EffArea, CFSAUID
        """
        df = pd.read_csv(csv_path)
        #df = df.nlargest(100, 'Burlington')
     
        
        for _, row in df.iterrows():
            b_type  = 'residential' if row["BUILDINGTY"] == 'RESIDENTIAL' else 'commercial'
            building_data = {
                "name": str(row.get("BUILDINGID", f"Building-{row.name}")),
                "building_type": b_type,
                "latitude": row["Latitude"],
                "longitude": row["Longitude"],
                "floor_area": row.get("EffArea", 0),
                "heating_demand": row.get("Burlington", 0),
                "cooling_demand": 0,
                "fsa": row.get("CFSAUID", None),  # requires fsa field in Building model
                "waste_heat_source_type": 'space_cooling' if b_type == 'commercial' else None,
                "load_type": 'heating_dominant' if b_type == 'residential' else 'cooling_dominant'


            }

            building, created = Building.objects.get_or_create(
                project=project,
                name=building_data["name"],
                defaults=building_data
            )
            if created:
                self.stdout.write(self.style.SUCCESS(
                    f'Created Building: {building.name} (FSA: {building.fsa}) at {building.latitude}, {building.longitude}'
                ))
        


