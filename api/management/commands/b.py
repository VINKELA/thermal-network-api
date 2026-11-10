# myapp/management/commands/create_burlington_fsa_network.py

from django.core.management.base import BaseCommand
from datetime import datetime, timedelta
from api.models import Country, Province, City, Project, Building

class Command(BaseCommand):
    help = 'Creates Burlington thermal network aggregated buildings by FSA with summed heating/cooling loads'

    def handle(self, *args, **options):
        # 1. Geographic hierarchy
        canada = self.create_country()
        ontario = self.create_province(canada)
        burlington = self.create_city(ontario, canada)

        # 2. Project
        project = self.create_project(canada, ontario, burlington)

        # 3. Create FSA aggregated buildings
        self.create_fsa_aggregated_buildings(project)

        self.stdout.write(self.style.SUCCESS(
            '\n✅ Successfully created Burlington FSA aggregated buildings for the thermal network'
        ))

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
            name="Burlington",
            province=province,
            country=country,
            latitude=43.3255,
            longitude=-79.7990,
            population=186948  # 2021 Census, replace as needed
        )
        if created:
            self.stdout.write(self.style.SUCCESS('Created Burlington'))
        return city

    def create_project(self, country, province, city):
        project, created = Project.objects.get_or_create(
            name="Burlington FSA Thermal Network",
            description="District heating/cooling: aggregated peak loads by Burlington FSAs",
            country=country,
            province=province,
            city=city,
            start_date=datetime.now().date(),
            end_date=datetime.now().date() + timedelta(days=365*3),
            budget=18000000.00,
            status="planning"
        )
        if created:
            self.stdout.write(self.style.SUCCESS('Created Project'))
        return project

    def create_fsa_aggregated_buildings(self, project):
        # --- Assumptions (tune these with real data) ---
        avg_household_size = 2.6            # people per household
        avg_dwelling_area_m2 = 120          # m² per household
        nonres_area_per_capita_m2 = 15      # m² per person (commercial/institutional/industrial)
        peak_heat_intensity_W_m2 = 60       # W/m²
        peak_cool_intensity_W_m2 = 45       # W/m²

        # --- Burlington FSAs: populations and centroid coords (approximate) ---
        # Replace populations with real counts and coords with polygon centroids for accuracy.
        fsas = [
            {
                "fsa": "L7L",  # Southeast / Appleby
                "population": 40000,
                "lat": 43.3610,
                "lon": -79.7600
            },
            {
                "fsa": "L7M",  # North-central (Tansley/Orchard)
                "population": 50000,
                "lat": 43.3830,
                "lon": -79.8170
            },
            {
                "fsa": "L7N",  # South-central (Roseland/Longmoor)
                "population": 20000,
                "lat": 43.3470,
                "lon": -79.7810
            },
            {
                "fsa": "L7P",  # Northwest (Brant Hills/Headon Forest)
                "population": 35000,
                "lat": 43.3690,
                "lon": -79.8350
            },
            {
                "fsa": "L7R",  # Downtown core
                "population": 10000,
                "lat": 43.3210,
                "lon": -79.8010
            },
            {
                "fsa": "L7S",  # Lakeshore corporate/condos
                "population": 5000,
                "lat": 43.3180,
                "lon": -79.7920
            },
            {
                "fsa": "L7T",  # Aldershot/Plains West
                "population": 26000,
                "lat": 43.3110,
                "lon": -79.8530
            },
        ]

        total_pop = sum(x["population"] for x in fsas)
        # Optional sanity check log
        self.stdout.write(self.style.WARNING(f'Using total Burlington population (FSAs sum): {total_pop}'))

        for item in fsas:
            pop = item["population"]

            # Residential floor area estimate
            households = pop / avg_household_size
            residential_floor_m2 = households * avg_dwelling_area_m2

            # Non-residential floor area estimate
            nonres_floor_m2 = pop * nonres_area_per_capita_m2

            # Total conditioned floor area
            total_floor_m2 = residential_floor_m2 + nonres_floor_m2

            # Peak loads (kW)
            peak_heat_kW = (total_floor_m2 * peak_heat_intensity_W_m2) / 1000.0
            peak_cool_kW = (total_floor_m2 * peak_cool_intensity_W_m2) / 1000.0

            building_data = {
                "name": f"{item['fsa']} Aggregated Thermal Load",
                "building_type": "aggregated_fsa",
                "latitude": item["lat"],
                "longitude": item["lon"],
                "floor_area": int(total_floor_m2),
                "heating_demand": int(peak_heat_kW),
                "cooling_demand": int(peak_cool_kW),
            }

            building, created = Building.objects.get_or_create(
                project=project,
                **building_data
            )

            if created:
                self.stdout.write(self.style.SUCCESS(
                    f"Created FSA {item['fsa']}: floor {building.floor_area} m², heat {building.heating_demand} kW, cool {building.cooling_demand} kW"
                ))
            else:
                self.stdout.write(self.style.WARNING(
                    f"FSA {item['fsa']} already exists; skipped."
                ))