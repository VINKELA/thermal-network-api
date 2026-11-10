# myapp/management/commands/generate_hourly_profiles.py

from django.core.management.base import BaseCommand
from datetime import datetime, timedelta
import random
from api.models import Building, Boiler, LoadProfile

class Command(BaseCommand):
    help = "Generate hourly heat load for buildings and waste heat for boilers (Nov 2024 – Feb 2025)"

    def handle(self, *args, **options):
        buildings = Building.objects.all()
        boilers = Boiler.objects.all()

        if not buildings.exists() and not boilers.exists():
            self.stdout.write(self.style.WARNING("No buildings or boilers found in the database."))
            return

        for building in buildings:
            if building.heating_demand > 0:
                self.stdout.write(self.style.SUCCESS(f"\n🏢 Processing Building: {building.name}"))
                self.generate_hourly_heat_load(building)

        for boiler in boilers:
            self.stdout.write(self.style.SUCCESS(f"\n🔥 Processing Boiler: {boiler.name}"))
            self.generate_hourly_waste_heat(boiler)

        self.stdout.write(self.style.SUCCESS("\n✅ Hourly profiles generated for all assets."))

    def generate_hourly_heat_load(self, building):
        start_date = datetime(2024, 11, 1)
        end_date = datetime(2025, 2, 28, 23)

        current_time = start_date
        while current_time <= end_date:
            heat_load = self.calc_heat_load(building, current_time)
            LoadProfile.objects.create(
                building=building,
                profile_type="heat_load",
                value=heat_load,
                unit="kW",
                start_date=current_time,
                end_date=current_time
            )
            current_time += timedelta(hours=1)

    def generate_hourly_waste_heat(self, boiler):
        start_date = datetime(2024, 11, 1)
        end_date = datetime(2025, 2, 28, 23)

        current_time = start_date
        while current_time <= end_date:
            waste_heat = self.calc_boiler_output(boiler, current_time)
            LoadProfile.objects.create(
                boiler=boiler,  # assumes LoadProfile has FK to Boiler
                profile_type="waste_heat",
                value=waste_heat,
                unit="kW",
                start_date=current_time,
                end_date=current_time
            )
            current_time += timedelta(hours=1)

    def calc_heat_load(self, building, dt):
        base = building.heating_demand
        month_factor = {11: 0.9, 12: 1.0, 1: 1.1, 2: 1.0}.get(dt.month, 1)
        hour_factor = 1.2 if 6 <= dt.hour <= 9 or 17 <= dt.hour <= 20 else 0.8
        noise = random.uniform(0.9, 1.1)
        return round(base * month_factor * hour_factor * noise, 2)

    def calc_boiler_output(self, boiler, dt):
        month_factor = {11: 0.85, 12: 1.0, 1: 1.05, 2: 1.0}.get(dt.month, 1)
        hour_factor = 1.1 if 6 <= dt.hour <= 9 or 17 <= dt.hour <= 20 else 0.7
        noise = random.uniform(0.95, 1.05)
        return round(boiler.capacity * boiler.efficiency * month_factor * hour_factor * noise, 2)