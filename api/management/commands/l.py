import datetime
import random
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand
from django.apps import apps
from django.db import transaction

# --- Configuration ---
NUM_BUILDINGS = 50
PROFILE_TYPE = 'heating'
# *** We will generate 52 weekly aggregate records for 2024 ***
YEAR = 2024 

class Command(BaseCommand):
    help = f'Generates and loads 52 WEEKLY aggregate load profiles ({PROFILE_TYPE}) for existing buildings ID 1 to {NUM_BUILDINGS} for the year {YEAR}.'

    def handle(self, *args, **options):
        # 1. Dynamically fetch models (assuming your app is named 'api')
        try:
            Building = apps.get_model('api', 'Building')
            LoadProfile = apps.get_model('api', 'LoadProfile')
            # Fetching Project/Country is not needed since buildings already exist
        except LookupError as e:
            self.stdout.write(self.style.ERROR(f"Model lookup failed. Check app name 'api' and ensure models.py is correct: {e}"))
            return

        # 2. Identify the target buildings (IDs 1 through 50)
        target_ids = list(range(1, NUM_BUILDINGS + 1))
        existing_buildings_qs = Building.objects.filter(id__in=target_ids)
        building_ids = list(existing_buildings_qs.values_list('id', flat=True))
        
        if len(building_ids) != NUM_BUILDINGS:
            missing_ids = set(target_ids) - set(building_ids)
            self.stdout.write(self.style.ERROR(
                f"Required {NUM_BUILDINGS} buildings (IDs 1-50) not found. Found {len(building_ids)}. Missing IDs: {sorted(list(missing_ids))}"
            ))
            self.stdout.write(self.style.NOTICE("Please ensure buildings with IDs 1 to 50 exist before running this command."))
            return
        
        self.stdout.write(self.style.SUCCESS(f"Successfully identified {len(building_ids)} buildings (IDs 1-50)."))

        # 3. Generate Load Profile Data (52 weekly periods)
        # Generate weekly time periods for 2024
        start_of_year = datetime.datetime(YEAR, 1, 1, tzinfo=datetime.timezone.utc)
        end_of_year = datetime.datetime(YEAR + 1, 1, 1, tzinfo=datetime.timezone.utc)
        
        # Create weekly date ranges using pandas
        weekly_index = pd.date_range(start=start_of_year, end=end_of_year, freq='W-MON', inclusive='left', tz='utc')
        
        time_periods = []
        for i in range(len(weekly_index)):
            start = weekly_index[i].date()
            # Calculate end date, ensuring we don't exceed the end of the year
            if i + 1 < len(weekly_index):
                end = (weekly_index[i+1] - datetime.timedelta(days=1)).date()
            else:
                end = (end_of_year - datetime.timedelta(days=1)).date()
            time_periods.append({'start_date': start, 'end_date': end})
            
        
        # Simple mock annual load curve (peak in winter, low in summer)
        weeks_of_year = np.arange(len(time_periods))
        # Use a sine wave to simulate seasonality (low in summer/middle of the year, high in winter/ends)
        annual_cycle = np.cos(weeks_of_year * 2 * np.pi / len(time_periods)) * 0.4 + 0.6
        
        all_load_profiles = []
        
        for b_id in building_ids:
            # Base load is proportional to the building's ID (to create variance)
            base_load = (b_id % 10 + 5) * 1000 # Increased base load for weekly/aggregate data
            
            # Combine cycle and add noise
            load_multiplier = annual_cycle * (random.uniform(0.9, 1.1))
            
            # Final mock aggregate load values (e.g., average kW over the week)
            mock_loads = base_load * load_multiplier
            
            # Create LoadProfile objects
            for idx, load_val in enumerate(mock_loads):
                period = time_periods[idx]
                
                load_profile_instance = LoadProfile(
                    building_id=b_id,
                    profile_type=PROFILE_TYPE,
                    start_date=period['start_date'],
                    end_date=period['end_date'],
                    unit='kW', # Using kW as the unit for consistency
                    value=float(max(10, load_val + random.uniform(-50, 50))) # Ensure positive value
                )
                
                all_load_profiles.append(load_profile_instance)

        total_records = len(all_load_profiles)
        self.stdout.write(self.style.NOTICE(f"Total records prepared for insertion: {total_records} (52 weekly records * {NUM_BUILDINGS})"))

        # 4. Insert Data using bulk_create
        self.stdout.write(self.style.NOTICE("Starting bulk insertion..."))
        
        try:
            with transaction.atomic():
                # Delete existing profiles for these buildings/type to prevent duplicates
                LoadProfile.objects.filter(building_id__in=building_ids, profile_type=PROFILE_TYPE).delete()
                
                LoadProfile.objects.bulk_create(all_load_profiles, batch_size=5000)
            self.stdout.write(self.style.SUCCESS(f'Successfully loaded {total_records} {PROFILE_TYPE} weekly load profiles for {YEAR}.'))
        except Exception as e:
            self.stdout.write(self.style.ERROR(f'Bulk insertion failed. Check your database settings and permissions: {e}'))

        # Check final count
        final_count = LoadProfile.objects.filter(profile_type=PROFILE_TYPE).count()
        self.stdout.write(self.style.NOTICE(f'Current total LoadProfile records of type {PROFILE_TYPE}: {final_count}'))