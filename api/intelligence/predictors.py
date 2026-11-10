import numpy as np
import calendar
from datetime import datetime, timedelta
from django.utils import timezone 
import requests
from .heating_models import MODEL_CATALOG
from datetime import datetime
from api.models import *
from django.db import transaction

class BuildingEstimator:
# Registry of available models for the Frontend to list
    AVAILABLE_MODELS = MODEL_CATALOG + [
        {'id': 'ontario_heating_v1', 'name': 'Legacy Ontario monthly-weight profile'},
        {'id': 'flat_average_v1', 'name': 'Simple Flat Average (Baseline)'},
        {
                'id': 'banerjee_stochastic_v1', 
                'name': 'Legacy stochastic schedule model (synthetic weather)'
        }
    ]
    # --- MODEL A: Ontario Climate (Leap Year Aware) ---
    @staticmethod
    def _model_ontario_v1(total_annual_demand_kwh, year):
        
        # 1. Configuration
        ontario_weights = {
            1: 1.0, 2: 0.95, 3: 0.8, 4: 0.5, 5: 0.2, 6: 0.05,
            7: 0.0, 8: 0.0, 9: 0.1, 10: 0.4, 11: 0.7, 12: 0.95
        }
        
        daily_pattern = np.array([
            0.4, 0.4, 0.4, 0.5, 0.6, 0.8, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5,
            0.5, 0.5, 0.6, 0.7, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4
        ])

        # 2. Time Loop
        # We start at Jan 1, 00:00 of the chosen year
        current_time = datetime(year, 1, 1, 0, 0, 0)
        
        # We collect raw weights first
        raw_weights = []
        timestamps = []

        # Run while we are still in the requested year
        while current_time.year == year:
            month_weight = ontario_weights[current_time.month]
            hour_weight = daily_pattern[current_time.hour]
            
            # If it's Feb 29, Python handles the date correctly automatically.
            # We just apply the Feb weight (0.95).
            
            raw_weights.append(month_weight * hour_weight)
            timestamps.append(current_time)
            
            current_time += timedelta(hours=1)

        # 3. Scaling
        # Normalize so the sum of all hourly values equals the total annual demand
        raw_weights = np.array(raw_weights)
        total_weight = np.sum(raw_weights)
        
        if total_weight == 0:
            scaling_factor = 0
        else:
            scaling_factor = total_annual_demand_kwh / total_weight

        final_values = raw_weights * scaling_factor

        # 4. Pack Result
        result = []
        for i, val in enumerate(final_values):
            result.append({
                'time': timestamps[i],
                'value': round(val, 4)
            })
            
        return result

    # --- MODEL B: Flat Average ---
    @staticmethod
    def _model_flat_average(total_annual_demand_kwh, year):
        current_time = datetime(year, 1, 1, 0, 0, 0)
        timestamps = []
        
        # Simply count how many hours are in this year
        while current_time.year == year:
            timestamps.append(current_time)
            current_time += timedelta(hours=1)
            
        # Distribute demand equally
        hours_in_year = len(timestamps) # 8760 or 8784
        hourly_val = total_annual_demand_kwh / hours_in_year
        
        return [{'time': t, 'value': round(hourly_val, 4)} for t in timestamps]
    @staticmethod
    def _model_banerjee_v1(total_annual_demand_kwh, year, building_type='commercial'):
        """
        Implementation of the Ahijit Banerjee model:
        Uses stochastic weather generation + schedule-based internal gains 
        to shape the load profile.
        """
        # 1. Setup Time Array (0 to 8760)
        number_of_hours = (366 if calendar.isleap(year) else 365) * 24
        hours = np.arange(number_of_hours)

        # 2. Weather Simulation (Southern Ontario approx)
        # T_annual varies seasonally, T_daily varies by hour, + Noise
        temp_annual = (8.5) - (18.5) * np.cos(2 * np.pi * (hours - 744) / 8760)
        temp_daily = 4.0 * np.sin(2 * np.pi * (hours - 9) / 24)
        
        # Set seed for consistency across runs for the same building
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 2.0, number_of_hours)
        
        t_outdoor = temp_annual + temp_daily + noise

        # 3. Schedule & Setpoint Logic
        setpoint = np.zeros(number_of_hours)
        internal_gains = np.zeros(number_of_hours)

        # Vectorized approach is faster, but loops are clearer for complex logic
        # We'll use a fast loop approach
        for i in range(number_of_hours):
            hour_of_day = i % 24
            day_of_week = (datetime(year, 1, 1).weekday() + i // 24) % 7 # Monday=0

            if building_type == 'residential':
                # Residential Logic: Steady 21C, slight night setback, evening gains
                if hour_of_day >= 23 or hour_of_day < 6:
                    setpoint[i] = 19.0 # Night
                else:
                    setpoint[i] = 21.0 # Day
                
                # Evening Cooking/TV Gains (5pm - 9pm)
                if 17 <= hour_of_day <= 21:
                    internal_gains[i] = 1.5
            
            else: 
                # Commercial Logic (Default): 9-5 Mon-Fri
                # Night Setback (16C)
                setpoint[i] = 16.0 
                
                if day_of_week < 5: # Mon-Fri
                    # Morning Warm-up Spike (5am-7am)
                    if 5 <= hour_of_day < 7:
                        setpoint[i] = 24.0 # Boost
                    # Work Hours (7am-6pm)
                    elif 7 <= hour_of_day <= 18:
                        setpoint[i] = 21.0
                        internal_gains[i] = 3.0 # Computers/People heat
        
        # 4. Calculate Raw Load Shape
        # Load = (Setpoint - Outdoor) - Gains
        raw_load = (setpoint - t_outdoor) - internal_gains
        
        # Heating cannot be negative (Cooling is not calculated here)
        raw_load = np.maximum(0, raw_load)

        # 5. Scale to Annual kWh
        total_shape_sum = np.sum(raw_load)
        
        if total_shape_sum == 0:
            scaling_factor = 0
        else:
            scaling_factor = total_annual_demand_kwh / total_shape_sum
            
        hourly_values = raw_load * scaling_factor

        # 6. Format Output for Tasks
        timeseries_data = []
        
        # Create timezone-aware start date
        start_time_naive = datetime(year, 1, 1, 0, 0, 0)
        try:
            current_time = timezone.make_aware(start_time_naive)
        except ValueError:
            current_time = start_time_naive

        for val in hourly_values:
            timeseries_data.append({
                'time': current_time,
                'value': round(float(val), 4)
            })
            current_time += timedelta(hours=1)

        return timeseries_data

    @staticmethod
    def generate_heating_profile(total_annual_demand_kwh, model_type='ontario_heating_v1', year=2024, building_type='commercial'):
        """
        Factory method to switch between models
        """
        if model_type == 'ontario_heating_v1':
            return BuildingEstimator._model_ontario_v1(total_annual_demand_kwh, year)
        
        elif model_type == 'banerjee_stochastic_v1':
            return BuildingEstimator._model_banerjee_v1(total_annual_demand_kwh, year, building_type)
        
        elif model_type == 'flat_average_v1':
            return BuildingEstimator._model_flat_average(total_annual_demand_kwh, year)
        else:
            raise ValueError(f'Unknown legacy heating model: {model_type}')
    @staticmethod
    def get_available_models():
        return BuildingEstimator.AVAILABLE_MODELS
    @staticmethod
    def fetch_burlington_weather_data(year=2025):
        """
        Fetches 8760 hours of historical weather data for Burlington, Ontario
        using the Open-Meteo ERA5-Land reanalysis dataset.
        
        Returns a list of dictionaries: [{'timestamp': datetime, 'temperature': float}]
        """
        # Coordinates for Burlington, Ontario
        latitude = 43.3255
        longitude = -79.7990
        
        # We use the archive API to get a complete, finalized past year
        # timezone=America/Toronto ensures the timestamps align with local daily cycles
        url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={latitude}&longitude={longitude}"
            f"&start_date={year}-01-01&end_date={year}-12-31"
            f"&hourly=temperature_2m"
            f"&timezone=America%2FToronto"
        )
        
        print(f"Fetching weather data for Burlington ({year})...")
        response = requests.get(url)
        
        if response.status_code != 200:
            raise ConnectionError(f"Failed to fetch weather data: {response.text}")
            
        data = response.json()
        
        times = data['hourly']['time']
        temperatures = data['hourly']['temperature_2m']
        
        weather_data = []
        
        for i in range(len(times)):
            # Sometimes the API might return a null temperature for missing data
            temp = temperatures[i]
            if temp is None:
                # Fallback to the previous hour's temperature if missing
                temp = temperatures[i-1] if i > 0 else 0.0 
                
            weather_data.append({
                # Convert "YYYY-MM-DDTHH:MM" to a Python datetime object
                'timestamp': datetime.fromisoformat(times[i]),
                'temperature': float(temp)
            })
            
        print(f"Successfully generated {len(weather_data)} hourly weather records.")
        return weather_data

    # --- How to use it together with the load profile generator ---
    # if __name__ == "__main__":
    #     my_building = Building.objects.get(id=1)
    #     
    #     # Fetch Burlington data (2025 is a complete historical year)
    #     weather_2025 = fetch_burlington_weather_data(2025)
    #     
    #     # Pass the fetched data directly into your load profile generator
    #     records_created = generate_hourly_load_profiles(my_building, weather_2025)
    #     print(f"Created {records_created} load profiles.")
    @staticmethod

    def generate_hourly_load_profiles(building, weather_data, network=None, source_model='ontario_degree_hour_v1'):
        """
        Estimates 8760 hourly heating and cooling loads for a Building based on 
        floor area, building type, and outdoor air temperature.
        
        :param building: Building model instance
        :param weather_data: List of dicts, e.g., [{'timestamp': datetime, 'temperature': float}, ...]
        :param network: Optional Network instance for network-scoped profiles.
        :param source_model: Identifier for the generation model used.
        """
        if not building.floor_area:
            print(f"Building {building.id} lacks a floor_area. Cannot estimate loads.")
            return 0

        # 1. Ontario (Hamilton/Toronto) Climate Constants
        T_DESIGN_HEAT = -18.0  # Celsius
        T_DESIGN_COOL = 30.0   # Celsius
        T_BASE_HEAT = 15.0     # Heating balance point
        T_BASE_COOL = 18.0     # Cooling balance point
        
        # Delta T maximums for the linear interpolation
        MAX_DT_HEAT = T_BASE_HEAT - T_DESIGN_HEAT  # 33.0 C
        MAX_DT_COOL = T_DESIGN_COOL - T_BASE_COOL  # 12.0 C

        # 2. Peak Load Densities (W/m2) by Building Type based on NRCan/OBC typicals
        LOAD_DENSITIES = {
            'residential': {'heat': 40.0, 'cool': 35.0},
            'commercial':  {'heat': 50.0, 'cool': 60.0},
            'industrial':  {'heat': 65.0, 'cool': 50.0},
            'public':      {'heat': 45.0, 'cool': 55.0},
            'other':       {'heat': 45.0, 'cool': 45.0},
        }
        
        b_type = building.building_type if building.building_type in LOAD_DENSITIES else 'other'
        q_peak_h = LOAD_DENSITIES[b_type]['heat']
        q_peak_c = LOAD_DENSITIES[b_type]['cool']
        
        area = building.floor_area
        profiles_to_create = []

        # 3. Calculate hourly profiles
        for data in weather_data:
            T_out = data['temperature']
            timestamp = data['timestamp']
            # End date is start date + 1 hour for hourly profiling
            end_time = timestamp + timedelta(hours=1) 
            
            heating_kw = 0.0
            cooling_kw = 0.0
            
            # Calculate Heating
            if T_out < T_BASE_HEAT:
                ratio = (T_BASE_HEAT - T_out) / MAX_DT_HEAT
                # Prevent going over 100% design load if temps drop below -18C
                ratio = min(ratio, 1.0) 
                heating_kw = (area * q_peak_h * ratio) / 1000.0
                
            # Calculate Cooling
            if T_out > T_BASE_COOL:
                ratio = (T_out - T_BASE_COOL) / MAX_DT_COOL
                # Prevent going over 100% design load if temps rise above 30C
                ratio = min(ratio, 1.0)
                cooling_kw = (area * q_peak_c * ratio) / 1000.0
                
            # Prepare Heating LoadProfile
            if heating_kw >= 0:
                profiles_to_create.append(
                    LoadProfile(
                        profile_type='heating',
                        value=round(heating_kw, 2),
                        unit='kW',
                        start_date=timestamp,
                        end_date=end_time,
                        building=building,
                        network=network,
                        source_model=source_model or 'ontario_degree_hour_v1'
                    )
                )
                

        # 4. Bulk Insert to Database (Batch size of 2000 prevents SQLite/Postgres max variable limits)
        # Using a transaction block ensures all hours are written safely
        with transaction.atomic():
            LoadProfile.objects.bulk_create(profiles_to_create, batch_size=2000)
            
        return len(profiles_to_create)
