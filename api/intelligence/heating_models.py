"""Canadian heating profile models. All arrays contain hourly mean delivered kW.

The NECB schedule-derived SLP is a research archetype, not a measured utility SLP.
No model in this module uses random noise or silently substitutes another model.
"""
import calendar
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from .heating_thermo import generate_thermo_shape, THESIS_SOURCE

NECB_COMMIT = 'c8c1b9ebb30c5f1a441231bf3eae76b18fbbe907'
NECB_SOURCE = ('https://github.com/NatLabRockies/openstudio-standards/blob/'
               + NECB_COMMIT + '/lib/openstudio-standards/standards/necb/NECB2011/data/schedules.json')
MODEL_CATALOG = [
    dict(id='thermo_rc_ca_v1', name='THERMO-derived RC model (published parameters)', family='Dynamic physical / reduced-order RC',
         description='Two thermal masses and smooth thermostat hysteresis; published detached/attached median parameters.',
         use_case='Explore thermal inertia and recovery effects on hourly peaks at equal annual heat.',
         limitation='Forward circuit only, not the trained THERMO neural network. Commercial use extrapolates a residential shape; no measured validation.',
         source='', source_label=THESIS_SOURCE),
    dict(id='hdd_daily_ca_v1', name='Heating degree-day (HDD) model', family='Degree days',
         description='Daily mean temperature below the balance point; uniform space heating within each day.',
         use_case='Low-data annual/seasonal planning; baseline for peak-sensitivity studies.',
         limitation='Does not resolve intraday space-heating peaks.',
         source='https://doi.org/10.1038/s41597-024-04129-8'),
    dict(id='heat_balance_ca_v1', name='Steady-state heat-balance model', family='Physical heat balance',
         description='Hourly transmission and ventilation losses minus scheduled internal gains.',
         use_case='Envelope, ventilation and operating-schedule sensitivity.',
         limitation='Single-zone steady state: no thermal storage, solar gains or recovery dynamics.',
         source='https://bigladdersoftware.com/epx/docs/24-1/engineering-reference/zone-air-heat-balance.html'),
    dict(id='necb_slp_ca_v1', name='NECB schedule-derived research template (SLP)', family='Schedule-derived archetype',
         description='Monthly typical-temperature/day-type templates using published Canadian NECB schedules.',
         use_case='Repeatable archetype profiles for preliminary Canadian district studies.',
         limitation='Research-derived SLP, not an official or measured Canadian heat-utility profile; smooths cold spells.',
         source=NECB_SOURCE),
]
MODEL_IDS = {item['id'] for item in MODEL_CATALOG}


def normalize_energy(shape, annual_kwh):
    """Scale a nonnegative hourly shape without discarding annual energy."""
    shape = np.asarray(shape, dtype=float)
    if not np.isfinite(shape).all() or np.any(shape < 0):
        raise ValueError('Profile weights must be finite and nonnegative.')
    if annual_kwh == 0:
        return np.zeros_like(shape)
    if shape.sum() <= 0:
        raise ValueError('No heating hours: check weather, balance temperature and gains.')
    return shape * (annual_kwh / shape.sum())


@lru_cache(maxsize=1)
def load_necb_schedules():
    path = Path(__file__).parent / 'data' / 'necb2011_schedules.json'
    return json.loads(path.read_text(encoding='utf-8'))['tables']['schedules']['table']


def build_schedule(timestamps, group, category):
    """Expand NECB hourly weekday/Saturday/Sunday values in local standard time."""
    name = f'NECB-{group}-{category}'
    rows = [row for row in load_necb_schedules() if row['name'] == name]
    by_day = {}
    for token in ('Wkdy', 'Sat', 'Sun'):
        match = next((row for row in rows if token in row['day_types'].split('|')), None)
        if match is None:
            raise ValueError(f'Missing published schedule: {name}/{token}')
        values = match['values']
        by_day[token] = values * 24 if len(values) == 1 else values
        if len(by_day[token]) != 24:
            raise ValueError(f'Invalid schedule length: {name}')
    return np.array([by_day['Wkdy' if t.weekday() < 5 else 'Sat' if t.weekday() == 5 else 'Sun'][t.hour]
                     for t in timestamps], dtype=float)


def generate_hdd_profile(temperatures, annual_space_kwh, balance_temperature_c):
    daily_mean = temperatures.reshape(-1, 24).mean(axis=1)
    daily_degree_days = np.maximum(balance_temperature_c - daily_mean, 0)
    return normalize_energy(np.repeat(daily_degree_days, 24), annual_space_kwh)


def generate_heat_balance_profile(temperatures, setpoints, occupancy, area_m2, volume_m3, options):
    # 0.33 Wh/(m3 K): volumetric heat capacity of air. ACH is effective unrecuperated air exchange.
    loss_coefficient_kw_k = (options['envelope_w_m2k'] * area_m2
                              + 0.33 * options['air_changes_per_hour'] * volume_m3) / 1000
    gains_kw = options['internal_gains_w_m2'] * area_m2 * occupancy / 1000
    return np.maximum(loss_coefficient_kw_k * (setpoints - temperatures) - gains_kw, 0)


def generate_standard_load_profile(temperatures, timestamps, setpoints, annual_space_kwh, balance_temperature_c):
    """Monthly typical-day heating templates; schedule temperatures provide daily modulation."""
    months = np.array([t.month for t in timestamps])
    typical_temperature = np.empty_like(temperatures)
    for month in range(1, 13):
        typical_temperature[months == month] = temperatures[months == month].mean()
    # The 22 C occupied NECB setpoint maps to the requested effective balance point.
    weights = np.maximum(typical_temperature * -1 + balance_temperature_c + setpoints - 22, 0)
    return normalize_energy(weights, annual_space_kwh)


def generate_building_profile(model_id, temperatures, timestamps, building, options, shape_cache=None):
    if model_id not in MODEL_IDS:
        raise ValueError(f'Unknown heating model: {model_id}')
    expected = (366 if calendar.isleap(timestamps[0].year) else 365) * 24
    temperatures = np.asarray(temperatures, dtype=float)
    if len(temperatures) != expected or not np.isfinite(temperatures).all():
        raise ValueError('A complete finite hourly weather year is required.')
    group = options['schedule_group']
    if group == 'auto':
        # Explicit research mapping of coarse database types; users can override it.
        group = 'G' if building['building_type'].lower() == 'residential' else 'A'
    setpoints = build_schedule(timestamps, group, 'Thermostat Setpoint-Heating')
    occupancy = build_schedule(timestamps, group, 'Occupancy')
    dhw_shape = build_schedule(timestamps, group, 'Service Water Heating')
    annual = building['annual_kwh']
    space_energy = building.get('annual_space_kwh', annual * (1 - options['dhw_fraction']))
    water_energy = building.get('annual_dhw_kwh', annual * options['dhw_fraction'])
    if model_id == 'hdd_daily_ca_v1':
        space = generate_hdd_profile(temperatures, space_energy, options['balance_temperature_c'])
    elif model_id == 'thermo_rc_ca_v1':
        key = (group, options['thermo_archetype'])
        if shape_cache is not None and key in shape_cache:
            shape = shape_cache[key]
        else:
            shape = generate_thermo_shape(temperatures, setpoints, options['thermo_archetype'])
            if shape_cache is not None:
                shape_cache[key] = shape
        space = normalize_energy(shape, space_energy)
    elif model_id == 'necb_slp_ca_v1':
        space = generate_standard_load_profile(temperatures, timestamps, setpoints, space_energy,
                                              options['balance_temperature_c'])
    else:
        space = generate_heat_balance_profile(temperatures, setpoints, occupancy,
                                               building['floor_area'], building['volume_m3'], options)
        if options['normalize_physical']:
            space = normalize_energy(space, space_energy)
    dhw = normalize_energy(dhw_shape, water_energy)
    return space + dhw
