"""Validated historical reanalysis for Canadian study locations; no synthetic fallback."""
import calendar
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
from django.conf import settings

LOCATIONS = [
    dict(id='burlington', name='Burlington, Ontario', latitude=43.3255, longitude=-79.7990, utc_offset=-5),
    dict(id='toronto', name='Toronto, Ontario', latitude=43.6532, longitude=-79.3832, utc_offset=-5),
    dict(id='ottawa', name='Ottawa, Ontario', latitude=45.4215, longitude=-75.6972, utc_offset=-5),
    dict(id='montreal', name='Montréal, Quebec', latitude=45.5019, longitude=-73.5674, utc_offset=-5),
    dict(id='winnipeg', name='Winnipeg, Manitoba', latitude=49.8951, longitude=-97.1384, utc_offset=-6),
    dict(id='calgary', name='Calgary, Alberta', latitude=51.0447, longitude=-114.0719, utc_offset=-7),
    dict(id='vancouver', name='Vancouver, British Columbia', latitude=49.2827, longitude=-123.1207, utc_offset=-8),
    dict(id='halifax', name='Halifax, Nova Scotia', latitude=44.6488, longitude=-63.5752, utc_offset=-4),
]


def validate_weather_payload(payload, year, location):
    zone = timezone(timedelta(hours=location['utc_offset']))
    start = datetime(year, 1, 1, tzinfo=zone)
    expected = (366 if calendar.isleap(year) else 365) * 24
    hourly = payload.get('hourly', {})
    times = hourly.get('time', [])
    values = hourly.get('temperature_2m', [])
    if len(times) != expected or len(values) != expected or any(v is None for v in values):
        raise ValueError('Weather provider did not return a complete hourly year; no gaps were filled.')
    timestamps = [datetime.fromisoformat(t).replace(tzinfo=zone) for t in times]
    if any(t != start + timedelta(hours=i) for i, t in enumerate(timestamps)):
        raise ValueError('Weather timestamps must be consecutive hourly local-standard-time values.')
    temperatures = np.asarray(values, dtype=float)
    if not np.isfinite(temperatures).all() or np.any((temperatures < -90) | (temperatures > 65)):
        raise ValueError('Weather temperatures are missing or outside physical bounds.')
    return timestamps, temperatures


def fetch_canadian_weather(year, location_id):
    location = next((x for x in LOCATIONS if x['id'] == location_id), None)
    if location is None:
        raise ValueError('Select a supported Canadian weather location.')
    if not 1950 <= year < datetime.now(timezone.utc).year:
        raise ValueError('Weather year must be a completed year from 1950 onward.')
    params = dict(latitude=location['latitude'], longitude=location['longitude'],
                  start_date=f'{year}-01-01', end_date=f'{year}-12-31',
                  hourly='temperature_2m', models='era5_land',
                  timezone=f"Etc/GMT+{abs(location['utc_offset'])}")
    cache_dir = Path(settings.BASE_DIR) / 'data' / 'weather_cache'
    cache_path = cache_dir / f'era5_land_{location_id}_{year}.json'
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding='utf-8'))
    else:
        response = requests.get('https://archive-api.open-meteo.com/v1/archive', params=params, timeout=(10, 60))
        response.raise_for_status()
        payload = response.json()
        validate_weather_payload(payload, year, location)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Readers never see a partial cache write.
        import uuid
        temporary = cache_path.with_suffix(f'.{uuid.uuid4().hex}.tmp')
        temporary.write_text(json.dumps(payload), encoding='utf-8')
        temporary.replace(cache_path)
    timestamps, temperatures = validate_weather_payload(payload, year, location)
    return timestamps, temperatures, dict(
        provider='Open-Meteo / Copernicus ERA5-Land reanalysis', location=location,
        year=year, hours=len(timestamps), parameters=params,
        timezone=f"UTC{location['utc_offset']:+d} fixed local standard time; no daylight-saving shifts",
        sha256=hashlib.sha256(json.dumps(payload['hourly'], sort_keys=True).encode()).hexdigest(),
        source='https://open-meteo.com/en/docs/historical-weather-api',
        note='Reanalysis at the selected Canadian location, not station observations. Shared weather for this study.')
