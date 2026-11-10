"""Project-scoped orchestration shared by comparison and the existing generation task."""
import gzip
import json
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.cache import cache
from django.db import transaction
from django.shortcuts import get_object_or_404
from rest_framework import serializers

from api.models import Building, LoadProfile, Network, Node, Project
from .heating_metrics import calculate_profile_metrics, screen_equivalent_feeder
from .heating_models import MODEL_CATALOG, MODEL_IDS, NECB_SOURCE, generate_building_profile
from .heating_weather import LOCATIONS, fetch_canadian_weather
from .heating_energy import calculate_annual_heat, ENERGY_PRESETS, ENERGY_SOURCES
from .heating_thermo import THERMO_PARAMETERS, THESIS_SOURCE


class HeatingStudyRequest(serializers.Serializer):
    project_id = serializers.IntegerField(min_value=1)
    network_id = serializers.IntegerField(min_value=1, required=False, allow_null=True)
    building_ids = serializers.ListField(child=serializers.IntegerField(min_value=1), required=False, max_length=500)
    models = serializers.ListField(child=serializers.ChoiceField(choices=sorted(MODEL_IDS)),
                                   default=lambda: [m['id'] for m in MODEL_CATALOG], min_length=1, max_length=len(MODEL_IDS))
    reference = serializers.ChoiceField(choices=sorted(MODEL_IDS) + ['existing'], default='hdd_daily_ca_v1')
    year = serializers.IntegerField(min_value=1950, max_value=datetime.now(timezone.utc).year - 1, default=2024)
    location = serializers.ChoiceField(choices=[l['id'] for l in LOCATIONS], default='burlington')
    energy_basis = serializers.ChoiceField(choices=['canadian_scenario', 'existing_or_intensity', 'intensity', 'imported_annual'], default='existing_or_intensity')
    energy_scenario = serializers.ChoiceField(choices=['low', 'central', 'high'], default='central')
    commercial_energy_preset = serializers.ChoiceField(choices=[k for k in ENERGY_PRESETS if k != 'residential'], default='office')
    residential_total_intensity_kwh_m2 = serializers.FloatField(min_value=0, max_value=3000, required=False, allow_null=True)
    commercial_total_intensity_kwh_m2 = serializers.FloatField(min_value=0, max_value=3000, required=False, allow_null=True)
    residential_space_fraction = serializers.FloatField(min_value=0, max_value=1, default=.636)
    residential_water_fraction = serializers.FloatField(min_value=0, max_value=1, default=.172)
    commercial_space_fraction = serializers.FloatField(min_value=0, max_value=1, default=.566)
    commercial_water_fraction = serializers.FloatField(min_value=0, max_value=1, default=.057)
    space_heat_conversion = serializers.FloatField(min_value=.1, max_value=6, default=.85)
    water_heat_conversion = serializers.FloatField(min_value=.1, max_value=6, default=.85)
    thermo_archetype = serializers.ChoiceField(choices=['detached', 'attached'], default='detached')
    intensity_kwh_m2 = serializers.FloatField(min_value=0, max_value=2000, default=120)
    dhw_fraction = serializers.FloatField(min_value=0, max_value=.9, default=.15)
    balance_temperature_c = serializers.FloatField(min_value=5, max_value=25, default=18)
    envelope_w_m2k = serializers.FloatField(min_value=.05, max_value=10, default=.8)
    air_changes_per_hour = serializers.FloatField(min_value=0, max_value=10, default=.5)
    internal_gains_w_m2 = serializers.FloatField(min_value=0, max_value=100, default=4)
    storey_height_m = serializers.FloatField(min_value=2, max_value=10, default=3)
    schedule_group = serializers.ChoiceField(choices=['auto'] + list('ABCDEFGHI'), default='auto')
    normalize_physical = serializers.BooleanField(default=True)
    delta_t_c = serializers.FloatField(min_value=1, max_value=80, default=30)
    peak_threshold_fraction = serializers.FloatField(min_value=.01, max_value=1, default=.9)

    def validate(self, attrs):
        import math
        if any(isinstance(v, float) and not math.isfinite(v) for v in attrs.values()):
            raise serializers.ValidationError('All numeric inputs must be finite.')
        for sector in ('residential', 'commercial'):
            if attrs[sector + '_space_fraction'] + attrs[sector + '_water_fraction'] > 1:
                raise serializers.ValidationError(f'{sector.title()} space heating and DHW fractions must sum to at most 1.')
        attrs['models'] = list(dict.fromkeys(attrs['models']))
        if attrs['reference'] != 'existing' and attrs['reference'] not in attrs['models']:
            raise serializers.ValidationError('The reference model must also be selected.')
        return attrs


def consumer_building_queryset(options):
    project = get_object_or_404(Project, pk=options['project_id'])
    network = None
    buildings = Building.objects.filter(project=project)
    if options.get('network_id'):
        network = get_object_or_404(Network, pk=options['network_id'], project=project)
        ids = Node.objects.filter(network=network, building__isnull=False).exclude(
            node_type__in=['producer', 'boiler', 'junction']).values_list('building_id', flat=True)
        buildings = buildings.filter(id__in=ids)
    else:
        from django.db.models import Q
        buildings = buildings.filter(Q(waste_heat_source_type='') | Q(waste_heat_source_type__isnull=True))
    return project, network, buildings


def resolve_study_buildings(options):
    project, network, buildings = consumer_building_queryset(options)
    if options.get('building_ids'):
        buildings = buildings.filter(id__in=options['building_ids'])
        if buildings.count() != len(set(options['building_ids'])):
            raise ValueError('Some selected buildings are outside this project/network consumer scope.')
    buildings = list(buildings.order_by('id')[:501])
    if not buildings:
        raise ValueError('No consumer buildings in this selection.')
    if len(buildings) > 500:
        raise ValueError('Select a network or at most 500 building IDs for a reproducible comparison.')
    return project, network, buildings


def read_existing_profiles(buildings, network, timestamps):
    """Use only exact-scope, complete, unique, one-hour kW series; never sum competing models."""
    start, end = timestamps[0], timestamps[-1] + timedelta(hours=1)
    rows = LoadProfile.objects.filter(building_id__in=[b.id for b in buildings], network=network,
                                     profile_type='heating', start_date__gte=start, start_date__lt=end)
    values = {b.id: np.full(len(timestamps), np.nan) for b in buildings}
    invalid, sources = set(), {b.id: set() for b in buildings}
    for row in rows.values('building_id', 'start_date', 'end_date', 'value', 'unit', 'source_model').iterator(chunk_size=10000):
        bid = row['building_id']
        position = (row['start_date'] - start).total_seconds() / 3600
        if (not position.is_integer() or row['end_date'] - row['start_date'] != timedelta(hours=1)
                or row['unit'] != 'kW' or not np.isfinite(row['value']) or row['value'] < 0):
            invalid.add(bid)
            continue
        index = int(position)
        if np.isfinite(values[bid][index]):
            invalid.add(bid)
        values[bid][index] = row['value']
        sources[bid].add(row['source_model'])
    complete = {bid: arr for bid, arr in values.items()
                if bid not in invalid and np.isfinite(arr).all() and len(sources[bid]) == 1}
    return complete, {str(bid): sorted(sources[bid]) for bid in complete}


def prepare_building_inputs(buildings, existing, options):
    inputs = []
    for building in buildings:
        area = building.floor_area
        if area is None or not np.isfinite(area) or area <= 0:
            raise ValueError(f'Building {building.id} requires a positive floor area in m².')
        basis = options['energy_basis']
        energy_details = {}
        if basis == 'canadian_scenario':
            energy_details = calculate_annual_heat(area, building.building_type, options)
            annual = energy_details['annual_kwh']
            source = energy_details['energy_source']
        elif basis == 'imported_annual':
            annual = building.heating_demand
            source = 'User explicitly interpreted Building.heating_demand as annual delivered kWh'
            if annual is None or not np.isfinite(annual) or annual < 0:
                raise ValueError(f'Building {building.id} has no valid imported annual energy.')
        elif basis == 'existing_or_intensity' and building.id in existing:
            annual = float(existing[building.id].sum())
            source = 'Integrated complete existing hourly kW profile (not necessarily measured)'
        else:
            annual = area * options['intensity_kwh_m2']
            source = 'User-assumed delivered energy intensity × existing floor area'
        volume = building.volume_m3
        if volume is None or not np.isfinite(volume) or volume <= 0:
            volume = area * options['storey_height_m']
        inputs.append(dict(id=building.id, name=building.name or f'Building {building.id}',
                           building_type=building.building_type, floor_area=area, volume_m3=volume,
                           annual_kwh=annual, energy_source=source,
                           volume_source='Building.volume_m3' if volume == building.volume_m3 else 'Floor area × assumed storey height'))
        inputs[-1].update(energy_details)
    return inputs


def run_heating_study(options, user_id=None, weather=None):
    from api.utils.pipe_sizing import RuntimeDesignCriteria
    project, network, buildings = resolve_study_buildings(options)
    timestamps, temperatures, provenance = weather or fetch_canadian_weather(options['year'], options['location'])
    existing, existing_sources = read_existing_profiles(buildings, network, timestamps)
    inputs = prepare_building_inputs(buildings, existing, options)
    shape_cache = {}
    matrices = {model: np.array([generate_building_profile(model, temperatures, timestamps, b, options, shape_cache)
                                for b in inputs]) for model in options['models']}
    if options['reference'] == 'existing':
        if len(existing) != len(buildings):
            raise ValueError(f'Existing reference is incomplete, duplicated or mixed-source: {len(existing)}/{len(buildings)} buildings valid. Select a generated reference or repair/import the data.')
        reference_matrix = np.array([existing[b.id] for b in buildings])
    else:
        reference_matrix = matrices[options['reference']]
    reference = reference_matrix.sum(axis=0)
    criteria = RuntimeDesignCriteria(network.id if network else None)
    threshold = float(reference.max()) * options['peak_threshold_fraction']
    results = []
    for model, matrix in matrices.items():
        total = matrix.sum(axis=0)
        metrics = calculate_profile_metrics(total, reference, float(matrix.max(axis=1).sum()), threshold,
                                             criteria.CP, options['delta_t_c'])
        results.append(dict(model=model, name=next(m['name'] for m in MODEL_CATALOG if m['id'] == model),
                            metrics=metrics, hourly_kw=total.tolist(),
                            duration_curve_kw=np.sort(total)[::-1].tolist(),
                            feeder=screen_equivalent_feeder(total, reference, criteria, options['delta_t_c']),
                            buildings=[dict(id=b['id'], name=b['name'], annual_kwh=float(row.sum()),
                                            peak_kw=float(row.max())) for b, row in zip(inputs, matrix)]))
    run_id = str(uuid.uuid4())
    response = dict(run_id=run_id, project_id=project.id, network_id=network.id if network else None,
                    building_count=len(buildings), options=options, results=results,
                    timestamps=[t.isoformat() for t in timestamps], reference_hourly_kw=reference.tolist(),
                    reference_label='Existing stored profiles (provenance below; not assumed measured)' if options['reference'] == 'existing' else 'Generated model baseline — differences are not measured accuracy',
                    threshold_kw=threshold, weather=provenance, building_inputs=inputs,
                    existing_coverage=dict(complete_buildings=len(existing), selected_buildings=len(buildings), sources=existing_sources),
                    energy_sources=dict(Counter(b['energy_source'] for b in inputs)),
                    design=dict(cp_kj_kgk=criteria.CP, rho_kg_m3=criteria.RHO,
                                max_velocity_m_s=criteria.MAX_VELOCITY, target_pressure_pa_m=criteria.TARGET_PRESSURE_DROP,
                                source='Selected network constants' if network else 'Existing project sizing defaults'),
                    sources=[NECB_SOURCE, 'https://open-meteo.com/en/docs/historical-weather-api'] + (ENERGY_SOURCES if options['energy_basis'] == 'canadian_scenario' else []),
                    energy_summary={key: sum(b.get(key, 0) for b in inputs) for key in ('total_final_energy_kwh', 'space_input_kwh', 'dhw_input_kwh', 'other_final_energy_kwh', 'annual_space_kwh', 'annual_dhw_kwh')},
                    thermo_provenance=dict(source=THESIS_SOURCE, archetype=options['thermo_archetype'], parameters=THERMO_PARAMETERS[options['thermo_archetype']],
                        solver='RK4, 2.5-minute steps, hourly mean output, 14-day repeated-first-week warm-up',
                        interpretation='Energy-conserving capacitor sign; complete heater branch gated by u. Heating only; no trained encoder; normalized median-house shape.'),
                    assumptions=[
                        'Canadian scenarios combine historical national benchmarks from different years; they are not local measured demand or forecasts.',
                        'Low/high = 75%/125% of central total energy; analyst sensitivity bounds, not statistical confidence limits. Fractions and conversion factors remain fixed unless edited.',
                        'Residential shares: NRCan 2019 63.6% space/17.2% water; commercial shares: NRCan 2017 56.6%/5.7%, sector proxies, not subtype-specific observations.',
                        'Useful heat = final energy × end-use fraction × conversion. Default 0.85 is an explicit scenario assumption, not a measured Canadian stock efficiency; >1 means assumed seasonal COP.',
                        'THERMO-derived: published residential median circuit parameters, energy-conserving interpretation of printed equations; no encoder training or transfer of thesis validation statistics.',
                        'THERMO commercial use extrapolates a residential archetype shape. No geometry/vintage inference; all same-archetype schedules are synchronized, with no stochastic diversity.',
                        'Canadian weather location is selected explicitly; one shared reanalysis series per study.',
                        '120 kWh/m²/year, DHW fraction, envelope coefficient, ACH, gains and storey height are editable research assumptions, not Canadian measured benchmarks.',
                        'Auto schedule mapping: residential → NECB G, other coarse building types → NECB A; no holiday calendar.',
                        'Normalized models share annual energy by construction; annual agreement is not validation.',
                        'NECB SLP is a monthly schedule-derived research template, not an official measured utility SLP.',
                        'Hourly data cannot establish subhourly DHW peaks; resolution sensitivity covers only 1, 3, 6 and 24 hours.',
                        'Equivalent feeder screening excludes topology, distribution losses, pumps, storage and terminal pressure; no unserved-energy or whole-network cost claim.',
                    ])
    if user_id is not None:
        cache.set(f'heating-study:{run_id}', dict(user_id=user_id, response=response, matrices=matrices), 3600)
    return response, matrices


def persist_selected_profile(response, matrix, model_id):
    """Archive exact replaced scope, then atomically install ONE active alternative."""
    options = response['options']
    _, network, buildings = resolve_study_buildings(options)
    if [b.id for b in buildings] != [b['id'] for b in response['building_inputs']]:
        raise ValueError('Building selection changed. Run the comparison again before applying.')
    timestamps = [datetime.fromisoformat(t) for t in response['timestamps']]
    folder = Path(settings.BASE_DIR) / 'data' / 'load_profile_runs' / str(uuid.uuid4())
    folder.mkdir(parents=True)
    (folder / 'study.json').write_text(json.dumps(response), encoding='utf-8')
    query = LoadProfile.objects.filter(building_id__in=[b.id for b in buildings], network=network,
                                      profile_type='heating', start_date__gte=timestamps[0],
                                      start_date__lt=timestamps[-1] + timedelta(hours=1))
    with transaction.atomic():
        # Lock the project on databases supporting row locks; SQLite serializes writes.
        Project.objects.select_for_update().get(pk=options['project_id'])
        with gzip.open(folder / 'previous_profiles.json.gz', 'wt', encoding='utf-8') as archive:
            archive.write('[')
            for index, row in enumerate(query.values().iterator(chunk_size=10000)):
                if index:
                    archive.write(',')
                archive.write(json.dumps(row, default=str))
            archive.write(']')
        query.delete()
        for building, values in zip(buildings, matrix):
            rows = [LoadProfile(building=building, network=network, profile_type='heating', unit='kW',
                                source_model=model_id, value=float(v), start_date=t,
                                end_date=t + timedelta(hours=1)) for t, v in zip(timestamps, values)]
            LoadProfile.objects.bulk_create(rows, batch_size=2000)
    return dict(created=len(buildings) * len(timestamps), model=model_id, archive_id=folder.name)
