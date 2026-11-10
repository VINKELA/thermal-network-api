"""Compare imported Burlington alternatives without regenerating or mixing models."""
import uuid
import numpy as np
from django.core.cache import cache
from rest_framework import serializers
from api.models import BurlingtonComparisonProfile
from .heating_studies import HeatingStudyRequest, consumer_building_queryset
from .heating_metrics import calculate_profile_metrics, screen_equivalent_feeder

CATALOG = [
    dict(id='burlington_envelope', name='Envelope transmission', family='Physical / steady-state transmission',
         description='Component U-values × exposed areas × hourly positive indoor–outdoor temperature difference; indoor temperature 20°C.',
         use_case='Explore geometry and assumed envelope heat loss.', limitation='Literature U-value proxies, not measured properties; omits ventilation, ground, gains and hot water.',
         source='https://idobe.engineering.ualberta.ca/wp-content/uploads/eSim-2024-proceedings-compilation-of-full-papers_organized.pdf#page=29'),
    dict(id='burlington_peak', name='Peak-scaled hourly temperature', family='Peak-calibrated temperature model',
         description='Hourly positive temperature deficit below 20°C, scaled to the supplied annual peak separately in each year.',
         use_case='Generate hourly scenarios when a peak estimate is available.', limitation='Source peaks assumed W; same peak reused each year; generated energy is not measured.', source='',
         source_label='2024IndividualBurlingtonHeatLoad_latlon_with_fsa2.csv; Heat_load (Burlington), otherwise BackUpHeat'),
    dict(id='burlington_hdd18', name='Daily HDD18', family='Degree-day / energy allocation',
         description='Daily degree days below 18°C; daily heat distributed equally over 24 hours. Annual energy matches the peak-scaled model.',
         use_case='Transparent seasonal baseline and daily-averaging sensitivity.', limitation='Suppresses intraday peaks; annual agreement is imposed, not validation. Base temperature also differs from the hourly models.',
         source='https://climate.meteo.gc.ca/glossary_e.html'),
]
MODEL_IDS = [m['id'] for m in CATALOG]


class BurlingtonStudyRequest(HeatingStudyRequest):
    energy_basis = serializers.ChoiceField(choices=["stored_burlington"], default="stored_burlington")
    models = serializers.ListField(child=serializers.ChoiceField(choices=MODEL_IDS), default=lambda: MODEL_IDS.copy(), min_length=1, max_length=3)
    reference = serializers.ChoiceField(choices=MODEL_IDS, default='burlington_hdd18')
    year = serializers.ChoiceField(choices=[2021, 2022, 2023], default=2023)
    normalize_physical = serializers.BooleanField(default=False)


def available_buildings(options):
    project, network, buildings = consumer_building_queryset(options)
    return project, network, buildings.filter(comparison_profiles__isnull=False).distinct()


def run_burlington_comparison(options, user_id):
    from api.utils.pipe_sizing import RuntimeDesignCriteria
    project, network, buildings = available_buildings(options)
    if options.get('building_ids'):
        buildings = buildings.filter(id__in=options['building_ids'])
        if buildings.count() != len(set(options['building_ids'])):
            raise ValueError('Select only buildings with imported comparison data in this project/network.')
    buildings = list(buildings.order_by('id'))
    if not buildings or len(buildings) > 10:
        raise ValueError('Choose the Burlington study containing the ten imported comparison buildings.')
    options = dict(options, building_ids=[b.id for b in buildings], energy_basis='stored_burlington')
    # Always load all three so normalization has a fixed target even if peak-scaled is hidden.
    records = {(p.building_id, p.model_id): p for p in BurlingtonComparisonProfile.objects.filter(
        building__in=buildings, year=options['year'])}
    if len(records) != len(buildings) * 3:
        raise ValueError('Incomplete imported alternatives for this year. Reimport the sample.')
    first = next(iter(records.values()))
    timestamps = first.timestamps
    if len(timestamps) != 8760 or any(p.timestamps != timestamps for p in records.values()):
        raise ValueError('Stored hourly timestamps are incomplete or misaligned.')
    matrices = {model: np.array([records[b.id, model].hourly_kw for b in buildings], dtype=float) for model in MODEL_IDS}
    if any(m.shape != (len(buildings), 8760) or not np.isfinite(m).all() or np.any(m < 0) for m in matrices.values()):
        raise ValueError('Invalid stored hourly loads.')
    if options['normalize_physical']:
        target = matrices['burlington_peak'].sum(axis=1)
        for model, matrix in matrices.items():
            energy = matrix.sum(axis=1)
            if np.any((energy == 0) & (target > 0)):
                raise ValueError('Cannot normalize a zero-energy profile to a positive target.')
            matrices[model] = matrix * np.divide(target, energy, out=np.zeros_like(target), where=energy > 0)[:, None]
    matrices = {m: matrices[m] for m in options['models']}
    reference = matrices[options['reference']].sum(axis=0)
    criteria = RuntimeDesignCriteria(network.id if network else None)
    threshold = float(reference.max()) * options['peak_threshold_fraction']
    inputs = [dict(id=b.id, name=b.name, floor_area=b.floor_area, building_type=b.building_type,
                   **records[b.id, 'burlington_envelope'].provenance) for b in buildings]
    results = []
    for model, matrix in matrices.items():
        total = matrix.sum(axis=0)
        results.append(dict(model=model, name=next(m['name'] for m in CATALOG if m['id'] == model),
            metrics=calculate_profile_metrics(total, reference, float(matrix.max(axis=1).sum()), threshold, criteria.CP, options['delta_t_c']),
            hourly_kw=total.tolist(), duration_curve_kw=np.sort(total)[::-1].tolist(),
            feeder=screen_equivalent_feeder(total, reference, criteria, options['delta_t_c']),
            buildings=[dict(id=b.id, name=b.name, annual_kwh=float(row.sum()), peak_kw=float(row.max())) for b, row in zip(buildings, matrix)]))
    response = dict(run_id=str(uuid.uuid4()), project_id=project.id, network_id=network.id if network else None,
        building_count=len(buildings), options=options, timestamps=timestamps, results=results,
        reference_hourly_kw=reference.tolist(), reference_label='Generated baseline; differences are not measured accuracy',
        threshold_kw=threshold, building_inputs=inputs, existing_coverage=dict(complete_buildings=len(buildings), selected_buildings=len(buildings), sources={str(b.id): MODEL_IDS for b in buildings}),
        weather=dict(location=dict(name='Burlington, Ontario'), year=options['year'], hours=8760,
            provider='User-supplied Burlington temperature CSV', timezone='Original hourly labels; timezone unverified',
            note='player_heatload_burlington_2021to2023.csv; temperature interpreted as °C. Station provenance unverified.', sha256=first.provenance['weather_sha256']),
        energy_sources={'Stored generated profiles; no measured consumption': len(buildings)}, energy_summary={},
        design=dict(cp_kj_kgk=criteria.CP, rho_kg_m3=criteria.RHO, max_velocity_m_s=criteria.MAX_VELOCITY, target_pressure_pa_m=criteria.TARGET_PRESSURE_DROP),
        sources=[m['source'] for m in CATALOG if m['source']],
        assumptions=[
            'Ten purposively selected buildings with positive source peaks span residential and nonresidential types and demand levels; not a statistically representative city sample.',
            'All three alternatives use the same retained building IDs and 2021–2023 temperature records, with 8,760 hourly values per year.',
            'Source peaks are assumed watts, not verified. Peak-scaled and HDD annual energies depend on this assumption.',
            'Envelope: 20°C indoors, 3 m per storey, 40% glazing, all perimeter walls exposed. Literature component U-values are proxies, not measured building properties.',
            'HDD18 is flat within each day and energy-matched to peak-scaled. Envelope annual energy is independent unless equal-energy comparison is selected.',
            'Equal-energy mode normalizes each building and year to its peak-scaled annual energy; original imported series remain unchanged.',
            'Shared temperature shapes impose synchronized peaks; no occupant diversity, DHW, measured validation or subhourly peaks.',
            'Feeder sizing is an equivalent-feeder screen, not connected branch analysis.'])
    cache.set('heating-study:' + response['run_id'], dict(user_id=user_id, response=response, matrices=matrices), 3600)
    return response
