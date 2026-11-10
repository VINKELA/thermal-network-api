"""Import a deterministic ten-building sample and all 90 model/year series."""
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from api.models import ImportedBuildingHeating, BurlingtonComparisonProfile


def select_representative_buildings(frame):
    """Stratify by use/form, then select nearest peak-demand quantiles without replacement."""
    # Positive source peaks are required for a useful three-way sizing comparison.
    frame = frame[frame.Heat_load.gt(0)].copy()
    residential = frame.BUILDINGTY.eq('RESIDENTIAL')
    specs = [('Residential detached', residential & frame.CALCTYPE.eq('detached'), [.1, .5, .9]),
             ('Residential townhouse', residential & frame.CALCTYPE.eq('townhouse'), [.5]),
             ('Residential low-rise', residential & frame.BUILDINGSU.eq('Low-rise Residential'), [.5]),
             ('Residential high-rise', residential & frame.BUILDINGSU.eq('High-rise Residential'), [.5]),
             ('Commercial', frame.BUILDINGTY.eq('COMMERCIAL'), [.1, .35, .65, .9])]
    selected = []
    for label, mask, quantiles in specs:
        for quantile in quantiles:
            pool = frame[mask & ~frame.BUILDINGID.isin([r['BUILDINGID'] for r in selected])].sort_values(['Heat_load', 'BUILDINGID'])
            if pool.empty:
                raise CommandError(f'No eligible building for {label}.')
            row = pool.iloc[round((len(pool)-1)*quantile)].to_dict()
            row['selection_reason'] = f'{label}; source-peak quantile {quantile:g}'
            selected.append(row)
    return selected


class Command(BaseCommand):
    help = __doc__

    def add_arguments(self, parser):
        parser.add_argument('--project-id', type=int, required=True)
        parser.add_argument('--data-dir', type=Path, required=True)

    def handle(self, *args, **options):
        root = options['data_dir']
        frame = pd.read_csv(root/'burlington_clean_building_profiles/retained_buildings.csv', dtype={'BUILDINGID': str})
        selected = select_representative_buildings(frame)
        links = {p.source_building_id: p.building_id for p in ImportedBuildingHeating.objects.filter(building__project_id=options['project_id'])}
        folders = {'burlington_envelope': ('burlington_clean_building_profiles', 'transmission_heat_kw'),
                   'burlington_peak': ('burlington_peak_scaled_profiles', 'heat_load_kw'),
                   'burlington_hdd18': ('burlington_hdd_profiles', 'heat_load_kw')}
        weather_bytes = (root/'burlington_hdd_profiles/hourly_hdd_weights.csv').read_bytes()
        weather_hash = hashlib.sha256(weather_bytes).hexdigest()
        indexes = {model: pd.read_csv(root/folder/'profile_index.csv', dtype={'BUILDINGID': str}).set_index('BUILDINGID') for model, (folder, _) in folders.items()}
        records = []
        for row in selected:
            source_id = row['BUILDINGID']
            if source_id not in links:
                raise CommandError(f'Building {source_id} has not been imported in this project.')
            for model, (folder, column) in folders.items():
                path = root/folder/indexes[model].loc[source_id, 'profile_file']
                profile = pd.read_csv(path)
                dates = pd.to_datetime(profile.Date)
                for year in (2021, 2022, 2023):
                    mask = dates.dt.year.eq(year)
                    times = dates[mask]
                    values = profile.loc[mask, column].to_numpy(dtype=float)
                    expected = pd.date_range(f'{year}-01-01', periods=8760, freq='h')
                    if not np.array_equal(times.to_numpy(), expected.to_numpy()) or not np.isfinite(values).all() or np.any(values < 0):
                        raise CommandError(f'Incomplete or invalid series: {path}, {year}')
                    provenance = {key: row[key] for key in ('selection_reason', 'ua_w_k', 'envelope_m2', 'wall_u_w_m2k', 'roof_u_w_m2k', 'window_u_w_m2k')}
                    provenance.update(source_building_id=source_id, source_file=str(path.relative_to(root)),
                        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), weather_sha256=weather_hash,
                        source_peak_units='W assumed; unverified', weather_checksum_file='burlington_hdd_profiles/hourly_hdd_weights.csv')
                    records.append(dict(building_id=links[source_id], model_id=model, year=year,
                        timestamps=[t.isoformat() for t in times], hourly_kw=values.tolist(), provenance=provenance))
        # Validation precedes any mutation; rerunning updates the same keys rather than duplicating data.
        with transaction.atomic():
            selected_ids = [links[row['BUILDINGID']] for row in selected]
            BurlingtonComparisonProfile.objects.filter(building__project_id=options['project_id']).exclude(building_id__in=selected_ids).delete()
            for record in records:
                key = {k: record.pop(k) for k in ('building_id', 'model_id', 'year')}
                BurlingtonComparisonProfile.objects.update_or_create(**key, defaults=record)
        manifest = pd.DataFrame(selected)[['BUILDINGID','BUILDINGTY','BUILDINGSU','EffArea','Heat_load','selection_reason']]
        manifest['app_building_id'] = manifest.BUILDINGID.map(links)
        manifest.to_csv(root/'burlington_comparison_sample.csv', index=False)
        self.stdout.write(self.style.SUCCESS('Loaded 10 buildings × 3 models × 3 years = 90 series; 788,400 hourly values.'))
