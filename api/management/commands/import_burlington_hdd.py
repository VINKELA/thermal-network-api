"""Import the complete retained inventory with exact compact hourly HDD profiles."""
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from api.models import (Project, ProjectMembership, Building, Country, Province, City,
                        ImportedHeatingShape, ImportedBuildingHeating)


class Command(BaseCommand):
    help = __doc__

    def add_arguments(self, parser):
        parser.add_argument('--owner', required=True)
        parser.add_argument('--data-dir', type=Path, required=True)

    def handle(self, *args, **options):
        owner=get_user_model().objects.get(username=options['owner'])
        root=options['data_dir']
        buildings=pd.read_csv(root/'burlington_clean_building_profiles/retained_buildings.csv',dtype={'BUILDINGID':str})
        annual=pd.read_csv(root/'burlington_hdd_profiles/annual_hdd_results.csv',dtype={'BUILDINGID':str})
        weather=pd.read_csv(root/'burlington_hdd_profiles/hourly_hdd_weights.csv')
        assert len(buildings)==46996 and buildings.BUILDINGID.is_unique
        assert len(annual)==len(buildings)*3 and not annual.duplicated(['BUILDINGID','year']).any()
        assert set(annual.BUILDINGID)==set(buildings.BUILDINGID)
        assert np.isfinite(buildings[['Latitude','Longitude','EffArea']]).all().all()
        if Project.objects.filter(name='Burlington HDD Study').exists():
            raise CommandError('Project already exists; refusing duplicate import.')
        annual=annual.set_index(['BUILDINGID','year'])
        mapping={'RESIDENTIAL':'residential','COMMERCIAL':'commercial','MERCANTILE':'commercial',
                 'INDUSTRIAL':'industrial','ASSEMBLY':'public','GOVERNMENT':'public'}
        with transaction.atomic():
            country,_=Country.objects.get_or_create(code='CAN',defaults={'name':'Canada'})
            province,_=Province.objects.get_or_create(name='Ontario',country=country)
            city,_=City.objects.get_or_create(name='Burlington',province=province,country=country,
                                            defaults={'latitude':43.3255,'longitude':-79.799})
            project=Project.objects.create(name='Burlington HDD Study',city=city,province=province,country=country,
                description='46,996 retained buildings; HDD18 synthetic hourly heating profiles 2021–2023. Annual energies from peak-scaled profiles, source peaks assumed W. Imported inventory is not a connected network.')
            ProjectMembership.objects.create(project=project,user=owner,role='owner')
            for year,group in weather.groupby('year'):
                assert len(group)==8760 and np.isclose(group.hourly_weight.sum(),1)
                ImportedHeatingShape.objects.create(project=project,year=int(year),
                    timestamps=group.Date.tolist(),hourly_weights=group.hourly_weight.tolist(),
                    provenance={'method':'daily HDD18; flat within day','timezone':'source labels retained; timezone unverified',
                                'energy':'synthetic peak-scaled annual energy; original peaks assumed watts',
                                'source_manifest':json.loads((root/'burlington_hdd_profiles/manifest.json').read_text())})
            for start in range(0,len(buildings),1000):
                chunk=buildings.iloc[start:start+1000]
                objects=[]
                for row in chunk.itertuples():
                    peaks=annual.loc[row.BUILDINGID].hdd_peak_kw
                    floors=float(row.Storys)
                    objects.append(Building(project=project,name='Burlington '+row.BUILDINGID,
                        building_type=mapping.get(row.BUILDINGTY,'other'),latitude=row.Latitude,longitude=row.Longitude,
                        fsa=None if pd.isna(row.CFSAUID) else row.CFSAUID,tag='Burlington HDD18 imported',
                        floor_area=row.EffArea,height=floors*3,volume_m3=row.SHAPEAREA*floors*3,
                        number_of_floors=round(floors) if math.isclose(floors,round(floors),abs_tol=1e-6) else None,
                        heating_demand=float(peaks.max()),load_type='heating_dominant'))
                Building.objects.bulk_create(objects,batch_size=500)
                links=[]
                for obj,row in zip(objects,chunk.itertuples()):
                    data=annual.loc[row.BUILDINGID]
                    links.append(ImportedBuildingHeating(building=obj,source_building_id=row.BUILDINGID,
                        annual_energy_kwh={str(y):float(data.loc[y].annual_energy_target_kwh) for y in (2021,2022,2023)},
                        annual_peak_kw={str(y):float(data.loc[y].hdd_peak_kw) for y in (2021,2022,2023)}))
                ImportedBuildingHeating.objects.bulk_create(links,batch_size=500)
                if start%10000==0:
                    self.stdout.write(f'Imported {min(start+1000,len(buildings))}/{len(buildings)}')
        self.stdout.write(self.style.SUCCESS(f'Project {project.id}: {len(buildings)} buildings, 3 full hourly years linked.'))
