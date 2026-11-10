"""Authenticated inventory access to exact compact HDD profiles."""
from datetime import datetime, timedelta
from django.db.models import Q
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import NotFound, ValidationError
from .models import Building, ImportedHeatingShape, ImportedBuildingHeating, Node
from .access_control import apply_project_access


class ImportedHeatingView(APIView):
    def get(self, request):
        try:
            project = int(request.query_params.get('project_id', ''))
            year = int(request.query_params.get('year', '2023'))
        except ValueError:
            raise ValidationError('A valid project_id and year are required.')
        shapes = apply_project_access(ImportedHeatingShape.objects.all(), request.user, 'project_id').filter(project_id=project)
        if not shapes.exists():
            raise NotFound('Imported profiles not found.')
        action = request.query_params.get('action', 'aggregate')
        buildings = Building.objects.filter(project_id=project, imported_heating__isnull=False)
        network = request.query_params.get('network_id')
        if network and network not in ('all','null'):
            buildings = buildings.filter(id__in=Node.objects.filter(network_id=network, network__project_id=project).values('building_id'))
        if action == 'directory':
            if request.query_params.get('type') == 'boiler':
                return Response({'results': [], 'next': None, 'count': 0})
            search = request.query_params.get('search','').strip()
            if search:
                buildings = buildings.filter(Q(name__icontains=search) | Q(fsa__icontains=search))
            try:
                page=max(1,int(request.query_params.get('page',1)))
            except ValueError:
                raise ValidationError('Invalid page')
            count=buildings.count()
            rows=[dict(id=b.id, building=b.id, node_type='building', name=b.name,
                       building_type=b.building_type) for b in buildings.order_by('id')[(page-1)*10:page*10]]
            return Response({'results':rows,'next':page+1 if page*10<count else None,'count':count})
        shape=shapes.filter(year=year).first()
        if shape is None:
            raise ValidationError('Select one of the imported years: 2021, 2022, 2023.')
        if action == 'individual':
            building=buildings.filter(pk=request.query_params.get('building')).first()
            if building is None:
                raise NotFound('Building not found in this project.')
            energy=building.imported_heating.annual_energy_kwh[str(year)]
            return Response({'results':[dict(start_date=t,
                end_date=(datetime.fromisoformat(t)+timedelta(hours=1)).isoformat(),
                value=energy*w,unit='kW',profile_type='heating',source_model='burlington_hdd18_import_v1')
                for t,w in zip(shape.timestamps,shape.hourly_weights)]})
        if action != 'aggregate':
            raise ValidationError('Unknown profile action')
        energies=ImportedBuildingHeating.objects.filter(building__in=buildings).values_list('annual_energy_kwh',flat=True)
        total=sum(e[str(year)] for e in energies)
        return Response({'results':[dict(timestamp=t,total_supply=0,total_demand=total*w,net_balance=-total*w)
                         for t,w in zip(shape.timestamps,shape.hourly_weights)]})
