"""Authenticated endpoints for non-destructive comparison and explicit application."""
import requests
from django.core.cache import cache
from rest_framework import serializers
from rest_framework.response import Response
from rest_framework.views import APIView

from .access_control import assert_can_edit_project, assert_can_view_project
from .intelligence.burlington_comparison import CATALOG as MODEL_CATALOG, MODEL_IDS, BurlingtonStudyRequest, available_buildings, run_burlington_comparison
from .intelligence.heating_studies import HeatingStudyRequest, consumer_building_queryset, persist_selected_profile, run_heating_study
from .intelligence.heating_weather import LOCATIONS
from .intelligence.heating_energy import ENERGY_PRESETS


class HeatingModelCatalogView(APIView):
    def get(self, request):
        return Response(dict(models=MODEL_CATALOG, locations=LOCATIONS, energy_presets=ENERGY_PRESETS))


class HeatingProfileComparisonView(APIView):
    def post(self, request):
        serializer = BurlingtonStudyRequest(data=request.data)
        serializer.is_valid(raise_exception=True)
        options = serializer.validated_data
        assert_can_view_project(request.user, options['project_id'])
        try:
            result = run_burlington_comparison(options, request.user.id)
        except requests.RequestException:
            return Response({'detail': 'Historical weather could not be retrieved. Retry later; no profiles were changed.'}, status=503)
        except ValueError as exc:
            raise serializers.ValidationError({'detail': str(exc)}) from exc
        return Response(result)


class HeatingStudyBuildingsView(APIView):
    def get(self, request):
        serializer = BurlingtonStudyRequest(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        options = serializer.validated_data
        assert_can_view_project(request.user, options['project_id'])
        _, _, buildings = available_buildings(options)
        buildings = buildings.filter(floor_area__gt=0)
        search = request.query_params.get('search', '').strip()[:100]
        if search:
            buildings = buildings.filter(name__icontains=search)
        return Response(dict(count=buildings.count(), results=list(buildings.order_by('id').values('id', 'name', 'floor_area')[:50])))


class ApplyHeatingProfileView(APIView):
    def post(self, request):
        class ApplyRequest(serializers.Serializer):
            run_id = serializers.UUIDField()
            model = serializers.ChoiceField(choices=sorted(MODEL_IDS))
        validation = ApplyRequest(data=request.data)
        validation.is_valid(raise_exception=True)
        run_id = str(validation.validated_data['run_id'])
        model_id = validation.validated_data['model']
        study = cache.get(f'heating-study:{run_id}')
        if not study or study['user_id'] != request.user.id:
            return Response({'detail': 'Comparison expired or is unavailable. Run it again.'}, status=404)
        assert_can_edit_project(request.user, study['response']['project_id'])
        if model_id not in study['matrices']:
            raise serializers.ValidationError('This model was not generated in the comparison.')
        try:
            result = persist_selected_profile(study['response'], study['matrices'][model_id], model_id)
        except ValueError as exc:
            raise serializers.ValidationError({'detail': str(exc)}) from exc
        cache.delete(f'heating-study:{run_id}')
        return Response(result)
