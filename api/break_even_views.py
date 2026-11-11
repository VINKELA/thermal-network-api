"""Authenticated, bounded background runs with owner-scoped snapshots on disk."""
import calendar
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import threading
import time
import uuid

import numpy as np
from celery import states
from django.conf import settings
from django.db import close_old_connections
from django.db.models import Q
from rest_framework import serializers
from rest_framework.exceptions import NotFound
from rest_framework.response import Response
from rest_framework.views import APIView
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError

from .access_control import assert_can_view_project
from .break_even_tasks import run_break_even_study
from .models import Building, BuildingCluster, ImportedBuildingHeating, ImportedHeatingShape, LoadProfile, Network, Node
from .utils.break_even import demo_profile, run_study, schema, validate_options

logger = logging.getLogger(__name__)
# On Windows an open polling read can deny delete/replace access to its target.
# Keep file handles out of the writer's replacement window in this process.
_snapshot_lock = threading.RLock()


class StudyRequest(serializers.Serializer):
    project_id = serializers.IntegerField(min_value=1)
    mode = serializers.ChoiceField(choices=['demo', 'custom', 'network'], default='demo')
    network_id = serializers.IntegerField(min_value=1, required=False)
    year = serializers.IntegerField(min_value=1950, max_value=2200, default=2025)
    network_profile = serializers.ChoiceField(choices=['active', 'imported'], default='active')
    source_model = serializers.CharField(max_length=100, required=False, allow_blank=True, default='')
    hourly_kw = serializers.ListField(child=serializers.FloatField(min_value=0, max_value=1000000),
                                     min_length=8760, max_length=8784, required=False)
    profile_label = serializers.CharField(max_length=200, default='Uploaded hourly profile')
    options = serializers.JSONField(default=dict)
    residual_credit = serializers.BooleanField(default=False)

    def validate(self, attrs):
        try:
            attrs['options'] = validate_options(attrs['options'])
        except ValueError as exc:
            raise serializers.ValidationError({'options': str(exc)}) from exc
        if attrs['mode'] == 'network' and not attrs.get('network_id'):
            raise serializers.ValidationError('Choose a network.')
        if attrs['mode'] == 'custom':
            values = attrs.get('hourly_kw', [])
            if len(values) not in (8760, 8784) or not np.isfinite(values).all() or not any(values):
                raise serializers.ValidationError('Upload 8760 or 8784 finite hourly kW values with positive demand.')
        return attrs


def folder(run_id):
    return Path(settings.BASE_DIR) / 'data' / 'break_even_runs' / str(uuid.UUID(str(run_id)))


def atomic_json(path, data):
    content = json.dumps(data, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    try:
        temporary.write_text(content, encoding='utf-8')
        with _snapshot_lock:
            for attempt in range(8):
                try:
                    temporary.replace(path)
                    break
                except OSError as exc:
                    # Other processes (including scanners) do not share our lock.
                    # Retry Windows access/sharing/lock violations, not all I/O errors.
                    if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 7:
                        raise
                    time.sleep(min(.01 * 2 ** attempt, .2))
    finally:
        # A failed write must leave the old JSON intact and no stale partial snapshot.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            logger.warning('Could not remove temporary study snapshot %s', temporary, exc_info=True)


def read_json(path):
    with _snapshot_lock:
        content = path.read_text(encoding='utf-8')
    return json.loads(content)


def network_profile(data):
    """Read one unambiguous, complete annual heating series per consumer."""
    network = Network.objects.filter(pk=data['network_id'], project_id=data['project_id']).first()
    if not network:
        raise NotFound('Network not found.')
    nodes = Node.objects.filter(network=network).exclude(node_type__in=['producer', 'boiler', 'junction'])
    if nodes.filter(network_node__isnull=False).exists():
        raise ValueError('Nested networks need an explicitly aggregated CSV profile for this study.')
    ids = set(nodes.filter(building__isnull=False).values_list('building_id', flat=True))
    clusters = nodes.filter(cluster__isnull=False).values_list('cluster_id', flat=True)
    ids.update(BuildingCluster.objects.filter(cluster_id__in=clusters).values_list('building_id', flat=True))
    if not ids or len(ids) > 500:
        raise ValueError('Select a network with 1–500 assigned consumer buildings, or upload an aggregate CSV.')
    buildings = list(Building.objects.filter(pk__in=ids, project_id=network.project_id)
                     .filter(Q(waste_heat_source_type='') | Q(waste_heat_source_type__isnull=True)).values_list('pk', flat=True))
    if set(buildings) != ids:
        raise ValueError('Consumer assignments contain a heat source or cross-project building.')
    year = data['year']
    hours = 8784 if calendar.isleap(year) else 8760
    provenance = dict(mode='network', network_id=network.id, project_id=network.project_id,
        year=year, building_ids=sorted(ids), name=network.name,
        network_profile=data['network_profile'], source_model=data['source_model'],
        scope='Aggregate heating demand of assigned consumers, modeled as one equivalent trunk; distance is an independent study input.')
    if data['network_profile'] == 'imported':
        shape = ImportedHeatingShape.objects.filter(project_id=network.project_id, year=year).first()
        records = list(ImportedBuildingHeating.objects.filter(building_id__in=ids))
        if not shape or len(records) != len(ids) or len(shape.hourly_weights) != hours:
            raise ValueError('The imported annual heating dataset does not completely cover this network and year.')
        weights = np.asarray(shape.hourly_weights, dtype=float)
        if not np.isfinite(weights).all() or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError('Imported heating weights are invalid.')
        energies = [r.annual_energy_kwh.get(str(year)) for r in records]
        if any(v is None or not np.isfinite(v) or v < 0 for v in energies):
            raise ValueError('An imported building has no valid energy value for this year.')
        profile = weights / sum(weights) * sum(energies)
        provenance.update(dataset='Imported heating', source=shape.provenance)
    else:
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
        query = LoadProfile.objects.filter(building_id__in=ids, profile_type='heating',
            start_date__gte=start, start_date__lt=end)
        if data['source_model']:
            query = query.filter(source_model=data['source_model'])
        # Choose a complete scope per building; never add a shared profile to a network-specific one.
        specific = set(query.filter(network=network).values_list('building_id', flat=True))
        query = query.filter(Q(network=network) | Q(network__isnull=True, building_id__in=ids - specific))
        values = {bid: np.full(hours, np.nan) for bid in ids}
        models = {bid: set() for bid in ids}
        for row in query.values('building_id', 'start_date', 'end_date', 'value', 'unit', 'source_model').iterator(chunk_size=10000):
            index = (row['start_date'] - start).total_seconds() / 3600
            bid = row['building_id']
            if (not index.is_integer() or not 0 <= index < hours or row['end_date'] - row['start_date'] != timedelta(hours=1)
                or row['unit'] != 'kW' or not np.isfinite(row['value']) or row['value'] < 0):
                raise ValueError('Saved profiles must contain exact one-hour, nonnegative kW intervals on a UTC annual grid.')
            if np.isfinite(values[bid][int(index)]):
                raise ValueError('Duplicate heating intervals or competing models exist. Specify a source model or upload a CSV.')
            values[bid][int(index)] = row['value']
            models[bid].add(row['source_model'])
        if any(not np.isfinite(v).all() for v in values.values()) or any(len(m) != 1 for m in models.values()):
            raise ValueError('Saved hourly profiles do not completely cover every consumer for this year. Choose imported heating or upload a complete CSV.')
        profile = sum(values.values())
        provenance.update(dataset='Active LoadProfile', source_models={str(k): sorted(v) for k, v in models.items()})
    if max(profile) <= 0:
        raise ValueError('The selected profile has no heating demand.')
    return profile.tolist(), provenance


def execute(run_id, owner_id, data, progress_callback=None):
    close_old_connections()
    state = dict(run_id=run_id, task_id=run_id, runner='celery', user_id=owner_id, project_id=data['project_id'], status='running',
                 message='Preparing the study input snapshot', updated_at=time.time())
    def progress(message):
        state.update(message=message, updated_at=time.time())
        atomic_json(folder(run_id) / 'status.json', state)
        if progress_callback:
            progress_callback(message)
    try:
        progress('Reading hourly heating demand')
        if data['mode'] == 'network':
            profile, provenance = network_profile(data)
        elif data['mode'] == 'custom':
            profile, provenance = data['hourly_kw'], dict(mode='custom', name=data['profile_label'])
        else:
            profile = demo_profile(data['options']['demo_peak_kw']).tolist()
            provenance = dict(mode='demo', name='Synthetic seasonal / daily annual heating profile',
                              note='Illustrative 8760-hour profile, not measured building data.')
        result = run_study(profile, data['options'], data['residual_credit'], progress)
        result.update(run_id=run_id, task_id=run_id, runner='celery', project_id=data['project_id'], profile=provenance)
        atomic_json(folder(run_id) / 'result.json', result)
        state.update(status='complete', message='Break-even study complete', updated_at=time.time())
    except (ValueError, NotFound) as exc:
        state.update(status='failed', message=str(exc), updated_at=time.time())
    except OSError:
        # Let Celery retry failed snapshot I/O; never replace valid results with
        # a generic calculation failure merely because a file was locked.
        raise
    except Exception:
        logger.exception('Break-even study %s failed', run_id)
        state.update(status='failed', message='The study could not finish. Check the API log and retry.', updated_at=time.time())
    finally:
        try:
            atomic_json(folder(run_id) / 'status.json', state)
        finally:
            close_old_connections()
    return dict(run_id=run_id, status=state['status'])


class BreakEvenStartView(APIView):
    def get(self, request):
        return Response(schema())

    def post(self, request):
        validation = StudyRequest(data=request.data)
        validation.is_valid(raise_exception=True)
        data = validation.validated_data
        assert_can_view_project(request.user, data['project_id'])
        if data['mode'] == 'network' and not Network.objects.filter(pk=data['network_id'], project_id=data['project_id']).exists():
            raise NotFound('Network not found.')
        run_id = str(uuid.uuid4())
        state = dict(run_id=run_id, task_id=run_id, runner='celery', user_id=request.user.id,
                     project_id=data['project_id'], status='queued',
                     message='Queued for the Celery break-even worker', updated_at=time.time())
        try:
            # Save inputs before publication so a queued run is reproducible even
            # if its worker starts after the web server restarts.
            atomic_json(folder(run_id) / 'request.json', dict(user_id=request.user.id, request=data))
            atomic_json(folder(run_id) / 'status.json', state)
            run_break_even_study.apply_async(args=[run_id, request.user.id, data], task_id=run_id,
                                             queue='break_even', retry=False)
        except (OperationalError, RedisError, OSError):
            logger.exception('Could not queue break-even study %s', run_id)
            state.update(status='failed', message='The study could not be queued. Check Redis, the Celery worker and study-folder access.', updated_at=time.time())
            try:
                atomic_json(folder(run_id) / 'status.json', state)
            except OSError:
                logger.exception('Could not save queue failure for break-even study %s', run_id)
            return Response({'detail': state['message']}, status=503)
        return Response(dict(run_id=run_id, task_id=run_id, runner='celery', status='queued'), status=202)


class BreakEvenStatusView(APIView):
    def get(self, request, run_id):
        path = folder(run_id)
        if not (path / 'status.json').exists():
            raise NotFound('Study not found.')
        state = read_json(path / 'status.json')
        if state['user_id'] != request.user.id:
            raise NotFound('Study not found.')
        assert_can_view_project(request.user, state['project_id'])
        state.pop('user_id')
        if state['status'] in ('queued', 'running') and state.get('runner') == 'celery':
            try:
                task = run_break_even_study.AsyncResult(state['task_id'])
                if task.state in (states.FAILURE, states.REVOKED):
                    state.update(status='failed', message='The Celery worker could not complete or save this study. Check the worker log, then retry.')
                elif task.state == states.RETRY:
                    state['message'] = 'Celery is retrying the study after a temporary worker or file-access error.'
            except (OperationalError, RedisError, OSError):
                # Previously persisted state remains readable during a broker outage.
                logger.warning('Celery status unavailable for break-even study %s', run_id)
        elif state['status'] == 'running' and time.time() - state['updated_at'] > 900:
            state.update(status='interrupted', message='This older local job was interrupted. Start a new Celery study.')
        if state['status'] == 'complete':
            state['result'] = read_json(path / 'result.json')
        return Response(state)
