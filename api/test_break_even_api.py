"""Request validation, saved-profile integrity and study ownership checks."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
from unittest.mock import patch
import uuid

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import path
from rest_framework.test import APIClient
from kombu.exceptions import OperationalError
from redis.exceptions import RedisError
from api.break_even_tasks import run_break_even_study

from api.break_even_views import (BreakEvenStartView, BreakEvenStatusView, StudyRequest,
    atomic_json, execute, folder, network_profile, read_json)
from api.models import (Building, ImportedBuildingHeating, ImportedHeatingShape,
    LoadProfile, Network, Node, Project, ProjectMembership)

urlpatterns = [path('break-even/', BreakEvenStartView.as_view()),
               path('break-even/<uuid:run_id>/', BreakEvenStatusView.as_view())]


def windows_lock_error(code=5):
    error = PermissionError('Simulated Windows file lock')
    error.winerror = code
    return error


class SnapshotConcurrencyTests(SimpleTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.target = Path(self.temp.name) / 'status.json'
        atomic_json(self.target, {'status': 'running'})

    def test_transient_windows_replacement_errors_are_retried(self):
        original_replace = Path.replace
        calls = []
        def locked_replace(source, destination):
            calls.append(source)
            if len(calls) <= 3:
                raise windows_lock_error((5, 32, 33)[len(calls) - 1])
            return original_replace(source, destination)
        with patch.object(Path, 'replace', locked_replace), patch('api.break_even_views.time.sleep') as sleep:
            atomic_json(self.target, {'status': 'complete'})
        self.assertEqual(len(calls), 4)
        self.assertEqual(sleep.call_count, 3)
        self.assertEqual(read_json(self.target), {'status': 'complete'})
        self.assertEqual(list(self.target.parent.glob('*.tmp')), [])

    def test_persistent_error_is_bounded_and_preserves_previous_snapshot(self):
        with patch.object(Path, 'replace', side_effect=windows_lock_error()) as replace, patch('api.break_even_views.time.sleep'):
            with self.assertRaises(PermissionError):
                atomic_json(self.target, {'status': 'complete'})
        self.assertEqual(replace.call_count, 8)
        self.assertEqual(read_json(self.target), {'status': 'running'})
        self.assertEqual(list(self.target.parent.glob('*.tmp')), [])

    def test_unrelated_io_errors_are_not_retried(self):
        with patch.object(Path, 'replace', side_effect=OSError('Disk full')) as replace, patch('api.break_even_views.time.sleep') as sleep:
            with self.assertRaises(OSError):
                atomic_json(self.target, {'status': 'complete'})
        replace.assert_called_once()
        sleep.assert_not_called()
        self.assertEqual(read_json(self.target), {'status': 'running'})
        self.assertEqual(list(self.target.parent.glob('*.tmp')), [])

    def test_invalid_json_value_preserves_previous_snapshot(self):
        with self.assertRaises(ValueError):
            atomic_json(self.target, {'value': float('nan')})
        self.assertEqual(read_json(self.target), {'status': 'running'})
        self.assertEqual(list(self.target.parent.glob('*.tmp')), [])

    def test_polling_reader_closes_its_handle_before_writer_replaces(self):
        reading = threading.Event()
        release_reader = threading.Event()
        writer_ready = threading.Event()
        replaced = threading.Event()
        errors, values = [], []
        original_read, original_replace = Path.read_text, Path.replace
        def slow_read(path, *args, **kwargs):
            if path != self.target:
                return original_read(path, *args, **kwargs)
            with path.open(encoding='utf-8') as stream:
                reading.set()
                if not release_reader.wait(5):
                    raise RuntimeError('Reader was not released')
                return stream.read()
        def observe_replace(source, destination):
            replaced.set()
            return original_replace(source, destination)
        def reader():
            try:
                values.append(read_json(self.target))
            except Exception as exc:
                errors.append(exc)
        def writer():
            try:
                writer_ready.set()
                atomic_json(self.target, {'status': 'complete'})
            except Exception as exc:
                errors.append(exc)
        r, w = threading.Thread(target=reader), threading.Thread(target=writer)
        with patch.object(Path, 'read_text', slow_read), patch.object(Path, 'replace', observe_replace):
            r.start()
            try:
                self.assertTrue(reading.wait(5))
                w.start()
                self.assertTrue(writer_ready.wait(5))
                self.assertFalse(replaced.wait(.05))
            finally:
                release_reader.set()
                r.join(5)
                if w.ident is not None:
                    w.join(5)
        self.assertFalse(r.is_alive())
        self.assertFalse(w.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(values, [{'status': 'running'}])
        self.assertEqual(read_json(self.target), {'status': 'complete'})


@override_settings(ROOT_URLCONF='api.test_break_even_api')
class BreakEvenAPITests(TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.setting = override_settings(BASE_DIR=self.temp.name)
        self.setting.enable()
        self.addCleanup(self.setting.disable)
        self.user = get_user_model().objects.create_user(username='break-even-owner')
        self.other = get_user_model().objects.create_user(username='break-even-other')
        self.project = Project.objects.create(name='Break-even test')
        ProjectMembership.objects.create(project=self.project, user=self.user, role='viewer')
        self.network = Network.objects.create(project=self.project, name='Test network', network_type='district_heating')
        self.building = Building.objects.create(project=self.project, name='Consumer', building_type='residential')
        Node.objects.create(network=self.network, building=self.building, node_type='consumer')
        self.client = APIClient()
        self.client.force_authenticate(self.user)

    def request(self, **changes):
        return dict(project_id=self.project.id, mode='network', network_id=self.network.id,
                    year=2024, network_profile='active', source_model='', **changes)

    def test_schema_requires_authentication(self):
        self.client.force_authenticate(None)
        self.assertIn(self.client.get('/break-even/').status_code, (401, 403))
        self.client.force_authenticate(self.user)
        response = self.client.get('/break-even/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['defaults']['reference_cost'], 4000)

    def test_unauthorized_project_and_network_cannot_start(self):
        self.client.force_authenticate(self.other)
        self.assertEqual(self.client.post('/break-even/', {'project_id': self.project.id}, format='json').status_code, 404)
        self.client.force_authenticate(self.user)
        other_project = Project.objects.create(name='Other project')
        other_network = Network.objects.create(project=other_project, name='Private', network_type='district_heating')
        response = self.client.post('/break-even/', {**self.request(), 'network_id': other_network.id}, format='json')
        self.assertEqual(response.status_code, 404)

    def test_invalid_inputs_fail_before_starting_job(self):
        for payload in ({'options': {'alpha': 2}}, {'mode': 'custom', 'hourly_kw': [1] * 24},
                        {'mode': 'network'}, {'options': {'years': 2.5}}):
            response = self.client.post('/break-even/', {'project_id': self.project.id, **payload}, format='json')
            self.assertEqual(response.status_code, 400)
        validation = StudyRequest(data={'project_id': self.project.id, 'mode': 'custom', 'hourly_kw': [float('nan')] * 8760})
        self.assertFalse(validation.is_valid())

    def test_start_queues_snapshot_without_changing_network(self):
        with patch('api.break_even_views.run_break_even_study.apply_async') as enqueue:
            response = self.client.post('/break-even/', {'project_id': self.project.id}, format='json')
            self.assertEqual(response.status_code, 202)
            self.assertTrue((folder(response.data['run_id']) / 'status.json').exists())
            self.assertTrue((folder(response.data['run_id']) / 'request.json').exists())
            self.assertEqual(response.data['status'], 'queued')
            self.assertEqual(response.data['runner'], 'celery')
            self.assertEqual(enqueue.call_args.kwargs['task_id'], response.data['run_id'])
            self.assertEqual(enqueue.call_args.kwargs['queue'], 'break_even')
        self.assertEqual(Network.objects.count(), 1)
        self.assertEqual(LoadProfile.objects.count(), 0)

    def test_only_owner_with_current_project_access_can_read_saved_run(self):
        run = str(uuid.uuid4())
        atomic_json(folder(run) / 'status.json', dict(run_id=run, user_id=self.user.id,
            project_id=self.project.id, status='complete'))
        atomic_json(folder(run) / 'result.json', {'test': 'saved result'})
        self.assertEqual(self.client.get(f'/break-even/{run}/').data['result']['test'], 'saved result')
        self.client.force_authenticate(self.other)
        self.assertEqual(self.client.get(f'/break-even/{run}/').status_code, 404)
        self.client.force_authenticate(self.user)
        ProjectMembership.objects.filter(project=self.project).delete()
        self.assertEqual(self.client.get(f'/break-even/{run}/').status_code, 404)

    def test_worker_saves_complete_reopenable_annual_snapshot(self):
        run = str(uuid.uuid4())
        validation = StudyRequest(data={'project_id': self.project.id, 'mode': 'demo'})
        validation.is_valid(raise_exception=True)
        task = run_break_even_study.apply(args=[run, self.user.id, validation.validated_data], task_id=run, throw=True)
        self.assertEqual(task.result['status'], 'complete')
        response = self.client.get(f'/break-even/{run}/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data['status'], 'complete')
        result = response.data['result']
        self.assertEqual(result['summary']['hours'], 8760)
        self.assertEqual(result['inputs']['options']['reference_cost'], 4000)
        self.assertEqual(len(result['selected']), 4)
        self.assertEqual(result['run_id'], run)
        self.assertNotIn('user_id', response.data)
        self.assertEqual(result['runner'], 'celery')
        self.assertEqual(Network.objects.count(), 1)
        self.assertEqual(LoadProfile.objects.count(), 0)

    def test_queue_connection_failure_is_reported_without_starting_local_thread(self):
        with patch('api.break_even_views.run_break_even_study.apply_async', side_effect=OperationalError('Broker down')):
            with self.assertLogs('api.break_even_views', level='ERROR'):
                response = self.client.post('/break-even/', {'project_id': self.project.id}, format='json')
        self.assertEqual(response.status_code, 503)
        snapshots = list((Path(self.temp.name) / 'data' / 'break_even_runs').glob('*/status.json'))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(read_json(snapshots[0])['status'], 'failed')

    def test_missing_profile_error_survives_a_locked_final_status_file(self):
        run = str(uuid.uuid4())
        validation = StudyRequest(data=self.request())
        validation.is_valid(raise_exception=True)
        original_replace = Path.replace
        failures = []
        def locked_final_update(source, target):
            snapshot = read_json(source)
            if target.name == 'status.json' and snapshot.get('status') == 'failed' and len(failures) < 2:
                failures.append(source)
                raise windows_lock_error()
            return original_replace(source, target)
        with patch.object(Path, 'replace', locked_final_update), patch('api.break_even_views.time.sleep'):
            task = run_break_even_study.apply(args=[run, self.user.id, validation.validated_data], task_id=run, throw=True)
        self.assertEqual(task.result['status'], 'failed')
        self.assertEqual(len(failures), 2)
        response = self.client.get(f'/break-even/{run}/')
        self.assertEqual(response.data['status'], 'failed')
        self.assertIn('completely cover', response.data['message'])
        self.assertEqual(list(folder(run).glob('*.tmp')), [])

    def test_queued_job_does_not_expire_while_waiting_for_worker(self):
        run = str(uuid.uuid4())
        atomic_json(folder(run) / 'status.json', dict(run_id=run, task_id=run, runner='celery', user_id=self.user.id,
            project_id=self.project.id, status='queued', message='Queued', updated_at=0))
        with patch('api.break_even_views.run_break_even_study.AsyncResult', return_value=SimpleNamespace(state='PENDING')):
            response = self.client.get(f'/break-even/{run}/')
        self.assertEqual(response.data['status'], 'queued')

    def test_worker_failure_and_backend_outage_are_visible_without_losing_snapshot(self):
        run = str(uuid.uuid4())
        atomic_json(folder(run) / 'status.json', dict(run_id=run, task_id=run, runner='celery', user_id=self.user.id,
            project_id=self.project.id, status='running', message='Calculating', updated_at=0))
        with patch('api.break_even_views.run_break_even_study.AsyncResult', return_value=SimpleNamespace(state='FAILURE')):
            self.assertEqual(self.client.get(f'/break-even/{run}/').data['status'], 'failed')
        with patch('api.break_even_views.run_break_even_study.AsyncResult', side_effect=RedisError('Backend down')):
            with self.assertLogs('api.break_even_views', level='WARNING'):
                response = self.client.get(f'/break-even/{run}/')
        self.assertEqual(response.data['status'], 'running')

    def test_queued_worker_rechecks_project_access_before_reading_demand(self):
        run = str(uuid.uuid4())
        validation = StudyRequest(data={'project_id': self.project.id})
        validation.is_valid(raise_exception=True)
        ProjectMembership.objects.filter(project=self.project).delete()
        from rest_framework.exceptions import NotFound
        with self.assertRaises(NotFound), patch('api.break_even_views.execute') as worker:
            run_break_even_study.apply(args=[run, self.user.id, validation.validated_data], task_id=run, throw=True)
        worker.assert_not_called()

    def test_active_profile_rejects_gaps_and_competing_models(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        LoadProfile.objects.create(network=self.network, building=self.building, profile_type='heating',
            start_date=start, end_date=start + timedelta(hours=1), value=100, unit='kW')
        with self.assertRaisesRegex(ValueError, 'completely cover'):
            network_profile(self.request())
        LoadProfile.objects.create(network=self.network, building=self.building, profile_type='heating',
            start_date=start, end_date=start + timedelta(hours=1), value=100, unit='kW', source_model='another-model')
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            network_profile(self.request())

    def test_imported_leap_year_preserves_annual_heat_and_network_scope(self):
        ImportedHeatingShape.objects.create(project=self.project, year=2024,
            timestamps=[], hourly_weights=[1] * 8784)
        ImportedBuildingHeating.objects.create(building=self.building, source_building_id='test',
            annual_energy_kwh={'2024': 87840}, annual_peak_kw={'2024': 10})
        values, source = network_profile({**self.request(), 'network_profile': 'imported'})
        self.assertEqual(len(values), 8784)
        self.assertAlmostEqual(sum(values), 87840)
        self.assertEqual(source['building_ids'], [self.building.id])

    def test_complete_active_leap_year_is_not_rescaled(self):
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        LoadProfile.objects.bulk_create([LoadProfile(network=self.network, building=self.building,
            profile_type='heating', start_date=start + timedelta(hours=i),
            end_date=start + timedelta(hours=i + 1), value=10, unit='kW') for i in range(8784)])
        values, source = network_profile(self.request())
        self.assertEqual(len(values), 8784)
        self.assertAlmostEqual(sum(values), 87840)
