"""Scientific invariants, API isolation and non-destructive comparison regression tests."""
import calendar
import tempfile
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np
from django.contrib.auth import get_user_model
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase, override_settings
from rest_framework.test import APIClient

from api.models import Building, LoadProfile, Network, Project, ProjectMembership
from api.intelligence.heating_metrics import calculate_profile_metrics, longest_event, screen_equivalent_feeder
from api.intelligence.heating_models import MODEL_IDS, build_schedule, generate_building_profile, generate_heat_balance_profile
from api.intelligence.heating_studies import HeatingStudyRequest, read_existing_profiles
from api.intelligence.heating_weather import LOCATIONS, validate_weather_payload


def example_weather(year=2024):
    count = (366 if calendar.isleap(year) else 365) * 24
    timestamps = [datetime(year, 1, 1, tzinfo=timezone(timedelta(hours=-5))) + timedelta(hours=i) for i in range(count)]
    hours = np.arange(count)
    temperatures = 8 - 16 * np.cos(2 * np.pi * hours / count) + 4 * np.sin(2 * np.pi * hours / 24)
    return timestamps, temperatures, {'provider': 'Deterministic test fixture', 'year': year}


def valid_options(**overrides):
    serializer = HeatingStudyRequest(data={'project_id': 1, **overrides})
    serializer.is_valid(raise_exception=True)
    return serializer.validated_data


class HeatingModelTests(SimpleTestCase):
    def test_models_conserve_energy_in_both_calendar_lengths(self):
        for year in (2023, 2024):
            times, temperatures, _ = example_weather(year)
            for model in MODEL_IDS:
                values = generate_building_profile(model, temperatures, times,
                    dict(building_type='residential', floor_area=200, volume_m3=600, annual_kwh=24000), valid_options())
                self.assertEqual(len(values), len(times))
                self.assertTrue(np.all(np.isfinite(values) & (values >= 0)))
                self.assertAlmostEqual(values.sum(), 24000, places=6)

    def test_hdd_is_uniform_within_each_day_without_dhw(self):
        times, temperatures, _ = example_weather()
        values = generate_building_profile('hdd_daily_ca_v1', temperatures, times,
            dict(building_type='commercial', floor_area=200, volume_m3=600, annual_kwh=24000), valid_options(dhw_fraction=0))
        self.assertTrue(np.all(np.ptp(values.reshape(-1, 24), axis=1) == 0))

    def test_physical_law_matches_hand_calculation_and_does_not_cap_extremes(self):
        # H=(0.8*200 + .33*.5*600)/1000=.259 kW/K; no gains.
        values = generate_heat_balance_profile(np.array([0., -40.]), np.array([20., 20.]),
                                               np.zeros(2), 200, 600, valid_options())
        np.testing.assert_allclose(values, [.259 * 20, .259 * 60])

    def test_published_schedule_weekend_and_all_groups(self):
        times, _, _ = example_weather()
        weekday = build_schedule(times, 'A', 'Thermostat Setpoint-Heating')
        self.assertEqual(weekday[12], 22)  # January 1 2024, Monday (holidays deliberately not applied).
        self.assertEqual(weekday[5 * 24 + 12], 18)
        for group in 'ABCDEFGHI':
            for category in ('Thermostat Setpoint-Heating', 'Occupancy', 'Service Water Heating'):
                self.assertEqual(len(build_schedule(times, group, category)), 8784)

    def test_zero_energy_and_unknown_model(self):
        times, temperatures, _ = example_weather()
        building = dict(building_type='residential', floor_area=200, volume_m3=600, annual_kwh=0)
        for model in MODEL_IDS:
            self.assertEqual(generate_building_profile(model, temperatures, times, building, valid_options()).sum(), 0)
        with self.assertRaises(ValueError):
            generate_building_profile('typo', temperatures, times, building, valid_options())

    def test_nonfinite_and_invalid_inputs_are_rejected(self):
        for data in ({'intensity_kwh_m2': float('nan')}, {'delta_t_c': 0}, {'year': 9999},
                     {'models': ['typo']}, {'models': []}, {'dhw_fraction': -1}):
            self.assertFalse(HeatingStudyRequest(data={'project_id': 1, **data}).is_valid())

    def test_metrics_use_signed_bias_and_shared_threshold(self):
        reference = np.arange(24, dtype=float)
        metrics = calculate_profile_metrics(reference * 2, reference, 60, 20, 4.19, 30)
        self.assertAlmostEqual(metrics['annual_energy_bias_pct'], 100)
        self.assertAlmostEqual(metrics['peak_bias_pct'], 100)
        self.assertEqual(metrics['hours_above_threshold'], 13)
        self.assertEqual(metrics['longest_peak_event_hours'], 13)
        self.assertAlmostEqual(metrics['coincidence_factor'], 46 / 60)
        self.assertAlmostEqual(metrics['peak_mass_flow_kg_s'], 46 / (4.19 * 30))
        self.assertEqual(longest_event(np.array([False, False])), 0)
        zeros = calculate_profile_metrics(np.zeros(24), np.zeros(24), 0, 0, 4.19, 30)
        self.assertIsNone(zeros['peak_bias_pct'])
        self.assertIsNone(zeros['cvrmse_pct'])

    def test_feeder_detects_undersizing_against_independent_reference(self):
        from api.utils.pipe_sizing import DesignCriteria
        result = screen_equivalent_feeder(np.ones(24), np.full(24, 10000), DesignCriteria, 30)
        self.assertEqual(result['reference_violation_hours'], 24)
        impossible = screen_equivalent_feeder(np.full(24, 1e9), np.ones(24), DesignCriteria, 30)
        self.assertIsNone(impossible['dn'])

    def test_weather_rejects_gaps_and_duplicate_timestamps(self):
        times, temperatures, _ = example_weather()
        payload = {'hourly': {'time': [t.replace(tzinfo=None).isoformat() for t in times], 'temperature_2m': temperatures.tolist()}}
        validated, _ = validate_weather_payload(payload, 2024, LOCATIONS[0])
        self.assertEqual(validated[0].utcoffset(), timedelta(hours=-5))
        payload['hourly']['time'][1] = payload['hourly']['time'][0]
        with self.assertRaises(ValueError):
            validate_weather_payload(payload, 2024, LOCATIONS[0])


class HeatingStudyApiTests(TestCase):
    def setUp(self):
        cache.clear()
        self.client = APIClient()
        self.user = get_user_model().objects.create_user(username='researcher', password='test-only-password')
        self.project = Project.objects.create(name='Canadian study')
        ProjectMembership.objects.create(project=self.project, user=self.user, role='editor')
        self.building = Building.objects.create(project=self.project, name='Existing Canadian building',
            building_type='residential', floor_area=200, heating_demand=24000)
        self.client.force_authenticate(self.user)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.override = override_settings(BASE_DIR=self.temp.name)
        self.override.enable()
        self.addCleanup(self.override.disable)
        self.weather = patch('api.intelligence.heating_studies.fetch_canadian_weather', return_value=example_weather())
        self.weather.start()
        self.addCleanup(self.weather.stop)

    def compare(self, **extra):
        return self.client.post('/api/heating-profile-comparison/', {'project_id': self.project.id,
                               'energy_basis': 'imported_annual', **extra}, format='json')

    def test_comparison_is_read_only_and_models_are_distinct(self):
        response = self.compare()
        self.assertEqual(response.status_code, 200, response.data)
        self.assertEqual(LoadProfile.objects.count(), 0)
        results = response.data['results']
        self.assertEqual(len(results), len(MODEL_IDS))
        self.assertNotEqual(results[0]['hourly_kw'], results[1]['hourly_kw'])
        self.assertNotEqual(results[0]['hourly_kw'], results[2]['hourly_kw'])

    def test_cross_project_network_and_building_selection_rejected(self):
        other = Project.objects.create(name='Other')
        network = Network.objects.create(project=other, name='Other network')
        building = Building.objects.create(project=other, floor_area=100)
        self.assertEqual(self.compare(network_id=network.id).status_code, 404)
        self.assertEqual(self.compare(building_ids=[building.id]).status_code, 400)
        self.assertEqual(self.client.post('/api/heating-profile-comparison/', {'project_id': other.id}, format='json').status_code, 404)

    def test_incomplete_reference_rejected(self):
        self.assertEqual(self.compare(reference='existing').status_code, 400)

    def test_apply_replaces_only_exact_scope_and_archives(self):
        start = example_weather()[0][0]
        old = LoadProfile.objects.create(building=self.building, profile_type='heating', unit='kW',
            value=7, start_date=start, end_date=start + timedelta(hours=1), source_model='old')
        network = Network.objects.create(project=self.project, name='Preserve network')
        preserved = LoadProfile.objects.create(building=self.building, network=network, profile_type='heating',
            unit='kW', value=9, start_date=start, end_date=start + timedelta(hours=1))
        run = self.compare().data
        response = self.client.post('/api/heating-profile-apply/', {'run_id': run['run_id'], 'model': 'necb_slp_ca_v1'}, format='json')
        self.assertEqual(response.status_code, 200, response.data)
        self.assertFalse(LoadProfile.objects.filter(pk=old.pk).exists())
        self.assertTrue(LoadProfile.objects.filter(pk=preserved.pk).exists())
        self.assertEqual(LoadProfile.objects.filter(network__isnull=True).count(), 8784)
        self.assertEqual(set(LoadProfile.objects.filter(network__isnull=True).values_list('source_model', flat=True)), {'necb_slp_ca_v1'})
        import gzip, json
        from pathlib import Path
        archive = Path(self.temp.name) / 'data/load_profile_runs' / response.data['archive_id'] / 'previous_profiles.json.gz'
        with gzip.open(archive, 'rt') as stream:
            self.assertEqual(json.load(stream)[0]['value'], 7)

    def test_viewer_can_compare_but_cannot_apply(self):
        ProjectMembership.objects.filter(user=self.user).update(role='viewer')
        response = self.compare()
        self.assertEqual(response.status_code, 200)
        apply = self.client.post('/api/heating-profile-apply/', {'run_id': response.data['run_id'], 'model': 'hdd_daily_ca_v1'}, format='json')
        self.assertEqual(apply.status_code, 403)

    def test_unknown_model_returns_validation_error_before_weather(self):
        self.assertEqual(self.compare(models=['unknown']).status_code, 400)
        response = self.client.post(f'/api/projects/{self.project.id}/generate_profiles/', {'model_type': 'unknown'}, format='json')
        self.assertEqual(response.status_code, 400)

    def test_weather_failure_leaves_existing_profiles_unchanged(self):
        import requests
        with patch('api.intelligence.heating_studies.fetch_canadian_weather', side_effect=requests.Timeout):
            self.assertEqual(self.compare().status_code, 503)
        self.assertEqual(LoadProfile.objects.count(), 0)

    def test_building_picker_enforces_project_scope_and_search(self):
        other = Project.objects.create(name='Private project')
        Building.objects.create(project=other, name='Other building', floor_area=300)
        response = self.client.get('/api/heating-study-buildings/', {'project_id': self.project.id, 'search': 'Canadian'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual([b['id'] for b in response.data['results']], [self.building.id])
        self.assertEqual(self.client.get('/api/heating-study-buildings/', {'project_id': other.id}).status_code, 404)

    def test_apply_rejects_malformed_run_id(self):
        response = self.client.post('/api/heating-profile-apply/', {'run_id': 'invalid', 'model': 'hdd_daily_ca_v1'}, format='json')
        self.assertEqual(response.status_code, 400)

    def test_background_generation_dispatches_selected_model(self):
        from api.tasks import generate_profiles_for_project
        with patch('api.intelligence.heating_weather.fetch_canadian_weather', return_value=example_weather()):
            generate_profiles_for_project(self.project.id, 'necb_slp_ca_v1', 2024, model_options={'energy_basis': 'imported_annual'})
        rows = LoadProfile.objects.filter(building=self.building)
        self.assertEqual(rows.count(), 8784)
        self.assertEqual(set(rows.values_list('source_model', flat=True)), {'necb_slp_ca_v1'})
        self.assertAlmostEqual(sum(rows.values_list('value', flat=True)), 24000, places=4)

class CanadianEnergyAndThermoTests(SimpleTestCase):
    def test_final_energy_is_split_before_useful_heat_conversion(self):
        from api.intelligence.heating_energy import calculate_annual_heat
        options = valid_options(energy_basis='canadian_scenario', residential_total_intensity_kwh_m2=200,
                                residential_space_fraction=.6, residential_water_fraction=.2,
                                space_heat_conversion=.8, water_heat_conversion=2.)
        energy = calculate_annual_heat(100, 'residential', options)
        self.assertEqual(energy['total_final_energy_kwh'], 20000)
        self.assertEqual(energy['annual_space_kwh'], 9600)
        self.assertEqual(energy['annual_dhw_kwh'], 8000)
        self.assertEqual(energy['annual_kwh'], 17600)
        high = calculate_annual_heat(100, 'residential', {**options, 'energy_scenario': 'high'})
        self.assertEqual(high['annual_kwh'], 22000)
        with self.assertRaises(ValueError):
            calculate_annual_heat(100, 'industrial', options)

    def test_shares_and_nonfinite_overrides_are_rejected(self):
        for fields in ({'commercial_space_fraction': .95, 'commercial_water_fraction': .2},
                       {'residential_total_intensity_kwh_m2': float('nan')}):
            serializer = HeatingStudyRequest(data={'project_id': 1, **fields})
            self.assertFalse(serializer.is_valid())

    def test_circuit_conserves_energy_and_off_heater_has_no_heat_flow(self):
        from api.intelligence.heating_thermo import thermal_circuit_rates, THERMO_PARAMETERS
        p = THERMO_PARAMETERS['detached']
        for control in (0., .4, 1.):
            exterior, interior, outdoor = 10., 20., -5.
            de, di, _, heat = thermal_circuit_rates((exterior, interior, control), outdoor, 22., p)
            self.assertAlmostEqual(p['c_we'] * de + p['c_wi'] * di,
                                   (outdoor - exterior) / p['r_ext'] + heat, places=10)
            if control == 0:
                self.assertEqual(heat, 0.)
        de, di, _, heat = thermal_circuit_rates((20., 20., 0.), 20., 20., p)
        self.assertAlmostEqual(de, 0.)
        self.assertAlmostEqual(di, 0.)

    def test_thermo_step_refinement_and_archetype_difference(self):
        from api.intelligence.heating_thermo import generate_thermo_shape
        temperatures = np.repeat([-15., 5., -5., 10.], 24)
        setpoints = np.tile(np.repeat([18., 22.], 12), 4)
        coarse = generate_thermo_shape(temperatures, setpoints, 'detached', 24)
        fine = generate_thermo_shape(temperatures, setpoints, 'detached', 48)
        attached = generate_thermo_shape(temperatures, setpoints, 'attached', 24)
        np.testing.assert_allclose(coarse, fine, rtol=.002, atol=.002)
        self.assertGreater(np.max(np.abs(coarse - attached)), .1)

    def test_four_models_use_explicit_space_and_water_targets(self):
        times, temperatures, _ = example_weather(2023)
        inputs = dict(building_type='residential', floor_area=200, volume_m3=600,
                      annual_kwh=10000, annual_space_kwh=7000, annual_dhw_kwh=3000)
        for model in MODEL_IDS:
            profile = generate_building_profile(model, temperatures, times, inputs, valid_options(dhw_fraction=0))
            self.assertAlmostEqual(profile.sum(), 10000, places=6)
