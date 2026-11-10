"""Scientific and authorization checks for branch decision sensitivity."""
from copy import deepcopy
import tempfile
from unittest.mock import patch
import uuid
import numpy as np
from django.test import SimpleTestCase, TestCase, override_settings
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from api.models import Project, ProjectMembership, LoadProfile, Edge
from api.test_heating_profiles import example_weather
from api.intelligence.heating_topology import demonstration_case, validate_radial_case
from api.intelligence.heating_sensitivity import (SensitivityRequest, run_pipe_sensitivity, aggregate_branch_profiles,
    calculate_design_benchmark, hydraulic_arrays, summarize_stability, build_experiment_cases)
from api.utils.pipe_sizing import DesignCriteria, HydraulicFormulas


def options(**kwargs):
    s=SensitivityRequest(data={'project_id':1,'years':[2023],**kwargs});s.is_valid(raise_exception=True);return s.validated_data


class BranchSensitivityScienceTests(SimpleTestCase):
    def test_coincident_aggregation_not_sum_of_individual_peaks(self):
        buildings=[dict(id=1,building_type='residential'),dict(id=2,building_type='commercial')]
        branches=[dict(building_ids=[1,2]),dict(building_ids=[1])]
        loads=aggregate_branch_profiles(np.array([[10.,0.],[0.,10.]]),buildings,branches,'mixed')
        np.testing.assert_equal(loads,[[10.,10.],[10.,0.]])
        subset=aggregate_branch_profiles(np.array([[10.,0.],[0.,10.]]),buildings,branches,'nonresidential')
        np.testing.assert_equal(subset,[[0.,10.],[0.,0.]])

    def test_topology_rejects_loops_disconnection_and_duplicate_buildings(self):
        case=demonstration_case()
        for mode in ('cycle','disconnect','duplicate'):
            nodes=deepcopy(case['nodes']);edges=deepcopy(case['branches'])
            if mode=='cycle': edges.append(dict(id=99,start=1,end=2,length_m=10.))
            elif mode=='disconnect': edges.pop()
            else: nodes[1]['building_ids']=[1]
            with self.assertRaises(ValueError): validate_radial_case(nodes,edges,case['buildings'])

    def test_hydraulics_match_existing_scalar_formulas(self):
        pipe=DesignCriteria.STD_PIPES[4];loads=np.array([0.,.1,1.,10.,100.,500.])
        v,g=hydraulic_arrays(loads,pipe,DesignCriteria,30.)
        for i,load in enumerate(loads):
            expected=HydraulicFormulas.calculate_head_loss_pa(float(v[i]),pipe['di'],1.,DesignCriteria)
            self.assertAlmostEqual(g[i],expected,places=9)

    def test_design_load_has_no_annual_normalization(self):
        o=options();b=[dict(id=1,building_type='residential',floor_area=200.,volume_m3=600.,annual_kwh=1.)]
        branch=[dict(building_ids=[1])]
        value=calculate_design_benchmark(b,branch,o['common'],o,'mixed')[0]
        self.assertAlmostEqual(value,.259*(21+18.4))
        b[0]['annual_kwh']=1000000
        self.assertAlmostEqual(calculate_design_benchmark(b,branch,o['common'],o,'mixed')[0],value)

    def test_ranges_validated_and_oat_only_changes_one_parameter(self):
        s=SensitivityRequest(data={'project_id':1,'ach_low':.8,'ach_high':.9});self.assertFalse(s.is_valid())
        s=SensitivityRequest(data={'project_id':1,'energy_low':float('nan')});self.assertFalse(s.is_valid())
        cases=build_experiment_cases(options())
        physical=[c for c in cases if not c['normalize_physical']]
        self.assertEqual(len(physical),7)
        self.assertTrue(all(len(c['physical'])<=1 for c in physical))

    def test_no_fit_is_not_stable(self):
        s=summarize_stability([dict(composition='mixed',experiment='shape',branches=[dict(peak_kw=100,dn=None)])],[dict(id=1,name='trunk')])
        self.assertFalse(s[0]['stable']);self.assertEqual(s[0]['no_fit_count'],1)


class BranchSensitivityRunTests(TestCase):
    def test_complete_run_energy_resolution_weather_and_shared_tests(self):
        o=options(years=[2023,2024],compositions=['mixed'],experiments=['shape'],resolutions=[1,24])
        def weather(year, location): return example_weather(year)
        r=run_pipe_sensitivity(o,weather_loader=weather)
        self.assertEqual(len(r['scenarios']),12)
        for year in (2023,2024):
            rows=[s for s in r['scenarios'] if s['year']==year]
            energies=[s['branches'][0]['annual_kwh'] for s in rows]
            np.testing.assert_allclose(energies,energies[0])
            for model in o['models']:
                h1=next(s for s in rows if s['model']==model and s['resolution_hours']==1)
                h24=next(s for s in rows if s['model']==model and s['resolution_hours']==24)
                self.assertLessEqual(h24['branches'][0]['peak_kw'],h1['branches'][0]['peak_kw']+1e-8)
        self.assertEqual(LoadProfile.objects.count(),0);self.assertEqual(Edge.objects.count(),0)
        self.assertIn(r['coldest_tested_year'],[2023,2024])
        # Every selection of the same branch/DN sees exactly the same common stress test.
        grouped={}
        for s in r['scenarios']:
            for b in s['branches']:
                if b['dn'] is not None:
                    key=(b['branch_id'],b['dn'])
                    if key in grouped:self.assertEqual(grouped[key],b['common_violation_hours'])
                    grouped[key]=b['common_violation_hours']

    def test_input_energy_and_raw_physical_are_distinct_experiments(self):
        o=options(compositions=['residential'],experiments=['input'],resolutions=[1])
        r=run_pipe_sensitivity(o,weather_loader=lambda year,location:example_weather(year))
        rows=r['scenarios'];low=next(s for s in rows if s['label']=='Low annual heat' and s['model']=='hdd_daily_ca_v1')
        high=next(s for s in rows if s['label']=='High annual heat' and s['model']=='hdd_daily_ca_v1')
        self.assertAlmostEqual(high['branches'][0]['annual_kwh']/low['branches'][0]['annual_kwh'],1.25/.75)
        raw=[s for s in rows if not s['normalized']]
        self.assertEqual(len(raw),7)
        self.assertGreater(len({round(s['branches'][0]['annual_kwh']) for s in raw}),1)


class SensitivityAccessTests(TestCase):
    def setUp(self):
        self.user=get_user_model().objects.create_user(username='studyuser',password='test-only')
        self.project=Project.objects.create(name='Study')
        ProjectMembership.objects.create(project=self.project,user=self.user,role='viewer')
        self.client=APIClient();self.client.force_authenticate(self.user)
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.override=override_settings(BASE_DIR=self.temp.name);self.override.enable();self.addCleanup(self.override.disable)

    def test_other_project_start_is_denied(self):
        other=Project.objects.create(name='Private')
        with patch('api.heating_sensitivity_views.prepare_sensitivity_case') as prepare:
            response=self.client.post('/api/pipe-sensitivity/',{'project_id':other.id},format='json')
            self.assertIn(response.status_code,[403,404]);prepare.assert_not_called()

    def test_result_owner_and_membership_are_checked(self):
        from api.heating_sensitivity_views import study_path,write_study_json
        run=str(uuid.uuid4())
        write_study_json(study_path(run)/'status.json',dict(user_id=self.user.id,project_id=self.project.id,status='complete',run_id=run))
        write_study_json(study_path(run)/'result.json',{'test':'result'})
        response=self.client.get(f'/api/pipe-sensitivity/{run}/');self.assertEqual(response.status_code,200)
        other=get_user_model().objects.create_user(username='otherstudy');self.client.force_authenticate(other)
        self.assertEqual(self.client.get(f'/api/pipe-sensitivity/{run}/').status_code,404)
        self.client.force_authenticate(self.user);ProjectMembership.objects.all().delete()
        self.assertIn(self.client.get(f'/api/pipe-sensitivity/{run}/').status_code,[403,404])
