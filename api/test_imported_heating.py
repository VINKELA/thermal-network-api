from django.contrib.auth import get_user_model
from django.test import TestCase
from rest_framework.test import APIClient
from api.models import Project,ProjectMembership,Building,ImportedHeatingShape,ImportedBuildingHeating


class ImportedHeatingTests(TestCase):
    def setUp(self):
        self.user=get_user_model().objects.create_user(username='owner')
        self.other=get_user_model().objects.create_user(username='other')
        self.project=Project.objects.create(name='Imported')
        ProjectMembership.objects.create(project=self.project,user=self.user,role='owner')
        self.building=Building.objects.create(project=self.project,name='Burlington 33568')
        ImportedBuildingHeating.objects.create(building=self.building,source_building_id='33568',
            annual_energy_kwh={'2023':100},annual_peak_kw={'2023':75})
        ImportedHeatingShape.objects.create(project=self.project,year=2023,
            timestamps=['2023-01-01 00:00:00','2023-01-01 01:00:00'],hourly_weights=[.25,.75])
        self.client=APIClient()
        self.client.force_authenticate(self.user)

    def query(self,**params):
        return self.client.get('/api/imported-heating/',{'project_id':self.project.id,'year':2023,**params})

    def test_exact_expansion_and_project_aggregate(self):
        r=self.query(action='individual',building=self.building.id)
        self.assertEqual(r.status_code,200)
        self.assertEqual([v['value'] for v in r.data['results']],[25,75])
        self.assertEqual(r.data['results'][0]['end_date'],'2023-01-01T01:00:00')
        r=self.query(action='aggregate')
        self.assertEqual([v['total_demand'] for v in r.data['results']],[25,75])

    def test_directory_without_network_and_zero_energy(self):
        self.assertEqual(self.query(action='directory').data['count'],1)
        ImportedBuildingHeating.objects.filter(building=self.building).update(annual_energy_kwh={'2023':0})
        self.assertEqual([v['value'] for v in self.query(action='individual',building=self.building.id).data['results']],[0,0])

    def test_unauthorized_user_cannot_read_import(self):
        self.client.force_authenticate(self.other)
        self.assertEqual(self.query(action='directory').status_code,404)
        self.assertEqual(self.query(action='individual',building=self.building.id).status_code,404)

    def test_invalid_year_rejected(self):
        self.assertEqual(self.query(action='aggregate',year=2024).status_code,400)
