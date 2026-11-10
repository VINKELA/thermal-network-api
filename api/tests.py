from datetime import timedelta

from django.core.exceptions import ValidationError
from django.test import TestCase
from django.utils import timezone

from api.models import Building, Edge, LoadProfile, Network, Node, Project
from api.utils.network_costing import build_network_compare_metrics


class ProjectIsolationTests(TestCase):
    def setUp(self):
        self.project_a = Project.objects.create(name="Project A")
        self.project_b = Project.objects.create(name="Project B")
        self.network_a = Network.objects.create(
            project=self.project_a,
            name="Network A",
            network_type="district_heating",
        )
        self.network_b = Network.objects.create(
            project=self.project_b,
            name="Network B",
            network_type="district_heating",
        )

    def test_node_rejects_building_from_another_project(self):
        other_building = Building.objects.create(project=self.project_b, name="Other Project Building")

        with self.assertRaises(ValidationError):
            Node.objects.create(
                network=self.network_a,
                building=other_building,
                node_type="building",
            )

    def test_edge_rejects_node_from_another_network(self):
        node_a = Node.objects.create(network=self.network_a, name="A", node_type="building")
        node_b = Node.objects.create(network=self.network_b, name="B", node_type="building")

        with self.assertRaises(ValidationError):
            Edge.objects.create(
                network=self.network_a,
                start_node=node_a,
                end_node=node_b,
                length=10.0,
            )

    def test_load_profile_rejects_mixed_project_anchors(self):
        other_building = Building.objects.create(project=self.project_b, name="Other Project Building")
        now = timezone.now()

        with self.assertRaises(ValidationError):
            LoadProfile.objects.create(
                building=other_building,
                network=self.network_a,
                profile_type="heating",
                unit="kW",
                value=12.0,
                start_date=now,
                end_date=now + timedelta(hours=1),
            )

    def test_network_compare_metrics_rejects_mixed_projects(self):
        with self.assertRaises(ValueError):
            build_network_compare_metrics([self.network_a.id, self.network_b.id])
