from unittest.mock import patch

from django.test import SimpleTestCase, TestCase

from api.models import (
    Edge,
    EdgeConnectionAlgorithm,
    EdgeConnectionAlgorithmEdge,
    Network,
    Node,
    Project,
)
from api.utils.network_versioning import build_route_data, create_route_tracing_version


class RouteStrategyTests(SimpleTestCase):
    def test_direct_strategy_is_explicit(self):
        route = build_route_data((43.0, -79.0), (43.001, -79.0), "direct")
        self.assertEqual(route["source"], "direct")
        self.assertEqual(route["mode"], "direct")

    @patch("api.utils.network_versioning.get_google_maps_route", return_value=None)
    def test_provider_failure_is_strict_by_default(self, _mock):
        with self.assertRaisesMessage(ValueError, "No valid driving corridor route"):
            build_route_data((43.0, -79.0), (43.001, -79.0), "driving")

    @patch("api.utils.network_versioning.get_google_maps_route", return_value=None)
    def test_direct_fallback_is_labelled(self, _mock):
        route = build_route_data(
            (43.0, -79.0),
            (43.001, -79.0),
            "driving",
            fallback_policy="direct",
        )
        self.assertEqual(route["source"], "direct_fallback")
        self.assertEqual(route["mode"], "direct")

    @patch("api.utils.network_versioning.get_google_maps_route")
    def test_requested_mode_is_not_silently_changed(self, google_route):
        google_route.return_value = None
        with self.assertRaises(ValueError):
            build_route_data((43.0, -79.0), (43.001, -79.0), "walking")
        google_route.assert_called_once_with((43.0, -79.0), (43.001, -79.0), "walking")


class RouteTracingVersionTests(TestCase):
    def test_direct_route_version_records_provenance(self):
        project = Project.objects.create(name="Routing research")
        network = Network.objects.create(project=project, name="Connected network")
        start = Node.objects.create(network=network, name="A", latitude=43.0, longitude=-79.0, node_type="consumer")
        end = Node.objects.create(network=network, name="B", latitude=43.001, longitude=-79.0, node_type="consumer")
        edge = Edge.objects.create(network=network, start_node=start, end_node=end, length=111.2, name="A-B")
        connection = EdgeConnectionAlgorithm.objects.create(network=network, name="MST", algorithm="mst")
        EdgeConnectionAlgorithmEdge.objects.create(edge_connection_algorithm=connection, edge=edge)

        output, translation, routes = create_route_tracing_version(connection, "direct")

        self.assertEqual(len(routes), 1)
        self.assertEqual(translation.parameters["route_source_counts"], {"direct": 1})
        self.assertEqual(translation.parameters["route_count"], 1)
        self.assertEqual(output.derivation_stage, Network.DerivationStage.ROUTE_TRACING)
        self.assertEqual(routes[0].edge.other_information["route_source"], "direct")
