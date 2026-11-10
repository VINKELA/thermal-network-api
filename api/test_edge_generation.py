import numpy as np
from django.test import SimpleTestCase, TestCase

from api.models import EdgeAlgorithm, Network, Node, Project
from api.utils.network_versioning import (
    _beta_skeleton_pairs,
    _edge_pairs_for_method,
    _project_coordinates_to_local_meters,
    generate_edges_for_network,
)


class EdgePairGenerationTests(SimpleTestCase):
    def setUp(self):
        self.nodes = [object(), object(), object(), object()]
        self.square = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])

    def test_complete_graph_contains_every_pair(self):
        pairs = _edge_pairs_for_method("complete", self.nodes, self.square, {})
        self.assertEqual(len(pairs), 6)

    def test_mutual_knn_only_keeps_reciprocal_neighbors(self):
        line = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
        pairs = _edge_pairs_for_method("mutual_knn", self.nodes[:3], line, {"k": 1})
        self.assertEqual(pairs, [(0, 1)])

    def test_beta_one_matches_gabriel_and_beta_two_matches_rng(self):
        gabriel = _edge_pairs_for_method("gabriel", self.nodes, self.square, {})
        rng = _edge_pairs_for_method("rng", self.nodes, self.square, {})
        self.assertEqual(_beta_skeleton_pairs(self.square, 1.0), gabriel)
        self.assertEqual(_beta_skeleton_pairs(self.square, 2.0), rng)

    def test_projection_returns_metric_scale(self):
        projected = _project_coordinates_to_local_meters(np.array([[43.0, -79.0], [43.0, -78.999]]))
        self.assertAlmostEqual(np.linalg.norm(projected[1] - projected[0]), 81.3, delta=0.5)


class EdgeGenerationServiceTests(TestCase):
    def setUp(self):
        project = Project.objects.create(name="Edge generation research")
        self.network = Network.objects.create(project=project, name="Candidate network")
        for index, (latitude, longitude) in enumerate(((43.0, -79.0), (43.001, -79.0), (43.0, -78.999))):
            Node.objects.create(
                network=self.network,
                name=f"N{index}",
                latitude=latitude,
                longitude=longitude,
                node_type="building",
            )

    def test_generation_rejects_topology_selection_methods(self):
        with self.assertRaisesMessage(ValueError, "not an edge-generation method"):
            generate_edges_for_network(self.network, "mst")

    def test_generation_records_connectivity_diagnostics(self):
        algorithm = generate_edges_for_network(self.network, "complete")
        self.assertEqual(algorithm.edges.count(), 3)
        self.assertTrue(algorithm.parameters["diagnostics"]["is_connected"])
        self.assertEqual(algorithm.parameters["coordinate_space"], "local_equirectangular_m")
