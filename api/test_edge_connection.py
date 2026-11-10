from django.test import TestCase

from api.models import Edge, EdgeAlgorithm, Network, Node, Project
from api.utils.network_versioning import select_connection_edges


class EdgeConnectionSelectionTests(TestCase):
    def setUp(self):
        project = Project.objects.create(name="Connection research")
        self.network = Network.objects.create(project=project, name="Candidate network")
        self.nodes = [
            Node.objects.create(
                network=self.network,
                name=name,
                latitude=43.0 + index * 0.001,
                longitude=-79.0,
                node_type=node_type,
            )
            for index, (name, node_type) in enumerate((
                ("Plant", "producer"),
                ("A", "building"),
                ("J", "junction"),
                ("B", "building"),
            ))
        ]
        self.algorithm = EdgeAlgorithm.objects.create(
            network=self.network,
            name="Test candidates",
            algorithm="complete",
        )
        self.edges = {}
        self._edge(0, 1, 1)
        self._edge(1, 2, 1)
        self._edge(2, 3, 1)
        self._edge(0, 3, 4)
        self._edge(0, 2, 2.5)
        self._edge(1, 3, 2.5)

    def _edge(self, start, end, length):
        edge = Edge.objects.create(
            network=self.network,
            edge_algorithm=self.algorithm,
            start_node=self.nodes[start],
            end_node=self.nodes[end],
            length=length,
            name=f"{start}-{end}",
        )
        self.edges[tuple(sorted((start, end)))] = edge

    def _pairs(self, edges):
        node_index = {node.id: index for index, node in enumerate(self.nodes)}
        return {
            tuple(sorted((node_index[edge.start_node_id], node_index[edge.end_node_id])))
            for edge in edges
        }

    def test_mst_is_minimum_spanning_tree(self):
        selected = select_connection_edges(self.algorithm, "mst")
        self.assertEqual(self._pairs(selected), {(0, 1), (1, 2), (2, 3)})

    def test_shortest_path_tree_is_rooted_at_source(self):
        selected = select_connection_edges(
            self.algorithm,
            "shortest_path_tree",
            {"source_node_id": self.nodes[0].id},
        )
        self.assertEqual(len(selected), len(self.nodes) - 1)
        self.assertIn((0, 1), self._pairs(selected))

    def test_steiner_tree_can_use_junction_and_skip_nonterminal(self):
        selected = select_connection_edges(
            self.algorithm,
            "steiner_tree",
            {"terminal_node_ids": [self.nodes[0].id, self.nodes[3].id]},
        )
        self.assertEqual(self._pairs(selected), {(0, 1), (1, 2), (2, 3)})

    def test_redundant_mst_has_no_bridge(self):
        selected = select_connection_edges(self.algorithm, "redundant_mst")
        selected_pairs = self._pairs(selected)
        self.assertGreater(len(selected_pairs), len(self.nodes) - 1)
        self.assertIn((0, 3), selected_pairs)

    def test_all_candidates_is_not_a_connection_method(self):
        with self.assertRaisesMessage(ValueError, "not an active edge-connection method"):
            select_connection_edges(self.algorithm, "all_candidates")

    def test_disconnected_candidates_are_rejected(self):
        Edge.objects.filter(edge_algorithm=self.algorithm).exclude(name__in=["0-1", "2-3"]).delete()
        with self.assertRaisesMessage(ValueError, "Candidate graph is disconnected"):
            select_connection_edges(self.algorithm, "mst")

    def test_legacy_custom_is_not_available_for_new_runs(self):
        with self.assertRaisesMessage(ValueError, "not an active edge-connection method"):
            select_connection_edges(self.algorithm, "custom")

    def test_self_loop_candidate_is_rejected(self):
        Edge.objects.create(
            network=self.network,
            edge_algorithm=self.algorithm,
            start_node=self.nodes[0],
            end_node=self.nodes[0],
            length=0,
            name="self-loop",
        )
        with self.assertRaisesMessage(ValueError, "is a self-loop"):
            select_connection_edges(self.algorithm, "mst")
