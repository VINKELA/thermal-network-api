from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from api.models import EdgeAlgorithm, EdgeConnectionAlgorithm
from api.utils.network_versioning import create_edge_connection_version


class Command(BaseCommand):
    help = "Create every feasible edge-generation/connection-method combination."

    def add_arguments(self, parser):
        parser.add_argument("--project", type=int, help="Limit edge-generation runs to a project ID.")
        parser.add_argument("--edge-algorithm-ids", nargs="+", type=int, help="Specific edge-generation run IDs.")
        parser.add_argument("--methods", nargs="+", help="Connection methods to try.")
        parser.add_argument("--source-node-id", type=int, help="Root node for shortest_path_tree.")
        parser.add_argument("--dry-run", action="store_true", help="Validate combinations without writing networks.")
        parser.add_argument("--recreate", action="store_true", help="Create a new run even when the combination exists.")

    def handle(self, *args, **options):
        queryset = EdgeAlgorithm.objects.select_related("network", "network__project").order_by("id")
        if options["project"]:
            queryset = queryset.filter(network__project_id=options["project"])
        if options["edge_algorithm_ids"]:
            queryset = queryset.filter(id__in=options["edge_algorithm_ids"])
        edge_algorithms = list(queryset)
        if not edge_algorithms:
            raise CommandError("No edge-generation runs matched the requested scope.")

        active_methods = EdgeConnectionAlgorithm.selection_values()
        methods = options["methods"] or [value for value, _ in EdgeConnectionAlgorithm.selection_choices()]
        unknown = set(methods).difference(active_methods)
        if unknown:
            raise CommandError(f"Unsupported connection methods: {sorted(unknown)}")

        generation_labels = dict(EdgeAlgorithm.EdgeAlgorithmType.choices)
        connection_labels = dict(EdgeConnectionAlgorithm.EdgeConnectionAlgorithmType.choices)
        created = []
        skipped = []
        failed = []

        for edge_algorithm in edge_algorithms:
            for method in methods:
                params = {}
                if method == EdgeConnectionAlgorithm.EdgeConnectionAlgorithmType.SHORTEST_PATH_TREE:
                    if options["source_node_id"] is None:
                        skipped.append((edge_algorithm.algorithm, method, "source_node_id was not provided"))
                        continue
                    params["source_node_id"] = options["source_node_id"]

                if not options["recreate"] and self._combination_exists(edge_algorithm.id, method):
                    skipped.append((edge_algorithm.algorithm, method, "combination already exists"))
                    continue

                generation_label = str(generation_labels.get(edge_algorithm.algorithm, edge_algorithm.algorithm))
                connection_label = str(connection_labels.get(method, method))
                name = f"{connection_label} - {generation_label}"
                try:
                    if options["dry_run"]:
                        from api.utils.network_versioning import select_connection_edges
                        selected = select_connection_edges(edge_algorithm, method, params)
                        created.append((edge_algorithm.algorithm, method, None, len(selected)))
                    else:
                        with transaction.atomic():
                            network, connection = create_edge_connection_version(
                                edge_algorithm,
                                method,
                                name=name,
                                parameters=params,
                            )
                        created.append((edge_algorithm.algorithm, method, network.id, connection.edges_connection.count()))
                except Exception as exc:
                    failed.append((edge_algorithm.algorithm, method, str(exc)))

        mode = "VALID" if options["dry_run"] else "CREATED"
        for generation, method, network_id, edge_count in created:
            target = "dry-run" if network_id is None else f"network={network_id}"
            self.stdout.write(self.style.SUCCESS(f"{mode}|{method}-{generation}|{target}|edges={edge_count}"))
        for generation, method, reason in skipped:
            self.stdout.write(self.style.WARNING(f"SKIPPED|{method}-{generation}|{reason}"))
        for generation, method, reason in failed:
            self.stdout.write(self.style.ERROR(f"INFEASIBLE|{method}-{generation}|{reason}"))

        self.stdout.write(
            f"SUMMARY|created_or_valid={len(created)}|skipped={len(skipped)}|infeasible={len(failed)}"
        )

    @staticmethod
    def _combination_exists(edge_algorithm_id, method):
        for connection in EdgeConnectionAlgorithm.objects.filter(algorithm=method).only("parameters"):
            if (connection.parameters or {}).get("source_edge_algorithm_id") == edge_algorithm_id:
                return True
        return False
