from django.core.management.base import BaseCommand, CommandError

from api.models import EdgeConnectionAlgorithm, EdgeTranslation
from api.utils.network_versioning import create_route_tracing_version


class Command(BaseCommand):
    help = "Create one route-traced network for each selected connection run."

    def add_arguments(self, parser):
        parser.add_argument("--project", type=int, help="Limit connection runs to a project ID.")
        parser.add_argument("--connection-ids", nargs="+", type=int)
        parser.add_argument("--strategy", default="driving", choices=sorted(EdgeTranslation.routing_values()))
        parser.add_argument("--fallback-policy", default="error", choices=["error", "direct"])
        parser.add_argument("--no-google", action="store_true")
        parser.add_argument("--recreate", action="store_true")

    def handle(self, *args, **options):
        queryset = EdgeConnectionAlgorithm.objects.select_related("network", "network__project").order_by("id")
        if options["project"]:
            queryset = queryset.filter(network__project_id=options["project"])
        if options["connection_ids"]:
            queryset = queryset.filter(id__in=options["connection_ids"])
        connections = list(queryset)
        if not connections:
            raise CommandError("No connection runs matched the requested scope.")

        created = []
        skipped = []
        failed = []
        for connection in connections:
            if not options["recreate"] and self._route_exists(connection.id, options["strategy"]):
                skipped.append((connection.id, "route already exists"))
                continue
            parameters = {
                "use_google": not options["no_google"],
                "fallback_policy": options["fallback_policy"],
            }
            name = f"{connection.name} - {options['strategy'].replace('_', ' ').title()} Route"
            try:
                network, translation, routes = create_route_tracing_version(
                    connection,
                    options["strategy"],
                    name=name,
                    parameters=parameters,
                )
                created.append((connection.id, network.id, translation.id, len(routes), translation.total_edge_length))
            except Exception as exc:
                failed.append((connection.id, str(exc)))

        for connection_id, network_id, translation_id, route_count, length in created:
            self.stdout.write(self.style.SUCCESS(
                f"CREATED|connection={connection_id}|network={network_id}|translation={translation_id}|routes={route_count}|length_m={length:.1f}"
            ))
        for connection_id, reason in skipped:
            self.stdout.write(self.style.WARNING(f"SKIPPED|connection={connection_id}|{reason}"))
        for connection_id, reason in failed:
            self.stdout.write(self.style.ERROR(f"FAILED|connection={connection_id}|{reason}"))
        self.stdout.write(f"SUMMARY|created={len(created)}|skipped={len(skipped)}|failed={len(failed)}")

    @staticmethod
    def _route_exists(connection_id, strategy):
        for translation in EdgeTranslation.objects.filter(translation=strategy).only("parameters"):
            if (translation.parameters or {}).get("source_edge_connection_algorithm_id") == connection_id:
                return True
        return False
