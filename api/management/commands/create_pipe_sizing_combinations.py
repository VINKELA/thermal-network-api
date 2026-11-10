from django.core.management.base import BaseCommand, CommandError

from api.models import EdgeTranslation, Network, PipeSizing
from api.views import run_pipe_sizing_version


METHODS = ('velocity_limited', 'pressure_gradient', 'lifecycle_cost')


class Command(BaseCommand):
    help = "Create each validated pipe-sizing method for each selected route tracing."

    def add_arguments(self, parser):
        parser.add_argument('--project', type=int, help='Limit route tracings to a project ID.')
        parser.add_argument('--route-ids', nargs='+', type=int)
        parser.add_argument('--methods', nargs='+', choices=METHODS, default=list(METHODS))
        parser.add_argument('--year', type=int, default=2025)
        parser.add_argument('--recreate', action='store_true')

    def handle(self, *args, **options):
        queryset = EdgeTranslation.objects.select_related('network', 'network__project').order_by('id')
        if options['project']:
            queryset = queryset.filter(network__project_id=options['project'])
        if options['route_ids']:
            queryset = queryset.filter(id__in=options['route_ids'])
        routes = list(queryset)
        if not routes:
            raise CommandError('No route tracings matched the requested scope.')

        existing = self._existing_pairs()
        created = []
        skipped = []
        failed = []
        for route in routes:
            for method in options['methods']:
                pair = (route.id, method)
                if not options['recreate'] and pair in existing:
                    skipped.append((route.id, method, 'sizing already exists'))
                    continue
                result, response_status = run_pipe_sizing_version(
                    route.id,
                    {
                        'method': method,
                        'year': options['year'],
                        'name_prefix': method.replace('_', ' ').title(),
                    },
                    source_type='route',
                )
                if 200 <= response_status < 300 and result.get('status') == 'success':
                    created.append((route.id, method, result['network_id'], result['pipe_sizing_id']))
                    existing.add(pair)
                else:
                    reason = '; '.join(result.get('errors') or []) or result.get('message') or str(result)
                    failed.append((route.id, method, reason))

        for route_id, method, network_id, sizing_id in created:
            self.stdout.write(self.style.SUCCESS(
                f'CREATED|route={route_id}|method={method}|network={network_id}|sizing={sizing_id}'
            ))
        for route_id, method, reason in skipped:
            self.stdout.write(self.style.WARNING(f'SKIPPED|route={route_id}|method={method}|{reason}'))
        for route_id, method, reason in failed:
            self.stdout.write(self.style.ERROR(f'FAILED|route={route_id}|method={method}|{reason}'))
        self.stdout.write(
            f'SUMMARY|routes={len(routes)}|methods={len(options["methods"])}|'
            f'created={len(created)}|skipped={len(skipped)}|failed={len(failed)}'
        )

    @staticmethod
    def _existing_pairs():
        pairs = set()
        queryset = PipeSizing.objects.select_related('network').exclude(network=None)
        for sizing in queryset:
            parameters = sizing.network.derivation_parameters or {}
            route_id = parameters.get('source_edge_translation_id')
            if route_id:
                pairs.add((int(route_id), sizing.method))
        return pairs
