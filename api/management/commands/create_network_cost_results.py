from django.core.management.base import BaseCommand, CommandError

from api.models import NetworkCostResult, PipeSizing
from api.utils.network_costing import build_network_cost_summary


class Command(BaseCommand):
    help = 'Create or update stored network-cost results for completed pipe-sizing runs.'

    def add_arguments(self, parser):
        parser.add_argument('--sizing-ids', nargs='+', type=int)
        parser.add_argument('--project', type=int)

    def handle(self, *args, **options):
        queryset = PipeSizing.objects.select_related('network').exclude(network=None).order_by('id')
        if options['sizing_ids']:
            queryset = queryset.filter(id__in=options['sizing_ids'])
        if options['project']:
            queryset = queryset.filter(network__project_id=options['project'])
        sizings = list(queryset)
        if not sizings:
            raise CommandError('No completed pipe-sizing runs matched the requested scope.')
        created = updated = failed = 0
        for sizing in sizings:
            try:
                summary = build_network_cost_summary(sizing.network_id)
                _, was_created = NetworkCostResult.objects.update_or_create(
                    network=sizing.network,
                    defaults={'pipe_sizing': sizing, 'summary': summary},
                )
                created += int(was_created)
                updated += int(not was_created)
            except Exception as exc:
                failed += 1
                self.stdout.write(self.style.ERROR(f'FAILED|sizing={sizing.id}|network={sizing.network_id}|{exc}'))
        self.stdout.write(f'SUMMARY|requested={len(sizings)}|created={created}|updated={updated}|failed={failed}')
