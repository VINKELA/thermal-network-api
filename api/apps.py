import os
import sys
import threading

from django.apps import AppConfig
from django.core.management import call_command
from django.db import OperationalError, ProgrammingError, connection


class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'
    _reference_seed_started = False

    def ready(self):
        if os.environ.get("THERMAL_NETWORK_SKIP_REFERENCE_SEED", "").lower() in {"1", "true", "yes"}:
            return

        management_commands = {"makemigrations", "migrate", "collectstatic", "test", "check", "shell", "dbshell", "load_reference_data"}
        if any(command in sys.argv for command in management_commands):
            return

        if "runserver" in sys.argv and os.environ.get("RUN_MAIN") not in {"true", "1"}:
            return

        if ApiConfig._reference_seed_started:
            return
        ApiConfig._reference_seed_started = True
        threading.Thread(target=self._seed_reference_data, daemon=True).start()

    def _seed_reference_data(self):
        try:
            existing_tables = set(connection.introspection.table_names())
            required_tables = {"api_deareferencedata", "api_carbontaxrate", "api_ontarionaturalgasrate"}
            if not required_tables.issubset(existing_tables):
                return

            from api.models import CarbonTaxRate, DEAReferenceData, OntarioNaturalGasRate

            if (
                not DEAReferenceData.objects.exists()
                or not CarbonTaxRate.objects.exists()
                or not OntarioNaturalGasRate.objects.exists()
            ):
                call_command("load_reference_data", quiet=True)
        except (OperationalError, ProgrammingError):
            return
