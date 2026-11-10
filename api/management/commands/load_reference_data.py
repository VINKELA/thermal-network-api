from pathlib import Path

import pandas as pd
from django.conf import settings
from django.core.management.base import BaseCommand
from django.db import transaction

from api.models import CarbonTaxRate, DEAReferenceData, OntarioNaturalGasRate


DEFAULT_CARBON_TRAJECTORY = {
    2026: 95.0,
    2027: 100.0,
    2028: 100.0,
    2029: 100.0,
    2030: 115.0,
    2031: 118.0,
    2032: 121.0,
    2033: 124.0,
    2034: 127.0,
    2035: 130.0,
}


class Command(BaseCommand):
    help = "Load DEA reference data and Canadian carbon tax schedules."

    def add_arguments(self, parser):
        parser.add_argument("--refresh", action="store_true", help="Clear and reload all reference data.")
        parser.add_argument("--quiet", action="store_true", help="Suppress summary output.")
        parser.add_argument("--gas-kwh-per-m3", type=float, default=10.55, help="Natural gas higher heating value used to convert m3 to kWh.")
        parser.add_argument("--boiler-efficiency", type=float, default=0.90, help="Boiler efficiency used to convert gas commodity price to useful heat price.")

    @transaction.atomic
    def handle(self, *args, **options):
        data_dir = Path(settings.BASE_DIR) / "data"
        refresh = options["refresh"]
        quiet = options["quiet"]

        if refresh:
            DEAReferenceData.objects.all().delete()
            CarbonTaxRate.objects.all().delete()
            OntarioNaturalGasRate.objects.all().delete()

        dea_created = 0
        if not DEAReferenceData.objects.exists():
            dea_created = self._load_dea_data(data_dir)

        carbon_created = 0
        if not CarbonTaxRate.objects.exists():
            carbon_created = self._load_carbon_schedule(data_dir)

        gas_created = 0
        if not OntarioNaturalGasRate.objects.exists():
            gas_created = self._load_ontario_gas_rates(
                data_dir,
                options["gas_kwh_per_m3"],
                options["boiler_efficiency"],
            )

        if not quiet:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Reference data ready. DEA rows loaded: {dea_created}; carbon years loaded: {carbon_created}; Ontario gas rows loaded: {gas_created}."
                )
            )

    def _load_dea_data(self, data_dir):
        rows = []
        for csv_name in ["deagenerationflat.csv", "deatransportflat.csv"]:
            path = data_dir / csv_name
            if not path.exists():
                self.stdout.write(self.style.WARNING(f"DEA file not found: {path}"))
                continue

            frame = pd.read_csv(path, encoding="latin1")
            frame.columns = [str(col).strip().lstrip("\ufeff") for col in frame.columns]
            frame["year"] = pd.to_numeric(frame.get("year"), errors="coerce")
            frame["val"] = pd.to_numeric(frame.get("val"), errors="coerce")
            frame = frame.dropna(subset=["Technology", "par", "year", "val"])

            for item in frame.itertuples(index=False):
                rows.append(
                    DEAReferenceData(
                        technology=str(getattr(item, "Technology", "")).strip(),
                        category=self._clean(getattr(item, "cat", "")),
                        parameter=str(getattr(item, "par", "")).strip(),
                        unit=self._clean(getattr(item, "unit", "")),
                        estimate_type=self._clean(getattr(item, "est", "ctrl")) or "ctrl",
                        target_year=int(getattr(item, "year")),
                        value=float(getattr(item, "val")),
                    )
                )

        DEAReferenceData.objects.bulk_create(rows, batch_size=2000)
        return len(rows)

    def _load_carbon_schedule(self, data_dir):
        path = data_dir / "carbon_tax_schedule.csv"
        if path.exists():
            frame = pd.read_csv(path)
            frame["year"] = pd.to_numeric(frame.get("year"), errors="coerce")
            frame["price_cad_per_tonne"] = pd.to_numeric(frame.get("price_cad_per_tonne"), errors="coerce")
            frame = frame.dropna(subset=["year", "price_cad_per_tonne"])
            rows = [
                CarbonTaxRate(
                    year=int(item.year),
                    price_cad_per_tonne=float(item.price_cad_per_tonne),
                )
                for item in frame.itertuples(index=False)
            ]
        else:
            rows = self._fallback_carbon_rows()
            self.stdout.write(
                self.style.WARNING(
                    f"Carbon schedule file not found at {path}; loaded default 2026 Canadian industrial trajectory."
                )
            )

        CarbonTaxRate.objects.bulk_create(rows, batch_size=100, ignore_conflicts=True)
        return len(rows)

    def _load_ontario_gas_rates(self, data_dir, kwh_per_m3, boiler_efficiency):
        path = data_dir / "ontario_natural_gas_rates.csv"
        if not path.exists():
            self.stdout.write(self.style.WARNING(f"Ontario gas-rate file not found: {path}"))
            return 0

        frame = pd.read_csv(path)
        frame.columns = [str(col).strip().lstrip("\ufeff") for col in frame.columns]
        frame["effective_date"] = pd.to_datetime(frame.get("effective_date"), errors="coerce")
        for column in [
            "commodity_price_cents_per_m3",
            "gas_cost_adjustment_cents_per_m3",
            "effective_price_cents_per_m3",
        ]:
            frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
        frame = frame.dropna(subset=["effective_date", "effective_price_cents_per_m3"])

        rows = []
        for item in frame.itertuples(index=False):
            effective_price = float(getattr(item, "effective_price_cents_per_m3"))
            cad_per_m3 = effective_price / 100.0
            cad_per_mwh = cad_per_m3 / (kwh_per_m3 / 1000.0)
            useful_heat_price = cad_per_mwh / max(boiler_efficiency, 0.01)
            effective_date = getattr(item, "effective_date").date()
            rows.append(OntarioNaturalGasRate(
                utility=self._clean(getattr(item, "utility", "")) or "Enbridge Gas Inc.",
                region=self._clean(getattr(item, "region", "")) or "Union South Rate Zone",
                effective_date=effective_date,
                year=effective_date.year,
                commodity_price_cents_per_m3=self._float_or_none(getattr(item, "commodity_price_cents_per_m3", None)),
                gas_cost_adjustment_cents_per_m3=self._float_or_none(getattr(item, "gas_cost_adjustment_cents_per_m3", None)),
                effective_price_cents_per_m3=effective_price,
                cad_per_m3=cad_per_m3,
                cad_per_mwh=cad_per_mwh,
                cad_per_mwh_useful_heat=useful_heat_price,
                kwh_per_m3=kwh_per_m3,
                assumed_boiler_efficiency=boiler_efficiency,
                source_name=self._clean(getattr(item, "source_name", "")) or "Ontario Energy Board",
                source_url=self._clean(getattr(item, "source_url", "")) or "https://www.oeb.ca/consumer-information-and-protection/natural-gas-rates",
            ))

        OntarioNaturalGasRate.objects.bulk_create(rows, batch_size=200, ignore_conflicts=True)
        return len(rows)

    def _fallback_carbon_rows(self):
        rows = []
        last_price = None
        for year in range(2026, 2046):
            if year in DEFAULT_CARBON_TRAJECTORY:
                price = DEFAULT_CARBON_TRAJECTORY[year]
            else:
                price = round((last_price or DEFAULT_CARBON_TRAJECTORY[2035]) * 1.015, 2)
            last_price = price
            rows.append(CarbonTaxRate(year=year, price_cad_per_tonne=price))
        return rows

    def _clean(self, value):
        if value is None or pd.isna(value):
            return None
        text = str(value).strip()
        return text or None

    def _float_or_none(self, value):
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
