from types import SimpleNamespace

from django.test import SimpleTestCase

from api.utils.network_costing import (
    _present_value_factor,
    _pumping_lifecycle_cost,
    _pumping_lifecycle_metrics,
    _thermal_loss_kw,
)


class NetworkCostFormulaTests(SimpleTestCase):
    def constants(self):
        return SimpleNamespace(
            pump_efficiency=0.75,
            hours_per_year=4000,
            electricity_cost=0.15,
            lifespan_years=30,
            discount_rate=0.04,
            thermal_loss_coefficient_w_m_k=0.25,
            supply_temperature_c=80,
            return_temperature_c=50,
            ambient_temperature_c=10,
        )

    def test_present_value_factor_discounts_future_costs(self):
        self.assertLess(_present_value_factor(0.04, 30), 30)
        self.assertEqual(_present_value_factor(0, 30), 30)

    def test_pumping_cost_uses_discounted_lifecycle(self):
        constants = self.constants()
        discounted = _pumping_lifecycle_cost(0.01, 100000, constants)
        undiscounted = (0.01 * 100000 / (0.75 * 1000)) * 4000 * 0.15 * 30
        self.assertGreater(discounted, 0)
        self.assertLess(discounted, undiscounted)

    def test_twin_pipe_heat_loss_uses_supply_and_return(self):
        self.assertAlmostEqual(_thermal_loss_kw(100, self.constants()), 2.75)

    def test_hourly_pumping_uses_flow_cube(self):
        constants = self.constants()
        result = _pumping_lifecycle_metrics(0.01, 100000, constants, [1.0, 0.5, 0.0])
        peak_power_kw = 0.01 * 100000 / (0.75 * 1000)
        self.assertAlmostEqual(result['annual_energy_kwh'], peak_power_kw * (1 + 0.5 ** 3))
