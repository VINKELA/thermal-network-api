from types import SimpleNamespace

from django.test import SimpleTestCase

from api.utils.pipe_sizing import DesignCriteria, HydraulicFormulas, NetworkSizer


class HydraulicFormulaTests(SimpleTestCase):
    def test_mass_flow_uses_heat_balance(self):
        self.assertAlmostEqual(HydraulicFormulas.calculate_mass_flow(419, 20), 5.0)

    def test_mass_flow_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            HydraulicFormulas.calculate_mass_flow(-1, 20)
        with self.assertRaises(ValueError):
            HydraulicFormulas.calculate_mass_flow(100, 0)

    def test_laminar_friction_factor_and_zero_flow(self):
        self.assertAlmostEqual(HydraulicFormulas.calculate_friction_factor(1000, 0.05), 0.064)
        self.assertEqual(HydraulicFormulas.calculate_head_loss_pa(0, 0.05, 100), 0)

    def test_rougher_pipe_has_more_pressure_loss(self):
        smooth = HydraulicFormulas.calculate_head_loss_pa(1.5, 0.05, 100, roughness=1e-6)
        rough = HydraulicFormulas.calculate_head_loss_pa(1.5, 0.05, 100, roughness=1e-3)
        self.assertGreater(rough, smooth)


class CatalogueSelectionTests(SimpleTestCase):
    def make_sizer(self):
        sizer = NetworkSizer.__new__(NetworkSizer)
        sizer.criteria = SimpleNamespace(
            RHO=DesignCriteria.RHO,
            CP=DesignCriteria.CP,
            MU=DesignCriteria.MU,
            MAX_VELOCITY=2.5,
            TARGET_PRESSURE_DROP=100,
            STD_PIPES=DesignCriteria.STD_PIPES,
            SUPPLY_TEMPERATURE_C=80,
            RETURN_TEMPERATURE_C=50,
        )
        sizer.catalog = sizer._validated_catalog()
        return sizer

    def test_velocity_selection_uses_smallest_feasible_catalogue_pipe(self):
        sizer = self.make_sizer()
        selected = sizer._select_pipe_by_velocity(1.0, 2.5)
        previous = [p for p in sizer.catalog if p['di'] < selected['di']]
        self.assertLessEqual(selected['velocity'], 2.5)
        self.assertTrue(all((1 / sizer.criteria.RHO) / (3.141592653589793 * p['di'] ** 2 / 4) > 2.5 for p in previous))

    def test_infeasible_catalogue_raises(self):
        sizer = self.make_sizer()
        with self.assertRaises(ValueError):
            sizer._select_pipe_by_velocity(10000, 2.5)

    def test_design_delta_t_comes_from_network_temperatures(self):
        self.assertEqual(self.make_sizer()._design_delta_t(), 30)

    def test_physical_segment_selector_uses_pressure_gradient_method(self):
        sizer = self.make_sizer()
        sizer.sizing_config = SimpleNamespace(method='pressure_gradient')
        selected = sizer._select_pipe_for_segment(1.0, 100)
        self.assertLessEqual(
            HydraulicFormulas.calculate_head_loss_pa(
                selected['velocity'], selected['di'], 1, sizer.criteria, selected['roughness']
            ),
            sizer.criteria.TARGET_PRESSURE_DROP,
        )
