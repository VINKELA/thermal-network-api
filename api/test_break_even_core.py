"""Scientific regression checks, runnable with unittest without a database."""
import json
import unittest
import numpy as np
from api.utils.break_even import (DEFAULTS, asset_cost, crossing_summary, demo_profile,
    evaluate, find_crossings, installed_cost, prepare_strategies, run_study, storage_dispatch, strategy_zone_analysis, validate_options)


class BreakEvenFormulaTests(unittest.TestCase):
    def options(self, **changes):
        return validate_options({**DEFAULTS, **changes})

    def test_reference_anchor_and_diameter_examples(self):
        options = self.options()
        self.assertEqual(installed_cost(200, options), 4000)
        self.assertAlmostEqual(installed_cost(100, options), 3400)
        self.assertAlmostEqual(installed_cost(300, options), 4600)
        for alpha in (0, .2, 1):
            self.assertEqual(installed_cost(200, self.options(alpha=alpha, exponent_n=2)), 4000)

    def test_alpha_zero_and_exponent_zero_remove_diameter_cost_difference(self):
        for changes in ({'alpha': 0}, {'exponent_n': 0}):
            options = self.options(**changes)
            self.assertEqual(installed_cost(100, options), installed_cost(300, options))

    def test_bands_share_rate_and_upper_boundary_is_inclusive(self):
        options = self.options(diameter_band_width_mm=100)
        for diameter in (1, 50, 80, 99.99, 100):
            self.assertAlmostEqual(installed_cost(diameter, options), 3400)
        self.assertAlmostEqual(installed_cost(100.001, options), 4000)
        self.assertAlmostEqual(installed_cost(200, options), 4000)
        self.assertAlmostEqual(installed_cost(200.001, options), 4600)
        self.assertAlmostEqual(installed_cost(.07, self.options(diameter_band_width_mm=.01)),
                               installed_cost(.07, self.options()))

    def test_legacy_options_remain_continuous_and_invalid_widths_fail(self):
        self.assertAlmostEqual(installed_cost(50, self.options()), 3100)
        raw = dict(DEFAULTS)
        raw.pop('diameter_band_width_mm')
        self.assertEqual(validate_options(raw)['diameter_band_width_mm'], 0)
        for width in (-1, 3001, float('nan'), True):
            with self.assertRaises(ValueError):
                self.options(diameter_band_width_mm=width)

    def test_bands_change_asset_cost_but_keep_real_diameter_and_operation(self):
        continuous = self.options(small_diameter_mm=50, large_diameter_mm=80)
        banded = dict(continuous, diameter_band_width_mm=100)
        cases = prepare_strategies(np.array([100., 200.]), continuous)[0]
        rows = [evaluate(cases[k], 1000, banded) for k in ('small', 'large')]
        self.assertEqual(rows[0]['components']['pipe_capex'], rows[1]['components']['pipe_capex'])
        self.assertGreater(rows[0]['max_pressure_pa'], rows[1]['max_pressure_pa'])
        for key, row in zip(('small', 'large'), rows):
            original = evaluate(cases[key], 1000, continuous)
            for field in ('annual_pump_kwh', 'max_pressure_pa', 'max_velocity', 'diameter_mm'):
                self.assertEqual(row[field], original[field])

    def test_invalid_or_nonfinite_options_rejected(self):
        for changes in ({'alpha': 1.1}, {'alpha': float('nan')}, {'years': 1.5},
            {'cp': None}, {'electricity_price': True}, {'large_diameter_mm': 50},
            {'return_c': 90}, {'hp_cop': 19}, {'unknown': 1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.options(**changes)

    def test_asset_timing_and_end_horizon_replacement(self):
        options = self.options(years=30, discount_rate=0)
        cost = asset_cost(100, 15, .02, options)
        self.assertEqual(cost['replacement_pv'], 100)  # year 15, not year 30
        self.assertEqual(cost['maintenance_pv'], 60)
        self.assertEqual(cost['total'], 260)
        self.assertEqual(asset_cost(100, 20, 0, options, residual=True)['residual_pv'], 50)

    def test_discounting_reduces_future_costs(self):
        undiscounted = asset_cost(100, 15, .02, self.options(discount_rate=0))
        discounted = asset_cost(100, 15, .02, self.options(discount_rate=.04))
        self.assertLess(discounted['total'], undiscounted['total'])

    def test_fixed_diameter_laminar_hand_calculation_both_pipe_legs(self):
        options = self.options(loss_coefficient=0, terminal_pressure_pa=0, minor_loss_k=0)
        case = prepare_strategies(np.array([1.]), options)[0]['small']
        length, diameter = 100, options['small_diameter_mm'] / 1000
        flow = 1 / (options['rho'] * options['cp'] * 30)
        velocity = flow / (np.pi * diameter ** 2 / 4)
        re = options['rho'] * velocity * diameter / options['mu']
        self.assertLess(re, 2300)
        expected_pressure = 2 * (64 / re) * length / diameter * options['rho'] * velocity ** 2 / 2
        row = evaluate(case, length, options)
        self.assertAlmostEqual(row['max_pressure_pa'], expected_pressure)
        self.assertAlmostEqual(row['annual_pump_kwh'], expected_pressure * flow / (1000 * options['pump_efficiency']))

    def test_twin_pipe_heat_loss_costed_once_and_increases_flow(self):
        options = self.options()
        case = prepare_strategies(np.array([100., 100.]), options)[0]['large']
        row = evaluate(case, 1000, options)
        expected_loss = .25 * 1000 * (80 + 50 - 20) / 1000 * 2
        self.assertAlmostEqual(row['annual_heat_loss_kwh'], expected_loss)
        self.assertAlmostEqual(row['annual_source_kwh'], 200 + expected_loss)
        self.assertGreater(row['max_velocity'], evaluate(case, 1000, options, couple_losses=False)['max_velocity'])

    def test_hp_conserves_energy_and_cop_changes_electricity(self):
        load = np.array([100., 200.])
        options = self.options(loss_coefficient=0)
        cases = prepare_strategies(load, options)[0]
        hp = evaluate(cases['hp'], 100, options)
        self.assertAlmostEqual(hp['annual_source_kwh'] + hp['annual_hp_kwh'], sum(load))
        self.assertAlmostEqual(hp['annual_hp_kwh'], 75)
        self.assertAlmostEqual(hp['components']['node_capex'], 20000 + 600 * 200)

    def storage_options(self, **changes):
        return self.options(tes_capacity_kwh=50, tes_delivery_cap_kw=50,
            tes_charge_kw=100, tes_discharge_kw=100, tes_charge_efficiency=1,
            tes_discharge_efficiency=1, tes_loss_per_hour=0, **changes)

    def test_storage_meets_early_peak_and_recharges_without_free_heat(self):
        result = storage_dispatch(np.array([100., 0.] * 5), self.storage_options())
        self.assertTrue(result['feasible'])
        self.assertAlmostEqual(result['initial_kwh'], result['final_kwh'], places=6)
        self.assertAlmostEqual(sum(result['profile']), 500)
        self.assertAlmostEqual(result['initial_kwh'], 50, places=6)
        self.assertFalse(np.any((result['charges'] > 0) & (result['discharges'] > 0)))

    def test_storage_cannot_shave_a_sustained_energy_deficit(self):
        result = storage_dispatch(np.array([100.] * 100), self.storage_options())
        self.assertFalse(result['feasible'])
        self.assertGreater(result['unmet_kwh'], 4900)

    def test_storage_losses_and_efficiencies_require_extra_source_heat(self):
        options = self.options(tes_capacity_kwh=100, tes_delivery_cap_kw=70,
            tes_charge_kw=70, tes_discharge_kw=70, tes_loss_per_hour=.01)
        result = storage_dispatch(np.array([100., 0.] * 8), options)
        self.assertTrue(result['feasible'])
        self.assertGreater(sum(result['profile']), 800)
        self.assertAlmostEqual(result['initial_kwh'], result['final_kwh'], places=5)

    def test_storage_capacity_increases_tank_cost_and_volume(self):
        load = np.array([100., 0.])
        small = prepare_strategies(load, self.options(tes_capacity_kwh=100))
        large = prepare_strategies(load, self.options(tes_capacity_kwh=200))
        self.assertAlmostEqual(large[2], 2 * small[2])
        self.assertGreater(large[0]['tes']['node_capex'], small[0]['tes']['node_capex'])

    def test_infeasible_options_excluded_from_recommendation(self):
        result = run_study([1000., 1000.], self.options(available_pressure_pa=1000))
        self.assertIsNone(result['summary']['winner'])
        self.assertTrue(all(not row['feasible'] for row in result['selected']))

    def test_analytical_crossing_statuses(self):
        self.assertEqual(crossing_summary(100, 110, 200, 205)['length_m'], 20)
        self.assertEqual(crossing_summary(100, 110, 50, 55)['status'], 'no_positive_crossing')
        self.assertEqual(crossing_summary(100, 110, 200, 210)['status'], 'parallel')
        self.assertEqual(crossing_summary(100, 110, 100, 110)['status'], 'coincident')

    def test_numerical_crossing_agrees_with_analytical_without_pipe_loss_coupling(self):
        options = self.options(loss_coefficient=0, maximum_length_m=5000,
            max_velocity=10, available_pressure_pa=10000000)
        result = run_study([1000., 300., 1000., 300.], options)
        for pair in result['comparisons']:
            root = pair['analytical']['length_m']
            if root is not None and 0 <= root <= 5000:
                self.assertTrue(pair['numerical'])
                self.assertAlmostEqual(pair['numerical'][0]['length_m'], root, places=4)

    def test_cost_breakdown_reconciles_and_snapshot_is_json_safe(self):
        result = run_study([100., 0., 200., 30.], self.options())
        for row in result['selected']:
            self.assertAlmostEqual(sum(row['components'].values()), row['npc'])
        json.dumps(result, allow_nan=False)
        changed = run_study([100., 0., 200., 30.], self.options(alpha=.5))
        self.assertNotEqual(result['input_sha256'], changed['input_sha256'])

    def test_demo_is_full_year_with_exact_requested_peak(self):
        profile = demo_profile(1234)
        self.assertEqual(len(profile), 8760)
        self.assertAlmostEqual(max(profile), 1234)
        self.assertGreater(min(profile), 0)

    def test_root_search_distinguishes_multiple_roots_and_coincident_curves(self):
        function = lambda x: (x - 1) * (x - 3)
        roots, status = find_crossings([0, 2, 4], [3, -1, 3], function)
        self.assertEqual(status, 'crossing')
        np.testing.assert_allclose(roots, [1, 3], atol=1e-8)
        self.assertEqual(find_crossings([0, 1, 2], [0, 0, 0], lambda x: 0), ([], 'coincident'))

    def test_thermal_loss_only_crossing_matches_independent_hand_calculation(self):
        options = self.options(years=1, discount_rate=0, maximum_length_m=100000,
            hp_fixed_cost=100, hp_cost_per_kw=0, hp_om=0, hp_life=100)
        result = run_study([100., 100., 100., 100.], options)
        view = next(v for v in result['critical_lengths'] if v['key'] == 'thermal')
        # 0.25 W/mK * (110 - 60) K / 1000 * 4 h * 0.05 CAD/kWh.
        expected = 100 / (.25 * 50 / 1000 * 4 * .05)
        self.assertEqual(len(view['crossings']), 1)
        self.assertAlmostEqual(view['crossings'][0]['length_m'], expected, places=4)
        self.assertTrue(view['diagnostic'])

    def test_pumping_intersection_uses_asset_premium_and_pv_electricity_savings(self):
        options = self.options()
        load = np.array([100., 200., 0., 50.])
        cases = prepare_strategies(load, options)[0]
        result = run_study(load, options)
        view = next(v for v in result['critical_lengths'] if v['key'] == 'pump')
        point = next(p for p in view['points'] if p['length_m'] == options['selected_length_m'])
        a, b = [evaluate(cases[k], options['selected_length_m'], options) for k in ('small', 'large')]
        self.assertAlmostEqual(point['difference'], b['npc'] - a['npc'])
        self.assertAlmostEqual(point['right'], a['components']['pumping_pv'] - b['components']['pumping_pv'])

    def test_focused_plots_include_exact_crossings_and_match_total_comparisons(self):
        result = run_study([1000., 300., 1000., 300.], self.options())
        self.assertEqual({v['key'] for v in result['critical_lengths']}, {'pump', 'thermal', 'hp', 'tes'})
        for view in result['critical_lengths']:
            for root in view['crossings']:
                point = next(p for p in view['points'] if p['length_m'] == root['length_m'])
                self.assertAlmostEqual(point['left'], point['right'], places=4)
                self.assertEqual(point['feasible'], root['feasible'])
            if view['key'] in ('hp', 'tes'):
                pair = next(p for p in result['comparisons'] if p['a'] == 'large' and p['b'] == view['key'])
                np.testing.assert_allclose([r['length_m'] for r in view['crossings']],
                                           [r['length_m'] for r in pair['numerical']], atol=1e-5)

    def make_zone_fixture(self, costs, maximum=4000, feasible=None, service=None):
        keys = list(costs)
        lengths = np.linspace(0, maximum, 81)
        pairs = []
        for index, a in enumerate(keys):
            for b in keys[index + 1:]:
                difference = lambda length, a=a, b=b: costs[a](length) - costs[b](length)
                roots, _ = find_crossings(lengths, [difference(x) for x in lengths], difference)
                pairs.append(dict(a=a, b=b, numerical=[dict(length_m=x) for x in roots]))
        def evaluate_at(key, length):
            ok = feasible(key, length) if feasible else True
            return dict(npc=costs[key](length), feasible=ok, service_feasible=service(key, length) if service else True,
                        reasons=[] if ok else ['Pressure limit exceeded.'])
        return strategy_zone_analysis(keys, lengths, pairs, evaluate_at)

    def test_three_winning_zones_and_dominated_hp_pipe_crossing(self):
        analysis = self.make_zone_fixture(dict(
            small=lambda x: 1000 + 2000 * x,
            large=lambda x: 1000 * x,
            hp=lambda x: 735000 + 325 * x,
            tes=lambda x: 360000 + 450 * x))
        selected = analysis['maps']['small,large,hp,tes']
        self.assertEqual([z['winner'] for z in selected['zones']], ['large', 'tes', 'hp'])
        self.assertEqual([t['label'] for t in selected['transitions']], ['Lcrit,1', 'Lcrit,2'])
        self.assertAlmostEqual(selected['transitions'][0]['length_m'], 360000 / 550, places=5)
        self.assertAlmostEqual(selected['transitions'][1]['length_m'], 3000, places=5)
        ignored = next(p for p in selected['pairwise_crossings'] if p['a'] == 'large' and p['b'] == 'hp')
        self.assertEqual(ignored['classification'], 'dominated_crossing')
        self.assertEqual(ignored['cheaper_keys'], ['tes'])
        self.assertEqual(ignored['best_key'], 'tes')

    def test_strategy_subset_recomputes_the_overall_winner_and_critical_boundary(self):
        analysis = self.make_zone_fixture(dict(large=lambda x: 1000 * x,
            hp=lambda x: 735000 + 325 * x, tes=lambda x: 360000 + 450 * x))
        pair_only = analysis['maps']['large,hp']
        self.assertEqual([z['winner'] for z in pair_only['zones']], ['large', 'hp'])
        self.assertAlmostEqual(pair_only['transitions'][0]['length_m'], 735000 / 675, places=5)
        self.assertEqual(pair_only['pairwise_crossings'][0]['classification'], 'winner_change')
        self.assertEqual(len(analysis['maps']), 7)

    def test_feasibility_limit_is_not_labeled_as_a_cost_intersection(self):
        analysis = self.make_zone_fixture(dict(large=lambda x: 100 + x, hp=lambda x: 200 + 2 * x),
            maximum=10, feasible=lambda key, length: key != 'large' or length <= 5)
        selected = analysis['maps']['large,hp']
        self.assertEqual([z['winner'] for z in selected['zones']], ['large', 'hp'])
        self.assertAlmostEqual(selected['transitions'][0]['length_m'], 5, places=6)
        self.assertEqual(selected['transitions'][0]['kind'], 'feasibility_change')
        self.assertTrue(selected['transitions'][0]['label'].startswith('Limit'))
        self.assertIsNone(selected['transitions'][0]['cost_cad'])

    def test_no_feasible_zone_is_explicit(self):
        analysis = self.make_zone_fixture(dict(small=lambda x: x, large=lambda x: 100 + x),
            maximum=10, feasible=lambda key, length: length <= 5)
        selected = analysis['maps']['small,large']
        self.assertEqual([z['winner'] for z in selected['zones']], ['small', None])
        self.assertEqual(selected['transitions'][0]['kind'], 'feasibility_change')

    def test_identical_costs_remain_an_optimal_tie_without_spurious_boundaries(self):
        analysis = self.make_zone_fixture(dict(small=lambda x: 100 + x, large=lambda x: 100 + x), maximum=10)
        selected = analysis['maps']['small,large']
        self.assertEqual(len(selected['zones']), 1)
        self.assertEqual(selected['zones'][0]['tied_keys'], ['small', 'large'])
        self.assertEqual(selected['transitions'], [])
        self.assertEqual(selected['pairwise_crossings'], [])

    def test_actual_model_zone_winners_are_cheapest_feasible_at_interval_midpoints(self):
        options = self.options()
        load = np.array([1000., 300., 1000., 300.])
        result = run_study(load, options)
        cases = prepare_strategies(load, options)[0]
        for zone in result['strategy_analysis']['maps']['small,large,hp,tes']['zones']:
            midpoint = (zone['start_m'] + zone['end_m']) / 2
            eligible = [evaluate(case, midpoint, options) for case in cases.values()]
            eligible = [row for row in eligible if row['feasible']]
            expected = min(eligible, key=lambda row: row['npc'])['key'] if eligible else None
            self.assertEqual(zone['winner'], expected)

    def test_economic_zones_ignore_pressure_limits_without_inventing_a_cost_crossing(self):
        analysis = self.make_zone_fixture(dict(large=lambda x: 100 + x, hp=lambda x: 200 + 2 * x),
            maximum=10, feasible=lambda key, length: key != 'large' or length <= 5)
        self.assertEqual([z['winner'] for z in analysis['maps']['large,hp']['zones']], ['large', 'hp'])
        economic = analysis['economic_maps']['large,hp']
        self.assertEqual([z['winner'] for z in economic['zones']], ['large'])
        self.assertEqual(economic['transitions'], [])

    def test_economic_zones_still_exclude_storage_that_cannot_meet_heating_demand(self):
        analysis = self.make_zone_fixture(dict(large=lambda x: 100 + x, tes=lambda x: 0), maximum=10,
            feasible=lambda key, length: key != 'tes', service=lambda key, length: key != 'tes')
        self.assertEqual([z['winner'] for z in analysis['economic_maps']['large,tes']['zones']], ['large'])

    def test_economic_three_strategy_map_has_ignored_pairwise_crossing_between_winning_boundaries(self):
        analysis = self.make_zone_fixture(dict(large=lambda x: 1000 * x,
            hp=lambda x: 735000 + 325 * x, tes=lambda x: 360000 + 450 * x))
        economic = analysis['economic_maps']['large,hp,tes']
        self.assertEqual([z['winner'] for z in economic['zones']], ['large', 'tes', 'hp'])
        ignored = next(p for p in economic['pairwise_crossings'] if p['a'] == 'large' and p['b'] == 'hp')
        self.assertEqual(ignored['classification'], 'dominated_crossing')
        self.assertTrue(economic['transitions'][0]['length_m'] < ignored['length_m'] < economic['transitions'][1]['length_m'])


if __name__ == '__main__':
    unittest.main()
