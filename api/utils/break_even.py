"""Reproducible, fixed-diameter trunk comparison. Costs are real base-year CAD.

Pure calculation module: no ORM writes, no dependence on Django settings.
The installed rate covers BOTH legs per ROUTE metre, including ordinary fittings.
"""
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from decimal import Decimal, ROUND_CEILING
from itertools import combinations
import numpy as np


# key, label (units), default, minimum, maximum, input step
FIELD_GROUPS = [
    ('Installed pipe cost', [
        ('diameter_band_width_mm', 'Diameter cost band width (mm; 0 = continuous)', 0, 0, 3000, 10),
        ('reference_cost', 'Reference installed cost (CAD / route m)', 4000, 0, 1000000, 100),
        ('reference_diameter_mm', 'Reference internal diameter (mm)', 200, 1, 3000, 5),
        ('alpha', 'Diameter-dependent cost fraction, alpha (0–1)', .30, 0, 1, .05),
        ('exponent_n', 'Diameter scaling exponent, n', 1, 0, 5, .1),
        ('small_diameter_mm', 'Small pipe internal diameter (mm)', 100, 1, 3000, 5),
        ('large_diameter_mm', 'Large pipe internal diameter (mm)', 150, 1, 3000, 5),
    ]),
    ('Study and electricity', [
        ('selected_length_m', 'Distance to compare (route m)', 1000, 0, 100000, 100),
        ('maximum_length_m', 'Maximum plotted distance (route m)', 5000, 1, 100000, 100),
        ('years', 'Study lifetime (whole years)', 30, 1, 100, 1),
        ('discount_rate', 'Real discount rate (fraction / year)', .04, 0, .3, .005),
        ('electricity_price', 'Electricity (CAD / kWh)', .15, 0, 10, .01),
        ('source_heat_price', 'Source heat, excluding node HP (CAD / thermal kWh)', .05, 0, 10, .01),
        ('demo_peak_kw', 'Illustrative annual profile peak (kW)', 1000, 1, 1000000, 100),
    ]),
    ('Hydraulics and heat losses', [
        ('rho', 'Water density (kg / m³)', 997, 100, 2000, 1),
        ('cp', 'Specific heat (kJ / kg K)', 4.184, .1, 10, .001),
        ('mu', 'Dynamic viscosity (Pa s)', .001, .00001, .1, .0001),
        ('roughness_m', 'Absolute roughness (m)', .000045, 0, .01, .000001),
        ('pump_efficiency', 'Combined pump and motor efficiency', .75, .01, 1, .01),
        ('max_velocity', 'Maximum allowable velocity (m / s)', 2.5, .1, 10, .1),
        ('available_pressure_pa', 'Maximum total pump differential (Pa)', 1000000, 1000, 10000000, 10000),
        ('terminal_pressure_pa', 'Terminal pressure requirement (Pa)', 50000, 0, 1000000, 1000),
        ('minor_loss_k', 'Combined supply / return fitting loss coefficient, K', 10, 0, 10000, 1),
        ('supply_c', 'Baseline supply temperature (°C)', 80, 1, 150, 1),
        ('return_c', 'Baseline return temperature (°C)', 50, 0, 149, 1),
        ('ground_c', 'Ground temperature (°C)', 10, -30, 100, 1),
        ('loss_coefficient', 'Heat loss coefficient per pipe leg (W / m K)', .25, 0, 10, .01),
    ]),
    ('Pumps and asset accounting', [
        ('pump_fixed_cost', 'Pump installed fixed cost (CAD)', 2000, 0, 10000000, 100),
        ('pump_cost_per_kw', 'Pump installed capacity cost (CAD / electric kW)', 800, 0, 100000, 100),
        ('pump_om', 'Pump annual maintenance / installed cost', .02, 0, .5, .005),
        ('pump_life', 'Pump service life (whole years)', 15, 1, 100, 1),
        ('pipe_om', 'Pipe annual maintenance / installed cost', .005, 0, .5, .001),
        ('pipe_life', 'Pipe service life (whole years)', 50, 1, 150, 1),
    ]),
    ('Node heat pump', [
        ('hp_cop', 'Node heating COP (constant scenario assumption)', 4, 1.01, 20, .1),
        ('hp_supply_c', 'HP strategy trunk supply temperature (°C)', 60, 1, 149, 1),
        ('hp_return_c', 'HP strategy trunk return temperature (°C)', 20, 0, 148, 1),
        ('hp_fixed_cost', 'HP fixed installed cost, A (CAD)', 20000, 0, 10000000, 1000),
        ('hp_cost_per_kw', 'HP installed capacity cost, B (CAD / output kW)', 600, 0, 100000, 50),
        ('hp_om', 'HP annual maintenance / installed cost', .02, 0, .5, .005),
        ('hp_life', 'HP service life (whole years)', 20, 1, 100, 1),
    ]),
    ('Node thermal storage', [
        ('tes_capacity_kwh', 'Usable storage energy (kWh)', 3000, 0, 10000000, 100),
        ('tes_delivery_cap_kw', 'Trunk delivery limit at node, before losses (kW)', 700, 0, 1000000, 50),
        ('tes_charge_kw', 'Maximum charging power (kW)', 500, 0, 1000000, 50),
        ('tes_discharge_kw', 'Maximum discharging power (kW)', 500, 0, 1000000, 50),
        ('tes_charge_efficiency', 'Charging efficiency', .95, .01, 1, .01),
        ('tes_discharge_efficiency', 'Discharging efficiency', .95, .01, 1, .01),
        ('tes_loss_per_hour', 'Standing loss fraction per hour', .0005, 0, .1, .0001),
        ('tes_delta_t', 'Usable tank temperature difference (K)', 30, 1, 100, 1),
        ('tes_usable_fraction', 'Usable tank volume fraction', .9, .01, 1, .05),
        ('tes_aux_kw', 'Auxiliary electric power while charging / discharging (kW)', 1, 0, 10000, .1),
        ('tes_fixed_cost', 'TES fixed installed cost, A (CAD)', 10000, 0, 10000000, 1000),
        ('tes_cost_per_m3', 'Installed tank cost (CAD / m³)', 1000, 0, 100000, 100),
        ('tes_charge_cost_per_kw', 'Charging equipment (CAD / kW)', 50, 0, 100000, 10),
        ('tes_discharge_cost_per_kw', 'Discharging equipment (CAD / kW)', 50, 0, 100000, 10),
        ('tes_om', 'TES annual maintenance / installed cost', .01, 0, .5, .005),
        ('tes_life', 'TES service life (whole years)', 25, 1, 100, 1),
    ]),
]
DEFAULTS = {key: value for _, fields in FIELD_GROUPS for key, _, value, *_ in fields}
STRATEGIES = [('small', 'A · Small pipe'), ('large', 'B · Larger pipe'),
              ('hp', 'C · Small pipe + heat pump'), ('tes', 'D · Small pipe + TES')]


def schema():
    # Continuous scaling is the default; existing banded snapshots remain reproducible.
    return dict(defaults=dict(DEFAULTS), groups=[dict(title=title, fields=[dict(
        key=k, label=label, min=lo, max=hi, step=step) for k, label, _, lo, hi, step in fields])
        for title, fields in FIELD_GROUPS])


def validate_options(raw):
    if not isinstance(raw, dict):
        raise ValueError('Study options must be an object.')
    unknown = set(raw) - set(DEFAULTS)
    if unknown:
        raise ValueError('Unknown study options: ' + ', '.join(sorted(unknown)))
    options = dict(DEFAULTS)
    for _, fields in FIELD_GROUPS:
        for key, label, default, lo, hi, _ in fields:
            value = raw.get(key, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f'{label}: enter a finite number from {lo} to {hi}.')
            options[key] = value
    for key in ('years', 'pipe_life', 'pump_life', 'hp_life', 'tes_life'):
        if options[key] != int(options[key]):
            raise ValueError(f'{key} must be a whole number of years.')
        options[key] = int(options[key])
    if options['small_diameter_mm'] >= options['large_diameter_mm']:
        raise ValueError('The large diameter must exceed the small diameter.')
    if options['selected_length_m'] > options['maximum_length_m']:
        raise ValueError('The comparison distance must not exceed the plotted distance.')
    for supply, ret in [('supply_c', 'return_c'), ('hp_supply_c', 'hp_return_c')]:
        if options[supply] <= options[ret] or options[ret] < options['ground_c']:
            raise ValueError('Each supply must exceed its return temperature; returns must be at least the ground temperature.')
    if options['hp_supply_c'] >= options['supply_c']:
        raise ValueError('The node HP scenario requires trunk supply below the common heating delivery temperature.')
    # Ideal upper bound using the warmest source-side temperature: deliberately permissive.
    carnot = (options['supply_c'] + 273.15) / (options['supply_c'] - options['hp_supply_c'])
    if options['hp_cop'] >= carnot:
        raise ValueError('The selected node COP exceeds the ideal heating COP for these temperatures.')
    return options


def cost_diameter(diameter_mm, options):
    """Upper-inclusive bands (0, w], (w, 2w], ...; hydraulics retain actual D."""
    width = options.get('diameter_band_width_mm', 0)
    if width <= 0:
        return diameter_mm
    # Decimal arithmetic avoids promoting an exact decimal boundary by one band.
    steps = (Decimal(str(diameter_mm)) / Decimal(str(width))).to_integral_value(rounding=ROUND_CEILING)
    return float(steps * Decimal(str(width)))


def installed_cost(diameter_mm, options):
    return options['reference_cost'] * ((1 - options['alpha']) + options['alpha'] *
        (cost_diameter(diameter_mm, options) / options['reference_diameter_mm']) ** options['exponent_n'])


def asset_cost(capex, life, om_rate, options, residual=False):
    """End-year cash flows; replacements strictly before the horizon."""
    years, rate = options['years'], options['discount_rate']
    pv_factor = sum((1 + rate) ** -y for y in range(1, years + 1))
    replacement_years = list(range(life, years, life))
    replacement = sum(capex / (1 + rate) ** y for y in replacement_years)
    last = replacement_years[-1] if replacement_years else 0
    salvage = capex * max(0, last + life - years) / life / (1 + rate) ** years if residual else 0.0
    return dict(initial=capex, maintenance_pv=capex * om_rate * pv_factor,
                replacement_pv=replacement, residual_pv=salvage,
                total=capex + capex * om_rate * pv_factor + replacement - salvage)


def demo_profile(peak_kw):
    t = np.arange(8760)
    season = .5 + .5 * np.cos(2 * np.pi * (t / 24 - 15) / 365)
    daily = .75 + .25 * np.cos(2 * np.pi * (t % 24 - 7) / 24)
    values = .12 + .88 * season * daily
    return values * peak_kw / values.max()


def storage_dispatch(load, options):
    """Greedy maximum-recharge peak shaving, with a periodic annual SOC.

    Solve the monotone annual state map for its greatest fixed point. An empty
    state may not provide free heat: unmet demand is checked after finding SOC.
    No simultaneous charge/discharge. Not a tariff-optimized dispatch.
    """
    cap = options['tes_capacity_kwh']
    delivery = options['tes_delivery_cap_kw']
    eta_c, eta_d = options['tes_charge_efficiency'], options['tes_discharge_efficiency']
    retention = 1 - options['tes_loss_per_hour']
    charge_limit, discharge_limit = options['tes_charge_kw'], options['tes_discharge_kw']
    loads = np.asarray(load).tolist()

    def simulate(initial, record=False):
        energy, unmet = initial, 0.0
        supplied, states, charges, discharges = [], [initial], [], []
        for demand in loads:
            energy *= retention
            if demand > delivery:
                discharge = min(demand - delivery, discharge_limit, energy * eta_d)
                charge = 0.0
                energy -= discharge / eta_d
                source = min(demand, delivery)
                unmet += max(0.0, demand - source - discharge)
            else:
                discharge = 0.0
                charge = min(delivery - demand, charge_limit, max(0.0, cap - energy) / eta_c)
                energy += charge * eta_c
                source = demand + charge
            if record:
                supplied.append(source)
                states.append(energy)
                charges.append(charge)
                discharges.append(discharge)
        return energy, unmet, supplied, states, charges, discharges

    lo, hi = 0.0, cap
    for _ in range(35):
        mid = (lo + hi) / 2
        end = simulate(mid)[0]
        if end >= mid:
            lo = mid
        else:
            hi = mid
    initial = (lo + hi) / 2
    end, unmet, supplied, states, charges, discharges = simulate(initial, True)
    return dict(profile=np.asarray(supplied), states=states, charges=np.asarray(charges),
        discharges=np.asarray(discharges), initial_kwh=initial, final_kwh=end,
        unmet_kwh=unmet, cyclic_error_kwh=abs(initial - end),
        feasible=unmet <= 1e-5 and abs(initial - end) <= max(1e-5, cap * 1e-7))


def prepare_strategies(load, options):
    storage = storage_dispatch(load, options)
    volume = 3600 * options['tes_capacity_kwh'] / (
        options['rho'] * options['cp'] * options['tes_delta_t'] * options['tes_usable_fraction'])
    hp_cap = options['hp_fixed_cost'] + options['hp_cost_per_kw'] * float(max(load))
    tes_cap = (options['tes_fixed_cost'] + options['tes_cost_per_m3'] * volume
        + options['tes_charge_cost_per_kw'] * options['tes_charge_kw']
        + options['tes_discharge_cost_per_kw'] * options['tes_discharge_kw'])
    cases = {}
    for key, label in STRATEGIES:
        hp = key == 'hp'
        tes = key == 'tes'
        cases[key] = dict(key=key, label=label,
            diameter_mm=options['large_diameter_mm'] if key == 'large' else options['small_diameter_mm'],
            supply=options['hp_supply_c'] if hp else options['supply_c'],
            ret=options['hp_return_c'] if hp else options['return_c'],
            profile=load * (1 - 1 / options['hp_cop']) if hp else storage['profile'] if tes else load,
            hp_kwh=float(sum(load) / options['hp_cop']) if hp else 0.0,
            aux_kwh=float(np.count_nonzero((storage['charges'] + storage['discharges']) > 1e-8)) * options['tes_aux_kw'] if tes else 0.0,
            node_capex=hp_cap if hp else tes_cap if tes else 0.0,
            node_life=options['hp_life'] if hp else options['tes_life'],
            node_om=options['hp_om'] if hp else options['tes_om'],
            physical_ok=storage['feasible'] if tes else True,
            node_reason=f"Storage cannot meet annual demand: {storage['unmet_kwh']:.3f} kWh unmet." if tes and not storage['feasible'] else '')
    return cases, storage, volume


def evaluate(case, length_m, options, residual=False, couple_losses=True):
    """Hourly water properties are held constant; both pipe legs share an ID."""
    diameter = case['diameter_mm'] / 1000
    loss_kw = options['loss_coefficient'] * length_m * (
        case['supply'] + case['ret'] - 2 * options['ground_c']) / 1000
    heat = case['profile'] + (loss_kw if couple_losses else 0)
    flow = heat / (options['rho'] * options['cp'] * (case['supply'] - case['ret']))
    velocity = 4 * flow / (np.pi * diameter ** 2)
    re = options['rho'] * velocity * diameter / options['mu']
    # Swamee–Jain above transition, laminar Darcy factor below 2300;
    # linear interpolation avoids a discontinuity in the transition interval.
    safe_re = np.maximum(re, 1.0)
    turbulent = .25 / np.log10(options['roughness_m'] / (3.7 * diameter) + 5.74 / safe_re ** .9) ** 2
    laminar = 64 / safe_re
    blend = np.clip((re - 2300) / 1700, 0, 1)
    friction = np.where(re < 2300, laminar, (1 - blend) * laminar + blend * turbulent)
    pressure = 2 * friction * length_m / diameter * options['rho'] * velocity ** 2 / 2
    pressure += options['minor_loss_k'] * options['rho'] * velocity ** 2 / 2
    # Fitting construction costs are already in the installed rate; hydraulic
    # resistance is a separate physical input, not an extra construction charge.
    pressure += np.where(flow > 0, options['terminal_pressure_pa'], 0)
    pump_kw = pressure * flow / (1000 * options['pump_efficiency'])
    peak_pump = float(np.max(pump_kw))
    pump_energy = float(np.sum(pump_kw))
    source_energy = float(np.sum(case['profile'])) + loss_kw * len(flow)
    source_cost = source_energy * options['source_heat_price']
    pump_cost = pump_energy * options['electricity_price']
    hp_cost = case['hp_kwh'] * options['electricity_price']
    aux_cost = case['aux_kwh'] * options['electricity_price']
    annual_energy = source_cost + pump_cost + hp_cost + aux_cost
    pv_factor = sum((1 + options['discount_rate']) ** -y for y in range(1, options['years'] + 1))
    pipe_cap = installed_cost(case['diameter_mm'], options) * length_m
    pump_cap = options['pump_fixed_cost'] + options['pump_cost_per_kw'] * peak_pump
    assets = [asset_cost(pipe_cap, options['pipe_life'], options['pipe_om'], options, residual),
        asset_cost(pump_cap, options['pump_life'], options['pump_om'], options, residual),
        asset_cost(case['node_capex'], case['node_life'], case['node_om'], options, residual)]
    components = dict(pipe_capex=pipe_cap, pump_capex=pump_cap, node_capex=case['node_capex'],
        source_heat_pv=source_cost * pv_factor, pumping_pv=pump_cost * pv_factor,
        hp_electricity_pv=hp_cost * pv_factor, tes_auxiliary_pv=aux_cost * pv_factor,
        maintenance_pv=sum(a['maintenance_pv'] for a in assets),
        replacements_pv=sum(a['replacement_pv'] for a in assets),
        residual_credit_pv=-sum(a['residual_pv'] for a in assets))
    max_v, max_pressure = float(max(velocity)), float(max(pressure))
    reasons = []
    if not case['physical_ok']:
        reasons.append(case['node_reason'])
    if max_v > options['max_velocity'] + 1e-8:
        reasons.append('Maximum velocity exceeded.')
    if max_pressure > options['available_pressure_pa'] + 1e-6:
        reasons.append('Available pump differential exceeded.')
    return dict(key=case['key'], label=case['label'], length_m=float(length_m),
        diameter_mm=case['diameter_mm'], installed_cad_per_m=installed_cost(case['diameter_mm'], options),
        npc=sum(components.values()), components=components,
        pipe_asset_pv=assets[0]['total'], pump_asset_pv=assets[1]['total'], node_asset_pv=assets[2]['total'],
        heat_loss_pv=loss_kw * len(flow) * options['source_heat_price'] * pv_factor,
        annual_source_kwh=source_energy, annual_heat_loss_kwh=loss_kw * len(flow),
        annual_pump_kwh=pump_energy, annual_hp_kwh=case['hp_kwh'], annual_aux_kwh=case['aux_kwh'],
        annual_energy_cost=annual_energy, peak_pump_kw=peak_pump,
        max_velocity=max_v, max_pressure_pa=max_pressure, feasible=not reasons,
        service_feasible=case['physical_ok'], reasons=reasons)


def crossing_summary(a0, a1, b0, b1):
    intercept = a0 - b0
    slope = (a1 - a0) - (b1 - b0)
    if abs(slope) < 1e-7:
        return dict(length_m=None, status='coincident' if abs(intercept) < 1e-5 else 'parallel')
    root = -intercept / slope
    return dict(length_m=root if root >= 0 else None,
                status='crossing' if root >= 0 else 'no_positive_crossing')


def find_crossings(lengths, differences, difference):
    """Bracketed sign-change roots, with explicit coincident-series handling."""
    if all(abs(value) <= 1e-6 for value in differences):
        return [], 'coincident'
    roots = []
    for index, length in enumerate(lengths):
        if abs(differences[index]) <= 1e-6:
            roots.append(float(length))
        elif index and differences[index - 1] * differences[index] < 0:
            lo, hi, flo = float(lengths[index - 1]), float(length), differences[index - 1]
            for _ in range(40):
                mid = (lo + hi) / 2
                value = difference(mid)
                if flo * value <= 0:
                    hi = mid
                else:
                    lo, flo = mid, value
            roots.append((lo + hi) / 2)
    unique = []
    for root in roots:
        if not unique or abs(root - unique[-1]) > 1e-5:
            unique.append(root)
    return unique, 'crossing' if unique else 'no_crossing_in_tested_range'


def critical_length_views(cases, options, residual, lengths):
    """Four explicitly defined intersections; diagnostics do not replace NPC."""
    cache = {}
    def at(key, length):
        identity = (key, float(length))
        if identity not in cache:
            cache[identity] = evaluate(cases[key], length, options, residual)
        return cache[identity]

    definitions = [
        dict(key='pump', symbol='Lcrit,pump', title='Pipe upsizing versus pumping savings',
            a='small', b='large', left_label='Extra pipe + pump asset cost (B − A)',
            right_label='Pumping electricity savings (A − B)',
            definition='Extra lifecycle pipe and pump asset cost of B intersects its discounted pumping-electricity savings relative to A. Includes asset maintenance, replacements and selected residual treatment.',
            diagnostic=True),
        dict(key='thermal', symbol='Lcrit,thermal-loss', title='Heat-loss savings versus HP equipment',
            a='large', b='hp', left_label='HP equipment lifecycle cost',
            right_label='Avoided pipe heat-loss cost (B − C)',
            definition='HP equipment lifecycle cost intersects discounted pipe heat-loss savings from the lower-temperature C trunk relative to B. Isolated diagnostic: excludes HP electricity, changes in source heat extraction, pipe-price differences and pumping. Use Lcrit,HP for the complete comparison.',
            diagnostic=True),
        dict(key='hp', symbol='Lcrit,HP', title='Larger pipe versus heat-pump strategy',
            a='large', b='hp', left_label='B · Larger pipe: total lifecycle cost',
            right_label='C · Small pipe + HP: total lifecycle cost',
            definition='Complete discounted lifecycle cost of B intersects C, including construction, equipment, source heat, pumping, node electricity, maintenance and replacements.',
            diagnostic=False),
        dict(key='tes', symbol='Lcrit,TES', title='Larger pipe versus storage strategy',
            a='large', b='tes', left_label='B · Larger pipe: total lifecycle cost',
            right_label='D · Small pipe + TES: total lifecycle cost',
            definition='Complete discounted lifecycle cost of B intersects D using the chronological cyclic storage dispatch. A crossing is a usable design comparison only where both alternatives satisfy the modeled limits.',
            diagnostic=False),
    ]
    views = []
    for definition in definitions:
        def point(length):
            a, b = at(definition['a'], length), at(definition['b'], length)
            if definition['key'] == 'pump':
                left = b['pipe_asset_pv'] + b['pump_asset_pv'] - a['pipe_asset_pv'] - a['pump_asset_pv']
                right = a['components']['pumping_pv'] - b['components']['pumping_pv']
            elif definition['key'] == 'thermal':
                left, right = b['node_asset_pv'], a['heat_loss_pv'] - b['heat_loss_pv']
            else:
                left, right = a['npc'], b['npc']
            return dict(length_m=float(length), left=left, right=right, difference=left - right,
                        feasible=a['feasible'] and b['feasible'])
        sampled = [point(length) for length in lengths]
        roots, status = find_crossings(lengths, [p['difference'] for p in sampled], lambda length: point(length)['difference'])
        crossings = []
        extra = []
        for root in roots:
            row = point(root)
            crossings.append(dict(length_m=root, cost_cad=(row['left'] + row['right']) / 2,
                                  feasible=row['feasible'], difference_cad=row['difference']))
            # Resolve the local intersection even when it lies very near the origin.
            span = max(.25, root * .4)
            extra.extend(max(0, min(options['maximum_length_m'], root + offset * span))
                         for offset in (-1, -.5, -.1, 0, .1, .5, 1))
        xs = sorted(set(float(v) for v in list(lengths) + extra))
        views.append(dict(**definition, status=status, crossings=crossings, points=[point(x) for x in xs]))
    return views


def strategy_zone_analysis(keys, lengths, comparisons, evaluate_at):
    """Build the lower feasible envelope for all nonempty strategy subsets.

    Pairwise roots and feasibility boundaries partition distance into intervals
    with constant cost ranking. Evaluate each open interval, then merge equal
    winners. A feasibility boundary is never mislabeled as an economic crossing.
    The current hydraulic screens are monotone in length; transitions are refined
    from the supplied sample grid. General nonmonotone screens need finer sampling.
    """
    maximum = float(max(lengths))
    tolerance_m = max(1e-6, maximum * 1e-10)
    cache = {}
    def row(key, length):
        identity = (key, float(length))
        if identity not in cache:
            cache[identity] = evaluate_at(key, float(length))
        return cache[identity]
    def candidates(length):
        return [dict(key=key, npc=row(key, length)['npc'], feasible=row(key, length)['feasible'],
                     service_feasible=row(key, length).get('service_feasible', True)) for key in keys]

    events = [0.0, maximum]
    pairwise = []
    for pair in comparisons:
        for crossing in pair['numerical']:
            length = float(crossing['length_m'])
            if 0 <= length <= maximum:
                a, b = pair['a'], pair['b']
                cost = (row(a, length)['npc'] + row(b, length)['npc']) / 2
                pairwise.append(dict(a=a, b=b, length_m=length, cost_cad=cost,
                                     candidates=candidates(length)))
                events.append(length)
    feasibility_boundaries = []
    for key in keys:
        previous = row(key, lengths[0])['feasible']
        for index in range(1, len(lengths)):
            current = row(key, lengths[index])['feasible']
            if current != previous:
                lo, hi = float(lengths[index - 1]), float(lengths[index])
                left_feasible = previous
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if row(key, mid)['feasible'] == left_feasible:
                        lo = mid
                    else:
                        hi = mid
                boundary = (lo + hi) / 2
                events.append(boundary)
                failed = row(key, hi if not current else lo)
                feasibility_boundaries.append(dict(key=key, length_m=boundary,
                    feasible_before=previous, feasible_after=current, reasons=failed.get('reasons', [])))
            previous = current
    events.sort()
    boundaries = []
    for value in events:
        if not boundaries or value - boundaries[-1] > tolerance_m:
            boundaries.append(value)
    # Include exact transition coordinates and local neighbors in the chart data.
    xs = set(float(value) for value in lengths)
    for index, boundary in enumerate(boundaries):
        xs.add(boundary)
        if index:
            xs.add((boundaries[index - 1] + boundary) / 2)
    curves = {key: [dict(length_m=x, npc=row(key, x)['npc'], feasible=row(key, x)['feasible'])
                    for x in sorted(xs)] for key in keys}
    intervals = [dict(start_m=a, end_m=b, candidates=candidates((a + b) / 2))
                 for a, b in zip(boundaries, boundaries[1:]) if b - a > tolerance_m]
    maps = {}
    economic_maps = {}
    for count in range(1, len(keys) + 1):
        for subset in combinations(keys, count):
            zones = []
            for interval in intervals:
                eligible = [r for r in interval['candidates'] if r['key'] in subset and r['feasible']]
                best = min(eligible, key=lambda r: r['npc']) if eligible else None
                winner = best['key'] if best else None
                cost_tolerance = max(1e-5, abs(best['npc']) * 1e-10) if best else 1e-5
                tied = [r['key'] for r in eligible if abs(r['npc'] - best['npc']) <= cost_tolerance] if best else []
                if zones and zones[-1]['winner'] == winner and zones[-1]['tied_keys'] == tied:
                    zones[-1]['end_m'] = interval['end_m']
                else:
                    zones.append(dict(start_m=interval['start_m'], end_m=interval['end_m'],
                                      winner=winner, tied_keys=tied))
            transitions = []
            economic_count = 0
            for index, (before, after) in enumerate(zip(zones, zones[1:]), start=1):
                length = before['end_m']
                related = next((p for p in pairwise if abs(p['length_m'] - length) <= tolerance_m
                    and {p['a'], p['b']} == {before['winner'], after['winner']}), None)
                if related:
                    kind = 'cost_crossing'
                    economic_count += 1
                    label = f'Lcrit,{economic_count}'
                elif before['winner'] is None or after['winner'] is None:
                    kind, label = 'feasibility_change', f'Limit {index}'
                else:
                    kind, label = 'feasibility_change', f'Limit {index}'
                reasons = [r for event in feasibility_boundaries if abs(event['length_m'] - length) <= tolerance_m
                           and event['key'] in subset for r in event['reasons']]
                transitions.append(dict(length_m=length, from_key=before['winner'], to_key=after['winner'],
                    kind=kind, label=label, cost_cad=related['cost_cad'] if related else None, reasons=reasons))
            crossings = []
            for pair in pairwise:
                if pair['a'] not in subset or pair['b'] not in subset:
                    continue
                eligible = [r for r in pair['candidates'] if r['key'] in subset and r['feasible']]
                winner = min(eligible, key=lambda r: r['npc']) if eligible else None
                pair_feasible = all(next(r for r in pair['candidates'] if r['key'] == key)['feasible'] for key in (pair['a'], pair['b']))
                cost_tolerance = max(1e-4, abs(pair['cost_cad']) * 1e-9)
                on_envelope = pair_feasible and winner is not None and pair['cost_cad'] <= winner['npc'] + cost_tolerance
                transition = next((t for t in transitions if t['kind'] == 'cost_crossing'
                                   and abs(t['length_m'] - pair['length_m']) <= tolerance_m
                                   and {t['from_key'], t['to_key']} == {pair['a'], pair['b']}), None)
                classification = ('infeasible_crossing' if not pair_feasible else 'winner_change' if transition
                                  else 'optimal_tie' if on_envelope else 'dominated_crossing')
                cheaper = [r['key'] for r in eligible if r['npc'] < pair['cost_cad'] - cost_tolerance]
                crossings.append(dict(a=pair['a'], b=pair['b'], length_m=pair['length_m'], cost_cad=pair['cost_cad'],
                    classification=classification, feasible=pair_feasible, cheaper_keys=cheaper,
                    best_key=winner['key'] if winner else None, label=transition['label'] if transition else None))
            maps[','.join(subset)] = dict(strategies=list(subset), zones=zones, transitions=transitions,
                                         pairwise_crossings=crossings)
    # Pure economic zones disregard hydraulic screens but still require every
    # strategy to meet the same heating demand (no under-delivery advantage).
    for count in range(1, len(keys) + 1):
        for subset in combinations(keys, count):
            zones = []
            for interval in intervals:
                eligible = [r for r in interval['candidates'] if r['key'] in subset and r['service_feasible']]
                best = min(eligible, key=lambda r: r['npc']) if eligible else None
                winner = best['key'] if best else None
                cost_tolerance = max(1e-5, abs(best['npc']) * 1e-10) if best else 1e-5
                tied = [r['key'] for r in eligible if abs(r['npc'] - best['npc']) <= cost_tolerance] if best else []
                if zones and zones[-1]['winner'] == winner and zones[-1]['tied_keys'] == tied:
                    zones[-1]['end_m'] = interval['end_m']
                else:
                    zones.append(dict(start_m=interval['start_m'], end_m=interval['end_m'], winner=winner, tied_keys=tied))
            transitions = []
            for index, (before, after) in enumerate(zip(zones, zones[1:]), start=1):
                length = before['end_m']
                related = next((p for p in pairwise if abs(p['length_m'] - length) <= tolerance_m
                    and {p['a'], p['b']} == {before['winner'], after['winner']}), None)
                transitions.append(dict(length_m=length, from_key=before['winner'], to_key=after['winner'],
                    kind='cost_crossing' if related else 'service_change', label=f'Lcrit,{index}' if related else f'Service limit {index}',
                    cost_cad=related['cost_cad'] if related else None, reasons=[]))
            crossings = []
            for pair in pairwise:
                if pair['a'] not in subset or pair['b'] not in subset:
                    continue
                eligible = [r for r in pair['candidates'] if r['key'] in subset and r['service_feasible']]
                winner = min(eligible, key=lambda r: r['npc']) if eligible else None
                pair_served = all(next(r for r in pair['candidates'] if r['key'] == key)['service_feasible'] for key in (pair['a'], pair['b']))
                pair_feasible = all(next(r for r in pair['candidates'] if r['key'] == key)['feasible'] for key in (pair['a'], pair['b']))
                cost_tolerance = max(1e-4, abs(pair['cost_cad']) * 1e-9)
                on_envelope = pair_served and winner is not None and pair['cost_cad'] <= winner['npc'] + cost_tolerance
                transition = next((t for t in transitions if t['kind'] == 'cost_crossing' and abs(t['length_m'] - pair['length_m']) <= tolerance_m
                                   and {t['from_key'], t['to_key']} == {pair['a'], pair['b']}), None)
                classification = ('service_infeasible_crossing' if not pair_served else 'winner_change' if transition
                                  else 'optimal_tie' if on_envelope else 'dominated_crossing')
                cheaper = [r['key'] for r in eligible if r['npc'] < pair['cost_cad'] - cost_tolerance]
                crossings.append(dict(a=pair['a'], b=pair['b'], length_m=pair['length_m'], cost_cad=pair['cost_cad'],
                    classification=classification, feasible=pair_feasible, cheaper_keys=cheaper,
                    best_key=winner['key'] if winner else None, label=transition['label'] if transition else None))
            economic_maps[','.join(subset)] = dict(strategies=list(subset), zones=zones, transitions=transitions,
                                                  pairwise_crossings=crossings)
    return dict(strategy_order=list(keys), maps=maps, economic_maps=economic_maps, curves=curves,
        feasibility_boundaries=feasibility_boundaries,
        note='Zones show the cheapest feasible strategy over the tested distance range. Economic crossings and operating-limit changes are labeled separately. Pairwise crossings above a cheaper feasible alternative do not change the winner. Sampling can miss tangencies or narrow multiple crossings.')


def run_study(profile, raw_options, residual=False, progress=lambda message: None):
    options = validate_options(raw_options)
    load = np.asarray(profile, dtype=float)
    if load.ndim != 1 or not 1 <= len(load) <= 8784 or not np.isfinite(load).all() or np.min(load) < 0 or np.max(load) <= 0:
        raise ValueError('The hourly heating profile must be finite, nonnegative, and have positive demand.')
    progress('Simulating cyclic storage operation')
    cases, storage, volume = prepare_strategies(load, options)
    maximum = options['maximum_length_m']
    lengths = np.unique(np.append(np.linspace(0, maximum, 81), options['selected_length_m']))
    curves = {key: [] for key in cases}
    progress('Comparing lifecycle costs and operating limits over distance')
    for length in lengths:
        for key, case in cases.items():
            row = evaluate(case, length, options, residual)
            curves[key].append(dict(length_m=float(length), npc=row['npc'], feasible=row['feasible']))
    comparisons = []
    keys = list(cases)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            analytical = crossing_summary(*[
                evaluate(cases[k], length, options, residual, couple_losses=False)['npc']
                for k in (a, b) for length in (0, 1)])
            def difference(length):
                return evaluate(cases[a], length, options, residual)['npc'] - evaluate(cases[b], length, options, residual)['npc']
            differences = [ra['npc'] - rb['npc'] for ra, rb in zip(curves[a], curves[b])]
            roots, status = find_crossings(lengths, differences, difference)
            unique = [dict(length_m=length, feasible=all(
                evaluate(cases[k], length, options, residual)['feasible'] for k in (a, b))) for length in roots]
            comparisons.append(dict(a=a, b=b, analytical=analytical, numerical=unique,
                status=status))
    selected = [evaluate(case, options['selected_length_m'], options, residual) for case in cases.values()]
    eligible = [r for r in selected if r['feasible']]
    winner = min(eligible, key=lambda r: r['npc'])['key'] if eligible else None
    sensitivity = []
    for alpha in sorted(set([0, options['alpha'], 1])):
        for multiplier in (.5, 1, 1.5):
            varied = dict(options, alpha=alpha, electricity_price=options['electricity_price'] * multiplier)
            rows = [evaluate(c, options['selected_length_m'], varied, residual) for c in cases.values()]
            feasible = [r for r in rows if r['feasible']]
            best = min(feasible, key=lambda r: r['npc']) if feasible else None
            sensitivity.append(dict(alpha=alpha, electricity_multiplier=multiplier,
                winner=best['key'] if best else None, npc=best['npc'] if best else None))
    progress('Saving reproducible study results')
    critical_lengths = critical_length_views(cases, options, residual, lengths)
    strategy_analysis = strategy_zone_analysis(list(cases), lengths, comparisons,
        lambda key, length: evaluate(cases[key], length, options, residual))
    payload = dict(options=options, residual_credit=residual, hourly_kw=load.tolist())
    digest = sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()
    return dict(model_version='trunk-break-even-v1.3', created_at=datetime.now(timezone.utc).isoformat(),
        input_sha256=digest, inputs=payload, summary=dict(hours=len(load), annual_heat_kwh=float(sum(load)),
        peak_kw=float(max(load)), average_kw=float(np.mean(load)), load_factor=float(np.mean(load) / max(load)),
        winner=winner, selected_length_m=options['selected_length_m']),
        selected=selected, curves=curves, comparisons=comparisons, sensitivity=sensitivity,
        critical_lengths=critical_lengths,
        strategy_analysis=strategy_analysis,
        storage=dict(initial_kwh=storage['initial_kwh'], final_kwh=storage['final_kwh'],
            cyclic_error_kwh=storage['cyclic_error_kwh'], unmet_kwh=storage['unmet_kwh'],
            feasible=storage['feasible'], volume_m3=volume, state_kwh=storage['states'],
            trunk_kw=storage['profile'].tolist(), charge_kw=storage['charges'].tolist(), discharge_kw=storage['discharges'].tolist()),
        assumptions=[
            f'Installed cost diameter bands use upper-inclusive intervals of {options["diameter_band_width_mm"]} mm; zero selects continuous scaling. The cost curve is evaluated at the band upper diameter, without rounding the reference diameter. Actual diameters are retained for hydraulics. Band rates are illustrative, not quotations.',
            'Installed cost covers both equal-diameter pipe legs, trenching, ordinary fittings and installation per route metre. No extra construction allowance is added.',
            'All default costs, alpha, n and equipment coefficients are illustrative editable assumptions, not calibrated quotations. Diameters are internal diameters, not nominal DN.',
            'Real base-year CAD, constant annual tariffs and demand; end-year discounting. Maintenance, replacements and optional straight-line residual credits are included.',
            'Source heat price is per thermal kWh at the trunk inlet, including the chosen source conversion cost. Do not add source electricity again.',
            'Each strategy serves the same heating load at the baseline supply temperature. The node HP extracts Qout - Qout/COP from its independently specified lower-temperature trunk; COP is a fixed scenario assumption.',
            'A constant per-leg heat-loss coefficient keeps both pipe legs warm all year. Full-model losses increase trunk flow as well as source energy; ground and fluid properties are constant.',
            'Storage uses maximum-recharge chronological peak shaving with periodic annual SOC, efficiency, standing losses and charge/discharge limits. It is not a tariff-optimized dispatch. Delivery cap is heat arriving at the node; upstream pipe losses are additional.',
            'Hydraulic screens cover velocity and total supply/return friction, specified fitting K and terminal differential. Elevation, detailed pump curves and network interactions require subsequent design checks.',
            'Analytical crossings freeze flow against pipe heat losses, but retain their linear energy cost. Numerical crossings include loss-driven flow and pumping, and are searched on an 81-point grid with sign-change refinement; tangencies may be missed.',
            'This is a single equivalent trunk study, not a whole-network routing or equipment placement optimization. No existing network design is modified.'
        ])
