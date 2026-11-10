"""Transparent profile differences and single-feeder hydraulic screening."""
import math

import numpy as np


def percentage_difference(value, reference):
    return 100 * (value - reference) / reference if reference else None


def longest_event(mask):
    padded = np.r_[False, mask, False].astype(int)
    return int(np.max(np.flatnonzero(np.diff(padded) == -1) - np.flatnonzero(np.diff(padded) == 1), initial=0))


def calculate_profile_metrics(values, reference, individual_peak_sum, threshold_kw, cp, delta_t):
    values, reference = np.asarray(values), np.asarray(reference)
    peak, reference_peak = float(values.max()), float(reference.max())
    difference = values - reference
    duration_difference = np.sort(values) - np.sort(reference)
    upper_count = max(1, math.ceil(len(values) * .01))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    high = values > threshold_kw
    metrics = dict(
        annual_energy_kwh=float(values.sum()), peak_kw=peak,
        peak_bias_pct=percentage_difference(peak, reference_peak),
        annual_energy_bias_pct=percentage_difference(float(values.sum()), float(reference.sum())),
        p95_kw=float(np.quantile(values, .95)), p99_kw=float(np.quantile(values, .99)),
        p999_kw=float(np.quantile(values, .999)),
        coincidence_factor=peak / individual_peak_sum if individual_peak_sum else None,
        load_factor=float(values.mean()) / peak if peak else None,
        hours_above_threshold=int(high.sum()), longest_peak_event_hours=longest_event(high),
        mae_kw=float(np.mean(np.abs(difference))), rmse_kw=rmse,
        cvrmse_pct=100 * rmse / float(reference.mean()) if reference.mean() else None,
        ldc_rmse_kw=float(np.sqrt(np.mean(duration_difference ** 2))),
        upper_1pct_ldc_rmse_kw=float(np.sqrt(np.mean(duration_difference[-upper_count:] ** 2))),
        peak_mass_flow_kg_s=peak / (cp * delta_t),
        peak_flow_bias_pct=percentage_difference(peak, reference_peak),
    )
    for quantile, name in ((.95, 'p95'), (.99, 'p99'), (.999, 'p999')):
        metrics[f'{name}_bias_pct'] = percentage_difference(float(np.quantile(values, quantile)),
                                                          float(np.quantile(reference, quantile)))
    # At hourly resolution, only coarser averaging can be assessed.
    metrics['resolution_peaks_kw'] = {str(hours): float(values.reshape(-1, hours).mean(axis=1).max())
                                       for hours in (1, 3, 6, 24)}
    return metrics


def screen_equivalent_feeder(values, reference, criteria, delta_t):
    """Screen one hypothetical feeder against the SAME reference flow, not a network simulation."""
    from api.utils.pipe_sizing import HydraulicFormulas
    catalog = sorted(criteria.STD_PIPES, key=lambda p: float(p['di']))
    for attr in ('CP', 'RHO', 'MU', 'MAX_VELOCITY', 'TARGET_PRESSURE_DROP'):
        if not math.isfinite(getattr(criteria, attr)) or getattr(criteria, attr) <= 0:
            raise ValueError(f'Invalid network design constant: {attr}')
    if not catalog or any(not math.isfinite(float(p['di'])) or float(p['di']) <= 0 for p in catalog):
        raise ValueError('Pipe catalog must contain positive internal diameters.')

    def hydraulic_state(load, pipe):
        mass_flow = load / (criteria.CP * delta_t)
        velocity = mass_flow / (criteria.RHO * math.pi * float(pipe['di']) ** 2 / 4)
        if velocity <= 0:
            return 0., 0.
        reynolds = HydraulicFormulas.calculate_reynolds(velocity, float(pipe['di']), criteria)
        friction = HydraulicFormulas.calculate_friction_factor(reynolds, float(pipe['di']),
                                                                float(pipe.get('roughness', .000045)))
        return velocity, friction * criteria.RHO * velocity ** 2 / (2 * float(pipe['di']))

    selected = None
    for pipe in catalog:
        velocity, gradient = hydraulic_state(float(max(values)), pipe)
        if velocity <= criteria.MAX_VELOCITY and gradient <= criteria.TARGET_PRESSURE_DROP:
            selected = pipe
            break
    if selected is None:
        return dict(status='No catalog pipe meets both constraints', dn=None,
                    reference_violation_hours=None, catalog_cost_cad_per_m=None)
    violations = sum(v > criteria.MAX_VELOCITY or p > criteria.TARGET_PRESSURE_DROP
                     for v, p in (hydraulic_state(float(load), selected) for load in reference))
    return dict(status='Equivalent feeder only', dn=selected['dn'], internal_diameter_m=selected['di'],
                reference_violation_hours=violations, catalog_cost_cad_per_m=selected.get('cost'),
                peak_velocity_m_s=velocity, peak_pressure_gradient_pa_m=gradient)
