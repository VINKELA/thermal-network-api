"""Forward THERMO-derived thermal circuit; no trained encoder is supplied.

Banerjee (2026), Fig. 3.1, Eqs. 3.2.1–3.2.2 and Table 3.2.
Use energy-conserving capacitor signs and gate the complete heater branch.
These documented interpretations resolve ambiguities in printed Eq. 3.2.1a/d.
"""
from math import exp
import numpy as np

THESIS_SOURCE = 'Ahijit Banerjee (2026), McMaster MASc thesis, Chapter 3, Table 3.2; supplied Ahijit_Thesis.pdf'
THERMO_PARAMETERS = {
    'detached': dict(r_ext=1.5, r_we=1.5, r_wi=1.4, r_1=2., r_2=.1, r_3=1., r_supply=.4, c_we=6., c_wi=2., alpha=.8, beta=1., deadband=1.5),
    'attached': dict(r_ext=3.6, r_we=2.5, r_wi=2.8, r_1=3.8, r_2=.1, r_3=2.8, r_supply=1.4, c_we=3., c_wi=.5, alpha=1., beta=1., deadband=1.6),
}


def sigmoid(value):
    return 1. / (1. + exp(-max(-60., min(60., value))))


def thermal_circuit_rates(state, outdoor_c, setpoint_c, parameters):
    """Return dTe/dh, dTi/dh, du/dh and heater kW; R in K/kW, C in kWh/K."""
    exterior, interior, control = state
    p = parameters
    control = min(1., max(0., control))
    ge = 1. / (p['r_we'] + p['r_1'])
    gi = 1. / (p['r_2'] + p['r_wi'])
    gh = control / (p['r_supply'] + control * p['r_3'])
    junction = (ge * exterior + gi * interior + gh * 60.) / (ge + gi + gh)
    heat_kw = max(0., gh * (60. - junction))
    air = junction + heat_kw * p['r_3']
    error = air - setpoint_c
    actuator_rate = p['alpha'] * (sigmoid(-p['beta'] * (error + p['deadband'])) * (1 - control)
                                 - sigmoid(p['beta'] * (error - p['deadband'])) * control)
    return ((outdoor_c - exterior) / p['r_ext'] + ge * (junction - exterior)) / p['c_we'], gi * (junction - interior) / p['c_wi'], actuator_rate, heat_kw


def integrate_thermo_hour(state, outdoor_c, setpoint_c, parameters, substeps=24):
    """RK4 with 2.5-minute internal steps and an integrated hourly mean heat output."""
    step = 1. / substeps
    energy = 0.
    for _ in range(substeps):
        k1 = thermal_circuit_rates(state, outdoor_c, setpoint_c, parameters)
        k2 = thermal_circuit_rates(tuple(state[j] + step * k1[j] / 2 for j in range(3)), outdoor_c, setpoint_c, parameters)
        k3 = thermal_circuit_rates(tuple(state[j] + step * k2[j] / 2 for j in range(3)), outdoor_c, setpoint_c, parameters)
        k4 = thermal_circuit_rates(tuple(state[j] + step * k3[j] for j in range(3)), outdoor_c, setpoint_c, parameters)
        state = tuple(state[j] + step * (k1[j] + 2*k2[j] + 2*k3[j] + k4[j]) / 6 for j in range(3))
        state = (state[0], state[1], max(0., min(1., state[2])))
        energy += step * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3]) / 6
    return state, energy


def generate_thermo_shape(temperatures, setpoints, archetype='detached', substeps=24):
    """Published median-house shape, subsequently scaled to each building's annual heat.

No area scaling of R/C is claimed. Commercial use is residential-shape extrapolation.
The repeated first week is a 14-day numerical warm-up, not additional weather data.
"""
    parameters = THERMO_PARAMETERS[archetype]
    state = (float(setpoints[0]), float(setpoints[0]), 0.)
    for hour in range(14 * 24):
        index = hour % min(168, len(temperatures))
        state, _ = integrate_thermo_hour(state, temperatures[index], setpoints[index], parameters, substeps)
    result = np.empty(len(temperatures))
    for index, (outdoor, setpoint) in enumerate(zip(temperatures, setpoints)):
        state, result[index] = integrate_thermo_hour(state, float(outdoor), float(setpoint), parameters, substeps)
    if not np.isfinite(result).all() or np.any(result < 0):
        raise ValueError('THERMO integration failed: check circuit parameters.')
    return result
