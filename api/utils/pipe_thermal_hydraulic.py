"""
pipe_thermal_hydraulic.py

A small Python library for 1D pipe thermal + hydraulic calculations.

Features
- Given: pipe length L, node load profile (W as array), velocity (m/s), temperature delta across the node (K), supply temperature (C), ambient temperature (C), insulation/pipe thermal properties
- Computes: thermal energy delivered, thermal energy lost along the pipe (to ambient), required mass & volumetric flow, pipe diameter to achieve given velocity, Reynolds number, friction factor, pressure drop (Darcy-Weisbach), and pump power required.

Dependencies: numpy

Usage summary (see functions below for details):
- process a time-series load (array) with timestep dt (seconds)
- call compute_branch_requirements(...) to run the end-to-end calculation and get results per timestep and summaries

"""

from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict, Any

# Physical constants for water (defaults, can be overridden)
DEFAULT_RHO = 997.0         # kg/m^3
DEFAULT_CP = 4184.0         # J/(kg*K)
DEFAULT_MU = 1.0e-3         # Pa.s (dynamic viscosity)
#

def compute_mass_flow_from_heat(heat_w: float, delta_t: float, cp: float = DEFAULT_CP) -> float:
    """Return required mass flow (kg/s) to carry heat_w [W] with temperature difference delta_t [K].

    If heat_w is zero, returns 0. Avoids division by zero by requiring delta_t>0.
    """
    if delta_t <= 0:
        raise ValueError("delta_t must be > 0 K")
    return heat_w / (cp * delta_t)


def diameter_from_massflow_and_velocity(mdot: float, v: float, rho: float = DEFAULT_RHO) -> float:
    """Return internal diameter [m] required to achieve mass flow mdot [kg/s] at mean velocity v [m/s].

    If mdot==0 returns 0.
    """
    if mdot <= 0:
        return 0.0
    q = mdot / rho
    area = q / v
    if area <= 0:
        raise ValueError("Computed cross-sectional area <=0")
    return np.sqrt(4 * area / np.pi)


def reynolds_number(rho: float, v: float, D: float, mu: float = DEFAULT_MU) -> float:
    if D <= 0 or v <= 0:
        return 0.0
    return rho * v * D / mu


def swamee_jain_friction_factor(Re: float, D: float, eps: float) -> float:
    """Swamee-Jain explicit approximation for turbulent friction factor.
    eps = absolute roughness [m]
    valid for Re>3000 typically. Returns f (Darcy).
    """
    if Re <= 0:
        raise ValueError("Re must be positive for friction factor calculation")
    if Re < 2300:
        return 64.0 / Re
    # avoid division by zero in log
    A = (eps / (3.7 * D))
    B = 5.74 / (Re ** 0.9)
    denom = np.log10(A + B)
    f = 0.25 / (denom ** 2)
    return float(f)


def pressure_drop_darcy(f: float, L: float, D: float, rho: float, v: float) -> float:
    """Return pressure drop (Pa) over length L [m] for pipe diameter D [m], velocity v [m/s], density rho and Darcy friction factor f.
    DeltaP = f * (L/D) * 0.5 * rho * v^2
    """
    if D <= 0 or L <= 0 or v <= 0:
        return 0.0
    return f * (L / D) * 0.5 * rho * v * v


def pump_power_required(delta_p: float, q_vol: float, pump_efficiency: float = 0.7) -> float:
    """Return shaft power [W] required by pump to overcome delta_p [Pa] at volumetric flow q_vol [m^3/s].
    Accounts for pump efficiency (0<efficiency<=1)."""
    if pump_efficiency <= 0 or pump_efficiency > 1:
        raise ValueError("pump_efficiency must be in (0,1]")
    if delta_p <= 0 or q_vol <= 0:
        return 0.0
    shaft = delta_p * q_vol
    return shaft / pump_efficiency


def thermal_loss_per_length(U: float, perimeter: float, T_mean: float, T_amb: float) -> float:
    """Return heat loss per unit length [W/m] = U [W/m2K] * perimeter [m] * (T_mean - T_amb)"""
    return U * perimeter * (T_mean - T_amb)


@dataclass
class PipeSpec:
    length: float                  # m
    roughness: float = 1.5e-6      # m (default: commercial steel ~1.5e-6) 
    insulation_U: Optional[float] = None  # overall U [W/m2K]. If None, a simple default is used (uninsulated)
    outer_diameter: Optional[float] = None  # m. If not given, calculated from ID + typical thickness not considered.


def default_uninsulated_U(D_internal: float) -> float:
    """Return a rough default U [W/m2K] for an uninsulated small pipe (very approximate).
    This is a very rough number and should be overridden for engineering calculations.
    """
    # Uninsulated pipe convective coefficient plus conduction through thin wall is roughly 10-40 W/m2K.
    # We'll pick 20 W/m2K as a mid value for small pipe in air.
    return 20.0


def compute_branch_requirements(
    load_profile: np.ndarray,
    dt: float,
    delta_t: float,
    v: float,
    supply_temp_c: float,
    pipe: PipeSpec,
    rho: float = DEFAULT_RHO,
    cp: float = DEFAULT_CP,
    mu: float = DEFAULT_MU,
    pump_efficiency: float = 0.7,
) -> Dict[str, Any]:
    """
    End-to-end calculation for a pipe branch supplying a node with a given time-series heat demand.

    Parameters
    - load_profile: array-like of demande heat at the node [W] for each timestep
    - dt: timestep in seconds
    - delta_t: temperature drop across the node (supply - return) in K
    - v: assumed mean velocity [m/s] inside the branch (used to compute ID)
    - supply_temp_c: supply temperature [C]
    - pipe: PipeSpec object
    - rho, cp, mu: fluid properties
    - pump_efficiency: pump mechanical efficiency

    Returns a dict with keys:
    - mdot_profile [kg/s]
    - diam_profile [m]
    - Re_profile
    - f_profile
    - delta_p_profile [Pa]
    - pump_power_profile [W]
    - heat_loss_profile [W] (total heat lost along the branch to ambient)
    - heat_delivered_profile [W] (what reaches the node = load_profile)
    - summary: aggregated scalars (peak values, totals)
    """
    loads = np.asarray(load_profile, dtype=float)
    if loads.ndim != 1:
        raise ValueError("load_profile must be a 1D array of heat demand (W)")
    n = loads.size

    # T_mean (approximate mean fluid temperature in the branch)
    T_supply = supply_temp_c
    T_return = supply_temp_c - delta_t
    T_mean = (T_supply + T_return) / 2.0

    # Determine U (overall heat transfer coeff). If pipe.insulation_U provided use it.
    # Need perimeter: for internal diameter we compute later; use outer diameter ~ internal for perimeter.

    mdot_profile = np.zeros(n)
    diam_profile = np.zeros(n)
    Re_profile = np.zeros(n)
    f_profile = np.zeros(n)
    delta_p_profile = np.zeros(n)
    pump_power_profile = np.zeros(n)
    heat_loss_profile = np.zeros(n)

    for i in range(n):
        load = loads[i]
        # compute required mass flow to carry the demanded load ignoring losses first
        mdot = compute_mass_flow_from_heat(load, delta_t, cp)

        # estimate heat loss per length using U and perimeter
        # compute instantaneous diameter needed to carry mdot at velocity v
        D = diameter_from_massflow_and_velocity(mdot, v, rho)
        if D <= 0:
            # zero flow case
            mdot_profile[i] = 0.0
            diam_profile[i] = 0.0
            Re_profile[i] = 0.0
            f_profile[i] = 0.0
            delta_p_profile[i] = 0.0
            pump_power_profile[i] = 0.0
            heat_loss_profile[i] = 0.0
            continue

        perimeter = np.pi * D  # approximate wetted perimeter (outer ~ inner for thin-walled tube)
        U = pipe.insulation_U if pipe.insulation_U is not None else default_uninsulated_U(D)

        qloss_per_m = thermal_loss_per_length(U, perimeter, T_mean, T_supply - (delta_t/2.0))
        # note: we used T_mean and ambient roughly supply-(dT/2) as local, but user can supply actual ambient
        # total loss along branch
        qloss_total = qloss_per_m * pipe.length

        # The demanded load is what must be delivered to node; to supply that we must send extra to cover losses
        mdot_with_losses = compute_mass_flow_from_heat(load + qloss_total, delta_t, cp)

        # recompute diameter for the increased mdot
        D = diameter_from_massflow_and_velocity(mdot_with_losses, v, rho)
        Re = reynolds_number(rho, v, D, mu)
        f = swamee_jain_friction_factor(Re, D, pipe.roughness) if Re > 0 else 0.0
        delta_p = pressure_drop_darcy(f, pipe.length, D, rho, v)
        q_vol = mdot_with_losses / rhod
        pump_power = pump_power_required(delta_p, q_vol, pump_efficiency)

        # store
        mdot_profile[i] = mdot_with_losses
        diam_profile[i] = D
        Re_profile[i] = Re
        f_profile[i] = f
        delta_p_profile[i] = delta_p
        pump_power_profile[i] = pump_power
        heat_loss_profile[i] = qloss_total

    summary = {
        'mdot_peak': float(np.max(mdot_profile)),
        'diam_peak_m': float(np.max(diam_profile)),
        'delta_p_peak_Pa': float(np.max(delta_p_profile)),
        'pump_power_peak_W': float(np.max(pump_power_profile)),
        'total_energy_delivered_J': float(np.sum(loads) * dt),
        'total_energy_loss_J': float(np.sum(heat_loss_profile) * dt),
    }

    return {
        'mdot_profile': mdot_profile,
        'diam_profile': diam_profile,
        'Re_profile': Re_profile,
        'f_profile': f_profile,
        'delta_p_profile': delta_p_profile,
        'pump_power_profile': pump_power_profile,
        'heat_loss_profile': heat_loss_profile,
        'heat_delivered_profile': loads.copy(),
        'summary': summary,
    }


# End of file
