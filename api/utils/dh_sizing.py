# dh_sizing.py
# District heating pipe sizing from load profiles with Peak, Diversity,
# LP-style coverage, and discrete MINLP diameter selection.
#
# Units:
# - Loads (profiles): kW
# - Temperature: K or °C differences
# - Flow: m^3/s, kg/s
# - Diameter: m
# - Length: m

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
import math


@dataclass
class FluidProps:
    rho: float = 997.0           # kg/m^3 (water near ~20–60°C)
    cp: float = 4180.0           # J/(kg·K)
    mu: float = 0.0006           # Pa·s (dynamic viscosity; ~60°C water)
    # You can override for other temps/fluids.


@dataclass
class PipeParams:
    length_m: float = 100.0      # Pipe length considered for sizing
    eps_m: float = 1.5e-5        # Absolute roughness (m) (e.g., steel ~1.5e-5 m)
    vmax: float = 2.0            # Max velocity (m/s)
    headloss_max_Pa_per_m: float = 400.0  # Max allowable pressure drop per meter (Pa/m)
    pump_eff: float = 0.65       # Pump wire-to-water efficiency (fraction)
    # Catalog of inner diameters (m); sample set — replace with your standard set
    diameter_catalog_m: Tuple[float, ...] = (
        0.025, 0.032, 0.040, 0.050, 0.065, 0.080, 0.100, 0.125, 0.150, 0.200,
        0.250, 0.300, 0.350, 0.400
    )


def aggregate_load_kw(load_profiles_kw: np.ndarray) -> np.ndarray:
    """
    Sum concurrent loads across all nodes for each timestep.
    load_profiles_kw: shape (T, N) in kW
    returns: shape (T,) in kW
    """
    if load_profiles_kw.ndim != 2:
        raise ValueError("load_profiles_kw must be 2D (T, N).")
    return np.sum(load_profiles_kw, axis=1)


def design_load_peak_kw(load_profiles_kw: np.ndarray) -> float:
    """
    Network peak coincident load from time series (kW).
    """
    return float(np.max(aggregate_load_kw(load_profiles_kw)))


def design_load_diversity_kw(
    load_profiles_kw: np.ndarray,
    diversity_factor: Optional[float] = None
) -> float:
    """
    Diversity-based design load (kW).
    - If diversity_factor is provided: DF * sum of individual peaks.
    - Else: falls back to coincident peak from time series (same as peak).
    """
    if diversity_factor is not None:
        indiv_peaks_kw = np.max(load_profiles_kw, axis=0)  # per-node peaks
        return float(np.sum(indiv_peaks_kw) * diversity_factor)
    # No DF provided: use coincident peak from measured/simulated profiles
    return design_load_peak_kw(load_profiles_kw)


def design_load_lp_kw(
    load_profiles_kw: np.ndarray,
    coverage: float = 0.95
) -> float:
    """
    LP-style coverage design load (kW).
    Interpretable as the minimum capacity H that can 'cover' at least a
    fraction 'coverage' of timesteps. This equals the coverage-quantile
    of the coincident load time series.
    """
    if not (0.0 < coverage <= 1.0):
        raise ValueError("coverage must be in (0, 1].")
    agg_kw = aggregate_load_kw(load_profiles_kw)
    # p-quantile (e.g., 95th percentile for coverage=0.95)
    p = 100.0 * coverage
    return float(np.percentile(agg_kw, p))


def heat_to_flow(
    design_load_kw: float,
    deltaT_K: float,
    fluid: FluidProps
) -> Dict[str, float]:
    """
    Convert design heat load to mass and volumetric flow.
    returns dict with keys: Q_W, m_dot_kg_s, q_m3_s
    """
    Q_W = design_load_kw * 1e3
    m_dot = Q_W / (fluid.cp * deltaT_K)
    q = m_dot / fluid.rho
    return {"Q_W": Q_W, "m_dot_kg_s": m_dot, "q_m3_s": q}


def diameter_from_velocity(q_m3_s: float, V_m_s: float) -> float:
    """
    Hydraulic diameter from volumetric flow and target mean velocity.
    D = sqrt(4 Q / (pi V))
    """
    if V_m_s <= 0:
        raise ValueError("Velocity must be positive.")
    return math.sqrt(4.0 * q_m3_s / (math.pi * V_m_s))


def reynolds_number(D_m: float, V_m_s: float, fluid: FluidProps) -> float:
    """
    Re = (rho * V * D) / mu
    """
    return (fluid.rho * V_m_s * D_m) / fluid.mu


def friction_factor_haaland(Re: float, rel_rough: float) -> float:
    """
    Haaland explicit approximation for Darcy friction factor (turbulent).
    1/sqrt(f) = -1.8 log10[ ( (eps/D)/3.7 )^1.11 + 6.9/Re ]
    """
    if Re <= 0:
        return 0.0
    term = ((rel_rough / 3.7) ** 1.11) + (6.9 / Re)
    inv_sqrt_f = -1.8 * math.log10(term)
    return 1.0 / (inv_sqrt_f ** 2)


def headloss_Pa_per_m(D_m: float, V_m_s: float, fluid: FluidProps, eps_m: float) -> float:
    """
    Darcy–Weisbach: dp/dx = f * (rho * V^2 / (2D))
    Convert to Pa/m.
    """
    Re = reynolds_number(D_m, V_m_s, fluid)
    rel_rough = eps_m / D_m if D_m > 0 else 0.0
    f = friction_factor_haaland(Re, rel_rough)
    return f * (fluid.rho * V_m_s * V_m_s) / (2.0 * D_m)


def pumping_power_W(
    q_m3_s: float,
    D_m: float,
    pipe: PipeParams,
    fluid: FluidProps
) -> float:
    """
    Approximate pump power for the given pipe segment:
    P = (Δp * Q) / η, with Δp = (dp/dx) * L.
    """
    # Compute velocity from Q and D
    area = math.pi * (D_m ** 2) / 4.0
    V = q_m3_s / area
    dpdx = headloss_Pa_per_m(D_m, V, fluid, pipe.eps_m)
    delta_p = dpdx * pipe.length_m
    return (delta_p * q_m3_s) / max(1e-6, pipe.pump_eff)


def default_pipe_cost_per_m(D_m: float) -> float:
    """
    Simple convex surrogate CAPEX cost per meter for pipe (inner diameter-based proxy).
    You should replace with your project-specific cost curve.
    """
    # Example: c(D) = a*D^1.2 + b, with a,b tuned to your market
    a = 1500.0
    b = 50.0
    return a * (D_m ** 1.2) + b


def select_diameter_minlp(
    q_m3_s: float,
    pipe: PipeParams,
    fluid: FluidProps,
    cost_per_m: Callable[[float], float] = default_pipe_cost_per_m,
    pump_energy_price_per_Wh: float = 0.00015,  # ~0.15 $/kWh
    annual_operating_hours: float = 2500.0
) -> Dict[str, float]:
    """
    Mixed-integer nonlinear selection: choose D from a discrete catalog that:
    - Satisfies velocity and headloss constraints.
    - Minimizes [CAPEX (pipe) + OPEX (pumping)] for this segment.

    Returns dict with best 'D_m' and breakdown of objective.
    """
    best = None
    for D in pipe.diameter_catalog_m:
        area = math.pi * (D ** 2) / 4.0
        V = q_m3_s / max(1e-12, area)

        # Velocity and headloss checks
        if V > pipe.vmax:
            continue

        dpdx = headloss_Pa_per_m(D, V, fluid, pipe.eps_m)
        if dpdx > pipe.headloss_max_Pa_per_m:
            continue

        # Costs
        capex = cost_per_m(D) * pipe.length_m
        P_pump = pumping_power_W(q_m3_s, D, pipe, fluid)
        annual_energy_Wh = P_pump * annual_operating_hours  # W·h
        opex = pump_energy_price_per_Wh * annual_energy_Wh

        total_cost = capex + opex

        if (best is None) or (total_cost < best["total_cost"]):
            best = {
                "D_m": D,
                "V_m_s": V,
                "dpdx_Pa_per_m": dpdx,
                "capex": capex,
                "pumping_power_W": P_pump,
                "annual_opex_energy_Wh": annual_energy_Wh,
                "annual_opex_cost": opex,
                "total_cost": total_cost,
            }

    if best is None:
        raise RuntimeError(
            "No feasible diameter in catalog meets velocity/headloss constraints. "
            "Adjust vmax/headloss limits or expand the catalog."
        )
    return best


def diameter_from_design_load(
    design_load_kw: float,
    deltaT_K: float,
    fluid: FluidProps,
    target_velocity_m_s: float
) -> Dict[str, float]:
    """
    Convenience: convert design load to flow and diameter given a target velocity.
    """
    flows = heat_to_flow(design_load_kw, deltaT_K, fluid)
    D = diameter_from_velocity(flows["q_m3_s"], target_velocity_m_s)
    return {"D_m": D, **flows}

def heat_loss_W(
    D_inner_m: float,
    insulation_thickness_m: float,
    U_W_m2K: float,
    length_m: float,
    T_fluid_C: float,
    T_ambient_C: float
) -> float:
    """
    Estimate total steady-state heat loss from a pipe segment.

    Parameters
    ----------
    D_inner_m : float
        Inner pipe diameter (m) — hydraulic diameter.
    insulation_thickness_m : float
        Thickness of insulation layer (m).
    U_W_m2K : float
        Overall heat transfer coefficient (W/m²·K).
        This is based on pipe material, insulation conductivity, burial depth, etc.
    length_m : float
        Length of the pipe segment (m).
    T_fluid_C : float
        Average fluid temperature (°C) in the pipe.
    T_ambient_C : float
        Ambient temperature outside the insulation (°C) — ground or air.

    Returns
    -------
    float
        Estimated total heat loss for the segment (W).
    """
    # Outer diameter including insulation
    D_outer_m = D_inner_m + 2.0 * insulation_thickness_m

    # Heat loss per unit length (W/m)
    q_per_m = U_W_m2K * math.pi * D_outer_m * (T_fluid_C - T_ambient_C)

    # Total segment heat loss
    return q_per_m * length_m

# -------- Example usage --------
if __name__ == "__main__":
    # Synthetic example: 8760 hours, 4 buildings (kW)
    rng = np.random.default_rng(42)
    T = 8760
    N = 4
    base_kw = np.array([150, 120, 80, 60])
    variability = rng.normal(0.0, 0.25, size=(T, N))  # ±25% noise
    load_profiles_kw = np.clip(base_kw * (1.0 + variability), 0.0, None)

    fluid = FluidProps(rho=985.0, cp=4180.0, mu=0.0005)  # ~70°C water
    pipe = PipeParams(length_m=200.0, vmax=1.8, headloss_max_Pa_per_m=300.0)

    deltaT = 20.0  # K

    # 1) Peak coincident load
    peak_kw = design_load_peak_kw(load_profiles_kw)

    # 2) Diversity-factor-based: DF * sum(individual peaks)
    df = 0.6
    div_kw = design_load_diversity_kw(load_profiles_kw, diversity_factor=df)

    # 3) LP-style coverage (e.g., cover 97.5% of hours)
    lp_kw = design_load_lp_kw(load_profiles_kw, coverage=0.975)

    # Convert each design load to a diameter at chosen velocity
    V_target = 1.5  # m/s
    peak_result = diameter_from_design_load(peak_kw, deltaT, fluid, V_target)
    div_result = diameter_from_design_load(div_kw, deltaT, fluid, V_target)
    lp_result = diameter_from_design_load(lp_kw, deltaT, fluid, V_target)

    print("Peak design (kW):", round(peak_kw, 1), "Diameter (m):", round(peak_result["D_m"], 4))
    print("Diversity design (kW):", round(div_kw, 1), "Diameter (m):", round(div_result["D_m"], 4))
    print("LP coverage design (kW):", round(lp_kw, 1), "Diameter (m):", round(lp_result["D_m"], 4))

    # 4) MINLP selection from catalog for the LP design flow
    q_lp = lp_result["q_m3_s"]
    best = select_diameter_minlp(q_lp, pipe, fluid)
    print("\nMINLP pick from catalog:")
    for k, v in best.items():
        print(f"  {k}: {v}")