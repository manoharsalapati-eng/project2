"""Non-linear lumped model of an extrudate conveyor (belt) dryer.

States  x = [X, Tp, Ta, Y, B]
    X   core moisture            [kg water / kg dry solids]
    Tp  product temperature      [degC]
    Ta  exhaust (chamber) air T  [degC]
    Y   exhaust absolute humidity [kg water / kg dry air]
    B   Maillard browning index  [BI]
Inputs  u = [Tin, G]  inlet air temperature [degC], dry-air flow [kg/s]
Disturbances d = Ambient(Xin, Yamb, Tamb, Fs)

Every function is vectorised: leading array dimensions are batch dimensions, so a
whole set of trajectories (MPC sensitivities, EKF Jacobians, the three plant twins)
is integrated in one call.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

P_ATM = 101325.0
CP_A, CP_V, CP_W, CP_S = 1.006, 1.88, 4.18, 1.6   # kJ/(kg K)
LAM0 = 2501.0                                      # kJ/kg latent heat at 0 degC
RGAS = 8.314
TS = 5.0                                           # control / estimator sample time [s]


@dataclass(frozen=True)
class Params:
    Fs: float = 0.25       # dry-solids throughput from the extruder [kg/s]
    Ms: float = 150.0      # dry-solids holdup on the belt [kg]  (tau = 600 s)
    Ma: float = 120.0      # dry-air holdup in the drying chamber [kg]
    Tfeed: float = 45.0    # extrudate temperature entering the dryer [degC]
    hA0: float = 22.0      # convective conductance at G0 [kW/K]; hA = hA0 sqrt(G/G0)
    G0: float = 5.0
    XD: float = 0.05       # internal-diffusion (falling-rate) moisture constant [kg/kg]
    UA: float = 0.25       # wall heat loss [kW/K]
    oswA: float = 0.12     # Oswin sorption isotherm
    oswN: float = 0.35
    kB: float = 0.6        # Maillard browning kinetics
    EaB: float = 110000.0
    TrefB: float = 80.0
    awPk: float = 0.65
    awW: float = 0.25
    Bin: float = 10.0
    fan: float = 0.12      # fan power coefficient [kW/(kg/s)^3]
    eta: float = 0.92      # heater efficiency


@dataclass(frozen=True)
class Limits:
    TinMin: float = 60.0
    TinMax: float = 200.0
    Gmin: float = 2.0
    Gmax: float = 8.0
    dTin: float = 5.0      # slew limit per 5 s step
    dG: float = 0.25


NOMINAL = Params()
LIMITS = Limits()


class Ambient(NamedTuple):
    """Disturbance / exogenous inputs; floats or arrays broadcastable to the batch."""
    Xin: float | np.ndarray
    Yamb: float | np.ndarray
    Tamb: float | np.ndarray
    Fs: float | np.ndarray


NOMINAL_D = Ambient(0.40, 0.0080, 20.0, NOMINAL.Fs)


def psat(T):
    return 610.94 * np.exp(17.625 * T / (T + 243.04))


def dln_psat(T):
    return 17.625 * 243.04 / (T + 243.04) ** 2


def humidity_from_rh(T, RH):
    pv = RH * psat(T)
    return 0.622 * pv / (P_ATM - pv)


def rh_from_humidity(T, Y):
    return (Y * P_ATM / (0.622 + Y)) / psat(T)


def to_wb(X):
    return 100.0 * X / (1.0 + X)


def water_activity(X, p: Params):
    r = (np.maximum(X, 1e-4) / p.oswA) ** (1.0 / p.oswN)
    return r / (1.0 + r)


class _Derived(NamedTuple):
    hA_coef: float
    inv_oswA: float
    inv_oswN: float
    inv_Ms: float
    inv_Ma: float
    neg_Ea_R: float
    inv_Tref: float


_DERIVED: dict[Params, _Derived] = {}


def _derived(p: Params) -> _Derived:
    c = _DERIVED.get(p)
    if c is None:
        c = _DERIVED[p] = _Derived(p.hA0 / np.sqrt(p.G0), 1.0 / p.oswA, 1.0 / p.oswN, 1.0 / p.Ms, 1.0 / p.Ma,
                                   -p.EaB / RGAS, 1.0 / (p.TrefB + 273.15))
    return c


def rates(x, u, d: Ambient, p: Params):
    """dx/dt for the five-state dryer model (shape (..., 5))."""
    c = _derived(p)
    X = np.maximum(x[..., 0], 1e-3)
    Tp, Ta = x[..., 1], x[..., 2]
    Y = np.maximum(x[..., 3], 1e-5)
    B = x[..., 4]
    Tin, G = u[..., 0], u[..., 1]
    hA = c.hA_coef * np.sqrt(G)                       # hA = hA0 sqrt(G/G0)
    cpH = CP_A + CP_V * Y
    r = (X * c.inv_oswA) ** c.inv_oswN                # Oswin isotherm, solved for water activity
    a = r / (1.0 + r)
    pv = a * psat(Tp)
    Ys = 0.622 * pv / np.maximum(P_ATM - pv, 1000.0)  # surface equilibrium humidity
    Rd = (hA / cpH) * (X / (X + p.XD)) * (Ys - Y)     # drying rate [kg/s], k_yA = hA / cpH (Lewis)
    dT = Ta - Tp
    Fs, Xin = d.Fs, d.Xin
    out = np.empty(np.broadcast_shapes(X.shape, G.shape, np.shape(Xin), np.shape(Fs), np.shape(d.Yamb)) + (5,))
    out[..., 0] = (Fs * (Xin - X) - Rd) * c.inv_Ms
    out[..., 1] = (Fs * (CP_S + CP_W * Xin) * (p.Tfeed - Tp) + hA * dT - Rd * (LAM0 - (CP_W - CP_V) * Tp)) / (p.Ms * (CP_S + CP_W * X))
    out[..., 2] = (G * (CP_A + CP_V * d.Yamb) * (Tin - Ta) - (hA + Rd * CP_V) * dT - p.UA * (Ta - d.Tamb)) * c.inv_Ma / cpH
    out[..., 3] = (G * (d.Yamb - Y) + Rd) * c.inv_Ma
    kT = p.kB * np.exp(c.neg_Ea_R * (1.0 / (Tp + 273.15) - c.inv_Tref))
    out[..., 4] = (Fs * c.inv_Ms) * (p.Bin - B) + kT * np.exp(-((a - p.awPk) / p.awW) ** 2)
    return out


def _clip_states(y):
    y[..., 0] = np.maximum(y[..., 0], 1e-3)
    y[..., 3] = np.maximum(y[..., 3], 1e-5)
    y[..., 4] = np.maximum(y[..., 4], 0.0)
    return y


def rk4(x, u, d: Ambient, p: Params, h: float, n: int):
    """Classical RK4, n steps of size h (used for the true plants and the EKF)."""
    y = np.array(x, dtype=float)[..., :5]
    for _ in range(n):
        k1 = rates(y, u, d, p)
        k2 = rates(y + 0.5 * h * k1, u, d, p)
        k3 = rates(y + 0.5 * h * k2, u, d, p)
        k4 = rates(y + h * k3, u, d, p)
        y = _clip_states(y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))
    return y


# ---- L-stable 2nd-order Rosenbrock (ROS2, W-method) for the stiff air/evaporation modes ----
ROS_G = 1.0 + 1.0 / np.sqrt(2.0)
JAC_EPS = np.array([1e-5, 1e-3, 1e-3, 1e-6, 1e-3])
_EYE5 = np.eye(5)
_DIAG_EPS = np.diag(JAC_EPS)


def _expand(d: Ambient) -> Ambient:
    """Add an axis so batched disturbances broadcast against the Jacobian perturbation axis."""
    return Ambient(*(np.asarray(v)[..., None] if np.ndim(v) else v for v in d))


def ros2_matrix(y, u, d: Ambient, p: Params, h: float):
    """Return (W^-1, f(y)) with W = I - g h J; J by forward differences, all perturbations in one batch."""
    f0 = rates(y, u, d, p)
    if y.shape != f0.shape:
        y = np.broadcast_to(y, f0.shape)
    fp = rates(y[..., None, :] + _DIAG_EPS, u[..., None, :], _expand(d), p)      # row j: f(y + eps_j e_j)
    J = np.swapaxes((fp - f0[..., None, :]) / JAC_EPS[:, None], -1, -2)          # J[i, j] = dfi/dxj
    return np.linalg.inv(_EYE5 - ROS_G * h * J), f0


def ros2_step(y, u, d: Ambient, p: Params, h: float, Winv, f0=None):
    """One ROS2 step with a given (possibly frozen) W^-1."""
    if f0 is None:
        f0 = rates(y, u, d, p)
    if y.shape != f0.shape:
        y = np.broadcast_to(y, f0.shape)
    k1 = (Winv @ f0[..., None])[..., 0]
    f1 = rates(y + h * k1, u, d, p)
    k2 = (Winv @ (f1 - 2.0 * k1)[..., None])[..., 0]
    return _clip_states(y + 1.5 * h * k1 + 0.5 * h * k2)


def ros2(x, u, d: Ambient, p: Params, h: float, n: int):
    """ROS2 keeps 2nd order with an approximate Jacobian (W-method), so W is built once per call."""
    y = np.array(x, dtype=float)[..., :5]
    Winv, f0 = ros2_matrix(y, u, d, p, h)
    for _ in range(n):
        y = ros2_step(y, u, d, p, h, Winv, f0)
        f0 = None
    return y


def power(u, d: Ambient, p: Params):
    """Heater duty, fan power and their sum [kW]."""
    Q = u[..., 1] * (CP_A + CP_V * d.Yamb) * (u[..., 0] - d.Tamb) / p.eta
    F = p.fan * u[..., 1] ** 3
    return Q, F, Q + F


def steady_state(u, p: Params = NOMINAL, d: Ambient = NOMINAL_D, x0=None):
    """Steady state(s) for input(s) u, by marching the L-stable ROS2 with large steps."""
    x = np.array([0.0989, 49.63, 57.97, 0.0231, 12.9] if x0 is None else x0, dtype=float)
    u = np.asarray(u, dtype=float)
    x = np.broadcast_to(x, u.shape[:-1] + (5,)).copy()
    for _ in range(80):
        x = ros2(x, u, d, p, 60.0, 1)
    return x


def recipe_tin(G: float, sp_wb: float, p: Params = NOMINAL) -> float:
    """Inlet temperature that hits the moisture set-point at steady state for a fixed airflow."""
    lo, hi = LIMITS.TinMin, LIMITS.TinMax
    for _ in range(2):                      # two vectorised grid refinements instead of a bisection loop
        grid = np.linspace(lo, hi, 48)
        u = np.stack([grid, np.full_like(grid, G)], axis=-1)
        w = to_wb(steady_state(u, p)[:, 0])   # moisture falls as Tin rises
        i = int(np.searchsorted(-w, -sp_wb))
        if i <= 0:
            return float(grid[0])
        if i >= grid.size:
            return float(grid[-1])
        lo, hi = grid[i - 1], grid[i]
        wlo, whi = w[i - 1], w[i]
    return float(lo + (wlo - sp_wb) / (wlo - whi) * (hi - lo))
