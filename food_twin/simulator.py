"""Three parallel plant twins (open-loop, PID, MPC), each with its own EKF soft sensor and KPIs."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .disturbances import SENSOR, Disturbances, Sample
from .ekf import EKF
from .mpc import MPC, MPC_DT, NB, Prediction, predict, u_to_z
from .physics import LIMITS, NOMINAL, TS, Ambient, Params, humidity_from_rh, power, recipe_tin, rh_from_humidity, rk4, steady_state, to_wb
from .pid import PID

LANES = ('ol', 'pid', 'mpc')
LANE_NAME = {'ol': 'Open-loop', 'pid': 'PID', 'mpc': 'MPC'}
HIST_KEYS = ('t', 'Xw', 'XwHat', 'XwSd', 'Tp', 'TpHat', 'TpSd', 'Ta', 'TaMeas', 'TaHat', 'RH', 'RHMeas', 'RHHat',
             'B', 'BHat', 'BSd', 'Xin', 'XinHat', 'XinSd', 'Tin', 'G', 'Pw', 'offspec')
MAX_HIST = 4320                         # 6 h of 5 s samples


@dataclass
class Config:
    sp: float = 9.0          # moisture set-point [% w.b.]
    tol: float = 1.0         # spec band +/- [% w.b.]
    Tmax: float = 60.0       # hard product-temperature limit [degC]
    backoff: float = 0.8     # MPC constraint back-off [degC]
    z_sigma: float = 2.0     # MPC chance-constraint tightening (x EKF sigma_Tp)
    Bmax: float = 22.0       # browning limit [BI]
    mismatch: float = 0.05   # plant-model mismatch fraction
    wE: float = 1.0          # MPC energy weight
    wDu: float = 2.0         # MPC move suppression
    Kp: float = 10.0         # PID gain [degC per % w.b.]
    Ti: float = 240.0        # PID integral time [s]
    Td: float = 15.0         # PID derivative time [s]
    Gfix: float = 4.0        # airflow used by open-loop and PID [kg/s]
    TinOL: float = 0.0       # open-loop recipe inlet temperature (derived)


@dataclass
class MPCCall:
    sp: float
    Bmax: float
    Tmax: float
    backoff: float
    z_sigma: float
    sigma_Tp: float


def plant_params(mis: float) -> Params:
    return replace(NOMINAL, hA0=NOMINAL.hA0 * (1 - mis), kB=NOMINAL.kB * (1 + 2 * mis),
                   XD=NOMINAL.XD * (1 + 1.5 * mis), UA=NOMINAL.UA * (1 + 2 * mis))


def new_kpi():
    return dict(time=0.0, iae=0.0, energy=0.0, mass=0.0, water=0.0, offMass=0.0, tAbove=0.0, TpMax=-np.inf, se=0.0, cover=0, n=0)


class Lane:
    def __init__(self, mode: str, pp: Params, u0):
        self.mode = mode
        self.u = np.array(u0, float)
        self.u_cmd = self.u.copy()
        self.x = steady_state(self.u, pp)
        self.ekf = EKF(NOMINAL)
        self.pid = PID()
        self.mpc = MPC(NOMINAL)
        self.hist = {k: [] for k in HIST_KEYS}
        self.reason = ''
        self.plan_t = 0.0
        self.nis_avg = None
        self.k = new_kpi()

    def reset_kpi(self):
        self.k = new_kpi()

    def step(self, x_new, x_old, s: Sample, amb_meas: Ambient, t: float, cfg: Config, step_idx: int, pp: Params):
        self.x = x = x_new
        # noisy exhaust-air sensors
        nm = s.noise_mult
        Ta_meas = x[2] + SENSOR['sTa'] * nm * s.nTa
        RH_true = 100 * min(1.0, float(rh_from_humidity(x[2], x[3])))
        RH_meas = float(np.clip(RH_true + SENSOR['sRH'] * nm * s.nRH, 0, 100))
        # EKF soft sensor
        self.ekf.predict(self.u, amb_meas)
        self.ekf.update([Ta_meas, RH_meas])
        self.nis_avg = self.ekf.nis if self.nis_avg is None else 0.97 * self.nis_avg + 0.03 * self.ekf.nis
        e = self.ekf.x
        # controller
        if self.mode == 'pid':
            self.pid.Kp, self.pid.Ti, self.pid.Td = cfg.Kp, cfg.Ti, cfg.Td
            self.u_cmd = np.array([np.clip(self.pid.step(float(to_wb(e[0])), cfg.sp, self.u[0]), LIMITS.TinMin, LIMITS.TinMax), cfg.Gfix])
        elif self.mode == 'mpc':
            if step_idx % int(MPC_DT / TS) == 0:
                self.mpc.wE, self.mpc.wDu = cfg.wE, cfg.wDu
                self.plan_t = t
                self.u_cmd = self.mpc.solve(e[:5], Ambient(e[5], amb_meas.Yamb, amb_meas.Tamb, s.amb.Fs), self.u,
                                            MPCCall(cfg.sp, cfg.Bmax, cfg.Tmax, cfg.backoff, cfg.z_sigma, self.ekf.sd(1)))
        else:
            self.u_cmd = np.array([cfg.TinOL, cfg.Gfix])
        u_prev = self.u
        du = np.clip(self.u_cmd - u_prev, [-LIMITS.dTin, -LIMITS.dG], [LIMITS.dTin, LIMITS.dG])
        self.u = np.clip(u_prev + du, [LIMITS.TinMin, LIMITS.Gmin], [LIMITS.TinMax, LIMITS.Gmax])
        # KPIs on the TRUE plant
        pw = float(power(u_prev, s.amb, pp)[2])
        Xw = float(to_wb(x[0]))
        mass_out = s.amb.Fs * (1 + x[0]) * TS
        water = max(0.0, s.amb.Fs * (s.amb.Xin - 0.5 * (x[0] + x_old[0])) * TS - pp.Ms * (x[0] - x_old[0]))
        self.reason = ('thermal limit exceeded' if x[1] > cfg.Tmax else 'over-browned' if x[4] > cfg.Bmax else
                       'too wet' if Xw > cfg.sp + cfg.tol else 'over-dried' if Xw < cfg.sp - cfg.tol else '')
        off = self.reason != ''
        K = self.k
        K['time'] += TS
        K['iae'] += abs(Xw - cfg.sp) * TS / 60
        K['energy'] += pw * TS
        K['mass'] += mass_out
        K['water'] += water
        if off:
            K['offMass'] += mass_out
        if x[1] > cfg.Tmax:
            K['tAbove'] += TS
        K['TpMax'] = max(K['TpMax'], float(x[1]))
        Xw_hat = float(to_wb(e[0]))
        Xw_sd = 100 / (1 + e[0]) ** 2 * self.ekf.sd(0)
        if K['time'] > 300:                                    # skip the EKF start-up transient
            K['se'] += (Xw_hat - Xw) ** 2
            K['n'] += 1
            K['cover'] += abs(Xw_hat - Xw) <= 2 * Xw_sd
        rec = dict(t=t, Xw=Xw, XwHat=Xw_hat, XwSd=Xw_sd, Tp=x[1], TpHat=e[1], TpSd=self.ekf.sd(1), Ta=x[2], TaMeas=Ta_meas,
                   TaHat=e[2], RH=RH_true, RHMeas=RH_meas, RHHat=100 * float(rh_from_humidity(e[2], e[3])), B=x[4], BHat=e[4],
                   BSd=self.ekf.sd(4), Xin=float(to_wb(s.amb.Xin)), XinHat=float(to_wb(e[5])), XinSd=100 / (1 + e[5]) ** 2 * self.ekf.sd(5),
                   Tin=u_prev[0], G=u_prev[1], Pw=pw, offspec=int(off))
        for k in HIST_KEYS:
            self.hist[k].append(float(rec[k]))

    def trim(self, n_max: int):
        n = len(self.hist['t']) - n_max
        if n > 0:
            for k in HIST_KEYS:
                del self.hist[k][:n]

    def kpis(self) -> dict:
        K = self.k
        return dict(iae=K['iae'], spe=K['energy'] / K['mass'] if K['mass'] else np.nan,
                    epw=K['energy'] / K['water'] if K['water'] else np.nan,
                    off=100 * K['offMass'] / K['mass'] if K['mass'] else np.nan,
                    tmax=K['TpMax'], above=K['tAbove'] / 60,
                    rmse=np.sqrt(K['se'] / K['n']) if K['n'] else np.nan, cov=100 * K['cover'] / K['n'] if K['n'] else np.nan)


class Simulator:
    def __init__(self, seed: int = 20261025, cfg: Config | None = None):
        self.seed = seed
        self.t = 0.0
        self.step_idx = 0
        self.cfg = replace(cfg) if cfg else Config()
        self.cfg.TinOL = recipe_tin(self.cfg.Gfix, self.cfg.sp)
        self.pp = plant_params(self.cfg.mismatch)
        self.dist = Disturbances(seed)
        u0 = [self.cfg.TinOL, self.cfg.Gfix]
        self.lanes = {m: Lane(m, self.pp, u0) for m in LANES}
        self.amb_T, self.amb_Y = 20.0, 0.0080
        self.d: Sample | None = None
        self.shared = {k: [] for k in ('t', 'Tamb', 'RHamb', 'Fs')}

    # ---- configuration helpers ----
    def set_gfix(self, G: float):
        self.cfg.Gfix = G
        self.cfg.TinOL = recipe_tin(G, self.cfg.sp)

    def set_setpoint(self, sp: float):
        self.cfg.sp = sp
        self.cfg.TinOL = recipe_tin(self.cfg.Gfix, sp)

    def set_mismatch(self, m: float):
        self.cfg.mismatch = m
        self.pp = plant_params(m)

    def reset_kpi(self):
        for lane in self.lanes.values():
            lane.reset_kpi()

    def amb_meas(self) -> Ambient:
        return Ambient(0.0, self.amb_Y, self.amb_T, self.d.amb.Fs if self.d else NOMINAL.Fs)

    def hold_forecast(self, lane: Lane) -> Prediction:
        """Model forecast from the EKF estimate with the current inputs held (what-if view for OL / PID)."""
        e = lane.ekf.x
        z = np.tile(u_to_z(lane.u), NB)
        a = self.amb_meas()
        return predict(e[:5], z, Ambient(e[5], a.Yamb, a.Tamb, a.Fs), NOMINAL)

    # ---- one 5 s step for all three twins ----
    def step(self):
        s = self.dist.step(self.t)
        self.d = s
        self.t += TS
        self.step_idx += 1
        # intake weather station (noisy, first-order filtered)
        Tm = s.amb.Tamb + SENSOR['sTamb'] * s.nTamb
        RHm = float(np.clip(s.RHamb + SENSOR['sRHamb'] / 100 * s.nRHamb, 0.01, 1))
        self.amb_T += 0.3 * (Tm - self.amb_T)
        self.amb_Y += 0.3 * (float(humidity_from_rh(Tm, RHm)) - self.amb_Y)
        amb = Ambient(0.0, self.amb_Y, self.amb_T, s.amb.Fs)
        # the three true plants are integrated as one batch (RK4, 1 s substeps)
        lanes = list(self.lanes.values())
        X_old = np.array([lane.x for lane in lanes])
        X_new = rk4(X_old, np.array([lane.u for lane in lanes]), s.amb, self.pp, 1.0, int(TS))
        for lane, xn, xo in zip(lanes, X_new, X_old):
            lane.step(xn, xo, s, amb, self.t, self.cfg, self.step_idx, self.pp)
        S = self.shared
        S['t'].append(self.t)
        S['Tamb'].append(s.amb.Tamb)
        S['RHamb'].append(100 * s.RHamb)
        S['Fs'].append(s.amb.Fs * 3600)
        if len(S['t']) > MAX_HIST + 200:
            n = len(S['t']) - MAX_HIST
            for k in S:
                del S[k][:n]
            for lane in lanes:
                lane.trim(MAX_HIST)

    def run(self, seconds: float, events: dict[float, list[str]] | None = None):
        """Advance the simulation; `events` maps a simulation time [s] to disturbance types to inject then."""
        end = self.t + seconds
        while self.t < end - 1e-9:
            for kind in (events or {}).get(self.t, []):
                self.dist.inject(kind, self.t)
            self.step()
