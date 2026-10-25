"""Constrained nonlinear MPC: move-blocked inputs, real-time-iteration SQP and an ADMM (OSQP-style) QP.

    min  sum_k  wX (X_k - X_sp)^2 + wE P_k / P_ref + wB max(0, B_k - B_max + 1)^2  +  wDu sum ||dz||^2
    s.t. Tp_k <= Tmax - backoff - z_sigma * sigma_Tp      (hard, every horizon step)
         60 <= Tin <= 200 degC,  2 <= G <= 8 kg/s

Sensitivities come from finite differences, all perturbations integrated as one batch. After the
SQP step a safety filter scales the inlet-temperature plan down until the nonlinear prediction is
feasible, so the hard limit is honoured on the model even if the QP linearisation was optimistic.
"""
from __future__ import annotations

import time
from typing import NamedTuple

import numpy as np

from .physics import CP_A, CP_V, LIMITS, NOMINAL, Ambient, Params, power, ros2_matrix, ros2_step, to_wb

MPC_DT = 15.0                              # MPC sample time [s]
BLOCKS = (2, 4, 8, 16, 30)                 # move-blocking lengths (steps)
N = sum(BLOCKS)                            # 60 steps = 15 min horizon
NB = len(BLOCKS)
NZ = 2 * NB
BLOCK_OF = np.repeat(np.arange(NB), BLOCKS)
BLOCK_START = np.concatenate([[0], np.cumsum(BLOCKS)[:-1]])
U_LO = np.array([LIMITS.TinMin, LIMITS.Gmin])
U_SPAN = np.array([LIMITS.TinMax - LIMITS.TinMin, LIMITS.Gmax - LIMITS.Gmin])
CON_ROWS = np.array([k for k in range(N) if k < 8 or (k - 8) % 2 == 0])   # Tp rows kept in the QP


def z_to_u(z):
    return U_LO + np.asarray(z).reshape(*np.shape(z)[:-1], NB, 2) * U_SPAN


def u_to_z(u):
    return (np.asarray(u, float) - U_LO) / U_SPAN


class Prediction(NamedTuple):
    Xw: np.ndarray   # outlet moisture [% w.b.]   (M, N)
    Tp: np.ndarray   # product temperature [degC]
    B: np.ndarray    # browning index
    Pw: np.ndarray   # heater + fan power [kW]

    def row(self, i):
        return Prediction(self.Xw[i:i + 1], self.Tp[i:i + 1], self.B[i:i + 1], self.Pw[i:i + 1])


def predict(x0, Z, d: Ambient, p: Params = NOMINAL, substeps: int = 1) -> Prediction:
    """Predict M input plans at once. The Rosenbrock matrix is rebuilt at every move-block boundary."""
    Z = np.atleast_2d(Z)
    M = Z.shape[0]
    U = z_to_u(Z)                                        # (M, NB, 2)
    h = MPC_DT / substeps
    x = np.broadcast_to(np.asarray(x0, float)[:5], (M, 5)).copy()
    Xw, Tp, B = np.empty((M, N)), np.empty((M, N)), np.empty((M, N))
    for j, (k0, length) in enumerate(zip(BLOCK_START, BLOCKS)):
        u = U[:, j]
        for k in range(k0, k0 + length):
            if (k - k0) % 4 == 0:                        # refresh W every minute of prediction
                Winv, f0 = ros2_matrix(x, u, d, p, h)
            else:
                f0 = None
            for _ in range(substeps):
                x = ros2_step(x, u, d, p, h, Winv, f0)
                f0 = None
            Xw[:, k], Tp[:, k], B[:, k] = to_wb(x[:, 0]), x[:, 1], x[:, 4]
    Pw = power(U, d, p)[2][:, BLOCK_OF]
    bad = ~(np.isfinite(Xw).all(1) & np.isfinite(Tp).all(1))
    if bad.any():
        Xw[bad], Tp[bad], B[bad] = 1e3, 1e3, 1e3
    return Prediction(Xw, Tp, B, Pw)


def admm_qp(H, q, A, lo, up, iters=200):
    """min 1/2 x'Hx + q'x  s.t.  lo <= A x <= up   (dense ADMM, OSQP iteration)."""
    n = q.size
    sigma, alpha, rho = 1e-6, 1.6, 0.1
    Kinv = np.linalg.inv(H + sigma * np.eye(n) + rho * A.T @ A)
    x = np.zeros(n)
    z = np.clip(0.0, lo, up)
    y = np.zeros(A.shape[0])
    for it in range(iters):
        xt = Kinv @ (sigma * x - q + A.T @ (rho * z - y))
        zt = A @ xt
        zr = alpha * zt + (1 - alpha) * z
        zn = np.clip(zr + y / rho, lo, up)
        y = y + rho * (zr - zn)
        prim = np.abs(zt - zn).max()
        z = zn
        x = alpha * xt + (1 - alpha) * x
        if prim < 2e-4 and it > 15:
            break
    return x


class MPC:
    def __init__(self, p: Params = NOMINAL, sqp_iters: int = 1, substeps: int = 1):
        # defaults = classic real-time iteration (one SQP step per sample, 15 s ROS2 steps); raise both for
        # a more converged / more accurate solve at about 3x the cost
        self.p, self.sqp_iters, self.substeps = p, sqp_iters, substeps
        self.wX = 6.0      # moisture tracking weight [1/(% w.b.)^2]
        self.wE = 1.0      # energy weight (per P_ref)
        self.wDu = 2.0     # move suppression
        self.wB = 0.5      # browning soft limit
        self.Pref = 400.0  # kW
        self.z = None
        self.plan = None
        self.diag = dict(ms=0.0, iters=0, active=0, Tlim=float('nan'), maxTp=float('nan'), J=float('nan'), filtered=False)

    def _predict(self, x0, Z, d):
        return predict(x0, Z, d, self.p, self.substeps)

    def cost(self, pr: Prediction, Z, z_prev, sp, Bmax):
        Z = np.atleast_2d(Z)
        e = pr.Xw - sp
        J = (self.wX * e * e + self.wE * pr.Pw / self.Pref).sum(1) / N
        b = np.maximum(pr.B - (Bmax - 1), 0.0)
        J += self.wB * (b * b).sum(1) / N
        Zb = Z.reshape(-1, NB, 2)
        prev = np.concatenate([np.broadcast_to(z_prev, (Zb.shape[0], 1, 2)), Zb[:, :-1]], axis=1)
        return J + self.wDu * ((Zb - prev) ** 2).sum((1, 2))

    def solve(self, x0, d: Ambient, u_applied, cfg) -> np.ndarray:
        t0 = time.perf_counter()
        p, sp, Bmax = self.p, cfg.sp, cfg.Bmax
        z_prev = u_to_z(u_applied)
        z = self.z.copy() if self.z is not None else np.tile(z_prev, NB)
        Tlim = cfg.Tmax - cfg.backoff - cfg.z_sigma * cfg.sigma_Tp
        z_cool = np.tile([0.0, z_prev[1]], NB)          # minimum heat at the current airflow
        lim = None
        cur = None
        merit = lambda pr, Z: self.cost(pr, Z, z_prev, sp, Bmax) + 5.0 * np.maximum(pr.Tp - lim, 0.0).sum(1)
        iters = 0
        for _ in range(self.sqp_iters):
            iters += 1
            hh = np.where(z + 0.01 > 1, -0.01, 0.01)
            rows = [z[None, :], z + np.diag(hh)] + ([z_cool[None, :]] if lim is None else [])
            pr = self._predict(x0, np.vstack(rows), d)
            if lim is None:
                # feasibility relaxation: never demand a Tp below the lowest achievable trajectory
                lim = np.maximum(Tlim, pr.Tp[-1] + 0.3)
            base = pr.row(0)
            cur = base
            JX = (pr.Xw[1:NZ + 1] - base.Xw) / hh[:, None]
            JT = (pr.Tp[1:NZ + 1] - base.Tp) / hh[:, None]
            JB = (pr.B[1:NZ + 1] - base.B) / hh[:, None]
            # Gauss-Newton Hessian and gradient
            e = base.Xw[0] - sp
            wx = self.wX / N
            H = wx * JX @ JX.T
            g = wx * JX @ e
            eb = base.B[0] - (Bmax - 1)
            act = eb > -0.5
            if act.any():
                wb = self.wB / N
                H += wb * JB[:, act] @ JB[:, act].T
                g += wb * JB[:, act] @ np.maximum(eb[act], 0.0)
            # exact energy gradient (power depends only on its own block's input)
            U = z_to_u(z)
            c = (CP_A + CP_V * d.Yamb) / p.eta
            w = self.wE * np.asarray(BLOCKS) / N / self.Pref
            g[0::2] += w * U[:, 1] * c * U_SPAN[0]
            g[1::2] += w * (c * (U[:, 0] - d.Tamb) + 3 * p.fan * U[:, 1] ** 2) * U_SPAN[1]
            # move suppression (exact quadratic)
            for j in range(NB):
                for v in range(2):
                    i, ip = 2 * j + v, 2 * (j - 1) + v
                    dz = z[i] - (z_prev[v] if j == 0 else z[ip])
                    H[i, i] += self.wDu
                    g[i] += self.wDu * dz
                    if j > 0:
                        H[ip, ip] += self.wDu
                        H[i, ip] -= self.wDu
                        H[ip, i] -= self.wDu
                        g[ip] -= self.wDu * dz
            H += 1e-3 * np.eye(NZ)
            # constraints: trust-region box + linearised hard Tp limit
            TR = 0.25
            A = np.vstack([np.eye(NZ), JT[:, CON_ROWS].T / 5.0])
            lo = np.concatenate([np.maximum(-z, -TR), np.full(CON_ROWS.size, -1e9)])
            up = np.concatenate([np.minimum(1 - z, TR), (lim[CON_ROWS] - base.Tp[0, CON_ROWS]) / 5.0])
            step = admm_qp(H, g, A, lo, up)
            # backtracking on an exact-penalty merit function, all candidate step lengths in one batch
            m0 = merit(base, z)[0]
            alphas = np.array([1.0, 0.5, 0.25, 0.125])
            Zc = np.clip(z + alphas[:, None] * step, 0.0, 1.0)
            prc = self._predict(x0, Zc, d)
            ok = np.flatnonzero(merit(prc, Zc) <= m0 + 1e-9)
            if ok.size == 0:
                break
            z = Zc[ok[0]]
            cur = prc.row(ok[0])
        # safety filter: guarantee the nonlinear prediction honours the hard Tp limit
        viol = lambda pr: (pr.Tp - lim).max(1)
        filtered = False
        if viol(cur)[0] > 0.02:
            filtered = True
            s_lo, s_hi = 0.0, 1.0
            scale = lambda s: np.where(np.arange(NZ) % 2 == 0, z * (1 - np.asarray(s)[:, None]), z)
            for _ in range(2):                           # two batched grid refinements instead of a bisection loop
                s = np.linspace(s_lo, s_hi, 13)[1:]
                ok = viol(self._predict(x0, scale(s), d)) <= 0
                if not ok.any():
                    s_hi = 1.0
                    break
                i = int(np.argmax(ok))
                s_lo, s_hi = (s[i - 1] if i > 0 else s_lo), s[i]
            z = scale([s_hi])[0]
            cur = self._predict(x0, z, d)
        self.z = z
        U = z_to_u(z)[BLOCK_OF]
        self.plan = dict(Xw=cur.Xw[0], Tp=cur.Tp[0], B=cur.B[0], Pw=cur.Pw[0], Tin=U[:, 0], G=U[:, 1], lim=lim)
        self.diag = dict(ms=1e3 * (time.perf_counter() - t0), iters=iters, active=int((cur.Tp[0] > lim - 0.15).sum()),
                         Tlim=Tlim, maxTp=float(cur.Tp[0].max()), J=float(self.cost(cur, z, z_prev, sp, Bmax)[0]), filtered=filtered)
        return z_to_u(z)[0]
