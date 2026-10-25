"""Extended Kalman Filter soft sensor.

Augmented state  z = [X, Tp, Ta, Y, B, Xin]   (Xin = unmeasured raw-feed moisture, random walk)
Measurements     y = [Ta (degC), RH_exhaust (%)]
Browning does not affect either measurement: it is unobservable but detectable (stable washout),
so its estimate and covariance are propagated through the model.
"""
from __future__ import annotations

import numpy as np

from .disturbances import SENSOR
from .physics import NOMINAL, P_ATM, TS, Ambient, Params, dln_psat, psat, rh_from_humidity, rk4

NX = 6
_EPS = np.array([1e-4, 0.01, 0.01, 1e-5, 0.01, 1e-4])


class EKF:
    def __init__(self, p: Params = NOMINAL):
        self.p = p
        self.x = np.array([0.14, 50.0, 58.0, 0.020, 18.0, 0.34])        # deliberately wrong initial guess
        self.P = np.diag([0.03, 4.0, 2.0, 0.004, 6.0, 0.05]) ** 2
        # process noise includes a model-error allowance so the 2-sigma bands stay ~95 % consistent under mismatch
        self.Q = np.diag([8e-4, 0.2, 0.25, 6e-5, 0.1, 2.5e-3]) ** 2
        self.R = np.diag([SENSOR['sTa'], SENSOR['sRH']]) ** 2
        self.innov = np.zeros(2)
        self.nis = 0.0

    def f(self, Z, u, amb: Ambient):
        """Discrete transition over one sample for a batch of augmented states Z (..., 6)."""
        d = Ambient(Z[..., 5], amb.Yamb, amb.Tamb, amb.Fs)
        y = rk4(Z[..., :5], u, d, self.p, TS / 2, 2)
        return np.concatenate([y, Z[..., 5:6]], axis=-1)

    def predict(self, u, amb: Ambient):
        # base point + one finite-difference perturbation per state, integrated as a single batch
        Z = np.vstack([self.x, self.x + np.diag(_EPS)])
        FZ = self.f(Z, np.asarray(u, float), amb)
        F = ((FZ[1:] - FZ[0]) / _EPS[:, None]).T
        self.x = FZ[0]
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_meas):
        x = self.x
        Ta, Y = x[2], max(x[3], 1e-5)
        rh = 100 * rh_from_humidity(Ta, Y)
        H = np.zeros((2, NX))
        H[0, 2] = 1.0
        H[1, 2] = -rh * dln_psat(Ta)
        H[1, 3] = 100 * P_ATM * 0.622 / ((0.622 + Y) ** 2 * psat(Ta))
        S = H @ self.P @ H.T + self.R
        Si = np.linalg.inv(S)
        K = self.P @ H.T @ Si
        r = np.asarray(z_meas, float) - np.array([Ta, rh])
        self.innov = r
        self.nis = float(r @ Si @ r)
        self.x = x + K @ r
        IKH = np.eye(NX) - K @ H
        P = IKH @ self.P @ IKH.T + K @ self.R @ K.T        # Joseph form
        self.P = 0.5 * (P + P.T)
        self.x[0] = np.clip(self.x[0], 0.01, 1.5)
        self.x[3] = np.clip(self.x[3], 1e-4, 0.3)
        self.x[4] = max(self.x[4], 0.0)
        self.x[5] = np.clip(self.x[5], 0.05, 1.5)

    def sd(self, i: int) -> float:
        return float(np.sqrt(max(self.P[i, i], 0.0)))
