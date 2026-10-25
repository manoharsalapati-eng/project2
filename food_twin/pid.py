"""Velocity-form PID with derivative on the measurement (inherently anti-windup when the output is clamped)."""
from __future__ import annotations

from .physics import TS


class PID:
    def __init__(self, Kp: float = 10.0, Ti: float = 240.0, Td: float = 15.0):
        self.Kp, self.Ti, self.Td = Kp, Ti, Td
        self.e1 = self.pv1 = self.pv2 = None

    def step(self, pv: float, sp: float, u: float) -> float:
        e = pv - sp                     # reverse acting: product too wet -> more heat
        if self.e1 is None:
            self.e1, self.pv1, self.pv2 = e, pv, pv
        du = self.Kp * ((e - self.e1) + TS / self.Ti * e + self.Td / TS * (pv - 2 * self.pv1 + self.pv2))
        self.pv2, self.pv1, self.e1 = self.pv1, pv, e
        return u + du
