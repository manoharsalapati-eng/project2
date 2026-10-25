"""Shared disturbance generator: stochastic fluctuations, injected events and sensor-noise draws.

All three plant twins consume the same draws (common random numbers), so differences in
their KPIs come from the controllers alone. The generator is a bit-exact port of the
browser version (mulberry32 + Marsaglia polar), so a given seed gives the same weather in both.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .physics import NOMINAL, TS, Ambient, rh_from_humidity

EVENT_TYPES = {
    #           label                     duration [s]  magnitude                         short label   description
    'wet':   dict(label='Wet-feed slug',          dur=720, mag=0.07,  short='wet feed',  desc='+3.4 % w.b. raw-feed moisture · 12 min'),
    'humid': dict(label='Ambient humidity spike', dur=900, mag=0.009, short='humid air', desc='+9 g/kg ambient humidity, +3 °C · 15 min'),
    'surge': dict(label='Throughput surge',       dur=600, mag=0.25,  short='surge',     desc='+25 % extruder throughput · 10 min'),
    'cold':  dict(label='Cold snap',              dur=900, mag=-12.0, short='cold snap', desc='−12 °C intake air · 15 min'),
    'noise': dict(label='Sensor noise burst',     dur=300, mag=4.0,   short='noise',     desc='5× exhaust sensor noise · 5 min'),
}

SENSOR = dict(sTa=0.4, sRH=1.2, sTamb=0.15, sRHamb=0.8)   # 1-sigma sensor noise


def mulberry32(seed: int):
    state = seed & 0xFFFFFFFF

    def rand() -> float:
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (61 | t)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return rand


def make_gauss(rand):
    spare = None

    def gauss() -> float:
        nonlocal spare
        if spare is not None:
            s, spare = spare, None
            return s
        while True:
            u, v = rand() * 2 - 1, rand() * 2 - 1
            s = u * u + v * v
            if 0 < s < 1:
                break
        m = math.sqrt(-2 * math.log(s) / s)
        spare = v * m
        return u * m
    return gauss


@dataclass
class Event:
    type: str
    start: float
    end: float


@dataclass
class Sample:
    """Disturbances and shared noise draws for one 5 s step."""
    amb: Ambient
    RHamb: float
    noise_mult: float
    nTa: float
    nRH: float
    nTamb: float
    nRHamb: float


@dataclass
class Disturbances:
    seed: int
    stoch_feed: bool = True
    weather: bool = True
    noise_scale: float = 1.0
    events: list[Event] = field(default_factory=list)

    def __post_init__(self):
        self.gauss = make_gauss(mulberry32(self.seed))
        self.ouX = self.ouY = self.ouT = 0.0
        self.filt = {k: 0.0 for k in EVENT_TYPES}

    def inject(self, kind: str, t: float):
        self.events.append(Event(kind, t, t + EVENT_TYPES[kind]['dur']))

    def active(self, kind: str, t: float) -> bool:
        return any(e.type == kind and e.start <= t < e.end for e in self.events)

    def step(self, t: float) -> Sample:
        g, a = self.gauss, TS
        # Ornstein-Uhlenbeck fluctuations (draws are always consumed so toggles don't desync the stream)
        nX, nY, nT = g(), g(), g()
        self.ouX += -self.ouX * a / 400 + 0.012 * math.sqrt(2 * a / 400) * nX
        self.ouY += -self.ouY * a / 900 + 0.0008 * math.sqrt(2 * a / 900) * nY
        self.ouT += -self.ouT * a / 1200 + 0.8 * math.sqrt(2 * a / 1200) * nT
        for k in self.filt:
            target = EVENT_TYPES[k]['mag'] if self.active(k, t) else 0.0
            tau = 1 if k == 'noise' else 40
            self.filt[k] += (target - self.filt[k]) * min(1.0, a / tau)
        f = self.filt
        Tamb = 20 + (self.ouT if self.weather else 0) + f['cold'] + 3 * f['humid'] / 0.009
        Yamb = max(0.002, 0.0080 + (self.ouY if self.weather else 0) + f['humid'])
        amb = Ambient(Xin=0.40 + (self.ouX if self.stoch_feed else 0) + f['wet'], Yamb=Yamb, Tamb=Tamb,
                      Fs=NOMINAL.Fs * (1 + f['surge']))
        s = Sample(amb=amb, RHamb=min(1.0, float(rh_from_humidity(Tamb, Yamb))), noise_mult=self.noise_scale * (1 + f['noise']),
                   nTa=g(), nRH=g(), nTamb=g(), nRHamb=g())
        self.events = [e for e in self.events if t < e.end + 3600 * 6]
        return s
