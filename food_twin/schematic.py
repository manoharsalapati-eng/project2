"""Inline SVG process schematic of one loop (extruder -> belt dryer -> exhaust), filled with live values."""
from __future__ import annotations

import base64
from html import escape

import numpy as np

from .physics import NOMINAL, Ambient, power
from .simulator import Simulator

THEME = {
    'light': dict(ink='#0b0b0b', ink2='#52514e', muted='#898781', axis='#c3c2b7', s2='#f2f1ed', sunk='#efeee9', surf='#fcfcfb',
                  est='#6250d6', warn='#fab219', good='#006300', crit='#b42626'),
    'dark': dict(ink='#f0efec', ink2='#c3c2b7', muted='#898781', axis='#383835', s2='#232321', sunk='#141413', surf='#1a1a19',
                 est='#9085e9', warn='#fab219', good='#0ca30c', crit='#e66767'),
}
_BROWN = [(0, (236, 220, 170)), (14, (226, 190, 120)), (24, (196, 140, 70)), (36, (130, 78, 34))]
CTRL = {'mpc': ('MPC (EKF-based)', 'manipulates T in and G'), 'pid': ('PID on EKF moisture', 'manipulates T in, G fixed'),
        'ol': ('Open-loop recipe', 'T in, G fixed')}


def _brown(b: float):
    for (b0, c0), (b1, c1) in zip(_BROWN, _BROWN[1:]):
        if b <= b1 or b1 == _BROWN[-1][0]:
            f = min(max((b - b0) / (b1 - b0), 0.0), 1.0)
            return tuple(round(v + f * (w - v)) for v, w in zip(c0, c1))
    return _BROWN[0][1]


def svg(sim: Simulator, lane: str, theme: str = 'light') -> str:
    C = THEME[theme]
    L = sim.lanes[lane]
    h = L.hist
    if not h['t']:
        return ''
    S = sim.shared
    e = lambda v, d: f'{v:,.{d}f}'
    pw = power(L.u, Ambient(0, sim.amb_Y, sim.amb_T, 0), NOMINAL)[0]
    Fs = sim.d.amb.Fs if sim.d else NOMINAL.Fs
    status, status_col, status_sub = (('✕ Off-spec', C['crit'], L.reason) if L.reason else ('✓ In spec', C['good'], f"{h['Xw'][-1]:.2f} % w.b."))
    ctrl, ctrl_sub = CTRL[lane]
    # belt pieces: colour ramps from fresh extrudate to the current outlet browning; position follows simulated time
    c_in, c_out = np.array(_brown(NOMINAL.Bin)), np.array(_brown(L.x[4]))
    span, n = 346.0, 24
    offset = (sim.t * 0.58 * Fs / NOMINAL.Fs) % span
    pieces = []
    for i in range(n):
        x = 262 + (i * span / n + offset) % span
        c = (c_in + (x - 262) / span * (c_out - c_in)).round().astype(int)
        pieces.append(f'<rect x="{x:.1f}" y="188" width="11" height="7" rx="2.5" fill="rgb({c[0]},{c[1]},{c[2]})"/>')
    air_op = min(max(0.25 + (L.u[1] - 2) / 6 * 0.75, 0.25), 1.0)
    arrows = ''.join(f'<line x1="{300 + i * 52}" y1="168" x2="{300 + i * 52}" y2="186" stroke="{C["ink2"]}" stroke-width="1.5" '
                     f'opacity="{air_op:.2f}" marker-end="url(#arw)"/>' for i in range(6))
    zig = lambda y: 'M40,%d ' % y + ' '.join(f'L{x + 4},{y - 6} L{x + 8},{y}' for x in range(40, 204, 8))
    flame_op = min(max(0.25 + pw / 700, 0.25), 1.0)
    fan_rot = (sim.t * L.u[1] * 1.8) % 360

    def box(x, label, val, unit, true):
        return (f'<rect x="{x}" y="246" width="128" height="74" rx="7" fill="{C["surf"]}" stroke="{C["est"]}" stroke-width="1.2"/>'
                f'<text x="{x + 10}" y="263" font-size="10.5" fill="{C["muted"]}">{label}</text>'
                f'<text x="{x + 10}" y="285" font-size="16" font-weight="600" fill="{C["ink"]}">{val}</text>'
                f'<text x="{x + 10}" y="300" font-size="10.5" fill="{C["muted"]}">{unit}</text>'
                f'<text x="{x + 10}" y="314" font-size="10.5" fill="{C["muted"]}">{true}</text>')

    t = lambda x, y, s, size=11, col='muted', extra='': f'<text x="{x}" y="{y}" font-size="{size}" fill="{C[col]}" {extra}>{escape(s)}</text>'
    st = f'fill="{C["s2"]}" stroke="{C["axis"]}" stroke-width="1.5"'
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 330" width="100%" role="img" style="font-family:system-ui,-apple-system,'Segoe UI',sans-serif"
 aria-label="Process schematic: twin-screw extruder feeding a hot-air conveyor dryer with heater, exhaust fan and exhaust temperature and humidity sensors">
<defs><marker id="arw" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{C['ink2']}"/></marker></defs>
<polygon points="42,18 96,18 82,44 56,44" {st}/>{t(69, 12, 'raw feed + water', extra='text-anchor="middle"')}
<rect x="8" y="46" width="26" height="40" rx="4" {st}/>{t(21, 70, 'M', 12, 'ink2', 'text-anchor="middle" font-weight="600"')}
<rect x="34" y="44" width="172" height="44" rx="7" fill="{C['sunk']}" stroke="{C['axis']}" stroke-width="1.5"/>
<path d="{zig(60)}" fill="none" stroke="{C['ink2']}" stroke-width="1.5" opacity=".7"/><path d="{zig(78)}" fill="none" stroke="{C['ink2']}" stroke-width="1.5" opacity=".7"/>
<polygon points="206,52 222,59 222,73 206,80" {st}/>{t(120, 104, 'Twin-screw extruder', 12, 'ink2', 'text-anchor="middle"')}
<path d="M222,66 C246,70 252,140 258,186" fill="none" stroke="{C['axis']}" stroke-width="1.5" stroke-dasharray="2 4"/>
{t(10, 130, 'Raw-feed moisture Xin (unmeasured)')}
{t(10, 148, f"EKF {e(h['XinHat'][-1], 1)} ± {e(2 * h['XinSd'][-1], 1)} % w.b.", 13, 'ink')}
{t(10, 166, f"true {e(h['Xin'][-1], 1)} % w.b.")}
{t(10, 190, 'Throughput (dry solids)')}{t(10, 208, f'{Fs * 3600:,.0f} kg/h', 13, 'ink')}
{t(236, 40, 'Fresh air')}{t(236, 54, f"{S['Tamb'][-1]:.1f} °C · {S['RHamb'][-1]:.0f} % RH", 11, 'ink2')}
<line x1="236" y1="78" x2="296" y2="78" stroke="{C['ink2']}" stroke-width="1.5" marker-end="url(#arw)"/>
<rect x="300" y="60" width="92" height="36" rx="6" {st}/>
<path d="M322,90 C314,82 320,74 324,68 C325,74 330,75 330,80 C333,77 333,73 332,70 C338,76 338,86 330,90 Z" fill="{C['warn']}" opacity="{flame_op:.2f}"/>
{t(364, 76, 'Heater', 12, 'ink2', 'text-anchor="middle" font-weight="600"')}{t(364, 90, f'{pw:,.0f} kW', 11, 'muted', 'text-anchor="middle"')}
<line x1="346" y1="96" x2="346" y2="128" stroke="{C['ink2']}" stroke-width="1.5" marker-end="url(#arw)"/>{t(352, 116, f'T in {L.u[0]:.1f} °C', 11, 'ink2')}
<rect x="420" y="16" width="124" height="44" rx="6" fill="{C['surf']}" stroke="{C['ink2']}" stroke-width="1.5"/>
{t(482, 33, ctrl, 12, 'ink', 'text-anchor="middle" font-weight="600"')}{t(482, 50, ctrl_sub, 10.5, 'muted', 'text-anchor="middle"')}
<path d="M440,60 V78 H392" fill="none" stroke="{C['ink2']}" stroke-dasharray="4 3"/><path d="M544,38 H586 V72" fill="none" stroke="{C['ink2']}" stroke-dasharray="4 3"/>
<rect x="232" y="128" width="404" height="104" rx="10" {st}/>{t(244, 146, 'Conveyor belt dryer', 12, 'ink2', 'font-weight="600"')}{t(244, 161, 'hot air through the product bed')}
{arrows}
<circle cx="258" cy="206" r="9" fill="{C['surf']}" stroke="{C['axis']}" stroke-width="1.5"/><circle cx="612" cy="206" r="9" fill="{C['surf']}" stroke="{C['axis']}" stroke-width="1.5"/>
<line x1="258" y1="197" x2="612" y2="197" stroke="{C['axis']}" stroke-width="1.5"/><line x1="258" y1="215" x2="612" y2="215" stroke="{C['axis']}" stroke-width="1.5"/>
{''.join(pieces)}
<line x1="586" y1="128" x2="586" y2="106" stroke="{C['ink2']}" stroke-width="1.5"/>
<circle cx="586" cy="90" r="16" fill="{C['surf']}" stroke="{C['ink2']}" stroke-width="1.5"/>
<g transform="translate(586 90) rotate({fan_rot:.0f})"><path d="M0,0 C4,-6 4,-12 0,-13 C-4,-12 -4,-6 0,0 M0,0 C6,4 12,4 13,0 C12,-4 6,-4 0,0 M0,0 C-4,6 -4,12 0,13 C4,12 4,6 0,0 M0,0 C-6,-4 -12,-4 -13,0 C-12,4 -6,4 0,0" fill="{C['ink2']}" opacity=".75"/></g>
{t(564, 100, 'exhaust fan', 10.5, 'muted', 'text-anchor="end"')}{t(564, 116, f'G {L.u[1]:.2f} kg/s', 11, 'ink2', 'text-anchor="end"')}
<line x1="602" y1="90" x2="734" y2="90" stroke="{C['ink2']}" stroke-width="1.5"/><line x1="734" y1="90" x2="734" y2="22" stroke="{C['ink2']}" stroke-width="1.5" marker-end="url(#arw)"/>
{t(726, 16, 'exhaust', 11, 'muted', 'text-anchor="end"')}
<line x1="640" y1="90" x2="640" y2="78" stroke="{C['ink2']}"/><circle cx="640" cy="66" r="12" fill="{C['surf']}" stroke="{C['ink2']}" stroke-width="1.2"/>
{t(640, 70, 'TT', 10, 'ink2', 'text-anchor="middle" font-weight="600"')}{t(640, 44, f"{h['TaMeas'][-1]:.1f} °C", 12, 'ink', 'text-anchor="middle"')}
<line x1="690" y1="90" x2="690" y2="102" stroke="{C['ink2']}"/><circle cx="690" cy="114" r="12" fill="{C['surf']}" stroke="{C['ink2']}" stroke-width="1.2"/>
{t(690, 118, 'MT', 10, 'ink2', 'text-anchor="middle" font-weight="600"')}{t(690, 142, f"{h['RHMeas'][-1]:.1f} % RH", 12, 'ink', 'text-anchor="middle"')}
<path d="M621,206 L664,246" stroke="{C['axis']}" stroke-width="1.5" stroke-dasharray="2 4"/>
<rect x="648" y="246" width="96" height="36" rx="5" {st}/>{t(696, 268, 'product out', 11.5, 'ink2', 'text-anchor="middle"')}
<text x="696" y="302" text-anchor="middle" font-size="12" font-weight="600" fill="{status_col}">{escape(status)}</text>
{t(696, 318, status_sub, 10.5, 'muted', 'text-anchor="middle"')}
{box(232, 'EKF · core moisture', f"{e(h['XwHat'][-1], 2)} ± {e(2 * h['XwSd'][-1], 2)}", '% w.b. (2σ)', f"true {e(h['Xw'][-1], 2)}")}
{box(370, 'EKF · product temp.', f"{e(h['TpHat'][-1], 1)} ± {e(2 * h['TpSd'][-1], 1)}", '°C (2σ)', f"true {e(h['Tp'][-1], 1)}")}
{box(508, 'EKF · browning index', f"{e(h['BHat'][-1], 1)} ± {e(2 * h['BSd'][-1], 1)}", 'BI (2σ)', f"true {e(h['B'][-1], 1)}")}
</svg>'''


def img_tag(sim: Simulator, lane: str, theme: str = 'light') -> str:
    """The schematic as a data-URI <img> (Streamlit's HTML sanitiser strips inline SVG)."""
    data = base64.b64encode(svg(sim, lane, theme).encode('utf-8')).decode('ascii')
    return (f'<img alt="Process schematic of the {lane} loop: extruder, belt dryer, heater, exhaust fan and sensors with live values" '
            f'style="width:100%;height:auto;display:block" src="data:image/svg+xml;base64,{data}">')
