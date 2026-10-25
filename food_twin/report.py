"""KPI table, telemetry export and summary text shared by the CLI and the dashboard."""
from __future__ import annotations

import io

import numpy as np

from .simulator import LANE_NAME, LANES, Simulator

KPI_COLUMNS = [  # key, header, short header (terminal), decimals, lower-is-better
    ('iae', 'IAE moisture [% w.b.·min]', 'IAE %·min', 1, True),
    ('spe', 'Specific energy [kJ/kg product]', 'kJ/kg prod', 0, True),
    ('epw', 'Energy per water [kJ/kg H2O]', 'kJ/kg H2O', 0, True),
    ('off', 'Off-spec rejection [% mass]', 'off-spec %', 1, True),
    ('tmax', 'Max product temp. [°C]', 'Tp max °C', 1, None),
    ('above', 'Time above Tmax [min]', '>Tmax min', 1, True),
    ('rmse', 'Soft-sensor RMSE [% w.b.]', 'EKF RMSE', 3, None),
    ('cov', '±2σ coverage [%]', '2σ cover %', 0, None),
]
CSV_KEYS = ('Xw', 'XwHat', 'XwSd', 'Tp', 'TpHat', 'TpSd', 'Ta', 'TaMeas', 'RH', 'RHMeas', 'B', 'BHat', 'BSd', 'Xin', 'XinHat', 'Tin', 'G', 'Pw', 'offspec')


def kpi_records(sim: Simulator, short: bool = False) -> list[dict]:
    """One dict per strategy: formatted values, with ' ★' on the strict winner of each lower-is-better column."""
    vals = {l: sim.lanes[l].kpis() for l in LANES}
    rows = []
    for l in LANES:
        row = {'Strategy': LANE_NAME[l]}
        for key, long_head, short_head, dec, lower in KPI_COLUMNS:
            head = short_head if short else long_head
            v = vals[l][key]
            s = '–' if not np.isfinite(v) else f'{v:,.{dec}f}'
            if lower and np.isfinite(v):
                others = sorted(vals[o][key] for o in LANES if np.isfinite(vals[o][key]))
                if len(others) > 1 and others[1] - others[0] > 1e-9 and v == others[0]:
                    s += ' ★'
            row[head] = s
        rows.append(row)
    return rows


def kpi_text(sim: Simulator) -> str:
    rows = kpi_records(sim, short=True)
    heads = list(rows[0].keys())
    widths = [max(len(h), *(len(r[h]) for r in rows)) for h in heads]
    line = lambda cells: '  '.join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))
    return '\n'.join([line(heads), line(['-' * w for w in widths])] + [line([r[h] for h in heads]) for r in rows])


def summary(sim: Simulator) -> str:
    p, q, o = (sim.lanes[l].kpis() for l in ('pid', 'mpc', 'ol'))
    if not np.isfinite(p['spe']) or sim.t < 120:
        return 'Collecting data…'
    de = 100 * (q['spe'] - p['spe']) / p['spe']
    di = 100 * (q['iae'] - p['iae']) / p['iae'] if p['iae'] > 0 else 0.0
    return (f"MPC vs PID: specific energy {de:+.1f} %, moisture IAE {di:+.0f} %, off-spec {q['off']:.1f} % vs {p['off']:.1f} % "
            f"(open-loop {o['off']:.1f} %), time above Tmax {q['above']:.1f} min vs {p['above']:.1f} min.")


def to_csv(sim: Simulator) -> str:
    S = sim.shared
    n = len(S['t'])
    head = ['t_s', 'Tamb_C', 'RHamb_pct', 'throughput_kg_h'] + [f'{l}_{k}' for l in LANES for k in CSV_KEYS]
    buf = io.StringIO()
    buf.write(','.join(head) + '\n')
    for i in range(n):
        row = [f"{S['t'][i]:g}", f"{S['Tamb'][i]:.3f}", f"{S['RHamb'][i]:.3f}", f"{S['Fs'][i]:.1f}"]
        for l in LANES:
            h = sim.lanes[l].hist
            j = len(h['t']) - n + i
            row += [f'{h[k][j]:.5f}' if j >= 0 else '' for k in CSV_KEYS]
        buf.write(','.join(row) + '\n')
    return buf.getvalue()
