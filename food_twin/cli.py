"""Headless scenario runs.

    python -m food_twin --minutes 150 --events wet@20,humid@60,surge@90,cold@120
    python -m food_twin --gfix 3 --events wet@20,humid@20 --csv run.csv --html run.html
"""
from __future__ import annotations

import argparse
import time

from .disturbances import EVENT_TYPES
from .physics import TS
from .report import kpi_text, summary, to_csv
from .simulator import Config, Simulator


def parse_events(spec: str) -> dict[float, list[str]]:
    """'wet@20,humid@60' -> {1200.0: ['wet'], 3600.0: ['humid']} (minutes, snapped to the 5 s grid)."""
    out: dict[float, list[str]] = {}
    for item in filter(None, (s.strip() for s in spec.split(','))):
        kind, _, minute = item.partition('@')
        if kind not in EVENT_TYPES:
            raise SystemExit(f'unknown event {kind!r}; choose from {", ".join(EVENT_TYPES)}')
        t = round(float(minute) * 60 / TS) * TS
        out.setdefault(t, []).append(kind)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m food_twin', description='Food dryer digital twin: open-loop vs PID vs constrained MPC.')
    ap.add_argument('--minutes', type=float, default=150, help='simulated duration (default 150)')
    ap.add_argument('--seed', type=int, default=12345)
    ap.add_argument('--events', default='wet@20,humid@60,surge@90,cold@120',
                    help=f'comma list of type@minute; types: {", ".join(EVENT_TYPES)}')
    ap.add_argument('--sp', type=float, default=9.0, help='moisture set-point [%% w.b.]')
    ap.add_argument('--tmax', type=float, default=60.0, help='max product temperature [degC]')
    ap.add_argument('--gfix', type=float, default=4.0, help='fixed airflow for open-loop and PID [kg/s]')
    ap.add_argument('--mismatch', type=float, default=0.05, help='plant-model mismatch fraction')
    ap.add_argument('--noise', type=float, default=1.0, help='sensor-noise multiplier')
    ap.add_argument('--no-stochastic', action='store_true', help='switch off random feed-moisture and weather drift')
    ap.add_argument('--sqp', type=int, default=1, help='SQP iterations per MPC solve (default 1 = real-time iteration)')
    ap.add_argument('--substeps', type=int, default=1, help='ROS2 substeps per 15 s MPC prediction step')
    ap.add_argument('--csv', help='write telemetry of all three loops to this CSV file')
    ap.add_argument('--html', help='write an interactive Plotly report to this HTML file')
    a = ap.parse_args(argv)

    sim = Simulator(a.seed, Config(sp=a.sp, Tmax=a.tmax, Gfix=a.gfix, mismatch=a.mismatch))
    sim.dist.noise_scale = a.noise
    sim.dist.stoch_feed = sim.dist.weather = not a.no_stochastic
    for lane in sim.lanes.values():
        lane.mpc.sqp_iters, lane.mpc.substeps = a.sqp, a.substeps
    events = parse_events(a.events)
    t0 = time.perf_counter()
    total = a.minutes * 60
    while sim.t < total - 1e-9:
        sim.run(min(600.0, total - sim.t), events)
        print(f'\r  t = {sim.t / 60:6.1f} / {a.minutes:g} min', end='', flush=True)
    print(f'\r  simulated {a.minutes:g} min in {time.perf_counter() - t0:.1f} s (seed {a.seed})\n')
    print(kpi_text(sim))
    print('\n' + summary(sim))
    if a.csv:
        with open(a.csv, 'w', encoding='utf-8') as f:
            f.write(to_csv(sim))
        print(f'telemetry written to {a.csv}')
    if a.html:
        from .plots import comparator_figure, ekf_figure
        win = min(a.minutes, 360)
        parts = [comparator_figure(sim, 'mpc', win).to_html(full_html=False, include_plotlyjs='cdn'),
                 ekf_figure(sim, 'mpc', win).to_html(full_html=False, include_plotlyjs=False)]
        with open(a.html, 'w', encoding='utf-8') as f:
            f.write('<!doctype html><meta charset="utf-8"><title>Food dryer twin report</title>'
                    '<body style="font-family:system-ui,sans-serif;max-width:1400px;margin:auto;padding:16px">'
                    f'<h2>Control-loop comparator</h2>{parts[0]}<h2>EKF soft sensor · MPC loop</h2>{parts[1]}'
                    f'<pre>{kpi_text(sim)}\n\n{summary(sim)}</pre></body>')
        print(f'report written to {a.html}')


if __name__ == '__main__':
    main()
