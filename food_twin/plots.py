"""Plotly figures shared by the Streamlit dashboard and the CLI's HTML report."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .disturbances import EVENT_TYPES
from .mpc import MPC_DT, N as HORIZON_STEPS
from .simulator import LANE_NAME, LANES, Simulator

# validated categorical slots (blue / orange / aqua) + violet for the estimator; dark steps chosen for the dark surface
PALETTE = {
    'light': dict(ol='#2a78d6', pid='#eb6834', mpc='#1baf7a', est='#6250d6', truth='#0b0b0b', meas='#898781', crit='#d03b3b',
                  good='#0ca30c', ink2='#52514e', grid='#e1e0d9', event='rgba(11,11,11,0.06)', forecast='rgba(11,11,11,0.035)',
                  spec='rgba(12,163,12,0.08)', surface='#fcfcfb', template='plotly_white'),
    'dark': dict(ol='#3987e5', pid='#d95926', mpc='#199e70', est='#9085e9', truth='#f0efec', meas='#898781', crit='#d03b3b',
                 good='#0ca30c', ink2='#c3c2b7', grid='#2c2c2a', event='rgba(255,255,255,0.06)', forecast='rgba(255,255,255,0.04)',
                 spec='rgba(12,163,12,0.10)', surface='#1a1a19', template='plotly_dark'),
}
HORIZON = HORIZON_STEPS * MPC_DT


def _window(sim: Simulator, win_min: float):
    now = sim.t
    t0 = max(0.0, now - win_min * 60)
    t1 = max(win_min * 60, now) + HORIZON
    return t0, t1, now


def _slice(h: dict, t0: float):
    t = np.asarray(h['t'])
    i0 = max(0, int(np.searchsorted(t, t0)) - 1)
    return i0, t[i0:] / 60.0


def _decorate(fig, sim: Simulator, t0, t1, now, C, rows, cols):
    fig.update_annotations(font_size=12)          # subplot titles and limit labels
    groups: dict[int, list] = {}
    for ev in sim.dist.events:
        if ev.end >= t0 and ev.start <= now:
            groups.setdefault(round(ev.start / 60), []).append(ev)
            fig.add_vrect(x0=max(ev.start, t0) / 60, x1=min(ev.end, now) / 60, fillcolor=C['event'], line_width=0, layer='below',
                          row='all', col='all')
    for evs in groups.values():          # one label per start time, on the first panel only
        label = ' + '.join(dict.fromkeys(EVENT_TYPES[e.type]['short'] for e in evs))
        fig.add_annotation(x=max(evs[0].start, t0) / 60, y=1, xref='x', yref='y domain', text=label, showarrow=False,
                           xanchor='left', yanchor='top', font=dict(size=10, color=C['meas']))
    fig.add_vrect(x0=now / 60, x1=min(t1, now + HORIZON) / 60, fillcolor=C['forecast'], line_width=0, layer='below', row='all', col='all')
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.add_vline(x=now / 60, line_width=1, line_color=C['meas'], row=r, col=c)
    fig.update_xaxes(range=[t0 / 60, t1 / 60], showgrid=False, ticksuffix=' min', zeroline=False)
    fig.update_yaxes(gridcolor=C['grid'], zeroline=False)


def _layout(fig, C, height):
    fig.update_layout(template=C['template'], height=height, margin=dict(l=10, r=10, t=56, b=10), hovermode='x unified',
                      legend=dict(orientation='h', yanchor='bottom', y=1.06, x=0, font_size=12), font=dict(size=12),
                      plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    return fig


def comparator_figure(sim: Simulator, focus: str = 'mpc', win_min: float = 60, theme: str = 'light') -> go.Figure:
    """True plant values for all three strategies, with the MPC's current plan to the right of 'now'."""
    C = PALETTE[theme]
    cfg = sim.cfg
    panels = [('Xw', 'Outlet moisture · % w.b.', 2), ('Tp', 'Product temperature · °C', 1), ('B', 'Browning index · BI', 1),
              ('Pw', 'Heater + fan power · kW', 0), ('Tin', 'Inlet air temperature (MV 1) · °C', 1), ('G', 'Drying-air flow (MV 2) · kg/s', 2)]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[p[1] for p in panels], vertical_spacing=0.16, horizontal_spacing=0.06)
    t0, t1, now = _window(sim, win_min)
    mpc = sim.lanes['mpc']
    plan = mpc.mpc.plan
    order = [l for l in LANES if l != focus] + [focus]
    for n, (key, _, dec) in enumerate(panels):
        r, c = n // 3 + 1, n % 3 + 1
        for lane in order:
            h = sim.lanes[lane].hist
            i0, tm = _slice(h, t0)
            fig.add_trace(go.Scatter(x=tm, y=h[key][i0:], mode='lines', name=LANE_NAME[lane], legendgroup=lane, showlegend=n == 0,
                                     line=dict(color=C[lane], width=2.4 if lane == focus else 1.6),
                                     hovertemplate=f'%{{y:.{dec}f}}'), row=r, col=c)
        if plan is not None:
            is_input = key in ('Tin', 'G')
            k = np.arange(HORIZON_STEPS) + (0 if is_input else 1)
            fig.add_trace(go.Scatter(x=(mpc.plan_t + k * MPC_DT) / 60, y=plan[key], mode='lines', name='MPC plan', legendgroup='plan',
                                     showlegend=n == 0, line=dict(color=C['mpc'], width=1.8, dash='dash', shape='hv' if is_input else 'linear'),
                                     hovertemplate=f'%{{y:.{dec}f}}'), row=r, col=c)
    fig.add_hrect(y0=cfg.sp - cfg.tol, y1=cfg.sp + cfg.tol, fillcolor=C['spec'], line_width=0, layer='below', row=1, col=1)
    fig.add_hline(y=cfg.sp, line_width=1, line_color=C['ink2'], row=1, col=1)
    fig.add_hline(y=cfg.Tmax, line_width=1.6, line_color=C['crit'], row=1, col=2, annotation_text=f'T max {cfg.Tmax:.1f} °C',
                  annotation_position='top left', annotation_font_size=10)
    if np.isfinite(mpc.mpc.diag['Tlim']):
        fig.add_hline(y=mpc.mpc.diag['Tlim'], line_width=1, line_dash='dot', line_color=C['meas'], row=1, col=2)
    fig.add_hline(y=cfg.Bmax, line_width=1.6, line_color=C['crit'], row=1, col=3, annotation_text=f'B max {cfg.Bmax:.0f}',
                  annotation_position='top left', annotation_font_size=10)
    _decorate(fig, sim, t0, t1, now, C, 2, 3)
    return _layout(fig, C, 640)


def lane_forecast(sim: Simulator, lane: str):
    """MPC lane: its optimal predicted trajectory. OL / PID: model forecast with current inputs held."""
    L = sim.lanes[lane]
    k = np.arange(1, HORIZON_STEPS + 1)
    if lane == 'mpc':
        if L.mpc.plan is None:
            return None
        p = L.mpc.plan
        return dict(t=(L.plan_t + k * MPC_DT) / 60, Xw=p['Xw'], Tp=p['Tp'], B=p['B'], label='MPC prediction')
    pr = sim.hold_forecast(L)
    return dict(t=(sim.t + k * MPC_DT) / 60, Xw=pr.Xw[0], Tp=pr.Tp[0], B=pr.B[0], label='Hold-input forecast')


def ekf_figure(sim: Simulator, lane: str = 'mpc', win_min: float = 60, theme: str = 'light') -> go.Figure:
    """EKF soft-sensor view of one loop: hidden states vs estimates with +/-2 sigma bands."""
    C = PALETTE[theme]
    cfg = sim.cfg
    panels = [('Xw', 'XwHat', 'XwSd', None, 'Core moisture X (hidden) · % w.b.', 2),
              ('Tp', 'TpHat', 'TpSd', None, 'Product temperature Tp (hidden) · °C', 1),
              ('B', 'BHat', 'BSd', None, 'Browning index B (unmeasured quality)', 1),
              ('Xin', 'XinHat', 'XinSd', None, 'Raw-feed moisture Xin (unmeasured) · % w.b.', 2),
              ('Ta', 'TaHat', None, 'TaMeas', 'Exhaust air temperature (measured) · °C', 2),
              ('RH', 'RHHat', None, 'RHMeas', 'Exhaust air RH (measured) · %', 1)]
    fig = make_subplots(rows=2, cols=3, subplot_titles=[p[4] for p in panels], vertical_spacing=0.16, horizontal_spacing=0.06)
    t0, t1, now = _window(sim, win_min)
    h = sim.lanes[lane].hist
    i0, tm = _slice(h, t0)
    fc = lane_forecast(sim, lane)
    shown = set()

    def once(name):
        first = name not in shown
        shown.add(name)
        return first

    for n, (kt, ke, ks, km, _, dec) in enumerate(panels):
        r, c = n // 3 + 1, n % 3 + 1
        y = np.asarray(h[ke][i0:])
        if ks:
            sd2 = 2 * np.asarray(h[ks][i0:])
            fig.add_trace(go.Scatter(x=np.concatenate([tm, tm[::-1]]), y=np.concatenate([y + sd2, (y - sd2)[::-1]]), fill='toself',
                                     fillcolor=C['est'], opacity=0.18, line_width=0, name='±2σ', legendgroup='band', showlegend=once('band'),
                                     hoverinfo='skip'), row=r, col=c)
        if km:
            fig.add_trace(go.Scatter(x=tm, y=h[km][i0:], mode='markers', marker=dict(color=C['meas'], size=3, opacity=0.6), name='Measured (noisy)',
                                     legendgroup='meas', showlegend=once('meas'), hovertemplate=f'%{{y:.{dec}f}}'), row=r, col=c)
        fig.add_trace(go.Scatter(x=tm, y=h[kt][i0:], mode='lines', line=dict(color=C['truth'], width=1.5), opacity=0.85, name='True plant',
                                 legendgroup='true', showlegend=once('true'), hovertemplate=f'%{{y:.{dec}f}}'), row=r, col=c)
        fig.add_trace(go.Scatter(x=tm, y=y, mode='lines', line=dict(color=C['est'], width=2), name='EKF estimate', legendgroup='ekf',
                                 showlegend=once('ekf'), hovertemplate=f'%{{y:.{dec}f}}'), row=r, col=c)
        if fc is not None and kt in fc:
            fig.add_trace(go.Scatter(x=fc['t'], y=fc[kt], mode='lines', line=dict(color=C['est'], width=1.8, dash='dash'), name=fc['label'],
                                     legendgroup='fc', showlegend=once('fc'), hovertemplate=f'%{{y:.{dec}f}}'), row=r, col=c)
    fig.add_hline(y=cfg.sp, line_width=1, line_color=C['ink2'], row=1, col=1)
    fig.add_hline(y=cfg.Tmax, line_width=1.6, line_color=C['crit'], row=1, col=2)
    fig.add_hline(y=cfg.Bmax, line_width=1.6, line_color=C['crit'], row=1, col=3)
    fig.update_yaxes(rangemode='tozero', row=2, col=3)
    _decorate(fig, sim, t0, t1, now, C, 2, 3)
    return _layout(fig, C, 640)
