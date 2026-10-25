"""Live Streamlit dashboard for the food-process digital twin.

    pip install -r requirements.txt
    streamlit run app.py
"""
from __future__ import annotations

import math
import random
import time

import pandas as pd
import streamlit as st

from food_twin import EVENT_TYPES, LANE_NAME, LANES, Config, Simulator
from food_twin import plots, schematic
from food_twin.mpc import MPC_DT, N as HORIZON_STEPS, NB
from food_twin.physics import TS
from food_twin.report import kpi_records, summary, to_csv

REFRESH = 1.0          # wall-clock seconds between live updates
DEFAULTS = Config()
TUNING = [  # Config field, label, min, max, step
    ('sp', 'Moisture set-point [% w.b.]', 7.0, 11.0, 0.1),
    ('tol', 'Spec tolerance ± [% w.b.]', 0.4, 2.0, 0.1),
    ('Tmax', 'Max product temperature [°C] (hard constraint)', 52.0, 70.0, 0.5),
    ('Bmax', 'Browning limit [BI]', 15.0, 40.0, 1.0),
    ('wE', 'MPC energy weight', 0.0, 3.0, 0.1),
    ('backoff', 'MPC constraint back-off [°C]', 0.0, 3.0, 0.1),
    ('wDu', 'MPC move suppression', 0.2, 8.0, 0.2),
    ('Kp', 'PID gain Kp [°C per % w.b.]', 2.0, 25.0, 0.5),
    ('Ti', 'PID integral time Ti [s]', 60.0, 900.0, 10.0),
    ('Td', 'PID derivative time Td [s]', 0.0, 120.0, 5.0),
    ('Gfix', 'Fixed airflow for open-loop & PID [kg/s]', 2.5, 8.0, 0.1),
]

st.set_page_config(page_title='Food Dryer Digital Twin', layout='wide')
ss = st.session_state


def new_sim(seed: int) -> Simulator:
    cfg = Config(**{k: float(ss.get('cfg_' + k, getattr(DEFAULTS, k))) for k, *_ in TUNING}, mismatch=ss.get('mis', 5) / 100)
    sim = Simulator(seed, cfg)
    sim.dist.stoch_feed = ss.get('stoch_feed', True)
    sim.dist.weather = ss.get('weather', True)
    sim.dist.noise_scale = ss.get('noise', 1.0)
    sim.run(30)                                   # a few samples so the first render has data
    return sim


def theme() -> str:
    try:
        return 'dark' if st.context.theme.type == 'dark' else 'light'
    except Exception:
        return 'light'


if 'sim' not in ss:
    ss.seed, ss.running, ss.debt = 20261025, True, 0.0
    ss.sim = new_sim(ss.seed)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.subheader('Simulation')
    c1, c2, c3 = st.columns(3)
    if c1.button('Pause' if ss.running else 'Run', type='primary', width='stretch'):
        ss.running = not ss.running
        st.rerun()
    if c2.button('+1 min', width='stretch', help='Advance one simulated minute'):
        ss.sim.run(60)
    if c3.button('Reset', width='stretch', help='Restart all three twins with a new random seed'):
        ss.seed = random.randrange(2 ** 31)
        ss.sim = new_sim(ss.seed)
    sim = ss.sim
    speed = st.select_slider('Speed [simulated min per second]', options=[0.5, 1, 2, 5], value=1, key='speed')
    focus = st.segmented_control('Inspect loop', LANES, format_func=LANE_NAME.get, default='mpc', key='focus') or 'mpc'
    win = st.segmented_control('Chart window', [30, 60, 120], format_func=lambda m: '2 h' if m == 120 else f'{m} min',
                               default=60, key='win') or 60

    st.subheader('Disturbances & sensors')
    sim.dist.stoch_feed = st.toggle('Random feed-moisture fluctuation', True, key='stoch_feed')
    sim.dist.weather = st.toggle('Ambient weather drift', True, key='weather')
    sim.dist.noise_scale = st.slider('Sensor noise × (σ = 0.4 °C / 1.2 % RH)', 0.0, 3.0, 1.0, 0.1, key='noise')
    mis = st.slider('Plant–model mismatch [%]', 0, 15, 5, key='mis')
    if abs(mis / 100 - sim.cfg.mismatch) > 1e-9:
        sim.set_mismatch(mis / 100)

    st.subheader('Targets & tuning')
    for key, label, lo, hi, step in TUNING:
        v = st.slider(label, lo, hi, float(getattr(DEFAULTS, key)), step, key='cfg_' + key)
        if v != getattr(sim.cfg, key):
            if key == 'Gfix':
                sim.set_gfix(v)
            elif key == 'sp':
                sim.set_setpoint(v)
            else:
                setattr(sim.cfg, key, v)
    st.caption(f'Open-loop recipe: T in {sim.cfg.TinOL:.1f} °C at G = {sim.cfg.Gfix:.1f} kg/s')

# ---------------------------------------------------------------- header + disturbance buttons
st.title('Food process digital twin: EKF soft sensor & constrained MPC')
st.caption('A twin-screw extruder feeds wet extrudate onto a hot-air belt dryer. Three identical plant twins run side by side '
           'under the same disturbances and the same sensor noise, one per strategy: open-loop recipe, PID and constrained '
           'nonlinear MPC. Each twin has its own Extended Kalman Filter that infers the hidden core moisture, product temperature '
           'and browning from noisy exhaust-air temperature and humidity.')

st.markdown('**Disturbance injection** · each event hits all three twins at the same moment')
cols = st.columns(len(EVENT_TYPES) + 1)
for col, (kind, ev) in zip(cols, EVENT_TYPES.items()):
    if col.button(ev['label'], help=ev['desc'], width='stretch', key='ev_' + kind):
        sim.dist.inject(kind, sim.t)
if cols[-1].button('Compound upset', help='Wet-feed slug and humidity spike at once', width='stretch', key='ev_combo'):
    sim.dist.inject('wet', sim.t)
    sim.dist.inject('humid', sim.t)
st.caption('Try it: set the fixed airflow for open-loop & PID to 3 kg/s, then trigger a compound upset. '
           'PID overshoots T max and rejects product; MPC holds the limit with no off-spec.')


# ---------------------------------------------------------------- live section
@st.fragment(run_every=REFRESH if ss.running else None)
def live():
    sim = ss.sim
    t_start = time.perf_counter()
    if ss.running:
        per_refresh = speed * 60 * REFRESH / TS
        ss.debt = min(ss.debt + per_refresh, 2 * per_refresh)
        while ss.debt >= 1 and time.perf_counter() - t_start < 0.6 * REFRESH:
            sim.step()
            ss.debt -= 1
    th = theme()
    L = sim.lanes[focus]
    h = L.hist
    active = [f"{EVENT_TYPES[e.type]['short']} ({(e.end - sim.t) / 60:.0f} min left)" for e in sim.dist.events if e.start <= sim.t < e.end]
    st.caption(f"t = {sim.t / 60:.1f} min · {'running' if ss.running else 'paused'} at {speed:g} sim-min/s · seed {ss.seed} · "
               f"active disturbances: {', '.join(active) if active else 'none'}")

    left, right = st.columns([1.55, 1])
    with left:
        st.html(schematic.img_tag(sim, focus, th))
    with right:
        K = L.kpis()
        m1, m2 = st.columns(2)
        m1.metric('Outlet moisture (true)', f"{h['Xw'][-1]:.2f} %", help=f"EKF {h['XwHat'][-1]:.2f} ± {2 * h['XwSd'][-1]:.2f} % w.b.")
        m2.metric('Product temperature', f"{h['Tp'][-1]:.1f} °C", help=f'limit {sim.cfg.Tmax:.1f} °C')
        m3, m4 = st.columns(2)
        m3.metric('Heater + fan', f"{h['Pw'][-1]:.0f} kW", help='so far: ' + (f"{K['spe']:.0f} kJ/kg" if math.isfinite(K['spe']) else '–'))
        m4.metric('Off-spec (cumulative)', f"{K['off']:.1f} %" if math.isfinite(K['off']) else '–', help=f"now: {L.reason or 'in spec'}")
        st.markdown(f'**{LANE_NAME[focus]} loop** · status: ' + (f':red[✕ off-spec, {L.reason}]' if L.reason else ':green[✓ in spec]'))
        dg = sim.lanes['mpc'].mpc.diag
        st.markdown('**MPC solver** (real-time-iteration SQP + ADMM QP)')
        st.dataframe(pd.DataFrame({'': ['Solve time', 'Horizon', 'Tightened limit Tmax − δ − 2σ', 'Predicted peak Tp',
                                        'Active Tp constraints', 'Safety filter', 'Applied T in / G'],
                                   'value': [f"{dg['ms']:.0f} ms", f'{HORIZON_STEPS} × {MPC_DT:.0f} s, {NB} move blocks',
                                             f"{dg['Tlim']:.2f} °C", f"{dg['maxTp']:.2f} °C", f"{dg['active']} of {HORIZON_STEPS}",
                                             'engaged' if dg['filtered'] else 'idle',
                                             f"{sim.lanes['mpc'].u[0]:.1f} °C / {sim.lanes['mpc'].u[1]:.2f} kg/s"]}),
                     hide_index=True, width='stretch')

    st.subheader('Performance KPIs')
    st.dataframe(pd.DataFrame(kpi_records(sim, short=True)), hide_index=True, width='stretch')
    st.caption('Accumulated on the true plant. Lower is better except coverage (target ≈ 95 %); ★ marks the best value. ' + summary(sim))

    st.subheader('Control-loop comparator')
    st.caption('True plant values for all three strategies. Dashed lines right of "now" are the MPC\'s current plan over its 15-minute horizon.')
    st.plotly_chart(plots.comparator_figure(sim, focus, win, th), key='cmp', config={'displayModeBar': False})

    st.subheader(f'EKF soft sensor · {LANE_NAME[focus]} loop')
    st.caption('Hidden states estimated from exhaust-air temperature and RH only. Band = ±2σ from the EKF covariance. Dashed = '
               + ("the MPC's optimal prediction." if focus == 'mpc' else 'model forecast from the EKF state with the current inputs held.'))
    st.plotly_chart(plots.ekf_figure(sim, focus, win, th), key='ekf', config={'displayModeBar': False})

    e = L.ekf
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('**EKF health**')
        st.dataframe(pd.DataFrame({'': ['Innovation Ta / RH', 'NIS (running mean, ≈ 2 expected)', 'σ core moisture',
                                        'σ product temp. / browning', 'σ raw-feed moisture'],
                                   'value': [f'{e.innov[0]:.2f} °C / {e.innov[1]:.2f} %', f'{L.nis_avg:.2f}',
                                             f'{100 / (1 + e.x[0]) ** 2 * e.sd(0):.3f} % w.b.', f'{e.sd(1):.2f} °C / {e.sd(4):.2f} BI',
                                             f'{100 / (1 + e.x[5]) ** 2 * e.sd(5):.2f} % w.b.']}), hide_index=True, width='stretch')
    with c2:
        st.markdown(f'**Telemetry · {LANE_NAME[focus]} loop** (every 30 s, newest first)')
        idx = list(range(len(h['t']) - 1, -1, -6))[:8]
        st.dataframe(pd.DataFrame({
            'min': [h['t'][i] / 60 for i in idx], 'X true': [h['Xw'][i] for i in idx],
            'X EKF ±2σ': [f"{h['XwHat'][i]:.2f} ± {2 * h['XwSd'][i]:.2f}" for i in idx], 'Tp true': [h['Tp'][i] for i in idx],
            'Tp EKF': [h['TpHat'][i] for i in idx], 'Ta meas': [h['TaMeas'][i] for i in idx], 'RH meas': [h['RHMeas'][i] for i in idx],
            'B true': [h['B'][i] for i in idx], 'B EKF': [h['BHat'][i] for i in idx], 'T in': [h['Tin'][i] for i in idx],
            'G': [h['G'][i] for i in idx], 'kW': [h['Pw'][i] for i in idx],
            'status': ['✕ off-spec' if h['offspec'][i] else '✓ ok' for i in idx]}).round(2), hide_index=True, width='stretch')
        st.download_button('Download telemetry CSV (all loops)', data=lambda: to_csv(ss.sim), file_name='food-dryer-twin-telemetry.csv',
                           mime='text/csv', on_click='ignore')


live()

with st.expander('Model, estimator and controller equations'):
    st.markdown(r'''
**Plant (5 states, lumped belt-dryer zone).** Core moisture $X$, product temperature $T_p$, exhaust air temperature $T_a$,
exhaust humidity $Y$, browning index $B$; inputs inlet air temperature $T_{in}$ and dry-air flow $G$.

$$a_w = \frac{r}{1+r},\; r = (X/A)^{1/n} \qquad Y_s = \frac{0.622\,a_w p_{sat}(T_p)}{P - a_w p_{sat}(T_p)} \qquad
R_d = \frac{hA}{c_{pH}}\,\frac{X}{X+X_D}\,(Y_s - Y),\; hA = hA_0\sqrt{G/G_0}$$

$$M_s\dot X = F_s(X_{in}-X) - R_d \qquad M_s(c_{ps}+c_{pw}X)\dot T_p = F_s(c_{ps}+c_{pw}X_{in})(T_{feed}-T_p) + hA(T_a-T_p) - R_d\lambda(T_p)$$

$$M_a c_{pH}\dot T_a = G(c_{pa}+c_{pv}Y_{amb})(T_{in}-T_a) - (hA + R_d c_{pv})(T_a-T_p) - UA(T_a-T_{amb}) \qquad M_a\dot Y = G(Y_{amb}-Y) + R_d$$

$$\dot B = \tfrac{F_s}{M_s}(B_{in}-B) + k_B\,e^{-\frac{E_a}{R}\left(\frac{1}{T_p}-\frac{1}{T_{ref}}\right)}\,e^{-\left(\frac{a_w-0.65}{0.25}\right)^2}$$

True plants: RK4 with 1 s steps, parameters offset by the mismatch slider. MPC predictor: L-stable Rosenbrock (ROS2) for the stiff air/evaporation modes.

**EKF (6 states).** $z=[X,T_p,T_a,Y,B,X_{in}]$, $X_{in}$ a random walk; measurements $T_a$ and exhaust RH with σ = 0.4 °C / 1.2 %.
Finite-difference Jacobians, Joseph-form update. Browning is unobservable but detectable, so its band reflects the kinetic uncertainty.

**Controllers.** Open-loop: steady-state recipe at the fixed airflow. PID: velocity form on the EKF moisture estimate,
manipulating $T_{in}$. MPC: minimises $\sum_k w_X(X_k-X_{sp})^2 + w_E P_k/P_{ref} + w_B\max(0,B_k-B_{max}+1)^2 + w_{\Delta u}\|\Delta z\|^2$
subject to $T_{p,k}\le T_{max}-\delta-2\sigma_{T_p}$ at every step of a 60 × 15 s horizon, with 5 move blocks; one SQP iteration per
sample (real-time iteration) with an ADMM QP, followed by a safety filter that lowers $T_{in}$ until the nonlinear prediction is feasible.

**KPIs.** IAE on true outlet moisture; specific energy = ∫(heater + fan) dt ÷ product mass; off-spec = mass outside the moisture
band, above $B_{max}$ or above $T_{max}$; soft-sensor RMSE and ±2σ coverage skip the first 5 minutes.
''')
