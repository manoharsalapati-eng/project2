# Food Process Digital Twin: EKF Soft Sensor & MPC Simulator

A digital twin of an Industry 4.0 food line: a twin-screw extruder feeding wet extrudate onto a hot-air conveyor (belt) dryer. Three identical plant twins run side by side under the **same** disturbances and the **same** sensor noise, so the KPIs compare the control strategies fairly:

| Loop | Manipulates | Uses |
|---|---|---|
| Open-loop | nothing (fixed recipe `T_in`, `G`) | steady-state design point |
| PID | inlet air temperature `T_in` (airflow fixed) | EKF moisture estimate |
| Constrained MPC | `T_in` **and** airflow `G` | full EKF state + disturbance estimate |

The project ships in two equivalent forms:

- **Python** (`food_twin/` package + `app.py` Streamlit dashboard). The physics, EKF, PID and MPC are written by hand in NumPy, with no control or optimisation libraries.
- **Browser** (`food-digital-twin-mpc.html`): a single dependency-free file. Open it in any browser.

Both use the same model, tuning and random-number generator, so a given seed produces the same disturbances and sensor noise in both.

## Python quick start

```bash
pip install -r requirements.txt
streamlit run app.py                      # live dashboard
python -m food_twin                       # headless run: KPI table in the terminal
python -m food_twin --gfix 3 --events wet@20,humid@20 --minutes 60 --csv run.csv --html run.html
python -m pytest -q                       # tests (pip install pytest)
```

The dashboard streams the process schematic, the KPI scoreboard, six comparator charts (all three loops plus the MPC's 15-minute plan), six EKF soft-sensor charts (hidden states vs estimates with ±2σ bands), solver and EKF diagnostics, and a telemetry table with CSV export. Disturbances can be injected from buttons; noise, plant–model mismatch, targets and PID/MPC tuning are sidebar sliders.

### Package layout

| Module | Contents |
|---|---|
| `food_twin/physics.py` | 5-state non-linear dryer model (moisture, product temperature, exhaust air temperature and humidity, Maillard browning), psychrometrics, RK4 and an L-stable Rosenbrock (ROS2) integrator. All functions are vectorised over batch dimensions. |
| `food_twin/disturbances.py` | Ornstein–Uhlenbeck feed-moisture and weather fluctuations, injectable events, shared sensor-noise draws |
| `food_twin/ekf.py` | 6-state Extended Kalman Filter soft sensor (adds the unmeasured feed moisture as a random walk) using exhaust T and RH only. Joseph-form update. |
| `food_twin/pid.py` | Velocity-form PID with derivative on the measurement |
| `food_twin/mpc.py` | Nonlinear MPC: 60 × 15 s horizon, 5 move blocks, energy + tracking objective, hard product-temperature limit tightened by the EKF's 2σ. Solved by real-time-iteration SQP with an ADMM (OSQP-style) QP, plus a safety filter on the nonlinear prediction. |
| `food_twin/simulator.py` | The three twins, KPIs and history |
| `food_twin/plots.py`, `schematic.py`, `report.py`, `cli.py` | Plotly figures, the SVG schematic, the KPI table / CSV, and the command line |

Speed: one MPC solve takes about 30 ms, and the CLI simulates 150 minutes in about 22 s in the test environment. The MPC builds all finite-difference sensitivities and line-search candidates as single batched NumPy integrations.

### Example results (`python -m food_twin`, seed 12345)

Wet-feed slug, humidity spike, throughput surge and cold snap over 150 min, at the default 4 kg/s airflow for open-loop and PID:

| | IAE (% w.b.·min) | kJ/kg product | Off-spec | Peak product temp. (limit 60 °C) |
|---|---|---|---|---|
| Open-loop | 86.4 | 1,473 | 18.3 % | 54.6 °C |
| PID | 48.5 | 1,521 | 5.7 % | 56.6 °C |
| MPC | 34.7 | 1,392 | 0.0 % | 58.6 °C |

Compound upset (wet feed + humid air together) with open-loop and PID at 3 kg/s: PID peaks at 63.0 °C, spends 7.7 min above the limit and rejects 32 % of product. MPC stays at 58.6 °C with no off-spec.

## Things to try

- Let it run for about 30 simulated minutes. The MPC lowers the airflow and raises the inlet temperature until product temperature rides just under the tightened limit. That cuts specific energy by roughly 5–10 % versus PID at the default settings, with lower moisture IAE.
- Set **Fixed airflow for open-loop & PID** to 3 kg/s and press **Compound upset**. PID overshoots `T_max` and rejects product; MPC holds the limit with no off-spec.
- Raise the **plant–model mismatch** or **sensor noise** and watch the EKF bands widen and the MPC back off further from the limit.

## Modelling notes

- The dryer is a lumped (well-mixed) zone with plausible but uncalibrated parameters. It is not a distributed belt model.
- Browning does not affect either exhaust measurement. It is unobservable but detectable: the EKF propagates it through the model, and its ±2σ band reflects the kinetic uncertainty.
- The Python MPC defaults to one SQP iteration per sample with 15 s Rosenbrock steps. The browser version does two iterations with 7.5 s steps. In closed loop the difference is within a few percent on every KPI, and `--sqp 2 --substeps 2` gets close to the browser settings.
