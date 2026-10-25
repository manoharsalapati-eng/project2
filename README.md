# Food Process Digital Twin: EKF Soft Sensor & MPC Simulator

`food-digital-twin-mpc.html` is a single-file, dependency-free simulator of an Industry 4.0 food line: a twin-screw extruder feeding wet extrudate onto a hot-air conveyor (belt) dryer. Open it in any modern browser. There is nothing to install or build.

Three identical plant twins run side by side under the **same** disturbances and the **same** sensor noise (common random numbers), so the KPIs compare the strategies fairly:

| Loop | Manipulates | Uses |
|---|---|---|
| Open-loop | nothing (fixed recipe `T_in`, `G`) | steady-state design point |
| PID | inlet air temperature `T_in` (airflow fixed) | EKF moisture estimate |
| Constrained MPC | `T_in` **and** airflow `G` | full EKF state + disturbance estimate |

## What is inside (all pure JavaScript)

1. **Non-linear plant model** with five coupled states: core moisture `X`, product temperature `T_p`, exhaust air temperature `T_a`, exhaust humidity `Y` and Maillard browning index `B`. It includes an Oswin sorption isotherm, psychrometrics, a Lewis-analogy mass transfer coefficient, a falling-rate (internal diffusion) factor, and Arrhenius plus water-activity browning kinetics. The true plants differ from the controller model by an adjustable plant–model mismatch.
2. **Extended Kalman Filter soft sensor** (6 states, including the unmeasured raw-feed moisture as a random walk). It uses only noisy exhaust-air **temperature** and **RH**, and reports estimates with ±2σ covariance bounds, NIS and coverage diagnostics. Browning is unobservable but detectable, and its covariance is shown honestly.
3. **Constrained nonlinear MPC**: a 15-minute horizon (60 × 15 s) with 5 move blocks, an economic objective (energy + moisture tracking + soft browning limit + move suppression), and a **hard** product-temperature limit tightened by the EKF's 2σ. It is solved by real-time-iteration SQP with an ADMM (OSQP-style) QP, followed by a safety filter that guarantees the nonlinear prediction is feasible. The stiff air/evaporation modes are integrated with an L-stable Rosenbrock (ROS2) method.
4. **Disturbances**: random feed-moisture and weather fluctuations, plus injection buttons for a wet-feed slug, an ambient humidity spike, a throughput surge, a cold snap, a sensor-noise burst and a compound upset.
5. **Live telemetry**: 12 streaming charts with synchronized crosshair tooltips, MPC plan and forecast overlays, a process schematic, KPI scoreboard (IAE, specific energy in kJ/kg, kJ/kg water, off-spec rejection %, max product temperature, time above `T_max`, soft-sensor RMSE and coverage), a telemetry table and CSV export.

## Things to try

- Let it run for about 30 simulated minutes. The MPC lowers the airflow and raises the inlet temperature until product temperature rides just under the tightened limit. That cuts specific energy by roughly 5–10 % versus PID at the default settings, with lower moisture IAE.
- Set **Fixed airflow for open-loop & PID** to 3 kg/s and press **Compound upset**. PID overshoots `T_max` and rejects product; MPC holds the limit with no off-spec.
- Raise the **plant–model mismatch** or **sensor noise** and watch the EKF bands widen and the MPC back off further from the limit.
