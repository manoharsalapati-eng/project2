"""Regression tests for the physics, estimator and controllers (python -m pytest -q)."""
import numpy as np
import pytest

from food_twin import Config, Simulator
from food_twin.cli import main as cli_main, parse_events
from food_twin.disturbances import mulberry32
from food_twin.physics import NOMINAL, NOMINAL_D, recipe_tin, rk4, ros2, steady_state, to_wb


def test_recipe_hits_setpoint_at_steady_state():
    tin = recipe_tin(5.0, 9.0)
    assert tin == pytest.approx(96.0, abs=0.2)
    x = steady_state(np.array([tin, 5.0]))
    assert to_wb(x[0]) == pytest.approx(9.0, abs=0.01)
    assert 45 < x[1] < 55            # product near wet-bulb-ish temperature at the design point


def test_rosenbrock_tracks_fine_rk4_through_stiff_transients():
    x0 = steady_state(np.array([96.0, 5.0]))
    a, b, err = x0.copy(), x0.copy(), np.zeros(5)
    for k in range(60):
        u = np.array([[150, 2.5], [60, 8], [200, 2], [120, 5]][k // 15], float)
        a = rk4(a, u, NOMINAL_D, NOMINAL, 0.25, 60)
        b = ros2(b, u, NOMINAL_D, NOMINAL, 7.5, 2)
        err = np.maximum(err, np.abs(a - b))
    assert err[0] < 1e-3 and err[1] < 0.6 and err[2] < 1.2


def test_prng_is_bit_exact_with_browser_version():
    r = mulberry32(12345)
    assert [round(r(), 12) for _ in range(3)] == [0.979728267761, 0.3067522645, 0.484205421526]


def test_ekf_soft_sensor_converges_and_is_consistent():
    sim = Simulator(7, Config(mismatch=0.0))
    sim.run(40 * 60)
    k = sim.lanes['pid'].kpis()
    assert k['rmse'] < 0.15                 # % w.b.
    assert k['cov'] > 85                    # truth inside the 2-sigma band most of the time


def test_mpc_holds_temperature_limit_where_pid_does_not():
    sim = Simulator(12345, Config(Gfix=3.0))
    sim.run(60 * 60, parse_events('wet@20,humid@20'))
    mpc, pid = sim.lanes['mpc'].kpis(), sim.lanes['pid'].kpis()
    assert mpc['above'] == 0 and mpc['tmax'] < sim.cfg.Tmax
    assert pid['above'] > 0
    assert mpc['off'] < pid['off'] and mpc['iae'] < pid['iae']


def test_cli_writes_csv(tmp_path, capsys):
    out = tmp_path / 'run.csv'
    cli_main(['--minutes', '3', '--csv', str(out)])
    lines = out.read_text().splitlines()
    assert lines[0].startswith('t_s,') and len(lines) == 1 + 36
    assert 'MPC' in capsys.readouterr().out
