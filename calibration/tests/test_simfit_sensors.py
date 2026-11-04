"""fit-sim and export-sim from a vehicle's own sensors (PROJECT_STATUS.md #96, simfit/sensors.py): a simulator's
window that moves exactly as the truth did reads exactly what the sensors read; the residual vector keeps its length
whatever the simulator does; the score says what the drift was measured against; a physics-free smoother starts the
windows from the right velocity."""
import numpy as np
import pytest

import state_smoother as sm
from simfit import core
from simfit import sensors as SN

from test_measurements import TRUTH, SPEC, as_run, truth_frame


def _run(**kw):
    df, nu, _ = truth_frame(n=700, dvl=False, **kw)
    run = as_run(df, anchors=nu[:, :3])
    return run, nu, df


def test_a_window_that_moves_as_the_truth_reads_what_the_sensors_read():
    run, nu, df = _run(pos_every=10, depth_every=2)
    wins = core.windows(run, 30.0, 30.0)
    for a, n in wins:
        parts = SN.window_parts(run, a, n, nu[a:a + n])
        assert set(parts) == {"gyro.p", "gyro.q", "gyro.r", "main.d", "usbl.n", "usbl.e", "usbl.d"}
        assert max(np.abs(r).max() for r in parts.values()) < 1e-9


def test_a_simulators_heading_error_doesnt_turn_its_path_away_from_the_fixes():
    """FS-real-U1 (#96): over 30 s a simulator's own heading drifts, and a path turned by it read as a surge error."""
    from scipy.spatial.transform import Rotation
    run, nu, df = _run(pos_every=10, depth_every=2)
    wins = core.windows(run, 30.0, 30.0)
    Q = df[["qx", "qy", "qz", "qw"]].values
    turned = (Rotation.from_euler("z", 30, degrees=True) * Rotation.from_quat(Q)).as_quat()
    rw = [(run, wins)]
    sc, lay = SN.scales(rw), SN.layout(rw)[0]
    good = SN.residuals(run, wins, [(nu[a:a + n], Q[a:a + n]) for a, n in wins], sc, lay)
    off = SN.residuals(run, wins, [(nu[a:a + n], turned[a:a + n]) for a, n in wins], sc, lay)
    assert max(np.abs(r).max() for r in off) < 1e-9
    assert all(np.array_equal(x, y) for x, y in zip(good, off))


def test_the_residuals_keep_their_length_and_a_blow_up_is_a_penalty():
    run, nu, df = _run(pos_every=10)
    wins = core.windows(run, 30.0, 30.0)
    rw = [(run, wins)]
    sc, lay = SN.scales(rw), SN.layout(rw)
    Q = df[["qx", "qy", "qz", "qw"]].values
    good = [(nu[a:a + n], Q[a:a + n]) for a, n in wins]
    bad = [(np.full((n, 6), np.nan), Q[a:a + n]) for a, n in wins]
    r1 = np.concatenate(SN.residuals(run, wins, good, sc, lay[0]))
    r2 = np.concatenate(SN.residuals(run, wins, bad, sc, lay[0]))
    assert r1.shape == r2.shape and np.abs(r1).max() < 1e-9 and np.all(r2 == SN.PENALTY)


def test_the_score_is_against_the_fixes_and_beats_standing_still():
    run, nu, df = _run(pos_every=10)
    wins = core.windows(run, 60.0, 30.0)
    Q = df[["qx", "qy", "qz", "qw"]].values
    perfect = core.score([(run, wins)], [[(nu[a:a + n], Q[a:a + n]) for a, n in wins]])
    still = core.baselines([(run, wins)])["no motion"]
    assert perfect["drift_against"] == "usbl"
    h = next(k for k in ("60s", "30s", "10s") if k in perfect)
    assert perfect[h]["drift_median_m"] < 1e-6 < still[h]["drift_median_m"]
    assert np.isnan(perfect[h]["velocity_rmse"][0])            # no DVL: u not measured, said as nan
    assert perfect["sensors"]["usbl.n"]["skill"] == pytest.approx(1.0)


def test_window_rows_start_from_the_estimated_velocity():
    run, nu, _ = _run()
    rows = core.window_rows(run, [(100, 50)])
    assert np.allclose(rows[0][0, :6], np.r_[nu[100, :3], nu[100, 3:6]])


def test_a_physics_free_smoother_finds_the_velocity_from_fixes():
    df, nu, _ = truth_frame(n=1500, dvl=False, pos_every=10, pos_noise=0.05)
    run = as_run(df, sensors=("usbl", "main"))
    sigma, _ = sm.choose_accel_sigma([run], None, None, {"usbl": 0.05, "main": 0.002})
    res = sm.smooth(run, None, None, sigma, {"usbl": 0.05, "main": 0.002})
    err = np.sqrt(np.mean((res.velocity[50:-50] - nu[50:-50, :3]) ** 2))
    # under 10 cm/s with no physics at all, from 1 Hz fixes with 5 cm noise on a vehicle accelerating hard between
    # them (8 cm/s measured); with Strathe's physics the same data gives 2 cm/s (test_measurements), which is why
    # fit-sim / export-sim use a Strathe calibration's physics when they have one
    assert res.fallback is None and err < 0.10
    phys = sm.smooth(run, TRUTH, SPEC.vehicle, 0.03, {"usbl": 0.05, "main": 0.002})
    assert np.sqrt(np.mean((phys.velocity[50:-50] - nu[50:-50, :3]) ** 2)) < err / 2
