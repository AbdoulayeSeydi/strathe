"""thruster_model.py -- validated against the real BlueROV2 dataset where it's available."""
import glob
import os

import numpy as np
import pandas as pd
import pytest

from thruster_model import (BLUEROV2_HEAVY_ALLOCATION, BLUEROV2_HEAVY_ALLOCATION_REFERENCE_FRAME, FLU_TO_FRD_WRENCH,
                            KGF, ThrustCurve, ThrusterConfig, allocation_from_geometry, bluerov2_heavy,
                            voltage_matching)

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "Bluerov2_trajectory_dataset", "dataset")
HAVE_DATA = bool(glob.glob(os.path.join(DATA, "*.csv")))


@pytest.mark.skipif(not HAVE_DATA, reason="BlueROV2 dataset not downloaded (data/download_dataset.sh)")
def test_raw_pwm_reproduces_the_datasets_forces_on_every_row():
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DATA, "*.csv")))], ignore_index=True)
    pwm = d[[f"pwm{i}" for i in range(1, 9)]].values
    ref = d[["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]].values
    assert len(d) > 180_000
    # allocation matrix entries are stored to 4 decimals; forces reach ~100 N
    np.testing.assert_allclose(bluerov2_heavy(frame="reference").wrench(pwm), ref, atol=5e-3)
    # the preset is the same forces in FRD: y and z of force and torque flipped (PROJECT_STATUS.md #62)
    np.testing.assert_allclose(bluerov2_heavy().wrench(pwm), ref @ FLU_TO_FRD_WRENCH, atol=5e-3)


@pytest.mark.skipif(not HAVE_DATA, reason="BlueROV2 dataset not downloaded (data/download_dataset.sh)")
def test_reference_dataset_is_ros_flu_enu():
    """Why the preset isn't the as-recovered matrix: the dataset's world z points up (every depth is
    negative) and its body velocities rotate onto the world track with the same up-pointing z."""
    from scipy.spatial.transform import Rotation
    for f in sorted(glob.glob(os.path.join(DATA, "*.csv")))[:4]:
        d = pd.read_csv(f)
        assert d["z"].max() < 0
        vw = Rotation.from_quat(d[["qx", "qy", "qz", "qw"]].values).apply(d[["Bu", "Bv", "Bw"]].values)
        dz = np.gradient(d["z"].values, 0.1)
        assert np.corrcoef(vw[:, 2], dz)[0, 1] > 0.9


# ArduSub's AP_Motors6DOF SUB_FRAME_VECTORED_6DOF (BlueROV2 Heavy) motor factors, FRD/NED:
# (roll, pitch, yaw, throttle, forward, lateral) for a positive motor output. Throttle -1 means a
# positive output pushes the vehicle down (+z).
ARDUSUB_HEAVY_FACTORS = [(0, 0, 1, 0, -1, 1), (0, 0, -1, 0, -1, -1), (0, 0, -1, 0, 1, 1), (0, 0, 1, 0, 1, -1),
                         (1, -1, 0, -1, 0, 0), (-1, -1, 0, -1, 0, 0), (1, 1, 0, -1, 0, 0), (-1, 1, 0, -1, 0, 0)]


def test_preset_matches_ardusub_motor_numbering_in_frd():
    """Each preset column pushes the way ArduSub says that motor pushes, on every axis it uses; the
    as-recovered (FLU) matrix gets sway, heave, pitch and yaw backwards."""
    for i, (roll, pitch, yaw, thr, fwd, lat) in enumerate(ARDUSUB_HEAVY_FACTORS):
        want = np.array([fwd, lat, -thr, roll, pitch, yaw], dtype=float)
        got = np.sign(np.round(BLUEROV2_HEAVY_ALLOCATION[:, i], 3))
        used = want != 0
        np.testing.assert_array_equal(got[used], want[used], err_msg=f"motor {i + 1}")
        assert not np.array_equal(np.sign(np.round(BLUEROV2_HEAVY_ALLOCATION_REFERENCE_FRAME[:, i], 3))[used],
                                  want[used])


def test_curve_shape_and_deadband():
    c = ThrustCurve.t200_bluerov2_dataset()
    assert c(1500) == 0 and c(1472) == 0 and c(1528) == 0
    # direction-corrected convention: PWM above 1500 pushes forward
    assert c(1900) == pytest.approx(47.9, abs=0.05) and c(1100) == pytest.approx(-37.2, abs=0.05)
    assert c(900) == c(1100) and c(2100) == c(1900)  # saturates outside the ESC range


def test_dataset_curve_matches_published_t200_at_about_15_volts():
    """Forward and reverse maxima must independently imply the same voltage."""
    c = ThrustCurve.t200_bluerov2_dataset()
    v_fwd = voltage_matching(c.max_forward_n)
    assert 14.5 < v_fwd < 15.5
    rescaled = c.at_voltage(v_fwd)
    assert rescaled.max_reverse_n == pytest.approx(c.max_reverse_n, rel=0.02)


def test_rescaling_to_published_voltages():
    c = ThrustCurve.t200_bluerov2_dataset()
    c16 = c.at_voltage(16.0)
    assert c16.max_forward_n == pytest.approx(5.25 * KGF) and c16.max_reverse_n == pytest.approx(4.10 * KGF)
    assert c16(1500) == 0
    with pytest.raises(ValueError):
        c.at_voltage(24.0)


def test_direction_mirroring():
    cfg = bluerov2_heavy()
    pwm = np.full(8, 1600.0)
    np.testing.assert_array_equal(cfg.effective_pwm(pwm), [1400, 1400, 1600, 1600, 1400, 1600, 1600, 1400])


def test_neutral_pwm_gives_zero_wrench_and_shapes_broadcast():
    cfg = bluerov2_heavy()
    np.testing.assert_array_equal(cfg.wrench(np.full(8, 1500.0)), np.zeros(6))
    assert cfg.wrench(np.full((5, 8), 1600.0)).shape == (5, 6)
    with pytest.raises(ValueError, match="8 PWM"):
        cfg.wrench(np.full(6, 1500.0))


def test_config_validation():
    c = ThrustCurve.t200_bluerov2_dataset()
    with pytest.raises(ValueError, match="rank"):
        ThrusterConfig(np.zeros((6, 8)), np.ones(8), c)
    with pytest.raises(ValueError, match="directions"):
        ThrusterConfig(BLUEROV2_HEAVY_ALLOCATION, np.ones(7), c)
    with pytest.raises(ValueError, match="6 x n"):
        ThrusterConfig(np.zeros((5, 8)), np.ones(8), c)
    with pytest.raises(ValueError, match="strictly increasing"):
        ThrustCurve(np.array([1500.0, 1400.0]), np.array([0.0, 1.0]))


def test_allocation_from_geometry_reproduces_the_bluerov2_matrix():
    """Positions/directions consistent with the preset (FRD): motor 5 is the front-right vertical
    thruster, at (x, y) = (0.12, 0.218), pushing down (+z) for a positive output; and a 45-degree
    horizontal one."""
    B = allocation_from_geometry([[0.12, 0.218, 0.0]], [[0.0, 0.0, 1.0]])
    np.testing.assert_allclose(B[:, 0], BLUEROV2_HEAVY_ALLOCATION[:, 4], atol=1e-3)
    B = allocation_from_geometry([[0.0, 0.0, 0.0]], [[1.0, 1.0, 0.0]])
    np.testing.assert_allclose(B[:3, 0], [0.7071, 0.7071, 0.0], atol=1e-4)


def test_per_sample_voltage_matches_the_one_voltage_rescale():
    """voltage_column (PROJECT_STATUS.md #63): a log at a steady 13.8 V gives exactly what voltage: 13.8 gives,
    and lower voltage means less thrust in both directions."""
    rng = np.random.default_rng(3)
    pwm = rng.uniform(1100, 1900, (300, 8))
    base, fixed = bluerov2_heavy(), bluerov2_heavy(13.8)
    np.testing.assert_allclose(base.wrench_at_voltage(pwm, np.full(300, 13.8)), fixed.wrench(pwm), atol=1e-9)
    lo = base.wrench_at_voltage(pwm, np.full(300, 12.0))
    hi = base.wrench_at_voltage(pwm, np.full(300, 16.0))
    assert np.all(np.abs(lo[:, 0]) <= np.abs(hi[:, 0]) + 1e-9) and np.abs(hi[:, 0]).sum() > 1.2 * np.abs(lo[:, 0]).sum()


def test_voltage_column_spec_rules():
    from data_ingestion import ThrusterSpec
    ok = ThrusterSpec(pwm_columns=["a"], voltage_column="v", preset="bluerov2_heavy")
    assert ok.voltage_column == "v" and "voltage_column" not in ok.block()
    with pytest.raises(ValueError, match="not both"):
        ThrusterSpec(pwm_columns=["a"], voltage_column="v", preset="bluerov2_heavy", voltage=14)
    with pytest.raises(ValueError, match="curve_csv"):
        ThrusterSpec(pwm_columns=["a"], voltage_column="v", layout=[], curve_csv="c.csv")
