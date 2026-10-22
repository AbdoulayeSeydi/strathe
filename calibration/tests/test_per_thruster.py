"""#91: per-thruster thrust curves and limits (open since the first URI run): a layout entry's curve_csv, thrust_scale
and max_forward_n / max_reverse_n."""
import numpy as np
import pytest

from thruster_model import thruster_block

LAYOUT = [{"position_m": [0, 0.1, 0], "direction": [1, 0, 0]}, {"position_m": [0, -0.1, 0], "direction": [1, 0, 0]},
          {"position_m": [0, 0, 0], "direction": [0, 0, 1]}]


def test_a_scaled_and_a_capped_thruster():
    lay = [dict(LAYOUT[0]), {**LAYOUT[1], "thrust_scale": 0.5}, {**LAYOUT[2], "max_forward_n": 10.0}]
    cfg = thruster_block({"layout": lay})
    shared = thruster_block({"layout": LAYOUT})
    pwm = np.array([1900.0, 1900.0, 1900.0])
    t, t0 = cfg.thrusts(pwm), shared.thrusts(pwm)
    assert t[0] == pytest.approx(t0[0]) and t[1] == pytest.approx(0.5 * t0[1]) and t[2] == pytest.approx(10.0)
    f = cfg.forward_limits()
    assert f[1] == pytest.approx(0.5 * shared.curve.max_forward_n) and f[2] == 10.0
    # what the thrusters can deliver respects each one's own limit
    w = cfg.achievable_wrench(np.array([0, 0, 100.0, 0, 0, 0]))
    assert w[2] == pytest.approx(10.0)
    assert "per thruster" in __import__("thruster_model").describe(cfg)


def test_a_thruster_with_its_own_table(tmp_path):
    (tmp_path / "big.csv").write_text("pwm_us,thrust_n\n1100,-80\n1500,0\n1900,100\n")
    cfg = thruster_block({"layout": [dict(LAYOUT[0]), dict(LAYOUT[1]), {**LAYOUT[2], "curve_csv": "big.csv"}]},
                         str(tmp_path))
    t = cfg.thrusts(np.array([1700.0, 1700.0, 1700.0]))
    assert t[2] == pytest.approx(50.0) and t[0] == pytest.approx(cfg.curve(1700.0))
    assert cfg.reverse_limits()[2] == 80.0
    with pytest.raises(ValueError, match="can't be rescaled by battery voltage"):
        cfg.wrench_at_voltage(np.array([[1700.0, 1700.0, 1700.0]]), np.array([15.0]))


def test_per_thruster_settings_survive_the_simulation_view_and_are_checked():
    lay = [dict(LAYOUT[0]), {**LAYOUT[1], "thrust_scale": 0.5}, dict(LAYOUT[2])]
    sim = thruster_block({"layout": lay}).for_simulation()
    assert sim.thrust_scales == (1.0, 0.5, 1.0)
    with pytest.raises(ValueError, match="allowed"):
        thruster_block({"layout": [{**LAYOUT[0], "power": 3}]})
    with pytest.raises(ValueError, match="must be positive"):
        thruster_block({"layout": [{**LAYOUT[0], "thrust_scale": 0}]})
