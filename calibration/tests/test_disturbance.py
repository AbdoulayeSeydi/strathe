"""disturbance.py and its use in FleetSim (PROJECT_STATUS.md #51)."""
import math
import os
import warnings

import numpy as np
import pytest
import yaml

from disturbance import (DEFAULT_FIT_PATH, AxisDisturbance, DisturbanceConfig, RollPitchDisturbance,
                         episode_seed, load_disturbance, params_sha)
from dynamics_model import CalibrationParameters
from fleet_belief import BeliefSpec
from fleet_env import FleetConfig, FleetSim
from optimizer import load_vehicle_spec

VEHICLE = load_vehicle_spec("vehicle_spec.yaml").vehicle
TRUTH_PATH = "calibrated_parameters_fleet_truth.yaml"
with open(TRUTH_PATH, encoding="utf-8") as _f:
    TRUTH = CalibrationParameters(**yaml.safe_load(_f)["parameters"])


def _cfg(sr=1.0, tr=2.0, sp=0.5, tp=0.5, scale=1.0):
    return DisturbanceConfig(AxisDisturbance(sr, tr), AxisDisturbance(sp, tp), scale=scale)


def test_ou_process_has_the_configured_spread_and_memory():
    d = RollPitchDisturbance(_cfg(sr=1.2, tr=2.0, sp=0.4, tp=0.5), dt=0.1, seed=0)
    x = np.array([d.next() for _ in range(200_000)])
    assert np.all(x[:, [0, 1, 2, 5]] == 0)          # only roll (Tx) and pitch (Ty)
    for col, sigma, tau in ((3, 1.2, 2.0), (4, 0.4, 0.5)):
        s = x[:, col]
        assert s.std() == pytest.approx(sigma, rel=0.05)
        lag1 = np.corrcoef(s[:-1], s[1:])[0, 1]
        assert lag1 == pytest.approx(math.exp(-0.1 / tau), abs=0.01)
        assert abs(s.mean()) < 0.1 * sigma


def test_scale_multiplies_sigma_and_zero_is_inactive():
    a = np.array([RollPitchDisturbance(_cfg(scale=2.0), 0.1, 1).next() for _ in range(1)])
    b = np.array([RollPitchDisturbance(_cfg(scale=1.0), 0.1, 1).next() for _ in range(1)])
    assert np.allclose(a, 2 * b)
    assert not _cfg(scale=0.0).active and _cfg().active


@pytest.mark.parametrize("bad", [dict(sr=-1), dict(tr=0), dict(tp=float("nan")), dict(scale=-0.5)])
def test_invalid_config_is_rejected(bad):
    with pytest.raises(ValueError):
        _cfg(**bad).validate()


def test_episode_seed_depends_only_on_episode_identity():
    s = lambda *a: np.random.default_rng(episode_seed(*a)).integers(1 << 30)
    assert s(3, 0, 2.5, 0.35) == s(3, 0, 2.5, 0.35)
    assert len({s(3, 0, 2.5, 0.35), s(3, 1, 2.5, 0.35), s(4, 0, 2.5, 0.35), s(3, 0, 3.0, 0.35),
                s(3, 0, 2.5, -0.35)}) == 5


def _episode(sim, actions, **reset):
    sim.reset(comm_seed=7, initial_separation=3.0, closing_speed=0.35, **reset)
    out = []
    for a in actions:
        _, _, term, trunc, info = sim.step(a)
        out.append((np.array(info["positions"]), [c.nu_array().copy() for c in sim.cores]))
        if term or trunc:
            break
    return out


# surge/sway/yaw commands only: any roll or pitch motion must come from the disturbance
_A = np.array([0.1, 0.1, 0.0, 0.0, 0.0, 0.1])
ACTIONS = [[_A, -_A]] * 60


def test_disturbance_moves_the_vehicles_and_is_paired_across_belief_models():
    clean = FleetSim(TRUTH, VEHICLE, FleetConfig())
    noisy_a = FleetSim(TRUTH, VEHICLE, FleetConfig(disturbance=_cfg()), belief=BeliefSpec("oracle", "oracle"))
    noisy_b = FleetSim(TRUTH, VEHICLE, FleetConfig(disturbance=_cfg()),
                       belief=BeliefSpec("last_beacon", "last_beacon"))
    c, a, b = _episode(clean, ACTIONS), _episode(noisy_a, ACTIONS), _episode(noisy_b, ACTIONS)
    roll_rate = lambda ep: np.array([nu[0][3] for _, nu in ep])
    # the clean run only rolls a little, through sway/yaw coupling
    assert np.abs(roll_rate(a)).max() > 5 * np.abs(roll_rate(c)).max()
    # same episode, same actions, different belief model -> identical truth trajectory
    for (pa, na), (pb, nb) in zip(a, b):
        assert np.array_equal(pa, pb) and all(np.array_equal(x, y) for x, y in zip(na, nb))


def test_scale_zero_and_none_are_the_clean_simulation():
    c = _episode(FleetSim(TRUTH, VEHICLE, FleetConfig()), ACTIONS)
    z = _episode(FleetSim(TRUTH, VEHICLE, FleetConfig(disturbance=_cfg(scale=0.0))), ACTIONS)
    assert all(np.array_equal(p, q) for (p, _), (q, _) in zip(c, z))


def test_beacons_carry_the_commanded_thrust_not_the_disturbance():
    sim = FleetSim(TRUTH, VEHICLE, FleetConfig(disturbance=_cfg(), beacon_interval_s=0.1))
    sim.reset(comm_seed=1, initial_separation=3.0, closing_speed=0.0)
    sent = []
    orig = sim.channel.send
    sim.channel.send = lambda **kw: (sent.append(kw["payload"].tau.copy()), orig(**kw))[1]
    sim.step([np.full(6, 0.2), np.full(6, -0.2)])
    assert len(sent) == 2
    assert np.allclose(sent[0], 0.2 * sim.cfg.action_bound) and np.allclose(sent[1], -0.2 * sim.cfg.action_bound)


def test_mujoco_and_stand_in_see_the_same_disturbance():
    pytest.importorskip("mujoco")
    si = _episode(FleetSim(TRUTH, VEHICLE, FleetConfig(disturbance=_cfg())), ACTIONS[:30])
    mj = _episode(FleetSim(TRUTH, VEHICLE, FleetConfig(disturbance=_cfg(), dynamics_backend="mujoco")), ACTIONS[:30])
    for (ps, ns), (pm, nm) in zip(si, mj):
        assert np.allclose(ps, pm, atol=5e-3)
        assert np.allclose(ns[0][3:5], nm[0][3:5], atol=5e-3)


def test_load_disturbance_checks_the_file_and_the_parameter_set(tmp_path):
    f = tmp_path / "fit.yaml"
    f.write_text(yaml.safe_dump({"roll": {"sigma_nm": 1.0, "tau_s": 2.0}, "pitch": {"sigma_nm": 0.5, "tau_s": 1.0},
                                 "fitted_for": {"params_sha": params_sha(TRUTH_PATH)}}))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_disturbance(str(f), scale=0.5, truth_params_path=TRUTH_PATH)
    assert cfg.roll.sigma_nm == 1.0 and cfg.scale == 0.5
    with pytest.warns(UserWarning, match="different parameter set"):
        load_disturbance(str(f), truth_params_path="calibrated_parameters_fleet_truth_alt.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"roll": {"sigma_nm": 1.0}}))
    with pytest.raises(ValueError, match="not a roll/pitch disturbance file"):
        load_disturbance(str(bad))


@pytest.mark.skipif(not os.path.exists(DEFAULT_FIT_PATH), reason="shipped fit not present")
def test_shipped_fit_is_for_the_fleet_truth_and_moves_the_motion_toward_the_real_logs():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = load_disturbance(DEFAULT_FIT_PATH, truth_params_path=TRUTH_PATH)
    raw = yaml.safe_load(open(DEFAULT_FIT_PATH, encoding="utf-8"))
    assert cfg.active
    for ax in ("roll", "pitch"):   # held-out: rate spread and 1 s memory closer to the logs than without it
        real, fit, none = raw["real_heldout"][ax], raw["sim_heldout"][ax], raw["sim_heldout_no_disturbance"][ax]
        assert abs(fit["rate_std"] - real["rate_std"]) < abs(none["rate_std"] - real["rate_std"])
        assert abs(fit["rate_acf"][5] - real["rate_acf"][5]) < abs(none["rate_acf"][5] - real["rate_acf"][5])
    # Not asserted: angle spread. With the current truth the model already tilts more than the
    # real vehicle before any disturbance (PROJECT_STATUS.md #54), which a disturbance can't undo.
