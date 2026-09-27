"""
Clean vs noisy simulation (sensor noise), buoyancy changes, and SAC/TD3 in
the fleet: the models themselves, their wiring into both simulators, the
experiment runners, the command line, and the policy loader.
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import strathe_cli as cli
from arena import Arena
from buoyancy import BuoyancyConfig, with_buoyancy_offset
from build_two_agent_cores import load_calibrated_parameters
from fleet_belief import BeliefSpec
from fleet_env import FleetConfig, FleetSim
from optimizer import initial_parameters, load_vehicle_spec
from sensor_noise import (PRESETS, SensorNoiseConfig, SensorSuite, euler_from_matrix, load_sensor_noise,
                          matrix_from_euler, resolve_sensor_noise)

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
TRUTH = load_calibrated_parameters(os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml"), quiet=True)
SPEC = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml"))
VEHICLE = SPEC.vehicle
GENERIC = initial_parameters(SPEC)
TYPICAL = load_sensor_noise("typical")


# ----------------------------------------------------------------- sensor noise: the model

def test_presets_load_and_describe_themselves():
    for name in PRESETS:
        cfg = load_sensor_noise(name)
        assert cfg.active and cfg.source.endswith(".yaml") and "DVL" in cfg.describe()
    assert load_sensor_noise("tank").position == "external"
    assert TYPICAL.position == "dead_reckoning"
    # the tank preset is what the reference logs contain: millimetres and milliradians
    tank = load_sensor_noise("tank")
    assert tank.velocity_std_mps < 0.01 and tank.position_std_m < 0.01 and tank.heading_std_rad < 0.01


@pytest.mark.parametrize("which,scale", [(None, 1), ("none", 1), ("", 1), ("typical", 0)])
def test_off_means_none(which, scale):
    assert resolve_sensor_noise(which, scale) is None


@pytest.mark.parametrize("sensors,msg", [
    ({"velocity_std_mps": -0.1}, ">= 0"),
    ({"heading_std_rad": 5.0}, "units"),            # degrees given as radians
    ({"depth_std_m": 3.0}, "units"),                 # centimetres given as metres
    ({"velocity_dropout_prob": 0.9}, "between"),
    ({"position": "gps"}, "position must be"),
    ({"velocity_std_mps": float("nan")}, "finite"),
    ({"magic": 1.0}, "unknown sensor-noise keys"),
])
def test_bad_noise_files_are_refused_with_a_reason(tmp_path, sensors, msg):
    p = tmp_path / "n.yaml"
    p.write_text(yaml.safe_dump({"sensors": sensors}))
    with pytest.raises(ValueError, match=msg):
        load_sensor_noise(str(p))


def test_not_a_noise_file_and_missing_file(tmp_path):
    p = tmp_path / "x.yaml"
    p.write_text("roll: {sigma_nm: 1}\n")
    with pytest.raises(ValueError, match="not a sensor-noise file"):
        load_sensor_noise(str(p))
    with pytest.raises(ValueError, match="none, tank, typical"):
        load_sensor_noise(str(tmp_path / "nope.yaml"))
    with pytest.raises(ValueError, match="scale"):
        SensorNoiseConfig(velocity_std_mps=0.01, scale=50).validate()


def test_euler_helpers_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(200):
        r, p, y = rng.uniform(-3, 3), rng.uniform(-1.4, 1.4), rng.uniform(-3, 3)
        R = matrix_from_euler(r, p, y)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        r2, p2, y2 = euler_from_matrix(R)
        np.testing.assert_allclose(matrix_from_euler(r2, p2, y2), R, atol=1e-12)
    from scipy.spatial.transform import Rotation
    R = Rotation.from_euler("xyz", [0.3, -0.2, 1.1]).as_matrix()
    np.testing.assert_allclose(euler_from_matrix(R), [0.3, -0.2, 1.1], atol=1e-12)


def _drive(suite, steps=300, dt=0.1, heading_rate=0.05, speed=0.4):
    """A vehicle turning slowly at constant speed; returns true and measured positions."""
    pos, yaw = np.zeros(3), 0.0
    suite.start(pos, matrix_from_euler(0, 0, yaw), np.array([speed, 0, 0, 0, 0, heading_rate]))
    true, nav = [], []
    for _ in range(steps):
        R = matrix_from_euler(0, 0, yaw)
        pos = pos + R @ np.array([speed, 0, 0]) * dt
        yaw += heading_rate * dt
        m = suite.update(pos, matrix_from_euler(0, 0, yaw), np.array([speed, 0, 0, 0, 0, heading_rate]))
        true.append(pos.copy())
        nav.append(m.position.copy())
    return np.array(true), np.array(nav)


def test_zero_sized_dead_reckoning_is_exact():
    cfg = SensorNoiseConfig(position="dead_reckoning", velocity_std_mps=0.0, scale=1.0)
    true, nav = _drive(SensorSuite(cfg, 0.1, 1))
    np.testing.assert_allclose(nav, true, atol=1e-12)


def test_dead_reckoning_drifts_with_a_bias_and_external_positioning_does_not():
    dr = SensorNoiseConfig(velocity_bias_mps=0.02, heading_bias_rad=0.03, position="dead_reckoning")
    ext = SensorNoiseConfig(velocity_bias_mps=0.02, heading_bias_rad=0.03, position="external", position_std_m=0.01)
    errs_dr, errs_ext = [], []
    for seed in range(20):
        t, n = _drive(SensorSuite(dr, 0.1, seed))
        errs_dr.append(np.linalg.norm(n - t, axis=1))
        t, n = _drive(SensorSuite(ext, 0.1, seed))
        errs_ext.append(np.linalg.norm(n - t, axis=1))
    dr_err, ext_err = np.mean(errs_dr, axis=0), np.mean(errs_ext, axis=0)
    assert dr_err[-1] > 3 * dr_err[29]            # grows with time (drift)
    assert dr_err[-1] > 0.1                        # 30 s at a 2 cm/s + 1.7 deg error: tens of cm
    assert ext_err.max() < 0.05 and abs(ext_err[-1] - ext_err[29]) < 0.01   # bounded, no drift


def test_noise_size_scales_linearly_and_the_random_stream_is_fixed():
    base = dict(velocity_std_mps=0.01, velocity_bias_mps=0.005, gyro_std_rps=0.01, depth_std_m=0.01,
                position="external", position_std_m=0.05)
    errs = []
    for scale in (1.0, 2.0):
        s = SensorSuite(SensorNoiseConfig(**base, scale=scale), 0.1, 7)
        R = np.eye(3)
        s.start(np.zeros(3), R, np.zeros(6))
        m = [s.update(np.zeros(3), R, np.zeros(6)) for _ in range(20)]
        errs.append(np.array([np.r_[x.nu, x.position] for x in m]))
    np.testing.assert_allclose(errs[1], 2 * errs[0], rtol=1e-12, atol=1e-15)


def test_dvl_dropouts_hold_the_last_reading():
    cfg = SensorNoiseConfig(velocity_std_mps=0.01, velocity_dropout_prob=0.5, position="external")
    s = SensorSuite(cfg, 0.1, 3)
    s.start(np.zeros(3), np.eye(3), np.zeros(6))
    v = np.array([s.update(np.zeros(3), np.eye(3), np.zeros(6)).nu[:3] for _ in range(2000)])
    held = np.mean(np.all(v[1:] == v[:-1], axis=1))
    assert 0.45 < held < 0.55


# ----------------------------------------------------------------- sensor noise in the arena

def test_arena_clean_is_unchanged_by_inactive_options():
    a = Arena(TRUTH, VEHICLE)
    b = Arena(TRUTH, VEHICLE, sensor_noise=load_sensor_noise("typical", 0.0), buoyancy=BuoyancyConfig(0.0))
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    rng = np.random.default_rng(0)
    for _ in range(100):
        act = rng.uniform(-1, 1, 6)
        oa, *_ = a.step(act)
        ob, *_ = b.step(act)
        np.testing.assert_array_equal(oa, ob)


@pytest.mark.parametrize("task", ["waypoint", "station_keeping", "docking", "obstacle_avoidance"])
def test_arena_noise_changes_what_the_policy_sees_not_the_task(task):
    clean = Arena(TRUTH, VEHICLE, task=task)
    noisy = Arena(TRUTH, VEHICLE, task=task, sensor_noise=TYPICAL)
    oc, _ = clean.reset(seed=5)
    on, _ = noisy.reset(seed=5)
    np.testing.assert_array_equal(clean._waypoint, noisy._waypoint)   # same target with noise on or off
    assert oc.shape == on.shape
    diffs = []
    for _ in range(60):
        act = np.full(6, 0.2)
        oc, _, _, _, ic = clean.step(act)
        on, _, _, _, inn = noisy.step(act)
        diffs.append(np.abs(oc - on).max())
        np.testing.assert_allclose(ic["position"], inn["position"])    # the true motion is the same
        assert ic["nav_error_m"] == 0.0 and inn["nav_error_m"] >= 0.0
    assert max(diffs) > 1e-3
    assert inn["distance"] == pytest.approx(float(np.linalg.norm(noisy._waypoint - inn["position"])))


def test_arena_noise_on_mujoco():
    a = Arena(TRUTH, VEHICLE, backend="mujoco", sensor_noise=TYPICAL, buoyancy=BuoyancyConfig(-2.0))
    a.reset(seed=1)
    for _ in range(30):
        o, r, t, tr, info = a.step(np.full(6, 0.1))
    assert np.all(np.isfinite(o)) and info["buoyancy_offset_n"] == -2.0 and info["nav_error_m"] > 0


# ----------------------------------------------------------------- buoyancy

def _sink(backend, offset, steps=50):
    a = Arena(TRUTH, VEHICLE, backend=backend, buoyancy=BuoyancyConfig(offset) if offset else None)
    a.reset(seed=1)
    for _ in range(steps):
        o, r, t, tr, info = a.step(np.zeros(6))
    return info["position"][2]


def test_buoyancy_direction_and_engines_agree():
    z_heavy, z_neutral, z_light = (_sink("stand_in", o) for o in (-5.0, 0.0, 5.0))
    assert z_heavy > z_neutral > z_light     # NED: + is deeper, so heavier sinks and lighter floats up
    # MuJoCo integrates the same equations; the two agree to a fraction of a millimetre over 5 s
    assert _sink("mujoco", -5.0) == pytest.approx(z_heavy, abs=2e-3)
    assert _sink("mujoco", 5.0) < _sink("mujoco", 0.0) < _sink("mujoco", -5.0)


def test_buoyancy_offset_is_added_to_the_calibration():
    p = with_buoyancy_offset(TRUTH, -3.0)
    assert p.net_buoyancy_z == pytest.approx(TRUTH.net_buoyancy_z - 3.0)
    assert with_buoyancy_offset(TRUTH, 0.0) is TRUTH
    changed = [k for k in vars(TRUTH) if getattr(p, k) != getattr(TRUTH, k)]
    assert changed == ["net_buoyancy_z"]         # nothing else about the vehicle changes


@pytest.mark.parametrize("offset,msg", [(80.0, "units"), ((5.0, -5.0), "min, max"), ("x", "number"),
                                         (float("inf"), "finite")])
def test_bad_buoyancy_is_refused(offset, msg):
    with pytest.raises(ValueError, match=msg):
        BuoyancyConfig(offset).validate()
    with pytest.raises(ValueError, match="unknown buoyancy keys"):
        BuoyancyConfig.from_dict({"offset": 1})


def test_buoyancy_range_is_drawn_per_episode_and_paired():
    cfg = FleetConfig(buoyancy=BuoyancyConfig((-5.0, 5.0)))
    a = FleetSim(TRUTH, VEHICLE, cfg, belief=BeliefSpec("last_beacon", "last_beacon"))
    b = FleetSim(TRUTH, VEHICLE, cfg, belief=BeliefSpec("physics", "calibrated", TRUTH))
    offs = []
    for seed in range(12):
        a.reset(comm_seed=seed, initial_separation=3.0, closing_speed=0.0)
        b.reset(comm_seed=seed, initial_separation=3.0, closing_speed=0.0)
        assert a.buoyancy_offset_n == b.buoyancy_offset_n           # the same episode, whatever the belief
        offs.append(a.buoyancy_offset_n)
    assert -5 <= min(offs) < -1 and 1 < max(offs) <= 5


@pytest.mark.parametrize("known", [False, True])
def test_buoyancy_known_to_beliefs(known):
    sim = FleetSim(TRUTH, VEHICLE, FleetConfig(buoyancy=BuoyancyConfig(-4.0, known_to_beliefs=known)),
                   belief=BeliefSpec("physics", "calibrated", TRUTH))
    sim.reset(comm_seed=1, initial_separation=3.0, closing_speed=0.0)
    assert sim.cores[0].params.net_buoyancy_z == pytest.approx(TRUTH.net_buoyancy_z - 4.0)
    belief_nb = sim.beliefs[0]._core.params.net_buoyancy_z
    assert belief_nb == pytest.approx(TRUTH.net_buoyancy_z - (4.0 if known else 0.0))
    # back to no change on the next episode of a sim without buoyancy
    sim.cfg.buoyancy = None
    sim.reset(comm_seed=2, initial_separation=3.0, closing_speed=0.0)
    assert sim.cores[0].params is TRUTH and sim.beliefs[0]._core.params is TRUTH


# ----------------------------------------------------------------- sensor noise in the fleet

def _run(sim, seed, steps=80, act=0.15):
    obs = sim.reset(comm_seed=seed, initial_separation=3.0, closing_speed=0.2)
    out, infos = [np.stack(obs)], []
    for k in range(steps):
        obs, r, term, trunc, info = sim.step([np.full(6, act), np.full(6, -act)])
        out.append(np.stack(obs))
        infos.append(info)
        if term or trunc:
            break
    return np.array(out), infos


def test_fleet_clean_unchanged_by_inactive_options():
    a = FleetSim(TRUTH, VEHICLE, FleetConfig())
    b = FleetSim(TRUTH, VEHICLE, FleetConfig(sensor_noise=load_sensor_noise("typical", 0.0),
                                             buoyancy=BuoyancyConfig(0.0)))
    oa, ia = _run(a, 3)
    ob, ib = _run(b, 3)
    np.testing.assert_array_equal(oa, ob)
    assert all(i["nav_errors"] == [0.0, 0.0] for i in ia)


def test_fleet_noise_is_paired_across_beliefs_and_reaches_beacons():
    cfg = FleetConfig(sensor_noise=TYPICAL, beacon_interval_s=0.5)
    a = FleetSim(TRUTH, VEHICLE, cfg, belief=BeliefSpec("last_beacon", "last_beacon"))
    b = FleetSim(TRUTH, VEHICLE, cfg, belief=BeliefSpec("physics", "calibrated", TRUTH))
    _, ia = _run(a, 9)
    _, ib = _run(b, 9)
    for x, y in zip(a._sensors, b._sensors):
        np.testing.assert_array_equal(x.bias_v, y.bias_v)
        assert x.bias_h == y.bias_h
    np.testing.assert_allclose([i["nav_errors"] for i in ia], [i["nav_errors"] for i in ib])  # same actions
    assert max(max(i["nav_errors"]) for i in ia) > 0.01
    # a last-beacon belief is exactly the teammate's believed position at its last beacon, not the truth
    sim = FleetSim(TRUTH, VEHICLE, FleetConfig(sensor_noise=TYPICAL, beacon_interval_s=0.1),
                   belief=BeliefSpec("last_beacon", "last_beacon"))
    sim.reset(comm_seed=1, initial_separation=3.0, closing_speed=0.0)
    for _ in range(5):
        sim.step([np.zeros(6), np.zeros(6)])
    b0 = sim.beliefs[0]._beacon
    assert b0 is not None and np.linalg.norm(b0.position - sim.cores[1].position_array()) > 0


def test_fleet_noise_on_mujoco_with_everything_on():
    from current import CurrentConfig
    from disturbance import DEFAULT_FIT_PATH, load_disturbance
    from fleet_experiment_schema import resolve_thrusters
    cfg = FleetConfig(dynamics_backend="mujoco", sensor_noise=load_sensor_noise("typical", 2.0),
                      buoyancy=BuoyancyConfig((-3.0, 3.0), known_to_beliefs=True),
                      disturbance=load_disturbance(DEFAULT_FIT_PATH, 2.0),
                      current=CurrentConfig(speed_mps=0.2, direction_deg=45.0), thrusters=resolve_thrusters("bluerov2_heavy"))
    sim = FleetSim(TRUTH, VEHICLE, cfg, belief=BeliefSpec("physics", "generic", GENERIC))
    obs, infos = _run(sim, 4, steps=40)
    assert np.all(np.isfinite(obs)) and infos[-1]["buoyancy_offset_n"] != 0.0


# ----------------------------------------------------------------- fitting sensor noise from logs

def test_fit_sensor_noise_recovers_known_noise(tmp_path):
    import fit_sensor_noise as fsn
    rng = np.random.default_rng(0)
    d = tmp_path / "logs"
    d.mkdir()
    t = np.arange(600) * 0.1
    for k in range(3):
        df = pd.DataFrame({
            "x": np.sin(0.1 * t) + rng.normal(0, 0.01, t.size), "y": 0.1 * t + rng.normal(0, 0.01, t.size),
            "z": 1 + 0.01 * t + rng.normal(0, 0.002, t.size),
            "Bu": 0.3 * np.cos(0.2 * t) + rng.normal(0, 0.005, t.size), "Bv": rng.normal(0, 0.005, t.size),
            "Bw": rng.normal(0, 0.005, t.size), "Bp": rng.normal(0, 0.02, t.size), "Bq": rng.normal(0, 0.02, t.size),
            "Br": 0.05 + rng.normal(0, 0.02, t.size), "qx": 0.0, "qy": 0.0, "qz": np.sin(0.025 * t),
            "qw": np.cos(0.025 * t)})
        df.loc[::10, ["Bu", "Bv", "Bw"]] = np.nan            # 10% DVL dropouts
        df.to_csv(d / f"run{k}.csv", index=False)
    out = tmp_path / "n.yaml"
    assert fsn.main(["--data-dir", str(d), "--out", str(out)]) == 0
    cfg = load_sensor_noise(str(out))
    assert cfg.velocity_std_mps == pytest.approx(0.005, rel=0.2)
    assert cfg.gyro_std_rps == pytest.approx(0.02, rel=0.2)
    assert cfg.depth_std_m == pytest.approx(0.002, rel=0.25)
    assert cfg.position == "external" and cfg.position_std_m == pytest.approx(0.01, rel=0.2)
    assert cfg.velocity_dropout_prob == pytest.approx(0.1, abs=0.03)
    assert fsn.main(["--data-dir", str(tmp_path / "empty"), "--out", str(out)]) == 2


# ----------------------------------------------------------------- algorithms: detection and training

@pytest.fixture(scope="module")
def tiny_policies(tmp_path_factory):
    """One tiny SAC and one tiny TD3 fleet policy, trained with the real training code."""
    import train_fleet_policy as tfp
    from stable_baselines3.common.vec_env import VecMonitor
    d = tmp_path_factory.mktemp("pols")
    out = {}
    for alg in ("sac", "td3"):
        env = VecMonitor(tfp.build_training_env(1, 0))
        m = tfp.make_model(alg, env, seed=0, verbose=0)
        m.learning_starts = 50          # take real gradient steps in a short test
        m.learn(200)
        m.save(str(d / f"tiny_{alg}"))
        out[alg] = str(d / f"tiny_{alg}")
    return out


def test_algorithm_is_detected_from_the_file(tiny_policies, tmp_path):
    from policy_io import PolicyLoadError, detect_algorithm, load_policy, resolve_algorithm
    assert detect_algorithm(os.path.join(CAL, "fleet_policy_seed1.zip")) == "ppo"
    assert detect_algorithm(tiny_policies["sac"]) == "sac" and detect_algorithm(tiny_policies["td3"]) == "td3"
    model, alg = load_policy(tiny_policies["td3"])
    assert alg == "td3" and model.predict(np.zeros((2, 13), np.float32), deterministic=True)[0].shape == (2, 6)
    with pytest.raises(PolicyLoadError, match="trained with SAC, not PPO"):
        resolve_algorithm(tiny_policies["sac"], "ppo")
    bad = tmp_path / "bad.zip"
    bad.write_bytes(b"not a zip")
    with pytest.raises(PolicyLoadError, match="isn't a PPO, SAC or TD3"):
        resolve_algorithm(str(bad))
    with pytest.raises(PolicyLoadError, match="algorithm must be"):
        resolve_algorithm(tiny_policies["sac"], "dqn")


def test_sac_td3_policies_meet_the_fleet_contract_and_reload_identically(tiny_policies):
    from validate_fleet_policy import FleetPolicyError, load_fleet_policy, main as validate_main
    rng = np.random.default_rng(0)
    obs = rng.normal(size=(1000, 13)).astype(np.float32)
    for alg, path in tiny_policies.items():
        m1 = load_fleet_policy(path)             # detected
        m2 = load_fleet_policy(path, alg)        # explicit
        a1, a2 = m1.predict(obs, deterministic=True)[0], m2.predict(obs, deterministic=True)[0]
        np.testing.assert_array_equal(a1, a2)
        assert np.all(np.isfinite(a1)) and np.abs(a1).max() <= 1.0
        assert validate_main([path]) == 0
    with pytest.raises(FleetPolicyError, match="not SAC"):
        load_fleet_policy(tiny_policies["td3"], "sac")


@pytest.mark.parametrize("alg", ["sac", "td3"])
def test_train_fleet_policy_cli_for_off_policy_algorithms(tmp_path, monkeypatch, alg):
    import train_fleet_policy as tfp
    out = tmp_path / f"p_{alg}"
    monkeypatch.setattr(sys, "argv", ["train_fleet_policy.py", "--algorithm", alg, "--timesteps", "300",
                                      "--n-sims", "1", "--mode", "noisy", "--buoyancy", "-2", "2", "--out", str(out)])
    tfp.main()
    meta = json.loads((tmp_path / f"p_{alg}.json").read_text(encoding="utf-8"))
    assert meta["algorithm"] == alg and meta["timesteps"] == 300
    assert meta["hyperparameters"]["gradient_steps"] == 4 and meta["hyperparameters"]["net_arch"] == [256, 256]
    td = meta["training_distribution"]
    assert td["mode"] == "noisy" and td["sensor_noise"]["preset"] == "typical"
    assert td["disturbance"]["file"] == "bluerov2_rollpitch.yaml" and td["buoyancy"]["offset_n"] == [-2.0, 2.0]
    from policy_io import detect_algorithm
    assert detect_algorithm(str(out)) == alg


def test_default_timesteps_per_algorithm():
    import train_fleet_policy as tfp
    assert tfp.DEFAULT_TIMESTEPS == {"ppo": 3_000_000, "sac": 1_000_000, "td3": 1_000_000}
    with pytest.raises(ValueError):
        tfp.make_model("dqn", None, 0)


# ----------------------------------------------------------------- statistics

def test_paired_by_averages_seeds_within_a_group_and_pairs_episodes():
    from fleet_stats import paired_by
    rows = []
    for ep in range(10):
        for pol, grp, succ in (("p1", "ppo", 1), ("p2", "ppo", 0), ("s1", "sac", 1)):
            rows.append({"policy": pol, "policy_group": grp, "belief": "b", "comm": "L2", "backend": "stand_in",
                         "episode_key": str(ep), "success": succ, "collision": 0})
    (c,) = paired_by(rows, "policy_group", "sac", "ppo", match=("belief", "comm", "backend"))
    assert c["n_pairs"] == 10 and c["success_a"] == 1.0 and c["success_b"] == 0.5
    assert c["success_diff"] == 0.5 and c["mcnemar_p"] is None     # averaged sides: no McNemar
    rows2 = [dict(r, mode=m, success=int(m == "clean" or int(r["episode_key"]) % 2)) for r in rows[:10]
             for m in ("clean", "noisy")]
    (d,) = paired_by(rows2, "mode", "noisy", "clean", match=("policy", "belief", "comm", "backend"))
    assert d["success_diff"] < 0 and d["mcnemar_p"] is not None and d["success_diff_ci95"][1] <= 0


# ----------------------------------------------------------------- experiment config and runner

def _exp(tmp_path, **over):
    base = {"schema_version": 2, "name": "t", "policies": [{"model_path": os.path.join(CAL, "fleet_policy_seed1")}],
            "beliefs": ["calibrated", "last_beacon"], "compare": {"a": "calibrated", "b": "last_beacon"},
            "communication": {"severities": ["L2"], "beacon_interval_s": [1.0]},
            "initial_conditions": {"separations_m": [3.0], "closing_speeds_mps": [0.0, 0.35]},
            "evaluation": {"comm_seeds": [1], "workers": 1}, "output_dir": str(tmp_path / "out")}
    base.update(over)
    return base


def test_schema_modes_and_blocks(tmp_path):
    from fleet_experiment_schema import FleetExperimentConfig
    os.chdir(CAL)
    c = FleetExperimentConfig(**_exp(tmp_path))
    assert c.modes() == ["clean"] and c.variant("clean") == (None, None, 1.0)
    assert c.policies[0].algorithm == "ppo" and c.policies[0].group == "ppo"
    n = FleetExperimentConfig(**_exp(tmp_path, mode="noisy"))
    noise, dist, scale = n.variant("noisy")
    assert noise.source == "typical_dvl_ahrs.yaml" and dist.endswith("bluerov2_rollpitch.yaml")
    # the defaults are written into the config, so the saved experiment.yaml reruns the same thing
    again = FleetExperimentConfig(**n.model_dump(mode="json", exclude_none=True))
    assert again.variant("noisy")[0].source == noise.source and again.variant("noisy")[1] == dist
    b = FleetExperimentConfig(**_exp(tmp_path, mode="both", sensor_noise={"preset": "tank", "scale": 2},
                                     disturbance={"roll_pitch": "none"}))
    assert b.modes() == ["clean", "noisy"] and b.variant("clean") == (None, None, 0.0)
    assert b.variant("noisy")[0].source == "bluerov2_tank.yaml" and b.variant("noisy")[1] is None
    explicit = FleetExperimentConfig(**_exp(tmp_path, sensor_noise={"preset": "typical"}))
    assert explicit.variant("clean")[0] is not None          # an explicit block applies in clean mode too
    with pytest.raises(Exception, match="none, tank, typical"):
        FleetExperimentConfig(**_exp(tmp_path, sensor_noise={"preset": "loud"}))
    with pytest.raises(Exception, match="units"):
        FleetExperimentConfig(**_exp(tmp_path, buoyancy={"offset_n": 100}))
    with pytest.raises(Exception, match="mode"):
        FleetExperimentConfig(**_exp(tmp_path, mode="sometimes"))


def test_fleet_experiment_both_modes_and_policy_groups(tmp_path, tiny_policies):
    import run_fleet_experiment as rfe
    os.chdir(CAL)
    cfg = _exp(tmp_path, mode="both", buoyancy={"offset_n": [-2, 2]},
               policies=[{"model_path": os.path.join(CAL, "fleet_policy_seed1")},
                         {"model_path": tiny_policies["sac"], "label": "sac_tiny"},
                         {"model_path": tiny_policies["td3"]}])
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump(cfg))
    m = rfe.run_experiment(str(p))
    assert {c["belief"] for c in m["clean_vs_noisy"]} == {"calibrated", "last_beacon"}
    assert {c["a"] for c in m["policy_group_contrasts"]} == {"sac_tiny", "td3"}
    ep = pd.read_csv(tmp_path / "out" / "episodes.csv")
    assert set(ep["mode"]) == {"clean", "noisy"} and set(ep["policy_group"]) == {"ppo", "sac_tiny", "td3"}
    assert (ep.loc[ep["mode"] == "clean", "mean_nav_error_m"] == 0).all()
    assert (ep.loc[ep["mode"] == "noisy", "mean_nav_error_m"] > 0).all()
    # clean and noisy halves see the same buoyancy (it belongs to the vehicle, not the mode)
    key = ["policy", "belief", "comm_base", "episode_key"]
    piv = ep.pivot_table(index=key, columns="mode", values="buoyancy_offset_n")
    np.testing.assert_allclose(piv["clean"], piv["noisy"])
    rep = (tmp_path / "out" / "summary_report.md").read_text(encoding="utf-8")
    assert "Does noise matter? Clean vs noisy" in rep and "Policy groups compared" in rep
    assert "Sensor noise on" in rep and "Buoyancy changed" in rep and "L2@1s noisy" in rep
    saved = yaml.safe_load((tmp_path / "out" / "experiment.yaml").read_text())
    assert saved["mode"] == "both" and saved["sensor_noise"]["preset"] == "typical"


# ----------------------------------------------------------------- the command line

def test_cli_fleet_flags_write_noise_buoyancy_and_labelled_policies(tmp_path, tiny_policies):
    dest = tmp_path / "x.yaml"
    rc = cli.main(["fleet", "--mode", "both", "--noise", "tank", "--noise-scale", "2", "--buoyancy", "-3",
                   "--buoyancy-known-to-beliefs", "--policies", f"mine={tiny_policies['sac']}",
                   os.path.join(CAL, "fleet_policy_seed1"), "--save-config", str(dest)])
    assert rc == 0
    exp = yaml.safe_load(dest.read_text())
    assert exp["mode"] == "both" and exp["sensor_noise"] == {"preset": "tank", "scale": 2.0}
    assert exp["buoyancy"] == {"offset_n": -3.0, "known_to_beliefs": True}
    assert exp["policies"][0]["label"] == "mine" and "label" not in exp["policies"][1]
    for bad, msg in ((["--noise", "loud"], "not found"), (["--buoyancy", "99"], "units"),
                     (["--noise", "typical", "--noise-scale", "-1"], "0 or more"),
                     (["--buoyancy", "1", "2", "3"], "one value")):
        assert cli.main(["fleet", *bad, "--save-config", str(tmp_path / "y.yaml")]) == 2, bad


def test_cli_arena_and_train_fleet_pass_the_new_options(tmp_path):
    calls = []
    params = os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml")
    assert cli.main(["arena", "--algorithm", "td3", "--mode", "noisy", "--noise", "tank", "--buoyancy", "-2", "2",
                     "--disturbance", "none", "--compare-clean-training", "--out", str(tmp_path / "a")],
                    runner=lambda argv: calls.append(argv) or 0) == 0
    a = calls[-1]
    assert a[a.index("--algorithm") + 1] == "td3" and a[a.index("--mode") + 1] == "noisy"
    assert a[a.index("--noise") + 1] == "tank" and a[a.index("--disturbance") + 1] == "none"
    assert a[a.index("--buoyancy") + 1:a.index("--buoyancy") + 3] == ["-2.0", "2.0"]
    assert "--compare-clean-training" in a
    import run_arena_experiment as rae
    assert a[1] == "run_arena_experiment.py"
    rae.build_parser().parse_args(a[2:])          # the script accepts exactly what the CLI passes
    assert cli.main(["train-fleet", "--params", params, "--algorithm", "sac", "--mode", "noisy", "--buoyancy",
                     "-1", "--buoyancy-known-to-beliefs", "--out", str(tmp_path / "p")],
                    runner=lambda argv: calls.append(argv) or 0) == 0
    t = calls[-1]
    assert t[t.index("--algorithm") + 1] == "sac" and "--timesteps" not in t and "--buoyancy-known-to-beliefs" in t
    assert cli.main(["fit-noise", "--data", str(tmp_path), "--out", str(tmp_path / "n.yaml"), "--position",
                     "external"], runner=lambda argv: calls.append(argv) or 0) == 0
    assert calls[-1][1] == "fit_sensor_noise.py" and "external" in calls[-1]


def test_sim_mode_resolution():
    import sim_mode
    assert sim_mode.resolve("clean") == ("none", "none")
    assert sim_mode.resolve("noisy") == ("typical", "fitted")
    assert sim_mode.resolve("noisy", "tank", "none") == ("tank", "none")
    assert sim_mode.resolve("clean", "typical") == ("typical", "none")
    with pytest.raises(ValueError):
        sim_mode.resolve("loud")
    assert sim_mode.buoyancy_from_flags(None) is None and sim_mode.buoyancy_from_flags([0]) is None
    assert sim_mode.buoyancy_from_flags([-1, 1], True).known_to_beliefs


def test_arena_experiment_end_to_end_noisy_sac_both_engines(tmp_path):
    import run_arena_experiment as rae
    out = tmp_path / "arena"
    rc = rae.main(["--algorithm", "sac", "--mode", "noisy", "--compare-clean-training", "--buoyancy", "-2",
                   "--timesteps", "1000", "--seeds", "1", "--episodes", "3", "--backend", "both", "--workers", "1",
                   "--out", str(out)])
    assert rc == 0
    res = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert res["meta"]["algorithm"] == "sac" and res["meta"]["noise"] == "typical"
    assert set(res["noise_trained_vs_clean_trained_by_engine"]) == {"stand_in", "mujoco"}
    ep = pd.read_csv(out / "episodes.csv")
    assert (ep["mean_nav_error_m"] > 0).all() and (ep["buoyancy_offset_n"] == -2.0).all()
    rep = (out / "report.md").read_text(encoding="utf-8")
    assert "Does training with the noise pay off?" in rep and "SAC trained" in rep and "navigation error" in rep
    assert rae.main(["--compare-clean-training", "--timesteps", "1000", "--out", str(tmp_path / "b")]) == 2


@pytest.mark.parametrize("alg", ["ppo", "sac", "td3"])
def test_train_multiseed_takes_noise_and_buoyancy(alg):
    """The specialist single-vehicle training script (Claim B) passes the simulation settings to its Arena."""
    import train_multiseed as tm
    r = tm.train_one_seed(TRUTH, VEHICLE, np.array([1.0, 1.0, 0.5]), 0, 300 if alg != "ppo" else 256, 2,
                          algorithm=alg, env_kwargs={"sensor_noise": TYPICAL, "buoyancy": BuoyancyConfig(-2.0)})
    assert np.isfinite(r["mean_reward"]) and 0 <= r["success_rate"] <= 1
