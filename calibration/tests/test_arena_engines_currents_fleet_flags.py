"""
The single-vehicle arena on MuJoCo and with the roll/pitch disturbance; fleet
experiments set entirely from flags; per-recording water currents in
calibration and their estimation from the logs.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import strathe_cli as cli
from arena import Arena
from build_two_agent_cores import load_calibrated_parameters
from current import CurrentConfig
from disturbance import DEFAULT_FIT_PATH, load_disturbance
from optimizer import initial_parameters, load_vehicle_spec

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
TRUTH = load_calibrated_parameters(os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml"), quiet=True)
SPEC = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml"))
DIST = load_disturbance(DEFAULT_FIT_PATH)


# ----------------------------------------------------------------- arena: engines and disturbance

def test_arena_mujoco_matches_the_stand_in():
    a = Arena(TRUTH, SPEC.vehicle, backend="stand_in")
    b = Arena(TRUTH, SPEC.vehicle, backend="mujoco")
    oa, _ = a.reset(seed=4)
    ob, _ = b.reset(seed=4)
    np.testing.assert_allclose(oa, ob, atol=1e-9)
    rng = np.random.default_rng(0)
    for _ in range(150):
        act = rng.uniform(-1, 1, 6)
        ra, rb = a.step(act), b.step(act)
        if ra[2] or ra[3]:
            break
    assert np.abs(ra[4]["position"] - rb[4]["position"]).max() < 0.05
    with pytest.raises(ValueError, match="backend must be"):
        Arena(TRUTH, SPEC.vehicle, backend="bullet")


def test_disturbance_changes_motion_but_not_targets_and_is_seeded():
    clean = Arena(TRUTH, SPEC.vehicle, task="station_keeping")
    rough = Arena(TRUTH, SPEC.vehicle, task="station_keeping", disturbance=DIST)
    rough2 = Arena(TRUTH, SPEC.vehicle, task="station_keeping", disturbance=DIST)
    clean.reset(seed=9), rough.reset(seed=9), rough2.reset(seed=9)
    np.testing.assert_array_equal(clean._waypoint, rough._waypoint)  # same targets with it on or off
    rolls, pushes = [], []
    for _ in range(200):
        a = np.zeros(6)
        _, _, _, _, i1 = rough.step(a)
        _, _, _, _, i2 = rough2.step(a)
        _, _, _, _, i0 = clean.step(a)
        np.testing.assert_array_equal(i1["disturbance"], i2["disturbance"])  # same seed, same torques
        assert i0["disturbance"] is None
        pushes.append(i1["disturbance"])
        rolls.append(rough._core.state.orientation.as_euler("xyz")[0])
    pushes = np.array(pushes)
    assert np.all(pushes[:, [0, 1, 2, 5]] == 0) and pushes[:, 3].std() > 0.3 and pushes[:, 4].std() > 0.3
    assert np.degrees(np.std(rolls)) > 1.0  # it actually rocks the vehicle


@pytest.mark.parametrize("backend", ["stand_in", "mujoco"])
def test_everything_on_at_once_survives_full_scale_random_commands(backend):
    from fleet_experiment_schema import resolve_thrusters
    rough = load_disturbance(DEFAULT_FIT_PATH, scale=3.0)
    env = Arena(TRUTH, SPEC.vehicle, task="docking", backend=backend, disturbance=rough,
                current=CurrentConfig(speed_mps=(0.0, 1.0), direction_deg=(0, 360), vertical_mps=0.1,
                                      speed_std_mps=0.1, direction_std_deg=30, time_scale_s=5),
                thrusters=resolve_thrusters("bluerov2_heavy"))
    rng = np.random.default_rng(1)
    env.reset(seed=1)
    for _ in range(600):
        obs, r, term, trunc, info = env.step(np.sign(rng.normal(size=6)))
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        if term or trunc:
            env.reset()


def test_gym_checks_pass_on_mujoco_with_disturbance():
    from gymnasium.utils.env_checker import check_env
    check_env(Arena(TRUTH, SPEC.vehicle, backend="mujoco", disturbance=DIST), skip_render_check=True)


def test_arena_cli_passes_engine_and_disturbance(tmp_path, capsys):
    calls = []
    assert cli.main(["arena", "--backend", "both", "--disturbance", "fitted", "--disturbance-scale", "2",
                     "--out", str(tmp_path / "a")], runner=lambda c: calls.append(c) or 0) == 0
    a = calls[0]
    assert a[a.index("--backend") + 1] == "both" and a[a.index("--disturbance") + 1] == "fitted"
    assert a[a.index("--disturbance-scale") + 1] == "2.0"
    assert cli.main(["arena", "--disturbance", str(tmp_path / "nope.yaml")], runner=lambda c: 0) == 2
    assert "disturbance file not found" in capsys.readouterr().out


# ----------------------------------------------------------------- fleet: every setting as a flag

def test_fleet_flags_build_a_valid_experiment(tmp_path, monkeypatch):
    from fleet_experiment_schema import load_fleet_experiment
    monkeypatch.setattr(cli, "JOBS_DIR", str(tmp_path))
    out = tmp_path / "mine.yaml"
    rc = cli.main(["fleet", "--comm", "L0", "L3b", "--beacon-interval", "0.5", "2", "--beliefs", "calibrated",
                   "last_beacon", "oracle", "--compare", "calibrated", "last_beacon", "--target-separation", "3",
                   "--tolerance", "0.3", "--hold-steps", "25", "--episode-steps", "300", "--collision-radius", "0.6",
                   "--separations", "3.5", "5", "--closing-speeds", "-0.2", "0.2", "--seeds", "3", "--backend",
                   "both", "--disturbance", "fitted", "--disturbance-scale", "1.5", "--name", "my study",
                   "--save-config", str(out)], runner=lambda c: 0)
    assert rc == 0
    cwd = os.getcwd()
    os.chdir(CAL)
    try:
        cfg = load_fleet_experiment(str(out))
    finally:
        os.chdir(cwd)
    assert cfg.communication.severities == ["L0", "L3b"] and cfg.communication.beacon_interval_s == [0.5, 2.0]
    assert [b for b in cfg.beliefs] == ["calibrated", "last_beacon", "oracle"]
    assert (cfg.compare.a, cfg.compare.b) == ("calibrated", "last_beacon")
    t = cfg.task
    assert (t.target_separation_m, t.tolerance_m, t.hold_steps, t.max_episode_steps, t.collision_radius_m) == \
        (3.0, 0.3, 25, 300, 0.6)
    assert cfg.initial_conditions.separations_m == [3.5, 5.0] and cfg.evaluation.comm_seeds == [1, 2, 3]
    assert cfg.dynamics_backend == "both" and cfg.disturbance.roll_pitch == "fitted" and cfg.disturbance.scale == 1.5
    assert cfg.name == "my_study"


@pytest.mark.parametrize("argv,msg", [
    (["--hold-steps", "500"], "hold_steps (500) > max_episode_steps"),
    (["--beliefs", "last_beacon", "oracle", "--compare", "calibrated", "generic"], "compare refers to belief"),
    (["--separations", "0.3"], "inside collision_radius_m"),
    (["--seeds", "0"], "--seeds must be at least 1"),
    (["--disturbance", "missing.yaml"], "disturbance file not found"),
])
def test_fleet_flags_refuse_impossible_experiments_before_running(tmp_path, monkeypatch, capsys, argv, msg):
    monkeypatch.setattr(cli, "JOBS_DIR", str(tmp_path))
    started = []
    assert cli.main(["fleet", *argv], runner=lambda c: started.append(c) or 0) == 2
    assert msg in capsys.readouterr().out and not started


def test_fleet_beliefs_without_the_default_pair_drop_the_headline(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "JOBS_DIR", str(tmp_path))
    out = tmp_path / "x.yaml"
    assert cli.main(["fleet", "--beliefs", "last_beacon", "oracle", "--save-config", str(out)], runner=lambda c: 0) == 0
    assert yaml.safe_load(out.read_text())["compare"] is None


# ----------------------------------------------------------------- water currents per recording

def _synthetic_dives(folder, currents, n=500, seed=0):
    import thruster_model as tm
    from dynamics_model import SimulationConfig
    from fast_dynamics import FastCore
    os.makedirs(folder, exist_ok=True)
    th = tm.bluerov2_heavy().for_simulation()
    rng = np.random.default_rng(seed)
    names = []
    for i, c in enumerate(currents):
        core = FastCore(TRUTH, SPEC.vehicle, SimulationConfig(dt=0.1, physics_dt=0.005))
        core.reset()
        core.current_world = np.array(c, float)
        tau, rows = np.zeros(6), []
        for k in range(n):
            if k % 20 == 0:
                tau = th.achievable_wrench(rng.uniform(-1, 1, 6) * np.array([40, 40, 30, 3, 3, 10]))
            st = core.state
            rows.append([*tau, *st.nu, *st.orientation.as_quat()])
            core.step(tau)
        name = f"sea-0{i + 1}-06-2025-run1.csv"
        pd.DataFrame(rows, columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp", "Bq", "Br",
                                    "qx", "qy", "qz", "qw"]).to_csv(os.path.join(folder, name), index=False)
        names.append(name)
    return names


def test_current_estimated_from_logs_with_a_good_model(tmp_path):
    from estimate_current import estimate_currents, write_outputs
    from objective import DatasetConfig
    truth_c = [(0.2, -0.05, 0.0), (-0.1, 0.15, 0.03)]
    names = _synthetic_dives(tmp_path / "sea", truth_c)
    files = [str(tmp_path / "sea" / n) for n in names]
    rows = estimate_currents(files, TRUTH, SPEC.vehicle, DatasetConfig(10.0), workers=2)
    for r, c in zip(rows, truth_c):
        assert np.linalg.norm([r["vx_mps"] - c[0], r["vy_mps"] - c[1], r["vz_mps"] - c[2]]) < 0.02
        assert r["looks_constant"] and r["error_explained_pct"] > 80
    md = write_outputs(rows, str(tmp_path / "sea" / "currents.csv"), "the true model")
    assert "Every recording's current looks constant" in open(md, encoding="utf-8").read()


def test_estimate_flags_a_current_that_changes_and_an_unreliable_model(tmp_path):
    """First half in one current, second half in another: the halves disagree and the file says so."""
    from estimate_current import estimate_run, verdict
    from objective import DatasetConfig
    names = _synthetic_dives(tmp_path / "a", [(0.25, 0.0, 0.0)], n=600)
    names2 = _synthetic_dives(tmp_path / "b", [(-0.25, 0.0, 0.0)], n=600, seed=0)
    a = pd.read_csv(tmp_path / "a" / names[0])
    b = pd.read_csv(tmp_path / "b" / names2[0])
    mixed = pd.concat([a.iloc[:300], b.iloc[300:]])
    p = tmp_path / "mixed-01-06-2025-run1.csv"
    mixed.to_csv(p, index=False)
    r = estimate_run((str(p), TRUTH.to_vector(), SPEC.vehicle, DatasetConfig(10.0)))
    assert r["looks_constant"] is False and "changed during this recording" in verdict(r)
    r["model"] = "default"
    assert "rough" in verdict(r)


def test_currents_file_reaches_every_step(tmp_path):
    from objective import DatasetConfig, load_run, read_currents_file
    names = _synthetic_dives(tmp_path / "sea", [(0.2, 0.0, 0.0), (0.0, 0.2, 0.0)], n=100)
    pd.DataFrame({"file": names, "vx_mps": [0.2, 0.0], "vy_mps": [0.0, 0.2], "vz_mps": [0.0, 0.0]}).to_csv(
        tmp_path / "sea" / "currents.csv", index=False)
    by = read_currents_file(str(tmp_path / "sea" / "currents.csv"))
    dc = DatasetConfig(10.0, current_world_mps=(0.9, 0.9, 0.0), current_by_file=by)
    r0 = load_run(str(tmp_path / "sea" / names[0]), dc)
    from scipy.spatial.transform import Rotation
    np.testing.assert_allclose(Rotation.from_quat(r0.orientation_quat).apply(r0.current_body),
                               np.tile([0.2, 0.0, 0.0], (100, 1)), atol=1e-12)  # its own current, not the global one
    pd.DataFrame({"file": ["x"], "speed": [1]}).to_csv(tmp_path / "bad.csv", index=False)
    with pytest.raises(ValueError, match="needs columns"):
        read_currents_file(str(tmp_path / "bad.csv"))
    # strathe calibrate picks up a currents.csv sitting next to the logs, and passes it to the fit
    calls = []
    cli.main(["calibrate", "--data", str(tmp_path / "sea"), "--out", str(tmp_path / "res"), "--skip-identifiability"],
             runner=lambda c: calls.append(c) or 0)
    fit = [c for c in calls if c[1] == "run_calibration.py"][0]
    assert fit[fit.index("--currents-file") + 1] == str(tmp_path / "sea" / "currents.csv")


def test_estimate_current_rounds_in_calibrate(tmp_path):
    calls = []
    rc = cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "r"), "--estimate-current",
                   "--current-rounds", "2", "--skip-identifiability"], runner=lambda c: calls.append(c) or 0)
    assert rc == 0
    assert [c[1] for c in calls] == ["preflight_check.py", "run_calibration.py", "estimate_current.py",
                                     "run_calibration.py", "estimate_current.py", "run_calibration.py",
                                     "calibration_report.py"]
    r1, e1, r2, e2, final = calls[1], calls[2], calls[3], calls[4], calls[5]
    assert "--currents-file" not in r1  # round 1: still water
    assert e1[e1.index("--params") + 1].endswith(os.path.join("current_round1", "calibrated_parameters.yaml"))
    assert r2[r2.index("--currents-file") + 1].endswith(os.path.join("current_round1", "currents.csv"))
    assert final[final.index("--currents-file") + 1].endswith(os.path.join("current_round2", "currents.csv"))
    assert final[final.index("--output-dir") + 1] == str(tmp_path / "r")
    assert cli.main(["calibrate", "--data", str(tmp_path), "--estimate-current", "--current-world-mps", "0.1", "0",
                     "0"], runner=lambda c: 0) == 2
