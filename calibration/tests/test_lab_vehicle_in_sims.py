"""
A lab's own vehicle in the simulators: thruster limits in the one-vehicle
Arena and the two-vehicle fleet (both engines), a lab's vehicle description
in fleet experiments, the single-vehicle experiment runner, and the
identifiability check `strathe calibrate` now runs.
"""
import json
import os
import re

import numpy as np
import pytest
import yaml

import thruster_model as tm
from arena import Arena
from build_two_agent_cores import load_calibrated_parameters
from current import CurrentConfig
from fleet_belief import BeliefSpec
from fleet_env import FleetConfig, FleetSim
from optimizer import initial_parameters, load_vehicle_spec

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
TRUTH = load_calibrated_parameters(os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml"))
SPEC = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml"))
HEAVY = tm.bluerov2_heavy().for_simulation()
SIX = tm.thruster_block({"layout": [
    {"position_m": [0.1335, -0.1335, 0], "direction": [-1, -1, 0], "reversed": True},
    {"position_m": [0.1335, 0.1335, 0], "direction": [-1, 1, 0], "reversed": True},
    {"position_m": [-0.1335, -0.1335, 0], "direction": [1, -1, 0]},
    {"position_m": [-0.1335, 0.1335, 0], "direction": [1, 1, 0]},
    {"position_m": [0, -0.11, 0], "direction": [0, 0, -1]},
    {"position_m": [0, 0.11, 0], "direction": [0, 0, -1], "reversed": True}]})


def _within_limits(cfg, wrenches):
    t = np.asarray(wrenches) @ np.linalg.pinv(cfg.allocation).T
    return t.max() <= cfg.curve.max_forward_n + 1e-6 and -t.min() <= cfg.curve.max_reverse_n + 1e-6


# ----------------------------------------------------------------- one vehicle

@pytest.mark.parametrize("thrusters", [None, HEAVY, SIX])
def test_arena_applies_thruster_limits(thrusters):
    env = Arena(TRUTH, SPEC.vehicle, task="station_keeping", thrusters=thrusters,
                current=CurrentConfig(speed_mps=(0.0, 0.3), direction_deg=(0, 360)))
    env.reset(seed=3)
    rng = np.random.default_rng(0)
    applied = []
    for _ in range(300):
        a = rng.uniform(-1, 1, 6)
        _, r, term, trunc, info = env.step(a)
        assert np.isfinite(r) and np.all(np.isfinite(info["position"]))
        applied.append((a, info["tau_applied"]))
        if term or trunc:
            env.reset()
    acts = np.array([np.clip(a, -1, 1) for a, _ in applied]) * 30.0
    taus = np.array([t for _, t in applied])
    if thrusters is None:
        np.testing.assert_allclose(taus, acts, atol=1e-5)  # original behaviour, untouched
    else:
        assert _within_limits(thrusters, taus)
        assert np.abs(taus).max() < np.abs(acts).max()  # full-scale random commands exceed what thrusters give
    if thrusters is SIX:
        assert np.abs(taus[:, 4]).max() < 1e-9  # no pitch authority, no pitch torque


def test_arena_without_thrusters_is_bit_identical_to_before():
    """thrusters=None must not change a single number (every earlier result used it)."""
    a, b = Arena(TRUTH, SPEC.vehicle), Arena(TRUTH, SPEC.vehicle, thrusters=None)
    oa, _ = a.reset(seed=5)
    ob, _ = b.reset(seed=5)
    rng = np.random.default_rng(1)
    for _ in range(100):
        act = rng.uniform(-1, 1, 6)
        ra, rb = a.step(act), b.step(act)
        np.testing.assert_array_equal(ra[0], rb[0])


# ----------------------------------------------------------------- two vehicles

@pytest.mark.parametrize("backend", ["stand_in", "mujoco"])
def test_fleet_applies_thruster_limits_and_beacons_carry_the_delivered_force(backend):
    pytest.importorskip("mujoco") if backend == "mujoco" else None
    cfg = FleetConfig(dynamics_backend=backend, thrusters=SIX,
                      current=CurrentConfig(speed_mps=0.2, direction_deg=(0, 360)))
    trajs = []
    for belief in (BeliefSpec("physics", "calibrated", TRUTH), BeliefSpec("last_beacon", "last_beacon")):
        sim = FleetSim(TRUTH, SPEC.vehicle, cfg, belief=belief)
        sim.reset(comm_seed=4, initial_separation=3.0, closing_speed=0.2)
        rng = np.random.default_rng(2)
        pos = []
        for _ in range(60):
            obs, r, term, trunc, info = sim.step([rng.uniform(-1, 1, 6), rng.uniform(-1, 1, 6)])
            assert all(np.all(np.isfinite(o)) for o in obs)
            for t in sim._last_tau:
                assert _within_limits(SIX, [t]) and abs(t[4]) < 1e-9
            pos.append(np.concatenate(info["positions"]))
            if term or trunc:
                break
        trajs.append(np.array(pos))
    n = min(len(t) for t in trajs)
    np.testing.assert_array_equal(trajs[0][:n], trajs[1][:n])  # truth still independent of the belief model


def test_limits_matter_physically():
    """A full-scale pitch command: 30 N*m unlimited, at most ~18 N*m on a Heavy, nothing on the 6-thruster."""
    pitch = np.array([0, 0, 0, 0, 30.0, 0])
    assert HEAVY.achievable_wrench(pitch)[4] == pytest.approx(HEAVY.max_wrench()[4], rel=1e-6)
    assert HEAVY.max_wrench()[4] < 20
    assert SIX.achievable_wrench(pitch)[4] == pytest.approx(0, abs=1e-9)


# ----------------------------------------------------------------- a lab's own vehicle in a fleet experiment

def _tiny_experiment(tmp_path, **over):
    exp = {"schema_version": 2, "name": "lab_vehicle", "policies": [{"model_path": "fleet_policy_seed1"}],
           "beliefs": ["calibrated", "generic", "last_beacon"], "communication": {"severity": "L2"},
           "initial_conditions": {"separations_m": [3.0, 4.5], "closing_speeds_mps": [0.0]},
           "evaluation": {"comm_seeds": [1], "workers": 1}, "output_dir": str(tmp_path / "out"), **over}
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(exp))
    return str(p)


def test_fleet_experiment_with_a_lab_vehicle_description_and_thrusters(tmp_path, monkeypatch):
    from fleet_experiment_schema import ConfigError, load_fleet_experiment
    from run_fleet_experiment import run_experiment
    monkeypatch.chdir(CAL)
    raw = yaml.safe_load(open(os.path.join(CAL, "vehicle_spec.yaml"), encoding="utf-8"))
    raw["fixed_parameters"]["mass_kg"]["value"] = 15.0  # a heavier vehicle than the reference 11.2 kg
    spec_path = tmp_path / "my_vehicle_spec.yaml"
    spec_path.write_text(yaml.safe_dump(raw))
    thr = os.path.join(CAL, "templates", "ingest_csv_pwm_custom_thrusters.yaml")
    cfg_path = _tiny_experiment(tmp_path, vehicle_spec=str(spec_path), thrusters=thr)
    cfg = load_fleet_experiment(cfg_path)
    assert load_vehicle_spec(cfg.vehicle_spec).vehicle.mass_kg == 15.0
    assert cfg.thruster_config().n_thrusters == 8
    run_experiment(cfg_path)
    report = open(tmp_path / "out" / "summary_report.md", encoding="utf-8").read()
    assert "Thruster limits on:" in report and f"Vehicle description: `{spec_path}`" in report
    meta = json.load(open(tmp_path / "out" / "metadata.json", encoding="utf-8"))
    assert meta["resolved_config"]["thrusters"] == thr
    for bad, msg in (({"vehicle_spec": "nope.yaml"}, "vehicle_spec file not found"),
                     ({"thrusters": "bluerov9"}, "only preset is bluerov2_heavy"),
                     ({"thrusters": {"layout": []}}, "non-empty list")):
        with pytest.raises(ConfigError, match=msg):
            load_fleet_experiment(_tiny_experiment(tmp_path, **bad))


# ----------------------------------------------------------------- one vehicle, end to end

def test_single_vehicle_experiment_end_to_end(tmp_path):
    import run_arena_experiment as rae
    out = tmp_path / "arena"
    rc = rae.main(["--seeds", "1", "--timesteps", "2048", "--episodes", "6", "--compare-generic",
                   "--current-speed", "0", "0.2", "--thrusters", "bluerov2_heavy", "--workers", "2",
                   "--task", "waypoint", "--out", str(out)])
    assert rc == 0
    res = json.load(open(out / "results.json", encoding="utf-8"))
    labels = [s["controller"] for s in res["summary"]]
    assert labels == ["trained on your calibration", "trained on default physics", "hand-written controller",
                      "random commands"]
    assert all(s["episodes"] == 6 for s in res["summary"])
    assert res["calibrated_vs_default"]["n_pairs"] == 6
    report = open(out / "report.md", encoding="utf-8").read()
    assert "Does the calibration matter" in report and "thruster limits" in report and "water current" in report
    assert "teammate" not in report  # a single vehicle has no teammate model
    rows = open(out / "episodes.csv", encoding="utf-8").read().splitlines()
    assert len(rows) == 1 + 4 * 6
    # every controller saw the same episodes (same seeds -> same currents)
    import csv
    eps = list(csv.DictReader(open(out / "episodes.csv", encoding="utf-8")))
    by_seed = {}
    for e in eps:
        by_seed.setdefault(e["episode_seed"], set()).add(e["current_speed_mps"])
    assert all(len(v) == 1 for v in by_seed.values())
    # evaluating your own policy instead of training
    pol = res["policies"][0]["policy"]
    assert rae.main(["--policy", pol, "--episodes", "3", "--workers", "1", "--out", str(tmp_path / "mine")]) == 0
    mine = json.load(open(tmp_path / "mine" / "results.json", encoding="utf-8"))
    assert mine["summary"][0]["controller"] == "your policy" and mine["calibrated_vs_default"] is None


# ----------------------------------------------------------------- identifiability, with the fit's own objective

def _synthetic_result(tmp_path, objective):
    params = initial_parameters(SPEC)
    d = tmp_path / "result"
    d.mkdir()
    (d / "calibrated_parameters.yaml").write_text(yaml.safe_dump(
        {"parameters": {n: float(getattr(params, n)) for n in params.param_names}}))
    (d / "metrics.json").write_text(json.dumps({"dataset": {"heldout_dates": ["03-03-2024"]},
                                                "objective": objective}))
    return str(d)


def test_identifiability_uses_the_calibrations_own_objective(tmp_path, monkeypatch):
    monkeypatch.chdir(CAL)
    from automated_identifiability_report import identifiability_for_result
    held = {n: float(getattr(initial_parameters(SPEC), n)) for n in initial_parameters(SPEC).param_names
            if n not in ("Xu", "Xuu", "Nr", "Nrr", "Zw")}
    obj = {"axis_scaling": "zero_rms", "prediction_horizon_samples": 30, "excluded_axes": [],
           "torque_reference_offset_m": [0, 0, -0.085], "current_world_mps": [0.05, 0.0, 0.0],
           "held_parameters": held, "sampling_rate_hz": 10.0}
    d = _synthetic_result(tmp_path, obj)
    serial = identifiability_for_result(d, "synth_test_data", "vehicle_spec.yaml", workers=1)
    assert serial["free_parameters"] == ["Xu", "Zw", "Nr", "Xuu", "Nrr"] or \
        set(serial["free_parameters"]) == {"Xu", "Xuu", "Nr", "Nrr", "Zw"}
    assert serial["objective"]["current_world_mps"] == [0.05, 0.0, 0.0]
    assert serial["objective"]["prediction_horizon_samples"] == 30
    assert serial["n_train_runs_used"] == 4 and np.isfinite(serial["condition_number"])
    parallel = identifiability_for_result(d, "synth_test_data", "vehicle_spec.yaml", workers=2,
                                          output=str(tmp_path / "p.json"))
    assert parallel["condition_number"] == pytest.approx(serial["condition_number"], rel=1e-9)
    assert json.load(open(os.path.join(d, "identifiability_report.json"), encoding="utf-8"))["free_parameters"]


def test_barely_pinned_groups_and_pairs_reach_the_plain_report(tmp_path, monkeypatch):
    """Two parameters whose effects are identical must come out as a pair and as a weak group, in plain words."""
    monkeypatch.chdir(CAL)
    import automated_identifiability_report as air
    from calibration_report import build_report
    held = {n: float(getattr(initial_parameters(SPEC), n)) for n in initial_parameters(SPEC).param_names
            if n not in ("Xu", "Xuu", "Nr")}
    d = _synthetic_result(tmp_path, {"axis_scaling": "none", "held_parameters": held, "sampling_rate_hz": 10.0})

    free = [n for n in initial_parameters(SPEC).param_names if n not in held]

    def fake_jacobian(fun, x0, method, bounds):
        rng = np.random.default_rng(0)
        col = rng.normal(size=200)
        w = bounds[1] - bounds[0]
        # Xu and Xuu move the residuals identically once scaled by their allowed ranges
        cols = [col if n in ("Xu", "Xuu") else rng.normal(size=200) for n in free]
        return np.stack([c / w[i] for i, c in enumerate(cols)], 1)
    import scipy.optimize._numdiff as nd
    monkeypatch.setattr(nd, "approx_derivative", fake_jacobian)
    r = air.identifiability_for_result(d, "synth_test_data", "vehicle_spec.yaml")
    assert r["free_parameters"] == ["Xu", "Nr", "Xuu"] or set(r["free_parameters"]) == {"Xu", "Xuu", "Nr"}
    assert r["condition_number"] == float("inf")
    assert set(r["weak_combinations"][0]["parameters"]) == {"Xu", "Xuu"}
    rep = build_report(d, "vehicle_spec.yaml")
    text = open(rep["report"], encoding="utf-8").read()
    assert "Groups of values your data barely pins down" in text
    assert re.search(r"forward drag \(slow speeds\), forward drag \(fast speeds\)|forward drag \(fast speeds\), "
                     r"forward drag \(slow speeds\)", text)
