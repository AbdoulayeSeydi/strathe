"""
Fleet v2 experiment config + end-to-end runner tests.

Includes regression tests for the two fleet v1 runner bugs: task settings
were validated but never reached the sweep, and any comm override crashed
with KeyError: '<severity>_custom'.
"""
import csv
import json
import os

import numpy as np
import pytest
import yaml

from fleet_experiment_schema import ConfigError, load_fleet_experiment
from fleet_vec_env import CommCondition, EpisodeSampler, FleetVecEnv
from fleet_belief import BeliefSpec
from fleet_env import FleetConfig, FleetSim
from comm_channel import BandwidthConfig, PacketLossConfig
from build_two_agent_cores import load_calibrated_parameters
from optimizer import load_vehicle_spec

TRUTH = "calibrated_parameters_fleet_truth.yaml"


@pytest.fixture(scope="module")
def tiny_policy(tmp_path_factory):
    """A barely-trained but contract-valid per-vehicle PPO policy."""
    from stable_baselines3 import PPO
    vehicle = load_vehicle_spec("vehicle_spec.yaml").vehicle
    truth = load_calibrated_parameters(TRUTH)
    sampler = EpisodeSampler([CommCondition("L0", PacketLossConfig(0, 1000, 0, 0), BandwidthConfig(1e9))],
                             lambda rng: BeliefSpec("last_beacon", "lb"))
    env = FleetVecEnv([FleetSim(truth, vehicle, FleetConfig(max_episode_steps=30, task=_task(5)))], sampler)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=64, n_epochs=1, seed=0, device="cpu")
    model.learn(128)
    path = str(tmp_path_factory.mktemp("pol") / "tiny_policy")
    model.save(path)
    return path


def _task(hold):
    from formation_hold_task import FormationHoldConfig
    return FormationHoldConfig(hold_steps=hold)


def _config(tmp_path, policy, **over):
    cfg = {
        "schema_version": 2, "name": "t",
        "policies": [{"model_path": policy}],
        "truth_params": TRUTH,
        "beliefs": ["calibrated", "generic", "last_beacon", "oracle"],
        "communication": {"severities": ["L2"], "beacon_interval_s": 1.0},
        "task": {"max_episode_steps": 40, "hold_steps": 5},
        "initial_conditions": {"separations_m": [2.5, 4.0], "closing_speeds_mps": [0.0, 0.5]},
        "evaluation": {"comm_seeds": [1], "workers": 1},
        "output_dir": str(tmp_path / "out"),
    }
    for k, v in over.items():
        cfg[k] = v
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


# ---------------------------------------------------------------- schema

def test_valid_config_loads(tmp_path, tiny_policy):
    cfg = load_fleet_experiment(_config(tmp_path, tiny_policy))
    assert cfg.communication.beacon_interval_s == [1.0]
    assert [l for l, _, _ in cfg.communication.levels()] == ["L2"]


def test_v1_config_gets_a_clear_message(tmp_path):
    p = tmp_path / "v1.yaml"
    p.write_text(yaml.safe_dump({"vehicles": [{"model_path": "x"}], "communication": {"severity": "L2"},
                                 "output_dir": "o"}))
    with pytest.raises(ConfigError, match="fleet v1 config"):
        load_fleet_experiment(str(p))


@pytest.mark.parametrize("override,match", [
    ({"communication": {"severities": ["L1", "L2"], "overrides": {"near_field_prob": 0.4}}}, "ONE severity"),
    ({"communication": {"severities": ["L2"], "beacon_interval_s": 0.25}}, "multiple of the 0.1 s tick"),
    ({"communication": {}}, "at least one severity"),
    ({"task": {"hold_steps": 50, "max_episode_steps": 40}}, "success would be impossible"),
    ({"task": {"target_separation_m": 0.6, "tolerance_m": 0.2}}, "must exceed collision_radius_m"),
    ({"compare": {"a": "calibrated", "b": "made_up"}}, "not in beliefs"),
    ({"initial_conditions": {"separations_m": [0.3, 3.0]}}, "inside collision_radius_m"),
    ({"beliefs": ["calibrated", "calibrated"]}, "unique"),
    ({"beliefs": ["calibrated", "generic", {"physics_params": "nope.yaml", "label": "x"}]}, "not found"),
    ({"beliefs": ["calibrated", "generic", {"physics_alpha": 1.5}]}, "less than or equal"),
    ({"truth_params": "missing.yaml"}, "truth_params file not found"),
])
def test_invalid_configs_are_rejected(tmp_path, tiny_policy, override, match):
    with pytest.raises(ConfigError, match=match):
        load_fleet_experiment(_config(tmp_path, tiny_policy, **override))


def test_overrides_resolve_on_top_of_a_severity(tmp_path, tiny_policy):
    cfg = load_fleet_experiment(_config(tmp_path, tiny_policy, communication={
        "severity": "L2", "overrides": {"near_field_prob": 0.9, "far_field_prob": 0.9}}))
    ((label, plc, _),) = cfg.communication.levels()
    assert label == "L2_custom" and plc.near_field_prob == 0.9 and plc.far_distance_m == 1000.0


def test_overrides_without_severity_are_custom(tmp_path, tiny_policy):
    cfg = load_fleet_experiment(_config(tmp_path, tiny_policy, communication={"overrides": {"far_field_prob": 0.2}}))
    ((label, plc, _),) = cfg.communication.levels()
    assert label == "custom" and plc.far_field_prob == 0.2 and plc.near_field_prob == 0.0


# ---------------------------------------------------------------- end to end

def _run(path):
    from run_fleet_experiment import run_experiment
    return run_experiment(path)


def test_end_to_end_outputs(tmp_path, tiny_policy):
    path = _config(tmp_path, tiny_policy, beliefs=["calibrated", {"physics_alpha": 0.5}, "generic",
                                                  "last_beacon", "oracle"])
    metrics = _run(path)
    out = tmp_path / "out"
    for f in ("summary_report.md", "summary.csv", "episodes.csv", "metrics.json", "metadata.json",
              "experiment.yaml", "plots/success_by_belief.png", "plots/dose_response.png"):
        assert (out / f).exists(), f
    assert len(metrics["summary"]) == 5
    (c,) = metrics["paired_contrast"]
    assert c["a"] == "calibrated" and c["b"] == "generic" and c["n_pairs"] == 4
    assert metrics["dose_response"] and metrics["dose_response"][0]["alphas"] == [0.0, 0.5, 1.0]
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    assert "git_uncommitted_changes" in meta and meta["n_episodes"] == 20
    with open(out / "episodes.csv") as f:
        assert len(list(csv.DictReader(f))) == 20
    report = (out / "summary_report.md").read_text(encoding="utf-8")
    assert "Headline" in report and "calibrated" in report
    # the resolved config is itself a valid config (reproducibility round-trip)
    again = load_fleet_experiment(str(out / "experiment.yaml"))
    assert again.communication.beacon_interval_s == [1.0]


def test_task_settings_reach_the_simulation(tmp_path, tiny_policy):
    """Fleet v1 regression: target separation used to be silently ignored."""
    results = []
    for target in (2.0, 3.5):
        d = tmp_path / f"t{target}"
        d.mkdir()
        m = _run(_config(d, tiny_policy, beliefs=["calibrated", "generic"],
                         task={"target_separation_m": target, "max_episode_steps": 40, "hold_steps": 5}))
        results.append({s["belief"]: s for s in m["summary"]})
    with open(tmp_path / "t2.0" / "out" / "episodes.csv") as f:
        e2 = [float(r["mean_abs_separation_error_m"]) for r in csv.DictReader(f)]
    with open(tmp_path / "t3.5" / "out" / "episodes.csv") as f:
        e35 = [float(r["mean_abs_separation_error_m"]) for r in csv.DictReader(f)]
    assert not np.allclose(e2, e35)


def test_comm_override_reaches_the_simulation(tmp_path, tiny_policy):
    """Fleet v1 regression: overrides crashed with KeyError; they must run and change delivery."""
    m = _run(_config(tmp_path, tiny_policy, beliefs=["calibrated", "generic"],
                     communication={"severity": "L2", "overrides": {"near_field_prob": 1.0, "far_field_prob": 1.0}}))
    assert all(s["delivery_rate"] == 0.0 for s in m["summary"])
    assert all(s["comm"] == "L2_custom@1s" for s in m["summary"])


def test_beacon_interval_sweep_creates_one_comm_cell_each(tmp_path, tiny_policy):
    m = _run(_config(tmp_path, tiny_policy, beliefs=["calibrated", "generic"],
                     communication={"severity": "L0", "beacon_interval_s": [0.5, 2.0]}))
    assert sorted({s["comm"] for s in m["summary"]}) == ["L0@0.5s", "L0@2s"]
