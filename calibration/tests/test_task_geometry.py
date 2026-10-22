"""#91: train-fleet and arena task geometry for vehicles bigger than a BlueROV2 (open since the first URI run)."""
import json
import os
import sys
import types

import pytest

import strathe_cli as cli

CAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_training_records_its_geometry_and_refuses_one_that_cant_work(tmp_path, monkeypatch):
    import train_fleet_policy as tfp
    out = tmp_path / "big"
    monkeypatch.setattr(sys, "argv", ["train_fleet_policy.py", "--algorithm", "ppo", "--timesteps", "64", "--n-sims", "1",
                                      "--target-separation", "6", "--tolerance", "0.6", "--collision-radius", "1.5",
                                      "--separations", "5", "9", "--episode-steps", "300", "--out", str(out)])
    tfp.main()
    td = json.loads((tmp_path / "big.json").read_text())["training_distribution"]
    assert td["task"] == {"target_separation": 6.0, "tolerance": 0.6, "safety_radius": 1.5, "hold_steps": 20}
    assert td["initial_separation_range_m"] == [5.0, 9.0]
    monkeypatch.setattr(sys, "argv", ["train_fleet_policy.py", "--timesteps", "64", "--collision-radius", "3",
                                      "--out", str(tmp_path / "x")])       # starting 2-5.5 m apart, 3 m collisions
    with pytest.raises(SystemExit):
        tfp.main()


def test_strathe_forwards_the_geometry(tmp_path):
    import run_arena_experiment as rae
    calls = []
    params = os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml")
    assert cli.main(["train-fleet", "--params", params, "--target-separation", "6", "--tolerance", "0.6",
                     "--separations", "5", "9", "--out", str(tmp_path / "p")], runner=lambda a: calls.append(a) or 0) == 0
    t = calls[-1]
    assert t[t.index("--target-separation") + 1] == "6.0" and t[t.index("--separations") + 1:][:2] == ["5.0", "9.0"]
    assert cli.main(["arena", "--target-radius", "1.5", "--target-range", "10", "--out", str(tmp_path / "a")],
                    runner=lambda a: calls.append(a) or 0) == 0
    a = calls[-1]
    assert a[a.index("--target-radius") + 1] == "1.5" and a[a.index("--target-range") + 1] == "10.0"
    rae.build_parser().parse_args(a[2:])


def test_the_arena_uses_and_describes_its_geometry():
    import run_arena_experiment as rae
    ek = {"spec": os.path.join(CAL, "vehicle_spec.yaml"), "task": "waypoint", "max_episode_steps": 50,
          "waypoint_radius": 1.5, "waypoint_range": 10.0}
    from build_two_agent_cores import load_calibrated_parameters
    env = rae.make_env(load_calibrated_parameters(os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml"),
                                                  quiet=True), ek)
    assert env.waypoint_radius == 1.5 and env.waypoint_range == 10.0
    assert "within 1.5 m" in rae.task_plain({"task": "waypoint", "target_radius_m": 1.5, "target_range_m": 10})
    assert "up to 10 m away" in rae.task_plain({"task": "waypoint", "target_radius_m": 1.5, "target_range_m": 10})
    assert "within 0.5 m" in rae.task_plain({"task": "docking"})            # unchanged by default


def test_the_fleet_report_cautions_when_the_policy_was_trained_for_another_geometry(tmp_path):
    import run_fleet_experiment as rfe
    (tmp_path / "pol.json").write_text(json.dumps({"training_distribution": {"task": {"target_separation": 2.0}}}))
    cfg = types.SimpleNamespace(policies=[types.SimpleNamespace(model_path=str(tmp_path / "pol.zip"))],
                                task=types.SimpleNamespace(target_separation_m=6.0, tolerance_m=0.6))
    note = rfe.trained_geometry_note(cfg)
    assert "trained to hold 2 m" in note and "--target-separation 6" in note
    cfg.task.target_separation_m = 2.05
    assert rfe.trained_geometry_note(cfg) is None
