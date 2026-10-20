"""Multi-start calibration (PROJECT_STATUS.md #79): `strathe calibrate --restarts N` starts the fit from N points and
keeps the lowest training cost, so a hard calibration doesn't depend on one random start (F25's other half: #76 made
the start repeatable, this makes it matter less)."""
import json
import os
import types

import numpy as np
import pytest

import calibration_report
import optimizer
import strathe_cli as cli
from optimizer import CalibrationConfig, calibrate, load_vehicle_spec


def _fits(calls):
    return [c for c in calls if c[1] == "run_calibration.py"]


def _restarts(argv):
    return int(argv[argv.index("--n-restarts") + 1]) if "--n-restarts" in argv else 1


def test_restarts_reach_the_fit(tmp_path, capsys):
    calls = []
    assert cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "o"), "--restarts", "3"],
                    runner=lambda c: calls.append(c) or 0) == 0
    assert _restarts(_fits(calls)[0]) == 3
    assert "from 3 points" in capsys.readouterr().out
    calls.clear()
    cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "o2")], runner=lambda c: calls.append(c) or 0)
    assert _restarts(_fits(calls)[0]) == 1  # one start unless asked
    assert cli.main(["calibrate", "--data", str(tmp_path), "--restarts", "0"], runner=lambda c: 0) == 2


def test_with_current_estimation_only_the_first_round_restarts(tmp_path):
    """Later rounds start from the previous round's result: restarting them would throw the chosen start away."""
    calls = []
    cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "o"), "--estimate-current",
              "--current-rounds", "2", "--restarts", "4", "--seed", "7"], runner=lambda c: calls.append(c) or 0)
    fits = _fits(calls)
    assert [_restarts(f) for f in fits] == [4, 1, 1]  # round 1, round 2, the final fit
    assert all(f[f.index("--seed") + 1] == "7" for f in fits)


def _fake_least_squares(costs, successes):
    it = iter(zip(costs, successes))

    def fake(fun, x0, **kw):
        cost, ok = next(it)
        return types.SimpleNamespace(x=np.asarray(x0, float), cost=cost, success=ok, status=1 if ok else 0, nfev=5,
                                     message="ok" if ok else "budget", optimality=0.0)
    return fake


def _spec():
    return load_vehicle_spec(os.path.join(os.path.dirname(optimizer.__file__), "vehicle_spec.yaml"))


def test_the_lowest_training_cost_is_kept_even_if_it_stopped_at_its_budget(monkeypatch):
    monkeypatch.setattr(optimizer, "least_squares", _fake_least_squares([3.9, 2.0, 4.2], [True, False, True]))
    monkeypatch.setattr(optimizer, "mean_run_rmse", lambda *a, **k: 0.0)
    r = calibrate(_spec(), [], CalibrationConfig(n_restarts=3, seed=1, initialization_strategy="perturbed"))
    assert r.cost == 2.0 and r.converged is False  # the best start ran out of budget: said so, not hidden
    assert [x.cost for x in r.restarts] == [3.9, 2.0, 4.2]
    starts = {tuple(x.x0) for x in r.restarts}
    assert len(starts) == 3  # three different starting points


def test_one_start_behaves_as_before(monkeypatch):
    monkeypatch.setattr(optimizer, "least_squares", _fake_least_squares([2.0], [True]))
    monkeypatch.setattr(optimizer, "mean_run_rmse", lambda *a, **k: 0.0)
    r = calibrate(_spec(), [], CalibrationConfig(n_restarts=1, seed=1, initialization_strategy="perturbed"))
    assert r.converged is True and r.cost == 2.0
    monkeypatch.setattr(optimizer, "least_squares", _fake_least_squares([2.0, 3.0], [False, False]))
    with pytest.raises(RuntimeError, match="failed to converge"):  # nothing converged: the CLI keeps the checkpoint
        calibrate(_spec(), [], CalibrationConfig(n_restarts=2, seed=1, initialization_strategy="perturbed"))


def test_the_report_lists_each_starts_cost(tmp_path):
    p = tmp_path / "r.md"
    calibration_report.write_report(str(p), "x", [], None, [], [], None, ["Check the logs."], True, seed=5,
                                    restarts=[2.018, 2.217, 2.0181])
    text = p.read_text()
    assert "from 3 points" in text and "2.018, 2.217" in text and "depends on where the fit starts" in text
    calibration_report.write_report(str(p), "x", [], None, [], [], None, ["Check the logs."], True, seed=5,
                                    restarts=[2.018, 2.0181])
    assert "doesn't matter" in p.read_text()
    calibration_report.write_report(str(p), "x", [], None, [], [], None, ["Check the logs."], True, seed=5)
    assert "--restarts 3" in p.read_text()  # one start: the report says how to try more


def test_the_report_reads_the_first_rounds_starts(tmp_path):
    (tmp_path / "current_round1").mkdir()
    (tmp_path / "current_round1" / "metrics.json").write_text(json.dumps({"optimizer": {"restart_costs": [1.0, 2.0]}}))
    final = {"optimizer": {"restart_costs": [1.0]}}
    assert calibration_report.restart_costs(str(tmp_path), final) == [1.0, 2.0]
    assert calibration_report.restart_costs(str(tmp_path / "nowhere"), final) == [1.0]


def test_when_no_start_converges_the_finished_result_still_lists_each_starts_cost(monkeypatch, tmp_path):
    """URI's bay run: all four starts ran out of budget, the CLI finished it from the checkpoint and the report lost
    the four costs. They now go through the checkpoint."""
    import yaml as _yaml
    from finalize_checkpoint import finalize
    from run_calibration import note_restart_costs
    monkeypatch.setattr(optimizer, "least_squares", _fake_least_squares([2.3, 2.6], [False, False]))
    with pytest.raises(optimizer.RestartsFailed, match="failed to converge") as e:
        calibrate(_spec(), [], CalibrationConfig(n_restarts=2, seed=1, initialization_strategy="perturbed"))
    assert e.value.restart_costs == [2.3, 2.6]
    p = optimizer.initial_parameters(_spec())
    ck = tmp_path / "checkpoint.yaml"
    ck.write_text(_yaml.safe_dump({"call_count": 5, "best_cost_so_far": 2.3, "best_parameters_so_far":
                                   {n: float(getattr(p, n)) for n in p.param_names}}))
    note_restart_costs(str(ck), e.value)
    note_restart_costs(str(tmp_path / "missing.yaml"), e.value)  # no checkpoint: nothing to add, no crash
    m = finalize(str(ck), "synth_test_data", "vehicle_spec.yaml", ["03-03-2024"], str(tmp_path), 10.0)
    assert m["optimizer"]["restart_costs"] == [2.3, 2.6]
    assert calibration_report.restart_costs(str(tmp_path), m) == [2.3, 2.6]


def test_restarts_every_round_reach_every_fit(tmp_path, capsys):
    """B8 on URI's data: the good/bad split happens in a later current round, which one start can't fix."""
    calls = []
    assert cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "o"), "--estimate-current",
                     "--current-rounds", "2", "--restarts", "4", "--restarts-every-round", "--seed", "7"],
                    runner=lambda c: calls.append(c) or 0) == 0
    fits = _fits(calls)
    assert [_restarts(f) for f in fits] == [4, 4, 4]  # round 1, round 2, the final fit
    assert "every round takes about 4x" in capsys.readouterr().out
    calls.clear()
    assert cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "o2"), "--restarts", "3",
                     "--restarts-every-round"], runner=lambda c: calls.append(c) or 0) == 0
    assert [_restarts(f) for f in _fits(calls)] == [3]
    assert "only matters with --estimate-current" in capsys.readouterr().out
    assert cli.main(["calibrate", "--data", str(tmp_path), "--restarts-every-round"], runner=lambda c: 0) == 2
    assert "needs --restarts 2 or more" in capsys.readouterr().out


def test_a_warm_start_keeps_the_previous_result_as_its_first_point(monkeypatch):
    seen = []

    def fake(fun, x0, **kw):
        seen.append(np.array(x0, float))
        return types.SimpleNamespace(x=np.asarray(x0, float), cost=1.0, success=True, status=1, nfev=5,
                                     message="ok", optimality=0.0)
    monkeypatch.setattr(optimizer, "least_squares", fake)
    monkeypatch.setattr(optimizer, "mean_run_rmse", lambda *a, **k: 0.0)
    spec = _spec()
    lb, ub = optimizer.parameter_bounds(spec)
    prev = (lb + ub) / 2  # inside every bound, so no clipping hides a perturbation
    calibrate(spec, [], CalibrationConfig(n_restarts=3, seed=1, initialization_strategy="perturbed", x0_override=prev))
    free = lb < ub
    assert np.allclose(seen[0], prev[free])  # the previous round's result, unchanged
    assert not np.allclose(seen[1], prev[free]) and not np.allclose(seen[2], prev[free])
    seen.clear()
    calibrate(spec, [], CalibrationConfig(n_restarts=1, seed=1, initialization_strategy="perturbed", x0_override=prev))
    assert not np.allclose(seen[0], prev[free])  # one start behaves as before: perturbed around it


def test_the_report_lists_later_rounds_starts(tmp_path):
    for k, costs in ((1, [3.0, 2.9]), (2, [2.5, 2.4, 2.6]), (3, [2.2, 2.2001])):
        (tmp_path / f"current_round{k}").mkdir()
        (tmp_path / f"current_round{k}" / "metrics.json").write_text(json.dumps({"optimizer": {"restart_costs": costs}}))
    (tmp_path / "metrics.json").write_text(json.dumps({"optimizer": {"restart_costs": [2.1, 2.3]}}))
    rounds = calibration_report.later_round_restart_costs(str(tmp_path))
    assert rounds == [("round 2", [2.5, 2.4, 2.6]), ("round 3", [2.2, 2.2001]), ("the final fit", [2.1, 2.3])]
    p = tmp_path / "r.md"
    calibration_report.write_report(str(p), "x", [], None, [], [], None, ["Check the logs."], True, seed=5,
                                    restarts=[3.0, 2.9], later_restarts=rounds)
    text = p.read_text()
    assert "round 2: 2.5, 2.4, 2.6, a point around it found a lower cost" in text
    assert "round 3: 2.2, 2.2, the previous result stayed best" in text
    assert calibration_report.later_round_restart_costs(str(tmp_path / "nowhere")) == []
