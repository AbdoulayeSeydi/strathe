"""A single-start fit that ends no better than the starting description is retried from more starting points, and a
fit that stays no better says so at the top of the report (PROJECT_STATUS.md #93: found by the one-session stress
test, where one start's training cost was 1599 against 20 for the same day's other split)."""
import glob
import json
import os
import sys
import pandas as pd

import calibration_report as CR
import run_calibration as RC

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)


def _logs(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    for f in sorted(glob.glob(os.path.join(CAL, "synth_test_data", "*.csv"))):
        pd.read_csv(f).iloc[:150].to_csv(d / os.path.basename(f), index=False)
    return d


def _fake(loaded, n, cost, rmse):
    """What calibrate() returns, without fitting: the starting values, at the given cost and training error."""
    import numpy as np
    import types
    from optimizer import CalibrationLog, CalibrationResult, RestartResult, initial_parameters
    p = initial_parameters(loaded)
    x = p.to_vector()
    raw = types.SimpleNamespace(nfev=3, njev=3, status=2, message="fake", optimality=0.0)
    rs = [RestartResult(x0=x, x=x, cost=cost + i, success=True, status=2, nfev=3, message="fake", raw_result=raw)
          for i in range(n)]
    log = CalibrationLog("now", "trf", "soft_l1", n, n, cost, 3, 0.1)
    return CalibrationResult(parameters=p, converged=True, cost=cost, n_iterations=3, train_rmse=rmse, restarts=rs,
                             stability={"mean": cost, "std": 0.0, "convergence_rate": 1.0, "n_converged": n,
                                        "n_total": n}, log=log, scipy_result=raw)


def _run(tmp_path, monkeypatch, first_fails: bool):
    calls = []

    def calibrate(loaded, runs, config):
        calls.append((config.n_restarts, config.seed))
        if first_fails and len(calls) == 1:
            return _fake(loaded, 1, 1599.0, 10.0)   # stuck: no better than the starting description
        return _fake(loaded, config.n_restarts, 20.0, 0.01)
    monkeypatch.setattr(RC, "calibrate", calibrate)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run_calibration.py", "--data-dir", str(_logs(tmp_path)), "--output-dir", str(out),
                                      "--max-nfev", "2", "--initialization-strategy", "perturbed", "--seed", "3",
                                      "--spec", os.path.join(CAL, "vehicle_spec.yaml")])
    try:
        RC.main()
    except SystemExit:
        pass
    return calls, json.loads((out / "metrics.json").read_text())


def test_a_failed_single_start_is_retried_and_the_better_fit_kept(tmp_path, monkeypatch):
    calls, m = _run(tmp_path, monkeypatch, first_fails=True)
    assert calls == [(1, 3), (RC.AUTO_RESTARTS, 4)]   # the more starts get their own seed
    auto = m["optimizer"]["auto_restart"]
    assert auto["more_starts"] == RC.AUTO_RESTARTS and auto["seed_of_more_starts"] == 4
    assert auto["best_cost"] < auto["first_cost"]
    assert len(m["optimizer"]["restart_costs"]) == 1 + RC.AUTO_RESTARTS
    # the fake fits keep the starting values, so the report still (rightly) says the fit failed, from all 4 starts
    assert "even from 4 starting points" in CR.fit_note(m)


def test_a_working_fit_isnt_retried_and_metrics_stay_as_before(tmp_path, monkeypatch):
    calls, m = _run(tmp_path, monkeypatch, first_fails=False)
    assert calls == [(1, 3)] and "auto_restart" not in m["optimizer"]


def test_the_report_says_when_the_fit_failed():
    m = {"rmse": {"default_train": 0.895, "calibrated_train": 0.908}}
    assert CR.fit_note(m).startswith("> ❌ **The fit failed.**")
    m["optimizer"] = {"auto_restart": {"more_starts": 3, "first_cost": 1599.0, "best_cost": 1598.0}}
    assert "even from 4 starting points" in CR.fit_note(m)
    assert CR.fit_note({"rmse": {"default_train": 0.9, "calibrated_train": 0.16}}) is None
