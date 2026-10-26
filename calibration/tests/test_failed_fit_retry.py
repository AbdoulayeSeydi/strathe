"""A single-start fit that ends no better than the starting description (stuck), or where the training cost is rough
(the replay is chaotic there, so the optimizer's slopes are noise: it stalls or wanders until its budget runs out), is
retried from more starting points; one that ran out of budget at a smooth, better answer is not. A fit that stays no
better says so at the top of the report (PROJECT_STATUS.md #93: stuck was found by the one-session stress test, where
one start's training cost was 1599 against 20 for the same day's other split; wandering by #94's, cost 103 against 39)."""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

import calibration_report as CR
import run_calibration as RC
from optimizer import end_roughness

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
SMOOTH, ROUGH = 3e-15, 7.6e-8   # end_roughness of a sound fit, and of the wandering one (#94's CSV route)


def _logs(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()
    for f in sorted(glob.glob(os.path.join(CAL, "synth_test_data", "*.csv"))):
        pd.read_csv(f).iloc[:150].to_csv(d / os.path.basename(f), index=False)
    return d


def _fake(loaded, cost, rmse, success=True, rough=SMOOTH, moved=True):
    """What calibrate() returns for one start, without fitting. moved: the values left the starting description (the
    patched mean_run_rmse scores them 0.5 against the start's 1.0). A start that didn't converge is raised, as
    calibrate() does when no start converged."""
    import types
    from dynamics_model import CalibrationParameters
    from optimizer import CalibrationLog, CalibrationResult, RestartResult, RestartsFailed, initial_parameters
    p = initial_parameters(loaded)
    x = p.to_vector()
    if moved:
        x = x.copy()
        x[p.param_names.index("Xu")] *= 1.01
    raw = types.SimpleNamespace(nfev=3, njev=3, status=2 if success else 0, message="fake", optimality=0.0)
    rs = [RestartResult(x0=x, x=x, cost=cost, success=success, status=raw.status, nfev=3, message="fake",
                        raw_result=raw, roughness=rough)]
    if not success:
        raise RestartsFailed(1, rs)
    log = CalibrationLog("now", "trf", "soft_l1", 1, 1, cost, 3, 0.1)
    return CalibrationResult(parameters=CalibrationParameters.from_vector(x), converged=True, cost=cost, n_iterations=3,
                             train_rmse=rmse, restarts=rs, stability={"mean": cost, "std": 0.0, "convergence_rate": 1.0,
                                                                      "n_converged": 1, "n_total": 1},
                             log=log, scipy_result=raw)


def _run(tmp_path, monkeypatch, starts, extra=()):
    """starts: what each calibrate() call returns, in order, as _fake keyword arguments."""
    from optimizer import initial_parameters
    start = initial_parameters(RC.load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml"))).to_vector()
    calls = []

    def calibrate(loaded, runs, config):
        calls.append((config.n_restarts, config.seed, config.measure_roughness))
        if config.on_iteration:   # the real fit checkpoints every evaluation
            config.on_iteration(len(calls), start, starts[len(calls) - 1]["cost"])
        return _fake(loaded, **starts[len(calls) - 1])
    monkeypatch.setattr(RC, "calibrate", calibrate)
    monkeypatch.setattr(RC, "mean_run_rmse",
                        lambda params, *a, **k: 1.0 if np.allclose(params.to_vector(), start) else 0.5)
    out = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", ["run_calibration.py", "--data-dir", str(_logs(tmp_path)), "--output-dir", str(out),
                                      "--max-nfev", "2", "--initialization-strategy", "perturbed", "--seed", "3",
                                      "--spec", os.path.join(CAL, "vehicle_spec.yaml"), *extra])
    code = 0
    try:
        RC.main()
    except SystemExit as e:
        code = e.code
    m = out / "metrics.json"
    return calls, (json.loads(m.read_text()) if m.exists() else None), code, out


STUCK = dict(cost=1599.0, rmse=10.0, rough=1.5e-3, moved=False)
GOOD = dict(cost=20.0, rmse=0.01)


def test_a_stuck_single_start_is_retried_and_the_better_fit_kept(tmp_path, monkeypatch):
    calls, m, code, _ = _run(tmp_path, monkeypatch, [STUCK, GOOD])
    assert calls == [(1, 3, True), (1, 4, True)]   # the next start, with its own seed, was sound: no need for more
    auto = m["optimizer"]["auto_restart"]
    assert auto["why"] == "no_better" and auto["more_starts"] == 1 and auto["seeds_of_more_starts"] == [4]
    assert auto["first_cost"] == 1599.0 and auto["best_cost"] == 20.0 and auto["first_roughness"] == 1.5e-3
    assert m["optimizer"]["restart_costs"] == [1599.0, 20.0] and m["optimizer"]["n_restarts"] == 2 and code == 0
    assert "stopped no better than where it started" in CR.fit_note({"optimizer": m["optimizer"]})


def test_a_working_fit_isnt_retried_and_metrics_stay_as_before(tmp_path, monkeypatch):
    calls, m, code, _ = _run(tmp_path, monkeypatch, [GOOD])
    assert calls == [(1, 3, True)] and "auto_restart" not in m["optimizer"] and code == 0


def test_a_fit_that_wandered_out_of_budget_on_a_rough_cost_is_retried(tmp_path, monkeypatch):
    calls, m, code, _ = _run(tmp_path, monkeypatch, [dict(cost=103.0, rmse=0.5, success=False, rough=ROUGH), GOOD])
    assert calls == [(1, 3, True), (1, 4, True)]
    auto = m["optimizer"]["auto_restart"]
    assert auto["why"] == "rough" and not auto["first_converged"] and auto["first_cost"] == 103.0
    assert auto["best_cost"] == 20.0 and auto["best_converged"] and m["optimizer"]["converged"] and code == 0
    assert m["optimizer"]["restart_costs"] == [103.0, 20.0]
    assert "too sensitive" in CR.fit_note({"optimizer": m["optimizer"]})


def test_a_fit_out_of_budget_at_a_smooth_better_answer_is_finished_as_before(tmp_path, monkeypatch):
    """Common and harmless (112 of 276 results): no retry; run_calibration exits non-zero and strathe finishes the
    result from the checkpoint, as before #93."""
    calls, m, code, _ = _run(tmp_path, monkeypatch, [dict(cost=43.0, rmse=0.5, success=False)])
    assert calls == [(1, 3, True)] and code == 1 and m is None


def test_a_converged_fit_on_a_rough_cost_is_retried(tmp_path, monkeypatch):
    calls, m, _, _ = _run(tmp_path, monkeypatch, [dict(cost=60.0, rmse=0.4, rough=ROUGH), GOOD])
    assert calls == [(1, 3, True), (1, 4, True)] and m["optimizer"]["auto_restart"]["why"] == "rough"


def test_when_every_start_fails_the_lowest_cost_is_finished_with_the_retry_noted(tmp_path, monkeypatch):
    """No start converged: run_calibration exits non-zero and strathe finishes from the checkpoint (finalize_checkpoint);
    the retry's record is in the checkpoint, so the finished result still says it was retried."""
    import finalize_checkpoint as FC
    from checkpoint_io import read_checkpoint
    bad = dict(cost=150.0, rmse=0.5, success=False, rough=ROUGH)
    calls, m, code, out = _run(tmp_path, monkeypatch, [dict(cost=103.0, rmse=0.5, success=False, rough=ROUGH)] +
                               [bad] * RC.AUTO_RESTARTS)
    assert [c[1] for c in calls] == [3, 4, 5, 6] and code == 1 and m is None   # all AUTO_RESTARTS tried
    ck = read_checkpoint(str(out / "checkpoint.yaml"))
    assert ck["auto_restart"]["why"] == "rough" and not ck["auto_restart"]["best_converged"]
    assert ck["auto_restart"]["best_cost"] == 103.0   # the first start was still the lowest
    assert ck["restart_costs"] == [103.0, 150.0, 150.0, 150.0]
    assert FC.main(["--checkpoint", str(out / "checkpoint.yaml"), "--data-dir", str(tmp_path / "logs"),
                    "--spec", os.path.join(CAL, "vehicle_spec.yaml"), "--reason", "budget"]) == 0
    fm = json.loads((out / "metrics.json").read_text())
    assert fm["optimizer"]["auto_restart"]["why"] == "rough" and not fm["optimizer"]["converged"]


def test_a_retry_that_isnt_sound_keeps_trying(tmp_path, monkeypatch):
    """A start that converged but higher than the first, or lower but on a rough cost, doesn't stop the retry."""
    calls, m, _, _ = _run(tmp_path, monkeypatch, [STUCK, dict(cost=1700.0, rmse=0.01),
                                                  dict(cost=30.0, rmse=0.01, rough=ROUGH), dict(cost=25.0, rmse=0.01)])
    assert [c[1] for c in calls] == [3, 4, 5, 6] and m["optimizer"]["restart_costs"] == [1599.0, 1700.0, 30.0, 25.0]
    assert m["optimizer"]["auto_restart"]["best_cost"] == 25.0


def test_several_starts_arent_retried(tmp_path, monkeypatch):
    calls, _, _, _ = _run(tmp_path, monkeypatch, [STUCK], extra=("--n-restarts", "2"))
    assert calls == [(2, 3, False)]   # --restarts N (--n-restarts here) is left as asked, unprobed


def test_the_report_says_when_the_fit_failed():
    m = {"rmse": {"default_train": 0.895, "calibrated_train": 0.908}}
    assert CR.fit_note(m).startswith("> ❌ **The fit failed.**")
    m["optimizer"] = {"auto_restart": {"more_starts": 3, "first_cost": 1599.0, "best_cost": 1598.0}}
    assert "even from 4 starting points" in CR.fit_note(m)
    assert CR.fit_note({"rmse": {"default_train": 0.9, "calibrated_train": 0.16}}) is None


def test_end_roughness_tells_a_smooth_cost_from_a_jumpy_one():
    lb, ub = np.zeros(3), np.ones(3)
    x = np.full(3, 0.5)
    smooth = end_roughness(lambda v: v - 0.2, x, lb, ub, ())
    jumpy = end_roughness(lambda v: (v - 0.2) * (1 + 1e-3 * np.sin(1e12 * v)), x, lb, ub, ())
    assert smooth < RC.AUTO_RESTART_ROUGHNESS < jumpy
    assert end_roughness(lambda v: v * 0, x, lb, ub, ()) == 0.0   # a perfect fit isn't rough


def test_end_roughness_stays_inside_the_bounds():
    seen = []
    end_roughness(lambda v: seen.append(v.copy()) or v + 1, np.array([1.0, 0.0]), np.zeros(2), np.ones(2), ())
    assert len(seen) == 7 and all((0 <= v).all() and (v <= 1).all() for v in seen)
