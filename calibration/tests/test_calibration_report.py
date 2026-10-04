"""calibration_report.py: verdicts, flags and advice on real and synthetic results."""
import json
import os
import shutil

import numpy as np
import pytest
import yaml

from calibration_report import BAD, CAUTION, GOOD, build_report
from optimizer import initial_parameters, load_vehicle_spec

HERE = os.path.dirname(__file__)
REAL = os.path.join(HERE, "..", "..", "calibration_output_10hz_coriolis")
DATA = os.path.join(HERE, "..", "..", "data", "raw", "Bluerov2_trajectory_dataset")


def _result_dir(tmp_path, params, per_dof=None, rmse=None, zero=None, excluded=None):
    d = tmp_path / "result"
    d.mkdir()
    (d / "calibrated_parameters.yaml").write_text(yaml.safe_dump({"parameters": params}))
    if per_dof is not None:
        m = {"per_dof": {"default_heldout": per_dof[0], "calibrated_heldout": per_dof[1]}}
        if rmse:
            m["rmse"] = rmse
        if zero is not None:
            m["skill_vs_zero"] = {"zero_heldout": zero}
        if excluded is not None:
            m["objective"] = {"excluded_axes": excluded}
        (d / "metrics.json").write_text(json.dumps(m))
    return str(d)


def _generic_dict():
    p = initial_parameters(load_vehicle_spec("vehicle_spec.yaml"))
    return {n: float(v) for n, v in zip(p.param_names, p.to_vector())}


@pytest.mark.skipif(not os.path.isdir(DATA), reason="public BlueROV2 dataset not present")
def test_real_converged_calibration(tmp_path):
    shutil.copytree(REAL, tmp_path / "r")
    r = build_report(str(tmp_path / "r"), identifiability="identifiability_report.json", data_dir=DATA)
    v = {x["axis"]: x for x in r["verdicts"]}
    assert v["r"]["verdict"] == GOOD and v["r"]["skill"] > 0.5  # yaw: real predictive skill
    # PROJECT_STATUS.md #50: roll and pitch beat the default model by 80+% but are
    # worse than predicting no motion -- the report must say "don't rely on it"
    for ax in "pq":
        assert v[ax]["improvement_pct"] > 80 and v[ax]["skill"] < 0
        assert v[ax]["verdict"] == BAD and "assuming the vehicle isn't moving" in v[ax]["why"]
    assert {p["name"] for p in r["implausible"]} == {"Ieff_roll", "Ieff_pitch"}
    assert len(r["pinned"]) == 9
    assert r["generalization_ratio"] == pytest.approx(0.9966, abs=1e-3)  # median of the per-axis ratios
    assert any("free-decay" in s and "roll rate" in s for s in r["next_steps"])
    text = open(r["report"], encoding="utf-8").read()
    # per-axis skill, never one average over the axes (the average was "65% more accurately", carried by roll and
    # pitch, which are no better than predicting no motion; PROJECT_STATUS.md #63)
    assert "more accurately than the default model on average" not in text
    assert "better than simply assuming no motion" in text and "Parameters your data can't tell apart" in text
    assert "No sign of memorising" in text
    assert "| p |" in text and "Skill" in text


def test_worse_than_default_axis_gets_a_clear_warning_and_advice(tmp_path):
    d = _result_dir(tmp_path, _generic_dict(),
                    per_dof=([0.2, 0.2, 0.4, 1.2, 2.4, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.5]),
                    rmse={"calibrated_train": 0.1, "calibrated_heldout": 0.15})
    r = build_report(d)
    v = {x["axis"]: x for x in r["verdicts"]}
    assert v["r"]["verdict"] == BAD and "WORSE" in v["r"]["why"]
    # without a no-motion baseline nothing can be trusted outright
    assert v["u"]["verdict"] == CAUTION and "couldn't be compared" in v["u"]["why"]
    assert any("Record more" in s and "turning rate" in s for s in r["next_steps"])
    assert "50% higher than on training runs" in open(r["report"], encoding="utf-8").read()


def test_easier_unseen_sessions_are_not_called_generalisation(tmp_path):
    """DFKI (#62/#63): the held-out runs were free, the training runs tethered, so held-out error came out lower.
    That can't show generalisation, and the report must not say it does."""
    d = _result_dir(tmp_path, _generic_dict(), per_dof=([0.15, 0.22, 0.21, 1.09, 1.08, 0.44],
                                                         [0.155, 0.033, 0.078, 0.136, 0.166, 0.196]),
                    zero=[0.392, 0.086, 0.128, 0.109, 0.164, 0.343],
                    rmse={"calibrated_train": 0.214, "calibrated_heldout": 0.139})
    mpath = os.path.join(d, "metrics.json")
    m = json.load(open(mpath))
    m["per_dof"]["calibrated_train"] = [0.228, 0.126, 0.080, 0.181, 0.248, 0.317]
    json.dump(m, open(mpath, "w"))
    r = build_report(d)
    text = open(r["report"], encoding="utf-8").read()
    assert r["generalization_ratio"] == pytest.approx(0.67, abs=0.01)
    assert "Can't tell from these sessions" in text and "generalises" not in text
    assert "4 of 6 axes better than simply assuming no motion" in text and "sideways speed (sway) +0.62" in text


def test_small_improvement_is_caution(tmp_path):
    d = _result_dir(tmp_path, _generic_dict(), per_dof=([1.0] * 6, [0.8] * 6))
    r = build_report(d)
    assert all(x["verdict"] == CAUTION and "only 20%" in x["why"] for x in r["verdicts"])


def test_without_accuracy_numbers_the_report_still_flags_problems(tmp_path):
    params = _generic_dict()
    params["Ieff_roll"] = 3.0
    r = build_report(_result_dir(tmp_path, params))
    assert r["verdicts"] == [] and r["implausible"][0]["name"] == "Ieff_roll"
    assert "No held-out accuracy numbers" in open(r["report"], encoding="utf-8").read()


def test_accuracy_computed_from_logs_when_metrics_lack_it(tmp_path):
    params = _generic_dict()
    params["Xu"] = -30.0
    r = build_report(_result_dir(tmp_path, params), data_dir="synth_test_data", heldout_dates=["03-03-2024"])
    assert len(r["verdicts"]) == 6
    assert all(np.isfinite(x["improvement_pct"]) for x in r["verdicts"])


def test_beating_default_is_not_enough_skill_decides(tmp_path):
    # roll 90% better than default, but worse than predicting no motion
    d = _result_dir(tmp_path, _generic_dict(),
                    per_dof=([0.5, 0.5, 0.5, 2.0, 2.0, 1.0], [0.1, 0.4, 0.1, 0.2, 0.12, 0.2]),
                    zero=[0.4, 0.45, 0.3, 0.15, 0.15, 0.5])
    v = {x["axis"]: x for x in build_report(d)["verdicts"]}
    assert v["p"]["verdict"] == BAD and v["p"]["skill"] == pytest.approx(1 - 0.2 / 0.15)
    assert v["u"]["verdict"] == GOOD and v["r"]["verdict"] == GOOD
    assert v["q"]["verdict"] == CAUTION and "only partly" in v["q"]["why"]   # skill 0.2
    assert v["v"]["verdict"] == CAUTION                                        # skill 0.11, improvement 20%


def test_excluded_axes_are_reported_as_not_calibrated(tmp_path):
    d = _result_dir(tmp_path, _generic_dict(), per_dof=([0.5] * 6, [0.1] * 6), zero=[0.4] * 6,
                    excluded=["p", "q"])
    r = build_report(d)
    v = {x["axis"]: x for x in r["verdicts"]}
    assert v["p"]["verdict"] == BAD and "not calibrated" in v["p"]["why"]
    assert v["q"]["excluded"] and v["u"]["verdict"] == GOOD
    assert any("free-decay" in s for s in r["next_steps"])
