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


def _result_dir(tmp_path, params, per_dof=None, rmse=None):
    d = tmp_path / "result"
    d.mkdir()
    (d / "calibrated_parameters.yaml").write_text(yaml.safe_dump({"parameters": params}))
    if per_dof is not None:
        m = {"per_dof": {"default_heldout": per_dof[0], "calibrated_heldout": per_dof[1]}}
        if rmse:
            m["rmse"] = rmse
        (d / "metrics.json").write_text(json.dumps(m))
    return str(d)


def _generic_dict():
    p = initial_parameters(load_vehicle_spec("vehicle_spec.yaml"))
    return {n: float(v) for n, v in zip(p.param_names, p.to_vector())}


def test_real_converged_calibration(tmp_path):
    shutil.copytree(REAL, tmp_path / "r")
    r = build_report(str(tmp_path / "r"), identifiability="identifiability_report.json")
    v = {x["axis"]: x for x in r["verdicts"]}
    assert v["r"]["verdict"] == GOOD                       # 51% better, nothing flagged
    assert v["p"]["verdict"] == CAUTION                     # 82% better, but Ieff_roll = 5.0 (19x reference)
    assert {p["name"] for p in r["implausible"]} == {"Ieff_roll", "Ieff_pitch"}
    assert len(r["pinned"]) == 9
    assert r["generalization_ratio"] == pytest.approx(1.0077, abs=1e-3)
    assert not any("Record more" in s for s in r["next_steps"])  # every axis already >= 30% better
    text = open(r["report"], encoding="utf-8").read()
    assert "65% more accurately" in text and "Parameters your data can't tell apart" in text


def test_worse_than_default_axis_gets_a_clear_warning_and_advice(tmp_path):
    d = _result_dir(tmp_path, _generic_dict(),
                    per_dof=([0.2, 0.2, 0.4, 1.2, 2.4, 0.4], [0.1, 0.1, 0.1, 0.2, 0.3, 0.5]),
                    rmse={"calibrated_train": 0.1, "calibrated_heldout": 0.15})
    r = build_report(d)
    v = {x["axis"]: x for x in r["verdicts"]}
    assert v["r"]["verdict"] == BAD and "WORSE" in v["r"]["why"]
    assert v["u"]["verdict"] == GOOD
    assert any("Record more" in s and "turning rate" in s for s in r["next_steps"])
    assert "50% worse than on training runs" in open(r["report"], encoding="utf-8").read()


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
