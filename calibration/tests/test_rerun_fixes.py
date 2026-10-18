"""Fixes from the full URI re-run (PROJECT_STATUS.md #76):
- F25: a calibration's random starting point is seeded, recorded, and repeatable with --seed;
- F9: a numerical blow-up is named as one, not blamed on the evaluation budget;
- F24: current estimation survives a recording on which the parameters blow up;
- a fleet config error is one plain line per problem.
(F23, unpatched Stonefish listing no parts, is in the C++ replay tool and was checked against a real unpatched build.)
"""
import os
import types

import pandas as pd
import pytest

import calibration_report
import estimate_current as ec
import run_calibration
import strathe_cli as cli


def _fits(calls):
    return [c for c in calls if c[1] == "run_calibration.py"]


def test_calibrate_records_a_seed_and_repeats_a_given_one(tmp_path, capsys):
    calls = []
    assert cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "o"), "--seed", "1234"],
                    runner=lambda c: calls.append(c) or 0) == 0
    fit = _fits(calls)[0]
    assert fit[fit.index("--seed") + 1] == "1234"
    assert "--seed 1234" in capsys.readouterr().out
    calls.clear()
    cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "o2")], runner=lambda c: calls.append(c) or 0)
    seed = int(_fits(calls)[0][_fits(calls)[0].index("--seed") + 1])
    assert 0 <= seed < 2 ** 31 and f"--seed {seed}" in capsys.readouterr().out


def test_every_current_round_starts_from_the_same_seed(tmp_path):
    calls = []
    cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "o"), "--estimate-current",
              "--current-rounds", "2", "--seed", "7"], runner=lambda c: calls.append(c) or 0)
    fits = _fits(calls)
    assert len(fits) == 3 and all(f[f.index("--seed") + 1] == "7" for f in fits)  # 2 rounds + the final fit
    assert all("--seed" not in c for c in calls if c[1] in ("estimate_current.py", "finalize_checkpoint.py"))


def test_a_blow_up_is_named_and_the_best_result_kept(tmp_path, capsys):
    calls = []

    def runner(cmd):
        calls.append(cmd)
        if cmd[1] == "run_calibration.py":
            d = cmd[cmd.index("--output-dir") + 1]
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "checkpoint.yaml"), "w").write("x")
            return run_calibration.EXIT_BLEW_UP
        return 0
    cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "o")], runner=runner)
    out = capsys.readouterr().out
    assert "blew up (NaN/inf)" in out and "usually the evaluation budget" not in out
    fin = next(c for c in calls if c[1] == "finalize_checkpoint.py")
    assert "blew up" in fin[fin.index("--reason") + 1] and "--seed" not in fin


def test_a_budget_stop_is_still_called_one(tmp_path, capsys):
    def runner(cmd):
        if cmd[1] == "run_calibration.py":
            d = cmd[cmd.index("--output-dir") + 1]
            os.makedirs(d, exist_ok=True)
            open(os.path.join(d, "checkpoint.yaml"), "w").write("x")
            return 1
        return 0
    cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "o")], runner=runner)
    assert "usually the evaluation budget" in capsys.readouterr().out


def _row(name):
    return {"file": name, "vx_mps": 0.05, "vy_mps": -0.02, "vz_mps": 0.0, "speed_mps": 0.054, "direction_deg": 338.0,
            "first_half": None, "second_half": None, "half_difference_mps": 0.01, "looks_constant": True,
            "heading_range_deg": 200.0, "error_explained_pct": 12.0, "error_still_water": 1.0,
            "error_with_current": 0.88, "samples": 100}


def test_an_unstable_recording_gets_no_current_and_the_others_still_do(monkeypatch, tmp_path):
    def fake(job):
        if "unstable" in job[0]:
            raise FloatingPointError("Simulation produced NaN/inf")
        return _row(os.path.basename(job[0]))
    monkeypatch.setattr(ec, "_estimate_run", fake)
    params = types.SimpleNamespace(to_vector=lambda: [0.0])
    rows = ec.estimate_currents(["a-run1.csv", "unstable-run1.csv"], params, None, None, workers=1)
    assert rows[0]["speed_mps"] == pytest.approx(0.054) and not rows[0].get("unstable")
    assert rows[1]["unstable"] and (rows[1]["vx_mps"], rows[1]["vy_mps"], rows[1]["vz_mps"]) == (0.0, 0.0, 0.0)
    assert "blew up" in ec.verdict(rows[1])
    for r in rows:
        r["model"] = "calibration"
    md = ec.write_outputs(rows, str(tmp_path / "currents.csv"), "a calibration")
    assert "No current for 1 recording" in open(md).read()
    from objective import read_currents_file
    assert dict(read_currents_file(str(tmp_path / "currents.csv")))["unstable-run1.csv"] == (0.0, 0.0, 0.0)


def test_the_report_says_how_to_repeat_the_fit(tmp_path):
    p = tmp_path / "r.md"
    calibration_report.write_report(str(p), "x", [], None, [], [], None, ["Check the logs."], True, seed=4242)
    assert "--seed 4242" in p.read_text()
    calibration_report.write_report(str(p), "x", [], None, [], [], None, ["Check the logs."], True)
    assert "--seed" not in p.read_text()


def test_a_bad_fleet_setting_is_one_plain_line(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rc = cli.main(["fleet", "--link", "optical", "--optical-terminals", "omni", "--optical-interval", "0.05",
                   "--save-config", str(tmp_path / "bad.yaml")], runner=lambda c: 0)
    out = capsys.readouterr().out
    assert "- communication.optical: optical beacon_interval_s must be a positive multiple of the 0.1 s tick" in out
    assert "pydantic" not in out and "validation error for" not in out
