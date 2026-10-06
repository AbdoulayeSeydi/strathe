"""
`strathe calibrate` end to end, for real: every step runs as its own process, as for a lab, on logs
whose dates are not the reference dataset's. Every other calibrate test replaces the steps with a
stub, which let the data check, the identifiability step and the report keep the reference dataset's
held-out dates as their defaults: every lab's calibration stopped at the data check (PROJECT_STATUS.md
#62). Small on purpose: six short recordings and the smallest optimizer budget.
"""
import glob
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
ROOT = os.path.dirname(CAL)
DATES = {"03-03-2024": "05-01-2030", "12-01-2024": "06-01-2030", "14-02-2024": "07-01-2030"}


def test_calibrate_runs_every_step_on_logs_with_their_own_dates(tmp_path):
    data = tmp_path / "logs"
    data.mkdir()
    for f in sorted(glob.glob(os.path.join(CAL, "synth_test_data", "*.csv"))):
        name = os.path.basename(f)
        for old, new in DATES.items():
            name = name.replace(old, new)
        pd.read_csv(f).iloc[:150].to_csv(data / name, index=False)
    out = tmp_path / "result"
    # output to a pipe that can't encode the symbols strathe prints, as on Windows (cp1252) and in its web page's
    # job logs: this crashed calibrate on Windows before the launcher switched to UTF-8
    env = {k: v for k, v in os.environ.items() if k != "PYTHONIOENCODING"}
    env.update({"MLFLOW_DISABLE_AGENT_HINT": "1", "LC_ALL": "C", "LANG": "C", "PYTHONCOERCECLOCALE": "0",
                "PYTHONUTF8": "0"})
    r = subprocess.run([sys.executable, os.path.join(ROOT, "strathe.py"), "calibrate", "--data", str(data),
                        "--name", "e2e", "--out", str(out), "--max-nfev", "1", "--workers", "1"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=900)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    assert "Traceback" not in log, log[-3000:]
    for step in ("[1/5]", "[2/5]", "[3/5]", "[4/5]", "[5/5]"):
        assert step in log
    for f in ("calibrated_parameters.yaml", "metrics.json", "identifiability_report.json", "calibration_report.md"):
        assert (out / f).exists(), f
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["dataset"]["heldout_dates"] == ["07-01-2030"]  # the lab's own last session
    report = (out / "calibration_report.md").read_text(encoding="utf-8")
    assert "Bottom line" in report
    # the comparison with the real vehicle (validation.py, #65): numbers, plots and the report's section
    assert (out / "validation" / "validation.json").exists()
    assert "## How the simulation compares with the real vehicle" in report


def test_one_session_is_explained_in_plain_words(tmp_path):
    """A lab with one day of logs is told what to do, at ingestion (a warning) and at calibration."""
    import pytest
    import yaml
    from data_ingestion import ingest, load_ingest_spec
    from data_split import split_by_date
    with pytest.raises(ValueError, match="one session .*morning and afternoon"):
        split_by_date([f"rov-05-01-2030-run{i}.csv" for i in (1, 2, 3)])
    raw = tmp_path / "raw"
    raw.mkdir()
    for i in (1, 2):
        d = pd.read_csv(os.path.join(CAL, "synth_test_data", "odom-03-03-2024-run1.csv")).iloc[:50]
        d.insert(0, "t", [k * 0.1 for k in range(len(d))])
        d.to_csv(raw / f"log{i}.csv", index=False)
    cols = ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")
    spec = {"vehicle": "one", "source": {"files": "raw/*.csv"}, "time": {"column": "t", "unit": "s"},
            "session_date": {"from": "constant", "value": "2030-01-05"},
            "body_velocity": {"columns": dict(zip("uvwpqr", ("Bu", "Bv", "Bw", "Bp", "Bq", "Br")))},
            "orientation": {"quaternion": {k: f"q{k}" for k in "xyzw"}},
            "wrench": {"columns": {c: c for c in cols}}}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    s = ingest(load_ingest_spec(str(tmp_path / "spec.yaml")), str(tmp_path / "out"))
    assert any("one session" in w for w in s["warnings"])
