"""#90: an interrupted simulator fit is finished from its best point so far (`--finish`), as `strathe calibrate
--finish` does. The bluerov2_gz Gazebo fits ran 7-10 h."""
import json

import yaml

from simfit import cli
from test_simfit_external import WRONG, _description, _logs, _model


def _fit(tmp_path, *extra):
    data = _logs(tmp_path, seconds=60)
    desc = _description(tmp_path, _model(tmp_path / "vehicle.json", WRONG))
    return ["fit", "--sim", "external", "--simulator", desc, "--data", data, "--reference-point", "0", "0", "0",
            "--heldout-dates", "02-01-2030", "--workers", "2", "--out", str(tmp_path / "out"), *extra]


def test_a_fit_keeps_its_best_point_and_finish_reuses_it(tmp_path, capsys):
    args = _fit(tmp_path, "--max-iterations", "3")
    assert cli.main(args) == 0
    out = tmp_path / "out"
    ck = json.loads((out / "my_sim_fit_checkpoint.json").read_text())
    s = yaml.safe_load((out / "my_sim_fit.yaml").read_text())
    assert ck["complete"] and ck["values"] == s["after"] and "--finish" in capsys.readouterr().out
    # a finished fit, finished again: the same values, no new fit, nothing reported as interrupted
    assert cli.main(args + ["--finish"]) == 0
    s2 = yaml.safe_load((out / "my_sim_fit.yaml").read_text())
    assert s2["after"] == s["after"] and s2["finished_from_checkpoint"] == []
    assert "evaluation 1:" not in capsys.readouterr().out

    # interrupted: the checkpoint holds a best point that isn't the end
    ck.update(complete=False, values={k: v * 1.1 for k, v in ck["values"].items()},
              x=[v + 0.0953 for v in ck["x"]], status="interrupted: finished from its best point so far, not converged")
    (out / "my_sim_fit_checkpoint.json").write_text(json.dumps(ck))
    assert cli.main(args + ["--finish"]) == 0
    s3 = yaml.safe_load((out / "my_sim_fit.yaml").read_text())
    assert s3["finished_from_checkpoint"] == ["first"]
    report = (out / "my_sim_fit_report.md").read_text()
    assert "Finished from a checkpoint (`--finish`)" in report and "without `--finish`" in report


def test_finish_refuses_a_checkpoint_from_other_settings_and_says_when_there_is_none(tmp_path, capsys):
    args = _fit(tmp_path, "--max-iterations", "1")
    assert cli.main(args + ["--finish"]) == 2
    assert "no checkpoint" in capsys.readouterr().out
    assert cli.main(args) == 0
    assert cli.main(args + ["--finish", "--fit", "drag"]) == 2
    assert "written for other settings" in capsys.readouterr().out
    assert cli.main(args + ["--finish", "--heldout-dates", "01-01-2030"]) == 2
    assert "written for another heldout" in capsys.readouterr().out


def test_a_new_fit_removes_the_old_checkpoint(tmp_path):
    args = _fit(tmp_path, "--max-iterations", "1")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "my_sim_fit_checkpoint_cg_held.json").write_text("{}")
    assert cli.main(args) == 0
    assert not (tmp_path / "out" / "my_sim_fit_checkpoint_cg_held.json").exists()
