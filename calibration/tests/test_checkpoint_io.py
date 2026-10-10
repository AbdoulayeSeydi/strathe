"""
Checkpoints survive an interruption (checkpoint_io.py, PROJECT_STATUS.md #70). A URI calibration stopped mid-write left
an empty checkpoint.yaml (it was rewritten with open(path, "w") every evaluation) and its best result was lost.
"""
import os

import pytest
import yaml

import checkpoint_io
import finalize_checkpoint
import run_calibration
from checkpoint_io import CheckpointError, read_checkpoint, write_checkpoint
from optimizer import initial_parameters, load_vehicle_spec


def _vec():
    return initial_parameters(load_vehicle_spec("vehicle_spec.yaml")).to_vector()


def test_the_calibration_callback_writes_a_readable_checkpoint(tmp_path):
    path = str(tmp_path / "checkpoint.yaml")
    cb = run_calibration.make_checkpoint_callback(path)
    v = _vec()
    cb(1, v, 5.0)
    cb(2, v * 1.01, 7.0)  # worse: the best stays the first
    ck = read_checkpoint(path)
    assert ck["call_count"] == 2 and ck["best_cost_so_far"] == 5.0
    assert run_calibration.load_checkpoint_warm_start(path) == pytest.approx(v)
    assert [f for f in os.listdir(tmp_path) if f.endswith(".tmp")] == []


def test_a_write_stopped_halfway_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    path = str(tmp_path / "checkpoint.yaml")
    write_checkpoint(path, {"call_count": 1, "best_parameters_so_far": {"a": 1.0}})

    def dies_halfway(data, f, **kw):
        f.write("call_count: 2\nbest_param")  # the start of the new checkpoint, then the run is stopped
        raise KeyboardInterrupt

    monkeypatch.setattr(checkpoint_io.yaml, "dump", dies_halfway)
    with pytest.raises(KeyboardInterrupt):
        write_checkpoint(path, {"call_count": 2, "best_parameters_so_far": {"a": 2.0}})
    monkeypatch.undo()
    assert read_checkpoint(path) == {"call_count": 1, "best_parameters_so_far": {"a": 1.0}}
    assert os.listdir(tmp_path) == ["checkpoint.yaml"]  # no temporary file left behind


def test_json_checkpoints_too(tmp_path):
    path = str(tmp_path / "progress.json")
    write_checkpoint(path, {"completed_runs": 3}, fmt="json")
    assert yaml.safe_load(open(path)) == {"completed_runs": 3}


@pytest.mark.parametrize("content,match", [
    ("", "is empty"),
    ("call_count: 2\nbest_parameters_so_far: {a: [1,\n", "damaged"),
    ("call_count: 2\n", "isn't a calibration checkpoint"),
])
def test_a_damaged_checkpoint_is_explained(tmp_path, content, match):
    path = tmp_path / "checkpoint.yaml"
    path.write_text(content)
    with pytest.raises(CheckpointError, match=match):
        read_checkpoint(str(path))
    with pytest.raises(CheckpointError, match=match):
        run_calibration.load_checkpoint_warm_start(str(path))


def test_finishing_from_an_empty_checkpoint_says_why(tmp_path, capsys):
    (tmp_path / "checkpoint.yaml").write_text("")
    rc = finalize_checkpoint.main(["--checkpoint", str(tmp_path / "checkpoint.yaml"), "--data-dir", str(tmp_path)])
    assert rc == 2 and "is empty" in capsys.readouterr().out
    with pytest.raises(CheckpointError, match="no checkpoint"):
        read_checkpoint(str(tmp_path / "missing.yaml"))
