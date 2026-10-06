"""
Roll and pitch from free-decay tank tests (free_decay.py, PROJECT_STATUS.md #64), on synthetic tests with known
answers (free_decay_synth.py integrates the truth with scipy's adaptive solver, not free_decay.py's own steps).
"""
import json
import math
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import free_decay as fd
from free_decay_synth import TRUTH, tank_test

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
ROOT = os.path.dirname(CAL)
SPEC = fd.spec_values(os.path.join(CAL, "vehicle_spec.yaml"))


def _close(got, want, rel):
    assert abs(got - want) <= rel * abs(want), (got, want)


def test_recovers_known_roll_and_pitch():
    r = fd.analyse([("roll.csv", tank_test("roll")), ("pitch.csv", tank_test("pitch", seed=1))], 10.0, SPEC)
    assert r["warnings"] == []
    roll, pitch = r["axes"]["roll"], r["axes"]["pitch"]
    assert roll["held"] and pitch["held"] and roll["releases"] == pitch["releases"] == 4
    _close(roll["Kp"], TRUTH["Kp"], 0.1)
    _close(roll["Kpp"], TRUTH["Kpp"], 0.15)
    _close(pitch["Mq"], TRUTH["Mq"], 0.1)
    _close(pitch["Mqq"], TRUTH["Mqq"], 0.15)
    _close(r["hold"]["cg_cb_offset_z"], TRUTH["cg_cb_offset_z"], 0.05)
    assert abs(roll["rests_at_deg"] - TRUTH["rest_roll_deg"]) < 0.2
    assert abs(pitch["rests_at_deg"] - TRUTH["rest_pitch_deg"]) < 0.2
    assert roll["r2_angle"] > 0.99
    assert set(r["hold"]) == {"Kp", "Kpp", "Mq", "Mqq", "cg_cb_offset_z"}


def test_works_at_50_hz():
    r = fd.analyse([("roll.csv", tank_test("roll", rate_hz=50))], 50.0, SPEC)
    _close(r["hold"]["Kp"], TRUTH["Kp"], 0.1)
    _close(r["hold"]["cg_cb_offset_z"], TRUTH["cg_cb_offset_z"], 0.05)
    assert "Mq" not in r["hold"]  # no pitch releases, nothing held for pitch
    assert any("pitch: no releases" in w for w in r["warnings"])


def test_release_starts_when_it_is_let_go_not_during_the_hold():
    """A hand holds it tilted for 6 s first: the release starts at the end of the hold, not during it."""
    df = tank_test("roll", hold_s=6.0)
    starts = [r.start for r in fd.find_releases(df, 10.0)]
    assert len(starts) == len(df.attrs["releases"])
    for got, true in zip(starts, df.attrs["releases"]):
        assert 0 <= got - true <= 2, (starts, df.attrs["releases"])


def test_thrust_on_is_never_a_release():
    df = tank_test("roll")
    df.loc[:, "Fx"] = 5.0  # driving forwards the whole time
    assert fd.find_releases(df, 10.0) == []
    r = fd.analyse([("driving.csv", df)], 10.0, SPEC)
    assert r["hold"] == {}
    assert any("roll: no releases" in w for w in r["warnings"])


def test_a_catch_mid_swing_ends_the_release():
    """Caught 2.5 s after letting go and tilted again: each release ends before the hand puts energy back in."""
    df = tank_test("roll", decay_s=2.5, starts_deg=(20, 20, 20))
    rels = fd.find_releases(df, 10.0)
    assert len(rels) == 3
    for rel, true in zip(rels, df.attrs["releases"]):
        assert rel.start + len(rel.angle) <= true + 25 + 1  # within its own 2.5 s


def test_tether_tugs_are_refused_not_held():
    """Tugs from outside make a swing the vehicle's own physics doesn't explain. Fitted anyway, a damping term goes
    to zero while the fit can still explain most of the motion; the axis must not be held."""
    for tug in (1.0, 3.0):
        r = fd.analyse([("tugged.csv", tank_test("roll", tug_nm=tug))], 10.0, SPEC)
        assert r["axes"]["roll"]["held"] is False, tug
        assert "Kp" not in r["hold"]
        assert any("roll: not held" in w for w in r["warnings"])


def test_a_steady_wobble_near_the_swing_frequency_is_refused():
    r = fd.analyse([("wobble.csv", tank_test("roll", wobble_deg=5))], 10.0, SPEC)
    assert r["axes"]["roll"]["held"] is False
    assert any("edge of what the fit allows" in w for w in r["warnings"])


def test_too_few_releases_are_not_held():
    r = fd.analyse([("two.csv", tank_test("roll", starts_deg=(15, -20)))], 10.0, SPEC)
    assert r["axes"]["roll"]["releases"] == 2 and r["axes"]["roll"]["held"] is False
    assert any("only 2 release(s)" in w for w in r["warnings"])


def test_roll_and_pitch_disagreeing_flags_the_inertia():
    """One vehicle has one righting arm: if pitch says otherwise, the spec's pitch inertia is off (here the true one
    is 0.40, the spec says 0.23) and the warning gives the ratio the tests imply."""
    r = fd.analyse([("roll.csv", tank_test("roll")), ("pitch.csv", tank_test("pitch", inertia=0.40, seed=2))],
                   10.0, SPEC)
    w = [x for x in r["warnings"] if "different righting arms" in x]
    assert w and "about 1.54" in w[0]
    assert "cg_cb_offset_z" in r["hold"]


def test_hold_file_and_bounds(tmp_path):
    p = tmp_path / "free_decay.yaml"
    p.write_text(yaml.safe_dump({"hold": {"Kp": -0.05, "Kpp": -1.0, "cg_cb_offset_z": 0.02}}))
    held = fd.hold_values(str(p))
    names = ["Kp", "Kpp", "cg_cb_offset_z"]
    out, notes = fd.within_bounds(held, names, np.array([-10, -10, -0.1]), np.array([-0.1, -0.2, 0.1]))
    assert out == {"Kp": -0.1, "Kpp": -1.0, "cg_cb_offset_z": 0.02}
    assert len(notes) == 1 and "Kp" in notes[0] and "widen the bounds" in notes[0]
    p.write_text(yaml.safe_dump({"hold": {"Xu": -3.0}}))
    with pytest.raises(ValueError, match="doesn't measure"):
        fd.hold_values(str(p))


def test_free_decay_setup_overrides_excluded_axis_values(tmp_path):
    from optimizer import initial_parameters, load_vehicle_spec
    from run_calibration import free_decay_setup
    spec = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml"))
    p = tmp_path / "free_decay.yaml"
    p.write_text(yaml.safe_dump({"hold": {"Kp": -0.6, "Kpp": -1.2, "cg_cb_offset_z": 0.018}}))
    info = free_decay_setup(str(p), spec, initial_parameters(spec))
    assert info["held"] == {"Kp": -0.6, "Kpp": -1.2, "cg_cb_offset_z": 0.018} and info["notes"] == []
    p.write_text(yaml.safe_dump({"hold": {}}))
    with pytest.raises(ValueError, match="holds nothing"):
        free_decay_setup(str(p), spec, initial_parameters(spec))


def _write_logs(folder, frames, rate_hz=10.0):
    folder.mkdir()
    for name, df in frames:
        df.to_csv(folder / name, index=False)
    (folder / "ingest_report.json").write_text(json.dumps({"sampling_rate_hz": rate_hz}))


def test_strathe_free_decay_then_calibrate_holds_the_measured_values(tmp_path):
    """The lab flow: `strathe free-decay` on the tank tests, then `strathe calibrate --free-decay` on the mission
    logs. The fit holds the measured values exactly, metrics.json and the report say so, and roll/pitch damping
    isn't listed as stuck at a limit."""
    tank = tmp_path / "tank"
    _write_logs(tank, [("tank-roll.csv", tank_test("roll")), ("tank-pitch.csv", tank_test("pitch", seed=1))])
    env = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
    r = subprocess.run([sys.executable, os.path.join(ROOT, "strathe.py"), "free-decay", "--data", str(tank),
                        "--out", str(tmp_path / "fd")], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "roll: 4 release(s)" in r.stdout and "-> held" in r.stdout
    fd_yaml = tmp_path / "fd" / "free_decay.yaml"
    held = yaml.safe_load(fd_yaml.read_text(encoding="utf-8"))["hold"]
    assert (tmp_path / "fd" / "free_decay_report.md").exists()

    logs = tmp_path / "logs"
    logs.mkdir()
    import glob
    for i, f in enumerate(sorted(glob.glob(os.path.join(CAL, "synth_test_data", "*.csv")))):
        pd.read_csv(f).iloc[:150].to_csv(logs / os.path.basename(f), index=False)
    out = tmp_path / "cal"
    r = subprocess.run([sys.executable, os.path.join(ROOT, "strathe.py"), "calibrate", "--data", str(logs),
                        "--out", str(out), "--max-nfev", "1", "--workers", "1", "--skip-identifiability",
                        "--free-decay", str(fd_yaml)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env, timeout=900)
    log = r.stdout + r.stderr
    assert r.returncode == 0, log[-3000:]
    params = yaml.safe_load((out / "calibrated_parameters.yaml").read_text(encoding="utf-8"))["parameters"]
    for k, v in held.items():  # held exactly (clamped into the spec's bounds, which these are inside)
        assert params[k] == pytest.approx(v, abs=1e-12), k
    obj = json.loads((out / "metrics.json").read_text(encoding="utf-8"))["objective"]
    assert obj["free_decay"]["held"] == held
    report = (out / "calibration_report.md").read_text(encoding="utf-8")
    assert "Roll and pitch from tank tests" in report
    assert "roll damping (`Kp`" not in report.split("## Be careful with")[1].split("##")[0]


def test_report_tailors_roll_pitch_advice_once_measured():
    from calibration_report import axis_verdicts, next_steps
    zero = np.ones(6)
    v = axis_verdicts(np.full(6, 2.0), np.array([0.3, 0.3, 0.3, 1.1, 1.1, 0.3]), set(), set(), zero_rmse=zero,
                      measured_axes=("p", "q"))
    roll = next(x for x in v if x["axis"] == "p")
    assert roll["measured"] and "free-decay tests" in roll["why"]
    steps = next_steps(v, [], [], None)
    assert any("Your free-decay tests measured" in s for s in steps)
    assert not any("(a free-decay test)" in s for s in steps)


def test_ardusub_tank_test_through_the_template(tmp_path):
    """The template on ArduSub DataFlash logs, written the way ArduSub records them (attitude in centidegrees, servo
    PWM): `strathe free-decay --ingest-spec` converts them and recovers the known values."""
    pytest.importorskip("pymavlink")
    sys.path.insert(0, HERE)
    from ardupilot_log_writer import write_bin
    (tmp_path / "tank_tests").mkdir()
    for name, df in (("roll", tank_test("roll")), ("pitch", tank_test("pitch", seed=1))):
        driving = (df[fd.WRENCH].abs() > 0).any(axis=1).to_numpy()
        rng = np.random.default_rng(3)
        for i in range(1, 9):
            df[f"pwm{i}"] = np.where(driving, rng.uniform(1400, 1600, len(df)).round(), 1500)
        write_bin(tmp_path / "tank_tests" / f"{name}.bin", df)
    spec = yaml.safe_load(open(os.path.join(CAL, "templates", "ingest_free_decay_ardupilot.yaml"), encoding="utf-8"))
    spec["source"]["files"] = str(tmp_path / "tank_tests" / "*.bin")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    r = subprocess.run([sys.executable, os.path.join(ROOT, "strathe.py"), "free-decay", "--ingest-spec",
                        str(tmp_path / "spec.yaml"), "--out", str(tmp_path / "fd")], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"},
                       timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    out = yaml.safe_load((tmp_path / "fd" / "free_decay.yaml").read_text(encoding="utf-8"))
    assert out["axes"]["roll"]["releases"] == out["axes"]["pitch"]["releases"] == 4
    _close(out["hold"]["Kp"], TRUTH["Kp"], 0.1)
    _close(out["hold"]["Mq"], TRUTH["Mq"], 0.1)
    _close(out["hold"]["cg_cb_offset_z"], TRUTH["cg_cb_offset_z"], 0.05)


def test_a_calibration_that_blows_up_on_unseen_recordings_is_reported_not_a_crash(monkeypatch):
    """Roll/pitch values held from the reference logs' free decays made the refit diverge on 9 of 14 held-out
    recordings' 3 s windows, and the finishing step crashed with a traceback (#64). Diverged recordings are now
    counted, left out of the skill, and the report opens with a warning."""
    import objective
    from calibration_report import diverged_note
    from optimizer import initial_parameters, load_vehicle_spec
    params = initial_parameters(load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml")))
    calls = iter([np.ones(6), FloatingPointError("NaN"), np.full(6, 3.0)])

    def fake(*a, **k):
        x = next(calls)
        if isinstance(x, Exception):
            raise x
        return x
    monkeypatch.setattr(objective, "rmse_by_dof", fake)
    mean, diverged = objective.stable_rmse_by_dof(params, [1, 2, 3], None)
    assert diverged == 1 and np.allclose(mean, 2.0)
    monkeypatch.setattr(objective, "rmse_by_dof", lambda *a, **k: (_ for _ in ()).throw(FloatingPointError()))
    mean, diverged = objective.stable_rmse_by_dof(params, [1, 2], None)
    assert diverged == 2 and np.isnan(mean).all()
    note = diverged_note({"skill_vs_zero": {"diverged_heldout": {"full_replay": 0, "short_windows": 9, "runs": 14}}})
    assert "9 of 14" in note and "Don't simulate with this calibration" in note
    assert diverged_note({"skill_vs_zero": {"diverged_heldout": {"full_replay": 0, "short_windows": 0,
                                                                 "runs": 14}}}) is None
    assert diverged_note({}) is None
