"""#90: when the first fit makes the attitude worse by moving the centre of gravity, fit-sim fits again with it held and
keeps the better of the two; Gazebo's world copy runs the result at a safe physics step."""
import json
import os
import types

import pytest
import yaml

from simfit import core, runner
from simfit.core import prefer_cg_held
from simfit.knobs import KnobSpec


def _r(d60, att, diverged=0):
    return {"10s": {"drift_median_m": d60 / 3, "attitude_error_median_deg": att}, "30s": {"drift_median_m": d60 / 1.5},
            "60s": {"drift_median_m": d60}, "diverged": diverged, "windows": 100}


ORIGINAL = _r(12.0, [3.9, 7.8, 41.4])


def test_the_choice_between_the_two_fits():
    tuned = _r(6.0, [7.3, 17.3, 49.6])
    assert prefer_cg_held(ORIGINAL, tuned, _r(5.4, [3.5, 6.2, 40.9]))[0]              # bluerov2_gz: better everywhere
    assert prefer_cg_held(ORIGINAL, tuned, _r(6.5, [3.5, 6.2, 40.9]))[0]              # within 10% of the drift
    ok, why = prefer_cg_held(ORIGINAL, tuned, _r(8.0, [3.5, 6.2, 40.9]))
    assert not ok and "position accuracy" in why
    ok, why = prefer_cg_held(ORIGINAL, tuned, _r(5.4, [7.3, 17.3, 49.6]))
    assert not ok and "didn't make the attitude better" in why
    ok, why = prefer_cg_held(ORIGINAL, tuned, _r(5.4, [3.5, 6.2, 40.9], diverged=5))
    assert not ok and "blew up" in why


class FakeSim:
    """Scores depend on the values: a moved centre of gravity tilts the vehicle worse."""
    sim, label, groups = "fake", "Fake", ("drag", "cg")

    def __init__(self, path):
        self.model_path, self.notes, self.base, self.dt = path, [], None, 0.1

    def info(self):
        return {}

    def knob_specs(self, info, groups, axisymmetric):
        out = [KnobSpec("drag", "drag", 1.0)]
        return out + ([KnobSpec("cg_z", "cg", 0.0, "linear", -0.1, 0.1)] if "cg" in groups else [])

    def set_base(self, values):
        self.base = values

    def write_calibrated(self, values, out_dir, suffix):
        p = os.path.join(out_dir, f"model_{suffix}.json")
        json.dump(values, open(p, "w"))
        return p

    def changes(self, values):
        return [f"- {k}: {v}" for k, v in values.items()]


def _run(tmp_path, monkeypatch, held_drift, extra=()):
    fits = []

    def fake_fit(simulate, train, knobs, work, workers, max_iterations, log=None, weighting=None):
        names = knobs.names()
        fits.append({s.name: s.start for s in knobs.specs})
        values = {"drag": 2.0, "cg_z": 0.05} if "cg_z" in names else {"drag": 1.9}
        return {"values": values, "start_cost": 10.0, "cost": 1.0, "evaluations": 3, "status": "ok", "axis_scales": [1] * 6}

    def fake_score(adapter, held, models, work, densities):
        out = {}
        for label, values in models:
            if values is None:
                out[label] = ORIGINAL
            elif values.get("cg_z"):
                out[label] = _r(6.0, [7.3, 17.3, 49.6])
            else:
                out[label] = _r(held_drift, [3.5, 6.2, 40.9])
        return out, []
    run = types.SimpleNamespace(dt=0.1, date="d2")
    split = types.SimpleNamespace(train_groups=["d1"], heldout_groups=["d2"])
    monkeypatch.setattr(core, "load_runs", lambda data, held: ([run], [run], split))
    monkeypatch.setattr(core, "fit", fake_fit)
    monkeypatch.setattr(core, "windows", lambda *a: [])
    monkeypatch.setattr(core, "baselines", lambda rw: {"no motion": _r(3.3, [3.9, 6.3, 95.1])})
    monkeypatch.setattr(core, "plot", lambda *a, **k: None)
    monkeypatch.setattr(runner, "_score_models", fake_score)
    ap = __import__("argparse").ArgumentParser()
    runner.add_shared_arguments(ap, str(tmp_path / "out"))
    runner.add_fit_arguments(ap)
    a = ap.parse_args(["--data", "x", "--reference-point", "0", "0", "0", "--out", str(tmp_path / "out"), *extra])
    assert runner.run_fit(FakeSim(str(tmp_path / "model.json")), a) == 0
    out = tmp_path / "out"
    return fits, yaml.safe_load((out / "fake_fit.yaml").read_text()), (out / "fake_fit_report.md").read_text(), out


def test_attitude_worse_after_a_cg_fit_refits_with_it_held_and_keeps_the_better(tmp_path, monkeypatch):
    fits, s, report, out = _run(tmp_path, monkeypatch, held_drift=5.4)
    assert len(fits) == 2 and "cg_z" not in fits[1] and fits[1]["drag"] == 2.0   # warm start from the first fit
    assert s["cg_rerun"]["chosen"] == "held" and s["after"] == {"drag": 1.9}
    assert json.load(open(out / "model_calibrated.json")) == {"drag": 1.9}           # the one to use
    assert json.load(open(out / "model_calibrated_cg_tuned.json")) == {"drag": 2.0, "cg_z": 0.05}
    assert "Two fits were run" in report and "model_calibrated_cg_tuned.json" in report
    assert "(centre of gravity held)" in report and "(centre of gravity tuned)" in report


def test_the_first_fit_is_kept_when_holding_the_cg_costs_too_much(tmp_path, monkeypatch):
    fits, s, report, out = _run(tmp_path, monkeypatch, held_drift=9.0)
    assert s["cg_rerun"]["chosen"] == "tuned" and s["after"] == {"drag": 2.0, "cg_z": 0.05}
    assert json.load(open(out / "model_calibrated_cg_held.json")) == {"drag": 1.9}
    assert "is still the first fit" in report and "position accuracy" in report


def test_no_rerun_when_asked_or_when_the_cg_wasnt_tuned(tmp_path, monkeypatch):
    fits, s, report, _ = _run(tmp_path, monkeypatch, 5.4, ["--no-cg-rerun"])
    assert len(fits) == 1 and s["cg_rerun"] is None and "Be careful" in report and "--no-cg-rerun" in report
    fits, s, _, _ = _run(tmp_path, monkeypatch, 5.4, ["--fit", "drag"])
    assert len(fits) == 1 and s["cg_rerun"] is None
    fits, s, _, _ = _run(tmp_path, monkeypatch, 5.4, ["--hold", "cg_z=0.01"])
    assert len(fits) == 1 and s["cg_rerun"] is None
