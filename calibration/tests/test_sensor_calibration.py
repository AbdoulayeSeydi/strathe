"""The sensor suite through the whole calibration (PROJECT_STATUS.md #96): a vehicle with no DVL is calibrated from its
position fixes and depth, the axes it can't see are held and said to be, nothing it was told not to use reaches the fit,
and the report says what the sensors could calibrate."""
import json
import os

import numpy as np
import pytest
import yaml

import calibration_report as cr
from optimizer import CalibrationConfig
from run_calibration import PipelineConfig, run_pipeline

from test_measurements import SPEC, TRUTH, truth_frame

DATES = ("01-10-2026", "02-10-2026")


def write_dataset(d, dvl=False, n=700, runs_per_day=2, late_samples=0, **kw):
    d.mkdir(parents=True, exist_ok=True)
    k = 0
    for date in DATES:
        for r in range(runs_per_day):
            df, _, _ = truth_frame(n=n, seed=10 * k + 1, dvl=dvl, **kw)
            if late_samples:   # each fix logged this many samples after the vehicle was there
                cols = [c for c in df.columns if c.startswith("pos_")]
                df[cols] = df[cols].shift(late_samples)
            df.to_csv(d / f"synth-{date}-run{r + 1}.csv", index=False)
            k += 1
    return d


def calibrate(data, out, sensors=("auto",), nfev=40, horizon_s=None, rounds=2, fixed=None, spec="vehicle_spec.yaml",
              unseen="infer"):
    # starts at the truth (vehicle_spec.yaml's initial guess is the truth these recordings were made with): the
    # plumbing is what's tested here; recovering the truth from elsewhere is stress/run_96.py's
    # x0_override: an explicit start, which the failed-fit retry (#93) leaves alone. From the truth a first fit can't
    # beat its own start, so the retry would fire and refit these 80 s of data from random points
    cc = CalibrationConfig(max_nfev=nfev, seed=3, initialization_strategy="fixed", ftol=1e-4, xtol=1e-4, gtol=1e-4,
                           x0_override=TRUTH.to_vector(),
                           horizon_samples=None if horizon_s is None else int(horizon_s * 10),
                           fixed_params=fixed)
    cfg = PipelineConfig(data_dir=str(data), spec_path=str(spec), output_dir=str(out),
                         sampling_rate_hz=10.0, heldout_dates=[DATES[1]], heldout_fraction=0.5,
                         calibration_config=cc, workers=1, sensors=tuple(sensors), anchor_rounds=rounds, unseen=unseen,
                         # these recordings' torques are the truth's own (vehicle_spec.yaml's data convention shifts
                         # the reference dataset's by 8.5 cm, which would make them another vehicle's)
                         torque_reference_offset_m=(0.0, 0.0, 0.0))
    try:
        return run_pipeline(cfg)
    except Exception as e:   # a capped budget may not converge: what matters here is what it wrote and held
        if "converge" not in str(e):
            raise
        return None


@pytest.fixture(scope="module")
def no_dvl(tmp_path_factory):
    base = tmp_path_factory.mktemp("nodvl")
    data = write_dataset(base / "data", n=400, pos_every=10, depth_every=2)
    rep = calibrate(data, base / "out")
    return base, rep


def test_a_vehicle_with_no_dvl_calibrates_from_position_and_depth(no_dvl):
    base, rep = no_dvl
    assert rep is not None and rep.sensor_suite is not None
    ss = rep.sensor_suite
    assert ss["axes_seen_by"] == {"u": ["position"], "v": ["position"], "w": ["depth", "position"]}
    assert ss["held_unobserved"] == {}
    assert set(ss["channels"]) == {"gyro.p", "gyro.q", "gyro.r", "main.d", "usbl.n", "usbl.e", "usbl.d"}
    # at the truth, every sensor is predicted (all but) exactly on the held-out day
    sk = ss["fit"]["skill_heldout"]
    assert all(sk[k] > 0.95 for k in ("usbl.n", "usbl.e", "usbl.d", "main.d", "gyro.r")), sk
    m = json.load(open(base / "out" / "metrics.json", encoding="utf-8"))
    assert "sensor_suite" in m and m["objective"]["prediction_horizon_samples"] == 100
    assert np.isnan(np.array(m["per_dof"]["calibrated_heldout"][:3], dtype=float)).all()  # no DVL: not measured
    p = yaml.safe_load(open(base / "out" / "calibrated_parameters.yaml", encoding="utf-8"))
    assert p["sensors"]["held_unobserved"] == {}


def test_the_report_says_what_the_sensors_calibrated(no_dvl):
    base, _ = no_dvl
    out = cr.build_report(str(base / "out"), "vehicle_spec.yaml")
    text = open(out["report"], encoding="utf-8").read()
    assert "## What your sensors could calibrate" in text
    assert "Surge: calibrated from position fixes." in text and "Heave: calibrated from depth and position fixes." \
        in text
    surge = next(v for v in out["verdicts"] if v["axis"] == "u")
    assert surge["unit"] == "m" and "measured by the position fixes" in surge["why"]
    # judged by the sensors that measured them, with their skill (the description here IS the truth, so "better than
    # the default" can't be earned: that part of the verdict isn't what's tested)
    for v in out["verdicts"]:
        assert v["skill"] is not None and v["skill"] > 0.95, v
    assert {v["axis"]: v["unit"] for v in out["verdicts"]} == {"u": "m", "v": "m", "w": "m", "p": "rad/s",
                                                               "q": "rad/s", "r": "rad/s"}


def test_depth_only_infers_surge_and_sway_and_says_so(tmp_path):
    data = write_dataset(tmp_path / "data", n=400)
    calibrate(data, tmp_path / "out", sensors=("main",), rounds=1)
    m = json.load(open(tmp_path / "out" / "metrics.json", encoding="utf-8"))
    ss = m["sensor_suite"]
    assert ss["axes_seen_by"] == {"u": [], "v": [], "w": ["depth"]}
    assert ss["inferred_axes"] == ["u", "v"] and not {"Xu", "Xuu", "Yv", "Yvv"} & set(ss["held_unobserved"])
    out = cr.build_report(str(tmp_path / "out"), "vehicle_spec.yaml")
    v = {x["axis"]: x for x in out["verdicts"]}
    assert v["u"]["verdict"] == cr.BAD and "inferred, not measured" in v["u"]["why"]   # nothing here can check them
    text = open(out["report"], encoding="utf-8").read()
    assert "Surge: inferred, not measured." in text and "`--unseen hold`" in text
    assert "predict less well than they would with position fixes" not in text


def test_with_heave_unseen_too_nothing_is_inferred(tmp_path):
    import measurements as ms
    seen = {"u": [], "v": [], "w": []}
    assert ms.inferred_axes(seen) == set()
    assert {"Xu", "Yv", "Zw"} <= set(ms.unobservable_holds(seen, TRUTH, ms.inferred_axes(seen)))
    seen["w"] = ["depth"]
    assert ms.inferred_axes(seen) == {"u", "v"}
    assert not {"Xu", "Yv", "Zw"} & set(ms.unobservable_holds(seen, TRUTH, ms.inferred_axes(seen)))


def test_depth_only_holds_surge_and_sway_when_asked_and_says_so(tmp_path):
    data = write_dataset(tmp_path / "data", n=400)
    rep = calibrate(data, tmp_path / "out", sensors=("main",), rounds=1, unseen="hold")
    m = json.load(open(tmp_path / "out" / "metrics.json", encoding="utf-8"))
    ss = m["sensor_suite"]
    assert ss["axes_seen_by"] == {"u": [], "v": [], "w": ["depth"]}
    held = ss["held_unobserved"]
    assert {"Xu", "Xuu", "Xu_dot", "Yv", "Yvv", "Yv_dot"} <= set(held)
    params = yaml.safe_load(open(tmp_path / "out" / "calibrated_parameters.yaml", encoding="utf-8"))["parameters"]
    from optimizer import initial_parameters, load_vehicle_spec
    p0 = initial_parameters(load_vehicle_spec("vehicle_spec.yaml"))
    for k in held:   # held exactly at the description's values, not fitted
        assert params[k] == pytest.approx(getattr(p0, k), abs=0.0)
    out = cr.build_report(str(tmp_path / "out"), "vehicle_spec.yaml")
    v = {x["axis"]: x for x in out["verdicts"]}
    assert v["u"]["verdict"] == cr.BAD and "no sensor in your logs sees this motion" in v["u"]["why"]
    assert v["v"]["verdict"] == cr.BAD and v["w"]["unit"] == "m"
    text = open(out["report"], encoding="utf-8").read()
    assert "Surge: not calibrated." in text and "forward drag" in text
    assert "predict less well than they would with position fixes" in text   # the measured cost of no horizontal sensor (#96, R-3)
    del rep


def test_a_hidden_dvl_never_reaches_the_fit(tmp_path):
    """The DVL is in the files but not among the sensors asked for: the fit, its metrics and its report know nothing
    of it. Changing the hidden velocity changes nothing."""
    data = write_dataset(tmp_path / "data", dvl=True, n=400)
    calibrate(data, tmp_path / "a", sensors=("usbl", "main"), rounds=1)
    for f in data.glob("*.csv"):
        import pandas as pd
        d = pd.read_csv(f, float_precision="round_trip")   # every other column written back exactly as it was
        d[["Bu", "Bv", "Bw"]] = 99.0 * np.random.default_rng(0).normal(size=(len(d), 3))
        d.to_csv(f, index=False)
    calibrate(data, tmp_path / "b", sensors=("usbl", "main"), rounds=1)
    pa = yaml.safe_load(open(tmp_path / "a" / "calibrated_parameters.yaml", encoding="utf-8"))["parameters"]
    pb = yaml.safe_load(open(tmp_path / "b" / "calibrated_parameters.yaml", encoding="utf-8"))["parameters"]
    # equal to float noise: a leak of 99 m/s noise would move them by far more than this (Windows CI differed at 1e-8)
    assert pa == pytest.approx(pb, rel=1e-6, abs=1e-9)
    ma = json.load(open(tmp_path / "a" / "metrics.json", encoding="utf-8"))
    mb = json.load(open(tmp_path / "b" / "metrics.json", encoding="utf-8"))
    fa, fb = ma["sensor_suite"]["fit"], mb["sensor_suite"]["fit"]
    assert {k: set(v) for k, v in fa.items()} == {k: set(v) for k, v in fb.items()}
    for k in fa:
        assert fa[k] == pytest.approx(fb[k], rel=1e-6, abs=1e-9, nan_ok=True), k
    assert "dvl.u" not in ma["sensor_suite"]["channels"]


def test_a_dvl_at_every_sample_still_takes_the_original_fit(tmp_path):
    data = write_dataset(tmp_path / "data", dvl=True, n=300, runs_per_day=1)
    rep = calibrate(data, tmp_path / "out", rounds=1)
    m = json.load(open(tmp_path / "out" / "metrics.json", encoding="utf-8"))
    assert "sensor_suite" not in m and m["objective"]["prediction_horizon_samples"] is None
    del rep


def test_the_stress_scorer_predicts_in_the_water_the_truth_moved_in(tmp_path):
    """stress/synthetic_96.truth_skill scores a calibration in the current the truth vehicle was driven in: the truth's
    own values score ~1 there, and in still water they don't (the first sweep scored B-current-known and B-dvl-water
    without it, and their exact fits read 0.47 in sway)."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stress"))
    import synthetic_96 as S
    df, _, _ = truth_frame(n=600, seed=4)
    src = tmp_path / "synth-01-10-2026-run1.csv"
    df.to_csv(src, index=False)
    current = (0.15, -0.05, 0.0)
    S.truth_dataset(str(tmp_path / "ds"), [str(src)], sensors={"dvl": {}}, current_world=current)
    truth = S.load_params(S.TRUTH_BLUEROV2)
    args = (truth, S.SPEC_BLUEROV2, str(tmp_path / "ds" / "data"), str(tmp_path / "ds" / "truth"), [src.name])
    same = S.truth_skill(*args, current_world=current)
    still = S.truth_skill(*args)
    assert min(same["skill"]) > 0.999
    assert min(still["skill"]) < 0.99


def test_late_fixes_are_taken_off_unless_the_lab_says_otherwise(tmp_path):
    """A USBL reporting 1 s late (#96): calibration finds the delay, takes it off before the fit, says so, and the
    validation's reloaded recordings get the same delay off. With delay_s: 0 in the description, the fixes are taken
    as logged and the report says they still look late."""
    from objective import DatasetConfig, load_run
    from run_calibration import sensor_runs_from_metrics
    data = write_dataset(tmp_path / "data", n=400, pos_every=10, depth_every=2, late_samples=10)
    rep = calibrate(data, tmp_path / "found", nfev=5, rounds=1)
    d = rep.sensor_suite["fix_delays"]["usbl"]
    assert d["late"] and d["applied_from"] == "found in the fixes" and 0.8 <= d["applied_s"] <= 1.2, d
    text = open(cr.build_report(str(tmp_path / "found"), "vehicle_spec.yaml")["report"], encoding="utf-8").read()
    assert f"so {d['applied_s']:.1f} s was taken off them before the fit" in text
    m = json.load(open(tmp_path / "found" / "metrics.json", encoding="utf-8"))
    run = load_run(str(sorted(data.glob("*.csv"))[-1]), DatasetConfig(10.0, (0.0, 0.0, 0.0), sensors=("auto",)))
    sensor_runs_from_metrics([run], m, TRUTH, SPEC.vehicle)
    assert next(c for c in run.channels if c.name == "usbl").shifted_s == d["applied_s"]

    raw = yaml.safe_load(open("vehicle_spec.yaml", encoding="utf-8"))
    raw["sensors"] = {"usbl": {"delay_s": 0}}
    spec = tmp_path / "on_time.yaml"
    spec.write_text(yaml.safe_dump(raw), encoding="utf-8")
    # fixes left 1 s late make the truth a worse fit than it is, so a 5-evaluation budget can't settle them all:
    # what's tested here is what's taken off and said, so all but one are held
    names, x = list(TRUTH.param_names), TRUTH.to_vector()
    held = {n: float(v) for n, v in zip(names[1:], x[1:])}
    said = calibrate(data, tmp_path / "said", nfev=5, rounds=1, spec=spec, fixed=held)
    g = said.sensor_suite["fix_delays"]["usbl"]
    assert g["applied_s"] == 0.0 and g["applied_from"] == "the vehicle description's sensors.usbl.delay_s", g
    text = open(cr.build_report(str(tmp_path / "said"), str(spec))["report"], encoding="utf-8").read()
    assert "0 s of delay taken off, as the vehicle description's sensors.usbl.delay_s says." in text
    assert "check the value you gave" in text
    # the fixes taken off their delay are predicted better on the held-out day than the same fixes as logged
    assert rep.sensor_suite["fit"]["skill_heldout"]["usbl.n"] > said.sensor_suite["fit"]["skill_heldout"]["usbl.n"]
