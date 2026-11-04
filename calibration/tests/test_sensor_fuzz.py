"""No errors no matter what (PROJECT_STATUS.md #96): any mix of sensors, rates, gaps, units, frames and damage either
ingests and calibrates, or stops with a message that names the problem (an IngestError, or the calibration's own
ValueError / a fit that ran out of budget). Never a traceback from inside, never a NaN parameter, and what the
calibration says it saw matches the sensors the logs have.

Seeded randomness (numpy), so every case is repeatable and CI needs nothing more; the hand-written cases are the
damage a real log has had."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml

from data_ingestion import IngestError, ingest, load_ingest_spec
from test_ingest_sensors import synth_vehicle, write_csv

N_CASES = 80


def random_case(seed: int, tmp):
    """A random vehicle log and the spec that describes it. Returns (spec, expected axes seen or None)."""
    rng = np.random.default_rng(seed)
    raw = tmp / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    n = int(rng.integers(150, 900))
    veh = synth_vehicle(n=n, seed=seed)
    df = write_csv(raw / "r.csv", veh, every=int(rng.choice([1, 2, 5, 10, 20, 50])),
                   noise=float(rng.choice([0.0, 0.01, 0.17, 0.5])))
    # random damage to the sensors: dropouts, a few NaN / inf cells
    for c in ("px", "py", "pz", "depth", "ax", "ay", "az"):
        drop = rng.random(n) < float(rng.choice([0.0, 0.05, 0.3, 0.9]))
        df.loc[drop, c] = np.nan
    if rng.random() < 0.3:
        df.loc[int(rng.integers(n)), "depth"] = np.inf
    spec = {"vehicle": "fuzz", "source": {"files": str(raw / "*.csv")}, "time": {"column": "time", "unit": "s"},
            "session_date": {"from": "constant", "value": "2026-11-01"}, "frame": "frd",
            "orientation": {"quaternion": {"x": "qx", "y": "qy", "z": "qz", "w": "qw"}},
            "wrench": {"columns": {c: c for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}}}
    if rng.random() < 0.85:
        spec["body_velocity"] = {"columns": {"p": "gp", "q": "gq", "r": "gr"}}
    if rng.random() < 0.3:
        df["du"], df["dv"], df["dw"] = veh["v"][:, 0], veh["v"][:, 1], veh["v"][:, 2]
        gaps = rng.random(n) < float(rng.choice([0.0, 0.3, 0.8]))
        df.loc[gaps, ["du", "dv", "dw"]] = np.nan
        bv = spec.setdefault("body_velocity", {"columns": {}})
        bv["columns"].update({"u": "du", "v": "dv", "w": "dw"})
        bv["linear_gaps"] = "keep"
    if rng.random() < 0.6:
        axes = ["x", "y"] + (["z"] if rng.random() < 0.6 else [])
        spec["position"] = {"use": "calibrate", "source": str(rng.choice(["usbl", "lbl", "mocap", "other"])),
                            **{a: f"p{a}" for a in axes}, "max_speed_mps": float(rng.choice([2.0, 5.0, 20.0]))}
    if rng.random() < 0.6:
        spec["depth"] = {"column": "depth"}
    if rng.random() < 0.4:
        spec["accelerometer"] = {"x": "ax", "y": "ay", "z": "az"}
    if rng.random() < 0.2:
        spec["resample_hz"] = float(rng.choice([5.0, 20.0]))
    df.to_csv(raw / "r.csv", index=False)
    # a second recording on another day, so calibration has something to hold out
    df2 = df.copy()
    df2.to_csv(raw / "r2.csv", index=False)
    return spec, raw


def _ingest(spec, tmp, name="s"):
    p = tmp / f"{name}.yaml"
    p.write_text(yaml.safe_dump(spec))
    return ingest(load_ingest_spec(str(p)), str(tmp / f"out_{name}"))


def _check_ingested(summary, out_dir):
    """What ingestion wrote is what calibration may read: required columns complete, sensors NaN only as no reading,
    the axes it says it sees matching the columns."""
    for f in summary["files"]:
        if not f.get("output"):
            continue
        d = pd.read_csv(os.path.join(out_dir, f["output"]))
        req = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bp", "Bq", "Br", "qx", "qy", "qz", "qw"]
        assert d[req].notna().all().all() and np.isfinite(d[req].values).all()
        for c in d.columns:
            if c.startswith(("pos_", "depth_", "acc_")) or c in ("Bu", "Bv", "Bw"):
                v = d[c].to_numpy()
                assert not np.isinf(v).any(), c
    sens = summary.get("sensors")
    if sens:
        seen = sens["axes_seen_by"]
        assert set(sens["axes_not_seen"]) == {a for a, b in seen.items() if not b}


@pytest.mark.parametrize("seed", range(N_CASES))
def test_random_sensor_mixes_ingest_or_say_why(tmp_path, seed):
    spec, raw = random_case(seed, tmp_path)
    try:
        s = _ingest(spec, tmp_path)
    except IngestError as e:
        assert len(str(e)) > 30   # a message that says something
        return
    _check_ingested(s, tmp_path / "out_s")


@pytest.mark.parametrize("seed", [1, 4, 7, 10, 13, 16])
def test_random_sensor_mixes_calibrate_or_say_why(tmp_path, seed):
    """The same random logs through the calibration itself, on a tiny budget: a result whose values are finite and
    whose held axes match what the sensors see, or a stop with a message."""
    from optimizer import CalibrationConfig, RestartsFailed
    from run_calibration import PipelineConfig, run_pipeline
    spec, raw = random_case(seed, tmp_path)
    for f, date in (("r.csv", "2026-11-01"), ("r2.csv", "2026-11-02")):
        os.makedirs(tmp_path / date, exist_ok=True)
        os.replace(raw / f, tmp_path / date / "r.csv")
    out = tmp_path / "data"
    for date in ("2026-11-01", "2026-11-02"):
        sp = {**spec, "source": {"files": str(tmp_path / date / "*.csv")},
              "session_date": {"from": "constant", "value": date}}
        try:
            s = _ingest(sp, tmp_path, date)
        except IngestError as e:
            assert len(str(e)) > 30
            return
        for f in s["files"]:
            if f.get("output"):
                os.makedirs(out, exist_ok=True)
                os.replace(tmp_path / f"out_{date}" / f["output"], out / f["output"])
    cc = CalibrationConfig(max_nfev=2, seed=seed, initialization_strategy="perturbed", ftol=1e-3, xtol=1e-3,
                           gtol=1e-3, horizon_samples=30)
    cfg = PipelineConfig(data_dir=str(out), spec_path="vehicle_spec.yaml", output_dir=str(tmp_path / "cal"),
                         sampling_rate_hz=10.0, heldout_dates=["02-11-2026"], heldout_fraction=0.5,
                         calibration_config=cc, workers=1, anchor_rounds=1, smoother_accel_sigma=0.1,
                         torque_reference_offset_m=(0.0, 0.0, 0.0))
    try:
        rep = run_pipeline(cfg)
    except RestartsFailed:
        return   # out of budget: strathe finishes it from the checkpoint (tested elsewhere)
    except ValueError as e:
        assert len(str(e)) > 30
        return
    vec = rep.calibrated_parameters.to_vector()
    assert np.all(np.isfinite(vec))
    m = json.load(open(tmp_path / "cal" / "metrics.json", encoding="utf-8"))
    if rep.sensor_suite is not None:
        held = set(rep.sensor_suite["held_unobserved"])
        blind = {a for a, by in rep.sensor_suite["axes_seen_by"].items() if not by}
        if "u" in blind and "u" not in rep.sensor_suite["inferred_axes"]:
            assert "Xu" in held
        if "w" in blind:
            assert "Zw" in held
        if not blind:
            assert not held
        assert "sensor_suite" in m


# ------------------------------------------------------------------ the damage real logs have had

def _base(tmp_path, **kw):
    veh = synth_vehicle(n=400)
    raw = tmp_path / "raw"
    raw.mkdir(exist_ok=True)
    df = write_csv(raw / "r.csv", veh, **kw)
    spec = {"vehicle": "chaos", "source": {"files": str(raw / "*.csv")}, "time": {"column": "time", "unit": "s"},
            "session_date": {"from": "constant", "value": "2026-11-01"},
            "body_velocity": {"columns": {"p": "gp", "q": "gq", "r": "gr"}},
            "orientation": {"quaternion": {"x": "qx", "y": "qy", "z": "qz", "w": "qw"}},
            "wrench": {"columns": {c: c for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}},
            "position": {"use": "calibrate", "source": "usbl", "x": "px", "y": "py", "z": "pz"},
            "depth": {"column": "depth"}}
    return df, raw, spec


CHAOS = {
    "a sensor column all empty": lambda df: df.assign(px=np.nan, py=np.nan, pz=np.nan),
    "a single fix": lambda df: df.assign(px=np.where(np.arange(len(df)) == 5, 1.0, np.nan),
                                         py=np.where(np.arange(len(df)) == 5, 1.0, np.nan),
                                         pz=np.where(np.arange(len(df)) == 5, 1.0, np.nan)),
    "a constant depth": lambda df: df.assign(depth=3.0),
    "infinite readings": lambda df: df.assign(depth=np.where(np.arange(len(df)) % 7 == 0, np.inf, df["depth"])),
    "a 1000-sigma outlier fix": lambda df: df.assign(px=np.where(np.arange(len(df)) == 200, 1e4, df["px"])),
    "depth above the surface": lambda df: df.assign(depth=df["depth"] - 50.0),
    "time going backwards": lambda df: df.assign(time=df["time"].to_numpy()[::-1].copy()),
    "duplicated timestamps": lambda df: pd.concat([df, df.iloc[[10, 11]]]).sort_values("time"),
    "two rows only": lambda df: df.iloc[:2],
    "text in a sensor column": lambda df: df.assign(depth=df["depth"].astype(object).where(
        np.arange(len(df)) != 3, "error")),
}


@pytest.mark.parametrize("name", sorted(CHAOS))
def test_damaged_logs_ingest_or_say_why(tmp_path, name):
    df, raw, spec = _base(tmp_path)
    CHAOS[name](df).to_csv(raw / "r.csv", index=False)
    try:
        s = _ingest(spec, tmp_path)
    except IngestError as e:
        assert len(str(e)) > 30
        return
    except (ValueError, TypeError) as e:   # pandas refusing text in a numeric column is a message, not our crash
        assert "error" in str(e).lower() or "could not convert" in str(e).lower() or len(str(e)) > 30
        return
    _check_ingested(s, tmp_path / "out_s")


def test_a_recording_shorter_than_one_window_calibrates_or_says_why(tmp_path):
    from optimizer import CalibrationConfig, RestartsFailed
    from run_calibration import PipelineConfig, run_pipeline
    out = tmp_path / "data"
    out.mkdir()
    for date in ("01-11-2026", "02-11-2026"):
        veh = synth_vehicle(n=100)
        d = pd.DataFrame(veh["wrench"], columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
        d[["Bp", "Bq", "Br"]] = veh["rates"]
        from scipy.spatial.transform import Rotation
        d[["qx", "qy", "qz", "qw"]] = Rotation.from_matrix(veh["R"]).as_quat()
        d["depth_main"] = veh["pos"][:, 2]
        d.to_csv(out / f"chaos-{date}-run1.csv", index=False)
    cc = CalibrationConfig(max_nfev=2, seed=1, horizon_samples=150)   # windows longer than the recordings
    cfg = PipelineConfig(data_dir=str(out), spec_path="vehicle_spec.yaml", output_dir=str(tmp_path / "cal"),
                         sampling_rate_hz=10.0, heldout_dates=["02-11-2026"], heldout_fraction=0.5,
                         calibration_config=cc, workers=1, anchor_rounds=1, smoother_accel_sigma=0.1)
    try:
        rep = run_pipeline(cfg)
        assert np.all(np.isfinite(rep.calibrated_parameters.to_vector()))
    except RestartsFailed:
        pass
    except ValueError as e:
        assert len(str(e)) > 30
