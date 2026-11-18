"""
stress/run_98.py -- stress test of small sensor delays found automatically (PROJECT_STATUS.md #98).

    python stress/run_98.py detect [ROW]   # the delay scan alone, as calibrate runs it (minutes a row)
    python stress/run_98.py e2e [ROW]      # full calibrations: does taking the delay off help (about 5 hours)
    python stress/run_98.py all
    python stress/run_98.py list

The standard proof chosen on 14 Nov 2026 (about 25 rows): the key variables only (delay size, number of recordings,
fix rate, depth, on-time and real data, a given delay_s, today's 0.5-2 s behaviour, every path). Not varied: fix
noise, dropouts, current, lever arms, description error, sample rate and the combinations of them (LIMITS.md).

The truth is hidden, as in #96 (synthetic_96.py): a known vehicle replays real recordings' thrust and orientation,
and its sensors report late by a chosen delay. D rows run state_smoother.fix_delays on the training runs with the
description's values and the smoother noise calibrate would use. E rows calibrate twice on the same data, once
finding the delay and once told `delay_s: 0` (the "before", in the same call: a row is never judged against another
row's later result, which is how #96's B-applied rows broke), and once on the same recordings with the sensor on time.

Bars are fixed here and hashed (BARS_HASH) before any row runs; a failure is a finding, never a moved bar.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, CAL)
sys.path.insert(0, HERE)
from common import PY, record  # noqa: E402
import synthetic_96 as S  # noqa: E402

F = "98"
WORK = os.path.expanduser("~/strathe_sensor_data/r98")
TRAIN, HELD = ("12-01-2024",), ("14-02-2024",)
HELD_FILE = "odom-14-02-2024-run1.csv"
URI_SPEC = os.path.expanduser("~/strathe_uri_lab_v2/data/alpha_rise_spec.yaml")
REAL = {   # real recordings whose position and depth sensors are on time (motion capture, a Bar30), and URI's GPS
    "D-real-tank": (os.path.expanduser("~/strathe_sensor_data/real/r1"), S.SPEC_BLUEROV2),
    "D-real-dfki": (os.path.expanduser("~/strathe_sensor_data/dfki/d1"), S.SPEC_BLUEROV2),
    "D-real-kth": (os.path.expanduser("~/strathe_sensor_data/marinarium/k96/k1"), S.SPEC_BLUEROV2),
    "D-real-uri": (os.path.expanduser("~/strathe_sensor_data/real/uri_nodvl"), URI_SPEC),
}

# --- the bars (fixed before any row ran) ------------------------------------------------------------------------------
BARS = {
    "never_wrong_s": 0.1,         # a delay taken off is within this of the truth
    "found_from": {"position": 0.3, "depth": 0.1},   # truths at or above these are taken off (A0's floor)
    "one_recording_found_from": 0.5,   # with one recording the scan before #98 decides: 0.5 s and up
    "gap_won_back": 0.5,          # end to end: skill >= before + this share of the gap to the sensor on time
    "given_within": 0.005,        # a given delay_s: skill within this of the same delay found, or of before (0 given)
    "regression_found_s": (0.7, 1.2),   # fixes 1 s late: found in this range and taken off, as before #98
}
BARS_HASH = hashlib.sha256(json.dumps(BARS, sort_keys=True).encode()).hexdigest()[:12]

NOMINAL = {"position": {"name": "usbl", "rate_hz": 1.0, "noise": 0.17, "dropout": 0.065},
           "depth": {"name": "main", "rate_hz": 10.0, "noise": 0.0024}}


def _sensors(position=None, depth=None) -> dict:
    out = json.loads(json.dumps(NOMINAL))
    out["position"].update(position or {})
    out["depth"].update(depth or {})
    return out


# D rows: (variable, value, sensors, training recordings, seed, sensor checked, truth delay)
DETECT = {
    **{f"D-pos-{x:g}": ("delay", x, _sensors(position={"delay_s": x}), 8, 96, "usbl", x) for x in (0.1, 0.2, 0.3, 0.4)},
    **{f"D-pos-0.3-s{s}": ("seed", s, _sensors(position={"delay_s": 0.3}), 8, s, "usbl", 0.3) for s in (962, 963)},
    **{f"D-pos-0.3-n{n}": ("recordings", n, _sensors(position={"delay_s": 0.3}), n, 96, "usbl", 0.3) for n in (1, 4, 16)},
    **{f"D-pos-0.3-rate{r:g}": ("fix rate", r, _sensors(position={"delay_s": 0.3, "rate_hz": r}), 8, 96, "usbl", 0.3)
       for r in (0.2, 10.0)},
    **{f"D-depth-{x:g}": ("depth delay", x, _sensors(depth={"delay_s": x}), 8, 96, "main", x) for x in (0.1, 0.2)},
    **{f"D-ontime-s{s}": ("on time", s, NOMINAL, 8, s, "both", 0.0) for s in (96, 962, 963)},
}
# E rows: (variable, value, what, sensors, the sensor delayed or None)
E2E = {
    "E-pos-0.3": ("delay end to end", 0.3, "fixes 0.3 s late: found and taken off, against delay_s: 0 and on time",
                  _sensors(position={"delay_s": 0.3}), "usbl"),
    "E-depth-0.2": ("delay end to end", 0.2, "depth 0.2 s late: found and taken off, against delay_s: 0 and on time",
                    _sensors(depth={"delay_s": 0.2}), "main"),
    "E-given-0.3": ("given delay_s", 0.3, "fixes 0.3 s late, delay_s: 0.3 given: taken as given",
                    _sensors(position={"delay_s": 0.3}), "usbl"),
    "E-given-0": ("given delay_s", 0.0, "fixes 0.3 s late, delay_s: 0 given: nothing taken off, as said",
                  _sensors(position={"delay_s": 0.3}), "usbl"),
    "E-regress-1": ("today's behaviour", 1.0, "fixes 1 s late: found and taken off, as before #98",
                    _sensors(position={"delay_s": 1.0}), "usbl"),
    "E-paths": ("every path", 0.3, "validate, report and a fit finished from its checkpoint apply the same delay",
                _sensors(position={"delay_s": 0.3}), "usbl"),
}


def all_rows() -> set:
    return set(DETECT) | set(REAL) | set(E2E) | set(ATTITUDE) | set(FIX_ROWS) | set(SLOW) | set(JUDGED)


# --- D rows ------------------------------------------------------------------------------------------------------------
def _dataset(name: str, sensors: dict, n_train: int, seed: int, held: bool = False) -> str:
    root = os.path.join(WORK, "data", name)
    if not os.path.exists(os.path.join(root, "truth")):
        shutil.rmtree(root, ignore_errors=True)
        src = S.reference_sources(TRAIN, n_train) + (S.reference_sources(HELD, 1) if held else [])
        S.truth_dataset(root, src, sensors=sensors, seed=seed)
    return root


def scan(data_dir: str, spec_path: str, files=None) -> dict:
    """state_smoother.fix_delays on these recordings, with what calibrate gives it: the description's values, its
    sensor mounts, the smoother noise (measurements.smoother_noise) and the chosen process noise."""
    import measurements as ms
    import state_smoother as sm
    from objective import DatasetConfig, load_run
    from optimizer import initial_parameters, load_vehicle_spec
    from run_calibration import _sensor_mounts, spec_torque_reference_offset
    spec = load_vehicle_spec(spec_path)
    params = initial_parameters(spec)
    dc = DatasetConfig(sampling_rate_hz=10.0, torque_reference_offset_m=tuple(spec_torque_reference_offset(spec)),
                       sensors=("auto",), lever_arms=_sensor_mounts(spec), actuators=spec.vehicle.actuators)
    files = files or sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    runs = [load_run(f, dc) for f in files]
    noise, _ = ms.smoother_noise(data_dir, ms.sensor_mounts(spec.raw), runs)
    sigma, _ = sm.choose_accel_sigma(runs, params, spec.vehicle, noise)
    return sm.fix_delays(runs, params, spec.vehicle, sigma, noise)


def _judge_detect(kind: str, truth: float, d: dict, n: int) -> tuple:
    took = d["delay_s"] if d.get("late") else 0.0
    if truth == 0:
        return took == 0.0, "on time: nothing taken off"
    if took:
        ok = abs(took - truth) <= BARS["never_wrong_s"] + 1e-9
        return ok, f"taken off within {BARS['never_wrong_s']} s of the truth ({truth} s)"
    floor = BARS["one_recording_found_from"] if n == 1 else BARS["found_from"][kind]
    must = truth >= floor - 1e-9
    return not must, (f"at {truth} s, at or above the {floor} s floor: must be found and taken off"
                      if must else f"under the {floor} s floor: not taken off (said)")


def detect(only=None) -> int:
    for row, (var, val, sensors, n, seed, sensor, truth) in DETECT.items():
        if only and not any(row == o or row.startswith(o) for o in only):
            continue
        t0 = time.time()
        root = _dataset(row, sensors, n, seed)
        res = scan(os.path.join(root, "data"), S.SPEC_BLUEROV2)
        names = ("usbl", "main") if sensor == "both" else (sensor,)
        oks, expects = [], []
        for name in names:
            kind = res[name]["kind"]
            ok, expect = _judge_detect(kind, truth if name == sensor or sensor == "both" else 0.0, res[name], n)
            oks.append(ok)
            expects.append(f"{name}: {expect}")
        nums = {"bars": BARS_HASH, "found": {k: res[k] for k in names}}
        record(F, row, var, val, f"{'; '.join(names)} late by {truth} s, {n} recordings, seed {seed}",
               "synthetic BlueROV2 (12 Jan recordings)", nums, "; ".join(expects), all(oks), time.time() - t0)
    for row, (data, spec) in REAL.items():
        if only and not any(row == o or row.startswith(o) for o in only):
            continue
        t0 = time.time()
        res = scan(data, spec)
        uri = row == "D-real-uri"
        took = {k: (d["delay_s"] if d.get("late") else 0.0) for k, d in res.items() if "delay_s" in d}
        # URI's real GPS delay isn't known (its recordings disagree: A0): the bar is that it isn't taken off, as today
        ok = all(v == 0.0 for v in took.values())
        expect = ("nothing taken off: the GPS's delay isn't known and its two recordings disagree, so it's said"
                  if uri else "real sensors that are on time: nothing taken off")
        record(F, row, "real data", os.path.basename(data), f"real {row[7:]}: every position and depth sensor",
               data, {"bars": BARS_HASH, "found": res}, expect, ok, time.time() - t0)
    return 0


# --- E rows ------------------------------------------------------------------------------------------------------------
def _spec_with(name: str, sensors: dict) -> str:
    raw = yaml.safe_load(open(S.SPEC_BLUEROV2, encoding="utf-8"))
    raw["sensors"] = sensors
    path = os.path.join(WORK, "specs", f"{name}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yaml.safe_dump(raw, open(path, "w", encoding="utf-8"), sort_keys=False)
    return path


def _fit(name: str, data_root: str, spec: str, nfev: int = 80, extra=()) -> dict:
    import run_96 as R
    out = os.path.join(WORK, "fits", name)
    if os.path.exists(os.path.join(out, "calibrated_parameters.yaml")) and not os.environ.get("R98_REFIT"):
        return {"exit": 0, "seconds": 0.0, "out": out, "reused": True}
    run = R._calibrate(os.path.join(data_root, "data"), out, list(extra), nfev=nfev, spec=spec, held=HELD)
    run["out"] = out
    return run


def _skill(fit: dict, root: str) -> dict:
    import run_96 as R
    if not os.path.exists(os.path.join(fit["out"], "calibrated_parameters.yaml")):
        return {"exit": fit["exit"], "failed": True}
    sc = R._score(fit["out"], root)
    return {"mean_skill": sc["mean_skill"], "skill": sc["skill"], "fix_delays": sc["fix_delays"]}


def _applied(sk: dict, sensor: str) -> float:
    d = (sk.get("fix_delays") or {}).get(sensor) or {}
    return float(d.get("applied_s") or 0.0)


def e2e(only=None) -> int:
    rows = [r for r in E2E if not only or any(r == o or r.startswith(o) for o in only)]
    if not rows:
        return 0
    ontime_root = _dataset("E-ontime", NOMINAL, 4, 96, held=True)
    ontime = _skill(_fit("E-ontime", ontime_root, S.SPEC_BLUEROV2), ontime_root)
    for row in rows:
        var, val, what, sensors, sensor = E2E[row]
        t0 = time.time()
        root = _dataset("E-" + "-".join(f"{k}{v.get('delay_s', 0):g}" for k, v in sensors.items()), sensors, 4, 96,
                        held=True)
        nums = {"bars": BARS_HASH, "on_time": ontime.get("mean_skill")}
        before = _skill(_fit(f"{row}-before", root, _spec_with(f"{row}-before", {sensor: {"delay_s": 0.0}})), root)
        nums["before"] = before.get("mean_skill")
        if row in ("E-pos-0.3", "E-depth-0.2", "E-regress-1"):
            auto = _skill(_fit(f"{row}-auto", root, S.SPEC_BLUEROV2), root)
            took = _applied(auto, sensor)
            nums.update(found=auto.get("fix_delays"), after=auto.get("mean_skill"), taken_off_s=took)
            if "failed" in auto or "failed" in before or "failed" in ontime:
                ok, expect = False, "three calibrations"
            else:
                bar = before["mean_skill"] + BARS["gap_won_back"] * (ontime["mean_skill"] - before["mean_skill"])
                near = (BARS["regression_found_s"][0] <= took <= BARS["regression_found_s"][1] if row == "E-regress-1"
                        else abs(took - val) <= BARS["never_wrong_s"] + 1e-9)
                ok = near and auto["mean_skill"] >= bar
                expect = (f"taken off near {val} s; mean skill >= {bar:.4f} (before {before['mean_skill']} + "
                          f"{BARS['gap_won_back']} of the gap to on time {ontime['mean_skill']})")
        elif row == "E-given-0.3":
            given = _skill(_fit(f"{row}-given", root, _spec_with(row, {sensor: {"delay_s": 0.3}})), root)
            nums.update(after=given.get("mean_skill"), taken_off_s=_applied(given, sensor),
                        applied_from=((given.get("fix_delays") or {}).get(sensor) or {}).get("applied_from"))
            auto = _skill(_fit("E-pos-0.3-auto", root, S.SPEC_BLUEROV2), root)
            ok = (nums["taken_off_s"] == 0.3 and "description" in str(nums["applied_from"])
                  and "failed" not in auto and abs(given["mean_skill"] - auto["mean_skill"]) <= 2 * BARS["never_wrong_s"])
            expect = "0.3 s taken off as the description says; skill close to the same delay found (E-pos-0.3)"
        elif row == "E-given-0":
            ok = _applied(before, sensor) == 0.0 and "failed" not in before
            expect = "nothing taken off, as the description says (the fixes are still found late, and said)"
        else:   # E-paths
            ok, expect, extra = _paths(root, sensor)
            nums.update(extra)
        record(F, row, var, val, what, "synthetic BlueROV2 (4 recordings of 12 Jan, 14 Feb held out)", nums, expect,
               bool(ok), time.time() - t0)
    return 0


def _paths(root: str, sensor: str) -> tuple:
    """validate and report reload the delay a calibration took off; a fit out of budget, finished from its checkpoint,
    takes the same one off."""
    short = _fit("E-paths-finish", root, S.SPEC_BLUEROV2, nfev=4)
    auto = os.path.join(WORK, "fits", "E-pos-0.3-auto")
    out = {}
    m = json.load(open(os.path.join(auto, "metrics.json")))
    took = ((m.get("sensor_suite") or {}).get("fix_delays") or {}).get(sensor, {}).get("applied_s")
    out["calibrate_took_s"] = took
    ms_ = json.load(open(os.path.join(short["out"], "metrics.json"))) if os.path.exists(
        os.path.join(short["out"], "metrics.json")) else {}
    out["finished_took_s"] = ((ms_.get("sensor_suite") or {}).get("fix_delays") or {}).get(sensor, {}).get("applied_s")
    v = subprocess.run([PY, "strathe_cli.py", "validate", "--result", auto, "--data", os.path.join(root, "data")],
                       cwd=CAL, capture_output=True, text=True)
    out["validate_exit"] = v.returncode
    rep = subprocess.run([PY, "strathe_cli.py", "report", "--result", auto, "--data", os.path.join(root, "data")],
                         cwd=CAL, capture_output=True, text=True)
    out["report_exit"] = rep.returncode
    text = ""
    for f in glob.glob(os.path.join(auto, "*report*.md")):
        text += open(f, encoding="utf-8").read()
    out["report_says_taken_off"] = f"{took:.1f} s was taken off" in text if took else None
    ok = (bool(took) and out["finished_took_s"] == took and v.returncode == 0 and rep.returncode == 0
          and out["report_says_taken_off"])
    return ok, "the same delay taken off by calibrate, a finished fit, and named in the report; validate and report run", out



# --- G rows: a gyro's and an orientation's latency, given in the ingest spec (Part B, after B0) -----------------------
# B0 found the logs can't tell a late gyro from a late orientation (so nothing is looked for); these rows prove the
# given keys. Their own bars, hashed apart from BARS (whose hash results already carry), fixed before any G row ran.
BARS_ATTITUDE = {
    "gyro_lag_left_s": 0.05,      # after ingest with the gyro's delay given, the gyro check reads at most this lag
    "gap_won_back": 0.5,          # held-out skill >= said + this share of the gap to the sensors on time
    "orientation_err_deg": 0.01,  # with the orientation's delay given, the ingested attitude is the truth's to this
    "orientation_no_worse": 0.002,   # and held-out skill isn't worse than with it said only, by more than this
    "subsample_rms_cut": 0.5,     # a delay under one sample: the gyro's RMS error to the truth at least halved
}
BARS_ATTITUDE_HASH = hashlib.sha256(json.dumps(BARS_ATTITUDE, sort_keys=True).encode()).hexdigest()[:12]
ATTITUDE = {   # tag: (gyro late s, orientation late s, given gyro delay, given orientation delay, raw rate Hz)
    "G-ontime": (0.0, 0.0, None, None, 10.0),
    "G-gyro-0.2-said": (0.2, 0.0, None, None, 10.0),
    "G-gyro-0.2-given": (0.2, 0.0, 0.2, None, 10.0),
    "G-orient-0.2-said": (0.0, 0.2, None, None, 10.0),
    "G-orient-0.2-given": (0.0, 0.2, None, 0.2, 10.0),
    "G-given-0": (0.0, 0.0, 0.0, 0.0, 10.0),
    "G-subsample-0.05": (0.05, 0.0, 0.05, None, 50.0),
    "G-refuse": (0.0, 0.0, -0.1, None, 10.0),
}


def _attitude_raw(tag: str) -> tuple:
    """Lab-style CSVs (time, thrust, gyro, quaternion, DVL) of the truth replaying 4 of 12 Jan's recordings and 1 of
    14 Feb's, with the gyro and/or the orientation logged late; and the truth's own velocity per recording."""
    import pandas as pd
    from scipy.spatial.transform import Rotation, Slerp
    from dynamics_model import SimulationConfig, TrajectoryInputs, simulate_trajectory
    from objective import DatasetConfig, load_run
    from optimizer import load_vehicle_spec
    from run_calibration import spec_torque_reference_offset
    g, o, _, _, rate = ATTITUDE[tag]
    root = os.path.join(WORK, "attitude", tag)
    raw = os.path.join(root, "raw")
    if os.path.exists(os.path.join(root, "raw.done")):
        return root, json.load(open(os.path.join(root, "truth_by_source.json")))
    shutil.rmtree(root, ignore_errors=True)
    os.makedirs(raw)
    spec = load_vehicle_spec(S.SPEC_BLUEROV2)
    truth = S.load_params(S.TRUTH_BLUEROV2)
    dc = DatasetConfig(sampling_rate_hz=10.0, torque_reference_offset_m=spec_torque_reference_offset(spec))
    by_source = {}
    for src in S.reference_sources(TRAIN, 4) + S.reference_sources(HELD, 1):
        run = load_run(src, dc)
        nu = simulate_trajectory(truth, spec.vehicle, SimulationConfig(dt=run.dt),
                                 TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                                  roll_pitch_series=run.roll_pitch)).nu
        t10 = np.arange(run.n_samples) * run.dt
        t = np.arange(0.0, t10[-1] + 1e-9, 1.0 / rate)
        rot = Slerp(t10, Rotation.from_quat(run.orientation_quat))
        df = pd.DataFrame({"time": 1.7e9 + t})
        for k, c in enumerate(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")):
            df[c] = np.interp(t, t10, run.forces_torques[:, k])
        for k, c in enumerate(("gp", "gq", "gr")):   # the gyro logged g s late: at t it reports the rate at t - g
            df[c] = np.interp(t - g, t10, nu[:, 3 + k])
        q = rot(np.clip(t - o, 0.0, t10[-1])).as_quat()
        for k, c in enumerate(("qx", "qy", "qz", "qw")):
            df[c] = q[:, k]
        for k, c in enumerate(("u", "v", "w")):
            df[c] = np.interp(t, t10, nu[:, k])
        name = os.path.basename(src)
        df.to_csv(os.path.join(raw, name), index=False)
        by_source[name] = nu.tolist()
    json.dump(by_source, open(os.path.join(root, "truth_by_source.json"), "w"))
    open(os.path.join(root, "raw.done"), "w").close()
    return root, by_source


def _attitude_spec(tag: str, root: str) -> str:
    _, _, gg, go, rate = ATTITUDE[tag]
    bv = {"columns": {"u": "u", "v": "v", "w": "w", "p": "gp", "q": "gq", "r": "gr"}}
    if gg is not None:
        bv["angular_delay_s"] = gg
    ori = {"quaternion": {"x": "qx", "y": "qy", "z": "qz", "w": "qw"}}
    if go is not None:
        ori["delay_s"] = go
    spec = {"vehicle": "synthetic_bluerov2", "source": {"files": os.path.join(root, "raw", "*.csv")},
            "time": {"column": "time", "unit": "s"}, "session_date": {"from": "filename"}, "frame": "frd",
            "body_velocity": bv, "orientation": ori,
            "wrench": {"columns": {c: c for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}}}
    if rate != 10.0:
        spec["resample_hz"] = 10.0
    path = os.path.join(root, "ingest.yaml")
    yaml.safe_dump(spec, open(path, "w"), sort_keys=False)
    return path


def _ingest(tag: str) -> tuple:
    """(ingest summary, data folder, truth folder): the raw CSVs through `strathe ingest`, and each output file's
    truth velocity written under its own name (as synthetic_96.truth_skill reads it)."""
    import pandas as pd
    from data_ingestion import ingest, load_ingest_spec
    root, by_source = _attitude_raw(tag)
    data, truth = os.path.join(root, "data"), os.path.join(root, "truth")
    shutil.rmtree(data, ignore_errors=True)
    shutil.rmtree(truth, ignore_errors=True)
    summary = ingest(load_ingest_spec(_attitude_spec(tag, root)), data)
    os.makedirs(truth)
    for f in summary["files"]:
        if f.get("output"):
            nu = np.array(by_source[os.path.basename(f["file"])])
            out = pd.read_csv(os.path.join(data, f["output"]))
            n = min(len(out), len(nu))
            pd.DataFrame(nu[:n], columns=["Bu", "Bv", "Bw", "Bp", "Bq", "Br"]).to_csv(
                os.path.join(truth, f["output"]), index=False)
    return summary, data, truth


def _attitude_fit(tag: str, data: str, truth: str) -> dict:
    import run_96 as R
    from dynamics_model import CalibrationParameters
    out = os.path.join(WORK, "attitude_fits", tag)
    if not os.path.exists(os.path.join(out, "calibrated_parameters.yaml")):
        R._calibrate(data, out, ["--sensors", "velocity"], nfev=80, spec=S.SPEC_BLUEROV2, held=HELD)
    if not os.path.exists(os.path.join(out, "calibrated_parameters.yaml")):
        return {"failed": True}
    p = CalibrationParameters(**yaml.safe_load(open(os.path.join(out, "calibrated_parameters.yaml")))["parameters"])
    held = sorted(f for f in os.listdir(data) if f.endswith(".csv") and HELD[0] in f)
    sk = S.truth_skill(p, S.SPEC_BLUEROV2, data, truth, held)
    return {"mean_skill": sk["mean_skill"], "skill": sk["skill"]}


def _gyro_lag(summary: dict) -> float:
    lags = [v for f in summary["files"] for v in (((f.get("sensor_checks") or {}).get("gyro_vs_orientation") or {})
                                                   .get("lag_s") or {}).values() if v is not None]
    return float(max(np.abs(lags))) if lags else float("nan")


def attitude(only=None) -> int:
    import pandas as pd
    from data_ingestion import IngestError
    from scipy.spatial.transform import Rotation
    rows = [r for r in ATTITUDE if not only or any(r == o or r.startswith(o) for o in only)]
    cache = {}

    def run_tag(tag):
        if tag not in cache:
            s, data, truth = _ingest(tag)
            cache[tag] = (s, data, truth, None)
        return cache[tag]

    def fitted(tag):
        s, data, truth, fit = run_tag(tag)
        if fit is None:
            fit = _attitude_fit(tag, data, truth)
            cache[tag] = (s, data, truth, fit)
        return fit

    for row in rows:
        t0 = time.time()
        nums = {"bars": BARS_ATTITUDE_HASH}
        try:
            if row == "G-refuse":
                try:
                    run_tag(row)
                    ok, expect = False, "a negative gyro delay is refused"
                except (IngestError, ValueError) as e:
                    nums["refused"] = str(e)[:240]
                    ok, expect = "0 s or more" in str(e), "a negative gyro delay is refused, saying it must be 0 s or more"
            elif row == "G-given-0":
                s0, d0, _, _ = run_tag(row)
                sp, dp, _, _ = run_tag("G-ontime")
                same = all(pd.read_csv(os.path.join(d0, f)).equals(pd.read_csv(os.path.join(dp, f)))
                           for f in sorted(os.listdir(dp)) if f.endswith(".csv"))
                said = all(f.get("gyro_delay_s") == 0 and f.get("orientation_delay_s") == 0 for f in s0["files"]
                           if f.get("output"))
                nums.update(identical=same, said=said)
                ok, expect = same and said, "0 given: the ingested logs are identical to none given, and the report says 0"
            elif row == "G-subsample-0.05":
                s, data, _, _ = run_tag(row)
                _, by_source = _attitude_raw(row)
                errs = {}
                for f in s["files"]:
                    if not f.get("output"):
                        continue
                    nu = np.array(by_source[os.path.basename(f["file"])])
                    got = pd.read_csv(os.path.join(data, f["output"]))[["Bp", "Bq", "Br"]].to_numpy()
                    n = min(len(got), len(nu)) - 2
                    late = np.stack([np.interp(np.arange(n) * 0.1 - 0.05, np.arange(len(nu)) * 0.1, nu[:, 3 + k])
                                     for k in range(3)], 1)
                    rms = lambda x: float(np.sqrt(np.mean((x[:n] - nu[:n, 3:6]) ** 2)))   # noqa: E731
                    errs[f["output"]] = (rms(got), rms(late))
                cut = float(np.mean([a / b for a, b in errs.values() if b > 0]))
                nums.update(rms_after_over_before=round(cut, 3))
                ok = cut <= 1 - BARS_ATTITUDE["subsample_rms_cut"]
                expect = f"50 Hz raw, gyro 0.05 s late, given: RMS error to the truth at most {1 - BARS_ATTITUDE['subsample_rms_cut']} of the late gyro's"
            elif row == "G-ontime":
                f = fitted(row)
                nums.update(mean_skill=f.get("mean_skill"))
                ok, expect = "failed" not in f, "the reference: the sensors on time, calibrated"
            elif row.endswith("-said"):
                s, _, _, _ = run_tag(row)
                warn = " ".join(((f.get("sensor_checks") or {}).get("gyro_vs_orientation") or {}).get("warning", "")
                                for f in s["files"])
                f = fitted(row)
                nums.update(mean_skill=f.get("mean_skill"), gyro_lag_seen_s=round(_gyro_lag(s), 3))
                ok = "failed" not in f and ("angular_delay_s" in warn if "gyro" in row else True)
                expect = ("not given: nothing shifted, and ingest says the gyro looks late and which key to give"
                          if "gyro" in row else "not given: nothing shifted")
            else:   # -given
                s, data, _, _ = run_tag(row)
                f, said, ref = fitted(row), fitted(row.replace("-given", "-said")), fitted("G-ontime")
                nums.update(mean_skill=f.get("mean_skill"), said=said.get("mean_skill"), on_time=ref.get("mean_skill"),
                            gyro_lag_left_s=round(_gyro_lag(s), 3))
                if "failed" in f or "failed" in said or "failed" in ref:
                    ok, expect = False, "three calibrations"
                elif "gyro" in row:
                    bar = said["mean_skill"] + BARS_ATTITUDE["gap_won_back"] * (ref["mean_skill"] - said["mean_skill"])
                    ok = nums["gyro_lag_left_s"] <= BARS_ATTITUDE["gyro_lag_left_s"] and f["mean_skill"] >= bar
                    expect = (f"after ingest the gyro check reads <= {BARS_ATTITUDE['gyro_lag_left_s']} s; mean skill >= "
                              f"{bar:.4f} (said {said['mean_skill']} + half the gap to on time {ref['mean_skill']})")
                else:
                    _, by_source = _attitude_raw("G-ontime")
                    ref_q = {}
                    for ff in run_tag("G-ontime")[0]["files"]:
                        if ff.get("output"):
                            ref_q[ff["output"]] = pd.read_csv(os.path.join(run_tag("G-ontime")[1], ff["output"]))[["qx", "qy", "qz", "qw"]].to_numpy()
                    err = 0.0
                    for ff in s["files"]:
                        if ff.get("output") and ff["output"] in ref_q:
                            q = pd.read_csv(os.path.join(data, ff["output"]))[["qx", "qy", "qz", "qw"]].to_numpy()
                            n = min(len(q), len(ref_q[ff["output"]])) - 3
                            rel = Rotation.from_quat(q[:n]).inv() * Rotation.from_quat(ref_q[ff["output"]][:n])
                            err = max(err, float(np.degrees(np.max(rel.magnitude()))))
                    nums["orientation_err_deg"] = round(err, 5)
                    ok = err <= BARS_ATTITUDE["orientation_err_deg"] and f["mean_skill"] >= said["mean_skill"] - BARS_ATTITUDE["orientation_no_worse"]
                    expect = (f"after ingest the orientation is the on-time one to {BARS_ATTITUDE['orientation_err_deg']} deg; "
                              f"mean skill >= said ({said['mean_skill']}) - {BARS_ATTITUDE['orientation_no_worse']}")
        except Exception as e:   # a crash is a failure, said
            ok, expect = False, "runs"
            nums["error"] = f"{type(e).__name__}: {e}"[:300]
        record(F, row, "given attitude delay", row[2:], ATTITUDE[row].__repr__(), "synthetic BlueROV2, lab-style CSVs through ingest",
               nums, expect, bool(ok), time.time() - t0)
    return 0



# --- F rows: recordings that disagree (after D-real-dfki failed) ------------------------------------------------------
# DELAY_DISAGREE_PER_RUN was chosen on A0's numbers after D-real-dfki failed (PROJECT_STATUS.md #98). These rows, with
# their own bars hashed apart, were fixed before any ran: the disagreeing cases have nothing taken off, and every
# consistent delay and on-time sensor of the D rows gives the same answer as before.
BARS_FIX = {"disagree_taken_off_s": 0.0, "never_wrong_s": 0.1, "found_from": {"position": 0.3, "depth": 0.1}}
BARS_FIX_HASH = hashlib.sha256(json.dumps(BARS_FIX, sort_keys=True).encode()).hexdigest()[:12]
MIXED = {   # row: (sensor changed, the delay in half the recordings)
    "F-depth-mixed": ("depth", 0.4), "F-pos-mixed": ("position", 0.4),
}
REGRESS = ["D-pos-0.3", "D-pos-0.4", "D-pos-0.3-s962", "D-pos-0.3-s963", "D-pos-0.3-n4", "D-pos-0.3-rate10",
           "D-depth-0.1", "D-depth-0.2", "D-ontime-s96"]
FIX_ROWS = ["F-real-dfki", *MIXED, *[f"F-{r[2:]}" for r in REGRESS], "F-real-tank", "F-real-kth", "F-real-uri"]


def _mixed_dataset(row: str) -> str:
    """8 of 12 Jan's recordings: the sensor late by the delay in recordings 1, 3, 5, 7 and on time in 2, 4, 6, 8."""
    sensor, x = MIXED[row]
    root = os.path.join(WORK, "data", row)
    if not os.path.exists(os.path.join(root, "truth")) or len(os.listdir(os.path.join(root, "truth"))) < 8:
        shutil.rmtree(root, ignore_errors=True)
        src = S.reference_sources(TRAIN, 8)
        S.truth_dataset(root, src[0::2], sensors=_sensors(**{sensor: {"delay_s": x}}), seed=96)
        S.truth_dataset(root, src[1::2], sensors=NOMINAL, seed=97)
    return root


def fix_rows(only=None) -> int:
    for row in FIX_ROWS:
        if only and not any(row == o or row.startswith(o) for o in only):
            continue
        t0 = time.time()
        nums = {"bars": BARS_FIX_HASH}
        if row in MIXED or row == "F-real-dfki":
            if row == "F-real-dfki":
                data, spec, name = REAL["D-real-dfki"][0], REAL["D-real-dfki"][1], "bar30"
            else:
                data, spec = os.path.join(_mixed_dataset(row), "data"), S.SPEC_BLUEROV2
                name = "main" if MIXED[row][0] == "depth" else "usbl"
            d = scan(data, spec)[name]
            took = d["delay_s"] if d.get("late") else 0.0
            nums["found"] = d
            ok = took == BARS_FIX["disagree_taken_off_s"] and d.get("disagreement_per_run", 0) > 0
            from state_smoother import DELAY_DISAGREE_PER_RUN
            ok = ok and d["disagreement_per_run"] > DELAY_DISAGREE_PER_RUN
            expect = "recordings that disagree: nothing taken off, and the disagreement over the bar (said in the report)"
            what = (f"{name}: half the recordings {MIXED[row][1]} s late, half on time" if row in MIXED
                    else "DFKI's real Bar30 (logged apart from its motion capture)")
        elif row.startswith("F-real-"):
            data, spec = REAL["D-real-" + row[7:]]
            res = scan(data, spec)
            nums["found"] = res
            ok = all((d["delay_s"] if d.get("late") else 0.0) == 0.0 for d in res.values() if "delay_s" in d)
            expect, what = "real sensors (and URI's unknown GPS): nothing taken off, as before", f"real {row[7:]}"
        else:   # a D row again, with the guard
            src_row = "D-" + row[2:]
            var, val, sensors, n, seed, sensor, truth = DETECT[src_row]
            root = _dataset(src_row, sensors, n, seed)
            res = scan(os.path.join(root, "data"), S.SPEC_BLUEROV2)
            names = ("usbl", "main") if sensor == "both" else (sensor,)
            oks, expects = [], []
            for nm in names:
                ok1, e1 = _judge_detect(res[nm]["kind"], truth, res[nm], n)
                oks.append(ok1)
                expects.append(f"{nm}: {e1}")
            nums["found"] = {k: res[k] for k in names}
            ok, expect, what = all(oks), "as " + src_row + ": " + "; ".join(expects), f"{src_row} again, with the guard"
        record(F, row, "recordings disagree", row[2:], what, "synthetic and real", nums, expect, bool(ok), time.time() - t0)
    return 0


# --- S rows: fixes too slow to show a small delay (after D-pos-0.3-rate0.2 failed) ------------------------------------
# DELAY_SLOW_FIX_S was chosen after D-pos-0.3-rate0.2 failed and a window experiment (120, 300, 600 s) found 0.2 Hz
# fixes don't show 0.3 s however long the window (PROJECT_STATUS.md #98). These rows and their bars were fixed before
# any ran: at 0.2 Hz nothing is taken off and the report says to give `delay_s:`; at 1 and 10 Hz, and on time, it doesn't.
BARS_SLOW = {"slow_taken_off_s": 0.0, "note_when_slow": True, "note_otherwise": False}
BARS_SLOW_HASH = hashlib.sha256(json.dumps(BARS_SLOW, sort_keys=True).encode()).hexdigest()[:12]
SLOW = {   # row: (the D row whose data it scans, slow)
    "S-pos-0.3-rate0.2": ("D-pos-0.3-rate0.2", True), "S-pos-0.3-rate1": ("D-pos-0.3", False),
    "S-pos-0.3-rate10": ("D-pos-0.3-rate10", False), "S-ontime": ("D-ontime-s96", False),
}


def slow_rows(only=None) -> int:
    from calibration_report import sensors_section
    for row, (src_row, slow) in SLOW.items():
        if only and not any(row == o or row.startswith(o) for o in only):
            continue
        t0 = time.time()
        var, val, sensors, n, seed, sensor, truth = DETECT[src_row]
        root = _dataset(src_row, sensors, n, seed)
        d = scan(os.path.join(root, "data"), S.SPEC_BLUEROV2)["usbl"]
        took = d["delay_s"] if d.get("late") else 0.0
        shown = {**d, "applied_s": took, **({"applied_from": "found in the fixes"} if took else {})}
        text = "\n".join(sensors_section({"sensor_suite": {"channels": ["usbl.n"], "fix_delays": {"usbl": shown}}}))
        noted = "too far apart" in text and "`delay_s:`" in text
        nums = {"bars": BARS_SLOW_HASH, "found": d, "noted": noted}
        if slow:
            ok = took == BARS_SLOW["slow_taken_off_s"] and noted == BARS_SLOW["note_when_slow"]
            expect = "fixes at 0.2 Hz: nothing taken off, and the report says they're too slow and to give delay_s"
        else:
            ok1, e1 = _judge_detect("position", truth, d, n)
            ok = ok1 and noted == BARS_SLOW["note_otherwise"]
            expect = f"as {src_row} ({e1}), and no slow-fix note"
        record(F, row, "fix rate", row[2:], f"{src_row}'s data", "synthetic BlueROV2 (12 Jan recordings)", nums,
               expect, bool(ok), time.time() - t0)
    return 0


# --- J rows: two #98 rows judged again, the way they should have been (after the follow-up PR) ------------------------
# G-gyro-0.2-given was judged by ingest's lag check, which reads p and q -0.3 s even on time on this known truth (its
# roll and pitch are replayed from a real recording); E-paths compared a fit stopped in round 1 with a full fit, which
# differ by design. These rows and their bars were fixed before any ran; the old rows stay as they failed.
BARS_JUDGED = {
    "gyro_err_given_rad_s": 1e-4,   # the delay given: the ingested gyro within this of the truth's rates (CSV rounding)
    "gyro_err_said_min_rad_s": 1e-3,   # not given: at least this far off, so the judge can see a late gyro
    "finished_equals_noted": True,   # a fit stopped mid-round, finished from its checkpoint, takes off what it noted
    "report_names_it": True,
}
BARS_JUDGED_HASH = hashlib.sha256(json.dumps(BARS_JUDGED, sort_keys=True).encode()).hexdigest()[:12]
JUDGED = {   # row: what
    "J-gyro-0.2-given": "G-gyro-0.2-given's data: the ingested gyro against the truth's rates",
    "J-gyro-0.2-said": "G-gyro-0.2-said's data: the same judge sees the late gyro",
    "J-finish-round1": "fixes 0.3 s late: calibrate stopped in round 1, then --finish",
    "J-finish-later": "fixes 0.3 s late: calibrate stopped after a later round scanned the delay again, then --finish",
}


def _gyro_err(tag: str) -> float:
    import pandas as pd
    summary, data, truth = _ingest(tag)
    worst = 0.0
    for f in summary["files"]:
        if not f.get("output"):
            continue
        got = pd.read_csv(os.path.join(data, f["output"]))[["Bp", "Bq", "Br"]].to_numpy()
        tru = pd.read_csv(os.path.join(truth, f["output"]))[["Bp", "Bq", "Br"]].to_numpy()
        n = min(len(got), len(tru)) - 5   # the last samples: nothing after the end to read at t + delay
        worst = max(worst, float(np.abs(got[:n] - tru[:n]).max()))
    return worst


def _stop_and_finish(row: str, root: str, later: bool) -> dict:
    """calibrate in the background; stopped once a checkpoint is newer than the delays noted in round 1 (or in a later
    round, noted again), then the same command with --finish."""
    import signal
    out = os.path.join(WORK, "fits", row)
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    argv = [PY, "strathe_cli.py", "calibrate", "--data", os.path.join(root, "data"), "--spec", S.SPEC_BLUEROV2,
            "--out", out, "--name", row, "--max-nfev", "80", "--seed", "7", "--workers", "3",
            "--heldout-dates", *HELD, "--skip-identifiability", "--unseen", "hold"]
    noted, ck = os.path.join(out, "applied_delays.json"), os.path.join(out, "checkpoint.yaml")
    log = open(out + ".log", "w", encoding="utf-8")
    proc = subprocess.Popen(argv, cwd=CAL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    notes, last = 0, None
    res = {"stopped": False}
    while proc.poll() is None:
        time.sleep(2)
        if os.path.exists(noted):
            m = os.path.getmtime(noted)
            if m != last:
                notes, last = notes + 1, m
        if notes >= (2 if later else 1) and os.path.exists(ck) and os.path.getmtime(ck) > last:
            res["noted"] = json.load(open(noted))
            res["notes_seen"] = notes
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait()
            res["stopped"] = True
            break
    log.close()
    if not res["stopped"]:
        res["error"] = "calibrate finished before it could be stopped"
        return res
    with open(out + ".finish.log", "w", encoding="utf-8") as f:
        res["finish_exit"] = subprocess.run(argv + ["--finish"], cwd=CAL, stdout=f, stderr=subprocess.STDOUT).returncode
    m = json.load(open(os.path.join(out, "metrics.json"))) if os.path.exists(os.path.join(out, "metrics.json")) else {}
    d = ((m.get("sensor_suite") or {}).get("fix_delays") or {}).get("usbl") or {}
    res["finished_applied_s"] = d.get("applied_s")
    res["applied_from_checkpoint"] = d.get("applied_from_checkpoint")
    text = "".join(open(f, encoding="utf-8").read() for f in glob.glob(os.path.join(out, "*report*.md")))
    took = res["noted"].get("usbl")
    res["report_says_it"] = bool(took) and f"{took:.1f} s was taken off" in text
    return res


def judged_rows(only=None) -> int:
    for row, what in JUDGED.items():
        if only and not any(row == o or row.startswith(o) for o in only):
            continue
        t0 = time.time()
        nums = {"bars": BARS_JUDGED_HASH}
        try:
            if row.startswith("J-gyro"):
                err = _gyro_err("G-" + row[2:])
                nums["gyro_max_err_rad_s"] = err
                if row.endswith("given"):
                    ok = err <= BARS_JUDGED["gyro_err_given_rad_s"]
                    expect = f"the ingested gyro within {BARS_JUDGED['gyro_err_given_rad_s']:g} rad/s of the truth's rates"
                else:
                    ok = err >= BARS_JUDGED["gyro_err_said_min_rad_s"]
                    expect = f"not given: at least {BARS_JUDGED['gyro_err_said_min_rad_s']:g} rad/s off (the judge sees it)"
            else:
                sensors = _sensors(position={"delay_s": 0.3})
                root = _dataset("E-" + "-".join(f"{k}{v.get('delay_s', 0):g}" for k, v in sensors.items()), sensors,
                                4, 96, held=True)
                r = _stop_and_finish(row, root, later=row.endswith("later"))
                nums.update(r)
                took = (r.get("noted") or {}).get("usbl")
                ok = (r.get("stopped") and r.get("finish_exit") == 0 and took is not None
                      and (r.get("finished_applied_s") == took) == BARS_JUDGED["finished_equals_noted"]
                      and r.get("report_says_it") == BARS_JUDGED["report_names_it"])
                expect = "finished from the checkpoint: the delay its round noted is the one taken off, and the report says it"
        except Exception as e:   # noqa: BLE001 -- a crash is a failed row, with its reason
            nums["error"] = f"{type(e).__name__}: {e}"[:300]
            ok, expect = False, "ran"
        record(F, row, "judged again", row[2:], what, "synthetic BlueROV2", nums, expect, bool(ok), time.time() - t0)
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        print("\n".join(sorted(all_rows())))
        sys.exit(0)
    if cmd == "detect":
        sys.exit(detect(sys.argv[2:] or None))
    if cmd == "e2e":
        sys.exit(e2e(sys.argv[2:] or None))
    if cmd == "fix":
        sys.exit(fix_rows(sys.argv[2:] or None))
    if cmd == "judged":
        sys.exit(judged_rows(sys.argv[2:] or None))
    if cmd == "slow":
        sys.exit(slow_rows(sys.argv[2:] or None))
    if cmd == "attitude":
        sys.exit(attitude(sys.argv[2:] or None))
    if cmd == "all":
        detect()
        sys.exit(e2e())
    print(__doc__)
    sys.exit(2)
