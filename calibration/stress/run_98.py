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
    return set(DETECT) | set(REAL) | set(E2E)


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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    if cmd == "list":
        print("\n".join(sorted(all_rows())))
        sys.exit(0)
    if cmd == "detect":
        sys.exit(detect(sys.argv[2:] or None))
    if cmd == "e2e":
        sys.exit(e2e(sys.argv[2:] or None))
    if cmd == "all":
        detect()
        sys.exit(e2e())
    print(__doc__)
    sys.exit(2)
