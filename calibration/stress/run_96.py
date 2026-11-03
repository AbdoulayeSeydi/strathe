"""
stress/run_96.py -- stress test of the sensor suite (PROJECT_STATUS.md #96): calibrating a vehicle from what it measures.

    python stress/run_96.py fast        # the sensor matrix (64 mixes), and every product step without a DVL (an hour)
    python stress/run_96.py sweep [ROW] # synthetic truth recovery, one variable at a time (hours)
    python stress/run_96.py rescore [ROW]  # the same rows scored again from their finished fits (no refit)
    python stress/run_96.py matrix [ROW]   # the sensor matrix alone (128 rows: 64 mixes on two vehicles)
    python stress/run_96.py fs-real        # fit-sim from real sensors against the calibration, truth hidden
    python stress/run_96.py real DIR    # the real-data rows, scored from finished fits in DIR (see real_fits())
    python stress/run_96.py list

The proof is a hidden truth:
- Synthetic (B rows): a vehicle whose physics is known exactly (the shipped BlueROV2 calibration) replays real
  recordings' logged thrust and orientation; its sensors read that motion with chosen rates, noise, dropouts and
  latency (synthetic_96.py). The calibration starts from the vehicle description, not the truth, and is scored on a
  day it never saw against the truth's own velocity, which it was never given.
- Real (R, U rows): real recordings with their DVL or motion-capture velocity removed before calibrating, then
  scored against it.

Bars are fixed here, before any result: a row passes or fails on them, and a failure is a finding, never a moved bar.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, CAL)
sys.path.insert(0, HERE)
from common import PY, record  # noqa: E402
import synthetic_96 as S  # noqa: E402

F = "96"
WORK = os.path.expanduser("~/strathe_sensor_data")
TRAIN, HELD = ("12-01-2024",), ("14-02-2024",)
HELD_FILE = "odom-14-02-2024-run1.csv"

# --- the bars ---------------------------------------------------------------------------------------------------------
PERFECT_OF_V = 0.02    # perfect sensors: mean held-out truth skill within this of the DVL fit's
SEEN_SKILL = 0.90      # a sensor set within the usual range: every axis its sensors see at least this skill
DROPOUT_COST = 0.02    # SOLAQUA's 6.5% fix dropouts cost at most this much mean skill against none

NOMINAL = {"position": {"name": "usbl", "rate_hz": 1.0, "noise": 0.17, "dropout": 0.065},
           "depth": {"name": "main", "rate_hz": 10.0, "noise": 0.0024}}


def _with(**changes) -> dict:
    """The nominal sensors with some changed: changes are {sensor: {key: value}} or {sensor: None} to remove it."""
    out = json.loads(json.dumps(NOMINAL))
    for k, v in changes.items():
        if v is None:
            out.pop(k, None)
        elif k in out and isinstance(v, dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


# row: (variable, value, sensors, calibrate flags, kind of bar, what). kind: perfect | seen | limit | depth_only
SWEEP = {
    "B-ref-velocity": ("sensor set", "velocity", {"dvl": {}}, ["--sensors", "velocity"], "reference",
                       "the DVL fit, today's method, on the same data: the reference"),
    "B-set-mocap": ("sensor set", "mocap 10 Hz + depth", {"position": {"name": "mocap", "rate_hz": 10.0},
                                                          "depth": {"name": "main"}}, [], "perfect",
                    "no DVL: perfect position and depth"),
    "B-set-nominal": ("sensor set", "USBL-like + depth", NOMINAL, [], "seen",
                      "no DVL: 1 Hz fixes, 0.17 m noise, 6.5% dropouts (SOLAQUA), depth 0.24 cm"),
    "B-set-2d": ("sensor set", "horizontal fixes + depth", _with(position={"axes": "ne"}), [], "seen",
                 "fixes with no vertical (a 2-D system) and depth"),
    "B-set-depth": ("sensor set", "depth only", _with(position=None), [], "depth_only",
                    "depth and the gyro only: surge and sway can't be seen and must be held"),
    "B-set-dvlgaps": ("sensor set", "DVL 30% dropouts + fixes", _with(dvl={"dropout": 0.3}), ["--sensors", "all"],
                      "seen", "a DVL losing bottom lock 30% of the time, with the fixes"),
    "B-set-accel": ("sensor set", "+ accelerometer", _with(accelerometer={"noise": 0.02}), ["--use-accelerometer"],
                    "seen", "the nominal set plus the accelerometer (exploratory: reported against nominal)"),
    **{f"B-rate-{r:g}": ("position rate", r, _with(position={"rate_hz": r}), [], "seen" if r >= 0.5 else "limit",
                         f"fixes at {r:g} Hz") for r in (10.0, 2.0, 0.5, 0.2)},
    **{f"B-noise-{x:g}": ("position noise", x, _with(position={"noise": x}), [], "seen" if x <= 0.5 else "limit",
                          f"fix noise {x:g} m") for x in (0.0, 0.05, 0.5, 1.0)},
    **{f"B-dropout-{x:g}": ("position dropout", x, _with(position={"dropout": x}), [], "seen",
                            f"{100 * x:g}% of fixes lost") for x in (0.0, 0.3)},
    "B-dropout-surface": ("position dropout", "surface only", _with(position={"surface_only_below_m": 0.5}), [],
                          "limit", "fixes only within 0.5 m of the shallowest point (GPS at the surface)"),
    **{f"B-latency-{x:g}": ("position latency", x, _with(position={"delay_s": x}), [], "seen" if x <= 0.3 else "limit",
                            f"fixes {x:g} s late, uncorrected") for x in (0.3, 1.0)},
    **{f"B-depthnoise-{x:g}": ("depth noise", x, _with(depth={"noise": x}), [], "seen", f"depth noise {100 * x:g} cm")
       for x in (0.02, 0.1)},
    "B-depthrate-1": ("depth rate", 1.0, _with(depth={"rate_hz": 1.0}), [], "seen", "depth at 1 Hz"),
    **{f"B-gyronoise-{x:g}": ("gyro noise", x, _with(gyro={"noise": x}), [], "seen", f"gyro noise {x:g} rad/s")
       for x in (0.0056, 0.0175)},
    "B-gyrobias-0.0087": ("gyro bias", 0.0087, _with(gyro={"bias": 0.0087}), [], "limit", "gyro bias 0.5 deg/s"),
    **{f"B-horizon-{h:g}": ("horizon", h, NOMINAL, ["--prediction-horizon-s", str(h)], "seen" if 5 <= h <= 20 else
                            "limit", f"{h:g} s prediction windows") for h in (3.0, 5.0, 20.0, 60.0)},
    **{f"B-rounds-{k}": ("anchor rounds", k, NOMINAL, ["--anchor-rounds", str(k)], "seen", f"{k} anchor round(s)")
       for k in (1, 3)},
    # added 2026-11-02 (the plan's sweep values not yet run), bars fixed before running them
    "B-set-dvlgaps-0.7": ("sensor set", "DVL 70% dropouts + fixes", _with(dvl={"dropout": 0.7}), ["--sensors", "all"],
                          "seen", "a DVL that reads only 30% of the time, with the fixes"),
    **{f"B-accelbias-{x:g}": ("accelerometer bias", x, _with(accelerometer={"noise": 0.02, "bias": x}),
                              ["--use-accelerometer"], "seen", f"the accelerometer with a {x:g} m/s^2 bias")
       for x in (0.05, 0.2)},
    "B-rounds-4": ("anchor rounds", 4, NOMINAL, ["--anchor-rounds", "4"], "seen", "4 anchor rounds"),
    "B-gyronoise-0.00017": ("gyro noise", 0.00017, _with(gyro={"noise": 0.00017}), [], "seen",
                            "gyro noise 0.01 deg/s (a fibre-optic gyro)"),
    "B-grade-rtk": ("sensor noise", "grade rtk_f9p", _with(position={"name": "usbl", "noise": 0.0085}), [], "seen",
                    "RTK fixes (0.85 cm) at 1 Hz, their noise from the grade rtk_f9p in the description"),
    "B-grade-gnss": ("sensor noise", "grade gnss_f9p", _with(position={"name": "usbl", "noise": 1.27}), [], "limit",
                     "standalone GNSS fixes (1.27 m) at 1 Hz, their noise from the grade gnss_f9p"),
    "B-redundant-depth": ("redundant sensors", "two depth sensors",
                          _with(depth=[{"name": "main", "noise": 0.0024}, {"name": "aux", "noise": 0.02}]), [], "seen",
                          "two pressure sensors (0.24 cm and 2 cm), each its own channel"),
    "B-redundant-position": ("redundant sensors", "USBL + GPS",
                             _with(position=[{"name": "usbl", "rate_hz": 1.0, "noise": 0.17, "dropout": 0.065},
                                             {"name": "gps", "rate_hz": 1.0, "noise": 1.27, "offset": 3}]), [], "seen",
                             "a USBL (0.17 m) and a GPS (1.27 m) at once, each its own channel"),
    "B-dvlnoise-0.01": ("DVL noise", 0.01, _with(dvl={"dropout": 0.3, "noise": 0.01}), ["--sensors", "all"], "seen",
                        "a DVL with 1 cm/s noise (A50-class) losing lock 30% of the time, with the fixes"),
    "B-grade-nucleus": ("sensor noise", "grade nucleus1000", _with(dvl={"dropout": 0.3, "noise": 0.005}),
                        ["--sensors", "all"], "seen", "a Nucleus 1000-class DVL (0.5 cm/s), its noise from the grade"),
    "B-grade-bar02": ("sensor noise", "grade bar02", _with(depth={"noise": 0.00016}), [], "seen",
                      "a Bar02 depth sensor (0.16 mm), its noise from the grade"),
    # the delay check's second threshold (0.5 per fix), chosen after B-latency-1 failed the first (2.0), tested here on
    # recordings and noise it wasn't chosen on; bar fixed before running: said at 1 s and 2 s, not at 0 s
    **{f"B-latency-fresh-{x:g}": ("position latency", f"{x:g} s, fresh recordings", _with(position={"delay_s": x}), [],
                                  "fresh_delay", f"fixes {x:g} s late on other recordings and noise (the delay check's "
                                  f"threshold wasn't chosen on these)") for x in (0.0, 1.0, 2.0)},
    "B-current-estimated": ("current", "0.15 m/s estimated", NOMINAL, ["--estimate-current"], "seen",
                            "the same 0.16 m/s current, not told: --estimate-current"),
    **{f"B-sigma-{x:g}": ("smoother accel sigma", x, NOMINAL, ["--smoother-accel-sigma", str(x)], "seen",
                          f"the smoother's process noise forced to {x:g} m/s^2") for x in (0.01, 1.0)},
    "B-restarts-3": ("restarts", 3, NOMINAL, ["--restarts", "3"], "seen", "three starting points"),
    "B-current-known": ("current", "0.15 m/s known", NOMINAL, ["--current-world-mps", "0.15", "-0.05", "0"], "seen",
                        "a 0.16 m/s current the calibration is told"),
    "B-current-ignored": ("current", "0.15 m/s ignored", NOMINAL, [], "limit",
                          "the same current, not told (it shows as drag)"),
}
ARM = [0.3, 0.0, -0.2]
SWEEP.update({
    "B-lever-known": ("lever arm", "given", _with(position={"lever_arm": ARM}), [], "seen",
                      "the USBL transponder 0.36 m from the reference point, given in the description's sensors:"),
    "B-lever-ignored": ("lever arm", "ignored", _with(position={"lever_arm": ARM}), [], "limit",
                        "the same transponder, its mount not given"),
    "B-dvl-water": ("DVL track", "water, given", {"dvl": {"track": "water", "dropout": 0.3}}, ["--sensors", "all",
                    "--current-world-mps", "0.15", "-0.05", "0"], "seen",
                    "a water-tracking DVL in a 0.16 m/s current, given as track: water"),
    "B-dvl-water-ignored": ("DVL track", "water, not given", {"dvl": {"track": "water", "dropout": 0.3}},
                            ["--sensors", "all", "--current-world-mps", "0.15", "-0.05", "0"], "limit",
                            "the same DVL taken as bottom-tracking"),
})
# the smoother's reading noise (measurements.smoother_noise): these logs have no ingest report, so B-set-nominal uses the
# defaults (position 0.2 m, depth 0.02 m); the truth is 0.17 m and 0.24 cm
SWEEP.update({
    "B-grade-usbl": ("sensor noise", "grade ugps_g2, 85 m", NOMINAL, [], "seen",
                     "the USBL's grade (0.2% of an 85 m range: 0.17 m, its true noise) in the description's sensors:"),
    "B-noise-given": ("sensor noise", "given, right", NOMINAL, [], "seen",
                      "the true noise given as noise: (USBL 0.17 m, depth 0.24 cm)"),
    "B-noise-overconfident": ("sensor noise", "given, 30x too small", NOMINAL, [], "limit",
                              "the USBL's noise given as 5 mm: the smoother trusts the fixes far too much"),
})
SPEC_SENSORS = {"B-lever-known": {"usbl": {"position_m": ARM}}, "B-dvl-water": {"dvl": {"track": "water"}},
                "B-grade-rtk": {"usbl": {"grade": "rtk_f9p"}}, "B-grade-gnss": {"usbl": {"grade": "gnss_f9p"}},
                "B-grade-nucleus": {"dvl": {"grade": "nucleus1000"}}, "B-grade-bar02": {"main": {"grade": "bar02"}},
                "B-grade-usbl": {"usbl": {"grade": "ugps_g2", "range_m": 85}},
                "B-noise-given": {"usbl": {"noise": 0.17}, "main": {"noise": 0.0024}},
                "B-noise-overconfident": {"usbl": {"noise": 0.005}}}
CURRENT_ROWS = {"B-current-known": (0.15, -0.05, 0.0), "B-current-ignored": (0.15, -0.05, 0.0),
                "B-current-estimated": (0.15, -0.05, 0.0),
                "B-dvl-water": (0.15, -0.05, 0.0), "B-dvl-water-ignored": (0.15, -0.05, 0.0)}


def _spec_for(row: str) -> str:
    """The BlueROV2 description, with a sensors: section for the rows that give one."""
    if row not in SPEC_SENSORS:
        return S.SPEC_BLUEROV2
    raw = yaml.safe_load(open(S.SPEC_BLUEROV2, encoding="utf-8"))
    raw["sensors"] = SPEC_SENSORS[row]
    path = os.path.join(WORK, "specs", f"{row}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    yaml.safe_dump(raw, open(path, "w", encoding="utf-8"), sort_keys=False)
    return path


def _dataset(row: str, sensors: dict) -> str:
    root = os.path.join(WORK, "synthetic", row)
    if not os.path.exists(os.path.join(root, "data", HELD_FILE)):
        shutil.rmtree(root, ignore_errors=True)
        src = S.reference_sources(TRAIN, 2) + S.reference_sources(HELD, 1)
        if row.startswith("B-latency-fresh"):   # other training recordings, other noise
            src = S.reference_sources(TRAIN)[2:4] + S.reference_sources(HELD, 1)
        S.truth_dataset(root, src, sensors=sensors, seed=961 if row.startswith("B-latency-fresh") else 96,
                        current_world=CURRENT_ROWS.get(row))
    return root


def _calibrate(data: str, out: str, flags: list, nfev: int = 80, spec: str = S.SPEC_BLUEROV2, held=HELD) -> dict:
    t0 = time.time()
    log = out + ".log"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run([PY, "strathe_cli.py", "calibrate", "--data", data, "--spec", spec, "--out", out,
                             "--name", os.path.basename(out), "--max-nfev", str(nfev), "--seed", "7", "--workers", "3",
                             "--heldout-dates", *held, "--skip-identifiability", *flags],
                            cwd=CAL, stdout=f, stderr=subprocess.STDOUT).returncode
    return {"exit": rc, "seconds": round(time.time() - t0, 1), "log": log}


def _score(out: str, root: str, current=None) -> dict:
    from dynamics_model import CalibrationParameters
    p = CalibrationParameters(**yaml.safe_load(open(os.path.join(out, "calibrated_parameters.yaml")))["parameters"])
    sk = S.truth_skill(p, S.SPEC_BLUEROV2, os.path.join(root, "data"), os.path.join(root, "truth"), [HELD_FILE],
                       current_world=current)
    err = S.range_error(p, S.load_params(S.TRUTH_BLUEROV2))
    m = json.load(open(os.path.join(out, "metrics.json")))
    ss = m.get("sensor_suite") or {}
    return {"skill": sk["skill"], "mean_skill": sk["mean_skill"],
            "median_param_error_of_range": round(float(np.median(list(err.values()))), 4),
            "worst_params": sorted(((k, round(v, 3)) for k, v in err.items()), key=lambda x: -x[1])[:3],
            "axes_seen_by": ss.get("axes_seen_by"), "held_unobserved": sorted(ss.get("held_unobserved") or {}),
            "fix_delays": ss.get("fix_delays"),
            "converged": (m.get("optimizer") or {}).get("converged"), "params": p}


def _reference(name: str) -> dict:
    """A finished row's numbers (results_96.csv), e.g. the DVL fit's."""
    from common import load_rows
    rows = [r for r in load_rows(F) if r["row"] == name]
    return json.loads(rows[-1]["numbers"]) if rows else {}


def sweep(only=None, rescore: bool = False) -> int:
    from optimizer import initial_parameters, load_vehicle_spec
    default = initial_parameters(load_vehicle_spec(S.SPEC_BLUEROV2))
    for row, (var, val, sensors, flags, kind, what) in SWEEP.items():
        if only and row not in only and not any(row.startswith(o) for o in only):
            continue
        root = _dataset(row, sensors)
        out = os.path.join(WORK, "fits", row)
        if rescore:   # the scorer changed, not the fit: the fit's own exit and time are kept
            if not os.path.exists(os.path.join(out, "calibrated_parameters.yaml")):
                continue
            old = _reference(row)
            run = {"exit": old.get("exit", 0), "seconds": old.get("seconds", 0.0)}
        else:
            run = _calibrate(os.path.join(root, "data"), out, flags, spec=_spec_for(row))
        nums = {"exit": run["exit"], "seconds": run["seconds"]}
        if not os.path.exists(os.path.join(out, "calibrated_parameters.yaml")):
            record(F, row, var, val, what, "synthetic BlueROV2 (reference recordings)", nums, "a calibration", False,
                   run["seconds"])
            continue
        sc = _score(out, root, CURRENT_ROWS.get(row))
        nums.update({k: v for k, v in sc.items() if k != "params"})
        nums["default_skill"] = S.truth_skill(default, S.SPEC_BLUEROV2, os.path.join(root, "data"),
                                              os.path.join(root, "truth"), [HELD_FILE],
                                              current_world=CURRENT_ROWS.get(row))["skill"]
        ok, expect = _judge(row, kind, nums, sc)
        record(F, row, var, val, what, "synthetic BlueROV2 (reference recordings)", nums, expect, ok, run["seconds"])
    return 0


def _judge(row: str, kind: str, nums: dict, sc: dict) -> tuple:
    sk = nums["skill"]
    seen = nums.get("axes_seen_by") or {}
    ref = _reference("B-ref-velocity")
    v_mean = ref.get("mean_skill")
    blind = [a for a, by in seen.items() if not by]
    held = set(nums.get("held_unobserved") or [])
    held_ok = (("u" not in blind or {"Xu", "Xuu", "Xu_dot"} <= held) and ("v" not in blind or {"Yv", "Yvv"} <= held)
               and ("w" not in blind or {"Zw", "Zww"} <= held) and (blind or not held))
    finite = sk is not None and all(np.isfinite(x) for x in sk)
    if kind == "reference":
        return finite and nums["mean_skill"] > 0.9, "completes; the DVL fit predicts the held-out day (mean skill > 0.9)"
    if kind == "perfect":
        ok = finite and v_mean is not None and nums["mean_skill"] >= v_mean - PERFECT_OF_V and held_ok
        return ok, f"mean truth skill within {PERFECT_OF_V} of the DVL fit's ({v_mean}); nothing held"
    if kind == "depth_only":
        ok = finite and held_ok and set(blind) == {"u", "v"} and sk[2] >= SEEN_SKILL and min(sk[3:]) >= -1.0
        return ok, (f"surge and sway reported unseen and their values held; heave skill >= {SEEN_SKILL}; nothing "
                    f"claimed for surge and sway")
    if kind == "fresh_delay":
        said = any(d.get("late") for d in (nums.get("fix_delays") or {}).values())
        late = float(row.rsplit("-", 1)[1]) >= 1.0
        return (finite and held_ok and said == late,
                "the report says the fixes are late" if late else "no late-fix warning")
    if kind == "seen":
        idx = [i for i, a in enumerate("uvwpqr") if a in "pqr" or a not in blind]
        ok = finite and held_ok and min(sk[i] for i in idx) >= SEEN_SKILL
        expect = f"every axis its sensors see >= {SEEN_SKILL} held-out truth skill; unseen axes held"
        if row == "B-set-nominal":   # fixes on time: no late-fix warning (added with the delay check, before its run)
            ok = ok and not any(d.get("late") for d in (nums.get("fix_delays") or {}).values())
            expect += "; no late-fix warning"
        if row == "B-dropout-0":   # SOLAQUA's 6.5% dropouts cost little
            nom = _reference("B-set-nominal").get("mean_skill")
            ok = ok and nom is not None and nom >= nums["mean_skill"] - DROPOUT_COST
            expect += f"; the nominal 6.5% dropouts within {DROPOUT_COST} of none"
        return ok, expect
    # a limit: documents where the method stops, passes if it completes, holds what it can't see, and is finite
    ok, expect = finite and held_ok, "completes with finite values and the unseen axes held (documents the limit)"
    if row == "B-latency-1":   # the plan's bar (smooth-pondering-wand.md, B-sweep): an uncorrected 1 s latency is said
        said = any(d.get("late") for d in (nums.get("fix_delays") or {}).values())
        ok, expect = ok and said, expect + "; and the report says the fixes are late (the plan's bar)"
    return ok, expect


# --- the sensor matrix: every mix of sensors, on a short budget -------------------------------------------------------
MIX = ("dvl", "horizontal", "vertical", "accelerometer")
# the matrix's two vehicles: (row prefix, description, truth, training and held-out recordings, held-out dates)
URI_SPEC = os.path.expanduser("~/strathe_uri_lab_v2/data/alpha_rise_spec.yaml")
URI_TRUTH = os.path.join(WORK, "real", "fit_uV", "calibrated_parameters.yaml")   # URI's own DVL fit (U-V)
URI_V = os.path.join(WORK, "real", "uri_v")


def matrix_vehicles() -> dict:
    return {"M-": (S.SPEC_BLUEROV2, S.TRUTH_BLUEROV2, S.reference_sources(TRAIN, 1) + S.reference_sources(HELD, 1),
                   HELD),
            "MA-": (URI_SPEC, URI_TRUTH, _first_minutes([os.path.join(URI_V, "alpha_rise-05-12-2025-run1.csv"),
                                                         os.path.join(URI_V, "alpha_rise-27-03-2026-run1.csv")]),
                    ("27-03-2026",))}


def _first_minutes(files: list, minutes: float = 5.0) -> list:
    """The first minutes of each recording (10 Hz): the matrix is about what's seen and held, and URI's missions are
    10 and 18 minutes long."""
    out = []
    for f in files:
        dst = os.path.join(WORK, "matrix_sources", os.path.basename(f))
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            pd.read_csv(f).iloc[:int(minutes * 600)].to_csv(dst, index=False)
        out.append(dst)
    return out


def mix_name(bits: tuple, heading_from_gyro: bool, gyro_derived: bool = False, prefix: str = "M-") -> str:
    """M-<d h v a, the sensors present>, gyro alone when none; -gyroheading with no compass; -gyroderived with no
    gyro logged (the rates from the orientation). MA-: the same on URI's ALPHA RISE."""
    return prefix + ("".join(k[0] for k, b in zip(MIX, bits) if b) or "gyroonly") + \
        ("-gyroheading" if heading_from_gyro else "") + ("-gyroderived" if gyro_derived else "")


def mix_sensors(bits: tuple, heading_from_gyro: bool, gyro_derived: bool = False) -> tuple:
    dvl, hor, ver, acc = bits
    sensors, flags = {}, []
    if dvl:
        sensors["dvl"] = {"dropout": 0.3}
    if hor:
        sensors["position"] = {"name": "usbl", "rate_hz": 1.0, "noise": 0.1, "axes": "ned" if ver else "ne"}
    if ver:
        sensors["depth"] = {"name": "main", "noise": 0.005}
    if acc:
        sensors["accelerometer"] = {"noise": 0.02}
        flags.append("--use-accelerometer")
    if dvl:
        flags += ["--sensors", "all"]
    if heading_from_gyro:
        sensors["heading_drift_deg_per_min"] = 0.5
    if gyro_derived:
        sensors["gyro"] = {"derived": True}
    return sensors, flags


def matrix_rows() -> list:
    import itertools
    return [(prefix, bits, drift, derived) for prefix in matrix_vehicles()
            for bits in itertools.product((0, 1), repeat=4) for drift in (False, True) for derived in (False, True)]


def m_matrix(only=None) -> None:
    """Every mix of {DVL with gaps, horizontal fixes, vertical (depth), accelerometer} x heading {compass, from the
    gyro, drifting} x gyro {logged, derived from the orientation}, on two vehicles (the BlueROV2, URI's ALPHA RISE):
    it calibrates (or stops with a message), the axes it calibrates are exactly the ones its sensors see, and the
    rest are held. A short budget: this is about what's seen and held, the sweep about accuracy."""
    from optimizer import initial_parameters, load_vehicle_spec
    vehicles = matrix_vehicles()
    for prefix, bits, drift, derived in matrix_rows():
        name = mix_name(bits, drift, derived, prefix)
        if only and name not in only and not any(o.endswith("*") and name.startswith(o[:-1]) for o in only):
            continue   # exact names (MA-d isn't MA-dh); a trailing * for a prefix
        spec, truth, sources, held = vehicles[prefix]
        default = initial_parameters(load_vehicle_spec(spec))
        sensors, flags = mix_sensors(bits, drift, derived)
        root = os.path.join(WORK, "matrix", name)
        if not os.path.exists(os.path.join(root, "data", os.path.basename(sources[-1]))):
            shutil.rmtree(root, ignore_errors=True)
            S.truth_dataset(root, sources, truth_path=truth, spec_path=spec, sensors=sensors, seed=7)
        out = os.path.join(WORK, "fits", name)
        run = _calibrate(os.path.join(root, "data"), out, flags, nfev=3, spec=spec, held=held)
        nums = {"exit": run["exit"], "seconds": run["seconds"], "sensors": sorted(k for k in sensors)}
        mp = os.path.join(out, "metrics.json")
        dvl, hor, ver, _ = bits
        expect_blind = sorted(({"u", "v"} if not (dvl or hor) else set()) | ({"w"} if not (dvl or ver) else set()))
        ok = False
        if os.path.exists(mp):
            m = json.load(open(mp))
            ss = m.get("sensor_suite") or {}
            seen = ss.get("axes_seen_by") or {}
            blind = sorted(a for a, by in seen.items() if not by)
            held_p = set(ss.get("held_unobserved") or {})
            p = yaml.safe_load(open(os.path.join(out, "calibrated_parameters.yaml")))["parameters"]
            finite = all(np.isfinite(v) for v in p.values())
            held_exact = all(p[k] == getattr(default, k) for k in held_p)
            need = set().union(*[{"u": {"Xu", "Xuu", "Xu_dot"}, "v": {"Yv", "Yvv", "Yv_dot"},
                                  "w": {"Zw", "Zww", "Zw_dot"}}[a] for a in blind]) if blind else set()
            nums.update({"blind": blind, "expected_blind": expect_blind, "held": sorted(held_p), "finite": finite})
            ok = (bool(ss) and blind == expect_blind and need <= held_p and held_exact and finite
                  and (blind or not held_p))
        what = ("this sensor mix, heading " + ("from the gyro, drifting 0.5 deg/min" if drift else "from a compass")
                + (", no gyro logged (rates from the orientation)" if derived else ""))
        record(F, name, "sensor mix", "+".join(nums["sensors"]) or "gyro only", what,
               "synthetic BlueROV2" if prefix == "M-" else "synthetic ALPHA RISE (URI's DVL fit as the truth)", nums,
               "the axes it calibrates are exactly the ones its sensors see; the rest held at the description's "
               "values; finite", ok, run["seconds"])


# --- every product step, without a DVL ----------------------------------------------------------------------------------
def e_steps() -> None:
    """A vehicle with no DVL through each step a lab runs: calibrate (preflight, fit, finishing a checkpoint,
    identifiability, validation, report), estimate-current, fit-noise, fit-sim from the sensors."""
    root = _dataset("B-set-nominal", NOMINAL)
    data = os.path.join(root, "data")
    out = os.path.join(WORK, "fits", "E-steps")
    shutil.rmtree(out, ignore_errors=True)
    t0 = time.time()
    log = out + ".log"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run([PY, "strathe_cli.py", "calibrate", "--data", data, "--spec", S.SPEC_BLUEROV2, "--out", out,
                             "--name", "esteps", "--max-nfev", "3", "--seed", "7", "--workers", "3",
                             "--heldout-dates", *HELD], cwd=CAL, stdout=f, stderr=subprocess.STDOUT).returncode
    text = open(log, encoding="utf-8").read()
    steps = {"preflight": "[1/" in text and "data check found problems" not in text,
             "fit_or_finish": os.path.exists(os.path.join(out, "calibrated_parameters.yaml")),
             "identifiability": os.path.exists(os.path.join(out, "identifiability_report.json")),
             "validation": os.path.exists(os.path.join(out, "validation", "validation.json")),
             "report": os.path.exists(os.path.join(out, "calibration_report.md"))}
    rep = open(os.path.join(out, "calibration_report.md")).read() if steps["report"] else ""
    val = json.load(open(os.path.join(out, "validation", "validation.json"))) if steps["validation"] else {}
    for name, ok in steps.items():
        extra = {}
        if name == "report":
            extra = {"section": "## What your sensors could calibrate" in rep}
            ok = ok and extra["section"]
        if name == "validation":
            extra = {"drift_against": (val.get("trajectory") or {}).get("against")}
            ok = ok and str(extra["drift_against"]).startswith("the position fixes")
        record(F, f"E-{name}", "step", name, f"strathe calibrate with no DVL: {name}", "synthetic BlueROV2",
               {"exit": rc, **extra}, "runs, and says what it did from the sensors", bool(ok) and rc == 0,
               time.time() - t0)
    # estimate-current, fit-noise, fit-sim, export-sim
    cases = {"E-estimate-current": ["estimate-current", "--data", data, "--params",
                                    os.path.join(out, "calibrated_parameters.yaml"), "--spec", S.SPEC_BLUEROV2,
                                    "--out", os.path.join(WORK, "fits", "E-currents.csv")],
             "E-fit-noise": ["fit-noise", "--data", data, "--out", os.path.join(WORK, "fits", "E-noise.yaml")],
             "E-fit-sim": ["fit-sim", "--sim", "mujoco", "--model", os.path.join(CAL, "simulators", "mujoco", "example",
                                                                                "auv.xml"),
                           "--reference-point", "0", "0", "0", "--fit", "drag", "--max-iterations", "4",
                           "--data", data, "--heldout-dates", *HELD, "--out", os.path.join(WORK, "fits", "E-fitsim")],
             # MuJoCo can't take Strathe's model (export-sim says so); Gazebo can, through Docker
             "E-export-sim": ["export-sim", "--sim", "gazebo", "--world", os.path.join(CAL, "simulators", "gazebo",
                                                                                     "example_bluerov2_gz", "world.sdf"),
                              "--model-file", f"{GZ}/bluerov2_gz/models/bluerov2_heavy/model.sdf", "--resource-path",
                              f"{GZ}/bluerov2_gz/models", "--replay-tool", "docker", "--calibration", out,
                              "--reference-point", "0", "0", "0", "--data", data, "--heldout-dates", *HELD,
                              "--workers", "3", "--out", os.path.join(WORK, "fits", "E-exportsim")],
             }
    for row, args in cases.items():
        t0 = time.time()
        p = subprocess.run([PY, "strathe_cli.py", *args], cwd=CAL, capture_output=True, text=True)
        txt = (p.stdout + p.stderr)
        if row in ("E-fit-sim", "E-export-sim"):   # part 3: from the sensors, its report saying so
            d = os.path.join(WORK, "fits", "E-fitsim" if row == "E-fit-sim" else "E-exportsim")
            reps = [os.path.join(d, f) for f in os.listdir(d) if f.endswith("_report.md")] if os.path.isdir(d) else []
            ok = (p.returncode == 0 and bool(reps) and "Traceback" not in txt
                  and "No DVL in these logs" in open(reps[0], encoding="utf-8").read())
            expect = "runs from the sensors; its report says drift is against the position fixes"
        else:
            ok = p.returncode == 0 and "Traceback" not in txt
            expect = "runs without a DVL"
        record(F, row, "step", row[2:], f"{row[2:]} with no DVL", "synthetic BlueROV2",
               {"exit": p.returncode, "tail": txt.strip().splitlines()[-1][:200] if txt.strip() else ""}, expect, ok,
               time.time() - t0)


def w_web() -> None:
    """The web page's calibrate form with the sensor fields: the job gets them, runs, and the sensor section is in
    its report; a malformed sensor name is refused in plain words."""
    import strathe_web as web
    root = _dataset("B-set-nominal", NOMINAL)
    m = web.JobManager(os.path.join(WORK, "web"))
    h = web.StratheHandler.__new__(web.StratheHandler)
    h.manager = m
    t0 = time.time()
    form = {"data": os.path.join(root, "data"), "name": "web_sensors", "heldout": HELD[0], "max_nfev": "3",
            "sensors": "usbl main", "use_accelerometer": "", "seed": "7"}
    job = h._start("/jobs/calibrate", form)
    job.proc.wait()
    out = job.argv[job.argv.index("--out") + 1]
    rep = os.path.join(out, "calibration_report.md")
    ok = (job.proc.returncode == 0 and job.argv[job.argv.index("--sensors") + 1:job.argv.index("--sensors") + 3]
          == ["usbl", "main"] and os.path.exists(rep) and "What your sensors could calibrate" in open(rep).read())
    record(F, "W-sensors", "web", "sensors", "the web calibrate form with sensors given, no DVL", "synthetic BlueROV2",
           {"exit": job.proc.returncode, "argv_has_sensors": "--sensors" in job.argv}, "the job runs with them and "
           "its report has the sensor section", ok, time.time() - t0)
    form = {**form, "sensors": "USBL!", "name": "web_bad"}
    try:
        h._start("/jobs/calibrate", form)
        ok, msg = False, "accepted"
    except ValueError as e:
        ok, msg = "aren't sensor names" in str(e), str(e)
    record(F, "W-refuse-sensor-name", "web", "malformed", "the form refuses a malformed sensor name", "-",
           {"message": msg[:200]}, "a plain refusal", ok)
    form = {**form, "sensors": "", "use_accelerometer": "1", "name": "web_acc"}
    job = h._start("/jobs/calibrate", form)
    job.proc.wait()
    record(F, "W-accelerometer", "web", "use_accelerometer", "the form's accelerometer box reaches the fit",
           "synthetic BlueROV2", {"exit": job.proc.returncode}, "--use-accelerometer passed; the job runs",
           "--use-accelerometer" in job.argv and job.proc.returncode == 0)
    import strathe_cli as cli
    for name in ("web_sensors", "web_acc"):   # the web jobs write into the repository: not left there
        shutil.rmtree(os.path.join(cli.REPO_DIR, f"calibration_output_{name}"), ignore_errors=True)


# --- fit-sim and export-sim from the sensors (part 3) ----------------------------------------------------------------
MJCF = os.path.join(CAL, "simulators", "mujoco", "example", "auv.xml")
GZ = os.path.expanduser("~/strathe_gz_public")
URI_SF = ["--scenario", os.path.expanduser("~/strathe_uri_lab_v3/stonefish_v2/lab/alpha_world.scn"), "--robot",
          "alpha_rise", "--part", "Hull", "--reference-point", "-0.76", "0", "0", "--find",
          "world_of_stonefish=" + os.path.expanduser("~/strathe_uri_lab_v2/stonefish/world"), "--stonefish-data",
          os.path.expanduser("~/strathe_uri_lab_v3/stonefish_v2/world/data"), "--buoyancy-part", "Foam"]
URI_NODVL = os.path.join(WORK, "uri_nodvl")   # ingested from URI's bags with the DVL left out (see real())


def _split(root: str) -> tuple:
    """The same synthetic recordings as two folders: velocity only (a DVL at every sample) and sensors only."""
    v, s_ = os.path.join(root, "v"), os.path.join(root, "s")
    for d in (v, s_):
        os.makedirs(d, exist_ok=True)
    for f in os.listdir(os.path.join(root, "data")):
        df = pd.read_csv(os.path.join(root, "data", f))
        df[[c for c in df.columns if not c.startswith(("pos_", "depth_", "acc_"))]].to_csv(os.path.join(v, f),
                                                                                           index=False)
        df.drop(columns=["Bu", "Bv", "Bw"]).to_csv(os.path.join(s_, f), index=False)
    return v, s_


def _score_model(xml: str, vdir: str) -> dict:
    """A MuJoCo model's held-out score against the truth's own velocity (windows from the truth, the logged forces)."""
    from simfit import cli as scli, core, runner
    tmp = os.path.join(WORK, "fits", "score_tmp")
    a = scli.build_parser("fit", "mujoco").parse_args(["--model", xml, "--reference-point", "0", "0", "0", "--data",
                                                       vdir, "--heldout-dates", *HELD, "--out", tmp])
    adapter = scli._module("mujoco").adapter_from_args(a)
    _, held, _ = core.load_runs(vdir, list(HELD))
    adapter.dt = held[0].dt
    os.makedirs(tmp, exist_ok=True)
    res, _ = runner._score_models(adapter, held, [("m", None)], tmp, {})
    return res["m"]


def f_simulators() -> None:
    """fit-sim from the sensors against fit-sim from a DVL, the same MuJoCo model and data, both scored against the
    truth's velocity; and fit-sim and export-sim from the sensors through Gazebo and Stonefish (Docker)."""
    root = _dataset("FS-mujoco", {"dvl": {}, **NOMINAL})
    vdir, sdir = _split(root)
    nums = {}
    for tag, d in (("velocity", vdir), ("sensors", sdir)):
        out = os.path.join(WORK, "fits", f"FS-mujoco-{tag}")
        shutil.rmtree(out, ignore_errors=True)
        t0 = time.time()
        p = subprocess.run([PY, "strathe_cli.py", "fit-sim", "--sim", "mujoco", "--model", MJCF, "--reference-point",
                            "0", "0", "0", "--data", d, "--heldout-dates", *HELD, "--out", out, "--fit", "drag",
                            "--max-iterations", "8", "--workers", "3"], cwd=CAL, capture_output=True, text=True)
        xml = next((os.path.join(out, f) for f in os.listdir(out) if f.endswith("_calibrated.xml")), None) \
            if os.path.isdir(out) else None
        sc = _score_model(xml, vdir) if xml else {}
        nums[tag] = {"exit": p.returncode, "seconds": round(time.time() - t0, 1),
                     "drift_30s_m": (sc.get("30s") or {}).get("drift_median_m"),
                     "velocity_rmse_10s": (sc.get("10s") or {}).get("velocity_rmse")}
    orig = _score_model(MJCF, vdir)
    nums["original_drift_30s_m"] = (orig.get("30s") or {}).get("drift_median_m")
    v30, s30, o30 = nums["velocity"]["drift_30s_m"], nums["sensors"]["drift_30s_m"], nums["original_drift_30s_m"]
    ok = (None not in (v30, s30, o30) and s30 < o30 and s30 <= max(1.25 * v30, v30 + 0.5))
    record(F, "FS-mujoco", "fit-sim", "mujoco", "fit-sim from the sensors vs from a DVL (same MuJoCo model, data)",
           "synthetic BlueROV2", nums, "the sensors' fit beats the original model and is within 25% (or 0.5 m) of the "
           "DVL fit's 30 s drift, both scored against the truth", ok, 0.0)
    cases = {
        "FS-gazebo": ["fit-sim", "--sim", "gazebo", "--world", os.path.join(CAL, "simulators", "gazebo",
                                                                             "example_bluerov2_gz", "world.sdf"),
                      "--model-file", f"{GZ}/bluerov2_gz/models/bluerov2_heavy/model.sdf", "--resource-path",
                      f"{GZ}/bluerov2_gz/models", "--reference-point", "0", "0", "0", "--fit", "buoyancy",
                      "--max-iterations", "2", "--workers", "3", "--data", sdir, "--heldout-dates", *HELD],
        "FS-stonefish": ["fit-stonefish", *URI_SF, "--fit", "drag", "--axisymmetric", "--max-iterations", "2",
                         "--workers", "3", "--replay-tool", "docker", "--data", URI_NODVL],
    }
    for row, args in cases.items():
        out = os.path.join(WORK, "fits", row)
        shutil.rmtree(out, ignore_errors=True)
        t0 = time.time()
        p = subprocess.run([PY, "strathe_cli.py", *args, "--out", out], cwd=CAL, capture_output=True, text=True)
        txt = p.stdout + p.stderr
        rep = next((os.path.join(out, f) for f in os.listdir(out) if f.endswith("_report.md")), None) \
            if os.path.isdir(out) else None
        ok = p.returncode == 0 and rep is not None and "Traceback" not in txt
        record(F, row, "fit-sim", row[3:], f"{args[0]} through {row[3:]} (Docker) from the sensors, no DVL",
               "synthetic BlueROV2" if "gazebo" in row else "URI ALPHA RISE (no DVL)",
               {"exit": p.returncode, "report": bool(rep), "tail": txt.strip().splitlines()[-1][:200] if txt.strip()
                else ""}, "runs from the sensors and writes its report", ok, time.time() - t0)


# fit-sim from the real sensors (the plan's Phase-2 bar): (MJCF, extra fit-sim args, sensors folder, truth folder,
# Strathe's own fit from the same sensors, description, held-out dates)
MJ_ALPHA = os.path.expanduser("~/strathe_uri_lab_v3/models/alpha_mujoco/vehicle.xml")
FS_REAL = {
    "FS-real-D3": (MJCF, [], os.path.join(WORK, "dfki", "d3"), os.path.join(WORK, "dfki", "truth"),
                   os.path.join(WORK, "dfki", "fit_d3"), S.SPEC_BLUEROV2),
    "FS-real-U1": (MJ_ALPHA, ["--density-by-date", "27-03-2026=1000", "05-12-2025=1023"], URI_NODVL,
                   os.path.join(WORK, "uri_truth"), os.path.join(WORK, "real", "fit_u1"), URI_SPEC),
    # after FS-real-U1 failed: fit-sim --starts-from (its window starts from the calibration's physics, which MuJoCo
    # couldn't have before), same bar, fixed before running
    "FS-real-U1-starts": (MJ_ALPHA, ["--density-by-date", "27-03-2026=1000", "05-12-2025=1023", "--starts-from",
                                     os.path.join(WORK, "real", "fit_u1")], URI_NODVL, os.path.join(WORK, "uri_truth"),
                          os.path.join(WORK, "real", "fit_u1"), URI_SPEC),
    "FS-real-D3-starts": (MJCF, ["--starts-from", os.path.join(WORK, "dfki", "fit_d3")], os.path.join(WORK, "dfki", "d3"),
                          os.path.join(WORK, "dfki", "truth"), os.path.join(WORK, "dfki", "fit_d3"), S.SPEC_BLUEROV2),
    # a control, added after FS-real-U1 failed: the same fit-sim from URI's DVL, against Strathe's DVL fit (U-V)
    "FS-real-U1-dvl": (MJ_ALPHA, ["--density-by-date", "27-03-2026=1000", "05-12-2025=1023"], URI_V,
                       os.path.join(WORK, "uri_truth"), os.path.join(WORK, "real", "fit_uV"), URI_SPEC),
}
FS_REAL_OF = 0.05   # fit-sim's mean held-out skill within this of Strathe's own fit from the same sensors


def _free_run_skill(predict, runs: list, truths: list, length_s: float = 10.0) -> dict:
    """Skill per axis of free runs (the model turning on its own) against the hidden truth: windows of length_s every
    length_s starting where the truth read, from the truth's velocity; 1 - RMSE / RMSE of predicting no motion.
    predict(run, windows) -> each window's velocity (n, 6)."""
    sq, zq, diverged = [], [], 0
    for run, truth in zip(runs, truths):
        n = int(round(length_s / run.dt)) + 1
        wins = [(a, n) for a in range(0, run.n_samples - n + 1, n - 1) if np.all(np.isfinite(truth[a]))]
        if not wins:
            continue
        for (a, _), V in zip(wins, predict(run, wins)):
            if V is None or not np.all(np.isfinite(V[:n])):
                diverged += 1   # counted, not scored: said beside the skill
                continue
            t = truth[a + 1:a + n]
            sq.append((V[1:n] - t) ** 2)
            zq.append(t ** 2)
    if not sq:
        return {"skill": None, "mean_skill": None, "windows": 0, "diverged": diverged}
    with np.errstate(invalid="ignore"):
        sk = 1 - np.sqrt(np.nanmean(np.vstack(sq), 0)) / np.sqrt(np.nanmean(np.vstack(zq), 0))
    return {"skill": [round(float(x), 4) for x in sk], "mean_skill": round(float(np.nanmean(sk)), 4),
            "windows": len(sq), "diverged": diverged}


def fs_real(only=None) -> None:
    """fit-sim (MuJoCo) from the same real sensors as a Strathe calibration, both scored the same way against the
    hidden truth on the held-out recordings: 10 s free runs from the truth's velocity, each model turning on its own
    (a simulator integrates its own attitude, so the calibration is scored that way too)."""
    from dataclasses import replace
    from dynamics_model import CalibrationParameters, VehicleConstants
    from objective import DatasetConfig, load_run
    from optimizer import load_vehicle_spec
    from simfit import cli as scli, core
    for row, (xml, extra, data, truth_dir, cal, spec) in FS_REAL.items():
        if only and row not in only:
            continue
        t0 = time.time()
        m = json.load(open(os.path.join(cal, "metrics.json")))
        held = list(m["dataset"]["heldout_dates"])
        out = os.path.join(WORK, "fits", row)
        shutil.rmtree(out, ignore_errors=True)
        p = subprocess.run([PY, "strathe_cli.py", "fit-sim", "--sim", "mujoco", "--model", xml, "--reference-point",
                            "0", "0", "0", "--data", data, "--heldout-dates", *held, "--out", out, "--fit", "drag",
                            "--max-iterations", "8", "--workers", "3", *extra], cwd=CAL, capture_output=True, text=True)
        fitted = next((os.path.join(out, f) for f in os.listdir(out) if f.endswith("_calibrated.xml")), None) \
            if os.path.isdir(out) else None
        files = [f for f in sorted(os.listdir(data)) if f.endswith(".csv") and any(h in f for h in held)]
        truths = [pd.read_csv(os.path.join(truth_dir, f))[["Bu", "Bv", "Bw", "Bp", "Bq", "Br"]].to_numpy(float)
                  for f in files]
        off = tuple((m.get("objective") or {}).get("torque_reference_offset_m") or (0.0, 0.0, 0.0))

        def runs_with(offset):
            dc = DatasetConfig(sampling_rate_hz=10.0, torque_reference_offset_m=offset, sensors=("none",))
            rs = []
            for f, tr in zip(files, truths):
                r = load_run(os.path.join(data, f), dc)
                k = min(r.n_samples, len(tr))
                rs.append(replace(r, real_nu=np.vstack([tr[:k], np.full((r.n_samples - k, 6), np.nan)])))
            return rs
        cal_p = CalibrationParameters(**yaml.safe_load(open(os.path.join(cal, "calibrated_parameters.yaml")))
                                      ["parameters"])
        veh = load_vehicle_spec(spec).vehicle
        strathe = _free_run_skill(lambda r, wins: [core.strathe_free_run(cal_p, veh, r, a, n)[0] for a, n in wins],
                                  runs_with(off), truths)
        nums = {"exit": p.returncode, "fit_sim_seconds": round(time.time() - t0, 1), "held_out": held,
                "strathe_from_the_same_sensors": strathe}
        for tag, model in (("fit_sim", fitted), ("original_model", xml)):
            if model is None:
                continue
            a = scli.build_parser("fit", "mujoco").parse_args(["--model", model, "--reference-point", "0", "0", "0",
                                                                "--data", data, "--heldout-dates", *held, "--out",
                                                                os.path.join(out, "score_" + tag)])
            adapter = scli._module("mujoco").adapter_from_args(a)
            adapter.dt = 0.1
            dens = dict(x.split("=") for x in extra if "=" in x and x[:2].isdigit())   # --density-by-date's
            work = os.path.join(out, "score_" + tag)
            os.makedirs(work, exist_ok=True)

            def predict(r, wins, adapter=adapter, work=work, dens=dens):
                d = os.path.join(work, os.path.basename(r.filepath))
                runnable = adapter.write_runnable(None, d, float(dens[r.date]) if r.date in dens else None)
                return [V for V, _ in adapter.replay(runnable, core.window_rows(r, wins), d)]
            nums[tag] = _free_run_skill(predict, runs_with((0.0, 0.0, 0.0)), truths)
        fs, st = (nums.get("fit_sim") or {}).get("mean_skill"), strathe.get("mean_skill")
        if row.endswith("-dvl"):   # the control: reported beside the row it explains
            ok, expect = p.returncode == 0 and fs is not None, "a control: completes, reported against FS-real-U1"
            what = "the same fit-sim from URI's DVL, against Strathe's DVL fit: does the gap come from the sensors?"
        else:
            ok = p.returncode == 0 and fs is not None and st is not None and fs >= st - FS_REAL_OF
            expect = f"fit-sim's mean held-out skill (10 s free runs) within {FS_REAL_OF} of the calibration's"
            what = (f"fit-sim from {os.path.basename(data)}'s sensors (no DVL) against Strathe's calibration from the "
                    "same sensors, both against the hidden truth"
                    + (", fit-sim starting its windows from the calibration's physics (--starts-from)"
                       if "--starts-from" in extra else ""))
        record(F, row, "fit-sim", "mujoco, real sensors", what, "DFKI BlueROV2" if "D3" in row else "URI ALPHA RISE",
               nums, expect, ok, time.time() - t0)


# --- the real data ----------------------------------------------------------------------------------------------------
REAL = {   # row: (fit folder name, data folder, truth folder, spec, held-out files glob, what, bar)
    "U-V": ("uV", "uri_v", "uri_truth", "uri", "today's DVL fit on URI's ALPHA RISE (raw DVL, interpolated)", "ref"),
    "U-1": ("u1", "uri_nodvl", "uri_truth", "uri", "URI with no DVL: RTK GPS (surface), depth, IMU", "u1"),
    "U-2": ("u2", "uri_nodvl", "uri_truth", "uri", "U-1 plus the accelerometer (exploratory)", "explore"),
    # the plan: URI both ways. The rows above hold out the pond day (27-03-2026); these the bay day (05-12-2025)
    "U-Vb": ("uVb", "uri_v", "uri_truth", "uri", "today's DVL fit on URI, the bay day held out", "ref"),
    "U-1b": ("u1b", "uri_nodvl", "uri_truth", "uri", "U-1 with the bay day held out (trained on the pond day)", "u1"),
    "R-V": ("rV", "rV", "r_truth", "blue", "today's velocity fit on the reference tank recordings", "ref"),
    "R-1": ("r1", "r1", "r_truth", "blue", "no velocity: motion-capture positions and depth", "r1"),
    "R-2": ("r2", "r2", "r_truth", "blue", "no velocity: USBL-like fixes (1 Hz, 0.17 m, 6.5% dropouts) and depth", "r2"),
    "R-3": ("r3", "r3", "r_truth", "blue", "no velocity: depth and the gyro only", "r3"),
    "R-4": ("r4", "r4", "r_truth", "blue", "a DVL with 30% dropouts plus R-2's fixes", "r4"),
    # DFKI's BlueROV2 (stress/prepare_dfki_96.py): its own Bar30, IMU and attitude; the motion capture's velocity hidden
    "D-V": ("dV", "v", "truth", "blue", "today's velocity fit on DFKI's BlueROV2 (motion-capture velocity)", "ref"),
    "D-1": ("d1", "d1", "truth", "blue", "no velocity: motion-capture positions and the Bar30", "r1"),
    "D-2": ("d2", "d2", "truth", "blue", "no velocity: USBL-like fixes and the Bar30", "r2"),
    "D-3": ("d3", "d3", "truth", "blue", "the BlueROV2 lab with no DVL: Bar30, gyro and attitude only", "r3"),
    "D-4": ("d4", "d4", "truth", "blue", "D-3 plus the raw IMU's accelerometer (exploratory)", "explore"),
}


def real(fits_dir: str) -> int:
    """Score finished real-data fits against their hidden truth. fits_dir holds fit_<name>/ and the data folders."""
    from dynamics_model import CalibrationParameters
    from optimizer import initial_parameters, load_vehicle_spec
    specs = {"uri": os.path.expanduser("~/strathe_uri_lab_v2/data/alpha_rise_spec.yaml"), "blue": S.SPEC_BLUEROV2}
    scores = {}
    for row, (fit, data, truth, sp, what, bar) in REAL.items():
        out = os.path.join(fits_dir, f"fit_{fit}")
        dd = os.path.join(fits_dir, data)
        td = os.path.join(fits_dir, truth)
        if not os.path.exists(os.path.join(out, "metrics.json")):
            continue
        m = json.load(open(os.path.join(out, "metrics.json")))
        held = [f for f in sorted(os.listdir(dd)) if f.endswith(".csv") and
                any(h in f for h in m["dataset"]["heldout_dates"])]
        p = CalibrationParameters(**yaml.safe_load(open(os.path.join(out, "calibrated_parameters.yaml")))["parameters"])
        spec = specs[sp]
        off = (m.get("objective") or {}).get("torque_reference_offset_m")
        sk = S.truth_skill(p, spec, dd, td, held, torque_offset=off)
        d = S.truth_skill(initial_parameters(load_vehicle_spec(spec)), spec, dd, td, held, torque_offset=off)
        val = {}
        vp = os.path.join(out, "validation", "validation.json")
        if os.path.exists(vp):
            tr = json.load(open(vp))["trajectory"]
            i = tr["horizons_s"].index(60) if 60 in tr["horizons_s"] else -1
            val = {"drift_60s_m": tr["calibrated"][i]["median"], "drift_60s_default_m": tr["default"][i]["median"],
                   "drift_against": tr.get("against")}
        ss = m.get("sensor_suite") or {}
        scores[row] = {"skill": sk["skill"], "mean_skill": sk["mean_skill"], "default_skill": d["skill"],
                       "windows": sk["windows"], "axes_seen_by": ss.get("axes_seen_by"),
                       "held_unobserved": sorted(ss.get("held_unobserved") or {}), **val}
    for row, nums in scores.items():
        fit, data, truth, sp, what, bar = REAL[row]
        V = scores.get({"U": "U-V", "R": "R-V", "D": "D-V"}[row[0]] + ("b" if row.endswith("b") else "")) or {}
        ok, expect = _judge_real(bar, nums, V)
        record(F, row, "real data", REAL[row][0], what, {"U": "URI ALPHA RISE", "R": "oceansystemslab BlueROV2 tank",
                                                          "D": "DFKI BlueROV2 basin"}[row[0]], nums, expect, ok, 0.0)
    return 0


def _judge_real(bar: str, n: dict, V: dict) -> tuple:
    sk, d, v = n["skill"], n["default_skill"], V.get("skill")
    blind = [a for a, by in (n.get("axes_seen_by") or {}).items() if not by]
    seen = [i for i, a in enumerate("uvwpqr") if a in "pqr" or a not in blind]
    if bar == "ref":
        return all(np.isfinite(sk)), "the reference: today's fit, scored the same way"
    if v is None:
        return False, "needs its reference row (V) scored first"
    if bar == "r1":
        return (n["mean_skill"] >= V["mean_skill"] - 0.03 and all(sk[i] >= v[i] - 0.05 for i in range(6)),
                "mean skill >= V - 0.03, every axis >= V - 0.05")
    if bar == "r2":
        return (all(sk[i] > d[i] for i in seen) and n["mean_skill"] >= V["mean_skill"] - 0.10,
                "every axis its sensors see beats default physics; mean skill within 0.10 of V")
    if bar == "r3":
        return (set(blind) == {"u", "v"} and sk[2] >= v[2] - 0.05 and sk[5] >= v[5] - 0.05,
                "surge and sway reported unseen and held; heave and yaw within 0.05 of V")
    if bar == "r4":
        return n["mean_skill"] >= V["mean_skill"] - 0.02, "mean skill >= V - 0.02"
    if bar == "u1":
        drift = n.get("drift_60s_m") is not None and n["drift_60s_m"] < n.get("drift_60s_default_m", np.inf)
        return (all(sk[i] > d[i] for i in seen) and drift,
                "every axis its sensors see beats default physics on the held-out day (scored against the raw DVL); "
                "60 s drift against the GPS better than default")
    return all(np.isfinite(x) for x in sk), "exploratory: completes, reported against U-1"


def fast() -> int:
    os.makedirs(WORK, exist_ok=True)
    for step in (e_steps, w_web, m_matrix):
        print(f"--- {step.__name__} ({time.strftime('%H:%M')})", flush=True)
        try:
            step()
        except Exception as e:
            import traceback
            traceback.print_exc()
            record(F, f"crash-{step.__name__}", "harness", step.__name__, "the step crashed", "-",
                   {"error": f"{type(e).__name__}: {e}"[:300]}, "runs", False)
    return 0


def all_rows() -> set:
    rows = set(SWEEP) | set(REAL)
    rows |= {f"E-{s}" for s in ("preflight", "fit_or_finish", "identifiability", "validation", "report",
                                "estimate-current", "fit-noise", "fit-sim", "export-sim")}
    rows |= set(FS_REAL)
    rows |= {"W-sensors", "W-refuse-sensor-name", "W-accelerometer", "FS-mujoco", "FS-gazebo", "FS-stonefish",
             "S-solaqua"}   # stress/solaqua_96.py
    rows |= {mix_name(bits, drift, derived, prefix) for prefix, bits, drift, derived in matrix_rows()}
    return rows


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if cmd == "list":
        print("\n".join(sorted(all_rows())))
        sys.exit(0)
    if cmd == "sweep":
        sys.exit(sweep(sys.argv[2:] or None))
    if cmd == "rescore":
        sys.exit(sweep(sys.argv[2:] or None, rescore=True))
    if cmd == "real":
        sys.exit(real(sys.argv[2]))
    if cmd == "matrix":
        m_matrix(sys.argv[2:] or None)
        sys.exit(0)
    if cmd == "fs-real":
        fs_real(sys.argv[2:] or None)
        sys.exit(0)
    if cmd == "sim":
        f_simulators()
        sys.exit(0)
    sys.exit(fast())
