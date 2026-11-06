"""
stress/run_97.py -- stress test of Fin Steer (PROJECT_STATUS.md #97): fin-steered vehicles, proved on vehicles Strathe wasn't written from.

    python stress/run_97.py fast          # faults, refusals, fuzz, mutants and the physics identities (minutes)
    python stress/run_97.py truth [ROW]   # independent truths: fit with the truth hidden, fly on unseen recordings (hours)
    python stress/run_97.py recover [ROW] # every layout and option through Strathe's own model (plumbing, labelled so)
    python stress/run_97.py real [ROW]    # OOI's REMUS logs (stress/prepare_ooi_97.py) and rudder checks
    python stress/run_97.py truth-finish ROW   # a truth row whose fit is done: finalize its checkpoint
    python stress/run_97.py sim [ROW]     # every simulator's own physics recovers its own fin lift (Docker, HOLOOCEAN_SRC)
    python stress/run_97.py more [ROW]    # mass entries, vectored, all pairs, layout x source x engine, export, cross engine, the lab's journey
    python stress/run_97.py list

The proof is a truth the fit never sees (stress/fin_truth.py): Prestero's REMUS, Fossen's remus100 and HoloOcean's Torpedo, flown closed
loop. A row calibrates from the vehicle's size and logs, as a lab would, then flies itself for 30 s on recordings it never saw.

Bars are fixed here, before any result: a row passes or fails on them, and a failure is a finding, never a moved bar. The numbers below
were proposed in the plan before the first truth ran, and the median/90th-percentile statistic was fixed after the go/no-go prototype
(Fossen, rich: heading 2.3 / 7.6, pitch 0.3 / 1.3, depth 0.18 / 0.66), so these bars had that much sight of the results.
"""
from __future__ import annotations

import hashlib
import json
import os
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

F = "97"
WORK = os.path.expanduser("~/strathe_fin_data/proof")

# --- the bars ---------------------------------------------------------------------------------------------------------
FLY = {"heading_deg": 5.0, "pitch_deg": 2.0, "depth_m": 0.5}   # median error after 30 s flying itself on unseen recordings
TURN_OF_HOLD = 0.5       # median turn-rate error at most this share of assuming the rate holds
RECOVER_CLEAN = 0.02     # same-model recovery of fin lift, relative, clean logs
RECOVER_NOISY = 0.10     # the same at grade noise
PATHS_AGREE = 1e-11      # fast, reference and oracle physics, relative
SPEED_NO_FINS = 1.02     # hover's step against before fins
SPEED_FOUR_FINS = 3.0    # four fins against none
FORMAT_DEG = 0.06        # a logged fin angle through any log format, degrees: half of one PWM step at 10 us per degree
FORMAT_REL = 1e-6        # the same for rpm, gimbal, moving mass and buoyancy, relative to the signal's size
SIM_RECOVER = 0.15       # fit-sim from twice the true fin lift lands within this of it, and drifts less on unseen recordings
EXPORT_EXACT = 1e-6      # the lift an export writes, against the lift the calibration found
FLY_UNSTABLE_S = 5.0     # Prestero's vehicle doubles an error every 0.7 s, so it can't be flown 30 s by any model: FLY's numbers at this horizon, and better than
                         # the default and than holding course. Set after T-prestero-rich failed and before any result at this horizon was seen.
BARS = {"fly": FLY, "turn_of_hold": TURN_OF_HOLD, "recover_clean": RECOVER_CLEAN, "recover_noisy": RECOVER_NOISY, "paths": PATHS_AGREE,
        "speed_no_fins": SPEED_NO_FINS, "speed_four_fins": SPEED_FOUR_FINS, "format_deg": FORMAT_DEG, "format_rel": FORMAT_REL, "sim_recover": SIM_RECOVER, "export_exact": EXPORT_EXACT, "fly_unstable_s": FLY_UNSTABLE_S}
BARS_HASH = hashlib.sha256(json.dumps(BARS, sort_keys=True).encode()).hexdigest()[:12]
NFEV = 40

GRADE = {"Bu": .01, "Bv": .01, "Bw": .01, "Bp": .002, "Bq": .002, "Br": .002}   # a survey DVL and a fibre gyro, per sample

# (row, engine, training mission, options, bar, what). `fly`: the flight bars. `warned`: fewer manoeuvres must be flagged, and the
# recordings like its training flown within the bars.
TRUTH = [
    ("T-prestero-rich", "prestero", "rich", {}, "fly", "Prestero's REMUS, manoeuvres"),
    ("T-fossen-rich", "fossen", "rich", {}, "fly", "Fossen's remus100, manoeuvres"),
    ("T-holoocean-rich", "holoocean", "rich", {}, "fly", "HoloOcean's Torpedo class, manoeuvres"),
    ("T-prestero-mixed", "prestero", "mixed", {}, "fly", "manoeuvres and survey lines"),
    ("T-fossen-mixed", "fossen", "mixed", {}, "fly", "manoeuvres and survey lines"),
    ("T-prestero-survey", "prestero", "survey", {}, "warned", "survey lines only: must warn"),
    ("T-fossen-survey", "fossen", "survey", {}, "warned", "survey lines only: must warn"),
    ("T-prestero-slow", "prestero", "rich", {"rpm": 1000.0}, "fly", "a slow vehicle"),
    ("T-prestero-fast", "prestero", "rich", {"rpm": 2000.0}, "fly", "a fast vehicle"),
    ("T-fossen-slow", "fossen", "rich", {"rpm": 800.0}, "fly", "a slow vehicle"),
    ("T-fossen-fast", "fossen", "rich", {"rpm": 1400.0}, "fly", "a fast vehicle"),
    ("T-prestero-rate-5", "prestero", "rich", {"rate": 5.0}, "fly", "logged at 5 Hz"),
    ("T-prestero-rate-2", "prestero", "rich", {"rate": 2.0}, "fly", "logged at 2 Hz"),
    ("T-prestero-rate-1", "prestero", "rich", {"rate": 1.0}, "fly", "logged at 1 Hz, as OOI's REMUS"),
    ("T-prestero-noise-grade", "prestero", "rich", {"noise": 1.0}, "fly", "survey-grade sensor noise"),
    ("T-prestero-noise-3x", "prestero", "rich", {"noise": 3.0}, "fly", "three times that noise"),
    ("T-fossen-noise-grade", "fossen", "rich", {"noise": 1.0}, "fly", "survey-grade sensor noise"),
    ("T-prestero-inertia-1.2", "prestero", "rich", {"inertia": 1.2}, "fly", "inertia 20% high in the description"),
    ("T-prestero-inertia-0.8", "prestero", "rich", {"inertia": .8}, "fly", "inertia 20% low in the description"),
    ("T-prestero-thrust-unknown", "prestero", "rich", {"kt": 1.5, "thrust_free": True}, "fly", "propeller curve 50% off, scale fitted"),
    ("T-prestero-thrust-wrong", "prestero", "rich", {"kt": 1.5}, "fly", "propeller curve 50% off, not fitted"),
    ("T-fossen-current", "fossen", "rich", {"current": (.2, 30.0)}, "fly", "a 0.2 m/s current, given to the fit"),
    ("T-prestero-long", "prestero", "rich", {"train_s": 2400.0}, "fly", "40 minutes of training"),
    ("T-prestero-short", "prestero", "rich", {"train_s": 300.0}, "fly", "5 minutes of training"),
]


def truth_rows() -> list:
    return [r[0] for r in TRUTH]


def _noisy(path: str, scale: float, seed: int) -> None:
    d = pd.read_csv(path)
    rng = np.random.default_rng(seed)
    for c, s in GRADE.items():
        d[c] = d[c] + rng.normal(0, s * scale, len(d))
    d.to_csv(path, index=False)


def _calibrate(data: str, spec: str, out: str, rate: float, flags=()) -> int:
    from fin_truth import HELD_DATE
    with open(out + ".log", "w") as f:
        return subprocess.run([PY, "strathe_cli.py", "calibrate", "--data", data, "--spec", spec, "--name", os.path.basename(out), "--out", out,
                               "--sampling-rate-hz", f"{rate:g}", "--max-nfev", str(NFEV), "--seed", "7", "--workers", "2",
                               "--heldout-dates", HELD_DATE, *flags], cwd=CAL, stdout=f, stderr=subprocess.STDOUT).returncode


def judge_flight(v: dict) -> tuple:
    """(passed, numbers) from validation.json's fin_free_run: the calibrated model against the flight bars."""
    f = v["fin_free_run"]
    c = f["calibrated"]
    n = {k: c[k]["median"] for k in FLY}
    n.update({f"{k}_p90": c[k]["p90"] for k in FLY})
    n["turn_rate"] = c["turn_rate_rad_s"]["median"]
    n["hold_turn_rate"] = f["straight_on"]["turn_rate_rad_s"]["median"]
    n["default"] = {k: f["default"][k]["median"] for k in FLY}
    n["straight_on"] = {k: f["straight_on"][k]["median"] for k in FLY}
    n["windows"], n["diverged"] = f["windows"], f["diverged"]["calibrated"]
    ok = all(n[k] is not None and n[k] <= FLY[k] for k in FLY) and not n["diverged"]
    ok = ok and n["turn_rate"] is not None and n["turn_rate"] <= TURN_OF_HOLD * max(n["hold_turn_rate"], 1e-9)
    return ok, n


def run_truth(row: tuple, finish: bool = False) -> bool:
    """finish: the fit is done and its checkpoint is there; only finalize it. A row's hours are in the fit, so a fault after it needn't cost them."""
    import fin_truth as T
    rid, engine, kind, o, bar, what = row
    t0 = time.time()
    if not T.available(engine):
        return record(F, rid, "truth", engine, what, "not available here", {"bars": BARS_HASH}, "NOT VERIFIED: the engine isn't here", False)
    d = os.path.join(WORK, rid)
    data, out = d + "/data", d + "/fit"
    fly = {}
    for k in ("rpm", "current"):
        if k in o:
            fly[k] = o[k]
    rate = o.get("rate", 10.0)
    if not finish:
        T.write_set(engine, data, kind, train_s=o.get("train_s", 900.0), rate=rate, **fly)
        if o.get("noise"):
            for i, f in enumerate(sorted(os.listdir(data))):
                _noisy(os.path.join(data, f), o["noise"], 100 + i)
    spec = d + "/spec.yaml" if finish else T.vehicle_spec(engine, d + "/spec.yaml", o.get("inertia", 1.0), bool(o.get("thrust_free")), o.get("kt", 1.0))
    flags = ["--sensors", "velocity"] + (["--finish"] if finish else [])
    if "current" in o:
        import math
        v, b = o["current"]
        flags += ["--current-world-mps", f"{v * math.cos(math.radians(b)):.4f}", f"{v * math.sin(math.radians(b)):.4f}", "0"]
    rc = _calibrate(data, spec, out, rate, flags)
    secs = time.time() - t0
    vj = os.path.join(out, "validation", "validation.json")
    if not os.path.exists(vj):
        return record(F, rid, "truth", engine, what, kind, {"exit": rc, "bars": BARS_HASH}, "a calibration and its validation", False, secs)
    v = json.load(open(vj))
    m = json.load(open(os.path.join(out, "metrics.json")))
    p = yaml.safe_load(open(os.path.join(out, "calibrated_parameters.yaml")))["parameters"]
    ok, n = judge_flight(v)
    n.update(exit=rc, bars=BARS_HASH, few=m["fins"]["few_manoeuvres"], held=sorted(m["fins"]["held"]), lift=[p["fin_lift_slope_a"], p["fin_lift_slope_b"]],
             inflow=[p["fin_inflow_a"], p["fin_inflow_b"]])
    if bar == "warned":
        ok = bool(n["few"])
        exp = "too few manoeuvres flagged"
    else:
        exp = f"30 s median heading <= {FLY['heading_deg']}, pitch <= {FLY['pitch_deg']}, depth <= {FLY['depth_m']}; turn rate <= {TURN_OF_HOLD} of holding"
    return record(F, rid, "truth", engine, what, kind, n, exp, ok, secs)


def more_rows() -> list:
    import fin_more_97
    return fin_more_97.more_rows()


def all_rows() -> list:
    """Every row, importable with no data (CI has none)."""
    import fin_rows_97 as R
    return truth_rows() + R.fast_rows() + R.recover_rows() + R.real_rows() + R.sim_rows() + more_rows()


def main(argv) -> int:
    cmd = argv[1] if len(argv) > 1 else "list"
    only = argv[2:]
    pick = lambda r: not only or any(r.startswith(o) for o in only)
    import fin_rows_97 as R
    if cmd == "list":
        print("\n".join(all_rows()))
        return 0
    if cmd == "truth":
        return 0 if all([run_truth(r) for r in TRUTH if pick(r[0])]) else 1
    if cmd == "truth-finish":
        return 0 if all([run_truth(r, finish=True) for r in TRUTH if pick(r[0])]) else 1
    if cmd == "fast":
        return R.run_fast(pick)
    if cmd == "recover":
        return R.run_recover(pick)
    if cmd == "real":
        return R.run_real(pick)
    if cmd == "sim":
        return R.run_sims(pick)
    if cmd == "more":
        import fin_more_97
        return fin_more_97.run_more(pick)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
