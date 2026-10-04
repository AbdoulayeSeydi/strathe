"""
Roll and pitch from free-decay tank tests (PROJECT_STATUS.md #64).

Ordinary mission logs can't calibrate roll and pitch: a BlueROV2 is passively stable, pilots don't command roll or
pitch, and most of the roll/pitch in the logs comes from the tether and the water, not the thrusters (#50). The
standard way to identify them is a free-decay test: tilt the vehicle, let go with the thrusters stopped, and record
how it swings back. The protocol is calibration/FREE_DECAY_PROTOCOL.md.

This module:
1. finds the releases in logs already converted to Strathe's CSV layout: stretches with the thrusters stopped that
   start at a large angle and swing back towards level (find_releases);
2. fits, one axis at a time, the same roll/pitch physics the calibration uses (dynamics_model.py), with the inertia
   held at vehicle_spec.yaml's value:
       I * rate_dot = K * rate + Kq * |rate| * rate - zg * W * sin(angle - rest)   [roll: times cos(pitch)]
       angle_dot   = rate
   K, Kq = Kp, Kpp (roll) or Mq, Mqq (pitch); zg = cg_cb_offset_z; W = weight; rest = the angle the vehicle rests at;
3. writes free_decay.yaml, whose `hold:` block `strathe calibrate --free-decay` holds fixed in the main fit.

What a decay can and can't tell apart: with no forcing, only ratios to the inertia are observable (the period
gives zg*W/I, the decay gives K/I and Kq/I). The fitted values are therefore "given the spec's inertia", which is the
inertia the main calibration also uses, so holding them reproduces the measured swing exactly. If roll and pitch give
different zg, the spec's roll/pitch inertia ratio is off; the file says so. The period and damping ratio are reported
as well: they don't depend on that assumption.

    python free_decay.py --data-dir <Strathe CSVs> --spec vehicle_spec.yaml --out free_decay_out
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import yaml
from scipy.optimize import least_squares

from dynamics_model import quaternion_to_roll_pitch

G = 9.81
AXES = {
    # name: (rate column, angle index (0 roll, 1 pitch), inertia param, linear, quadratic)
    "roll": ("Bp", 0, "Ieff_roll", "Kp", "Kpp"),
    "pitch": ("Bq", 1, "Ieff_pitch", "Mq", "Mqq"),
}
WRENCH = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]

# Detection defaults (deg, s, N, N*m); see find_releases
MIN_START_DEG = 8.0
MAX_START_DEG = 35.0          # beyond this the vehicle is probably at the surface or being handled
QUIET_FORCE_N = 0.5
QUIET_TORQUE_NM = 0.1
MAX_SEGMENT_S = 15.0
MIN_SEGMENT_S = 1.5
HALF_WITHIN_S = 4.0          # a release has to swing back to half its starting angle within this

# What makes an axis's fit good enough to hold in the main calibration
MIN_RELEASES = 3
MIN_R2 = 0.8
MAX_K_KQ_CORR = 0.95         # above this, linear and quadratic damping aren't told apart
ZG_AGREE_REL = 0.25          # roll and pitch zg further apart than this: the inertia ratio is off

# Fit bounds: wider than vehicle_spec.yaml's on purpose, so the test says what it measured. Values outside the
# spec's bounds are reported, and clamped (with a note) only when the main calibration holds them.
FIT_BOUNDS = {"K": (-50.0, 0.0), "Kq": (-50.0, 0.0), "zg": (1e-4, 0.3), "rest": (-math.radians(20), math.radians(20))}


@dataclass
class Release:
    file: str
    axis: str
    start: int                 # sample index in its file
    angle: np.ndarray          # rad, this axis
    rate: np.ndarray           # rad/s, this axis
    other_angle: np.ndarray    # rad: pitch for a roll release (its cos scales the roll moment), else zeros


def angles_from_quaternion(df) -> np.ndarray:
    roll, pitch = quaternion_to_roll_pitch(df["qx"].to_numpy(), df["qy"].to_numpy(),
                                           df["qz"].to_numpy(), df["qw"].to_numpy())
    return np.column_stack([roll, pitch])


def quiet_mask(df, force_n=QUIET_FORCE_N, torque_nm=QUIET_TORQUE_NM) -> np.ndarray:
    """Samples with the thrusters stopped: every logged force and torque near zero."""
    w = df[WRENCH].to_numpy(dtype=float)
    return (np.abs(w[:, :3]) <= force_n).all(1) & (np.abs(w[:, 3:]) <= torque_nm).all(1)


def _runs(mask: np.ndarray):
    """(start, end) of each run of True, end exclusive."""
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _segment_end(a: np.ndarray, w: np.ndarray, s: int, stop: int, rate_hz: float) -> int:
    """Where a release starting at s stops being a free decay: the run ends, the time cap, the vehicle settles, or
    a swing comes back bigger than the one before it (a hand or the tether put energy in). Returns an exclusive
    index."""
    stop = min(stop, s + int(round(MAX_SEGMENT_S * rate_hz)))
    peak = abs(a[s])
    settle = max(math.radians(0.5), 0.05 * peak)
    settle_n = int(round(1.0 * rate_hz))
    prev = abs(a[s])
    below = 0
    for i in range(s + 1, stop):
        below = below + 1 if abs(a[i]) < settle else 0
        if below >= settle_n:
            return i + 1
        if i + 1 < stop and np.sign(w[i]) != np.sign(w[i + 1]) and w[i] != 0:  # a swing's extreme
            ext = abs(a[i])
            if ext > prev + max(math.radians(1.0), 0.1 * prev):
                # energy came in during the half swing before this extreme: end at the previous extreme
                back = i - 1
                while back > s and not (np.sign(w[back - 1]) != np.sign(w[back])):
                    back -= 1
                return max(back, s + 1)
            prev = ext
    return stop


def find_releases(df, rate_hz: float, file: str = "", min_start_deg: float = MIN_START_DEG,
                  force_n: float = QUIET_FORCE_N, torque_nm: float = QUIET_TORQUE_NM,
                  max_start_deg: float = MAX_START_DEG) -> list[Release]:
    """The free-decay releases in one recording, both axes.

    A release starts where the vehicle is let go: inside a stretch with the thrusters stopped, the angle is
    `min_start_deg` to `max_start_deg` from where the vehicle rests and swings back to half of that within
    HALF_WITHIN_S. Beyond `max_start_deg` the vehicle is probably at the surface or being handled. If it was held
    tilted first, the start is the last moment of the hold (the last sample near the peak angle). It ends at the
    first of: thrust resumes, MAX_SEGMENT_S, the vehicle settles, or a swing grows (a hand or the tether)."""
    ang = angles_from_quaternion(df)
    quiet = quiet_mask(df, force_n, torque_nm)
    out = []
    for axis, (rate_col, idx, *_rest) in AXES.items():
        a_raw = ang[:, idx]
        w = df[rate_col].to_numpy(dtype=float)
        min_a = math.radians(min_start_deg)
        # where it rests: still, thrusters stopped, and near level. Without "near level", a vehicle held tilted for
        # long stretches would pull the estimate towards the held angle (the fit then finds the exact rest angle)
        still = quiet & (np.abs(w) < 0.05) & (np.abs(a_raw) < min_a)
        rest = float(np.median(a_raw[still])) if still.sum() >= 5 else 0.0
        a = a_raw - rest
        other = ang[:, 1] if axis == "roll" else np.zeros(len(df))
        half_n = int(round(HALF_WITHIN_S * rate_hz))
        for r0, r1 in _runs(quiet):
            i = r0
            while i < r1:
                if abs(a[i]) < min_a:
                    i += 1
                    continue
                window = np.abs(a[i:min(r1, i + half_n)])
                below_half = np.flatnonzero(window < 0.5 * abs(a[i]))
                if not len(below_half):
                    i += 1
                    continue
                j_half = i + int(below_half[0])
                peak = float(np.max(np.abs(a[i:j_half])))
                near = np.flatnonzero(np.abs(a[i:j_half]) >= peak - max(math.radians(0.5), 0.05 * peak))
                s = i + int(near[-1])  # the moment of release
                e = _segment_end(a, w, s, r1, rate_hz)
                if (e - s) / rate_hz >= MIN_SEGMENT_S and abs(a[s]) <= math.radians(max_start_deg):
                    out.append(Release(file, axis, s, a_raw[s:e], w[s:e], other[s:e]))
                i = max(e, s + 1)
    return out


def simulate_release(params, rel: Release, inertia: float, weight: float, dt: float, step_s: float = 0.02):
    """Angle and rate over one release, from its first sample's logged angle and rate. RK4 in steps of at most
    step_s (the calibration's own physics step), written out inline: this runs a few hundred times per fit."""
    K, Kq, zg, rest = (float(x) for x in params)
    substeps = max(1, math.ceil(dt / step_s - 1e-9))
    h = dt / substeps
    kI, qI = K / inertia, Kq / inertia
    sin = math.sin
    n = len(rel.angle)
    angle, rate = np.empty(n), np.empty(n)
    a, w = float(rel.angle[0]), float(rel.rate[0])
    cos_other = np.cos(rel.other_angle).tolist()
    for k in range(n):
        angle[k], rate[k] = a, w
        g = zg * weight * cos_other[k] / inertia
        for _ in range(substeps):
            d1 = kI * w + qI * abs(w) * w - g * sin(a - rest)
            w2 = w + 0.5 * h * d1
            d2 = kI * w2 + qI * abs(w2) * w2 - g * sin(a + 0.5 * h * w - rest)
            w3 = w + 0.5 * h * d2
            d3 = kI * w3 + qI * abs(w3) * w3 - g * sin(a + 0.5 * h * w2 - rest)
            w4 = w + h * d3
            d4 = kI * w4 + qI * abs(w4) * w4 - g * sin(a + h * w3 - rest)
            a += h / 6 * (w + 2 * w2 + 2 * w3 + w4)
            w += h / 6 * (d1 + 2 * d2 + 2 * d3 + d4)
            if not abs(w) < 1e3:  # diverged: let the residual say so instead of overflowing
                angle[k:], rate[k:] = 1e3, 1e3
                return angle, rate
    return angle, rate


def _residuals(params, releases, inertia, weight, dt, s_angle, s_rate):
    res = []
    for rel in releases:
        a, w = simulate_release(params, rel, inertia, weight, dt)
        res.append((a - rel.angle) / s_angle)
        res.append((w - rel.rate) / s_rate)
    return np.concatenate(res)


def fit_axis(releases: list[Release], inertia: float, weight: float, rate_hz: float) -> dict:
    """Fit linear damping, quadratic damping, zg and the rest angle to one axis's releases."""
    dt = 1.0 / rate_hz
    s_angle = max(float(np.sqrt(np.mean(np.concatenate([r.angle for r in releases]) ** 2))), 1e-3)
    s_rate = max(float(np.sqrt(np.mean(np.concatenate([r.rate for r in releases]) ** 2))), 1e-3)
    names = ["K", "Kq", "zg", "rest"]
    lb = np.array([FIT_BOUNDS[n][0] for n in names])
    ub = np.array([FIT_BOUNDS[n][1] for n in names])
    rest0 = float(np.median(np.concatenate([r.angle[-max(1, len(r.angle) // 4):] for r in releases])))
    best = None
    for K0, Kq0, zg0 in ((-0.5, -0.5, 0.02), (-3.0, -1.0, 0.01), (-0.1, -0.1, 0.05)):
        x0 = np.clip([K0, Kq0, zg0, rest0], lb + 1e-9, ub - 1e-9)
        r = least_squares(_residuals, x0, bounds=(lb, ub), args=(releases, inertia, weight, dt, s_angle, s_rate),
                          x_scale=ub - lb, max_nfev=200)
        if best is None or r.cost < best.cost:
            best = r
    K, Kq, zg, rest = best.x
    # quality: how much of the swing the model explains, per release and overall
    sse = sst = 0.0
    per = []
    for rel in releases:
        a, _ = simulate_release(best.x, rel, inertia, weight, dt)
        e = float(np.sum((a - rel.angle) ** 2))
        t = float(np.sum((rel.angle - rest) ** 2))
        sse, sst = sse + e, sst + t
        per.append(round(1 - e / t, 3) if t > 0 else None)
    r2 = 1 - sse / sst if sst > 0 else float("nan")
    # uncertainty from the Jacobian at the optimum (residual variance x (J'J)^-1)
    J = best.jac
    dof = max(1, len(best.fun) - len(best.x))
    s2 = 2 * best.cost / dof
    try:
        cov = np.linalg.inv(J.T @ J) * s2
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
        corr = cov[0, 1] / (std[0] * std[1]) if std[0] > 0 and std[1] > 0 else float("nan")
    except np.linalg.LinAlgError:
        std, corr = np.full(4, np.nan), float("nan")
    # model-independent summary: small-swing natural period and damping ratio
    k_stiff = zg * weight
    wn = math.sqrt(k_stiff / inertia)
    zeta = -K / (2 * math.sqrt(k_stiff * inertia))
    period = 2 * math.pi / (wn * math.sqrt(1 - zeta ** 2)) if zeta < 1 else None
    return {"K": float(K), "Kq": float(Kq), "zg": float(zg), "rest_rad": float(rest),
            "std": {n: float(s) for n, s in zip(names, std)}, "corr_K_Kq": float(corr), "r2_angle": float(r2),
            "r2_per_release": per, "natural_period_s": None if period is None else float(period),
            "damping_ratio_small_swing": float(zeta), "converged": bool(best.success)}


def load_logs(data_dir: str):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    files = [f for f in files if os.path.basename(f) not in ("currents.csv",)]
    import pandas as pd
    out = []
    for f in files:
        df = pd.read_csv(f)
        missing = [c for c in WRENCH + ["Bp", "Bq", "qx", "qy", "qz", "qw"] if c not in df.columns]
        if missing:
            raise ValueError(f"{f} is missing column(s) {missing}: is it in Strathe's CSV layout (`strathe ingest`)?")
        out.append((f, df))
    return out


def spec_values(spec_path: str) -> dict:
    """Mass, inertia and bounds from vehicle_spec.yaml (the same loader the calibration uses)."""
    from optimizer import load_vehicle_spec
    loaded = load_vehicle_spec(spec_path)
    flat = loaded.flat_calibration_targets
    return {"mass_kg": loaded.vehicle.mass_kg,
            "inertia": {n: float(flat[n]["initial_guess"]) for n in ("Ieff_roll", "Ieff_pitch")},
            "bounds": {n: tuple(float(b) for b in flat[n]["bounds"])
                       for n in ("Kp", "Kpp", "Mq", "Mqq", "cg_cb_offset_z")}}


def analyse(logs, rate_hz: float, spec: dict, min_start_deg: float = MIN_START_DEG,
            max_start_deg: float = MAX_START_DEG) -> dict:
    """Find the releases in every log, fit each axis that has enough, and decide what the calibration can hold."""
    weight = spec["mass_kg"] * G
    releases = []
    for f, df in logs:
        releases += find_releases(df, rate_hz, os.path.basename(f), min_start_deg, max_start_deg=max_start_deg)
    result = {"sampling_rate_hz": rate_hz, "mass_kg": spec["mass_kg"], "inertia_assumed": spec["inertia"],
              "axes": {}, "hold": {}, "warnings": [], "notes": []}
    zg = {}
    for axis, (_col, _idx, inertia_name, k_name, kq_name) in AXES.items():
        rels = [r for r in releases if r.axis == axis]
        entry = {"releases": len(rels), "files": sorted({r.file for r in rels})}
        result["axes"][axis] = entry
        if not rels:
            result["warnings"].append(f"{axis}: no releases found (thrusters stopped, starting {min_start_deg:g}-"
                                      f"{max_start_deg:g} deg from rest, swinging back). See FREE_DECAY_PROTOCOL.md.")
            continue
        fit = fit_axis(rels, spec["inertia"][inertia_name], weight, rate_hz)
        entry["start_angles_deg"] = [round(math.degrees(r.angle[0] - fit["rest_rad"]), 1) for r in rels]  # from rest
        entry.update({k_name: round(fit["K"], 4), kq_name: round(fit["Kq"], 4),
                      "cg_cb_offset_z": round(fit["zg"], 5),
                      "std": {k_name: round(fit["std"]["K"], 4), kq_name: round(fit["std"]["Kq"], 4),
                              "cg_cb_offset_z": round(fit["std"]["zg"], 5)},
                      f"corr_{k_name}_{kq_name}": round(fit["corr_K_Kq"], 3),
                      "rests_at_deg": round(math.degrees(fit["rest_rad"]), 2),
                      "natural_period_s": None if fit["natural_period_s"] is None else round(fit["natural_period_s"], 3),
                      "damping_ratio_small_swing": round(fit["damping_ratio_small_swing"], 3),
                      "r2_angle": round(fit["r2_angle"], 3), "r2_per_release": fit["r2_per_release"],
                      "converged": fit["converged"]})
        problems = []
        if len(rels) < MIN_RELEASES:
            problems.append(f"only {len(rels)} release(s); at least {MIN_RELEASES} are needed")
        if not fit["r2_angle"] >= MIN_R2:
            problems.append(f"the fitted swing explains {fit['r2_angle']:.0%} of the measured one (needs "
                            f"{MIN_R2:.0%}): something other than the vehicle's own righting and drag was acting "
                            "(the tether, a hand, thrust under the quiet threshold)")
        if not fit["converged"]:
            problems.append("the fit didn't converge")
        at_edge = [n for n, key in ((k_name, "K"), (kq_name, "Kq"), ("cg_cb_offset_z", "zg"))
                   if min(abs(fit[key] - FIT_BOUNDS[key][0]), abs(fit[key] - FIT_BOUNDS[key][1]))
                   < 1e-3 * (FIT_BOUNDS[key][1] - FIT_BOUNDS[key][0])]
        if at_edge:
            # a swing kept going by something outside (a tether, a wave) is fitted by turning a damping term off;
            # the fit quality can still look good, so this is checked separately (synthetic tests, #64)
            problems.append(f"{', '.join(at_edge)} reached the edge of what the fit allows: something other than "
                            "the vehicle's own righting and drag is shaping the swing")
        if problems:
            result["warnings"].append(f"{axis}: not held in the calibration: " + "; ".join(problems) + ".")
            entry["held"] = False
            continue
        entry["held"] = True
        result["hold"][k_name] = round(fit["K"], 4)
        result["hold"][kq_name] = round(fit["Kq"], 4)
        zg[axis] = (fit["zg"], fit["std"]["zg"])
        if abs(fit["corr_K_Kq"]) > MAX_K_KQ_CORR:
            result["warnings"].append(
                f"{axis}: linear and quadratic damping are hard to tell apart in these releases "
                f"(correlation {fit['corr_K_Kq']:+.2f}); their combined effect is right, the split isn't. Release "
                f"from two different angles (about 10 and 25 deg) to separate them.")
        span = (min(abs(a) for a in entry["start_angles_deg"]), max(abs(a) for a in entry["start_angles_deg"]))
        entry["start_angle_span_deg"] = [round(span[0], 1), round(span[1], 1)]
    if zg:
        if len(zg) == 2:
            (z1, s1), (z2, s2) = zg["roll"], zg["pitch"]
            w1, w2 = (1 / s1 ** 2, 1 / s2 ** 2) if s1 > 0 and s2 > 0 else (1.0, 1.0)
            combined = (w1 * z1 + w2 * z2) / (w1 + w2)
            if abs(z1 - z2) > ZG_AGREE_REL * max(z1, z2):
                result["warnings"].append(
                    f"roll and pitch give different righting arms ({z1 * 100:.2f} vs {z2 * 100:.2f} cm). One vehicle "
                    f"has one, so the inertias in vehicle_spec.yaml are probably off: pitch over roll inertia is "
                    f"about {spec['inertia']['Ieff_pitch'] * z1 / z2 / spec['inertia']['Ieff_roll']:.2f} by these "
                    f"tests, against {spec['inertia']['Ieff_pitch'] / spec['inertia']['Ieff_roll']:.2f} in the "
                    f"spec. The calibration holds their weighted mean ({combined * 100:.2f} cm).")
        else:
            combined = next(iter(zg.values()))[0]
        result["hold"]["cg_cb_offset_z"] = round(float(combined), 5)
    result["notes"].append("Values are given the inertia in vehicle_spec.yaml (inertia_assumed): a free decay "
                           "measures ratios to the inertia, and the main calibration uses the same inertia, so "
                           "holding them reproduces the measured swing. The period and damping ratio don't depend "
                           "on it.")
    return result


def hold_values(path: str) -> dict:
    """The `hold:` block of a free_decay.yaml: what `strathe calibrate --free-decay` holds fixed."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    hold = raw.get("hold") or {}
    allowed = {"Kp", "Kpp", "Mq", "Mqq", "cg_cb_offset_z"}
    bad = set(hold) - allowed
    if bad:
        raise ValueError(f"{path}: hold has parameter(s) a free-decay test doesn't measure: {sorted(bad)}")
    return {k: float(v) for k, v in hold.items()}


def within_bounds(hold: dict, names: list, lb, ub) -> tuple[dict, list]:
    """Clamp held values into vehicle_spec.yaml's bounds, saying so (the optimizer can't hold a value outside
    them)."""
    out, notes = {}, []
    for k, v in hold.items():
        i = names.index(k)
        c = float(min(max(v, lb[i]), ub[i]))
        if c != v:
            notes.append(f"{k}: the tank test gave {v:g}, outside vehicle_spec.yaml's bounds [{lb[i]:g}, {ub[i]:g}]; "
                         f"held at {c:g}. If you trust the test, widen the bounds")
        out[k] = c
    return out, notes


def write_report(result: dict, path: str) -> None:
    lines = ["# Free-decay tests: roll and pitch", ""]
    held = [a for a, e in result["axes"].items() if e.get("held")]
    if held:
        lines.append(f"**Held in the calibration:** {', '.join(held)}. `strathe calibrate --free-decay "
                     f"free_decay.yaml` holds these values fixed instead of fitting them from mission logs.")
    else:
        lines.append("**Nothing to hold:** no axis had a usable fit. The warnings below say why.")
    lines.append("")
    for axis, e in result["axes"].items():
        lines.append(f"## {axis.capitalize()}")
        if not e["releases"]:
            lines += ["No releases found.", ""]
            continue
        k_name, kq_name = AXES[axis][3], AXES[axis][4]
        lines += [
            f"- **Releases:** {e['releases']}, let go {min(abs(a) for a in e['start_angles_deg']):.0f} to "
            f"{max(abs(a) for a in e['start_angles_deg']):.0f} deg from where it rests.",
            f"- **The swing:** period {e['natural_period_s']} s for small swings, damping ratio "
            f"{e['damping_ratio_small_swing']} (1 = no overshoot). The vehicle rests at {e['rests_at_deg']:+.1f} deg.",
            f"- **Fit:** explains {e['r2_angle']:.0%} of the measured swing.",
            f"- **Values:** {k_name} {e[k_name]} ± {e['std'][k_name]}, {kq_name} {e[kq_name]} ± {e['std'][kq_name]}, "
            f"righting arm (cg_cb_offset_z) {e['cg_cb_offset_z'] * 100:.2f} ± {e['std']['cg_cb_offset_z'] * 100:.2f} cm.",
            ""]
    if result["warnings"]:
        lines += ["## Warnings", ""] + [f"- {w}" for w in result["warnings"]] + [""]
    lines += ["## Notes", ""] + [f"- {n}" for n in result["notes"]] + [""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def rate_from_folder(data_dir: str):
    path = os.path.join(data_dir, "ingest_report.json")
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f).get("sampling_rate_hz")
        except (OSError, ValueError):
            return None
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="the tank-test logs in Strathe's CSV layout")
    ap.add_argument("--spec", required=True, help="vehicle_spec.yaml (mass, inertia, bounds)")
    ap.add_argument("--out", required=True, help="output folder (free_decay.yaml, free_decay_report.md)")
    ap.add_argument("--sampling-rate-hz", type=float, default=None,
                    help="default: from ingest_report.json in the data folder")
    ap.add_argument("--min-start-deg", type=float, default=MIN_START_DEG,
                    help=f"smallest starting angle that counts as a release (default {MIN_START_DEG:g})")
    ap.add_argument("--max-start-deg", type=float, default=MAX_START_DEG,
                    help=f"largest; beyond it the vehicle is probably at the surface (default {MAX_START_DEG:g})")
    args = ap.parse_args(argv)
    rate = args.sampling_rate_hz or rate_from_folder(args.data_dir)
    if not rate:
        print("❌ no sampling rate: pass --sampling-rate-hz, or convert the logs with `strathe ingest` "
              "(it records the rate in ingest_report.json)")
        return 2
    logs = load_logs(args.data_dir)
    if not logs:
        print(f"❌ no CSV logs in {args.data_dir}")
        return 2
    result = analyse(logs, float(rate), spec_values(args.spec), args.min_start_deg, args.max_start_deg)
    result["data_dir"] = os.path.abspath(args.data_dir)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "free_decay.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=False, allow_unicode=True)
    write_report(result, os.path.join(args.out, "free_decay_report.md"))
    for axis, e in result["axes"].items():
        if e["releases"] and "r2_angle" in e:
            print(f"{axis}: {e['releases']} release(s), period {e['natural_period_s']} s, damping ratio "
                  f"{e['damping_ratio_small_swing']}, fit explains {e['r2_angle']:.0%}"
                  + (" -> held" if e.get("held") else " -> not held"), flush=True)
        else:
            print(f"{axis}: no releases found", flush=True)
    for w in result["warnings"]:
        print(f"⚠️  {w}", flush=True)
    print(f"Wrote {os.path.join(args.out, 'free_decay.yaml')} and free_decay_report.md", flush=True)
    if result["hold"]:
        print(f"Next: strathe calibrate ... --free-decay {os.path.join(args.out, 'free_decay.yaml')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
