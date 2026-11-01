#!/usr/bin/env python3
"""
calibration_report.py -- a calibration result explained for someone who
doesn't do system identification.

    python calibration_report.py --result ../calibration_output_x \
        [--identifiability identifiability_report.json] [--data-dir <logs>] [--spec vehicle_spec.yaml]

Writes <result>/calibration_report.md answering, in plain language:
  1. Is the calibrated simulator better than the default one, on sessions it never saw?
  2. Which motions can I trust it for? (a verdict per axis, with the reason)
  3. What in the result should I be careful with? (parameters stuck at their
     limits, physically implausible values, parameters the data can't separate)
  4. What should I do next to make it better?
A short technical appendix keeps the numbers for specialists.

Verdicts are based on SKILL: how much better than simply assuming "no
motion on this axis" the calibrated model predicts each motion on unseen
sessions (skill = 1 - error / no-motion error). Beating the default model
is not enough: in the reference dataset the default model is so poor at
roll and pitch that a calibration can beat it by 80% and still predict
roll and pitch worse than "no motion" (PROJECT_STATUS.md #50).

Inputs: the calibration output folder (calibrated_parameters.yaml, and
metrics.json if run_calibration.py wrote one), optionally the JSON from
automated_identifiability_report.py. If metrics.json has no per-axis
held-out numbers (e.g. MuJoCo/HoloOcean outputs), pass --data-dir and they are
computed with the stand-in model on the held-out dates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Optional

import numpy as np
import yaml

from dynamics_model import CalibrationParameters
from optimizer import initial_parameters, load_vehicle_spec, parameter_bounds

AXES = [("u", "forward speed (surge)"), ("v", "sideways speed (sway)"), ("w", "up/down speed (heave)"),
        ("p", "roll rate"), ("q", "pitch rate"), ("r", "turning rate (yaw)")]
AXES_INDEX = {k: i for i, (k, _) in enumerate(AXES)}
LABELS = dict(AXES)

PLAIN = {
    "Ieff_roll": "resistance to rolling (inertia + added inertia)", "Ieff_pitch": "resistance to pitching",
    "Ieff_yaw": "resistance to turning", "Xu": "forward drag (slow speeds)", "Yv": "sideways drag (slow speeds)",
    "Zw": "vertical drag (slow speeds)", "Kp": "roll damping", "Mq": "pitch damping", "Nr": "turning damping",
    "Xuu": "forward drag (fast speeds)", "Yvv": "sideways drag (fast speeds)", "Zww": "vertical drag (fast speeds)",
    "Kpp": "roll damping (fast rates)", "Mqq": "pitch damping (fast rates)", "Nrr": "turning damping (fast rates)",
    "Xu_dot": "water carried along when accelerating forward", "Yv_dot": "water carried along sideways",
    "Zw_dot": "water carried along vertically", "net_buoyancy_z": "net buoyancy (float vs sink)",
    "cg_cb_offset_z": "self-righting (centre of gravity below buoyancy)",
    "Xq_dot": "surge-pitch coupling", "Zq_dot": "heave-pitch coupling",
    "cg_cb_offset_x": "centre of gravity ahead of buoyancy (nose-down trim)", "Zuw": "hull lift (grows with speed)",
    "Muw": "hull lift's pitching/turning moment", "Muq": "pitch/turn damping that grows with speed",
}
# which parameters mainly drive each axis (for "why" and "what next")
AXIS_PARAMS = {"u": ["Xu", "Xuu", "Xu_dot"], "v": ["Yv", "Yvv", "Yv_dot", "Zuw"],
               "w": ["Zw", "Zww", "Zw_dot", "net_buoyancy_z", "Zuw"],
               "p": ["Ieff_roll", "Kp", "Kpp", "cg_cb_offset_z"],
               "q": ["Ieff_pitch", "Mq", "Mqq", "cg_cb_offset_z", "cg_cb_offset_x", "Muw", "Muq"],
               "r": ["Ieff_yaw", "Nr", "Nrr", "Muw", "Muq"]}
# a fitted value more than this far from the vehicle's own independent measurement (a calibration target's
# `reference:` in its vehicle_spec.yaml; PROJECT_STATUS.md #28, #70) is flagged as physically implausible
MAX_PLAUSIBLE_RATIO = 5.0
GOOD_SKILL = 0.3   # explains at least ~half the motion's variance: 1 - (1 - 0.3)^2 = 0.51

GOOD, CAUTION, BAD = "Trust it", "Use with care", "Don't rely on it"
IDENT_FILENAME = "identifiability_report.json"

def per_axis_heldout(metrics: Optional[dict], params, spec, data_dir: Optional[str], heldout_dates) -> Optional[tuple]:
    """(default_rmse[6], calibrated_rmse[6], zero_rmse[6] or None) on held-out
    sessions. zero_rmse is the error of predicting no motion; None when neither
    metrics.json nor the logs can provide it."""
    d = c = z = None
    if metrics and "per_dof" in metrics and "calibrated_heldout" in metrics["per_dof"]:
        pd_ = metrics["per_dof"]
        d, c = np.array(pd_["default_heldout"]), np.array(pd_["calibrated_heldout"])
        if (metrics.get("skill_vs_zero") or {}).get("zero_heldout") is not None:
            z = np.array(metrics["skill_vs_zero"]["zero_heldout"])
    if z is None and data_dir is not None:
        from data_split import discover_csv_files, split_by_date
        from objective import DatasetConfig, load_run, rmse_by_dof, stable_rmse_by_dof, zero_baseline_rmse
        files = discover_csv_files(data_dir)[0]
        split = split_by_date(files, heldout_dates=heldout_dates)
        # same torque convention the calibration used (metrics.json), else the spec's
        offset = ((metrics or {}).get("objective") or {}).get("torque_reference_offset_m")
        if offset is None:
            from run_calibration import spec_torque_reference_offset
            offset = spec_torque_reference_offset(spec)
        current = ((metrics or {}).get("objective") or {}).get("current_world_mps")
        rate = ((metrics or {}).get("objective") or {}).get("sampling_rate_hz") or 10.0
        by_file = ((metrics or {}).get("objective") or {}).get("current_by_file")
        by_file = None if not by_file else tuple((f, tuple(c)) for f, c in by_file)
        runs = [load_run(f, DatasetConfig(float(rate), tuple(offset), None if current is None else tuple(current),
                                          by_file)) for f in split.heldout_files]
        z = zero_baseline_rmse(runs)
        if c is None:
            d = np.mean([rmse_by_dof(initial_parameters(spec), r, spec.vehicle) for r in runs], axis=0)
            c = stable_rmse_by_dof(params, runs, spec.vehicle)[0]  # diverged recordings left out
    if c is None:
        return None
    return d, c, z


def pinned_parameters(params, spec) -> list:
    lb, ub = parameter_bounds(spec)
    names = params.param_names
    v = params.to_vector()
    out = []
    for i, n in enumerate(names):
        span = ub[i] - lb[i]
        if span <= 0:
            continue
        if abs(v[i] - lb[i]) < 1e-3 * span or abs(v[i] - ub[i]) < 1e-3 * span:
            out.append({"name": n, "value": float(v[i]), "bound": "lower" if abs(v[i] - lb[i]) < abs(v[i] - ub[i]) else "upper",
                        "bounds": [float(lb[i]), float(ub[i])]})
    return out


def physical_references(spec) -> dict:
    """{parameter: (value, source)} from the vehicle description's own `reference:` entries: independent measurements
    of THIS vehicle. A vehicle without them gets no plausibility check, not another vehicle's numbers."""
    return {n: (float(e["reference"]["value"]), str(e["reference"]["source"]))
            for n, e in spec.flat_calibration_targets.items() if e.get("reference")}


def implausible_parameters(params, spec) -> list:
    out = []
    for n, (ref, src) in physical_references(spec).items():
        val = getattr(params, n)
        ratio = val / ref
        if ratio > MAX_PLAUSIBLE_RATIO or ratio < 1 / MAX_PLAUSIBLE_RATIO:
            out.append({"name": n, "value": val, "reference": ref, "source": src, "ratio": ratio})
    return out


def sensor_axis_numbers(metrics: Optional[dict]) -> dict:
    """The sensor suite (#96): for each linear axis no DVL measured on the held-out recordings, the held-out RMSE of the
    sensors that see it instead, as (default, calibrated, no-motion, what measured it, unit), or held: True when no
    sensor saw it. Surge and sway share the horizontal position fixes (n, e), heave the depth (and vertical fixes)."""
    ss = (metrics or {}).get("sensor_suite") or {}
    fit = ss.get("fit") or {}
    if not fit:
        return {}
    seen = ss.get("axes_seen_by") or {}
    kinds = {}
    for lab in ss.get("channels", []):
        name, ax = lab.rsplit(".", 1)
        kinds.setdefault(name, set()).add(ax)

    def pooled(labels, key):
        vals = [fit[key].get(lab) for lab in labels]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.sqrt(np.mean(np.square(vals)))) if vals else float("nan")
    out = {}
    for axis in ("u", "v", "w"):
        by = seen.get(axis) or []
        if not by:
            out[axis] = {"held": True}
            continue
        if axis in "uv":
            labels = [f"{n}.{a}" for n, axs in kinds.items() if {"n", "e"} <= axs and n not in ("dvl", "gyro")
                      for a in ("n", "e")]
            what = "the position fixes (north/east, m)"
        else:
            labels = [f"{n}.d" for n, axs in kinds.items() if "d" in axs and n not in ("dvl", "gyro")]
            what = "depth (m)"
        out[axis] = {"held": False, "default": pooled(labels, "default_heldout"),
                     "calibrated": pooled(labels, "calibrated_heldout"), "zero": pooled(labels, "still_heldout"),
                     "from": what, "labels": labels}
    return out


def axis_verdicts(default_rmse, calib_rmse, pinned_names, implausible_names, zero_rmse=None,
                  excluded_axes=(), measured_axes=(), sensor_axes=None) -> list:
    """One verdict per axis. The main test is skill against predicting "no
    motion" (see module docstring); improvement over the default model is
    reported too but can't earn a "Trust it" on its own. sensor_axes (#96): the
    linear axes judged by the sensors that see them instead of a DVL (sensor_axis_numbers)."""
    verdicts = []
    sensor_axes = sensor_axes or {}
    for i, (k, label) in enumerate(AXES):
        via = sensor_axes.get(k)
        if via is not None and via.get("held"):
            v, why = BAD, ("not calibrated: no sensor in your logs sees this motion (no DVL, and no position fixes or "
                           "depth for it), so its values are your vehicle description's (SENSORS.md)")
            verdicts.append({"axis": k, "label": label, "improvement_pct": float("nan"), "skill": None, "verdict": v,
                             "why": why, "excluded": True, "measured": False, "default_rmse": float("nan"),
                             "calibrated_rmse": float("nan"), "zero_rmse": None, "unit": "-", "from": None})
            continue
        if via is not None:   # judged by its own sensors, in their units
            default_rmse = np.array(default_rmse, dtype=float)
            calib_rmse = np.array(calib_rmse, dtype=float)
            default_rmse[i], calib_rmse[i] = via["default"], via["calibrated"]
            if zero_rmse is not None:
                zero_rmse = np.array(zero_rmse, dtype=float)
                zero_rmse[i] = via["zero"]
        if not (np.isfinite(default_rmse[i]) and np.isfinite(calib_rmse[i])):
            verdicts.append({"axis": k, "label": label, "improvement_pct": float("nan"), "skill": None,
                             "verdict": CAUTION, "why": "it couldn't be checked: the held-out recordings have no "
                             "reading of this motion", "excluded": False, "measured": False,
                             "default_rmse": float("nan"), "calibrated_rmse": float("nan"), "zero_rmse": None,
                             "unit": "-", "from": None})
            continue
        imp = 100 * (default_rmse[i] - calib_rmse[i]) / default_rmse[i]
        skill = None if zero_rmse is None or not np.isfinite(zero_rmse[i]) else float(1 - calib_rmse[i] / zero_rmse[i])
        involved = [p for p in AXIS_PARAMS[k] if p in pinned_names or p in implausible_names]
        if k in excluded_axes:
            v, why = BAD, ("not calibrated: your logs don't explain this motion, so it was left out of the fit and "
                           "the simulator uses the reference values from vehicle_spec.yaml")
        elif skill is not None and skill <= 0:
            if k in measured_axes:
                v, why = BAD, (f"it predicts this motion no better than assuming no motion (skill {skill:+.2f}). Its "
                               f"damping and righting were measured in your free-decay tests, so what's wrong is the "
                               f"thruster-driven part: the roll/pitch the model gets from the thrusters doesn't match "
                               f"these logs (LIMITS.md)")
                verdicts.append({"axis": k, "label": label, "improvement_pct": imp, "skill": skill, "verdict": v,
                                 "why": why, "excluded": k in excluded_axes, "measured": True,
                                 "default_rmse": float(default_rmse[i]), "calibrated_rmse": float(calib_rmse[i]),
                                 "zero_rmse": None if zero_rmse is None else float(zero_rmse[i])})
                continue
            v, why = BAD, (f"it predicts this motion no better than simply assuming the vehicle isn't moving on this "
                           f"axis (skill {skill:+.2f}), even though it is {imp:.0f}% better than the default model"
                           if imp > 0 else
                           f"it predicts this motion worse than assuming no motion (skill {skill:+.2f}) and WORSE "
                           f"than the default model")
        elif imp < 10:
            v, why = BAD, ("the calibrated model predicts this motion no better than the default model"
                           if imp >= 0 else "the calibrated model predicts this motion WORSE than the default model")
        elif skill is None:
            v, why = CAUTION, (f"it improves prediction by {'only ' if imp < 30 else ''}{imp:.0f}% over the default "
                               f"model, but it couldn't be compared with simply assuming no motion (re-run with "
                               f"--data-dir)")
        elif skill < GOOD_SKILL or imp < 30 or involved:
            v = CAUTION
            reasons = []
            if skill < GOOD_SKILL:
                reasons.append(f"it only partly predicts this motion (skill {skill:+.2f}; 1 = perfect, 0 = no better "
                               f"than assuming no motion)")
            elif imp < 30:
                reasons.append(f"it improves prediction by only {imp:.0f}% over the default model")
            if involved:
                reasons.append("it relies on " + ", ".join(PLAIN[p] for p in involved)
                               + " which hit their allowed limits or look physically implausible")
            why = "; ".join(reasons)
        else:
            v, why = GOOD, (f"it predicts this motion well on unseen sessions (skill {skill:+.2f}; 1 = perfect) and "
                            f"{imp:.0f}% more accurately than the default model")
        if via is not None:
            why += f" (measured by {via['from']}: there's no DVL)"
        unit = ("m" if via is not None else "m/s") if k in "uvw" else "rad/s"
        verdicts.append({"axis": k, "label": label, "improvement_pct": imp, "skill": skill, "verdict": v, "why": why,
                         "excluded": k in excluded_axes, "measured": k in measured_axes,
                         "default_rmse": float(default_rmse[i]), "calibrated_rmse": float(calib_rmse[i]),
                         "zero_rmse": None if zero_rmse is None else float(zero_rmse[i]), "unit": unit,
                         "from": None if via is None else via["from"]})
    return verdicts


def fit_note(metrics) -> Optional[str]:
    """A fit no better than the starting description on its own training recordings failed; one that was retried from
    more starting points says so (#93)."""
    r = (metrics or {}).get("rmse") or {}
    auto = ((metrics or {}).get("optimizer") or {}).get("auto_restart")
    if r.get("calibrated_train") is not None and r.get("default_train") and \
            r["calibrated_train"] >= 0.9 * r["default_train"]:
        return (f"> ❌ **The fit failed.** On the recordings it trained on, the calibrated model is no better than your "
                f"vehicle description's starting values (error {r['calibrated_train']:.3g} against "
                f"{r['default_train']:.3g}){', even from ' + str(1 + auto['more_starts']) + ' starting points' if auto else ''}"
                f". Its values aren't a calibration. Check the description's ranges (a value stuck at one, below) and "
                f"the logs' data checks, or run again with --restarts 4.")
    if auto:
        first = ("ended where the simulation is too sensitive to these values to fit (a tiny change in them changed "
                 "its training error by far more than it should, so the optimizer's slopes were noise)"
                 if auto.get("why") == "rough" else "stopped no better than where it started")
        return (f"> ℹ️ The first fit {first} (training cost {auto['first_cost']:.4g}), so Strathe tried "
                f"{auto['more_starts']} more starting points and kept the lowest training cost "
                f"({auto['best_cost']:.4g}).")
    return None


def restart_costs(result_dir, metrics):
    """Each starting point's training cost (#79): the first fit's, which with --estimate-current is round 1's."""
    first = os.path.join(result_dir, "current_round1", "metrics.json")
    if os.path.exists(first):
        try:
            with open(first, encoding="utf-8") as f:
                metrics = json.load(f)
        except (OSError, ValueError):
            return None
    return ((metrics or {}).get("optimizer") or {}).get("restart_costs")


def whole_fit_starts(result_dir) -> Optional[dict]:
    """--restart-whole-fit's record (strathe_cli.WHOLE_FIT_FILE): each start's seed and round-by-round costs."""
    try:
        with open(os.path.join(result_dir, "whole_fit_starts.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def vehicle_description_note(spec, data_dir: Optional[str]) -> Optional[str]:
    """When the calibration used the shipped BlueROV2 Heavy description but the logs' thrusters aren't that vehicle's
    (ingest_report.json): the plain warning, else None. Fitting a different vehicle with a BlueROV2's mass, inertia
    and limits pushes values to their limits and can blow the simulation up (a 33 kg AUV: PROJECT_STATUS.md #70)."""
    if (spec.raw.get("vehicle") or {}).get("name") != "bluerov2_heavy" or not data_dir:
        return None
    path = os.path.join(data_dir, "ingest_report.json")
    try:
        thrusters = json.load(open(path, encoding="utf-8")).get("thrusters") if os.path.exists(path) else None
    except (OSError, ValueError):
        return None
    if not thrusters or thrusters.startswith("BlueROV2 Heavy"):
        return None
    return (f"This calibration used the shipped vehicle description, a BlueROV2 Heavy "
            f"({spec.vehicle.mass_kg:g} kg, with its inertia and limits), but your logs' thrusters are "
            f"{thrusters.split(';')[0]}. If your vehicle isn't a BlueROV2, describe it first and calibrate again "
            f"with --spec: copy calibration/templates/vehicle_spec_template.yaml (LAB_GUIDE.md step 4). Values "
            f"stuck at their limits and poor skill below are what a mismatched description produces.")


def next_steps(verdicts, pinned, implausible, ident, vehicle_note: Optional[str] = None) -> list:
    steps = []
    if vehicle_note:
        steps.append("Describe your own vehicle (mass, inertia, drag and their limits) in a copy of "
                     "calibration/templates/vehicle_spec_template.yaml and calibrate again with --spec: this "
                     "calibration used a BlueROV2 Heavy's description (see \"Be careful with\"). Do this before "
                     "acting on the steps below.")
    # more data helps axes the model predicts poorly; axes flagged only because a
    # parameter hit a limit need the limits reviewed instead (next step)
    unexplained = [v for v in verdicts if v.get("excluded") or (v.get("skill") is not None and v["skill"] <= 0)]
    measured = [v for v in unexplained if v.get("measured")]
    unexplained = [v for v in unexplained if not v.get("measured")]
    weak = [v for v in verdicts if v not in unexplained and
            (v["improvement_pct"] < 30 or (v.get("skill") is not None and v["skill"] < GOOD_SKILL))]
    if unexplained:
        steps.append("Your logs don't explain " + ", ".join(v["label"] for v in unexplained)
                     + ": that motion isn't caused by the recorded thruster commands (tether pull, waves or contact "
                       "are common causes). To calibrate it, record tests where the thrusters cause it: tip the "
                       "vehicle and let it settle (a free-decay test), and command deliberate roll/pitch moves, "
                       "with the tether slack. Until then, treat the simulator's values for those motions as "
                       "reference values, not measurements.")
    if measured:
        steps.append("Your free-decay tests measured " + ", ".join(v["label"] for v in measured)
                     + " (damping and righting), and these logs still aren't predicted: the roll/pitch the model gets "
                       "from the thrusters is wrong. Record deliberate roll/pitch commands with the tether slack: those "
                       "are what calibrate the thruster-driven part. Until then, compare this calibration with one "
                       "made without --free-decay before simulating attitude-sensitive tasks.")
    if weak:
        steps.append("Record more manoeuvres that exercise " + ", ".join(v["label"] for v in weak)
                     + ": short, sharp, varied commands on those axes, in several separate sessions.")
    if pinned:
        steps.append(f"{len(pinned)} parameter(s) stopped at the edge of their allowed range. Review those limits in "
                     "vehicle_spec.yaml with someone who knows the vehicle; if the physics allows, widen them and "
                     "re-run. Until then treat those values as 'at least this large', not as measurements.")
    if implausible:
        steps.append("Fix physically implausible values by holding them at independently measured values "
                     "(CAD model, pendulum test) and re-calibrating the rest.")
    if ident and (ident.get("correlated_pairs") or ident.get("weak_combinations")):
        steps.append("Some parameters can trade off against each other (listed above). That's harmless for simulation, "
                     "but don't report those individual values as vehicle properties. Manoeuvres that move the "
                     "axes they belong to in different ways (different speeds, and turning while moving) are what "
                     "separate them.")
    steps.append("Validate on hardware: run the same commands on the real vehicle and in the calibrated simulator "
                 "and compare. Held-out logs are strong evidence; a live test is the real proof.")
    return steps


def water_note(metrics: Optional[dict], result_dir: str) -> Optional[str]:
    """One plain paragraph on the water current the calibration assumed, or None for still water."""
    obj = (metrics or {}).get("objective") or {}
    by_file, cur = obj.get("current_by_file"), obj.get("current_world_mps")
    if by_file:
        import glob
        speeds = [float(np.hypot(c[0], c[1])) for _, c in by_file]
        data_dir = ((metrics or {}).get("reproducibility") or {}).get("data_dir") or ""
        est = sorted(glob.glob(os.path.join(result_dir, "current_round*", "current_estimates.md")))
        est = est or [p for p in (os.path.join(data_dir, "current_estimates.md"),) if data_dir and os.path.exists(p)]
        return (f"Each of {len(by_file)} recordings was calibrated with its own water current "
                f"({100 * min(speeds):.0f}-{100 * max(speeds):.0f} cm/s)"
                + (f"; how each was estimated and how reliable it is: `{est[-1]}`" if est else "")
                + ". A current that changed during a recording isn't modelled.")
    if cur and any(cur):
        return (f"The recordings were calibrated in one known water current of "
                f"{100 * float(np.hypot(cur[0], cur[1])):.0f} cm/s (vertical {100 * float(cur[2]):+.0f} cm/s).")
    return None


UNSEEN_SESSIONS = "recording sessions the calibration never saw"
UNSEEN_WITHIN = "recordings the calibration never saw (from the same session, not another day)"   # one session, #93


def skill_sentence(verdicts, unseen: str = UNSEEN_SESSIONS) -> str:
    """Per-axis skill on unseen sessions, in words: never one average over the axes, which a badly wrong default
    on one or two axes would dominate (PROJECT_STATUS.md #50, #63)."""
    rated = [v for v in verdicts if v.get("skill") is not None and not v.get("excluded")]
    if not rated:
        return ("Each verdict below compares it with the default model; re-run with --data-dir to also compare "
                "it with simply assuming no motion.")
    useful = sorted((v for v in rated if v["skill"] > 0), key=lambda v: -v["skill"])
    if not useful:
        return (f"On {unseen}, it predicts no axis better than simply assuming "
                "the vehicle isn't moving.")
    parts = ", ".join(f"{v['label']} {v['skill']:+.2f}" for v in useful)
    rest = len(rated) - len(useful)
    return (f"On {unseen}, it predicts {len(useful)} of {len(rated)} axes better "
            f"than simply assuming no motion (skill, where 1 is perfect: {parts})"
            + (f"; on the other {rest} it does no better." if rest else ".")
            + " Beating the default model alone can flatter a poor default, so each verdict below checks both.")


def load_validation(result_dir: str) -> Optional[dict]:
    path = os.path.join(result_dir, VALIDATION_DIR, "validation.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def free_decay_note(metrics: Optional[dict]) -> Optional[str]:
    """One plain paragraph on roll/pitch values held from free-decay tank tests, or None."""
    fd = ((metrics or {}).get("objective") or {}).get("free_decay")
    if not fd:
        return None
    held = fd.get("held") or {}
    words = ", ".join(f"{PLAIN.get(n, n)} (`{n}` = {v:.4g})" for n, v in held.items())
    text = (f"These values were measured in free-decay tank tests (`{fd.get('file')}`) and held fixed, not fitted "
            f"from these logs: {words}.")
    if fd.get("notes"):
        text += " " + " ".join(n + "." for n in fd["notes"])
    return text


def diverged_note(metrics: Optional[dict]) -> Optional[str]:
    """A warning when the calibrated simulation blew up on some held-out recordings, or None."""
    dv = ((metrics or {}).get("skill_vs_zero") or {}).get("diverged_heldout") or {}
    full, short, runs = dv.get("full_replay", 0), dv.get("short_windows", 0), dv.get("runs")
    if not (full or short):
        return None
    where = []
    if full:
        where.append(f"{full} of {runs} unseen recordings replayed whole")
    if short:
        where.append(f"{short} of {runs} unseen recordings replayed in a few seconds at a time")
    return ("**Unstable:** with these values the simulated vehicle diverges (its motion blows up) on "
            + " and on ".join(where) + ". The skill numbers below cover only the recordings where it stayed "
            "stable. Don't simulate with this calibration: re-run it, and if values were held fixed "
            "(`--free-decay`, `--exclude-axes`), compare with a calibration that doesn't hold them.")


VALIDATION_DIR = "validation"


def _m(s, unit="", digits=2):
    return "n/a" if s is None or s.get("median") is None else f"{s['median']:.{digits}f}{unit}"


def validation_section(v: dict) -> list:
    """The report's section from validation.py's validation.json: position drift, orientation with the model
    turning on its own, the 1 s response, command steps, and the plots (PROJECT_STATUS.md #65)."""
    L = ["## How the simulation compares with the real vehicle\n",
         f"On the {v['heldout_recordings']} recordings the calibration never saw"
         + (f" ({', '.join(v['heldout_dates'])})" if v.get("heldout_dates") else "")
         + f". The numbers are in `{VALIDATION_DIR}/validation.json`.\n"]
    plots = set(v.get("plots") or [])
    tr = v["trajectory"]
    against = tr.get("against")
    if against and not against.startswith("the logged velocity"):   # the sensor suite (#96): real fixes
        L.append(f"**Position.** How far the simulated vehicle's displacement over each window is from what "
                 f"{against} measured, starting each window from the calibration's estimate of the real state. This "
                 f"is checked against an independent measurement, not against a velocity. Median over the windows "
                 f"(90% of windows under the second number):\n")
    else:
        L.append("**Position.** How far the simulated vehicle ends up from where its logged velocity says it went, "
                 "starting each window from the real state. Median over the windows (90% of windows under the second "
                 "number):\n")
    L += ["| After | Calibrated | Default physics | Assuming it doesn't move |", "|---|---|---|---|"]
    for i, H in enumerate(tr["horizons_s"]):
        if H not in (10, 30, 60) or tr["calibrated"][i]["median"] is None:
            continue
        c = tr["calibrated"][i]
        L.append(f"| {H} s | {c['median']:.2f} m ({c['p90']:.2f} m) | {_m(tr['default'][i], ' m')} | "
                 f"{_m(tr['no_motion'][i], ' m')} |")
    L.append("")
    for f, alt in (("drift.png", "Position drift against horizon"), ("trajectory_typical.png", "Path")):
        if f in plots:
            L.append(f"![{alt}]({VALIDATION_DIR}/{f})\n")
    o = v["orientation"]
    last = len(o["report_s"]) - 1
    L.append(f"**Orientation, with the model turning and tilting on its own** (the fit itself is scored with the "
             f"logged orientation fed in). Median error after {' and '.join(f'{s:g} s' for s in o['report_s'])}, "
             f"restarting from the real state every {o['window_s']:g} s:\n")
    L += ["| Angle | Calibrated | Default physics | Holding the starting attitude |", "|---|---|---|---|"]
    worse, better = [], []
    for name in ("roll", "pitch", "yaw"):
        cells = [" / ".join(_m(x, "°", 1) for x in o[k][name]) for k in ("calibrated", "default", "hold_start")]
        L.append(f"| {name.capitalize()} | " + " | ".join(cells) + " |")
        c, h = o["calibrated"][name][last]["median"], o["hold_start"][name][last]["median"]
        if c is not None and h is not None:
            (better if c < h else worse).append(name)
    L.append("")
    if better:
        L.append(f"- **{', '.join(better).capitalize()}:** the simulation tracks the real vehicle better than holding "
                 f"the starting attitude.")
    if worse:
        L.append(f"- **{', '.join(worse).capitalize()}:** worse than simply holding the starting attitude. Don't trust "
                 f"the simulated {' and '.join(worse)} on their own (LIMITS.md).")
    L.append("")
    if "orientation_typical.png" in plots:
        L.append(f"![Orientation]({VALIDATION_DIR}/orientation_typical.png)\n")
    re = v["response"]
    L.append(f"**Response over {re['horizon_s']:g} s.** The velocity change over the next second (that is, the "
             f"average acceleration), predicted against measured, from {re['windows']} moments. Skill compares "
             f"with assuming the velocity stays the same (1 = perfect, 0 = no better):\n")
    L += ["| Motion | Calibrated error | Default physics | Velocity stays the same | Skill |", "|---|---|---|---|---|"]
    axes = ["u", "v", "w", "p", "q", "r"]
    for j, ax in enumerate(axes):
        L.append(f"| {LABELS.get(ax, ax)} | {re['calibrated']['rmse'][j]:.3f} | {re['default']['rmse'][j]:.3f} | "
                 f"{re['same_velocity_rmse'][j]:.3f} | {re['calibrated']['skill_vs_same_velocity'][j]:+.2f} |")
    L.append("\nErrors in m/s (surge, sway, heave) and rad/s (roll, pitch, turning).\n")
    st = {ax: x for ax, x in (v.get("steps") or {}).items() if x.get("steps")}
    if st:
        L.append("**Command steps.** When a force or torque jumps and then holds: how long the velocity takes to make "
                 "63% of its change, and how big the change is, model against real (medians):\n")
        L += ["| Motion | Steps | Real vehicle | Calibrated | Calibrated size / real |", "|---|---|---|---|---|"]
        fast = 0
        for ax, x in st.items():
            rr, cr, sz = x.get("real_rise_s"), x.get("calibrated_rise_s"), x.get("calibrated_size_ratio")
            L.append(f"| {LABELS.get(ax, ax)} | {x['steps']} | {'n/a' if rr is None else f'{rr:.1f} s'} | "
                     f"{'n/a' if cr is None else f'{cr:.1f} s'} | {'n/a' if sz is None else f'{sz:.2f}'} |")
            if rr and cr is not None and cr <= rr / 1.5:
                fast += 1
        L.append("")
        if 2 * fast > len(st):
            L.append(f"- **The simulated vehicle responds faster than the real one:** at least 1.5 times as fast on "
                     f"{fast} of {len(st)} axes. Thruster spin-up and command lag aren't modelled (LIMITS.md); a "
                     f"controller tuned in simulation may be too aggressive on the real vehicle.\n")
        if "steps.png" in plots:
            L.append(f"![Command steps]({VALIDATION_DIR}/steps.png)\n")
    vel = [f for f in ("velocity_typical.png", "velocity_best.png", "velocity_worst.png") if f in plots]
    if vel:
        L.append(f"**Velocity over whole recordings**, as the calibration is scored: "
                 + ", ".join(f"[{f.split('_')[1].split('.')[0]}]({VALIDATION_DIR}/{f})" for f in vel) + ".\n")
        L.append(f"![Velocity, typical recording]({VALIDATION_DIR}/{vel[0]})\n")
    div = {k: x for k, x in (("position", tr.get("diverged", {}).get("calibrated")),
                             ("orientation", o.get("diverged", {}).get("calibrated")),
                             ("response", re.get("diverged", {}).get("calibrated")))}
    if any((sum(x) if isinstance(x, list) else x) for x in div.values() if x):
        L.append("- **Some windows diverged** (the simulated motion blew up) and are left out: "
                 + ", ".join(f"{k} {sum(x) if isinstance(x, list) else x}" for k, x in div.items() if x) + ".\n")
    return L


def sensors_section(metrics: Optional[dict]) -> list:
    """What the vehicle's sensors could calibrate (#96): which sensors, each axis and what measured it, what was held
    and why, and each sensor's own held-out skill."""
    ss = (metrics or {}).get("sensor_suite")
    if not ss:
        return []
    L = ["## What your sensors could calibrate\n"]
    names = sorted({lab.rsplit(".", 1)[0] for lab in ss.get("channels", [])})
    L.append("Calibrated from what the vehicle measured: " + ", ".join(names) + ". No velocity was assumed: each "
             "sensor is compared with what the simulated vehicle would have read (SENSORS.md).\n")
    for line in ss.get("plain", []):
        L.append(f"- {line}")
    L.append("- Roll, pitch, yaw: calibrated from the gyro.")
    held = ss.get("held_unobserved") or {}
    if held:
        L.append("\nHeld at your vehicle description's values, because no sensor sees their motion: "
                 + ", ".join(f"{PLAIN.get(k, k)} (`{k}`)" for k in held) + ".")
    fit = (ss.get("fit") or {}).get("skill_heldout") or {}
    if fit:
        L.append("\nEach sensor's own held-out skill (1 - error / error of predicting no motion; 1 = perfect, 0 = no "
                 "better than assuming the vehicle didn't move):\n")
        L.append("| Sensor axis | Skill | Default physics' skill |")
        L.append("|---|---|---|")
        dflt = (ss.get("fit") or {}).get("skill_heldout_default") or {}
        for lab, v in fit.items():
            d = dflt.get(lab)
            L.append(f"| {lab} | {'n/a' if v is None else f'{v:+.2f}'} | {'n/a' if d is None else f'{d:+.2f}'} |")
    sm_ = ss.get("smoother") or {}
    if sm_.get("fallbacks"):
        L.append("\n> ⚠️ The starting-velocity estimate fell back to the DVL (or rest) on: "
                 + "; ".join(sm_["fallbacks"]))
    L.append("")
    return L


def write_report(path, name, verdicts, gen_gap, pinned, implausible, ident, steps, converged, water=None,
                 free_decay=None, diverged=None, validation=None, vehicle_note=None, seed=None, restarts=None,
                 whole_fit=None, current_rounds=False, split_note=None, fit_note=None, sensors=None):
    L = [f"# Calibration report: {name}\n", f"{datetime.now():%Y-%m-%d}\n"]
    if fit_note:     # the fit failed, or was retried from more starting points (#93)
        L.append(fit_note + "\n")
    if split_note:   # one session: an easier check than an unseen day (#93)
        L.append("> ⚠️ " + split_note + "\n")
    if diverged:
        L.append("> ⚠️ " + diverged + "\n")
    if verdicts:
        good = [v for v in verdicts if v["verdict"] == GOOD]
        L.append("## Bottom line\n")
        L.append(f"It is trustworthy for **{len(good)} of 6** motion axes"
                 + (f": {', '.join(v['label'] for v in good)}." if good else ".")
                 + " " + skill_sentence(verdicts, UNSEEN_WITHIN if split_note else UNSEEN_SESSIONS) + "\n")
        L.append("## Can I trust it for this motion?\n")
        L.append("| Motion | Verdict | Why |")
        L.append("|---|---|---|")
        icon = {GOOD: "✅", CAUTION: "⚠️", BAD: "❌"}
        for v in verdicts:
            L.append(f"| {v['label']} | {icon[v['verdict']]} {v['verdict']} | {v['why']} |")
        L.append("")
    else:
        L.append("## Bottom line\n\nNo held-out accuracy numbers were available (re-run with --data-dir to compute "
                 "them). The checks below still apply.\n")
    if sensors:
        L += sensors
    if converged is False:
        L.append("> ⚠️ The optimizer stopped before fully converging. The result is usable but may improve with a "
                 "longer run.\n")
    if gen_gap is not None:
        L.append("## Did it learn the vehicle, or just memorise the training runs?\n")
        if gen_gap < 0.85:
            verdict = ("Can't tell from these sessions: the error on unseen sessions is lower than on the training "
                       "runs, which usually means the unseen sessions were easier (gentler, or recorded in different "
                       "conditions), not that the model learned more. Hold out sessions like the ones you'll simulate.")
        elif gen_gap <= 1.15:
            verdict = ("No sign of memorising: the error on unseen sessions is about the same as on the training "
                       "runs.")
        else:
            verdict = (f"Error on unseen sessions is {100 * (gen_gap - 1):.0f}% higher than on training runs, a sign "
                       f"of over-fitting to those particular recordings.")
        L.append(f"{verdict} (Unseen / training error, median over the six axes: {gen_gap:.2f}; about 1 is what "
                 f"you want.)\n")
    if validation:
        L += validation_section(validation)
    if water:
        L.append("## Water current\n")
        L.append(water + "\n")
    if free_decay:
        L.append("## Roll and pitch from tank tests\n")
        L.append(free_decay + "\n")
    L.append("## Be careful with\n")
    if vehicle_note:
        L.append("**The vehicle description isn't your vehicle's.** " + vehicle_note + "\n")
    if not (vehicle_note or pinned or implausible or (ident and (ident.get("correlated_pairs") or ident.get("weak_combinations")))):
        L.append("Nothing flagged.\n")
    if pinned:
        L.append(f"**Values stuck at a limit ({len(pinned)}).** The fit pushed these as far as allowed and stopped, "
                 "so the true value is probably beyond the limit:\n")
        for p in pinned:
            L.append(f"- {PLAIN.get(p['name'], p['name'])} (`{p['name']}` = {p['value']:.4g}, at its {p['bound']} "
                     f"limit of {p['bounds']})")
        L.append("")
    if implausible:
        L.append("**Physically implausible values.** These disagree strongly with independent measurements:\n")
        for p in implausible:
            L.append(f"- {PLAIN[p['name']]}: fitted {p['value']:.3g}, but {p['source']} gives {p['reference']:.3g} "
                     f"({p['ratio']:.0f}x). The fit is probably using this to compensate for something else.")
        L.append("")
    if ident and ident.get("correlated_pairs"):
        L.append("**Parameters your data can't tell apart.** In these pairs, raising one and lowering the other gives "
                 "almost the same motion, so the data pins down their combined effect but not each value:\n")
        for c in ident["correlated_pairs"]:
            L.append(f"- {PLAIN.get(c['param_a'], c['param_a'])} vs {PLAIN.get(c['param_b'], c['param_b'])} "
                     f"(correlation {c['correlation']:+.2f})")
        L.append("")
    if ident and ident.get("weak_combinations"):
        L.append("**Groups of values your data barely pins down.** Each group below can shift together with almost "
                 "no change in the predicted motion, so treat the individual values as uncertain even where they "
                 "look precise:\n")
        for w in ident["weak_combinations"]:
            L.append("- " + ", ".join(PLAIN.get(p, p) for p in w["parameters"])
                     + f" (constrained about {1 / max(w['relative_strength'], 1e-12):,.0f}x less than the "
                       f"best-constrained combination)")
        L.append("")
    L.append("## What to do next\n")
    for i, s in enumerate(steps, 1):
        L.append(f"{i}. {s}")
    L.append("")
    if whole_fit and whole_fit.get("starts"):
        done = [st for st in whole_fit["starts"] if st.get("final_cost") is not None]
        parts = [f"seed {st['seed']}: " + " -> ".join("?" if c is None else f"{c:.4g}" for c in st["round_costs"])
                 for st in done]
        finals = [st["final_cost"] for st in done]
        spread = (max(finals) - min(finals)) / max(min(finals), 1e-12) if finals else 0.0
        chosen = next((st for st in done if st["start"] == whole_fit.get("chosen")), None)
        L.append(f"The whole fit (every current round) ran from {len(whole_fit['starts'])} starting points and kept "
                 f"the lowest final training cost"
                 + (f" (seed {chosen['seed']})" if chosen else "") + ". Training cost by round: " + "; ".join(parts)
                 + ". " + ("They ended within 1% of each other: these starts agree on the result." if spread < 0.01 else
                           "They ended in different places: this calibration depends on where the fit starts. Each "
                           "start's full result is in its startN folder.") + "\n")
    elif restarts and len(restarts) > 1:
        costs = ", ".join(f"{c:.4g}" for c in restarts)
        spread = (max(restarts) - min(restarts)) / max(min(restarts), 1e-12)
        L.append(f"The fit started from {len(restarts)} points (seed {seed}) and kept the lowest training cost. Their "
                 f"costs: {costs}. "
                 + (("They ended within 1% of each other after the first round. With current estimation, later rounds "
                     "can still go two ways from starts this close (URI's B8: 4.760 and 4.781 after round 1 ended at "
                     "4.17 and 3.86); `--restart-whole-fit` runs every round from each start and compares the final "
                     "results." if current_rounds else
                     "They ended within 1% of each other: these starts agree.")
                    if spread < 0.01 else
                    "They ended in different places: this calibration depends on where the fit starts. More starts "
                    "(--restarts) make it more likely the best one is found.") + "\n")
    elif seed is not None:
        L.append(f"The fit started from a randomly perturbed point (seed {seed}): `strathe calibrate ... --seed {seed}` "
                 f"repeats it exactly. On a hard calibration another seed can land elsewhere; `--restarts 3` starts "
                 f"from three points and keeps the lowest training cost.\n")
    if verdicts:
        L.append("## Technical appendix\n")
        L.append("Held-out RMSE per axis (m/s or rad/s; with no DVL, the error of the sensors that measured it, in "
                 "their units): predicting no motion, default physics, calibrated physics. "
                 "Skill = 1 - calibrated / no-motion.\n")
        L.append("| Axis | Unit | No motion | Default | Calibrated | Improvement vs default | Skill |")
        L.append("|---|---|---|---|---|---|---|")
        for v in verdicts:
            if not np.isfinite(v["calibrated_rmse"]):
                L.append(f"| {v['axis']} | - | n/a | n/a | n/a | n/a | n/a |")
                continue
            z = "n/a" if v["zero_rmse"] is None else f"{v['zero_rmse']:.4f}"
            sk = "n/a" if v["skill"] is None else f"{v['skill']:+.2f}"
            unit = v.get("unit") or ("m/s" if v["axis"] in "uvw" else "rad/s")
            L.append(f"| {v['axis']} | {unit} | {z} | {v['default_rmse']:.4f} | {v['calibrated_rmse']:.4f} | "
                     f"{v['improvement_pct']:+.1f}% | {sk} |")
        if ident:
            L.append(f"\nIdentifiability: Jacobian condition number {ident.get('condition_number', float('nan')):.3g} "
                     f"from {ident.get('n_train_runs_used', '?')} training runs"
                     + (f", {len(ident['free_parameters'])} free parameters, same objective as the fit"
                        if ident.get("free_parameters") else "") + ".")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def build_report(result_dir: str, spec_path: str = "vehicle_spec.yaml", identifiability: Optional[str] = None,
                 data_dir: Optional[str] = None, heldout_dates=None) -> dict:
    spec = load_vehicle_spec(spec_path)
    with open(os.path.join(result_dir, "calibrated_parameters.yaml"), encoding="utf-8") as f:
        params = CalibrationParameters(**yaml.safe_load(f)["parameters"])
    mpath = os.path.join(result_dir, "metrics.json")
    metrics = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else None
    if identifiability is None and os.path.exists(os.path.join(result_dir, IDENT_FILENAME)):
        identifiability = os.path.join(result_dir, IDENT_FILENAME)  # written by `strathe calibrate`
    ident = json.load(open(identifiability, encoding="utf-8")) if identifiability else None
    if heldout_dates is None:  # the dates the calibration itself held out
        heldout_dates = ((metrics or {}).get("dataset") or {}).get("heldout_dates")  # None: the default split
    axes = per_axis_heldout(metrics, params, spec, data_dir, list(heldout_dates) if heldout_dates else None)
    fd_held = set(((metrics or {}).get("objective", {}).get("free_decay") or {}).get("held") or {})
    pinned = [p for p in pinned_parameters(params, spec) if p["name"] not in fd_held]  # measured, not fitted
    implausible = implausible_parameters(params, spec)
    measured_axes = tuple(ax for ax, ps in (("p", ("Kp", "Kpp")), ("q", ("Mq", "Mqq"))) if fd_held & set(ps))
    names_flagged = {p["name"] for p in pinned}, {p["name"] for p in implausible}
    excluded = tuple((metrics or {}).get("objective", {}).get("excluded_axes", ()))
    sensor_axes = {k: v for k, v in sensor_axis_numbers(metrics).items()
                   if axes is None or not np.isfinite(axes[1][AXES_INDEX[k]])}
    verdicts = (axis_verdicts(axes[0], axes[1], *names_flagged, zero_rmse=axes[2], excluded_axes=excluded,
                              measured_axes=measured_axes, sensor_axes=sensor_axes)
                if axes else [])
    gen_gap = None
    if metrics and "rmse" in metrics and "calibrated_train" in metrics["rmse"]:
        # median of the per-axis ratios: one overall error mixes m/s with rad/s and lets one axis dominate
        pd_ = metrics.get("per_dof") or {}
        if pd_.get("calibrated_heldout") and pd_.get("calibrated_train"):
            with np.errstate(invalid="ignore", divide="ignore"):
                ratios = np.array(pd_["calibrated_heldout"], dtype=float) / np.array(pd_["calibrated_train"], dtype=float)
            ratios = ratios[np.isfinite(ratios)]   # u, v, w unmeasured with no DVL (#96)
            gen_gap = float(np.median(ratios)) if len(ratios) else None
        else:
            gen_gap = metrics["rmse"]["calibrated_heldout"] / metrics["rmse"]["calibrated_train"]
    converged = None
    if metrics:
        converged = metrics.get("optimizer", {}).get("converged", metrics.get("converged"))
    vehicle_note = vehicle_description_note(spec, data_dir)
    steps = next_steps(verdicts, pinned, implausible, ident, vehicle_note)
    out = os.path.join(result_dir, "calibration_report.md")
    write_report(out, os.path.basename(os.path.abspath(result_dir)), verdicts, gen_gap, pinned, implausible,
                 ident, steps, converged, water_note(metrics, result_dir), free_decay_note(metrics),
                 diverged_note(metrics), load_validation(result_dir), vehicle_note,
                 ((metrics or {}).get("reproducibility") or {}).get("seed"), restart_costs(result_dir, metrics),
                 whole_fit_starts(result_dir), os.path.isdir(os.path.join(result_dir, "current_round1")),
                 ((metrics or {}).get("dataset") or {}).get("split_warning"), fit_note(metrics),
                 sensors_section(metrics))
    return {"report": out, "verdicts": verdicts, "pinned": pinned, "implausible": implausible,
            "generalization_ratio": gen_gap, "next_steps": steps, "vehicle_note": vehicle_note}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", required=True, help="calibration output folder")
    ap.add_argument("--spec", default="vehicle_spec.yaml")
    ap.add_argument("--identifiability", help="JSON from automated_identifiability_report.py")
    ap.add_argument("--data-dir", help="logs, to compute held-out per-axis accuracy (and the no-motion baseline) "
                                       "if metrics.json lacks them")
    ap.add_argument("--heldout-dates", "--heldout-recordings", dest="heldout_dates", nargs="+", default=None,
                    help="default: the dates the calibration held out (metrics.json)")
    args = ap.parse_args(argv)
    r = build_report(args.result, args.spec, args.identifiability, args.data_dir, args.heldout_dates)
    print(f"Wrote {r['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
