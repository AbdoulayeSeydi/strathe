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
}
# which parameters mainly drive each axis (for "why" and "what next")
AXIS_PARAMS = {"u": ["Xu", "Xuu", "Xu_dot"], "v": ["Yv", "Yvv", "Yv_dot"], "w": ["Zw", "Zww", "Zw_dot", "net_buoyancy_z"],
               "p": ["Ieff_roll", "Kp", "Kpp", "cg_cb_offset_z"], "q": ["Ieff_pitch", "Mq", "Mqq", "cg_cb_offset_z"],
               "r": ["Ieff_yaw", "Nr", "Nrr"]}
# independent physical reference values where we have them (see PROJECT_STATUS.md #28)
PHYSICAL_REFERENCE = {"Ieff_roll": (0.26, "BlueROV2 Heavy rigid-body Ixx, public Gazebo model"),
                      "Ieff_pitch": (0.23, "BlueROV2 Heavy rigid-body Iyy, public Gazebo model")}
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
        from objective import DatasetConfig, load_run, rmse_by_dof, zero_baseline_rmse
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
            c = np.mean([rmse_by_dof(params, r, spec.vehicle) for r in runs], axis=0)
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


def implausible_parameters(params) -> list:
    out = []
    for n, (ref, src) in PHYSICAL_REFERENCE.items():
        val = getattr(params, n)
        ratio = val / ref
        if ratio > MAX_PLAUSIBLE_RATIO or ratio < 1 / MAX_PLAUSIBLE_RATIO:
            out.append({"name": n, "value": val, "reference": ref, "source": src, "ratio": ratio})
    return out


def axis_verdicts(default_rmse, calib_rmse, pinned_names, implausible_names, zero_rmse=None,
                  excluded_axes=(), measured_axes=()) -> list:
    """One verdict per axis. The main test is skill against predicting "no
    motion" (see module docstring); improvement over the default model is
    reported too but can't earn a "Trust it" on its own."""
    verdicts = []
    for i, (k, label) in enumerate(AXES):
        imp = 100 * (default_rmse[i] - calib_rmse[i]) / default_rmse[i]
        skill = None if zero_rmse is None else float(1 - calib_rmse[i] / zero_rmse[i])
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
        verdicts.append({"axis": k, "label": label, "improvement_pct": imp, "skill": skill, "verdict": v, "why": why,
                         "excluded": k in excluded_axes, "measured": k in measured_axes,
                         "default_rmse": float(default_rmse[i]), "calibrated_rmse": float(calib_rmse[i]),
                         "zero_rmse": None if zero_rmse is None else float(zero_rmse[i])})
    return verdicts


def next_steps(verdicts, pinned, implausible, ident) -> list:
    steps = []
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


def skill_sentence(verdicts) -> str:
    """Per-axis skill on unseen sessions, in words: never one average over the axes, which a badly wrong default
    on one or two axes would dominate (PROJECT_STATUS.md #50, #63)."""
    rated = [v for v in verdicts if v.get("skill") is not None and not v.get("excluded")]
    if not rated:
        return ("Each verdict below compares it with the default model; re-run with --data-dir to also compare "
                "it with simply assuming no motion.")
    useful = sorted((v for v in rated if v["skill"] > 0), key=lambda v: -v["skill"])
    if not useful:
        return ("On recording sessions the calibration never saw, it predicts no axis better than simply assuming "
                "the vehicle isn't moving.")
    parts = ", ".join(f"{v['label']} {v['skill']:+.2f}" for v in useful)
    rest = len(rated) - len(useful)
    return (f"On recording sessions the calibration never saw, it predicts {len(useful)} of {len(rated)} axes better "
            f"than simply assuming no motion (skill, where 1 is perfect: {parts})"
            + (f"; on the other {rest} it does no better." if rest else ".")
            + " Beating the default model alone can flatter a poor default, so each verdict below checks both.")


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


def write_report(path, name, verdicts, gen_gap, pinned, implausible, ident, steps, converged, water=None,
                 free_decay=None):
    L = [f"# Calibration report: {name}\n", f"{datetime.now():%Y-%m-%d}\n"]
    if verdicts:
        good = [v for v in verdicts if v["verdict"] == GOOD]
        L.append("## Bottom line\n")
        L.append(f"It is trustworthy for **{len(good)} of 6** motion axes"
                 + (f": {', '.join(v['label'] for v in good)}." if good else ".")
                 + " " + skill_sentence(verdicts) + "\n")
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
    if water:
        L.append("## Water current\n")
        L.append(water + "\n")
    if free_decay:
        L.append("## Roll and pitch from tank tests\n")
        L.append(free_decay + "\n")
    L.append("## Be careful with\n")
    if not (pinned or implausible or (ident and (ident.get("correlated_pairs") or ident.get("weak_combinations")))):
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
    if verdicts:
        L.append("## Technical appendix\n")
        L.append("Held-out RMSE per axis (m/s or rad/s): predicting no motion, default physics, calibrated physics. "
                 "Skill = 1 - calibrated / no-motion.\n")
        L.append("| Axis | No motion | Default | Calibrated | Improvement vs default | Skill |")
        L.append("|---|---|---|---|---|---|")
        for v in verdicts:
            z = "n/a" if v["zero_rmse"] is None else f"{v['zero_rmse']:.4f}"
            sk = "n/a" if v["skill"] is None else f"{v['skill']:+.2f}"
            L.append(f"| {v['axis']} | {z} | {v['default_rmse']:.4f} | {v['calibrated_rmse']:.4f} | "
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
    implausible = implausible_parameters(params)
    measured_axes = tuple(ax for ax, ps in (("p", ("Kp", "Kpp")), ("q", ("Mq", "Mqq"))) if fd_held & set(ps))
    names_flagged = {p["name"] for p in pinned}, {p["name"] for p in implausible}
    excluded = tuple((metrics or {}).get("objective", {}).get("excluded_axes", ()))
    verdicts = (axis_verdicts(axes[0], axes[1], *names_flagged, zero_rmse=axes[2], excluded_axes=excluded,
                              measured_axes=measured_axes)
                if axes else [])
    gen_gap = None
    if metrics and "rmse" in metrics and "calibrated_train" in metrics["rmse"]:
        # median of the per-axis ratios: one overall error mixes m/s with rad/s and lets one axis dominate
        pd_ = metrics.get("per_dof") or {}
        if pd_.get("calibrated_heldout") and pd_.get("calibrated_train"):
            gen_gap = float(np.median(np.array(pd_["calibrated_heldout"]) / np.array(pd_["calibrated_train"])))
        else:
            gen_gap = metrics["rmse"]["calibrated_heldout"] / metrics["rmse"]["calibrated_train"]
    converged = None
    if metrics:
        converged = metrics.get("optimizer", {}).get("converged", metrics.get("converged"))
    steps = next_steps(verdicts, pinned, implausible, ident)
    out = os.path.join(result_dir, "calibration_report.md")
    write_report(out, os.path.basename(os.path.abspath(result_dir)), verdicts, gen_gap, pinned, implausible,
                 ident, steps, converged, water_note(metrics, result_dir), free_decay_note(metrics))
    return {"report": out, "verdicts": verdicts, "pinned": pinned, "implausible": implausible,
            "generalization_ratio": gen_gap, "next_steps": steps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", required=True, help="calibration output folder")
    ap.add_argument("--spec", default="vehicle_spec.yaml")
    ap.add_argument("--identifiability", help="JSON from automated_identifiability_report.py")
    ap.add_argument("--data-dir", help="logs, to compute held-out per-axis accuracy (and the no-motion baseline) "
                                       "if metrics.json lacks them")
    ap.add_argument("--heldout-dates", nargs="+", default=None,
                    help="default: the dates the calibration held out (metrics.json)")
    args = ap.parse_args(argv)
    r = build_report(args.result, args.spec, args.identifiability, args.data_dir, args.heldout_dates)
    print(f"Wrote {r['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
