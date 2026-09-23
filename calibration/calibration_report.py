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

GOOD, CAUTION, BAD = "Trust it", "Use with care", "Don't rely on it"


def per_axis_heldout(metrics: Optional[dict], params, spec, data_dir: Optional[str], heldout_dates) -> Optional[tuple]:
    """(default_rmse[6], calibrated_rmse[6]) on held-out sessions."""
    if metrics and "per_dof" in metrics and "calibrated_heldout" in metrics["per_dof"]:
        pd_ = metrics["per_dof"]
        return np.array(pd_["default_heldout"]), np.array(pd_["calibrated_heldout"])
    if data_dir is None:
        return None
    from data_split import discover_csv_files, split_by_date
    from objective import DatasetConfig, load_run, rmse_by_dof
    files = discover_csv_files(data_dir)[0]
    split = split_by_date(files, heldout_dates=heldout_dates)
    runs = [load_run(f, DatasetConfig(10.0)) for f in split.heldout_files]
    default = initial_parameters(spec)
    d = np.mean([rmse_by_dof(default, r, spec.vehicle) for r in runs], axis=0)
    c = np.mean([rmse_by_dof(params, r, spec.vehicle) for r in runs], axis=0)
    return d, c


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


def axis_verdicts(default_rmse, calib_rmse, pinned_names, implausible_names) -> list:
    verdicts = []
    for i, (k, label) in enumerate(AXES):
        imp = 100 * (default_rmse[i] - calib_rmse[i]) / default_rmse[i]
        involved = [p for p in AXIS_PARAMS[k] if p in pinned_names or p in implausible_names]
        if imp < 10:
            v, why = BAD, ("the calibrated model predicts this motion no better than the default model"
                           if imp >= 0 else "the calibrated model predicts this motion WORSE than the default model")
        elif imp < 30 or involved:
            v = CAUTION
            why = (f"it improves prediction by {imp:.0f}%, but relies on " + ", ".join(f"{PLAIN[p]}" for p in involved)
                   + (" which hit their allowed limits or look physically implausible" if involved else "")
                   if involved else f"it improves prediction by only {imp:.0f}%")
        else:
            v, why = GOOD, f"it predicts this motion {imp:.0f}% more accurately than the default model on unseen sessions"
        verdicts.append({"axis": k, "label": label, "improvement_pct": imp, "verdict": v, "why": why,
                         "default_rmse": float(default_rmse[i]), "calibrated_rmse": float(calib_rmse[i])})
    return verdicts


def next_steps(verdicts, pinned, implausible, ident) -> list:
    steps = []
    # more data helps axes the model predicts poorly; axes flagged only because a
    # parameter hit a limit need the limits reviewed instead (next step)
    weak = [v for v in verdicts if v["improvement_pct"] < 30]
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
    if ident and ident.get("correlated_pairs"):
        steps.append("Some parameters can trade off against each other (listed above). That's harmless for simulation, "
                     "but don't report those individual values as vehicle properties.")
    steps.append("Validate on hardware: run the same commands on the real vehicle and in the calibrated simulator "
                 "and compare. Held-out logs are strong evidence; a live test is the real proof.")
    return steps


def write_report(path, name, verdicts, gen_gap, pinned, implausible, ident, steps, converged):
    L = [f"# Calibration report: {name}\n", f"{datetime.now():%Y-%m-%d}\n"]
    if verdicts:
        good = [v for v in verdicts if v["verdict"] == GOOD]
        overall = np.mean([v["improvement_pct"] for v in verdicts])
        L.append("## Bottom line\n")
        L.append(f"On recording sessions the calibration never saw, the calibrated simulator predicts your vehicle's "
                 f"motion **{overall:.0f}% more accurately** than the default model, averaged over the six motion axes. "
                 f"It is trustworthy for **{len(good)} of 6** axes.\n")
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
        verdict = ("It generalises: accuracy on unseen sessions is as good as on the training runs."
                   if gen_gap < 1.15 else
                   f"Accuracy on unseen sessions is {100 * (gen_gap - 1):.0f}% worse than on training runs, a sign of "
                   f"over-fitting to those particular recordings.")
        L.append(f"{verdict} (held-out / training error ratio {gen_gap:.2f}; 1.0 is ideal.)\n")
    L.append("## Be careful with\n")
    if not (pinned or implausible or (ident and ident.get("correlated_pairs"))):
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
    L.append("## What to do next\n")
    for i, s in enumerate(steps, 1):
        L.append(f"{i}. {s}")
    L.append("")
    if verdicts:
        L.append("## Technical appendix\n")
        L.append("Held-out RMSE per axis (m/s or rad/s), default vs calibrated physics:\n")
        L.append("| Axis | Default | Calibrated | Improvement |")
        L.append("|---|---|---|---|")
        for v in verdicts:
            L.append(f"| {v['axis']} | {v['default_rmse']:.4f} | {v['calibrated_rmse']:.4f} | {v['improvement_pct']:+.1f}% |")
        if ident:
            L.append(f"\nIdentifiability: Jacobian condition number {ident.get('condition_number', float('nan')):.3g} "
                     f"from {ident.get('n_train_runs_used', '?')} training runs.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def build_report(result_dir: str, spec_path: str = "vehicle_spec.yaml", identifiability: Optional[str] = None,
                 data_dir: Optional[str] = None, heldout_dates=("16-02-2024", "19-02-2024")) -> dict:
    spec = load_vehicle_spec(spec_path)
    with open(os.path.join(result_dir, "calibrated_parameters.yaml"), encoding="utf-8") as f:
        params = CalibrationParameters(**yaml.safe_load(f)["parameters"])
    mpath = os.path.join(result_dir, "metrics.json")
    metrics = json.load(open(mpath, encoding="utf-8")) if os.path.exists(mpath) else None
    ident = json.load(open(identifiability, encoding="utf-8")) if identifiability else None
    axes = per_axis_heldout(metrics, params, spec, data_dir, list(heldout_dates))
    pinned = pinned_parameters(params, spec)
    implausible = implausible_parameters(params)
    names_flagged = {p["name"] for p in pinned}, {p["name"] for p in implausible}
    verdicts = axis_verdicts(*axes, *names_flagged) if axes else []
    gen_gap = None
    if metrics and "rmse" in metrics and "calibrated_train" in metrics["rmse"]:
        gen_gap = metrics["rmse"]["calibrated_heldout"] / metrics["rmse"]["calibrated_train"]
    converged = None
    if metrics:
        converged = metrics.get("optimizer", {}).get("converged", metrics.get("converged"))
    steps = next_steps(verdicts, pinned, implausible, ident)
    out = os.path.join(result_dir, "calibration_report.md")
    write_report(out, os.path.basename(os.path.abspath(result_dir)), verdicts, gen_gap, pinned, implausible,
                 ident, steps, converged)
    return {"report": out, "verdicts": verdicts, "pinned": pinned, "implausible": implausible,
            "generalization_ratio": gen_gap, "next_steps": steps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", required=True, help="calibration output folder")
    ap.add_argument("--spec", default="vehicle_spec.yaml")
    ap.add_argument("--identifiability", help="JSON from automated_identifiability_report.py")
    ap.add_argument("--data-dir", help="logs, to compute held-out per-axis accuracy if metrics.json lacks it")
    ap.add_argument("--heldout-dates", nargs="+", default=["16-02-2024", "19-02-2024"])
    args = ap.parse_args(argv)
    r = build_report(args.result, args.spec, args.identifiability, args.data_dir, args.heldout_dates)
    print(f"Wrote {r['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
