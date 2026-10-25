"""
stonefish_export.py -- put a Strathe calibration into a lab's own Stonefish vehicle model (`strathe export-stonefish`,
PROJECT_STATUS.md #74).

`fit-stonefish` tunes Stonefish's settings by replaying the logs in Stonefish (an hour or more). This instead converts
the values `strathe calibrate` already found into Stonefish's own settings on one part (usually the hull), in seconds,
then scores the original and the exported model on the recordings the calibration never saw:
  - mass: the part's <mass> moves so the vehicle weighs the calibration's mass;
  - net buoyancy: the buoyant part (--buoyancy-part, a box, cylinder or sphere) is resized;
  - drag: with the patch (STONEFISH_AXIS_DAMPING_PATCH, #80), Strathe's own per-axis linear + quadratic damping, all six
    axes, replacing the vehicle's mesh drag, at the logs' reference point: the same law, so nothing is converted.
    Without it, Stonefish's mesh drag is cubic in speed (0.5 rho Cd S |v|^3; measured, PROJECT_STATUS.md #74) plus a
    linear skin term (rho Cf T v), so Cd and Cf are fitted to Strathe's linear + quadratic drag over the speeds the
    logs actually cover, after what the other parts already contribute;
  - added mass per axis (needs calibration/stonefish/stonefish_added_mass.patch);
  - the centre of gravity relative to the centre of buoyancy (the part's <cg> moves);
  - the effective roll/pitch/yaw inertia (rigid + added): the part's added inertia, and its rigid <inertia> where the
    added inertia alone can't get low enough.
Each change is checked by reloading the vehicle in Stonefish. What Stonefish has no setting for (its rotational drag
comes from the same mesh coefficients; hull lift; thruster lag; the off-diagonal added mass) is listed in the report.

Outputs, in the output folder:
  <vehicle file>_strathe.scn   the lab's own vehicle file with only the tuned part (and buoyancy part) changed
  stonefish_export.yaml        Strathe's values, Stonefish's before and after, and the held-out scores
  stonefish_export_report.md   the same, readable; drift.png
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import yaml
from scipy.optimize import nnls

from stonefish_fit import (EVAL_STRIDE_S, EVAL_WINDOW_S, HORIZONS_S, PartSettings, Replay, Scenario,
                           StonefishFitError, _fmt, _load_runs, _plot, _table, baselines, find_tool, score,
                           strathe_free_run, window_rows, windows)

G = 9.81
AXES = "xyz"
INERTIA_FLOOR = 0.05  # the part's rigid inertia is never set below this fraction of Stonefish's own


@dataclass
class Export:
    settings: PartSettings
    rows: list = field(default_factory=list)   # [quantity, Strathe's value, Stonefish before, Stonefish after, unit]
    notes: list = field(default_factory=list)
    drag: dict = field(default_factory=dict)   # per axis: Cd, Cf, the speeds matched and the force at them


def _parts(info: dict) -> dict:
    return {p["name"].split("/")[-1]: p for p in info["parts"]}


def drag_coefficients(linear: float, quadratic: float, speeds: np.ndarray, rho: float, projected: float,
                      tangential: float, others: Callable) -> tuple:
    """(Cd, Cf) for one axis of the tuned part: Stonefish's 0.5 rho Cd S U^3 + rho Cf T U (plus what the other parts
    already give) matched to Strathe's |linear| U + |quadratic| U^2, least squares over the logged speeds, Cd, Cf >= 0."""
    U = np.asarray(speeds, dtype=float)
    target = abs(linear) * U + abs(quadratic) * U ** 2 - others(U)
    A = np.column_stack([0.5 * rho * max(projected, 1e-9) * U ** 3, rho * max(tangential, 1e-9) * U])
    (cd, cf), _ = nnls(A, np.maximum(target, 0.0))
    return float(cd), float(cf)


def axis_speeds(runs: list, axis: int, n_max: int = 4000) -> np.ndarray:
    """The speeds along one body axis the logs cover (what the drag is matched over); a short grid if they barely move."""
    v = np.abs(np.concatenate([r.real_nu[:, axis] for r in runs])) if runs else np.zeros(0)
    v = v[v > 1e-3]
    if len(v) < 20 or np.percentile(v, 95) < 0.05:
        return np.linspace(0.01, 0.2, 20)
    if len(v) > n_max:
        v = np.sort(v)[np.linspace(0, len(v) - 1, n_max).astype(int)]
    return v


def export_settings(params, mass: float, measure: Callable, part: str, buoyancy_part: Optional[str], rho: float,
                    speeds: list, reference_point=(0.0, 0.0, 0.0)) -> Export:
    """Stonefish settings for the tuned part from a Strathe calibration. `measure(settings)` reloads the vehicle in
    Stonefish with those settings and returns the replay tool's --info (None = as the scenario has it)."""
    i0 = measure(None)
    parts, base = _parts(i0), i0["base"]
    if part not in parts:
        raise StonefishFitError(f"Stonefish doesn't report a part '{part}' (it has: {', '.join(parts) or 'none'})")
    hull = parts[part]
    for k in ("projected_area", "tangential_area", "inertia"):
        if k not in hull:
            raise StonefishFitError("the replay tool is older than export-stonefish: rebuild it "
                                    "(cmake --build calibration/stonefish/build)")
    patched = bool(i0.get("patched"))
    exact_damping = bool(i0.get("axis_damping"))
    e = Export(PartSettings())
    s = e.settings

    # Mass: the part takes the difference.
    s.mass = hull["mass"] + (mass - base["mass"])
    if s.mass <= 0.05 * hull["mass"]:
        raise StonefishFitError(f"the calibration's mass ({mass:g} kg) would leave part '{part}' with {s.mass:.3g} kg: "
                                f"is --mass the vehicle's mass in air?")

    # Net buoyancy, in the water the calibration was trained in.
    if buoyancy_part:
        if buoyancy_part not in parts:
            raise StonefishFitError(f"no part '{buoyancy_part}' (--buoyancy-part) in Stonefish's vehicle")
        bp = parts[buoyancy_part]
        need = (mass * G + params.net_buoyancy_z) / (rho * G)
        new_v = bp["volume"] + need - base["volume"]
        if new_v <= 0.05 * bp["volume"]:
            e.notes.append(f"Net buoyancy not carried: {params.net_buoyancy_z:+.1f} N would need part '{buoyancy_part}' "
                           f"to shrink to {new_v:.4g} m³. Check the calibration's net buoyancy, or use a bigger part.")
        else:
            s.buoyancy_scale = new_v / bp["volume"]
    else:
        e.notes.append("Net buoyancy not carried: give --buoyancy-part (a buoyant box, cylinder or sphere part).")

    # Drag per axis, after what the other external parts give.
    others_ext = [p for n, p in parts.items() if p.get("external") and n != part]
    lin, quad = [params.Xu, params.Yv, params.Zw], [params.Xuu, params.Yvv, params.Zww]
    if exact_damping:
        # Strathe's damping as it is: the patch applies it at the reference point instead of every part's mesh drag
        s.axis_damping = {"linear": lin, "linear_rot": [params.Kp, params.Mq, params.Nr], "quadratic": quad,
                          "quadratic_rot": [params.Kpp, params.Mqq, params.Nrr],
                          "point": [float(x) for x in reference_point]}
    cd, cf = [], []
    for i in range(3 if not exact_damping else 0):
        def others(U, i=i):
            return sum(0.5 * rho * p["quadratic_drag"][i] * p["projected_area"][i] * U ** 3
                       + rho * p["viscous_drag"][i] * p["tangential_area"][i] * U for p in others_ext)
        c_d, c_f = drag_coefficients(lin[i], quad[i], speeds[i], rho, hull["projected_area"][i],
                                     hull["tangential_area"][i], others)
        cd.append(c_d), cf.append(c_f)
        U = np.percentile(speeds[i], [50, 95])
        strathe = abs(lin[i]) * U + abs(quad[i]) * U ** 2
        sf = (0.5 * rho * c_d * hull["projected_area"][i] * U ** 3 + rho * c_f * hull["tangential_area"][i] * U
              + others(U))
        e.drag[AXES[i]] = {"quadratic_drag": c_d, "viscous_drag": c_f, "speeds_m_s": U.round(3).tolist(),
                           "strathe_force_n": strathe.round(2).tolist(), "stonefish_force_n": sf.round(2).tolist()}
    if not exact_damping:
        s.quadratic_drag, s.viscous_drag = cd, cf

    # Added mass per axis (the compound adds up its external parts' values).
    am = [params.Xu_dot, params.Yv_dot, params.Zw_dot]
    if patched:
        other_am = np.sum([p["added_mass"] for p in others_ext], axis=0) if others_ext else np.zeros(3)
        s.added_mass = [max(a - o, 0.01) for a, o in zip(am, other_am)]
        s.added_inertia = list(hull["added_inertia"])
        s.anisotropic = True
    else:
        e.notes.append("Added mass not carried: the replay tool is built against Stonefish without "
                       "stonefish_added_mass.patch, which averages added mass over the three axes.")

    # Centre of gravity relative to the centre of buoyancy (x only where the calibration fitted it).
    i1 = measure(s)
    d = np.array(i1["base"]["cg"]) - np.array(i1["base"]["cb"])
    # With Strathe's damping carried as it is, the whole model is Strathe's, so its fore-aft offset (0 unless fitted) goes
    # too: Stonefish's own few millimetres, against Strathe's small righting arm, pitched ALPHA RISE 23-36 deg (#80).
    tx = params.cg_cb_offset_x if (params.cg_cb_offset_x or exact_damping) else d[0]
    if not params.cg_cb_offset_x and not exact_damping:
        e.notes.append("Fore-aft CG offset kept as Stonefish has it: the calibration didn't fit cg_cb_offset_x "
                       "(only with the torpedo-hull terms, vehicle_spec.yaml).")
    s.cg_shift = [float((tx - d[0]) * mass / s.mass), 0.0, float((params.cg_cb_offset_z - d[2]) * mass / s.mass)]
    if exact_damping and buoyancy_part:
        # Strathe's model has its CG at the logs' reference point (its rigid-body terms and restoring moment are written
        # there), with the CB at the calibrated offset from it. Stonefish's vehicle keeps its own geometry, so the
        # logged thrust and the damping, both at the reference point, would pitch it about a CG ~10 cm away: ALPHA RISE
        # pitched 14-27 deg after 10 s against Strathe's 10-11 (#80). So the CG goes to the reference point (the
        # tuned part's <cg>) and the CB to the calibrated offset from it (the buoyancy part moves).
        ref = np.array(reference_point, dtype=float)
        target_cb = ref - np.array([params.cg_cb_offset_x, 0.0, params.cg_cb_offset_z])
        s.cg_shift, s.buoyancy_shift = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        for _ in range(3):  # moving one changes the other's mass or volume share slightly
            info = measure(s)
            b, bp = info["base"], _parts(info)[buoyancy_part]
            s.cg_shift = [c + float(x) for c, x in zip(s.cg_shift, (ref - np.array(b["cg"])) * mass / s.mass)]
            share = max(bp["volume"], 1e-9) / max(b["volume"], 1e-9)
            s.buoyancy_shift = [c + float(x) for c, x in zip(s.buoyancy_shift, (target_cb - np.array(b["cb"])) / share)]
        e.notes.append(f"Centre of gravity moved to the reference point and part '{buoyancy_part}' moved to put the "
                       f"centre of buoyancy at the calibrated offset from it: Strathe's model is written about that "
                       f"point, so the vehicle's geometry follows it.")
    elif exact_damping:
        e.notes.append("Centre of gravity left where Stonefish has it: give --buoyancy-part, so the CG can go to the "
                       "reference point with the CB at the calibrated offset (what Strathe's model assumes; otherwise "
                       "pitch drifts, #80).")

    # Effective inertia: added inertia first, then the rigid inertia where that isn't enough. Two passes, since moving
    # mass and the CG change the vehicle's inertia about its CG.
    target = [params.Ieff_roll, params.Ieff_pitch, params.Ieff_yaw]
    rigid0 = list(hull["inertia"])
    floor_hit = set()
    for _ in range(3):
        cur = measure(s)["base"]["inertia_body_axes"]
        aI = list(s.added_inertia) if s.added_inertia is not None else None
        I = list(s.inertia) if s.inertia is not None else list(rigid0)
        for k in range(3):
            delta = target[k] - cur[k]
            if aI is not None:
                a = aI[k] + delta
                if a < 0:
                    I[k] += a
                    a = 0.0
                aI[k] = a
            else:
                I[k] += delta
            if I[k] < INERTIA_FLOOR * rigid0[k]:
                I[k] = INERTIA_FLOOR * rigid0[k]
                floor_hit.add(k)
        s.added_inertia, s.inertia = aI, I
    for k in sorted(floor_hit):
        e.notes.append(f"{['Roll', 'Pitch', 'Yaw'][k]} inertia: Strathe's {target[k]:.3g} kg·m² is below what Stonefish's "
                       f"vehicle allows (its other parts and their distances from the CG); the part's own inertia "
                       f"stopped at {INERTIA_FLOOR:.0%} of Stonefish's.")

    # What Stonefish has after the change, against Strathe's values.
    i2 = measure(s)
    b0, b2 = base, i2["base"]
    rho_m = i0.get("water_density") or rho

    def net(b):
        return rho * G * b["volume"] - b["mass"] * G
    e.rows += [["mass", mass, b0["mass"], b2["mass"], "kg"],
               [f"net buoyancy (water {rho:g} kg/m³)", params.net_buoyancy_z, net(b0), net(b2), "N"],
               ["CG − CB, x", params.cg_cb_offset_x if (params.cg_cb_offset_x or exact_damping) else None,
                b0["cg"][0] - b0["cb"][0], b2["cg"][0] - b2["cb"][0], "m"],
               ["CG − CB, z", params.cg_cb_offset_z, b0["cg"][2] - b0["cb"][2], b2["cg"][2] - b2["cb"][2], "m"]]
    for k, n in enumerate(("roll", "pitch", "yaw")):
        e.rows.append([f"{n} inertia (rigid + added)", target[k], b0["inertia_body_axes"][k],
                       b2["inertia_body_axes"][k], "kg·m²"])
    if patched:
        aniso = b2.get("anisotropic_added_mass") or [None] * 3
        for k, n in enumerate("xyz"):
            e.rows.append([f"added mass {n}", am[k], b0["added_mass"][k], aniso[k], "kg"])
    if rho_m != rho:
        e.notes.append(f"Buoyancy and drag converted for water of {rho:g} kg/m³ (the training recordings'); the "
                       f"scenario's water is {rho_m:g}, so the vehicle is that much more or less buoyant there, as the "
                       f"real one would be.")
    if params.Zuw or params.Muw or params.Muq:
        e.notes.append("Hull lift (Zuw, Muw, Muq) not carried: Stonefish has no hull-lift term.")
    if params.thruster_time_constant_s:
        e.notes.append(f"Thruster lag ({params.thruster_time_constant_s:.2f} s) not carried: set it on your thrusters.")
    if params.Xq_dot or params.Zq_dot:
        e.notes.append("Off-diagonal added mass (Xq_dot, Zq_dot) not carried: Stonefish's added mass is diagonal.")
    if not exact_damping:
        e.notes.append("Drag converted, not carried: Stonefish's mesh drag grows with speed cubed and couples the axes, "
                       "and its rotational drag comes from the same coefficients over the part's mesh. Stonefish built "
                       "with calibration/stonefish/stonefish_added_mass.patch takes Strathe's damping as it is.")
    return e


def load_calibration(path: str, mass: Optional[float] = None) -> tuple:
    """(CalibrationParameters, mass in kg, the yaml's path) from a calibration's output folder or its
    calibrated_parameters.yaml."""
    from dynamics_model import CalibrationParameters
    pfile = os.path.join(path, "calibrated_parameters.yaml") if os.path.isdir(path) else path
    if not os.path.exists(pfile):
        raise StonefishFitError(f"no calibrated_parameters.yaml at {path}: give your calibration's output folder")
    doc = yaml.safe_load(open(pfile, encoding="utf-8"))
    mass = mass or doc.get("vehicle_mass_kg")
    if not mass:
        raise StonefishFitError("this calibration doesn't record the vehicle's mass (it's older than #74): give --mass, "
                                "the mass in your vehicle_spec.yaml")
    try:
        return CalibrationParameters(**(doc.get("parameters") or {})), float(mass), pfile
    except TypeError as e:  # an incomplete or hand-edited file: say which values are missing (#88)
        raise StonefishFitError(f"{pfile} isn't a complete calibration ({str(e).split(': ', 1)[-1]}): give the "
                                f"output folder of `strathe calibrate`")


def training_density(train_dates: list, densities: dict, model_rho: float) -> float:
    """The water the calibration was trained in (--density-by-date), or the scenario's when mixed or not given. The
    training groups are dates, or recordings named with their date (one session, #93)."""
    from data_split import DATE_PATTERN
    days = [(DATE_PATTERN.search(str(g)) or [g])[0] for g in train_dates]
    rhos = {densities.get(d, model_rho) for d in days}
    return rhos.pop() if len(rhos) == 1 else model_rho


def run_export(a) -> int:
    cal = a.calibration
    if not (a.params or cal):
        raise StonefishFitError("give --calibration <your calibration's output folder> (or --params "
                                "calibrated_parameters.yaml)")
    from dynamics_model import VehicleConstants
    params, mass, pfile = load_calibration(a.params or cal, a.mass)
    data = a.data or (os.path.join(cal, "ingested_logs") if cal else None)
    if not data or not os.path.isdir(data):
        raise StonefishFitError("the calibration's logs aren't there: give --data <folder of Strathe CSVs>")
    heldout = a.heldout_dates
    if not heldout and cal and os.path.exists(os.path.join(cal, "metrics.json")):
        heldout = json.load(open(os.path.join(cal, "metrics.json"), encoding="utf-8"))["dataset"].get("heldout_dates")

    scenario = Scenario(a.scenario, dict(kv.split("=", 1) for kv in (a.find or [])), a.part, a.buoyancy_part)
    scenario.discover()
    replay = Replay(a.replay_tool_argv, a.stonefish_data or os.path.dirname(os.path.abspath(a.scenario)), a.robot,
                    tuple(a.reference_point), a.start_depth, a.physics_hz)
    out = a.out
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="strathe_stonefish_export_")
    try:
        train, held, split = _load_runs(data, heldout)
        replay.dt = (train or held)[0].dt
        densities = dict((d, float(r)) for d, r in (kv.split("=", 1) for kv in (a.density_by_date or [])))
        counter = {"n": 0}

        def measure(settings):
            counter["n"] += 1
            return replay.info(scenario.write_runnable(os.path.join(work, f"info{counter['n']}"), settings))
        rho = training_density(split.train_groups, densities, measure(None).get("water_density") or 1025.0)
        print(f"Converting {os.path.basename(pfile)} into part '{a.part}' of {os.path.basename(scenario.vehicle_file)} "
              f"(mass {mass:g} kg, water {rho:g} kg/m³)...", flush=True)
        e = export_settings(params, float(mass), measure, a.part, a.buoyancy_part, rho,
                            [axis_speeds(train or held, i) for i in range(3)], a.reference_point)
        s = e.settings
        vehicle_out = os.path.join(out, os.path.splitext(os.path.basename(scenario.vehicle_file))[0] + "_strathe.scn")
        scenario.write_calibrated_vehicle(vehicle_out, s)

        results = {}
        if split.warning:   # one session: an easier check than an unseen day (#93)
            print("  note: " + split.warning, flush=True)
        if held:
            print(f"Scoring the original and exported models on the held-out recordings "
                  f"({', '.join(split.heldout_groups)})...", flush=True)
            runs_wins = [(run, windows(run, EVAL_WINDOW_S, EVAL_STRIDE_S)) for run in held]
            for m, (label, settings) in enumerate((("Stonefish, original", None),
                                                    ("Stonefish with Strathe's values", s))):
                preds = []
                for i, (run, wins) in enumerate(runs_wins):
                    scn = scenario.write_runnable(os.path.join(work, f"eval{m}_scn{i}"), settings,
                                                  densities.get(run.date))
                    preds.append(replay.run(scn, window_rows(run, wins), os.path.join(work, f"eval{m}_run{i}")))
                results[label] = score(runs_wins, preds)
            results["Strathe's own model"] = score(runs_wins, [[strathe_free_run(params, VehicleConstants(float(mass)),
                                                                                 run, a_, n) for a_, n in wins]
                                                               for run, wins in runs_wins])
            results.update(baselines(runs_wins))
            _plot(results, os.path.join(out, "drift.png"))
        else:
            e.notes.append("Not scored: no held-out recordings (give --heldout-dates).")

        summary = {"calibration": os.path.abspath(pfile), "scenario": os.path.abspath(a.scenario), "robot": a.robot,
                   "part": a.part, "buoyancy_part": a.buoyancy_part, "mass_kg": float(mass), "water_density": rho,
                   "reference_point_m": list(a.reference_point), "train_dates": split.train_groups,
                   "heldout_dates": split.heldout_groups,
                   **({"split_warning": split.warning} if split.warning else {}),   # one session (#93)
                   "settings": {k: v for k, v in s.__dict__.items()
                                                                        if v is not None},
                   "drag_by_axis": e.drag, "comparison": [dict(zip(("quantity", "strathe", "stonefish_before",
                                                                    "stonefish_after", "unit"), r)) for r in e.rows],
                   "notes": e.notes, "exported_vehicle_file": vehicle_out, "heldout": results,
                   "axis_verdict_10s": axis_verdict(results["Stonefish, original"],
                                                    results["Stonefish with Strathe's values"]) if results else None,
                   "recommendation": recommendation(s)}
        yaml.safe_dump(json.loads(json.dumps(summary)), open(os.path.join(out, "stonefish_export.yaml"), "w"),
                       sort_keys=False)
        open(os.path.join(out, "stonefish_export_report.md"), "w", encoding="utf-8").write(
            "\n".join(_report(scenario, a, e, results, split, vehicle_out, rho)) + "\n")
        print(f"Wrote {vehicle_out}, stonefish_export.yaml and stonefish_export_report.md in {out}")
        for name, r in results.items():
            hs = [f"{r[f'{h}s']['drift_median_m']:.2f}" for h in HORIZONS_S if f"{h}s" in r]
            print(f"  {name:36s} drift {' / '.join(hs)} m")
        o, x = results.get("Stonefish, original"), results.get("Stonefish with Strathe's values")
        if o and x:
            v = axis_verdict(o, x)
            print("  after 10 s, by axis: " + "; ".join(f"{k}: {', '.join(n)}" for k, n in v.items() if n))
        print("  " + recommendation(s))
        for n in e.notes:
            print("  note: " + n)
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


AXIS_CHECKS = (("forward speed (surge)", "velocity_rmse", 0), ("sideways speed (sway)", "velocity_rmse", 1),
               ("up/down speed (heave)", "velocity_rmse", 2), ("roll", "attitude_error_median_deg", 0),
               ("pitch", "attitude_error_median_deg", 1), ("heading (yaw)", "attitude_error_median_deg", 2))
RECOMMENDATION = ("Strathe's values are not a calibrated Stonefish model: they balance against Strathe's own drag and "
                  "righting, not Stonefish's. Its mesh drag grows with speed cubed and couples the axes (surge), and "
                  "its roll and pitch damping come from the same mesh coefficients, so a transplanted centre of "
                  "gravity no longer balances them (roll, pitch). To calibrate your Stonefish model, run "
                  "`strathe fit-stonefish`, which fits it through Stonefish itself (on URI's ALPHA RISE: 53-66% less "
                  "error after 60 s, against 17-28% for this conversion; PROJECT_STATUS.md #73-#75).")


RECOMMENDATION_EXACT = ("With the patch's damping, Stonefish runs Strathe's own model, with the centre of gravity at the "
                        "logs' reference point as Strathe's model has it: on URI's ALPHA RISE the speeds matched "
                        "Strathe's, and the error after 10 s fell 56-68% against the drag conversion (#80). Over a "
                        "minute `strathe fit-stonefish`, which fits your model through Stonefish itself, can still be "
                        "better (21.6 and 13.8 m after 60 s on ALPHA RISE, against 36.0 and 17.6 m for this export).")


def recommendation(settings) -> str:
    """What to do next: depends on whether Strathe's damping went in as it is (the patch) or was converted."""
    return RECOMMENDATION_EXACT if settings.axis_damping is not None else RECOMMENDATION


def axis_verdict(original: dict, exported: dict, horizon: str = "10s", margin: float = 0.05) -> dict:
    """{'better': [...], 'worse': [...], 'same': [...]}: the exported model against the original, per axis, by velocity
    error (linear axes) and attitude error (roll, pitch, heading) after `horizon`; within `margin` is 'same'."""
    out = {"better": [], "worse": [], "same": []}
    o, x = original.get(horizon), exported.get(horizon)
    if not (o and x):
        return out
    for name, key, i in AXIS_CHECKS:
        a, b = o[key][i], x[key][i]
        out["better" if b < (1 - margin) * a else "worse" if b > (1 + margin) * a else "same"].append(name)
    return out


def _num(x, unit):
    return "n/a (not fitted)" if x is None else f"{x:.4g} {unit}"


def _report(scenario, a, e: Export, results: dict, split, vehicle_out: str, rho: float) -> list:
    L = [f"# Your Strathe calibration in Stonefish: {os.path.basename(scenario.vehicle_file)}\n"]
    if split.warning:   # one session (#93)
        L.append("> ⚠️ " + split.warning + "\n")
    L.append("## Bottom line\n")
    o, x = results.get("Stonefish, original", {}), results.get("Stonefish with Strathe's values", {})
    h = next((f"{h}s" for h in reversed(HORIZONS_S) if f"{h}s" in o and f"{h}s" in x), None)
    if h:
        ch = 100 * (x[h]["drift_median_m"] - o[h]["drift_median_m"]) / max(o[h]["drift_median_m"], 1e-9)
        L.append(f"On recordings the calibration never saw ({', '.join(split.heldout_groups)}), Stonefish with Strathe's "
                 f"values has **{x[h]['drift_median_m']:.2f} m** of position error after {h[:-1]} s, against "
                 f"**{o[h]['drift_median_m']:.2f} m** for your original model ({ch:+.0f}%). Check every horizon below: "
                 f"a change can help over a minute and cost over seconds.\n")
        v = axis_verdict(o, x)
        if any(v.values()):
            L.append("After 10 s, by axis: " + "; ".join(f"**{k}**: {', '.join(n)}" for k, n in v.items() if n) + ".\n")
    else:
        L.append("Not scored: no held-out recording is long enough for a 60 s window (or none was held out).\n")
    L.append(f"**{recommendation(e.settings)}**\n")
    if h:
        L += ["## How the models compare on unseen recordings\n",
              f"Windows of {EVAL_WINDOW_S:g} s every {EVAL_STRIDE_S:g} s; each starts from the logged state and is "
              f"driven by the logged forces, the model turning and tilting on its own, in still water. Medians.\n"]
        L += _table(results)
        L += ["", "![Drift](drift.png)", "",
              "\"Keeps its starting velocity\" is a hard baseline over a closed-loop mission: the logged forces react to "
              "the real vehicle, so any open-loop model's error compounds. These numbers rank models; they don't "
              "promise minute-ahead prediction.\n"]
    L += ["## Strathe's values, and what Stonefish has before and after\n",
          "| Quantity | Strathe | Stonefish before | Stonefish after |", "|---|---|---|---|"]
    for q, sv, b, af, unit in e.rows:
        L.append(f"| {q} | {_num(sv, unit)} | {_num(b, unit)} | {_num(af, unit)} |")
    if e.settings.axis_damping is not None:
        d = e.settings.axis_damping
        L += ["", "**Drag:** Strathe's per-axis damping as it is, replacing the vehicle's mesh drag, at the logs' "
                  f"reference point ({_fmt(d['point'])} m): linear {_fmt(d['linear'])} / {_fmt(d['linear_rot'])}, "
                  f"quadratic {_fmt(d['quadratic'])} / {_fmt(d['quadratic_rot'])} (surge, sway, heave / roll, pitch, "
                  f"yaw).\n"]
    else:
        L += ["", f"**Drag** (Stonefish's mesh drag is cubic in speed, so it's matched to Strathe's over the speeds "
                  f"your logs cover; water {rho:g} kg/m³):\n",
              "| Axis | Cd | Cf | at speeds (median, 95th pct) | Strathe's force | Stonefish's force |",
              "|---|---|---|---|---|---|"]
    for ax, d in e.drag.items():
        L.append(f"| {ax} | {d['quadratic_drag']:.4g} | {d['viscous_drag']:.4g} | "
                 f"{', '.join(f'{u:.2f}' for u in d['speeds_m_s'])} m/s | "
                 f"{', '.join(f'{f:.1f}' for f in d['strathe_force_n'])} N | "
                 f"{', '.join(f'{f:.1f}' for f in d['stonefish_force_n'])} N |")
    L += ["", "## What changed in your vehicle file\n",
          f"`{os.path.basename(vehicle_out)}` is `{os.path.basename(scenario.vehicle_file)}` with only part `{a.part}`"
          + (f" and the buoyancy part `{a.buoyancy_part}`" if a.buoyancy_part else "") + " changed:\n"]
    s = e.settings
    L.append(f"- mass: {s.mass:.4g} kg")
    if s.axis_damping is not None:
        L.append("- `damping_linear`, `damping_linear_rot`, `damping_quadratic`, `damping_quadratic_rot`, `damping_point` "
                 "on its `<hydrodynamics>`: Strathe's damping, used instead of the mesh drag (needs Stonefish with "
                 "`calibration/stonefish/stonefish_added_mass.patch`; without it they're ignored)")
    else:
        L.append(f"- quadratic drag {_fmt(s.quadratic_drag)}, viscous drag {_fmt(s.viscous_drag)}")
    if s.added_mass is not None:
        L.append(f"- added mass {_fmt(s.added_mass)}, added inertia {_fmt(s.added_inertia)}, "
                 f"`anisotropic_added_mass=\"true\"` (needs Stonefish with "
                 f"`calibration/stonefish/stonefish_added_mass.patch`)")
    if s.inertia is not None:
        L.append(f"- rigid inertia {_fmt(s.inertia)} kg·m²")
    if s.cg_shift is not None:
        L.append(f"- centre of gravity moved by {s.cg_shift[0]:+.3f} m (x), {s.cg_shift[2]:+.3f} m (z)")
    if s.buoyancy_scale is not None:
        L.append(f"- buoyant volume of `{a.buoyancy_part}`: x{s.buoyancy_scale:.3f}")
    if s.buoyancy_shift is not None and any(s.buoyancy_shift):
        L.append(f"- `{a.buoyancy_part}` moved by {s.buoyancy_shift[0]:+.3f} m (x), {s.buoyancy_shift[2]:+.3f} m (z), "
                 f"so the centre of buoyancy sits at the calibrated offset from the reference point")
    if e.notes:
        L += ["", "## Not carried, and why\n"] + [f"- {n}" for n in e.notes]
    return L


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calibration", help="your Strathe calibration's output folder (calibrated_parameters.yaml, "
                                          "ingested_logs/, metrics.json)")
    ap.add_argument("--params", help="or: a calibrated_parameters.yaml (then give --data)")
    ap.add_argument("--data", help="the logs in Strathe's CSV layout (default: the calibration's ingested_logs/)")
    ap.add_argument("--mass", type=float, help="the vehicle's mass, kg (default: the calibration's)")
    ap.add_argument("--heldout-dates", "--heldout-recordings", dest="heldout_dates", nargs="+",
                    help="dates or recordings to score on (default: the calibration's)")
    ap.add_argument("--scenario", required=True, help="your Stonefish scenario (the world file with the robot in it)")
    ap.add_argument("--robot", required=True, help="the robot's name in the scenario (after $(arg ...) substitution)")
    ap.add_argument("--part", required=True, help="the robot part that takes the values, e.g. its hull")
    ap.add_argument("--reference-point", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                    help="where the logs' velocity and forces are measured, in the robot's frame (forward-right-down, "
                         "m from its origin), e.g. the base_link of your URDF")
    ap.add_argument("--buoyancy-part", help="a buoyant box/cylinder/sphere part resized for the net buoyancy")
    ap.add_argument("--find", nargs="+", metavar="PKG=PATH", help="folders for $(find PKG) in ROS-style scenarios")
    ap.add_argument("--stonefish-data", help="Stonefish's data folder for meshes (default: the scenario's folder)")
    ap.add_argument("--density-by-date", nargs="+", metavar="DATE=RHO",
                    help="water density per recording date (e.g. 27-03-2026=1000 for fresh water)")
    ap.add_argument("--start-depth", type=float, default=3.0, help="depth the windows are replayed at, m")
    ap.add_argument("--physics-hz", type=float, default=200.0, help="Stonefish's physics rate (default 200 Hz)")
    ap.add_argument("--replay-tool", help="the built strathe_stonefish_replay; or docker: Strathe's Stonefish image (setup-sim "
                    "stonefish --docker, once); or docker:IMAGE (default: calibration/stonefish/build/, else the "
                    "Docker image if it's built)")
    ap.add_argument("--out", default="stonefish_export", help="output folder")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        a.replay_tool_argv = find_tool(a.replay_tool)
        return run_export(a)
    except StonefishFitError as e:
        print(f"❌ {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
