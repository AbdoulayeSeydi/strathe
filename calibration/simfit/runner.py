"""
simfit/runner.py -- `strathe fit-sim` and `strathe export-sim` for any simulator (PROJECT_STATUS.md #81).

The same steps as fit-stonefish (#72-#80), with the simulator behind an adapter (calibration/simfit/adapters/):
  fit-sim     fit the lab's own model's settings (knobs) to the training recordings by replaying them in the
              simulator; score original and calibrated on the held-out recordings; write the calibrated model.
  export-sim  put a Strathe calibration into the lab's model where the simulator has the same physics (an adapter's
              export_values), and score it the same way.
Outputs, in --out: <model>_calibrated.<ext> (or _strathe), <sim>_fit.yaml / <sim>_export.yaml, the report, drift.png.

An adapter provides:
  sim, label                        a short name ("gazebo") and the reports' name ("Gazebo")
  groups                            the knob groups it can fit, in order (--fit chooses among them)
  info() -> dict                    what the model and simulator report (mass, version, ...)
  knob_specs(info, groups, axisymmetric) -> [KnobSpec]
  set_base(values)                  start every model from these values (--from-strathe)
  write_runnable(values, folder, rho) -> model path   (values None: the model as it is, or its base)
  replay(model, rows_by_window, folder) -> [(V, Q)]
  write_calibrated(values, out_dir, suffix) -> path   (the lab's own file, only those values changed)
  changes(values) -> [report lines]
  export_values(params, mass, info, train, rho) -> (values, notes)   (raises SimFitError where there's no mapping)
  notes                             report lines the adapter adds (what it couldn't do, and why)
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np
import yaml

from simfit import core
from simfit.core import EVAL_STRIDE_S, EVAL_WINDOW_S, HORIZONS_S, SimFitError
from simfit.knobs import Knobs, parse_holds


def add_shared_arguments(ap, out_default: str) -> None:
    ap.add_argument("--data", required=True, help="your logs in Strathe's CSV layout (`strathe ingest`, or "
                                                  "`--ingest-spec` on `strathe fit-sim`)")
    ap.add_argument("--reference-point", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                    help="where the logs' velocity and forces are measured, in the model's frame (forward-right-down, "
                         "m from its origin), e.g. your URDF's base_link")
    ap.add_argument("--heldout-dates", "--heldout-recordings", dest="heldout_dates", nargs="+",
                    help="dates or recordings to hold out (default: the latest 30%% of dates, else of recordings, else "
                         "the last 30%% of the one recording; calibration/ONE_SESSION.md)")
    ap.add_argument("--density-by-date", nargs="+", metavar="DATE=RHO",
                    help="water density per recording date (e.g. 27-03-2026=1000 for fresh water)")
    ap.add_argument("--compare-params", help="also score your Strathe calibration (calibrated_parameters.yaml)")
    ap.add_argument("--compare-mass", type=float, help="its vehicle mass, kg (default: the calibration's)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="simulator runs at once")
    ap.add_argument("--spec", help="a fin vehicle's description: says what its fin channels mean (FINS.md)")
    ap.add_argument("--out", default=out_default, help="output folder")


def add_fit_arguments(ap) -> None:
    ap.add_argument("--fit", nargs="+", help="what to fit, by group (default: every group the simulator has; the "
                                             "simulator's guide lists them)")
    ap.add_argument("--hold", nargs="+", metavar="NAME=VALUE",
                    help="hold settings at values you measured instead of fitting them, e.g. added_mass_x=10.6")
    ap.add_argument("--weighting", choices=["variation", "magnitude"], default="variation",
                    help="each axis's error is divided by its velocity's variation (default) or its magnitude (RMS)")
    ap.add_argument("--axisymmetric", action="store_true",
                    help="one value for sway and heave (a torpedo-shaped vehicle)")
    ap.add_argument("--from-strathe", metavar="CALIBRATION",
                    help="start from your Strathe calibration's values put into the model (as export-sim), then fit")
    ap.add_argument("--starts-from", metavar="CALIBRATION",
                    help="no DVL: start each window from the state smoother with your Strathe calibration's physics, "
                         "without putting its values into the model (any simulator; --from-strathe does both)")
    ap.add_argument("--max-iterations", type=int, default=15, help="least-squares iterations (each: one Jacobian)")
    ap.add_argument("--finish", action="store_true",
                    help="an interrupted fit: don't fit again, use its best point so far (the checkpoint in --out), then "
                         "score it and write the model and report as usual. Give the same command as the interrupted run")
    ap.add_argument("--no-cg-rerun", action="store_true",
                    help="don't fit again with the centre of gravity held when the first fit makes the attitude worse "
                         "(by default it does, and keeps the better of the two)")


def _densities(a) -> dict:
    out = {}
    for kv in a.density_by_date or []:
        d, _, r = kv.partition("=")
        try:
            out[d] = float(r)
        except ValueError:
            raise SimFitError(f"--density-by-date takes DATE=RHO (e.g. 27-03-2026=1000), got {kv!r}")
    return out


def _score_models(adapter, held, models: list, work: str, densities: dict) -> dict:
    """{label: score} for [(label, values)] on the held-out recordings' 60 s windows."""
    runs_wins = [(run, core.windows(run, EVAL_WINDOW_S, EVAL_STRIDE_S)) for run in held]
    results = {}
    for m, (label, values) in enumerate(models):
        preds = []
        for i, (run, wins) in enumerate(runs_wins):
            d = os.path.join(work, f"eval{m}_{i}")
            model = adapter.write_runnable(values, d, densities.get(run.date))
            preds.append(adapter.replay(model, core.window_rows(run, wins), d))
        results[label] = core.score(runs_wins, preds)
    return results, runs_wins


def _strathe_own(results, runs_wins, params_file, mass) -> None:
    from dynamics_model import CalibrationParameters, VehicleConstants
    doc = yaml.safe_load(open(params_file, encoding="utf-8"))
    p = CalibrationParameters(**doc["parameters"])
    m = mass or doc.get("vehicle_mass_kg")
    if not m:
        raise SimFitError("give --compare-mass: the calibration doesn't record the vehicle's mass")
    act = getattr(runs_wins[0][0], "actuators", None) if runs_wins else None
    results["Strathe's own model"] = core.score(runs_wins, [[core.strathe_free_run(p, VehicleConstants(float(m), actuators=act), run, a_, n)
                                                             for a_, n in wins] for run, wins in runs_wins])


def _bottom_line(results, original: str, new: str, held_groups, what: str) -> list:
    o, x = results[original], results[new]
    h = next((f"{h}s" for h in reversed(HORIZONS_S) if f"{h}s" in o and f"{h}s" in x), None)
    if not h:
        return ["Not scored: no held-out recording is long enough for a 60 s window.\n"]
    ch = 100 * (x[h]["drift_median_m"] - o[h]["drift_median_m"]) / max(o[h]["drift_median_m"], 1e-9)
    L = [f"On recordings it never saw ({', '.join(held_groups)}), {what} has **{x[h]['drift_median_m']:.2f} m** of "
         f"position error after {h[:-1]} s, against **{o[h]['drift_median_m']:.2f} m** for your original model "
         f"({ch:+.0f}%). Check every horizon below: a change can help over a minute and cost over seconds.\n"]
    return L + _warnings(results, original, new, what)


_warnings = core.report_warnings  # moved to simfit.core (#90): fit-stonefish uses it too


def _compare_section(results) -> list:
    against = next((r.get("drift_against") for r in results.values() if r.get("drift_against")), None)
    note = ([f"No DVL in these logs (#96, SENSORS.md): each window starts from the vehicle's estimated velocity, and "
             f"drift is measured against its own position fixes ({against}); velocity errors on u, v, w are nan where "
             f"nothing measured them.\n"] if against else [])
    return ["## How the models compare on unseen recordings\n", *note,
            f"Windows of {EVAL_WINDOW_S:g} s every {EVAL_STRIDE_S:g} s; each starts from the logged state and is driven "
            f"by the logged forces, the model turning and tilting on its own, in still water. Medians.\n",
            *core.table(results), "", "![Drift](drift.png)", "",
            "\"Keeps its starting velocity\" is a hard baseline over a closed-loop mission: the logged forces react to the "
            "real vehicle, so any open-loop model's error compounds. These numbers rank models; they don't promise "
            "minute-ahead prediction.\n"]


def _print_results(results) -> None:
    for name, r in results.items():
        hs = [f"{r[f'{h}s']['drift_median_m']:.2f}" for h in HORIZONS_S if f"{h}s" in r]
        print(f"  {name:36s} drift {' / '.join(hs)} m" if hs else f"  {name:36s} not scored: no held-out recording is long enough for a 60 s window")


def _calibration_noise(pfile: str, data: str) -> dict:
    """The reading noise the Strathe calibration's smoother used (its description's noise: and grade:, and what was
    measured: metrics.json), else what ingestion measured in these logs."""
    from measurements import measured_noise
    from run_calibration import smoother_noise_of
    mp = os.path.join(os.path.dirname(os.path.abspath(pfile)), "metrics.json")
    if os.path.exists(mp):
        ss = json.load(open(mp, encoding="utf-8")).get("sensor_suite") or {}
        if ss:
            return smoother_noise_of(ss)
    return measured_noise(data)


def _calibration_delays(pfile: str, runs) -> None:
    """The delay the Strathe calibration took off a late position sensor's fixes (metrics.json), taken off these runs'
    fixes too, so the simulator is fitted to the same fixes (#96)."""
    mp = os.path.join(os.path.dirname(os.path.abspath(pfile)), "metrics.json")
    if not runs or getattr(runs[0], "channels", None) is None or not os.path.exists(mp):
        return
    from state_smoother import shift_fixes
    for name, d in ((json.load(open(mp, encoding="utf-8")).get("sensor_suite") or {}).get("fix_delays") or {}).items():
        if d.get("applied_s"):
            shift_fixes(runs, name, d["applied_s"])
            print(f"  {name}'s fixes: {d['applied_s']:g} s of delay taken off, as the calibration did.", flush=True)


def _physics_starts(runs, params, mass, noise=None) -> None:
    """No DVL (#96): with a Strathe calibration at hand, each window starts from the smoother run with its physics
    (about 4x closer to the truth than physics-free in tests/test_simfit_sensors.py) instead of without."""
    if mass and runs and getattr(runs[0], "channels", None) is not None:
        from dynamics_model import VehicleConstants
        from simfit.sensors import set_starts
        set_starts(runs, params, VehicleConstants(float(mass)), noise)
        print("  Window starts from the smoother with your Strathe calibration's physics.", flush=True)


def run_fit(adapter, a) -> int:
    out = a.out
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"strathe_{adapter.sim}_")
    try:
        train, held, split = core.load_runs(a.data, a.heldout_dates, getattr(a, "spec", None))
        if not held:
            raise SimFitError("no held-out recordings: give --heldout-dates, or logs from at least two sessions")
        adapter.dt = train[0].dt
        densities = _densities(a)
        notes = list(getattr(adapter, "notes", []))
        exported = None
        if a.from_strathe and getattr(a, "starts_from", None):
            raise SimFitError("give --from-strathe or --starts-from, not both: --from-strathe already starts each window "
                              "from its calibration's physics")
        if getattr(a, "starts_from", None):   # #96: the physics-aided starts alone, for a model Strathe's can't go into
            from stonefish_export import load_calibration
            sp, smass, pfile = load_calibration(a.starts_from, a.compare_mass)
            if getattr(train[0], "channels", None) is None:
                print("  note: --starts-from changes nothing here: these logs have a DVL, and windows start from it",
                      flush=True)
            _calibration_delays(pfile, train + held)
            _physics_starts(train + held, sp, smass, _calibration_noise(pfile, a.data))
            a.compare_params, a.compare_mass = a.compare_params or pfile, a.compare_mass or smass
        if a.from_strathe:
            from stonefish_export import load_calibration, training_density
            sp, smass, pfile = load_calibration(a.from_strathe, a.compare_mass)
            _calibration_delays(pfile, train + held)
            _physics_starts(train + held, sp, smass, _calibration_noise(pfile, a.data))
            info0 = adapter.info()
            rho = training_density(split.train_groups, densities, info0.get("water_density") or 1025.0)
            try:
                values, enotes = adapter.export_values(sp, smass, info0, train, rho)
            except SimFitError as e:  # e.g. MuJoCo: say what to do here, not what export-sim would (#88, F3)
                raise SimFitError(f"--from-strathe needs Strathe's values in the {adapter.label} model, and that isn't "
                                  f"possible: {e} Run fit-sim without --from-strathe: it starts from your own model.")
            adapter.set_base(values)
            exported = {"calibration": pfile, "values": values, "notes": enotes}
            a.compare_params, a.compare_mass = a.compare_params or pfile, a.compare_mass or smass
            print(f"Starting from {pfile}'s values, put into the {adapter.label} model (mass {smass:g} kg).", flush=True)
        info = adapter.info()
        groups = tuple(a.fit or getattr(adapter, "default_groups", None) or adapter.groups)
        groups = tuple(g for g in groups if g in adapter.groups) if not a.fit else groups
        if not a.fit and getattr((train or held)[0], "actuators", None) is not None and "fins" in adapter.groups and "fins" not in groups:
            groups = (*groups, "fins")   # a fin vehicle's fins are what's being tuned first of all
        unknown = set(groups) - set(adapter.groups)
        if unknown:
            raise SimFitError(f"--fit {', '.join(sorted(unknown))}: the {adapter.label} model has "
                              f"{', '.join(adapter.groups)}")
        specs = adapter.knob_specs(info, groups, a.axisymmetric)
        notes = list(getattr(adapter, "notes", notes))
        if split.warning:   # one session: an easier check than an unseen day (#93)
            notes.insert(0, split.warning)
        if not specs:
            raise SimFitError(f"nothing to fit: the {adapter.label} model has no settings in {', '.join(groups)}")
        knobs = Knobs(specs, parse_holds(a.hold))
        if knobs.held:
            print("  held: " + ", ".join(f"{n} = {v:g}" for n, v in knobs.held.items()), flush=True)
        print(f"Fitting {', '.join(sorted({s.group for s in specs}, key=groups.index))} in the {adapter.label} model "
              f"({len(knobs.names())} values) from {len(train)} training recording(s); {len(held)} held out "
              f"({', '.join(split.heldout_groups)}).", flush=True)
        for n in notes:
            print("  note: " + n, flush=True)

        fingerprint = {"simulator": adapter.sim, "model": os.path.abspath(adapter.model_path),
                       "data": os.path.abspath(a.data), "heldout": list(split.heldout_groups)}
        checkpoints = {"first": os.path.join(out, f"{adapter.sim}_fit_checkpoint.json"),
                       "cg_held": os.path.join(out, f"{adapter.sim}_fit_checkpoint_cg_held.json")}
        finished = []   # the fits finished from a checkpoint (--finish): reported as not converged
        if not a.finish:
            for c in checkpoints.values():   # a new fit: last run's checkpoints no longer apply
                if os.path.exists(c):
                    os.remove(c)

        def fit_with(kn, which):
            if a.finish:
                if which == "cg_held" and not os.path.exists(checkpoints[which]):
                    return None
                r = core.finish_from_checkpoint(checkpoints[which], kn, fingerprint)
                if r.get("finished_from_checkpoint"):
                    finished.append(which)
                print(f"Finishing from {os.path.basename(checkpoints[which])}: the best of {r['evaluations']} evaluations "
                      f"(cost {r['cost']:.4f}).", flush=True)
                return r

            def simulate(x, i, run, rows, d):
                model = adapter.write_runnable(kn.values(x), os.path.join(d, f"m{i}"), densities.get(run.date))
                return adapter.replay(model, rows, os.path.join(d, f"r{i}"))
            print(f"  (the best point so far is saved to {checkpoints[which]}: if this is interrupted, run the same "
                  f"command with --finish)", flush=True)
            return core.fit(simulate, train, kn, work, a.workers, a.max_iterations, log=lambda s: print(s, flush=True),
                            weighting=a.weighting, checkpoint=checkpoints[which], fingerprint=fingerprint)

        def check_step(values):
            """Gazebo (#90): if the fitted damping is too stiff for the world's step, score at a safe one."""
            if not hasattr(adapter, "safe_step"):
                return False
            step = adapter.safe_step(values)
            if step is None or (adapter.step_override and step >= adapter.step_override):
                return False
            for n in adapter.after_fit_notes(values):
                if n not in notes:
                    notes.append(n)
                    print("  note: " + n, flush=True)
            adapter.step_override = step
            return True

        f = fit_with(knobs, "first")
        values = f["values"]
        check_step(values)

        def score(models):
            base = adapter.base
            out_ = {}
            for label, vals in models:
                adapter.set_base(None if vals == "original" else base)  # "original" is the lab's own model
                out_.update(_score_models(adapter, held, [(label, None if vals in ("original", "base") else vals)],
                                          work, densities)[0])
            adapter.set_base(base)
            return out_

        print("Scoring the original and calibrated models on the held-out recordings...", flush=True)
        orig, cal = f"{adapter.label}, original", f"{adapter.label}, calibrated by Strathe"
        models = [(orig, "original")] + ([(f"{adapter.label} with Strathe's values", "base")] if exported else []) + \
            [(cal, values)]
        results = score(models)
        runs_wins = [(run, core.windows(run, EVAL_WINDOW_S, EVAL_STRIDE_S)) for run in held]

        # #90: the first fit traded attitude for position (it moved the centre of gravity): fit again with it held,
        # starting from the first fit's other values, and keep the better of the two
        rerun = None
        cg_tuned = [sp.name for sp in specs if sp.group == "cg" and sp.name not in knobs.held]
        worse = core.attitude_worse(results[orig], results[cal])
        skipped_rerun = False
        if worse and cg_tuned and not a.no_cg_rerun and a.finish and not os.path.exists(checkpoints["cg_held"]):
            skipped_rerun = True   # finishing doesn't start a new fit; the report says how to get the second one
        elif worse and cg_tuned and not a.no_cg_rerun:
            print(f"The calibrated model turns or tilts worse than your original ({', '.join(worse)}), and the fit moved "
                  f"the centre of gravity. Fitting again with it held (about as long again; --no-cg-rerun skips "
                  f"this)...", flush=True)
            from dataclasses import replace
            specs_b = [replace(sp, start=float(values[sp.name])) if sp.transform != "linear" and sp.name in values
                       and float(values[sp.name]) > 0 else sp for sp in specs if sp.group != "cg"]
            knobs_b = Knobs(specs_b, {k: v for k, v in knobs.held.items() if k in {sp.name for sp in specs_b}})
            fb = fit_with(knobs_b, "cg_held")
            values_b = fb["values"]
            tuned_label, held_label = cal + " (centre of gravity tuned)", cal + " (centre of gravity held)"
            if check_step(values_b):   # a shorter step: score everything again at it, so the rows compare
                results = score(models)
            results[tuned_label] = results.pop(cal)
            results.update(score([(held_label, values_b)]))
            use_held, why = core.prefer_cg_held(results[orig], results[tuned_label], results[held_label])
            rerun = {"held_values": values_b, "fit": {k: fb[k] for k in ("start_cost", "cost", "evaluations", "status")},
                     "chosen": "held" if use_held else "tuned", "why": why, "attitude_worse": worse}
            cal = held_label if use_held else tuned_label
            other_values, values = (values, values_b) if use_held else (values_b, values)
            if use_held:
                f = fb
        model_out = adapter.write_calibrated(values, out, "calibrated")
        other_out = None
        if rerun:
            other_out = adapter.write_calibrated(other_values, out, "calibrated_cg_tuned" if rerun["chosen"] == "held"
                                                 else "calibrated_cg_held")
        world_out = adapter.write_deliverable(values, out) if hasattr(adapter, "write_deliverable") else None
        if a.compare_params:
            _strathe_own(results, runs_wins, a.compare_params, a.compare_mass)
        results.update(core.baselines(runs_wins))

        start = {sp.name: sp.start for sp in specs}
        summary = {"simulator": adapter.sim, "model": adapter.model_path, "reference_point_m": list(a.reference_point),
                   "fitted": sorted({sp.group for sp in specs}, key=groups.index), "axisymmetric": bool(a.axisymmetric),
                   "held": knobs.held, "weighting": a.weighting, "notes": notes,
                   "from_strathe": exported, "axis_scales": f["axis_scales"],
                   "before": start, "after": values, "train_dates": split.train_groups,
                   "heldout_dates": split.heldout_groups,
                   "fit": {k: f[k] for k in ("start_cost", "cost", "evaluations", "status")},
                   "cg_rerun": None if not rerun else {**rerun, "other_model_file": other_out},
                   "finished_from_checkpoint": finished,
                   "physics_step_s": getattr(adapter, "step_override", None),
                   "calibrated_world": world_out,
                   "info": info, "calibrated_model_file": model_out, "heldout": results}
        yaml.safe_dump(json.loads(json.dumps(summary)), open(os.path.join(out, f"{adapter.sim}_fit.yaml"), "w"),
                       sort_keys=False)
        core.plot(results, os.path.join(out, "drift.png"), f"Unseen recordings: original vs calibrated {adapter.label} model")
        L = [f"# {adapter.label} model calibrated from your logs: {os.path.basename(adapter.model_path)}\n",
             "## Bottom line\n", *_bottom_line(results, orig, cal, split.heldout_groups, "the calibrated model")]
        if rerun:
            L.append(f"**Two fits were run.** The first moved the centre of gravity and made the attitude worse "
                     f"({', '.join(rerun['attitude_worse'])}), so Strathe fitted again with the centre of gravity held. "
                     + (f"`{os.path.basename(model_out)}` is the one with it held: {rerun['why']}. The first is "
                        f"`{os.path.basename(other_out)}`." if rerun["chosen"] == "held" else
                        f"`{os.path.basename(model_out)}` is still the first fit: {rerun['why']}. The one with it held is "
                        f"`{os.path.basename(other_out)}`.") + " Both are in the table below.\n")
        if world_out:
            step = getattr(adapter, "step_override", None)
            L.append(f"**To run it:** `calibrated_world/` is a copy of your world with the calibrated model in it"
                     + (f", at a {step:g} s physics step (your world's is too long for the fitted damping; see the notes)"
                        if step else "") + f". Your own files are unchanged. From this folder: "
                     f"`{adapter.run_command(os.path.basename(world_out))}` (`calibrated_world/RUN_ME.md` has the "
                     f"Windows line).\n")
        L += [*_compare_section(results), "## What changed in your model\n",
              f"`{os.path.basename(model_out)}` is `{os.path.basename(adapter.model_path)}` with only these values "
              f"changed:\n", *adapter.changes(values)]
        if knobs.held:
            L.append("- held at your values: " + ", ".join(f"{n} = {v:g}" for n, v in knobs.held.items()))
        if exported:
            L.append(f"- started from your Strathe calibration (`{exported['calibration']}`), put into the model as "
                     f"export-sim does; the fit tuned the rest from there")
        L += ["", "## How the fit went\n",
              f"Fitted {len(knobs.names())} values on {', '.join(split.train_groups)}. Cost {summary['fit']['start_cost']:.3f} -> "
              f"{summary['fit']['cost']:.3f} after {summary['fit']['evaluations']} evaluations ({summary['fit']['status']})."]
        if rerun:
            rf = rerun["fit"]
            L.append(f"The refit with the centre of gravity held: cost {rf['start_cost']:.3f} -> {rf['cost']:.3f} after "
                     f"{rf['evaluations']} evaluations ({rf['status']}).")
        if finished:
            L.append(f"**Finished from a checkpoint (`--finish`):** the {' and the '.join('first fit' if w == 'first' else 'refit with the centre of gravity held' for w in finished)} "
                     f"stopped before converging, and this is its best point so far. Run the command again without "
                     f"`--finish` for a converged fit.")
        if skipped_rerun:
            L.append("**The refit with the centre of gravity held wasn't run** (`--finish` doesn't start a fit): run the "
                     "command again without `--finish` to get it.")
        L += [f"- {n}" for n in notes]
        L += ["", "Fitting can trade one axis for another, or tailor values to the training sessions: odd values (a "
              "drag far from the shape's, a value at its limit) mean the logs don't pin them down. More varied "
              "manoeuvres at several speeds help."]
        open(os.path.join(out, f"{adapter.sim}_fit_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
        print(f"Wrote {model_out}, {adapter.sim}_fit.yaml and {adapter.sim}_fit_report.md in {out}"
              + (f"; {os.path.relpath(world_out, out)} runs it" if world_out else ""))
        _print_results(results)
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_export(adapter, a) -> int:
    from stonefish_export import load_calibration, training_density
    cal = a.calibration
    if not (a.params or cal):
        raise SimFitError("give --calibration <your calibration's output folder> (or --params calibrated_parameters.yaml)")
    params, mass, pfile = load_calibration(a.params or cal, a.mass)
    data = a.data or (os.path.join(cal, "ingested_logs") if cal else None)
    if not data or not os.path.isdir(data):
        raise SimFitError("the calibration's logs aren't there: give --data <folder of Strathe CSVs>")
    heldout = a.heldout_dates
    if not heldout and cal and os.path.exists(os.path.join(cal, "metrics.json")):
        heldout = json.load(open(os.path.join(cal, "metrics.json"), encoding="utf-8"))["dataset"].get("heldout_dates")
    out = a.out
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"strathe_{adapter.sim}_export_")
    try:
        train, held, split = core.load_runs(data, heldout, getattr(a, "spec", None))
        adapter.dt = (train or held)[0].dt
        _calibration_delays(pfile, train + held)
        _physics_starts(train + held, params, mass, _calibration_noise(pfile, data))   # no DVL (#96): physics starts
        no_dvl = False
        densities = _densities(a)
        info = adapter.info()
        rho = training_density(split.train_groups, densities, info.get("water_density") or 1025.0)
        print(f"Putting {os.path.basename(pfile)} into the {adapter.label} model {os.path.basename(adapter.model_path)} "
              f"(mass {mass:g} kg, water {rho:g} kg/m³)...", flush=True)
        values, notes = adapter.export_values(params, float(mass), info, train, rho)
        model_out = adapter.write_calibrated(values, out, "strathe")
        results = {}
        new = f"{adapter.label} with Strathe's values"
        if split.warning:
            notes.insert(0, split.warning)
            print("  note: " + split.warning, flush=True)
        if held:
            print(f"Scoring the original and exported models on the held-out recordings "
                  f"({', '.join(split.heldout_groups)})...", flush=True)
            results, runs_wins = _score_models(adapter, held, [(f"{adapter.label}, original", None), (new, values)],
                                               work, densities)
            _strathe_own(results, runs_wins, pfile, mass)
            results.update(core.baselines(runs_wins))
            core.plot(results, os.path.join(out, "drift.png"), f"Unseen recordings: original vs exported {adapter.label} model")
        elif no_dvl:
            notes.append("Not scored: these logs have no DVL reading at every sample, and the export's score is "
                         "against logged velocity (SENSORS.md).")
        else:
            notes.append("Not scored: no held-out recordings (give --heldout-dates).")
        summary = {"simulator": adapter.sim, "calibration": os.path.abspath(pfile), "model": adapter.model_path,
                   "mass_kg": float(mass), "water_density": rho, "reference_point_m": list(a.reference_point),
                   "train_dates": split.train_groups, "heldout_dates": split.heldout_groups, "values": values,
                   "notes": notes, "exported_model_file": model_out, "heldout": results}
        yaml.safe_dump(json.loads(json.dumps(summary)), open(os.path.join(out, f"{adapter.sim}_export.yaml"), "w"),
                       sort_keys=False)
        L = [f"# Your Strathe calibration in {adapter.label}: {os.path.basename(adapter.model_path)}\n", "## Bottom line\n"]
        if results:
            L += _bottom_line(results, f"{adapter.label}, original", new, split.heldout_groups,
                              f"{adapter.label} with Strathe's values")
            L += _compare_section(results)
        L += ["## What changed in your model\n",
              f"`{os.path.basename(model_out)}` is `{os.path.basename(adapter.model_path)}` with only these values "
              f"changed:\n", *adapter.changes(values)]
        if notes:
            L += ["", "## Not carried, and why\n"] + [f"- {n}" for n in notes]
        open(os.path.join(out, f"{adapter.sim}_export_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
        print(f"Wrote {model_out}, {adapter.sim}_export.yaml and {adapter.sim}_export_report.md in {out}")
        _print_results(results)
        for n in notes:
            print("  note: " + n)
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)
