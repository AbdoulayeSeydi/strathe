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
    ap.add_argument("--heldout-dates", nargs="+", help="recording dates to hold out (default: the latest 30%%)")
    ap.add_argument("--density-by-date", nargs="+", metavar="DATE=RHO",
                    help="water density per recording date (e.g. 27-03-2026=1000 for fresh water)")
    ap.add_argument("--compare-params", help="also score your Strathe calibration (calibrated_parameters.yaml)")
    ap.add_argument("--compare-mass", type=float, help="its vehicle mass, kg (default: the calibration's)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="simulator runs at once")
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
    ap.add_argument("--max-iterations", type=int, default=15, help="least-squares iterations (each: one Jacobian)")


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
    results["Strathe's own model"] = core.score(runs_wins, [[core.strathe_free_run(p, VehicleConstants(float(m)), run, a_, n)
                                                             for a_, n in wins] for run, wins in runs_wins])


def _bottom_line(results, original: str, new: str, held_groups, what: str) -> list:
    o, x = results[original], results[new]
    h = next((f"{h}s" for h in reversed(HORIZONS_S) if f"{h}s" in o and f"{h}s" in x), None)
    if not h:
        return ["Not scored: no held-out recording is long enough for a 60 s window.\n"]
    ch = 100 * (x[h]["drift_median_m"] - o[h]["drift_median_m"]) / max(o[h]["drift_median_m"], 1e-9)
    return [f"On recordings it never saw ({', '.join(held_groups)}), {what} has **{x[h]['drift_median_m']:.2f} m** of "
            f"position error after {h[:-1]} s, against **{o[h]['drift_median_m']:.2f} m** for your original model "
            f"({ch:+.0f}%). Check every horizon below: a change can help over a minute and cost over seconds.\n"]


def _compare_section(results) -> list:
    return ["## How the models compare on unseen recordings\n",
            f"Windows of {EVAL_WINDOW_S:g} s every {EVAL_STRIDE_S:g} s; each starts from the logged state and is driven "
            f"by the logged forces, the model turning and tilting on its own, in still water. Medians.\n",
            *core.table(results), "", "![Drift](drift.png)", "",
            "\"Keeps its starting velocity\" is a hard baseline over a closed-loop mission: the logged forces react to the "
            "real vehicle, so any open-loop model's error compounds. These numbers rank models; they don't promise "
            "minute-ahead prediction.\n"]


def _print_results(results) -> None:
    for name, r in results.items():
        hs = [f"{r[f'{h}s']['drift_median_m']:.2f}" for h in HORIZONS_S if f"{h}s" in r]
        print(f"  {name:36s} drift {' / '.join(hs)} m")


def run_fit(adapter, a) -> int:
    out = a.out
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix=f"strathe_{adapter.sim}_")
    try:
        train, held, split = core.load_runs(a.data, a.heldout_dates)
        if not held:
            raise SimFitError("no held-out recordings: give --heldout-dates, or logs from at least two sessions")
        adapter.dt = train[0].dt
        densities = _densities(a)
        notes = list(getattr(adapter, "notes", []))
        exported = None
        if a.from_strathe:
            from stonefish_export import load_calibration, training_density
            sp, smass, pfile = load_calibration(a.from_strathe, a.compare_mass)
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
        unknown = set(groups) - set(adapter.groups)
        if unknown:
            raise SimFitError(f"--fit {', '.join(sorted(unknown))}: the {adapter.label} model has "
                              f"{', '.join(adapter.groups)}")
        specs = adapter.knob_specs(info, groups, a.axisymmetric)
        notes = list(getattr(adapter, "notes", notes))
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

        def simulate(x, i, run, rows, d):
            model = adapter.write_runnable(knobs.values(x), os.path.join(d, f"m{i}"), densities.get(run.date))
            return adapter.replay(model, rows, os.path.join(d, f"r{i}"))
        f = core.fit(simulate, train, knobs, work, a.workers, a.max_iterations, log=lambda s: print(s, flush=True),
                     weighting=a.weighting)
        values = f["values"]
        model_out = adapter.write_calibrated(values, out, "calibrated")

        print("Scoring the original and calibrated models on the held-out recordings...", flush=True)
        orig, cal = f"{adapter.label}, original", f"{adapter.label}, calibrated by Strathe"
        base = adapter.base
        adapter.set_base(None)  # "original" is the lab's own model, not the --from-strathe start
        results, runs_wins = _score_models(adapter, held, [(orig, None)], work, densities)
        adapter.set_base(base)
        later = ([(f"{adapter.label} with Strathe's values", None)] if exported else []) + [(cal, values)]
        results.update(_score_models(adapter, held, later, work, densities)[0])
        if a.compare_params:
            _strathe_own(results, runs_wins, a.compare_params, a.compare_mass)
        results.update(core.baselines(runs_wins))

        start = {s.name: s.start for s in specs}
        summary = {"simulator": adapter.sim, "model": adapter.model_path, "reference_point_m": list(a.reference_point),
                   "fitted": sorted({s.group for s in specs}, key=groups.index), "axisymmetric": bool(a.axisymmetric),
                   "held": knobs.held, "weighting": a.weighting, "notes": notes,
                   "from_strathe": exported, "axis_scales": f["axis_scales"],
                   "before": start, "after": values, "train_dates": split.train_groups,
                   "heldout_dates": split.heldout_groups,
                   "fit": {k: f[k] for k in ("start_cost", "cost", "evaluations", "status")},
                   "info": info, "calibrated_model_file": model_out, "heldout": results}
        yaml.safe_dump(json.loads(json.dumps(summary)), open(os.path.join(out, f"{adapter.sim}_fit.yaml"), "w"),
                       sort_keys=False)
        core.plot(results, os.path.join(out, "drift.png"), f"Unseen recordings: original vs calibrated {adapter.label} model")
        L = [f"# {adapter.label} model calibrated from your logs: {os.path.basename(adapter.model_path)}\n",
             "## Bottom line\n", *_bottom_line(results, orig, cal, split.heldout_groups, "the calibrated model"),
             *_compare_section(results), "## What changed in your model\n",
             f"`{os.path.basename(model_out)}` is `{os.path.basename(adapter.model_path)}` with only these values "
             f"changed:\n", *adapter.changes(values)]
        if knobs.held:
            L.append("- held at your values: " + ", ".join(f"{n} = {v:g}" for n, v in knobs.held.items()))
        if exported:
            L.append(f"- started from your Strathe calibration (`{exported['calibration']}`), put into the model as "
                     f"export-sim does; the fit tuned the rest from there")
        L += ["", "## How the fit went\n",
              f"Fitted {len(knobs.names())} values on {', '.join(split.train_groups)}. Cost {f['start_cost']:.3f} -> "
              f"{f['cost']:.3f} after {f['evaluations']} evaluations ({f['status']})."]
        L += [f"- {n}" for n in notes]
        L += ["", "Fitting can trade one axis for another, or tailor values to the training sessions: odd values (a "
              "drag far from the shape's, a value at its limit) mean the logs don't pin them down. More varied "
              "manoeuvres at several speeds help."]
        open(os.path.join(out, f"{adapter.sim}_fit_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
        print(f"Wrote {model_out}, {adapter.sim}_fit.yaml and {adapter.sim}_fit_report.md in {out}")
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
        train, held, split = core.load_runs(data, heldout)
        adapter.dt = (train or held)[0].dt
        densities = _densities(a)
        info = adapter.info()
        rho = training_density(split.train_groups, densities, info.get("water_density") or 1025.0)
        print(f"Putting {os.path.basename(pfile)} into the {adapter.label} model {os.path.basename(adapter.model_path)} "
              f"(mass {mass:g} kg, water {rho:g} kg/m³)...", flush=True)
        values, notes = adapter.export_values(params, float(mass), info, train, rho)
        model_out = adapter.write_calibrated(values, out, "strathe")
        results = {}
        new = f"{adapter.label} with Strathe's values"
        if held:
            print(f"Scoring the original and exported models on the held-out recordings "
                  f"({', '.join(split.heldout_groups)})...", flush=True)
            results, runs_wins = _score_models(adapter, held, [(f"{adapter.label}, original", None), (new, values)],
                                               work, densities)
            _strathe_own(results, runs_wins, pfile, mass)
            results.update(core.baselines(runs_wins))
            core.plot(results, os.path.join(out, "drift.png"), f"Unseen recordings: original vs exported {adapter.label} model")
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
