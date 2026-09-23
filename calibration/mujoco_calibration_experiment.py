"""
Runs the ACTUAL least_squares fitting process against MuJoCo's
forward-sim simulate_fn, directly testing whether an optimizer searching
parameter space gets misled by drift-dominated residual noise -- the
specific concern raised in conversation that B2.1's single-evaluation
RMSE check (fixed, already-good parameters) could not itself test.

SCALE WARNING: MuJoCo's real per-step wall-clock cost has NOT yet been
measured (that's B2.3's separate job) -- HoloOcean's own backend needed
a ~400x-per-step overhead workaround for training. Given method="trf"'s
numerical Jacobian needs ~N+1 (N=22 free parameters) evaluations PER
ITERATION, a full-scale refit here could be extremely slow and is not
attempted. Defaults to a SMALL subset of training runs (--n-train-runs)
and a LOW --max-nfev -- explicitly a small-scale, DIRECTIONAL
experiment, not a production refit. Scale up only after seeing how long
this takes; the printed per-call timing gives an early, rougher signal
for that, though B2.3 remains the real measurement.

NOTE on optimizer.py's own train_rmse field (see conversation): this
script does NOT rely on CalibrationResult.train_rmse, since it may not
reflect the simulate_fn actually used for fitting. RMSE is recomputed
explicitly below via objective.mean_run_rmse with the correct
simulate_fn passed every time.

METHOD: runs calibrate() twice from the SAME initial guess (vehicle_spec
.yaml's own initial_guess values, matching the original calibration's
own starting point) -- once with simulate_fn=None (stand-in, as an
in-script control confirming this scaled-down setup still produces a
sensible fit) and once with simulate_fn=mujoco's forward-sim. Compares
the two fitted parameter vectors directly, then cross-evaluates both on
held-out data (PHASE1_HELDOUT_DATES) through BOTH simulate_fns (4
combinations total) to see whether a MuJoCo-fit set of parameters
actually generalizes, or is overfit to whatever MuJoCo-specific
artifacts (e.g. gimbal-lock-proximity drift) it minimized during
fitting.

*** UNVERIFIED against real data in this sandbox -- no real CSV files,
no real calibrated_parameters_a1.yaml, in this sandbox. This has NOT
been run against real data before being handed off.
"""
import argparse
import time

import numpy as np
import mujoco

import objective
from objective import DatasetConfig
from optimizer import (
    load_vehicle_spec, initial_parameters, CalibrationConfig, calibrate,
)
from mujoco_backend import make_mujoco_simulate_fn

MJCF_PATH = "b0_single_body_placeholder.xml"
VEHICLE_SPEC_PATH = "vehicle_spec.yaml"


def _load_runs(data_dir, dates, sampling_rate_hz, limit=None):
    from data_split import discover_csv_files, extract_date
    all_files, _ = discover_csv_files(data_dir)
    selected = sorted(f for f in all_files if extract_date(f) in dates)
    if limit is not None:
        selected = selected[:limit]
    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)
    runs = [objective.load_run(f, dataset_config) for f in selected]
    return runs, selected


def _make_progress_callback(label):
    start_time = time.monotonic()
    best = {"cost": float("inf"), "param_vec": None, "n_calls": 0}

    def on_iteration(call_count, param_vec, cost):
        best["n_calls"] = call_count
        if cost < best["cost"]:
            best["cost"] = cost
            best["param_vec"] = param_vec.copy()
        if call_count == 1 or call_count % 5 == 0:
            elapsed = time.monotonic() - start_time
            print(f"    [{label}] call {call_count:>4}: cost={cost:.6f}, "
                  f"best_so_far={best['cost']:.6f}, elapsed={elapsed:.1f}s")

    return on_iteration, best


def _calibrate_with_fallback(loaded, train_runs, config, best, label):
    # Matches build_calibrated_parameters.py's own established pattern
    # exactly (see conversation): calibrate() raises RuntimeError if no
    # restart meets scipy's strict convergence tolerances -- expected,
    # not a bug, per this project's own documented precedent (esp. given
    # MuJoCo's cost landscape has never been fit before, at ANY max_nfev).
    # Falls back to the best point seen during the search rather than
    # losing all fitting progress to an exception.
    from dynamics_model import CalibrationParameters

    class _FallbackResult:
        def __init__(self, parameters, cost, converged, nfev):
            self.parameters = parameters
            self.cost = cost
            self.converged = converged

            class _Raw:
                pass
            self.scipy_result = _Raw()
            self.scipy_result.nfev = nfev

    try:
        result = calibrate(loaded, train_runs, config)
        return result
    except RuntimeError as e:
        if best["param_vec"] is None:
            raise RuntimeError(
                f"[{label}] No usable point found at all (not even a non-converged best "
                f"point) -- something is genuinely wrong, not just an exhausted budget: {e}"
            ) from e
        print(f"  [{label}] Did not reach strict scipy convergence within max_nfev -- "
              f"per this project's own established precedent, using best point found "
              f"(cost={best['cost']:.6f}) instead.")
        params = CalibrationParameters.from_vector(best["param_vec"])
        return _FallbackResult(params, best["cost"], converged=False, nfev=best["n_calls"])


def run_experiment(data_dir: str, sampling_rate_hz: float, n_train_runs: int, max_nfev: int,
                    vehicle_spec_path: str = VEHICLE_SPEC_PATH):
    from phase2_config import PHASE1_TRAIN_DATES, PHASE1_HELDOUT_DATES

    loaded = load_vehicle_spec(vehicle_spec_path)
    vehicle = loaded.vehicle
    x0 = initial_parameters(loaded).to_vector()

    train_runs, train_files = _load_runs(data_dir, PHASE1_TRAIN_DATES, sampling_rate_hz, limit=n_train_runs)
    heldout_runs, heldout_files = _load_runs(data_dir, PHASE1_HELDOUT_DATES, sampling_rate_hz)

    print(f"Training on {len(train_runs)} run(s) (of {len(PHASE1_TRAIN_DATES)} available train dates, "
          f"capped at --n-train-runs={n_train_runs} for tractability):")
    for f in train_files:
        print(f"  {f}")
    print(f"Evaluating held-out generalization on all {len(heldout_runs)} PHASE1_HELDOUT_DATES runs.")
    print(f"max_nfev={max_nfev} (LOW, by design -- see this script's own docstring)")

    # --- Fit 1: stand-in, as an in-script control ---
    print("\n" + "=" * 70)
    print("FIT 1: stand-in (simulate_fn=None) -- control, confirms this scaled-down setup works")
    print("=" * 70)
    t0 = time.monotonic()
    standin_on_iteration, standin_best = _make_progress_callback("stand-in")
    standin_config = CalibrationConfig(
        max_nfev=max_nfev, simulate_fn=None, seed=0,
        on_iteration=standin_on_iteration,
    )
    standin_result = _calibrate_with_fallback(loaded, train_runs, standin_config, standin_best, "stand-in")
    standin_fit_time = time.monotonic() - t0
    print(f"  Done in {standin_fit_time:.1f}s, cost={standin_result.cost:.6f}, "
          f"converged={standin_result.converged}")

    # --- Fit 2: MuJoCo forward-sim -- the actual experiment ---
    print("\n" + "=" * 70)
    print("FIT 2: MuJoCo forward-sim -- the actual experiment")
    print("=" * 70)
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco_simulate_fn = make_mujoco_simulate_fn(model, data)
    t0 = time.monotonic()
    mujoco_on_iteration, mujoco_best = _make_progress_callback("mujoco")
    mujoco_config = CalibrationConfig(
        max_nfev=max_nfev, simulate_fn=mujoco_simulate_fn, seed=0,
        on_iteration=mujoco_on_iteration,
    )
    mujoco_result = _calibrate_with_fallback(loaded, train_runs, mujoco_config, mujoco_best, "mujoco")
    mujoco_fit_time = time.monotonic() - t0
    print(f"  Done in {mujoco_fit_time:.1f}s, cost={mujoco_result.cost:.6f}, "
          f"converged={mujoco_result.converged}")
    if mujoco_result.scipy_result.nfev > 0:
        per_call_estimate = mujoco_fit_time / mujoco_result.scipy_result.nfev
        print(f"  ~{per_call_estimate:.3f}s per real function evaluation (rough early signal only -- "
              f"B2.3 is the real per-step measurement)")

    # --- Compare fitted parameter vectors directly ---
    print("\n" + "=" * 70)
    print("PARAMETER COMPARISON: stand-in-fit vs MuJoCo-fit (both from the same x0)")
    print("=" * 70)
    param_names = initial_parameters(loaded).param_names
    standin_vec = standin_result.parameters.to_vector()
    mujoco_vec = mujoco_result.parameters.to_vector()
    print(f"{'param':<10} {'x0':>12} {'stand-in fit':>14} {'MuJoCo fit':>14} {'abs diff':>12}")
    for name, x0_val, sv, mv in zip(param_names, x0, standin_vec, mujoco_vec):
        print(f"{name:<10} {x0_val:>12.4f} {sv:>14.4f} {mv:>14.4f} {abs(sv - mv):>12.4f}")

    # --- Cross-evaluate both fits on held-out data through BOTH simulate_fns ---
    print("\n" + "=" * 70)
    print("HELD-OUT GENERALIZATION: 4-way cross-evaluation")
    print("=" * 70)
    model2 = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data2 = mujoco.MjData(model2)
    mujoco_simulate_fn_eval = make_mujoco_simulate_fn(model2, data2)

    combos = [
        ("stand-in fit", standin_result.parameters, "stand-in eval", None),
        ("stand-in fit", standin_result.parameters, "MuJoCo eval", mujoco_simulate_fn_eval),
        ("MuJoCo fit", mujoco_result.parameters, "stand-in eval", None),
        ("MuJoCo fit", mujoco_result.parameters, "MuJoCo eval", mujoco_simulate_fn_eval),
    ]
    print(f"{'fitted with':<14} {'evaluated with':<16} {'held-out RMSE':>15}")
    heldout_rmses = {}
    for fit_label, params, eval_label, sim_fn in combos:
        rmse = objective.mean_run_rmse(params, heldout_runs, vehicle, simulate_fn=sim_fn)
        print(f"{fit_label:<14} {eval_label:<16} {rmse:>15.6f}")
        heldout_rmses[(fit_label, eval_label)] = rmse

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("Look at the 'MuJoCo fit' + 'MuJoCo eval' row specifically: does it come in")
    print("close to 'stand-in fit' + 'stand-in eval' (the original methodology's own")
    print("apples-to-apples number)? If yes -- fitting THROUGH MuJoCo produces parameters")
    print("that generalize about as well as fitting through the stand-in normally does --")
    print("that's real, positive evidence for the calibration-suitability question.")
    print("If 'MuJoCo fit' + 'MuJoCo eval' is close but 'MuJoCo fit' + 'stand-in eval' is")
    print("much worse, that suggests the MuJoCo fit found a genuinely different parameter")
    print("regime that only looks good when evaluated the same way it was fit -- a real")
    print("overfitting-to-the-backend signal, not just fitting noise.")
    print("This is a SMALL-SCALE, single-seed run -- one directional data point, not a")
    print("definitive verdict. Re-run with different --n-train-runs or seeds before")
    print("treating any single result here as conclusive.")

    return {
        "standin_result": standin_result, "mujoco_result": mujoco_result,
        "standin_fit_time": standin_fit_time, "mujoco_fit_time": mujoco_fit_time,
        "heldout_rmses": heldout_rmses,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--n-train-runs", type=int, default=3,
                         help="Small by default -- see this script's own SCALE WARNING.")
    parser.add_argument("--max-nfev", type=int, default=20,
                         help="Low by default -- each real nfev is one full MuJoCo forward-sim "
                              "across all training runs.")
    args = parser.parse_args()
    run_experiment(args.data_dir, args.sampling_rate_hz, args.n_train_runs, args.max_nfev)