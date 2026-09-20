"""
A1.1, calibrated half — builds the CALIBRATED CalibrationParameters,
reusing the exact same discover_csv_files -> split_by_date -> load_run ->
calibrate sequence already proven in run_calibration.py /
run_full_pipeline.py — not a new, parallel implementation of that
plumbing.

## Why calibrate(), not calibrate_parallel()
A real, direct test against synthetic data (before this script was ever
handed over) reproduced the EXACT known "hidden per-iteration Jacobian-
evaluation multiplier" phenomenon PROJECT_STATUS.md's finding #14 already
documents for this exact model (22 free parameters, TRF, bounds): a
requested max_nfev=60 does not actually cap real function-call volume
anywhere near 60 — 560+ real, individually-normal-cost calls had still
not reached scipy's own termination check when manually interrupted.
calibrate_parallel() has no way to gracefully handle this: it explicitly
refuses on_iteration (checkpointing) since multiple processes can't safely
share one checkpoint file, so a run that exhausts its budget without
strict convergence just raises RuntimeError with nothing usable returned.
calibrate() supports on_iteration, so this script reuses
run_full_pipeline.py's own proven pattern instead of inventing a new
one: track the best (lowest-cost) point seen via on_iteration, and if
calibrate() raises because no restart hit strict convergence, fall back
to that best point rather than failing outright — exactly matching
run_full_pipeline.py's own documented reasoning that hitting max_nfev
without strict convergence is EXPECTED for a bounded budget, not a
failure.

Holds out PHASE1_HELDOUT_DATES exactly, via assert_uses_phase1_split(),
so this calibration is built on the SAME split as every other real result
in this project, per A1.1's explicit requirement. Additionally verifies
the resulting TRAIN dates match PHASE1_TRAIN_DATES exactly — not strictly
required by assert_uses_phase1_split() (which only checks the heldout
side), but a real, cheap extra check: if the data directory ever contains
an unexpected 5th date, this catches it rather than silently training on
data outside this project's known collection sessions.
"""
from pathlib import Path
from datetime import datetime, timezone
import time

from data_split import discover_csv_files, split_by_date
from dynamics_model import CalibrationParameters
from objective import DatasetConfig, load_run
from optimizer import load_vehicle_spec, initial_parameters, CalibrationConfig, calibrate
from phase2_config import PHASE1_TRAIN_DATES, PHASE1_HELDOUT_DATES, assert_uses_phase1_split

VEHICLE_SPEC_PATH = "vehicle_spec.yaml"
DEFAULT_SAMPLING_RATE_HZ = 10.0  # empirically determined for the real dataset — see
                                  # PROJECT_STATUS.md finding #11. Override if this
                                  # script is ever pointed at a different dataset.


def build_calibrated_parameters(data_dir: str, n_restarts: int = 1, max_nfev: int = 60,
                                 seed: int = 42, output_path: str = None,
                                 max_train_runs: int = None):
    """
    max_train_runs: truncates to this many training files, same protective
    mechanism run_full_pipeline.py already uses (--max-train-runs) after
    directly observing single evaluations on the FULL dataset take
    multiple hours. NOT optional-in-spirit for a first real run at this
    dataset's actual measured per-call cost (72.9s/call on the full 25
    training files, confirmed via time_real_eval_cost.py) — None (the
    default) uses everything, which is a genuine multi-hour-or-more
    commitment at that cost, not a quick check.
    """
    assert_uses_phase1_split(PHASE1_HELDOUT_DATES)

    csv_paths, duplicates = discover_csv_files(data_dir)
    if not csv_paths:
        raise RuntimeError(f"No CSV files found under {data_dir}")
    if duplicates:
        print(f"Note: {len(duplicates)} duplicate filename(s) skipped (kept first occurrence).")

    split = split_by_date(csv_paths, heldout_dates=PHASE1_HELDOUT_DATES)
    print(f"Train dates:   {split.train_groups}  ({split.n_train} files)")
    print(f"Heldout dates: {split.heldout_groups}  ({split.n_heldout} files)")

    expected_train = sorted(PHASE1_TRAIN_DATES)
    if sorted(split.train_groups) != expected_train:
        raise RuntimeError(
            f"Train-date mismatch: expected exactly {expected_train} after holding out "
            f"{PHASE1_HELDOUT_DATES}, but got {sorted(split.train_groups)}. This means "
            f"the data directory contains date(s) beyond this project's known 4 "
            f"collection dates — investigate before trusting this calibration run."
        )

    loaded_spec = load_vehicle_spec(VEHICLE_SPEC_PATH)
    dataset_config = DatasetConfig(sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ)

    train_files = split.train_files
    if max_train_runs is not None and max_train_runs < len(train_files):
        print(f"Truncating to {max_train_runs} of {len(train_files)} training files "
              f"(--max-train-runs) — this is what actually protects against a multi-hour "
              f"run, not a small --max-nfev alone. See run_full_pipeline.py's own identical "
              f"reasoning.")
        train_files = train_files[:max_train_runs]

    train_runs = [load_run(fp, dataset_config) for fp in train_files]
    print(f"Loaded {len(train_runs)} training run(s).")

    # Best-point tracking, matching run_full_pipeline.py's exact proven
    # pattern — see module docstring for why calibrate_parallel() can't do
    # this and calibrate() can.
    #
    # LIVE PROGRESS PRINTING — a real gap in the first version of this
    # script: at this dataset's confirmed real per-call cost (72.9s/call
    # on the full training set), a run can legitimately take many minutes
    # to hours with ZERO output if this callback doesn't print anything,
    # which is indistinguishable from a genuine hang. Every call now
    # prints elapsed time and current best cost, so "still working" and
    # "actually stuck" are never ambiguous again.
    best = {"cost": float("inf"), "param_vec": None}
    start_time = time.monotonic()

    def track_best(call_count, param_vec, cost):
        if cost < best["cost"]:
            best["cost"] = cost
            best["param_vec"] = param_vec.copy()
        elapsed = time.monotonic() - start_time
        print(f"  [call {call_count:>4}] elapsed={elapsed:>7.1f}s  "
              f"this_cost={cost:.6g}  best_cost_so_far={best['cost']:.6g}", flush=True)

    config = CalibrationConfig(
        loss="soft_l1",
        n_restarts=n_restarts,
        seed=seed,
        max_nfev=max_nfev,
        initialization_strategy="perturbed",
        on_iteration=track_best,
    )

    print(f"Running calibrate() with {n_restarts} restart(s), max_nfev={max_nfev} per restart...")
    print("(Real per-evaluation cost has already been confirmed normal on this dataset — "
          "see the module docstring. A real, completed run may still take a while: this "
          "project's own documented experience is that reaching strict scipy convergence "
          "for this 22-parameter model can require FAR more real evaluations than max_nfev "
          "suggests, per PROJECT_STATUS.md finding #14. That's expected, not a hang.)")
    try:
        result = calibrate(loaded_spec, train_runs, config)
        print(f"Converged: {result.converged}  cost={result.cost:.6g}  "
              f"restarts converged: {result.stability['n_converged']}/{result.stability['n_total']}")
        params = result.parameters
        converged = result.converged
        cost = result.cost
    except RuntimeError as e:
        if best["param_vec"] is None:
            raise RuntimeError(
                f"No usable point found at all (not even a non-converged best point) — "
                f"something is genuinely wrong, not just an exhausted budget: {e}"
            ) from e
        print(f"\nDid not reach strict scipy convergence within max_nfev={max_nfev} — "
              f"per run_full_pipeline.py's own established precedent, this is EXPECTED "
              f"for a bounded budget on this model, not a failure. Using the best point "
              f"found (cost={best['cost']:.6g}) instead. Raise --max-nfev / --n-restarts "
              f"for a more fully converged result.")
        params = CalibrationParameters.from_vector(best["param_vec"])
        converged = False
        cost = best["cost"]

    if output_path:
        _save_calibrated_parameters(params, output_path, converged=converged, cost=cost,
                                     data_dir=data_dir, n_restarts=n_restarts, max_nfev=max_nfev)
        print(f"\nSaved to {output_path}")

    return params


def _save_calibrated_parameters(params: CalibrationParameters, output_path: str,
                                 converged: bool, cost: float, data_dir: str,
                                 n_restarts: int, max_nfev: int) -> None:
    """Same shape run_full_pipeline.py already writes ({"parameters": {...}}),
    so anything already set up to read that format (or A1.2, once it exists)
    reads this file without needing a second, different loader — plus a
    provenance block so a LATER, more-converged re-run doesn't get silently
    confused with this quick one."""
    import yaml as _yaml
    from datetime import datetime, timezone

    params_dict = {name: float(getattr(params, name)) for name in params.param_names}
    payload = {
        "parameters": params_dict,
        "provenance": {
            "converged": converged,
            "cost": cost,
            "data_dir": data_dir,
            "n_restarts": n_restarts,
            "max_nfev": max_nfev,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "note": ("NOT a fully-converged result if converged=False above — "
                     "a quick A1.1 fit, not a Claim-A-grade calibration. Re-run "
                     "with a larger --max-nfev/--n-restarts for anything more rigorous."),
        },
    }
    with open(output_path, "w", encoding="utf-8") as f:
        _yaml.dump(payload, f, sort_keys=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Folder of real logged BlueROV2 CSVs")
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=60,
                         help="Per-restart budget. Kept small by default deliberately — "
                              "see the module docstring on why a larger number here doesn't "
                              "mean what it sounds like it means for this model.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-runs", type=int, default=5,
                         help="Truncates to this many training files, matching "
                              "run_full_pipeline.py's own default (5) and its own stated "
                              "reason: a single evaluation on the FULL dataset has been "
                              "confirmed to take 72.9s/call at this project's real scale — "
                              "with the confirmed real-call overshoot on top of that, the "
                              "full 25-file set is a genuine multi-hour-or-more commitment, "
                              "not a quick check. Pass a large number (e.g. 1000) to use "
                              "the full dataset once you're ready for that.")
    parser.add_argument("--output", default="calibrated_parameters_a1.yaml",
                         help="Where to save the result, so A1.2 can load it without "
                              "re-running this calibration every time. Pass '' to skip saving.")
    args = parser.parse_args()

    params = build_calibrated_parameters(
        args.data_dir, n_restarts=args.n_restarts, max_nfev=args.max_nfev, seed=args.seed,
        output_path=args.output or None, max_train_runs=args.max_train_runs,
    )
    print("\nCalibrated CalibrationParameters:")
    for name in params.param_names:
        print(f"  {name:16s} = {getattr(params, name)}")