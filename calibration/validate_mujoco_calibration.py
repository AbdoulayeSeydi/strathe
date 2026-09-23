"""
Validates a new MuJoCo-fitted calibration against calibrated_parameters_a1.yaml,
per the specific usability criteria discussed in conversation: convergence
alone isn't sufficient, physically plausible parameters and held-out
predictive quality matter too.

PART 1: automated bound-pinning check -- which parameters landed at or
very near their real bounds (from vehicle_spec.yaml, via
optimizer.parameter_bounds -- NOT eyeballed round numbers, which can be
wrong when a bound itself isn't a round number).

PART 2: teacher-forced held-out RMSE, for BOTH parameter sets, on the
SAME PHASE1_HELDOUT_DATES -- this is the one check from the plan that
didn't already exist as a script: b2_1_heldout_rmse.py uses forward-sim
(correct for that script's own purpose), but "does the calibration
actually generalize under the SAME mode it was fit in" needs
teacher_forced=True specifically (mujoco_backend.make_mujoco_calibration_simulate_fn).

Does NOT check forward-simulation stability -- that's already
b1_6_acceptance_test.py's own job; run it separately, once per parameter
file, rather than duplicating that logic here:
    python3 b1_6_acceptance_test.py --data-dir <dir> --sampling-rate-hz 10.0 \
        --calibrated-params <path>

*** UNVERIFIED against real data in this sandbox -- no real CSV files or
real calibrated_parameters_a1.yaml / calibration_output_mujoco_full in
this sandbox.
"""
import argparse

import numpy as np
import yaml
import mujoco

import objective
from objective import DatasetConfig
from optimizer import load_vehicle_spec, parameter_bounds, initial_parameters
from dynamics_model import CalibrationParameters
from mujoco_backend import make_mujoco_calibration_simulate_fn

MJCF_PATH = "b0_single_body_placeholder.xml"
VEHICLE_SPEC_PATH = "vehicle_spec.yaml"
NEAR_BOUND_REL_TOL = 1e-3  # within 0.1% of the bound counts as "pinned"


def load_params_yaml(path: str) -> CalibrationParameters:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "parameters" in data:
        return CalibrationParameters(**data["parameters"])
    if "best_parameters_so_far" in data:
        return CalibrationParameters(**data["best_parameters_so_far"])
    raise ValueError(f"{path}: no 'parameters' or 'best_parameters_so_far' key found.")


def check_bound_pinning(params: CalibrationParameters, loaded_spec, label: str):
    """Reports two DISTINCT things, not conflated into one: parameters
    AT/NEAR a bound (within NEAR_BOUND_REL_TOL of it -- "pinned"), and
    parameters OUTSIDE the given range entirely (only possible when
    checking a parameter set against a DIFFERENT spec than the one it
    was actually fit under -- e.g. a widened-bounds fit checked against
    the ORIGINAL, narrower spec). These are different findings: "pinned"
    means the optimizer wants more room; "outside" means it already TOOK
    more room than this spec would have allowed. A value can be far
    outside a range without being "near" either of its edges (confirmed
    directly: -120 vs bounds [-80, -1.6] is 51% past the lower bound in
    relative terms, nowhere near NEAR_BOUND_REL_TOL) -- conflating the
    two into a single check under-reports exactly the case a widened-vs-
    original comparison exists to catch."""
    lb, ub = parameter_bounds(loaded_spec)
    param_names = initial_parameters(loaded_spec).param_names
    vec = params.to_vector()

    print(f"\n--- Bound check: {label} ---")
    n_pinned = 0
    n_outside = 0
    for name, val, low, high in zip(param_names, vec, lb, ub):
        width = high - low
        dist_to_low = abs(val - low) / width if width > 0 else 0
        dist_to_high = abs(val - high) / width if width > 0 else 0
        if val < low - 1e-9:
            print(f"  {name:<16} = {val:>12.4f}  <-- OUTSIDE range, past LOWER bound ({low})")
            n_outside += 1
        elif val > high + 1e-9:
            print(f"  {name:<16} = {val:>12.4f}  <-- OUTSIDE range, past UPPER bound ({high})")
            n_outside += 1
        elif dist_to_low < NEAR_BOUND_REL_TOL:
            print(f"  {name:<16} = {val:>12.4f}  <-- AT LOWER BOUND ({low})")
            n_pinned += 1
        elif dist_to_high < NEAR_BOUND_REL_TOL:
            print(f"  {name:<16} = {val:>12.4f}  <-- AT UPPER BOUND ({high})")
            n_pinned += 1
    print(f"  {n_pinned}/{len(param_names)} pinned at a bound, {n_outside}/{len(param_names)} outside the range entirely.")
    return n_pinned, n_outside


def teacher_forced_heldout_rmse(params, vehicle, heldout_runs, simulate_fn, label):
    per_run = []
    for run in heldout_runs:
        rmse = objective.rmse_for_run(params, run, vehicle, simulate_fn=simulate_fn)
        per_run.append(rmse)
    mean_rmse = float(np.mean(per_run))
    print(f"  {label}: mean held-out RMSE (teacher_forced=True) = {mean_rmse:.6f}")
    return mean_rmse, per_run


def run_validation(data_dir: str, sampling_rate_hz: float,
                    a1_params_path: str, new_params_path: str,
                    vehicle_spec_path: str = VEHICLE_SPEC_PATH,
                    widened_params_path: str = None, widened_spec_path: str = None):
    from data_split import discover_csv_files, extract_date
    from phase2_config import PHASE1_HELDOUT_DATES

    loaded_spec = load_vehicle_spec(vehicle_spec_path)
    vehicle = loaded_spec.vehicle

    a1_params = load_params_yaml(a1_params_path)
    new_params = load_params_yaml(new_params_path)
    widened_params = load_params_yaml(widened_params_path) if widened_params_path else None

    print("=" * 70)
    print("PART 1: BOUND-PINNING CHECK")
    print("=" * 70)
    n_pinned_a1, n_outside_a1 = check_bound_pinning(a1_params, loaded_spec, f"A1 ({a1_params_path})")
    n_pinned_new, n_outside_new = check_bound_pinning(new_params, loaded_spec, f"NEW ({new_params_path})")
    n_pinned_widened = None
    n_pinned_widened_vs_original = n_outside_widened_vs_original = None
    if widened_params is not None:
        # Two DIFFERENT checks, deliberately: against the WIDENED spec's own
        # (wider) bounds -- does it still want even more room -- and against
        # the ORIGINAL (narrower) spec's bounds -- confirms how far past the
        # old limits it actually moved, which is the number that matters for
        # the A/B/C classification, not just its status under its own bounds.
        widened_spec = load_vehicle_spec(widened_spec_path) if widened_spec_path else loaded_spec
        n_pinned_widened, _ = check_bound_pinning(
            widened_params, widened_spec, f"WIDENED, vs ITS OWN (widened) bounds ({widened_params_path})")
        n_pinned_widened_vs_original, n_outside_widened_vs_original = check_bound_pinning(
            widened_params, loaded_spec, f"WIDENED, vs ORIGINAL (narrow) bounds ({widened_params_path})")

    print("\n" + "=" * 70)
    print("PART 2: TEACHER-FORCED HELD-OUT RMSE")
    print("=" * 70)
    all_files, _ = discover_csv_files(data_dir)
    heldout_files = [f for f in all_files if extract_date(f) in PHASE1_HELDOUT_DATES]
    if not heldout_files:
        raise FileNotFoundError(f"No files found matching PHASE1_HELDOUT_DATES {PHASE1_HELDOUT_DATES}")
    print(f"Held-out runs: {len(heldout_files)} (dates: {PHASE1_HELDOUT_DATES})")

    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)
    heldout_runs = [objective.load_run(f, dataset_config) for f in heldout_files]

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    simulate_fn = make_mujoco_calibration_simulate_fn(model, data)

    a1_rmse, a1_per_run = teacher_forced_heldout_rmse(a1_params, vehicle, heldout_runs, simulate_fn, "A1")
    new_rmse, new_per_run = teacher_forced_heldout_rmse(new_params, vehicle, heldout_runs, simulate_fn, "NEW")
    widened_rmse = None
    if widened_params is not None:
        widened_rmse, widened_per_run = teacher_forced_heldout_rmse(
            widened_params, vehicle, heldout_runs, simulate_fn, "WIDENED")

    print(f"\n{'':20} {'A1':>12} {'NEW':>12} {'WIDENED':>12}")
    row = f"{'held-out RMSE':<20} {a1_rmse:>12.6f} {new_rmse:>12.6f}"
    if widened_rmse is not None:
        row += f" {widened_rmse:>12.6f}"
    print(row)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Bound-pinning (vs. own fitting bounds): A1 {n_pinned_a1}/22, NEW {n_pinned_new}/22"
          + (f", WIDENED {n_pinned_widened}/22" if n_pinned_widened is not None else ""))
    if n_pinned_widened_vs_original is not None:
        print(f"WIDENED vs. ORIGINAL (narrow) bounds: {n_pinned_widened_vs_original}/22 pinned near the "
              f"old edge, {n_outside_widened_vs_original}/22 moved COMPLETELY PAST the old range -- "
              f"this second number is the one that shows how much room the widening actually used.")

    if new_rmse < a1_rmse:
        print(f"NEW vs A1 held-out RMSE: improves by {(a1_rmse - new_rmse) / a1_rmse * 100:.1f}%.")
    else:
        print(f"NEW vs A1 held-out RMSE: WORSE by {(new_rmse - a1_rmse) / a1_rmse * 100:.1f}%.")

    if widened_rmse is not None:
        delta_pct = (new_rmse - widened_rmse) / new_rmse * 100
        if widened_rmse < new_rmse:
            print(f"WIDENED vs NEW held-out RMSE: further improves by {delta_pct:.1f}%.")
        else:
            print(f"WIDENED vs NEW held-out RMSE: WORSE by {-delta_pct:.1f}% -- the extra bound room "
                  f"may be overfitting to the training subset rather than generalizing.")
        print("\nIf WIDENED's held-out RMSE improved only slightly (or got worse) despite parameters")
        print("moving substantially past their old bounds, that's real evidence these parameters")
        print("are weakly identified by this data -- not that the original bounds were wrong. If")
        print("held-out RMSE improved by a comparable margin to the training RMSE gain, that's a")
        print("stronger case the original bounds were genuinely constraining.")

    print("\nThis does NOT check forward-simulation stability -- run b1_6_acceptance_test.py "
          "separately for each parameter file for that (see this script's own docstring).")

    return {
        "n_pinned_a1": n_pinned_a1, "n_pinned_new": n_pinned_new, "n_pinned_widened": n_pinned_widened,
        "n_pinned_widened_vs_original": n_pinned_widened_vs_original,
        "a1_heldout_rmse": a1_rmse, "new_heldout_rmse": new_rmse, "widened_heldout_rmse": widened_rmse,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--a1-params", default="calibrated_parameters_a1.yaml")
    parser.add_argument("--new-params", required=True,
                         help="e.g. ../calibration_output_mujoco_full/calibrated_parameters.yaml")
    parser.add_argument("--widened-params", default=None,
                         help="Optional third parameter set, e.g. "
                              "../calibration_output_mujoco_widened_bounds/calibrated_parameters.yaml")
    parser.add_argument("--widened-spec", default="vehicle_spec_widened_bounds.yaml",
                         help="Used only if --widened-params is given -- the spec its OWN bounds "
                              "were fit under, to check whether it still wants even more room.")
    args = parser.parse_args()
    run_validation(args.data_dir, args.sampling_rate_hz, args.a1_params, args.new_params,
                   widened_params_path=args.widened_params, widened_spec_path=args.widened_spec)