"""
B2.1 -- runs the actual Claim-A methodology through MuJoCo, using
parameters already fitted via the stand-in (does NOT re-fit against
MuJoCo -- evaluate-before-refit, the same approach used for HoloOcean's
first proof-of-concept). Evaluates held-out RMSE on PHASE1_HELDOUT_DATES
using mujoco_backend's simulate_fn -- the DEFAULT (forward-sim,
teacher_forced=False) mode, per the task's own literal wording.

Also computes the stand-in's OWN RMSE on the SAME held-out runs,
alongside MuJoCo's -- not because this is B2.2's full three-way
comparison (that also wants HoloOcean's already-recorded number, not
available here), but because MuJoCo's number alone is not interpretable
without SOME baseline for what "reasonable" looks like on this specific
held-out data.

Uses objective.py's own, unmodified rmse_for_run/mean_run_rmse --
already confirmed (B1.1) to work correctly with mujoco_backend's
simulate_fn without any changes to objective.py itself. This script
applies that confirmed integration to real held-out data for the first
time, rather than re-verifying the integration itself.

*** UNVERIFIED against real data in this sandbox -- no real CSV files,
no real calibrated_parameters_a1.yaml (mine differs from Abdoul's real
one, per this whole session's own repeated NOTE about differing cost
values), in this sandbox.
"""
import argparse

import numpy as np
import mujoco

import objective
from objective import DatasetConfig
from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec
from mujoco_backend import make_mujoco_simulate_fn

MJCF_PATH = "b0_single_body_placeholder.xml"


def run_b2_1(data_dir: str, sampling_rate_hz: float,
             calibrated_params_path: str = "calibrated_parameters_a1.yaml"):
    from data_split import discover_csv_files, extract_date
    from phase2_config import PHASE1_HELDOUT_DATES

    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters(calibrated_params_path)
    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)

    all_files, duplicates = discover_csv_files(data_dir)
    if duplicates:
        print(f"NOTE: {len(duplicates)} duplicate-named file(s) found, later ones skipped.")
    selected_files = [f for f in all_files if extract_date(f) in PHASE1_HELDOUT_DATES]
    if not selected_files:
        raise FileNotFoundError(
            f"No CSV files found under {data_dir!r} matching PHASE1_HELDOUT_DATES "
            f"{PHASE1_HELDOUT_DATES}. Found {len(all_files)} total files with dates: "
            f"{sorted(set(extract_date(f) for f in all_files))}"
        )
    print(f"PHASE1_HELDOUT_DATES: {PHASE1_HELDOUT_DATES}")
    print(f"Found {len(selected_files)} held-out run file(s):")
    for f in selected_files:
        print(f"  {f}")

    runs = [objective.load_run(f, dataset_config) for f in selected_files]

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco_simulate_fn = make_mujoco_simulate_fn(model, data)  # default: forward-sim, NOT teacher-forced

    print(f"\n{'run':<70} {'stand-in RMSE':>15} {'MuJoCo RMSE':>15}")
    per_run_results = []
    for run, filepath in zip(runs, selected_files):
        standin_rmse = objective.rmse_for_run(params, run, vehicle, simulate_fn=None)
        mujoco_rmse = objective.rmse_for_run(params, run, vehicle, simulate_fn=mujoco_simulate_fn)
        print(f"{filepath:<70} {standin_rmse:>15.6f} {mujoco_rmse:>15.6f}")
        per_run_results.append({"filepath": filepath, "standin_rmse": standin_rmse, "mujoco_rmse": mujoco_rmse})

    standin_mean_rmse = objective.mean_run_rmse(params, runs, vehicle, simulate_fn=None)
    # Fresh model/data for the aggregate call -- avoids any residual state
    # from the per-run loop above affecting this independent computation.
    model2 = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data2 = mujoco.MjData(model2)
    mujoco_simulate_fn2 = make_mujoco_simulate_fn(model2, data2)
    mujoco_mean_rmse = objective.mean_run_rmse(params, runs, vehicle, simulate_fn=mujoco_simulate_fn2)

    print(f"\n{'='*70}")
    print("B2.1 SUMMARY")
    print(f"Stand-in held-out RMSE (mean across {len(runs)} runs): {standin_mean_rmse:.6f}")
    print(f"MuJoCo   held-out RMSE (mean across {len(runs)} runs): {mujoco_mean_rmse:.6f}")
    ratio = mujoco_mean_rmse / standin_mean_rmse if standin_mean_rmse > 0 else float("inf")
    print(f"Ratio (MuJoCo / stand-in): {ratio:.2f}x")
    print()
    print("This is NOT B2.2's full three-way comparison (HoloOcean's own held-out RMSE")
    print("is not available here) -- it's the minimum context needed to interpret MuJoCo's")
    print("number at all. Per the teacher-forced-vs-forward-simulation caveat, exact")
    print("closeness is not expected -- this is a directional check: is MuJoCo's error")
    print("the same rough order of magnitude as the stand-in's own held-out RMSE, or is")
    print("something clearly broken (e.g. orders of magnitude larger, non-finite, or")
    print("wildly inconsistent across runs).")
    return per_run_results, standin_mean_rmse, mujoco_mean_rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--calibrated-params", default="calibrated_parameters_a1.yaml")
    args = parser.parse_args()
    run_b2_1(args.data_dir, args.sampling_rate_hz, args.calibrated_params)