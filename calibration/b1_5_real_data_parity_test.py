"""
B1.5 -- Single-body parity test, teacher-forced first, against REAL
logged runs and REAL MuJoCo. Unlike everything tested so far (either
isolated MuJoCo API spikes, or mujoco_backend.py's own unit tests
against a FAKE mujoco harness with synthetic random data), this is the
first genuine end-to-end test of this whole file: real data, real
physics engine, both at once.

For each of a handful of real logged runs, drives BOTH
dynamics_model.simulate_trajectory() (the stand-in reference) and
mujoco_backend.simulate_trajectory_mujoco() (real MuJoCo, teacher-forced
per B1.1's staging) on the IDENTICAL trajectory, and compares the
resulting velocity trajectories directly.

Per the task's own framing: teacher-forced orientation isolates whether
the force/damping computation itself matches the stand-in, before any
orientation-integration difference (B1.6's later concern) can confound
the comparison.

Reuses ONE model/data instance across every run, matching
make_mujoco_simulate_fn's own "reuse across every optimizer evaluation"
design intent -- this also implicitly tests that state correctly resets
between calls with REAL MuJoCo, not just the fake harness
test_shared_model_timestep_is_restored_after_the_call already covered.

*** UNVERIFIED -- no mujoco, and no real data files, in this sandbox.
The data-loading calls (discover_csv_files, load_run) reuse this
project's own existing, already-used utilities exactly as
build_calibrated_parameters.py and others already call them -- not new,
unverified code, but I have not personally run this specific script.
"""
import argparse

import numpy as np
import mujoco

from data_split import discover_csv_files, extract_date
from objective import load_run, DatasetConfig
from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec
from dynamics_model import SimulationConfig, TrajectoryInputs, simulate_trajectory
from mujoco_backend import simulate_trajectory_mujoco

MJCF_PATH = "b0_single_body_placeholder.xml"


def run_parity_test(data_dir: str, sampling_rate_hz: float, dates: list, physics_dt: float,
                     calibrated_params_path: str = "calibrated_parameters_a1.yaml"):
    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters(calibrated_params_path)
    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)

    all_files, duplicates = discover_csv_files(data_dir)
    if duplicates:
        print(f"NOTE: {len(duplicates)} duplicate-named file(s) found, later ones skipped: {duplicates}")

    selected_files = [f for f in all_files if extract_date(f) in dates]
    if not selected_files:
        raise FileNotFoundError(
            f"No CSV files found under {data_dir!r} matching dates {dates}. "
            f"Found {len(all_files)} total files with dates: "
            f"{sorted(set(extract_date(f) for f in all_files))}"
        )
    print(f"Found {len(selected_files)} real run file(s) matching {dates}:")
    for f in selected_files:
        print(f"  {f}")

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    results = []
    for filepath in selected_files:
        print(f"\n--- {filepath} ---")
        run = load_run(filepath, dataset_config)
        print(f"n_samples={run.n_samples}, dt={run.dt:.6f}, date={run.date}")

        sim_config = SimulationConfig(dt=run.dt, physics_dt=physics_dt)
        trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                        roll_pitch_series=run.roll_pitch)

        reference = simulate_trajectory(params, vehicle, sim_config, trajectory)
        mujoco_result = simulate_trajectory_mujoco(params, vehicle, sim_config, trajectory, model, data)

        diff = mujoco_result.nu - reference.nu
        max_abs_diff = np.max(np.abs(diff))
        rmse = np.sqrt(np.mean(diff ** 2))
        print(f"max abs diff: {max_abs_diff:.6e}")
        print(f"RMSE (mujoco vs stand-in): {rmse:.6e}")
        print(f"reference nu[-1]: {reference.nu[-1]}")
        print(f"mujoco nu[-1]:    {mujoco_result.nu[-1]}")

        results.append({"filepath": filepath, "max_abs_diff": max_abs_diff, "rmse": rmse,
                         "n_samples": run.n_samples})

    print(f"\n{'='*70}")
    print("B1.5 SUMMARY")
    for r in results:
        print(f"  {r['filepath']}: max_abs_diff={r['max_abs_diff']:.4e}, "
              f"rmse={r['rmse']:.4e}, n_samples={r['n_samples']}")
    max_over_all = max(r["max_abs_diff"] for r in results)
    print(f"\nOverall max abs diff across all runs: {max_over_all:.4e}")
    if max_over_all < 1e-6:
        print("PASSED: mujoco_backend matches the stand-in to near-machine "
              "precision on REAL data.")
    else:
        print("Review needed -- see individual run diffs above. This does NOT "
              "necessarily mean a bug: real trajectories may have a much larger "
              "n_substeps than the synthetic tests' small hand-picked examples, "
              "and floating-point accumulation over many more steps could "
              "plausibly explain a larger (but still small relative to the "
              "trajectory's own scale) gap. Compare max_abs_diff against the "
              "real_nu column's own typical magnitude before concluding "
              "anything is wrong.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, help="Directory containing real logged run CSVs")
    parser.add_argument("--sampling-rate-hz", type=float, required=True,
                         help="Real sampling rate for the logged data -- no default, must match the actual data")
    parser.add_argument("--dates", nargs="+", default=None,
                         help="Specific DD-MM-YYYY dates to test (default: PHASE1_TRAIN_DATES)")
    parser.add_argument("--physics-dt", type=float, default=0.005,
                         help="Sub-step physics resolution, matching SimulationConfig's own default")
    parser.add_argument("--calibrated-params", default="calibrated_parameters_a1.yaml")
    args = parser.parse_args()

    if args.dates is None:
        from phase2_config import PHASE1_TRAIN_DATES
        dates = PHASE1_TRAIN_DATES
        print(f"No --dates given, using PHASE1_TRAIN_DATES: {dates}")
    else:
        dates = args.dates

    run_parity_test(args.data_dir, args.sampling_rate_hz, dates, args.physics_dt, args.calibrated_params)