"""
Directly compares real-data forward-simulation results BEFORE and AFTER
the gimbal-lock fix (routing through compute_acceleration_from_rotation_matrix
directly, vs the old path of extracting roll/pitch via Euler decomposition
then calling compute_acceleration()).

METHOD: rather than hand-write a replica of the old loop (risking
divergence from what the old code actually did -- a mistake worth
avoiding explicitly), this temporarily monkey-patches
mujoco_backend.compute_acceleration_from_rotation_matrix with a wrapper
that first extracts roll/pitch from R (replicating the OLD behavior
exactly) and then calls the scalar compute_acceleration(). Everything
else in the real, unmodified simulate_trajectory_mujoco loop stays
identical -- only this one computation step differs between the two runs.

Prediction, stated before running (see conversation): g_eta itself was
verified mathematically identical between the two paths across a wide
range of angles, including 89.99 degrees pitch -- the one CONFIRMED
discrepancy is narrow, occurring only right at scipy's own gimbal-lock
detection threshold (~89.999999 degrees empirically). This means the fix
is expected to be small in its effect on real-data numbers, NOT a
resolution of the separately-confirmed, physics-inherent near-universal
gimbal-lock-proximity finding. This script exists to check that
prediction against real data, not to assume it.

*** UNVERIFIED against real data in this sandbox -- no real CSV files here.
"""
import argparse

import numpy as np
import mujoco

from objective import load_run, DatasetConfig
from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec
from dynamics_model import SimulationConfig, TrajectoryInputs, compute_acceleration
import mujoco_backend

MJCF_PATH = "b0_single_body_placeholder.xml"


def run_comparison(data_dir: str, sampling_rate_hz: float, dates: list, physics_dt: float,
                    calibrated_params_path: str = "calibrated_parameters_a1.yaml"):
    from data_split import discover_csv_files, extract_date

    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters(calibrated_params_path)
    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)

    all_files, _ = discover_csv_files(data_dir)
    selected_files = [f for f in all_files if extract_date(f) in dates]
    print(f"Found {len(selected_files)} real run file(s) matching {dates}")

    original_matrix_fn = mujoco_backend.compute_acceleration_from_rotation_matrix

    def old_style_via_extraction(params, vehicle, v, tau, R):
        # Replicates the OLD behavior exactly: extract roll/pitch from R
        # (discarding yaw, via the same Euler decomposition the original
        # code used), then call the scalar compute_acceleration().
        from scipy.spatial.transform import Rotation
        roll, pitch, _yaw = Rotation.from_matrix(R).as_euler("xyz")
        return compute_acceleration(params, vehicle, v, tau, roll, pitch)

    results = []
    for filepath in selected_files:
        run = load_run(filepath, dataset_config)
        sim_config = SimulationConfig(dt=run.dt, physics_dt=physics_dt)
        trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                        roll_pitch_series=run.roll_pitch)

        model_new = mujoco.MjModel.from_xml_path(MJCF_PATH)
        data_new = mujoco.MjData(model_new)
        new_result = mujoco_backend.simulate_trajectory_mujoco(params, vehicle, sim_config, trajectory, model_new, data_new)

        mujoco_backend.compute_acceleration_from_rotation_matrix = old_style_via_extraction
        try:
            model_old = mujoco.MjModel.from_xml_path(MJCF_PATH)
            data_old = mujoco.MjData(model_old)
            old_result = mujoco_backend.simulate_trajectory_mujoco(params, vehicle, sim_config, trajectory, model_old, data_old)
        finally:
            mujoco_backend.compute_acceleration_from_rotation_matrix = original_matrix_fn

        new_finite = np.all(np.isfinite(new_result.nu))
        old_finite = np.all(np.isfinite(old_result.nu))
        if new_finite and old_finite:
            diff = np.max(np.abs(new_result.nu - old_result.nu))
        else:
            diff = float("nan")

        print(f"{filepath}: new_finite={new_finite}, old_finite={old_finite}, "
              f"max_diff_new_vs_old={diff:.4e}" if new_finite and old_finite else
              f"{filepath}: new_finite={new_finite}, old_finite={old_finite}")
        results.append({"filepath": filepath, "new_finite": new_finite, "old_finite": old_finite, "diff": diff})

    print(f"\n{'='*70}")
    n_both_finite = sum(1 for r in results if r["new_finite"] and r["old_finite"])
    n_new_finite_old_not = sum(1 for r in results if r["new_finite"] and not r["old_finite"])
    print(f"{n_both_finite}/{len(results)} runs: both old and new finite")
    print(f"{n_new_finite_old_not}/{len(results)} runs: NEW finite but OLD was not (fix mattered directly)")
    finite_diffs = [r["diff"] for r in results if r["new_finite"] and r["old_finite"]]
    if finite_diffs:
        print(f"Among both-finite runs: median diff={np.median(finite_diffs):.4e}, "
              f"max diff={np.max(finite_diffs):.4e}")
    print("\nInterpretation: if n_new_finite_old_not > 0, the fix directly prevented at least one "
          "real divergence -- concrete, unambiguous value. If both stayed finite everywhere with small "
          "diffs, that would confirm the prediction that this specific fix is a small, narrow correction "
          "rather than a resolution of the separately-confirmed, physics-inherent instability -- still "
          "worth having (it's still eliminating a confirmed real glitch), just not the whole story.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--dates", nargs="+", default=None)
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--calibrated-params", default="calibrated_parameters_a1.yaml")
    args = parser.parse_args()

    if args.dates is None:
        from phase2_config import PHASE1_TRAIN_DATES
        dates = PHASE1_TRAIN_DATES
    else:
        dates = args.dates

    run_comparison(args.data_dir, args.sampling_rate_hz, dates, args.physics_dt, args.calibrated_params)