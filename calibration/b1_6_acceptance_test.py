"""
B1.6 acceptance test -- formalizes the task's own literal bar
("confirm the resulting trajectories are still reasonable -- not
necessarily identical to the teacher-forced numbers") as a single,
permanent, re-runnable check, rather than leaving that claim scattered
across several ad-hoc diagnostic scripts from the investigation.

Reasonableness, per everything established in this investigation (see
mujoco_backend.py's own consolidated docstring for the full derivation):
  1. Finite output on every real run -- no NaN/inf.
  2. Velocity stays within a range a real BlueROV2 could physically
     achieve, regardless of how far orientation has drifted from the
     logged trajectory.

Explicitly NOT checked here (by design, not oversight): closeness to
the teacher-forced/logged trajectory. Large orientation drift is
expected and understood -- see the docstring's CONCLUSION section --
not a failure condition for this test.

*** UNVERIFIED against real data in this sandbox -- no real CSV files here.
"""
import argparse

import numpy as np
import mujoco

from objective import load_run, DatasetConfig
from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec
from dynamics_model import SimulationConfig, TrajectoryInputs
from mujoco_backend import simulate_trajectory_mujoco

MJCF_PATH = "b0_single_body_placeholder.xml"
MAX_PLAUSIBLE_LINEAR_SPEED = 5.0   # m/s
MAX_PLAUSIBLE_ANGULAR_SPEED = 10.0  # rad/s


def run_acceptance_test(data_dir: str, sampling_rate_hz: float, dates: list, physics_dt: float,
                          calibrated_params_path: str = "calibrated_parameters_a1.yaml"):
    from data_split import discover_csv_files, extract_date

    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters(calibrated_params_path)
    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)

    all_files, _ = discover_csv_files(data_dir)
    selected_files = [f for f in all_files if extract_date(f) in dates]
    if not selected_files:
        raise FileNotFoundError(f"No CSV files found under {data_dir!r} matching dates {dates}.")

    print(f"B1.6 acceptance test: {len(selected_files)} real run(s), dates {dates}")

    results = []
    for filepath in selected_files:
        run = load_run(filepath, dataset_config)
        sim_config = SimulationConfig(dt=run.dt, physics_dt=physics_dt)
        trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                        roll_pitch_series=run.roll_pitch)

        model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        data = mujoco.MjData(model)
        result = simulate_trajectory_mujoco(params, vehicle, sim_config, trajectory, model, data)

        finite = np.all(np.isfinite(result.nu))
        if finite:
            max_linear = np.max(np.linalg.norm(result.nu[:, :3], axis=1))
            max_angular = np.max(np.linalg.norm(result.nu[:, 3:], axis=1))
            plausible = max_linear < MAX_PLAUSIBLE_LINEAR_SPEED and max_angular < MAX_PLAUSIBLE_ANGULAR_SPEED
        else:
            max_linear = max_angular = float("nan")
            plausible = False

        passed = finite and plausible
        status = "PASS" if passed else "FAIL"
        print(f"  {status}  {filepath}: finite={finite}, "
              f"max_speed=({max_linear:.2f} m/s, {max_angular:.2f} rad/s)")
        results.append({"filepath": filepath, "finite": finite, "plausible": plausible, "passed": passed})

    n_passed = sum(1 for r in results if r["passed"])
    print(f"\n{n_passed}/{len(results)} runs passed (finite AND physically plausible velocity)")
    all_passed = n_passed == len(results)
    if all_passed:
        print("B1.6 ACCEPTANCE: PASSED")
    else:
        print("B1.6 ACCEPTANCE: FAILED -- investigate the specific failing run(s) above directly, "
              "don't assume this is the same already-understood drift pattern without checking.")
    return results, all_passed


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

    _, all_passed = run_acceptance_test(args.data_dir, args.sampling_rate_hz, dates, args.physics_dt, args.calibrated_params)
    raise SystemExit(0 if all_passed else 1)