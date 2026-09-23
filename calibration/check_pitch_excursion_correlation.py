"""
Tests the gimbal-lock-proximity hypothesis directly: does the WILD
variation in B1.6's orientation drift across real runs (some runs drift
<10 degrees, others 100+) correlate with how close the vehicle's
self-integrated orientation actually comes to pitch=+/-90 during the
trajectory -- not just at the final step (already showed a strong 0.861
correlation using only final-pitch values, see conversation), but at
ANY point, since a run could swing through high pitch and back,
understating the true peak if only the final value is checked.

METHOD: wraps mujoco_backend._current_rotation_matrix with a spy that
extracts (roll, pitch) from the returned R for REPORTING purposes only,
then calls through to the REAL, UNMODIFIED function -- this measures the
actual trajectory without any risk of a hand-written loop replica
diverging from the real code path (a mistake worth avoiding explicitly,
having made it before this session).

CORRECTED (see conversation): an earlier version of this script spied on
_extract_roll_pitch_from_mujoco_quat, which was the right target when
this script was first written. The subsequent gimbal-lock fix changed
the forward-sim loop to call _current_rotation_matrix directly instead,
bypassing that function entirely -- meaning the old spy silently
recorded nothing (n_substep_readings=0, all-nan output) rather than
failing loudly. Fixed to spy on the function actually called now.

This is NOT proposing a fix inside mujoco_backend.py. If confirmed, the
mechanism lives in compute_acceleration()'s own roll/pitch scalar
interface (dynamics_model.py), shared by the stand-in, HoloOcean's
backend, and this MuJoCo backend alike -- not something patchable in
this file specifically.

*** UNVERIFIED -- no mujoco, and no real data files, in this sandbox.
"""
import argparse

import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

from objective import load_run, DatasetConfig
from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec
from dynamics_model import SimulationConfig, TrajectoryInputs
import mujoco_backend

MJCF_PATH = "b0_single_body_placeholder.xml"


def run_check(data_dir: str, sampling_rate_hz: float, dates: list, physics_dt: float,
              calibrated_params_path: str = "calibrated_parameters_a1.yaml"):
    from data_split import discover_csv_files, extract_date

    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters(calibrated_params_path)
    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)

    all_files, _ = discover_csv_files(data_dir)
    selected_files = [f for f in all_files if extract_date(f) in dates]
    print(f"Found {len(selected_files)} real run file(s) matching {dates}")

    original_fn = mujoco_backend._current_rotation_matrix
    recorded_pitches = []

    def spy(data):
        R = original_fn(data)
        roll, pitch, _yaw = Rotation.from_matrix(R).as_euler("xyz")
        recorded_pitches.append(pitch)
        return R

    results = []
    try:
        mujoco_backend._current_rotation_matrix = spy

        for filepath in selected_files:
            run = load_run(filepath, dataset_config)
            sim_config = SimulationConfig(dt=run.dt, physics_dt=physics_dt)
            trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                            roll_pitch_series=run.roll_pitch)

            recorded_pitches.clear()
            model = mujoco.MjModel.from_xml_path(MJCF_PATH)
            data = mujoco.MjData(model)
            forward_result = mujoco_backend.simulate_trajectory_mujoco(
                params, vehicle, sim_config, trajectory, model, data)

            finite = np.all(np.isfinite(forward_result.nu))
            peak_abs_pitch_deg = np.degrees(np.max(np.abs(recorded_pitches))) if recorded_pitches else float("nan")
            final_abs_pitch_deg = np.degrees(abs(recorded_pitches[-1])) if recorded_pitches else float("nan")

            print(f"{filepath}: finite={finite}, peak|pitch|={peak_abs_pitch_deg:.2f} deg, "
                  f"final|pitch|={final_abs_pitch_deg:.2f} deg "
                  f"(n_substep_readings={len(recorded_pitches)})")

            results.append({
                "filepath": filepath, "finite": finite,
                "peak_abs_pitch_deg": peak_abs_pitch_deg,
                "final_abs_pitch_deg": final_abs_pitch_deg,
            })
    finally:
        mujoco_backend._current_rotation_matrix = original_fn

    print(f"\n{'='*70}")
    print("Sorted by peak |pitch| reached during the trajectory (not just final):")
    for r in sorted(results, key=lambda r: r["peak_abs_pitch_deg"]):
        near_lock = " <-- within 20 deg of gimbal lock at some point" if r["peak_abs_pitch_deg"] > 70 else ""
        print(f"  {r['filepath']}: peak|pitch|={r['peak_abs_pitch_deg']:6.2f} deg, "
              f"final|pitch|={r['final_abs_pitch_deg']:6.2f} deg{near_lock}")

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

    run_check(args.data_dir, args.sampling_rate_hz, dates, args.physics_dt, args.calibrated_params)