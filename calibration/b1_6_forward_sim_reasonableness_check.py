"""
B1.6 -- genuine forward simulation, real-data reasonableness check. Per
the task's own explicit framing (matching dynamics_model.py's own
documented caveat): forward-simulated results are NOT expected to match
the teacher-forced parity numbers exactly, since MuJoCo is now
self-integrating orientation from real physics rather than being pinned
to logged values every step. This script checks that the results are
still REASONABLE, not that they're identical to B1.5's own confirmed
parity numbers.

For each real run: runs BOTH teacher_forced=True (B1.5's own confirmed
path) and the new default (genuine forward simulation), and reports:
  - Whether forward-sim produces finite output at all (the most basic bar).
  - How far the two diverge, and whether that divergence GROWS smoothly
    over time (expected, since orientation-integration error compounds)
    or jumps suddenly (would suggest a real bug, not expected drift).
  - How far the SELF-INTEGRATED roll/pitch drifts from the LOGGED
    roll/pitch over the trajectory -- this is the direct, physical
    meaning of "genuine forward simulation": MuJoCo's own orientation,
    evolved from real angular velocity, versus what was actually logged.

*** UNVERIFIED -- no mujoco, and no real data files, in this sandbox.
"""
import argparse

import numpy as np
import mujoco

from objective import load_run, DatasetConfig
from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec
from dynamics_model import SimulationConfig, TrajectoryInputs
from mujoco_backend import simulate_trajectory_mujoco, _extract_roll_pitch_from_mujoco_quat

MJCF_PATH = "b0_single_body_placeholder.xml"


def run_check(data_dir: str, sampling_rate_hz: float, dates: list, physics_dt: float,
              calibrated_params_path: str = "calibrated_parameters_a1.yaml"):
    from data_split import discover_csv_files, extract_date

    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters(calibrated_params_path)
    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)

    all_files, _ = discover_csv_files(data_dir)
    selected_files = [f for f in all_files if extract_date(f) in dates]
    if not selected_files:
        raise FileNotFoundError(f"No CSV files found under {data_dir!r} matching dates {dates}.")

    print(f"Found {len(selected_files)} real run file(s) matching {dates}")

    results = []
    for filepath in selected_files:
        print(f"\n--- {filepath} ---")
        run = load_run(filepath, dataset_config)
        sim_config = SimulationConfig(dt=run.dt, physics_dt=physics_dt)
        trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                        roll_pitch_series=run.roll_pitch)

        model_tf = mujoco.MjModel.from_xml_path(MJCF_PATH)
        data_tf = mujoco.MjData(model_tf)
        teacher_forced_result = simulate_trajectory_mujoco(
            params, vehicle, sim_config, trajectory, model_tf, data_tf, teacher_forced=True)

        model_fwd = mujoco.MjModel.from_xml_path(MJCF_PATH)
        data_fwd = mujoco.MjData(model_fwd)
        forward_result = simulate_trajectory_mujoco(params, vehicle, sim_config, trajectory, model_fwd, data_fwd)

        finite = np.all(np.isfinite(forward_result.nu))
        print(f"n_samples={run.n_samples}, forward-sim finite: {finite}")
        if not finite:
            print("NOT FINITE -- skipping further comparison for this run")
            results.append({"filepath": filepath, "finite": False})
            continue

        diff = np.abs(forward_result.nu - teacher_forced_result.nu)
        max_diff_per_step = np.max(diff, axis=1)

        # Physical plausibility, independent of any comparison to
        # teacher-forced: does the self-integrated velocity itself stay
        # in a range a real BlueROV2 could actually achieve, regardless
        # of how far orientation has drifted from the logged values?
        max_linear_speed = np.max(np.linalg.norm(forward_result.nu[:, :3], axis=1))
        max_angular_speed = np.max(np.linalg.norm(forward_result.nu[:, 3:], axis=1))
        plausible = max_linear_speed < 5.0 and max_angular_speed < 10.0

        final_roll, final_pitch = _extract_roll_pitch_from_mujoco_quat(data_fwd)
        logged_final_roll, logged_final_pitch = run.roll_pitch[-1]
        roll_drift_deg = np.degrees(abs(final_roll - logged_final_roll))
        pitch_drift_deg = np.degrees(abs(final_pitch - logged_final_pitch))

        print(f"max abs diff vs teacher-forced: {np.max(max_diff_per_step):.4e}")
        print(f"diff at 25%/50%/75%/100% through the trajectory: "
              f"{max_diff_per_step[len(max_diff_per_step)//4]:.4e} / "
              f"{max_diff_per_step[len(max_diff_per_step)//2]:.4e} / "
              f"{max_diff_per_step[3*len(max_diff_per_step)//4]:.4e} / "
              f"{max_diff_per_step[-1]:.4e}")
        print(f"max linear speed reached: {max_linear_speed:.3f} m/s, "
              f"max angular speed reached: {max_angular_speed:.3f} rad/s "
              f"({'PLAUSIBLE' if plausible else 'IMPLAUSIBLE -- check this run directly'})")
        print(f"self-integrated final roll/pitch: {np.degrees(final_roll):.2f} deg / {np.degrees(final_pitch):.2f} deg")
        print(f"logged final roll/pitch:          {np.degrees(logged_final_roll):.2f} deg / {np.degrees(logged_final_pitch):.2f} deg")
        print(f"drift from logged orientation: roll={roll_drift_deg:.2f} deg, pitch={pitch_drift_deg:.2f} deg")

        results.append({
            "filepath": filepath, "finite": True,
            "max_diff": np.max(max_diff_per_step),
            "roll_drift_deg": roll_drift_deg, "pitch_drift_deg": pitch_drift_deg,
            "max_linear_speed": max_linear_speed, "max_angular_speed": max_angular_speed,
            "plausible": plausible,
            "n_samples": run.n_samples,
        })

    print(f"\n{'='*70}")
    print("B1.6 SUMMARY")
    n_finite = sum(1 for r in results if r["finite"])
    print(f"{n_finite}/{len(results)} runs produced finite forward-simulated output")
    n_plausible = sum(1 for r in results if r["finite"] and r["plausible"])
    print(f"{n_plausible}/{n_finite} finite runs stayed within physically plausible velocity bounds "
          f"(<5 m/s linear, <10 rad/s angular)")
    for r in results:
        if r["finite"]:
            flag = "" if r["plausible"] else "  <-- IMPLAUSIBLE VELOCITY"
            print(f"  {r['filepath']}: max_diff_vs_teacher_forced={r['max_diff']:.4e}, "
                  f"orientation_drift=({r['roll_drift_deg']:.1f}, {r['pitch_drift_deg']:.1f}) deg, "
                  f"max_speed=({r['max_linear_speed']:.2f} m/s, {r['max_angular_speed']:.2f} rad/s), "
                  f"n_samples={r['n_samples']}{flag}")
        else:
            print(f"  {r['filepath']}: NOT FINITE")
    print("\nCorrection to an earlier, overconfident claim: I previously said a divergence that "
          "'grows smoothly' would be the healthy pattern and a sudden jump would be a red flag. "
          "Real results often don't look like either -- they can rise and fall non-monotonically "
          "as two similar-but-not-identical oscillating trajectories (roll/pitch restoring moments "
          "behave somewhat like a pendulum) drift in and out of phase. That's a plausible, benign "
          "explanation for non-monotonic patterns, not a confirmed one -- I have not independently "
          "verified it. The physical-plausibility check above (velocity staying in a range a real "
          "BlueROV2 could achieve) is a more direct signal than the shape of the divergence curve: "
          "a plausible run with large orientation drift is far less concerning than an implausible "
          "one, regardless of how the divergence curve looks.")
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