"""
Diagnoses WHERE and under what conditions simulate_trajectory_mujoco
diverges from the stand-in on a SPECIFIC real run, for the runs that
showed large/catastrophic divergence in b1_5_real_data_parity_test.py.

Does NOT assume which of several candidate explanations is right:
  - Compounding floating-point error over far more steps than any prior
    test (real runs: 2000-6000 samples x ~20 substeps/sample by default
    = 40,000-120,000+ total MuJoCo steps, vs a maximum of ~100 in any
    test in this whole session before now).
  - Euler-angle gimbal lock: this file's roll/pitch/yaw=0 quaternion
    construction uses 'xyz' intrinsic Euler angles, numerically
    ill-conditioned near pitch=+/-90 degrees -- checked directly below,
    not assumed either way.
  - An undiscovered bug that only manifests under real-world conditions
    (large/rapid rotation, extreme forces, etc.) this session's
    synthetic/hand-picked tests never happened to exercise.

METHOD: finds the single step where the ABSOLUTE difference between
mujoco_backend's and the stand-in's nu makes its LARGEST jump from one
step to the next -- threshold-free (no tuned cutoff to get wrong),
reveals whether the divergence is SUDDEN (bug/instability-like) or
GRADUAL (compounding-error-like) directly from the data. Reports the
full state (roll, pitch, tau, both nu vectors) at and around that point,
plus the roll/pitch range across the whole trajectory.
"""
import argparse

import numpy as np
import mujoco

from objective import load_run, DatasetConfig
from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec
from dynamics_model import SimulationConfig, TrajectoryInputs, simulate_trajectory
from mujoco_backend import simulate_trajectory_mujoco

MJCF_PATH = "b0_single_body_placeholder.xml"


def diagnose(filepath: str, sampling_rate_hz: float, physics_dt: float,
             calibrated_params_path: str = "calibrated_parameters_a1.yaml"):
    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters(calibrated_params_path)
    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)

    run = load_run(filepath, dataset_config)
    print(f"Run: {filepath}")
    print(f"n_samples={run.n_samples}, dt={run.dt}")

    roll_deg = np.degrees(run.roll_pitch[:, 0])
    pitch_deg = np.degrees(run.roll_pitch[:, 1])
    print(f"\nRoll range:  [{roll_deg.min():.2f}, {roll_deg.max():.2f}] degrees")
    print(f"Pitch range: [{pitch_deg.min():.2f}, {pitch_deg.max():.2f}] degrees")
    near_gimbal_lock = np.abs(pitch_deg) > 80
    if np.any(near_gimbal_lock):
        idx = np.where(near_gimbal_lock)[0]
        print(f"FLAG: {len(idx)}/{run.n_samples} samples have |pitch| > 80 degrees "
              f"(first at sample {idx[0]}) -- near gimbal lock for the 'xyz' Euler "
              f"decomposition this file uses.")
    else:
        print("No samples near gimbal lock (|pitch| > 80 degrees) -- rules this "
              "specific explanation out for this run.")

    sim_config = SimulationConfig(dt=run.dt, physics_dt=physics_dt)
    trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                    roll_pitch_series=run.roll_pitch)

    reference = simulate_trajectory(params, vehicle, sim_config, trajectory)

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco_result = simulate_trajectory_mujoco(params, vehicle, sim_config, trajectory, model, data)

    abs_diff = np.max(np.abs(mujoco_result.nu - reference.nu), axis=1)  # worst DOF at each step
    jumps = abs_diff[1:] - abs_diff[:-1]
    worst_jump_step = int(np.argmax(jumps)) + 1  # +1: jumps[i] is the jump INTO step i+1
    overall_max_step = int(np.argmax(abs_diff))

    print(f"\nOverall max abs diff: {np.max(abs_diff):.4e} (at step {overall_max_step})")
    print(f"Largest single-step jump: {jumps[worst_jump_step - 1]:.4e}, "
          f"landing at step {worst_jump_step} of {len(abs_diff)}")
    print(f"  diff just before the jump (step {worst_jump_step - 1}): {abs_diff[worst_jump_step - 1]:.4e}")
    print(f"  diff just after the jump  (step {worst_jump_step}):     {abs_diff[worst_jump_step]:.4e}")
    jump_ratio = (abs_diff[worst_jump_step] / abs_diff[worst_jump_step - 1]
                  if abs_diff[worst_jump_step - 1] > 1e-12 else float("inf"))
    print(f"  ratio: {jump_ratio:.2f}x "
          f"({'SUDDEN, bug/instability-like' if jump_ratio > 5 else 'gradual, compounding-error-like'})")

    if overall_max_step != worst_jump_step:
        print(f"\nNOTE: the overall max (step {overall_max_step}) and the biggest single-step "
              f"jump (step {worst_jump_step}) are DIFFERENT locations -- the peak was reached "
              f"gradually rather than via one sudden jump. Reporting BOTH windows below, not "
              f"just the jump location, so neither is left uninvestigated.")

    def _print_window(center: int, label: str):
        k = center
        print(f"\n--- {label}: step {k} ---")
        print(f"  roll={np.degrees(run.roll_pitch[k,0]):.2f} deg, pitch={np.degrees(run.roll_pitch[k,1]):.2f} deg")
        print(f"  tau: {run.forces_torques[k]}")
        print(f"  reference nu[{k}]: {reference.nu[k]}")
        print(f"  mujoco nu[{k}]:    {mujoco_result.nu[k]}")
        if k > 0:
            print(f"  reference nu[{k-1}]: {reference.nu[k-1]}")
            print(f"  mujoco nu[{k-1}]:    {mujoco_result.nu[k-1]}")
        window_start = max(0, k - 5)
        window_end = min(len(abs_diff), k + 6)
        print(f"  abs diff (worst DOF) for steps {window_start}-{window_end - 1}:")
        for i in range(window_start, window_end):
            marker = "  <-- HERE" if i == k else ""
            print(f"    step {i}: {abs_diff[i]:.6e}{marker}")

    _print_window(worst_jump_step, "Largest single-step jump")
    if overall_max_step != worst_jump_step:
        _print_window(overall_max_step, "Overall maximum (separate location)")

    # Per-DOF breakdown at the overall max -- which axes actually diverge,
    # not just the single worst-DOF summary number.
    k = overall_max_step
    dof_names = ["u (surge)", "v (sway)", "w (heave)", "p (roll rate)", "q (pitch rate)", "r (yaw rate)"]
    print(f"\nPer-DOF breakdown at step {k} (the overall maximum):")
    for i, name in enumerate(dof_names):
        print(f"  {name:<16s} reference={reference.nu[k,i]:>9.5f}  mujoco={mujoco_result.nu[k,i]:>9.5f}  "
              f"diff={abs(mujoco_result.nu[k,i]-reference.nu[k,i]):.5f}")

    # Coarse, whole-trajectory overview -- shows whether an elevated region
    # rises gradually from near-zero, or has a sharper onset further back
    # than any single narrow window would reveal.
    print(f"\nCoarse overview of abs diff across the full {len(abs_diff)}-step trajectory "
          f"(every {max(1, len(abs_diff)//40)}th step):")
    stride = max(1, len(abs_diff) // 40)
    for i in range(0, len(abs_diff), stride):
        marker = "  <-- overall max region" if abs(i - overall_max_step) < stride else ""
        print(f"  step {i:>5d}: {abs_diff[i]:.4e}{marker}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filepath", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--calibrated-params", default="calibrated_parameters_a1.yaml")
    args = parser.parse_args()
    diagnose(args.filepath, args.sampling_rate_hz, args.physics_dt, args.calibrated_params)