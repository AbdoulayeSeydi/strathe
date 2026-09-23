"""
Comprehensive per-substep diagnostic, per Abdoul's own specified plan.

IMPORTANT CORRECTION carried over from the conversation (see there for
full context): run7's earlier "2.6 million" catastrophic blowup was
from B1.5's TEACHER-FORCED parity test, caused by the (already-fixed)
xfrc_applied R.T wrench-conversion bug -- not from genuine forward
simulation. Every test of run7's FORWARD-SIM behavior has shown it
finite throughout, with large orientation DRIFT (not a numerical
explosion) as the vehicle's self-integrated pitch swings toward gimbal
lock. This script looks for "where does the trajectory start departing
from ordinary behavior" (acceleration, energy, conditioning), not "where
does a number blow up to infinity" -- because the latter doesn't
currently happen in forward-sim mode on this data.

METHOD: replays each real run through the REAL, unmodified
simulate_trajectory_mujoco (forward-sim, teacher_forced=False), spying
on compute_acceleration_from_rotation_matrix to record every quantity
Abdoul's plan asks for, at every substep, without altering execution.
Also independently recomputes the Coriolis term via the real
_coriolis_centripetal_force (pure, side-effect-free) purely for logging.

Records per substep: roll, pitch, yaw, u,v,w,p,q,r, Tx,Ty,Tz,
||tau_rot||, ||nu_rot||, u_dot..r_dot, kinetic energy (0.5 v^T M v,
constant M), and reports cond(M)/eigenvalues once (M doesn't vary with
state -- it's built purely from calibrated params/vehicle, not from
orientation, velocity, or time).

*** UNVERIFIED against real data in this sandbox -- no real CSV files here.
"""
import argparse

import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

from objective import load_run, DatasetConfig
from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec
from dynamics_model import (
    SimulationConfig, TrajectoryInputs,
    _mass_and_damping_arrays, _coriolis_centripetal_force,
)
import mujoco_backend

MJCF_PATH = "b0_single_body_placeholder.xml"


def run_diagnostic(filepath: str, sampling_rate_hz: float, physics_dt: float,
                    params, vehicle, M):
    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)
    run = load_run(filepath, dataset_config)
    sim_config = SimulationConfig(dt=run.dt, physics_dt=physics_dt)
    trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                    roll_pitch_series=run.roll_pitch)

    log = {k: [] for k in [
        "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r",
        "Tx", "Ty", "Tz", "tau_rot_norm", "nu_rot_norm",
        "u_dot", "v_dot", "w_dot", "p_dot", "q_dot", "r_dot",
        "kinetic_energy", "coriolis_rot_norm", "damping_rot_norm", "g_eta_rot_norm",
    ]}

    original_fn = mujoco_backend.compute_acceleration_from_rotation_matrix

    def spy(params, vehicle, v, tau, R):
        v_dot, damping_force, g_eta = original_fn(params, vehicle, v, tau, R)
        roll, pitch, yaw = Rotation.from_matrix(R).as_euler("xyz")
        coriolis = _coriolis_centripetal_force(M, v)
        log["roll"].append(roll); log["pitch"].append(pitch); log["yaw"].append(yaw)
        log["u"].append(v[0]); log["v"].append(v[1]); log["w"].append(v[2])
        log["p"].append(v[3]); log["q"].append(v[4]); log["r"].append(v[5])
        log["Tx"].append(tau[3]); log["Ty"].append(tau[4]); log["Tz"].append(tau[5])
        log["tau_rot_norm"].append(np.linalg.norm(tau[3:]))
        log["nu_rot_norm"].append(np.linalg.norm(v[3:]))
        log["u_dot"].append(v_dot[0]); log["v_dot"].append(v_dot[1]); log["w_dot"].append(v_dot[2])
        log["p_dot"].append(v_dot[3]); log["q_dot"].append(v_dot[4]); log["r_dot"].append(v_dot[5])
        log["kinetic_energy"].append(0.5 * v @ M @ v)
        log["coriolis_rot_norm"].append(np.linalg.norm(coriolis[3:]))
        log["damping_rot_norm"].append(np.linalg.norm(damping_force[3:]))
        log["g_eta_rot_norm"].append(np.linalg.norm(g_eta[3:]))
        return v_dot, damping_force, g_eta

    mujoco_backend.compute_acceleration_from_rotation_matrix = spy
    try:
        model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        data = mujoco.MjData(model)
        result = mujoco_backend.simulate_trajectory_mujoco(params, vehicle, sim_config, trajectory, model, data)
    finally:
        mujoco_backend.compute_acceleration_from_rotation_matrix = original_fn

    for k in log:
        log[k] = np.array(log[k])
    log["finite"] = np.all(np.isfinite(result.nu))
    log["n_substeps"] = len(log["roll"])
    return log


def print_summary(label, log):
    print(f"\n=== {label} ===")
    print(f"  finite: {log['finite']}, n_substeps: {log['n_substeps']}")
    print(f"  peak |pitch|: {np.degrees(np.max(np.abs(log['pitch']))):.2f} deg")
    print(f"  peak |roll|:  {np.degrees(np.max(np.abs(log['roll']))):.2f} deg")
    print(f"  peak |p,q,r|: {np.max(np.abs(log['p'])):.3f}, {np.max(np.abs(log['q'])):.3f}, {np.max(np.abs(log['r'])):.3f} rad/s")
    print(f"  peak |p_dot,q_dot,r_dot|: {np.max(np.abs(log['p_dot'])):.3f}, "
          f"{np.max(np.abs(log['q_dot'])):.3f}, {np.max(np.abs(log['r_dot'])):.3f} rad/s^2")
    print(f"  kinetic energy: min={np.min(log['kinetic_energy']):.4f}, max={np.max(log['kinetic_energy']):.4f}, "
          f"final={log['kinetic_energy'][-1]:.4f}")
    print(f"  peak ||coriolis_rot||: {np.max(log['coriolis_rot_norm']):.4f}")
    print(f"  peak ||damping_rot||:  {np.max(log['damping_rot_norm']):.4f}")
    print(f"  peak ||g_eta_rot||:    {np.max(log['g_eta_rot_norm']):.4f}")


def print_detailed_window(label, log, center, half_width=10):
    start = max(0, center - half_width)
    end = min(log["n_substeps"], center + half_width)
    print(f"\n--- {label}: detailed window, substeps {start}-{end - 1} ---")
    header = (f"{'step':>5} {'roll':>7} {'pitch':>7} {'yaw':>7} {'p':>7} {'q':>7} {'r':>7} "
              f"{'Tx':>7} {'Ty':>7} {'Tz':>7} {'p_dot':>9} {'q_dot':>9} {'r_dot':>9} {'KE':>8}")
    print(header)
    for i in range(start, end):
        print(f"{i:>5} {np.degrees(log['roll'][i]):>7.2f} {np.degrees(log['pitch'][i]):>7.2f} "
              f"{np.degrees(log['yaw'][i]):>7.2f} {log['p'][i]:>7.3f} {log['q'][i]:>7.3f} {log['r'][i]:>7.3f} "
              f"{log['Tx'][i]:>7.2f} {log['Ty'][i]:>7.2f} {log['Tz'][i]:>7.2f} "
              f"{log['p_dot'][i]:>9.3f} {log['q_dot'][i]:>9.3f} {log['r_dot'][i]:>9.3f} "
              f"{log['kinetic_energy'][i]:>8.3f}")


def main(data_dir: str, sampling_rate_hz: float, physics_dt: float,
         calibrated_params_path: str = "calibrated_parameters_a1.yaml"):
    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters(calibrated_params_path)

    M, _, _ = _mass_and_damping_arrays(params, vehicle)
    print("=== Mass matrix M (constant -- built from calibrated params/vehicle only, ===")
    print("=== does NOT vary with orientation, velocity, or time) ===")
    eigenvalues = np.linalg.eigvalsh(M)
    cond_M = np.linalg.cond(M)
    print(f"eigenvalues: {eigenvalues}")
    print(f"cond(M): {cond_M:.4e}")
    M_rot = M[3:, 3:]
    eigenvalues_rot = np.linalg.eigvalsh(M_rot)
    cond_M_rot = np.linalg.cond(M_rot)
    print(f"M_rot (rotational 3x3 sub-block) eigenvalues: {eigenvalues_rot}")
    print(f"cond(M_rot): {cond_M_rot:.4e}")
    print("NOTE: M_rot is diagonal (Xq_dot/Zq_dot couple linear-angular, not angular-angular),")
    print("so cond(M_rot) is just the ratio of the largest to smallest of Ieff_roll/pitch/yaw.")

    run_files = {
        "run7 (large drift, near gimbal lock)": f"{data_dir}/odom-12-01-2024-run7.csv",
        "run11 (moderate drift, no gimbal lock)": f"{data_dir}/odom-12-01-2024-run11.csv",
        "run9 (small drift)": f"{data_dir}/odom-12-01-2024-run9.csv",
        "run4 (large drift, near gimbal lock)": f"{data_dir}/odom-12-01-2024-run4.csv",
    }

    logs = {}
    for label, filepath in run_files.items():
        logs[label] = run_diagnostic(filepath, sampling_rate_hz, physics_dt, params, vehicle, M)
        print_summary(label, logs[label])

    # Detailed window for run7, centered on where pitch first crosses 45 degrees --
    # not "where it blows up" (it doesn't), but "where the excursion begins".
    run7_log = logs["run7 (large drift, near gimbal lock)"]
    crossing_indices = np.where(np.abs(np.degrees(run7_log["pitch"])) > 45)[0]
    if len(crossing_indices) > 0:
        center = crossing_indices[0]
        print_detailed_window("run7", run7_log, center, half_width=15)
    else:
        print("\nrun7 never crossed 45 degrees pitch in this run -- no detailed window to show.")

    print(f"\n{'='*70}")
    print("Compare the summaries above directly: does run7/run4 (large drift) show")
    print("substantially larger peak p_dot/q_dot/r_dot, kinetic energy, or Coriolis/damping")
    print("norms than run9/run11 (smaller drift)? If yes, that points toward the rotational")
    print("dynamics itself. If accelerations stay comparable across all four despite the")
    print("very different drift outcomes, that points toward the trajectories simply")
    print("evolving differently under sensitive, compounding integration -- not a moment")
    print("of unreasonable acceleration.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--calibrated-params", default="calibrated_parameters_a1.yaml")
    args = parser.parse_args()
    main(args.data_dir, args.sampling_rate_hz, args.physics_dt, args.calibrated_params)