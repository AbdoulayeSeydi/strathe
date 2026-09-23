"""
Pure-Python, MuJoCo-FREE open-loop forward integrator, using
compute_acceleration() directly with a completely independent numerical
method (plain explicit Euler, no full-override, no wrench injection, no
MuJoCo at all) -- to isolate whether the near-universal gimbal-lock
proximity found via check_pitch_excursion_correlation.py (24/25 real
runs peaked within 20 degrees of pitch=+/-90 during MuJoCo-based genuine
forward simulation, most within 1-2 degrees) is a property of the
CALIBRATED PHYSICS ITSELF (compute_acceleration(), shared by the
stand-in, HoloOcean's backend, and mujoco_backend.py alike), or
something specific to MuJoCo's own implementation.

If this ALSO shows near-universal gimbal-lock proximity on the same
real data: strong, independent evidence the finding is inherent to
replaying LOGGED, CLOSED-LOOP-GENERATED control forces OPEN-LOOP (with
no corrective feedback) over a long horizon -- a well-known general
instability, not a MuJoCo-specific bug. If it does NOT: points at
something specific to MuJoCo's own numerics instead.

Uses the SAME confirmed body-frame right-multiply quaternion
integration formula verified in b1_3_velocity_frame_check.py.

*** UNVERIFIED against real data in this sandbox -- no real CSV files
here. The integration logic itself is tested against synthetic data
below before handoff.
"""
import argparse

import numpy as np
from scipy.spatial.transform import Rotation


def pure_python_open_loop_forward_sim(params, vehicle, run, physics_dt):
    from dynamics_model import compute_acceleration

    n_samples = run.n_samples
    dt = run.dt
    n_substeps = max(1, int(np.ceil(dt / physics_dt)))
    sub_dt = dt / n_substeps

    roll0, pitch0 = run.roll_pitch[0]
    R = Rotation.from_euler("xyz", [roll0, pitch0, 0.0]).as_matrix()
    v = run.nu0.copy()

    peak_abs_pitch = abs(pitch0)
    n_finite_checks = 0

    for k in range(n_samples - 1):
        tau = run.forces_torques[k]
        for _ in range(n_substeps):
            roll, pitch, _yaw = Rotation.from_matrix(R).as_euler("xyz")
            peak_abs_pitch = max(peak_abs_pitch, abs(pitch))

            v_dot, _damping, _g_eta = compute_acceleration(params, vehicle, v, tau, roll, pitch)
            v = v + v_dot * sub_dt
            n_finite_checks += 1
            if not np.all(np.isfinite(v)):
                return peak_abs_pitch, v, False

            incremental = Rotation.from_rotvec(v[3:] * sub_dt).as_matrix()
            R = R @ incremental

    return peak_abs_pitch, v, True


def _self_test():
    """Structural test against synthetic data -- confirms the
    integration logic runs correctly and produces sane output, without
    needing real data or MuJoCo. Does NOT test whether the real-data
    finding replicates -- that needs the real CSVs."""
    from dataclasses import dataclass

    @dataclass
    class FakeRun:
        nu0: np.ndarray
        forces_torques: np.ndarray
        roll_pitch: np.ndarray
        dt: float
        n_samples: int

    rng = np.random.default_rng(0)
    n = 30
    run = FakeRun(
        nu0=np.array([0.3, 0.1, -0.1, 0.05, 0.05, 0.0]),
        forces_torques=rng.uniform(-3, 3, size=(n, 6)),
        roll_pitch=np.tile([0.05, -0.05], (n, 1)),
        dt=0.1,
        n_samples=n,
    )

    from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
    from optimizer import load_vehicle_spec

    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters("calibrated_parameters_a1.yaml")

    peak_pitch, final_v, finite = pure_python_open_loop_forward_sim(params, vehicle, run, physics_dt=0.02)
    print(f"Self-test: peak|pitch|={np.degrees(peak_pitch):.2f} deg, finite={finite}")
    assert finite, "self-test trajectory diverged -- integration logic itself may be broken"
    print("Self-test PASSED: integration logic runs correctly on synthetic data.")


def run_check(data_dir: str, sampling_rate_hz: float, dates: list, physics_dt: float,
              calibrated_params_path: str = "calibrated_parameters_a1.yaml"):
    from data_split import discover_csv_files, extract_date
    from objective import load_run, DatasetConfig
    from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
    from optimizer import load_vehicle_spec

    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters(calibrated_params_path)
    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)

    all_files, _ = discover_csv_files(data_dir)
    selected_files = [f for f in all_files if extract_date(f) in dates]
    print(f"Found {len(selected_files)} real run file(s) matching {dates}")

    results = []
    for filepath in selected_files:
        run = load_run(filepath, dataset_config)
        peak_abs_pitch, final_v, finite = pure_python_open_loop_forward_sim(params, vehicle, run, physics_dt)
        peak_deg = np.degrees(peak_abs_pitch)
        near_lock = peak_deg > 70
        print(f"{filepath}: finite={finite}, peak|pitch|={peak_deg:.2f} deg"
              f"{'  <-- within 20 deg of gimbal lock' if near_lock else ''}")
        results.append({"filepath": filepath, "finite": finite, "peak_deg": peak_deg})

    print(f"\n{'='*70}")
    n_near_lock = sum(1 for r in results if r["finite"] and r["peak_deg"] > 70)
    n_finite = sum(1 for r in results if r["finite"])
    print(f"{n_finite}/{len(results)} finite; {n_near_lock}/{n_finite} finite runs came within "
          f"20 deg of gimbal lock at some point")
    print("\nIf this count is similarly high to the MuJoCo-based check (24/25), that is strong "
          "evidence the near-universal gimbal-lock proximity is inherent to replaying logged, "
          "closed-loop-generated forces open-loop over a long horizon -- NOT a MuJoCo-specific "
          "issue. If this count is notably LOWER, that would point at something specific to "
          "MuJoCo's own numerics instead, worth investigating directly.")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Run the synthetic self-test only, no real data needed")
    parser.add_argument("--data-dir")
    parser.add_argument("--sampling-rate-hz", type=float)
    parser.add_argument("--dates", nargs="+", default=None)
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--calibrated-params", default="calibrated_parameters_a1.yaml")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
    else:
        if args.dates is None:
            from phase2_config import PHASE1_TRAIN_DATES
            dates = PHASE1_TRAIN_DATES
        else:
            dates = args.dates
        run_check(args.data_dir, args.sampling_rate_hz, dates, args.physics_dt, args.calibrated_params)