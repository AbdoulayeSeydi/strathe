#!/usr/bin/env python3
"""
Minimal debug script -- prints raw values at the first few real samples
of one run, to diagnose why pitch_residual_test.py's sanity check failed
(residual mean=7.5, range -368 to +480, far too large to be a genuine
small residual). Prints EVERYTHING needed to spot the bug directly,
rather than guessing at it remotely.

Usage:
    python3 debug_pitch_residual.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset/dataset \
        --spec vehicle_spec.yaml \
        --calibrated-params calibrated_parameters_a1.yaml \
        --sampling-rate-hz 10.0
"""
import argparse

import yaml

from data_split import discover_csv_files
from dynamics_model import CalibrationParameters, compute_acceleration
from objective import DatasetConfig, load_run
from optimizer import load_vehicle_spec


def load_calibrated_parameters(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CalibrationParameters(**data["parameters"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--calibrated-params", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    args = parser.parse_args()

    loaded_spec = load_vehicle_spec(args.spec)
    calibrated = load_calibrated_parameters(args.calibrated_params)
    vehicle = loaded_spec.vehicle

    csv_paths, duplicates = discover_csv_files(args.data_dir)
    run = load_run(csv_paths[0], DatasetConfig(sampling_rate_hz=args.sampling_rate_hz))

    print(f"File: {csv_paths[0]}")
    print(f"run.dt = {run.dt}")
    print(f"run.sample_rate_hz = {run.sample_rate_hz}")
    print(f"run.n_samples = {run.n_samples}")
    print(f"real_nu.shape = {run.real_nu.shape}")
    print(f"roll_pitch.shape = {run.roll_pitch.shape}")
    print(f"forces_torques.shape = {run.forces_torques.shape}")

    print("\nFirst 5 samples, full detail:")
    for t in range(5):
        v_t = run.real_nu[t]
        u, v, w, p, q, r = v_t
        roll, pitch = run.roll_pitch[t]
        tau_t = run.forces_torques[t]

        q_dot_real = (run.real_nu[t + 1, 4] - run.real_nu[t, 4]) / run.dt

        v_dot_model, damping_force, g_eta = compute_acceleration(
            calibrated, vehicle, v_t, tau_t, roll, pitch
        )

        print(f"\n--- t={t} ---")
        print(f"  v_t (u,v,w,p,q,r) = {v_t}")
        print(f"  roll={roll:.4f}, pitch={pitch:.4f}")
        print(f"  tau_t (Fx,Fy,Fz,Tx,Ty,Tz) = {tau_t}")
        print(f"  real_nu[t+1,4] (next q) = {run.real_nu[t+1, 4]:.6f}, real_nu[t,4] (this q) = {q:.6f}")
        print(f"  q_dot_real = {q_dot_real:.6f}")
        print(f"  v_dot_model (full 6) = {v_dot_model}")
        print(f"  q_dot_model (index 4) = {v_dot_model[4]:.6f}")
        print(f"  damping_force (full 6) = {damping_force}")
        print(f"  g_eta (full 6) = {g_eta}")
        print(f"  residual = q_dot_real - q_dot_model = {q_dot_real - v_dot_model[4]:.6f}")


if __name__ == "__main__":
    main()