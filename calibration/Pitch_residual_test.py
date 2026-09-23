#!/usr/bin/env python3
"""
Pitch residual-structure test -- the one physics investigation left open
from this session's sway/added-mass work (see conversation): Mq/Mqq
remains the only damping pair whose fitted-vs-kinematic correlation gap
SHRANK after Coriolis terms were added, the signature of genuine missing
physics rather than data-coverage. Xq_dot/Zq_dot were already introduced
for pitch coupling, and are pinned in every single parameter-reduction
experiment run this session (control, Experiment A, B1) -- consistent
with the model wanting more pitch coupling than current bounds allow,
but never directly tested for WHAT specifically is missing.

METHOD: an INSTANTANEOUS residual, not an integrated-trajectory one --
computes q_dot_real(t) via finite difference of the real logged Bq
column, and q_dot_model(t) by calling compute_acceleration() DIRECTLY at
the real, logged (u,v,w,p,q,r,roll,pitch,tau) state at that same instant
-- no integration, so no drift contaminates the comparison; this isolates
the DYNAMICS MODEL's own error from anything about how it's integrated
over time. Deliberately does NOT reuse simulate_trajectory()'s own debug
output for this reason -- that would carry the SIMULATED (drifting) nu,
not the real logged one, at every sample past the first.

Regresses this residual against candidate cross-axis kinematic terms
NOT currently in the model, following the same logic already used to
close the sway question: does the CURRENT model's leftover pitch error
contain structure that maps onto a physically specific candidate, or is
it unstructured (noise) -- which would argue against adding anything
further. Includes q and q*|q| too, as a direct sanity check: if the
residual STILL correlates with these, that's evidence Mq/Mqq themselves
aren't fully explaining what they should, even after fitting.

Usage:
    python3 pitch_residual_test.py \\
        --data-dir ../data/raw/Bluerov2_trajectory_dataset/dataset \\
        --spec vehicle_spec.yaml \\
        --calibrated-params calibrated_parameters_a1.yaml \\
        --sampling-rate-hz 10.0 \\
        --heldout-dates 16-02-2024 19-02-2024

*** UNVERIFIED against real data -- built without local access to a
working dynamics_model.py/objective.py reconstruction this turn (sandbox
reset since earlier in this session); compute_acceleration()'s signature
was confirmed directly against the real file first, but the full script
has not been end-to-end tested the way every earlier handoff this
session was. Treat the first real run's output with a bit more scrutiny
than usual -- if anything about the shapes/values looks obviously wrong
(e.g. residuals many orders of magnitude off from q_dot's own scale),
stop and flag it rather than trusting the correlation numbers.
"""
from __future__ import annotations

import argparse

import numpy as np
import yaml

from data_split import split_by_date, discover_csv_files
from dynamics_model import CalibrationParameters, compute_acceleration
from objective import DatasetConfig, load_run
from optimizer import load_vehicle_spec

DOF_INDEX = {"u": 0, "v": 1, "w": 2, "p": 3, "q": 4, "r": 5}


def load_calibrated_parameters(path: str) -> CalibrationParameters:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CalibrationParameters(**data["parameters"])


def compute_pitch_residuals(params, vehicle, runs):
    """
    Returns (residual, regressors_dict) concatenated across all runs.
    residual[i] = q_dot_real(t_i) - q_dot_model(t_i), an INSTANTANEOUS
    comparison at the real logged state, not an integrated trajectory.
    """
    all_residuals = []
    all_regressors = {name: [] for name in
                       ["u", "w", "p", "r", "q", "q_absq", "uw", "pr", "up", "wp", "ur", "wr"]}

    for run in runs:
        real_nu = run.real_nu          # (n_samples, 6): u,v,w,p,q,r
        roll_pitch = run.roll_pitch    # (n_samples, 2): roll, pitch
        tau = run.forces_torques       # (n_samples, 6)
        dt = run.dt
        n = run.n_samples

        for t in range(n - 1):  # need t+1 for the finite difference
            v_t = real_nu[t]
            u, v, w, p, q, r = v_t
            roll, pitch = roll_pitch[t]

            q_dot_real = (real_nu[t + 1, 4] - real_nu[t, 4]) / dt

            v_dot_model, _damping, _restoring = compute_acceleration(
                params, vehicle, v_t, tau[t], roll, pitch
            )
            q_dot_model = v_dot_model[4]

            residual = q_dot_real - q_dot_model
            all_residuals.append(residual)
            all_regressors["u"].append(u)
            all_regressors["w"].append(w)
            all_regressors["p"].append(p)
            all_regressors["r"].append(r)
            all_regressors["q"].append(q)
            all_regressors["q_absq"].append(q * abs(q))
            all_regressors["uw"].append(u * w)
            all_regressors["pr"].append(p * r)
            all_regressors["up"].append(u * p)
            all_regressors["wp"].append(w * p)
            all_regressors["ur"].append(u * r)
            all_regressors["wr"].append(w * r)

    residual = np.array(all_residuals)
    regressors = {name: np.array(vals) for name, vals in all_regressors.items()}
    return residual, regressors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--calibrated-params", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=None)
    args = parser.parse_args()

    loaded_spec = load_vehicle_spec(args.spec)
    calibrated = load_calibrated_parameters(args.calibrated_params)
    vehicle = loaded_spec.vehicle

    csv_paths, duplicates = discover_csv_files(args.data_dir)
    for dup_path, kept_path in duplicates:
        print(f"Duplicate filename skipped: {dup_path} (kept {kept_path})")
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)
    train_runs = [load_run(fp, dataset_config) for fp in split.train_files]

    print(f"Computing instantaneous pitch residuals across {len(train_runs)} training runs "
          f"(no integration -- real logged state at every sample)...")
    residual, regressors = compute_pitch_residuals(calibrated, vehicle, train_runs)
    n_samples = len(residual)
    print(f"Total samples: {n_samples}")

    print(f"\nq_dot_real - q_dot_model residual: mean={residual.mean():.6f}, "
          f"std={residual.std():.6f}, min={residual.min():.6f}, max={residual.max():.6f}")
    print("(Sanity check: this should be a small, roughly-zero-centered number on the same")
    print("order of magnitude as typical q_dot values -- if it's many orders of magnitude")
    print("off, or the mean is far from zero, something upstream is wrong -- stop and check")
    print("before trusting the correlations below.)")

    print("\n" + "=" * 70)
    print("CORRELATION: pitch residual vs. candidate cross-axis terms")
    print("=" * 70)
    correlations = []
    for name, values in regressors.items():
        corr = np.corrcoef(residual, values)[0, 1]
        correlations.append((name, corr))
    correlations.sort(key=lambda x: -abs(x[1]))

    print(f"{'term':<10}{'corr':>10}   meaning")
    meanings = {
        "u": "surge velocity (raw, linear) -- NOT currently a direct model term",
        "w": "heave velocity (raw, linear) -- NOT currently a direct model term",
        "p": "roll rate (raw, linear) -- NOT currently a pitch-coupled term",
        "r": "yaw rate (raw, linear) -- NOT currently a pitch-coupled term",
        "q": "pitch rate -- ALREADY modeled via Mq; residual correlation here would mean Mq isn't fully capturing it",
        "q_absq": "pitch rate squared (signed) -- ALREADY modeled via Mqq; same caveat as q above",
        "uw": "surge*heave product -- a genuine Coriolis-type cross term, NOT in the current model",
        "pr": "roll*yaw product -- gyroscopic-type cross term, NOT in the current model",
        "up": "surge*roll product -- NOT in the current model",
        "wp": "heave*roll product -- NOT in the current model",
        "ur": "surge*yaw product -- NOT in the current model",
        "wr": "heave*yaw product -- NOT in the current model",
    }
    for name, corr in correlations:
        print(f"{name:<10}{corr:>10.3f}   {meanings[name]}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    strongest_name, strongest_corr = correlations[0]
    if abs(strongest_corr) < 0.2:
        print(f"Strongest correlation ({strongest_name}: {strongest_corr:.3f}) is weak. The current")
        print("model's pitch residual does NOT show strong structured dependence on any of these")
        print("candidate terms -- consistent with the residual being closer to noise than a clean")
        print("missing physical mechanism. This would argue AGAINST adding a new pitch coupling")
        print("term based on this evidence -- Mq/Mqq's earlier-confirmed correlation gap may")
        print("reflect something more diffuse (e.g. genuinely just a harder-to-fit axis) than a")
        print("single missing term this test could isolate.")
    else:
        already_modeled = strongest_name in ("q", "q_absq")
        if already_modeled:
            print(f"Strongest correlation ({strongest_name}: {strongest_corr:.3f}) is with a term")
            print("ALREADY in the model (Mq/Mqq) -- suggesting the existing damping isn't fully")
            print("capturing its own relationship to q, not that a NEW cross-axis term is missing.")
            print("Worth checking whether Mq/Mqq's functional form (linear+quadratic in q alone)")
            print("is the right one, before considering a structurally different addition.")
        else:
            print(f"Strongest correlation ({strongest_name}: {strongest_corr:.3f}) is with a term NOT")
            print(f"currently in the model -- {meanings[strongest_name]}")
            print("This is real, structured evidence worth taking seriously as a candidate for a")
            print("physically-motivated model extension -- though a single correlation number is a")
            print("starting point for that conversation, not a decision to add the term yet.")


if __name__ == "__main__":
    main()