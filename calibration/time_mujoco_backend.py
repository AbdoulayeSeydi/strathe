#!/usr/bin/env python3
"""
Measures real per-step cost of the MuJoCo-backed simulate_fn against the
analytical stand-in, on a small truncated slice of real data, then
extrapolates to what a full calibration run would actually cost --
mirrors time_holoocean_backend.py's exact methodology (same truncation
approach, same initial_parameters() point, same joint_residuals() call,
same extrapolation logic) so the two are directly, honestly comparable,
not just similar.

Uses mujoco_backend.make_mujoco_calibration_simulate_fn (teacher_forced=
True) specifically, not forward-sim -- this benchmark is about the cost
of using MuJoCo as a FITTING backend (the actual B2.3 question: does
MuJoCo need the same train-on-stand-in/validate-on-real-backend split
HoloOcean required), and teacher_forced=True is the only MuJoCo mode
confirmed calibration-safe this session.

Usage:
    python time_mujoco_backend.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml \
        --sampling-rate-hz 10.0 \
        --n-steps-timed 20 \
        --n-runs-timed 3

*** UNVERIFIED against real data or a real mujoco/dynamics_model.py in
this sandbox (sandbox reset earlier this session) -- built by mirroring
time_holoocean_backend.py's own confirmed-working structure as closely
as possible, not independently tested end-to-end the way every earlier
handoff this session was. Worth a moment's extra scrutiny on first run:
the printed per-step ms for the stand-in half should roughly match the
range already established elsewhere this session (~0.45-0.72ms/step on
the full dataset; this uses a much smaller slice, so some difference is
expected, but not an order of magnitude).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import mujoco

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, joint_residuals
from optimizer import load_vehicle_spec, initial_parameters
from dynamics_model import TrajectoryInputs
from mujoco_backend import make_mujoco_calibration_simulate_fn

MJCF_PATH = "b0_single_body_placeholder.xml"
HOLOOCEAN_REFERENCE_SLOWDOWN = 400.0  # per B2.3's own task text ("~400x") -- override with your
                                        # own actual time_holoocean_backend.py output if you have it


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=["16-02-2024", "19-02-2024"])
    parser.add_argument("--n-steps-timed", type=int, default=20,
                         help="Truncate each timed run to this many steps -- matches "
                              "time_holoocean_backend.py's own default exactly")
    parser.add_argument("--n-runs-timed", type=int, default=3,
                         help="How many runs to include in the timed joint_residuals call")
    parser.add_argument("--assumed-evals-per-restart", type=int, default=40,
                         help="For the extrapolation estimate -- matches "
                              "time_holoocean_backend.py's own default")
    parser.add_argument("--mjcf-path", default=MJCF_PATH)
    parser.add_argument("--holoocean-slowdown", type=float, default=HOLOOCEAN_REFERENCE_SLOWDOWN,
                         help="HoloOcean's own measured slowdown, for the final three-way "
                              "comparison -- pass your own actual time_holoocean_backend.py "
                              "output here if you have a precise number; defaults to B2.3's "
                              "own approximate '~400x' if not.")
    args = parser.parse_args()

    csv_paths, _ = discover_csv_files(args.data_dir)
    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)
    dataset_config = DatasetConfig(sampling_rate_hz=args.sampling_rate_hz)

    all_train_runs = [load_run(fp, dataset_config) for fp in split.train_files]
    avg_full_run_length = np.mean([r.n_samples for r in all_train_runs])
    n_train_runs = len(all_train_runs)

    loaded_spec = load_vehicle_spec(args.spec)
    params = initial_parameters(loaded_spec)  # literature values -- matches
                                                # time_holoocean_backend.py's own point exactly,
                                                # NOT calibrated_parameters_a1.yaml
    vehicle = loaded_spec.vehicle

    # Same truncation pattern as time_holoocean_backend.py, verbatim --
    # n_samples is auto-computed in __post_init__ from forces_torques.shape[0],
    # not a constructor arg.
    timed_runs = []
    for run in all_train_runs[:args.n_runs_timed]:
        n = min(args.n_steps_timed, run.n_samples)
        truncated = type(run)(
            filepath=run.filepath, forces_torques=run.forces_torques[:n],
            roll_pitch=run.roll_pitch[:n], real_nu=run.real_nu[:n], nu0=run.nu0,
            dt=run.dt, sample_rate_hz=run.sample_rate_hz, date=run.date,
        )
        timed_runs.append(truncated)

    print(f"Timing against {len(timed_runs)} run(s), {args.n_steps_timed} steps each\n")

    param_vec = params.to_vector()

    print("=== Timing the STAND-IN (baseline) ===")
    t0 = time.monotonic()
    _ = joint_residuals(param_vec, timed_runs, vehicle)
    standin_elapsed = time.monotonic() - t0
    standin_per_step_ms = standin_elapsed / (len(timed_runs) * args.n_steps_timed) * 1000
    print(f"Stand-in: {standin_elapsed:.4f}s total, {standin_per_step_ms:.4f}ms/step\n")

    print("=== Creating MuJoCo model/data (one-time cost, not counted in per-step timing) ===")
    t0 = time.monotonic()
    model = mujoco.MjModel.from_xml_path(args.mjcf_path)
    data = mujoco.MjData(model)
    model_creation_time = time.monotonic() - t0
    print(f"Model creation took {model_creation_time:.3f}s (one-time only, NOT repeated per evaluation)\n")

    simulate_fn = make_mujoco_calibration_simulate_fn(model, data)

    print("=== Timing the MUJOCO-BACKED simulate_fn (teacher_forced=True) ===")
    t0 = time.monotonic()
    _ = joint_residuals(param_vec, timed_runs, vehicle, simulate_fn=simulate_fn)
    mujoco_elapsed = time.monotonic() - t0
    per_step_ms = mujoco_elapsed / (len(timed_runs) * args.n_steps_timed) * 1000
    print(f"MuJoCo-backed: {mujoco_elapsed:.4f}s total, {per_step_ms:.4f}ms/step\n")

    slowdown = mujoco_elapsed / standin_elapsed if standin_elapsed > 0 else float("inf")
    print("=" * 70)
    print("EXTRAPOLATION TO A FULL CALIBRATION RUN")
    print("=" * 70)
    print(f"MuJoCo is {slowdown:.2f}x slower than the stand-in per step (this measurement)")
    print(f"Real dataset: {n_train_runs} training runs, average {avg_full_run_length:.0f} steps/run")

    per_eval_seconds = per_step_ms / 1000 * avg_full_run_length * n_train_runs
    print(f"\nEstimated cost of ONE full joint_residuals() evaluation "
          f"({n_train_runs} runs x ~{avg_full_run_length:.0f} steps): {per_eval_seconds:.1f}s "
          f"(~{per_eval_seconds/60:.1f} min)")

    for n_evals in [args.assumed_evals_per_restart, args.assumed_evals_per_restart * 5]:
        total_seconds = per_eval_seconds * n_evals
        label = "single restart" if n_evals == args.assumed_evals_per_restart else "5 restarts"
        print(f"Estimated total for ~{n_evals} evaluations ({label}): "
              f"{total_seconds/60:.1f} min ({total_seconds/3600:.2f} hours)")

    print("\n" + "=" * 70)
    print("B2.3: THREE-WAY COMPARISON")
    print("=" * 70)
    print(f"Stand-in:  1.00x (reference)")
    print(f"MuJoCo:    {slowdown:.2f}x   ({per_step_ms:.4f} ms/step, this measurement)")
    print(f"HoloOcean: {args.holoocean_slowdown:.0f}x   (per B2.3's own task text / "
          f"--holoocean-slowdown, not independently re-measured here)")
    print(f"\nMuJoCo is ~{args.holoocean_slowdown/slowdown:.0f}x FASTER than HoloOcean's own "
          f"measured overhead.")
    if slowdown < 20:
        print("Conclusion: MuJoCo does NOT need the train-on-stand-in/validate-on-real-backend")
        print("split HoloOcean required -- its overhead is low enough to plausibly train")
        print("directly on, unlike HoloOcean's confirmed ~400x.")
    else:
        print("Conclusion: MuJoCo's overhead here is higher than earlier measurements this")
        print("session suggested -- worth checking whether this small-slice benchmark is")
        print("representative before concluding it's safe to train directly on.")

    print("\nCaveat: this is a rough linear extrapolation from a small, short-run sample --")
    print("real per-step cost may not scale perfectly linearly. Treat this as a planning")
    print("estimate, not a guarantee -- same caveat time_holoocean_backend.py's own output gives.")


if __name__ == "__main__":
    main()