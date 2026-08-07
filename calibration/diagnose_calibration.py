#!/usr/bin/env python3
"""
Post-hoc diagnostics for an already-completed calibration run. Answers
what average RMSE alone can't:

  1. Active-bounds report (instant, free) — which calibrated parameters
     landed exactly on a bound, and which one. 12/23 pinned is a PATTERN
     worth investigating, not evidence the bounds are simply too tight —
     see module note below before reaching for --widen-bounds.

  2. Profiling (~1 min) — where a single simulate_trajectory() call
     actually spends its time. Don't guess where the runtime went.

  3. Per-DOF worst-run comparison (~1s) — average RMSE hides "surge fits
     great, yaw fits terribly." This shows the worst run's per-DOF
     breakdown directly.

  4. Jacobian-based identifiability analysis (~10 min on real data —
     evaluates the Jacobian ONCE at the already-calibrated point, NOT a
     full re-optimization, but still ~n_params+1 = 24 simulate calls).
     Reports the parameter correlation matrix and condition number —
     this is what actually answers "are Xu and Xuu distinguishable, or
     is the optimizer just trading them off against each other."

## Why hitting bounds doesn't automatically mean "widen the bounds"
Twelve parameters pinning to a boundary — almost entirely damping terms
(Yv, Zw, Kp, Mq, Xuu, Yvv, Zww, Kpp, Mqq) — is a specific pattern, not
generic noise. The stand-in dynamics model has no Coriolis/cross-coupling,
no thruster lag, no current, no nonlinear restoring terms beyond a
simplified buoyancy offset (see dynamics_model.py's assumptions section).
When real physics is missing from the model, the ONLY thing the optimizer
has to explain that missing physics with is damping and added mass — so
it pushes damping as high as the bounds allow, because that's the closest
available substitute for dynamics the model doesn't have. Widening the
bounds without addressing this would very likely just move the pinning
further out, not fix the underlying issue. This script's job is to give
evidence for or against that theory (correlation matrix, per-DOF
breakdown) before touching vehicle_spec.yaml's bounds.

Usage:
    python3 diagnose_calibration.py \\
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \\
        --spec vehicle_spec.yaml \\
        --calibrated-params ../calibration_output/calibrated_parameters.yaml \\
        --sampling-rate-hz 50 \\
        --heldout-dates 16-02-2024 19-02-2024 \\
        [--skip-jacobian]   # skip the ~10 min identifiability analysis
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np
import yaml

from data_split import split_by_date, discover_csv_files
from dynamics_model import CalibrationParameters, SimulationConfig, TrajectoryInputs, simulate_trajectory
from objective import DatasetConfig, joint_residuals, load_run, rmse_by_dof, rmse_for_run
from optimizer import load_vehicle_spec

try:
    import matplotlib
    matplotlib.use("Agg")  # no display needed — saving PNGs, not showing a window
    import matplotlib.pyplot as plt
    HAVE_MATPLOTLIB = True
except ImportError:
    HAVE_MATPLOTLIB = False

DOF_LABELS = ["u", "v", "w", "p", "q", "r"]


# ---------------------------------------------------------------------------
# 1. Active-bounds report — instant, no simulation
# ---------------------------------------------------------------------------

def load_calibrated_parameters(path: str) -> CalibrationParameters:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return CalibrationParameters(**data["parameters"])


def active_bounds_report(loaded_spec, calibrated: CalibrationParameters, tol_frac: float = 0.01) -> list:
    """Which calibrated parameters landed at (or within tol_frac of) a
    bound, and which bound. Pure arithmetic on already-computed values —
    no simulation calls."""
    flat = loaded_spec.flat_calibration_targets
    hits = []
    for name in calibrated.param_names:
        lo, hi = flat[name]["bounds"]
        val = getattr(calibrated, name)
        width = hi - lo
        tol = tol_frac * width if width > 0 else 0
        if abs(val - lo) <= tol:
            hits.append((name, "lower", lo, val))
        elif abs(val - hi) <= tol:
            hits.append((name, "upper", hi, val))
    return hits


# ---------------------------------------------------------------------------
# 2. Profiling — where does one evaluation actually spend its time
# ---------------------------------------------------------------------------

def profile_single_evaluation(params, run, vehicle) -> str:
    sim_config = SimulationConfig(dt=run.dt)
    trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                    roll_pitch_series=run.roll_pitch)
    profiler = cProfile.Profile()
    profiler.enable()
    simulate_trajectory(params, vehicle, sim_config, trajectory)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(15)
    return stream.getvalue()


# ---------------------------------------------------------------------------
# 3. Per-DOF worst-run comparison
# ---------------------------------------------------------------------------

def per_dof_worst_run_report(params, runs, vehicle) -> None:
    worst = max(runs, key=lambda r: rmse_for_run(params, r, vehicle))
    dof_rmse = rmse_by_dof(params, worst, vehicle)
    print(f"\nWorst-performing run: {worst.filepath}")
    print(f"  {'DOF':<6}{'RMSE':>10}")
    for lbl, val in zip(DOF_LABELS, dof_rmse):
        print(f"  {lbl:<6}{val:>10.4f}")


# ---------------------------------------------------------------------------
# 4. Jacobian-based identifiability analysis
# ---------------------------------------------------------------------------

def identifiability_analysis(param_vec: np.ndarray, runs: list, vehicle) -> dict:
    """
    Evaluates the Jacobian ONCE at the given point (already-calibrated
    parameters) — NOT a full re-optimization. Still costs ~n_params+1
    simulate calls, so this is the expensive step in this script.
    """
    from scipy.optimize._numdiff import approx_derivative

    call_count = {"n": 0}
    total = len(param_vec) + 1

    def counted_residuals(x, *args):
        call_count["n"] += 1
        t0 = time.monotonic()
        result = joint_residuals(x, *args)
        print(f"  Jacobian eval {call_count['n']}/{total} ({time.monotonic() - t0:.1f}s)")
        return result

    print(f"Evaluating Jacobian at the calibrated point ({total} evaluations expected, "
          f"using 2-point differences)...")
    t_start = time.monotonic()
    J = approx_derivative(counted_residuals, param_vec, args=(runs, vehicle), method="2-point")
    elapsed = time.monotonic() - t_start
    print(f"Jacobian evaluated in {elapsed:.1f}s total, shape {J.shape}")

    singular_values = np.linalg.svd(J, compute_uv=False)
    smallest = singular_values[-1]
    if smallest < 1e-12:
        condition_number = float("inf")
        print("Smallest singular value is ~0 — Jacobian is rank-deficient "
              "(at least one parameter direction has essentially no effect "
              "on the residuals at this point).")
    else:
        condition_number = float(singular_values[0] / smallest)

    # --- Multi-parameter identifiability: near-null-space directions ---
    # The pairwise correlation matrix (below) only catches TWO-parameter
    # degeneracies. A flat direction can involve 3+ parameters
    # simultaneously with no single pair crossing the correlation
    # threshold — exactly the pattern found for Zw_dot after adding
    # Coriolis terms (pinned at its bound, high overall condition number,
    # but no pairwise correlation >=0.7 in the existing analysis). The
    # right singular vectors corresponding to the SMALLEST singular values
    # directly reveal these: each is a parameter COMBINATION the data
    # barely constrains as a group, even if no single pair within it is
    # individually degenerate. Reuses the SAME Jacobian already computed
    # above — no additional expensive simulate_trajectory() calls.
    from dynamics_model import CalibrationParameters
    param_names = CalibrationParameters.from_vector(param_vec).param_names
    _, S, Vt = np.linalg.svd(J, full_matrices=False)
    n_flat_to_report = min(3, len(S))
    print(f"\nSmallest {n_flat_to_report} singular value(s) and their parameter-combination "
          f"directions (multi-parameter flat-direction check, not just pairwise):")
    near_null_directions = []
    for i in range(1, n_flat_to_report + 1):
        sv = S[-i]
        direction = Vt[-i]
        order = np.argsort(-np.abs(direction))
        print(f"  Singular value {sv:.4e} (rank {len(S) - i + 1}/{len(S)}) — "
              f"parameters with the largest role in this barely-constrained combination:")
        top_params = []
        for j in order[:5]:
            print(f"      {param_names[j]:<18}{direction[j]:+.3f}")
            top_params.append({"parameter": param_names[j], "loading": float(direction[j])})
        near_null_directions.append({"singular_value": float(sv), "top_loadings": top_params})

    JtJ = J.T @ J
    try:
        # Cov(params) ~ (J^T J)^-1 up to an unknown noise-variance scale
        # factor — that factor cancels out of the CORRELATION matrix even
        # though it affects the absolute covariance magnitude.
        cov = np.linalg.inv(JtJ)
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)
    except np.linalg.LinAlgError:
        corr = None

    return {
        "jacobian": J, "condition_number": condition_number, "correlation": corr,
        "near_null_directions": near_null_directions,
    }


def report_correlated_pairs(corr: np.ndarray, param_names: tuple, threshold: float = 0.8) -> list:
    if corr is None:
        return []
    pairs = []
    n = len(param_names)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(corr[i, j]) >= threshold:
                pairs.append((param_names[i], param_names[j], float(corr[i, j])))
    return sorted(pairs, key=lambda p: -abs(p[2]))


# ---------------------------------------------------------------------------
# 5. Real vs. simulated velocity plots — especially roll/pitch/yaw
# ---------------------------------------------------------------------------

DOF_FULL_LABELS = [
    "u — surge (m/s)", "v — sway (m/s)", "w — heave (m/s)",
    "p — roll rate (rad/s)", "q — pitch rate (rad/s)", "r — yaw rate (rad/s)",
]
ROTATIONAL_DOF_INDICES = {3, 4, 5}  # p, q, r — the weak axes per the RMSE breakdown


def plot_real_vs_simulated(params, run, vehicle, output_path: str, title_suffix: str = "") -> None:
    """
    One PNG, 6 stacked subplots (one per DOF), real vs. simulated velocity
    over the run's full duration. Rotational DOFs (p, q, r) get a light
    red background tint — RMSE numbers alone don't show WHERE in a
    trajectory the fit breaks down; this does.
    """
    if not HAVE_MATPLOTLIB:
        raise RuntimeError(
            "matplotlib is required for plotting and isn't installed. "
            "Run: pip install matplotlib"
        )

    sim_config = SimulationConfig(dt=run.dt)
    trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                    roll_pitch_series=run.roll_pitch)
    result = simulate_trajectory(params, vehicle, sim_config, trajectory)
    sim_nu, real_nu = result.nu, run.real_nu
    t = np.arange(len(real_nu)) * run.dt

    fig, axes = plt.subplots(6, 1, figsize=(11, 15), sharex=True)
    for i, (ax, label) in enumerate(zip(axes, DOF_FULL_LABELS)):
        if i in ROTATIONAL_DOF_INDICES:
            ax.set_facecolor("#fff0f0")  # light red = rotational / known-weak axis
        ax.plot(t, real_nu[:, i], label="real", linewidth=1.1, color="black")
        ax.plot(t, sim_nu[:, i], label="simulated", linewidth=1.1, color="tab:red", linestyle="--")
        ax.set_ylabel(label, fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Real vs. simulated velocity — {Path(run.filepath).name}{title_suffix}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def median_rmse_run(params, runs: list, vehicle):
    """The run closest to typical, not the worst-case outlier — worth
    plotting both, since the worst run alone can be misleadingly extreme."""
    ranked = sorted(runs, key=lambda r: rmse_for_run(params, r, vehicle))
    return ranked[len(ranked) // 2]


DOF_INDEX = {"u": 0, "v": 1, "w": 2, "p": 3, "q": 4, "r": 5}


def plot_force_breakdown(params, run, vehicle, dof: str, output_path: str) -> None:
    """
    Decomposes v_dot = (tau + damping_force - restoring_force) / M_diag
    for ONE DOF, term by term, instead of only looking at the combined
    result. This is what actually tests whether the restoring-force term
    is responsible for a slow, input-independent drift — the shape of the
    combined real-vs-simulated plot can only suggest that; this confirms
    or rules it out directly.
    """
    if not HAVE_MATPLOTLIB:
        raise RuntimeError("matplotlib is required for plotting — run: pip install matplotlib")

    idx = DOF_INDEX[dof]
    sim_config = SimulationConfig(dt=run.dt, debug=True)
    trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                    roll_pitch_series=run.roll_pitch)
    result = simulate_trajectory(params, vehicle, sim_config, trajectory)

    t = np.arange(run.n_samples) * run.dt
    tau = run.forces_torques[:, idx]
    damping = result.debug_info["damping_force"][:, idx]
    restoring = result.debug_info["restoring_force"][:, idx]  # this is g_eta; SUBTRACTED in the equation
    accel = result.debug_info["acceleration"][:, idx]

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    ax.plot(t, tau, label="tau (logged thruster torque)", linewidth=0.8, color="tab:blue")
    ax.plot(t, damping, label="damping force term", linewidth=0.8, color="tab:green")
    ax.plot(t, -restoring, label="-restoring force (g_eta, as it enters v_dot)", linewidth=0.8, color="tab:orange")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax.set_ylabel(f"contributions to {dof}_dot numerator")
    ax.set_title(f"Force breakdown — {dof} ({['u','v','w','p','q','r'][idx]}) — {Path(run.filepath).name}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t, accel, label="resulting acceleration", color="tab:red")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(t, run.real_nu[:, idx], label="real velocity (reference, right axis)",
                  color="black", alpha=0.4, linewidth=0.8)
    ax2.set_ylabel("acceleration", color="tab:red")
    ax2_twin.set_ylabel("real velocity (reference)", color="black")
    ax2.set_xlabel("time (s)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--calibrated-params", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--heldout-dates", nargs="*", default=None)
    parser.add_argument("--skip-jacobian", action="store_true",
                         help="Skip the ~10 min identifiability analysis (steps 1-3 only)")
    parser.add_argument("--correlation-threshold", type=float, default=0.8)
    parser.add_argument("--plot", action="store_true",
                         help="Save real-vs-simulated velocity plots (worst + median run)")
    parser.add_argument("--plot-output-dir", default="plots")
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

    print("=" * 70)
    print("1. ACTIVE BOUNDS REPORT")
    print("=" * 70)
    hits = active_bounds_report(loaded_spec, calibrated)
    print(f"{len(hits)}/{len(calibrated.param_names)} parameters at a bound:")
    for name, side, bound_val, actual_val in hits:
        print(f"  {name:<16} at {side} bound ({bound_val})")
    if not hits:
        print("  None — no parameters pinned to a bound.")

    print("\n" + "=" * 70)
    print("2. PROFILING (single simulate_trajectory() call)")
    print("=" * 70)
    profile_output = profile_single_evaluation(calibrated, train_runs[0], vehicle)
    print(profile_output)

    print("=" * 70)
    print("3. PER-DOF WORST-RUN COMPARISON")
    print("=" * 70)
    per_dof_worst_run_report(calibrated, train_runs, vehicle)

    if args.plot:
        print("\n" + "=" * 70)
        print("5. REAL VS. SIMULATED VELOCITY PLOTS")
        print("=" * 70)
        if not HAVE_MATPLOTLIB:
            print("matplotlib not installed — run: pip install matplotlib")
        else:
            out_dir = Path(args.plot_output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)

            worst = max(train_runs, key=lambda r: rmse_for_run(calibrated, r, vehicle))
            typical = median_rmse_run(calibrated, train_runs, vehicle)

            worst_path = out_dir / "worst_run.png"
            typical_path = out_dir / "typical_run.png"
            plot_real_vs_simulated(calibrated, worst, vehicle, str(worst_path), " (WORST run)")
            plot_real_vs_simulated(calibrated, typical, vehicle, str(typical_path), " (median/typical run)")
            print(f"Saved: {worst_path}")
            print(f"Saved: {typical_path}")
            print("Rotational DOFs (p, q, r) have a light red background — "
                  "compare how much wider the real/simulated gap is there "
                  "vs. the translational DOFs above them.")

            for dof in ("p", "q"):
                breakdown_path = out_dir / f"force_breakdown_{dof}.png"
                plot_force_breakdown(calibrated, worst, vehicle, dof, str(breakdown_path))
                print(f"Saved: {breakdown_path}")
            print("Force breakdown plots decompose tau / damping / restoring-force "
                  "contributions separately for roll and pitch, on the worst run — "
                  "shows directly which term is driving the mismatch, rather than "
                  "inferring it from the combined output shape.")

    if args.skip_jacobian:
        print("\n--skip-jacobian set — skipping identifiability analysis.")
        return

    print("\n" + "=" * 70)
    print("4. JACOBIAN-BASED IDENTIFIABILITY ANALYSIS")
    print("=" * 70)
    param_vec = calibrated.to_vector()
    result = identifiability_analysis(param_vec, train_runs, vehicle)
    print(f"\nCondition number: {result['condition_number']:.3e}")
    print("(Very large values, e.g. >1e6, indicate the Jacobian is close to "
          "rank-deficient — some parameter combinations are barely "
          "distinguishable from others given this data.)")

    pairs = report_correlated_pairs(result["correlation"], calibrated.param_names,
                                      threshold=args.correlation_threshold)
    print(f"\nParameter pairs with |correlation| >= {args.correlation_threshold}:")
    if pairs:
        for name_a, name_b, corr_val in pairs:
            print(f"  {name_a:<16} <-> {name_b:<16}  corr = {corr_val:+.3f}")
    else:
        print("  None — no strongly correlated pairs at this threshold.")


if __name__ == "__main__":
    main()