"""
evaluate_recovery_jitter.py — local sensitivity around ONE nominal
(separation, velocity) state. Answers: is an observed S/R/C/F outcome a
genuine region of state space, or an artifact of hitting one very
specific deterministic trajectory?

Builds a small, DENSE, deterministic grid of (d0 +/- jitter, v0 +/-
jitter) around the nominal point -- not random samples, so the
sharp-vs-graded pattern is directly visible in the printed matrix:
  S S S S F F F   -> sharp basin boundary
  S S S R R F F   -> graded degradation region

seed is fixed at 0 throughout (matching evaluate_recovery.py's
observation that this setup has zero stochasticity -- PacketLossConfig
has zero dropout, deterministic=True -- so "seed" cannot vary outcomes
here; only the nominal state itself is being varied, deliberately).
"""
import argparse
import csv

from evaluate_recovery import build_env, run_scenario, safe_csv_path


def main(model_path: str, d0: float, v0: float, d_radius: float, v_radius: float,
         n_points: int, csv_path: str):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env = build_env()
    true_original_step = env._orchestrator.step

    if n_points > 1:
        d_offsets = [d_radius * (2 * i / (n_points - 1) - 1) for i in range(n_points)]
        v_offsets = [v_radius * (2 * i / (n_points - 1) - 1) for i in range(n_points)]
    else:
        d_offsets, v_offsets = [0.0], [0.0]

    print(f"Nominal state: d0={d0}, v0={v0}")
    print(f"Jitter: d +/-{d_radius}m, v +/-{v_radius}m/s, {n_points}x{n_points} grid, "
          f"{n_points*n_points} total scenarios\n")

    all_results = []
    for d_off in d_offsets:
        for v_off in v_offsets:
            d_test, v_test = d0 + d_off, v0 + v_off
            result = run_scenario(env, model, true_original_step, d_test, v_test, seed=0)
            result["d_offset"], result["v_offset"] = d_off, v_off
            all_results.append(result)

    actual_csv_path = safe_csv_path(csv_path)
    with open(actual_csv_path, "w", newline="") as f:
        fieldnames = list(all_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"Full results ({len(all_results)} rows) written to {actual_csv_path}\n")

    def cell(d_off, v_off, key):
        return next(r[key] for r in all_results if r["d_offset"] == d_off and r["v_offset"] == v_off)

    header = f"{'d0+off':>10}" + "".join(f"{v0+vo:>8.3f}" for vo in v_offsets)

    print("CLASSIFICATION MATRIX (S=success, R=re-entered/no hold, C=collision, F=never recovered)")
    print(header)
    for d_off in d_offsets:
        print(f"{d0+d_off:>10.3f}" + "".join(f"{cell(d_off, vo, 'classification'):>8}" for vo in v_offsets))

    print("\nMIN_SEPARATION MATRIX")
    print(header)
    for d_off in d_offsets:
        print(f"{d0+d_off:>10.3f}" + "".join(f"{cell(d_off, vo, 'min_separation'):>8.2f}" for vo in v_offsets))

    print("\nInterpretation:")
    print("  Smooth blocks of the SAME letter -> sharp basin boundary; the original")
    print("  point's outcome is representative of a real surrounding region.")
    print("  Interleaved/mixed letters at this fine scale -> the original point sits")
    print("  near a genuine discontinuity or is itself fragile, not representative")
    print("  of a broader region -- worth treating its outcome with more caution.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--d0", type=float, required=True)
    parser.add_argument("--v0", type=float, required=True)
    parser.add_argument("--d-radius", type=float, default=0.05)
    parser.add_argument("--v-radius", type=float, default=0.02)
    parser.add_argument("--n-points", type=int, default=5, help="grid is n_points x n_points")
    parser.add_argument("--csv-path", default=None)
    args = parser.parse_args()
    csv_path = args.csv_path or f"jitter_d{args.d0}_v{args.v0}.csv"
    main(args.model_path, args.d0, args.v0, args.d_radius, args.v_radius, args.n_points, csv_path)