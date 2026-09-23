"""
evaluate_recovery_grid.py — maps the actual basin of attraction across a
grid of (initial separation, initial relative surge velocity) starting
states, multiple seeds per cell, using the SAME run_scenario() as
evaluate_recovery.py (imported directly, not duplicated).

Default grid matches the 8x4 proposal: separations 2.0-5.5 (step 0.5) x
velocities {-0.7,-0.15,+0.15,+0.7}, 5 seeds each = 160 scenario runs.

Each cell is classified per-seed as:
  S = full hold_steps reached (success)
  R = re-entered tolerance but never completed the hold
  F = never recovered into tolerance at all
printed as counts (e.g. "3S/1R/1F") so seed-to-seed variance is visible,
not collapsed into a single verdict.

Also serves item 3 (repeat a single point across many seeds) directly --
pass a single separation and single velocity with a higher --n-seeds to
concentrate all runs on one cell, e.g.:
  python3 evaluate_recovery_grid.py --model-path X --separations 5.0 \\
      --velocities 0.7 --n-seeds 20
"""
import argparse
import csv

from evaluate_recovery import build_env, run_scenario, safe_csv_path

DEFAULT_SEPARATIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
DEFAULT_VELOCITIES = [-0.7, -0.15, 0.15, 0.7]
DEFAULT_N_SEEDS = 5


def main(model_path: str, separations: list, velocities: list, n_seeds: int, csv_path: str):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env = build_env()
    true_original_step = env._orchestrator.step

    all_results = []
    total = len(separations) * len(velocities) * n_seeds
    done = 0
    for sep0 in separations:
        for vel0 in velocities:
            for seed in range(n_seeds):
                result = run_scenario(env, model, true_original_step, sep0, vel0, seed)
                all_results.append(result)
                done += 1
                if done % 10 == 0 or done == total:
                    print(f"  progress: {done}/{total} scenarios run", flush=True)

    actual_csv_path = safe_csv_path(csv_path)
    with open(actual_csv_path, "w", newline="") as f:
        fieldnames = list(all_results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nFull per-scenario results ({len(all_results)} rows) written to {actual_csv_path}\n")

    print("=" * 100)
    print("BASIN-OF-ATTRACTION MATRIX (counts per classification, out of n_seeds)")
    print("S=full hold achieved, R=re-entered tolerance but failed hold, "
          "C=collision, F=never recovered")
    print("=" * 100)
    header = f"{'d0 \\ v0':>10}" + "".join(f"{v:>16.2f}" for v in velocities)
    print(header)
    for sep0 in separations:
        row_cells = []
        for vel0 in velocities:
            cell_results = [r for r in all_results
                             if r["initial_separation"] == sep0 and r["initial_surge_velocity"] == vel0]
            n_s = sum(1 for r in cell_results if r["classification"] == "S")
            n_r = sum(1 for r in cell_results if r["classification"] == "R")
            n_c = sum(1 for r in cell_results if r["classification"] == "C")
            n_f = sum(1 for r in cell_results if r["classification"] == "F")
            row_cells.append(f"{n_s}S/{n_r}R/{n_c}C/{n_f}F")
        print(f"{sep0:>10.2f}" + "".join(f"{c:>16}" for c in row_cells))

    print("\n" + "=" * 100)
    print("MEAN MIN_SEPARATION MATRIX (collision-proximity risk, d_safe=0.5)")
    print("=" * 100)
    print(header)
    for sep0 in separations:
        row_cells = []
        for vel0 in velocities:
            cell_results = [r for r in all_results
                             if r["initial_separation"] == sep0 and r["initial_surge_velocity"] == vel0]
            mean_min_sep = sum(r["min_separation"] for r in cell_results) / len(cell_results)
            row_cells.append(f"{mean_min_sep:.3f}")
        print(f"{sep0:>10.2f}" + "".join(f"{c:>16}" for c in row_cells))

    print("\n" + "=" * 100)
    print("MEAN FRACTION_STEPS_SATURATED MATRIX")
    print("=" * 100)
    print(header)
    for sep0 in separations:
        row_cells = []
        for vel0 in velocities:
            cell_results = [r for r in all_results
                             if r["initial_separation"] == sep0 and r["initial_surge_velocity"] == vel0]
            mean_sat = sum(r["fraction_steps_saturated"] for r in cell_results) / len(cell_results)
            row_cells.append(f"{mean_sat:.2f}")
        print(f"{sep0:>10.2f}" + "".join(f"{c:>16}" for c in row_cells))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--separations", type=float, nargs="+", default=DEFAULT_SEPARATIONS)
    parser.add_argument("--velocities", type=float, nargs="+", default=DEFAULT_VELOCITIES)
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS,
                         help="Seeds per (separation, velocity) cell. Use a single "
                              "separation and velocity with a high --n-seeds (e.g. 20) "
                              "to concentrate all runs on one point instead of a grid.")
    parser.add_argument("--csv-path", default="recovery_grid.csv")
    args = parser.parse_args()
    main(args.model_path, args.separations, args.velocities, args.n_seeds, args.csv_path)