"""
classify_state_difficulty.py — uses the ALREADY-VALIDATED heuristic
controller (check_alternative_controller.py, confirmed 32/32 on the
sparse diagnostic grid) to measure actual recovery difficulty across a
FINE grid, producing a data-driven lookup table for curriculum staging.

Difficulty combines TWO signals, both rank-normalized (so no magic
thresholds are needed, and it auto-calibrates to whatever the actual
measured distribution turns out to be):
  1. time_difficulty  — steps_run to reach formation_achieved (any
     non-success gets a large penalty so it ranks as hardest, handled
     defensively; shouldn't occur within the validated range)
  2. safety_risk       — how close the heuristic itself came to
     collision (smaller min_separation = higher risk)

REVISION: the first version of this script used ONLY time_difficulty.
Staged curriculum training built on that scored 34.4% success (matching
the fixed-start baseline) across the full randomized-start grid, but
ALSO produced the worst collision rate of any run so far (12.5%, vs.
6.25% for plain broad randomization) — concentrated at moderate
separations paired with extreme velocities (e.g. d0=3.0 at v0=+/-0.7).
Time-to-success and collision-margin are different axes; a state the
heuristic solves quickly can still be one it passes close to the safety
boundary while doing so. This revision adds safety_risk as a second,
independently-weighted difficulty signal so those states are correctly
staged later rather than mixed in with genuinely easy ones.
"""
import argparse
import csv

import numpy as np

from check_alternative_controller import build_env, run_case

FAILURE_PENALTY_STEPS = 1000  # sorts any non-success into the hardest tier


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Maps values to their percentile rank in [0,1]. Robust to whatever
    scale/distribution the raw values actually have -- no threshold
    constants required."""
    n = len(values)
    order = np.argsort(values)
    ranks = np.empty(n)
    ranks[order] = np.arange(n) / max(n - 1, 1)
    return ranks


def sweep(separations, velocities, seed: int = 0):
    env = build_env()
    rows = []
    for d0 in separations:
        for v0 in velocities:
            result = run_case(env, d0, v0, seed)
            success = result["terminal_reason"] == "formation_achieved"
            time_raw = result["steps_run"] if success else FAILURE_PENALTY_STEPS
            rows.append({
                "d0": d0, "v0": v0, "steps_run": result["steps_run"],
                "terminal_reason": result["terminal_reason"],
                "min_separation": result["min_separation"],
                "success": success, "time_raw": time_raw,
            })
    return rows


def main(d_step: float, v_step: float, csv_path: str, safety_weight: float):
    separations = list(np.arange(2.0, 5.5 + 1e-9, d_step))
    velocities = list(np.arange(-0.7, 0.7 + 1e-9, v_step))
    print(f"Sweeping {len(separations)}x{len(velocities)}={len(separations)*len(velocities)} "
          f"points with the heuristic controller...")
    rows = sweep(separations, velocities)

    time_raw = np.array([r["time_raw"] for r in rows])
    min_sep = np.array([r["min_separation"] for r in rows])
    time_rank = rank_normalize(time_raw)
    safety_risk_rank = rank_normalize(-min_sep)  # smaller min_separation -> higher rank -> riskier
    difficulty = time_rank + safety_weight * safety_risk_rank

    for i, r in enumerate(rows):
        r["time_rank"] = time_rank[i]
        r["safety_risk_rank"] = safety_risk_rank[i]
        r["difficulty"] = difficulty[i]
        del r["time_raw"]  # intermediate value, not needed downstream

    from evaluate_recovery import safe_csv_path
    actual_path = safe_csv_path(csv_path)
    with open(actual_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written to {actual_path}")

    n_failures = sum(1 for r in rows if not r["success"])
    print(f"\n{n_failures}/{len(rows)} points were NOT cleanly solved by the heuristic "
          f"at this resolution (expected: 0 -- investigate if this is nonzero)")

    difficulties_sorted = sorted(difficulty)
    print(f"\nCombined difficulty distribution (safety_weight={safety_weight}):")
    print(f"  min={difficulties_sorted[0]:.3f}  "
          f"p25={np.percentile(difficulties_sorted, 25):.3f}  "
          f"p50={np.percentile(difficulties_sorted, 50):.3f}  "
          f"p75={np.percentile(difficulties_sorted, 75):.3f}  "
          f"max={difficulties_sorted[-1]:.3f}")

    print(f"\nThe 10 states with the highest safety_risk_rank (came closest to collision), "
          f"for a sanity check against the specific collision cells you observed:")
    worst_safety = sorted(rows, key=lambda r: -r["safety_risk_rank"])[:10]
    for r in worst_safety:
        print(f"  d0={r['d0']:.2f} v0={r['v0']:+.2f}: min_separation={r['min_separation']:.3f}, "
              f"safety_risk_rank={r['safety_risk_rank']:.3f}, difficulty={r['difficulty']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d-step", type=float, default=0.25)
    parser.add_argument("--v-step", type=float, default=0.1)
    parser.add_argument("--csv-path", default="state_difficulty.csv")
    parser.add_argument("--safety-weight", type=float, default=1.0,
                         help="How much closeness-to-collision counts relative to "
                              "time-to-success (both rank-normalized to [0,1]). "
                              "1.0 = equal weight. Raise this if collision rate is "
                              "still too high after this fix; lower it if the "
                              "curriculum is now too conservative/slow.")
    args = parser.parse_args()
    main(args.d_step, args.v_step, args.csv_path, args.safety_weight)