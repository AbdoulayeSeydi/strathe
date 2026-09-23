"""
Calibration stability check — runs build_calibrated_parameters() across
multiple seeds (and optionally different budgets) and prints a
comparison table of exactly the fields that matter for the surge-pitch
coupling question: is Xq_dot/Zq_dot's sign and order of magnitude
consistent across independent fits, or does it swing around (meaning
the data doesn't actually constrain it, and the "coupling" story built
on top of it shouldn't be trusted)?

REAL TIME COST, stated upfront: at the confirmed ~74s/call rate on the
real 5-file training set, each max_nfev=15 run takes ~74 minutes. This
script runs len(seeds) such calibrations SEQUENTIALLY — budget your time
accordingly before running with more than 1-2 seeds. Pass --max-nfev to
try a larger budget per run if you want to test convergence stability
too, not just seed stability (that will take proportionally longer per
run, per the already-confirmed ~21x real-call multiplier).
"""
import time

from build_calibrated_parameters import build_calibrated_parameters

FIELDS_TO_REPORT = ["Xu", "Xuu", "Xq_dot", "Zq_dot"]


def main(data_dir: str, seeds: list, max_nfev: int, max_train_runs: int):
    results = []
    for seed in seeds:
        print(f"\n{'='*60}\nRunning calibration with seed={seed}, max_nfev={max_nfev}...")
        t0 = time.monotonic()
        params = build_calibrated_parameters(
            data_dir, n_restarts=1, max_nfev=max_nfev, seed=seed,
            output_path=f"calibrated_parameters_stability_seed{seed}.yaml",
            max_train_runs=max_train_runs,
        )
        elapsed = time.monotonic() - t0
        row = {"seed": seed, "elapsed_s": elapsed}
        for field in FIELDS_TO_REPORT:
            row[field] = getattr(params, field)
        results.append(row)
        print(f"Completed in {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    header = f"{'seed':>6} " + " ".join(f"{f:>12}" for f in FIELDS_TO_REPORT) + f" {'time(s)':>10}"
    print(header)
    for row in results:
        line = f"{row['seed']:>6} " + " ".join(f"{row[f]:>12.4f}" for f in FIELDS_TO_REPORT)
        line += f" {row['elapsed_s']:>10.1f}"
        print(line)

    xq_vals = [row["Xq_dot"] for row in results]
    zq_vals = [row["Zq_dot"] for row in results]
    print(f"\nXq_dot: min={min(xq_vals):.4f}, max={max(xq_vals):.4f}, "
          f"all same sign: {all(v > 0 for v in xq_vals) or all(v < 0 for v in xq_vals)}")
    print(f"Zq_dot: min={min(zq_vals):.4f}, max={max(zq_vals):.4f}, "
          f"all same sign: {all(v > 0 for v in zq_vals) or all(v < 0 for v in zq_vals)}")
    print("\nConsistent sign AND comparable order of magnitude across seeds -> the "
          "coupling looks real, not fit noise. Sign flips or order-of-magnitude "
          "swings -> the data doesn't reliably constrain this parameter; don't "
          "build a training conclusion on top of it (same standard this project "
          "already applied to Ieff_roll/Ieff_pitch/net_buoyancy_z).")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 123],
                         help="Default: 3 seeds. Each takes real time (see module docstring) — "
                              "pass fewer (e.g. --seeds 42 7) to reduce total wall-clock cost.")
    parser.add_argument("--max-nfev", type=int, default=15)
    parser.add_argument("--max-train-runs", type=int, default=5)
    args = parser.parse_args()
    main(args.data_dir, args.seeds, args.max_nfev, args.max_train_runs)