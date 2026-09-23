"""
find_first_divergence.py -- reads the per-step CSVs already produced by
compare_ppo_vs_heuristic.py and finds, for each trace, the FIRST step
where ppo_Fx and heuristic_Fx disagree in SIGN (not just magnitude) --
the sharpest, least arbitrary definition of "pointing the wrong way"
relative to the counterfactual heuristic action at that exact state.

This deliberately does NOT re-run the model or environment: every
column needed (separation, d_dot, pitch, q, ppo_Fx/Ty, heuristic_Fx/Ty,
saturation) is already in the CSVs compare_ppo_vs_heuristic.py wrote.

Near-zero actions can trivially "disagree in sign" from noise alone;
SIGN_THRESHOLD excludes both values being within a small band of zero
before checking sign disagreement, so a genuine large-magnitude
opposite-direction disagreement is what gets reported.
"""
import argparse
import csv

SIGN_THRESHOLD = 3.0  # Newtons -- below this on EITHER side, a "sign disagreement" is noise, not signal


def find_first_divergence(csv_path: str):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        ppo_fx = float(row["ppo_Fx"])
        heur_fx = float(row["heuristic_Fx"])
        if abs(ppo_fx) < SIGN_THRESHOLD or abs(heur_fx) < SIGN_THRESHOLD:
            continue
        if (ppo_fx > 0) != (heur_fx > 0):
            return row, rows
    return None, rows  # no sign disagreement found anywhere in the trace


def print_context(csv_path: str, divergence_row, all_rows, context_steps: int = 2):
    print(f"\n{'='*90}\n{csv_path}\n{'='*90}")
    if divergence_row is None:
        print("No Fx sign disagreement found anywhere in this trace (above the noise threshold).")
        return

    div_step = int(divergence_row["step"])
    print(f"First Fx sign disagreement at step {div_step} "
          f"(of {len(all_rows)} steps total in this trace):\n")

    start = max(0, div_step - 1 - context_steps)
    end = min(len(all_rows), div_step + context_steps)
    header = (f"{'step':>5} {'sep':>7} {'d_dot':>8} {'pitch':>8} {'q':>8} "
              f"{'ppo_Fx':>9} {'heur_Fx':>9} {'ppo_Ty':>9} {'heur_Ty':>9} {'sat':>6}")
    print(header)
    for row in all_rows[start:end]:
        marker = " <-- DIVERGENCE" if int(row["step"]) == div_step else ""
        print(f"{int(row['step']):>5} {float(row['separation']):>7.3f} {float(row['d_dot']):>8.3f} "
              f"{float(row['pitch']):>8.3f} {float(row['q']):>8.3f} "
              f"{float(row['ppo_Fx']):>9.2f} {float(row['heuristic_Fx']):>9.2f} "
              f"{float(row['ppo_Ty']):>9.2f} {float(row['heuristic_Ty']):>9.2f} "
              f"{row['ppo_saturated']:>6}{marker}")

    print(f"\nAt divergence: separation={float(divergence_row['separation']):.3f}, "
          f"d_dot={float(divergence_row['d_dot']):.3f}, pitch={float(divergence_row['pitch']):.3f}, "
          f"q={float(divergence_row['q']):.3f}")
    print(f"  ppo_Fx={float(divergence_row['ppo_Fx']):.2f} vs heuristic_Fx={float(divergence_row['heuristic_Fx']):.2f} "
          f"(opposite signs)")
    print(f"  ppo_saturated={divergence_row['ppo_saturated']}")
    print(f"  fraction of trace remaining after divergence: "
          f"{(len(all_rows) - div_step) / len(all_rows):.2f}")


def find_first_saturation(csv_path: str):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["ppo_saturated"] == "True":
            return row, rows
    return None, rows


def main(csv_paths):
    for path in csv_paths:
        divergence_row, all_rows = find_first_divergence(path)
        saturation_row, _ = find_first_saturation(path)
        print_context(path, divergence_row, all_rows)
        if saturation_row is not None:
            sat_step = int(saturation_row["step"])
            div_step = int(divergence_row["step"]) if divergence_row else None
            print(f"\nFirst ppo_saturated=True at step {sat_step}"
                  + (f" (vs first sign-disagreement at step {div_step}, "
                     f"a gap of {div_step - sat_step} steps)" if div_step else " (no sign-disagreement found)"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_paths", nargs="+", help="One or more compare_ppo_vs_heuristic.py output CSVs")
    args = parser.parse_args()
    main(args.csv_paths)