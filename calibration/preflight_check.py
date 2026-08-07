#!/usr/bin/env python3
"""
Automated data-quality pre-flight check (item #6) — wraps
check_sampling_rate_windowed.py and check_coriolis_excitation.py as
subprocesses (neither exposes importable functions - only main()/CLI -
so this reuses them exactly as-is rather than duplicating their logic).

Meant to run BEFORE committing real compute to a calibration, catching
data-quality problems early rather than discovering them hours in.

Usage:
    python preflight_check.py \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --spec vehicle_spec.yaml
"""
import argparse
import subprocess
import sys


def run_check(name: str, cmd: list) -> bool:
    print(f"\n{'=' * 70}\nPRE-FLIGHT CHECK: {name}\n{'=' * 70}")
    result = subprocess.run(cmd, capture_output=False)
    passed = result.returncode == 0
    print(f"\n{'PASS' if passed else 'FAIL (non-zero exit code)'}: {name}")
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, default=10.0)
    parser.add_argument("--rate-candidates", type=float, nargs="+", default=[5, 10, 20, 25, 50, 100],
                         help="Candidate sampling rates (Hz) to test on a NEW, unknown dataset — "
                              "the point of this check is determining which one is actually "
                              "correct, not assuming one. Override if you have reason to expect "
                              "a specific rate not in this default list.")
    args = parser.parse_args()

    results = {}

    results["sampling_rate"] = run_check(
        "Sampling rate validation",
        [sys.executable, "check_sampling_rate_windowed.py",
         "--data-dir", args.data_dir,
         "--candidates", *[str(c) for c in args.rate_candidates]],
    )
    results["coriolis_excitation"] = run_check(
        "Coriolis-relevant velocity product excitation",
        [sys.executable, "check_coriolis_excitation.py",
         "--data-dir", args.data_dir, "--sampling-rate-hz", str(args.sampling_rate_hz)],
    )

    print(f"\n{'=' * 70}\nPRE-FLIGHT SUMMARY\n{'=' * 70}")
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'} — {name}")

    all_passed = all(results.values())
    if all_passed:
        print("\nAll pre-flight checks passed — safe to proceed with a real calibration run.")
    else:
        print("\nSome checks failed or need review — read the output above before committing "
              "real compute time to a calibration run.")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()