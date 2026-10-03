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
import glob
import json
import os
import subprocess
import sys


def sampling_rate_source(data_dir: str, requested_hz: float) -> tuple:
    """How the sampling rate can be checked for this folder: ('ingested', message, ok) when
    data_ingestion.py measured it from the logs' own timestamps; ('no_positions', message, True)
    when the files have no x/y/z columns to infer it from (the rate given is used as is);
    ('infer', None, None) to run check_sampling_rate_windowed.py on the logged positions."""
    report = os.path.join(data_dir, "ingest_report.json")
    if os.path.exists(report):
        with open(report, encoding="utf-8") as f:
            measured = json.load(f).get("sampling_rate_hz")
        if measured:
            ok = abs(measured - requested_hz) <= 0.01 * measured
            msg = (f"The logs' own timestamps give {measured:g} Hz (measured during conversion, "
                   f"ingest_report.json); calibration will use {requested_hz:g} Hz.")
            if not ok:
                msg += (f" These disagree: pass --sampling-rate-hz {measured:g} (strathe calibrate does this "
                        f"automatically), or the physics will be integrated at the wrong time step.")
            return "ingested", msg, ok
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if files:
        with open(files[0], encoding="utf-8") as f:
            header = f.readline().strip().split(",")
        if not {"x", "y", "z"} <= set(header):
            return ("no_positions", f"These logs have no position columns (x, y, z) to infer the rate from, and "
                    f"no timestamps; calibration will assume {requested_hz:g} Hz. If that's wrong, pass "
                    f"--sampling-rate-hz (converting logs with timestamps records it automatically).", True)
    return "infer", None, None


def run_check(name: str, cmd: list) -> bool:
    # flush first: when the output goes to a file (the web page's job logs), the check's own output
    # would otherwise land before this header
    print(f"\n{'=' * 70}\nPRE-FLIGHT CHECK: {name}\n{'=' * 70}", flush=True)
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
    parser.add_argument("--heldout-dates", nargs="*", default=None,
                         help="Forwarded to check_coriolis_excitation.py's own --heldout-dates. "
                              "A real gap found and fixed: without this, that check always fell "
                              "back to ITS OWN internal default dates, which don't exist in a "
                              "smaller/different dataset - producing a confusing, unrelated-"
                              "looking failure. Only forwarded when explicitly given, so behavior "
                              "against the standard full dataset stays exactly unchanged by default.")
    args = parser.parse_args()

    results = {}

    how, msg, ok = sampling_rate_source(args.data_dir, args.sampling_rate_hz)
    if how == "infer":
        results["sampling_rate"] = run_check(
            "Sampling rate validation",
            [sys.executable, "check_sampling_rate_windowed.py",
             "--data-dir", args.data_dir,
             "--candidates", *[str(c) for c in args.rate_candidates]],
        )
    else:
        print(f"\n{'=' * 70}\nPRE-FLIGHT CHECK: Sampling rate\n{'=' * 70}\n{msg}")
        print(f"\n{'PASS' if ok else 'FAIL'}: Sampling rate")
        results["sampling_rate"] = ok
    _coriolis_cmd = [sys.executable, "check_coriolis_excitation.py",
                      "--data-dir", args.data_dir, "--sampling-rate-hz", str(args.sampling_rate_hz)]
    if args.heldout_dates:
        _coriolis_cmd += ["--heldout-dates", *args.heldout_dates]
    results["coriolis_excitation"] = run_check(
        "Coriolis-relevant velocity product excitation",
        _coriolis_cmd,
    )
    results["schema"] = run_check(
        "Schema validation (required columns)",
        [sys.executable, "check_schema.py", "--data-dir", args.data_dir],
    )
    results["data_integrity"] = run_check(
        "Structural data integrity (empty files, malformed rows, NaN/Inf)",
        [sys.executable, "check_data_integrity.py", "--data-dir", args.data_dir],
    )
    results["physical_plausibility"] = run_check(
        "Physical/units plausibility",
        [sys.executable, "check_physical_plausibility.py",
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