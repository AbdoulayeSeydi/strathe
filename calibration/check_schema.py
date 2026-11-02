#!/usr/bin/env python3
"""
Schema validation (item #2 follow-up, high-confidence FAIL-level check) -
fast, header-only check that every CSV has the columns
objective.py's REQUIRED_COLUMNS actually needs, run BEFORE committing
to loading (and potentially failing deep inside) the full dataset.

Deliberately imports REQUIRED_COLUMNS directly from objective.py rather
than maintaining a second, separately-typed list - this project has
already fixed real bugs caused by exactly that kind of duplication
(see PROJECT_STATUS.md's "no duplicated schema" principle, same
reasoning as optimizer.py reusing CalibrationParameters.param_names
instead of its own hand-maintained list).

Complements, not duplicates, objective.py's own validate_run_dataframe():
that function already catches this deep inside load_run() - this check
exists so the failure happens fast, upfront, and for EVERY file at once,
not one at a time as each file happens to get loaded during a real
(possibly hours-long) calibration run.

Usage:
    python check_schema.py --data-dir ../data/raw/Bluerov2_trajectory_dataset
"""
import argparse
import csv
import sys
from pathlib import Path

from objective import REQUIRED_COLUMNS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    csv_files = sorted(Path(args.data_dir).rglob("*.csv"))
    if not csv_files:
        print(f"ERROR: no CSV files found under {args.data_dir}")
        sys.exit(1)

    print(f"Checking schema of {len(csv_files)} file(s) against "
          f"objective.py's REQUIRED_COLUMNS ({len(REQUIRED_COLUMNS)} columns)...\n")

    any_failed = False
    for filepath in csv_files:
        try:
            with open(filepath, encoding="utf-8", newline="") as f:
                header = next(csv.reader(f), [])
        except UnicodeDecodeError:
            print(f"  ERROR: {filepath.name}: not valid UTF-8 - cannot even read the header.")
            any_failed = True
            continue

        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if set(missing) == {"Bu", "Bv", "Bw"}:
            # no DVL (PROJECT_STATUS.md #96): calibrated from the vehicle's other sensors instead
            sensors = sorted({c.rsplit("_", 1)[0] if c.startswith("pos_") else c for c in header
                              if c.startswith(("pos_", "depth_", "acc_"))})
            print(f"  note: {filepath.name}: no body velocity (no DVL); calibrated from "
                  + (", ".join(sensors) + " and the gyro" if sensors else "the gyro alone (rotation only)")
                  + " (SENSORS.md).")
            missing = []
        if missing:
            any_failed = True
            print(f"  ERROR: {filepath.name}: missing required column(s): {missing}. "
                  f"This looks like it may be a different log format, a different vehicle's "
                  f"export, or a corrupted/truncated header row.")

    print()
    if any_failed:
        print("FAIL: one or more files are missing columns this pipeline requires. "
              "Fix the file(s) or confirm they're genuinely the right dataset before proceeding.")
        sys.exit(1)
    else:
        print(f"PASS: all {len(csv_files)} file(s) have every required column.")
        sys.exit(0)


if __name__ == "__main__":
    main()