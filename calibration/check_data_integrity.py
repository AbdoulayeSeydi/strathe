#!/usr/bin/env python3
"""
Structural data-integrity checks (item #2 follow-up) — the FAIL-level,
high-confidence half of garbage-input robustness: empty/near-empty
files, malformed CSV structure, non-numeric values, and NaN/Inf
poisoning. Deliberately does NOT attempt schema validation or physical
plausibility here (see check_physical_plausibility.py) - those are
lower-confidence, REVIEW-level checks that need vehicle-specific
context; these are unambiguous structural failures any real dataset
should never have, regardless of vehicle.

Follows preflight_check.py's existing contract exactly: prints
human-readable output, exits 0 (pass) or 1 (fail) - no other reporting
architecture, so it plugs into the same subprocess-based orchestration
already in place for the sampling-rate and Coriolis-excitation checks.

MIN_USABLE_ROWS is a real, confirmed number from this project's own
Coriolis-excitation work: below roughly this size, a run is too short to
meaningfully contribute grount-truth signal to a joint calibration
across many runs (not a hard requirement, but empirically the shortest
real runs already accepted in this project's own dataset).

Usage:
    python check_data_integrity.py --data-dir ../data/raw/Bluerov2_trajectory_dataset
"""
import argparse
import csv
import math
import sys
from pathlib import Path

MIN_USABLE_ROWS = 10  # a file with fewer real data rows than this isn't
# usable for calibration at all, regardless of what's in it


def check_file(filepath: Path) -> list:
    """Returns a list of (severity, message) tuples for this one file -
    empty list means the file is structurally clean."""
    issues = []

    try:
        with open(filepath, encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                issues.append(("ERROR", f"{filepath.name}: file is completely empty (no header row)."))
                return issues

            n_columns = len(header)
            rows = list(reader)
    except UnicodeDecodeError:
        issues.append(("ERROR", f"{filepath.name}: not valid UTF-8 text - "
                                 f"cannot be read as a CSV at all."))
        return issues

    if not rows:
        issues.append(("ERROR", f"{filepath.name}: header-only file, zero data rows."))
        return issues

    if len(rows) < MIN_USABLE_ROWS:
        issues.append(("ERROR", f"{filepath.name}: only {len(rows)} data row(s) - "
                                 f"below the minimum ({MIN_USABLE_ROWS}) to be usable "
                                 f"for calibration, even if otherwise well-formed."))

    malformed_row_numbers = []
    non_numeric_cells = []
    nan_inf_cells = []
    for row_idx, row in enumerate(rows, start=2):  # start=2: row 1 is the header
        if len(row) != n_columns:
            malformed_row_numbers.append(row_idx)
            continue  # can't meaningfully check cell contents of a malformed row
        for col_idx, cell in enumerate(row):
            try:
                value = float(cell)
            except ValueError:
                non_numeric_cells.append((row_idx, header[col_idx] if col_idx < len(header) else f"col{col_idx}", cell))
                continue
            if math.isnan(value) or math.isinf(value):
                nan_inf_cells.append((row_idx, header[col_idx] if col_idx < len(header) else f"col{col_idx}", value))

    if malformed_row_numbers:
        shown = malformed_row_numbers[:5]
        more = f" (+{len(malformed_row_numbers) - 5} more)" if len(malformed_row_numbers) > 5 else ""
        issues.append(("ERROR", f"{filepath.name}: {len(malformed_row_numbers)} row(s) have a "
                                 f"different column count than the header (expected {n_columns}) - "
                                 f"rows {shown}{more}. Likely a truncated write, a broken quote, or "
                                 f"an unescaped delimiter inside a field."))

    if non_numeric_cells:
        shown = non_numeric_cells[:5]
        more = f" (+{len(non_numeric_cells) - 5} more)" if len(non_numeric_cells) > 5 else ""
        detail = ", ".join(f"row {r} column '{c}'={v!r}" for r, c, v in shown)
        issues.append(("ERROR", f"{filepath.name}: {len(non_numeric_cells)} non-numeric value(s) "
                                 f"in what should be numeric data: {detail}{more}."))

    if nan_inf_cells:
        shown = nan_inf_cells[:5]
        more = f" (+{len(nan_inf_cells) - 5} more)" if len(nan_inf_cells) > 5 else ""
        detail = ", ".join(f"row {r} column '{c}'={v}" for r, c, v in shown)
        issues.append(("ERROR", f"{filepath.name}: {len(nan_inf_cells)} NaN/Inf value(s) found - "
                                 f"these will silently poison any calibration that uses this row: "
                                 f"{detail}{more}."))

    return issues


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    csv_files = sorted(Path(args.data_dir).rglob("*.csv"))
    if not csv_files:
        print(f"ERROR: no CSV files found under {args.data_dir}")
        sys.exit(1)

    print(f"Checking structural integrity of {len(csv_files)} file(s)...\n")
    all_issues = []
    for filepath in csv_files:
        issues = check_file(filepath)
        all_issues.extend(issues)
        for severity, message in issues:
            print(f"  {severity}: {message}")

    print()
    if all_issues:
        print(f"FAIL: {len(all_issues)} structural integrity issue(s) found across the dataset. "
              f"Fix or remove the affected file(s) before proceeding - a calibration run "
              f"using corrupted rows will produce a real but MEANINGLESS result, not an "
              f"obvious crash.")
        sys.exit(1)
    else:
        print(f"PASS: all {len(csv_files)} file(s) are structurally clean "
              f"(no empty files, no malformed rows, no non-numeric or NaN/Inf values).")
        sys.exit(0)


if __name__ == "__main__":
    main()