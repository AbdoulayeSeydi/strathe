#!/usr/bin/env python3
"""
Loads and inspects BlueROV2 trajectory-dataset CSVs (oceansystemslab format).

Usage:
    python3 data/loader.py data/raw/Bluerov2_trajectory_dataset/**/*.csv

For each file, reports:
  - whether all expected columns are present
  - sample count and (if a time-like column exists) approximate sample rate
  - NaN/gap check
  - MANEUVER DIVERSITY: min/max/std of body-frame linear + angular velocity,
    and how many distinct DOFs see meaningful excitation. This is the check
    that tells us whether this dataset is a real stress test (coupled,
    aggressive maneuvers) or just gentle single-axis motion.
"""
import sys
import glob
import numpy as np
import pandas as pd

EXPECTED_COLS = [
    "x", "y", "z",
    "qx", "qy", "qz", "qw",
    "Bu", "Bv", "Bw",
    "Bq", "Bp", "Br",
    "Ux", "Uy", "Uz", "Vx", "Vy", "Vz",
    "Fx", "Fy", "Fz", "Tx", "Ty", "Tz",
] + [f"pwm{i}" for i in range(1, 9)]

TIME_LIKE_CANDIDATES = ["time", "t", "stamp", "timestamp", "Time"]


def inspect_file(path: str):
    print(f"\n{'=' * 70}\n{path}\n{'=' * 70}")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  FAILED TO READ: {e}")
        return

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLS and c not in TIME_LIKE_CANDIDATES]
    print(f"  Rows: {len(df)}")
    if missing:
        print(f"  MISSING EXPECTED COLUMNS: {missing}")
    else:
        print("  All expected columns present.")
    if extra:
        print(f"  Extra/unrecognized columns: {extra}")

    time_col = next((c for c in TIME_LIKE_CANDIDATES if c in df.columns), None)
    if time_col:
        dt = np.diff(df[time_col].values)
        dt = dt[dt > 0]
        if len(dt):
            print(f"  Time column: '{time_col}' — median dt={np.median(dt):.4f}s "
                  f"(~{1/np.median(dt):.1f} Hz), duration={df[time_col].iloc[-1]-df[time_col].iloc[0]:.1f}s")
    else:
        print("  No explicit time column found — assuming fixed-rate samples (row index = time step).")

    nan_counts = df[[c for c in EXPECTED_COLS if c in df.columns]].isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if len(nan_cols):
        print(f"  NaNs found in columns: {dict(nan_cols)}")
    else:
        print("  No NaNs in expected columns.")

    vel_cols = [c for c in ["Bu", "Bv", "Bw", "Bq", "Bp", "Br"] if c in df.columns]
    if vel_cols:
        print("\n  Maneuver diversity (body-frame velocities):")
        excited_dofs = 0
        for c in vel_cols:
            v = df[c].dropna().values
            if len(v) == 0:
                continue
            vmin, vmax, vstd = v.min(), v.max(), v.std()
            rng = vmax - vmin
            note = "excited" if rng > 1e-3 else "~flat/near-zero"
            print(f"    {c}: min={vmin:.4f} max={vmax:.4f} std={vstd:.4f}  [{note}]")
            if rng > 1e-3:
                excited_dofs += 1
        print(f"  --> {excited_dofs}/{len(vel_cols)} DOFs show non-trivial motion in this file.")
        if excited_dofs <= 2:
            print("  FLAG: looks like single/few-axis motion — a gentle case, not a coupled stress test.")
        else:
            print("  Multiple DOFs excited — closer to the coupled, harder case we want for validation.")

    pwm_cols = [c for c in df.columns if c.startswith("pwm") and c[3:].isdigit()]
    if pwm_cols:
        print("\n  Thruster (PWM) utilization:")
        for c in pwm_cols:
            v = df[c].dropna().values
            if len(v):
                print(f"    {c}: min={v.min():.0f} max={v.max():.0f} std={v.std():.1f}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = []
    for pattern in sys.argv[1:]:
        paths.extend(glob.glob(pattern, recursive=True))
    if not paths:
        print("No files matched the given pattern(s).")
        sys.exit(1)

    for p in sorted(paths):
        inspect_file(p)

    print(f"\n{'=' * 70}\nInspected {len(paths)} file(s).\n{'=' * 70}")


if __name__ == "__main__":
    main()
