"""
Synthetic-data generator — NOT real BlueROV2 data. Exists only so I can
exercise the full discover_csv_files -> split_by_date -> load_run ->
calibrate_parallel pipeline end-to-end in this sandbox (no real dataset
available here) before handing build_calibrated_parameters.py over.

Uses the project's ACTUAL required schema (objective.py's REQUIRED_COLUMNS)
and the ACTUAL PHASE1_TRAIN_DATES/PHASE1_HELDOUT_DATES strings, so
split_by_date()'s "unknown heldout group" check and
assert_uses_phase1_split() both exercise their real logic, not a
simplified stand-in. The physics results from this data are meaningless
(random small forces, no real vehicle behind them) — this only proves the
CODE PATH runs without crashing, never that any calibration NUMBER is
correct.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATES = ["12-01-2024", "14-02-2024", "16-02-2024", "19-02-2024"]  # exact PHASE1 dates
N_ROWS = 60
N_RUNS_PER_DATE = 2


def make_synthetic_run(rng: np.random.Generator, n_rows: int = N_ROWS) -> pd.DataFrame:
    # Small forces/torques -> small, stable velocities. Not physically
    # meaningful, just finite and well-behaved so simulate_trajectory()
    # doesn't blow up (NaN/inf) on this synthetic input.
    forces_torques = rng.normal(0, 0.5, size=(n_rows, 6))
    real_nu = np.cumsum(rng.normal(0, 0.01, size=(n_rows, 6)), axis=0)
    # Identity-ish quaternion with tiny noise (valid, near-zero roll/pitch)
    quats = np.zeros((n_rows, 4))
    quats[:, 3] = 1.0  # qw = 1, qx=qy=qz=0 -> identity rotation
    quats[:, :3] += rng.normal(0, 0.01, size=(n_rows, 3))
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    quats = quats / norms

    df = pd.DataFrame({
        "Fx": forces_torques[:, 0], "Fy": forces_torques[:, 1], "Fz": forces_torques[:, 2],
        "Tx": forces_torques[:, 3], "Ty": forces_torques[:, 4], "Tz": forces_torques[:, 5],
        "Bu": real_nu[:, 0], "Bv": real_nu[:, 1], "Bw": real_nu[:, 2],
        "Bp": real_nu[:, 3], "Bq": real_nu[:, 4], "Br": real_nu[:, 5],
        "qx": quats[:, 0], "qy": quats[:, 1], "qz": quats[:, 2], "qw": quats[:, 3],
    })
    return df


def generate(data_dir: str, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)
    for date in DATES:
        for run_idx in range(1, N_RUNS_PER_DATE + 1):
            df = make_synthetic_run(rng)
            path = out / f"synth-{date}-run{run_idx}.csv"
            df.to_csv(path, index=False)
            print(f"wrote {path} ({len(df)} rows)")


if __name__ == "__main__":
    generate("synth_data_for_testing", seed=0)