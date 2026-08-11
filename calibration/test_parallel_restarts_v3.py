"""
Same as v2, but forces verbose=2 inside each worker (real, per-evaluation
scipy output) via a small monkeypatch-free approach: importing
calibrate_parallel's dependencies directly and building ONE single-restart
call manually with verbose=2, to see real, live per-evaluation progress
- distinguishing "hit the known, pre-existing performance mystery" from
"something new and different is actually wrong."

Deliberately tests with n_restarts=1 (not 4) first - simpler, isolates
whether even ONE restart shows real progress before trusting 4 at once.
"""
import os
from optimizer import (load_vehicle_spec, CalibrationConfig, calibrate,
                        initial_parameters, parameter_bounds)
from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run
import time


def main():
    csv_paths, _ = discover_csv_files('../data/raw/Bluerov2_trajectory_dataset')
    split = split_by_date(csv_paths, heldout_dates=['16-02-2024', '19-02-2024'])
    train_runs = [load_run(fp, DatasetConfig(sampling_rate_hz=10.0)) for fp in split.train_files[:3]]
    loaded = load_vehicle_spec('vehicle_spec.yaml')
    print(f"Loaded {len(train_runs)} training run(s). Running a SINGLE restart "
          f"(not parallel, not 4x) with verbose=2 for real, live visibility...", flush=True)

    # Deliberately using the ALREADY-PROVEN, sequential calibrate() here,
    # NOT calibrate_parallel() - isolating whether the underlying
    # single-restart computation itself is slow (the known mystery),
    # separate from anything about the new parallel code.
    config = CalibrationConfig(n_restarts=1, seed=42, max_nfev=50,
                                initialization_strategy='perturbed', verbose=2)
    t0 = time.monotonic()
    result = calibrate(loaded, train_runs, config)
    elapsed = time.monotonic() - t0
    print(f'Done in {elapsed:.1f}s, converged={result.converged}, cost={result.cost:.4f}')


if __name__ == "__main__":
    main()