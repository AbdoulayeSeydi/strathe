"""
Test script for calibrate_parallel() - MUST be run as a saved file, not
`python -c`, and MUST have the if __name__ == "__main__": guard below.

Why this matters specifically on Windows: the 'spawn' multiprocessing
start method (the only one Windows supports) re-imports/re-executes the
ENTIRE parent script to bootstrap each worker process. Without this
guard, code that CALLS calibrate_parallel() at module level would itself
run again inside every spawned worker - each of which would then try to
spawn its own workers, recursively. This is a well-known, serious Windows
multiprocessing gotcha, not specific to this code.
"""
from optimizer import load_vehicle_spec, calibrate_parallel, CalibrationConfig
from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run
import time


def main():
    csv_paths, _ = discover_csv_files('../data/raw/Bluerov2_trajectory_dataset')
    split = split_by_date(csv_paths, heldout_dates=['16-02-2024', '19-02-2024'])
    train_runs = [load_run(fp, DatasetConfig(sampling_rate_hz=10.0)) for fp in split.train_files[:3]]
    loaded = load_vehicle_spec('vehicle_spec.yaml')

    config = CalibrationConfig(n_restarts=4, seed=42, max_nfev=300, initialization_strategy='perturbed')
    t0 = time.monotonic()
    result = calibrate_parallel(loaded, train_runs, config)
    elapsed = time.monotonic() - t0
    print(f'Done in {elapsed:.1f}s, converged={result.converged}, cost={result.cost:.4f}, '
          f'n_converged={result.n_converged}/{result.n_restarts}')
    print(f'stability: {result.stability}')


if __name__ == "__main__":
    main()