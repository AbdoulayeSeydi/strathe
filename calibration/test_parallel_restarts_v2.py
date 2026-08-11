"""
Same test as before, but with REAL, live progress visibility this time -
each worker prints the moment it actually starts computing, so silence
vs. genuine progress is directly distinguishable, not something to guess
at. Still requires the if __name__ == "__main__": guard (Windows spawn).
"""
import os
from optimizer import load_vehicle_spec, calibrate_parallel, CalibrationConfig
from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run
import time


def main():
    csv_paths, _ = discover_csv_files('../data/raw/Bluerov2_trajectory_dataset')
    split = split_by_date(csv_paths, heldout_dates=['16-02-2024', '19-02-2024'])
    train_runs = [load_run(fp, DatasetConfig(sampling_rate_hz=10.0)) for fp in split.train_files[:3]]
    loaded = load_vehicle_spec('vehicle_spec.yaml')
    print(f"Loaded {len(train_runs)} training run(s). Starting 4 parallel restarts "
          f"(PID {os.getpid()} is this main process)...", flush=True)

    config = CalibrationConfig(n_restarts=4, seed=42, max_nfev=50, initialization_strategy='perturbed')
    t0 = time.monotonic()
    result = calibrate_parallel(loaded, train_runs, config)
    elapsed = time.monotonic() - t0
    print(f'Done in {elapsed:.1f}s, converged={result.converged}, cost={result.cost:.4f}, '
          f'n_converged={result.n_converged}/{result.n_restarts}')
    print(f'stability: {result.stability}')


if __name__ == "__main__":
    main()