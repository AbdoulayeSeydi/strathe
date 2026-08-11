"""
Corrected version of the sequential (calibrate(), not parallel) diagnostic
test — properly truncates each run to 150 steps, matching the convention
used everywhere else in this project (run_calibration_holoocean.py's
truncate_run()), rather than the earlier version's real, meaningful
oversight of using full-length real runs (some 6000+ samples), which
explained the earlier ~5min-per-iteration real cost.
"""
from optimizer import load_vehicle_spec, CalibrationConfig, calibrate
from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run
from run_calibration_holoocean import truncate_run
import time


def main():
    csv_paths, _ = discover_csv_files('../data/raw/Bluerov2_trajectory_dataset')
    split = split_by_date(csv_paths, heldout_dates=['16-02-2024', '19-02-2024'])
    dataset_config = DatasetConfig(sampling_rate_hz=10.0)
    train_runs = [truncate_run(load_run(fp, dataset_config), 150) for fp in split.train_files[:3]]
    loaded = load_vehicle_spec('vehicle_spec.yaml')
    total_steps = sum(r.n_samples for r in train_runs)
    print(f"Loaded {len(train_runs)} training run(s), truncated to 150 steps each "
          f"({total_steps} total steps). Running a single restart with verbose=2...", flush=True)

    config = CalibrationConfig(n_restarts=1, seed=42, max_nfev=50,
                                initialization_strategy='perturbed', verbose=2)
    t0 = time.monotonic()
    result = calibrate(loaded, train_runs, config)
    elapsed = time.monotonic() - t0
    print(f'Done in {elapsed:.1f}s, converged={result.converged}, cost={result.cost:.4f}')


if __name__ == "__main__":
    main()