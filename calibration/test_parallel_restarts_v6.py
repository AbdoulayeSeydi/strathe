"""
Same as v5, but fixed to use CalibrationResult's REAL fields this time -
n_converged/n_restarts don't exist on it directly; they're available via
result.stability (built by the proven _stability_report()) and
result.restarts (the raw per-restart list) instead.
"""
from optimizer import load_vehicle_spec, calibrate_parallel, CalibrationConfig
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
          f"({total_steps} total steps). Starting 4 parallel restarts...", flush=True)

    config = CalibrationConfig(n_restarts=4, seed=42, max_nfev=50, initialization_strategy='perturbed')
    t0 = time.monotonic()
    result = calibrate_parallel(loaded, train_runs, config)
    elapsed = time.monotonic() - t0
    n_converged = sum(1 for r in result.restarts if r.success)
    print(f'Done in {elapsed:.1f}s, converged={result.converged}, cost={result.cost:.4f}, '
          f'n_converged={n_converged}/{len(result.restarts)}, train_rmse={result.train_rmse:.4f}')
    print(f'stability: {result.stability}')


if __name__ == "__main__":
    main()