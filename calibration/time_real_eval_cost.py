"""
Measures real per-evaluation cost on the ACTUAL training data — not
synthetic — before choosing max_nfev/n_restarts for a bigger, real
calibration run. The whole reason this exists: guessing a budget and
discovering it takes hours longer than expected is exactly the mistake
this project's own history already made once (PROJECT_STATUS.md's
documented multi-hour surprise runs). Measure first.

Loads the real training split (same discover -> split -> load path
build_calibrated_parameters.py uses) and times a handful of direct
joint_residuals() calls — the same function scipy calls internally,
timed the same way that isolated the real bottleneck during testing.
"""
import time

from data_split import discover_csv_files, split_by_date
from objective import DatasetConfig, load_run, joint_residuals
from optimizer import load_vehicle_spec, initial_parameters
from phase2_config import PHASE1_TRAIN_DATES, PHASE1_HELDOUT_DATES

VEHICLE_SPEC_PATH = "vehicle_spec.yaml"
DEFAULT_SAMPLING_RATE_HZ = 10.0
N_TIMED_CALLS = 10


def main(data_dir: str, max_train_runs: int = None):
    csv_paths, duplicates = discover_csv_files(data_dir)
    split = split_by_date(csv_paths, heldout_dates=PHASE1_HELDOUT_DATES)
    print(f"Train dates: {split.train_groups} ({split.n_train} files total)")

    train_files = split.train_files
    if max_train_runs is not None and max_train_runs < len(train_files):
        print(f"Truncating to {max_train_runs} of {len(train_files)} training files "
              f"for this timing measurement — matching whatever --max-train-runs you "
              f"intend to actually use for the real run, so this number is representative.")
        train_files = train_files[:max_train_runs]

    loaded_spec = load_vehicle_spec(VEHICLE_SPEC_PATH)
    dataset_config = DatasetConfig(sampling_rate_hz=DEFAULT_SAMPLING_RATE_HZ)

    print(f"Loading {len(train_files)} real training run(s)...")
    t_load = time.monotonic()
    train_runs = [load_run(fp, dataset_config) for fp in train_files]
    total_samples = sum(r.n_samples for r in train_runs)
    print(f"Loaded in {time.monotonic() - t_load:.2f}s — {total_samples} total samples "
          f"across {len(train_runs)} runs (avg {total_samples/len(train_runs):.0f} rows/run)")

    params = initial_parameters(loaded_spec)
    vec = params.to_vector()

    print(f"\nTiming {N_TIMED_CALLS} direct joint_residuals() calls on this training set "
          f"(the exact function scipy calls internally)...")
    times = []
    for i in range(N_TIMED_CALLS):
        t0 = time.monotonic()
        joint_residuals(vec, train_runs, loaded_spec.vehicle)
        elapsed = time.monotonic() - t0
        times.append(elapsed)
        print(f"  call {i+1}: {elapsed*1000:.1f}ms")

    avg = sum(times) / len(times)
    print(f"\nAverage: {avg*1000:.1f}ms/call ({avg:.4f}s/call)")
    print(f"\nProjected wall time using the CONFIRMED real multiplier (21.5x, measured "
          f"directly on this exact model/optimizer setup — not a range anymore):")
    for max_nfev in (15, 30, 60, 100):
        real_calls = max_nfev * 21.5
        seconds = real_calls * avg
        print(f"  max_nfev={max_nfev:>4}: ~{real_calls:.0f} real calls, "
              f"~{seconds/60:.1f} minutes ({seconds/3600:.2f} hours)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--max-train-runs", type=int, default=None,
                         help="Match whatever you plan to pass to build_calibrated_parameters.py")
    args = parser.parse_args()
    main(args.data_dir, max_train_runs=args.max_train_runs)