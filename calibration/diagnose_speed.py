"""One-off timing diagnostic — not part of the permanent pipeline."""
import glob
import time

from data_split import split_by_date
from objective import DatasetConfig, load_run, joint_residuals
from optimizer import load_vehicle_spec, initial_parameters

csv_paths = sorted(glob.glob("../data/raw/Bluerov2_trajectory_dataset/**/*.csv", recursive=True))
split = split_by_date(csv_paths, heldout_dates=["16-02-2024", "19-02-2024"])

loaded = load_vehicle_spec("vehicle_spec.yaml")
dataset_config = DatasetConfig(sampling_rate_hz=50.0)

print(f"Loading 2 training runs to time against (out of {len(split.train_files)} total)...")
sample_runs = [load_run(fp, dataset_config) for fp in split.train_files[:2]]
n_samples = sum(r.n_samples for r in sample_runs)
print(f"Sample: {len(sample_runs)} runs, {n_samples} total timesteps")

params = initial_parameters(loaded)
vec = params.to_vector()
vehicle = loaded.vehicle

print("Timing a single joint_residuals() call...")
t0 = time.perf_counter()
joint_residuals(vec, sample_runs, vehicle)
elapsed = time.perf_counter() - t0
print(f"Single call: {elapsed:.3f}s for {n_samples} timesteps ({elapsed/n_samples*1e6:.2f} microseconds/timestep)")

all_train_samples_estimate = n_samples / len(sample_runs) * 26
per_call_full = elapsed * (all_train_samples_estimate / n_samples)
n_params = len(vec)
jacobian_evals = n_params + 1
print(f"\nEstimated per-call time on full 26-run training set: {per_call_full:.2f}s")
print(f"Estimated per-Jacobian time (~{jacobian_evals} evals): {per_call_full * jacobian_evals:.1f}s")
print(f"At ~20 optimizer iterations: ~{per_call_full * jacobian_evals * 20 / 60:.1f} minutes PER RESTART")
