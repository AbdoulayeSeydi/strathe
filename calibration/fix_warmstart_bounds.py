"""Clips an old calibrated_parameters.yaml's values to the CURRENT
vehicle_spec.yaml bounds - fixes the exact class of issue documented in
PROJECT_STATUS.md finding #30 (bounds tightened after the file was
originally saved), for warm-start files specifically (calibrated_
parameters.yaml's "parameters" key, not checkpoint.yaml's different
format)."""
import sys
import yaml

from optimizer import load_vehicle_spec, parameter_bounds, initial_parameters

warmstart_path = sys.argv[1]
spec_path = sys.argv[2] if len(sys.argv) > 2 else "vehicle_spec.yaml"

loaded = load_vehicle_spec(spec_path)
lb, ub = parameter_bounds(loaded)
param_names = initial_parameters(loaded).param_names

with open(warmstart_path, encoding="utf-8") as f:
    data = yaml.safe_load(f)

fixed_count = 0
for i, name in enumerate(param_names):
    val = data["parameters"][name]
    if val < lb[i] or val > ub[i]:
        clipped = min(max(val, lb[i]), ub[i])
        print(f"  {name}: {val} -> {clipped} (bounds [{lb[i]}, {ub[i]}])")
        data["parameters"][name] = clipped
        fixed_count += 1

with open(warmstart_path, "w", encoding="utf-8") as f:
    yaml.dump(data, f, sort_keys=False)

print(f"Fixed {fixed_count} out-of-bounds parameter(s) in {warmstart_path}")