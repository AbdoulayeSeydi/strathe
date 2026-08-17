"""Real recovery script - fix_warmstart_bounds.py's first (buggy) run
corrupted the warm-start file by writing 3 numpy float64 values back
without casting to plain Python floats first, making the file
unloadable via yaml.safe_load(). Uses yaml.unsafe_load() just for this
ONE-TIME recovery (safe here - this is our own, locally-generated data,
not untrusted external input), then rewrites EVERY value as a guaranteed
plain Python float, and re-applies the real bounds-clipping correctly
this time."""
import sys
import yaml

from optimizer import load_vehicle_spec, parameter_bounds, initial_parameters

warmstart_path = sys.argv[1]
spec_path = sys.argv[2] if len(sys.argv) > 2 else "vehicle_spec.yaml"

loaded = load_vehicle_spec(spec_path)
lb, ub = parameter_bounds(loaded)
param_names = initial_parameters(loaded).param_names

with open(warmstart_path, encoding="utf-8") as f:
    data = yaml.unsafe_load(f)  # tolerates the numpy-tagged values our
    # own earlier bug wrote - safe here, this is our own local data

fixed_count = 0
for i, name in enumerate(param_names):
    val = float(data["parameters"][name])  # force plain float regardless
    # of whether this specific value was corrupted or not
    clipped = float(min(max(val, float(lb[i])), float(ub[i])))
    if clipped != val:
        print(f"  {name}: {val} -> {clipped} (bounds [{float(lb[i])}, {float(ub[i])}])")
        fixed_count += 1
    data["parameters"][name] = clipped  # ALWAYS a plain float now,
    # whether clipped or not - guarantees a clean, safely-loadable file

with open(warmstart_path, "w", encoding="utf-8") as f:
    yaml.safe_dump(data, f, sort_keys=False)  # safe_dump refuses to
    # silently write an unsafe tag again - a real guardrail against
    # repeating this exact mistake

print(f"Recovered and re-clipped {fixed_count} parameter(s). "
      f"File is now guaranteed safe-loadable (verifying below)...")

# Real, direct verification this actually worked before declaring done
with open(warmstart_path, encoding="utf-8") as f:
    yaml.safe_load(f)
print("VERIFIED: file loads correctly via yaml.safe_load().")