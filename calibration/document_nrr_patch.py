# Safe, self-verifying patch: pure documentation update for Nrr, same
# treatment as Zw_dot/Zww. No bounds/code change - Nrr already works
# correctly and isn't even bound-pinned right now.

with open("vehicle_spec.yaml", encoding="utf-8") as f:
    content = f.read()

old_nrr = '''    Nrr: {unit: "N*m/(rad/s)^2", initial_guess: -2.0, bounds: [-30, -0.4],
          bounds_rationale: "wide net — low-confidence axis; upper bound excludes near-zero damping, see linear_damping comment", source: "literature", confidence: literature}'''

new_nrr = '''    Nrr: {unit: "N*m/(rad/s)^2", initial_guess: -2.0, bounds: [-30, -0.4],
          bounds_rationale: "wide net — low-confidence axis; upper bound excludes near-zero damping, see linear_damping comment. UPDATE: post-Coriolis real fit shows Nr/Nrr correlation at -0.861 vs. the fixed kinematic baseline of 0.864 (see identifiability_note below) - gap 0.003, the TIGHTEST match of any damping pair (was 0.006 pre-Coriolis, already small). Not bound-pinned in the current real result either. Clean, confirmed data-coverage case - no missing-physics symptom here.",
          source: "literature", confidence: literature}'''

count = content.count(old_nrr)
if count != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence, found {count}.")
else:
    content = content.replace(old_nrr, new_nrr, 1)
    with open("vehicle_spec.yaml", "w", encoding="utf-8") as f:
        f.write(content)
    print("Nrr documentation update applied successfully.")
