# Safe, self-verifying patch: pure documentation update to vehicle_spec.yaml.
# No bounds, values, or code behavior change - both parameters already
# work correctly. Fixes stale comments that predate real investigations
# already completed this session (finding #18 for Zw_dot; tonight's
# real diagnostic run for Zww).

with open("vehicle_spec.yaml", encoding="utf-8") as f:
    content = f.read()

old_zwdot = '''    Zw_dot: {unit: kg, initial_guess: 14.6, bounds: [0, 40],
             bounds_rationale: "~3x nominal literature estimate", source: "literature", confidence: literature}'''

new_zwdot = '''    Zw_dot: {unit: kg, initial_guess: 14.6, bounds: [0, 40],
             bounds_rationale: "~3x nominal literature estimate. UPDATE (PROJECT_STATUS.md finding #18): pinning at the lower bound in real fits is CONFIRMED a genuine practical (data-coverage) identifiability gap, not a modeling bug - two independent methods (Jacobian-based sensitivity analysis, and an SVD-based multi-parameter near-null-space check) both concluded this dataset's real maneuvers do not sufficiently excite heave acceleration to constrain this parameter. Bounds/initial_guess deliberately left unchanged - there is no fit-quality or physical basis to move them, unlike Ieff_roll/Ieff_pitch's later, different situation.",
             source: "literature", confidence: literature}'''

old_zww = '''    Zww: {unit: "N/(m/s)^2", initial_guess: -50.0, bounds: [-150, -3.0],
          bounds_rationale: "~3x nominal literature estimate; upper bound excludes near-zero damping, see linear_damping comment", source: "literature", confidence: literature}'''

new_zww = '''    Zww: {unit: "N/(m/s)^2", initial_guess: -50.0, bounds: [-150, -3.0],
          bounds_rationale: "~3x nominal literature estimate; upper bound excludes near-zero damping, see linear_damping comment. UPDATE: post-Coriolis real fit shows Zw/Zww correlation at -0.826 (was -0.88 pre-Coriolis) vs. the fixed kinematic baseline of 0.918 (see identifiability_note below) - the gap from baseline WIDENED (0.038 -> 0.092) after adding Coriolis, the opposite of what would be expected if Coriolis were resolving a missing-physics-driven inflation the way it was hoped to for Mq/Mqq. This pair's correlation looks like a genuine data-coverage issue, not a missing-physics symptom.",
          source: "literature", confidence: literature}'''

old_note_tail = '''    Zw/Zww and Kp/Kpp show milder versions of the same disagreement (2-rank
    shifts each) — worth watching too, not just Mq/Mqq.'''

new_note_tail = '''    Zw/Zww and Kp/Kpp show milder versions of the same disagreement (2-rank
    shifts each) — worth watching too, not just Mq/Mqq.
    UPDATE, post-Coriolis (real diagnostic run, PROJECT_STATUS.md): Zw/Zww's
    fitted correlation is now -0.826 (was -0.88), vs. its fixed kinematic
    baseline of 0.918. The gap from baseline WIDENED (0.038 -> 0.092), the
    OPPOSITE of what Coriolis resolving a missing-physics inflation would
    predict — Zw/Zww looks like genuine data-coverage, not a missing-physics
    symptom, unlike the open Mq/Mqq question above. Not yet re-checked for
    Mq/Mqq itself with this same before/after comparison.'''

replacements = [
    ("Zw_dot entry", old_zwdot, new_zwdot),
    ("Zww entry", old_zww, new_zww),
    ("identifiability_note tail", old_note_tail, new_note_tail),
]

missing = [label for label, old, new in replacements if content.count(old) != 1]
if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}.")
else:
    for label, old, new in replacements:
        content = content.replace(old, new, 1)
    with open("vehicle_spec.yaml", "w", encoding="utf-8") as f:
        f.write(content)
    print("Both documentation updates applied successfully.")
