"""
Safe, self-verifying patch: widens the bounds that check_p_bounds.py
found genuinely pinned (Ieff_roll/pitch/yaw upper bound, Kp lower bound)
plus one secondary near-pinned candidate (Nrr lower bound) - all
motivated by real, checked evidence, not a guess. Refuses if any anchor
isn't found exactly once.
"""
with open("vehicle_spec.yaml", encoding="utf-8") as f:
    content = f.read()

replacements = [
    (
        '    Ieff_roll:  {unit: "kg*m^2", initial_guess: 0.28, bounds: [0.05, 5.0],\n'
        '                 bounds_rationale: "deliberately wider than the old Ixx+Kp_dot combined bound (2.4) — see note above",',
        '    Ieff_roll:  {unit: "kg*m^2", initial_guess: 0.28, bounds: [0.05, 10.0],\n'
        '                 bounds_rationale: "widened from 5.0 to 10.0 — CONFIRMED pinned exactly at the old '
        'upper bound (check_p_bounds.py, 0.0% from bound) in a real HoloOcean-refined calibration where roll '
        'rate remained stuck (see PROJECT_STATUS.md), suggesting the model genuinely wants MORE roll inertia "'
        '"than previously allowed, not that roll is unfittable",',
    ),
    (
        '    Ieff_pitch: {unit: "kg*m^2", initial_guess: 0.28, bounds: [0.05, 5.0],\n'
        '                 bounds_rationale: "deliberately wider than the old Iyy+Mq_dot combined bound (2.4) — see note above",',
        '    Ieff_pitch: {unit: "kg*m^2", initial_guess: 0.28, bounds: [0.05, 10.0],\n'
        '                 bounds_rationale: "widened from 5.0 to 10.0 for consistency — ALSO confirmed pinned '
        'at the same old upper bound (check_p_bounds.py), even though pitch rate itself fit well; widening this '
        'range generally rather than only for roll",',
    ),
    (
        '    Ieff_yaw:   {unit: "kg*m^2", initial_guess: 0.32, bounds: [0.05, 5.0],',
        '    Ieff_yaw:   {unit: "kg*m^2", initial_guess: 0.32, bounds: [0.05, 10.0],',
    ),
    (
        '    Kp:  {unit: "N*m/(rad/s)", initial_guess: -0.3, bounds: [-5, -0.1],\n'
        '          bounds_rationale: "wide net — low-confidence axis, sparse published data; upper bound excludes near-zero damping, see comment above",',
        '    Kp:  {unit: "N*m/(rad/s)", initial_guess: -0.3, bounds: [-10, -0.1],\n'
        '          bounds_rationale: "widened lower bound from -5 to -10 — CONFIRMED pinned exactly at -5 '
        '(check_p_bounds.py, 0.1% from bound) in the same real calibration where roll rate remained stuck; '
        'the model wants STRONGER roll damping than previously allowed. Upper bound still excludes near-zero '
        'damping, see comment above",',
    ),
    (
        '    Nrr: {unit: "N*m/(rad/s)^2", initial_guess: -2.0, bounds: [-20, -0.4],',
        '    Nrr: {unit: "N*m/(rad/s)^2", initial_guess: -2.0, bounds: [-30, -0.4],',
    ),
]

missing = []
for old, new in replacements:
    if content.count(old) != 1:
        missing.append(old[:60] + "...")

if missing:
    print("REFUSING TO PATCH — the following expected text was not found exactly once:")
    for m in missing:
        print(f"  {m}")
    print("Do NOT proceed.")
else:
    for old, new in replacements:
        content = content.replace(old, new, 1)
    with open("vehicle_spec.yaml", "w", encoding="utf-8") as f:
        f.write(content)
    print("All 5 bound widenings applied successfully.")