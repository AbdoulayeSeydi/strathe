# Safe, self-verifying patch: fixes Ieff_roll/Ieff_pitch near their REAL,
# independently-sourced physical values (0.26, 0.23 kg*m^2 - a public
# BlueROV2 "heavy" config Gazebo model) instead of letting them float up
# to the non-physical 9-10 kg*m^2 recent fits converged to.
#
# Bounds are TIGHT but NOT exactly equal (avoids a real x_scale=ub-lb=0
# division-by-zero risk, confirmed directly before this patch was
# written) - genuinely pins the value with no meaningful room to drift,
# while staying numerically safe.

with open("vehicle_spec.yaml", encoding="utf-8") as f:
    content = f.read()

old_roll = '''    Ieff_roll:  {unit: "kg*m^2", initial_guess: 0.28, bounds: [0.05, 10.0],
                 bounds_rationale: "widened from 5.0 to 10.0 — CONFIRMED pinned exactly at the old upper bound (check_p_bounds.py, 0.0% from bound) in a real HoloOcean-refined calibration where roll rate remained stuck (see PROJECT_STATUS.md), suggesting the model genuinely wants MORE roll inertia than previously allowed, not that roll is unfittable",
                 source: "sum of box-approximation inertia + literature added-mass estimate", confidence: estimated}'''

new_roll = '''    Ieff_roll:  {unit: "kg*m^2", initial_guess: 0.26, bounds: [0.255, 0.265],
                 bounds_rationale: "FIXED near the real, independently-sourced BlueROV2 heavy-config value (0.26 kg*m^2, clydemcqueen/bluerov2_gz public Gazebo model) after two convergent findings (PROJECT_STATUS.md): (1) a sensitivity sweep showed near-ZERO RMSE sensitivity to this parameter's exact value, (2) a real re-fit with it fixed at this value lost none of the fit quality (cost still dropped ~97% with the other 20 parameters doing the work). Previous wide bounds let it drift to a non-physical 9-10 kg*m^2 with no real fit-quality justification - likely compensating for something else rather than reflecting genuine inertia. Bounds kept narrow-but-nonzero (not exactly equal) to avoid a real x_scale=ub-lb=0 division risk, confirmed directly before this change.",
                 source: "independently-sourced real BlueROV2 heavy-config value (clydemcqueen/bluerov2_gz)", confidence: literature}'''

old_pitch = '''    Ieff_pitch: {unit: "kg*m^2", initial_guess: 0.28, bounds: [0.05, 10.0],
                 bounds_rationale: "widened from 5.0 to 10.0 for consistency — ALSO confirmed pinned at the same old upper bound (check_p_bounds.py), even though pitch rate itself fit well; widening this range generally rather than only for roll",
                 source: "sum of box-approximation inertia + literature added-mass estimate", confidence: estimated}'''

new_pitch = '''    Ieff_pitch: {unit: "kg*m^2", initial_guess: 0.23, bounds: [0.225, 0.235],
                 bounds_rationale: "FIXED near the real, independently-sourced BlueROV2 heavy-config value (0.23 kg*m^2, clydemcqueen/bluerov2_gz public Gazebo model) - same reasoning as Ieff_roll above (near-zero RMSE sensitivity confirmed two ways; the earlier 9-10 kg*m^2 drift was not fit-quality-justified).",
                 source: "independently-sourced real BlueROV2 heavy-config value (clydemcqueen/bluerov2_gz)", confidence: literature}'''

missing = []
if content.count(old_roll) != 1:
    missing.append("Ieff_roll block")
if content.count(old_pitch) != 1:
    missing.append("Ieff_pitch block")

if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}.")
else:
    content = content.replace(old_roll, new_roll, 1)
    content = content.replace(old_pitch, new_pitch, 1)
    with open("vehicle_spec.yaml", "w", encoding="utf-8") as f:
        f.write(content)
    print("Inertia fix applied successfully - Ieff_roll/Ieff_pitch now fixed near real physical values.")