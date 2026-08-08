# Safe, self-verifying patch: tightens net_buoyancy_z's bounds around
# vehicle_spec.yaml's own original neutral-trim prior (0.0N), following
# the SAME confirmed pattern as Ieff_roll/Ieff_pitch: a real re-fit with
# it fixed there lost none of the fit quality (cost still dropped ~97%
# with the other 21 parameters doing the work) - the wide [-10,10] bound
# let it drift to the extreme +10N edge with no fit-quality justification.

with open("vehicle_spec.yaml", encoding="utf-8") as f:
    content = f.read()

old = '''    net_buoyancy_z: {unit: N, initial_guess: 0.0, bounds: [-10, 10],
                      bounds_rationale: "±10N is roughly ±1kg of trim error — generous for a hand-trimmed vehicle",
                      source: "assumed near-neutral trim", confidence: assumed}'''

new = '''    net_buoyancy_z: {unit: N, initial_guess: 0.0, bounds: [-1.0, 1.0],
                      bounds_rationale: "TIGHTENED from [-10,10] - CONFIRMED (PROJECT_STATUS.md): real sensitivity sweep showed near-ZERO RMSE sensitivity to this parameter (tied for lowest of all 22 parameters), AND a real re-fit with it fixed at 0.0 (this file's own original neutral-trim prior) lost none of the fit quality (cost still dropped ~97% with the other 21 parameters doing the work). The old wide bound let it drift to the extreme +10N edge (~1kg of trim) with no fit-quality justification - the same non-load-bearing-drift pattern already confirmed and fixed for Ieff_roll/Ieff_pitch. New bounds still allow +-1N (~0.1kg) of real trim variation, comfortably wider than a well-trimmed vehicle should need.",
                      source: "assumed near-neutral trim, confirmed non-load-bearing", confidence: assumed}'''

count = content.count(old)
if count != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence, found {count}.")
else:
    content = content.replace(old, new, 1)
    with open("vehicle_spec.yaml", "w", encoding="utf-8") as f:
        f.write(content)
    print("net_buoyancy_z bounds tightened successfully.")
