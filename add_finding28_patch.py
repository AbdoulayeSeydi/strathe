# Safe, self-verifying patch: adds finding #28 to PROJECT_STATUS.md.
with open("PROJECT_STATUS.md", encoding="utf-8") as f:
    content = f.read()

anchor = "## Honest open items"

finding_28 = '''### 28. Ieff_roll/Ieff_pitch fixed at real, independently-sourced physical values — confirmed non-load-bearing two different ways, real Coriolis-coupling implication for r

Following finding #27's r regression, investigated whether Ieff_roll/Ieff_pitch drifting to non-physical values (9.1, 10.0 kg*m^2 in the most recent fits) was a real, load-bearing part of the fit or an artifact of weak identifiability (the same class of issue as Zw_dot in finding #18).

**Independent research, not assumed:** found a real, published, independently-sourced BlueROV2 "heavy" configuration rigid-body inertia model (a public Gazebo simulation repo): Ixx=0.26, Iyy=0.23, Izz=0.37 kg*m^2 — remarkably close to this project's OWN original literature-based initial guesses (0.28, 0.28, 0.32), confirming the original prior was well-grounded. The fitted values (9-10 kg*m^2) are 30-40x higher than both sources — far beyond a plausible rotational-added-mass correction.

**Confirmed two independent ways that these parameters are not load-bearing:**
1. Built `check_parameter_sensitivity.py` (item #11 from an early, never-completed diagnostic checklist) — a general per-parameter perturb-and-measure-ΔRMSE sweep. Result: Ieff_roll/Ieff_pitch ranked LOWEST of all 22 parameters (sensitivity 0.020 and 0.000 respectively) — the objective barely notices their exact value.
2. Directly re-fit with them held fixed at the real values (0.26, 0.23) via `fixed_params`, then again via the standard (non-fixed_params) calibration path with tightened `vehicle_spec.yaml` bounds — BOTH produced essentially the same fit quality (cost still dropped ~97%, final cost 0.35-0.36 either way) as letting them float freely.

**Fix applied:** `vehicle_spec.yaml`'s Ieff_roll/Ieff_pitch bounds tightened to [0.255,0.265]/[0.225,0.235] (narrow but deliberately NOT identical — confirmed directly that exactly-equal bounds cause a real x_scale=ub-lb=0 division-by-zero risk in the optimizer). Verified via the actual production calibration path (not just the fixed_params override) that the fit still converges well with these tight bounds.

**Open, testable implication for r:** Coriolis's yaw equation depends directly on (Ieff_pitch - Ieff_roll) (finding #16's derivation) — non-physical drift in these two parameters could have been quietly corrupting yaw's specific coupling term even while contributing little to the AGGREGATE RMSE this sensitivity analysis measures. Not yet confirmed - the next real HoloOcean refinement segment (once run) should be checked specifically for whether r's regression pattern (finding #27) changes now that this drift is structurally prevented.

'''

occurrences = content.count(anchor)
if occurrences != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence, found {occurrences}.")
else:
    new_content = content.replace(anchor, finding_28 + anchor, 1)
    with open("PROJECT_STATUS.md", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Finding #28 inserted successfully.")