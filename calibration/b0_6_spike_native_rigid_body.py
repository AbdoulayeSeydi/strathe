"""
B0.6 -- Spike test 2: native rigid-body approach, for comparison.

Injects ONLY the force-like hydrodynamic terms -- damping and buoyancy
-- through the same xfrc_applied wrench mechanism as B0.5, and lets
MuJoCo's own engine handle gravity and its own rigid-body Coriolis
natively (using MuJoCo's placeholder mass matrix, not the stand-in's
calibrated one).

Added mass is intentionally OMITTED here, not approximated: it modifies
the effective mass matrix rather than constituting an externally
injectable wrench, and a lagged-force approximation would introduce a
second, unvalidated numerical approximation into what's meant to be a
clean, single-variable comparison (native vs. calibrated Coriolis).
Calibrated cross-coupling (Xq_dot, Zq_dot) is omitted for the same
reason -- both live inside the calibrated M, not in compute_acceleration()'s
force-like return values.

GRAVITY/BUOYANCY FIX (the actual point of this file, beyond B0.5):
the stand-in's g_eta is a small RESIDUAL after weight and buoyancy are
already assumed to nearly cancel (standard Fossen convention for a
submerged vehicle) -- it is NOT a standalone buoyancy force. MuJoCo's
native gravity applies the FULL, uncancelled weight (confirmed in B0.5:
qfrc_bias's z-component was exactly 109.872 N = 11.2 kg * 9.81 m/s^2).
Naively injecting g_eta on top of native gravity would double-count
weight and barely apply buoyancy -- the vehicle would sink like a rock
with a tiny nudge, not float. Fixed here by isolating ONLY the
gravity-dependent part of MuJoCo's qfrc_bias (evaluated at v=0, where
the velocity-dependent Coriolis/centrifugal terms vanish exactly,
leaving pure gravity) and cancelling that specific part, while leaving
qfrc_bias's Coriolis contribution (evaluated at the real test velocity)
completely untouched and native.

Reuses B0.5's exact TEST_CASES for a direct, apples-to-apples
comparison against the stand-in's fully-calibrated v_dot.

Built-in decomposition (NOT a pass/fail check -- corrected after an
initial wrong assumption; see conversation): at v=0, Coriolis is
genuinely zero for both native and calibrated physics, so the at-rest
cases isolate the mismatch from added mass and cross-coupling (Xq_dot,
Zq_dot) ALONE -- both are mass-matrix-structural effects, not
velocity-dependent ones, so they do NOT vanish at rest the way Coriolis
does. The nonzero-velocity cases show that SAME structural mismatch
combined with the Coriolis-source difference. Comparing the two groups
is itself informative for B0.7, even without a clean per-case
subtraction (the four test cases use different tau/v combinations, so
an exact "Coriolis-only" isolation isn't available from this test's
structure alone).

*** VERIFIED vs UNVERIFIED: the wrench formula (including the gravity/
Coriolis separation) is verified to machine precision against synthetic
data mimicking MuJoCo's shapes. The actual MuJoCo API calls (mj_forward,
data.qfrc_bias, data.xfrc_applied, data.qacc) are unrun -- no mujoco in
this sandbox.
Note: unlike B0.5, this script never explicitly reads MuJoCo's mass
matrix (no mj_fullM) -- known forces are injected and MuJoCo's own
internal dynamics respond naturally, since B0.6 doesn't target a
specific acceleration the way B0.5's full override does.
"""
import numpy as np
import mujoco

from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec
from dynamics_model import compute_acceleration
from b0_5_spike_full_override import MJCF_PATH, TEST_CASES


def run_spike_test():
    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters("calibrated_parameters_a1.yaml")

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    body_id = 1  # index 0 is MuJoCo's implicit world body (confirmed in B0.3)

    assert model.nv == 6, f"expected 6 velocity DOF for a single free body, got {model.nv}"

    results = []
    for name, v, tau in TEST_CASES:
        print(f"\n--- {name} ---")
        print(f"v={v}, tau={tau}")

        # Stand-in's fully-calibrated result (the B0.5 comparison target).
        # v_dot itself is NOT used to construct the injected wrench here --
        # only damping_force and g_eta, which don't depend on M at all.
        calibrated_v_dot, damping_force, g_eta = compute_acceleration(
            params, vehicle, v, tau, roll=0.0, pitch=0.0)
        print(f"calibrated v_dot (B0.5 reference): {calibrated_v_dot}")
        print(f"damping_force (injected):          {damping_force}")
        print(f"g_eta (injected, subtracted):       {g_eta}")

        # Identity-orientation test state, same scope limit as B0.5.
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0  # quaternion [w,x,y,z] = [1,0,0,0]
        data.xfrc_applied[:, :] = 0.0

        # Pass 1: full native bias (gravity + native Coriolis, mixed) at
        # the REAL test velocity.
        data.qvel[:] = v
        mujoco.mj_forward(model, data)
        qfrc_bias_full = data.qfrc_bias.copy()

        # Pass 2: bias at v=0, same orientation -- isolates gravity only,
        # since native Coriolis/centrifugal vanish exactly at zero velocity.
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        gravity_only_bias = data.qfrc_bias.copy()
        print(f"MuJoCo native gravity-only bias:    {gravity_only_bias}")
        print(f"MuJoCo native full bias (at test v): {qfrc_bias_full}")

        # Cancel ONLY gravity; native Coriolis (qfrc_bias_full - gravity_only_bias)
        # is left untouched by never being added to the injected wrench.
        injected_wrench = tau + gravity_only_bias + damping_force - g_eta
        data.xfrc_applied[body_id, :] = injected_wrench

        # Pass 3: restore the real test velocity, recompute with the
        # injected wrench now in effect.
        data.qvel[:] = v
        mujoco.mj_forward(model, data)
        native_qacc = data.qacc.copy()

        diff = native_qacc - calibrated_v_dot
        max_abs_diff = np.max(np.abs(diff))
        print(f"B0.6 native-Coriolis realized qacc: {native_qacc}")
        print(f"Difference from calibrated v_dot:   {diff}")
        print(f"Max abs difference:                 {max_abs_diff:.6e}")

        is_at_rest = np.allclose(v, 0.0)
        if is_at_rest:
            print(f"(at-rest case: isolates the mismatch from added mass + "
                  f"cross-coupling alone -- Coriolis is genuinely zero here for "
                  f"BOTH native and calibrated, but added mass/cross-coupling are "
                  f"mass-matrix-structural, not velocity-dependent, so they do NOT "
                  f"vanish at rest. This number is real data for B0.7, not a bug.)")
        else:
            print("(nonzero-velocity case: mismatch here combines the SAME "
                  "mass-matrix-structural effect seen at rest PLUS the "
                  "Coriolis-source difference -- both contribute together.)")

        results.append({"name": name, "v": v, "tau": tau,
                         "calibrated_v_dot": calibrated_v_dot,
                         "native_qacc": native_qacc,
                         "max_abs_diff": max_abs_diff,
                         "is_at_rest": is_at_rest})

    print(f"\n{'='*70}")
    print("B0.6 SUMMARY -- native vs. calibrated mismatch (max abs diff):")
    for r in results:
        tag = "(mass-matrix structure only: added mass + cross-coupling)" \
            if r["is_at_rest"] else "(mass-matrix structure + Coriolis-source, combined)"
        print(f"  {r['name']:<55s} max_abs_diff = {r['max_abs_diff']:.4e}  {tag}")
    return results


if __name__ == "__main__":
    run_spike_test()