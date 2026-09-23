"""
B0.5 -- Spike test 1: full-override approach.

For a handful of made-up test (velocity, force) pairs, computes the
DESIRED acceleration via the existing, already-validated
compute_acceleration() (using the real calibrated_parameters_a1.yaml,
not made-up parameters -- per the plan's own "preserves exact parity
with everything already validated" framing), then attempts to make
MuJoCo realize EXACTLY that acceleration by:

  1. Reading MuJoCo's own qfrc_bias (Coriolis, centrifugal, gravity)
     and mass matrix M_mj at the test state.
  2. Injecting a wrench via xfrc_applied computed as
         injected_wrench = M_mj @ desired_v_dot + qfrc_bias
     which exactly cancels MuJoCo's own bias force and replaces it with
     the stand-in's calibrated dynamics -- regardless of what MuJoCo's
     OWN mass/inertia happen to be (even B0.3's placeholder values),
     since M_mj is read back from MuJoCo itself at runtime, not assumed.

SCOPE, DELIBERATELY LIMITED: every test case here uses IDENTITY
orientation (roll=pitch=yaw=0, matching B0.3's actual starting state).
MuJoCo's freejoint stores angular velocity in BODY frame but LINEAR
velocity in WORLD frame, while compute_acceleration()'s v is fully
body-frame (standard marine/Fossen convention) -- these coincide
exactly only at identity orientation. Testing at a general orientation
would silently conflate two separate unknowns (does wrench injection
work? are the frames handled correctly?) in one test. The general
frame-conversion question is explicitly B1.2/B1.4's job, not this
spike test's.

*** VERIFIED vs UNVERIFIED, precisely:
- compute_acceleration() itself, loaded with the REAL calibrated
  parameters: run and confirmed working (see conversation).
- The wrench-injection formula (injected_wrench = M_mj @ desired_v_dot
  + qfrc_bias): verified to machine precision (~3.5e-17) against
  synthetic mass-matrix/bias-force data mimicking MuJoCo's shapes.
- mj_fullM's call signature: confirmed against MuJoCo's own changelog
  (not assumed) -- mjData.qM was removed in 3.11.0 in favor of the
  CSR-format mjData.M, and mj_fullM(m, dst, d->qM) became
  mj_fullM(m, d, dst). This project is on 3.12.0, past that break; a
  first run against the old signature failed loudly with an
  AttributeError, exactly the good failure mode.
- Every remaining MuJoCo API call below (mj_forward, data.qfrc_bias,
  data.xfrc_applied indexing, data.qacc) is still UNTESTED -- no
  mujoco in this sandbox. Correct per MuJoCo's documented API as I
  understand it, not run.
"""
import numpy as np
import mujoco

from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec
from dynamics_model import compute_acceleration

MJCF_PATH = "b0_single_body_placeholder.xml"

# A handful of made-up (v, tau) test cases. v=zeros isolates the
# external-force-plus-gravity/buoyancy case; the nonzero-v cases also
# exercise damping_force and coriolis_force, which are functions of v.
TEST_CASES = [
    ("at rest, pure surge force",
     np.zeros(6), np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ("at rest, combined force+torque",
     np.zeros(6), np.array([2.0, -1.0, 0.5, 0.0, 0.3, 0.0])),
    ("nonzero velocity, zero external force (damping+Coriolis only)",
     np.array([0.5, 0.2, -0.1, 0.1, 0.05, 0.02]), np.zeros(6)),
    ("nonzero velocity AND nonzero force (fully general case)",
     np.array([0.5, 0.2, -0.1, 0.1, 0.05, 0.02]),
     np.array([3.0, 0.0, -2.0, 0.0, 0.2, 0.0])),
]


def run_spike_test(tolerance: float = 1e-6):
    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters("calibrated_parameters_a1.yaml")

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    body_id = 1  # index 0 is MuJoCo's implicit world body (confirmed in B0.3)

    assert model.nv == 6, f"expected 6 velocity DOF for a single free body, got {model.nv}"

    all_passed = True
    for name, v, tau in TEST_CASES:
        print(f"\n--- {name} ---")
        print(f"v={v}, tau={tau}")

        # Desired acceleration from the already-validated stand-in physics.
        desired_v_dot, damping_force, g_eta = compute_acceleration(
            params, vehicle, v, tau, roll=0.0, pitch=0.0)
        print(f"desired v_dot (from compute_acceleration): {desired_v_dot}")

        # Set MuJoCo to the identity-orientation test state.
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0  # quaternion [w,x,y,z] = [1,0,0,0] -> identity orientation
        data.qvel[:] = v    # valid 1:1 at identity orientation only (see module docstring)
        data.xfrc_applied[:, :] = 0.0

        # First forward pass: get MuJoCo's own bias force and mass matrix
        # at this state, BEFORE any injected wrench (qfrc_bias does not
        # depend on xfrc_applied, so this is well-defined).
        mujoco.mj_forward(model, data)
        qfrc_bias = data.qfrc_bias.copy()
        M_mj = np.zeros((model.nv, model.nv))
        # mj_fullM's signature changed in MuJoCo 3.10.0: mjData.qM (legacy
        # sparse format) was removed in 3.11.0 in favor of the CSR-format
        # mjData.M, and mj_fullM now takes the whole `data` object rather
        # than a bare sparse array: mj_fullM(m, dst, d->qM) -> mj_fullM(m,
        # d, dst). Confirmed against MuJoCo's own changelog, not assumed --
        # this project is on 3.12.0, past that break.
        mujoco.mj_fullM(model, data, M_mj)
        print(f"MuJoCo's own qfrc_bias: {qfrc_bias}")

        # Compute and inject the cancelling + replacing wrench.
        injected_wrench = M_mj @ desired_v_dot + qfrc_bias
        data.xfrc_applied[body_id, :] = injected_wrench

        # Second forward pass: recompute qacc now that xfrc_applied is set.
        mujoco.mj_forward(model, data)
        realized_qacc = data.qacc.copy()

        max_abs_diff = np.max(np.abs(realized_qacc - desired_v_dot))
        print(f"realized qacc:                          {realized_qacc}")
        print(f"max abs difference from desired v_dot:  {max_abs_diff:.2e}")

        passed = max_abs_diff < tolerance
        all_passed = all_passed and passed
        print("PASS" if passed else "FAIL", f"(tolerance={tolerance:.0e})")

    print(f"\n{'='*70}")
    if all_passed:
        print(f"B0.5 SPIKE TEST PASSED: full-override approach realizes the "
              f"stand-in's calibrated acceleration exactly (within {tolerance:.0e}) "
              f"across all {len(TEST_CASES)} test cases, at identity orientation.")
    else:
        print("B0.5 SPIKE TEST FAILED: see individual cases above.")
    return all_passed


if __name__ == "__main__":
    run_spike_test()