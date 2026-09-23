"""
Cleanly isolates TWO variables that my first attempt at this test
conflated: whether the angular qacc mismatch found in run4 (step 1261)
is caused by (a) nonzero orientation itself, or (b) something about the
MjData object having already been through 25220 prior substeps of
accumulated simulation (warm-start state, solver history, etc.) rather
than a fresh object -- my earlier identity-orientation test changed
BOTH of these at once and so could not distinguish between them.

Three conditions, all using a FRESH MjData object and run4's actual
observed v/tau at step 1261, varying ONLY orientation:
  1. Identity orientation (0, 0) -- already run once, machine precision.
  2. run4's ACTUAL orientation at step 1261 (0.36 deg, -0.74 deg) --
     the genuinely new case this script adds.
  3. A deliberately LARGER rotation (10 deg, 10 deg) -- to see whether
     any mismatch, if present, scales with rotation angle.

All three use a FRESH data object -- so if condition 2 or 3 shows a
mismatch, that DIRECTLY implicates orientation itself (with freshness
held constant), not accumulated simulation history. If condition 2
matches condition 1 (machine precision), but the REAL diagnostic
(inspect_b1_5_internal_state.py, same orientation, same v/tau, but a
data object with 25220 prior substeps of history) does NOT match, that
would instead implicate accumulated internal MuJoCo state as the real
variable -- a different, separate hypothesis to test next.

*** UNVERIFIED -- no mujoco in this sandbox.
"""
import numpy as np
import mujoco

from dynamics_model import compute_acceleration
from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec

MJCF_PATH = "b0_single_body_placeholder.xml"

# run4's actual observed state and tau at step 1261 (see conversation)
V_REAL = np.array([0.893, 0.058, -0.427, -0.041, -0.934, -0.043])
TAU_REAL = np.array([114.23, -1.04, -1.10, 0.58, -8.48, 1.63])

CONDITIONS = [
    ("Identity orientation (0, 0) -- baseline, already confirmed", 0.0, 0.0),
    ("run4's ACTUAL orientation at step 1261 (0.36 deg, -0.74 deg)",
     np.radians(0.36), np.radians(-0.74)),
    ("Deliberately larger rotation (10 deg, 10 deg) -- scaling check",
     np.radians(10.0), np.radians(10.0)),
]


def run_single_condition(params, vehicle, label: str, roll: float, pitch: float):
    print(f"\n--- {label} ---")

    # FRESH model/data every time -- isolates orientation as the only variable
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    from scipy.spatial.transform import Rotation
    quat_xyzw = Rotation.from_euler("xyz", [roll, pitch, 0.0]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    R = Rotation.from_euler("xyz", [roll, pitch, 0.0]).as_matrix()

    v_dot, _, _ = compute_acceleration(params, vehicle, V_REAL, TAU_REAL, roll, pitch)

    data.qpos[:3] = 0.0
    data.qpos[3:7] = quat_wxyz
    data.qvel[:3] = R @ V_REAL[:3]  # body -> world, matching mujoco_backend.py's own fix
    data.qvel[3:] = V_REAL[3:]      # angular already body-frame, no rotation applied
    data.xfrc_applied[:, :] = 0.0

    mujoco.mj_forward(model, data)
    qfrc_bias = data.qfrc_bias.copy()
    M_mj = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M_mj)

    desired_qacc = np.concatenate([R @ v_dot[:3], v_dot[3:]])
    injected_wrench = M_mj @ desired_qacc + qfrc_bias
    data.xfrc_applied[1, :] = injected_wrench

    # DIRECT CHECK: does qfrc_bias itself change between the two
    # mj_forward() calls, even though qpos/qvel are unchanged and only
    # xfrc_applied was set? This should be impossible if qfrc_bias is
    # purely a function of qpos/qvel -- checking directly rather than
    # assuming.
    mujoco.mj_forward(model, data)
    qfrc_bias_second_call = data.qfrc_bias.copy()
    qacc_before_step = data.qacc.copy()
    qfrc_bias_diff = qfrc_bias_second_call - qfrc_bias
    print(f"  qfrc_bias (1st call, xfrc_applied=0):        {qfrc_bias}")
    print(f"  qfrc_bias (2nd call, xfrc_applied=injected): {qfrc_bias_second_call}")
    print(f"  qfrc_bias DIFFERENCE between the two calls:  {qfrc_bias_diff}")
    print(f"  max abs qfrc_bias difference:                {np.max(np.abs(qfrc_bias_diff)):.6e}")
    diff_before_step = qacc_before_step - desired_qacc
    print(f"  qacc IMMEDIATELY after mj_forward (before mj_step): {qacc_before_step}")
    print(f"  diff before mj_step (angular):                      {diff_before_step[3:]}")
    print(f"  max abs diff BEFORE mj_step (angular):               {np.max(np.abs(diff_before_step[3:])):.6e}")

    mujoco.mj_step(model, data)
    realized_qacc = data.qacc.copy()

    diff = realized_qacc - desired_qacc
    print(f"  desired qacc (mixed frame): {desired_qacc}")
    print(f"  realized qacc (AFTER mj_step): {realized_qacc}")
    print(f"  diff:                       {diff}")
    print(f"  max abs diff, LINEAR (0-2): {np.max(np.abs(diff[:3])):.6e}")
    print(f"  max abs diff, ANGULAR (3-5):{np.max(np.abs(diff[3:])):.6e}")
    return np.max(np.abs(diff))


def run_check():
    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters("calibrated_parameters_a1.yaml")

    results = []
    for label, roll, pitch in CONDITIONS:
        max_diff = run_single_condition(params, vehicle, label, roll, pitch)
        results.append((label, roll, pitch, max_diff))

    print(f"\n{'='*70}")
    print("SUMMARY")
    for label, roll, pitch, max_diff in results:
        print(f"  {label}: max_diff={max_diff:.4e}")

    identity_diff = results[0][3]
    actual_diff = results[1][3]
    larger_diff = results[2][3]

    if identity_diff < 1e-6 and actual_diff < 1e-6 and larger_diff < 1e-6:
        print("\nAll conditions match to machine precision, INCLUDING run4's actual "
              "orientation with a FRESH data object. This means orientation itself, "
              "isolated from accumulated simulation history, is NOT the cause. The "
              "real diagnostic's mismatch (same orientation, but 25220 prior substeps "
              "of history) must instead implicate something about ACCUMULATED "
              "internal MuJoCo state -- a different hypothesis to test next.")
    elif identity_diff < 1e-6 and (actual_diff >= 1e-6 or larger_diff >= 1e-6):
        print("\nIdentity matches, but nonzero orientation does NOT (even fresh). "
              "This confirms orientation itself is the cause, independent of "
              "accumulated history -- implicating the rotation-handling fix directly, "
              "not warm-start state.")
    else:
        print("\nUnexpected pattern -- even identity orientation shows a mismatch here, "
              "which contradicts the earlier isolate test. Investigate directly rather "
              "than trust either prior conclusion.")


if __name__ == "__main__":
    run_check()