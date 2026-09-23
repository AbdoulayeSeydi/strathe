"""
Confirms whether MuJoCo's freejoint ANGULAR velocity is world-frame or
body-frame -- the one remaining assumption ("as any physical IMU must
be" body-frame, reasoned about but never independently tested, matching
how HoloOcean treated its own IMU) that the divergence-bug fix is about
to depend on directly. B1.3 confirmed LINEAR velocity is world-frame;
this is the angular analog, not covered by that test.

METHOD: set qvel[3:]=[1,0,0] (unit angular rate about "X", whatever
frame that turns out to be) at two starting orientations -- 0 deg yaw
and 90 deg yaw -- step a tiny dt, and extract the WORLD-frame axis of
the resulting small rotation via R_new @ R_old.T's rotation vector.

NOTE: a pure-yaw test cannot distinguish frame convention for a Z-axis
angular velocity (yawing doesn't move the Z axis at all) -- verified by
hand before writing this script (see conversation). Testing rotation
about X specifically, since body-X visibly rotates to world-Y under a
90-degree yaw (confirmed in b1_3's own test), is what makes this
distinguishable at all.

  - If the extracted axis stays ~[1,0,0] regardless of starting yaw:
    WORLD-frame (matches B1.3's linear-velocity result).
  - If it's [1,0,0] at 0deg but ~[0,1,0] at 90deg: BODY-frame (matches
    the "as any physical IMU must be" assumption already coded in, but
    now actually confirmed rather than just reasoned about).

*** UNVERIFIED -- no mujoco in this sandbox. The axis-extraction
methodology itself is verified against synthetic data representing both
competing hypotheses, cleanly distinguishing them (see conversation).
"""
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

MJCF_PATH = "b0_single_body_placeholder.xml"


def _incremental_rotation_axis(R_old: np.ndarray, R_new: np.ndarray) -> np.ndarray:
    dR = R_new @ R_old.T
    rotvec = Rotation.from_matrix(dR).as_rotvec()
    angle = np.linalg.norm(rotvec)
    return rotvec / angle if angle > 1e-9 else np.zeros(3)


def run_single_test(model, data, yaw_deg: float, dt: float = 0.001):
    yaw_rad = np.radians(yaw_deg)
    quat_xyzw = Rotation.from_euler("xyz", [0, 0, yaw_rad]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    data.qpos[:3] = 0.0
    data.qpos[3:7] = quat_wxyz
    data.qvel[:] = 0.0
    data.qvel[3:] = [1.0, 0.0, 0.0]
    data.xfrc_applied[:, :] = 0.0

    R_old = Rotation.from_quat(quat_xyzw).as_matrix()

    model.opt.timestep = dt
    mujoco.mj_step(model, data)

    w, x, y, z = data.qpos[3:7]
    R_new = Rotation.from_quat([x, y, z, w]).as_matrix()

    return _incremental_rotation_axis(R_old, R_new)


def run_check():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    print("--- yaw = 0 deg, angular velocity = [1,0,0] ---")
    axis_0 = run_single_test(model, data, yaw_deg=0.0)
    print(f"resulting incremental rotation axis (world frame): {axis_0.round(4)}")

    print("\n--- yaw = 90 deg, angular velocity = [1,0,0] ---")
    axis_90 = run_single_test(model, data, yaw_deg=90.0)
    print(f"resulting incremental rotation axis (world frame): {axis_90.round(4)}")

    print(f"\n{'='*70}")
    world_signature = np.allclose(axis_0, [1, 0, 0], atol=1e-3) and np.allclose(axis_90, [1, 0, 0], atol=1e-3)
    body_signature = np.allclose(axis_0, [1, 0, 0], atol=1e-3) and np.allclose(axis_90, [0, 1, 0], atol=1e-3)

    if world_signature:
        print("RESULT: WORLD-FRAME. The rotation axis stayed fixed at world-X "
              "regardless of starting orientation. This means the CURRENT "
              "assumption in mujoco_backend.py (angular = body-frame, "
              "unrotated) is WRONG -- angular velocity would need the SAME "
              "R-based rotation treatment as linear velocity, not the direct "
              "pass-through currently coded.")
    elif body_signature:
        print("RESULT: BODY-FRAME. The rotation axis rotated WITH the body "
              "(world-X at 0deg, world-Y at 90deg) -- CONFIRMS the existing "
              "assumption in mujoco_backend.py (angular velocity used "
              "directly, no rotation). Safe to proceed with the divergence "
              "fix using this confirmed convention.")
    else:
        print(f"RESULT: AMBIGUOUS -- axis_0={axis_0.round(4)}, axis_90={axis_90.round(4)}. "
              f"Neither clean signature matched; investigate directly.")


if __name__ == "__main__":
    run_check()