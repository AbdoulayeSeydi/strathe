"""
B1.3's genuinely new piece: empirically confirms whether MuJoCo's
freejoint qvel LINEAR component is world-frame or body-frame -- the one
assumption in mujoco_backend.py that was previously only documented
convention, never independently tested, exactly analogous to
holoocean_backend.py's own diagnose_holoocean_frames.py confirming
VelocitySensor's world-frame convention via a direct push-and-observe
test before relying on it.

(Every OTHER piece of B1.3's own description -- compute v_dot via
compute_acceleration(), apply via B0.7's full-override method, call
mj_step, extract body-frame velocity into a SimulationResult -- was
already built and tested as part of B1.1's implementation. This script
covers only the specific, previously-unconfirmed gap.)

METHOD, mirroring HoloOcean's own methodology exactly: set the SAME
qvel[:3] = [1,0,0] at two different orientations (0deg and 90deg yaw),
step forward, and observe which WORLD direction the body actually
moves in.
  - If the motion direction stays FIXED regardless of orientation:
    qvel is WORLD-frame (HoloOcean's own "world-frame signature").
  - If the motion direction ROTATES WITH the body: qvel is BODY-frame.
Only X/Y position change is examined -- gravity only affects Z, so this
keeps the signal clean without needing to disable anything.

Angular velocity's body-frame convention is NOT independently tested
here -- treated the same way HoloOcean's own IMU angular velocity was
("as any physical IMU must be" body-frame, standard for all physics
engines' free joints), not because it's certain, but because it lacks
HoloOcean's specific historical precedent of being surprising. If a
future check finds this assumption wrong too, this is the file to
extend.

*** UNVERIFIED -- no mujoco in this sandbox. Everything here is
standard, small-dt, gravity-affects-only-Z physics reasoning; correct
per that reasoning, not run.
"""
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

MJCF_PATH = "b0_single_body_placeholder.xml"


def run_single_test(model, data, yaw_deg: float, dt: float = 0.01):
    yaw_rad = np.radians(yaw_deg)
    quat_xyzw = Rotation.from_euler("xyz", [0, 0, yaw_rad]).as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    data.qpos[:3] = 0.0
    data.qpos[3:7] = quat_wxyz
    data.qvel[:] = 0.0
    data.qvel[:3] = [1.0, 0.0, 0.0]
    data.xfrc_applied[:, :] = 0.0

    model.opt.timestep = dt
    mujoco.mj_step(model, data)

    dx, dy = data.qpos[0], data.qpos[1]
    return dx, dy


def run_check():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    print("--- yaw = 0 deg ---")
    dx0, dy0 = run_single_test(model, data, yaw_deg=0.0)
    print(f"position change after one step: dx={dx0:.6f}, dy={dy0:.6f}")

    print("\n--- yaw = 90 deg ---")
    dx90, dy90 = run_single_test(model, data, yaw_deg=90.0)
    print(f"position change after one step: dx={dx90:.6f}, dy={dy90:.6f}")

    print(f"\n{'='*70}")
    world_frame_signature = abs(dx0) > abs(dy0) and abs(dx90) > abs(dy90)
    body_frame_signature = abs(dx0) > abs(dy0) and abs(dy90) > abs(dx90)

    if world_frame_signature:
        print("RESULT: WORLD-FRAME. Motion direction stayed fixed (dominant in "
              "X) regardless of orientation -- matches HoloOcean's own "
              "'world-frame signature'. The assumption already coded into "
              "mujoco_backend.py's _set_teacher_forced_state / "
              "_extract_body_velocity_mujoco is CONFIRMED correct for the "
              "linear component.")
    elif body_frame_signature:
        print("RESULT: BODY-FRAME. Motion direction rotated with the body "
              "(dominant in X at 0deg, dominant in Y at 90deg). The current "
              "assumption in mujoco_backend.py is WRONG for the linear "
              "component -- _set_teacher_forced_state and "
              "_extract_body_velocity_mujoco's R @ / R.T @ rotations would "
              "need to be REMOVED, not corrected, since qvel would already "
              "be in body frame with no conversion needed.")
    else:
        print(f"RESULT: AMBIGUOUS -- dx0={dx0:.6f}, dy0={dy0:.6f}, "
              f"dx90={dx90:.6f}, dy90={dy90:.6f}. Neither clean signature "
              f"matched; investigate directly rather than trust either "
              f"assumption.")


if __name__ == "__main__":
    run_check()