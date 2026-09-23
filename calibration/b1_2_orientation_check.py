"""
B1.2 real-machine check -- simpler and earlier than B1.4's full dynamic-
rotation test. Confirms two things against REAL MuJoCo (not the fake
harness used for mujoco_backend.py's own unit tests):

  1. Setting data.qpos[3:7] directly and calling mj_forward doesn't
     silently renormalize, reorder, or otherwise mangle the quaternion
     -- i.e. what you put in is what comes back out, unchanged.
  2. mujoco_backend._extract_roll_pitch_from_mujoco_quat() then
     correctly recovers the ORIGINAL roll/pitch from that quaternion.

This does NOT yet test whether MuJoCo's own rigid-body DYNAMICS (a real
rotation building up from applied torque, not just a directly-assigned
static quaternion) behaves per the assumed convention -- that dynamic,
known-rotation test (10deg yaw, 10deg roll, checked against the
textbook-exact rotation matrix, the same test that caught HoloOcean's
real mismatch) is explicitly B1.4's job, not this script's.
"""
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

from mujoco_backend import _extract_roll_pitch_from_mujoco_quat, _roll_pitch_to_quat_wxyz

MJCF_PATH = "b0_single_body_placeholder.xml"

TEST_ANGLES = [
    (0.1, -0.2),
    (0.0, 0.0),
    (0.3, 0.05),
    (-0.4, 0.25),
    (0.05, -0.5),
]


def run_check():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    all_passed = True
    for roll, pitch in TEST_ANGLES:
        print(f"\n--- roll={roll}, pitch={pitch} ---")
        quat_wxyz_set = _roll_pitch_to_quat_wxyz(roll, pitch)
        data.qpos[3:7] = quat_wxyz_set
        mujoco.mj_forward(model, data)
        quat_wxyz_readback = data.qpos[3:7].copy()

        quat_diff = np.max(np.abs(quat_wxyz_readback - quat_wxyz_set))
        print(f"quaternion set:      {quat_wxyz_set}")
        print(f"quaternion readback: {quat_wxyz_readback}")
        print(f"max abs diff:        {quat_diff:.2e}")

        roll_out, pitch_out = _extract_roll_pitch_from_mujoco_quat(data)
        roll_diff = abs(roll_out - roll)
        pitch_diff = abs(pitch_out - pitch)
        print(f"extracted roll={roll_out:.6f} (diff {roll_diff:.2e}), "
              f"pitch={pitch_out:.6f} (diff {pitch_diff:.2e})")

        passed = quat_diff < 1e-9 and roll_diff < 1e-9 and pitch_diff < 1e-9
        all_passed = all_passed and passed
        print("PASS" if passed else "FAIL")

    print(f"\n{'='*60}")
    if all_passed:
        print("B1.2 CHECK PASSED: MuJoCo preserves a directly-set quaternion "
              "unchanged, and the extraction function correctly recovers the "
              "original roll/pitch. Note: this does NOT yet confirm MuJoCo's "
              "own rotational DYNAMICS follow the same convention -- that is "
              "B1.4's job.")
    else:
        print("B1.2 CHECK FAILED: see individual cases above.")
    return all_passed


if __name__ == "__main__":
    run_check()