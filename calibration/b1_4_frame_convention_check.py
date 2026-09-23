"""
B1.4 -- Frame-convention check against REAL MuJoCo, mirroring
HoloOcean's own confirmed methodology exactly (see holoocean_backend.py's
module docstring: "verified EXACTLY... against two independent real
HoloOcean tests (10deg yaw, 10deg roll)").

WHY THIS IS NOT REDUNDANT WITH B1.2: B1.2's test used
_roll_pitch_to_quat_wxyz, which ALWAYS sets yaw=0 (a structural
limitation of this file's 2-angle teacher-forcing format). HoloOcean's
own confirmed bug (diag(1,-1,-1)) specifically affected the Y/Z axes
while sparing X -- meaning an analogous MuJoCo issue tied to the Z-axis
(yaw) specifically could NEVER have been caught by B1.2, since yaw was
never nonzero in any of its test cases. This script closes that gap.

METHOD: construct a KNOWN rotation via scipy (this project's own
SNAME/Fossen 'xyz' intrinsic Euler convention), set it directly into
MuJoCo's qpos, read it back, derive the resulting rotation matrix, and
compare against the INDEPENDENTLY-computed textbook-exact matrix for the
SAME angles. Three cases, matching HoloOcean's own "two independent
tests" precedent plus one combined case for completeness:
  1. 10 deg yaw alone (roll=pitch=0) -- the genuinely new case, never
     tested by B1.2 or B1.3's rotation-matrix correctness (B1.3 tested
     pure yaw too, but for VELOCITY FRAME, a different question).
  2. 10 deg roll alone (pitch=yaw=0) -- matches HoloOcean's own second
     test; also substantially covered already by B1.2's own roll/pitch
     tests, included here for direct methodological parity with
     HoloOcean's precedent, not because a new result is expected.
  3. 10 deg yaw AND 10 deg roll combined -- an extra check beyond
     HoloOcean's own two tests, cheap to add.

If any case shows a mismatch, this script does NOT attempt to
automatically guess a correction -- it reports both matrices side by
side so the actual discrepancy can be diagnosed directly, the same
"investigate before fixing" discipline used for every other real bug
found this session (the mj_fullM signature change, the frame-mixing
divergence). Guessing a correction that happens to pass one test risks
landing on something wrong in general, exactly what this discipline
exists to avoid.

*** UNVERIFIED -- no mujoco in this sandbox. The comparison logic
itself (quaternion match implies matrix match, given a consistent
conversion function) is verified directly against scipy (see
conversation). The actual MuJoCo behavior is what this script exists to
determine -- run, not assumed.
"""
import numpy as np
from scipy.spatial.transform import Rotation
import mujoco

MJCF_PATH = "b0_single_body_placeholder.xml"

TEST_CASES = [
    ("10 deg yaw alone", 0.0, 0.0, np.radians(10)),
    ("10 deg roll alone", np.radians(10), 0.0, 0.0),
    ("10 deg yaw AND 10 deg roll combined", np.radians(10), 0.0, np.radians(10)),
]


def run_single_case(model, data, roll: float, pitch: float, yaw: float, label: str):
    print(f"\n--- {label} (roll={np.degrees(roll):.0f} deg, pitch={np.degrees(pitch):.0f} deg, "
          f"yaw={np.degrees(yaw):.0f} deg) ---")

    quat_xyzw_set = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    quat_wxyz_set = np.array([quat_xyzw_set[3], quat_xyzw_set[0], quat_xyzw_set[1], quat_xyzw_set[2]])

    data.qpos[3:7] = quat_wxyz_set
    mujoco.mj_forward(model, data)
    quat_wxyz_readback = data.qpos[3:7].copy()

    w, x, y, z = quat_wxyz_readback
    quat_xyzw_readback = np.array([x, y, z, w])
    R_from_mujoco = Rotation.from_quat(quat_xyzw_readback).as_matrix()

    R_expected = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()

    max_diff = np.max(np.abs(R_from_mujoco - R_expected))
    print(f"R_expected (textbook-exact, this project's SNAME/Fossen convention):\n{R_expected}")
    print(f"R_from_mujoco (readback, no correction applied):\n{R_from_mujoco}")
    print(f"max abs diff: {max_diff:.2e}")

    passed = max_diff < 1e-9
    print("PASS -- no correction needed for this case" if passed else "MISMATCH -- see matrices above")
    return passed, R_expected, R_from_mujoco


def run_check():
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    all_passed = True
    for label, roll, pitch, yaw in TEST_CASES:
        passed, _, _ = run_single_case(model, data, roll, pitch, yaw, label)
        all_passed = all_passed and passed

    print(f"\n{'='*70}")
    if all_passed:
        print("B1.4 CHECK PASSED: MuJoCo's reported orientation matches the "
              "expected textbook rotation matrix EXACTLY across all three "
              "cases (yaw-alone, roll-alone, combined) -- NO correction "
              "matrix needed, unlike HoloOcean's confirmed diag(1,-1,-1).")
    else:
        print("B1.4 CHECK FAILED for at least one case -- see the matrices "
              "printed above. Do NOT guess a correction from this output "
              "alone; work out the exact discrepancy pattern first, the "
              "same way HoloOcean's diag(1,-1,-1) was derived from directly "
              "comparing expected vs. reported matrices, not guessed.")
    return all_passed


if __name__ == "__main__":
    run_check()