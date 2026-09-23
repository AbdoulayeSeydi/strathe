"""
Run this AFTER adding the two functions from ADD_TO_dynamics_model.py
to your real dynamics_model.py. This independently confirms they match
YOUR real _restoring_forces() and compute_acceleration() to machine
precision -- don't just take my word for it, verify it yourself.

If this fails, do NOT proceed to use mujoco_backend.py's forward-sim
default -- something about the addition doesn't match your real file's
actual behavior, and that needs to be resolved first.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from dynamics_model import (
    _restoring_forces, _restoring_forces_from_rotation_matrix,
    compute_acceleration, compute_acceleration_from_rotation_matrix,
)
from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec

vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
params = load_calibrated_parameters("calibrated_parameters_a1.yaml")

print("=== Test 1: _restoring_forces_from_rotation_matrix vs your real _restoring_forces ===")
print("Using genuine rotations with real, nonzero yaw (not just yaw=0):")
max_diff_overall = 0.0
test_cases = [
    (0.3, 0.2, 0.0), (0.3, 0.2, 0.5), (0.3, 0.2, 1.2), (-0.4, 0.6, 2.0),
    (0.1, -0.7, 0.3), (np.radians(89.9), 0.1, 0.0), (0.1, np.radians(89.9), 1.5),
]
for roll, pitch, yaw in test_cases:
    R = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    r_ext, p_ext, _ = Rotation.from_matrix(R).as_euler("xyz")
    g1 = _restoring_forces(r_ext, p_ext, vehicle.mass_kg, params.net_buoyancy_z, params.cg_cb_offset_z)
    g2 = _restoring_forces_from_rotation_matrix(R, vehicle.mass_kg, params.net_buoyancy_z, params.cg_cb_offset_z)
    diff = np.max(np.abs(g1 - g2))
    max_diff_overall = max(max_diff_overall, diff)
    print(f"  roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}: diff={diff:.3e}")
print(f"Overall max diff: {max_diff_overall:.3e}")
assert max_diff_overall < 1e-9, "MISMATCH -- do not proceed, something doesn't match your real file"
print("PASSED")

print()
print("=== Test 2: compute_acceleration_from_rotation_matrix vs your real compute_acceleration ===")
rng = np.random.default_rng(0)
max_diff_overall = 0.0
for _ in range(20):
    roll, pitch, yaw = rng.uniform(-1.5, 1.5, 3)
    v = rng.uniform(-1, 1, 6)
    tau = rng.uniform(-50, 50, 6)
    R = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    r_ext, p_ext, _ = Rotation.from_matrix(R).as_euler("xyz")
    v_dot1, _, _ = compute_acceleration(params, vehicle, v, tau, r_ext, p_ext)
    v_dot2, _, _ = compute_acceleration_from_rotation_matrix(params, vehicle, v, tau, R)
    diff = np.max(np.abs(v_dot1 - v_dot2))
    max_diff_overall = max(max_diff_overall, diff)
print(f"Overall max diff across 20 random trials: {max_diff_overall:.3e}")
assert max_diff_overall < 1e-9, "MISMATCH -- do not proceed, something doesn't match your real file"
print("PASSED")
print()
print("Both new functions confirmed to match your REAL existing functions to machine precision.")
print("Safe to proceed with mujoco_backend.py's forward-sim path.")