#!/usr/bin/env python3
"""
Tests whether HoloOcean's Custom Dynamics angular acceleration input
follows a right-handed (standard robotics) or left-handed (Unreal Engine's
native) convention — in complete isolation from our own physics model.

Commands a PURE +Z world-frame angular acceleration, starting from
identity rotation and zero velocity, and checks which direction the
vehicle actually rotates. In a standard RIGHT-HANDED convention, positive
Z angular velocity/acceleration means rotation that turns +X TOWARD +Y
(counterclockwise viewed from above, looking down the +Z axis toward
origin). If HoloOcean instead rotates +X toward -Y, that's the
left-handed-convention signature.

Usage:
    python3 diagnose_handedness.py
"""
import numpy as np
import holoocean

env = holoocean.make("SimpleUnderwater-Hovering")
agent = env.agents[list(env.agents.keys())[0]]
agent.set_control_scheme(2)

# Start from a clean, known state: identity rotation, zero velocity.
agent.set_physics_state(location=[0, 0, 0], rotation=[0, 0, 0], velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

print("Commanding pure +Z world-frame angular acceleration for 10 ticks...")
command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])  # only ang_accel_z = +1.0
for i in range(10):
    state = env.step(command)

R = np.asarray(state["PoseSensor"])[:3, :3]
print(f"\nRotation matrix after 10 ticks of +Z angular accel:\n{R}")

# For a rotation about +Z, R[0,1] tells us the direction: if the body's
# +X axis has rotated TOWARD +Y (right-handed, standard convention),
# R[1,0] (how much of world-Y ends up along body-X) should be POSITIVE
# and R[0,1] should be NEGATIVE for a small positive-Z right-handed rotation
# (standard rotation matrix convention: R = [[cos,-sin,0],[sin,cos,0],[0,0,1]]
# for a right-handed +Z rotation by a small positive angle).
r01 = R[0, 1]
r10 = R[1, 0]
print(f"\nR[0,1] = {r01:.4f} (expect NEGATIVE for standard right-handed +Z rotation)")
print(f"R[1,0] = {r10:.4f} (expect POSITIVE for standard right-handed +Z rotation)")

if r10 > 0 and r01 < 0:
    print("\n=> RIGHT-HANDED convention confirmed. The angular acceleration input is NOT flipped.")
elif r10 < 0 and r01 > 0:
    print("\n=> LEFT-HANDED convention detected! HoloOcean rotates the OPPOSITE way from")
    print("   standard right-handed robotics convention for a 'positive Z' angular command.")
    print("   FIX: negate the angular acceleration command's Z component (and likely X, Y too)")
    print("   before sending to HoloOcean, OR negate on readback from IMUSensor — pick ONE")
    print("   consistently, not both.")
else:
    print("\n=> Ambiguous/near-zero result — rerun with more ticks or a larger commanded accel.")

imu = np.asarray(state["IMUSensor"])
print(f"\nIMUSensor body angular velocity (row 1): {imu[1]}")
print("Expect this to be POSITIVE on the Z component (index 2) if readback matches the")
print("commanded positive-Z input consistently (same-sign in and out).")