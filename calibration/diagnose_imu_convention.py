#!/usr/bin/env python3
"""
Isolates IMUSensor's angular velocity convention, independent of the
already-confirmed PoseSensor rotation-matrix fix.

Starts from rotation=[0,0,0] — which, per the confirmed PoseSensor fix,
means body frame and world frame coincide EXACTLY at that instant (the
corrected rotation matrix is identity there). Commands a PURE world-frame
+Z angular acceleration right from that instant, for just a couple of
ticks (before much rotation can accumulate and complicate things), and
checks IMUSensor's reported angular velocity directly.

If IMUSensor's own convention matches ours: expect angular velocity
concentrated on index 2 (r/yaw), POSITIVE, with p and q near zero.
Any other pattern (wrong sign, or the response showing up on a different
index) tells us exactly what correction IMUSensor itself needs.

Usage:
    python3 diagnose_imu_convention.py
"""
import numpy as np
import holoocean

env = holoocean.make("SimpleUnderwater-Hovering")
agent = env.agents[list(env.agents.keys())[0]]
agent.set_control_scheme(2)

# Start from a state where body frame == world frame exactly (confirmed).
agent.set_physics_state(location=[0, 0, 0], rotation=[0, 0, 0], velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

print("Commanding pure +Z world-frame angular acceleration for 3 ticks (small, to stay near t=0)...")
command = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
for i in range(3):
    state = env.step(command)
    imu = np.asarray(state["IMUSensor"])
    pose = np.asarray(state["PoseSensor"])
    print(f"\nTick {i+1}:")
    print(f"  IMUSensor angular velocity (row 1): {imu[1]}")
    print(f"  PoseSensor rotation matrix:\n{pose[:3,:3]}")

print("\nExpected if IMUSensor matches our convention: angular velocity concentrated")
print("on index 2 (r), POSITIVE, with indices 0,1 (p,q) near zero.")