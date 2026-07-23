#!/usr/bin/env python3
"""
Checks whether the Custom Dynamics angular acceleration INPUT's sign
convention follows the same X-stays/Y,Z-flip pattern already confirmed
for PoseSensor's rotation matrix, or something different — by testing
a pure world-frame +X angular acceleration (roll) the same clean way
Z was just tested (from a body==world starting point).

Usage:
    python3 diagnose_angular_input_sign.py
"""
import numpy as np
import holoocean

env = holoocean.make("SimpleUnderwater-Hovering")
agent = env.agents[list(env.agents.keys())[0]]
agent.set_control_scheme(2)
agent.set_physics_state(location=[0, 0, 0], rotation=[0, 0, 0], velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

print("Test A: pure +X world-frame angular accel (roll) for 3 ticks")
command_x = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
for i in range(3):
    state = env.step(command_x)
    imu = np.asarray(state["IMUSensor"])
    print(f"  Tick {i+1}: IMU angular velocity = {imu[1]}")

print()
agent.set_physics_state(location=[0, 0, 0], rotation=[0, 0, 0], velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
print("Test B: pure +Y world-frame angular accel (pitch) for 3 ticks")
command_y = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
for i in range(3):
    state = env.step(command_y)
    imu = np.asarray(state["IMUSensor"])
    print(f"  Tick {i+1}: IMU angular velocity = {imu[1]}")

print()
print("Expected if the pattern matches PoseSensor's (X stays, Y&Z flip):")
print("  Test A (+X command): expect POSITIVE p (index 0)")
print("  Test B (+Y command): expect NEGATIVE q (index 1)")