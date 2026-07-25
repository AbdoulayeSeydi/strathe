#!/usr/bin/env python3
"""
Checks whether IMUSensor/VelocitySensor have real, built-in noise by
holding the vehicle perfectly still (zero command, zero initial velocity)
and looking at how much the readings jitter across repeated ticks with
NOTHING driving them — isolates sensor noise from our own physics/feedback
loop entirely.

Usage:
    python diagnose_sensor_noise.py
"""
import numpy as np
import holoocean

env = holoocean.make("SimpleUnderwater-Hovering")
agent = env.agents[list(env.agents.keys())[0]]
agent.set_control_scheme(2)
agent.set_physics_state(location=[0, 0, 0], rotation=[0, 0, 0], velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

zero_command = np.zeros(6)
imu_readings = []
vel_readings = []
for i in range(20):
    state = env.step(zero_command)
    imu_readings.append(np.asarray(state["IMUSensor"])[1].copy())  # angular velocity row
    vel_readings.append(np.asarray(state["VelocitySensor"]).copy())

imu_readings = np.array(imu_readings)
vel_readings = np.array(vel_readings)

print("With ZERO command and ZERO initial velocity, held for 20 ticks:")
print(f"\nIMUSensor angular velocity — should be exactly 0 if noiseless:")
print(f"  mean: {imu_readings.mean(axis=0)}")
print(f"  std:  {imu_readings.std(axis=0)}")
print(f"  min:  {imu_readings.min(axis=0)}")
print(f"  max:  {imu_readings.max(axis=0)}")

print(f"\nVelocitySensor — should be exactly 0 if noiseless:")
print(f"  mean: {vel_readings.mean(axis=0)}")
print(f"  std:  {vel_readings.std(axis=0)}")
print(f"  min:  {vel_readings.min(axis=0)}")
print(f"  max:  {vel_readings.max(axis=0)}")

print("\nIf std is meaningfully nonzero here (with nothing at all driving the vehicle),")
print("that confirms real sensor noise is the source of the run-to-run RMSE variance,")
print("not a state-reset bug.")