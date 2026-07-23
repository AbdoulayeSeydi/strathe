#!/usr/bin/env python3
"""
Checks whether the Custom Dynamics LINEAR acceleration input needs the
same Y,Z-flip correction already confirmed for rotation and angular
acceleration — only ever tested with a pure-X push before (which
wouldn't reveal this either way).

Usage:
    python3 diagnose_linear_input_sign.py
"""
import numpy as np
import holoocean

env = holoocean.make("SimpleUnderwater-Hovering")
agent = env.agents[list(env.agents.keys())[0]]
agent.set_control_scheme(2)

for label, command in [
    ("+Y world-frame linear accel", np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])),
    ("+Z world-frame linear accel", np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])),
]:
    agent.set_physics_state(location=[0, 0, 0], rotation=[0, 0, 0], velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
    print(f"Test: {label}")
    for i in range(3):
        state = env.step(command)
    vel = np.asarray(state["VelocitySensor"])
    pos = np.asarray(state["PoseSensor"])[:3, 3]
    print(f"  After 3 ticks — VelocitySensor: {vel}, position: {pos}")
    print()

print("Expected if X-stays/Y,Z-flip pattern also applies to LINEAR input:")
print("  +Y command -> expect NEGATIVE velocity/position on index 1")
print("  +Z command -> expect NEGATIVE velocity/position on index 2")