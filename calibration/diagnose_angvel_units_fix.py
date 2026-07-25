#!/usr/bin/env python3
"""
Tests whether set_physics_state's angular_velocity parameter expects
DEGREES/second (like the confirmed-degrees rotation parameter), not
radians/second — the likely full explanation for the ~58.77x magnitude
gap found in diagnose_initial_angvel_lag.py (matches 180/pi very
closely, off only by the also-confirmed ~2.5%/tick decay).

Usage:
    python diagnose_angvel_units_fix.py
"""
import numpy as np

from holoocean_backend import make_deterministic_environment

BODY_FRAME_CORRECTION = np.diag([1.0, -1.0, -1.0])

env = make_deterministic_environment("SimpleUnderwater-Hovering")
agent = env.agents[list(env.agents.keys())[0]]
agent.set_control_scheme(2)

desired_body_omega_rad = np.array([0.4, -0.6, 0.3])  # radians/sec, our own convention
world_ang_vel0_rad = BODY_FRAME_CORRECTION @ desired_body_omega_rad

# THE FIX: convert to degrees/sec before passing, same as we already do for rotation
world_ang_vel0_deg = np.degrees(world_ang_vel0_rad)

print(f"Desired (radians/sec, our convention): {desired_body_omega_rad}")
print(f"Passing to set_physics_state (DEGREES/sec this time): {world_ang_vel0_deg}\n")

agent.set_physics_state(location=[0, 0, 0], rotation=[0, 0, 0],
                         velocity=[0, 0, 0], angular_velocity=world_ang_vel0_deg)

print("Reading back across 5 consecutive ticks with a ZERO command each time...")
for i in range(5):
    state = env.step(np.zeros(6))
    readback = np.asarray(state["IMUSensor"])[1]
    print(f"  Tick {i+1}: {readback}")

print(f"\nExpected NOW: much closer to desired {desired_body_omega_rad} (radians/sec), only off by")
print(f"the small, separately-confirmed ~2.5%/tick decay — not off by ~58x anymore.")