#!/usr/bin/env python3
"""
Follow-up to diagnose_initial_angvel.py: the first-tick readback matched
NEITHER the desired body-frame value NOR the uncorrected one — came back
near zero instead. Tests whether this is a simple one-tick lag (set_
physics_state not fully reflected until the 2nd or 3rd step) by reading
back across several consecutive steps instead of just one.

Usage:
    python diagnose_initial_angvel_lag.py
"""
import numpy as np

from holoocean_backend import make_deterministic_environment

BODY_FRAME_CORRECTION = np.diag([1.0, -1.0, -1.0])

env = make_deterministic_environment("SimpleUnderwater-Hovering")
agent = env.agents[list(env.agents.keys())[0]]
agent.set_control_scheme(2)

desired_body_omega = np.array([0.4, -0.6, 0.3])
world_ang_vel0_as_currently_coded = BODY_FRAME_CORRECTION @ desired_body_omega

print(f"Desired BODY-frame angular velocity: {desired_body_omega}")
print(f"Value passed to set_physics_state:   {world_ang_vel0_as_currently_coded}\n")

agent.set_physics_state(location=[0, 0, 0], rotation=[0, 0, 0],
                         velocity=[0, 0, 0], angular_velocity=world_ang_vel0_as_currently_coded)

print("Reading back across 5 consecutive ticks with a ZERO command each time...")
for i in range(5):
    state = env.step(np.zeros(6))
    readback = np.asarray(state["IMUSensor"])[1]
    print(f"  Tick {i+1}: {readback}")

print(f"\nCompare each tick against desired {desired_body_omega} and "
      f"uncorrected {world_ang_vel0_as_currently_coded} to see which (if either) it converges to.")