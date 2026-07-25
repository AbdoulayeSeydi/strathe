#!/usr/bin/env python3
"""
Directly tests set_physics_state's angular_velocity frame convention —
previously extrapolated from the pattern (rotation matrix + Custom
Dynamics angular input both needed diag(1,-1,-1)), never independently
confirmed. Gives a known BODY-frame angular velocity, applies our current
code's correction to compute what we pass to set_physics_state, steps
ONCE with a ZERO command (no dynamics — purely checking whether the
INITIAL condition was set correctly), and reads back IMUSensor
immediately.

Usage:
    python diagnose_initial_angvel.py
"""
import numpy as np
import holoocean

BODY_FRAME_CORRECTION = np.diag([1.0, -1.0, -1.0])  # our current, extrapolated assumption

env = holoocean.make("SimpleUnderwater-Hovering")
agent = env.agents[list(env.agents.keys())[0]]
agent.set_control_scheme(2)

# A clearly nonzero, multi-axis test vector — not just one axis, so we can
# see the full pattern (or lack thereof) at once, same as the earlier
# angular-input tests did.
desired_body_omega = np.array([0.4, -0.6, 0.3])
print(f"Desired BODY-frame angular velocity: {desired_body_omega}\n")

# Body == world at rotation=[0,0,0] (confirmed earlier), so R0 = identity —
# this isolates the test to ONLY the correction question, no rotation
# matrix conversion involved at all.
world_ang_vel0_as_currently_coded = BODY_FRAME_CORRECTION @ desired_body_omega
print(f"Value our current code would pass to set_physics_state: {world_ang_vel0_as_currently_coded}\n")

agent.set_physics_state(location=[0, 0, 0], rotation=[0, 0, 0],
                         velocity=[0, 0, 0], angular_velocity=world_ang_vel0_as_currently_coded)

# Step ONCE with a ZERO command — no forces, no torques, purely reading
# back what the initial condition actually was.
state = env.step(np.zeros(6))
readback = np.asarray(state["IMUSensor"])[1]

print(f"IMUSensor angular velocity immediately after (zero command, first step): {readback}")
print(f"\nExpected if our current correction is RIGHT: matches desired {desired_body_omega} closely")
print(f"If it instead matches world_ang_vel0_as_currently_coded ({world_ang_vel0_as_currently_coded}) "
      f"directly, our correction is WRONG and should be removed for this specific input.")
match_desired = np.allclose(readback, desired_body_omega, atol=0.05)
match_uncorrected = np.allclose(readback, world_ang_vel0_as_currently_coded, atol=0.05)
print(f"\nMatches desired (correction is RIGHT): {match_desired}")
print(f"Matches uncorrected value (correction is WRONG/unneeded here): {match_uncorrected}")