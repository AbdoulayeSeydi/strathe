#!/usr/bin/env python3
"""
Empirically determines VelocitySensor's frame convention (body vs. world)
by rotating the vehicle 90deg in yaw FIRST, then applying a purely
world-frame +X acceleration via the Custom Dynamics scheme, and checking
which axis VelocitySensor reports the motion on.

If VelocitySensor is WORLD-frame: it should report velocity on the SAME
axis (+X) regardless of the vehicle's rotation, since the applied
acceleration was specified in world frame throughout.

If VelocitySensor is BODY-frame: rotating 90deg in yaw first means a
world-frame +X push now looks like body-frame +Y (or -Y, depending on
rotation direction) motion, since body axes rotated with the vehicle.

Also inspects agent.set_physics_state()'s real signature via introspection,
since its exact parameters weren't discoverable just from calling it.

Usage:
    python3 diagnose_holoocean_frames.py
"""
import inspect
import numpy as np
import holoocean

env = holoocean.make("SimpleUnderwater-Hovering")
agent_name = list(env.agents.keys())[0]
agent = env.agents[agent_name]

print("=" * 70)
print("1. INSPECTING set_physics_state's REAL SIGNATURE")
print("=" * 70)
try:
    sig = inspect.signature(agent.set_physics_state)
    print(f"set_physics_state signature: {sig}")
except (TypeError, ValueError) as e:
    print(f"Could not introspect signature ({e}) — trying help() output:")
    help(agent.set_physics_state)

print()
print("=" * 70)
print("2. INSPECTING teleport's REAL SIGNATURE")
print("=" * 70)
try:
    sig = inspect.signature(agent.teleport)
    print(f"teleport signature: {sig}")
except (TypeError, ValueError) as e:
    print(f"Could not introspect signature ({e})")

print()
print("=" * 70)
print("3. FRAME TEST: rotate 90deg yaw, then push in WORLD +X, check VelocitySensor")
print("=" * 70)

# Rotate the vehicle 90 degrees in yaw (rotation is [roll, pitch, yaw] in degrees, per docs)
agent.teleport(location=None, rotation=[0, 0, 90])
agent.set_control_scheme(2)  # Custom Dynamics: 6-vector world-frame accel

# Step once with zero command to let the teleport settle and get a clean baseline
state = env.step(np.zeros(6))
print(f"After teleport to yaw=90deg — PoseSensor rotation block:\n{np.asarray(state['PoseSensor'])[:3,:3]}")

# Apply a pure WORLD-FRAME +X acceleration for several ticks
world_x_accel = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
for _ in range(10):
    state = env.step(world_x_accel)

vel = np.asarray(state["VelocitySensor"])
print(f"\nAfter 10 ticks of world-frame [2,0,0,0,0,0] accel, with vehicle yawed 90deg:")
print(f"VelocitySensor reports: {vel}")
print(f"\nIf VelocitySensor is WORLD-frame: expect motion concentrated on X (index 0)")
print(f"If VelocitySensor is BODY-frame:  expect motion concentrated on Y (index 1) instead,")
print(f"                                  since a world-X push looks like body-Y after a 90deg yaw")

print()
print("Diagnostic complete.")