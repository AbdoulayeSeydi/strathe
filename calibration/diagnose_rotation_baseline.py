#!/usr/bin/env python3
"""
Simplest possible check: what does PoseSensor's rotation matrix actually
look like at a known, tiny, unambiguous rotation — before drawing any
conclusion from the earlier confusing 90-degree/10-tick results.

Usage:
    python3 diagnose_rotation_baseline.py
"""
import numpy as np
import holoocean

env = holoocean.make("SimpleUnderwater-Hovering")
agent = env.agents[list(env.agents.keys())[0]]

print("=" * 70)
print("1. FRESH ENVIRONMENT — no teleport, no command at all")
print("=" * 70)
state = env.step(np.zeros(8))  # default scheme 0, all-zero thruster command, just to read state
R_fresh = np.asarray(state["PoseSensor"])[:3, :3]
print(f"Rotation matrix at creation (before any teleport):\n{R_fresh}")

print()
print("=" * 70)
print("2. TELEPORT TO EXPLICIT ZERO ROTATION")
print("=" * 70)
agent.teleport(location=[0, 0, 0], rotation=[0, 0, 0])
state = env.step(np.zeros(8))
R_zero = np.asarray(state["PoseSensor"])[:3, :3]
print(f"Rotation matrix after teleport(rotation=[0,0,0]):\n{R_zero}")

print()
print("=" * 70)
print("3. TELEPORT TO A SMALL, UNAMBIGUOUS 10-DEGREE YAW ONLY")
print("=" * 70)
agent.teleport(location=[0, 0, 0], rotation=[0, 0, 10])
state = env.step(np.zeros(8))
R_yaw10 = np.asarray(state["PoseSensor"])[:3, :3]
print(f"Rotation matrix after teleport(rotation=[0,0,10]):\n{R_yaw10}")
print(f"\nR[2,2] = {R_yaw10[2,2]:.4f} (expect close to +1.0 — a pure yaw should barely touch this)")
print(f"R[0,0] = {R_yaw10[0,0]:.4f} (expect close to cos(10deg) = 0.9848)")
print(f"R[0,1] = {R_yaw10[0,1]:.4f}, R[1,0] = {R_yaw10[1,0]:.4f}")
print(f"(right-handed +yaw: R[0,1]=-sin(10)=-0.1736, R[1,0]=+sin(10)=+0.1736)")
print(f"(left-handed +yaw:  R[0,1]=+sin(10)=+0.1736, R[1,0]=-sin(10)=-0.1736)")

print()
print("=" * 70)
print("4. SAME SMALL YAW, BUT ONLY ROLL THIS TIME (10-degree ROLL only, yaw/pitch=0)")
print("=" * 70)
agent.teleport(location=[0, 0, 0], rotation=[10, 0, 0])
state = env.step(np.zeros(8))
R_roll10 = np.asarray(state["PoseSensor"])[:3, :3]
print(f"Rotation matrix after teleport(rotation=[10,0,0]):\n{R_roll10}")
print(f"R[0,0] = {R_roll10[0,0]:.4f} (expect close to +1.0 — a pure roll should barely touch this)")