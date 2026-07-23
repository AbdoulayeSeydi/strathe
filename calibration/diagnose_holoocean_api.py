#!/usr/bin/env python3
"""
Discovers HoloOcean's actual API surface for the Custom Dynamics control
scheme, since several details can't be confirmed from documentation alone
(this needs to run against the REAL installed HoloOcean - can't be tested
in a sandbox without a GPU/display). Run this BEFORE building the actual
integration backend, rather than guessing at method names / sensor formats.

What this discovers:
  1. What env.agents actually exposes (looking for a way to set control
     scheme, and a teleport/reset method)
  2. The exact state dict keys/shapes/formats for PoseSensor and
     VelocitySensor after a step
  3. Whether HoveringAUV's Custom Dynamics (acceleration) control scheme
     can actually be set and stepped without error

Usage:
    python3 diagnose_holoocean_api.py
"""
import numpy as np
import holoocean

print("=" * 70)
print("1. CREATING ENVIRONMENT")
print("=" * 70)
env = holoocean.make("SimpleUnderwater-Hovering")
print(f"env type: {type(env)}")
print(f"env public attributes: {[a for a in dir(env) if not a.startswith('_')]}")

print()
print("=" * 70)
print("2. CHECKING AGENTS")
print("=" * 70)
print(f"env.agents: {env.agents}")
for name, agent in env.agents.items():
    print(f"\nAgent '{name}': type={type(agent)}")
    print(f"  public methods/attrs: {[a for a in dir(agent) if not a.startswith('_')]}")

print()
print("=" * 70)
print("3. STEPPING ONCE WITH DEFAULT (THRUSTER) COMMAND — CHECKING STATE FORMAT")
print("=" * 70)
command = np.array([0, 0, 0, 0, 0, 0, 0, 0])
state = env.step(command)
print(f"state keys: {list(state.keys())}")
for k, v in state.items():
    arr = np.asarray(v)
    print(f"  {k}: shape={arr.shape}, dtype={arr.dtype}")
    print(f"    value: {v}")

print()
print("=" * 70)
print("4. ATTEMPTING TO SET CUSTOM DYNAMICS CONTROL SCHEME (guessing the exact call)")
print("=" * 70)
agent_name = list(env.agents.keys())[0]
agent = env.agents[agent_name]
scheme_set_ok = False
try:
    if hasattr(agent, "set_control_scheme"):
        agent.set_control_scheme(2)
        print("SUCCESS: agent.set_control_scheme(2)")
        scheme_set_ok = True
    elif hasattr(env, "set_control_scheme"):
        env.set_control_scheme(agent_name, 2)
        print("SUCCESS: env.set_control_scheme(agent_name, 2)")
        scheme_set_ok = True
    else:
        print("NEITHER agent.set_control_scheme NOR env.set_control_scheme found.")
        print("Check the printed attribute lists above manually for the real method name.")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")

print()
print("=" * 70)
print("5. ATTEMPTING A CUSTOM DYNAMICS (ACCELERATION) COMMAND")
print("=" * 70)
if scheme_set_ok:
    try:
        accel_command = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])  # small upward world accel
        state2 = env.step(accel_command)
        print(f"SUCCESS — stepped with acceleration command, state keys: {list(state2.keys())}")
        for k, v in state2.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
else:
    print("Skipped — control scheme wasn't successfully set in step 4.")

print()
print("=" * 70)
print("6. CHECKING FOR TELEPORT / RESET METHODS")
print("=" * 70)
for name, agent in env.agents.items():
    for candidate in ["teleport", "set_state", "reset", "set_physics_state"]:
        print(f"  agent '{name}' has '{candidate}': {hasattr(agent, candidate)}")
print(f"  env has 'reset': {hasattr(env, 'reset')}")

print()
print("Diagnostic complete.")