#!/usr/bin/env python3
"""
Locates the real installed SimpleUnderwater-Hovering.json scenario file,
zeros out IMUSensor's noise parameters (AccelSigma, AngVelSigma,
AccelBiasSigma, AngVelBiasSigma), and confirms this actually produces a
noiseless IMU when passed to holoocean.make(scenario_cfg=...) instead of
the scenario name — before wiring this into the real backend.

Usage:
    python diagnose_noiseless_scenario.py
"""
import json
import glob
import os
import numpy as np
import holoocean

print("Searching for the installed scenario JSON file...")
# HoloOcean installs worlds under a per-user local app data folder; search broadly.
search_roots = [
    os.path.expanduser("~/AppData/Local/holoocean"),
    os.path.expanduser("~/.local/share/holoocean"),
]
found = None
for root in search_roots:
    if os.path.isdir(root):
        matches = glob.glob(os.path.join(root, "**", "*SimpleUnderwater-Hovering*.json"), recursive=True)
        if matches:
            found = matches[0]
            break

if found is None:
    print("Could not find the JSON file by searching common install locations.")
    print("Falling back to holoocean.packagemanager introspection...")
    import inspect
    pm_functions = [name for name in dir(holoocean.packagemanager) if not name.startswith("_")]
    print(f"Available packagemanager functions: {pm_functions}")
    raise SystemExit("Please report the above function list so we can find the right one.")

print(f"Found: {found}\n")
with open(found) as f:
    scenario = json.load(f)

# Locate and zero out IMUSensor's noise config, wherever it is in the structure
modified_count = 0
for agent in scenario.get("agents", []):
    for sensor in agent.get("sensors", []):
        if sensor.get("sensor_type") == "IMUSensor":
            cfg = sensor.setdefault("configuration", {})
            for key in ["AccelSigma", "AngVelSigma", "AccelBiasSigma", "AngVelBiasSigma"]:
                if key in cfg:
                    cfg[key] = 0.0
                    modified_count += 1

print(f"Zeroed {modified_count} noise parameter(s) in IMUSensor's config\n")

print(f"Scenario dict's top-level keys: {list(scenario.keys())}")
if "package_name" not in scenario:
    scenario["package_name"] = "Ocean"  # the package this world/scenario belongs to
    print("Added missing 'package_name': 'Ocean' — required by holoocean.make() when "
          "passing scenario_cfg directly, normally supplied separately when loading by name\n")

print("Creating environment from the MODIFIED scenario config...")
env = holoocean.make(scenario_cfg=scenario)
agent = env.agents[list(env.agents.keys())[0]]
agent.set_control_scheme(2)
agent.set_physics_state(location=[0, 0, 0], rotation=[0, 0, 0], velocity=[0, 0, 0], angular_velocity=[0, 0, 0])

zero_command = np.zeros(6)
imu_readings = []
for i in range(20):
    state = env.step(zero_command)
    imu_readings.append(np.asarray(state["IMUSensor"])[1].copy())

imu_readings = np.array(imu_readings)
print(f"IMUSensor angular velocity over 20 ticks, held still, NOISE ZEROED:")
print(f"  mean: {imu_readings.mean(axis=0)}")
print(f"  std:  {imu_readings.std(axis=0)} (expect ~0.0 now, was ~0.005-0.009 before)")