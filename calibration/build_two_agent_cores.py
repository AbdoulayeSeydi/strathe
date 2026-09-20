from __future__ import annotations

import numpy as np
import yaml

from arena_core import ArenaCore
from build_generic_parameters import build_generic_parameters
from dynamics_model import CalibrationParameters, SimulationConfig
from optimizer import load_vehicle_spec

VEHICLE_SPEC_PATH = "vehicle_spec.yaml"


def load_calibrated_parameters(yaml_path: str) -> CalibrationParameters:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    params = CalibrationParameters(**data["parameters"])
    params.validate()

    provenance = data.get("provenance")
    if provenance and not provenance.get("converged", True):
        print(f"NOTE: {yaml_path}'s calibration did NOT reach strict convergence "
              f"(cost={provenance.get('cost')}, max_nfev={provenance.get('max_nfev')}, "
              f"max_train_runs used: see build_calibrated_parameters.py's own run log) -- "
              f"fine for A1's comparative purpose, not a Claim-A-grade result. "
              f"See build_calibrated_parameters.py's provenance block for detail.")
    return params


def build_two_agent_cores(calibrated_params_path: str = "calibrated_parameters_a1.yaml",
                           dt: float = 0.1, physics_dt: float = 0.005,
                           core_class=ArenaCore):
    """Returns (core_generic, core_calibrated) -- two independently-usable
    core instances sharing the same vehicle constants and simulation
    resolution, differing only in calibration.

    core_class: C0.3 addition (see conversation) -- defaults to ArenaCore
    (the stand-in), preserving EXACT prior behavior for every existing
    caller that doesn't pass this. Pass mujoco_backend.MuJoCoArenaCore
    instead to build two agents backed by genuine, self-integrating
    MuJoCo dynamics -- both classes share the identical
    (params, vehicle, sim_config) constructor signature (MuJoCoArenaCore's
    own extra mjcf_path parameter has a default), so no other change is
    needed here."""
    loaded_spec = load_vehicle_spec(VEHICLE_SPEC_PATH)
    vehicle = loaded_spec.vehicle

    generic_params = build_generic_parameters()
    calibrated_params = load_calibrated_parameters(calibrated_params_path)

    sim_config = SimulationConfig(dt=dt, physics_dt=physics_dt)

    core_generic = core_class(generic_params, vehicle, sim_config)
    core_calibrated = core_class(calibrated_params, vehicle, sim_config)

    return core_generic, core_calibrated
