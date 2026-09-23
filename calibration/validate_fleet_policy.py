#!/usr/bin/env python3
"""
validate_fleet_policy.py -- pre-flight check that a policy fits the fleet v2
per-vehicle contract, before any compute is spent on it.

    python validate_fleet_policy.py my_policy.zip [--algorithm ppo]

## Fleet v2 policy contract
One policy, applied separately to each vehicle (parameter sharing):
  observation  Box(13,) float32 -- see fleet_env.py for the layout
  action       Box(6,)  float32, bounded [-1, 1] (scaled to 30 N / N*m)
  algorithms   PPO (validated by this project's own training), SAC, TD3
               (stable-baselines3 classes; accepted, not yet validated)
Not supported: recurrent policies (the fleet evaluator keeps no hidden
state between ticks), policies from other frameworks (wrap them in an SB3
class or ask us about an adapter).

Common mismatches get a specific explanation instead of a shape error:
  Box(24,)/Box(12,)  -> a fleet v1 centralized policy (drove both vehicles
                        from both vehicles' states; retired, see fleet_env.py)
  Box(11,)/Box(15,)  -> a single-vehicle Arena policy (waypoint, docking, ...)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from fleet_env import ACT_DIM, OBS_DIM

VALIDATED_ALGORITHMS = ("ppo",)
ACCEPTED_ALGORITHMS = ("ppo", "sac", "td3")

_KNOWN_FOREIGN_SHAPES = {
    (24,): "a fleet v1 CENTRALIZED policy (one network driving both vehicles from both "
           "vehicles' states). Fleet v2 needs a per-vehicle policy; train one with "
           "train_fleet_policy.py",
    (11,): "a single-vehicle Arena policy (waypoint / station_keeping / docking task), "
           "not a fleet policy",
    (15,): "a single-vehicle Arena obstacle_avoidance policy, not a fleet policy",
}


class FleetPolicyError(ValueError):
    """Policy doesn't meet the fleet v2 contract. Message says why and what to do."""


def _resolve(path: str) -> str:
    return path if path.endswith(".zip") else f"{path}.zip"


def load_fleet_policy(path: str, algorithm: str = "ppo"):
    alg = algorithm.lower()
    if alg not in ACCEPTED_ALGORITHMS:
        raise FleetPolicyError(f"algorithm must be one of {ACCEPTED_ALGORITHMS}, got {algorithm!r}")
    zip_path = _resolve(path)
    if not os.path.exists(zip_path):
        raise FleetPolicyError(f"policy file not found: {zip_path!r} (paths are relative to "
                               f"where you run the command; shipped policies live in calibration/)")
    from stable_baselines3 import PPO, SAC, TD3
    cls = {"ppo": PPO, "sac": SAC, "td3": TD3}[alg]
    try:
        model = cls.load(zip_path, device="cpu")
    except Exception as e:  # SB3 raises a variety of errors for wrong-class or corrupt files
        raise FleetPolicyError(f"could not load {zip_path!r} as a {alg.upper()} policy: "
                               f"{type(e).__name__}: {e}") from e
    check_fleet_policy(model, zip_path)
    return model


def check_fleet_policy(model, label: str = "<policy>") -> None:
    obs_space, act_space = model.observation_space, model.action_space
    obs_shape = getattr(obs_space, "shape", None)
    if obs_shape != (OBS_DIM,):
        hint = _KNOWN_FOREIGN_SHAPES.get(obs_shape)
        raise FleetPolicyError(
            f"{label}: observation shape {obs_shape}, fleet v2 needs ({OBS_DIM},)."
            + (f" This looks like {hint}." if hint else ""))
    act_shape = getattr(act_space, "shape", None)
    if act_shape != (ACT_DIM,):
        raise FleetPolicyError(f"{label}: action shape {act_shape}, fleet v2 needs ({ACT_DIM},) per vehicle")
    low, high = np.asarray(act_space.low), np.asarray(act_space.high)
    if not (np.allclose(low, -1.0) and np.allclose(high, 1.0)):
        raise FleetPolicyError(
            f"{label}: action bounds [{low.min()}, {high.max()}], fleet v2 needs [-1, 1] "
            f"(the environment scales to newtons itself; a policy emitting newtons would be scaled twice)")
    if "Recurrent" in type(model.policy).__name__:
        raise FleetPolicyError(f"{label}: recurrent policies are not supported by the fleet evaluator")
    probe = np.zeros((2, OBS_DIM), dtype=np.float32)
    probe[:, 11] = 1.0  # "no beacon yet" -- the very first observation of every episode
    actions, _ = model.predict(probe, deterministic=True)
    actions = np.asarray(actions)
    if actions.shape != (2, ACT_DIM) or not np.all(np.isfinite(actions)):
        raise FleetPolicyError(f"{label}: predict() on a real first observation returned "
                               f"shape {actions.shape} / non-finite values")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("policy")
    ap.add_argument("--algorithm", default="ppo")
    args = ap.parse_args(argv)
    try:
        load_fleet_policy(args.policy, args.algorithm)
    except FleetPolicyError as e:
        print(f"FAIL: {e}")
        return 1
    note = "" if args.algorithm.lower() in VALIDATED_ALGORITHMS else \
        f" (note: {args.algorithm.upper()} is accepted but not yet validated by a fleet training run)"
    print(f"PASS: {_resolve(args.policy)} -- observation ({OBS_DIM},), action ({ACT_DIM},) in [-1, 1]{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
