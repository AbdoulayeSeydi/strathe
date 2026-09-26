#!/usr/bin/env python3
"""
train_fleet_policy.py -- train one shared per-vehicle fleet v2 policy.

    python train_fleet_policy.py --timesteps 3000000 --seed 1 --out fleet_policy_seed1

Produces <out>.zip (a stock SB3 PPO policy: observation Box(13,), action
Box(6,)) and <out>.json (the exact training setup, git commit, runtime).

Training distribution, drawn fresh every episode:
  - comm severity uniformly from L0, L1, L2, L3a, and beacon interval
    uniformly from 0.5, 1, 2, 3 s (acoustic state updates are sparse)
  - belief condition (--belief mix, the default): a physics dead-reckoning model whose parameters are
    interpolated a random fraction alpha ~ U(0,1) of the way from the true
    vehicle's parameters to generic default parameters (60%), or
    last-beacon (15%), constant-velocity (10%), oracle (15%)
  - initial separation U(2.0, 5.5) m and closing speed U(-0.7, 0.7) m/s
Training across belief conditions (rather than on one) means evaluation
can compare conditions without favouring the one the policy was trained on.
But the observation doesn't say WHICH belief model is active, so a mixed
policy applies one average staleness correction to all of them (see
PROJECT_STATUS.md #47). --belief <kind> trains a policy specialized to one
belief model instead, as a real fleet with a fixed onboard predictor would be.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

from build_two_agent_cores import load_calibrated_parameters
from evaluate_comm_degradation_curve import COMM_CONFIGS
from fleet_belief import BeliefSpec, interpolate_params
from fleet_env import FleetConfig, FleetSim
from fleet_vec_env import CommCondition, EpisodeSampler, FleetVecEnv
from mlflow_run_utils import get_git_commit, get_git_dirty
from optimizer import initial_parameters, load_vehicle_spec
from keep_awake import keep_awake

TRUTH_PARAMS_PATH = "calibrated_parameters_fleet_truth.yaml"
VEHICLE_SPEC_PATH = "vehicle_spec.yaml"
TRAIN_SEVERITIES = ("L0", "L1", "L2", "L3a")
TRAIN_BEACON_INTERVALS_S = (0.5, 1.0, 2.0, 3.0)
BELIEF_MIX = {"physics_interpolated": 0.60, "last_beacon": 0.15, "constant_velocity": 0.10, "oracle": 0.15}


FIXED_BELIEFS = ("calibrated", "generic", "last_beacon", "constant_velocity", "oracle")


def make_fixed_belief_sampler(kind: str, truth, generic):
    """Every episode uses the same belief model -- a policy specialized to it,
    the way a real fleet would be trained with its own onboard predictor."""
    if kind == "calibrated":
        spec = BeliefSpec("physics", "calibrated", truth)
    elif kind == "generic":
        spec = BeliefSpec("physics", "generic", generic)
    elif kind in FIXED_BELIEFS:
        spec = BeliefSpec(kind, kind)
    else:
        raise ValueError(f"--belief must be 'mix' or one of {FIXED_BELIEFS}, got {kind!r}")
    return lambda rng: spec


def make_belief_sampler(truth, generic):
    kinds = list(BELIEF_MIX)
    probs = np.array([BELIEF_MIX[k] for k in kinds])

    def sample(rng: np.random.Generator) -> BeliefSpec:
        kind = kinds[rng.choice(len(kinds), p=probs)]
        if kind == "physics_interpolated":
            alpha = float(rng.uniform(0.0, 1.0))
            return BeliefSpec("physics", f"physics_alpha{alpha:.2f}", interpolate_params(truth, generic, alpha))
        return BeliefSpec(kind, kind)
    return sample


def comm_conditions(severities=TRAIN_SEVERITIES, intervals=TRAIN_BEACON_INTERVALS_S):
    return [CommCondition(f"{s}@{b:g}s", COMM_CONFIGS[s]["packet_loss_config"], COMM_CONFIGS[s]["bandwidth_config"],
                          beacon_interval_s=b)
            for s in severities for b in intervals]


def build_training_env(n_sims: int, seed: int, truth_path: str = TRUTH_PARAMS_PATH,
                       fleet_config: FleetConfig = None, belief: str = "mix",
                       spec_path: str = VEHICLE_SPEC_PATH) -> FleetVecEnv:
    spec = load_vehicle_spec(spec_path)
    vehicle = spec.vehicle
    truth = load_calibrated_parameters(truth_path)
    generic = initial_parameters(spec)
    belief_sampler = (make_belief_sampler(truth, generic) if belief == "mix"
                      else make_fixed_belief_sampler(belief, truth, generic))
    sampler = EpisodeSampler(comm_conditions(), belief_sampler)
    sims = [FleetSim(truth, vehicle, fleet_config or FleetConfig()) for _ in range(n_sims)]
    return FleetVecEnv(sims, sampler, seed=seed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timesteps", type=int, default=3_000_000, help="total per-vehicle transitions")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-sims", type=int, default=8, help="parallel simulations (x2 vehicles = SB3 slots)")
    ap.add_argument("--truth-params", default=TRUTH_PARAMS_PATH)
    ap.add_argument("--belief", default="mix",
                    help="'mix' (default: the documented mixture) or one fixed belief model for every episode: "
                         + ", ".join(FIXED_BELIEFS))
    ap.add_argument("--disturbance", default="none",
                    help="'none' (clean, default), 'fitted' (disturbances/bluerov2_rollpitch.yaml) or a fitted "
                         "disturbance YAML: random roll/pitch torques shaped like real unexplained motion")
    ap.add_argument("--disturbance-scale", type=float, default=1.0)
    ap.add_argument("--current-speed", type=float, nargs="+", metavar="MPS",
                    help="water current speed: one value, or MIN MAX to draw one per episode (default: still water)")
    ap.add_argument("--current-direction", type=float, nargs="+", default=[0.0, 360.0], metavar="DEG",
                    help="direction the water flows toward: one value, or MIN MAX (default 0 360: any direction)")
    ap.add_argument("--current-speed-std", type=float, default=0.0, help="slow speed fluctuation (m/s)")
    ap.add_argument("--current-direction-std", type=float, default=0.0, help="slow direction fluctuation (deg)")
    ap.add_argument("--current-time-scale", type=float, default=30.0, help="fluctuation time scale (s)")
    ap.add_argument("--current-unknown-to-beliefs", action="store_true",
                    help="physics teammate models don't know the current (default: they know its episode mean)")
    ap.add_argument("--spec", default=VEHICLE_SPEC_PATH, help="vehicle description (default: the BlueROV2 Heavy)")
    ap.add_argument("--thrusters", default=None,
                    help="limit commands to what the thrusters can deliver: bluerov2_heavy, or a YAML file with a "
                         "thrusters block (an ingest spec works). Default: no thruster limits")
    ap.add_argument("--out", required=True, help="output path without .zip")
    args = ap.parse_args()
    fleet_kwargs = {}
    if args.thrusters:
        from fleet_experiment_schema import resolve_thrusters
        fleet_kwargs["thrusters"] = resolve_thrusters(args.thrusters)
    if args.disturbance != "none":
        from disturbance import DEFAULT_FIT_PATH, load_disturbance
        dpath = DEFAULT_FIT_PATH if args.disturbance == "fitted" else args.disturbance
        fleet_kwargs["disturbance"] = load_disturbance(dpath, args.disturbance_scale,
                                                       truth_params_path=args.truth_params)
    if args.current_speed:
        from current import CurrentConfig
        pick = lambda xs, name: xs[0] if len(xs) == 1 else (xs[0], xs[1]) if len(xs) == 2 else ap.error(
            f"--{name} takes one value or MIN MAX")
        fleet_kwargs["current"] = CurrentConfig(
            speed_mps=pick(args.current_speed, "current-speed"),
            direction_deg=pick(args.current_direction, "current-direction"),
            speed_std_mps=args.current_speed_std, direction_std_deg=args.current_direction_std,
            time_scale_s=args.current_time_scale, known_to_beliefs=not args.current_unknown_to_beliefs)
        fleet_kwargs["current"].validate()
    fleet_config = FleetConfig(**fleet_kwargs) if fleet_kwargs else None

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import VecMonitor
    torch.set_num_threads(1)  # several trainings run side by side; don't oversubscribe cores

    env = VecMonitor(build_training_env(args.n_sims, args.seed, args.truth_params, fleet_config=fleet_config,
                                        belief=args.belief, spec_path=args.spec))
    hyper = dict(n_steps=256, batch_size=1024, n_epochs=10, learning_rate=3e-4, gamma=0.99,
                 gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
                 policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))
    model = PPO("MlpPolicy", env, seed=args.seed, verbose=1, device="cpu", **hyper)
    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    elapsed = time.time() - t0
    model.save(args.out)

    meta = {
        "policy": f"{args.out}.zip",
        "contract": {"observation": "Box(13,) float32", "action": "Box(6,) float32 in [-1, 1]", "per_vehicle": True},
        "algorithm": "ppo", "seed": args.seed, "timesteps": args.timesteps, "n_sims": args.n_sims,
        "hyperparameters": {k: v for k, v in hyper.items() if k != "policy_kwargs"} | {"net_arch": [128, 128]},
        "truth_params": args.truth_params, "vehicle_spec": args.spec,
        "thrusters": args.thrusters,
        "training_distribution": {"belief": args.belief, "severities": list(TRAIN_SEVERITIES),
                                  "beacon_intervals_s": list(TRAIN_BEACON_INTERVALS_S),
                                  "belief_mix": BELIEF_MIX if args.belief == "mix" else None,
                                  "initial_separation_range_m": list(FleetConfig().initial_separation_range),
                                  "initial_closing_speed_range_mps": list(FleetConfig().initial_closing_speed_range),
                                  "disturbance": (None if fleet_config is None or fleet_config.disturbance is None
                                                  else {"file": fleet_config.disturbance.source,
                                                        "scale": fleet_config.disturbance.scale}),
                                  "current": (None if fleet_config is None or fleet_config.current is None
                                              else fleet_config.current.describe())},
        "git_commit": get_git_commit(), "git_uncommitted_changes": get_git_dirty(),
        "python": platform.python_version(),
        "elapsed_s": round(elapsed, 1), "transitions_per_s": round(args.timesteps / elapsed, 1),
    }
    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"saved {args.out}.zip and {args.out}.json ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    with keep_awake("fleet policy training"):  # PROJECT_STATUS.md #49
        main()
