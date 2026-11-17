#!/usr/bin/env python3
"""
train_escort_policy.py -- train a BlueROV2 escort policy for the heterogeneous (BlueROV2 + REMUS) study.

    python train_escort_policy.py --seed 1 --out escort_policy_seed1                 # PPO, 3M steps, mixed models
    python train_escort_policy.py --algorithm td3 --seed 1 --out escort_td3_seed1    # 1M steps
    python train_escort_policy.py --belief calibrated --seed 1 --out escort_cal_seed1  # one fixed teammate model

Produces <out>.zip (a stock SB3 policy: observation Box(16,), action Box(4,)) and <out>.json (the training setup,
git commit, runtime). Hyperparameters are fleet v2's (train_fleet_policy.ALGORITHM_SETTINGS, NET_ARCH).

Training distribution, drawn fresh every episode:
  - a 60 s window of one of the training missions (escort_remus.TRAIN_SEEDS; never the evaluation missions), the ROV
    starting 2-6 m from the REMUS at a random bearing, moving with it
  - comm severity uniformly from L0, L1, L2, L3a and beacon interval from 0.5, 1, 2, 3 s
  - the teammate model (--belief mix, the default): a REMUS model a random fraction alpha ~ U(0, 1) of the way from
    the calibrated to the generic parameters (60%), last_beacon (15%), constant_velocity (10%), oracle (15%)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

import escort_env as EE
import escort_remus as ER
from fleet_belief import BeliefSpec, interpolate_params

TRAIN_SEVERITIES = ("L0", "L1", "L2", "L3a")
TRAIN_BEACON_INTERVALS_S = (0.5, 1.0, 2.0, 3.0)
BELIEF_MIX = {"physics_interpolated": 0.60, "last_beacon": 0.15, "constant_velocity": 0.10, "oracle": 0.15}
FIXED_BELIEFS = ("calibrated", "generic", "last_beacon", "constant_velocity", "oracle")
DEFAULT_TIMESTEPS = {"ppo": 3_000_000, "sac": 1_000_000, "td3": 1_000_000}
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "escort", "missions_cache")


def comm_conditions(severities=TRAIN_SEVERITIES, intervals=TRAIN_BEACON_INTERVALS_S) -> list:
    return [EE.CommCondition(s, b) for s in severities for b in intervals]


def belief_sampler(models: EE.RemusModels, kind: str = "mix"):
    if kind != "mix":
        if kind not in FIXED_BELIEFS:
            raise ValueError(f"--belief must be 'mix' or one of {FIXED_BELIEFS}, got {kind!r}")
        spec = models.belief(kind)
        return lambda rng: spec
    kinds = list(BELIEF_MIX)
    probs = np.array([BELIEF_MIX[k] for k in kinds])

    def sample(rng: np.random.Generator) -> BeliefSpec:
        k = kinds[rng.choice(len(kinds), p=probs)]
        if k == "physics_interpolated":
            alpha = float(rng.uniform(0.0, 1.0))
            return BeliefSpec("physics", f"physics_alpha{alpha:.2f}",
                              interpolate_params(models.calibrated, models.generic, alpha))
        return BeliefSpec(k, k)
    return sample


def make_model(algorithm: str, env, seed: int, verbose: int = 1):
    from stable_baselines3 import PPO, SAC, TD3
    from train_fleet_policy import ALGORITHM_SETTINGS, NET_ARCH
    hyper = dict(ALGORITHM_SETTINGS[algorithm])
    arch = NET_ARCH[algorithm]
    if algorithm == "ppo":
        return PPO("MlpPolicy", env, seed=seed, verbose=verbose, device="cpu",
                   policy_kwargs=dict(net_arch=dict(pi=arch, vf=arch)), **hyper)
    if algorithm == "sac":
        return SAC("MlpPolicy", env, seed=seed, verbose=verbose, device="cpu", policy_kwargs=dict(net_arch=arch), **hyper)
    from stable_baselines3.common.noise import NormalActionNoise
    std = hyper.pop("action_noise_std")
    return TD3("MlpPolicy", env, seed=seed, verbose=verbose, device="cpu", policy_kwargs=dict(net_arch=arch),
               action_noise=NormalActionNoise(np.zeros(EE.ACT_DIM), std * np.ones(EE.ACT_DIM)), **hyper)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algorithm", choices=["ppo", "sac", "td3"], default="ppo")
    ap.add_argument("--timesteps", type=int, default=None, help="default: 3M for ppo, 1M for sac/td3")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-envs", type=int, default=16, help="parallel environments in one process")
    ap.add_argument("--belief", default="mix", help="'mix' (default) or one of " + ", ".join(FIXED_BELIEFS))
    ap.add_argument("--missions", type=int, default=len(ER.TRAIN_SEEDS), help="training missions to draw from")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processes for building missions (once)")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="where precomputed missions are kept")
    ap.add_argument("--out", required=True, help="output path without .zip")
    args = ap.parse_args()
    timesteps = args.timesteps or DEFAULT_TIMESTEPS[args.algorithm]

    import torch
    from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    from escort_gym import EscortGymEnv
    torch.set_num_threads(1)

    lib = ER.MissionLibrary.build(list(ER.TRAIN_SEEDS)[:args.missions], cache_dir=args.cache, workers=args.workers)
    rov, remus, cfg = EE.load_rov(), EE.load_remus_models(), EE.EscortConfig()
    sampler = belief_sampler(remus, args.belief)
    comms = comm_conditions()
    seeds = np.random.SeedSequence(args.seed).generate_state(args.n_envs)
    env = VecMonitor(DummyVecEnv([
        (lambda s=int(s): EscortGymEnv(EE.EscortSim(rov, remus, lib, cfg), sampler, comms, seed=s)) for s in seeds]))
    model = make_model(args.algorithm, env, args.seed)
    t0 = time.time()
    model.learn(total_timesteps=timesteps, log_interval=(1 if args.algorithm == "ppo" else 50))
    elapsed = time.time() - t0
    model.save(args.out)

    from mlflow_run_utils import get_git_commit, get_git_dirty
    from train_fleet_policy import ALGORITHM_SETTINGS, NET_ARCH
    meta = {
        "policy": f"{args.out}.zip", "study": "escort (BlueROV2 escorting a REMUS 100)",
        "contract": {"observation": f"Box({EE.OBS_DIM},) float32", "action": f"Box({EE.ACT_DIM},) float32 in [-1, 1]"},
        "algorithm": args.algorithm, "seed": args.seed, "timesteps": timesteps, "n_envs": args.n_envs,
        "hyperparameters": ALGORITHM_SETTINGS[args.algorithm] | {"net_arch": NET_ARCH[args.algorithm]},
        "rov": {"params": EE.ROV_PARAMS, "spec": EE.ROV_SPEC, "thrusters": "bluerov2_heavy"},
        "remus": {"truth": "stress/prestero_remus.py, re-ballasted (escort_remus.py)", "spec": EE.REMUS_SPEC,
                  "calibrated": EE.REMUS_CALIBRATED, "buoyancy_change_n": ER.BUOYANCY_CHANGE_N},
        "missions": {"config": lib.cfg.__dict__, "seeds": [min(ER.TRAIN_SEEDS), min(ER.TRAIN_SEEDS) + args.missions - 1]},
        "task": cfg.__dict__,
        "training_distribution": {"belief": args.belief, "belief_mix": BELIEF_MIX if args.belief == "mix" else None,
                                  "severities": list(TRAIN_SEVERITIES), "beacon_intervals_s": list(TRAIN_BEACON_INTERVALS_S)},
        "git_commit": get_git_commit(), "git_uncommitted_changes": get_git_dirty(), "python": platform.python_version(),
        "elapsed_s": round(elapsed, 1), "transitions_per_s": round(timesteps / elapsed, 1),
    }
    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"saved {args.out}.zip and {args.out}.json ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    from keep_awake import keep_awake
    with keep_awake("escort policy training"):
        main()
