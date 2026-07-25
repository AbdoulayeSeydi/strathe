"""
Shared evaluation harness for Arena — used by Stage 2.2's sanity check
(does PPO learn anything at all) and Stage 2.4's real default-vs-calibrated
comparison. Built and tested independent of stable-baselines3/PPO, using a
simple hand-coded controller as a genuine POSITIVE CONTROL: a policy that
should reliably do better than random, so the evaluation harness itself
can be confirmed correct before any real learning algorithm touches it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from arena import Arena


@dataclass
class EvalResult:
    mean_reward: float
    mean_final_distance: float
    success_rate: float  # fraction of episodes that reached the waypoint (terminated, not truncated)
    n_episodes: int


def evaluate_policy(env: Arena, policy_fn: Optional[Callable[[np.ndarray], np.ndarray]],
                     n_episodes: int = 20, seed: int = 0) -> EvalResult:
    """
    Runs n_episodes and reports aggregate performance. policy_fn maps
    obs -> action; if None, samples RANDOM actions from the env's action
    space (the baseline every real policy should beat).
    """
    rewards = []
    final_distances = []
    successes = 0

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        episode_reward = 0.0
        terminated = truncated = False
        distance = info.get("distance", None)

        while not (terminated or truncated):
            if policy_fn is None:
                action = rng.uniform(env.action_space.low, env.action_space.high)
            else:
                action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            distance = info["distance"]

        rewards.append(episode_reward)
        final_distances.append(distance)
        if terminated:
            successes += 1

    return EvalResult(
        mean_reward=float(np.mean(rewards)),
        mean_final_distance=float(np.mean(final_distances)),
        success_rate=successes / n_episodes,
        n_episodes=n_episodes,
    )


def proportional_controller(obs: np.ndarray, gain: float = 0.3) -> np.ndarray:
    """
    A simple, hand-coded, NON-learned policy: pushes toward the waypoint
    (proportional to the body-frame position error), zero torque. NOT
    meant to be a good controller — a genuine POSITIVE CONTROL for testing
    evaluate_policy() itself.

    Returns a NORMALIZED action in [-1,1] (matching Arena's action_space
    convention — see arena.py's module docstring on why raw force-range
    actions are the wrong convention for PPO), not a raw force value.
    """
    pos_error_body = obs[6:9]
    force = np.clip(gain * pos_error_body, -1.0, 1.0)
    return np.concatenate([force, np.zeros(3)]).astype(np.float32)