"""
Shared evaluation harness for Arena — used by Stage 2.2's sanity check
(does PPO learn anything at all) and Stage 2.4's real default-vs-calibrated
comparison. Built and tested independent of stable-baselines3/PPO, using a
simple hand-coded controller as a genuine POSITIVE CONTROL: a policy that
should reliably do better than random, so the evaluation harness itself
can be confirmed correct before any real learning algorithm touches it.

## Success tracking (revised this session)
Previously used `if terminated: successes += 1` — this was WRONG for any
task where `terminated` can mean something other than success (e.g. a
collision in obstacle_avoidance). Now uses
`info.get("terminal_reason") == TERMINAL_REASON_GOAL` instead, which
every task (the three original ones, retrofitted, and the new
obstacle_avoidance task) sets correctly and explicitly. Confirmed
backward-compatible: the three original tasks set this on the exact same
condition that used to set `terminated=True` for a success, so
success_rate numbers are unchanged for them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from arena import Arena, TERMINAL_REASON_GOAL, TERMINAL_REASON_COLLISION


@dataclass
class EvalResult:
    mean_reward: float
    mean_final_distance: float
    success_rate: float  # fraction of episodes with terminal_reason == GOAL
    collision_rate: float  # fraction of episodes with terminal_reason == COLLISION
        # (always 0.0 for tasks that never produce a collision - not a
        # special case, just naturally zero since the key never appears)
    timeout_rate: float  # fraction of episodes that ended via truncation
    mean_episode_length: float  # ALL episodes, regardless of outcome -
        # a genuinely useful debugging signal on its own (e.g. "collisions
        # happen after 15 steps, timeouts after 400, successes after 120")
    mean_time_to_goal: Optional[float]  # mean episode length among
        # SUCCESSFUL episodes only. None if zero episodes succeeded.
    mean_min_obstacle_clearance: Optional[float]  # mean, across episodes,
        # of each episode's closest approach to an obstacle. None if no
        # episode ever reported "obstacle_clearance" in info (i.e. the
        # task doesn't have obstacles at all) - detected dynamically, not
        # via a hardcoded task-name check.
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
    episode_lengths = []
    time_to_goal_values = []
    min_clearance_values = []
    successes = 0
    collisions = 0
    timeouts = 0

    rng = np.random.default_rng(seed)

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        episode_reward = 0.0
        terminated = truncated = False
        distance = info.get("distance", None)
        step_count = 0
        episode_min_clearance = float("inf")

        while not (terminated or truncated):
            if policy_fn is None:
                action = rng.uniform(env.action_space.low, env.action_space.high)
            else:
                action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            distance = info["distance"]
            step_count += 1
            if "obstacle_clearance" in info:
                episode_min_clearance = min(episode_min_clearance, info["obstacle_clearance"])

        rewards.append(episode_reward)
        final_distances.append(distance)
        episode_lengths.append(step_count)

        terminal_reason = info.get("terminal_reason")
        if terminal_reason == TERMINAL_REASON_GOAL:
            successes += 1
            time_to_goal_values.append(step_count)
        elif terminal_reason == TERMINAL_REASON_COLLISION:
            collisions += 1
        if truncated:
            timeouts += 1

        if episode_min_clearance != float("inf"):
            min_clearance_values.append(episode_min_clearance)

    return EvalResult(
        mean_reward=float(np.mean(rewards)),
        mean_final_distance=float(np.mean(final_distances)),
        success_rate=successes / n_episodes,
        collision_rate=collisions / n_episodes,
        timeout_rate=timeouts / n_episodes,
        mean_episode_length=float(np.mean(episode_lengths)),
        mean_time_to_goal=float(np.mean(time_to_goal_values)) if time_to_goal_values else None,
        mean_min_obstacle_clearance=float(np.mean(min_clearance_values)) if min_clearance_values else None,
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