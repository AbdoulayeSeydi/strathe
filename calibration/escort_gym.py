"""
escort_gym.py -- the escort as a gymnasium environment for stock stable-baselines3 training.

Every reset draws a fresh episode (mission window, ROV start, comm seed), a comm condition and a teammate model, so
one policy is trained across conditions and evaluation can compare models without favouring the one it was trained
on (as fleet v2's mixed policies are). The observation doesn't say which model is active.
"""
from __future__ import annotations

from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np

from escort_env import ACT_DIM, OBS_DIM, CommCondition, EscortSim, sample_episode
from fleet_belief import BeliefSpec

OBSERVATION_SPACE = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)


class EscortGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sim, belief_sampler: Callable[[np.random.Generator], BeliefSpec],
                 comm_conditions: Sequence[CommCondition], seed: int = 0):
        """sim: one EscortSim, or a list of them (e.g. one per water current and sensor setting) to draw from
        uniformly at every reset (domain randomisation)."""
        self.sims = list(sim) if isinstance(sim, (list, tuple)) else [sim]
        self.sim = self.sims[0]
        self.belief_sampler = belief_sampler
        self.comm_conditions = list(comm_conditions)
        self.observation_space = OBSERVATION_SPACE
        self.action_space = ACTION_SPACE
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if len(self.sims) > 1:
            self.sim = self.sims[int(self._rng.integers(len(self.sims)))]
        comm = self.comm_conditions[int(self._rng.integers(len(self.comm_conditions)))]
        self.sim.set_comm(comm)
        self.sim.set_belief(self.belief_sampler(self._rng))
        ep = sample_episode(self._rng, self.sim.library, self.sim.cfg)
        obs = self.sim.reset(ep)
        return obs, {"comm": comm.label, "belief": self.sim.belief_spec.label}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.sim.step(action)
        if terminated or truncated:
            info = dict(info, episode_summary=self.sim.summary())
        return obs, reward, terminated, truncated, info
