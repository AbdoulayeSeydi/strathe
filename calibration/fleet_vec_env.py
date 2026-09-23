"""
fleet_vec_env.py -- trains ONE shared per-vehicle policy on FleetSim with
stock stable-baselines3 algorithms.

Each vehicle in each simulation is presented to SB3 as its own
environment slot (parameter sharing: the same network acts for every
vehicle, from that vehicle's own 13-float observation). Both vehicles in a
simulation receive the shared cooperative reward. So n_sims simulations
appear to SB3 as a VecEnv with n_sims * 2 slots, observation Box(13,) and
action Box(6,) -- a plain single-agent PPO/SAC/TD3 policy, which is also
the lab-facing policy contract.

Every reset draws a fresh episode from an EpisodeSampler, so one policy
is trained across comm severities and belief conditions instead of one
policy per condition (fleet v1's four separately-trained policies were a
confound in its own results).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from comm_channel import BandwidthConfig, PacketLossConfig
from fleet_belief import BeliefSpec
from fleet_env import ACT_DIM, N_AGENTS, OBS_DIM, FleetSim

OBSERVATION_SPACE = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
ACTION_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(ACT_DIM,), dtype=np.float32)


@dataclass
class CommCondition:
    label: str
    packet_loss_config: PacketLossConfig
    bandwidth_config: BandwidthConfig
    speed_of_sound_mps: float = 1500.0
    beacon_interval_s: Optional[float] = None  # None = keep the sim's configured interval


@dataclass
class EpisodeSampler:
    """What varies between training episodes. belief_sampler returns a
    BeliefSpec given the sim's RNG, so continuous conditions (e.g. a
    randomly interpolated model) are possible."""
    comm_conditions: Sequence[CommCondition]
    belief_sampler: Callable[[np.random.Generator], BeliefSpec]
    comm_weights: Optional[Sequence[float]] = None

    def sample_comm(self, rng: np.random.Generator) -> CommCondition:
        idx = rng.choice(len(self.comm_conditions), p=self.comm_weights)
        return self.comm_conditions[idx]


class FleetVecEnv(VecEnv):
    def __init__(self, sims: Sequence[FleetSim], sampler: EpisodeSampler, seed: int = 0):
        self.sims = list(sims)
        self.sampler = sampler
        self._rngs = [np.random.default_rng([seed, k]) for k in range(len(self.sims))]
        self._actions = None
        super().__init__(len(self.sims) * N_AGENTS, OBSERVATION_SPACE, ACTION_SPACE)

    def _reset_sim(self, k: int) -> list:
        rng = self._rngs[k]
        sim = self.sims[k]
        comm = self.sampler.sample_comm(rng)
        sim.set_comm(comm.packet_loss_config, comm.bandwidth_config, comm.speed_of_sound_mps,
                     comm.beacon_interval_s)
        sim.set_belief(self.sampler.belief_sampler(rng))
        return sim.reset(comm_seed=int(rng.integers(2**31)), rng=rng)

    def reset(self):
        obs = []
        for k in range(len(self.sims)):
            obs.extend(self._reset_sim(k))
        return np.stack(obs)

    def step_async(self, actions):
        self._actions = np.asarray(actions)

    def step_wait(self):
        obs, rews, dones, infos = [], [], [], []
        for k, sim in enumerate(self.sims):
            acts = [self._actions[k * N_AGENTS + i] for i in range(N_AGENTS)]
            o, r, term, trunc, info = sim.step(acts)
            done = term or trunc
            if done:
                terminal = o
                o = self._reset_sim(k)
            for i in range(N_AGENTS):
                slot_info = {"TimeLimit.truncated": bool(trunc and not term)}
                if done:
                    slot_info["terminal_observation"] = terminal[i]
                    slot_info["terminal_reason"] = info["terminal_reason"]
                obs.append(o[i])
                rews.append(r)
                dones.append(done)
                infos.append(slot_info)
        return np.stack(obs), np.array(rews, dtype=np.float32), np.array(dones), infos

    def close(self):
        pass

    # --- VecEnv plumbing SB3 requires; sims are not gym.Envs ------------------
    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None) for _ in self._indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("FleetVecEnv slots are not individual gym.Envs")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._indices(indices)]

    def _indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices
