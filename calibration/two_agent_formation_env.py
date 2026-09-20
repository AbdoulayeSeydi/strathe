from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym

from two_agent_orchestrator import TwoAgentOrchestrator, OrchestratorConfig
from comm_channel import PacketLossConfig, BandwidthConfig
from formation_hold_task import FormationHoldConfig

DEFAULT_ACTION_BOUND = 30.0


class TwoAgentFormationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self,
                 calibrated_params_path: str = "calibrated_parameters_a1.yaml",
                 packet_loss_config: Optional[PacketLossConfig] = None,
                 bandwidth_config: Optional[BandwidthConfig] = None,
                 task_config: Optional[FormationHoldConfig] = None,
                 orchestrator_config: Optional[OrchestratorConfig] = None,
                 action_bound: float = DEFAULT_ACTION_BOUND,
                 comm_seed: Optional[int] = None,
                 randomize_initial_state: bool = False,
                 initial_separation_range: tuple = (2.0, 5.5),
                 initial_velocity_range: tuple = (-0.7, 0.7),
                 curriculum_lookup=None,
                 anchor_probability: float = 0.0,
                 anchor_separation: Optional[float] = None,
                 anchor_velocity: float = 0.0,
                 dynamics_backend: str = "stand_in",
                 speed_of_sound_mps: float = 1500.0):
        """
        dynamics_backend: C0.3 addition (see conversation). Passed straight
        through to TwoAgentOrchestrator -- "stand_in" (default, exact prior
        behavior) or "mujoco" (genuine MuJoCo dynamics for both agents).
        Everything else about this env -- policy, observation, reward,
        action scaling, communication model, task/termination logic -- is
        completely unaffected by this parameter.
        """
        super().__init__()
        self._orchestrator = TwoAgentOrchestrator(
            calibrated_params_path=calibrated_params_path,
            packet_loss_config=packet_loss_config,
            bandwidth_config=bandwidth_config,
            task_config=task_config,
            orchestrator_config=orchestrator_config,
            comm_seed=comm_seed,
            dynamics_backend=dynamics_backend,
            speed_of_sound_mps=speed_of_sound_mps,  # F2.2
        )
        self._action_bound = action_bound
        self._randomize_initial_state = randomize_initial_state
        self._initial_separation_range = initial_separation_range
        self._initial_velocity_range = initial_velocity_range
        self._curriculum_lookup = curriculum_lookup
        self.curriculum_max_difficulty = float("inf")
        self.anchor_probability = anchor_probability
        self._anchor_separation = (anchor_separation if anchor_separation is not None
                                    else self._orchestrator.config.initial_separation)
        self._anchor_velocity = anchor_velocity

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32
        )

    def _sample_initial_state(self):
        if self.anchor_probability > 0 and self.np_random.uniform() < self.anchor_probability:
            return self._anchor_separation, self._anchor_velocity

        if self._curriculum_lookup is None:
            d0 = float(self.np_random.uniform(*self._initial_separation_range))
            v0 = float(self.np_random.uniform(*self._initial_velocity_range))
            return d0, v0

        max_attempts = 50
        for _ in range(max_attempts):
            d0 = float(self.np_random.uniform(*self._initial_separation_range))
            v0 = float(self.np_random.uniform(*self._initial_velocity_range))
            if self._curriculum_lookup.difficulty(d0, v0) <= self.curriculum_max_difficulty:
                return d0, v0
        return self._curriculum_lookup.easiest_point

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._randomize_initial_state:
            d0, v0 = self._sample_initial_state()
            extra_generic, extra_calibrated = self._orchestrator.reset(
                initial_separation=d0, initial_calibrated_surge_velocity=v0)
        else:
            extra_generic, extra_calibrated = self._orchestrator.reset()
        obs = self._build_full_obs(extra_generic, extra_calibrated)
        return obs, {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32),
                          self.action_space.low, self.action_space.high)
        action_generic_norm = action[:6]
        action_calibrated_norm = action[6:]

        tau_generic = action_generic_norm * self._action_bound
        tau_calibrated = action_calibrated_norm * self._action_bound

        (extra_generic, extra_calibrated), reward, terminated, truncated, info = (
            self._orchestrator.step(
                tau_generic, tau_calibrated,
                action_generic_for_reward=action_generic_norm,
                action_calibrated_for_reward=action_calibrated_norm,
            )
        )

        obs = self._build_full_obs(extra_generic, extra_calibrated)
        return obs, reward, terminated, truncated, info

    def _build_full_obs(self, extra_obs_generic: np.ndarray, extra_obs_calibrated: np.ndarray) -> np.ndarray:
        core_generic = self._orchestrator.core_generic
        core_calibrated = self._orchestrator.core_calibrated

        roll_g, pitch_g, _yaw_g = core_generic.state.orientation.as_euler("xyz")
        roll_c, pitch_c, _yaw_c = core_calibrated.state.orientation.as_euler("xyz")

        obs_generic = np.concatenate([core_generic.state.nu, [roll_g, pitch_g], extra_obs_generic])
        obs_calibrated = np.concatenate([core_calibrated.state.nu, [roll_c, pitch_c], extra_obs_calibrated])

        return np.concatenate([obs_generic, obs_calibrated]).astype(np.float32)
