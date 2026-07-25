"""
Arena — the Gym environment for Phase 2 RL training (Stage 2.1).

Wraps arena_core.ArenaCore (tested physics, see that module) in
gymnasium.Env's standard interface. Per phase2_config.py's Decision 2,
this uses the FAST stand-in physics (ArenaCore -> compute_acceleration)
for training — this class IS the TRAINING_BACKEND. Validation against
real HoloOcean physics happens SEPARATELY (Stage 2.3), not through this
same class, since HoloOcean needs a persistent env object and its own
integration, not this class's self-contained one.

Task: waypoint-following. The waypoint is either FIXED (pass one in,
e.g. derived from a real held-out trajectory's endpoint — see Stage 2.3's
grounding requirement) or randomized within a plausible range each
episode (useful for training itself, less so for the final real-data
comparison, which should use fixed, data-derived waypoints).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym

from arena_core import ArenaCore
from dynamics_model import CalibrationParameters, VehicleConstants, SimulationConfig

# Placeholder — TODO before real training: replace with the ACTUAL observed
# min/max force magnitude range from the real BlueROV2 dataset (Fx..Tz
# columns), so the agent can't apply nonphysical forces the real vehicle
# never produced. 30.0 is a rough guess based on forces seen during this
# session's diagnostic work, not a verified bound.
DEFAULT_ACTION_BOUND = 30.0


class Arena(gym.Env):
    """Gymnasium environment: waypoint-following with our calibratable
    6-DOF physics (see module docstring for the training/validation
    backend split this class deliberately does NOT try to unify)."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: CalibrationParameters,
        vehicle: VehicleConstants,
        dt: float = 0.1,
        physics_dt: float = 0.02,
        max_episode_steps: int = 200,
        waypoint_radius: float = 0.5,
        fixed_waypoint: Optional[np.ndarray] = None,
        waypoint_range: float = 3.0,
        action_bound: float = DEFAULT_ACTION_BOUND,
    ):
        super().__init__()
        sim_config = SimulationConfig(dt=dt, physics_dt=physics_dt)
        self._core = ArenaCore(params, vehicle, sim_config)
        self.max_episode_steps = max_episode_steps
        self.waypoint_radius = waypoint_radius
        self._fixed_waypoint = None if fixed_waypoint is None else np.array(fixed_waypoint, dtype=float)
        self.waypoint_range = waypoint_range

        self.action_space = gym.spaces.Box(
            low=-action_bound, high=action_bound, shape=(6,), dtype=np.float32
        )
        # obs = [u,v,w,p,q,r] (6) + position error in body frame (3) + roll,pitch (2) = 11
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        self._waypoint = np.zeros(3)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._core.reset()
        self._step_count = 0
        if self._fixed_waypoint is not None:
            self._waypoint = self._fixed_waypoint.copy()
        else:
            self._waypoint = self.np_random.uniform(-self.waypoint_range, self.waypoint_range, size=3)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32),
                          self.action_space.low, self.action_space.high)
        self._core.step(action)
        self._step_count += 1

        distance = float(np.linalg.norm(self._waypoint - self._core.state.position))
        reward = -distance
        terminated = distance < self.waypoint_radius
        truncated = self._step_count >= self.max_episode_steps
        if terminated:
            reward += 10.0

        info = {"distance": distance, "position": self._core.state.position.copy()}
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        pos_error_body = self._core.position_error_body_frame(self._waypoint)
        roll, pitch, _yaw = self._core.state.orientation.as_euler("xyz")
        obs = np.concatenate([self._core.state.nu, pos_error_body, [roll, pitch]])
        return obs.astype(np.float32)