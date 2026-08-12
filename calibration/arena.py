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

## Task architecture (refactored this session)
Each task's reward/termination logic lives in its own Task subclass
rather than inline if/elif branches in Arena.step() — the previous
structure worked for 3 tasks but was flagged as not scaling to more.
This refactor was a PURE migration for the three original tasks
(waypoint/station_keeping/docking) — zero behavior change, verified via
a 6000-trial randomized logic-equivalence test AND all 26 real
regression tests AND a live end-to-end product run before anything new
was added on top of it.

Task.reset() ALSO returns (spawn_override, waypoint_override) — added
after discovering ArenaCore.reset() already supports an explicit
starting position (a real capability that wasn't being used anywhere).
The three original tasks return (None, None) unconditionally, which
ArenaCore.reset(position=None) and Arena's own existing
fixed/randomized-waypoint fallback logic both already treat identically
to "no override given" — re-verified to be exactly byte-equivalent to
the pre-change behavior before this was trusted.

Observation space is deliberately left unchanged (11,) for the three
original tasks — a new task with a different observation contract is
free to differ (Arena already sets observation_space as a per-instance
attribute, not a shared class constant), but changing an EXISTING task's
observation shape would break real, tested downstream assumptions for no
benefit — an explicit compatibility decision, not an oversight.
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


# ---------------------------------------------------------------------------
# Task base class and the three original tasks, migrated unchanged
# ---------------------------------------------------------------------------

class Task:
    """Base class - each subclass encapsulates one task's reward,
    termination, and episode-generation logic.

    reset(arena) -> (spawn_override, waypoint_override): called from
    Arena.reset() BEFORE the core physics reset happens, so a task can
    control the episode's starting layout (needed for tasks like
    obstacle avoidance, where spawn/goal/obstacle must be generated
    together as one mutually-validated set). Either return value may be
    None to preserve Arena's own existing default behavior for that
    piece (zero spawn position; Arena's own fixed/randomized waypoint
    logic). The three original tasks return (None, None) unconditionally.

    step() computes reward and terminated from ONE consistent transition
    (never recomputed independently elsewhere), matching Arena.step()'s
    own real (reward, terminated, truncated, info) shape - truncated is
    task-independent (pure step-count), so Arena computes it once itself
    and passes it in, rather than asking every task to redundantly
    recompute the same thing."""

    def reset(self, arena: "Arena") -> tuple:
        return None, None

    def step(self, arena: "Arena", distance: float, truncated: bool) -> tuple:
        """Returns (reward, terminated, info) - info here is ONLY this
        task's own additions; Arena.step() merges in the common
        distance/position fields itself before returning to the caller."""
        raise NotImplementedError


class WaypointTask(Task):
    """Original, unchanged behavior: touch the target once, episode ends."""

    def step(self, arena, distance, truncated):
        reward = -distance
        terminated = distance < arena.waypoint_radius
        if terminated:
            reward += 10.0
        return reward, terminated, {}


class StationKeepingTask(Task):
    """Must STAY within waypoint_radius for hold_steps CONSECUTIVE steps
    before counting as success - a vehicle drifting through and
    continuing past should NOT succeed, the whole point of this task."""

    def __init__(self, hold_steps: int = 20):
        self.hold_steps = hold_steps
        self._consecutive_steps_in_radius = 0

    def reset(self, arena):
        self._consecutive_steps_in_radius = 0
        return None, None

    def step(self, arena, distance, truncated):
        reward = -distance
        if distance < arena.waypoint_radius:
            self._consecutive_steps_in_radius += 1
            # Extra shaping while close: penalize speed too, so the
            # policy learns to actually STOP near the target rather
            # than orbit through the radius repeatedly.
            speed = float(np.linalg.norm(arena._core.state.nu[:3]))
            reward -= 0.1 * speed
        else:
            self._consecutive_steps_in_radius = 0
        terminated = self._consecutive_steps_in_radius >= self.hold_steps
        if terminated:
            reward += 10.0
        return reward, terminated, {"consecutive_steps_in_radius": self._consecutive_steps_in_radius}


class DockingTask(Task):
    """Requires position, orientation alignment (roll/pitch within
    orientation_tol_rad), AND low speed (below max_speed) simultaneously
    - matching what a real dock actually requires, not just touching the
    target point at any angle/speed."""

    def __init__(self, orientation_tol_rad: float = 0.2, max_speed: float = 0.3):
        self.orientation_tol_rad = orientation_tol_rad
        self.max_speed = max_speed

    def step(self, arena, distance, truncated):
        reward = -distance
        roll, pitch, _yaw = arena._core.state.orientation.as_euler("xyz")
        orientation_error = float(np.sqrt(roll ** 2 + pitch ** 2))
        speed = float(np.linalg.norm(arena._core.state.nu[:3]))
        reward -= 0.5 * orientation_error  # shape toward alignment throughout, not just at the end
        terminated = (
            distance < arena.waypoint_radius
            and orientation_error < self.orientation_tol_rad
            and speed < self.max_speed
        )
        if terminated:
            reward += 10.0
        return reward, terminated, {"orientation_error": orientation_error, "speed": speed}


def _make_task(task: str, hold_steps: int, docking_orientation_tol_rad: float,
               docking_max_speed: float) -> Task:
    if task == "waypoint":
        return WaypointTask()
    if task == "station_keeping":
        return StationKeepingTask(hold_steps=hold_steps)
    if task == "docking":
        return DockingTask(orientation_tol_rad=docking_orientation_tol_rad,
                            max_speed=docking_max_speed)
    raise ValueError(f"Unknown task {task!r} - expected 'waypoint', 'station_keeping', or 'docking'")


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
        task: str = "waypoint",
        hold_steps: int = 20,
        docking_orientation_tol_rad: float = 0.2,
        docking_max_speed: float = 0.3,
    ):
        """
        task: "waypoint" (default, ORIGINAL behavior, unchanged) - touch
            the target once, episode ends. "station_keeping" - must STAY
            within waypoint_radius for hold_steps CONSECUTIVE steps before
            counting as success (a vehicle drifting through and continuing
            past should NOT succeed - the whole point of this task).
            "docking" - requires position, orientation alignment
            (roll/pitch within docking_orientation_tol_rad), AND low speed
            (below docking_max_speed) simultaneously, matching what a real
            dock actually requires - not just touching the target point at
            any angle/speed.
        """
        super().__init__()
        sim_config = SimulationConfig(dt=dt, physics_dt=physics_dt)
        self._core = ArenaCore(params, vehicle, sim_config)
        self.max_episode_steps = max_episode_steps
        self.waypoint_radius = waypoint_radius
        self._fixed_waypoint = None if fixed_waypoint is None else np.array(fixed_waypoint, dtype=float)
        self.waypoint_range = waypoint_range
        self._task = task
        self._task_obj = _make_task(task, hold_steps, docking_orientation_tol_rad, docking_max_speed)

        # Action space NORMALIZED to [-1,1] — NOT the raw force range. SB3's
        # default PPO policy initializes its action distribution centered
        # near 0 with std~1, so a wide raw range (e.g. +-30) would need the
        # policy to learn a large output-mean shift before acting
        # meaningfully at all — a well-known, textbook PPO/continuous-
        # control gotcha. Confirmed as the real cause here, not just
        # theorized: the first real training attempt (unnormalized, +-30
        # directly) showed the policy's action std barely moving from its
        # 1.0 initialization across 50k steps, explained_variance stuck near
        # zero, and final performance statistically indistinguishable from
        # random. Rescaling happens internally in step().
        self._action_bound = action_bound
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        # obs = [u,v,w,p,q,r] (6) + position error in body frame (3) + roll,pitch (2) = 11
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )

        self._waypoint = np.zeros(3)
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        spawn_override, waypoint_override = self._task_obj.reset(self)
        self._core.reset(position=spawn_override)
        if waypoint_override is not None:
            self._waypoint = np.array(waypoint_override, dtype=float)
        elif self._fixed_waypoint is not None:
            self._waypoint = self._fixed_waypoint.copy()
        else:
            self._waypoint = self.np_random.uniform(-self.waypoint_range, self.waypoint_range, size=3)
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32),
                          self.action_space.low, self.action_space.high)
        tau = action * self._action_bound  # rescale [-1,1] -> real force/torque units
        self._core.step(tau)
        self._step_count += 1

        distance = float(np.linalg.norm(self._waypoint - self._core.state.position))
        info = {"distance": distance, "position": self._core.state.position.copy()}
        truncated = self._step_count >= self.max_episode_steps

        reward, terminated, task_info = self._task_obj.step(self, distance, truncated)
        info.update(task_info)

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        pos_error_body = self._core.position_error_body_frame(self._waypoint)
        roll, pitch, _yaw = self._core.state.orientation.as_euler("xyz")
        obs = np.concatenate([self._core.state.nu, pos_error_body, [roll, pitch]])
        return obs.astype(np.float32)