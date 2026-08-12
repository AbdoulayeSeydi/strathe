"""
Arena — the Gym environment for Phase 2 RL training (Stage 2.1).

Wraps arena_core.ArenaCore (tested physics, see that module) in
gymnasium.Env's standard interface. Per phase2_config.py's Decision 2,
this uses the FAST stand-in physics (ArenaCore -> compute_acceleration)
for training — this class IS the TRAINING_BACKEND. Validation against
real HoloOcean physics happens SEPARATELY (Stage 2.3), not through this
same class, since HoloOcean needs a persistent env object and its own
integration, not this class's self-contained one.

## Task architecture
Each task's reward, termination, episode-generation, and (optionally)
extra observation components live in its own Task subclass rather than
inline if/elif branches in Arena — refactored this session, verified via
randomized logic-equivalence tests, the full real test suite, and live
end-to-end product runs at each step, given how much downstream code
depends on this file's exact behavior.

Task.reset(arena) -> (spawn_override, waypoint_override): either may be
None to preserve Arena's own existing default (zero spawn; Arena's own
fixed/randomized waypoint logic). The three original tasks
(waypoint/station_keeping/docking) return (None, None) unconditionally -
re-verified byte-equivalent to their pre-refactor behavior.

Task.step(arena, distance, truncated) -> (reward, terminated, info):
computed from ONE consistent transition. truncated is task-independent
(pure step-count) so Arena computes it once itself.

Task.extra_obs_dim()/extra_obs(arena): optional extension point for a
task whose observation needs more than Arena's base 11-dim
[nu(6), pos_error_body(3), roll, pitch]. Defaults to 0/empty for the
three original tasks, which keeps their observation_space at exactly
(11,) - unchanged, not just "usually" - the concatenation is skipped
entirely when extra_obs_dim() is 0, not merely zero-valued.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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

# Centralized terminal-reason constants - plain strings (not an Enum),
# a deliberate choice matching this project's existing JSON-heavy
# reporting throughout (strings serialize trivially; Enums need extra
# handling). Defined once here so a typo like "Goal" or "collison"
# can't silently break evaluation.
TERMINAL_REASON_GOAL = "goal"
TERMINAL_REASON_COLLISION = "collision"


# ---------------------------------------------------------------------------
# Task base class and the three original tasks, migrated unchanged
# ---------------------------------------------------------------------------

class Task:
    """Base class - see module docstring for the full contract."""

    def reset(self, arena: "Arena") -> tuple:
        return None, None

    def step(self, arena: "Arena", distance: float, truncated: bool) -> tuple:
        """Returns (reward, terminated, info) - info here is ONLY this
        task's own additions; Arena.step() merges in the common
        distance/position fields itself before returning to the caller."""
        raise NotImplementedError

    def extra_obs_dim(self) -> int:
        return 0

    def extra_obs(self, arena: "Arena") -> np.ndarray:
        return np.zeros(0, dtype=np.float32)


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
        reward -= 0.5 * orientation_error
        terminated = (
            distance < arena.waypoint_radius
            and orientation_error < self.orientation_tol_rad
            and speed < self.max_speed
        )
        if terminated:
            reward += 10.0
        return reward, terminated, {"orientation_error": orientation_error, "speed": speed}


# ---------------------------------------------------------------------------
# ObstacleAvoidanceTask - new this session
# ---------------------------------------------------------------------------

@dataclass
class EpisodeLayout:
    spawn: np.ndarray
    goal: np.ndarray
    obstacle_position: np.ndarray
    obstacle_radius: float


@dataclass
class ObstacleAvoidanceConfig:
    """Grouped, structured task configuration - see PROJECT_STATUS.md's
    frozen design doc for the reward equation and termination table this
    implements exactly.

    R = w_p*delta_d + w_g*1[goal] - w_c*1[collision] - w_u*||u||^2 - w_o*f(d_obstacle)
    where f(d) = max(0, (influence_radius - d) / influence_radius) when
    d < influence_radius, else 0.
    """
    world_range: float = 3.0
    obstacle_radius_min: float = 0.3
    obstacle_radius_max: float = 0.8
    influence_radius: float = 1.5
    min_spawn_goal_distance: float = 2.0
    difficulty: float = 1.0  # scales obstacle radius range; 1.0 = fixed
        # baseline. Reserved for future curriculum use - zero cost if
        # left untouched.

    reward_progress_weight: float = 1.0
    reward_goal_bonus: float = 10.0
    reward_collision_penalty: float = 10.0
    reward_control_weight: float = 0.01
    reward_obstacle_weight: float = 1.0

    def generate_episode(self, rng: np.random.Generator) -> EpisodeLayout:
        """Owns the regenerate-until-valid loop internally - the task
        itself never knows or cares how validity was achieved, only that
        it receives a guaranteed-valid layout. Constraints: spawn outside
        obstacle, goal outside obstacle, spawn-goal distance above
        min_spawn_goal_distance, obstacle doesn't cover spawn or goal."""
        radius_min = self.obstacle_radius_min * self.difficulty
        radius_max = self.obstacle_radius_max * self.difficulty
        max_attempts = 1000

        for _ in range(max_attempts):
            spawn = rng.uniform(-self.world_range, self.world_range, size=3)
            goal = rng.uniform(-self.world_range, self.world_range, size=3)
            obstacle_position = rng.uniform(-self.world_range, self.world_range, size=3)
            obstacle_radius = rng.uniform(radius_min, radius_max)

            spawn_goal_dist = np.linalg.norm(goal - spawn)
            spawn_obstacle_dist = np.linalg.norm(spawn - obstacle_position)
            goal_obstacle_dist = np.linalg.norm(goal - obstacle_position)

            if (spawn_goal_dist >= self.min_spawn_goal_distance
                    and spawn_obstacle_dist > obstacle_radius
                    and goal_obstacle_dist > obstacle_radius):
                return EpisodeLayout(spawn=spawn, goal=goal,
                                      obstacle_position=obstacle_position,
                                      obstacle_radius=obstacle_radius)

        raise RuntimeError(
            f"Could not generate a valid obstacle-avoidance episode layout after "
            f"{max_attempts} attempts - check that min_spawn_goal_distance "
            f"({self.min_spawn_goal_distance}) and world_range ({self.world_range}) "
            f"are compatible (e.g. min_spawn_goal_distance should be well under "
            f"2 * world_range)."
        )


class ObstacleAvoidanceTask(Task):
    """Route around a fixed (per-episode, randomized) obstacle to reach
    the goal. See ObstacleAvoidanceConfig for the frozen reward equation
    and this module's TERMINAL_REASON_* constants for the termination
    table: collision -> terminated, COLLISION; goal -> terminated, GOAL;
    neither -> not terminated (Arena's own truncated/timeout handling
    applies as normal, terminal_reason absent)."""

    def __init__(self, config: Optional[ObstacleAvoidanceConfig] = None):
        self.config = config or ObstacleAvoidanceConfig()
        self._obstacle_position = np.zeros(3)
        self._obstacle_radius = 0.0
        self._prev_distance = 0.0

    def reset(self, arena):
        layout = self.config.generate_episode(arena.np_random)
        self._obstacle_position = layout.obstacle_position
        self._obstacle_radius = layout.obstacle_radius
        self._prev_distance = float(np.linalg.norm(layout.goal - layout.spawn))
        return layout.spawn, layout.goal

    def step(self, arena, distance, truncated):
        cfg = self.config
        position = arena._core.state.position
        obstacle_center_distance = float(np.linalg.norm(position - self._obstacle_position))
        collision = obstacle_center_distance < self._obstacle_radius
        goal_reached = distance < arena.waypoint_radius

        progress = self._prev_distance - distance  # positive = real progress this step
        self._prev_distance = distance

        action = getattr(arena, "_last_action", np.zeros(6, dtype=np.float32))
        control_penalty = float(np.sum(np.square(action)))

        obstacle_shaping = 0.0
        if obstacle_center_distance < cfg.influence_radius:
            obstacle_shaping = max(
                0.0, (cfg.influence_radius - obstacle_center_distance) / cfg.influence_radius
            )

        reward = (
            cfg.reward_progress_weight * progress
            + (cfg.reward_goal_bonus if goal_reached else 0.0)
            - (cfg.reward_collision_penalty if collision else 0.0)
            - cfg.reward_control_weight * control_penalty
            - cfg.reward_obstacle_weight * obstacle_shaping
        )

        info = {
            "obstacle_clearance": obstacle_center_distance - self._obstacle_radius,
        }

        # Collision checked first: a collision should never be silently
        # overridden by simultaneously reaching the goal on the same step.
        if collision:
            terminated = True
            info["terminal_reason"] = TERMINAL_REASON_COLLISION
        elif goal_reached:
            terminated = True
            info["terminal_reason"] = TERMINAL_REASON_GOAL
        else:
            terminated = False

        return reward, terminated, info

    def extra_obs_dim(self) -> int:
        return 4  # obstacle relative position (3, body frame) + obstacle distance (1)

    def extra_obs(self, arena) -> np.ndarray:
        rel = arena._core.position_error_body_frame(self._obstacle_position)
        dist = float(np.linalg.norm(arena._core.state.position - self._obstacle_position))
        return np.concatenate([rel, [dist]]).astype(np.float32)


def _make_task(task: str, hold_steps: int, docking_orientation_tol_rad: float,
               docking_max_speed: float,
               obstacle_avoidance_config: Optional[ObstacleAvoidanceConfig]) -> Task:
    if task == "waypoint":
        return WaypointTask()
    if task == "station_keeping":
        return StationKeepingTask(hold_steps=hold_steps)
    if task == "docking":
        return DockingTask(orientation_tol_rad=docking_orientation_tol_rad,
                            max_speed=docking_max_speed)
    if task == "obstacle_avoidance":
        return ObstacleAvoidanceTask(config=obstacle_avoidance_config)
    raise ValueError(
        f"Unknown task {task!r} - expected 'waypoint', 'station_keeping', "
        f"'docking', or 'obstacle_avoidance'"
    )


class Arena(gym.Env):
    """Gymnasium environment: waypoint-following (or a variant task) with
    our calibratable 6-DOF physics (see module docstring for the
    training/validation backend split this class deliberately does NOT
    try to unify)."""

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
        obstacle_avoidance_config: Optional[ObstacleAvoidanceConfig] = None,
    ):
        """
        task: "waypoint" (default, ORIGINAL behavior, unchanged) - touch
            the target once, episode ends. "station_keeping" - must STAY
            within waypoint_radius for hold_steps CONSECUTIVE steps before
            counting as success. "docking" - requires position,
            orientation alignment, AND low speed simultaneously.
            "obstacle_avoidance" - route around a randomized obstacle to
            reach the goal; see ObstacleAvoidanceConfig/ObstacleAvoidanceTask.
        """
        super().__init__()
        sim_config = SimulationConfig(dt=dt, physics_dt=physics_dt)
        self._core = ArenaCore(params, vehicle, sim_config)
        self.max_episode_steps = max_episode_steps
        self.waypoint_radius = waypoint_radius
        self._fixed_waypoint = None if fixed_waypoint is None else np.array(fixed_waypoint, dtype=float)
        self.waypoint_range = waypoint_range
        self._task = task
        self._task_obj = _make_task(task, hold_steps, docking_orientation_tol_rad,
                                     docking_max_speed, obstacle_avoidance_config)

        # Action space NORMALIZED to [-1,1] — see PROJECT_STATUS.md for why
        # (a confirmed, real PPO/continuous-control gotcha, not theoretical).
        self._action_bound = action_bound
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )
        # obs = [u,v,w,p,q,r] (6) + position error in body frame (3) + roll,pitch (2) = 11
        # base dims, PLUS this task's own extra_obs_dim() (0 for the three
        # original tasks - their observation_space stays exactly (11,)).
        base_dim = 11
        extra_dim = self._task_obj.extra_obs_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(base_dim + extra_dim,), dtype=np.float32
        )

        self._waypoint = np.zeros(3)
        self._step_count = 0
        self._last_action = np.zeros(6, dtype=np.float32)

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
        self._last_action = action  # available to any task that needs it (e.g. control-effort reward)
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
        extra = self._task_obj.extra_obs(self)
        if extra.size > 0:
            obs = np.concatenate([obs, extra])
        return obs.astype(np.float32)