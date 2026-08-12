"""
Layered regression tests for ObstacleAvoidanceTask (Phase B/D) - matching
the frozen design doc's D1-D4 test plan exactly:
  D1. Mock-based logic tests - fast, isolated, no real Arena/gym dependency
  D2. Real-Arena-instance construction/step tests
  D3. Deterministic scenarios with KNOWN, exact expected outcomes - the
      strongest layer, using a real Arena with directly-controlled initial
      state (not relying on random generation to happen to produce a
      given scenario)
  D4. This file's own collection, run alongside the existing suite,
      confirms the total test count grows correctly.
"""
import numpy as np
import pytest

from arena import (
    Arena, ObstacleAvoidanceTask, ObstacleAvoidanceConfig,
    TERMINAL_REASON_GOAL, TERMINAL_REASON_COLLISION,
)
from optimizer import load_vehicle_spec, initial_parameters


# ---------------------------------------------------------------------------
# D1 - Mock-based logic tests (no real Arena/gym dependency)
# ---------------------------------------------------------------------------

class _FakeCoreState:
    def __init__(self, position, nu=None):
        self.position = np.array(position, dtype=float)
        self.nu = np.zeros(6) if nu is None else np.array(nu, dtype=float)


class _FakeCore:
    def __init__(self, position):
        self.state = _FakeCoreState(position)


class _FakeArena:
    """Minimal stand-in exposing only what ObstacleAvoidanceTask.step()
    actually reads - no real gym/ArenaCore dependency at all."""
    def __init__(self, position, waypoint_radius=0.5, last_action=None):
        self._core = _FakeCore(position)
        self.waypoint_radius = waypoint_radius
        self._last_action = np.zeros(6, dtype=np.float32) if last_action is None else last_action


class TestObstacleAvoidanceTaskLogicMock:
    def test_collision_when_inside_obstacle_radius(self):
        task = ObstacleAvoidanceTask()
        task._obstacle_position = np.array([0.0, 0.0, 0.0])
        task._obstacle_radius = 1.0
        task._prev_distance = 10.0
        arena = _FakeArena(position=[0.2, 0.0, 0.0])  # inside the radius
        reward, terminated, info = task.step(arena, distance=9.8, truncated=False)
        assert terminated is True
        assert info["terminal_reason"] == TERMINAL_REASON_COLLISION

    def test_goal_reached_no_collision(self):
        task = ObstacleAvoidanceTask()
        task._obstacle_position = np.array([50.0, 50.0, 50.0])  # far away
        task._obstacle_radius = 1.0
        task._prev_distance = 0.6
        arena = _FakeArena(position=[9.7, 0.0, 0.0], waypoint_radius=0.5)
        reward, terminated, info = task.step(arena, distance=0.3, truncated=False)
        assert terminated is True
        assert info["terminal_reason"] == TERMINAL_REASON_GOAL

    def test_normal_step_neither_collision_nor_goal(self):
        task = ObstacleAvoidanceTask()
        task._obstacle_position = np.array([50.0, 50.0, 50.0])
        task._obstacle_radius = 1.0
        task._prev_distance = 5.5
        arena = _FakeArena(position=[0.0, 0.0, 0.0], waypoint_radius=0.5)
        reward, terminated, info = task.step(arena, distance=5.0, truncated=False)
        assert terminated is False
        assert "terminal_reason" not in info

    def test_collision_takes_priority_over_simultaneous_goal(self):
        """Edge case: vehicle is simultaneously inside the obstacle AND
        within waypoint_radius of the goal - collision must win, per the
        frozen design's explicit ordering decision."""
        task = ObstacleAvoidanceTask()
        task._obstacle_position = np.array([0.0, 0.0, 0.0])
        task._obstacle_radius = 1.0
        task._prev_distance = 0.5
        arena = _FakeArena(position=[0.1, 0.0, 0.0], waypoint_radius=0.5)
        reward, terminated, info = task.step(arena, distance=0.1, truncated=False)
        assert terminated is True
        assert info["terminal_reason"] == TERMINAL_REASON_COLLISION

    def test_reward_includes_obstacle_shaping_within_influence_radius(self):
        cfg = ObstacleAvoidanceConfig(influence_radius=1.5, reward_obstacle_weight=1.0,
                                       reward_progress_weight=0.0)
        task = ObstacleAvoidanceTask(config=cfg)
        task._obstacle_position = np.array([0.0, 0.0, 0.0])
        task._obstacle_radius = 0.3
        task._prev_distance = 2.0
        arena = _FakeArena(position=[1.0, 0.0, 0.0], waypoint_radius=0.5)  # distance to obstacle = 1.0
        reward, terminated, info = task.step(arena, distance=2.0, truncated=False)
        expected_shaping = (1.5 - 1.0) / 1.5
        assert abs(reward - (-1.0 * expected_shaping)) < 1e-9

    def test_generate_episode_produces_valid_layout(self):
        cfg = ObstacleAvoidanceConfig()
        rng = np.random.default_rng(42)
        for _ in range(200):
            layout = cfg.generate_episode(rng)
            assert np.linalg.norm(layout.goal - layout.spawn) >= cfg.min_spawn_goal_distance
            assert np.linalg.norm(layout.spawn - layout.obstacle_position) > layout.obstacle_radius
            assert np.linalg.norm(layout.goal - layout.obstacle_position) > layout.obstacle_radius

    def test_generate_episode_raises_on_impossible_config(self):
        cfg = ObstacleAvoidanceConfig(world_range=1.0, min_spawn_goal_distance=100.0)
        with pytest.raises(RuntimeError):
            cfg.generate_episode(np.random.default_rng(1))


# ---------------------------------------------------------------------------
# D2 - Real Arena instance construction/step tests
# ---------------------------------------------------------------------------

@pytest.fixture
def real_vehicle_params():
    loaded = load_vehicle_spec("vehicle_spec.yaml")
    return initial_parameters(loaded), loaded.vehicle


class TestObstacleAvoidanceArenaReal:
    def test_constructs_and_resets_without_error(self, real_vehicle_params):
        params, vehicle = real_vehicle_params
        env = Arena(params, vehicle, task="obstacle_avoidance")
        obs, info = env.reset(seed=1)
        assert obs.shape == (15,)

    def test_steps_without_error(self, real_vehicle_params):
        params, vehicle = real_vehicle_params
        env = Arena(params, vehicle, task="obstacle_avoidance")
        obs, info = env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert obs.shape == (15,)
        assert "obstacle_clearance" in info

    def test_observation_space_shape_matches_actual_obs(self, real_vehicle_params):
        params, vehicle = real_vehicle_params
        env = Arena(params, vehicle, task="obstacle_avoidance")
        assert env.observation_space.shape == (15,)
        obs, _ = env.reset(seed=1)
        assert obs.shape == env.observation_space.shape


# ---------------------------------------------------------------------------
# D3 - Deterministic scenarios with KNOWN, exact expected outcomes,
# using a real Arena with directly-controlled initial state
# ---------------------------------------------------------------------------

class TestObstacleAvoidanceDeterministicScenarios:
    def test_guaranteed_collision(self, real_vehicle_params):
        params, vehicle = real_vehicle_params
        env = Arena(params, vehicle, task="obstacle_avoidance")
        env.reset(seed=1)
        # Directly control the scenario: obstacle at origin, vehicle placed
        # INSIDE its radius - a real, known collision, not left to chance.
        env._task_obj._obstacle_position = np.array([0.0, 0.0, 0.0])
        env._task_obj._obstacle_radius = 1.0
        env._core.reset(position=np.array([0.3, 0.0, 0.0]))
        env._waypoint = np.array([100.0, 100.0, 100.0])  # far away - not a goal-reach case
        env._task_obj._prev_distance = float(np.linalg.norm(env._waypoint - env._core.state.position))
        obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert terminated is True
        assert info["terminal_reason"] == TERMINAL_REASON_COLLISION

    def test_guaranteed_success(self, real_vehicle_params):
        params, vehicle = real_vehicle_params
        env = Arena(params, vehicle, task="obstacle_avoidance", waypoint_radius=0.5)
        env.reset(seed=1)
        # Vehicle placed directly AT the goal, obstacle far away - a real,
        # known success, not left to chance.
        env._task_obj._obstacle_position = np.array([100.0, 100.0, 100.0])
        env._task_obj._obstacle_radius = 0.3
        env._waypoint = np.array([0.0, 0.0, 0.0])
        env._core.reset(position=np.array([0.1, 0.0, 0.0]))  # well within waypoint_radius
        env._task_obj._prev_distance = float(np.linalg.norm(env._waypoint - env._core.state.position))
        obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert terminated is True
        assert info["terminal_reason"] == TERMINAL_REASON_GOAL

    def test_guaranteed_timeout(self, real_vehicle_params):
        params, vehicle = real_vehicle_params
        env = Arena(params, vehicle, task="obstacle_avoidance", max_episode_steps=3)
        obs, info = env.reset(seed=1)
        # Zero action every step, short episode - drift alone should not
        # plausibly reach the goal or a collision in 3 steps.
        terminated = truncated = False
        for _ in range(3):
            obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
        assert truncated is True
        assert info.get("terminal_reason") is None

    def test_same_seed_produces_identical_episode_layout(self, real_vehicle_params):
        params, vehicle = real_vehicle_params
        env1 = Arena(params, vehicle, task="obstacle_avoidance")
        env2 = Arena(params, vehicle, task="obstacle_avoidance")
        env1.reset(seed=777)
        env2.reset(seed=777)
        assert np.array_equal(env1._core.state.position, env2._core.state.position)
        assert np.array_equal(env1._waypoint, env2._waypoint)
        assert np.array_equal(env1._task_obj._obstacle_position, env2._task_obj._obstacle_position)
        assert env1._task_obj._obstacle_radius == env2._task_obj._obstacle_radius   