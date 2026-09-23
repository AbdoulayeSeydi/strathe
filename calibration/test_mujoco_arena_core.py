#!/usr/bin/env python3
"""
Standalone test of MuJoCoArenaCore against REAL MuJoCo (C0.3, see
conversation) -- before building the TwoAgentOrchestrator integration
on top of it. Mirrors the tests already run locally against a fake
harness; this is the first time this class has been run against the
real MuJoCo library at all. Uses REAL vehicle_spec.yaml/
calibrated_parameters_a1.yaml (via build_two_agent_cores' own
established loading pattern), not toy values, for a meaningful check.

Usage:
    pytest test_mujoco_arena_core.py      (or: python3 test_mujoco_arena_core.py)
"""
import numpy as np

try:
    import pytest
    _fixture = pytest.fixture
except ImportError:  # still runnable as a plain script without pytest installed
    def _fixture(*args, **kwargs):
        return (lambda f: f) if not (args and callable(args[0])) else args[0]

from mujoco_backend import MuJoCoArenaCore
from arena_core import ArenaCore
from build_two_agent_cores import build_two_agent_cores
from dynamics_model import SimulationConfig

CALIBRATED_PARAMS_PATH = "calibrated_parameters_a1.yaml"


def _load_real_setup():
    _core_generic, core_calibrated = build_two_agent_cores(CALIBRATED_PARAMS_PATH)
    return core_calibrated.params, core_calibrated.vehicle, SimulationConfig(dt=0.1, physics_dt=0.005)


# pytest fixtures -- these tests used to hand values to each other through
# return values, which pytest can't do (tests 2-4 errored as "fixture not
# found" and never ran under pytest).
@_fixture(scope="module")
def _setup():
    return _load_real_setup()


@_fixture
def params(_setup):
    return _setup[0]


@_fixture
def vehicle(_setup):
    return _setup[1]


@_fixture
def sim_config(_setup):
    return _setup[2]


def test_1_reset_interface():
    print("=" * 70)
    print("TEST 1: reset() interface")
    print("=" * 70)
    params, vehicle, sim_config = _load_real_setup()

    mujoco_core = MuJoCoArenaCore(params, vehicle, sim_config)
    state = mujoco_core.reset(position=np.array([1.0, 2.0, 3.0]),
                                nu=np.array([0.3, 0.1, 0.0, 0.0, 0.0, 0.05]))
    print(f"  position: {state.position} (expected [1,2,3])")
    print(f"  nu: {state.nu}")
    assert np.allclose(state.position, [1.0, 2.0, 3.0])
    assert np.allclose(state.nu, [0.3, 0.1, 0.0, 0.0, 0.0, 0.05])
    assert np.allclose(state.orientation.as_matrix(), np.eye(3))
    print("  PASS\n")


def test_2_position_updates_under_thrust(params, vehicle, sim_config):
    print("=" * 70)
    print("TEST 2: position updates under constant thrust")
    print("(this is exactly the bug found and fixed locally -- position")
    print("silently stayed frozen before an explicit mj_kinematics() call")
    print("was added at the end of step(); confirming the fix holds here)")
    print("=" * 70)
    mujoco_core = MuJoCoArenaCore(params, vehicle, sim_config)
    mujoco_core.reset(position=np.array([1.0, 2.0, 3.0]), nu=np.zeros(6))
    tau = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.3])
    state = None
    for _ in range(20):
        state = mujoco_core.step(tau)
    print(f"  final position: {state.position}")
    print(f"  final nu: {state.nu}")
    assert np.all(np.isfinite(state.position)) and np.all(np.isfinite(state.nu))
    assert not np.allclose(state.position, [1.0, 2.0, 3.0]), "BUG: position did not move under thrust"
    print("  PASS -- position correctly updates\n")


def test_3_compare_against_arenacore(params, vehicle, sim_config):
    print("=" * 70)
    print("TEST 3: MuJoCoArenaCore vs ArenaCore over a short horizon")
    print("(loose tolerance expected -- different integrators computing")
    print("the SAME underlying acceleration, not the same code path --")
    print("matching B1.6's own established 'plausibility, not precision'")
    print("pattern for this kind of forward-sim comparison)")
    print("=" * 70)
    arena_core = ArenaCore(params, vehicle, sim_config)
    arena_core.reset(position=np.zeros(3), nu=np.zeros(6))
    mujoco_core = MuJoCoArenaCore(params, vehicle, sim_config)
    mujoco_core.reset(position=np.zeros(3), nu=np.zeros(6))

    rng = np.random.default_rng(0)
    n_steps = 30
    max_pos_diff = 0.0
    max_nu_diff = 0.0
    for i in range(n_steps):
        tau_i = rng.uniform(-5, 5, 6)
        s_arena = arena_core.step(tau_i)
        s_mujoco = mujoco_core.step(tau_i)
        max_pos_diff = max(max_pos_diff, np.max(np.abs(s_arena.position - s_mujoco.position)))
        max_nu_diff = max(max_nu_diff, np.max(np.abs(s_arena.nu - s_mujoco.nu)))
        if i in (4, 9, 19, 29):
            print(f"  step {i + 1}: arena pos={s_arena.position}, mujoco pos={s_mujoco.position}")

    print(f"\n  max position diff across {n_steps} steps: {max_pos_diff:.6f}")
    print(f"  max nu diff across {n_steps} steps: {max_nu_diff:.6f}")
    print(f"  (both should be small but plausibly nonzero -- exact zero would be")
    print(f"  suspicious given genuinely different integrators; a LARGE value")
    print(f"  would indicate a real bug, not just integrator disagreement)")
    assert np.all(np.isfinite([max_pos_diff, max_nu_diff]))
    # A real bound, not just finiteness: after fixing the missing omega x v
    # transport term in mujoco_backend this gap is ~2e-4 m; with the bug it
    # was ~4.3e-3 m. 1e-3 m separates the two with margin on both sides.
    assert max_pos_diff < 1e-3, f"MuJoCo drifted {max_pos_diff:.5f} m from ArenaCore in 3 s"
    print("  PASS\n")


def test_4_longer_horizon_stays_finite(params, vehicle, sim_config):
    print("=" * 70)
    print("TEST 4: longer horizon (100 steps), stays finite, no NaN/inf")
    print("=" * 70)
    mujoco_core = MuJoCoArenaCore(params, vehicle, sim_config)
    mujoco_core.reset(position=np.array([4.0, 0.0, 0.0]), nu=np.zeros(6))
    rng = np.random.default_rng(1)
    state = None
    for _ in range(100):
        tau_i = rng.uniform(-10, 10, 6)
        state = mujoco_core.step(tau_i)
    print(f"  final position: {state.position}")
    print(f"  final nu: {state.nu}")
    assert np.all(np.isfinite(state.position)) and np.all(np.isfinite(state.nu))
    print("  PASS\n")


if __name__ == "__main__":
    test_1_reset_interface()
    params, vehicle, sim_config = _load_real_setup()
    test_2_position_updates_under_thrust(params, vehicle, sim_config)
    test_3_compare_against_arenacore(params, vehicle, sim_config)
    test_4_longer_horizon_stays_finite(params, vehicle, sim_config)
    print("=" * 70)
    print("ALL TESTS PASSED")
    print("=" * 70)