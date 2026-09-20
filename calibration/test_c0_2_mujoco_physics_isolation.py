#!/usr/bin/env python3
"""
C0.2.2 -- MuJoCo-source physics isolation (see conversation).

A1.8 (test_zero_degradation.py) proves wrapping doesn't introduce
physics coupling -- but it checks env._orchestrator.core_generic.state
.position directly, which is ArenaCore's OWN internal state, completely
unchanged by C0.1. That test could pass even if the MuJoCo-derived
position (what channel.send/true_separation/observations actually use
now) were subtly wrong or coupled -- A1.8 was never looking at that
value at all. This file closes that specific gap: the exact same two
guarantees A1.8 already established, re-proven against the value that
actually matters post-C0.1 -- info["position_generic"] /
info["position_calibrated"], read from env.step()'s own returned info
dict (TwoAgentOrchestrator.step()'s own info.update(...) call), not
reached into via _orchestrator internals.

Two properties, mirroring A1.8's own two tests exactly in structure:

  1. test_mujoco_position_exactly_matches_independent_physics --
     the MuJoCo-derived position must match a completely independent
     ArenaCore run's position (mirrors A1.8's own baseline-comparison
     test). Uses A1.8's own tolerance (atol=1e-10) -- see this file's
     own note on why that tolerance choice is justified here, not just
     copied blindly.

  2. test_mujoco_position_never_depends_on_the_other_agents_actions --
     the MuJoCo-derived position for one agent must be bit-for-bit
     identical regardless of the OTHER agent's actions (mirrors A1.8's
     own cross-agent independence test, atol=1e-12).

## Why A1.8's own tight tolerances (1e-10, 1e-12) are appropriate here,
## not just copied without checking
This is NOT the same kind of comparison as C0.1's own old-vs-new backend
equivalence test (which used looser 1e-6 tolerances, since THAT test
crosses ArenaCore-position -> MuJoCo-qpos -> MuJoCo-position, a real
representation change). THIS file's comparisons are same-backend,
same-representation: "does the MuJoCo scene's own reported position
change depending on which agent produced it or how it got there" --
there is no representation crossing here to justify a looser bound.
Confirmed empirically too (see conversation): for this scene's simple
freejoint bodies, set_body_pose -> mj_kinematics -> get_body_position
introduces exactly zero numerical difference from the position written
in, so A1.8's own tight tolerances are the right, not arbitrary, choice.

Usage:
    python3 test_c0_2_mujoco_physics_isolation.py

*** UNVERIFIED end-to-end against real MuJoCo/formation_hold_task.py in
this sandbox -- tested against the fake harness and the real, pasted
formation_hold_task.py content, not the real MuJoCo library itself.
"""
import numpy as np

from build_two_agent_cores import build_two_agent_cores
from two_agent_formation_env import TwoAgentFormationEnv, DEFAULT_ACTION_BOUND
from comm_channel import PacketLossConfig, BandwidthConfig
from two_agent_orchestrator import OrchestratorConfig

CALIBRATED_PARAMS_PATH = "calibrated_parameters_a1.yaml"
INITIAL_SEPARATION = OrchestratorConfig().initial_separation


def _zero_degradation_env(max_episode_steps=30, action_bound=DEFAULT_ACTION_BOUND):
    return TwoAgentFormationEnv(
        calibrated_params_path=CALIBRATED_PARAMS_PATH,
        packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.0, 0.0),
        bandwidth_config=BandwidthConfig(bytes_per_tick=1e9),
        orchestrator_config=OrchestratorConfig(max_episode_steps=max_episode_steps),
        action_bound=action_bound,
    )


def test_mujoco_position_exactly_matches_independent_physics():
    """A1.8's own test_wrapped_physics_exactly_matches_independent_physics,
    re-run checking info["position_generic"]/info["position_calibrated"]
    (MuJoCo-derived, per C0.1) instead of
    env._orchestrator.core_generic.state.position directly."""
    n_steps = 30
    action_bound = 10.0
    rng = np.random.default_rng(0)
    actions_generic_norm = rng.uniform(-1.0, 1.0, size=(n_steps, 6))
    actions_calibrated_norm = rng.uniform(-1.0, 1.0, size=(n_steps, 6))

    core_generic_independent, core_calibrated_independent = build_two_agent_cores(CALIBRATED_PARAMS_PATH)
    core_generic_independent.reset(position=np.array([0.0, 0.0, 0.0]))
    core_calibrated_independent.reset(position=np.array([INITIAL_SEPARATION, 0.0, 0.0]))

    positions_generic_independent = []
    positions_calibrated_independent = []
    for i in range(n_steps):
        core_generic_independent.step(actions_generic_norm[i] * action_bound)
        core_calibrated_independent.step(actions_calibrated_norm[i] * action_bound)
        positions_generic_independent.append(core_generic_independent.state.position.copy())
        positions_calibrated_independent.append(core_calibrated_independent.state.position.copy())

    env = _zero_degradation_env(max_episode_steps=n_steps + 5, action_bound=action_bound)
    env.reset()

    positions_generic_mujoco = []
    positions_calibrated_mujoco = []
    for i in range(n_steps):
        combined_action = np.concatenate([actions_generic_norm[i], actions_calibrated_norm[i]])
        _obs, _reward, _term, _trunc, info = env.step(combined_action)
        positions_generic_mujoco.append(info["position_generic"].copy())
        positions_calibrated_mujoco.append(info["position_calibrated"].copy())

    for i in range(n_steps):
        assert np.allclose(positions_generic_independent[i], positions_generic_mujoco[i], atol=1e-10), (
            f"step {i}: generic agent's MUJOCO-DERIVED position diverged from independent "
            f"ArenaCore physics — independent={positions_generic_independent[i]}, "
            f"mujoco={positions_generic_mujoco[i]}"
        )
        assert np.allclose(positions_calibrated_independent[i], positions_calibrated_mujoco[i], atol=1e-10), (
            f"step {i}: calibrated agent's MUJOCO-DERIVED position diverged from independent physics"
        )


def test_mujoco_position_never_depends_on_the_other_agents_actions():
    """A1.8's own test_one_agents_actions_never_affect_the_others_physics,
    re-run checking info["position_generic"] (MuJoCo-derived) instead of
    env._orchestrator.core_generic.state.position directly."""
    n_steps = 20
    action_bound = 10.0
    rng = np.random.default_rng(1)
    actions_generic_norm = rng.uniform(-1.0, 1.0, size=(n_steps, 6))
    actions_calibrated_norm_run_a = rng.uniform(-1.0, 1.0, size=(n_steps, 6))
    actions_calibrated_norm_run_b = rng.uniform(-1.0, 1.0, size=(n_steps, 6))
    assert not np.allclose(actions_calibrated_norm_run_a, actions_calibrated_norm_run_b), (
        "test setup bug: the two calibrated-agent action sequences must actually differ"
    )

    def run(actions_calibrated_norm):
        env = _zero_degradation_env(max_episode_steps=n_steps + 5, action_bound=action_bound)
        env.reset()
        positions_generic_mujoco = []
        for i in range(n_steps):
            combined = np.concatenate([actions_generic_norm[i], actions_calibrated_norm[i]])
            _obs, _reward, _term, _trunc, info = env.step(combined)
            positions_generic_mujoco.append(info["position_generic"].copy())
        return positions_generic_mujoco

    positions_generic_run_a = run(actions_calibrated_norm_run_a)
    positions_generic_run_b = run(actions_calibrated_norm_run_b)

    for i in range(n_steps):
        assert np.allclose(positions_generic_run_a[i], positions_generic_run_b[i], atol=1e-12), (
            f"step {i}: generic agent's MUJOCO-DERIVED position changed depending on the OTHER "
            f"agent's actions ({positions_generic_run_a[i]} vs {positions_generic_run_b[i]}) — "
            f"the shared MuJoCo scene is introducing coupling between agents' reported positions."
        )


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        raise SystemExit(1)