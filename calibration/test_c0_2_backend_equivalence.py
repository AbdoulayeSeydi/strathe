#!/usr/bin/env python3
"""
C0.2.3 -- Backend equivalence across the full edge-case matrix (see
conversation). Extends test_c0_1_pose_source_equivalence.py's own
comparison framework (same _OldPoseSourceOrchestrator, same
compare_traces, same tolerances -- POSITION_TOLERANCE etc. imported
directly, not redefined, so C0.2 never silently drifts from what C0.1
already established as the right bound for an ArenaCore-position ->
MuJoCo-qpos -> MuJoCo-position comparison) to the full scenario matrix:

  target separation, too-close-to-d_safe, just-outside-d_safe,
  just-inside-d_safe, too-far, positive/negative relative surge,
  lateral displacement, off-axis displacement, non-identity orientation,
  success-tolerance boundary, collision boundary.

Three of these (lateral, off-axis, orientation) cannot be expressed
through TwoAgentOrchestrator.reset()'s own public parameters
(initial_separation, initial_calibrated_surge_velocity are both
1-dimensional; ArenaCore.reset() itself has no orientation parameter at
all, always hardcoding Rotation.identity()) -- inject_custom_state()
below reproduces evaluate_recovery.py's own established pattern (direct
core.state override, then rebuild observations) rather than inventing a
new approach or expanding the production reset() API just for this test.

d_safe/target_separation/tolerance values below are read from a real
FormationHoldConfig() instance's own defaults, not hardcoded numbers
that could silently drift from the task's actual configuration.

Usage:
    python3 test_c0_2_backend_equivalence.py

*** UNVERIFIED end-to-end against real MuJoCo in this sandbox -- tested
against the fake harness and the real, pasted formation_hold_task.py
content, not the real MuJoCo library itself.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from two_agent_orchestrator import TwoAgentOrchestrator, _GENERIC_BODY_ID, _CALIBRATED_BODY_ID
from formation_hold_task import FormationHoldConfig
from arena_core import ArenaState
from test_c0_1_pose_source_equivalence import (
    _OldPoseSourceOrchestrator, compare_traces, make_deterministic_action_sequence,
    POSITION_TOLERANCE, SEPARATION_TOLERANCE, OBSERVATION_TOLERANCE, REWARD_TOLERANCE,
)

_CFG = FormationHoldConfig()  # real defaults, not hardcoded numbers


def inject_custom_state(orch, generic_position=None, generic_nu=None, generic_orientation=None,
                          calibrated_position=None, calibrated_nu=None, calibrated_orientation=None):
    """Resets normally first (comm channel/task/step-count all clean),
    then directly overrides ArenaState for whichever core(s) need a
    state TwoAgentOrchestrator.reset()'s own public parameters can't
    express. Re-syncs BOTH cores' poses into the pose source afterward --
    a no-op for _OldPoseSourceOrchestrator's own overridden
    _sync_pose_to_mujoco, but necessary for the real orchestrator so
    _build_observations() (which reads MuJoCo-derived position) sees the
    injected state, not stale pre-injection data."""
    orch.reset()

    if generic_position is not None or generic_nu is not None or generic_orientation is not None:
        cur = orch.core_generic.state
        orch.core_generic.state = ArenaState(
            nu=cur.nu.copy() if generic_nu is None else np.array(generic_nu, dtype=float),
            position=cur.position.copy() if generic_position is None else np.array(generic_position, dtype=float),
            orientation=cur.orientation if generic_orientation is None else generic_orientation,
        )

    if calibrated_position is not None or calibrated_nu is not None or calibrated_orientation is not None:
        cur = orch.core_calibrated.state
        orch.core_calibrated.state = ArenaState(
            nu=cur.nu.copy() if calibrated_nu is None else np.array(calibrated_nu, dtype=float),
            position=cur.position.copy() if calibrated_position is None else np.array(calibrated_position, dtype=float),
            orientation=cur.orientation if calibrated_orientation is None else calibrated_orientation,
        )

    orch._sync_pose_to_mujoco(_GENERIC_BODY_ID, orch.core_generic)
    orch._sync_pose_to_mujoco(_CALIBRATED_BODY_ID, orch.core_calibrated)
    return orch._build_observations()


def run_episode_custom(orchestrator, actions_generic, actions_calibrated, **custom_state_kwargs):
    """Same trace-recording shape as test_c0_1's own run_episode, but
    using inject_custom_state() for setup instead of
    orchestrator.reset(initial_separation=..., ...) -- needed for the
    scenarios that require lateral/off-axis/orientation state."""
    inject_custom_state(orchestrator, **custom_state_kwargs)
    trace = []
    for action_g, action_c in zip(actions_generic, actions_calibrated):
        (obs_g, obs_c), reward, terminated, truncated, info = orchestrator.step(action_g, action_c)
        trace.append({
            "position_generic": info["position_generic"].copy(),
            "position_calibrated": info["position_calibrated"].copy(),
            "true_separation": info["true_separation"],
            "obs_generic": obs_g.copy(),
            "obs_calibrated": obs_c.copy(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "terminal_reason": info.get("terminal_reason"),
        })
        if terminated or truncated:
            break
    return trace


def run_simple_scenario(label: str, initial_separation, initial_calibrated_surge_velocity,
                          n_steps: int, action_seed: int) -> bool:
    """For scenarios expressible via TwoAgentOrchestrator.reset()'s own
    public parameters -- mirrors test_c0_1's own run_one_scenario."""
    print("=" * 70)
    print(f"SCENARIO: {label}")
    print("=" * 70)

    actions_generic, actions_calibrated = make_deterministic_action_sequence(n_steps, seed=action_seed)

    old_orch = _OldPoseSourceOrchestrator(comm_seed=0)
    new_orch = TwoAgentOrchestrator(comm_seed=0)

    old_orch.reset(initial_separation=initial_separation,
                    initial_calibrated_surge_velocity=initial_calibrated_surge_velocity)
    new_orch.reset(initial_separation=initial_separation,
                    initial_calibrated_surge_velocity=initial_calibrated_surge_velocity)

    trace_old = _step_through(old_orch, actions_generic, actions_calibrated)
    trace_new = _step_through(new_orch, actions_generic, actions_calibrated)

    passed, failures = compare_traces(trace_old, trace_new)
    _print_result(label, passed, failures)
    return passed


def run_custom_state_scenario(label: str, custom_state_kwargs: dict, n_steps: int, action_seed: int) -> bool:
    """For scenarios needing direct state injection (lateral, off-axis,
    orientation) -- both orchestrators get the SAME injected state via
    the SAME inject_custom_state() call, so any difference found is
    attributable only to the pose-source swap, never to some difference
    in how the two scenarios were constructed."""
    print("=" * 70)
    print(f"SCENARIO: {label}")
    print("=" * 70)

    actions_generic, actions_calibrated = make_deterministic_action_sequence(n_steps, seed=action_seed)

    old_orch = _OldPoseSourceOrchestrator(comm_seed=0)
    new_orch = TwoAgentOrchestrator(comm_seed=0)

    trace_old = run_episode_custom(old_orch, actions_generic, actions_calibrated, **custom_state_kwargs)
    trace_new = run_episode_custom(new_orch, actions_generic, actions_calibrated, **custom_state_kwargs)

    passed, failures = compare_traces(trace_old, trace_new)
    _print_result(label, passed, failures)
    return passed


def _step_through(orchestrator, actions_generic, actions_calibrated):
    trace = []
    for action_g, action_c in zip(actions_generic, actions_calibrated):
        (obs_g, obs_c), reward, terminated, truncated, info = orchestrator.step(action_g, action_c)
        trace.append({
            "position_generic": info["position_generic"].copy(),
            "position_calibrated": info["position_calibrated"].copy(),
            "true_separation": info["true_separation"],
            "obs_generic": obs_g.copy(),
            "obs_calibrated": obs_c.copy(),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "terminal_reason": info.get("terminal_reason"),
        })
        if terminated or truncated:
            break
    return trace


def _print_result(label, passed, failures):
    if passed:
        print(f"PASS -- {label}: no meaningful behavioral regression.\n")
    else:
        print(f"FAIL -- {label}: {len(failures)} check(s) failed:")
        for f in failures[:20]:
            print(f"  {f}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        print()


def main():
    results = {}

    # --- Simple cases: expressible via reset()'s own public parameters ---
    results["target separation"] = run_simple_scenario(
        "target separation (d=target_separation)", _CFG.target_separation, 0.0, 50, 10)
    results["too close to d_safe"] = run_simple_scenario(
        "too close to d_safe (well inside collision)", _CFG.safety_radius * 0.5, 0.0, 30, 11)
    results["just outside d_safe"] = run_simple_scenario(
        "just outside d_safe", _CFG.safety_radius * 1.02, 0.0, 30, 12)
    results["just inside d_safe"] = run_simple_scenario(
        "just inside d_safe (immediate collision expected)", _CFG.safety_radius * 0.98, 0.0, 30, 13)
    results["too far"] = run_simple_scenario(
        "too far (well beyond influence_radius)", _CFG.influence_radius * 2.0, 0.0, 50, 14)
    results["positive relative surge"] = run_simple_scenario(
        "positive relative surge", _CFG.target_separation + 1.0, 0.5, 50, 15)
    results["negative relative surge"] = run_simple_scenario(
        "negative relative surge", _CFG.target_separation + 1.0, -0.5, 50, 16)
    results["success-tolerance boundary (just inside)"] = run_simple_scenario(
        "success-tolerance boundary, just inside", _CFG.target_separation + _CFG.tolerance * 0.98, 0.0, 50, 17)
    results["success-tolerance boundary (just outside)"] = run_simple_scenario(
        "success-tolerance boundary, just outside", _CFG.target_separation + _CFG.tolerance * 1.02, 0.0, 50, 18)
    results["collision boundary (just inside)"] = run_simple_scenario(
        "collision boundary, just inside d_safe", _CFG.safety_radius * 0.99, 0.0, 30, 19)
    results["collision boundary (just outside)"] = run_simple_scenario(
        "collision boundary, just outside d_safe", _CFG.safety_radius * 1.01, 0.0, 30, 20)

    # --- Custom-state cases: need direct injection, not expressible via reset() ---
    results["lateral displacement"] = run_custom_state_scenario(
        "lateral displacement (y-offset, not just x-separation)",
        {"calibrated_position": np.array([_CFG.target_separation, 1.5, 0.0])},
        50, 21)
    results["off-axis displacement"] = run_custom_state_scenario(
        "off-axis displacement (x, y, AND z offset combined)",
        {"calibrated_position": np.array([1.5, 1.0, 0.8])},
        50, 22)
    results["non-identity orientation"] = run_custom_state_scenario(
        "non-identity orientation (calibrated agent rotated)",
        {"calibrated_position": np.array([_CFG.target_separation, 0.0, 0.0]),
         "calibrated_orientation": Rotation.from_euler("xyz", [0.3, 0.2, 0.1])},
        50, 23)
    results["both agents rotated, lateral"] = run_custom_state_scenario(
        "both agents non-identity orientation AND lateral offset",
        {"generic_orientation": Rotation.from_euler("xyz", [-0.1, 0.15, 0.0]),
         "calibrated_position": np.array([_CFG.target_separation, -1.2, 0.3]),
         "calibrated_orientation": Rotation.from_euler("xyz", [0.25, -0.1, 0.2])},
        50, 24)

    print("=" * 70)
    print("C0.2.3 OVERALL RESULT")
    print("=" * 70)
    n_passed = sum(1 for v in results.values() if v)
    n_total = len(results)
    for label, passed in results.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    print()
    if n_passed == n_total:
        print(f"PASS -- {n_passed}/{n_total} scenarios show no meaningful behavioral regression "
              f"across the full edge-case matrix.")
    else:
        print(f"FAIL -- {n_total - n_passed}/{n_total} scenario(s) failed. Investigate before "
              f"treating C0.2 as complete.")


if __name__ == "__main__":
    main()