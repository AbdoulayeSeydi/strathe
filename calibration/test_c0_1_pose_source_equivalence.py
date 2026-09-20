#!/usr/bin/env python3
"""
C0.1 acceptance test (see conversation) -- confirms the MuJoCo pose-source
swap in two_agent_orchestrator.py introduces no meaningful behavioral
regression, per the explicit acceptance criterion agreed:

    same seed, same initial state, same actions
        -> old orchestrator vs new orchestrator
        -> position diff < tolerance
        -> separation diff < tolerance
        -> observation diff < tolerance
        -> reward diff < tolerance
        -> termination identical
        -> episode trace equivalent

NOT a claim of literal mathematical identity -- writing to MuJoCo's qpos
and reading back through mj_kinematics can introduce tiny representation/
normalization differences (quaternion round-trip, floating-point
propagation through a different code path). The tolerance below is
deliberately an empirical acceptance bound, not zero.

Method: _OldPoseSourceOrchestrator subclasses the REAL, current
TwoAgentOrchestrator and overrides ONLY _sync_pose_to_mujoco -- the
single method C0.1's own refactor consolidated all position-sourcing
into -- to return ArenaCore.state.position directly, bypassing MuJoCo
entirely. Everything else (reset/step/_build_observations, the comm
channel, the task, ArenaCore itself) is the REAL, unmodified code, not a
reimplementation -- so any difference this test finds is attributable
ONLY to the pose-source swap itself, never to some incidental
reimplementation difference.

Usage:
    python3 test_c0_1_pose_source_equivalence.py

*** UNVERIFIED end-to-end in this sandbox -- formation_hold_task.py was
never provided this session, so this script could not be run against the
REAL task logic here. Syntax-checked and structurally reasoned through
only. This is the one piece of C0.1 that most needs a real run before
trusting the "no regression" claim.
"""
import numpy as np

from two_agent_orchestrator import TwoAgentOrchestrator, OrchestratorConfig
from comm_channel import PacketLossConfig, BandwidthConfig

POSITION_TOLERANCE = 1e-6
SEPARATION_TOLERANCE = 1e-6
OBSERVATION_TOLERANCE = 1e-6
REWARD_TOLERANCE = 1e-9


class _OldPoseSourceOrchestrator(TwoAgentOrchestrator):
    """Test-only subclass reproducing the EXACT pre-C0.1 behavior --
    ArenaCore.state.position read directly, MuJoCo never actually
    consulted for anything. Overrides BOTH _sync_pose_to_mujoco AND
    _build_observations() -- overriding only the former was found (see
    conversation) to leave _build_observations() still reading MuJoCo's
    xpos directly, which this subclass's own bypassed pose-sync never
    populates, producing stale/zero position data there specifically
    even though position/separation elsewhere were already correct. The
    real, current TwoAgentOrchestrator does not have this problem --
    _sync_pose_to_mujoco always keeps the scene current before
    _build_observations() reads it -- this was purely an artifact of
    this test's own reference implementation needing to fully match the
    ORIGINAL pre-C0.1 code path, not partially. Still constructs the
    real MuJoCo model/data in __init__ (inherited, unchanged) -- harmless
    and unused here, kept only so no other code path needs
    special-casing."""

    def _sync_pose_to_mujoco(self, body_id, core):
        return core.state.position.copy()

    def _build_observations(self):
        obs_generic = self.task.extra_obs_for_agent(
            own_position=self.core_generic.state.position,
            comm_revealed_other_position=self._last_revealed_to_generic,
        )
        obs_calibrated = self.task.extra_obs_for_agent(
            own_position=self.core_calibrated.state.position,
            comm_revealed_other_position=self._last_revealed_to_calibrated,
        )
        return obs_generic, obs_calibrated


def make_deterministic_action_sequence(n_steps: int, seed: int = 0):
    """Fixed, reproducible actions for both agents -- deliberately
    nontrivial (nonzero on multiple axes) so the comparison actually
    exercises real motion, not a degenerate zero-action trajectory."""
    rng = np.random.default_rng(seed)
    actions_generic = rng.uniform(-5, 5, size=(n_steps, 6))
    actions_calibrated = rng.uniform(-5, 5, size=(n_steps, 6))
    return actions_generic, actions_calibrated


def run_episode(orchestrator, actions_generic, actions_calibrated,
                 initial_separation=None, initial_calibrated_surge_velocity=None):
    """Runs one full episode trace, recording everything the acceptance
    criterion checks at every step."""
    orchestrator.reset(initial_separation=initial_separation,
                        initial_calibrated_surge_velocity=initial_calibrated_surge_velocity)
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


def compare_traces(trace_old, trace_new):
    """Compares two episode traces step by step, per the explicit
    acceptance criterion. Returns (passed, list_of_failure_messages)."""
    failures = []

    if len(trace_old) != len(trace_new):
        failures.append(
            f"Episode length differs: old ran {len(trace_old)} steps, "
            f"new ran {len(trace_new)} steps -- termination diverged."
        )
        # Still compare the overlapping prefix, for diagnostic value.

    n_common = min(len(trace_old), len(trace_new))
    max_position_diff = 0.0
    max_separation_diff = 0.0
    max_obs_diff = 0.0
    max_reward_diff = 0.0

    for i in range(n_common):
        old, new = trace_old[i], trace_new[i]

        pos_diff = max(
            np.max(np.abs(old["position_generic"] - new["position_generic"])),
            np.max(np.abs(old["position_calibrated"] - new["position_calibrated"])),
        )
        max_position_diff = max(max_position_diff, pos_diff)
        if pos_diff >= POSITION_TOLERANCE:
            failures.append(f"Step {i}: position diff {pos_diff:.3e} >= tolerance {POSITION_TOLERANCE:.3e}")

        sep_diff = abs(old["true_separation"] - new["true_separation"])
        max_separation_diff = max(max_separation_diff, sep_diff)
        if sep_diff >= SEPARATION_TOLERANCE:
            failures.append(f"Step {i}: separation diff {sep_diff:.3e} >= tolerance {SEPARATION_TOLERANCE:.3e}")

        obs_diff = max(
            np.max(np.abs(old["obs_generic"] - new["obs_generic"])),
            np.max(np.abs(old["obs_calibrated"] - new["obs_calibrated"])),
        )
        max_obs_diff = max(max_obs_diff, obs_diff)
        if obs_diff >= OBSERVATION_TOLERANCE:
            failures.append(f"Step {i}: observation diff {obs_diff:.3e} >= tolerance {OBSERVATION_TOLERANCE:.3e}")

        reward_diff = abs(old["reward"] - new["reward"])
        max_reward_diff = max(max_reward_diff, reward_diff)
        if reward_diff >= REWARD_TOLERANCE:
            failures.append(f"Step {i}: reward diff {reward_diff:.3e} >= tolerance {REWARD_TOLERANCE:.3e}")

        if old["terminated"] != new["terminated"] or old["truncated"] != new["truncated"]:
            failures.append(
                f"Step {i}: termination flags differ -- old (terminated={old['terminated']}, "
                f"truncated={old['truncated']}), new (terminated={new['terminated']}, "
                f"truncated={new['truncated']})"
            )
        if old["terminal_reason"] != new["terminal_reason"]:
            failures.append(
                f"Step {i}: terminal_reason differs -- old={old['terminal_reason']!r}, "
                f"new={new['terminal_reason']!r}"
            )

    print(f"Max position diff across episode:    {max_position_diff:.3e} (tolerance {POSITION_TOLERANCE:.3e})")
    print(f"Max separation diff across episode:   {max_separation_diff:.3e} (tolerance {SEPARATION_TOLERANCE:.3e})")
    print(f"Max observation diff across episode:  {max_obs_diff:.3e} (tolerance {OBSERVATION_TOLERANCE:.3e})")
    print(f"Max reward diff across episode:        {max_reward_diff:.3e} (tolerance {REWARD_TOLERANCE:.3e})")

    return len(failures) == 0, failures


def run_one_scenario(label: str, initial_separation, initial_calibrated_surge_velocity,
                       n_steps: int, action_seed: int):
    print("=" * 70)
    print(f"SCENARIO: {label}")
    print("=" * 70)

    actions_generic, actions_calibrated = make_deterministic_action_sequence(n_steps, seed=action_seed)

    old_orch = _OldPoseSourceOrchestrator(comm_seed=0)
    new_orch = TwoAgentOrchestrator(comm_seed=0)

    trace_old = run_episode(old_orch, actions_generic, actions_calibrated,
                              initial_separation, initial_calibrated_surge_velocity)
    trace_new = run_episode(new_orch, actions_generic, actions_calibrated,
                              initial_separation, initial_calibrated_surge_velocity)

    passed, failures = compare_traces(trace_old, trace_new)
    if passed:
        print(f"PASS -- {label}: no meaningful behavioral regression.\n")
    else:
        print(f"FAIL -- {label}: {len(failures)} check(s) failed:")
        for f in failures[:20]:  # cap printed output -- a real failure could produce many lines
            print(f"  {f}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more")
        print()
    return passed


def main():
    scenarios = [
        ("default fixed start (4.0m, zero velocity)", None, None, 50, 1),
        ("near tolerance boundary (2.3m, small velocity)", 2.3, 0.15, 50, 2),
        ("far, receding (5.0m, +0.7 surge)", 5.0, 0.7, 50, 3),
        ("close, approaching fast (2.0m, -0.7 surge)", 2.0, -0.7, 50, 4),
    ]

    results = []
    for label, sep0, vel0, n_steps, seed in scenarios:
        results.append(run_one_scenario(label, sep0, vel0, n_steps, seed))

    print("=" * 70)
    print("C0.1 OVERALL RESULT")
    print("=" * 70)
    if all(results):
        print(f"PASS -- {len(results)}/{len(results)} scenarios show no meaningful behavioral "
              f"regression. The MuJoCo pose-source swap is behaviorally equivalent within "
              f"tolerance.")
    else:
        n_failed = sum(1 for r in results if not r)
        print(f"FAIL -- {n_failed}/{len(results)} scenario(s) failed. Investigate before "
              f"treating C0.1 as complete.")


if __name__ == "__main__":
    main()