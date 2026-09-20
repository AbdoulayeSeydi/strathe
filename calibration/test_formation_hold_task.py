"""
FormationHoldTask tests — mock-based, no real ArenaCore/orchestration
loop involved (that's A1.6, not built yet). Same layering discipline
ObstacleAvoidanceTask's own Phase D used: verify the frozen reward
equation and termination table against hand-computed expected values
before any real integration exists to hide a bug behind.
"""
import numpy as np

from formation_hold_task import (
    FormationHoldTask, FormationHoldConfig,
    TERMINAL_REASON_COLLISION, TERMINAL_REASON_FORMATION_ACHIEVED,
)


def _default_config():
    return FormationHoldConfig(
        target_separation=2.0, safety_radius=0.5, influence_radius=1.5,
        tolerance=0.2, hold_steps=3,
        reward_sep_weight=1.0, reward_ctrl_weight=0.01,
        reward_prox_weight=1.0, reward_coll_weight=10.0, success_bonus=10.0,
    )


def test_config_validates_safety_vs_influence_ordering():
    try:
        FormationHoldConfig(safety_radius=2.0, influence_radius=1.0).validate()
        raise AssertionError("expected ValueError for safety_radius >= influence_radius")
    except ValueError:
        pass


def test_config_validates_tolerance_vs_target_ordering():
    try:
        FormationHoldConfig(target_separation=0.1, tolerance=0.5).validate()
        raise AssertionError("expected ValueError for tolerance >= target_separation")
    except ValueError:
        pass


def test_reward_equation_exact_at_target_separation_no_control_no_proximity():
    """At exactly target_separation, outside influence_radius (so no
    proximity shaping), zero action -> reward should be exactly 0.0
    (no sep_cost, no ctrl_cost, no prox_cost, no coll_cost, no bonus yet)."""
    task = FormationHoldTask(_default_config())
    zero_action = np.zeros(6)
    reward, terminated, info = task.step(separation=2.0, action_a=zero_action,
                                          action_b=zero_action, truncated=False)
    assert abs(reward - 0.0) < 1e-9, f"expected reward exactly 0.0, got {reward}"
    assert not terminated
    assert info["reward_components"]["separation"] == 0.0
    assert info["reward_components"]["proximity"] == 0.0


def test_reward_equation_exact_hand_computed_case():
    """A fully hand-computed case, checked to 9 decimals — same tolerance
    ObstacleAvoidanceTask's own reward-shaping tests used."""
    cfg = _default_config()
    task = FormationHoldTask(cfg)
    # separation = 1.0 -> inside influence_radius (1.5), not colliding (>0.5)
    d = 1.0
    action_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # ||u_a||^2 = 1.0
    action_b = np.array([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])  # ||u_b||^2 = 4.0

    expected_sep_cost = cfg.reward_sep_weight * abs(d - cfg.target_separation)  # 1.0 * 1.0 = 1.0
    expected_ctrl_cost = cfg.reward_ctrl_weight * (1.0 + 4.0)                    # 0.01 * 5.0 = 0.05
    expected_prox_cost = cfg.reward_prox_weight * ((cfg.influence_radius - d) / cfg.influence_radius)
    # = 1.0 * (0.5 / 1.5) = 0.333333...
    expected_reward = -expected_sep_cost - expected_ctrl_cost - expected_prox_cost

    reward, terminated, info = task.step(separation=d, action_a=action_a,
                                          action_b=action_b, truncated=False)

    assert abs(reward - expected_reward) < 1e-9, f"expected {expected_reward}, got {reward}"
    assert abs(info["reward_components"]["separation"] - (-expected_sep_cost)) < 1e-9
    assert abs(info["reward_components"]["control"] - (-expected_ctrl_cost)) < 1e-9
    assert abs(info["reward_components"]["proximity"] - (-expected_prox_cost)) < 1e-9
    assert not terminated


def test_collision_terminates_with_correct_reason_and_large_penalty():
    task = FormationHoldTask(_default_config())
    zero = np.zeros(6)
    reward, terminated, info = task.step(separation=0.3, action_a=zero, action_b=zero, truncated=False)
    assert terminated
    assert info["terminal_reason"] == TERMINAL_REASON_COLLISION
    assert info["reward_components"]["collision"] == -10.0


def test_formation_achieved_requires_consecutive_steps_not_just_one():
    """hold_steps=3 in the default test config — one in-tolerance step
    must NOT be enough."""
    task = FormationHoldTask(_default_config())
    zero = np.zeros(6)
    # step 1: within tolerance (target=2.0, tolerance=0.2 -> [1.8, 2.2))
    _, terminated1, _ = task.step(separation=2.0, action_a=zero, action_b=zero, truncated=False)
    assert not terminated1
    # step 2: still within tolerance
    _, terminated2, _ = task.step(separation=2.0, action_a=zero, action_b=zero, truncated=False)
    assert not terminated2
    # step 3: third consecutive in-tolerance step -> NOW it should succeed
    reward3, terminated3, info3 = task.step(separation=2.0, action_a=zero, action_b=zero, truncated=False)
    assert terminated3
    assert info3["terminal_reason"] == TERMINAL_REASON_FORMATION_ACHIEVED
    assert info3["reward_components"]["success_bonus"] == 10.0


def test_dropping_out_of_tolerance_resets_the_consecutive_counter():
    """A real, deliberate edge case: 2 good steps, 1 bad step, 2 more good
    steps should NOT succeed at step 5 — the counter must reset, not just
    pause, on a bad step."""
    task = FormationHoldTask(_default_config())
    zero = np.zeros(6)
    task.step(separation=2.0, action_a=zero, action_b=zero, truncated=False)  # 1
    task.step(separation=2.0, action_a=zero, action_b=zero, truncated=False)  # 2
    task.step(separation=0.9, action_a=zero, action_b=zero, truncated=False)  # bad -> reset to 0
    _, terminated4, info4 = task.step(separation=2.0, action_a=zero, action_b=zero, truncated=False)  # 1 again
    assert not terminated4
    assert info4["consecutive_steps_in_tolerance"] == 1, (
        f"expected counter to have reset and be at 1, got {info4['consecutive_steps_in_tolerance']}"
    )


def test_simultaneous_collision_and_formation_achieved_is_collision_not_success():
    """The exact priority-ordering edge case ObstacleAvoidanceTask already
    established for its own two termination conditions — collision must
    win, never silently overridden by success. Constructed by setting
    safety_radius wider than tolerance would allow, so a within-tolerance
    separation is ALSO a collision."""
    cfg = FormationHoldConfig(target_separation=2.0, safety_radius=1.9,
                               influence_radius=2.0, tolerance=0.2, hold_steps=1)
    task = FormationHoldTask(cfg)
    zero = np.zeros(6)
    # separation=1.85: within tolerance of target (2.0 +/- 0.2 -> [1.8,2.2))
    # AND within safety_radius (1.9) -> collision. hold_steps=1 means formation
    # would trigger on the FIRST in-tolerance step if collision didn't win.
    _, terminated, info = task.step(separation=1.85, action_a=zero, action_b=zero, truncated=False)
    assert terminated
    assert info["terminal_reason"] == TERMINAL_REASON_COLLISION, (
        f"collision must take priority over formation-achieved, got {info['terminal_reason']}"
    )


def test_step_accepts_comm_revealed_position_and_records_it_for_audit():
    """A1.5's literal requirement: comm_revealed_other_position must be an
    explicit input to step() itself. Confirms it's accepted and shows up
    in info for audit purposes."""
    task = FormationHoldTask(_default_config())
    zero = np.zeros(6)
    comm_pos = np.array([5.0, 6.0, 7.0])
    _, _, info = task.step(separation=2.0, action_a=zero, action_b=zero,
                            truncated=False, comm_revealed_other_position=comm_pos)
    assert info["comm_revealed_other_position_at_step_time"] == [5.0, 6.0, 7.0]

    task.reset()
    _, _, info_none = task.step(separation=2.0, action_a=zero, action_b=zero,
                                 truncated=False, comm_revealed_other_position=None)
    assert info_none["comm_revealed_other_position_at_step_time"] is None


def test_step_reward_and_termination_are_invariant_to_comm_position():
    """THE behavioral proof of the critical invariant, now that
    comm_revealed_other_position is a real parameter on step(): reward,
    terminated, and reward_components must be bit-for-bit identical
    regardless of what this parameter is set to. Run on two FRESH,
    otherwise-identically-seeded task instances so internal state
    (_consecutive_steps_in_tolerance) can't itself be the source of any
    difference — isolating the comm parameter as the only thing that
    varies between the two runs."""
    cfg = _default_config()
    task_with_comm = FormationHoldTask(cfg)
    task_without_comm = FormationHoldTask(cfg)

    action_a = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    action_b = np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0])
    fake_comm_position = np.array([999.0, -999.0, 42.0])  # a wildly different value each call

    for step_i in range(5):
        # Deliberately vary comm_revealed_other_position across calls (None,
        # then a real array, then a DIFFERENT real array) — reward/terminated
        # must never move regardless.
        comm_val = [None, fake_comm_position, fake_comm_position * -1, None, fake_comm_position][step_i]

        r_with, term_with, info_with = task_with_comm.step(
            separation=1.9, action_a=action_a, action_b=action_b,
            truncated=False, comm_revealed_other_position=comm_val,
        )
        r_without, term_without, info_without = task_without_comm.step(
            separation=1.9, action_a=action_a, action_b=action_b,
            truncated=False, comm_revealed_other_position=None,
        )

        assert r_with == r_without, (
            f"step {step_i}: reward differed ({r_with} vs {r_without}) depending on "
            f"comm_revealed_other_position — the ground-truth invariant is broken."
        )
        assert term_with == term_without, f"step {step_i}: terminated differed"
        # Compare reward_components, excluding the audit-only field that's
        # SUPPOSED to differ (it literally records the comm value itself).
        comps_with = {k: v for k, v in info_with["reward_components"].items()}
        comps_without = {k: v for k, v in info_without["reward_components"].items()}
        assert comps_with == comps_without, (
            f"step {step_i}: reward_components differed depending on comm_revealed_other_position"
        )


def test_reset_clears_consecutive_counter():
    task = FormationHoldTask(_default_config())
    zero = np.zeros(6)
    task.step(separation=2.0, action_a=zero, action_b=zero, truncated=False)
    task.step(separation=2.0, action_a=zero, action_b=zero, truncated=False)
    task.reset()
    _, terminated, info = task.step(separation=2.0, action_a=zero, action_b=zero, truncated=False)
    assert not terminated
    assert info["consecutive_steps_in_tolerance"] == 1


def test_extra_obs_with_comm_position_available():
    task = FormationHoldTask(_default_config())
    own_pos = np.array([0.0, 0.0, 0.0])
    other_pos = np.array([3.0, 4.0, 0.0])
    obs = task.extra_obs_for_agent(own_pos, other_pos)
    assert obs.shape == (4,)
    assert np.allclose(obs[:3], [3.0, 4.0, 0.0])
    assert obs[3] == 1.0  # validity flag set


def test_extra_obs_with_no_comm_position_available():
    """The comm-limited case — dropped/delayed message. Must NOT silently
    fabricate a plausible-looking zero vector without flagging it."""
    task = FormationHoldTask(_default_config())
    own_pos = np.array([1.0, 2.0, 3.0])
    obs = task.extra_obs_for_agent(own_pos, None)
    assert obs.shape == (4,)
    assert obs[3] == 0.0  # validity flag CLEAR — policy can tell this apart from real data


def test_A1_4_and_A1_5_acceptance_check():
    """
    Not a generic regression test — a direct, explicit check against the
    PLAN's literal wording for A1.4 and A1.5, so 'all tests pass' actually
    means 'these two specific requirements are satisfied,' not just
    'the code doesn't crash.'

    A1.4: 'Define terminal-reason constants for this task... added to the
    existing TERMINAL_REASON_* vocabulary in arena.py, not a separate,
    parallel constant set.'
      -> Checked by confirming formation_hold_task._IMPORTED_FROM_ARENA
         is True. If arena.py is missing the required new line, or fails
         to import for any other reason, importing formation_hold_task
         itself already raised a loud, specific ImportError before this
         test could even run — so reaching this assertion at all is
         already partial evidence, and the flag makes it explicit rather
         than inferred.

    A1.5: 'Extend Task.step()'s signature... adding whatever the comm
    module currently reveals about the other agent... as an explicit
    input.'
      -> Checked two ways: (1) inspect.signature confirms
         comm_revealed_other_position is a real parameter of step(),
         satisfying the LITERAL requirement; (2) the invariance test
         above confirms it's genuinely inert for reward/termination,
         satisfying the DESIGN requirement (ground truth only) at the
         same time. Both must hold — literal compliance without the
         invariant would be a real regression, not a win.
    """
    import inspect
    import formation_hold_task as fht

    # A1.4
    assert getattr(fht, "_IMPORTED_FROM_ARENA", False) is True, (
        "A1.4 NOT satisfied: TERMINAL_REASON_* constants were not confirmed "
        "imported from arena.py."
    )

    # A1.5, part 1 — literal signature requirement
    sig = inspect.signature(fht.FormationHoldTask.step)
    assert "comm_revealed_other_position" in sig.parameters, (
        "A1.5 NOT satisfied: step() has no comm_revealed_other_position parameter."
    )

    # A1.5, part 2 — the invariant must still hold (checked fully in the
    # dedicated test above; a lightweight spot-check here too so THIS
    # specific acceptance test is self-contained and doesn't just trust
    # another test ran first).
    task_a = fht.FormationHoldTask()
    task_b = fht.FormationHoldTask()
    zero = np.zeros(6)
    r_a, t_a, _ = task_a.step(separation=1.5, action_a=zero, action_b=zero,
                               truncated=False, comm_revealed_other_position=np.array([1.0, 2.0, 3.0]))
    r_b, t_b, _ = task_b.step(separation=1.5, action_a=zero, action_b=zero,
                               truncated=False, comm_revealed_other_position=None)
    assert r_a == r_b and t_a == t_b, (
        "A1.5 NOT satisfied: reward/termination changed based on comm_revealed_other_position — "
        "the ground-truth ONLY invariant from A1.3's frozen design is broken."
    )

    print("  -> A1.4: constants confirmed imported from arena.py (not a parallel set)")
    print("  -> A1.5: comm_revealed_other_position is a real, explicit parameter of step()")
    print("  -> A1.5: confirmed inert for reward/termination (ground-truth invariant intact)")


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
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        raise SystemExit(1)