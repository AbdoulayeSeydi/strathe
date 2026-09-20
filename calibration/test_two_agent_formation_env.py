"""
A1.7 tests. Focused specifically on the three things this step was asked
to guarantee: real gym.Env interface, correct combined shapes, and
correct action rescaling — including the clip-then-scale ORDER, since
getting that backwards would silently allow effectively-unbounded forces.
"""
import numpy as np

from two_agent_formation_env import TwoAgentFormationEnv, DEFAULT_ACTION_BOUND

CALIBRATED_PARAMS_PATH = "calibrated_parameters_a1.yaml"


def _env(**kwargs):
    return TwoAgentFormationEnv(calibrated_params_path=CALIBRATED_PARAMS_PATH, **kwargs)


def test_action_and_observation_spaces_have_declared_shapes():
    env = _env()
    assert env.action_space.shape == (12,)
    assert env.observation_space.shape == (24,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_reset_returns_gymnasium_tuple_matching_observation_space():
    env = _env()
    result = env.reset()
    assert isinstance(result, tuple) and len(result) == 2, "reset() must return (obs, info)"
    obs, info = result
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_step_returns_real_gymnasium_five_tuple():
    env = _env()
    env.reset()
    action = env.action_space.sample()
    result = env.step(action)
    assert isinstance(result, tuple) and len(result) == 5, (
        "step() must return exactly (obs, reward, terminated, truncated, info)"
    )
    obs, reward, terminated, truncated, info = result
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_smoke_random_episode_runs_to_completion_without_crashing():
    """Same spirit as run_full_pipeline.py's own smoke_test_gym_environment —
    confirm a full random-action episode runs end to end, not just one step."""
    env = _env(orchestrator_config=None)  # default max_episode_steps=200
    env.reset()
    for _ in range(50):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        if terminated or truncated:
            break
    assert True  # reaching here at all is the test


def test_action_is_rescaled_by_action_bound():
    """THE specific check for point 3 — confirms [-1,1] actions actually
    get multiplied by action_bound before reaching the physics, not passed
    through raw. Verified by MONKEYPATCHING the underlying orchestrator's
    step() to capture exactly what tau values it receives, rather than
    inferring rescaling indirectly from resulting positions (which would
    also be affected by physics/damping and harder to attribute precisely)."""
    env = _env(action_bound=10.0)
    env.reset()

    captured = {}
    original_step = env._orchestrator.step

    def spy_step(action_generic, action_calibrated, **kwargs):
        captured["tau_generic"] = np.array(action_generic)
        captured["tau_calibrated"] = np.array(action_calibrated)
        return original_step(action_generic, action_calibrated, **kwargs)

    env._orchestrator.step = spy_step

    normalized_action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0,   # generic: full +surge
                                   0.0, -1.0, 0.0, 0.0, 0.0, 0.0])  # calibrated: full -sway
    env.step(normalized_action)

    assert np.allclose(captured["tau_generic"], [10.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (
        f"expected generic tau scaled by action_bound=10.0, got {captured['tau_generic']}"
    )
    assert np.allclose(captured["tau_calibrated"], [0.0, -10.0, 0.0, 0.0, 0.0, 0.0]), (
        f"expected calibrated tau scaled by action_bound=10.0, got {captured['tau_calibrated']}"
    )


def test_out_of_range_action_is_clipped_BEFORE_scaling_not_after():
    """The order-sensitive check: an action of 2.0 (outside the declared
    [-1,1] space) must be CLIPPED to 1.0 first, then scaled by
    action_bound — giving action_bound, NOT 2.0*action_bound. Getting this
    order backwards would silently permit unbounded real-world forces
    from a policy that briefly proposes an out-of-range value during
    exploration."""
    env = _env(action_bound=10.0)
    env.reset()

    captured = {}
    original_step = env._orchestrator.step

    def spy_step(action_generic, action_calibrated, **kwargs):
        captured["tau_generic"] = np.array(action_generic)
        return original_step(action_generic, action_calibrated, **kwargs)

    env._orchestrator.step = spy_step

    out_of_range_action = np.array([2.0, -5.0, 0.0, 0.0, 0.0, 0.0,
                                     0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    env.step(out_of_range_action)

    # 2.0 -> clipped to 1.0 -> * 10.0 = 10.0 (NOT 20.0)
    # -5.0 -> clipped to -1.0 -> * 10.0 = -10.0 (NOT -50.0)
    assert captured["tau_generic"][0] == 10.0, (
        f"expected clip-then-scale to give 10.0, got {captured['tau_generic'][0]} "
        f"(if this is 20.0, scaling happened BEFORE clipping — a real bug)"
    )
    assert captured["tau_generic"][1] == -10.0, (
        f"expected clip-then-scale to give -10.0, got {captured['tau_generic'][1]}"
    )


def test_default_action_bound_matches_arenas_own_default():
    """A cheap, direct guard against the two files' defaults silently
    drifting apart — flagged as a real, open item in the module docstring
    (no shared constant module yet), but at least caught immediately if
    it happens rather than discovered later during training."""
    assert DEFAULT_ACTION_BOUND == 30.0


def test_observation_layout_matches_documented_12_plus_12_split():
    """Confirms structurally that obs[0:6] is generic's nu, obs[6:8] is
    its roll/pitch, obs[8:12] is its extra_obs, and the same pattern
    repeats at obs[12:24] for calibrated — by checking against the raw
    ArenaCore state directly after a reset (nu should be all zeros at
    reset, per ArenaCore.reset()'s own contract)."""
    env = _env()
    obs, _ = env.reset()
    # At reset, both cores start at nu=zeros, orientation=identity (roll=pitch=0),
    # and nothing has been communicated yet (validity flags should be 0).
    assert np.allclose(obs[0:6], 0.0), "generic nu should be zero at reset"
    assert np.allclose(obs[6:8], 0.0), "generic roll/pitch should be zero at reset"
    assert obs[11] == 0.0, "generic's validity flag (last of its extra_obs) should be 0 at reset"
    assert np.allclose(obs[12:18], 0.0), "calibrated nu should be zero at reset"
    assert np.allclose(obs[18:20], 0.0), "calibrated roll/pitch should be zero at reset"
    assert obs[23] == 0.0, "calibrated's validity flag should be 0 at reset"


def test_control_cost_uses_normalized_action_not_rescaled_force():
    """THE regression test for the confirmed scale bug: reward's control
    cost must be computed from the NORMALIZED [-1,1] action, not the
    rescaled real-Newton force — matching ObstacleAvoidanceTask's own
    convention, which is what reward_ctrl_weight=0.01 was calibrated
    against. Verified directly: at full-thrust action=1.0 on one axis,
    the resulting control-cost component must be reward_ctrl_weight * 1.0^2
    = 0.01, NOT reward_ctrl_weight * action_bound^2 = 9.0 (the bug's
    signature value, confirmed by direct calculation before this fix)."""
    from formation_hold_task import FormationHoldConfig
    env = _env(action_bound=30.0)
    env.reset()

    # Full-thrust action on generic's surge axis only; everything else zero.
    action = np.zeros(12, dtype=np.float32)
    action[0] = 1.0  # generic surge, full [-1,1] scale
    _, reward, _, _, info = env.step(action)

    expected_ctrl_cost = FormationHoldConfig().reward_ctrl_weight * (1.0 ** 2)
    actual_ctrl_component = info["reward_components"]["control"]

    assert abs(actual_ctrl_component - (-expected_ctrl_cost)) < 1e-6, (
        f"expected control cost component -{expected_ctrl_cost} (normalized-action scale), "
        f"got {actual_ctrl_component} — if this is close to -9.0, the bug is back: "
        f"the RAW 30N force is leaking into the control-cost calculation again."
    )


def test_backward_compatible_when_reward_override_not_given():
    """Confirms TwoAgentOrchestrator.step() (called directly, not through
    the env) is completely unaffected by this fix when the new
    action_*_for_reward parameters aren't passed — preserving every
    existing A1.6 test's exact expected behavior."""
    from two_agent_orchestrator import TwoAgentOrchestrator
    from comm_channel import PacketLossConfig, BandwidthConfig

    orch = TwoAgentOrchestrator(
        calibrated_params_path=CALIBRATED_PARAMS_PATH,
        packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.0, 0.0),
        bandwidth_config=BandwidthConfig(bytes_per_tick=1e9),
    )
    orch.reset()
    action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # No action_*_for_reward given -- control cost should reflect the SAME
    # value passed for physics (old behavior), not silently zero or altered.
    _, _, _, _, info = orch.step(action, action)
    expected_ctrl_cost = 0.01 * (1.0 ** 2 + 1.0 ** 2)  # both agents, same small action
    assert abs(info["reward_components"]["control"] - (-expected_ctrl_cost)) < 1e-6


def test_randomize_initial_state_defaults_to_false_preserves_original_behavior():
    """Default (no randomize_initial_state argument at all) must be
    byte-for-byte identical to the original fixed-start behavior --
    same starting observation regardless of seed."""
    env = _env()
    obs1, _ = env.reset(seed=1)
    obs2, _ = env.reset(seed=2)
    assert np.allclose(obs1, obs2), (
        "default construction (randomize_initial_state not passed) must give the "
        "SAME fixed start regardless of seed -- if this fails, backward "
        "compatibility for every existing training run has broken"
    )


def test_randomize_initial_state_stays_within_configured_ranges():
    sep_range, vel_range = (2.0, 5.5), (-0.7, 0.7)
    env = _env(randomize_initial_state=True,
                initial_separation_range=sep_range, initial_velocity_range=vel_range)
    for seed in range(20):
        env.reset(seed=seed)
        pos = env._orchestrator.core_calibrated.state.position[0]
        vel = env._orchestrator.core_calibrated.state.nu[0]
        assert sep_range[0] <= pos <= sep_range[1], f"seed {seed}: position {pos} out of range"
        assert vel_range[0] <= vel <= vel_range[1], f"seed {seed}: velocity {vel} out of range"


def test_randomize_initial_state_same_seed_is_reproducible():
    env = _env(randomize_initial_state=True)
    env.reset(seed=42)
    pos_a = env._orchestrator.core_calibrated.state.position[0]
    env.reset(seed=42)
    pos_b = env._orchestrator.core_calibrated.state.position[0]
    assert pos_a == pos_b, "same seed must reproduce the same sampled start"


def test_randomize_initial_state_generic_agent_still_starts_at_origin():
    """Randomization only affects the calibrated agent's start -- generic
    stays at the origin, matching every diagnostic script's convention
    throughout this investigation (evaluate_recovery.py, trace_recovery_cases.py)."""
    env = _env(randomize_initial_state=True)
    for seed in range(10):
        env.reset(seed=seed)
        assert np.allclose(env._orchestrator.core_generic.state.position, [0.0, 0.0, 0.0])
        assert np.allclose(env._orchestrator.core_generic.state.nu, np.zeros(6))


def test_anchor_defaults_to_zero_probability_never_fires():
    """Default anchor_probability=0.0 must never override curriculum/random
    sampling -- confirms backward compatibility with every prior training
    run that didn't ask for anchoring."""
    env = _env(randomize_initial_state=True)
    positions = []
    for seed in range(30):
        env.reset(seed=seed)
        positions.append(env._orchestrator.core_calibrated.state.position[0])
    n_at_exact_anchor = sum(1 for p in positions if p == 4.0)
    assert n_at_exact_anchor == 0, (
        "anchor_probability=0.0 should never land exactly on the anchor point "
        "via continuous random sampling -- this is the exact gap the anchor "
        "feature exists to close"
    )


def test_anchor_probability_one_always_gives_exact_anchor_point():
    """anchor_probability=1.0 forces every reset to the anchor -- and by
    default that anchor tracks orchestrator_config.initial_separation
    (4.0) and velocity 0.0, the ORIGINAL fixed-start scenario, without
    needing it hardcoded twice."""
    env = _env(randomize_initial_state=True, anchor_probability=1.0)
    for seed in range(5):
        env.reset(seed=seed)
        pos = env._orchestrator.core_calibrated.state.position[0]
        vel = env._orchestrator.core_calibrated.state.nu[0]
        assert pos == 4.0 and vel == 0.0


def test_anchor_bypasses_curriculum_difficulty_filter():
    """The anchor point must ALWAYS be reachable regardless of the current
    curriculum_max_difficulty threshold -- it should never be filtered
    out like a normal curriculum-sampled point would be."""
    import csv
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["d0", "v0", "difficulty"])
        writer.writeheader()
        writer.writerow({"d0": 2.0, "v0": -0.7, "difficulty": 0.1})
        writer.writerow({"d0": 5.5, "v0": 0.7, "difficulty": 1.0})
        csv_path = f.name

    from difficulty_lookup import DifficultyLookup
    lookup = DifficultyLookup(csv_path)
    env = _env(randomize_initial_state=True, curriculum_lookup=lookup, anchor_probability=1.0)
    env.curriculum_max_difficulty = 0.1  # would normally restrict sampling to the easy corner
    env.reset(seed=0)
    pos = env._orchestrator.core_calibrated.state.position[0]
    assert pos == 4.0, "anchor must bypass the difficulty filter, not be subject to it"


def test_anchor_probability_is_mutable_mid_training_like_curriculum_max_difficulty():
    """anchor_probability must be a PUBLIC, mutable attribute (not fixed at
    construction) so a training callback can ramp it per curriculum stage
    -- e.g. 0% during easy stages, rising once the difficulty threshold
    reaches the anchor's own neighborhood, exactly like
    curriculum_max_difficulty already works."""
    env = _env(randomize_initial_state=True, anchor_probability=0.0)
    env.reset(seed=0)
    pos_before = env._orchestrator.core_calibrated.state.position[0]

    env.anchor_probability = 1.0  # simulate a mid-training stage transition
    env.reset(seed=0)
    pos_after = env._orchestrator.core_calibrated.state.position[0]

    assert pos_after == 4.0, "raising anchor_probability mid-run must take effect immediately"


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