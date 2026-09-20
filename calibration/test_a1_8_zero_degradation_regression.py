"""
A1.8 — Zero-degradation regression check.

## Why this compares against raw ArenaCore, not Arena
The plan's original wording says compare against "two independent
single-agent Arena runs" — but formation-hold has no fixed waypoint, so
Arena's own position_error_body_frame(waypoint) term has no meaning here
(see two_agent_formation_env.py's own docstring on why the observation
space is 24 dims, not 22). The real invariant worth proving is more
fundamental than "matches Arena": each agent's own physics must be
byte-for-byte identical whether it's stepped completely alone or
wrapped inside the full comm-channel + task + gym.Env machinery this
whole track has built. Two independent proofs of that below — an
external baseline comparison, and an internal cross-agent independence
check that doesn't need any baseline at all.
"""
import numpy as np

from build_two_agent_cores import build_two_agent_cores
from two_agent_formation_env import TwoAgentFormationEnv, DEFAULT_ACTION_BOUND
from comm_channel import PacketLossConfig, BandwidthConfig
from two_agent_orchestrator import OrchestratorConfig

CALIBRATED_PARAMS_PATH = "calibrated_parameters_a1.yaml"
# Must match TwoAgentOrchestrator's own default exactly — this test proves
# "wrapping changes nothing" by giving the independent baseline the SAME
# starting conditions the wrapped run actually uses, including this offset.
INITIAL_SEPARATION = OrchestratorConfig().initial_separation


def _zero_degradation_env(max_episode_steps=30, action_bound=DEFAULT_ACTION_BOUND):
    """Maximally permissive comm settings — zero packet loss, effectively
    unlimited bandwidth. NOT zero latency (never achievable for nonzero
    separation, per A1.6's own documented physical constraint) — but
    latency alone doesn't feed back into either agent's own PHYSICS at
    all, only into what each agent's OBSERVATION reveals, so it has no
    bearing on this specific check."""
    return TwoAgentFormationEnv(
        calibrated_params_path=CALIBRATED_PARAMS_PATH,
        packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.0, 0.0),
        bandwidth_config=BandwidthConfig(bytes_per_tick=1e9),
        orchestrator_config=OrchestratorConfig(max_episode_steps=max_episode_steps),
        action_bound=action_bound,
    )


def test_wrapped_physics_exactly_matches_independent_physics():
    """Baseline comparison: two ArenaCore instances stepped completely
    independently (no orchestrator, no comm channel, no task, no second
    agent in scope at all) versus the SAME two agents driven by the SAME
    real-unit forces through the full TwoAgentFormationEnv. Manually
    applies the identical action_bound rescaling to the independent
    baseline so this isolates "does wrapping introduce coupling" from
    "is rescaling correct" (already proven separately in
    test_two_agent_formation_env.py)."""
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

    positions_generic_wrapped = []
    positions_calibrated_wrapped = []
    for i in range(n_steps):
        combined_action = np.concatenate([actions_generic_norm[i], actions_calibrated_norm[i]])
        env.step(combined_action)
        positions_generic_wrapped.append(env._orchestrator.core_generic.state.position.copy())
        positions_calibrated_wrapped.append(env._orchestrator.core_calibrated.state.position.copy())

    for i in range(n_steps):
        assert np.allclose(positions_generic_independent[i], positions_generic_wrapped[i], atol=1e-10), (
            f"step {i}: generic agent's position diverged between independent and wrapped runs — "
            f"the wrapping is introducing physics coupling. independent={positions_generic_independent[i]}, "
            f"wrapped={positions_generic_wrapped[i]}"
        )
        assert np.allclose(positions_calibrated_independent[i], positions_calibrated_wrapped[i], atol=1e-10), (
            f"step {i}: calibrated agent's position diverged between independent and wrapped runs"
        )


def test_one_agents_actions_never_affect_the_others_physics():
    """The most direct possible zero-coupling proof, needing no external
    baseline at all: run generic's EXACT SAME action sequence twice,
    with calibrated's actions DIFFERING between the two runs. Generic's
    resulting position trajectory must be bit-for-bit identical either
    way — two vehicles' own equations of motion cannot depend on each
    other; only the REWARD and OBSERVATION are allowed to, via comm.
    If this fails, something in the wrapping is leaking the other
    agent's actions or state into this agent's own dynamics."""
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
        positions_generic = []
        for i in range(n_steps):
            combined = np.concatenate([actions_generic_norm[i], actions_calibrated_norm[i]])
            env.step(combined)
            positions_generic.append(env._orchestrator.core_generic.state.position.copy())
        return positions_generic

    positions_generic_run_a = run(actions_calibrated_norm_run_a)
    positions_generic_run_b = run(actions_calibrated_norm_run_b)

    for i in range(n_steps):
        assert np.allclose(positions_generic_run_a[i], positions_generic_run_b[i], atol=1e-12), (
            f"step {i}: generic agent's trajectory changed depending on the OTHER agent's "
            f"actions ({positions_generic_run_a[i]} vs {positions_generic_run_b[i]}) — the "
            f"wrapping is introducing physics coupling between agents."
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