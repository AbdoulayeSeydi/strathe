"""
A1.6 tests. Layered per the project's own established discipline: basic
wiring sanity first, then the specific physical properties this loop is
supposed to exhibit (staleness, divergence, ground-truth-only reward).
"""
import numpy as np

from two_agent_orchestrator import TwoAgentOrchestrator, OrchestratorConfig
from comm_channel import PacketLossConfig, BandwidthConfig

CALIBRATED_PARAMS_PATH = "calibrated_parameters_a1.yaml"


def _permissive_orchestrator(max_episode_steps=50, seed=0):
    """No packet loss, effectively unlimited bandwidth — A1.8's
    zero-degradation setting, needed by several tests below that aren't
    specifically about degradation itself."""
    return TwoAgentOrchestrator(
        calibrated_params_path=CALIBRATED_PARAMS_PATH,
        packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.0, 0.0),
        bandwidth_config=BandwidthConfig(bytes_per_tick=1e9),
        orchestrator_config=OrchestratorConfig(max_episode_steps=max_episode_steps),
        comm_seed=seed,
    )


def test_reset_returns_two_observations_of_expected_shape():
    orch = _permissive_orchestrator()
    obs_generic, obs_calibrated = orch.reset()
    assert obs_generic.shape == (4,)
    assert obs_calibrated.shape == (4,)
    # Nothing has been sent yet at reset — both should show "no info" (validity flag 0).
    assert obs_generic[3] == 0.0
    assert obs_calibrated[3] == 0.0


def test_step_runs_without_crashing_and_returns_expected_types():
    orch = _permissive_orchestrator()
    orch.reset()
    action_generic = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action_calibrated = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    (obs_g, obs_c), reward, terminated, truncated, info = orch.step(action_generic, action_calibrated)

    assert obs_g.shape == (4,)
    assert obs_c.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "true_separation" in info
    assert "reward_components" in info


def test_agent_always_lags_by_at_least_one_tick():
    """The specific, non-obvious physical consequence flagged in the
    module docstring: even under maximally permissive comm settings, a
    message sent at tick N cannot be delivered until tick N+1 at the
    earliest, because latency = distance/1500 is never exactly zero.
    Confirmed by checking the FIRST step reveals nothing yet (both
    validity flags 0), and only from the SECOND step onward does
    information start arriving."""
    orch = _permissive_orchestrator()
    orch.reset()
    action = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])

    (obs_g_1, obs_c_1), _, _, _, info_1 = orch.step(action, action)
    assert obs_g_1[3] == 0.0, "agent should NOT have received anything on the very first step"
    assert obs_c_1[3] == 0.0, "agent should NOT have received anything on the very first step"

    (obs_g_2, obs_c_2), _, _, _, info_2 = orch.step(action, action)
    assert obs_g_2[3] == 1.0, "by the second step, the first tick's beacon should have arrived"
    assert obs_c_2[3] == 1.0, "by the second step, the first tick's beacon should have arrived"


def test_revealed_position_is_the_senders_position_at_send_time_not_current():
    """Confirms the STALENESS property end-to-end: what generic learns
    about calibrated's position should match calibrated's position AT THE
    TIME IT WAS SENT (an earlier tick), not calibrated's current position
    (which has since moved, since a nonzero action is applied every
    step). This is the actual mechanism behind 'decision-making on stale
    information' — proven here, not just asserted in a docstring."""
    orch = _permissive_orchestrator()
    orch.reset()

    action_generic = np.zeros(6)
    action_calibrated = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # calibrated moves; generic doesn't

    positions_calibrated_over_time = []
    for _ in range(5):
        (obs_g, obs_c), reward, terminated, truncated, info = orch.step(action_generic, action_calibrated)
        positions_calibrated_over_time.append(info["position_calibrated"].copy())

    # By now, generic has received at least one beacon (from the lag test above).
    revealed_to_generic = info["revealed_to_generic"]
    assert revealed_to_generic is not None

    current_calibrated_position = info["position_calibrated"]
    # The revealed position should NOT match the CURRENT (most recent)
    # position, since calibrated has kept moving since that beacon was sent.
    assert not np.allclose(revealed_to_generic, current_calibrated_position), (
        "revealed position matched the CURRENT position exactly — staleness "
        "isn't being modeled, comm is leaking real-time ground truth."
    )
    # It SHOULD match one of the earlier, already-passed positions.
    matches_a_past_position = any(
        np.allclose(revealed_to_generic, p) for p in positions_calibrated_over_time
    )
    assert matches_a_past_position, (
        "revealed position doesn't match ANY of calibrated's actual past positions — "
        "something is corrupting the payload in transit."
    )


def test_generic_and_calibrated_agents_genuinely_diverge():
    """Same divergence proof as A1.2, now run through the FULL
    orchestrator (comm channel included) rather than the bare cores
    directly — confirming the wiring didn't accidentally collapse the
    two agents' distinct physics somewhere in the plumbing."""
    orch = _permissive_orchestrator(max_episode_steps=100)
    orch.reset()
    action = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.3])

    for _ in range(50):
        _, _, terminated, truncated, info = orch.step(action, action)
        if terminated or truncated:
            break

    pos_g = info["position_generic"]
    pos_c = info["position_calibrated"]
    divergence = float(np.linalg.norm(pos_g - pos_c))
    assert divergence > 1e-6, (
        f"generic and calibrated positions did not diverge (divergence={divergence}) "
        f"even under identical actions — something is wrong with the wiring."
    )


def test_reward_uses_ground_truth_separation_not_revealed_position():
    """Confirms the actual reward returned by the orchestrator matches
    what FormationHoldTask.step() would compute directly from the TRUE
    separation reported in info — not something derived from
    revealed_to_generic/revealed_to_calibrated, which may differ or be
    stale/None."""
    from formation_hold_task import FormationHoldTask, FormationHoldConfig
    orch = _permissive_orchestrator()
    orch.reset()
    action_a = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    action_b = np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0])

    _, reward, terminated, truncated, info = orch.step(action_a, action_b)

    reference_task = FormationHoldTask(FormationHoldConfig())
    expected_reward, expected_terminated, _ = reference_task.step(
        separation=info["true_separation"], action_a=action_a, action_b=action_b,
        truncated=truncated, comm_revealed_other_position=None,  # must not matter — already proven
    )
    assert abs(reward - expected_reward) < 1e-9
    assert terminated == expected_terminated


def test_reset_actually_clears_state_between_episodes():
    """A real, deliberate check that reset() doesn't leave stale comm
    messages or task state leaking from one episode into the next —
    the same class of bug the comm channel's own reset() docstring
    warns about."""
    orch = _permissive_orchestrator()
    orch.reset()
    action = np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    for _ in range(10):
        orch.step(action, action)

    obs_g, obs_c = orch.reset()
    assert obs_g[3] == 0.0, "reset() left a stale 'received something' flag set"
    assert obs_c[3] == 0.0, "reset() left a stale 'received something' flag set"
    assert orch.task._consecutive_steps_in_tolerance == 0
    assert np.allclose(orch.core_generic.state.position, [0.0, 0.0, 0.0])
    # calibrated resets to its deliberate starting OFFSET, not the origin —
    # see OrchestratorConfig.initial_separation. Checking against [0,0,0]
    # here would be checking for the exact bug that offset exists to fix.
    assert np.allclose(orch.core_calibrated.state.position,
                        [orch.config.initial_separation, 0.0, 0.0])


def test_delivery_flags_present_and_vary_under_real_packet_loss():
    """New for A1.10: info must expose delivered_to_generic_this_tick /
    delivered_to_calibrated_this_tick as booleans, distinct from the
    existing revealed_to_* fields (which only carry the LAST known
    position, not whether something arrived on this specific tick).
    Uses high, constant packet loss so both True and False actually
    occur in a short run, and checks the two directions aren't just
    mirroring each other (each direction draws its own independent
    Bernoulli loss)."""
    orch = TwoAgentOrchestrator(
        calibrated_params_path=CALIBRATED_PARAMS_PATH,
        packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.7, 0.7),
        bandwidth_config=BandwidthConfig(bytes_per_tick=1e9),
        comm_seed=0,
    )
    orch.reset()
    generic_flags, calibrated_flags = [], []
    for _ in range(15):
        (og, oc), reward, terminated, truncated, info = orch.step(np.zeros(6), np.zeros(6))
        assert isinstance(info["delivered_to_generic_this_tick"], bool)
        assert isinstance(info["delivered_to_calibrated_this_tick"], bool)
        generic_flags.append(info["delivered_to_generic_this_tick"])
        calibrated_flags.append(info["delivered_to_calibrated_this_tick"])
        if terminated or truncated:
            break
    assert any(generic_flags) and not all(generic_flags), \
        "expected a mix of True/False under 70% constant packet loss"
    assert any(calibrated_flags) and not all(calibrated_flags)
    assert generic_flags != calibrated_flags, \
        "the two directions should draw independent losses, not mirror each other"


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