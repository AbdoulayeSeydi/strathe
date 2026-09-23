"""
Fleet v2 tests: belief models, FleetSim, FleetVecEnv.

The two properties the v2 redesign exists for are tested directly:
  - DECENTRALIZATION: a vehicle's observation is unaffected by anything
    its teammate does unless it arrives through the acoustic channel.
  - PAIRING: for a named comm severity, the packet-drop pattern is
    identical across belief conditions, so conditions can be compared
    episode-by-episode (common random numbers).
"""
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from comm_channel import BandwidthConfig, PacketLossConfig
from dynamics_model import CalibrationParameters, SimulationConfig
from fast_dynamics import FastCore
from fleet_belief import (Beacon, BeliefSpec, ConstantVelocityBelief, LastBeaconBelief,
                          OracleBelief, PhysicsBelief, interpolate_params)
from fleet_env import FleetConfig, FleetSim, OBS_DIM
from fleet_vec_env import CommCondition, EpisodeSampler, FleetVecEnv
from formation_hold_task import FormationHoldConfig
from optimizer import initial_parameters, load_vehicle_spec

SPEC = load_vehicle_spec("vehicle_spec.yaml")
VEHICLE = SPEC.vehicle
GENERIC = initial_parameters(SPEC)
with open("calibrated_parameters_a1.yaml", encoding="utf-8") as _f:
    CALIBRATED = CalibrationParameters(**yaml.safe_load(_f)["parameters"])
DT = 0.1
NO_LOSS = (PacketLossConfig(0.0, 1000.0, 0.0, 0.0), BandwidthConfig(1e9))
TOTAL_LOSS = (PacketLossConfig(0.0, 1000.0, 1.0, 1.0), BandwidthConfig(1e9))
L2 = (PacketLossConfig(0.0, 1000.0, 0.3, 0.3), BandwidthConfig(1e9))


def _truth_run(params, tau, n_ticks, nu0=(0.2, 0, 0, 0, 0, 0.05)):
    core = FastCore(params, VEHICLE, SimulationConfig(dt=DT))
    core.reset(position=[1.0, 2.0, 0.5], nu=nu0)
    states = []
    for _ in range(n_ticks):
        core.step(tau)
        states.append((core.position_array(), np.array(core.R), core.nu_array()))
    return states


def _beacon(tick, state, tau):
    pos, R, nu = state
    return Beacon(send_tick=tick, position=pos, orientation=R, nu=nu, tau=np.array(tau, dtype=float))


# ---------------------------------------------------------------- beliefs

def test_no_estimate_before_first_beacon():
    for b in (LastBeaconBelief(DT), ConstantVelocityBelief(DT), OracleBelief(DT),
              PhysicsBelief(DT, GENERIC, VEHICLE)):
        b.on_tick(1, [])
        assert not b.has_estimate and b.estimate(np.zeros(3)) is None and b.age_s == float("inf")


def test_last_beacon_and_age():
    tau = [5, 0, 0, 0, 0, 0]
    s = _truth_run(GENERIC, tau, 5)
    b = LastBeaconBelief(DT)
    b.on_tick(2, [_beacon(1, s[0], tau)])
    np.testing.assert_allclose(b.estimate(), s[0][0])
    assert b.age_s == pytest.approx(0.1)
    b.on_tick(3, [])
    b.on_tick(4, [])
    assert b.age_s == pytest.approx(0.3)
    np.testing.assert_allclose(b.estimate(), s[0][0])


def test_constant_velocity_extrapolates_world_velocity():
    R = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    beacon = Beacon(send_tick=1, position=np.zeros(3), orientation=R,
                    nu=np.array([1.0, 0, 0, 0, 0, 0]), tau=np.zeros(6))
    b = ConstantVelocityBelief(DT)
    b.on_tick(3, [beacon])
    # body surge along +x, vehicle yawed 90 deg -> world +y, 0.2 s elapsed
    np.testing.assert_allclose(b.estimate(), [0.0, 0.2, 0.0], atol=1e-12)


@pytest.mark.parametrize("gap", [1, 3, 10, 30])
def test_physics_belief_with_true_model_is_exact_under_constant_thrust(gap):
    """With the true parameters and an unchanging thrust command, dead
    reckoning must reproduce the teammate's true trajectory exactly,
    however many beacons in a row are lost. This pins down the timing
    convention (beacon describes END of send_tick; propagate to now)."""
    tau = [8, -3, 2, 0.4, -0.2, 0.6]
    s = _truth_run(CALIBRATED, tau, 45)
    b = PhysicsBelief(DT, CALIBRATED, VEHICLE)
    # beacon sent at tick 1 (state index 0), delivered one tick later
    b.on_tick(2, [_beacon(1, s[0], tau)])
    for k in range(3, 3 + gap):
        b.on_tick(k, [])
    now = 2 + gap
    np.testing.assert_allclose(b.estimate(), s[now - 1][0], atol=1e-9)


def test_wrong_model_predicts_worse_than_true_model():
    tau = [15, 0, -5, 0, 1, 2]
    s = _truth_run(CALIBRATED, tau, 30)
    errs = {}
    for name, params in (("true", CALIBRATED), ("generic", GENERIC)):
        b = PhysicsBelief(DT, params, VEHICLE)
        b.on_tick(2, [_beacon(1, s[0], tau)])
        for k in range(3, 31):
            b.on_tick(k, [])
        errs[name] = np.linalg.norm(b.estimate() - s[29][0])
    assert errs["true"] < 1e-9 < errs["generic"]


def test_older_out_of_order_beacon_is_ignored():
    tau = [3, 0, 0, 0, 0, 0]
    s = _truth_run(GENERIC, tau, 10)
    b = LastBeaconBelief(DT)
    b.on_tick(6, [_beacon(5, s[4], tau)])
    b.on_tick(7, [_beacon(2, s[1], tau)])
    np.testing.assert_allclose(b.estimate(), s[4][0])
    assert b.age_s == pytest.approx(0.2)


def test_newest_of_several_simultaneous_beacons_wins():
    tau = [3, 0, 0, 0, 0, 0]
    s = _truth_run(GENERIC, tau, 10)
    b = LastBeaconBelief(DT)
    b.on_tick(8, [_beacon(4, s[3], tau), _beacon(7, s[6], tau), _beacon(5, s[4], tau)])
    np.testing.assert_allclose(b.estimate(), s[6][0])


def test_oracle_returns_truth_and_requires_it():
    tau = [3, 0, 0, 0, 0, 0]
    s = _truth_run(GENERIC, tau, 3)
    b = OracleBelief(DT)
    b.on_tick(2, [_beacon(1, s[0], tau)])
    np.testing.assert_allclose(b.estimate(np.array([9.0, 8.0, 7.0])), [9, 8, 7])
    with pytest.raises(ValueError):
        b.estimate(None)


def test_physics_belief_survives_a_diverging_model():
    b = PhysicsBelief(DT, GENERIC, VEHICLE)
    beacon = Beacon(send_tick=1, position=np.zeros(3), orientation=np.eye(3),
                    nu=np.array([1e6, 0, 0, 0, 0, 0]), tau=np.full(6, 1e9))
    b.on_tick(2, [beacon])
    for k in range(3, 10):
        b.on_tick(k, [])
    assert np.all(np.isfinite(b.estimate()))


def test_interpolate_params():
    np.testing.assert_allclose(interpolate_params(CALIBRATED, GENERIC, 0.0).to_vector(), CALIBRATED.to_vector())
    np.testing.assert_allclose(interpolate_params(CALIBRATED, GENERIC, 1.0).to_vector(), GENERIC.to_vector())
    mid = interpolate_params(CALIBRATED, GENERIC, 0.5)
    np.testing.assert_allclose(mid.to_vector(), 0.5 * (CALIBRATED.to_vector() + GENERIC.to_vector()))
    FastCore(mid, VEHICLE, SimulationConfig(dt=DT))  # mass matrix still positive-definite
    with pytest.raises(ValueError):
        interpolate_params(CALIBRATED, GENERIC, 1.5)


def test_belief_spec_validation():
    with pytest.raises(ValueError):
        BeliefSpec("physics", "no params")
    with pytest.raises(ValueError):
        BeliefSpec("telepathy", "bad kind")
    assert isinstance(BeliefSpec("physics", "p", GENERIC).build(DT, VEHICLE), PhysicsBelief)


# ---------------------------------------------------------------- FleetSim

def _sim(comm=NO_LOSS, belief=None, **cfg_kwargs):
    cfg_kwargs.setdefault("beacon_interval_s", 0.1)  # beacon every tick unless a test says otherwise
    return FleetSim(CALIBRATED, VEHICLE, FleetConfig(**cfg_kwargs), comm[0], comm[1],
                    belief=belief or BeliefSpec("last_beacon", "lb"))


def test_reset_places_vehicles_and_splits_closing_speed():
    sim = _sim()
    sim.reset(comm_seed=0, initial_separation=3.5, closing_speed=0.6)
    np.testing.assert_allclose(sim.cores[0].position_array(), [0, 0, 0])
    np.testing.assert_allclose(sim.cores[1].position_array(), [3.5, 0, 0])
    assert sim.cores[0].nu_array()[0] == pytest.approx(-0.3)
    assert sim.cores[1].nu_array()[0] == pytest.approx(0.3)


def test_random_reset_respects_configured_ranges():
    sim = _sim(initial_separation_range=(3.0, 4.0), initial_closing_speed_range=(0.1, 0.2))
    rng = np.random.default_rng(0)
    for _ in range(50):
        sim.reset(comm_seed=0, rng=rng)
        assert 3.0 <= sim.initial_separation <= 4.0 and 0.1 <= sim.closing_speed <= 0.2


def test_initial_observation_has_no_teammate_information():
    obs = _sim().reset(comm_seed=0, initial_separation=3.0, closing_speed=0.0)
    assert len(obs) == 2
    for o in obs:
        assert o.shape == (OBS_DIM,) and o.dtype == np.float32
        np.testing.assert_array_equal(o[8:11], 0.0)
        assert o[11] == 1.0 and o[12] == 0.0


def test_first_beacon_arrives_one_tick_after_sending():
    sim = _sim()
    sim.reset(comm_seed=0, initial_separation=3.0, closing_speed=0.0)
    obs, *_ = sim.step([np.zeros(6), np.zeros(6)])
    assert obs[0][12] == 0.0
    obs, *_ = sim.step([np.zeros(6), np.zeros(6)])
    assert obs[0][12] == 1.0 and obs[1][12] == 1.0
    # (z differs by ~1 mm: with zero thrust both vehicles drift vertically
    # under their net buoyancy -- real physics, not an observation error)
    np.testing.assert_allclose(obs[0][8:11], [3.0, 0, 0], atol=5e-3)   # teammate ahead
    np.testing.assert_allclose(obs[1][8:11], [-3.0, 0, 0], atol=5e-3)  # teammate behind


def test_observation_is_decentralized():
    """With every beacon lost, vehicle 0's observation must be identical no
    matter what vehicle 1 does -- no side channel from teammate truth."""
    rng = np.random.default_rng(0)
    a0 = rng.uniform(-1, 1, (40, 6))
    runs = []
    for a1_scale in (0.0, 1.0):
        sim = _sim(comm=TOTAL_LOSS)
        o = sim.reset(comm_seed=3, initial_separation=5.0, closing_speed=0.0)
        seq = [o[0]]
        for t in range(40):
            o, _, term, trunc, _ = sim.step([a0[t], a1_scale * np.ones(6)])
            seq.append(o[0])
            if term or trunc:
                break
        runs.append(np.array(seq))
    n = min(len(runs[0]), len(runs[1]))
    np.testing.assert_array_equal(runs[0][:n], runs[1][:n])


def test_teammate_actions_do_reach_observation_through_the_channel():
    """Sanity check that the decentralization test above can fail: with no
    packet loss, vehicle 1's actions DO change vehicle 0's observation."""
    seqs = []
    for a1 in (np.zeros(6), np.ones(6)):
        sim = _sim(comm=NO_LOSS)
        sim.reset(comm_seed=3, initial_separation=5.0, closing_speed=0.0)
        for _ in range(5):
            o, *_ = sim.step([np.zeros(6), a1])
        seqs.append(o[0])
    assert not np.allclose(seqs[0], seqs[1])


def test_oracle_belief_error_is_zero_and_last_beacon_is_not():
    for spec, zero in ((BeliefSpec("oracle", "o"), True), (BeliefSpec("last_beacon", "lb"), False)):
        sim = _sim(comm=L2, belief=spec)
        sim.reset(comm_seed=1, initial_separation=4.0, closing_speed=0.5)
        errs = []
        for _ in range(30):
            _, _, term, trunc, info = sim.step([np.full(6, 0.3), np.full(6, -0.3)])
            errs += [e for e in info["belief_errors"] if e is not None]
        assert (max(errs) < 1e-12) == zero


def test_true_model_physics_belief_is_exact_with_no_loss_and_constant_thrust():
    sim = _sim(comm=NO_LOSS, belief=BeliefSpec("physics", "true", CALIBRATED))
    sim.reset(comm_seed=1, initial_separation=4.0, closing_speed=0.4)
    act = [np.array([0.2, 0.1, 0, 0, 0, 0.1]), np.array([-0.3, 0, 0.1, 0, 0, 0])]
    for _ in range(20):
        _, _, _, _, info = sim.step(act)
    assert max(e for e in info["belief_errors"] if e is not None) < 1e-9


def test_drop_pattern_is_identical_across_belief_conditions():
    """Common random numbers: different belief models -> different
    observations -> different actions, yet the SAME beacons get dropped."""
    patterns = []
    for spec in (BeliefSpec("last_beacon", "lb"), BeliefSpec("oracle", "o"),
                 BeliefSpec("physics", "g", GENERIC)):
        sim = _sim(comm=L2, belief=spec)
        o = sim.reset(comm_seed=11, initial_separation=5.0, closing_speed=0.0)
        delivered = []
        for _ in range(60):
            # an observation-dependent "policy" so trajectories really differ
            acts = [np.tanh(o[i][8:14].sum()) * np.full(6, 0.2) for i in range(2)]
            o, _, term, trunc, info = sim.step(acts)
            delivered.append(tuple(info["delivered"]))
            if term or trunc:
                break
        patterns.append(delivered)
    n = min(map(len, patterns))
    assert n > 10
    assert patterns[0][:n] == patterns[1][:n] == patterns[2][:n]


def test_episode_length_is_configurable():
    sim = _sim(max_episode_steps=7, task=FormationHoldConfig(hold_steps=7))  # starts 3 m out of tolerance
    sim.reset(comm_seed=0, initial_separation=5.0, closing_speed=0.0)
    for t in range(7):
        _, _, term, trunc, _ = sim.step([np.zeros(6), np.zeros(6)])
    assert trunc and not term


def test_config_validation():
    with pytest.raises(ValueError):
        FleetConfig(initial_separation_range=(0.3, 5.0)).validate()  # inside collision radius
    with pytest.raises(ValueError):
        FleetConfig(max_episode_steps=0).validate()
    with pytest.raises(ValueError):
        FleetConfig(dynamics_backend="holoocean").validate()
    with pytest.raises(ValueError):
        _sim().reset(comm_seed=0, initial_separation=0.2)


def test_body_frame_relative_position():
    """Yaw vehicle 0 by 90 deg: a teammate at +x world must appear at -y body."""
    sim = _sim()
    sim.reset(comm_seed=0, initial_separation=3.0, closing_speed=0.0)
    sim.step([np.zeros(6), np.zeros(6)])
    sim.step([np.zeros(6), np.zeros(6)])
    sim.cores[0].R = Rotation.from_euler("z", 90, degrees=True).as_matrix().tolist()
    obs = sim._observations()
    np.testing.assert_allclose(obs[0][8:11], [0.0, -3.0, 0.0], atol=5e-3)  # z: buoyancy drift


def test_mujoco_truth_backend_tracks_stand_in():
    """MuJoCo full-override reproduces the stand-in dynamics (validated in
    Track B); fleet v2 must preserve that when MuJoCo is the truth engine."""
    finals = []
    for backend in ("stand_in", "mujoco"):
        sim = _sim(dynamics_backend=backend)
        sim.reset(comm_seed=0, initial_separation=3.0, closing_speed=0.2)
        for _ in range(20):
            sim.step([np.array([0.3, 0, 0.1, 0, 0, 0.05]), np.array([-0.2, 0.1, 0, 0, 0, 0])])
        finals.append(np.concatenate([c.position_array() for c in sim.cores]))
    np.testing.assert_allclose(finals[0], finals[1], atol=5e-3)


# ---------------------------------------------------------------- VecEnv

def _vec(n_sims=3):
    sampler = EpisodeSampler(
        comm_conditions=[CommCondition("L0", *NO_LOSS), CommCondition("L2", *L2)],
        belief_sampler=lambda rng: BeliefSpec("physics", "g", GENERIC) if rng.random() < 0.5
        else BeliefSpec("last_beacon", "lb"))
    sims = [FleetSim(CALIBRATED, VEHICLE, FleetConfig(max_episode_steps=25, task=FormationHoldConfig(hold_steps=20)))
            for _ in range(n_sims)]
    return FleetVecEnv(sims, sampler, seed=0)


def test_vec_env_shapes_and_shared_reward():
    env = _vec()
    obs = env.reset()
    assert obs.shape == (6, OBS_DIM) and env.num_envs == 6
    obs, rew, done, infos = env.step(np.zeros((6, 6)))
    assert obs.shape == (6, OBS_DIM) and rew.shape == (6,) and len(infos) == 6
    for k in range(3):
        assert rew[2 * k] == rew[2 * k + 1] and done[2 * k] == done[2 * k + 1]


def test_vec_env_auto_resets_and_reports_terminal_observation():
    env = _vec()
    env.reset()
    for _ in range(25):
        obs, rew, done, infos = env.step(np.zeros((6, 6)))
    assert done.all()
    for info in infos:
        assert info["terminal_observation"].shape == (OBS_DIM,)
        assert info["TimeLimit.truncated"]
    np.testing.assert_array_equal(obs[:, 12], 0.0)  # fresh episode: no beacon yet


def test_ppo_trains_on_vec_env_without_error():
    from stable_baselines3 import PPO
    model = PPO("MlpPolicy", _vec(), n_steps=32, batch_size=64, n_epochs=1, verbose=0, device="cpu")
    model.learn(total_timesteps=256)
    action, _ = model.predict(np.zeros(OBS_DIM, dtype=np.float32), deterministic=True)
    assert action.shape == (6,)


# ---------------------------------------------------------------- beacon interval, collision reward

def test_beacons_are_sent_only_every_beacon_interval():
    sim = _sim(beacon_interval_s=1.0)
    sim.reset(comm_seed=0, initial_separation=4.0, closing_speed=0.0)
    arrivals = []
    for t in range(1, 36):
        _, _, _, _, info = sim.step([np.zeros(6), np.zeros(6)])
        if info["delivered"][0]:
            arrivals.append(t)
    assert arrivals == [11, 21, 31]  # sent at ticks 10, 20, 30; one tick of latency


def test_belief_age_grows_between_sparse_beacons():
    sim = _sim(beacon_interval_s=2.0)
    sim.reset(comm_seed=0, initial_separation=4.0, closing_speed=0.0)
    ages = []
    for _ in range(40):
        obs, *_ = sim.step([np.zeros(6), np.zeros(6)])
        ages.append(obs[0][11] * sim.cfg.age_cap_s)
    # beacon sent at tick 20 arrives at 21 (age 0.1 s), then ages 0.1 s per tick
    assert ages[20] == pytest.approx(0.1) and ages[38] == pytest.approx(1.9)


def test_beacon_interval_must_be_whole_ticks():
    with pytest.raises(ValueError):
        FleetConfig(beacon_interval_s=0.15).validate()
    with pytest.raises(ValueError):
        FleetConfig(beacon_interval_s=0.0).validate()
    assert FleetConfig(beacon_interval_s=3.0).beacon_every_ticks == 30


def test_hold_steps_longer_than_episode_is_rejected():
    with pytest.raises(ValueError):
        FleetConfig(max_episode_steps=10, task=FormationHoldConfig(hold_steps=20)).validate()


def _crash(absorbing):
    sim = _sim(absorbing_collision_penalty=absorbing)
    sim.reset(comm_seed=0, initial_separation=1.0, closing_speed=-3.0)
    for t in range(1, 200):
        _, r, term, trunc, info = sim.step([np.array([1, 0, 0, 0, 0, 0]), np.array([-1, 0, 0, 0, 0, 0])])
        if term:
            return t, r, info
    raise AssertionError("expected a collision")


def test_collision_charges_the_rest_of_the_episode():
    t, r_abs, info = _crash(True)
    assert info["terminal_reason"] == "collision"
    _, r_plain, _ = _crash(False)
    expected_tail = (200 - t) * 1.0 * 2.0  # remaining steps * reward_sep_weight * target_separation
    assert r_plain - r_abs == pytest.approx(expected_tail)
    assert info["reward_components"]["collision_tail"] == pytest.approx(-expected_tail)


def test_crashing_early_never_beats_surviving():
    """The exploit this penalty closes: with it, crashing on tick 1 must
    score worse than drifting at the worst in-range separation all episode."""
    t, r_crash, _ = _crash(True)
    worst_survival = -200 * 1.0 * (5.5 - 2.0)  # max initial range, never corrected
    assert r_crash < worst_survival or t > 1
    assert r_crash <= -(200 - t) * 2.0
