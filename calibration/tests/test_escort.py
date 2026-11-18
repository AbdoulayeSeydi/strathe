"""Heterogeneous escort study (escort_remus.py, escort_env.py, run_escort_study.py)."""
from __future__ import annotations

import dataclasses
import math
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CAL, "stress"))

import escort_env as EE  # noqa: E402
import escort_remus as ER  # noqa: E402
import prestero_remus as PR  # noqa: E402
from fleet_belief import Beacon  # noqa: E402

SHORT = ER.MissionConfig(duration_s=75.0)


@pytest.fixture(scope="module", autouse=True)
def _in_calibration_dir():
    old = os.getcwd()
    os.chdir(CAL)                      # model files are referenced relative to calibration/
    yield
    os.chdir(old)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return ER.MissionLibrary.build([7, 8, 9], SHORT, cache_dir=str(tmp_path_factory.mktemp("missions")))


@pytest.fixture(scope="module")
def models(_in_calibration_dir):
    return EE.load_rov(), EE.load_remus_models()


def _sim(lib, models, **cfg):
    rov, remus = models
    return EE.EscortSim(rov, remus, lib, EE.EscortConfig(**cfg))


# ------------------------------------------------------------------ the REMUS truth

def test_autopilot_is_fin_truths():
    import fin_truth
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.uniform(-1, 1, 7)
        ours = ER.autopilot(a[0] * .3, a[1], a[2] * 4, a[3], a[4] * 4, a[5] * .3, a[6] * .4, .01)
        theirs = fin_truth.prestero_autopilot(a[0] * .3, a[1], a[2] * 4, a[3], a[4] * 4, a[5] * .3, a[6] * .4, .01)
        assert ours == theirs


def test_reballasting_is_a_smaller_B_in_presteros_equations(monkeypatch):
    rng = np.random.default_rng(1)
    x = np.r_[rng.normal(0, .3, 6), 0, 0, 10, rng.normal(0, .2, 3)]
    assert np.array_equal(ER.derivative(x, .05, -.03, 700, 0.0), PR.derivative(x, .05, -.03, 700))
    ours = ER.derivative(x, .05, -.03, 700, -6.0)
    monkeypatch.setattr(PR, "B", PR.B - 6.0)
    assert np.allclose(ours, PR.derivative(x, .05, -.03, 700), atol=1e-12)


def test_missions_are_reproducible_and_fly_a_survey():
    a, b = ER.fly_mission(3, SHORT), ER.fly_mission(3, SHORT)
    assert np.array_equal(a.pos, b.pos) and np.array_equal(a.fins, b.fins)
    assert not np.array_equal(a.pos, ER.fly_mission(4, SHORT).pos)
    m = ER.fly_mission(5)                                       # the default 120 s mission
    speed = np.linalg.norm(m.nu[:, :3], axis=1)
    assert 0.45 < np.median(speed) < 0.75 and speed.max() < 0.8
    assert np.abs(m.pos[:, 2] - 10.0).max() < 1.0                # holds depth
    assert 0.1 < np.mean([m.turning(k) for k in range(len(m))]) < 0.8
    assert np.abs(m.fins[:, :2]).max() <= ER.FIN_LIMIT_RAD + 1e-12
    v_world = np.array([m.world_velocity(k) for k in range(len(m))])   # R is body->world: it moves where it points
    assert np.allclose(np.diff(m.pos, axis=0) / SHORT.dt, (v_world[1:] + v_world[:-1]) / 2, atol=0.02)


def test_library_round_trip_and_refuses_another_config(lib, tmp_path):
    p = str(tmp_path / "m.npz")
    lib.save(p)
    back = ER.MissionLibrary.load(p, SHORT)
    assert len(back) == 3 and np.array_equal(back[1].R, lib[1].R) and back[2].seed == 9
    with pytest.raises(ValueError, match="built for"):
        ER.MissionLibrary.load(p, ER.MissionConfig())


# ------------------------------------------------------------------ models of the REMUS

def test_belief_parameters(models):
    _, remus = models
    raw = EE.load_remus_models(buoyancy_change_n=0.0)
    assert remus.calibrated.net_buoyancy_z == pytest.approx(raw.calibrated.net_buoyancy_z + ER.BUOYANCY_CHANGE_N)
    assert remus.generic.net_buoyancy_z == 0.0                  # describe-vehicle's neutral guess, untouched
    assert remus.vehicle.actuators is not None and len(remus.vehicle.actuators.fins) == 4
    mid = remus.belief("physics_alpha0.5").params
    assert mid.Nrr == pytest.approx((remus.calibrated.Nrr + remus.generic.Nrr) / 2)


def _predict(remus, params, m, k, ticks):
    from dynamics_model import SimulationConfig
    from fast_dynamics import FastCore
    from scipy.spatial.transform import Rotation
    c = FastCore(params, remus.vehicle, SimulationConfig(dt=.1, physics_dt=.005))
    c.reset(position=m.pos[k], nu=m.nu[k], orientation=Rotation.from_matrix(m.R[k]))
    c.controls = list(ER.controls_for(remus.basis, *m.fins[k]))
    for _ in range(ticks):
        c.step([0.0] * 6)
    return float(np.linalg.norm(c.position_array() - m.pos[k + ticks]))


def test_calibrated_model_predicts_the_remus_better_than_generic(lib, models):
    """The physics the study rests on, signs included: a sign slip in the fin mapping would turn the model the wrong way
    and lose to the generic one in turns."""
    _, remus = models
    m = lib[0]
    ks = range(0, len(m) - 31, 9)
    cal1 = [_predict(remus, remus.calibrated, m, k, 10) for k in ks]
    cal3 = [_predict(remus, remus.calibrated, m, k, 30) for k in ks]
    gen3 = [_predict(remus, remus.generic, m, k, 30) for k in ks]
    cv3 = [float(np.linalg.norm(m.pos[k] + m.world_velocity(k) * 3.0 - m.pos[k + 30])) for k in ks]
    assert np.median(cal1) < 0.03
    assert np.median(cal3) < 0.5 * np.median(gen3)
    assert np.median(cal3) < np.median(cv3)
    turning = [k for k in ks if m.turning(k)]
    assert turning, "the mission window should contain a turn"
    assert np.median([_predict(remus, remus.calibrated, m, k, 30) for k in turning]) < \
        np.median([float(np.linalg.norm(m.pos[k] + m.world_velocity(k) * 3.0 - m.pos[k + 30])) for k in turning])


def test_physics_belief_holds_a_fin_beacons_controls(models, lib):
    _, remus = models
    b = remus.belief("calibrated").build(.1, remus.vehicle)
    m = lib[0]
    ctl = ER.controls_for(remus.basis, *m.fins[20])
    bc = Beacon(send_tick=0, position=m.pos[20], orientation=m.R[20], nu=m.nu[20], tau=np.zeros(6), controls=ctl)
    b.on_tick(1, [bc])
    assert b._core.controls == [float(c) for c in ctl]
    b.on_tick(2, [])
    assert b._core.controls == [float(c) for c in ctl]
    hover = Beacon(send_tick=5, position=np.zeros(3), orientation=np.eye(3), nu=np.zeros(6), tau=np.zeros(6))
    assert hover.controls is None


# ------------------------------------------------------------------ the escort simulation

def test_severities_match_the_fleets():
    try:
        from evaluate_comm_degradation_curve import COMM_CONFIGS
    except ImportError:
        pytest.skip("gymnasium isn't installed here (the fleet module imports it)")
    for k, (plc, bwc) in EE.SEVERITIES.items():
        assert plc == COMM_CONFIGS[k]["packet_loss_config"] and bwc == COMM_CONFIGS[k]["bandwidth_config"]


def test_action_goes_through_the_bluerov2_heavys_thrusters(models):
    rov, _ = models
    assert rov.scale[0] == pytest.approx(105.2, abs=0.5) and rov.scale[3] == pytest.approx(28.1, abs=0.5)


def test_reset_starts_the_rov_at_its_offset_moving_with_the_remus(lib, models):
    sim = _sim(lib, models)
    sim.set_belief(models[1].belief("oracle"))
    ep = EE.EscortEpisode(1, 50, (3.0, -1.0, 0.2), 11)
    obs = sim.reset(ep)
    p, v, _ = sim.remus_true()
    assert np.allclose(sim.core.position_array(), p + np.array(ep.offset))
    assert np.allclose(sim.core.world_velocity(), v, atol=1e-9)
    assert np.allclose(obs[11:14], 0.0, atol=1e-6)                # no relative velocity at launch
    assert np.linalg.norm(obs[8:11]) == pytest.approx(np.linalg.norm(ep.offset), rel=1e-6)
    assert obs.shape == (EE.OBS_DIM,) and obs[15] == 1.0


def test_episodes_are_paired_across_models_and_actions(lib, models):
    """The REMUS's motion and the beacons that get through don't depend on the teammate model or on what the ROV does."""
    sim = _sim(lib, models)
    sim.set_comm(EE.CommCondition("L3a", 1.0))
    ep = EE.EscortEpisode(0, 10, (3.0, 0.0, 0.0), 5)
    rng = np.random.default_rng(0)
    seen = []
    for belief, act in (("oracle", lambda: np.zeros(4)), ("last_beacon", lambda: rng.uniform(-.3, .3, 4)),
                        ("calibrated", lambda: np.zeros(4))):
        sim.set_belief(models[1].belief(belief))
        sim.reset(ep)
        trace = []
        for _ in range(150):
            _, _, term, trunc, info = sim.step(act())
            trace.append((info["delivered"], tuple(sim.remus_true()[0])))
            if term or trunc:
                break
        seen.append(trace)
    n = min(len(t) for t in seen)
    assert seen[0][:n] == seen[1][:n] == seen[2][:n]
    assert 0 < sum(d for d, _ in seen[0]) < 150 * 0.7         # L3a loses some beacons


def test_oracle_error_is_zero_and_last_beacon_error_grows(lib, models):
    sim = _sim(lib, models)
    sim.set_comm(EE.CommCondition("L0", 3.0))
    pol = EE.EscortPursuit(models[0])
    ep = EE.EscortEpisode(2, 0, (3.0, 0.0, 0.0), 1)
    sim.set_belief(models[1].belief("oracle"))
    assert EE.run_episode(sim, pol, ep)["mean_belief_error_m"] == 0.0
    sim.set_belief(models[1].belief("last_beacon"))
    obs = sim.reset(ep)
    errs = []
    for _ in range(29):
        obs, _, _, _, info = sim.step(np.zeros(4))
        errs.append(info["belief_error"])
    assert errs[0] < errs[-1] and errs[-1] > 1.0                 # ~0.7 m/s for 2.9 s since the launch handover


def test_collision_ends_the_episode_and_charges_the_rest(lib, models):
    sim = _sim(lib, models)
    sim.set_belief(models[1].belief("oracle"))
    ep = EE.EscortEpisode(0, 0, (1.3, 0.0, 0.0), 1)
    sim.reset(ep)
    term, reward = False, 0.0
    while not term:
        _, reward, term, trunc, info = sim.step(np.array([1.0, 0, 0, 0]) * np.sign(sim.remus_true()[0][0] - sim.core.position_array()[0]))
        assert not trunc
    assert info["terminal_reason"] == "collision"
    remaining = sim.cfg.episode_steps - sim.tick
    assert reward < -remaining * sim.cfg.worst_tick_cost + 1e-9
    s = sim.summary()
    assert s["success"] == 0 and s["collision"] == 1
    assert s["in_band_frac"] < 1.0                                # ticks lost to the early end count as out of band
    with pytest.raises(RuntimeError):
        sim.step(np.zeros(4))


def test_the_task_is_feasible(lib, models):
    """With perfect knowledge the scripted escort holds the band: the REMUS is within the BlueROV2's reach."""
    sim = _sim(lib, models)
    sim.set_belief(models[1].belief("oracle"))
    pol = EE.EscortPursuit(models[0])
    eps = EE.evaluation_episodes(lib, sim.cfg, per_mission=1, seed=3)
    res = [EE.run_episode(sim, pol, e) for e in eps]
    assert all(r["success"] for r in res)
    assert all(r["max_rov_speed_mps"] < 1.0 for r in res)


def test_degraded_comms_separate_the_models(lib, models):
    """The scripted escort at 60% loss and 3 s beacons: calibrated beats generic and last-beacon on belief error."""
    sim = _sim(lib, models)
    sim.set_comm(EE.CommCondition("L3a", 3.0))
    pol = EE.EscortPursuit(models[0])
    eps = EE.evaluation_episodes(lib, sim.cfg, per_mission=2, seed=4)
    err = {}
    for b in ("calibrated", "generic", "last_beacon"):
        sim.set_belief(models[1].belief(b))
        err[b] = np.mean([EE.run_episode(sim, pol, e)["mean_belief_error_m"] for e in eps])
    assert err["calibrated"] < err["generic"] < err["last_beacon"]


def test_config_validation():
    with pytest.raises(ValueError):
        EE.EscortConfig(collision_m=2.5).validate()
    with pytest.raises(ValueError):
        EE.EscortConfig(start_range_m=(1.0, 4.0)).validate()
    EE.EscortConfig().validate()


def test_beacon_interval_must_be_whole_ticks(lib, models):
    sim = _sim(lib, models)
    with pytest.raises(ValueError):
        sim.set_comm(EE.CommCondition("L0", 0.25))
    with pytest.raises(ValueError):
        sim.set_comm(EE.CommCondition("L9", 1.0))


# ------------------------------------------------------------------ the study runner

def test_study_runs_and_reports(lib, models, tmp_path):
    import run_escort_study as RS
    RS._W.update(lib=lib, rov=models[0], remus=models[1], cfg=EE.EscortConfig(), policies={})
    eps = EE.evaluation_episodes(lib, EE.EscortConfig(), per_mission=1, seed=5)
    rows = []
    for b in ("oracle", "calibrated", "physics_alpha0.5", "generic", "last_beacon"):
        rows += RS.run_cell(("scripted", b, "L3a@3", eps))
    assert len(rows) == 5 * len(eps) and {r["comm"] for r in rows} == {"L3a@3s"}
    RS.analyse(rows, str(tmp_path))
    rep = (tmp_path / "report.md").read_text()
    for s in ("Success rate", "straight legs / turns", "calibrated - generic", "Dose-response"):
        assert s in rep
    assert (tmp_path / "contrasts.csv").exists() and (tmp_path / "summary_pooled.csv").exists()
    assert RS.parse_comm("L2@3").beacon_interval_s == 3.0 and RS.parse_comm("L3a@1s").severity == "L3a"


def test_gym_env(lib, models):
    pytest.importorskip("gymnasium")
    from escort_gym import EscortGymEnv
    import train_escort_policy as T
    env = EscortGymEnv(_sim(lib, models), T.belief_sampler(models[1]), T.comm_conditions(), seed=0)
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs) and "belief" in info
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (EE.OBS_DIM,) and isinstance(r, float)


# ------------------------------------------------------------------ what the beacon carries (escort_beliefs.py)

def _first_beacon(sim, lib, k=30):
    sim.reset(EE.EscortEpisode(0, k, (3.0, 0.0, 0.0), 1))
    return sim._beacon(0)


def test_beacons_carry_fins_and_orders(lib, models):
    sim = _sim(lib, models)
    b = _first_beacon(sim, lib)
    m = lib[0]
    assert b.fins == tuple(float(x) for x in m.fins[30])
    assert b.intent == (float(m.heading_order[30]), SHORT.depth_m)


def test_intent_model_reproduces_the_reported_elevator(lib, models):
    """The pitch integrator is set so the autopilot law gives back the elevator angle in the beacon."""
    sim = _sim(lib, models)
    remus = models[1]
    b = _first_beacon(sim, lib, k=100)
    bel = remus.belief("calibrated_intent").build(.1, remus.vehicle)
    bel.on_tick(0, [b])
    pitch, q, yaw, r, z = bel._state()
    th_d = float(np.clip(ER.DEPTH_GAIN * (z - b.intent[1]), -ER.PITCH_ORDER_LIMIT_RAD, ER.PITCH_ORDER_LIMIT_RAD))
    ds, dr, _ = ER.autopilot(pitch, q, yaw, r, b.intent[0], th_d, bel._ii, ER.AUTOPILOT_H)
    if abs(b.fins[1]) < ER.FIN_LIMIT_RAD - 1e-6 and abs(bel._ii) < .5:
        assert ds == pytest.approx(b.fins[1], abs=1e-3)   # one integrator step (|e_th| h) apart: the law integrates first
    assert dr == pytest.approx(b.fins[0], abs=1e-3)       # same law, same state: the same rudder


def test_no_fins_model_centres_the_fins_and_keeps_the_rpm(lib, models):
    sim = _sim(lib, models)
    remus = models[1]
    b = _first_beacon(sim, lib)
    bel = remus.belief("calibrated_nofins").build(.1, remus.vehicle)
    bel.on_tick(0, [b])
    assert bel._core.controls == pytest.approx(list(ER.controls_for(remus.basis, 0.0, 0.0, b.fins[2])))


def test_intent_beats_holding_the_fins_over_long_gaps(lib, models):
    """The predicted REMUS 7 s after a beacon, no beacon in between: flying the autopilot toward the orders stays
    closer than coasting on frozen fins."""
    sim = _sim(lib, models)
    remus = models[1]
    err = {"calibrated": [], "calibrated_intent": []}
    for k in range(0, len(lib[1]) - 80, 25):
        sim.reset(EE.EscortEpisode(1, 0, (3.0, 0.0, 0.0), 1))
        b = sim._beacon(k)
        b = dataclasses.replace(b, send_tick=0)
        for name in err:
            bel = remus.belief(name).build(.1, remus.vehicle)
            bel.on_tick(0, [b])
            for t in range(1, 71):
                bel.on_tick(t, [])
            err[name].append(float(np.linalg.norm(bel.estimate(None) - lib[1].pos[k + 70])))
    assert np.median(err["calibrated_intent"]) < 0.7 * np.median(err["calibrated"])


def test_report_has_the_beacon_contrasts(lib, models, tmp_path):
    import run_escort_study as RS
    RS._W.update(lib=lib, rov=models[0], remus=models[1], cfg=EE.EscortConfig(), policies={})
    eps = EE.evaluation_episodes(lib, EE.EscortConfig(), per_mission=1, seed=6)
    first = []
    for b in ("oracle", "calibrated", "generic"):
        first += RS.run_cell(("scripted", b, "L3a@3", eps))
    RS._write_csv(str(tmp_path / "first.csv"), first)
    back = RS.read_episodes(str(tmp_path / "first.csv"))
    assert [r["success"] for r in back] == [r["success"] for r in first]
    assert back[0]["mean_belief_error_m"] == pytest.approx(first[0]["mean_belief_error_m"], rel=1e-5)
    rows = back
    for b in RS.BEACON_BELIEFS:
        rows += RS.run_cell(("scripted", b, "L3a@3", eps))
    RS.analyse(rows, str(tmp_path))
    rep = (tmp_path / "report.md").read_text()
    assert "What the beacon carries" in rep and "calibrated_intent - calibrated" in rep
