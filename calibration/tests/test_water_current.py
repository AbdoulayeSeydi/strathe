"""Water current physics (dynamics_model._relative_to_water) in every engine."""
from dataclasses import replace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from arena_core import ArenaCore
from dynamics_model import (CalibrationParameters, SimulationConfig, TrajectoryInputs, compute_acceleration,
                            compute_acceleration_from_rotation_matrix, simulate_trajectory)
from fast_dynamics import FastCore, FastDynamics
from optimizer import load_vehicle_spec

SPEC = load_vehicle_spec("vehicle_spec.yaml")
VEHICLE = SPEC.vehicle
with open("calibrated_parameters_fleet_truth.yaml", encoding="utf-8") as _f:
    TRUTH = CalibrationParameters(**yaml.safe_load(_f)["parameters"])
# The truth vehicle is slightly heavy, so it also sinks at its terminal velocity; the
# kinematic checks below use a neutrally buoyant copy so only the current moves it.
NEUTRAL = replace(TRUTH, net_buoyancy_z=0.0)
SC = SimulationConfig(dt=0.1)
VC = np.array([0.25, -0.15, 0.0])          # world-frame current, m/s


def _cores():
    cores = {"stand_in (FastCore)": FastCore(NEUTRAL, VEHICLE, SC), "ArenaCore": ArenaCore(NEUTRAL, VEHICLE, SC)}
    try:
        from mujoco_backend import MuJoCoArenaCore
        cores["MuJoCo"] = MuJoCoArenaCore(NEUTRAL, VEHICLE, SC)
    except ImportError:
        pass
    return cores


def _world_velocity(core):
    st = core.state
    return st.orientation.apply(st.nu[:3])


@pytest.mark.parametrize("name", list(_cores()))
def test_an_unpowered_vehicle_ends_up_drifting_with_the_current(name):
    core = _cores()[name]
    core.reset()
    core.current_world = VC
    for _ in range(400):                      # 40 s
        core.step(np.zeros(6))
    assert np.allclose(_world_velocity(core), VC, atol=5e-3)


@pytest.mark.parametrize("name", list(_cores()))
def test_a_spinning_vehicle_moving_with_the_water_stays_with_it(name):
    """Zero relative velocity + yawing: no hydrodynamic force, so the world
    velocity must stay equal to the current. Fails without the -omega x c_b term."""
    core = _cores()[name]
    r = 0.6                                   # rad/s yaw rate
    core.reset(nu=np.array([*VC, 0.0, 0.0, r]))
    core.current_world = VC
    worst = 0.0
    for _ in range(50):                       # the spin decays within ~5 s, so check every step
        core.step(np.array([0, 0, 0, 0, 0, 0.0]))
        worst = max(worst, np.linalg.norm(_world_velocity(core) - VC))
    assert worst < 1e-3                       # ~2.5e-5 with the term, ~0.024 m/s without it


def test_holding_station_takes_exactly_the_drag_of_the_current():
    # neutral buoyancy and no surge/heave-pitch added-mass coupling, so surge is decoupled
    p = replace(NEUTRAL, Xq_dot=0.0, Zq_dot=0.0)
    cb = np.array([0.3, 0.0, 0.0])
    # at rest over ground in a 0.3 m/s current along +x, the water moves past at v_r = -0.3,
    # and holding station takes a thrust equal and opposite to that drag
    tau_x = -(p.Xu * -0.3 + p.Xuu * 0.3 * -0.3)
    vdot, _, _ = compute_acceleration(p, VEHICLE, np.zeros(6), np.array([tau_x, 0, 0, 0, 0, 0]), 0.0, 0.0,
                                      current_body=cb)
    assert np.allclose(vdot, 0.0, atol=1e-12)
    # without that thrust the vehicle accelerates downstream
    vdot0, _, _ = compute_acceleration(p, VEHICLE, np.zeros(6), np.zeros(6), 0.0, 0.0, current_body=cb)
    assert vdot0[0] > 0


def test_zero_current_is_bit_identical_to_still_water():
    rng = np.random.default_rng(0)
    for _ in range(20):
        v, tau = rng.normal(size=6) * 0.3, rng.normal(size=6) * 10
        a = compute_acceleration(TRUTH, VEHICLE, v, tau, 0.1, -0.2)[0]
        assert np.array_equal(a, compute_acceleration(TRUTH, VEHICLE, v, tau, 0.1, -0.2, current_body=np.zeros(3))[0])
        assert np.array_equal(a, compute_acceleration(TRUTH, VEHICLE, v, tau, 0.1, -0.2, current_body=None)[0])


def test_fast_and_reference_dynamics_agree_in_a_current():
    fd = FastDynamics(TRUTH, VEHICLE)
    rng = np.random.default_rng(1)
    for _ in range(50):
        v, tau, cb = rng.normal(size=6) * 0.3, rng.normal(size=6) * 10, rng.normal(size=3) * 0.2
        roll, pitch = rng.normal(size=2) * 0.2
        ref = compute_acceleration(TRUTH, VEHICLE, v, tau, roll, pitch, current_body=cb)[0]
        assert np.allclose(fd.acceleration_current(list(v), list(tau), roll, pitch, list(cb)), ref, atol=1e-10)
        R = Rotation.from_euler("xyz", [roll, pitch, rng.normal()]).as_matrix()
        via_R = compute_acceleration_from_rotation_matrix(TRUTH, VEHICLE, v, tau, R, current_world=R @ cb)[0]
        assert np.allclose(via_R, ref, atol=1e-10)


def test_teacher_forced_replay_fast_path_matches_reference_in_a_current():
    rng = np.random.default_rng(2)
    n = 200
    traj = TrajectoryInputs(nu0=np.zeros(6), forces_torques=rng.normal(size=(n, 6)) * 5,
                            roll_pitch_series=rng.normal(size=(n, 2)) * 0.05,
                            current_body_series=np.tile([0.2, -0.1, 0.0], (n, 1)))
    fast = simulate_trajectory(TRUTH, VEHICLE, SC, traj).nu
    ref = simulate_trajectory(TRUTH, VEHICLE, SimulationConfig(dt=0.1, debug=True), traj).nu
    assert np.allclose(fast, ref, atol=1e-9)
    still = simulate_trajectory(TRUTH, VEHICLE, SC, TrajectoryInputs(traj.nu0, traj.forces_torques,
                                                                     traj.roll_pitch_series)).nu
    assert not np.allclose(fast, still, atol=1e-3)
    with pytest.raises(ValueError):
        TrajectoryInputs(np.zeros(6), traj.forces_torques, traj.roll_pitch_series, np.zeros((n, 2))).validate()


def test_mujoco_and_stand_in_agree_in_a_current():
    pytest.importorskip("mujoco")
    from mujoco_backend import MuJoCoArenaCore
    a, b = FastCore(TRUTH, VEHICLE, SC), MuJoCoArenaCore(TRUTH, VEHICLE, SC)
    for c in (a, b):
        c.reset()
        c.current_world = VC
    rng = np.random.default_rng(3)
    for _ in range(100):
        tau = rng.normal(size=6) * np.array([20, 20, 20, 2, 2, 3])
        a.step(tau)
        b.step(tau)
    assert np.allclose(a.state.position, b.state.position, atol=0.02)
    assert np.allclose(a.state.nu, b.state.nu, atol=0.01)


# ----------------------------------------------------------------- configuration

from current import CurrentConfig, WaterCurrent, current_from_spec  # noqa: E402


@pytest.mark.parametrize("bad", [dict(speed_mps=-0.1), dict(speed_mps=(0.3, 0.1)), dict(speed_mps=5.0),
                                 dict(time_scale_s=0.1), dict(speed_std_mps=-1), dict(vertical_mps=2.0),
                                 dict(direction_deg=[1, 2, 3])])
def test_invalid_current_configs_are_rejected(bad):
    with pytest.raises(ValueError):
        CurrentConfig(**bad).validate()


def test_from_dict_and_describe():
    cfg = CurrentConfig.from_dict({"speed_mps": [0.0, 0.3], "direction_deg": 90,
                                   "variability": {"speed_std_mps": 0.05, "time_scale_s": 20}})
    assert cfg.active and cfg.speed_std_mps == 0.05 and cfg.time_scale_s == 20
    assert "0-0.3 m/s (drawn per episode)" in cfg.describe() and "toward 90 deg" in cfg.describe()
    assert not CurrentConfig().active
    with pytest.raises(ValueError, match="variability"):
        CurrentConfig.from_dict({"variability": {"gusts": 1}})


def test_constant_current_vector_and_direction_convention():
    wc = WaterCurrent(CurrentConfig(speed_mps=0.2, direction_deg=90, vertical_mps=0.05), 0.1, seed=0)
    assert np.allclose(wc.next(), [0.0, 0.2, 0.05], atol=1e-12)     # toward +y
    assert np.allclose(wc.mean_world, [0.0, 0.2, 0.05], atol=1e-12)


def test_ranges_are_drawn_per_episode_and_reproducibly():
    cfg = CurrentConfig(speed_mps=(0.1, 0.3), direction_deg=(0, 360))
    draws = [WaterCurrent(cfg, 0.1, seed=s) for s in range(200)]
    speeds = np.array([d.mean_speed for d in draws])
    assert speeds.min() >= 0.1 and speeds.max() <= 0.3 and speeds.std() > 0.04
    assert WaterCurrent(cfg, 0.1, seed=7).mean_speed == WaterCurrent(cfg, 0.1, seed=7).mean_speed


def test_fluctuation_has_the_configured_spread_and_time_scale():
    cfg = CurrentConfig(speed_mps=0.5, direction_deg=0, speed_std_mps=0.05, direction_std_deg=10, time_scale_s=20)
    wc = WaterCurrent(cfg, 0.1, seed=3)
    x = np.array([wc.next() for _ in range(300_000)])
    speed = np.linalg.norm(x[:, :2], axis=1)
    heading = np.degrees(np.arctan2(x[:, 1], x[:, 0]))
    assert speed.mean() == pytest.approx(0.5, abs=0.01) and speed.std() == pytest.approx(0.05, rel=0.1)
    assert heading.std() == pytest.approx(10, rel=0.1)
    lag = int(20 / 0.1)
    assert np.corrcoef(speed[:-lag], speed[lag:])[0, 1] == pytest.approx(np.exp(-1), abs=0.05)


def test_vehicle_spec_environment_is_still_water_for_the_tank_dataset():
    assert current_from_spec(SPEC) is None


# ----------------------------------------------------------------- fleet

from fleet_belief import BeliefSpec  # noqa: E402
from fleet_env import FleetConfig, FleetSim  # noqa: E402


def _drift(cfg_current, belief=None, steps=60, known=True):
    cur = None if cfg_current is None else CurrentConfig(**cfg_current, known_to_beliefs=known)
    sim = FleetSim(NEUTRAL, VEHICLE, FleetConfig(current=cur, beacon_interval_s=3.0),
                   belief=belief or BeliefSpec("oracle", "oracle"))
    sim.reset(comm_seed=5, initial_separation=3.0, closing_speed=0.0)
    out = []
    for _ in range(steps):
        _, _, term, trunc, info = sim.step([np.zeros(6), np.zeros(6)])
        out.append(info)
        if term or trunc:
            break
    return sim, out


def test_both_vehicles_drift_together_in_the_same_water():
    sim, infos = _drift({"speed_mps": 0.2, "direction_deg": 45})
    p0, p1 = infos[-1]["positions"]
    start_sep = 3.0
    assert np.linalg.norm(p0 - p1) == pytest.approx(start_sep, abs=1e-6)       # same water, same drift
    assert np.linalg.norm(p0[:2]) > 0.5                                          # ~0.2 m/s for 6 s
    assert np.allclose(infos[-1]["current_world"], [0.2 * np.cos(np.pi / 4), 0.2 * np.sin(np.pi / 4), 0])
    _, still = _drift(None)
    assert np.linalg.norm(still[-1]["positions"][0][:2]) < 1e-6 and still[-1]["current_world"] is None


def test_currents_are_paired_across_belief_models():
    a = _drift({"speed_mps": (0.0, 0.3), "direction_deg": (0, 360)}, BeliefSpec("oracle", "oracle"))[1]
    b = _drift({"speed_mps": (0.0, 0.3), "direction_deg": (0, 360)},
               BeliefSpec("last_beacon", "last_beacon"))[1]
    for x, y in zip(a, b):
        assert np.array_equal(x["current_world"], y["current_world"])
        assert all(np.array_equal(p, q) for p, q in zip(x["positions"], y["positions"]))


def test_a_physics_teammate_model_that_knows_the_current_predicts_drift():
    spec = BeliefSpec("physics", "calibrated", NEUTRAL)
    known = _drift({"speed_mps": 0.25, "direction_deg": 0}, spec, known=True)[1]
    unknown = _drift({"speed_mps": 0.25, "direction_deg": 0}, spec, known=False)[1]
    err = lambda infos: np.nanmean([e for i in infos for e in i["belief_errors"] if e is not None])
    assert err(known) < 0.02 < err(unknown)


def test_experiment_yaml_current_block(tmp_path):
    from fleet_experiment_schema import ConfigError, load_fleet_experiment
    import run_fleet_experiment as rfe
    base = yaml.safe_load(open("ci_smoke_experiment.yaml", encoding="utf-8"))
    base["output_dir"] = str(tmp_path / "out")
    base["current"] = {"speed_mps": [0.0, 0.2], "direction_deg": 30,
                       "variability": {"speed_std_mps": 0.02, "time_scale_s": 15}, "known_to_beliefs": False}
    f = tmp_path / "exp.yaml"
    f.write_text(yaml.safe_dump(base))
    cfg = load_fleet_experiment(str(f))
    fc = rfe.fleet_config_for(cfg, "stand_in")
    assert fc.current.speed_mps == (0.0, 0.2) and fc.current.known_to_beliefs is False
    base["current"] = {"speed_mps": -1}
    f.write_text(yaml.safe_dump(base))
    with pytest.raises(ConfigError):
        load_fleet_experiment(str(f))
    del base["current"]
    f.write_text(yaml.safe_dump(base))
    assert rfe.fleet_config_for(load_fleet_experiment(str(f)), "stand_in").current is None


# ----------------------------------------------------------------- single-vehicle environment

def test_arena_env_current_is_optional_and_seeded():
    from arena import Arena
    still = Arena(NEUTRAL, VEHICLE, task="station_keeping")
    moving = Arena(NEUTRAL, VEHICLE, task="station_keeping",
                   current=CurrentConfig(speed_mps=0.2, direction_deg=(0, 360)))
    for env in (still, moving):
        env.reset(seed=11)
    for _ in range(50):
        _, _, _, _, info_s = still.step(np.zeros(6))
        _, _, _, _, info_m = moving.step(np.zeros(6))
    assert info_s["current_world"] is None and np.linalg.norm(info_s["position"][:2]) < 1e-6
    assert np.linalg.norm(info_m["position"][:2]) > 0.5
    again = Arena(NEUTRAL, VEHICLE, task="station_keeping", current=CurrentConfig(speed_mps=0.2, direction_deg=(0, 360)))
    again.reset(seed=11)
    assert np.allclose(again.step(np.zeros(6))[4]["current_world"], moving._current.mean_world)


# ----------------------------------------------------------------- calibration in a known current

def test_calibration_in_a_known_current_recovers_drag_and_ignoring_it_does_not():
    """Open-water logs: the real drag acts on velocity relative to the water.
    Told the current, the fit recovers linear and quadratic surge drag exactly;
    ignoring a 0.2 m/s current leaves the whole current as unexplained error
    (~0.2 m/s RMSE) and biases both drag terms by ~5%."""
    from scipy.optimize import minimize
    from objective import RunData, rmse_by_dof
    rng = np.random.default_rng(4)
    n = 1500
    F = np.zeros((n, 6))
    x = 0.0
    for i in range(n):
        x = 0.97 * x + rng.normal() * 3
        F[i, 0] = x
    rp = np.zeros((n, 2))
    cb = np.tile([0.2, 0.0, 0.0], (n, 1))              # level vehicle, fixed heading: constant body current
    true = replace(NEUTRAL, Xu=-20.0, Xuu=-40.0, Xq_dot=0.0, Zq_dot=0.0)
    real = simulate_trajectory(true, VEHICLE, SC, TrajectoryInputs(np.zeros(6), F, rp, cb)).nu

    def fit(current):
        run = RunData("open_water", F, rp, real, real[0].copy(), 0.1, 10.0, current_body=current)
        r = minimize(lambda z: rmse_by_dof(replace(true, Xu=z[0], Xuu=z[1]), run, VEHICLE)[0], [-10.0, -10.0],
                     method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-10, "maxiter": 2000})
        return r.x, r.fun
    (xu, xuu), rmse = fit(cb)
    assert xu == pytest.approx(-20.0, abs=0.01) and xuu == pytest.approx(-40.0, abs=0.02) and rmse < 1e-3
    (xu, xuu), rmse = fit(None)
    assert rmse > 0.15 and (abs(xu + 20) > 0.5 or abs(xuu + 40) > 1.0)


def test_dataset_config_rotates_the_known_current_into_body_axes():
    from objective import DatasetConfig, load_run
    import glob
    f = sorted(glob.glob("synth_test_data/*.csv"))[0]
    plain = load_run(f, DatasetConfig(10.0))
    run = load_run(f, DatasetConfig(10.0, current_world_mps=(0.1, -0.2, 0.0)))
    assert plain.current_body is None and run.current_body.shape == (run.n_samples, 3)
    assert np.allclose(np.linalg.norm(run.current_body, axis=1), np.hypot(0.1, 0.2))
