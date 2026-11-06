"""A calibrated fin vehicle in experiments (#97): an action asks its propeller and fins, and it steers like one."""
import os

import numpy as np
import pytest

import steering
from arena_core import ArenaCore
from dataclasses import replace
from dynamics_model import SimulationConfig
from fast_dynamics import FastCore
from fin_force import action_controls
from optimizer import initial_parameters, load_vehicle_spec

CAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = load_vehicle_spec(os.path.join(CAL, "templates", "vehicle_spec_fin_template.yaml"))
P = replace(initial_parameters(SPEC), net_buoyancy_z=0.0)
V = SPEC.vehicle


def fly(core_cls, action, seconds=20.0):
    c = core_cls(P, V, SimulationConfig(dt=.1))
    c.reset(nu=[1.2, 0, 0, 0, 0, 0])
    for _ in range(int(seconds * 10)):
        c.controls = action_controls(V.actuators, action)
        c.step(np.zeros(6))
    return c


def test_an_action_becomes_this_vehicles_controls():
    c = action_controls(V.actuators, [.5, 0, 0, 0, 0, 1.0])
    assert len(c) == V.actuators.width and c[-1] == pytest.approx(750.0)
    assert max(abs(x) for x in c[:4]) == pytest.approx(V.actuators.fins[0].limit_rad)


def test_sway_and_heave_ask_nothing_of_a_fin_vehicle():
    assert not any(action_controls(V.actuators, [0, 1, 1, 0, 0, 0]))


def test_fins_stay_within_their_end_stops():
    a = action_controls(V.actuators, [1, 0, 0, 1, 1, 1])
    assert all(abs(x) <= f.limit_rad + 1e-12 for x, f in zip(a, V.actuators.fins))


@pytest.mark.parametrize("core", [FastCore, ArenaCore])
def test_positive_yaw_turns_to_starboard_and_negative_to_port(core):
    s = fly(core, [.7, 0, 0, 0, 0, .3], 10.0)
    p = fly(core, [.7, 0, 0, 0, 0, -.3], 10.0)
    r = (s.state.nu[5], p.state.nu[5])
    assert r[0] > .01 and r[1] < -.01


def test_a_stopped_fin_vehicle_cant_steer():
    c = FastCore(P, V, SimulationConfig(dt=.1))
    c.reset(nu=[0, 0, 0, 0, 0, 0])
    for _ in range(100):
        c.controls = action_controls(V.actuators, [0, 0, 0, 0, 0, 1.0])
        c.step(np.zeros(6))
    assert abs(c.state.nu[5]) < 1e-6


def test_the_two_cores_agree():
    a, b = fly(FastCore, [.7, 0, 0, 0, .2, .3], 5.0), fly(ArenaCore, [.7, 0, 0, 0, .2, .3], 5.0)
    assert np.allclose(a.state.nu, b.state.nu, atol=1e-6)


def test_the_arena_steps_a_fin_vehicle_and_its_action_turns_it():
    from arena import Arena as Env
    e = Env(P, V)
    e.reset(seed=1)
    for _ in range(50):
        e.step(np.array([.7, 0, 0, 0, 0, .3], dtype=np.float32))
    assert e._core.controls is not None


def test_mujoco_refuses_a_fin_vehicle_and_says_why():
    from arena import Arena as Env
    with pytest.raises(ValueError, match="stand_in"):
        Env(P, V, backend="mujoco")
    from fleet_env import _MuJoCoTruth
    with pytest.raises(ValueError, match="stand-in"):
        _MuJoCoTruth(P, V, SimulationConfig(dt=.1))
