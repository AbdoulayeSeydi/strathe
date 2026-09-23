"""
Regression test for the missing rotating-frame transport term in
mujoco_backend._inject_full_override_wrench (fixed 2026-09-23).

MuJoCo's free-joint linear velocity is world-frame, so realizing a
body-frame acceleration v_dot needs a world-frame acceleration of
R (v_dot + omega x v). Without the omega x v term, MuJoCo and the stand-in
disagreed by ~1.4 cm after 2 s of turning, and the gap did NOT shrink with a
finer physics_dt -- a frame error, not an integrator difference.
"""
import numpy as np
import yaml

from arena_core import ArenaCore
from dynamics_model import CalibrationParameters, SimulationConfig
from mujoco_backend import MuJoCoArenaCore
from optimizer import load_vehicle_spec

VEHICLE = load_vehicle_spec("vehicle_spec.yaml").vehicle
with open("calibrated_parameters_a1.yaml", encoding="utf-8") as _f:
    CALIBRATED = CalibrationParameters(**yaml.safe_load(_f)["parameters"])


def test_mujoco_matches_stand_in_while_turning():
    """Regression test for the missing omega x v transport term in
    mujoco_backend._inject_full_override_wrench: with it missing, MuJoCo and
    the stand-in disagreed by ~1.4 cm after 2 s of turning, and the gap did
    NOT shrink with a finer physics_dt. Both engines integrate the same
    equations, so the gap must shrink roughly in proportion to physics_dt."""
    tau = [9, -4, 3, 1, -1, 1.5]
    gaps = []
    for pdt in (0.005, 0.001):
        cfg = SimulationConfig(dt=0.1, physics_dt=pdt)
        a, m = ArenaCore(CALIBRATED, VEHICLE, cfg), MuJoCoArenaCore(CALIBRATED, VEHICLE, cfg)
        a.reset(nu=[-0.1, 0, 0, 0, 0, 0])
        m.reset(nu=[-0.1, 0, 0, 0, 0, 0])
        for _ in range(20):
            sa, sm = a.step(tau), m.step(tau)
        gaps.append(np.linalg.norm(sa.position - sm.position))
    assert gaps[0] < 3e-3
    assert gaps[1] < 0.35 * gaps[0]
