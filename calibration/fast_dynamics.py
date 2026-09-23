"""
fast_dynamics.py -- a faster, numerically-equivalent implementation of
ArenaCore's 6-DOF integrator, for RL training and fleet evaluation.

## Why this exists
ArenaCore.step() calls dynamics_model.compute_acceleration() once per
sub-step (20 sub-steps per 0.1s tick). That function rebuilds the 6x6
mass matrix, re-checks it with a Cholesky factorization, builds three
skew matrices and runs np.linalg.solve -- every sub-step -- and ArenaCore
wraps each sub-step in scipy Rotation objects. For parameters that never
change during an episode that is almost all overhead: the fleet env ran
at ~300 steps/s on an M1, too slow to train per-vehicle policies with
multiple seeds.

## What is identical, and how that is proven
Same equations, same integration order, same zero-order hold:
    M v_dot = tau + D(v) v - g(eta) - C(v) v
with M, D, g and C exactly as in dynamics_model.py, and the same
sub-step ordering as ArenaCore.step() (position and orientation advanced
with the start-of-sub-step velocity, then velocity). The differences are
purely representational: M is inverted once instead of solved each
sub-step, C(v)v is written as cross products instead of skew-matrix
products, and orientation is a 3x3 rotation matrix updated with
Rodrigues' formula instead of a scipy Rotation.

test_fast_dynamics.py checks acceleration against
dynamics_model.compute_acceleration() and full multi-step trajectories
against ArenaCore for random parameters, states and thrusts, to
floating-point tolerance. ArenaCore stays the reference implementation;
this file must never be treated as a second source of physics truth.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from arena_core import ArenaState
from dynamics_model import CalibrationParameters, VehicleConstants, SimulationConfig, _mass_and_damping_arrays

GRAVITY = 9.81  # matches dynamics_model._restoring_forces' default g


class FastDynamics:
    """Precomputed dynamics for one fixed parameter set."""

    def __init__(self, params: CalibrationParameters, vehicle: VehicleConstants):
        params.validate()
        vehicle.validate()
        # Reuse dynamics_model's own construction (including its
        # positive-definiteness check) so M can never drift from the reference.
        M, lin, quad = _mass_and_damping_arrays(params, vehicle)
        self.M = [[float(x) for x in row] for row in M]
        self.Minv = [[float(x) for x in row] for row in np.linalg.inv(M)]
        self.lin = [float(x) for x in lin]
        self.quad = [float(x) for x in quad]
        W = vehicle.mass_kg * GRAVITY
        self._W = W
        self._diff = W - (W + params.net_buoyancy_z)  # = -net_buoyancy_z, same expression as reference
        self._zg = params.cg_cb_offset_z

    def acceleration(self, v, tau, roll: float, pitch: float) -> list:
        """Body-frame v_dot (list of 6 floats). Equivalent to
        dynamics_model.compute_acceleration(...)[0]."""
        M, Minv, lin, quad = self.M, self.Minv, self.lin, self.quad
        # Momentum p = M v (full matrix -- keeps off-diagonal coupling exact).
        p = [M[i][0] * v[0] + M[i][1] * v[1] + M[i][2] * v[2]
             + M[i][3] * v[3] + M[i][4] * v[4] + M[i][5] * v[5] for i in range(6)]
        # C(v)v = [-(p_lin x w) ; -(p_lin x v_lin) - (p_ang x w)]
        pl0, pl1, pl2, pa0, pa1, pa2 = p
        u, vv, w, a, b, c = v
        cv0 = -(pl1 * c - pl2 * b)
        cv1 = -(pl2 * a - pl0 * c)
        cv2 = -(pl0 * b - pl1 * a)
        cv3 = -(pl1 * w - pl2 * vv) - (pa1 * c - pa2 * b)
        cv4 = -(pl2 * u - pl0 * w) - (pa2 * a - pa0 * c)
        cv5 = -(pl0 * vv - pl1 * u) - (pa0 * b - pa1 * a)
        # Restoring forces g(eta) -- same formula as _restoring_forces.
        sr, cr, sp, cp = math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch)
        diff, W, zg = self._diff, self._W, self._zg
        g0 = diff * sp
        g1 = -diff * cp * sr
        g2 = -diff * cp * cr
        g3 = zg * W * cp * sr
        g4 = zg * W * sp
        rhs = [
            tau[0] + (lin[0] + quad[0] * abs(u)) * u - g0 - cv0,
            tau[1] + (lin[1] + quad[1] * abs(vv)) * vv - g1 - cv1,
            tau[2] + (lin[2] + quad[2] * abs(w)) * w - g2 - cv2,
            tau[3] + (lin[3] + quad[3] * abs(a)) * a - g3 - cv3,
            tau[4] + (lin[4] + quad[4] * abs(b)) * b - g4 - cv4,
            tau[5] + (lin[5] + quad[5] * abs(c)) * c - cv5,
        ]
        return [Minv[i][0] * rhs[0] + Minv[i][1] * rhs[1] + Minv[i][2] * rhs[2]
                + Minv[i][3] * rhs[3] + Minv[i][4] * rhs[4] + Minv[i][5] * rhs[5]
                for i in range(6)]


def _rotvec_matrix(wx: float, wy: float, wz: float) -> list:
    """exp([w]x) via Rodrigues, with series expansions near zero angle."""
    th2 = wx * wx + wy * wy + wz * wz
    if th2 < 1e-12:
        A = 1.0 - th2 / 6.0
        B = 0.5 - th2 / 24.0
    else:
        th = math.sqrt(th2)
        A = math.sin(th) / th
        B = (1.0 - math.cos(th)) / th2
    return [
        [1.0 - B * (wy * wy + wz * wz), -A * wz + B * wx * wy, A * wy + B * wx * wz],
        [A * wz + B * wx * wy, 1.0 - B * (wx * wx + wz * wz), -A * wx + B * wy * wz],
        [-A * wy + B * wx * wz, A * wx + B * wy * wz, 1.0 - B * (wx * wx + wy * wy)],
    ]


def _matmul3(X: list, Y: list) -> list:
    return [[X[i][0] * Y[0][j] + X[i][1] * Y[1][j] + X[i][2] * Y[2][j] for j in range(3)]
            for i in range(3)]


def roll_pitch_from_matrix(R: list) -> tuple:
    """(roll, pitch) matching scipy Rotation.as_euler('xyz') (extrinsic x-y-z)
    away from gimbal lock (|pitch| = 90 deg)."""
    pitch = -math.asin(max(-1.0, min(1.0, R[2][0])))
    roll = math.atan2(R[2][1], R[2][2])
    return roll, pitch


class FastCore:
    """Drop-in for ArenaCore's reset()/step()/state API, backed by FastDynamics."""

    def __init__(self, params: CalibrationParameters, vehicle: VehicleConstants,
                 sim_config: SimulationConfig):
        sim_config.validate()
        self.params = params
        self.vehicle = vehicle
        self.sim_config = sim_config
        self.dyn = FastDynamics(params, vehicle)
        self._n_sub = sim_config.n_substeps
        self._sub_dt = sim_config.sub_dt
        self.reset()

    def reset(self, position=None, nu=None, orientation: Optional[Rotation] = None):
        self.nu = [0.0] * 6 if nu is None else [float(x) for x in nu]
        self.pos = [0.0] * 3 if position is None else [float(x) for x in position]
        if orientation is None:
            self.R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        else:
            self.R = [[float(x) for x in row] for row in orientation.as_matrix()]
        return self.state

    def step(self, tau):
        tau = [float(x) for x in tau]
        v = self.nu
        pos = self.pos
        R = self.R
        dt = self._sub_dt
        acc = self.dyn.acceleration
        try:
            for _ in range(self._n_sub):
                roll, pitch = roll_pitch_from_matrix(R)
                u, vv, w = v[0], v[1], v[2]
                pos = [pos[i] + (R[i][0] * u + R[i][1] * vv + R[i][2] * w) * dt for i in range(3)]
                R = _matmul3(R, _rotvec_matrix(v[3] * dt, v[4] * dt, v[5] * dt))
                vd = acc(v, tau, roll, pitch)
                v = [v[i] + vd[i] * dt for i in range(6)]
        except (OverflowError, ValueError) as e:
            # math.sin/sqrt raise instead of returning inf/nan once a diverging
            # state overflows; report it the same way ArenaCore does.
            raise FloatingPointError(f"FastCore.step() diverged: {e}") from e
        if not all(math.isfinite(x) for x in v) or not all(math.isfinite(x) for x in pos):
            raise FloatingPointError(
                "FastCore.step() produced NaN/inf -- check parameter bounds or "
                "whether this action/parameter combination made the simulation unstable."
            )
        self.nu, self.pos, self.R = v, pos, R
        return self.state

    # --- read accessors (cheap; the env uses these, not .state, in hot loops) ---
    def position_array(self) -> np.ndarray:
        return np.array(self.pos)

    def nu_array(self) -> np.ndarray:
        return np.array(self.nu)

    def roll_pitch(self) -> tuple:
        return roll_pitch_from_matrix(self.R)

    def world_velocity(self) -> np.ndarray:
        R, v = self.R, self.nu
        return np.array([R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2] for i in range(3)])

    @property
    def state(self) -> ArenaState:
        return ArenaState(nu=np.array(self.nu), position=np.array(self.pos),
                          orientation=Rotation.from_matrix(np.array(self.R)))
