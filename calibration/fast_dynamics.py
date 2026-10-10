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
        self._xg = params.cg_cb_offset_x  # hull terms (#71); all 0 for vehicles without them
        self._lift = (params.Zuw, params.Muw, params.Muq) if (params.Zuw or params.Muw or params.Muq) else None

    def acceleration(self, v, tau, roll: float, pitch: float) -> list:
        """Body-frame v_dot (list of 6 floats). Equivalent to
        dynamics_model.compute_acceleration(...)[0]."""
        return self.acceleration_trig(v, tau, math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch))

    def acceleration_current(self, v, tau, roll: float, pitch: float, cb) -> list:
        """acceleration() in a water current; cb = current in body axes (3).
        Same physics as dynamics_model._relative_to_water: forces from the
        velocity relative to the water, plus the -omega x cb transport term."""
        return self.acceleration_trig(v, tau, math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch), cb)

    def acceleration_trig(self, v, tau, sr: float, cr: float, sp: float, cp: float, cb=None) -> list:
        """acceleration() with sin/cos of roll and pitch precomputed (they are
        constant across a teacher-forced sample's sub-steps). cb: optional
        water current in body axes (see acceleration_current)."""
        if cb is not None and (cb[0] or cb[1] or cb[2]):
            p_, q_, r_ = v[3], v[4], v[5]
            vd = self.acceleration_trig([v[0] - cb[0], v[1] - cb[1], v[2] - cb[2], p_, q_, r_], tau, sr, cr, sp, cp)
            vd[0] -= q_ * cb[2] - r_ * cb[1]
            vd[1] -= r_ * cb[0] - p_ * cb[2]
            vd[2] -= p_ * cb[1] - q_ * cb[0]
            return vd
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
        diff, W, zg = self._diff, self._W, self._zg
        g0 = diff * sp
        g1 = -diff * cp * sr
        g2 = -diff * cp * cr
        g3 = zg * W * cp * sr
        g4 = zg * W * sp
        g5 = 0.0
        if self._xg:  # x offset of the centre of gravity (dynamics_model._restoring_forces)
            g4 += self._xg * W * cp * cr
            g5 = -self._xg * W * cp * sr
        rhs = [
            tau[0] + (lin[0] + quad[0] * abs(u)) * u - g0 - cv0,
            tau[1] + (lin[1] + quad[1] * abs(vv)) * vv - g1 - cv1,
            tau[2] + (lin[2] + quad[2] * abs(w)) * w - g2 - cv2,
            tau[3] + (lin[3] + quad[3] * abs(a)) * a - g3 - cv3,
            tau[4] + (lin[4] + quad[4] * abs(b)) * b - g4 - cv4,
            tau[5] + (lin[5] + quad[5] * abs(c)) * c - g5 - cv5,
        ]
        if self._lift is not None:  # hull lift and speed-dependent damping (dynamics_model._hull_lift_force)
            zuw, muw, muq = self._lift
            rhs[1] += zuw * u * vv
            rhs[2] += zuw * u * w
            rhs[4] += muw * u * w + muq * u * b
            rhs[5] += -muw * u * vv + muq * u * c
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
        self._dt = sim_config.dt
        from dynamics_model import ThrusterLag
        self._lag = ThrusterLag()  # thruster spin-up (params.thruster_time_constant_s; none at 0)
        # Water current in WORLD axes (m/s), or None for still water. Set it
        # before each step() (fleet_env / current.py update it per tick).
        self.current_world = None
        self.reset()

    def set_params(self, params: CalibrationParameters) -> None:
        """Swap the parameter set between episodes (e.g. a buoyancy change, buoyancy.py)."""
        if params is not self.params:
            self.dyn = FastDynamics(params, self.vehicle)
            self.params = params

    def reset(self, position=None, nu=None, orientation: Optional[Rotation] = None, applied_wrench=None):
        """applied_wrench: what the thrusters are producing at the start (default: stopped). Only matters with a
        thruster lag."""
        self._lag.reset(applied_wrench)
        self.nu = [0.0] * 6 if nu is None else [float(x) for x in nu]
        self.pos = [0.0] * 3 if position is None else [float(x) for x in position]
        if orientation is None:
            self.R = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        else:
            self.R = [[float(x) for x in row] for row in orientation.as_matrix()]
        return self.state

    def step(self, tau):
        tau = [float(x) for x in self._lag.step(tau, self.params.thruster_time_constant_s, self._dt)]
        v = self.nu
        pos = self.pos
        R = self.R
        dt = self._sub_dt
        acc = self.dyn.acceleration
        Vc = self.current_world
        if Vc is not None and not (Vc[0] or Vc[1] or Vc[2]):
            Vc = None
        try:
            for _ in range(self._n_sub):
                roll, pitch = roll_pitch_from_matrix(R)
                u, vv, w = v[0], v[1], v[2]
                pos = [pos[i] + (R[i][0] * u + R[i][1] * vv + R[i][2] * w) * dt for i in range(3)]
                if Vc is None:
                    R = _matmul3(R, _rotvec_matrix(v[3] * dt, v[4] * dt, v[5] * dt))
                    vd = acc(v, tau, roll, pitch)
                else:
                    # current in body axes at the START of the sub-step (same R as the kinematics)
                    cb = [R[0][i] * Vc[0] + R[1][i] * Vc[1] + R[2][i] * Vc[2] for i in range(3)]
                    R = _matmul3(R, _rotvec_matrix(v[3] * dt, v[4] * dt, v[5] * dt))
                    vd = self.dyn.acceleration_current(v, tau, roll, pitch, cb)
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


def simulate_velocity_teacher_forced(params: CalibrationParameters, vehicle: VehicleConstants,
                                     n_substeps: int, sub_dt: float, nu0, forces_torques,
                                     roll_pitch_series, current_body_series=None) -> np.ndarray:
    """dynamics_model.simulate_trajectory's integration loop (teacher-forced
    roll/pitch, Forward Euler sub-steps), on FastDynamics. Returns (N, 6)."""
    dyn = FastDynamics(params, vehicle)
    acc = dyn.acceleration_trig
    n = forces_torques.shape[0]
    out = np.empty((n, 6))
    out[0] = nu0
    v = [float(x) for x in nu0]
    F = forces_torques.tolist()
    RP = roll_pitch_series.tolist()
    CB = None if current_body_series is None else np.asarray(current_body_series, dtype=float).tolist()
    for k in range(n - 1):
        tau = F[k]
        roll, pitch = RP[k]
        cb = None if CB is None else CB[k]
        sr, cr, sp, cp = math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch)
        for _ in range(n_substeps):
            vd = acc(v, tau, sr, cr, sp, cp, cb)
            v = [v[0] + vd[0] * sub_dt, v[1] + vd[1] * sub_dt, v[2] + vd[2] * sub_dt,
                 v[3] + vd[3] * sub_dt, v[4] + vd[4] * sub_dt, v[5] + vd[5] * sub_dt]
        out[k + 1] = v
    return out
