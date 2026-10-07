"""
The tested physics/state-tracking CORE behind the Arena Gym environment
(see arena.py) — kept separate from gymnasium.Env's interface so it can be
fully tested independent of whether gymnasium is installed.

## New capability this introduces, not present anywhere else in the pipeline
Every physics function built so far (dynamics_model.simulate_trajectory,
holoocean_backend.simulate_trajectory_holoocean) only ever tracked BODY-
FRAME VELOCITY — never world-frame POSITION or self-integrated ORIENTATION,
because Phase 1's calibration only ever compared velocities against real
logged velocities. A live RL rollout has no pre-recorded real trajectory to
teacher-force orientation from (unlike the stand-in's batch
simulate_trajectory, which uses REAL LOGGED roll/pitch at every step) — it
must genuinely self-integrate both position and orientation as it goes,
exactly like holoocean_backend.py already does (for a different reason:
there, HoloOcean's own rigid-body engine does it; here, we do it ourselves
since Decision 2 in phase2_config.py trains on the fast stand-in, with no
HoloOcean underneath).

## Physics: SAME calibratable equations as everywhere else
Uses dynamics_model.compute_acceleration() directly — the identical,
already-validated function the stand-in AND the HoloOcean backend both
call. No re-implementation, no risk of drift from what's already proven.

## Orientation integration method
R_new = R_old * Rotation.from_rotvec(body_angular_velocity * dt) — this is
NOT an approximation for constant angular velocity over one timestep (which
matches this whole pipeline's existing zero-order-hold-per-step
convention) — composing a rotation vector this way is an exact way to
integrate a constant body-frame angular velocity, the same kind of
integrator real rigid-body simulators use internally.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from dynamics_model import CalibrationParameters, VehicleConstants, SimulationConfig, ThrusterLag, compute_acceleration


@dataclass
class ArenaState:
    """Everything needed to fully describe the vehicle's current state —
    kept as one dataclass so reset()/step() have a single, clear thing to
    mutate, and so a future renderer/logger has one object to read from."""
    nu: np.ndarray             # (6,) body-frame [u,v,w,p,q,r]
    position: np.ndarray       # (3,) world-frame position
    orientation: Rotation      # body->world rotation

    @classmethod
    def initial(cls) -> "ArenaState":
        return cls(nu=np.zeros(6), position=np.zeros(3), orientation=Rotation.identity())


class ArenaCore:
    """
    The tested physics core. step() advances the vehicle by ONE external
    dt (with internal sub-stepping for numerical stability, matching
    dynamics_model.py's existing SimulationConfig.n_substeps/sub_dt
    convention exactly), given a body-frame force/torque action.
    """

    def __init__(self, params: CalibrationParameters, vehicle: VehicleConstants,
                 sim_config: SimulationConfig):
        params.validate()
        vehicle.validate()
        sim_config.validate()
        self.params = params
        self.vehicle = vehicle
        self.sim_config = sim_config
        self.state = ArenaState.initial()
        self._lag = ThrusterLag()  # thruster spin-up (params.thruster_time_constant_s; none at 0)
        # Water current in WORLD axes (m/s), or None for still water
        # (dynamics_model._relative_to_water). Set before each step().
        self.current_world = None

    def set_params(self, params: CalibrationParameters) -> None:
        """Swap the parameter set between episodes (e.g. a buoyancy change, buoyancy.py)."""
        params.validate()
        self.params = params

    def reset(self, position: np.ndarray = None, nu: np.ndarray = None) -> ArenaState:
        self._lag.reset()
        self.state = ArenaState(
            nu=np.zeros(6) if nu is None else np.array(nu, dtype=float),
            position=np.zeros(3) if position is None else np.array(position, dtype=float),
            orientation=Rotation.identity(),
        )
        return self.state

    def step(self, tau: np.ndarray) -> ArenaState:
        """Advances by one external dt. tau is body-frame [Fx,Fy,Fz,Tx,Ty,Tz].

        Integrates velocity, position, AND orientation together at the
        SUB-STEP level, all using pre-update ("start of this sub-step")
        state — standard, consistent forward Euler. An earlier version of
        this function integrated position/orientation ONCE per external
        step using the ALREADY-UPDATED velocity, which silently
        double-counted each step's acceleration (caught by a direct test:
        3 steps of constant thrust gave position=6.0 instead of the
        correct 3.0 — fixed and reverified before this was ever handed off).

        tau is the commanded wrench; with a thruster lag (params.thruster_time_constant_s) the wrench applied over
        this step lags it (dynamics_model.ThrusterLag)."""
        tau = np.asarray(self._lag.step(np.asarray(tau, dtype=float), self.params.thruster_time_constant_s,
                                        self.sim_config.dt), dtype=float)
        v = self.state.nu.copy()
        position = self.state.position.copy()
        orientation = self.state.orientation

        n_substeps = self.sim_config.n_substeps
        sub_dt = self.sim_config.sub_dt

        for _ in range(n_substeps):
            roll, pitch, _yaw = orientation.as_euler("xyz")
            current_body = (None if self.current_world is None
                            else orientation.inv().apply(np.asarray(self.current_world, dtype=float)))

            # Kinematics FIRST, using state as it stood at the START of
            # this sub-step — no old/new velocity mixing.
            world_lin_vel = orientation.apply(v[:3])
            position = position + world_lin_vel * sub_dt
            if not np.all(np.isfinite(v)):
                raise FloatingPointError("ArenaCore.step() diverged (non-finite velocity) — physics_dt too "
                                         "large for this parameter set, or an unstable parameter combination.")
            orientation = orientation * Rotation.from_rotvec(v[3:] * sub_dt)

            # THEN update velocity using this sub-step's acceleration.
            v_dot, _damping, _restoring = compute_acceleration(
                self.params, self.vehicle, v, tau, roll, pitch, current_body
            )
            v = v + v_dot * sub_dt

        if not (np.all(np.isfinite(v)) and np.all(np.isfinite(position))):
            raise FloatingPointError(
                "ArenaCore.step() produced NaN/inf — check parameter bounds or "
                "whether this action/parameter combination made the simulation unstable."
            )

        self.state = ArenaState(nu=v, position=position, orientation=orientation)
        return self.state

    def position_error_body_frame(self, target_position: np.ndarray) -> np.ndarray:
        """World-frame vector to target, expressed in the CURRENT body
        frame — a natural observation component (the vehicle 'feels'
        the target relative to its own heading, not in absolute world
        coordinates, which is both more physically realistic and usually
        an easier signal for an RL policy to learn from)."""
        world_error = np.asarray(target_position, dtype=float) - self.state.position
        return self.state.orientation.inv().apply(world_error)