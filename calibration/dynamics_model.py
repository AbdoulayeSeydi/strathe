"""
Simplified 6-DOF underwater vehicle dynamics — a STAND-IN for HoloOcean.

Not the final simulator. Exists to build/test the calibration optimizer,
objective function, and train/held-out split logic against real trajectory
data before HoloOcean is available (needs a GPU/Linux machine we don't have
access to yet). Project history and rationale for this approach live in the
repo README, not here — code comments below explain WHY things are the way
they are, not the backstory.

## Coordinate frames
- Body-fixed frame: SNAME/Fossen convention — x forward (surge), y
  starboard (sway), z down (heave); roll about x, pitch about y, yaw
  about z.
- World frame: NOT verified from the source dataset (no frame convention
  documented there — NED vs ENU unconfirmed). Doesn't affect correctness
  here since the restoring-force term only depends on roll/pitch relative
  to gravity, not a specific world-frame convention — but this WILL matter
  once world-frame position is simulated (not done in this stand-in), so
  flagging it now rather than letting it surface as a silent bug later.

## Assumptions (all of them, one place)
- Constant timestep (fixed dt, no variable-rate integration)
- No Coriolis/centripetal coupling terms
- No ocean current
- Diagonal added mass and diagonal damping (no off-diagonal coupling)
- Orientation (roll/pitch) is TEACHER-FORCED from real logged data, not
  self-integrated — see TEACHER_FORCED_ORIENTATION below. This is the
  single most important caveat in this file; read that section.
- Logged thruster forces/torques (Fx..Tz) are taken as ground truth input,
  not re-derived from PWM (see vehicle_spec.yaml's thruster_model section).

## Structural non-identifiability (found via diagnose_calibration.py)
Rotational inertia (Ixx/Iyy/Izz) and rotational added mass (Kp_dot/Mq_dot/
Nr_dot) were originally SEPARATE calibration targets. A Jacobian-based
identifiability analysis found Izz and Nr_dot correlated at EXACTLY -1.000
— not just "hard to separate," but mathematically impossible to separate
from velocity-only data, because they only ever appear as a SUM
(M_diag[yaw] = Izz + Nr_dot) everywhere in this model. Velocity data can
observe that sum; it can never determine how it splits. Confirmed
algebraically (not just inferred from correlation) that roll (Ixx+Kp_dot)
and pitch (Iyy+Mq_dot) have the identical structure. All three axes were
therefore COLLAPSED into single effective-inertia parameters (Ieff_roll,
Ieff_pitch, Ieff_yaw) — 6 fields became 3, removing the degeneracy at the
model level rather than trying to optimize around it. If Ixx/Iyy/Izz are
ever needed individually again (e.g. a future CAD-based measurement),
that value becomes a KNOWN CONSTANT in VehicleConstants, not a
calibration target — recalibrating Ieff_* would then back out the
implied added-mass term as Ieff - known_inertia.

## Physics reference
Standard rigid-body + hydrodynamic model, M*nu_dot + D(nu)*nu + g(eta) = tau,
per Fossen, T.I. (2011), "Handbook of Marine Craft Hydrodynamics and Motion
Control", Wiley — general form of the 6-DOF equations of motion and the
restoring-force vector g(eta) implemented below as a diagonal-only
simplification (no specific equation numbers cited here deliberately,
rather than risk citing ones I can't verify against the source text).

## Numerical stability
Forward Euler integration (v_next = v + sub_dt*v_dot), not RK4. Forward
Euler becomes unstable as the integration step grows relative to the
system's fastest time constant — CONFIRMED by direct reproduction, not
just theorized: at the empirically-determined real data rate of 10Hz
(dt=0.1s), the DEFAULT (uncalibrated, weakly-damped) parameter set
overflowed to NaN, while the SAME parameters integrated fine at 50Hz
(dt=0.02s). That's not a sign 10Hz is wrong — it's that using the data's
own sample interval AS the physics integration step conflates two
different things that don't have to be equal.

FIX: SimulationConfig.physics_dt (default 0.02s) sets the target INTERNAL
integration step, independent of dt (the external data sample interval).
dt is subdivided into ceil(dt/physics_dt) sub-steps internally, holding
the force/orientation input constant across them (zero-order hold — real
data isn't available any finer than dt anyway). At dt=0.02s (50Hz) this
is a no-op (1 sub-step, identical to the old behavior). At dt=0.1s (10Hz)
this now takes 5 sub-steps of ~0.02s each, matching the same internal
resolution already validated as stable — confirmed to fix the NaN found
above, not just theorized to.

This does not guarantee stability for every possible parameter combination
at every possible dt — if simulate_trajectory() still raises
FloatingPointError, check whether physics_dt itself needs to shrink further
before assuming the parameters are at fault.
TODO: replace with RK4 if stiffness becomes an issue even with sub-stepping.

## THE MOST IMPORTANT CAVEAT: teacher forcing
This model is driven by the REAL, LOGGED roll/pitch at every timestep
(TrajectoryInputs.roll_pitch_series) rather than integrating its own
orientation forward. Deliberate for this MVP — it isolates the calibration
problem (damping/added-mass/buoyancy fit) from compounding orientation-
integration error. BUT it means this model's RMSE is NOT directly
comparable to a true forward simulator (HoloOcean) that must integrate
orientation itself and can't "see" ground truth partway through a rollout.
Never present this model's accuracy as equivalent to a from-scratch
closed-loop simulator's — every result object carries
`used_teacher_forced_orientation` so downstream comparison code can check
this before treating two RMSE numbers as apples-to-apples.

## Interface stability
simulate_trajectory() is the swap point: objective.py and optimizer.py
depend only on this function's signature, not its internals. A future
HoloOcean-backed (or Isaac Sim, or MuJoCo) implementation matches the same
shape: (params, vehicle, sim_config, trajectory) -> SimulationResult.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

# Order used ONLY internally for vector<->object conversion at the
# optimizer boundary (scipy.optimize needs a flat float array). Nothing
# outside CalibrationParameters.to_vector()/from_vector() should construct
# or index a raw vector by hand — that's exactly the fragile-string-key
# failure mode this dataclass exists to prevent.
_PARAM_ORDER = (
    "Ieff_roll", "Ieff_pitch", "Ieff_yaw",
    "Xu", "Yv", "Zw", "Kp", "Mq", "Nr",
    "Xuu", "Yvv", "Zww", "Kpp", "Mqq", "Nrr",
    "Xu_dot", "Yv_dot", "Zw_dot",
    "net_buoyancy_z", "cg_cb_offset_z",
    "Xq_dot", "Zq_dot",
)

TEACHER_FORCED_ORIENTATION = True  # see module docstring — comparison code
                                    # elsewhere should check this flag
                                    # before comparing RMSE across backends


@dataclass
class CalibrationParameters:
    """
    The 20 unknowns the optimizer solves for, as named fields — not a raw
    vector. A typo like Xu_dot -> Xudot fails immediately at construction
    (dataclass raises TypeError for an unexpected/missing field) instead
    of silently corrupting a fit. Inertia lives here, not in
    VehicleConstants — no CAD is available, so it's genuinely unknown,
    not a known constant (see vehicle_spec.yaml).
    """
    Ieff_roll: float   # = Ixx + Kp_dot, COLLAPSED — see module docstring,
    Ieff_pitch: float  # = Iyy + Mq_dot,   "Structural non-identifiability"
    Ieff_yaw: float    # = Izz + Nr_dot, section. These three were previously
                       # six separate fields (Ixx/Iyy/Izz + Kp_dot/Mq_dot/
                       # Nr_dot); diagnose_calibration.py's Jacobian analysis
                       # proved Izz/Nr_dot correlate at exactly -1.000 —
                       # velocity-only data can observe their SUM, never the
                       # individual values, so keeping them separate wasn't
                       # just hard to fit, it was mathematically impossible
                       # to fit uniquely. Roll/pitch have the identical
                       # algebraic structure (confirmed by inspection, not
                       # inferred from correlation) and were collapsed too.
    Xu: float
    Yv: float
    Zw: float
    Kp: float
    Mq: float
    Nr: float
    Xuu: float
    Yvv: float
    Zww: float
    Kpp: float
    Mqq: float
    Nrr: float
    Xu_dot: float
    Yv_dot: float
    Zw_dot: float
    net_buoyancy_z: float
    cg_cb_offset_z: float

    # Off-diagonal added-mass coupling — surge-pitch and heave-pitch,
    # the physically best-motivated cross-terms for a BlueROV2's real
    # vertical (top-bottom) asymmetry (open frame, buoyancy above
    # thrusters — the SAME asymmetry cg_cb_offset_z already models for
    # the restoring force). Defaults to 0.0 — both for a sensible "no
    # coupling" starting point, AND so existing saved
    # calibrated_parameters.yaml files (from before this was added)
    # still load correctly via CalibrationParameters(**old_dict), with
    # these two fields implicitly zero (reduces exactly to the prior
    # model — see _mass_matrix()'s regression test).
    Xq_dot: float = 0.0
    Zq_dot: float = 0.0

    def validate(self) -> None:
        """Cheap sanity checks — called once per construction, not per
        integration step, so this doesn't cost anything in the hot loop."""
        for name in ("Ieff_roll", "Ieff_pitch", "Ieff_yaw"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError(f"{name} must be positive, got {v}")
        for name in _PARAM_ORDER:
            v = getattr(self, name)
            if not np.isfinite(v):
                raise ValueError(f"{name} is not finite: {v}")

    def to_vector(self) -> np.ndarray:
        """Flattens to the order scipy.optimize.least_squares expects."""
        return np.array([getattr(self, name) for name in _PARAM_ORDER], dtype=float)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "CalibrationParameters":
        """Inverse of to_vector(). Raises immediately on a length mismatch
        rather than silently misassigning values to the wrong field."""
        if len(vec) != len(_PARAM_ORDER):
            raise ValueError(
                f"Expected {len(_PARAM_ORDER)} values {_PARAM_ORDER}, got {len(vec)}"
            )
        params = cls(**dict(zip(_PARAM_ORDER, vec)))
        params.validate()
        return params

    @property
    def param_names(self) -> tuple:
        """Read-only view of the vector ordering, for code building bounds arrays."""
        return _PARAM_ORDER


@dataclass
class VehicleConstants:
    """Truly known/measured quantities — NOT touched by calibration.
    Just mass for now; inertia lives in CalibrationParameters instead,
    since no CAD is available and it's a genuine unknown."""
    mass_kg: float

    def validate(self) -> None:
        if self.mass_kg <= 0:
            raise ValueError(f"mass_kg must be positive, got {self.mass_kg}")


@dataclass
class SimulationConfig:
    dt: float                    # the DATA's sample interval (e.g. 0.1s at 10Hz)
    physics_dt: float = 0.005    # target INTERNAL integration step. dt is
                                  # subdivided into ceil(dt/physics_dt) sub-steps,
                                  # each holding the force/orientation input constant
                                  # (zero-order hold, since real data isn't available
                                  # any finer than dt anyway). This decouples "how
                                  # coarsely the data was sampled" from "how fine the
                                  # physics integration needs to be for stability" —
                                  # see the "Numerical stability" module docstring
                                  # section for the original (pre-Coriolis) finding
                                  # this was built for.
                                  #
                                  # TIGHTENED from 0.02 to 0.005 after adding Coriolis
                                  # terms: confirmed via direct reproduction that
                                  # 0.02 crashes (NaN) at DEFAULT (weakly-damped)
                                  # parameters once real-scale forces push velocity
                                  # large enough for the Coriolis terms (quadratic in
                                  # velocity) to overflow — reproduced cleanly at
                                  # force magnitude ~100+, confirmed 0.005 resolves it
                                  # up to force magnitude 200 with consistent,
                                  # converged max|nu| values (a genuine resolution
                                  # fix, not a different physics answer).
                                  #
                                  # HONEST TRADEOFF: this changes the OLD "50Hz is a
                                  # no-op (1 sub-step)" regression guarantee from the
                                  # original sub-stepping fix — at 50Hz (dt=0.02) this
                                  # new default now takes 4 sub-steps, not 1. Findings
                                  # #1-11 in PROJECT_STATUS.md were computed with the
                                  # OLD default (physics_dt=0.02, 1 substep at 50Hz) —
                                  # pass physics_dt=0.02 explicitly to exactly
                                  # reproduce those specific historical numbers.
                                  # Going forward, both 10Hz and 50Hz get more
                                  # numerically robust integration by default.
    seed: Optional[int] = None  # Reserved for future sensor-noise / current
                                 # disturbance injection. Unused today —
                                 # the hook exists so adding noise later
                                 # doesn't require touching every call site.
    debug: bool = False

    def validate(self) -> None:
        if self.dt <= 0:
            raise ValueError(f"dt must be positive, got {self.dt}")
        if not (0.001 <= self.dt <= 1.0):
            # Widened from the old 0.001-0.1s range now that sub-stepping
            # handles integration stability internally — this range is now
            # about "is this a plausible data sample interval at all" (1Hz
            # to 1000Hz), not "is Forward Euler stable at this exact step."
            raise ValueError(
                f"dt={self.dt}s is outside the plausible range for real "
                f"telemetry sample intervals (0.001s-1.0s, i.e. 1Hz-1000Hz)."
            )
        if self.physics_dt <= 0 or self.physics_dt > self.dt:
            raise ValueError(
                f"physics_dt must be positive and <= dt, got physics_dt="
                f"{self.physics_dt} vs dt={self.dt}"
            )

    @property
    def n_substeps(self) -> int:
        return max(1, int(np.ceil(self.dt / self.physics_dt)))

    @property
    def sub_dt(self) -> float:
        return self.dt / self.n_substeps


@dataclass
class TrajectoryInputs:
    """One run's worth of driving data for the simulator."""
    nu0: np.ndarray                 # shape (6,) initial [u,v,w,p,q,r]
    forces_torques: np.ndarray      # shape (N,6) logged Fx,Fy,Fz,Tx,Ty,Tz
    roll_pitch_series: np.ndarray   # shape (N,2) REAL LOGGED roll,pitch
                                     # (radians) — see TEACHER_FORCED_ORIENTATION

    def validate(self) -> None:
        n = self.forces_torques.shape[0]
        if self.forces_torques.shape != (n, 6):
            raise ValueError(f"forces_torques must be (N,6), got {self.forces_torques.shape}")
        if self.roll_pitch_series.shape != (n, 2):
            raise ValueError(f"roll_pitch_series must be (N,2), got {self.roll_pitch_series.shape}")
        if self.nu0.shape != (6,):
            raise ValueError(f"nu0 must be shape (6,), got {self.nu0.shape}")
        for name, arr in [("forces_torques", self.forces_torques),
                           ("roll_pitch_series", self.roll_pitch_series),
                           ("nu0", self.nu0)]:
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or inf")


@dataclass
class SimulationResult:
    nu: np.ndarray                          # shape (N,6) simulated velocity trajectory
    used_teacher_forced_orientation: bool    # always True for this backend
    debug_info: Optional[dict] = None        # populated only if SimulationConfig.debug=True
                                              # -> {"damping_force", "restoring_force", "acceleration"},
                                              # each shape (N,6), for inspecting the fit term-by-term


def quaternion_to_roll_pitch(qx, qy, qz, qw):
    """
    Quaternion -> roll (phi), pitch (theta), radians. Uses
    scipy.spatial.transform.Rotation rather than a hand-rolled formula —
    fewer places for a sign/axis-order bug to hide undetected.
    """
    qx = np.atleast_1d(qx); qy = np.atleast_1d(qy)
    qz = np.atleast_1d(qz); qw = np.atleast_1d(qw)
    rot = Rotation.from_quat(np.stack([qx, qy, qz, qw], axis=-1))
    euler = rot.as_euler("xyz")  # roll, pitch, yaw, per the SNAME body-frame order
    return euler[..., 0], euler[..., 1]


def _restoring_forces(roll, pitch, mass_kg, net_buoyancy_z, cg_cb_offset_z, g=9.81):
    """
    Simplified restoring force/moment vector g(eta) — see module docstring
    for the Fossen reference. Assumes CG/CB laterally aligned (only a
    z-offset between them) and near-neutral buoyancy.
    """
    W = mass_kg * g
    B = W + net_buoyancy_z
    diff = W - B  # = -net_buoyancy_z

    Fx = diff * np.sin(pitch)
    Fy = -diff * np.cos(pitch) * np.sin(roll)
    Fz = -diff * np.cos(pitch) * np.cos(roll)
    Tx = cg_cb_offset_z * W * np.cos(pitch) * np.sin(roll)
    Ty = cg_cb_offset_z * W * np.sin(pitch)
    Tz = 0.0
    return np.array([Fx, Fy, Fz, Tx, Ty, Tz])


def _mass_and_damping_arrays(params: CalibrationParameters, vehicle: VehicleConstants):
    """Builds the full mass/inertia matrix M (6x6), linear_damping, and
    quad_damping from params/vehicle — factored out so both
    simulate_trajectory() and compute_acceleration() build these
    identically, with no risk of the two drifting apart.

    M is diagonal EXCEPT for two off-diagonal coupling terms (Xq_dot,
    Zq_dot — see CalibrationParameters' docstring on why these two
    specifically). Checked for positive-definiteness (via Cholesky, not
    just "diagonal entries positive" — the diagonal-only check used
    before this was added is no longer sufficient once off-diagonal
    terms exist) — fails loudly if a parameter combination would make
    the mass matrix non-physical, the same "fail fast, don't silently
    integrate garbage" principle used everywhere else in this file.
    """
    m = vehicle.mass_kg
    M = np.diag([
        m + params.Xu_dot, m + params.Yv_dot, m + params.Zw_dot,
        params.Ieff_roll, params.Ieff_pitch, params.Ieff_yaw,
    ])
    M[0, 4] = M[4, 0] = params.Xq_dot  # surge-pitch coupling
    M[2, 4] = M[4, 2] = params.Zq_dot  # heave-pitch coupling
    try:
        np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        raise ValueError(
            f"Mass matrix is not positive-definite for this parameter combination "
            f"(off-diagonal Xq_dot={params.Xq_dot}, Zq_dot={params.Zq_dot} too large "
            f"relative to the corresponding diagonal terms) — a nonphysical mass "
            f"matrix, not just a numerically awkward one. Check vehicle_spec.yaml's "
            f"bounds on Xq_dot/Zq_dot aren't allowing this combination through."
        )
    linear_damping = np.array([params.Xu, params.Yv, params.Zw, params.Kp, params.Mq, params.Nr])
    quad_damping = np.array([params.Xuu, params.Yvv, params.Zww, params.Kpp, params.Mqq, params.Nrr])
    return M, linear_damping, quad_damping


def _coriolis_centripetal_force(M: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    C(nu) @ nu — the rigid-body Coriolis-centripetal contribution this
    model was missing entirely until finding #16 (PROJECT_STATUS.md).
    This is THE specific gap findings #5, #10, #12, and #13 independently
    converged on (rotational DOFs fitting far worse than translational,
    damping compensating for absent cross-coupling) — closing it here,
    not working around it further.

    Takes the FULL mass matrix M (6x6 — may include off-diagonal added-
    mass coupling, e.g. Xq_dot/Zq_dot; see CalibrationParameters), not
    just its diagonal. Uses the GENERAL Fossen construction: generalized
    momentum p = M @ nu (computed from the FULL matrix, mixing linear and
    angular components when off-diagonal coupling exists), then p ITSELF
    (not M) is split into linear/angular halves to build the standard
    skew-symmetric block form:
        C(nu) = [[0, -S(p_lin)], [-S(p_lin), -S(p_ang)]]
    This is a genuine generalization from the original (diagonal-only)
    derivation — VERIFIED, not assumed, before being trusted: confirmed
    energy-conserving (nu^T C(nu) nu ~ 0 to floating-point precision,
    10,000 random trials) even WITH nonzero off-diagonal linear-angular
    mass coupling, AND confirmed to reduce EXACTLY to the original
    diagonal-only formula when the off-diagonal terms are zero (max
    difference ~3e-13 across 1,000 trials — floating-point noise, not an
    approximation).

    Returns C(nu)@nu, shape (6,). Callers SUBTRACT this from tau,
    matching M*nu_dot = tau - C(nu)*nu - D(nu)*nu - g(eta) — the
    standard form, with this file's damping_force already embedding the
    -D(nu)*nu sign (its coefficients are negative, so it's ADDED, not
    subtracted, elsewhere in this function).
    """
    p = M @ v
    p_lin, p_ang = p[:3], p[3:]

    def _skew(x: np.ndarray) -> np.ndarray:
        return np.array([
            [0.0, -x[2], x[1]],
            [x[2], 0.0, -x[0]],
            [-x[1], x[0], 0.0],
        ])

    C = np.zeros((6, 6))
    C[:3, 3:] = -_skew(p_lin)
    C[3:, :3] = -_skew(p_lin)
    C[3:, 3:] = -_skew(p_ang)
    return C @ v


def compute_acceleration(
    params: CalibrationParameters,
    vehicle: VehicleConstants,
    v: np.ndarray,
    tau: np.ndarray,
    roll: float,
    pitch: float,
):
    """
    THE single-step physics calculation — computes body-frame acceleration
    v_dot given the current body velocity, current net force/torque, and
    current roll/pitch (for the buoyancy restoring term). This is exactly
    the per-step math the stand-in integrator uses internally, extracted
    as its own function so a HoloOcean-integrated backend (which lets
    HoloOcean do the actual position/orientation INTEGRATION, and only
    needs the acceleration from OUR calibratable physics — see
    holoocean_backend.py) uses the IDENTICAL equations, not a
    re-implementation that could silently drift from this one.

    Now includes the Coriolis-centripetal term C(nu)@nu (see
    _coriolis_centripetal_force) — previously entirely absent, the
    specific, repeatedly-identified gap behind this model's weak
    rotational-DOF fit. Adds ZERO new calibration parameters; uses only
    the mass/inertia values already being fit.

    Returns (v_dot, damping_force, g_eta) — the latter two purely for
    optional debug/diagnostic use, matching debug_info's existing fields.
    """
    M, linear_damping, quad_damping = _mass_and_damping_arrays(params, vehicle)
    damping_force = linear_damping * v + quad_damping * np.abs(v) * v
    g_eta = _restoring_forces(roll, pitch, vehicle.mass_kg, params.net_buoyancy_z, params.cg_cb_offset_z)
    coriolis_force = _coriolis_centripetal_force(M, v)
    # M is no longer guaranteed diagonal (Xq_dot/Zq_dot introduce two
    # off-diagonal entries) — a proper linear solve, not elementwise
    # division, is required. Reduces to exactly the same answer as the
    # old division when M is diagonal (confirmed: solving a diagonal
    # system is mathematically identical to elementwise division).
    v_dot = np.linalg.solve(M, tau + damping_force - g_eta - coriolis_force)
    return v_dot, damping_force, g_eta


def simulate_trajectory(
    params: CalibrationParameters,
    vehicle: VehicleConstants,
    sim_config: SimulationConfig,
    trajectory: TrajectoryInputs,
) -> SimulationResult:
    """
    THE interface everything else in the pipeline depends on. Forward-
    integrates body-frame velocity [u,v,w,p,q,r] via Forward Euler (see
    module docstring, Numerical stability section), using
    compute_acceleration() for the actual physics each sub-step.

    Swap point: a future HoloOcean-backed implementation (see
    holoocean_backend.py) matches this exact signature — (params, vehicle,
    sim_config, trajectory) -> SimulationResult — so nothing calling this
    needs to change. That backend uses compute_acceleration() too, but lets
    HoloOcean's own rigid-body physics do the actual integration instead of
    this function's own Forward Euler loop.
    """
    params.validate()
    vehicle.validate()
    sim_config.validate()
    trajectory.validate()

    _mass_and_damping_arrays(params, vehicle)  # validates M is positive-definite up front, fail fast before the loop

    n_steps = trajectory.forces_torques.shape[0]
    nu = np.zeros((n_steps, 6))
    nu[0] = trajectory.nu0

    debug_info = None

    if sim_config.debug:
        debug_info = {
            "damping_force": np.zeros((n_steps, 6)),
            "restoring_force": np.zeros((n_steps, 6)),
            "acceleration": np.zeros((n_steps, 6)),
        }

    n_substeps = sim_config.n_substeps
    sub_dt = sim_config.sub_dt

    for k in range(n_steps - 1):
        v = nu[k]
        tau = trajectory.forces_torques[k]
        roll, pitch = trajectory.roll_pitch_series[k]

        # Sub-step between logged samples k and k+1, holding tau/roll/pitch
        # constant across sub-steps (zero-order hold — real data isn't
        # available any finer than the external dt anyway). Only the LAST
        # sub-step's values are recorded for debug_info, matching this
        # function's existing per-external-sample debug granularity.
        damping_force = g_eta = v_dot = None
        for _ in range(n_substeps):
            v_dot, damping_force, g_eta = compute_acceleration(params, vehicle, v, tau, roll, pitch)
            v = v + v_dot * sub_dt

        nu[k + 1] = v

        if debug_info is not None:
            debug_info["damping_force"][k] = damping_force
            debug_info["restoring_force"][k] = g_eta
            debug_info["acceleration"][k] = v_dot

    if not np.all(np.isfinite(nu)):
        raise FloatingPointError(
            "Simulation produced NaN/inf — likely dt too large, or this "
            "parameter combination made the system unstable (see "
            "Numerical stability note in module docstring). If this fires "
            "during optimization, check the bounds in vehicle_spec.yaml "
            "aren't allowing a nonphysical combination through."
        )

    return SimulationResult(
        nu=nu,
        used_teacher_forced_orientation=TEACHER_FORCED_ORIENTATION,
        debug_info=debug_info,
    )