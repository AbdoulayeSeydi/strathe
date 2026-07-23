"""
HoloOcean-integrated backend for simulate_trajectory() — matches the
stand-in's exact signature, so objective.py/optimizer.py/run_calibration.py
need ZERO changes to use this instead.

## Architecture — why this is NOT "calibrating HoloOcean's own physics"
Confirmed via HoloOcean's own docs: HoveringAUV's default-scheme physics
is NOT exposed for parameter adjustment at all ("modifying [these
variables] will have no effect on results"). The newer Fossen-dynamics
customization system is explicitly torpedo-vehicle-only as of the current
release. So instead: OUR calibratable dynamics_model.compute_acceleration()
(the exact same equations the stand-in model uses — see dynamics_model.py)
computes body-frame acceleration each tick; HoloOcean's HoveringAUV Custom
Dynamics control scheme (confirmed: agent.set_control_scheme(2)) takes that
acceleration and does the actual rigid-body INTEGRATION, collision
handling, and sensor simulation. Buoyancy/gravity/damping are disabled
internally under this scheme — deliberately, since OUR equations already
compute all of that.

Because we never touch HoloOcean's own physics configuration, the SAME
running environment can be reused across every optimizer evaluation and
restart — no per-parameter-guess restart cost.

## Confirmed API facts (via diagnose_holoocean_api.py / diagnose_holoocean_frames.py
   run against the real installed HoloOcean — NOT assumed)
- agent.set_control_scheme(2) enables Custom Dynamics (6-vector WORLD-frame
  linear+angular acceleration; buoyancy/gravity/damping disabled, collisions active)
- env.step(command) works directly with a single agent (no per-agent dict needed
  for this scenario shape)
- state['PoseSensor']: (4,4) world-frame homogeneous transform (position in
  column 3, rotation matrix in the top-left 3x3, body->world)
- state['VelocitySensor']: (3,) WORLD-FRAME linear velocity — CONFIRMED
  empirically (a world-frame push after a 90deg yaw still showed up on the
  original axis, not the rotated one — the world-frame signature)
- state['IMUSensor']: (4,3) — row 0 = linear acceleration (specific force),
  row 1 = angular velocity (BODY-frame, as any physical IMU must be — it's
  rigidly mounted to and rotates with the vehicle), rows 2-3 = bias estimates
  (matches the sensor's AccelBiasSigma/AngVelBiasSigma/ReturnBias config)
- agent.set_physics_state(location, rotation, velocity, angular_velocity) and
  agent.teleport(location=None, rotation=None) — real signatures confirmed

## Remaining assumption, not yet empirically isolated
set_physics_state's `angular_velocity` parameter's frame (world vs body) was
NOT independently tested the way VelocitySensor's readback frame was —
assumed WORLD-frame here for symmetry with VelocitySensor's confirmed
readback convention, and because that's how Unreal's underlying rigid-body
physics typically represents state internally. If initial spin direction
looks wrong in the real stress test, this is the first thing to revisit.

## A second known simplification, same one used for the stand-in
Converting BODY-frame angular acceleration to WORLD-frame via the same
rotation matrix used for linear acceleration (world_ang_accel = R @
body_ang_accel) is a practical approximation, not the fully rigorous
kinematic relationship between angular acceleration in rotating frames.
Reasonable for the same reason the stand-in's teacher-forcing was
reasonable — flagged here rather than silently assumed correct.

## This is a genuine forward simulation, NOT teacher-forced
Unlike the stand-in model (which feeds REAL LOGGED roll/pitch at every
step specifically to avoid compounding integration error), this backend
reads orientation/velocity back from HoloOcean's OWN physics each tick —
a real closed-loop simulation. RMSE from this backend is therefore NOT
directly comparable to the stand-in's RMSE without accounting for that
difference — this is a harder, more honest evaluation, and results may
look different (likely worse, due to compounding error) even with
identical calibrated parameters.
## Body-frame convention fix — confirmed via diagnose_rotation_baseline.py,
not a guess. HoloOcean's PoseSensor reports rotation matrices such that
body-frame Z points DOWN relative to a Z-up world frame at "zero rotation"
— i.e. HoloOcean's native body frame already uses the SAME SNAME/Fossen
convention as our own dynamics_model.py (body Z = down/heave), but
expresses it relative to Unreal's Z-up world. The correction
R_true = R_reported @ diag(1,-1,-1) was verified EXACTLY (not
approximately) against two independent real HoloOcean tests (10deg yaw,
10deg roll) — both recovered the textbook-exact standard rotation matrix
after this correction. Only the rotation matrix needs it; IMUSensor's
angular velocity is already reported in HoloOcean's native body frame,
which already matches our Fossen convention.

"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dynamics_model import (
    CalibrationParameters, VehicleConstants, SimulationConfig, TrajectoryInputs,
    SimulationResult, compute_acceleration,
)

CUSTOM_DYNAMICS_SCHEME = 2

# Confirmed via diagnose_rotation_baseline.py — see module docstring above.
_BODY_FRAME_CORRECTION = np.diag([1.0, -1.0, -1.0])


def _pose_to_rotation_and_position(pose: np.ndarray):
    """PoseSensor's 4x4 -> (3x3 CORRECTED rotation matrix, (3,) position).
    Applies _BODY_FRAME_CORRECTION so the returned R matches our own
    Fossen/SNAME body-frame convention exactly — see module docstring."""
    pose = np.asarray(pose)
    R_reported = pose[:3, :3]
    R = R_reported @ _BODY_FRAME_CORRECTION
    position = pose[:3, 3]
    return R, position


def _rotation_matrix_to_roll_pitch(R: np.ndarray):
    """Body->world rotation matrix -> (roll, pitch) in radians, matching
    dynamics_model.py's existing roll/pitch convention elsewhere."""
    roll, pitch, _yaw = Rotation.from_matrix(R).as_euler("xyz")
    return roll, pitch


def _extract_body_velocity(state: dict, R: np.ndarray) -> np.ndarray:
    """
    Builds body-frame [u,v,w,p,q,r] from HoloOcean's state dict:
      - VelocitySensor is WORLD-frame linear velocity (confirmed) -> rotate to body via R.T
      - IMUSensor row 1 is BODY-frame angular velocity (confirmed by IMU physics) -> use as-is
    """
    world_lin_vel = np.asarray(state["VelocitySensor"])
    body_lin_vel = R.T @ world_lin_vel
    body_ang_vel = np.asarray(state["IMUSensor"])[1]
    return np.concatenate([body_lin_vel, body_ang_vel])


def simulate_trajectory_holoocean(
    params: CalibrationParameters,
    vehicle: VehicleConstants,
    sim_config: SimulationConfig,
    trajectory: TrajectoryInputs,
    env,
    agent_name: str = "auv0",
    initial_location=None,
) -> SimulationResult:
    """
    Same signature/shape as dynamics_model.simulate_trajectory(), but
    drives a REAL, already-created HoloOcean environment via HoveringAUV's
    Custom Dynamics scheme instead of an internal Forward Euler integrator.

    `env` must already exist (holoocean.make(...)) — this function does
    NOT create or destroy it, precisely so it can be reused across every
    optimizer evaluation without per-guess restart cost.
    """
    params.validate()
    vehicle.validate()
    trajectory.validate()

    agent = env.agents[agent_name]
    agent.set_control_scheme(CUSTOM_DYNAMICS_SCHEME)

    n_steps = trajectory.forces_torques.shape[0]
    nu = np.zeros((n_steps, 6))

    # Set initial orientation from the run's real logged roll/pitch (yaw
    # unknown from a 2-angle roll/pitch series — assumed 0 for the initial
    # heading; only relative rotation from this point matters for physics).
    roll0, pitch0 = trajectory.roll_pitch_series[0]
    rotation_deg0 = [np.degrees(roll0), np.degrees(pitch0), 0.0]
    R0 = Rotation.from_euler("xyz", [roll0, pitch0, 0.0]).as_matrix()

    # nu0 is body-frame [u,v,w,p,q,r] — convert to world frame for
    # set_physics_state (assumed world-frame for both velocity fields;
    # confirmed for linear via VelocitySensor's readback convention, NOT
    # independently confirmed for angular — see module docstring).
    world_lin_vel0 = R0 @ trajectory.nu0[:3]
    # Applying the SAME diag(1,-1,-1) correction here too, extrapolating from
    # the confirmed pattern (rotation matrix + angular accel input both need
    # it) — but this SPECIFIC spot (set_physics_state's angular_velocity
    # parameter) has NOT been independently tested the way the other three
    # were. If initial spin direction looks wrong in a real run, check this
    # line first.
    world_ang_vel0 = _BODY_FRAME_CORRECTION @ (R0 @ trajectory.nu0[3:])
    agent.set_physics_state(
        location=initial_location if initial_location is not None else [0.0, 0.0, 0.0],
        rotation=rotation_deg0,
        velocity=world_lin_vel0,
        angular_velocity=world_ang_vel0,
    )

    nu[0] = trajectory.nu0
    R = R0

    debug_info = None
    if sim_config.debug:
        debug_info = {
            "damping_force": np.zeros((n_steps, 6)),
            "restoring_force": np.zeros((n_steps, 6)),
            "acceleration": np.zeros((n_steps, 6)),
        }

    for k in range(n_steps - 1):
        v = nu[k]
        tau = trajectory.forces_torques[k]
        roll, pitch = _rotation_matrix_to_roll_pitch(R)

        # OUR calibratable physics — identical equations to the stand-in
        # model, via the shared compute_acceleration() function.
        v_dot, damping_force, g_eta = compute_acceleration(params, vehicle, v, tau, roll, pitch)

        # Body-frame acceleration -> world-frame for HoloOcean's Custom
        # Dynamics input (confirmed: it wants global-frame accel).
        world_lin_accel = R @ v_dot[:3]  # linear input confirmed unflipped — see module docstring
        world_ang_accel_raw = R @ v_dot[3:]  # practical approximation — see module docstring
        # CONFIRMED via diagnose_angular_input_sign.py (tested independently on
        # all 3 axes): HoloOcean's Custom Dynamics angular acceleration INPUT
        # needs the SAME diag(1,-1,-1) correction as the rotation matrix — X
        # unchanged, Y and Z negated. This is a SEPARATE fix from the rotation
        # matrix one (confirmed via static teleport tests, no dynamics
        # involved) — they happen to share the same correction matrix because
        # both stem from one consistent underlying convention difference that
        # only affects rotation-related quantities, not linear ones (linear
        # acceleration input confirmed clean, no correction, via
        # diagnose_linear_input_sign.py).
        world_ang_accel = _BODY_FRAME_CORRECTION @ world_ang_accel_raw
        command = np.concatenate([world_lin_accel, world_ang_accel])

        state = env.step(command)

        R, _position = _pose_to_rotation_and_position(state["PoseSensor"])
        nu[k + 1] = _extract_body_velocity(state, R)

        if debug_info is not None:
            debug_info["damping_force"][k] = damping_force
            debug_info["restoring_force"][k] = g_eta
            debug_info["acceleration"][k] = v_dot

    if not np.all(np.isfinite(nu)):
        raise FloatingPointError(
            "HoloOcean-backed simulation produced NaN/inf — check parameter "
            "bounds, or whether HoloOcean itself reported an error mid-run."
        )

    return SimulationResult(nu=nu, used_teacher_forced_orientation=False, debug_info=debug_info)