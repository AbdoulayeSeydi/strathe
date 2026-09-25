"""
MuJoCo-integrated backend for simulate_trajectory() -- matches the
stand-in's exact signature (plus a model/data handle), so
objective.py/optimizer.py/run_calibration.py need ZERO changes to use
this instead. Mirrors holoocean_backend.py's shape and documentation
discipline directly.

## Architecture -- why this is full-override, not native rigid-body
B0.5 and B0.6 independently spiked both candidate approaches for wiring
dynamics_model.compute_acceleration() (the SAME calibratable equations
the stand-in and HoloOcean backends both use) into MuJoCo. B0.7's
decision record (track_b_config.py) documents the evidence: native
rigid-body diverged from the calibrated stand-in by >100% even at rest
(added mass and cross-coupling are first-order effects for this vehicle,
not small corrections), and by 10-20x more once realistic motion was
introduced -- at velocities SMALLER than what this project's trained
policies actually produce. This file implements full-override: MuJoCo's
own bias force (gravity + Coriolis) is read back and cancelled every
substep, replaced by compute_acceleration()'s calibrated result via an
injected wrench. assert_uses_full_override() is called below, not just
imported for documentation -- if this file ever drifted from B0.7's
decision, it fails loudly here, not silently downstream.

## This is TEACHER-FORCED, matching the stand-in -- NOT yet genuine
forward simulation (that's B1.6, a later, deliberate change)
Like dynamics_model.simulate_trajectory() and UNLIKE
holoocean_backend.py's simulate_trajectory_holoocean(), this function
feeds the REAL LOGGED roll/pitch (trajectory.roll_pitch_series) into
MuJoCo's own orientation state at every step, rather than letting
MuJoCo integrate orientation from angular velocity. Per B1.5's explicit
staging, this isolates whether the force/damping computation itself
matches the stand-in before any orientation-integration difference can
confound the comparison. used_teacher_forced_orientation=True below,
matching the stand-in's own TEACHER_FORCED_ORIENTATION flag -- NOT
HoloOcean's False.

## Frame conventions: ALL confirmed empirically, not assumed
Every frame-convention question this file depends on was independently
tested against real MuJoCo before being relied on here, mirroring
holoocean_backend.py's own discipline (each of its corrections came from
a dedicated diagnose_*.py script, not documented convention alone):

  1. Freejoint qvel LINEAR component is WORLD-frame -- confirmed via a
     push-and-observe test at two orientations (b1_3_velocity_frame_check.py),
     mirroring HoloOcean's own VelocitySensor test exactly.
  2. Freejoint qvel ANGULAR component is BODY-frame -- confirmed via an
     analogous rotation-axis test (b1_check_angular_velocity_frame.py).
  3. Quaternion storage/readback and the resulting rotation matrix match
     this project's SNAME/Fossen convention EXACTLY, with NO axis-sign
     correction needed (unlike HoloOcean's confirmed diag(1,-1,-1)) --
     confirmed via b1_2_orientation_check.py (static storage) and
     b1_4_frame_convention_check.py (yaw-alone, roll-alone, and combined
     rotations, all exact to floating-point precision).

compute_acceleration()'s v is fully body-frame (Fossen/SNAME convention,
matching the stand-in and HoloOcean both) -- so only the linear part of
any MuJoCo-facing acceleration is ever rotated (via R, the teacher-forced
orientation); the angular part is used directly, matching MuJoCo's own
convention with no conversion needed.

## Divergence bug: found, diagnosed, and fixed -- root cause confirmed
## against real MuJoCo, not guessed
Nonzero rotation previously diverged. The actual mechanism (verified
algebraically against real MuJoCo's own reported qacc at multiple
rotation angles, exact match before the fix was written): MuJoCo's
xfrc_applied TORQUE component must be specified in WORLD frame (its
documented Cartesian-wrench convention) -- MuJoCo internally converts it
via R.T before adding it as the generalized force conjugate to qvel[3:]
(confirmed body-frame). This function previously wrote the desired
BODY-FRAME generalized torque directly into xfrc_applied, so MuJoCo's own
R.T conversion then corrupted it -- invisible at identity (R=I), scaling
with rotation angle otherwise. Fixed in _inject_full_override_wrench by
rotating the angular part into world frame via R before injection, so
MuJoCo's own R.T recovers it exactly. No longer guarded; nonzero rotation
is correctly supported. NOTE: the fake-mujoco harness used in
test_mujoco_backend.py never modeled this R.T conversion (it always
treated xfrc_applied as directly the generalized force) -- it has been
updated accordingly; see that file's own docstring.

## B1.2 addition: MuJoCo-quat -> roll/pitch extraction (not yet wired in)
_extract_roll_pitch_from_mujoco_quat() reads data.qpos[3:7] and returns
(roll, pitch) via scipy, matching dynamics_model.py's own convention --
verified against scipy directly AND against real MuJoCo's actual
reported quaternion behavior (b1_2_orientation_check.py,
b1_4_frame_convention_check.py). Not called anywhere in
simulate_trajectory_mujoco yet -- that function stays fully teacher-forced
until B1.6 explicitly wires this in, per B1.5/B1.6's own staging.

## B3.1/B3.2/B3.3 addition: multi-body support (see conversation)
Every function below that reads/writes qpos/qvel/xfrc_applied now takes
an explicit body_id (default 1, preserving exact single-body behavior)
and, where needed, the body's own qpos_adr/dof_adr within the shared
array -- computed once via _body_qpos_dof_addr(), using MuJoCo's own
body_jntadr/jnt_qposadr/body_dofadr fields (confirmed directly against
MuJoCo's mjmodel.h source before use, not guessed). For body_id=1 in a
single-body scene, qpos_adr=dof_adr=0 exactly, so every existing
single-body call site (which never specified these, relying on the old
hardcoded [:7]/[:6]/_BODY_ID=1 slicing) behaves identically to before --
confirmed by construction, not just assumed, since the address
computation reduces to the same offsets in that case. B3.3 adds
_extract_body_position_mujoco -- position was previously always
zeroed/ignored (only velocity mattered for calibration RMSE); this is
the first function that reports it meaningfully, needed for future
multi-agent coordination work. NOTE on naming: B3.3's own task text uses
0-indexed "body 0"/"body 1" for the two AGENTS, which is NOT the same as
MuJoCo's own 1-indexed body_id (where 0 is always the implicit world
body) -- agent 0 = MuJoCo body_id 1 = auv_placeholder, agent 1 = MuJoCo
body_id 2 = auv_placeholder_2. b3_3_position_indexing_test.py makes this
mapping explicit rather than assumed, since conflating the two would be
exactly the kind of indexing bug this task exists to catch.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dynamics_model import (
    CalibrationParameters, VehicleConstants, SimulationConfig, TrajectoryInputs,
    SimulationResult, compute_acceleration, compute_acceleration_from_rotation_matrix,
)
from arena_core import ArenaState  # C0.3 addition -- one-directional (arena_core.py does
                                     # NOT import from this file), so not circular
from track_b_config import MUJOCO_DYNAMICS_INTEGRATION_APPROACH, assert_uses_full_override

import mujoco

_BODY_ID = 1  # index 0 is MuJoCo's implicit world body (confirmed in B0.3) -- kept as the
              # default body_id everywhere below, not removed, so every existing single-body
              # call site needs zero changes.


def _body_qpos_dof_addr(model, body_id: int):
    """Returns (qpos_adr, dof_adr) for a body's own freejoint -- the
    starting index into qpos (7 elements: xyz+quat, wxyz) and
    qvel/qacc/dof (6 elements: linear+angular) respectively, within the
    SHARED array covering every body in the scene. Confirmed exact
    MuJoCo API fields directly against MuJoCo's own mjmodel.h source
    (body_jntadr: start addr of joints; jnt_qposadr: start addr in qpos
    for a joint's data; body_dofadr: start addr of dofs) before use --
    getting this wrong would silently address the wrong body's state,
    exactly the failure B3.2's cross-talk test exists to catch. For
    body_id=1 in a single-body scene this returns (0, 0) -- the same
    offsets the old hardcoded [:7]/[:6] slicing always assumed."""
    joint_id = model.body_jntadr[body_id]
    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.body_dofadr[body_id]
    return qpos_adr, dof_adr


def _extract_roll_pitch_from_mujoco_quat(data, qpos_adr: int = 0):
    """MuJoCo qpos[qpos_adr+3:qpos_adr+7] (scalar-first [w,x,y,z],
    MuJoCo's documented convention) -> (roll, pitch), matching
    dynamics_model.py's own convention (quaternion_to_roll_pitch, via
    scipy.spatial.transform.Rotation's [x,y,z,w] ordering and 'xyz'
    intrinsic Euler angles). Verified against scipy directly AND against
    MuJoCo's REAL reported quaternion behavior (b1_2_orientation_check.py,
    b1_4_frame_convention_check.py -- yaw-alone, roll-alone, and combined
    rotations, all exact to floating-point precision, no correction
    needed unlike HoloOcean's confirmed diag(1,-1,-1)).

    NOT YET WIRED into simulate_trajectory_mujoco's main loop -- that
    function is still fully teacher-forced (B1.1/B1.5's staging); this
    extraction path is only needed once B1.6 switches to genuine
    self-integration. Built and unit-tested now as its own standalone
    piece, per B1.2's own scope, so B1.6 has a ready, already-verified
    building block to wire in rather than writing and testing it for the
    first time then."""
    w, x, y, z = data.qpos[qpos_adr + 3:qpos_adr + 7]
    quat_xyzw = np.array([x, y, z, w])
    roll, pitch, _yaw = Rotation.from_quat(quat_xyzw).as_euler("xyz")
    return roll, pitch


def _roll_pitch_to_quat_wxyz(roll: float, pitch: float) -> np.ndarray:
    """(roll, pitch) -> MuJoCo quaternion [w,x,y,z]. Yaw unknown from a
    2-angle roll/pitch series -- assumed 0, matching
    holoocean_backend.py's own identical limitation and comment."""
    quat_xyzw = Rotation.from_euler("xyz", [roll, pitch, 0.0]).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])


def _set_teacher_forced_state(data, roll: float, pitch: float, v_body: np.ndarray,
                                qpos_adr: int = 0, dof_adr: int = 0) -> np.ndarray:
    """Sets MuJoCo's qpos/qvel to match the real logged orientation and
    the current body-frame velocity, returning R (body->world) for the
    caller's own use. Frame conventions confirmed empirically (see
    conversation): freejoint qvel linear=world (B1.3), angular=body
    (dedicated check), and quaternion storage/rotation-matrix behavior
    matches this project's convention exactly with no correction needed
    (B1.2, B1.4). Nonzero rotation is now correctly supported -- see
    _inject_full_override_wrench's own docstring for how the previously
    confirmed divergence bug was fixed using these same confirmed
    conventions.

    qpos_adr/dof_adr (see _body_qpos_dof_addr) address THIS body's own
    7/6-element slice within the shared multi-body qpos/qvel arrays --
    default 0 preserves the exact single-body behavior."""
    data.qpos[qpos_adr:qpos_adr + 3] = 0.0  # position is irrelevant here -- this backend only ever reports nu, not position
    data.qpos[qpos_adr + 3:qpos_adr + 7] = _roll_pitch_to_quat_wxyz(roll, pitch)

    R = Rotation.from_euler("xyz", [roll, pitch, 0.0]).as_matrix()
    data.qvel[dof_adr:dof_adr + 3] = R @ v_body[:3]   # body -> world (confirmed convention, B1.3)
    data.qvel[dof_adr + 3:dof_adr + 6] = v_body[3:]   # angular already body-frame (confirmed convention)
    return R


def _extract_body_velocity_mujoco(data, R: np.ndarray, dof_adr: int = 0) -> np.ndarray:
    """MuJoCo qvel -> body-frame [u,v,w,p,q,r], mirroring
    holoocean_backend.py's own _extract_body_velocity naming/shape.
    dof_adr addresses this body's own 6-element slice, default 0
    preserves single-body behavior."""
    world_lin_vel = data.qvel[dof_adr:dof_adr + 3].copy()
    body_lin_vel = R.T @ world_lin_vel
    body_ang_vel = data.qvel[dof_adr + 3:dof_adr + 6].copy()  # already body-frame (assumed, see docstring)
    return np.concatenate([body_lin_vel, body_ang_vel])


def _extract_body_position_mujoco(data, qpos_adr: int = 0) -> np.ndarray:
    """MuJoCo qpos -> world-frame [x,y,z] position for ONE body, using
    its own qpos_adr (see _body_qpos_dof_addr). B3.3 addition (see
    conversation): position was previously always zeroed/ignored by this
    backend entirely (only nu/velocity mattered for calibration RMSE --
    see _set_teacher_forced_state's own comment, "position is irrelevant
    here"). This is the first function that reports it meaningfully,
    needed for future multi-agent coordination work where relative
    position between bodies matters. Deliberately separate from
    _extract_body_velocity_mujoco rather than folded into it -- position
    and velocity have different frame considerations (position needs no
    body/world rotation, velocity does) and different call-site needs
    (most existing call sites never want position at all)."""
    return data.qpos[qpos_adr:qpos_adr + 3].copy()


def set_body_pose(model, data, body_id: int, position: np.ndarray, orientation: Rotation) -> None:
    """
    C0.1 addition (see conversation) -- the public, representation-hiding
    write side of the shared-scene pose API. Writes a full, valid pose
    (position AND a complete orientation quaternion, never position
    alone leaving a stale/invalid quaternion behind) into body_id's own
    qpos slice, then propagates via mj_kinematics -- forward kinematics
    ONLY (confirmed directly against MuJoCo's own API reference: this is
    the minimal sub-component, lighter than mj_fwdKinematics, and
    triggers NO collision detection or dynamics at all) -- so
    get_body_position's own read of data.xpos is immediately correct and
    internally consistent. Deliberately does NOT call mj_forward: this
    step establishes MuJoCo as the shared-scene POSE source only, not
    yet the authority for dynamics, contacts, or termination -- that is
    explicitly later, separate scope (see conversation).

    orientation: a scipy.spatial.transform.Rotation (body->world),
    matching ArenaState's own convention exactly -- callers (e.g.
    TwoAgentOrchestrator) pass ArenaCore.state.orientation directly, with
    no conversion of their own. This function owns every MuJoCo-specific
    representation detail (qpos layout, the wxyz/xyzw quaternion
    ordering difference) internally, so callers never need to know
    MuJoCo's own conventions at all -- the clean API boundary this
    session's own design discussion called for.
    """
    qpos_adr, _ = _body_qpos_dof_addr(model, body_id)
    quat_xyzw = orientation.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    data.qpos[qpos_adr:qpos_adr + 3] = np.asarray(position, dtype=float)
    data.qpos[qpos_adr + 3:qpos_adr + 7] = quat_wxyz
    mujoco.mj_kinematics(model, data)


def get_body_position(model, data, body_id: int) -> np.ndarray:
    """
    C0.1 addition (see conversation) -- the public, representation-hiding
    read side of the shared-scene pose API. Returns body_id's current
    world-frame [x,y,z] position from MuJoCo's own computed global
    Cartesian pose (data.xpos, populated by set_body_pose's own internal
    mj_kinematics call) -- NOT raw qpos directly. This is deliberately
    more general than _extract_body_position_mujoco's own raw-qpos read
    above (which is only correct for a body whose qpos IS its world
    position outright, true for a simple freejoint body but not in
    general for a body nested in a kinematic chain) -- xpos is MuJoCo's
    own always-correct global position for ANY body, indexed directly by
    body_id like xfrc_applied.
    """
    return data.xpos[body_id].copy()


def _inject_full_override_wrench(model, data, desired_v_dot: np.ndarray, R: np.ndarray,
                                   body_id: int = _BODY_ID, dof_adr: int = 0,
                                   self_integrating_orientation: bool = False) -> None:
    """B0.7's confirmed full-override approach: cancel MuJoCo's own bias
    force (gravity + Coriolis, from ITS OWN mass matrix) and inject a
    wrench that makes the realized acceleration match desired_v_dot
    exactly -- verified to machine precision in b0_5_spike_full_override.py.
    Note: tau is NOT a separate argument here and is not added again --
    compute_acceleration() already folded tau's effect into desired_v_dot
    via M^-1 (see dynamics_model.compute_acceleration), so re-adding it
    would double-count it.

    ROOT CAUSE FOUND AND FIXED (see conversation -- verified algebraically
    against real MuJoCo's own reported qacc, exact match at multiple
    rotation angles before this fix was written, not guessed): MuJoCo's
    xfrc_applied TORQUE component must be specified in WORLD frame (its
    documented convention for a Cartesian wrench applied at a body's
    center of mass) -- MuJoCo internally converts it to the generalized
    force via R.T @ xfrc_applied[3:] before adding it to the equations of
    motion, since qvel[3:] (confirmed body-frame) requires its conjugate
    generalized force to also be body-frame. The LINEAR part needs no such
    conversion (confirmed: it matched exactly at every rotation tested).
    Previously, this function wrote the desired BODY-FRAME generalized
    torque directly into xfrc_applied[3:], so MuJoCo's own R.T conversion
    then corrupted it -- invisible at identity (R=I) but scaling with
    rotation angle, exactly matching every divergence observed on real
    data. Fixed by rotating the angular part into world frame via R
    before writing it, so MuJoCo's own R.T recovers it exactly:
    R.T @ (R @ x) = x.

    B3.2 addition (see conversation): model.qfrc_bias and the full mass
    matrix now cover EVERY body in the scene (shape (nv,) / (nv,nv), not
    just this one body's own 6). For mechanically disconnected free
    bodies (no shared joint, no contact -- confirmed via B3.1's own
    contact="disable" flag), the mass matrix is expected to be exactly
    block-diagonal and each body's own qfrc_bias slice independent of
    every other body's state -- this function extracts JUST this body's
    own 6x6 diagonal block and 6-element bias sub-vector via dof_adr,
    which is mathematically equivalent to a true single-body computation
    IF AND ONLY IF that block-diagonal/independence property actually
    holds. B3.2's own test verifies this directly (both by comparing
    trajectories AND by checking the off-diagonal blocks are exactly
    zero), rather than assuming it from this docstring's own physics
    argument alone."""
    assert_uses_full_override(MUJOCO_DYNAMICS_INTEGRATION_APPROACH)

    mujoco.mj_forward(model, data)
    qfrc_bias_full = data.qfrc_bias.copy()  # (nv,) -- ALL bodies in the scene
    M_mj_full = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, M_mj_full)  # confirmed 3.12.0 signature -- see b0_5's own fix

    # This body's own 6x6 diagonal block / 6-element bias slice --
    # see docstring above for why this is only correct if the two
    # bodies are truly mechanically independent, which B3.2 verifies.
    qfrc_bias = qfrc_bias_full[dof_adr:dof_adr + 6]
    M_mj = M_mj_full[dof_adr:dof_adr + 6, dof_adr:dof_adr + 6]

    # MuJoCo's free-joint linear qvel is WORLD-frame. When orientation
    # genuinely evolves (self_integrating_orientation=True: MuJoCoArenaCore
    # and forward-sim mode, which read body velocity back with the UPDATED
    # rotation), the world linear acceleration that realizes body-frame
    # v_b_dot is d/dt(R v_b) = R (v_b_dot + omega x v_b). That omega x v_b
    # transport term used to be missing on those paths: a systematic
    # sideways drift whenever a vehicle turned while moving (1.4 cm after
    # 2 s at ~0.8 rad/s yaw) that did NOT shrink with physics_dt, which is
    # how it was told apart from ordinary integrator differences -- see
    # tests/test_mujoco_transport.py.
    # Teacher-forced mode must NOT add it: it reads velocity back with the
    # PRE-step rotation, matching the stand-in's fixed-orientation update
    # v += v_dot*dt exactly (verified against real MuJoCo to ~1e-15).
    world_lin_acc = R @ desired_v_dot[:3]
    if self_integrating_orientation:
        body_lin_vel = R.T @ data.qvel[dof_adr:dof_adr + 3]
        body_ang_vel = data.qvel[dof_adr + 3:dof_adr + 6]
        world_lin_acc = world_lin_acc + R @ np.cross(body_ang_vel, body_lin_vel)
    desired_qacc_mujoco_frame = np.concatenate([world_lin_acc, desired_v_dot[3:]])
    generalized_wrench = M_mj @ desired_qacc_mujoco_frame + qfrc_bias
    # LINEAR part: xfrc_applied's force component is already world-frame,
    # matching the generalized force directly -- no further rotation.
    # ANGULAR part: rotate into world frame so MuJoCo's own internal R.T
    # conversion recovers the intended body-frame generalized torque.
    injected_wrench = np.concatenate([generalized_wrench[:3], R @ generalized_wrench[3:]])
    data.xfrc_applied[body_id, :] = injected_wrench


def _current_rotation_matrix(data, qpos_adr: int = 0) -> np.ndarray:
    """MuJoCo qpos[qpos_adr+3:qpos_adr+7] -> full 3x3 rotation matrix
    (body->world), including whatever yaw has evolved via
    self-integration -- unlike _extract_roll_pitch_from_mujoco_quat,
    which deliberately discards yaw for compute_acceleration()'s own
    roll/pitch-only interface, this is needed for correctly converting
    qvel/qacc between body and world frame, which depends on the FULL
    orientation, not just roll/pitch. qpos_adr addresses this body's own
    slice, default 0 preserves single-body behavior."""
    w, x, y, z = data.qpos[qpos_adr + 3:qpos_adr + 7]
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def simulate_trajectory_mujoco(
    params: CalibrationParameters,
    vehicle: VehicleConstants,
    sim_config: SimulationConfig,
    trajectory: TrajectoryInputs,
    model,
    data,
    *,
    teacher_forced: bool = False,
    body_id: int = _BODY_ID,
) -> SimulationResult:
    """
    Same signature/shape as dynamics_model.simulate_trajectory(), plus a
    MuJoCo model/data handle, plus an optional body_id (B3.2 addition,
    default _BODY_ID=1 -- every existing single-body call site needs no
    changes).

    B1.6: genuine forward simulation is now the DEFAULT
    (teacher_forced=False) -- MuJoCo self-integrates both orientation and
    velocity, matching holoocean_backend.py's own non-teacher-forced
    operation. Initial orientation comes from trajectory.roll_pitch_series[0]
    (yaw assumed 0, same known limitation as HoloOcean's own
    initialization -- a 2-angle series can't carry yaw), matching
    holoocean_backend.py's own init pattern exactly. At every substep,
    R and body-frame velocity are read back from MuJoCo's OWN, evolving
    state, feeding compute_acceleration_from_rotation_matrix() directly
    -- NOT the roll/pitch extraction path (see FIXES below).
    used_teacher_forced_orientation=False in the returned result,
    matching HoloOcean's own convention.

    teacher_forced=True preserves the ORIGINAL B1.1-B1.5 behavior exactly
    (real logged roll/pitch drives orientation every step; only velocity
    is integrated) -- kept, not deleted, because B1.5's parity test
    against dynamics_model.simulate_trajectory() remains a valuable
    regression check isolating the force/damping computation from any
    future orientation-integration question. Explicitly requested by
    passing teacher_forced=True; make_mujoco_simulate_fn's own closure
    does not pass it, so calibration use gets the new default.

    B3.2 addition: body_id selects WHICH body in a multi-body scene this
    call actuates and reports on -- qpos_adr/dof_adr are computed once
    from it (see _body_qpos_dof_addr) and threaded through every helper
    call below. Other bodies in the same scene are left completely
    untouched by this call; actuating them (e.g. with a different
    parameter set) requires a SEPARATE simulate_trajectory_mujoco call
    with their own body_id, sharing the same model/data -- see B3.2's own
    test script for exactly this pattern.

    `model`/`data` must already exist (mujoco.MjModel.from_xml_path(...),
    mujoco.MjData(model)) -- this function does not create or destroy
    them, mirroring holoocean_backend.py's own `env` reuse pattern so the
    same MuJoCo instance can be reused across every optimizer evaluation.
    """
    params.validate()
    vehicle.validate()
    sim_config.validate()
    trajectory.validate()

    qpos_adr, dof_adr = _body_qpos_dof_addr(model, body_id)

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
    original_timestep = model.opt.timestep
    model.opt.timestep = sub_dt  # match the stand-in's own sub_dt exactly, not MuJoCo's XML default

    try:
        if teacher_forced:
            for k in range(n_steps - 1):
                v = nu[k]
                tau = trajectory.forces_torques[k]
                roll, pitch = trajectory.roll_pitch_series[k]
                cb = None if trajectory.current_body_series is None else trajectory.current_body_series[k]

                damping_force = g_eta = v_dot = None
                for _ in range(n_substeps):
                    v_dot, damping_force, g_eta = compute_acceleration(params, vehicle, v, tau, roll, pitch, cb)

                    R = _set_teacher_forced_state(data, roll, pitch, v, qpos_adr, dof_adr)
                    _inject_full_override_wrench(model, data, v_dot, R, body_id, dof_adr)
                    mujoco.mj_step(model, data)
                    v = _extract_body_velocity_mujoco(data, R, dof_adr)

                nu[k + 1] = v

                if debug_info is not None:
                    debug_info["damping_force"][k] = damping_force
                    debug_info["restoring_force"][k] = g_eta
                    debug_info["acceleration"][k] = v_dot
        else:
            if trajectory.current_body_series is not None:
                raise NotImplementedError(
                    "a per-sample body-axes current only makes sense with teacher-forced orientation; "
                    "for forward simulation in a current use MuJoCoArenaCore.current_world")
            # B1.6: genuine forward simulation -- initialize once, then
            # let MuJoCo self-integrate for the entire trajectory.
            roll0, pitch0 = trajectory.roll_pitch_series[0]
            data.qpos[qpos_adr:qpos_adr + 3] = 0.0
            data.qpos[qpos_adr + 3:qpos_adr + 7] = _roll_pitch_to_quat_wxyz(roll0, pitch0)
            R0 = Rotation.from_euler("xyz", [roll0, pitch0, 0.0]).as_matrix()
            data.qvel[dof_adr:dof_adr + 3] = R0 @ trajectory.nu0[:3]
            data.qvel[dof_adr + 3:dof_adr + 6] = trajectory.nu0[3:]

            for k in range(n_steps - 1):
                tau = trajectory.forces_torques[k]

                damping_force = g_eta = v_dot = None
                for _ in range(n_substeps):
                    # Read CURRENT state back from MuJoCo's own, evolving
                    # orientation/velocity -- not overwritten from logged data.
                    # Uses compute_acceleration_from_rotation_matrix() directly
                    # with R -- NOT the roll/pitch extraction path. To be
                    # precise about what this avoids (see conversation): the
                    # near-universal gimbal-lock PROXIMITY itself is inherent
                    # to the calibrated physics under open-loop replay,
                    # confirmed independently of this extraction question via
                    # a MuJoCo-free integrator -- NOT something this fix
                    # resolves. What this fix does eliminate: a separate,
                    # smaller, CONFIRMED discrepancy that occurs specifically
                    # at the exact threshold where scipy's own gimbal-lock
                    # detection activates (~89.999999 degrees pitch in
                    # testing) and forcibly zeros the extracted yaw,
                    # producing a small but real (~1e-8 scale) discontinuity.
                    # Calling compute_acceleration_from_rotation_matrix()
                    # avoids this by never extracting an angle at all.
                    R = _current_rotation_matrix(data, qpos_adr)
                    v = _extract_body_velocity_mujoco(data, R, dof_adr)

                    v_dot, damping_force, g_eta = compute_acceleration_from_rotation_matrix(
                        params, vehicle, v, tau, R)
                    _inject_full_override_wrench(model, data, v_dot, R, body_id, dof_adr,
                                                 self_integrating_orientation=True)
                    mujoco.mj_step(model, data)  # actually integrates orientation now, not overwritten after

                R_final = _current_rotation_matrix(data, qpos_adr)
                nu[k + 1] = _extract_body_velocity_mujoco(data, R_final, dof_adr)

                if debug_info is not None:
                    debug_info["damping_force"][k] = damping_force
                    debug_info["restoring_force"][k] = g_eta
                    debug_info["acceleration"][k] = v_dot
    finally:
        model.opt.timestep = original_timestep  # never leak a mutated shared model's timestep

    if not np.all(np.isfinite(nu)):
        raise FloatingPointError(
            "MuJoCo-backed simulation produced NaN/inf -- check parameter "
            "bounds, or whether the injected wrench diverged."
        )

    return SimulationResult(nu=nu, used_teacher_forced_orientation=teacher_forced, debug_info=debug_info)


def make_mujoco_simulate_fn(model, data, body_id: int = _BODY_ID):
    """
    Binds a persistent MuJoCo model/data (and, per B3.2, a specific
    body_id) into objective.py's SimulateFn shape -- (params, vehicle,
    sim_config, trajectory) -> SimulationResult -- so it can be passed
    directly as simulate_fn to joint_residuals, mean_run_rmse,
    rmse_by_dof, etc., without any of those functions needing to know
    MuJoCo (or multi-body scenes) exist. Mirrors
    holoocean_backend.make_holoocean_simulate_fn(env, ...) exactly.

    `model`/`data` are created ONCE by the caller and reused across every
    call this returned function makes. For a multi-body scene, create
    ONE simulate_fn per body_id, sharing the same model/data -- see
    B3.2's own test script for exactly this pattern.

    Usage:
        import mujoco
        model = mujoco.MjModel.from_xml_path("b0_single_body_placeholder.xml")
        data = mujoco.MjData(model)
        simulate_fn = make_mujoco_simulate_fn(model, data)
        rmse = objective.mean_run_rmse(params, runs, vehicle, simulate_fn=simulate_fn)
    """
    def simulate_fn(params, vehicle, sim_config, trajectory):
        return simulate_trajectory_mujoco(params, vehicle, sim_config, trajectory, model, data,
                                            body_id=body_id)

    return simulate_fn


def make_mujoco_calibration_simulate_fn(model, data, body_id: int = _BODY_ID):
    """
    Same SimulateFn shape as make_mujoco_simulate_fn() above, but for
    actual parameter FITTING (i.e. passing as CalibrationConfig.simulate_fn
    into optimizer.calibrate()) specifically -- NOT for the held-out
    evaluate-only use case make_mujoco_simulate_fn() itself was built for
    (B2.1, B1.6's acceptance test), where forward-sim is the deliberate,
    correct choice.

    Exists because make_mujoco_simulate_fn()'s own default
    (teacher_forced=False, forward-sim) is UNSAFE for gradient-based
    fitting specifically: confirmed (see conversation) that forward-sim's
    finite-difference residual gradient becomes wildly unstable
    (>20x spread across step sizes) once a fit wanders near the
    near-gimbal-lock/near-inversion region this backend's own
    self-integrated orientation can reach -- while teacher_forced=True's
    gradient at that exact same point is numerically IDENTICAL to the
    stand-in's own (matching to 4 decimal places), since it never lets
    orientation self-integrate into that region at all. A person reaching
    for make_mujoco_simulate_fn() the obvious way when wiring up a real
    MuJoCo-backed calibration run would silently get the unsafe one --
    this function exists so the safe one is exactly as easy to reach for.

    Usage (identical to make_mujoco_simulate_fn(), only the fitting case):
        model = mujoco.MjModel.from_xml_path("b0_single_body_placeholder.xml")
        data = mujoco.MjData(model)
        simulate_fn = make_mujoco_calibration_simulate_fn(model, data)
        config = CalibrationConfig(simulate_fn=simulate_fn, ...)
        result = optimizer.calibrate(loaded, train_runs, config)
    """
    def simulate_fn(params, vehicle, sim_config, trajectory):
        return simulate_trajectory_mujoco(params, vehicle, sim_config, trajectory, model, data,
                                            teacher_forced=True, body_id=body_id)

    return simulate_fn


class MuJoCoArenaCore:
    """
    C0.3 addition (see conversation) -- a live, step-by-step MuJoCo-
    dynamics-backed replacement for arena_core.ArenaCore, mirroring its
    exact interface (state, reset(position=None, nu=None)->ArenaState,
    step(tau)->ArenaState, position_error_body_frame(target)) so
    TwoAgentOrchestrator can use either interchangeably with zero
    interface-level changes elsewhere.

    Extracts the EXACT substep integration logic already proven inside
    simulate_trajectory_mujoco's own forward-sim (teacher_forced=False)
    branch above -- read state from MuJoCo's own evolving qpos/qvel ->
    compute_acceleration_from_rotation_matrix -> inject the full-override
    wrench -> mj_step -- called ONE EXTERNAL STEP AT A TIME here
    (sim_config.n_substeps inner iterations per external step), rather
    than only across a pre-known batch trajectory. No new physics logic
    -- same functions, same order, same frame conventions already
    validated throughout B1/B2/B3; this class is a different CALLING
    PATTERN over that same logic, not a reimplementation of it.

    DELIBERATELY uses its OWN, INDEPENDENT single-body MuJoCo instance
    (b0_single_body_placeholder.xml by default), NOT C0.1's shared
    two-body scene -- B3.2/C0.2 already proved zero cross-talk between
    two bodies in a shared scene, so two fully independent instances are
    mathematically equivalent, while avoiding a real, confirmed hazard
    (see conversation): mj_step() advances an ENTIRE shared scene's
    clock at once, so sequentially running one agent's full substep loop
    before the other agent's wrench is even computed would silently
    double-step / stale-wrench that second agent, unless both agents'
    substeps were carefully interleaved before every shared mj_step()
    call (the pattern B3.2's own simulate_two_bodies_simultaneously
    function used, for a different purpose). Two independent instances
    sidestep that complexity entirely, at no physics cost given the
    already-proven zero coupling -- TwoAgentOrchestrator instantiates one
    per agent when using this backend.

    Forward-sim (teacher_forced=False in simulate_trajectory_mujoco's own
    terms), not teacher-forced: a live rollout has no real logged
    orientation to teacher-force FROM at all (per arena_core.py's own
    docstring on why ArenaCore -- and now this class -- must genuinely
    self-integrate), exactly like B2.1's own held-out evaluation already
    did via make_mujoco_simulate_fn. teacher_forced=True's own
    justification (sidestepping forward-sim's finite-difference GRADIENT
    instability near gimbal lock, for CALIBRATION fitting specifically)
    does not apply here: PPO never computes a gradient through the
    physics itself, only through its own policy network, treating the
    environment as a black-box reward source.
    """

    def __init__(self, params: CalibrationParameters, vehicle: VehicleConstants,
                 sim_config: SimulationConfig, mjcf_path: str = "b0_single_body_placeholder.xml"):
        params.validate()
        vehicle.validate()
        sim_config.validate()
        self.params = params
        self.vehicle = vehicle
        self.sim_config = sim_config
        self._model = mujoco.MjModel.from_xml_path(mjcf_path)
        self._data = mujoco.MjData(self._model)
        self._qpos_adr, self._dof_adr = _body_qpos_dof_addr(self._model, _BODY_ID)
        self.state = ArenaState.initial()
        # Water current in WORLD axes (m/s), or None for still water; the
        # calibrated forces act on the velocity relative to it
        # (dynamics_model._relative_to_water). Set before each step().
        self.current_world = None

    def reset(self, position: np.ndarray = None, nu: np.ndarray = None) -> ArenaState:
        """Same signature/defaults as ArenaCore.reset() exactly --
        orientation is always reset to identity here too, matching
        ArenaCore's own convention (see arena_core.py: 'orientation is
        irrelevant [to set at reset], reset always uses identity')."""
        position = np.zeros(3) if position is None else np.array(position, dtype=float)
        nu = np.zeros(6) if nu is None else np.array(nu, dtype=float)
        orientation = Rotation.identity()

        set_body_pose(self._model, self._data, _BODY_ID, position, orientation)
        self._data.qvel[self._dof_adr:self._dof_adr + 3] = orientation.apply(nu[:3])
        self._data.qvel[self._dof_adr + 3:self._dof_adr + 6] = nu[3:]

        self.state = ArenaState(nu=nu, position=position, orientation=orientation)
        return self.state

    def step(self, tau: np.ndarray) -> ArenaState:
        """Advances by one external dt (sim_config.n_substeps internal
        sub-steps, matching ArenaCore.step()'s own convention exactly).
        tau is body-frame [Fx,Fy,Fz,Tx,Ty,Tz], same as ArenaCore.step()."""
        tau = np.asarray(tau, dtype=float)
        original_timestep = self._model.opt.timestep
        self._model.opt.timestep = self.sim_config.sub_dt
        try:
            for _ in range(self.sim_config.n_substeps):
                R = _current_rotation_matrix(self._data, self._qpos_adr)
                v = _extract_body_velocity_mujoco(self._data, R, self._dof_adr)
                v_dot, _damping, _restoring = compute_acceleration_from_rotation_matrix(
                    self.params, self.vehicle, v, tau, R, current_world=self.current_world)
                _inject_full_override_wrench(self._model, self._data, v_dot, R, _BODY_ID, self._dof_adr,
                                             self_integrating_orientation=True)
                mujoco.mj_step(self._model, self._data)
        finally:
            self._model.opt.timestep = original_timestep

        # Explicit, defensive forward-kinematics refresh before reading
        # position back -- confirmed against MuJoCo's own documentation
        # that mj_step's own internal pipeline already includes this
        # (step 2 of its documented sequence), but given how much rests
        # on get_body_position() being correct here specifically (the
        # whole point of this class), this is called explicitly rather
        # than relying solely on an internal implementation detail this
        # session could not verify against real MuJoCo directly. Cheap:
        # pure kinematics, no collision/dynamics re-triggered.
        mujoco.mj_kinematics(self._model, self._data)

        R_final = _current_rotation_matrix(self._data, self._qpos_adr)
        v_final = _extract_body_velocity_mujoco(self._data, R_final, self._dof_adr)
        position_final = get_body_position(self._model, self._data, _BODY_ID)

        if not (np.all(np.isfinite(v_final)) and np.all(np.isfinite(position_final))):
            raise FloatingPointError(
                "MuJoCoArenaCore.step() produced NaN/inf -- check parameter bounds or "
                "whether this action/parameter combination made the simulation unstable."
            )

        self.state = ArenaState(nu=v_final, position=position_final,
                                  orientation=Rotation.from_matrix(R_final))
        return self.state

    def position_error_body_frame(self, target_position: np.ndarray) -> np.ndarray:
        """Identical formula to ArenaCore's own method of the same name --
        duplicated rather than imported to avoid a circular dependency
        (arena_core.py would need to import FROM mujoco_backend.py for
        this one method, while mujoco_backend.py otherwise has no
        dependency on arena_core.py at all) for a single, trivial,
        already-proven formula, not real logic worth that coupling."""
        world_error = np.asarray(target_position, dtype=float) - self.state.position
        return self.state.orientation.inv().apply(world_error)