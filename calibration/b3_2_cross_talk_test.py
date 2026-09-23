#!/usr/bin/env python3
"""
B3.2 -- Zero-cross-talk confirmation. Actuates BOTH bodies in the B3.1
two-body scene SIMULTANEOUSLY (one generic, one differently-parameterized
"calibrated" set -- deliberately different, not identical, so any real
cross-talk has the best chance of producing a visible, divergent effect
rather than being masked by both bodies behaving identically anyway),
and compares each body's resulting trajectory against a true standalone
single-body simulation given the identical forces. They should match to
machine precision -- any divergence means the shared scene is
introducing unintended coupling.

simulate_trajectory_mujoco() only drives ONE body per call (by design --
see mujoco_backend.py), so true SIMULTANEOUS actuation needs its own
loop here, built directly from the same body-id-aware helper functions
that function itself uses internally (_body_qpos_dof_addr,
_set_teacher_forced_state, _inject_full_override_wrench,
_extract_body_velocity_mujoco) -- not a reimplementation of their logic,
the SAME functions, called for both bodies before each shared mj_step().

Uses teacher_forced-style stepping specifically (real orientation fed in
every substep, only velocity integrated) -- the confirmed-reliable mode
this session established for calibration use, and the right choice here
too: this test is about scene-level coupling, not about forward-sim's
own separately-understood orientation-drift behavior, and teacher_forced
mode avoids that confound entirely.

Two independent checks, not just one:
  1. Trajectory comparison: shared-scene body 1/2 vs standalone body 1/2,
     given identical forces and parameters -- the direct, top-level
     question B3.2 asks.
  2. Mass-matrix block-diagonal check: confirms the OFF-diagonal blocks
     of the full 12x12 mass matrix are exactly zero at several sampled
     states -- direct physics-level evidence for WHY zero cross-talk is
     expected (no shared joint, no contact), not just trajectory-level
     agreement that could in principle mask a real-but-small coupling.

Uses synthetic (not real CSV) trajectories deliberately -- this is a
scene-wiring/indexing question, not a real-data validation question, so
a fixed-seed synthetic trajectory is sufficient and avoids depending on
data_split.py/objective.py's own real-data loading for a test that isn't
actually about real data at all.

Usage:
    python3 b3_2_cross_talk_test.py

*** UNVERIFIED against a real, working MuJoCo/dynamics_model.py in this
sandbox (sandbox reset earlier this session) -- built directly from
mujoco_backend.py's own real, confirmed structure and the same
CalibrationParameters/VehicleConstants/SimulationConfig/TrajectoryInputs
construction patterns used successfully throughout this session, but not
independently end-to-end tested the way most earlier handoffs were.
Worth a moment's extra scrutiny on first run.
"""
import numpy as np
import mujoco

from dynamics_model import (
    CalibrationParameters, VehicleConstants, SimulationConfig, TrajectoryInputs,
)
from mujoco_backend import (
    _body_qpos_dof_addr, _set_teacher_forced_state, _inject_full_override_wrench,
    _extract_body_velocity_mujoco, simulate_trajectory_mujoco,
)

TWO_BODY_MJCF = "b3_two_body_placeholder.xml"
SINGLE_BODY_MJCF = "b0_single_body_placeholder.xml"

# Deliberately DIFFERENT, not identical -- see module docstring for why.
GENERIC_PARAMS = CalibrationParameters(
    Ieff_roll=0.26, Ieff_pitch=0.23, Ieff_yaw=0.32,
    Xu=-11.7, Yv=-20.0, Zw=-31.8, Kp=-0.3, Mq=-0.3, Nr=-1.0,
    Xuu=-18.2, Yvv=-21.7, Zww=-50.0, Kpp=-0.5, Mqq=-0.5, Nrr=-2.0,
    Xu_dot=5.5, Yv_dot=12.7, Zw_dot=14.6, net_buoyancy_z=0.0,
    cg_cb_offset_z=0.02, Xq_dot=0.0, Zq_dot=0.0,
)
CALIBRATED_PARAMS = CalibrationParameters(
    Ieff_roll=0.265, Ieff_pitch=0.225, Ieff_yaw=2.43,
    Xu=-37.4, Yv=-80.0, Zw=-63.1, Kp=-10.0, Mq=-5.0, Nr=-5.26,
    Xuu=-51.7, Yvv=-101.9, Zww=-141.99, Kpp=-10.0, Mqq=-10.0, Nrr=-0.65,
    Xu_dot=30.0, Yv_dot=32.5, Zw_dot=21.5, net_buoyancy_z=1.0,
    cg_cb_offset_z=0.023, Xq_dot=-0.5, Zq_dot=-0.5,
)
VEHICLE = VehicleConstants(mass_kg=11.2)
SIM_CONFIG = SimulationConfig(dt=0.1, physics_dt=0.005)


def make_synthetic_trajectory(seed: int, n_steps: int = 30) -> TrajectoryInputs:
    rng = np.random.default_rng(seed)
    nu0 = rng.uniform(-0.2, 0.2, 6)
    forces_torques = np.column_stack([
        rng.uniform(-3, 3, n_steps), rng.uniform(-3, 3, n_steps), rng.uniform(-3, 3, n_steps),
        rng.uniform(-1, 1, n_steps), rng.uniform(-1, 1, n_steps), rng.uniform(-1, 1, n_steps),
    ])
    roll_pitch_series = np.column_stack([
        rng.uniform(-0.3, 0.3, n_steps), rng.uniform(-0.3, 0.3, n_steps),
    ])
    return TrajectoryInputs(nu0=nu0, forces_torques=forces_torques, roll_pitch_series=roll_pitch_series)


def simulate_two_bodies_simultaneously(model, data, params1, params2, trajectory1, trajectory2):
    """
    Drives BOTH bodies at once, teacher_forced-style: each substep,
    computes both bodies' own desired accelerations independently
    (compute_acceleration, imported lazily to avoid a hard dependency at
    module load time if only the mass-matrix check is needed), injects
    BOTH wrenches, then advances BOTH with a single shared mj_step() --
    the genuinely simultaneous scenario B3.2's own wording describes,
    not two sequential single-body calls.
    """
    from dynamics_model import compute_acceleration

    qpos_adr1, dof_adr1 = _body_qpos_dof_addr(model, 1)
    qpos_adr2, dof_adr2 = _body_qpos_dof_addr(model, 2)

    n_steps = trajectory1.forces_torques.shape[0]
    assert n_steps == trajectory2.forces_torques.shape[0], "both trajectories must have equal length for this test"

    nu1 = np.zeros((n_steps, 6)); nu1[0] = trajectory1.nu0
    nu2 = np.zeros((n_steps, 6)); nu2[0] = trajectory2.nu0

    n_substeps = SIM_CONFIG.n_substeps
    sub_dt = SIM_CONFIG.sub_dt
    original_timestep = model.opt.timestep
    model.opt.timestep = sub_dt

    try:
        for k in range(n_steps - 1):
            v1, v2 = nu1[k], nu2[k]
            tau1, tau2 = trajectory1.forces_torques[k], trajectory2.forces_torques[k]
            roll1, pitch1 = trajectory1.roll_pitch_series[k]
            roll2, pitch2 = trajectory2.roll_pitch_series[k]

            for _ in range(n_substeps):
                v_dot1, _, _ = compute_acceleration(params1, VEHICLE, v1, tau1, roll1, pitch1)
                v_dot2, _, _ = compute_acceleration(params2, VEHICLE, v2, tau2, roll2, pitch2)

                R1 = _set_teacher_forced_state(data, roll1, pitch1, v1, qpos_adr1, dof_adr1)
                R2 = _set_teacher_forced_state(data, roll2, pitch2, v2, qpos_adr2, dof_adr2)

                _inject_full_override_wrench(model, data, v_dot1, R1, body_id=1, dof_adr=dof_adr1)
                _inject_full_override_wrench(model, data, v_dot2, R2, body_id=2, dof_adr=dof_adr2)

                mujoco.mj_step(model, data)  # ONE shared step advances BOTH bodies together

                v1 = _extract_body_velocity_mujoco(data, R1, dof_adr1)
                v2 = _extract_body_velocity_mujoco(data, R2, dof_adr2)

            nu1[k + 1] = v1
            nu2[k + 1] = v2
    finally:
        model.opt.timestep = original_timestep

    return nu1, nu2


def check_mass_matrix_block_diagonal(model, data, n_samples: int = 5, seed: int = 0):
    """Directly verifies the OFF-diagonal blocks of the full mass matrix
    are exactly zero at several random states -- physics-level evidence
    for WHY zero cross-talk is expected, independent of the trajectory
    comparison."""
    rng = np.random.default_rng(seed)
    max_off_diag = 0.0
    for _ in range(n_samples):
        data.qpos[:] = 0.0
        data.qpos[3] = 1.0   # body 1 quat w=1 (identity)
        data.qpos[10] = 1.0  # body 2 quat w=1 (identity) -- qpos[7:14], w at offset 7+3=10
        data.qpos[7:10] = 2.0  # body 2's own xyz offset, matching its MJCF pos="2 0 0"
        data.qvel[:] = rng.uniform(-1, 1, model.nv)
        mujoco.mj_forward(model, data)
        M = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, data, M)
        off_diag_block = M[0:6, 6:12]  # body1-vs-body2 coupling terms
        max_off_diag = max(max_off_diag, np.max(np.abs(off_diag_block)))
    return max_off_diag


def main():
    print("=" * 70)
    print("PART 1: MASS-MATRIX BLOCK-DIAGONAL CHECK")
    print("=" * 70)
    model_shared = mujoco.MjModel.from_xml_path(TWO_BODY_MJCF)
    data_shared = mujoco.MjData(model_shared)
    max_off_diag = check_mass_matrix_block_diagonal(model_shared, data_shared)
    print(f"Max |off-diagonal block value| across 5 random states: {max_off_diag:.3e}")
    if max_off_diag < 1e-10:
        print("PASS -- mass matrix is exactly block-diagonal, confirming zero mechanical coupling.")
    else:
        print("FAIL -- nonzero coupling found in the mass matrix itself. Investigate before "
              "trusting any trajectory-level comparison below.")

    print("\n" + "=" * 70)
    print("PART 2: TRAJECTORY COMPARISON (standalone vs. shared-scene, simultaneous actuation)")
    print("=" * 70)
    trajectory1 = make_synthetic_trajectory(seed=1)
    trajectory2 = make_synthetic_trajectory(seed=2)

    print("Running STANDALONE single-body simulations...")
    model_s1 = mujoco.MjModel.from_xml_path(SINGLE_BODY_MJCF)
    data_s1 = mujoco.MjData(model_s1)
    result_standalone_1 = simulate_trajectory_mujoco(
        GENERIC_PARAMS, VEHICLE, SIM_CONFIG, trajectory1, model_s1, data_s1, teacher_forced=True)

    model_s2 = mujoco.MjModel.from_xml_path(SINGLE_BODY_MJCF)
    data_s2 = mujoco.MjData(model_s2)
    result_standalone_2 = simulate_trajectory_mujoco(
        CALIBRATED_PARAMS, VEHICLE, SIM_CONFIG, trajectory2, model_s2, data_s2, teacher_forced=True)

    print("Running SHARED-SCENE simulation, both bodies actuated SIMULTANEOUSLY...")
    model_shared2 = mujoco.MjModel.from_xml_path(TWO_BODY_MJCF)
    data_shared2 = mujoco.MjData(model_shared2)
    nu1_shared, nu2_shared = simulate_two_bodies_simultaneously(
        model_shared2, data_shared2, GENERIC_PARAMS, CALIBRATED_PARAMS, trajectory1, trajectory2)

    diff1 = np.max(np.abs(nu1_shared - result_standalone_1.nu))
    diff2 = np.max(np.abs(nu2_shared - result_standalone_2.nu))
    print(f"\nBody 1 (generic params): max |shared - standalone| = {diff1:.3e}")
    print(f"Body 2 (calibrated params): max |shared - standalone| = {diff2:.3e}")

    print("\n" + "=" * 70)
    print("B3.2 RESULT")
    print("=" * 70)
    if diff1 < 1e-8 and diff2 < 1e-8:
        print("PASS -- both bodies match their standalone equivalents to machine precision, "
              "even when actuated simultaneously with different parameter sets. No cross-talk.")
    else:
        print("FAIL -- at least one body's shared-scene trajectory diverges from its standalone "
              "equivalent. The shared scene is introducing unintended coupling -- investigate "
              "the body-id-aware indexing in mujoco_backend.py before proceeding to B3.3.")


if __name__ == "__main__":
    main()