#!/usr/bin/env python3
"""
B3.3 -- Position extraction and indexing check. Confirms each body's
position is correctly and STABLY indexed in the scene's shared qpos
array.

NAMING NOTE (see mujoco_backend.py's own docstring for this): this
task's own text uses 0-indexed "body 0"/"body 1" for the two AGENTS --
NOT the same as MuJoCo's own 1-indexed body_id, where index 0 is always
the implicit world body. Conflating the two would itself be exactly the
kind of indexing bug this task exists to catch, so the mapping is made
explicit throughout: agent 0 = MuJoCo body_id 1 = auv_placeholder,
agent 1 = MuJoCo body_id 2 = auv_placeholder_2.

Three checks, not one, since "correctly AND stably indexed" is really
two separate claims:

  1. DIRECT WRITE ISOLATION (the task's own literal wording: "moves only
     body 0... confirms body 1's recorded position is completely
     unaffected"): writes directly to agent 0's qpos slice, confirms
     agent 1's slice is BIT-FOR-BIT unchanged -- pure indexing, no
     dynamics involved at all, isolating the indexing claim itself from
     anything about the physics.

  2. STABILITY ACROSS CALLS: _body_qpos_dof_addr() returns model-level
     (not data-level) addresses -- they must never change across
     multiple calls or after stepping the simulation. Confirmed
     explicitly rather than assumed, since "stably indexed" is part of
     this task's own literal wording, not just "correctly" once.

  3. DYNAMICS-DRIVEN CONFIRMATION: steps the shared scene forward with
     ONLY agent 0 actuated (a real force, via the same wrench-injection
     machinery B3.2 already confirmed correct), and confirms agent 1's
     recorded position exactly matches a standalone single-body
     simulation of agent 1 alone under identical (zero-force) conditions
     -- this overlaps somewhat with B3.2's own velocity-level
     confirmation (position is just the integral of velocity, already
     shown independent), but is included as a direct, position-specific
     check rather than an assumption that B3.2's velocity result
     automatically extends to the newly-added position-extraction code
     path, which did not exist at B3.2's time.

Usage:
    python3 b3_3_position_indexing_test.py

*** UNVERIFIED against a real, working MuJoCo/dynamics_model.py in this
sandbox (sandbox reset earlier this session) -- built directly from
mujoco_backend.py's own real, confirmed structure, same as b3_2's own
test script, but not independently end-to-end tested against real
MuJoCo. Worth a moment's extra scrutiny on first run.
"""
import numpy as np
import mujoco

from dynamics_model import CalibrationParameters, VehicleConstants, SimulationConfig, TrajectoryInputs
from mujoco_backend import (
    _body_qpos_dof_addr, _set_teacher_forced_state, _inject_full_override_wrench,
    _extract_body_velocity_mujoco, _extract_body_position_mujoco, simulate_trajectory_mujoco,
)

TWO_BODY_MJCF = "b3_two_body_placeholder.xml"
SINGLE_BODY_MJCF = "b0_single_body_placeholder.xml"

# Explicit agent-to-MuJoCo-body-id mapping -- see module docstring.
AGENT_TO_MUJOCO_BODY_ID = {0: 1, 1: 2}

GENERIC_PARAMS = CalibrationParameters(
    Ieff_roll=0.26, Ieff_pitch=0.23, Ieff_yaw=0.32,
    Xu=-11.7, Yv=-20.0, Zw=-31.8, Kp=-0.3, Mq=-0.3, Nr=-1.0,
    Xuu=-18.2, Yvv=-21.7, Zww=-50.0, Kpp=-0.5, Mqq=-0.5, Nrr=-2.0,
    Xu_dot=5.5, Yv_dot=12.7, Zw_dot=14.6, net_buoyancy_z=0.0,
    cg_cb_offset_z=0.02, Xq_dot=0.0, Zq_dot=0.0,
)
VEHICLE = VehicleConstants(mass_kg=11.2)
SIM_CONFIG = SimulationConfig(dt=0.1, physics_dt=0.005)


def check_1_direct_write_isolation():
    print("=" * 70)
    print("CHECK 1: direct write isolation (this task's own literal wording)")
    print("=" * 70)
    model = mujoco.MjModel.from_xml_path(TWO_BODY_MJCF)
    data = mujoco.MjData(model)

    qpos_adr_agent0, _ = _body_qpos_dof_addr(model, AGENT_TO_MUJOCO_BODY_ID[0])
    qpos_adr_agent1, _ = _body_qpos_dof_addr(model, AGENT_TO_MUJOCO_BODY_ID[1])
    print(f"agent 0 (MuJoCo body_id {AGENT_TO_MUJOCO_BODY_ID[0]}) qpos_adr = {qpos_adr_agent0}")
    print(f"agent 1 (MuJoCo body_id {AGENT_TO_MUJOCO_BODY_ID[1]}) qpos_adr = {qpos_adr_agent1}")
    assert qpos_adr_agent0 != qpos_adr_agent1, "agents must NOT share the same qpos address"

    # Record agent 1's position BEFORE touching agent 0 at all.
    position_agent1_before = _extract_body_position_mujoco(data, qpos_adr_agent1).copy()

    # "Move" agent 0 -- a direct write, no dynamics involved, isolating
    # the indexing claim itself.
    data.qpos[qpos_adr_agent0:qpos_adr_agent0 + 3] = np.array([99.0, -42.0, 7.5])
    position_agent0_after = _extract_body_position_mujoco(data, qpos_adr_agent0)
    position_agent1_after = _extract_body_position_mujoco(data, qpos_adr_agent1)

    print(f"agent 0 position after write: {position_agent0_after}")
    print(f"agent 1 position before: {position_agent1_before}, after: {position_agent1_after}")

    assert np.array_equal(position_agent0_after, np.array([99.0, -42.0, 7.5])), \
        "agent 0's own write did not take effect at its own address"
    assert np.array_equal(position_agent1_before, position_agent1_after), \
        "agent 1's position CHANGED after only agent 0 was written -- indexing bug"
    print("PASS -- agent 1's position is bit-for-bit unaffected by agent 0's write.\n")


def check_2_address_stability():
    print("=" * 70)
    print("CHECK 2: address stability across multiple calls and after stepping")
    print("=" * 70)
    model = mujoco.MjModel.from_xml_path(TWO_BODY_MJCF)
    data = mujoco.MjData(model)

    addr_0_first = _body_qpos_dof_addr(model, AGENT_TO_MUJOCO_BODY_ID[0])
    addr_1_first = _body_qpos_dof_addr(model, AGENT_TO_MUJOCO_BODY_ID[1])

    # Re-query several times, including after mutating data (not model) --
    # these are MODEL-level addresses and must never depend on data state.
    for _ in range(5):
        data.qvel[:] = np.random.default_rng(0).uniform(-1, 1, model.nv)
        mujoco.mj_forward(model, data)
        addr_0_again = _body_qpos_dof_addr(model, AGENT_TO_MUJOCO_BODY_ID[0])
        addr_1_again = _body_qpos_dof_addr(model, AGENT_TO_MUJOCO_BODY_ID[1])
        assert addr_0_again == addr_0_first, f"agent 0's address changed: {addr_0_first} -> {addr_0_again}"
        assert addr_1_again == addr_1_first, f"agent 1's address changed: {addr_1_first} -> {addr_1_again}"

    print(f"agent 0 address stable at {addr_0_first} across 5 re-queries and mj_forward calls.")
    print(f"agent 1 address stable at {addr_1_first} across 5 re-queries and mj_forward calls.")
    print("PASS -- addresses are stable, as expected for model-level (not data-level) quantities.\n")


def check_3_dynamics_driven_confirmation():
    print("=" * 70)
    print("CHECK 3: dynamics-driven confirmation (position-specific, not assumed from B3.2)")
    print("=" * 70)
    trajectory_agent0 = TrajectoryInputs(
        nu0=np.array([0.3, 0.1, -0.2, 0.05, 0.02, 0.0]),
        forces_torques=np.random.default_rng(3).uniform(-3, 3, size=(20, 6)),
        roll_pitch_series=np.tile([0.1, -0.1], (20, 1)),
    )
    n_steps = trajectory_agent0.forces_torques.shape[0]
    n_substeps = SIM_CONFIG.n_substeps

    print("Running STANDALONE single-body simulation for agent 1 (untouched, never actuated)...")
    model_standalone = mujoco.MjModel.from_xml_path(SINGLE_BODY_MJCF)
    data_standalone = mujoco.MjData(model_standalone)
    qpos_adr_standalone, dof_adr_standalone = _body_qpos_dof_addr(model_standalone, 1)
    initial_position_standalone = _extract_body_position_mujoco(data_standalone, qpos_adr_standalone).copy()
    position_trace_standalone = _run_untouched_tracking_position(
        model_standalone, data_standalone, qpos_adr_standalone, dof_adr_standalone, n_steps, n_substeps)
    displacement_standalone = position_trace_standalone - initial_position_standalone

    print("Running SHARED-SCENE simulation, agent 0 actuated, agent 1 left untouched...")
    model_shared = mujoco.MjModel.from_xml_path(TWO_BODY_MJCF)
    data_shared = mujoco.MjData(model_shared)
    qpos_adr_0, dof_adr_0 = _body_qpos_dof_addr(model_shared, AGENT_TO_MUJOCO_BODY_ID[0])
    qpos_adr_1, dof_adr_1 = _body_qpos_dof_addr(model_shared, AGENT_TO_MUJOCO_BODY_ID[1])
    initial_position_agent1 = _extract_body_position_mujoco(data_shared, qpos_adr_1).copy()
    print(f"agent 1's own initial position in the shared scene: {initial_position_agent1} "
          f"(its own MJCF pos, e.g. 2 0 0 -- NOT necessarily the origin; real MuJoCo initializes "
          f"each free body's qpos to its own MJCF-declared position, confirmed directly)")

    position_trace_shared_agent1 = _run_two_body_agent0_only(
        model_shared, data_shared, GENERIC_PARAMS, trajectory_agent0,
        qpos_adr_0, dof_adr_0, qpos_adr_1, dof_adr_1)
    displacement_shared_agent1 = position_trace_shared_agent1 - initial_position_agent1

    # Compare DISPLACEMENT from each body's own initial position, not
    # absolute position -- see this function's own comment above and the
    # conversation for why absolute comparison is not apples-to-apples
    # when the two scenes place their bodies at different starting points.
    diff = np.max(np.abs(displacement_shared_agent1 - displacement_standalone))
    print(f"agent 1 DISPLACEMENT (not absolute position), max |shared - standalone|: {diff:.3e}")

    if diff < 1e-8:
        print("PASS -- agent 1's position matches its standalone equivalent exactly, "
              "even while agent 0 is actively actuated in the same shared scene.\n")
    else:
        print("FAIL -- agent 1's position diverges from its standalone equivalent. "
              "Investigate position indexing before proceeding.\n")
    return diff


def _run_untouched_tracking_position(model, data, qpos_adr, dof_adr, n_steps, n_substeps):
    """Tracks position for a body that is NEVER teacher-forced or given
    an injected wrench -- matching exactly what happens to a passive
    agent in the shared scene (its xfrc_applied stays zero throughout,
    but mj_forward still computes its own qfrc_bias, and mj_step still
    advances it under that bias -- a real, passive body under its own
    gravity/Coriolis, not an inert one). Fixed after a real bug found in
    this test's own first draft (see conversation): comparing this
    against a body that HAD been run through _set_teacher_forced_state
    (which resets qpos[:3] to 0 every single substep, by design, since
    position was never meaningful in teacher-forced mode before B3.3)
    was comparing two different methodologies, not a fair standalone
    equivalent."""
    positions = np.zeros((n_steps, 3))
    positions[0] = _extract_body_position_mujoco(data, qpos_adr)  # index 0 must be the TRUE initial
                                                                    # position, not left at the np.zeros
                                                                    # placeholder -- a real bug found and
                                                                    # fixed here (see conversation): this
                                                                    # was invisible for a body starting at
                                                                    # the origin, but not for one that
                                                                    # doesn't (e.g. agent 1 at pos="2 0 0")
    original_timestep = model.opt.timestep
    model.opt.timestep = SIM_CONFIG.sub_dt
    try:
        for k in range(n_steps - 1):
            for _ in range(n_substeps):
                mujoco.mj_forward(model, data)  # computes qfrc_bias -- matches what agent 0's
                                                  # own wrench injection triggers scene-wide
                mujoco.mj_step(model, data)
            positions[k + 1] = _extract_body_position_mujoco(data, qpos_adr)
    finally:
        model.opt.timestep = original_timestep
    return positions


def _run_two_body_agent0_only(model, data, params0, trajectory0, qpos_adr0, dof_adr0, qpos_adr1, dof_adr1):
    """Actuates ONLY agent 0; agent 1 is left completely untouched (its
    xfrc_applied stays at zero throughout -- it still moves under its own
    gravity/bias via MuJoCo's own stepping, exactly like a real passive
    body would). Returns agent 1's own position trace for comparison."""
    from dynamics_model import compute_acceleration
    n_steps = trajectory0.forces_torques.shape[0]
    positions_agent1 = np.zeros((n_steps, 3))
    positions_agent1[0] = _extract_body_position_mujoco(data, qpos_adr1)  # same fix as
                                                                            # _run_untouched_tracking_position
                                                                            # above -- index 0 must be the
                                                                            # TRUE initial position
    v0 = trajectory0.nu0.copy()
    n_substeps = SIM_CONFIG.n_substeps
    original_timestep = model.opt.timestep
    model.opt.timestep = SIM_CONFIG.sub_dt
    try:
        for k in range(n_steps - 1):
            tau0 = trajectory0.forces_torques[k]
            roll0, pitch0 = trajectory0.roll_pitch_series[k]
            for _ in range(n_substeps):
                v_dot0, _, _ = compute_acceleration(params0, VEHICLE, v0, tau0, roll0, pitch0)
                R0 = _set_teacher_forced_state(data, roll0, pitch0, v0, qpos_adr0, dof_adr0)
                _inject_full_override_wrench(model, data, v_dot0, R0, body_id=AGENT_TO_MUJOCO_BODY_ID[0], dof_adr=dof_adr0)
                # Agent 1: NOT touched at all -- xfrc_applied[body_id=2] stays zero.
                mujoco.mj_step(model, data)
                v0 = _extract_body_velocity_mujoco(data, R0, dof_adr0)
            positions_agent1[k + 1] = _extract_body_position_mujoco(data, qpos_adr1)
    finally:
        model.opt.timestep = original_timestep
    return positions_agent1


def main():
    check_1_direct_write_isolation()
    check_2_address_stability()
    diff = check_3_dynamics_driven_confirmation()

    print("=" * 70)
    print("B3.3 RESULT")
    print("=" * 70)
    if diff < 1e-8:
        print("PASS -- position is correctly and stably indexed for both agents.")
    else:
        print("FAIL -- see CHECK 3 output above.")


if __name__ == "__main__":
    main()