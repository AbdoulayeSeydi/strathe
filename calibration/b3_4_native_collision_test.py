#!/usr/bin/env python3
"""
B3.4 -- Native collision. Confirms two things, not one: that MuJoCo's
own collision detection and contact-force response actually work in
this scene once enabled (the "for free" capability itself), AND that
enabling it costs nothing in the non-colliding case B3.2 already
confirmed correct (i.e. this genuinely optional capability doesn't
perturb the required, already-proven behavior).

WHY this is expected to work with zero changes to mujoco_backend.py's
own wrench-injection logic, confirmed directly against MuJoCo's own
documentation (see conversation), not assumed: qfrc_bias is documented
as ONLY "Coriolis, centrifugal and gravitational forces". Contact force
is a separate quantity entirely (qfrc_constraint / efc_force via the
constraint Jacobian), added independently in MuJoCo's own equation of
motion. _inject_full_override_wrench cancels and replaces ONLY
qfrc_bias -- it structurally cannot touch contact forces, by
construction, regardless of whether they're enabled.

Four checks:
  1. NO collision at the scene's default 2m separation (a sanity check
     on the test itself -- confirms contacts being "enabled" doesn't
     mean "always active").
  2. Collision IS detected (data.ncon > 0) once the two bodies are
     deliberately moved into overlap.
  3. A REAL, nonzero contact force appears (data.cfrc_ext), not just
     detection without physical consequence.
  4. Zero cross-talk STILL holds at the default separation with contacts
     enabled -- reuses B3.2's own comparison directly, against the B3.4
     MJCF instead of B3.1's, confirming this optional capability doesn't
     perturb the already-proven required behavior.

Checks 1-3 need REAL MuJoCo's actual collision geometry computation to
mean anything -- my own fake harness (used for local testing throughout
this session) has no model of contacts at all, so it cannot meaningfully
exercise or validate this part; these checks are UNVERIFIED here and
need a real run to confirm. Check 4 reuses the same block-diagonal
dynamics logic already validated in B3.2, so it can be (and was)
verified locally.

Usage:
    python3 b3_4_native_collision_test.py

*** Checks 1-3 UNVERIFIED against real MuJoCo in this sandbox --
collision detection cannot be meaningfully faked. Check 4 WAS verified
locally against the same fake harness B3.2/B3.3 used.
"""
import numpy as np
import mujoco

from dynamics_model import CalibrationParameters, VehicleConstants, SimulationConfig, TrajectoryInputs
from mujoco_backend import (
    _body_qpos_dof_addr, _set_teacher_forced_state, _inject_full_override_wrench,
    _extract_body_velocity_mujoco, simulate_trajectory_mujoco,
)

CONTACTS_ENABLED_MJCF = "b3_4_two_body_contacts_enabled.xml"
SINGLE_BODY_MJCF = "b0_single_body_placeholder.xml"
AGENT_TO_MUJOCO_BODY_ID = {0: 1, 1: 2}

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


def check_1_no_collision_at_default_separation():
    print("=" * 70)
    print("CHECK 1: no collision at the scene's default 2m separation")
    print("=" * 70)
    model = mujoco.MjModel.from_xml_path(CONTACTS_ENABLED_MJCF)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    print(f"data.ncon at default separation: {data.ncon}")
    assert data.ncon == 0, "unexpected contact at the default, well-separated positions"
    print("PASS -- no contact detected, as expected at 2m separation.\n")


def check_2_and_3_collision_detected_and_produces_force():
    print("=" * 70)
    print("CHECK 2 & 3: collision detected AND produces a real, nonzero force when overlapping")
    print("=" * 70)
    model = mujoco.MjModel.from_xml_path(CONTACTS_ENABLED_MJCF)
    data = mujoco.MjData(model)

    qpos_adr_0, dof_adr_0 = _body_qpos_dof_addr(model, AGENT_TO_MUJOCO_BODY_ID[0])
    qpos_adr_1, dof_adr_1 = _body_qpos_dof_addr(model, AGENT_TO_MUJOCO_BODY_ID[1])

    # Move agent 1 directly on top of agent 0 -- their boxes (each
    # 0.3 x 0.15 x 0.15) fully overlap, well beyond any margin.
    data.qpos[qpos_adr_1:qpos_adr_1 + 3] = data.qpos[qpos_adr_0:qpos_adr_0 + 3]
    mujoco.mj_forward(model, data)

    print(f"data.ncon after moving agent 1 onto agent 0: {data.ncon}")
    assert data.ncon > 0, "no contact detected despite fully overlapping geoms -- collision detection not working"
    print("PASS -- collision detected.")

    # NOTE (see conversation): cfrc_ext requires an EXPLICIT
    # mj_rnePostConstraint() call after mj_forward() -- MuJoCo 2.0+ no
    # longer calls it automatically unless the model has force-related
    # sensors (this one doesn't), so cfrc_ext silently stays at its
    # stale zero value even with real, detected contacts. Using
    # qfrc_constraint instead: populated directly by mj_forward() itself
    # as part of the core constraint-solver output, no extra call needed.
    qfrc_constraint_0 = data.qfrc_constraint[dof_adr_0:dof_adr_0 + 6].copy()
    qfrc_constraint_1 = data.qfrc_constraint[dof_adr_1:dof_adr_1 + 6].copy()
    print(f"qfrc_constraint on agent 0: {qfrc_constraint_0}")
    print(f"qfrc_constraint on agent 1: {qfrc_constraint_1}")
    max_force = max(np.max(np.abs(qfrc_constraint_0)), np.max(np.abs(qfrc_constraint_1)))
    assert max_force > 1e-6, "contact detected but no real force resulted -- detection without physical response"
    print(f"PASS -- real, nonzero contact force present (max |qfrc_constraint| = {max_force:.3e}).\n")


def _run_two_body_agent0_only(model, data, params0, trajectory0, qpos_adr0, dof_adr0, qpos_adr1, dof_adr1):
    from dynamics_model import compute_acceleration
    n_steps = trajectory0.forces_torques.shape[0]
    nu1 = np.zeros((n_steps, 6))
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
                mujoco.mj_step(model, data)
                v0 = _extract_body_velocity_mujoco(data, R0, dof_adr0)
    finally:
        model.opt.timestep = original_timestep
    return v0


def make_synthetic_trajectory(seed, n_steps=20):
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


def check_4_zero_cross_talk_still_holds_with_contacts_enabled():
    print("=" * 70)
    print("CHECK 4: B3.2's zero cross-talk result still holds with contacts enabled")
    print("=" * 70)
    trajectory0 = make_synthetic_trajectory(seed=1)

    print("Running STANDALONE single-body simulation (agent 0's own parameters)...")
    model_standalone = mujoco.MjModel.from_xml_path(SINGLE_BODY_MJCF)
    data_standalone = mujoco.MjData(model_standalone)
    result_standalone = simulate_trajectory_mujoco(
        GENERIC_PARAMS, VEHICLE, SIM_CONFIG, trajectory0, model_standalone, data_standalone, teacher_forced=True)

    print("Running SHARED-SCENE (contacts ENABLED), agent 0 actuated, agent 1 untouched, "
          "at the default 2m separation (never overlapping)...")
    model_shared = mujoco.MjModel.from_xml_path(CONTACTS_ENABLED_MJCF)
    data_shared = mujoco.MjData(model_shared)
    qpos_adr_0, dof_adr_0 = _body_qpos_dof_addr(model_shared, AGENT_TO_MUJOCO_BODY_ID[0])
    qpos_adr_1, dof_adr_1 = _body_qpos_dof_addr(model_shared, AGENT_TO_MUJOCO_BODY_ID[1])
    final_v0_shared = _run_two_body_agent0_only(
        model_shared, data_shared, GENERIC_PARAMS, trajectory0, qpos_adr_0, dof_adr_0, qpos_adr_1, dof_adr_1)

    diff = np.max(np.abs(final_v0_shared - result_standalone.nu[-1]))
    print(f"agent 0 final velocity, max |shared(contacts enabled) - standalone|: {diff:.3e}")
    if diff < 1e-8:
        print("PASS -- enabling contacts costs nothing when bodies aren't actually touching.\n")
    else:
        print("FAIL -- enabling contacts changed behavior even without an actual collision.\n")
    return diff


def main():
    check_1_no_collision_at_default_separation()
    check_2_and_3_collision_detected_and_produces_force()
    diff = check_4_zero_cross_talk_still_holds_with_contacts_enabled()

    print("=" * 70)
    print("B3.4 RESULT")
    print("=" * 70)
    if diff < 1e-8:
        print("PASS -- native collision works, and costs nothing when not triggered.")
    else:
        print("FAIL -- see CHECK 4 output above.")


if __name__ == "__main__":
    main()