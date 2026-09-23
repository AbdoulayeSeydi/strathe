"""
Instruments the REAL, UNMODIFIED mujoco_backend.simulate_trajectory_mujoco
execution at a specific target step, by wrapping its actual
_inject_full_override_wrench function with a spy that prints internal
values (qfrc_bias, M_mj, the injected wrench, realized vs. expected
qacc) -- rather than reimplementing the simulation loop by hand, which
would risk subtly diverging from the real execution path and inspecting
the wrong thing.

Purpose: at run4's identified peak-divergence step (1261), check
whether MuJoCo's own qfrc_bias or the resulting injected wrench become
anomalously large -- e.g. from a real gyroscopic/Coriolis effect given
the vehicle's large built-up pitch rate interacting with the tiny
placeholder rotational inertia -- which the fake-physics harness used
for mujoco_backend.py's own unit tests cannot replicate (it never
computes a real gyroscopic term at all, unlike actual MuJoCo).

Already ruled out via the fake harness (see conversation): the formula
breaking at run4's force scale, and the formula breaking under
sustained angular velocity buildup matching run4's apparent history.
Both held to machine precision. This script checks what's left: real
MuJoCo's own internal computation at the actual problematic moment.
"""
import argparse

import numpy as np
import mujoco

from objective import load_run, DatasetConfig
from build_two_agent_cores import load_calibrated_parameters, VEHICLE_SPEC_PATH
from optimizer import load_vehicle_spec
from dynamics_model import SimulationConfig, TrajectoryInputs
import mujoco_backend

MJCF_PATH = "b0_single_body_placeholder.xml"


def inspect_step(filepath: str, sampling_rate_hz: float, target_step: int, physics_dt: float,
                  calibrated_params_path: str = "calibrated_parameters_a1.yaml"):
    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    params = load_calibrated_parameters(calibrated_params_path)
    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)
    run = load_run(filepath, dataset_config)

    n_substeps_expected = max(1, int(np.ceil(run.dt / physics_dt)))
    target_substep_start = target_step * n_substeps_expected
    target_substep_end = target_substep_start + n_substeps_expected

    call_counter = {"n": 0}
    original_inject = mujoco_backend._inject_full_override_wrench
    original_mj_step = mujoco.mj_step
    pending = {}

    def spy_inject(model, data, desired_v_dot, R):
        i = call_counter["n"]
        call_counter["n"] += 1
        if target_substep_start <= i < target_substep_end:
            mujoco.mj_forward(model, data)
            qfrc_bias = data.qfrc_bias.copy()
            M_mj = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, data, M_mj)
            desired_qacc = np.concatenate([R @ desired_v_dot[:3], desired_v_dot[3:]])
            injected_wrench = M_mj @ desired_qacc + qfrc_bias

            print(f"\n--- substep call #{i} (outer step {i // n_substeps_expected}, "
                  f"substep {i % n_substeps_expected}) ---")
            print(f"  desired_v_dot (body-frame):   {desired_v_dot}")
            print(f"  qfrc_bias (MuJoCo's own):     {qfrc_bias}")
            print(f"  max abs qfrc_bias:            {np.max(np.abs(qfrc_bias)):.4e}")
            print(f"  M_mj FULL matrix:")
            for row in M_mj:
                print(f"    {row}")
            off_diag_linear_angular = M_mj[:3, 3:]
            off_diag_angular_linear = M_mj[3:, :3]
            print(f"  M_mj linear-angular coupling block [:3,3:]: \n    {off_diag_linear_angular}")
            print(f"  M_mj angular-linear coupling block [3:,:3]: \n    {off_diag_angular_linear}")
            print(f"  max abs linear-angular coupling: {np.max(np.abs(off_diag_linear_angular)):.4e}")
            print(f"  injected_wrench:              {injected_wrench}")
            print(f"  max abs injected_wrench:      {np.max(np.abs(injected_wrench)):.4e}")
            pending["expected_qacc"] = desired_qacc
            pending["data"] = data
            pending["active"] = True
        else:
            pending["active"] = False

        return original_inject(model, data, desired_v_dot, R)

    def spy_mj_step(model, data):
        result = original_mj_step(model, data)
        if pending.get("active"):
            realized_qacc = data.qacc.copy()
            expected_qacc = pending["expected_qacc"]
            diff = realized_qacc - expected_qacc
            print(f"  realized qacc (MuJoCo's, post mj_step): {realized_qacc}")
            print(f"  expected qacc (mixed frame):            {expected_qacc}")
            print(f"  qacc diff:                               {diff}")
            print(f"  max abs qacc diff:                      {np.max(np.abs(diff)):.4e}")
            print(f"  qacc MATCH (atol=1e-6): {np.allclose(realized_qacc, expected_qacc, atol=1e-6)}")
            pending["active"] = False
        return result

    mujoco_backend._inject_full_override_wrench = spy_inject
    mujoco.mj_step = spy_mj_step
    try:
        model = mujoco.MjModel.from_xml_path(MJCF_PATH)
        data = mujoco.MjData(model)
        sim_config = SimulationConfig(dt=run.dt, physics_dt=physics_dt)
        trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                        roll_pitch_series=run.roll_pitch)

        print(f"Replaying {filepath} through the REAL simulate_trajectory_mujoco, "
              f"instrumenting outer step {target_step} (substep calls "
              f"{target_substep_start}-{target_substep_end - 1} of "
              f"{n_substeps_expected} substeps/step)...")
        result = mujoco_backend.simulate_trajectory_mujoco(params, vehicle, sim_config, trajectory, model, data)
        print(f"\nTotal substep calls made: {call_counter['n']}")
        print(f"Final nu: {result.nu[-1]}")
        print(f"All finite: {np.all(np.isfinite(result.nu))}")
    finally:
        mujoco_backend._inject_full_override_wrench = original_inject
        mujoco.mj_step = original_mj_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filepath", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--physics-dt", type=float, default=0.005)
    parser.add_argument("--calibrated-params", default="calibrated_parameters_a1.yaml")
    args = parser.parse_args()
    inspect_step(args.filepath, args.sampling_rate_hz, args.target_step, args.physics_dt, args.calibrated_params)