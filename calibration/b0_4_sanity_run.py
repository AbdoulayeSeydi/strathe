"""
B0.4 -- Trivial sanity run for Track B's single-body MJCF (B0.3).

Loads the model and steps it with ZERO applied force of any kind: no
xfrc_applied set, no actuators exist in this model at all (confirmed
below via model.nu == 0), so MuJoCo's own native gravity is the only
force acting on the body -- nothing calibrated, nothing from
compute_acceleration(), matching B0.3's "pure passive body" design.

Checks the resulting fall against the analytical free-fall solution
(z(t) = -0.5*g*t^2) to a loose tolerance. This is NOT a physics-accuracy
claim -- it's a cheap, real pass/fail check that the pipeline runs and
behaves plausibly, stronger than eyeballing numbers but far short of
B1's actual validation work.
"""
import numpy as np
import mujoco

MJCF_PATH = "b0_single_body_placeholder.xml"


def run_sanity_check(n_steps: int = 200, tolerance: float = 0.05):
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)

    assert model.nu == 0, "expected zero actuators for this pure passive body (B0.3 design)"

    dt = model.opt.timestep
    g = -model.opt.gravity[2]  # gravity vector is [0, 0, -9.81] -> g = 9.81 (positive)

    z_positions = []
    times = []
    for step in range(n_steps):
        # No xfrc_applied, no ctrl (none exist), no calibrated physics of
        # any kind -- mj_step applies ONLY MuJoCo's own native gravity
        # and rigid-body dynamics to this single free body.
        mujoco.mj_step(model, data)
        z_positions.append(data.qpos[2])  # qpos = [x, y, z, qw, qx, qy, qz] for a freejoint
        times.append((step + 1) * dt)

    z_positions = np.array(z_positions)
    times = np.array(times)
    z_analytical = -0.5 * g * times**2

    final_z_sim = z_positions[-1]
    final_z_analytical = z_analytical[-1]
    relative_error = abs(final_z_sim - final_z_analytical) / abs(final_z_analytical)

    print(f"model.nu (actuators): {model.nu}")
    print(f"model.opt.timestep: {dt}")
    print(f"gravity magnitude: {g}")
    print(f"Steps: {n_steps}, total simulated time: {times[-1]:.4f} s")
    print(f"\nFinal z position (simulated):            {final_z_sim:.6f} m")
    print(f"Final z position (analytical free-fall):  {final_z_analytical:.6f} m")
    print(f"Relative error at final step:              {relative_error:.4%}")
    print(f"\nFirst 5 z positions (sim):        {z_positions[:5]}")
    print(f"First 5 z positions (analytical): {z_analytical[:5]}")

    assert final_z_sim < 0, "body should have fallen (negative z) under gravity"
    assert relative_error < tolerance, (
        f"simulated free-fall deviates from analytical solution by "
        f"{relative_error:.2%}, expected < {tolerance:.0%} for this trivial case"
    )
    print(f"\nB0.4 SANITY CHECK PASSED: body falls under gravity, within "
          f"{tolerance:.0%} of the analytical free-fall solution.")


if __name__ == "__main__":
    run_sanity_check()