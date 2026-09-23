"""
Direct physics check — NOT inferred from RL training. Computes
nu_dot = M^-1 @ tau with tau = [surge_force, 0, 0, 0, 0, 0] and
EVERYTHING else zero (no damping, no Coriolis, no restoring force — those
all depend on nu or orientation, both held at their zero/reference
values here). This isolates the mass matrix's off-diagonal coupling
terms (Xq_dot, Zq_dot) as the ONLY possible source of a nonzero pitch
response — a clean causal demonstration, not something inferred from
watching a trained policy's behavior.

Run this against your REAL calibrated_parameters_a1.yaml (from the
actual 5-file BlueROV2 run, Xq_dot=-0.2297) to see the real magnitude —
any test run against a synthetic/placeholder calibration file will show
a much smaller, non-representative number.
"""
import numpy as np

from dynamics_model import _mass_and_damping_arrays
from build_generic_parameters import build_generic_parameters
from build_two_agent_cores import load_calibrated_parameters
from optimizer import load_vehicle_spec

VEHICLE_SPEC_PATH = "vehicle_spec.yaml"


def main(calibrated_params_path: str, surge_force: float = 30.0):
    vehicle = load_vehicle_spec(VEHICLE_SPEC_PATH).vehicle
    generic_params = build_generic_parameters()
    calibrated_params = load_calibrated_parameters(calibrated_params_path)

    tau = np.array([surge_force, 0.0, 0.0, 0.0, 0.0, 0.0])

    print(f"Pure surge force ({surge_force}N), everything else zero. "
          f"nu_dot = M^-1 @ tau, no damping/Coriolis/restoring involved.\n")

    for label, params in [("GENERIC", generic_params), ("CALIBRATED", calibrated_params)]:
        M, _, _ = _mass_and_damping_arrays(params, vehicle)
        nu_dot = np.linalg.solve(M, tau)
        print(f"{label}:")
        print(f"  Xq_dot (M[0,4], surge-pitch mass coupling): {M[0, 4]:.6f}")
        print(f"  Zq_dot (M[2,4], heave-pitch mass coupling): {M[2, 4]:.6f}")
        print(f"  nu_dot = {nu_dot}")
        print(f"  q_dot (pitch angular acceleration, index 4): {nu_dot[4]:.6f}")
        print()

    print("If CALIBRATED's q_dot is meaningfully nonzero and GENERIC's is exactly "
          "0.0 (guaranteed, since generic's Xq_dot=Zq_dot=0.0 by construction), "
          "that's the clean, direct causal demonstration — pure surge thrust "
          "produces pitch angular acceleration for the calibrated model, from "
          "the mass matrix alone, before any damping/Coriolis/RL behavior enters.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-params", default="calibrated_parameters_a1.yaml")
    parser.add_argument("--surge-force", type=float, default=30.0)
    args = parser.parse_args()
    main(args.calibrated_params, args.surge_force)