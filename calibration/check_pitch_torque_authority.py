"""
check_pitch_torque_authority.py — open-loop physics test, NO RL involved.

REVISION NOTE: an earlier version of this script swept a fixed set of
constant pitch torques (-30..+30) for 10 continuous seconds and found
max_abs_pitch ~1.5 rad at EVERY value tested -- which looked suspicious,
and was: since the mass/added-mass matrix M is velocity-independent, the
relationship q_dot = M_inv[4,0]*Fx + M_inv[4,4]*Ty is EXACTLY LINEAR,
meaning exactly one Ty perfectly cancels the coupling and any other
value leaves a small residual bias that a full 10-second CONSTANT hold
integrates into a large swept angle regardless of which wrong value was
tried. An arbitrary discrete sweep with 5-10-unit spacing was very
unlikely to land near that one exact value, so every tested point showed
similar runaway behavior for the same underlying reason -- not because
+/-30 was somehow uniquely necessary.

CORRECTED APPROACH: solve for the exact canceling torque analytically
(the system is linear at v=0, so this has a closed form -- no simulation
duration to confound it), then verify empirically that holding exactly
that torque keeps pitch near zero, and contrast against nearby "wrong"
values to show the actual sensitivity.

This directly answers: is the magnitude of torque genuinely required to
control pitch during full corrective surge SMALL (relative to the +/-30
actuator limit) or LARGE? If small, +/-30 is not genuinely required and
PPO's constant saturation (confirmed 12/12 in evaluate_recovery.py) is a
crude, not a necessary, strategy. If the exact canceling value is itself
close to +/-30, or if the system is so sensitive that only very precise
torque prevents runaway, that supports genuine authority-limitation.
"""
import numpy as np

from dynamics_model import _mass_and_damping_arrays
from build_two_agent_cores import build_two_agent_cores, load_calibrated_parameters
from optimizer import load_vehicle_spec

SURGE_FORCE = -30.0  # full corrective thrust, matching magnitudes actually
                       # observed driving real recovery attempts
N_STEPS = 100  # 10s at dt=0.1


def solve_exact_canceling_torque(calibrated_params_path: str, surge_force: float) -> float:
    """Closed-form: q_dot = M_inv[4,0]*Fx + M_inv[4,4]*Ty (tau's other
    components zero). Solve for Ty such that q_dot = 0 exactly."""
    vehicle = load_vehicle_spec("vehicle_spec.yaml").vehicle
    params = load_calibrated_parameters(calibrated_params_path)
    M, _, _ = _mass_and_damping_arrays(params, vehicle)
    M_inv = np.linalg.inv(M)
    return -M_inv[4, 0] * surge_force / M_inv[4, 4]


def run_constant_torque(calibrated_params_path: str, ty: float, n_steps: int = N_STEPS):
    _, core_calibrated = build_two_agent_cores(calibrated_params_path)
    core_calibrated.reset()
    tau = np.array([SURGE_FORCE, 0.0, 0.0, 0.0, ty, 0.0])
    pitches = []
    for _ in range(n_steps):
        core_calibrated.step(tau)
        _roll, pitch, _yaw = core_calibrated.state.orientation.as_euler("xyz")
        pitches.append(pitch)
    return pitches


def main(calibrated_params_path: str = "calibrated_parameters_a1.yaml"):
    exact_ty = solve_exact_canceling_torque(calibrated_params_path, SURGE_FORCE)
    print(f"Fixed surge force: {SURGE_FORCE}N")
    print(f"EXACT counter-torque that analytically cancels q_dot: {exact_ty:.4f} N*m")
    print(f"As a fraction of the +/-30 N*m actuator limit: {abs(exact_ty)/30.0*100:.2f}%\n")

    test_values = [
        ("exact", exact_ty),
        ("exact x0.5 (undershoot)", exact_ty * 0.5),
        ("exact x2.0 (overshoot)", exact_ty * 2.0),
        ("zero (no counter-torque)", 0.0),
        ("+/-30 (full actuator authority)", 30.0 if exact_ty <= 0 else -30.0),
    ]

    print(f"{'condition':>34} {'Ty used':>10} {'final_pitch':>12} {'max_abs_pitch':>14}")
    for label, ty in test_values:
        pitches = run_constant_torque(calibrated_params_path, ty)
        print(f"{label:>34} {ty:>10.4f} {pitches[-1]:>12.4f} {max(abs(p) for p in pitches):>14.4f}")

    print("\nInterpretation:")
    print(f"  If the exact canceling torque ({exact_ty:.4f} N*m) is a small fraction of")
    print("  +/-30, the vehicle does NOT genuinely need anywhere near full actuator")
    print("  authority to hold pitch steady under full corrective surge -- PPO")
    print("  saturating pitch torque in every recovery scenario (evaluate_recovery.py's")
    print("  12/12 finding) would then be a crude, not a physically necessary, strategy.")
    print("  Compare the 'exact' row's max_abs_pitch (should stay near 0) against the")
    print("  'zero' and '+/-30' rows (both likely still show significant excursions,")
    print("  for the SAME reason -- neither happens to be the one canceling value).")


def run_pd_controlled(calibrated_params_path: str, torque_limit: float,
                       kp: float = 50.0, kd: float = 20.0, n_steps: int = N_STEPS):
    """A CONSTANT feedforward torque can only cancel the pure M^-1 term at
    v=0 -- it goes stale immediately as surge velocity builds up and
    velocity-dependent (Coriolis) terms start contributing too, which the
    result above empirically confirms (the 'exact' row still diverges).
    This tests REACTIVE closed-loop control instead: at every step, apply
    Ty = clip(-kp*pitch - kd*q, -torque_limit, +torque_limit) -- a simple,
    fixed-gain PD stabilizer, saturated at torque_limit. Sweeping
    torque_limit answers: what's the SMALLEST actuator ceiling under which
    a reactive (not open-loop) controller can still hold pitch bounded?
    That's the fair comparison to what the RL policy is doing -- both are
    reactive -- and it directly tests genuine authority requirements
    rather than an open-loop feedforward's staleness."""
    _, core_calibrated = build_two_agent_cores(calibrated_params_path)
    core_calibrated.reset()
    pitches, ty_used, saturated_steps = [], [], 0
    for _ in range(n_steps):
        _roll, pitch, _yaw = core_calibrated.state.orientation.as_euler("xyz")
        q = core_calibrated.state.nu[4]
        ty_raw = -kp * pitch - kd * q
        ty = float(np.clip(ty_raw, -torque_limit, torque_limit))
        if abs(ty_raw) > torque_limit:
            saturated_steps += 1
        tau = np.array([SURGE_FORCE, 0.0, 0.0, 0.0, ty, 0.0])
        core_calibrated.step(tau)
        pitches.append(pitch)
        ty_used.append(ty)
    return pitches, ty_used, saturated_steps


def run_pd_sweep(calibrated_params_path: str):
    print("\n" + "=" * 70)
    print("CLOSED-LOOP (reactive PD) pitch stabilization under the same")
    print(f"{SURGE_FORCE}N sustained corrective surge -- sweeping the torque ceiling")
    print("=" * 70)
    torque_limits = [1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0]
    print(f"{'torque_limit':>13} {'final_pitch':>12} {'max_abs_pitch':>14} "
          f"{'%_steps_saturated':>18}")
    for limit in torque_limits:
        pitches, ty_used, sat_steps = run_pd_controlled(calibrated_params_path, limit)
        pct_sat = 100.0 * sat_steps / len(pitches)
        print(f"{limit:>13.1f} {pitches[-1]:>12.4f} {max(abs(p) for p in pitches):>14.4f} "
              f"{pct_sat:>18.1f}")
    print("\n  If max_abs_pitch drops to near-zero well before torque_limit=30 (e.g.")
    print("  already small at limit=5-10), a REACTIVE controller does not need full")
    print("  authority either -- strengthening the case that PPO's constant")
    print("  saturation is a learned inefficiency, not a physical necessity.")
    print("  If max_abs_pitch only becomes small as torque_limit approaches 30, that")
    print("  supports genuine authority-limitation even for a reactive controller.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrated-params", default="calibrated_parameters_a1.yaml")
    args = parser.parse_args()
    main(args.calibrated_params)
    run_pd_sweep(args.calibrated_params)