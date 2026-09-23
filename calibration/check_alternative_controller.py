"""
check_alternative_controller.py — tests whether a SIMPLE, hand-designed
feedback controller (no RL, no trained model at all) can recover from
the SAME failure-inducing initial conditions where the trained PPO
policy could not (robust_failure: d0=5.0, v0=+0.7) or barely could
(collision: d0=3.75, v0=0.0).

This directly answers the remaining open question: does a BETTER action
exist at these states that PPO simply never learned to find (supporting
the training-distribution/narrow-basin hypothesis), or does even a
reasonable, deliberately-designed alternative controller ALSO fail
(pointing toward a genuine dynamics/control-authority limitation)?

Controller design (calibrated agent only; generic held at zero thrust,
matching every other diagnostic in this investigation):
  Fx = clip(-K_SEP*(separation - target) - K_VELDOT*d_dot, -bound, bound)
    -- drives separation toward target, damps relative velocity
  Ty = clip(-KP_PITCH*pitch - KD_PITCH*q, -bound, bound)
    -- the SAME PD pitch stabilizer already validated in
       check_pitch_torque_authority.py (gains kept identical)
  All other action components (Fy, Fz, Tx, Tz) = 0.

No RL model is used -- this needs no stable_baselines3/torch, unlike
check_value_calibration.py, so it IS fully tested in this sandbox
before being handed over.
"""
import argparse

import numpy as np

from evaluate_recovery import build_env, reset_to_custom_state, DEFAULT_ACTION_BOUND

TARGET_SEPARATION = 2.0
TOLERANCE_LOW, TOLERANCE_HIGH = 1.8, 2.2
K_SEP = 15.0
K_VELDOT = 10.0
KP_PITCH = 50.0
KD_PITCH = 20.0


def alternative_action(separation: float, d_dot: float, pitch: float, q: float,
                        action_bound: float = DEFAULT_ACTION_BOUND) -> np.ndarray:
    fx = -K_SEP * (separation - TARGET_SEPARATION) - K_VELDOT * d_dot
    ty = -KP_PITCH * pitch - KD_PITCH * q
    fx = float(np.clip(fx, -action_bound, action_bound))
    ty = float(np.clip(ty, -action_bound, action_bound))

    cal_norm = np.zeros(6, dtype=np.float32)
    cal_norm[0] = fx / action_bound
    cal_norm[4] = ty / action_bound
    gen_norm = np.zeros(6, dtype=np.float32)  # generic held stationary
    return np.concatenate([gen_norm, cal_norm]).astype(np.float32)


def run_case(env, d0: float, v0: float, seed: int = 0, max_steps: int = 200) -> dict:
    obs = reset_to_custom_state(env, d0, v0, seed)
    dt = env._orchestrator.config.dt

    sep_history = []
    min_sep = float("inf")
    max_consec = 0
    terminal_reason = None
    final_sep = None
    steps_run = 0

    for step in range(1, max_steps + 1):
        core_c = env._orchestrator.core_calibrated
        core_g = env._orchestrator.core_generic
        current_sep = float(np.linalg.norm(core_c.state.position - core_g.state.position))
        d_dot = (current_sep - sep_history[-1]) / dt if sep_history else 0.0
        sep_history.append(current_sep)

        pitch = core_c.state.orientation.as_euler("xyz")[1]
        q = core_c.state.nu[4]

        action = alternative_action(current_sep, d_dot, pitch, q)
        obs, reward, terminated, truncated, info = env.step(action)
        steps_run = step

        final_sep = info["true_separation"]
        min_sep = min(min_sep, final_sep)
        max_consec = max(max_consec, info["consecutive_steps_in_tolerance"])

        if terminated:
            terminal_reason = info.get("terminal_reason")
            break
        if truncated:
            break

    return {
        "d0": d0, "v0": v0, "steps_run": steps_run, "terminal_reason": terminal_reason,
        "final_separation": final_sep, "min_separation": min_sep,
        "max_consecutive_in_tolerance": max_consec,
    }


def main(seed: int = 0):
    env = build_env()
    cases = [
        ("robust_failure_state", 5.00, 0.70),
        ("collision_state", 3.75, 0.0),
        ("success_state (sanity check)", 3.00, 0.0),
    ]
    print(f"Alternative hand-designed controller (NO RL):")
    print(f"  Fx = -{K_SEP}*(sep-{TARGET_SEPARATION}) - {K_VELDOT}*d_dot")
    print(f"  Ty = -{KP_PITCH}*pitch - {KD_PITCH}*q")
    print(f"  both saturated to +/-{DEFAULT_ACTION_BOUND}\n")
    for label, d0, v0 in cases:
        result = run_case(env, d0, v0, seed)
        print(f"{label} (d0={d0}, v0={v0}):")
        print(f"  steps_run={result['steps_run']}  terminal_reason={result['terminal_reason']}")
        print(f"  final_separation={result['final_separation']:.3f}  "
              f"min_separation={result['min_separation']:.3f}")
        print(f"  max_consecutive_in_tolerance={result['max_consecutive_in_tolerance']} (target=20)")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.seed)