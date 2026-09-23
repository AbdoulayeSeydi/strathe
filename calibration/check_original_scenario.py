"""
check_original_scenario.py -- directly verifies the ORIGINAL fixed-start
scenario (d0=4.0, v0=0.0 exactly) that A1.9 was scoped against from the
start. Not covered by evaluate_recovery_grid.py's grid (its velocities
are -0.7/-0.15/0.15/0.7, never exactly 0.0), so this hasn't actually
been directly re-checked for the curriculum-trained model despite being
the original, literal scope of A1.9.

Deterministic given fixed (d0, v0) and deterministic=True policy
(already established earlier in this investigation) -- one seed is
sufficient, no need for repeats.
"""
import argparse
from evaluate_recovery import build_env, run_scenario


def main(model_path):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env = build_env()
    true_original_step = env._orchestrator.step

    result = run_scenario(env, model, true_original_step, 4.0, 0.0, seed=0)
    print(f"ORIGINAL scenario (d0=4.0, v0=0.0 exactly):")
    print(f"  classification: {result['classification']}")
    print(f"  steps_run: {result['steps_run']}  terminal_reason: {result['terminal_reason']}")
    print(f"  final_separation: {result['final_separation']:.3f}  min_separation: {result['min_separation']:.3f}")
    print(f"  max_consecutive_in_tolerance: {result['max_consecutive_in_tolerance']} (target=20)")
    print(f"  fraction_steps_saturated: {result['fraction_steps_saturated']:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()
    main(args.model_path)