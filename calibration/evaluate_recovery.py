"""
evaluate_recovery.py — two dedicated recovery evaluations, run WITHOUT
touching the task/reward/training setup at all: only each ArenaCore's
initial position/velocity is overridden per scenario, using the
reset(position=..., nu=...) capability ArenaCore has always exposed.
Nothing about FormationHoldTask, the reward equation, or hold_steps
changes for this.

## Evaluation A — near-miss recovery
Start near the tolerance boundary (d=2.3, 2.5), with the calibrated
agent given a range of relative surge velocities. Question: once outside
[1.8, 2.2], does the policy restore d -> 2.0, or diverge?

## Evaluation B — far/receding recovery
Start at d=4.0, 5.0, ALREADY receding. Question: does the policy reverse
course, or keep pushing outward — the distinct second failure mode the
detailed trace revealed (steps ~140-200, sustained large positive Fx
while separation kept growing).

## Velocity magnitudes — grounded in the real trace, not guessed
"slow" (0.15 m/s) and "fast" (0.70 m/s) bracket the ACTUAL dsep/dt values
observed at the real divergence trace's two overshoot points: step 70
(dsep/dt=0.445) and step 165 (dsep/dt=1.332). Not arbitrary round numbers.

## Generic held stationary (nu=0) in every scenario
Deliberate simplification: attributes all relative velocity to the
calibrated agent specifically, matching how the real divergence trace's
Fx was doing almost all the observable work.
"""
import argparse

import numpy as np

from trace_episode_detailed import build_env

SLOW_VELOCITY = 0.15  # m/s -- brackets step 70's real dsep/dt=0.445 from below
FAST_VELOCITY = 0.70  # m/s -- brackets step 165's real dsep/dt=1.332 from below
TOLERANCE_LOW, TOLERANCE_HIGH = 1.8, 2.2  # FormationHoldConfig defaults


def reset_to_custom_state(env, calibrated_position_x: float, calibrated_surge_velocity: float, seed: int):
    """Normal env.reset() first (fresh comm channel, task state, step
    counter) — THEN override each core's position/velocity directly.
    This never touches task/reward/training code, only where each agent
    starts and how fast it's already moving."""
    env.reset(seed=seed)
    orch = env._orchestrator
    orch.core_generic.reset(position=np.array([0.0, 0.0, 0.0]), nu=np.zeros(6))
    cal_nu = np.zeros(6)
    cal_nu[0] = calibrated_surge_velocity
    orch.core_calibrated.reset(position=np.array([calibrated_position_x, 0.0, 0.0]), nu=cal_nu)
    extra_generic, extra_calibrated = orch._build_observations()
    return env._build_full_obs(extra_generic, extra_calibrated)


DEFAULT_ACTION_BOUND = 30.0  # matches TwoAgentFormationEnv's own default
SATURATION_THRESHOLD_FRACTION = 0.99  # a component counts as "saturated" at >=99% of action_bound


def safe_csv_path(requested_path: str) -> str:
    """Never silently overwrite a previous run's data. If requested_path
    already exists, append _1, _2, ... until finding a free name, and
    print what happened -- this is exactly the bug that lost the
    d0=3.0/v0=+0.7 collision-cell data from a real 160-row grid run when
    a later single-point run reused the same default filename."""
    import os
    if not os.path.exists(requested_path):
        return requested_path
    stem, ext = os.path.splitext(requested_path)
    i = 1
    while os.path.exists(f"{stem}_{i}{ext}"):
        i += 1
    new_path = f"{stem}_{i}{ext}"
    print(f"NOTE: '{requested_path}' already exists -- writing to '{new_path}' instead "
          f"so it isn't overwritten. Pass --csv-path explicitly to control this.")
    return new_path


def run_scenario(env, model, original_step, calibrated_position_x: float,
                  calibrated_surge_velocity: float, seed: int, max_steps: int = 200) -> dict:
    obs = reset_to_custom_state(env, calibrated_position_x, calibrated_surge_velocity, seed)

    captured_tau = {}

    def spy_step(action_generic, action_calibrated, **kwargs):
        captured_tau["calibrated"] = np.array(action_calibrated, dtype=float)
        return original_step(action_generic, action_calibrated, **kwargs)

    env._orchestrator.step = spy_step
    try:
        separations, pitches, pitch_rates, pitch_torques, fx_values, dsep_dts = [], [], [], [], [], []
        max_consec = 0
        reentered_at = None
        prev_sep = None
        terminal_reason = None
        steps_run = 0
        saturated_steps = 0
        cumulative_control_effort = 0.0
        saturation_threshold = SATURATION_THRESHOLD_FRACTION * DEFAULT_ACTION_BOUND

        for step in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            steps_run = step

            sep = info["true_separation"]
            dsep_dt = (sep - prev_sep) / env._orchestrator.config.dt if prev_sep is not None else 0.0
            prev_sep = sep
            consec = info["consecutive_steps_in_tolerance"]
            cal_pitch = env._orchestrator.core_calibrated.state.orientation.as_euler("xyz")[1]
            cal_q = env._orchestrator.core_calibrated.state.nu[4]
            tau_c = captured_tau["calibrated"]

            separations.append(sep)
            pitches.append(cal_pitch)
            pitch_rates.append(cal_q)
            pitch_torques.append(tau_c[4])
            fx_values.append(tau_c[0])
            dsep_dts.append(dsep_dt)
            max_consec = max(max_consec, consec)

            if np.max(np.abs(tau_c)) >= saturation_threshold:
                saturated_steps += 1
            # Unweighted sum-of-squares of the NORMALIZED action (tau_c/action_bound),
            # matching the reward's own control-cost quantity structurally (before
            # the 0.01 weight) -- an interpretable "how much effort was spent" figure.
            normalized_action = tau_c / DEFAULT_ACTION_BOUND
            cumulative_control_effort += float(np.sum(normalized_action ** 2))

            if reentered_at is None and TOLERANCE_LOW <= sep <= TOLERANCE_HIGH:
                reentered_at = step

            if terminated:
                terminal_reason = info.get("terminal_reason")
                break
            if truncated:
                break
    finally:
        env._orchestrator.step = original_step

    max_consec_target = env._orchestrator.task.config.hold_steps
    if terminal_reason == "collision":
        classification = "C"  # collision -- NOT equivalent to a merely-failed hold
    elif max_consec >= max_consec_target:
        classification = "S"  # full hold achieved
    elif reentered_at is not None:
        classification = "R"  # re-entered tolerance but never held it
    else:
        classification = "F"  # never recovered at all

    return {
        "initial_separation": calibrated_position_x,
        "initial_surge_velocity": calibrated_surge_velocity,
        "seed": seed,
        "classification": classification,
        "steps_run": steps_run,
        "terminal_reason": terminal_reason,
        "final_separation": separations[-1],
        "min_separation": min(separations),
        "max_separation": max(separations),
        "max_abs_pitch": max(abs(p) for p in pitches),
        "max_abs_pitch_rate": max(abs(q) for q in pitch_rates),
        "max_abs_pitch_torque": max(abs(t) for t in pitch_torques),
        "mean_fx": float(np.mean(fx_values)),
        "final_dsep_dt": dsep_dts[-1],
        "reentered_tolerance_at_step": reentered_at,
        "max_consecutive_in_tolerance": max_consec,
        "fraction_steps_saturated": saturated_steps / steps_run,
        "cumulative_control_effort": cumulative_control_effort,
    }


def print_result(label: str, result: dict) -> None:
    print(f"\n{label}  [{result['classification']}]")
    print(f"  initial: separation={result['initial_separation']:.2f}m, "
          f"surge_velocity={result['initial_surge_velocity']:+.2f}m/s")
    print(f"  steps_run: {result['steps_run']}, terminal_reason: {result['terminal_reason']}")
    print(f"  final_separation: {result['final_separation']:.3f}  "
          f"(min={result['min_separation']:.3f}, max={result['max_separation']:.3f})")
    print(f"  max_abs_pitch: {result['max_abs_pitch']:.3f} rad   "
          f"max_abs_pitch_rate: {result['max_abs_pitch_rate']:.3f} rad/s   "
          f"max_abs_pitch_torque: {result['max_abs_pitch_torque']:.2f}")
    print(f"  mean_Fx: {result['mean_fx']:+.2f}   final_dsep/dt: {result['final_dsep_dt']:+.3f}")
    print(f"  reentered_tolerance_at_step: {result['reentered_tolerance_at_step']}   "
          f"max_consecutive_in_tolerance: {result['max_consecutive_in_tolerance']}")
    print(f"  fraction_steps_saturated: {result['fraction_steps_saturated']:.2f}   "
          f"cumulative_control_effort: {result['cumulative_control_effort']:.2f}")


def main(model_path: str, seed: int) -> None:
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env = build_env()
    true_original_step = env._orchestrator.step

    print("=" * 78)
    print(f"EVALUATION A — near-miss recovery (d=2.3, 2.5)")
    print(f"velocities: slow=+/-{SLOW_VELOCITY} m/s, fast=+/-{FAST_VELOCITY} m/s "
          f"(grounded in the real trace's dsep/dt at the two overshoot points)")
    print("=" * 78)
    scenarios_a = [
        (2.3, -SLOW_VELOCITY, "approaching slowly"), (2.3, -FAST_VELOCITY, "approaching quickly"),
        (2.3, +SLOW_VELOCITY, "receding slowly"), (2.3, +FAST_VELOCITY, "receding quickly"),
        (2.5, -SLOW_VELOCITY, "approaching slowly"), (2.5, -FAST_VELOCITY, "approaching quickly"),
        (2.5, +SLOW_VELOCITY, "receding slowly"), (2.5, +FAST_VELOCITY, "receding quickly"),
    ]
    results_a = []
    for sep0, vel0, label in scenarios_a:
        result = run_scenario(env, model, true_original_step, sep0, vel0, seed)
        results_a.append(result)
        print_result(f"d0={sep0}, {label} (v={vel0:+.2f})", result)

    print("\n" + "=" * 78)
    print("EVALUATION B — far/receding recovery (d=4.0, 5.0, already moving apart)")
    print("=" * 78)
    scenarios_b = [
        (4.0, +SLOW_VELOCITY, "receding slowly"), (4.0, +FAST_VELOCITY, "receding quickly"),
        (5.0, +SLOW_VELOCITY, "receding slowly"), (5.0, +FAST_VELOCITY, "receding quickly"),
    ]
    results_b = []
    for sep0, vel0, label in scenarios_b:
        result = run_scenario(env, model, true_original_step, sep0, vel0, seed)
        results_b.append(result)
        print_result(f"d0={sep0}, {label} (v={vel0:+.2f})", result)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    n_reentered = sum(1 for r in results_a if r["reentered_tolerance_at_step"] is not None)
    n_full_hold = sum(1 for r in results_a if r["max_consecutive_in_tolerance"] >= 20)
    print(f"Evaluation A: {n_reentered}/{len(results_a)} scenarios re-entered "
          f"[{TOLERANCE_LOW},{TOLERANCE_HIGH}] at some point; "
          f"{n_full_hold}/{len(results_a)} reached the full hold_steps=20")

    n_reversed = sum(1 for r in results_b if r["final_dsep_dt"] < 0)
    n_correcting_fx = sum(1 for r in results_b if r["mean_fx"] < 0)
    print(f"Evaluation B: {n_reversed}/{len(results_b)} scenarios ended with dsep/dt < 0 "
          f"(receding motion reversed by episode end)")
    print(f"Evaluation B: {n_correcting_fx}/{len(results_b)} scenarios had net-negative "
          f"(correcting, toward-target) mean Fx across the episode")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.model_path, args.seed)