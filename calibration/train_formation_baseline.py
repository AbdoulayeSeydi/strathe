from __future__ import annotations

import argparse
import time

import numpy as np

from two_agent_formation_env import TwoAgentFormationEnv, DEFAULT_ACTION_BOUND
from formation_hold_task import TERMINAL_REASON_FORMATION_ACHIEVED, TERMINAL_REASON_COLLISION
from comm_channel import PacketLossConfig, BandwidthConfig
from two_agent_orchestrator import OrchestratorConfig
from mlflow_run_utils import get_git_commit, compute_per_agent_rewards
from evaluate_comm_degradation_curve import VEHICLE_SPEC_PATH
import mlflow_logging

CALIBRATED_PARAMS_PATH = "calibrated_parameters_a1.yaml"


def build_zero_degradation_env(max_episode_steps: int = 200) -> TwoAgentFormationEnv:
    """A1.9's explicit scope: comm degradation OFF. Same permissive
    settings as A1.8's regression check, reused rather than redefined —
    if these ever drift apart, that's a real inconsistency worth
    catching, not two independently-maintained 'zero degradation'
    definitions."""
    return TwoAgentFormationEnv(
        calibrated_params_path=CALIBRATED_PARAMS_PATH,
        packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.0, 0.0),
        bandwidth_config=BandwidthConfig(bytes_per_tick=1e9),
        randomize_initial_state=True, 
        orchestrator_config=OrchestratorConfig(max_episode_steps=max_episode_steps),
        action_bound=DEFAULT_ACTION_BOUND,
    )


def evaluate_policy(policy_fn, n_episodes: int, max_episode_steps: int, seed: int) -> dict:
    """policy_fn(obs) -> action (12-dim, in [-1,1]). Works for both a
    trained SB3 model's .predict() and a plain random-action baseline —
    same evaluation path for both, so the comparison is apples-to-apples,
    matching finding #37's own 'two baselines, same harness' discipline.

    Reports well beyond success_rate — per direct request: final
    separation error, minimum separation ever reached (did it get
    close even if it never held), the deepest consecutive-in-tolerance
    streak reached (progress toward the 20-step hold requirement, even
    on episodes that ultimately time out), and a full reward-component
    breakdown, all pulled directly from what FormationHoldTask.step()
    already reports in info — nothing here is a new computation
    independent of the actual reward/termination logic already tested
    in A1.3-A1.5.

    C0.5 addition (see conversation): also returns reward_generic/
    reward_calibrated, a per-agent decomposition of the combined reward
    -- see compute_per_agent_rewards' own docstring for why this split
    is principled rather than arbitrary. Purely a derived logging
    metric; does not change what the policy is trained or evaluated on.
    """
    from collections import Counter
    from formation_hold_task import FormationHoldConfig

    target_separation = FormationHoldConfig().target_separation
    env = build_zero_degradation_env(max_episode_steps=max_episode_steps)
    reward_ctrl_weight = env._orchestrator.task.config.reward_ctrl_weight
    rng = np.random.default_rng(seed)

    episode_rewards = []
    episode_lengths = []
    final_separation_errors = []
    min_separations = []
    max_consecutive_in_tolerance = []
    component_sums = {"separation": [], "control": [], "proximity": [], "collision": [], "success_bonus": []}
    terminal_reasons = []
    reward_generic_means, reward_calibrated_means = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        total_reward = 0.0
        steps = 0
        terminal_reason = None
        min_sep = float("inf")
        max_consec = 0
        this_episode_components = {k: 0.0 for k in component_sums}
        last_separation = None
        reward_generic_sum, reward_calibrated_sum = 0.0, 0.0

        for _ in range(max_episode_steps):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            r_g, r_c = compute_per_agent_rewards(reward, action[:6], action[6:], reward_ctrl_weight)
            reward_generic_sum += r_g
            reward_calibrated_sum += r_c

            sep = info.get("true_separation")
            if sep is not None:
                min_sep = min(min_sep, sep)
                last_separation = sep
            max_consec = max(max_consec, info.get("consecutive_steps_in_tolerance", 0))
            comps = info.get("reward_components", {})
            for k in this_episode_components:
                this_episode_components[k] += comps.get(k, 0.0)

            if terminated:
                terminal_reason = info.get("terminal_reason")
                break
            if truncated:
                break

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        min_separations.append(min_sep)
        max_consecutive_in_tolerance.append(max_consec)
        for k in component_sums:
            component_sums[k].append(this_episode_components[k])
        if last_separation is not None:
            final_separation_errors.append(abs(last_separation - target_separation))
        terminal_reasons.append(terminal_reason if terminal_reason else "timeout")
        reward_generic_means.append(reward_generic_sum / steps)
        reward_calibrated_means.append(reward_calibrated_sum / steps)

    reason_counts = Counter(terminal_reasons)
    successes = reason_counts.get(TERMINAL_REASON_FORMATION_ACHIEVED, 0)
    collisions = reason_counts.get(TERMINAL_REASON_COLLISION, 0)
    timeouts = reason_counts.get("timeout", 0)

    return {
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "success_rate": successes / n_episodes,
        "collision_rate": collisions / n_episodes,
        "timeout_rate": timeouts / n_episodes,
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_final_separation_error": float(np.mean(final_separation_errors)) if final_separation_errors else None,
        "mean_min_separation": float(np.mean(min_separations)),
        "mean_max_consecutive_steps_in_tolerance": float(np.mean(max_consecutive_in_tolerance)),
        "reward_component_means": {k: float(np.mean(v)) for k, v in component_sums.items()},
        "terminal_reason_counts": dict(reason_counts),
        "reward_generic": float(np.mean(reward_generic_means)),
        "reward_calibrated": float(np.mean(reward_calibrated_means)),
    }


def random_policy_fn(action_space):
    def _fn(obs):
        return action_space.sample()
    return _fn


def _print_eval(label: str, result: dict) -> None:
    print(f"\n{label}:")
    print(f"  mean_reward:                      {result['mean_reward']:.3f} (+/- {result['std_reward']:.3f})")
    print(f"  success_rate:                     {result['success_rate']:.0%}")
    print(f"  collision_rate:                   {result['collision_rate']:.0%}")
    print(f"  timeout_rate:                     {result['timeout_rate']:.0%}")
    print(f"  mean_episode_length:              {result['mean_episode_length']:.1f}")
    fse = result['mean_final_separation_error']
    print(f"  mean_final_separation_error:      {fse:.3f}" if fse is not None else "  mean_final_separation_error:      N/A")
    print(f"  mean_min_separation:              {result['mean_min_separation']:.3f}")
    print(f"  mean_max_consecutive_in_tolerance:{result['mean_max_consecutive_steps_in_tolerance']:.1f}  (target: 20)")
    print(f"  reward components (mean per episode):")
    for k, v in result["reward_component_means"].items():
        print(f"    {k:16s}: {v:.3f}")
    print(f"  terminal reason counts: {result['terminal_reason_counts']}")


def _print_decision_rule(trained_result: dict) -> None:
    """Prints which branch of the agreed decision rule this result falls
    into — computed here, not left for manual eyeballing of the numbers
    above."""
    sr = trained_result["success_rate"]
    consec = trained_result["mean_max_consecutive_steps_in_tolerance"]
    fse = trained_result["mean_final_separation_error"]

    print("\n" + "=" * 60)
    print("DECISION RULE")
    print("=" * 60)
    if 0.30 <= sr <= 0.80:
        print(f"success_rate = {sr:.0%} — in the 30-80% target range. The task/reward "
              f"formulation is working. Next: introduce initial-state randomization "
              f"and retrain — the generalization stage, not before.")
    elif sr > 0.80:
        print(f"success_rate = {sr:.0%} — ABOVE the 30-80% target range. Worth checking "
              f"whether the task has become too easy at this fixed starting condition "
              f"before treating this as unambiguously good news.")
    elif sr > 0.0:
        print(f"success_rate = {sr:.0%} — nonzero but below the 30% target. Real progress, "
              f"but check mean_max_consecutive_in_tolerance ({consec:.1f}/20) and "
              f"mean_final_separation_error ({fse:.3f}) too — a rising trend across "
              f"successive runs at increasing timestep counts matters more here than "
              f"this single number alone.")
    else:
        print(f"success_rate = 0%. Per the agreed rule, this alone does NOT mean "
              f"something is broken — check whether mean_final_separation_error "
              f"({fse:.3f}) and mean_max_consecutive_in_tolerance ({consec:.1f}/20) are "
              f"ALSO plateaued (compare directly against a prior run's own numbers via "
              f"compare_learning_curves.py, if you have one) before concluding the "
              f"reward scale, action dynamics, observation design, PPO config, or task "
              f"implementation need real investigation. A still-improving reward curve "
              f"with 0% success can still mean 'needs more time,' not 'broken.'")


def run_gate1(seed: int = 0) -> None:
    """Construction + a handful of real gradient updates. Fast, cheap —
    run this first, every time, before ever attempting Gate 2. Failing
    here means something structural is wrong (env/SB3 incompatibility),
    not a training-quality question."""
    from stable_baselines3 import PPO

    print("GATE 1 — construction and integration check")
    env = build_zero_degradation_env(max_episode_steps=50)

    print("Constructing PPO...")
    model = PPO("MlpPolicy", env, verbose=0, seed=seed)

    print("Running a tiny learn() call (500 timesteps) — proves the full "
          "PPO <-> env loop executes without crashing, not that anything "
          "useful has been learned yet...")
    t0 = time.monotonic()
    model.learn(total_timesteps=500)
    elapsed = time.monotonic() - t0
    print(f"Completed in {elapsed:.1f}s with no crash.")

    print("Confirming .predict() works and returns a valid action...")
    obs, _ = env.reset(seed=seed)
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (12,), f"expected action shape (12,), got {action.shape}"
    assert np.all(action >= -1.0) and np.all(action <= 1.0), (
        f"predicted action outside [-1,1]: {action}"
    )
    print(f"predict() returned a valid action: {action}")

    print("\nGATE 1 PASSED — PPO/env integration is structurally sound. "
          "Proceed to Gate 2 for a real learning check.")


def run_gate2(total_timesteps: int, eval_episodes: int, seed: int, run_name: str = None,
              ent_coef: float = 0.0) -> None:
    """Real training, then a fresh, honest comparison against a random
    baseline — computed in THIS run, not borrowed from anywhere else.

    ent_coef (SB3's PPO default: 0.0) — added specifically to test a real,
    evidence-backed hypothesis from the 150k/400k runs: `std` (action
    noise) collapsed from 0.99 to ~0.10 over 400k steps while
    success_rate stayed flat at 0% and mean_max_consecutive_in_tolerance
    stayed flatly 0.0 (never even one step in-tolerance, across 20 eval
    episodes) despite real reward improvement. That combination — reward
    improving via a control-cost collapse to near-zero, not via actually
    closing the gap — looks like premature entropy collapse into a
    low-effort local optimum, not slow convergence. A nonzero ent_coef
    directly counteracts that by penalizing low-entropy (low-exploration)
    policies during training. NOT set as the new default — this is a
    targeted experiment, not a decided fix, until it's actually tested.

    C0.5 addition (see conversation): logs both the random baseline and
    the trained-policy evaluation to MLflow via log_fleet_run(), one run
    each -- comm_severity_level="L0" always (this function's own,
    explicit zero-degradation scope, per build_zero_degradation_env()),
    dynamics_backend="stand_in" always (training happens here, and per
    Decision 2 training is always stand_in)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    run_name = run_name or str(total_timesteps)
    model_path = f"formation_hold_ppo_{run_name}"
    monitor_path = f"monitor_{run_name}.csv"
    # BUG FOUND AND FIXED (see conversation): stable_baselines3's Monitor
    # wrapper unconditionally appends its own ".monitor.csv" suffix to
    # whatever filename it's given (confirmed by real, empirical evidence:
    # a run with monitor_path="monitor_my_run2.csv" actually wrote
    # "monitor_my_run2.csv.monitor.csv" to disk) -- this is a PRE-EXISTING
    # bug in this file, predating C0.5 entirely; it was invisible before
    # because the original code only ever PRINTED monitor_path as a
    # message, never actually tried to open the file. log_fleet_run()
    # below does try to open it, which is what surfaced this. monitor_path
    # itself is left unchanged (still passed to Monitor()'s own
    # constructor below) to avoid any risk of double-suffixing --
    # monitor_path_on_disk is the actual, real path, used everywhere the
    # code needs to touch the real file.
    monitor_path_on_disk = monitor_path + ".monitor.csv"
    git_commit = get_git_commit()

    print(f"GATE 2 — real training check ({total_timesteps} timesteps, "
          f"run_name={run_name}, ent_coef={ent_coef})")

    print("\nStep 1/3: evaluating the RANDOM-ACTION baseline "
          f"({eval_episodes} episodes)...")
    env_for_space = build_zero_degradation_env()
    random_result = evaluate_policy(
        random_policy_fn(env_for_space.action_space), eval_episodes,
        max_episode_steps=200, seed=seed,
    )
    _print_eval("RANDOM BASELINE", random_result)
    mlflow_logging.log_fleet_run(
        script="train_formation_baseline.py",
        run_name=f"{run_name}_random_baseline",
        status="SUCCESS",
        status_reason="random baseline evaluation completed",
        git_commit=git_commit,
        vehicle_spec_path=VEHICLE_SPEC_PATH,
        comm_severity_level="L0",
        dynamics_backend="stand_in",
        coordination_success_rate=random_result["success_rate"],
        reward_generic=random_result["reward_generic"],
        reward_calibrated=random_result["reward_calibrated"],
        params={"n_agents": 2, "run_name": run_name},
        metrics={"collision_rate": random_result["collision_rate"],
                 "mean_max_consecutive_in_tolerance": random_result["mean_max_consecutive_steps_in_tolerance"]},
        tags={"policy_type": "random"},
    )

    print(f"\nStep 2/3: training PPO for {total_timesteps} timesteps "
          f"(ent_coef={ent_coef}) — logging per-episode reward/length to "
          f"{monitor_path}...")
    train_env = build_zero_degradation_env()
    train_env = Monitor(train_env, filename=monitor_path)
    model = PPO("MlpPolicy", train_env, verbose=1, seed=seed, ent_coef=ent_coef)
    t0 = time.monotonic()
    model.learn(total_timesteps=total_timesteps)
    elapsed = time.monotonic() - t0
    print(f"\nTraining completed in {elapsed:.1f}s ({elapsed/60:.1f} min).")

    # BUG FOUND AND FIXED (see conversation): model.save() and train_env.close()
    # used to happen much later in this function, AFTER the trained-policy
    # log_fleet_run() call below tried to attach model_path/monitor_path as
    # artifacts -- meaning every real run has been silently failing to log
    # its own model and training-curve artifacts to MLflow, since neither
    # file existed on disk yet at that point. Moved here, immediately after
    # training completes, so both files are real and present before anything
    # tries to read them. train_env.close() added so the Monitor wrapper's
    # CSV is actually flushed to disk too, not just assumed to be.
    train_env.close()
    model.save(model_path)

    print(f"\nStep 3/3: evaluating the TRAINED policy ({eval_episodes} episodes, "
          f"same seed as the random baseline for a fair comparison)...")

    def trained_policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    trained_result = evaluate_policy(trained_policy_fn, eval_episodes,
                                      max_episode_steps=200, seed=seed)
    _print_eval("TRAINED POLICY", trained_result)
    mlflow_logging.log_fleet_run(
        script="train_formation_baseline.py",
        run_name=f"{run_name}_trained",
        status="SUCCESS",
        status_reason=f"trained {total_timesteps} timesteps, ent_coef={ent_coef}, elapsed={elapsed:.1f}s",
        git_commit=git_commit,
        vehicle_spec_path=VEHICLE_SPEC_PATH,
        comm_severity_level="L0",
        dynamics_backend="stand_in",
        coordination_success_rate=trained_result["success_rate"],
        reward_generic=trained_result["reward_generic"],
        reward_calibrated=trained_result["reward_calibrated"],
        params={"n_agents": 2, "run_name": run_name, "total_timesteps": total_timesteps,
                "ent_coef": ent_coef, "algorithm": "PPO", "seed": seed},
        metrics={"collision_rate": trained_result["collision_rate"],
                 "mean_max_consecutive_in_tolerance": trained_result["mean_max_consecutive_steps_in_tolerance"],
                 "runtime_seconds": elapsed},
        artifacts=[f"{model_path}.zip", monitor_path_on_disk],
        tags={"policy_type": "trained"},
    )

    print("\n" + "=" * 60)
    print("COMPARISON (fresh numbers, this run — not borrowed from elsewhere)")
    print("=" * 60)
    print(f"  success_rate:  random={random_result['success_rate']:.0%}  "
          f"trained={trained_result['success_rate']:.0%}")
    print(f"  mean_reward:   random={random_result['mean_reward']:.3f}  "
          f"trained={trained_result['mean_reward']:.3f}")
    print(f"  mean_min_separation: random={random_result['mean_min_separation']:.3f}  "
          f"trained={trained_result['mean_min_separation']:.3f}")
    print(f"  mean_max_consecutive_in_tolerance: random="
          f"{random_result['mean_max_consecutive_steps_in_tolerance']:.1f}  "
          f"trained={trained_result['mean_max_consecutive_steps_in_tolerance']:.1f}")

    _print_decision_rule(trained_result)

    print(f"\nSaved trained model to {model_path}.zip")
    # BUG FOUND AND FIXED (see conversation): this used to hardcode a
    # reference to "the 150k -> 400k comparison" and "monitor_400k.csv" --
    # both specific to an earlier historical run on this project, meaning
    # neither meant anything to a stranger running their own --run-name.
    # Generalized to describe the actual pattern without assuming a prior
    # run exists.
    print(f"Saved per-episode training log to {monitor_path_on_disk} — to compare against "
          f"a PRIOR run of your own (if you have one), use:\n"
          f"  python3 compare_learning_curves.py --path-a <prior_monitor.csv> "
          f"--path-b {monitor_path_on_disk} --label-a <prior_label> --label-b {run_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gate1", "gate2"], default="gate1",
                         help="gate1 (default): fast construction/integration check. "
                              "gate2: real training + comparison against a random baseline. "
                              "ALWAYS run gate1 first, every time this changes.")
    parser.add_argument("--total-timesteps", type=int, default=20000,
                         help="Gate 2 only. Starting point, not a validated final number — "
                              "this project's own single-agent PPO scale was found through "
                              "real experimentation, not decided on paper (see "
                              "PROJECT_STATUS.md); expect to adjust this after seeing a "
                              "first real result.")
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None,
                         help="Tags saved model/log filenames. Defaults to str(total_timesteps) — "
                              "set explicitly if running the same timestep count twice.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
                         help="PPO's entropy bonus coefficient (SB3 default: 0.0). Raise this "
                              "to directly counteract premature action-noise collapse — see "
                              "the module docstring's ent_coef note for why this is worth "
                              "testing specifically, not a general-purpose knob to tune blindly.")
    args = parser.parse_args()

    if args.mode == "gate1":
        run_gate1(seed=args.seed)
    else:
        run_gate2(total_timesteps=args.total_timesteps, eval_episodes=args.eval_episodes,
                   seed=args.seed, run_name=args.run_name, ent_coef=args.ent_coef)