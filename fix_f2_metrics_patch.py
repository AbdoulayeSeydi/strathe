with open("calibration/train_multiseed.py", encoding="utf-8") as f:
    content = f.read()

replacements = []

# 1. train_one_seed() must return the FULL set of EvalResult fields,
# not just the original 3 - collision/timeout/episode-length/time-to-
# goal/clearance were computed in Phase C but silently discarded here.
old1 = '''    result = evaluate_policy(eval_env, policy_fn=policy_fn, n_episodes=n_eval_episodes, seed=seed)
    return {"seed": seed, "mean_reward": result.mean_reward,
            "mean_final_distance": result.mean_final_distance,
            "success_rate": result.success_rate}'''
new1 = '''    result = evaluate_policy(eval_env, policy_fn=policy_fn, n_episodes=n_eval_episodes, seed=seed)
    return {"seed": seed, "mean_reward": result.mean_reward,
            "mean_final_distance": result.mean_final_distance,
            "success_rate": result.success_rate,
            "collision_rate": result.collision_rate,
            "timeout_rate": result.timeout_rate,
            "mean_episode_length": result.mean_episode_length,
            "mean_time_to_goal": result.mean_time_to_goal,
            "mean_min_obstacle_clearance": result.mean_min_obstacle_clearance}'''
replacements.append(("train_one_seed return", old1, new1))

# 2. Per-seed print statement - show the new metrics too, with
# mean_min_obstacle_clearance only shown when it's not None (i.e. only
# meaningful for obstacle_avoidance, not printed as a misleading 0.0
# or omitted silently for other tasks).
old2 = '''            print(f"    reward={r['mean_reward']:.2f}, "
                  f"distance={r['mean_final_distance']:.2f}, success={r['success_rate']:.2f}")'''
new2 = '''            clearance_str = (f", clearance={r['mean_min_obstacle_clearance']:.2f}"
                              if r['mean_min_obstacle_clearance'] is not None else "")
            print(f"    reward={r['mean_reward']:.2f}, "
                  f"distance={r['mean_final_distance']:.2f}, success={r['success_rate']:.2f}, "
                  f"collision={r['collision_rate']:.2f}, timeout={r['timeout_rate']:.2f}, "
                  f"ep_len={r['mean_episode_length']:.1f}{clearance_str}")'''
replacements.append(("per-seed print", old2, new2))

# 3. summarize() - aggregate the new metrics across seeds too.
old3 = '''    def summarize(variant_results):
        rewards = [r["mean_reward"] for r in variant_results]
        successes = [r["success_rate"] for r in variant_results]
        return {
            "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
            "mean_success_rate": float(np.mean(successes)), "std_success_rate": float(np.std(successes)),
            "per_seed": variant_results,
        }'''
new3 = '''    def summarize(variant_results):
        rewards = [r["mean_reward"] for r in variant_results]
        successes = [r["success_rate"] for r in variant_results]
        collisions = [r["collision_rate"] for r in variant_results]
        timeouts = [r["timeout_rate"] for r in variant_results]
        ep_lengths = [r["mean_episode_length"] for r in variant_results]
        clearances = [r["mean_min_obstacle_clearance"] for r in variant_results
                      if r["mean_min_obstacle_clearance"] is not None]
        return {
            "mean_reward": float(np.mean(rewards)), "std_reward": float(np.std(rewards)),
            "mean_success_rate": float(np.mean(successes)), "std_success_rate": float(np.std(successes)),
            "mean_collision_rate": float(np.mean(collisions)),
            "mean_timeout_rate": float(np.mean(timeouts)),
            "mean_episode_length": float(np.mean(ep_lengths)),
            "mean_min_obstacle_clearance": float(np.mean(clearances)) if clearances else None,
            "per_seed": variant_results,
        }'''
replacements.append(("summarize()", old3, new3))

# 4. Final aggregate print block - show the new metrics too.
old4 = '''    for v in ["default", "calibrated"]:
        s = summary[v]
        print(f"{v:<12} reward={s['mean_reward']:.2f}±{s['std_reward']:.2f}, "
              f"success={s['mean_success_rate']:.2f}±{s['std_success_rate']:.2f} "
              f"(n={args.n_seeds} seeds)")'''
new4 = '''    for v in ["default", "calibrated"]:
        s = summary[v]
        clearance_str = (f", clearance={s['mean_min_obstacle_clearance']:.2f}"
                          if s['mean_min_obstacle_clearance'] is not None else "")
        print(f"{v:<12} reward={s['mean_reward']:.2f}±{s['std_reward']:.2f}, "
              f"success={s['mean_success_rate']:.2f}±{s['std_success_rate']:.2f}, "
              f"collision={s['mean_collision_rate']:.2f}, timeout={s['mean_timeout_rate']:.2f}, "
              f"ep_len={s['mean_episode_length']:.1f}{clearance_str} "
              f"(n={args.n_seeds} seeds)")'''
replacements.append(("final aggregate print", old4, new4))

missing = [label for label, old, new in replacements if content.count(old) != 1]
if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}")
    for label, old, new in replacements:
        print(f"  {label}: {content.count(old)} occurrence(s)")
else:
    for label, old, new in replacements:
        content = content.replace(old, new, 1)
    with open("calibration/train_multiseed.py", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"All {len(replacements)} changes applied successfully.")