#!/usr/bin/env python3
"""
Gate 1 (Integration) for SAC - per the frozen plan, tests ONLY whether
the wiring is correct, not whether SAC learns anything (that's Gate 2).
Each sub-item gets its own explicit PASS/FAIL so a failure is immediately
diagnosable, not lumped into one crash.

Budgets (locked in Phase C):
  Construction budget: total_timesteps=1 (with learning_starts=0
    override, specifically to test a real gradient step in isolation -
    the REAL, default learning_starts=100 is tested separately below,
    at the Integration budget, which is what it's actually for)
  Integration budget: total_timesteps=500 (5x the real, confirmed
    learning_starts=100 - Phase A)

Run from calibration/:
    python gate1_sac_integration.py
"""
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from optimizer import load_vehicle_spec, initial_parameters
from arena import Arena

CONSTRUCTION_BUDGET = 1
INTEGRATION_BUDGET = 500
REAL_LEARNING_STARTS = 100  # confirmed real SB3 default, Phase A


def make_env():
    loaded = load_vehicle_spec("vehicle_spec.yaml")
    params = initial_parameters(loaded)
    return Monitor(Arena(params, loaded.vehicle, task="waypoint", max_episode_steps=100)), loaded, params


results = {}


def report(name, passed, detail=""):
    results[name] = passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))


print("=" * 70)
print("GATE 1 (INTEGRATION) - SAC")
print("=" * 70)

# --- Sub-test 1: construction + one real gradient step ---
print("\n--- 1. Construction + one real gradient step ---")
try:
    env, loaded, params = make_env()
    model = SAC("MlpPolicy", env, verbose=0, seed=0, learning_starts=0)
    n_updates_before = getattr(model, "_n_updates", None)
    model.learn(total_timesteps=CONSTRUCTION_BUDGET)
    n_updates_after = getattr(model, "_n_updates", None)
    detail = f"_n_updates before={n_updates_before}, after={n_updates_after}"
    report("construction_and_gradient_step", n_updates_after is not None and n_updates_after > (n_updates_before or 0), detail)
except Exception as e:
    report("construction_and_gradient_step", False, f"{type(e).__name__}: {e}")

# --- Sub-test 2: replay buffer correctness at REAL learning_starts ---
print("\n--- 2. Replay buffer correctness (real learning_starts=100) ---")
try:
    env, loaded, params = make_env()
    model = SAC("MlpPolicy", env, verbose=0, seed=0)  # REAL default learning_starts=100

    n_updates_log = []

    from stable_baselines3.common.callbacks import BaseCallback

    class TrackUpdates(BaseCallback):
        def _on_step(self):
            n_updates_log.append((self.model.num_timesteps, getattr(self.model, "_n_updates", None)))
            return True

    model.learn(total_timesteps=INTEGRATION_BUDGET, callback=TrackUpdates())

    before_samples = [n for (t, n) in n_updates_log if t < REAL_LEARNING_STARTS and n is not None]
    updates_after_threshold = [n for (t, n) in n_updates_log if t >= REAL_LEARNING_STARTS and n]
    zero_before = all(n == 0 for n in before_samples)
    some_after = len(updates_after_threshold) > 0 and updates_after_threshold[-1] > 0
    report("zero_updates_before_learning_starts", zero_before,
           f"{len(before_samples)} sample(s) inspected before step {REAL_LEARNING_STARTS}, "
           f"all confirmed n_updates=0" if zero_before and before_samples else
           f"{len(before_samples)} sample(s) inspected, zero_before={zero_before}")
    report("updates_begin_after_learning_starts", some_after,
           f"final _n_updates={updates_after_threshold[-1] if updates_after_threshold else 'N/A'}")

    buffer_size = model.replay_buffer.size()
    report("buffer_fills", buffer_size > 0, f"buffer.size()={buffer_size}")

    obs_arr = model.replay_buffer.observations
    rew_arr = model.replay_buffer.rewards
    has_nan_inf = (not np.all(np.isfinite(obs_arr[:buffer_size]))) or (not np.all(np.isfinite(rew_arr[:buffer_size])))
    report("buffer_free_of_nan_inf", not has_nan_inf)

    try:
        sample = model.replay_buffer.sample(min(32, buffer_size))
        report("buffer_sampling_succeeds", True, f"sampled batch shape ok")
    except Exception as e:
        report("buffer_sampling_succeeds", False, f"{type(e).__name__}: {e}")

except Exception as e:
    report("replay_buffer_block", False, f"{type(e).__name__}: {e}")

# --- Sub-test 3: save -> reload -> evaluate consistency ---
print("\n--- 3. Save -> reload -> evaluate consistency ---")
try:
    from evaluate_policy_arena import evaluate_policy

    env, loaded, params = make_env()
    model = SAC("MlpPolicy", env, verbose=0, seed=0)
    model.learn(total_timesteps=INTEGRATION_BUDGET)

    eval_env1, _, _ = make_env()
    def policy_fn1(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action
    result_before = evaluate_policy(eval_env1.env, policy_fn=policy_fn1, n_episodes=3, seed=0)

    model.save("gate1_sac_test_model")
    model.save_replay_buffer("gate1_sac_test_buffer")

    loaded_model = SAC.load("gate1_sac_test_model")
    loaded_model.load_replay_buffer("gate1_sac_test_buffer")

    eval_env2, _, _ = make_env()
    def policy_fn2(obs):
        action, _ = loaded_model.predict(obs, deterministic=True)
        return action
    result_after = evaluate_policy(eval_env2.env, policy_fn=policy_fn2, n_episodes=3, seed=0)

    consistent = abs(result_before.mean_reward - result_after.mean_reward) < 1e-6
    report("save_reload_consistent", consistent,
           f"before={result_before.mean_reward:.6f}, after={result_after.mean_reward:.6f}")
    report("replay_buffer_reloaded", loaded_model.replay_buffer.size() == model.replay_buffer.size(),
           f"before={model.replay_buffer.size()}, after={loaded_model.replay_buffer.size()}")
except Exception as e:
    report("save_reload_block", False, f"{type(e).__name__}: {e}")

# --- Sub-test 4: resume training (reset_num_timesteps=False) ---
print("\n--- 4. Resume training ---")
try:
    env, loaded, params = make_env()
    model = SAC("MlpPolicy", env, verbose=0, seed=0)
    model.learn(total_timesteps=INTEGRATION_BUDGET)
    timesteps_before_resume = model.num_timesteps
    model.save("gate1_sac_resume_test")

    resumed_model = SAC.load("gate1_sac_resume_test", env=env)
    resumed_model.learn(total_timesteps=INTEGRATION_BUDGET, reset_num_timesteps=False)
    timesteps_after_resume = resumed_model.num_timesteps

    genuinely_continued = timesteps_after_resume == (timesteps_before_resume + INTEGRATION_BUDGET)
    report("resume_continues_not_restarts", genuinely_continued,
           f"before={timesteps_before_resume}, after={timesteps_after_resume}, "
           f"expected={timesteps_before_resume + INTEGRATION_BUDGET}")
except Exception as e:
    report("resume_block", False, f"{type(e).__name__}: {e}")

print("\n" + "=" * 70)
print("GATE 1 SUMMARY")
print("=" * 70)
all_passed = all(results.values())
for name, passed in results.items():
    print(f"  {'PASS' if passed else 'FAIL'} - {name}")
print(f"\n{'ALL GATE 1 CHECKS PASSED' if all_passed else 'GATE 1 FAILED - see above'}")