"""
check_value_calibration.py — tests hypothesis D (training distribution
never taught recovery from these states) WITHOUT retraining, by querying
the trained CRITIC's value estimate V(s) along the three traced
trajectories.

The logic: if V(s) stays falsely optimistic right up until collision or
endless drift, the critic never learned to correctly evaluate this
region of state space -- direct evidence the training distribution
never meaningfully visited it (consistent with D: PPO only ever trained
from one fixed start, so states far from that trajectory's reachable
set were never seen in any rollout). If V(s) instead turns clearly
pessimistic well before the outcome is decided, that means the policy
DOES recognize it's in trouble -- pointing more toward "correct
recognition, wrong/insufficient corrective action" (closer to A/B than D).

IMPORTANT — UNVERIFIED: this sandbox has no network access and could not
install stable_baselines3 or torch, so the SB3-specific call below
(model.policy.predict_values) could not be tested against a real model
the way every other script in this session was. It's believed correct
for SB3 2.9.0 (confirmed your installed version from earlier training
logs) based on that version's ActorCriticPolicy API, with a manual
fallback if it raises AttributeError. Run this cautiously and report
back the exact error if it fails -- don't assume it's correct just
because it's here.
"""
import argparse

import numpy as np

from evaluate_recovery import build_env, reset_to_custom_state

CASES = [
    ("collision", 3.75, 0.0),
    ("robust_failure", 5.00, 0.70),
    ("success", 3.00, 0.0),
]


def get_value(model, obs: np.ndarray) -> float:
    import torch
    from stable_baselines3.common.utils import obs_as_tensor

    obs_tensor = obs_as_tensor(np.asarray(obs)[None, :], model.device)
    with torch.no_grad():
        try:
            value = model.policy.predict_values(obs_tensor)
        except AttributeError:
            # Fallback for SB3 versions/configs without predict_values --
            # replicate ActorCriticPolicy.predict_values manually.
            features = model.policy.extract_features(obs_tensor)
            if isinstance(features, tuple):
                _, latent_vf = features
            else:
                latent_vf = model.policy.mlp_extractor.forward_critic(features)
            value = model.policy.value_net(latent_vf)
    return float(value.item())


def trace_with_value(env, model, original_step, d0: float, v0: float,
                      seed: int = 0, max_steps: int = 200):
    obs = reset_to_custom_state(env, d0, v0, seed)
    values, separations = [], []
    for _ in range(max_steps):
        v = get_value(model, obs)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        values.append(v)
        separations.append(info["true_separation"])
        if terminated or truncated:
            break
    return values, separations


def main(model_path: str, seed: int = 0):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)
    env = build_env()
    true_original_step = env._orchestrator.step

    for case_name, d0, v0 in CASES:
        values, seps = trace_with_value(env, model, true_original_step, d0, v0, seed)
        n = len(values)
        sample_idxs = list(range(0, n, max(1, n // 10)))
        if sample_idxs[-1] != n - 1:
            sample_idxs.append(n - 1)

        print(f"\n{case_name} (d0={d0}, v0={v0}), {n} steps:")
        print("  step:  " + "  ".join(f"{i:>6d}" for i in sample_idxs))
        print("  V(s):  " + "  ".join(f"{values[i]:+6.1f}" for i in sample_idxs))
        print("  sep:   " + "  ".join(f"{seps[i]:6.2f}" for i in sample_idxs))
        print(f"  final V(s)={values[-1]:+.2f}   final separation={seps[-1]:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.model_path, args.seed)