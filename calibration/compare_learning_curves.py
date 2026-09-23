"""
compare_learning_curves.py — reads two stable_baselines3 Monitor CSV logs
and reports a rolling-average reward comparison, so "did the extra
training actually help" is answered from real logged numbers, not by
re-reading two different console transcripts side by side.

Monitor's file format: line 1 is a JSON comment (starts with '#'), line 2
is the real header (r,l,t = reward, length, wall-clock time), then one
row per completed episode.
"""
import argparse

import numpy as np
import pandas as pd


def load_monitor_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, skiprows=1)


def rolling_mean(rewards: np.ndarray, window: int) -> np.ndarray:
    window = min(window, len(rewards)) or 1
    return pd.Series(rewards).rolling(window=window).mean().to_numpy()


def main(path_a: str, path_b: str, label_a: str, label_b: str, window: int) -> None:
    df_a = load_monitor_csv(path_a)
    df_b = load_monitor_csv(path_b)

    rewards_a = df_a["r"].to_numpy()
    rewards_b = df_b["r"].to_numpy()
    rolling_a = rolling_mean(rewards_a, window)
    rolling_b = rolling_mean(rewards_b, window)

    print(f"{label_a}: {len(df_a)} episodes logged, total timesteps {df_a['l'].sum():.0f}")
    print(f"{label_b}: {len(df_b)} episodes logged, total timesteps {df_b['l'].sum():.0f}")

    print(f"\n{label_a} — first 5 episode rewards: {np.round(rewards_a[:5], 1).tolist()}")
    print(f"{label_a} — last 5 episode rewards:  {np.round(rewards_a[-5:], 1).tolist()}")
    print(f"{label_a} — final rolling-{window} mean: {rolling_a[-1]:.2f}")

    print(f"\n{label_b} — first 5 episode rewards: {np.round(rewards_b[:5], 1).tolist()}")
    print(f"{label_b} — last 5 episode rewards:  {np.round(rewards_b[-5:], 1).tolist()}")
    print(f"{label_b} — final rolling-{window} mean: {rolling_b[-1]:.2f}")

    print(f"\nImprovement in final rolling-{window} mean, {label_a} -> {label_b}: "
          f"{rolling_b[-1] - rolling_a[-1]:+.2f}")

    # A crude but useful "is it still improving or has it plateaued" signal:
    # compare each run's own first-half vs second-half mean.
    for label, rewards in ((label_a, rewards_a), (label_b, rewards_b)):
        half = len(rewards) // 2
        if half == 0:
            continue
        first_half_mean = float(np.mean(rewards[:half]))
        second_half_mean = float(np.mean(rewards[half:]))
        print(f"\n{label} — first-half mean reward: {first_half_mean:.2f}, "
              f"second-half mean reward: {second_half_mean:.2f} "
              f"({'still improving' if second_half_mean > first_half_mean else 'flat or worse'} "
              f"within this run)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path-a", required=True)
    parser.add_argument("--path-b", required=True)
    parser.add_argument("--label-a", default="Run A")
    parser.add_argument("--label-b", default="Run B")
    parser.add_argument("--window", type=int, default=20,
                         help="Rolling-average window size, in episodes.")
    args = parser.parse_args()
    main(args.path_a, args.path_b, args.label_a, args.label_b, args.window)