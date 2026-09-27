# SAC and TD3 for the fleet: criteria fixed before any result was seen (2026-09-27)

Written while the first SAC/TD3 fleet policies were still training, before any
of them had been evaluated. The analysis reports against these criteria as
written; nothing here is changed after the results come in.

## What is trained

| | PPO (reference) | SAC | TD3 |
|---|---|---|---|
| Policies | the shipped `fleet_policy_seed1..3` | `algorithm_policies/fleet_sac_seed1..3` | `algorithm_policies/fleet_td3_seed1..3` |
| Steps per seed | 3,000,000 | 1,000,000 | 1,000,000 |
| Training mix | belief mix, L0–L3a, beacons 0.5–3 s, clean | same | same |
| Settings | `train_fleet_policy.ALGORITHM_SETTINGS["ppo"]` | `["sac"]` | `["td3"]` |

SAC and TD3 get a third of PPO's steps because each off-policy step costs far
more compute (gradient updates from a replay buffer). The wall-clock of each
run is recorded in its `.json`, so the comparison can be read either way.

## Gate 1: integration (every trained file)

1. `validate_fleet_policy.py` passes, with the algorithm read from the file.
2. Saved and reloaded, the policy gives identical actions on 1,000 recorded
   observations.
3. No NaN or infinite action on any evaluation episode.

## Gate 2: learning (pooled over the three seeds of each algorithm)

Evaluation grid: comm L2 and L3a, beacons every 1 s and 3 s, beliefs
`calibrated` and `last_beacon`, the standard 8 separations × 5 closing speeds
× 5 comm seeds (1,600 paired episodes per policy), stand-in physics, clean.

- **Floor:** random commands on the same grid.
- **Learned:** the algorithm closes at least 50% of the gap between the floor
  and PPO's success, pooled over the grid: success ≥ floor + 0.5 × (PPO − floor).
- **Parity with PPO:** the paired difference (algorithm − PPO, seeds averaged
  per episode) has a 95% CI that includes zero or lies above it, pooled over
  the grid.
- **Stability:** reported per seed; a seed below the floor + 25% of the gap is
  reported as a failed seed.

An algorithm is marked *validated for the fleet* if it passes Gate 1 and
"Learned". "Parity" is reported but not required: an algorithm can be usable
without matching PPO at a third of the steps.
