# Fleet experiment: fleet_v2_mujoco_truth

2026-09-23 01:40 · 3600 episodes · 22.1 min · **ran with uncommitted code changes**

## What was tested

Two simulated BlueROV2s must hold 2.0 m ± 0.2 m apart for 20 consecutive ticks (0.1 s each) within 200 ticks, without coming closer than 0.5 m. They share their state over an acoustic link that loses packets. Between beacons, each vehicle predicts where its teammate is with a **belief model**; that model is the only thing that changes between the conditions below. Both vehicles' true physics is `calibrated_parameters_fleet_truth.yaml` in every condition, and every condition sees the same starting positions and the same lost packets.

## Headline: `calibrated` vs `generic`

- **L3a@1s**: **calibrated did better**. Success 98% with `calibrated` vs 88% with `generic` (+9 pts, 95% CI +6 pts to +12 pts; McNemar p = 3.5e-10; 600 matched episodes). Difference in belief error -36.0 cm; collisions -1 pts.
- **L3a@3s**: **calibrated did better**. Success 66% with `calibrated` vs 48% with `generic` (+18 pts, 95% CI +14 pts to +23 pts; McNemar p = 5.6e-13; 600 matched episodes). Difference in belief error -95.9 cm; collisions -4 pts.

## All conditions

| Comm | Belief | Success (95% CI) | Collision | Mean belief error | Episodes |
|---|---|---|---|---|---|
| L3a@1s | `calibrated` | 98% (96%–98%) | 0% | 33.8 cm | 600 |
| L3a@1s | `generic` | 88% (86%–91%) | 1% | 69.7 cm | 600 |
| L3a@1s | `last_beacon` | 99% (97%–99%) | 0% | 40.5 cm | 600 |
| L3a@3s | `calibrated` | 66% (62%–69%) | 1% | 63.6 cm | 600 |
| L3a@3s | `generic` | 48% (44%–51%) | 5% | 159.6 cm | 600 |
| L3a@3s | `last_beacon` | 68% (64%–72%) | 0% | 71.8 cm | 600 |

## Belief error vs success across all belief models

Spearman correlation between a condition's mean belief error and its success rate (negative = lower error goes with more success; the oracle is excluded because its error is zero by construction).

- L3a@1s: rho = -0.50 over 3 belief models (p = 0.67)
- L3a@3s: rho = -0.50 over 3 belief models (p = 0.67)

## How to read this, and its limits

- Differences are **paired**: the same episode (start state, packet losses) is run under each belief model, so the comparison isn't blurred by luck of the draw. CIs are bootstrap intervals over episodes; McNemar's test counts episodes that one model won and the other lost.
- `calibrated` means a model whose parameters equal the simulated vehicle's true parameters. It shows the value of an accurate model; how close a real calibration gets to a real vehicle is measured separately, on held-out real logs (see README, calibration results).
- Simulation only. No hardware has been in the loop.
- Policies: fleet_policy_seed1, fleet_policy_seed2, fleet_policy_seed3. Results pool them; `episodes.csv` has the per-policy breakdown.
