# Fleet experiment: fleet_v2_mujoco_truth

2026-09-24 13:52 · 3600 episodes · 11.9 min · **ran with uncommitted code changes**

## What was tested

Two simulated BlueROV2s must hold 2.0 m ± 0.2 m apart for 20 consecutive ticks (0.1 s each) within 200 ticks, without coming closer than 0.5 m. They share their state over an acoustic link that loses packets. Between beacons, each vehicle predicts where its teammate is with a **belief model**; that model is the only thing that changes between the conditions below. Both vehicles' true physics is `calibrated_parameters_fleet_truth.yaml` in every condition, and every condition sees the same starting positions and the same lost packets.

Clean simulation: no roll/pitch disturbance (set `disturbance:` in the YAML to add one).

## Headline: `calibrated` vs `generic`

- **L3a@1s**: **calibrated did better**. Success 98% with `calibrated` vs 83% with `generic` (+15 pts, 95% CI +12 pts to +18 pts; McNemar p = 1.1e-21; 600 matched episodes). Difference in belief error -44.9 cm; collisions -1 pts.
- **L3a@3s**: **calibrated did better**. Success 68% with `calibrated` vs 55% with `generic` (+14 pts, 95% CI +9 pts to +18 pts; McNemar p = 6.1e-08; 600 matched episodes). Difference in belief error -95.7 cm; collisions -3 pts.

## All conditions

| Comm | Belief | Success (95% CI) | Collision | Mean belief error | Episodes |
|---|---|---|---|---|---|
| L3a@1s | `calibrated` | 98% (97%–99%) | 0% | 30.2 cm | 600 |
| L3a@1s | `generic` | 83% (80%–86%) | 1% | 75.1 cm | 600 |
| L3a@1s | `last_beacon` | 97% (96%–98%) | 1% | 40.8 cm | 600 |
| L3a@3s | `calibrated` | 68% (64%–72%) | 3% | 56.6 cm | 600 |
| L3a@3s | `generic` | 55% (50%–58%) | 6% | 152.3 cm | 600 |
| L3a@3s | `last_beacon` | 66% (63%–70%) | 3% | 71.3 cm | 600 |

## Belief error vs success across all belief models

Spearman correlation between a condition's mean belief error and its success rate (negative = lower error goes with more success; the oracle is excluded because its error is zero by construction).

- L3a@1s: rho = -1.00 over 3 belief models (p = 0)
- L3a@3s: rho = -1.00 over 3 belief models (p = 0)

## How to read this, and its limits

- Differences are **paired**: the same episode (start state, packet losses) is run under each belief model, so the comparison isn't blurred by luck of the draw. CIs are bootstrap intervals over episodes; McNemar's test counts episodes that one model won and the other lost.
- `calibrated` means a model whose parameters equal the simulated vehicle's true parameters. It shows the value of an accurate model; how close a real calibration gets to a real vehicle is measured separately, on held-out real logs (see README, calibration results).
- Simulation only. No hardware has been in the loop.
- Policies: fleet_policy_seed1, fleet_policy_seed2, fleet_policy_seed3. Results pool them; `episodes.csv` has the per-policy breakdown.
