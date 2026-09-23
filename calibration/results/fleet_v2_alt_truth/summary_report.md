# Fleet experiment: fleet_v2_alt_truth

2026-09-23 01:02 · 9600 episodes · 9.8 min · **ran with uncommitted code changes**

## What was tested

Two simulated BlueROV2s must hold 2.0 m ± 0.2 m apart for 20 consecutive ticks (0.1 s each) within 200 ticks, without coming closer than 0.5 m. They share their state over an acoustic link that loses packets. Between beacons, each vehicle predicts where its teammate is with a **belief model**; that model is the only thing that changes between the conditions below. Both vehicles' true physics is `calibrated_parameters_fleet_truth_alt.yaml` in every condition, and every condition sees the same starting positions and the same lost packets.

## Headline: `calibrated` vs `generic`

- **L2@1s**: **no clear difference**. Success 84% with `calibrated` vs 86% with `generic` (-1 pts, 95% CI -5 pts to +2 pts; McNemar p = 0.64; 600 matched episodes). Difference in belief error -12.6 cm; collisions -0 pts.
- **L2@3s**: **calibrated did better**. Success 74% with `calibrated` vs 55% with `generic` (+19 pts, 95% CI +14 pts to +24 pts; McNemar p = 5.8e-13; 600 matched episodes). Difference in belief error -40.0 cm; collisions -3 pts.
- **L3a@1s**: **calibrated did better**. Success 88% with `calibrated` vs 74% with `generic` (+14 pts, 95% CI +9 pts to +18 pts; McNemar p = 2.7e-10; 600 matched episodes). Difference in belief error -30.7 cm; collisions -2 pts.
- **L3a@3s**: **calibrated did better**. Success 58% with `calibrated` vs 40% with `generic` (+18 pts, 95% CI +14 pts to +23 pts; McNemar p = 2.1e-12; 600 matched episodes). Difference in belief error -73.3 cm; collisions -4 pts.

## All conditions

| Comm | Belief | Success (95% CI) | Collision | Mean belief error | Episodes |
|---|---|---|---|---|---|
| L2@1s | `calibrated` | 84% (81%–87%) | 0% | 21.8 cm | 600 |
| L2@1s | `generic` | 86% (82%–88%) | 0% | 34.4 cm | 600 |
| L2@1s | `last_beacon` | 96% (94%–97%) | 0% | 32.7 cm | 600 |
| L2@1s | `oracle` | 76% (72%–79%) | 0% | 0.0 cm | 600 |
| L2@3s | `calibrated` | 74% (71%–78%) | 2% | 61.9 cm | 600 |
| L2@3s | `generic` | 55% (51%–59%) | 5% | 101.9 cm | 600 |
| L2@3s | `last_beacon` | 80% (77%–83%) | 2% | 64.5 cm | 600 |
| L2@3s | `oracle` | 70% (66%–73%) | 0% | 0.0 cm | 600 |
| L3a@1s | `calibrated` | 88% (85%–90%) | 0% | 40.8 cm | 600 |
| L3a@1s | `generic` | 74% (71%–78%) | 3% | 71.5 cm | 600 |
| L3a@1s | `last_beacon` | 91% (89%–93%) | 0% | 49.5 cm | 600 |
| L3a@1s | `oracle` | 75% (72%–78%) | 0% | 0.0 cm | 600 |
| L3a@3s | `calibrated` | 58% (55%–62%) | 2% | 68.1 cm | 600 |
| L3a@3s | `generic` | 40% (36%–44%) | 6% | 141.4 cm | 600 |
| L3a@3s | `last_beacon` | 55% (51%–59%) | 2% | 73.6 cm | 600 |
| L3a@3s | `oracle` | 58% (54%–62%) | 0% | 0.0 cm | 600 |

## Belief error vs success across all belief models

Spearman correlation between a condition's mean belief error and its success rate (negative = lower error goes with more success; the oracle is excluded because its error is zero by construction).

- L2@1s: rho = +0.50 over 3 belief models (p = 0.67)
- L2@3s: rho = -0.50 over 3 belief models (p = 0.67)
- L3a@1s: rho = -0.50 over 3 belief models (p = 0.67)
- L3a@3s: rho = -1.00 over 3 belief models (p = 0)

## How to read this, and its limits

- Differences are **paired**: the same episode (start state, packet losses) is run under each belief model, so the comparison isn't blurred by luck of the draw. CIs are bootstrap intervals over episodes; McNemar's test counts episodes that one model won and the other lost.
- `calibrated` means a model whose parameters equal the simulated vehicle's true parameters. It shows the value of an accurate model; how close a real calibration gets to a real vehicle is measured separately, on held-out real logs (see README, calibration results).
- Simulation only. No hardware has been in the loop.
- Policies: fleet_policy_seed1, fleet_policy_seed2, fleet_policy_seed3. Results pool them; `episodes.csv` has the per-policy breakdown.
