# Fleet experiment: fleet_v2_belief_to_success

2026-09-24 13:34 · 38400 episodes · 21.5 min · **ran with uncommitted code changes**

## What was tested

Two simulated BlueROV2s must hold 2.0 m ± 0.2 m apart for 20 consecutive ticks (0.1 s each) within 200 ticks, without coming closer than 0.5 m. They share their state over an acoustic link that loses packets. Between beacons, each vehicle predicts where its teammate is with a **belief model**; that model is the only thing that changes between the conditions below. Both vehicles' true physics is `calibrated_parameters_fleet_truth.yaml` in every condition, and every condition sees the same starting positions and the same lost packets.

Clean simulation: no roll/pitch disturbance (set `disturbance:` in the YAML to add one).

## Headline: `calibrated` vs `generic`

- **L0@1s**: **no clear difference**. Success 100% with `calibrated` vs 100% with `generic` (+0 pts, 95% CI +0 pts to +0 pts; McNemar p = 1; 600 matched episodes). Difference in belief error -10.7 cm; collisions +0 pts.
- **L0@3s**: **calibrated did better**. Success 97% with `calibrated` vs 76% with `generic` (+21 pts, 95% CI +17 pts to +25 pts; McNemar p = 1.4e-24; 600 matched episodes). Difference in belief error -37.4 cm; collisions +0 pts.
- **L1@1s**: **no clear difference**. Success 100% with `calibrated` vs 100% with `generic` (+0 pts, 95% CI +0 pts to +0 pts; McNemar p = 1; 600 matched episodes). Difference in belief error -12.9 cm; collisions +0 pts.
- **L1@3s**: **calibrated did better**. Success 94% with `calibrated` vs 70% with `generic` (+25 pts, 95% CI +20 pts to +29 pts; McNemar p = 4.2e-27; 600 matched episodes). Difference in belief error -44.4 cm; collisions -1 pts.
- **L2@1s**: **calibrated did better**. Success 100% with `calibrated` vs 98% with `generic` (+2 pts, 95% CI +1 pts to +2 pts; McNemar p = 0.0039; 600 matched episodes). Difference in belief error -19.4 cm; collisions -0 pts.
- **L2@3s**: **calibrated did better**. Success 86% with `calibrated` vs 64% with `generic` (+22 pts, 95% CI +17 pts to +26 pts; McNemar p = 6.2e-19; 600 matched episodes). Difference in belief error -60.3 cm; collisions -3 pts.
- **L3a@1s**: **calibrated did better**. Success 98% with `calibrated` vs 84% with `generic` (+13 pts, 95% CI +10 pts to +16 pts; McNemar p = 1.1e-17; 600 matched episodes). Difference in belief error -44.0 cm; collisions -1 pts.
- **L3a@3s**: **calibrated did better**. Success 68% with `calibrated` vs 52% with `generic` (+16 pts, 95% CI +12 pts to +21 pts; McNemar p = 3.9e-11; 600 matched episodes). Difference in belief error -97.1 cm; collisions -2 pts.

## All conditions

| Comm | Belief | Success (95% CI) | Collision | Mean belief error | Episodes |
|---|---|---|---|---|---|
| L0@1s | `calibrated` | 100% (99%–100%) | 0% | 7.6 cm | 600 |
| L0@1s | `constant_velocity` | 100% (99%–100%) | 0% | 7.2 cm | 600 |
| L0@1s | `generic` | 100% (99%–100%) | 0% | 18.3 cm | 600 |
| L0@1s | `last_beacon` | 100% (99%–100%) | 0% | 18.7 cm | 600 |
| L0@1s | `oracle` | 100% (99%–100%) | 0% | 0.0 cm | 600 |
| L0@1s | `physics_alpha_0.25` | 100% (99%–100%) | 0% | 9.1 cm | 600 |
| L0@1s | `physics_alpha_0.5` | 100% (99%–100%) | 0% | 11.4 cm | 600 |
| L0@1s | `physics_alpha_0.75` | 100% (99%–100%) | 0% | 14.8 cm | 600 |
| L0@3s | `calibrated` | 97% (95%–98%) | 0% | 47.2 cm | 600 |
| L0@3s | `constant_velocity` | 71% (67%–74%) | 0% | 49.8 cm | 600 |
| L0@3s | `generic` | 76% (72%–79%) | 0% | 84.6 cm | 600 |
| L0@3s | `last_beacon` | 98% (96%–98%) | 0% | 45.0 cm | 600 |
| L0@3s | `oracle` | 100% (99%–100%) | 0% | 0.0 cm | 600 |
| L0@3s | `physics_alpha_0.25` | 92% (89%–94%) | 0% | 51.1 cm | 600 |
| L0@3s | `physics_alpha_0.5` | 85% (82%–88%) | 0% | 57.0 cm | 600 |
| L0@3s | `physics_alpha_0.75` | 89% (86%–91%) | 2% | 63.2 cm | 600 |
| L1@1s | `calibrated` | 100% (99%–100%) | 0% | 9.9 cm | 600 |
| L1@1s | `constant_velocity` | 100% (99%–100%) | 0% | 9.4 cm | 600 |
| L1@1s | `generic` | 100% (99%–100%) | 0% | 22.8 cm | 600 |
| L1@1s | `last_beacon` | 100% (99%–100%) | 0% | 20.9 cm | 600 |
| L1@1s | `oracle` | 100% (99%–100%) | 0% | 0.0 cm | 600 |
| L1@1s | `physics_alpha_0.25` | 100% (99%–100%) | 0% | 11.6 cm | 600 |
| L1@1s | `physics_alpha_0.5` | 100% (99%–100%) | 0% | 14.3 cm | 600 |
| L1@1s | `physics_alpha_0.75` | 100% (99%–100%) | 0% | 18.2 cm | 600 |
| L1@3s | `calibrated` | 94% (92%–96%) | 0% | 48.7 cm | 600 |
| L1@3s | `constant_velocity` | 78% (75%–81%) | 0% | 50.6 cm | 600 |
| L1@3s | `generic` | 70% (66%–73%) | 1% | 93.1 cm | 600 |
| L1@3s | `last_beacon` | 94% (92%–96%) | 0% | 48.6 cm | 600 |
| L1@3s | `oracle` | 100% (99%–100%) | 0% | 0.0 cm | 600 |
| L1@3s | `physics_alpha_0.25` | 89% (86%–91%) | 0% | 52.9 cm | 600 |
| L1@3s | `physics_alpha_0.5` | 86% (83%–89%) | 0% | 59.3 cm | 600 |
| L1@3s | `physics_alpha_0.75` | 85% (82%–87%) | 1% | 67.5 cm | 600 |
| L2@1s | `calibrated` | 100% (99%–100%) | 0% | 16.1 cm | 600 |
| L2@1s | `constant_velocity` | 100% (99%–100%) | 0% | 15.4 cm | 600 |
| L2@1s | `generic` | 98% (97%–99%) | 0% | 35.4 cm | 600 |
| L2@1s | `last_beacon` | 100% (99%–100%) | 0% | 27.0 cm | 600 |
| L2@1s | `oracle` | 100% (99%–100%) | 0% | 0.0 cm | 600 |
| L2@1s | `physics_alpha_0.25` | 100% (99%–100%) | 0% | 18.3 cm | 600 |
| L2@1s | `physics_alpha_0.5` | 100% (99%–100%) | 0% | 21.8 cm | 600 |
| L2@1s | `physics_alpha_0.75` | 99% (98%–100%) | 0% | 27.2 cm | 600 |
| L2@3s | `calibrated` | 86% (82%–88%) | 1% | 52.4 cm | 600 |
| L2@3s | `constant_velocity` | 84% (81%–87%) | 2% | 53.5 cm | 600 |
| L2@3s | `generic` | 64% (60%–68%) | 4% | 112.7 cm | 600 |
| L2@3s | `last_beacon` | 84% (81%–87%) | 1% | 57.5 cm | 600 |
| L2@3s | `oracle` | 100% (99%–100%) | 0% | 0.0 cm | 600 |
| L2@3s | `physics_alpha_0.25` | 84% (80%–86%) | 1% | 57.0 cm | 600 |
| L2@3s | `physics_alpha_0.5` | 84% (81%–87%) | 1% | 64.7 cm | 600 |
| L2@3s | `physics_alpha_0.75` | 75% (71%–78%) | 3% | 78.6 cm | 600 |
| L3a@1s | `calibrated` | 98% (96%–98%) | 1% | 30.3 cm | 600 |
| L3a@1s | `constant_velocity` | 95% (93%–96%) | 0% | 30.1 cm | 600 |
| L3a@1s | `generic` | 84% (81%–87%) | 1% | 74.3 cm | 600 |
| L3a@1s | `last_beacon` | 96% (94%–97%) | 1% | 40.9 cm | 600 |
| L3a@1s | `oracle` | 100% (99%–100%) | 0% | 0.0 cm | 600 |
| L3a@1s | `physics_alpha_0.25` | 98% (96%–98%) | 1% | 34.0 cm | 600 |
| L3a@1s | `physics_alpha_0.5` | 97% (95%–98%) | 0% | 40.6 cm | 600 |
| L3a@1s | `physics_alpha_0.75` | 94% (92%–96%) | 0% | 51.0 cm | 600 |
| L3a@3s | `calibrated` | 68% (65%–72%) | 3% | 56.9 cm | 600 |
| L3a@3s | `constant_velocity` | 66% (62%–69%) | 4% | 66.4 cm | 600 |
| L3a@3s | `generic` | 52% (48%–56%) | 5% | 154.0 cm | 600 |
| L3a@3s | `last_beacon` | 65% (61%–69%) | 3% | 71.5 cm | 600 |
| L3a@3s | `oracle` | 93% (90%–94%) | 0% | 0.0 cm | 600 |
| L3a@3s | `physics_alpha_0.25` | 70% (66%–73%) | 5% | 61.7 cm | 600 |
| L3a@3s | `physics_alpha_0.5` | 69% (65%–73%) | 4% | 74.5 cm | 600 |
| L3a@3s | `physics_alpha_0.75` | 64% (60%–68%) | 4% | 99.3 cm | 600 |

## Does belief-model error drive success? (dose-response)

Belief models placed on a line from the true parameters (alpha = 0) to generic defaults (alpha = 1). A negative slope means success falls as the model gets less accurate.

| Comm | Success at alpha = 0, 0.25, 0.5, 0.75, 1 | Slope per unit alpha (95% CI) |
|---|---|---|
| L0@1s | 100%, 100%, 100%, 100%, 100% | -0 pts (-0 pts to -0 pts) |
| L0@3s | 97%, 92%, 85%, 89%, 76% | -18 pts (-21 pts to -14 pts) |
| L1@1s | 100%, 100%, 100%, 100%, 100% | -0 pts (-0 pts to -0 pts) |
| L1@3s | 94%, 89%, 86%, 85%, 70% | -21 pts (-25 pts to -18 pts) |
| L2@1s | 100%, 100%, 100%, 99%, 98% | -2 pts (-2 pts to -1 pts) |
| L2@3s | 86%, 84%, 84%, 75%, 64% | -21 pts (-25 pts to -17 pts) |
| L3a@1s | 98%, 98%, 97%, 94%, 84% | -12 pts (-15 pts to -9 pts) |
| L3a@3s | 68%, 70%, 69%, 64%, 52% | -15 pts (-20 pts to -11 pts) |

## Belief error vs success across all belief models

Spearman correlation between a condition's mean belief error and its success rate (negative = lower error goes with more success; the oracle is excluded because its error is zero by construction).

- L0@1s: not defined (every belief model had the same success rate or error)
- L0@3s: rho = -0.61 over 7 belief models (p = 0.15)
- L1@1s: rho = -0.61 over 7 belief models (p = 0.14)
- L1@3s: rho = -0.79 over 7 belief models (p = 0.036)
- L2@1s: rho = -0.80 over 7 belief models (p = 0.03)
- L2@3s: rho = -0.72 over 7 belief models (p = 0.068)
- L3a@1s: rho = -0.63 over 7 belief models (p = 0.13)
- L3a@3s: rho = -0.71 over 7 belief models (p = 0.071)

## How to read this, and its limits

- Differences are **paired**: the same episode (start state, packet losses) is run under each belief model, so the comparison isn't blurred by luck of the draw. CIs are bootstrap intervals over episodes; McNemar's test counts episodes that one model won and the other lost.
- `calibrated` means a model whose parameters equal the simulated vehicle's true parameters. It shows the value of an accurate model; how close a real calibration gets to a real vehicle is measured separately, on held-out real logs (see README, calibration results).
- Simulation only. No hardware has been in the loop.
- Policies: fleet_policy_seed1, fleet_policy_seed2, fleet_policy_seed3. Results pool them; `episodes.csv` has the per-policy breakdown.
