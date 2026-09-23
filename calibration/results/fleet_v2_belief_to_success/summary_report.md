# Fleet experiment: fleet_v2_belief_to_success

2026-09-23 00:51 · 38400 episodes · 29.1 min · **ran with uncommitted code changes**

## What was tested

Two simulated BlueROV2s must hold 2.0 m ± 0.2 m apart for 20 consecutive ticks (0.1 s each) within 200 ticks, without coming closer than 0.5 m. They share their state over an acoustic link that loses packets. Between beacons, each vehicle predicts where its teammate is with a **belief model**; that model is the only thing that changes between the conditions below. Both vehicles' true physics is `calibrated_parameters_fleet_truth.yaml` in every condition, and every condition sees the same starting positions and the same lost packets.

## Headline: `calibrated` vs `generic`

- **L0@1s**: **calibrated did better**. Success 100% with `calibrated` vs 97% with `generic` (+3 pts, 95% CI +2 pts to +5 pts; McNemar p = 1.9e-06; 600 matched episodes). Difference in belief error -7.1 cm; collisions +0 pts.
- **L0@3s**: **no clear difference**. Success 90% with `calibrated` vs 88% with `generic` (+2 pts, 95% CI -1 pts to +6 pts; McNemar p = 0.19; 600 matched episodes). Difference in belief error -23.4 cm; collisions -1 pts.
- **L1@1s**: **calibrated did better**. Success 100% with `calibrated` vs 97% with `generic` (+3 pts, 95% CI +2 pts to +4 pts; McNemar p = 7.6e-06; 600 matched episodes). Difference in belief error -8.7 cm; collisions +0 pts.
- **L1@3s**: **calibrated did better**. Success 90% with `calibrated` vs 78% with `generic` (+12 pts, 95% CI +7 pts to +15 pts; McNemar p = 1.8e-08; 600 matched episodes). Difference in belief error -30.0 cm; collisions -0 pts.
- **L2@1s**: **calibrated did better**. Success 100% with `calibrated` vs 96% with `generic` (+4 pts, 95% CI +2 pts to +5 pts; McNemar p = 1.9e-05; 600 matched episodes). Difference in belief error -14.7 cm; collisions -0 pts.
- **L2@3s**: **calibrated did better**. Success 89% with `calibrated` vs 64% with `generic` (+25 pts, 95% CI +20 pts to +29 pts; McNemar p = 1.9e-26; 600 matched episodes). Difference in belief error -59.1 cm; collisions -2 pts.
- **L3a@1s**: **calibrated did better**. Success 97% with `calibrated` vs 86% with `generic` (+11 pts, 95% CI +8 pts to +14 pts; McNemar p = 6.8e-12; 600 matched episodes). Difference in belief error -35.1 cm; collisions -2 pts.
- **L3a@3s**: **calibrated did better**. Success 64% with `calibrated` vs 48% with `generic` (+15 pts, 95% CI +10 pts to +20 pts; McNemar p = 5.1e-09; 600 matched episodes). Difference in belief error -105.0 cm; collisions -4 pts.

## All conditions

| Comm | Belief | Success (95% CI) | Collision | Mean belief error | Episodes |
|---|---|---|---|---|---|
| L0@1s | `calibrated` | 100% (99%–100%) | 0% | 9.3 cm | 600 |
| L0@1s | `constant_velocity` | 100% (99%–100%) | 0% | 7.9 cm | 600 |
| L0@1s | `generic` | 97% (95%–98%) | 0% | 16.4 cm | 600 |
| L0@1s | `last_beacon` | 100% (99%–100%) | 0% | 18.6 cm | 600 |
| L0@1s | `oracle` | 100% (99%–100%) | 0% | 0.0 cm | 600 |
| L0@1s | `physics_alpha_0.25` | 98% (97%–99%) | 0% | 10.4 cm | 600 |
| L0@1s | `physics_alpha_0.5` | 99% (98%–100%) | 0% | 12.3 cm | 600 |
| L0@1s | `physics_alpha_0.75` | 98% (97%–99%) | 0% | 14.8 cm | 600 |
| L0@3s | `calibrated` | 90% (87%–92%) | 0% | 45.5 cm | 600 |
| L0@3s | `constant_velocity` | 90% (87%–92%) | 0% | 49.6 cm | 600 |
| L0@3s | `generic` | 88% (85%–90%) | 1% | 69.0 cm | 600 |
| L0@3s | `last_beacon` | 100% (99%–100%) | 0% | 45.7 cm | 600 |
| L0@3s | `oracle` | 84% (81%–87%) | 0% | 0.0 cm | 600 |
| L0@3s | `physics_alpha_0.25` | 83% (80%–86%) | 0% | 48.0 cm | 600 |
| L0@3s | `physics_alpha_0.5` | 77% (73%–80%) | 0% | 50.9 cm | 600 |
| L0@3s | `physics_alpha_0.75` | 78% (75%–81%) | 1% | 55.5 cm | 600 |
| L1@1s | `calibrated` | 100% (99%–100%) | 0% | 11.8 cm | 600 |
| L1@1s | `constant_velocity` | 98% (96%–99%) | 0% | 10.1 cm | 600 |
| L1@1s | `generic` | 97% (95%–98%) | 0% | 20.6 cm | 600 |
| L1@1s | `last_beacon` | 100% (99%–100%) | 0% | 20.8 cm | 600 |
| L1@1s | `oracle` | 98% (97%–99%) | 0% | 0.0 cm | 600 |
| L1@1s | `physics_alpha_0.25` | 100% (99%–100%) | 0% | 13.1 cm | 600 |
| L1@1s | `physics_alpha_0.5` | 99% (98%–100%) | 0% | 15.1 cm | 600 |
| L1@1s | `physics_alpha_0.75` | 98% (97%–99%) | 0% | 17.7 cm | 600 |
| L1@3s | `calibrated` | 90% (87%–92%) | 0% | 48.0 cm | 600 |
| L1@3s | `constant_velocity` | 86% (83%–89%) | 0% | 52.9 cm | 600 |
| L1@3s | `generic` | 78% (75%–81%) | 0% | 78.0 cm | 600 |
| L1@3s | `last_beacon` | 99% (98%–100%) | 0% | 49.2 cm | 600 |
| L1@3s | `oracle` | 84% (81%–87%) | 0% | 0.0 cm | 600 |
| L1@3s | `physics_alpha_0.25` | 87% (84%–90%) | 0% | 50.6 cm | 600 |
| L1@3s | `physics_alpha_0.5` | 85% (82%–87%) | 0% | 54.0 cm | 600 |
| L1@3s | `physics_alpha_0.75` | 84% (81%–87%) | 1% | 59.8 cm | 600 |
| L2@1s | `calibrated` | 100% (99%–100%) | 0% | 18.8 cm | 600 |
| L2@1s | `constant_velocity` | 97% (95%–98%) | 0% | 16.8 cm | 600 |
| L2@1s | `generic` | 96% (94%–97%) | 0% | 33.6 cm | 600 |
| L2@1s | `last_beacon` | 100% (99%–100%) | 0% | 27.1 cm | 600 |
| L2@1s | `oracle` | 93% (90%–94%) | 0% | 0.0 cm | 600 |
| L2@1s | `physics_alpha_0.25` | 100% (99%–100%) | 0% | 20.4 cm | 600 |
| L2@1s | `physics_alpha_0.5` | 100% (99%–100%) | 0% | 22.8 cm | 600 |
| L2@1s | `physics_alpha_0.75` | 98% (97%–99%) | 0% | 26.5 cm | 600 |
| L2@3s | `calibrated` | 89% (86%–91%) | 0% | 52.7 cm | 600 |
| L2@3s | `constant_velocity` | 80% (77%–83%) | 1% | 67.6 cm | 600 |
| L2@3s | `generic` | 64% (60%–68%) | 2% | 111.8 cm | 600 |
| L2@3s | `last_beacon` | 90% (88%–93%) | 0% | 57.7 cm | 600 |
| L2@3s | `oracle` | 82% (78%–85%) | 0% | 0.0 cm | 600 |
| L2@3s | `physics_alpha_0.25` | 88% (85%–90%) | 0% | 56.2 cm | 600 |
| L2@3s | `physics_alpha_0.5` | 82% (78%–84%) | 1% | 63.0 cm | 600 |
| L2@3s | `physics_alpha_0.75` | 80% (76%–83%) | 0% | 72.0 cm | 600 |
| L3a@1s | `calibrated` | 97% (95%–98%) | 0% | 33.9 cm | 600 |
| L3a@1s | `constant_velocity` | 94% (92%–96%) | 0% | 38.1 cm | 600 |
| L3a@1s | `generic` | 86% (83%–88%) | 2% | 69.0 cm | 600 |
| L3a@1s | `last_beacon` | 99% (97%–99%) | 0% | 40.7 cm | 600 |
| L3a@1s | `oracle` | 86% (83%–89%) | 0% | 0.0 cm | 600 |
| L3a@1s | `physics_alpha_0.25` | 97% (95%–98%) | 0% | 36.3 cm | 600 |
| L3a@1s | `physics_alpha_0.5` | 96% (94%–97%) | 0% | 40.4 cm | 600 |
| L3a@1s | `physics_alpha_0.75` | 94% (92%–95%) | 0% | 48.3 cm | 600 |
| L3a@3s | `calibrated` | 64% (60%–68%) | 0% | 64.1 cm | 600 |
| L3a@3s | `constant_velocity` | 52% (48%–56%) | 3% | 108.0 cm | 600 |
| L3a@3s | `generic` | 48% (45%–52%) | 4% | 169.2 cm | 600 |
| L3a@3s | `last_beacon` | 67% (63%–71%) | 1% | 70.6 cm | 600 |
| L3a@3s | `oracle` | 69% (65%–72%) | 0% | 0.0 cm | 600 |
| L3a@3s | `physics_alpha_0.25` | 63% (59%–67%) | 0% | 69.8 cm | 600 |
| L3a@3s | `physics_alpha_0.5` | 64% (60%–68%) | 1% | 78.6 cm | 600 |
| L3a@3s | `physics_alpha_0.75` | 61% (57%–64%) | 2% | 95.9 cm | 600 |

## Does belief-model error drive success? (dose-response)

Belief models placed on a line from the true parameters (alpha = 0) to generic defaults (alpha = 1). A negative slope means success falls as the model gets less accurate.

| Comm | Success at alpha = 0, 0.25, 0.5, 0.75, 1 | Slope per unit alpha (95% CI) |
|---|---|---|
| L0@1s | 100%, 98%, 99%, 98%, 97% | -3 pts (-4 pts to -1 pts) |
| L0@3s | 90%, 83%, 77%, 78%, 88% | -4 pts (-7 pts to -1 pts) |
| L1@1s | 100%, 100%, 99%, 98%, 97% | -3 pts (-4 pts to -2 pts) |
| L1@3s | 90%, 87%, 85%, 84%, 78% | -10 pts (-14 pts to -7 pts) |
| L2@1s | 100%, 100%, 100%, 98%, 96% | -3 pts (-5 pts to -2 pts) |
| L2@3s | 89%, 88%, 82%, 80%, 64% | -23 pts (-27 pts to -19 pts) |
| L3a@1s | 97%, 97%, 96%, 94%, 86% | -10 pts (-13 pts to -7 pts) |
| L3a@3s | 64%, 63%, 64%, 61%, 48% | -13 pts (-18 pts to -9 pts) |

## Belief error vs success across all belief models

Spearman correlation between a condition's mean belief error and its success rate (negative = lower error goes with more success; the oracle is excluded because its error is zero by construction).

- L0@1s: rho = -0.37 over 7 belief models (p = 0.41)
- L0@3s: rho = -0.59 over 7 belief models (p = 0.16)
- L1@1s: rho = +0.02 over 7 belief models (p = 0.97)
- L1@3s: rho = -0.96 over 7 belief models (p = 0.00045)
- L2@1s: rho = -0.04 over 7 belief models (p = 0.94)
- L2@3s: rho = -0.89 over 7 belief models (p = 0.0068)
- L3a@1s: rho = -0.57 over 7 belief models (p = 0.18)
- L3a@3s: rho = -0.71 over 7 belief models (p = 0.071)

## How to read this, and its limits

- Differences are **paired**: the same episode (start state, packet losses) is run under each belief model, so the comparison isn't blurred by luck of the draw. CIs are bootstrap intervals over episodes; McNemar's test counts episodes that one model won and the other lost.
- `calibrated` means a model whose parameters equal the simulated vehicle's true parameters. It shows the value of an accurate model; how close a real calibration gets to a real vehicle is measured separately, on held-out real logs (see README, calibration results).
- Simulation only. No hardware has been in the loop.
- Policies: fleet_policy_seed1, fleet_policy_seed2, fleet_policy_seed3. Results pool them; `episodes.csv` has the per-policy breakdown.
