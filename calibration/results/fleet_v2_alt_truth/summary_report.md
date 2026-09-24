# Fleet experiment: fleet_v2_alt_truth

2026-09-24 13:40 · 9600 episodes · 5.6 min · **ran with uncommitted code changes**

## What was tested

Two simulated BlueROV2s must hold 2.0 m ± 0.2 m apart for 20 consecutive ticks (0.1 s each) within 200 ticks, without coming closer than 0.5 m. They share their state over an acoustic link that loses packets. Between beacons, each vehicle predicts where its teammate is with a **belief model**; that model is the only thing that changes between the conditions below. Both vehicles' true physics is `calibrated_parameters_fleet_truth_alt.yaml` in every condition, and every condition sees the same starting positions and the same lost packets.

Clean simulation: no roll/pitch disturbance (set `disturbance:` in the YAML to add one).

## Headline: `calibrated` vs `generic`

- **L2@1s**: **no clear difference**. Success 98% with `calibrated` vs 99% with `generic` (-1 pts, 95% CI -3 pts to +0 pts; McNemar p = 0.14; 600 matched episodes). Difference in belief error -13.0 cm; collisions +0 pts.
- **L2@3s**: **calibrated did better**. Success 77% with `calibrated` vs 56% with `generic` (+21 pts, 95% CI +16 pts to +26 pts; McNemar p = 4.6e-16; 600 matched episodes). Difference in belief error -49.0 cm; collisions -5 pts.
- **L3a@1s**: **calibrated did better**. Success 94% with `calibrated` vs 82% with `generic` (+13 pts, 95% CI +9 pts to +16 pts; McNemar p = 5e-12; 600 matched episodes). Difference in belief error -32.8 cm; collisions -2 pts.
- **L3a@3s**: **calibrated did better**. Success 64% with `calibrated` vs 44% with `generic` (+19 pts, 95% CI +14 pts to +24 pts; McNemar p = 3e-13; 600 matched episodes). Difference in belief error -87.9 cm; collisions -5 pts.

## All conditions

| Comm | Belief | Success (95% CI) | Collision | Mean belief error | Episodes |
|---|---|---|---|---|---|
| L2@1s | `calibrated` | 98% (97%–99%) | 0% | 20.2 cm | 600 |
| L2@1s | `generic` | 99% (98%–100%) | 0% | 33.3 cm | 600 |
| L2@1s | `last_beacon` | 71% (67%–74%) | 0% | 32.9 cm | 600 |
| L2@1s | `oracle` | 84% (81%–87%) | 0% | 0.0 cm | 600 |
| L2@3s | `calibrated` | 77% (73%–80%) | 1% | 59.0 cm | 600 |
| L2@3s | `generic` | 56% (52%–60%) | 7% | 108.0 cm | 600 |
| L2@3s | `last_beacon` | 56% (52%–60%) | 4% | 68.2 cm | 600 |
| L2@3s | `oracle` | 76% (73%–80%) | 0% | 0.0 cm | 600 |
| L3a@1s | `calibrated` | 94% (92%–96%) | 1% | 37.4 cm | 600 |
| L3a@1s | `generic` | 82% (78%–84%) | 2% | 70.2 cm | 600 |
| L3a@1s | `last_beacon` | 69% (65%–72%) | 2% | 51.0 cm | 600 |
| L3a@1s | `oracle` | 80% (76%–83%) | 0% | 0.0 cm | 600 |
| L3a@3s | `calibrated` | 64% (60%–67%) | 2% | 63.3 cm | 600 |
| L3a@3s | `generic` | 44% (40%–48%) | 7% | 151.2 cm | 600 |
| L3a@3s | `last_beacon` | 42% (38%–46%) | 4% | 88.6 cm | 600 |
| L3a@3s | `oracle` | 73% (69%–77%) | 0% | 0.0 cm | 600 |

## Belief error vs success across all belief models

Spearman correlation between a condition's mean belief error and its success rate (negative = lower error goes with more success; the oracle is excluded because its error is zero by construction).

- L2@1s: rho = +0.50 over 3 belief models (p = 0.67)
- L2@3s: rho = -0.87 over 3 belief models (p = 0.33)
- L3a@1s: rho = -0.50 over 3 belief models (p = 0.67)
- L3a@3s: rho = -0.50 over 3 belief models (p = 0.67)

## How to read this, and its limits

- Differences are **paired**: the same episode (start state, packet losses) is run under each belief model, so the comparison isn't blurred by luck of the draw. CIs are bootstrap intervals over episodes; McNemar's test counts episodes that one model won and the other lost.
- `calibrated` means a model whose parameters equal the simulated vehicle's true parameters. It shows the value of an accurate model; how close a real calibration gets to a real vehicle is measured separately, on held-out real logs (see README, calibration results).
- Simulation only. No hardware has been in the loop.
- Policies: fleet_policy_seed1, fleet_policy_seed2, fleet_policy_seed3. Results pool them; `episodes.csv` has the per-policy breakdown.
