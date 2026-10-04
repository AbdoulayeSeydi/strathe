# Fleet experiment: pilot_dfki_slack_own

2026-10-03 22:42 · 3200 episodes · 2.4 min

## What was tested

Two simulated vehicles must hold 2.0 m ± 0.2 m apart for 20 consecutive ticks (0.1 s each) within 200 ticks, without coming closer than 0.5 m. They share their state over an acoustic link that loses packets. Between beacons, each vehicle predicts where its teammate is with a **belief model**; that model is the only thing that changes between the conditions below. Both vehicles' true physics is `calibrated_parameters.yaml` in every condition, and every condition sees the same starting positions and the same lost packets.

Exact sensor readings: no sensor noise (`mode: noisy`, or a `sensor_noise:` block, adds realistic DVL, gyro, attitude, compass and depth noise).

No roll/pitch disturbance (set `mode: noisy` or `disturbance:` in the YAML to add one).

Still water: no current (set `current:` in the YAML for open-water conditions).

**Thruster limits on:** custom layout: 6 thrusters, curve T200 @ ~15 V (BlueROV2 dataset) rescaled to 13.8 V (+43.2 N / -33.8 N); max force 96/96/86 N, max torque 7.4/0.0/25.6 N*m (surge/sway/heave, roll/pitch/yaw); can't push: pitch torque. Every command is limited to what these thrusters can deliver; the beacons carry the delivered force.

## Headline: `calibrated` vs `generic`

- **L2@1s**: **no clear difference**. Success 100% with `calibrated` vs 100% with `generic` (+0 pts, 95% CI +0 pts to +2 pts; McNemar p = 1; 200 matched episodes). Difference in belief error -18.8 cm; collisions +0 pts.
- **L2@3s**: **calibrated did better**. Success 82% with `calibrated` vs 70% with `generic` (+12 pts, 95% CI +5 pts to +20 pts; McNemar p = 0.0022; 200 matched episodes). Difference in belief error -29.1 cm; collisions -1 pts.
- **L3a@1s**: **calibrated did better**. Success 97% with `calibrated` vs 86% with `generic` (+10 pts, 95% CI +6 pts to +15 pts; McNemar p = 5.7e-06; 200 matched episodes). Difference in belief error -26.9 cm; collisions -1 pts.
- **L3a@3s**: **calibrated did better**. Success 64% with `calibrated` vs 42% with `generic` (+22 pts, 95% CI +14 pts to +29 pts; McNemar p = 6e-08; 200 matched episodes). Difference in belief error -55.0 cm; collisions -0 pts.

## All conditions

| Comm | Belief | Success (95% CI) | Collision | Mean belief error | Episodes |
|---|---|---|---|---|---|
| L2@1s | `calibrated` | 100% (98%–100%) | 0% | 13.6 cm | 200 |
| L2@1s | `generic` | 100% (97%–100%) | 0% | 32.5 cm | 200 |
| L2@1s | `last_beacon` | 62% (56%–69%) | 0% | 38.5 cm | 200 |
| L2@1s | `oracle` | 100% (98%–100%) | 0% | 0.0 cm | 200 |
| L2@3s | `calibrated` | 82% (76%–87%) | 0% | 50.1 cm | 200 |
| L2@3s | `generic` | 70% (63%–75%) | 2% | 79.2 cm | 200 |
| L2@3s | `last_beacon` | 69% (62%–75%) | 2% | 65.3 cm | 200 |
| L2@3s | `oracle` | 98% (96%–99%) | 0% | 0.0 cm | 200 |
| L3a@1s | `calibrated` | 97% (94%–99%) | 0% | 27.6 cm | 200 |
| L3a@1s | `generic` | 86% (81%–91%) | 2% | 54.6 cm | 200 |
| L3a@1s | `last_beacon` | 66% (60%–73%) | 4% | 58.7 cm | 200 |
| L3a@1s | `oracle` | 100% (97%–100%) | 0% | 0.0 cm | 200 |
| L3a@3s | `calibrated` | 64% (57%–70%) | 1% | 65.4 cm | 200 |
| L3a@3s | `generic` | 42% (35%–49%) | 2% | 120.5 cm | 200 |
| L3a@3s | `last_beacon` | 50% (43%–56%) | 1% | 80.7 cm | 200 |
| L3a@3s | `oracle` | 78% (72%–83%) | 0% | 0.0 cm | 200 |

## Belief error vs success across all belief models

Spearman correlation between a condition's mean belief error and its success rate (negative = lower error goes with more success; the oracle is excluded because its error is zero by construction).

- L2@1s: rho = -1.00 over 3 belief models (p = 0)
- L2@3s: rho = -0.50 over 3 belief models (p = 0.67)
- L3a@1s: rho = -1.00 over 3 belief models (p = 0)
- L3a@3s: rho = -1.00 over 3 belief models (p = 0)

## How to read this, and its limits

- Differences are **paired**: the same episode (start state, packet losses) is run under each belief model, so the comparison isn't blurred by luck of the draw. CIs are bootstrap intervals over episodes; McNemar's test counts episodes that one model won and the other lost.
- `calibrated` means a model whose parameters equal the simulated vehicle's true parameters. It shows the value of an accurate model; how close a real calibration gets to a real vehicle is measured separately, on held-out real logs (see README, calibration results).
- Simulation only. No hardware has been in the loop.
- Policies: fleet_policy. Results pool them; `episodes.csv` has the per-policy breakdown.
