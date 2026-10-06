# Fleet experiment: pilot_dfki_fleet_own

2026-10-03 15:10 · 3200 episodes · 1.2 min

## What was tested

Two simulated vehicles must hold 2.0 m ± 0.2 m apart for 20 consecutive ticks (0.1 s each) within 200 ticks, without coming closer than 0.5 m. They share their state over an acoustic link that loses packets. Between beacons, each vehicle predicts where its teammate is with a **belief model**; that model is the only thing that changes between the conditions below. Both vehicles' true physics is `calibrated_parameters.yaml` in every condition, and every condition sees the same starting positions and the same lost packets.

Exact sensor readings: no sensor noise (`mode: noisy`, or a `sensor_noise:` block, adds realistic DVL, gyro, attitude, compass and depth noise).

No roll/pitch disturbance (set `mode: noisy` or `disturbance:` in the YAML to add one).

Still water: no current (set `current:` in the YAML for open-water conditions).

**Thruster limits on:** custom layout: 6 thrusters, curve T200 @ ~15 V (BlueROV2 dataset) rescaled to 13.8 V (+43.2 N / -33.8 N); max force 96/96/86 N, max torque 7.4/0.0/25.6 N*m (surge/sway/heave, roll/pitch/yaw); can't push: pitch torque. Every command is limited to what these thrusters can deliver; the beacons carry the delivered force.

## Headline: `calibrated` vs `generic`

- **L2@1s**: **no clear difference**. Success 100% with `calibrated` vs 98% with `generic` (+2 pts, 95% CI +0 pts to +3 pts; McNemar p = 0.25; 200 matched episodes). Difference in belief error -13.1 cm; collisions +0 pts.
- **L2@3s**: **calibrated did better**. Success 66% with `calibrated` vs 55% with `generic` (+12 pts, 95% CI +3 pts to +20 pts; McNemar p = 0.0095; 200 matched episodes). Difference in belief error -27.5 cm; collisions +1 pts.
- **L3a@1s**: **calibrated did better**. Success 94% with `calibrated` vs 82% with `generic` (+12 pts, 95% CI +6 pts to +17 pts; McNemar p = 0.00012; 200 matched episodes). Difference in belief error -19.8 cm; collisions +0 pts.
- **L3a@3s**: **calibrated did better**. Success 57% with `calibrated` vs 46% with `generic` (+12 pts, 95% CI +3 pts to +20 pts; McNemar p = 0.0086; 200 matched episodes). Difference in belief error -36.1 cm; collisions -2 pts.

## All conditions

| Comm | Belief | Success (95% CI) | Collision | Mean belief error | Episodes |
|---|---|---|---|---|---|
| L2@1s | `calibrated` | 100% (98%–100%) | 0% | 10.9 cm | 200 |
| L2@1s | `generic` | 98% (96%–99%) | 0% | 24.0 cm | 200 |
| L2@1s | `last_beacon` | 94% (90%–97%) | 0% | 28.8 cm | 200 |
| L2@1s | `oracle` | 100% (98%–100%) | 0% | 0.0 cm | 200 |
| L2@3s | `calibrated` | 66% (59%–72%) | 2% | 41.4 cm | 200 |
| L2@3s | `generic` | 55% (48%–61%) | 1% | 69.0 cm | 200 |
| L2@3s | `last_beacon` | 56% (49%–63%) | 2% | 62.2 cm | 200 |
| L2@3s | `oracle` | 90% (84%–93%) | 0% | 0.0 cm | 200 |
| L3a@1s | `calibrated` | 94% (90%–97%) | 0% | 23.0 cm | 200 |
| L3a@1s | `generic` | 82% (77%–87%) | 0% | 42.8 cm | 200 |
| L3a@1s | `last_beacon` | 70% (64%–76%) | 2% | 47.2 cm | 200 |
| L3a@1s | `oracle` | 100% (98%–100%) | 0% | 0.0 cm | 200 |
| L3a@3s | `calibrated` | 57% (51%–64%) | 2% | 50.0 cm | 200 |
| L3a@3s | `generic` | 46% (39%–53%) | 4% | 86.0 cm | 200 |
| L3a@3s | `last_beacon` | 46% (39%–53%) | 3% | 71.7 cm | 200 |
| L3a@3s | `oracle` | 78% (71%–83%) | 0% | 0.0 cm | 200 |

## Belief error vs success across all belief models

Spearman correlation between a condition's mean belief error and its success rate (negative = lower error goes with more success; the oracle is excluded because its error is zero by construction).

- L2@1s: rho = -1.00 over 3 belief models (p = 0)
- L2@3s: rho = -1.00 over 3 belief models (p = 0)
- L3a@1s: rho = -1.00 over 3 belief models (p = 0)
- L3a@3s: rho = -0.87 over 3 belief models (p = 0.33)

## How to read this, and its limits

- Differences are **paired**: the same episode (start state, packet losses) is run under each belief model, so the comparison isn't blurred by luck of the draw. CIs are bootstrap intervals over episodes; McNemar's test counts episodes that one model won and the other lost.
- `calibrated` means a model whose parameters equal the simulated vehicle's true parameters. It shows the value of an accurate model; how close a real calibration gets to a real vehicle is measured separately, on held-out real logs (see README, calibration results).
- Simulation only. No hardware has been in the loop.
- Policies: dfki_fleet_policy. Results pool them; `episodes.csv` has the per-policy breakdown.
