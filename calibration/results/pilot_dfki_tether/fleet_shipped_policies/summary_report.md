# Fleet experiment: pilot_dfki_slack_shipped

2026-10-03 22:39 · 9600 episodes · 8.0 min

## What was tested

Two simulated vehicles must hold 2.0 m ± 0.2 m apart for 20 consecutive ticks (0.1 s each) within 200 ticks, without coming closer than 0.5 m. They share their state over an acoustic link that loses packets. Between beacons, each vehicle predicts where its teammate is with a **belief model**; that model is the only thing that changes between the conditions below. Both vehicles' true physics is `calibrated_parameters.yaml` in every condition, and every condition sees the same starting positions and the same lost packets.

Exact sensor readings: no sensor noise (`mode: noisy`, or a `sensor_noise:` block, adds realistic DVL, gyro, attitude, compass and depth noise).

No roll/pitch disturbance (set `mode: noisy` or `disturbance:` in the YAML to add one).

Still water: no current (set `current:` in the YAML for open-water conditions).

**Thruster limits on:** custom layout: 6 thrusters, curve T200 @ ~15 V (BlueROV2 dataset) rescaled to 13.8 V (+43.2 N / -33.8 N); max force 96/96/86 N, max torque 7.4/0.0/25.6 N*m (surge/sway/heave, roll/pitch/yaw); can't push: pitch torque. Every command is limited to what these thrusters can deliver; the beacons carry the delivered force.

**Caution: the policies were trained on a different vehicle.** fleet_policy_seed1, fleet_policy_seed2, fleet_policy_seed3 learned to hold formation on `calibrated_parameters_fleet_truth.yaml`, not on this experiment's vehicle (`calibrated_parameters.yaml`). Policies tuned to another vehicle hold formation less often, and that can blur or reverse the belief-model comparison below. For conclusions about your vehicle, train policies on it first: `python strathe.py train-fleet --params <calibration output>/calibrated_parameters.yaml --out my_policy` (about 40 min; add the same `--thrusters` as here), then run this experiment with `--policies my_policy`.

## Headline: `calibrated` vs `generic`

- **L2@1s**: **no clear difference**. Success 70% with `calibrated` vs 70% with `generic` (-0 pts, 95% CI -3 pts to +3 pts; McNemar p = 1; 600 matched episodes). Difference in belief error -9.4 cm; collisions -1 pts.
- **L2@3s**: **calibrated did better**. Success 61% with `calibrated` vs 53% with `generic` (+8 pts, 95% CI +4 pts to +13 pts; McNemar p = 0.00053; 600 matched episodes). Difference in belief error -19.4 cm; collisions -3 pts.
- **L3a@1s**: **calibrated did better**. Success 71% with `calibrated` vs 64% with `generic` (+7 pts, 95% CI +3 pts to +11 pts; McNemar p = 0.00028; 600 matched episodes). Difference in belief error -15.8 cm; collisions -2 pts.
- **L3a@3s**: **calibrated did better**. Success 53% with `calibrated` vs 47% with `generic` (+6 pts, 95% CI +3 pts to +10 pts; McNemar p = 0.0026; 600 matched episodes). Difference in belief error -31.1 cm; collisions +0 pts.

## All conditions

| Comm | Belief | Success (95% CI) | Collision | Mean belief error | Episodes |
|---|---|---|---|---|---|
| L2@1s | `calibrated` | 70% (66%–73%) | 0% | 10.5 cm | 600 |
| L2@1s | `generic` | 70% (66%–73%) | 1% | 19.9 cm | 600 |
| L2@1s | `last_beacon` | 70% (67%–74%) | 4% | 25.9 cm | 600 |
| L2@1s | `oracle` | 68% (64%–72%) | 0% | 0.0 cm | 600 |
| L2@3s | `calibrated` | 61% (57%–65%) | 4% | 41.0 cm | 600 |
| L2@3s | `generic` | 53% (49%–57%) | 8% | 60.3 cm | 600 |
| L2@3s | `last_beacon` | 50% (47%–54%) | 12% | 51.8 cm | 600 |
| L2@3s | `oracle` | 78% (75%–82%) | 0% | 0.0 cm | 600 |
| L3a@1s | `calibrated` | 71% (68%–75%) | 2% | 25.5 cm | 600 |
| L3a@1s | `generic` | 64% (60%–68%) | 4% | 41.3 cm | 600 |
| L3a@1s | `last_beacon` | 60% (56%–64%) | 7% | 42.5 cm | 600 |
| L3a@1s | `oracle` | 73% (70%–77%) | 0% | 0.0 cm | 600 |
| L3a@3s | `calibrated` | 53% (49%–57%) | 4% | 54.2 cm | 600 |
| L3a@3s | `generic` | 47% (43%–51%) | 4% | 85.3 cm | 600 |
| L3a@3s | `last_beacon` | 49% (45%–53%) | 8% | 61.5 cm | 600 |
| L3a@3s | `oracle` | 69% (65%–73%) | 0% | 0.0 cm | 600 |

## Belief error vs success across all belief models

Spearman correlation between a condition's mean belief error and its success rate (negative = lower error goes with more success; the oracle is excluded because its error is zero by construction).

- L2@1s: rho = +1.00 over 3 belief models (p = 0)
- L2@3s: rho = -0.50 over 3 belief models (p = 0.67)
- L3a@1s: rho = -1.00 over 3 belief models (p = 0)
- L3a@3s: rho = -1.00 over 3 belief models (p = 0)

## How to read this, and its limits

- Differences are **paired**: the same episode (start state, packet losses) is run under each belief model, so the comparison isn't blurred by luck of the draw. CIs are bootstrap intervals over episodes; McNemar's test counts episodes that one model won and the other lost.
- `calibrated` means a model whose parameters equal the simulated vehicle's true parameters. It shows the value of an accurate model; how close a real calibration gets to a real vehicle is measured separately, on held-out real logs (see README, calibration results).
- Simulation only. No hardware has been in the loop.
- Policies: fleet_policy_seed1, fleet_policy_seed2, fleet_policy_seed3. Results pool them; `episodes.csv` has the per-policy breakdown.
