# Fleet experiment: pilot_dfki_fleet_shipped_committed

2026-10-03 20:21 · 9600 episodes · 7.3 min

## What was tested

Two simulated vehicles must hold 2.0 m ± 0.2 m apart for 20 consecutive ticks (0.1 s each) within 200 ticks, without coming closer than 0.5 m. They share their state over an acoustic link that loses packets. Between beacons, each vehicle predicts where its teammate is with a **belief model**; that model is the only thing that changes between the conditions below. Both vehicles' true physics is `calibrated_parameters.yaml` in every condition, and every condition sees the same starting positions and the same lost packets.

Exact sensor readings: no sensor noise (`mode: noisy`, or a `sensor_noise:` block, adds realistic DVL, gyro, attitude, compass and depth noise).

No roll/pitch disturbance (set `mode: noisy` or `disturbance:` in the YAML to add one).

Still water: no current (set `current:` in the YAML for open-water conditions).

**Thruster limits on:** custom layout: 6 thrusters, curve T200 @ ~15 V (BlueROV2 dataset) rescaled to 13.8 V (+43.2 N / -33.8 N); max force 96/96/86 N, max torque 7.4/0.0/25.6 N*m (surge/sway/heave, roll/pitch/yaw); can't push: pitch torque. Every command is limited to what these thrusters can deliver; the beacons carry the delivered force.

**Caution: the policies were trained on a different vehicle.** fleet_policy_seed1, fleet_policy_seed2, fleet_policy_seed3 learned to hold formation on `calibrated_parameters_fleet_truth.yaml`, not on this experiment's vehicle (`calibrated_parameters.yaml`). Policies tuned to another vehicle hold formation less often, and that can blur or reverse the belief-model comparison below. For conclusions about your vehicle, train policies on it first: `python strathe.py train-fleet --params <repo>/calibration/results/pilot_dfki_tether/calibration/calibrated_parameters.yaml --out my_policy` (about 40 min; add the same `--thrusters` as here), then run this experiment with `--policies my_policy`.

## Headline: `calibrated` vs `generic`

- **L2@1s**: **no clear difference**. Success 63% with `calibrated` vs 61% with `generic` (+2 pts, 95% CI -1 pts to +5 pts; McNemar p = 0.27; 600 matched episodes). Difference in belief error -10.3 cm; collisions +0 pts.
- **L2@3s**: **generic did better**. Success 52% with `calibrated` vs 58% with `generic` (-6 pts, 95% CI -11 pts to -2 pts; McNemar p = 0.0037; 600 matched episodes). Difference in belief error -19.4 cm; collisions +0 pts.
- **L3a@1s**: **no clear difference**. Success 56% with `calibrated` vs 56% with `generic` (-0 pts, 95% CI -4 pts to +3 pts; McNemar p = 1; 600 matched episodes). Difference in belief error -16.0 cm; collisions -0 pts.
- **L3a@3s**: **calibrated did better**. Success 51% with `calibrated` vs 46% with `generic` (+5 pts, 95% CI +1 pts to +9 pts; McNemar p = 0.011; 600 matched episodes). Difference in belief error -28.7 cm; collisions -1 pts.

## All conditions

| Comm | Belief | Success (95% CI) | Collision | Mean belief error | Episodes |
|---|---|---|---|---|---|
| L2@1s | `calibrated` | 63% (59%–66%) | 0% | 9.8 cm | 600 |
| L2@1s | `generic` | 61% (57%–65%) | 0% | 20.1 cm | 600 |
| L2@1s | `last_beacon` | 58% (55%–62%) | 0% | 21.8 cm | 600 |
| L2@1s | `oracle` | 56% (52%–60%) | 0% | 0.0 cm | 600 |
| L2@3s | `calibrated` | 52% (48%–56%) | 2% | 38.6 cm | 600 |
| L2@3s | `generic` | 58% (55%–62%) | 2% | 58.0 cm | 600 |
| L2@3s | `last_beacon` | 59% (55%–63%) | 8% | 45.9 cm | 600 |
| L2@3s | `oracle` | 50% (46%–54%) | 0% | 0.0 cm | 600 |
| L3a@1s | `calibrated` | 56% (52%–60%) | 1% | 22.6 cm | 600 |
| L3a@1s | `generic` | 56% (52%–60%) | 1% | 38.6 cm | 600 |
| L3a@1s | `last_beacon` | 59% (55%–63%) | 3% | 36.3 cm | 600 |
| L3a@1s | `oracle` | 51% (47%–55%) | 0% | 0.0 cm | 600 |
| L3a@3s | `calibrated` | 51% (47%–55%) | 3% | 51.2 cm | 600 |
| L3a@3s | `generic` | 46% (42%–50%) | 4% | 79.9 cm | 600 |
| L3a@3s | `last_beacon` | 49% (45%–53%) | 6% | 56.3 cm | 600 |
| L3a@3s | `oracle` | 49% (45%–53%) | 0% | 0.0 cm | 600 |

## Belief error vs success across all belief models

Spearman correlation between a condition's mean belief error and its success rate (negative = lower error goes with more success; the oracle is excluded because its error is zero by construction).

- L2@1s: rho = -1.00 over 3 belief models (p = 0)
- L2@3s: rho = +0.50 over 3 belief models (p = 0.67)
- L3a@1s: rho = +0.50 over 3 belief models (p = 0.67)
- L3a@3s: rho = -1.00 over 3 belief models (p = 0)

## How to read this, and its limits

- Differences are **paired**: the same episode (start state, packet losses) is run under each belief model, so the comparison isn't blurred by luck of the draw. CIs are bootstrap intervals over episodes; McNemar's test counts episodes that one model won and the other lost.
- `calibrated` means a model whose parameters equal the simulated vehicle's true parameters. It shows the value of an accurate model; how close a real calibration gets to a real vehicle is measured separately, on held-out real logs (see README, calibration results).
- Simulation only. No hardware has been in the loop.
- Policies: fleet_policy_seed1, fleet_policy_seed2, fleet_policy_seed3. Results pool them; `episodes.csv` has the per-policy breakdown.
