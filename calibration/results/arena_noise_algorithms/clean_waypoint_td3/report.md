# Single-vehicle experiment: one vehicle, clean: td3

2026-09-27 15:31 · 100 evaluation episodes per controller · 21.9 min

## What was tested

One simulated vehicle must reach a target point (within 0.5 m) that is up to 3 m away in each direction, within 200 steps (0.1 s each). **The vehicle** is your calibrated physics (`calibrated_parameters_fleet_truth.yaml`, `vehicle_spec.yaml`), in still water, with no thruster limits (any command up to ±30 N / N·m per axis), with no roll/pitch disturbance, with exact sensor readings. Physics engine for the tests: stand-in. Every controller below drives that same vehicle through the same episodes (same targets).

Controllers:

- **trained on your calibration**: TD3 trained in the simulator on your calibrated physics, with the same conditions as the test (2 training seed(s), 150,000 steps each)
- **hand-written controller**: pushes toward the target in proportion to the distance; no training
- **random commands**: a floor every controller should beat

## Results

| Controller | Engine | Success (95% CI) | Mean time to succeed | Mean final distance | Mean effort* |
|---|---|---|---|---|---|
| trained on your calibration | stand-in | 100% (98%–100%) | 5.4 s | 0.48 m | 628 |
| hand-written controller | stand-in | 65% (55%–74%) | 15.0 s | 0.56 m | 355 |
| random commands | stand-in | 0% (0%–4%) | n/a | 2.95 m | 1797 |

*Effort: the delivered forces (N) and torques (N·m), in absolute value, summed over the episode and multiplied by the 0.1 s step. It's a relative measure of how hard the thrusters work (lower = gentler), not a physical energy.

## What this does and doesn't show

- It is simulation only: the "vehicle" is your calibrated model. How well that model matches the real vehicle is what your calibration report measures; a hardware trial is the final test.
- Success rates are for these episodes and this training budget; a longer `--timesteps` usually helps harder tasks (station keeping, docking).
- Reproduce: `python run_arena_experiment.py --task waypoint --algorithm td3 --seeds 2 --timesteps 150000 --episodes 100 --workers 2 --name one vehicle, clean: td3 --out results/arena_noise_algorithms/clean_waypoint_td3`
