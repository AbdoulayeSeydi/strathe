# Single-vehicle experiment: ppo_seed1

2026-09-27 15:33 · 100 evaluation episodes per controller · 2.0 min

## What was tested

One simulated vehicle must reach a target point (within 0.5 m) that is up to 3 m away in each direction, within 200 steps (0.1 s each). **The vehicle** is your calibrated physics (`calibrated_parameters_fleet_truth.yaml`, `vehicle_spec.yaml`), in still water, with no thruster limits (any command up to ±30 N / N·m per axis), pushed by random roll/pitch torques (`bluerov2_rollpitch.yaml` x1: roll 0.92 N·m, pitch 1.08 N·m typical size), with sensor noise (`typical_dvl_ahrs.yaml` x1: DVL ±1 cm/s (bias ±0.5, dropouts 2%), gyro ±0.57 deg/s (bias ±0.17), roll/pitch ±0.5 deg (bias ±0.5), heading ±2 deg (bias ±2), depth ±1 cm, position dead-reckoned from the DVL and compass (start fix ±0.2 m)): the controllers see what the vehicle's own sensors report and where it believes it is, and success is judged on the true position. Physics engine for the tests: stand-in. Every controller below drives that same vehicle through the same episodes (same targets, same disturbances, same sensor noise).

Controllers:

- **trained on your calibration**: PPO trained in the simulator on your calibrated physics, with the same conditions as the test (2 training seed(s), 150,000 steps each)
- **hand-written controller**: pushes toward the target in proportion to the distance; no training
- **random commands**: a floor every controller should beat

## Results

| Controller | Engine | Success (95% CI) | Mean time to succeed | Mean final distance | Mean effort* | Mean navigation error† |
|---|---|---|---|---|---|---|
| your policy | stand-in | 97% (92%–99%) | 6.5 s | 0.49 m | 595 | 26 cm |
| hand-written controller | stand-in | 54% (44%–63%) | 14.4 s | 0.59 m | 354 | 26 cm |
| random commands | stand-in | 0% (0%–4%) | n/a | 2.95 m | 1797 | 26 cm |

*Effort: the delivered forces (N) and torques (N·m), in absolute value, summed over the episode and multiplied by the 0.1 s step. It's a relative measure of how hard the thrusters work (lower = gentler), not a physical energy.

†Navigation error: how far the vehicle's believed position was from its true position, averaged over the episode. Final distance and success use the true position.

## What this does and doesn't show

- It is simulation only: the "vehicle" is your calibrated model. How well that model matches the real vehicle is what your calibration report measures; a hardware trial is the final test.
- Success rates are for these episodes and this training budget; a longer `--timesteps` usually helps harder tasks (station keeping, docking).
- Reproduce: `python run_arena_experiment.py --task waypoint --policy results/arena_noise_algorithms/clean_waypoint_ppo/policies/ppo_calibrated_seed1.zip --mode noisy --episodes 100 --workers 1 --out results/arena_noise_algorithms/algorithms_under_noise/ppo_seed1`
