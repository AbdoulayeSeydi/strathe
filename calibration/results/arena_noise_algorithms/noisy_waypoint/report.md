# Single-vehicle experiment: one vehicle, noisy: trained with vs without the noise

2026-09-27 14:31 · 100 evaluation episodes per controller · 50.1 min

## What was tested

One simulated vehicle must reach a target point (within 0.5 m) that is up to 3 m away in each direction, within 200 steps (0.1 s each). **The vehicle** is your calibrated physics (`calibrated_parameters_fleet_truth.yaml`, `vehicle_spec.yaml`), in still water, with no thruster limits (any command up to ±30 N / N·m per axis), pushed by random roll/pitch torques (`bluerov2_rollpitch.yaml` x1: roll 0.92 N·m, pitch 1.08 N·m typical size), with sensor noise (`typical_dvl_ahrs.yaml` x1: DVL ±1 cm/s (bias ±0.5, dropouts 2%), gyro ±0.57 deg/s (bias ±0.17), roll/pitch ±0.5 deg (bias ±0.5), heading ±2 deg (bias ±2), depth ±1 cm, position dead-reckoned from the DVL and compass (start fix ±0.2 m)): the controllers see what the vehicle's own sensors report and where it believes it is, and success is judged on the true position. Physics engine for the tests: stand-in and MuJoCo (policies are trained on the stand-in, then tested on each). Every controller below drives that same vehicle through the same episodes (same targets, same disturbances, same sensor noise).

Controllers:

- **trained on your calibration**: PPO trained in the simulator on your calibrated physics, with the same conditions as the test (2 training seed(s), 300,000 steps each)
- **trained clean**: the same training on your calibration, but with exact sensors and no roll/pitch disturbance, then tested on the same noisy vehicle
- **hand-written controller**: pushes toward the target in proportion to the distance; no training
- **random commands**: a floor every controller should beat

## Results

| Controller | Engine | Success (95% CI) | Mean time to succeed | Mean final distance | Mean effort* | Mean navigation error† |
|---|---|---|---|---|---|---|
| trained on your calibration | stand-in | 97% (94%–99%) | 5.8 s | 0.49 m | 650 | 26 cm |
| trained clean | stand-in | 99% (96%–100%) | 5.6 s | 0.48 m | 848 | 26 cm |
| hand-written controller | stand-in | 54% (44%–63%) | 14.4 s | 0.59 m | 354 | 26 cm |
| random commands | stand-in | 0% (0%–4%) | n/a | 2.95 m | 1797 | 26 cm |
| trained on your calibration | MuJoCo | 98% (94%–99%) | 5.8 s | 0.49 m | 646 | 26 cm |
| trained clean | MuJoCo | 98% (96%–99%) | 5.5 s | 0.48 m | 850 | 26 cm |
| hand-written controller | MuJoCo | 54% (44%–63%) | 14.4 s | 0.59 m | 354 | 26 cm |
| random commands | MuJoCo | 0% (0%–4%) | n/a | 2.95 m | 1797 | 26 cm |

The two engines agree to within 1 points of success on every controller.

*Effort: the delivered forces (N) and torques (N·m), in absolute value, summed over the episode and multiplied by the 0.1 s step. It's a relative measure of how hard the thrusters work (lower = gentler), not a physical energy.

†Navigation error: how far the vehicle's believed position was from its true position, averaged over the episode. Final distance and success use the true position.

## Does training with the noise pay off? (stand-in)

On your noisy test, policies trained with the noise succeed -2 points (-4 to -0, 95% CI; 200 paired episodes) compared with the same training on a clean simulation. That difference is clear of zero.

## Does training with the noise pay off? (MuJoCo)

On your noisy test, policies trained with the noise succeed -1 points (-2 to +0, 95% CI; 200 paired episodes) compared with the same training on a clean simulation. That interval includes zero: here, training clean did about as well.

## What this does and doesn't show

- It is simulation only: the "vehicle" is your calibrated model. How well that model matches the real vehicle is what your calibration report measures; a hardware trial is the final test.
- Success rates are for these episodes and this training budget; a longer `--timesteps` usually helps harder tasks (station keeping, docking).
- Reproduce: `python run_arena_experiment.py --task waypoint --mode noisy --compare-clean-training --seeds 2 --timesteps 300000 --episodes 100 --backend both --workers 3 --name one vehicle, noisy: trained with vs without the noise --out results/arena_noise_algorithms/noisy_waypoint`
