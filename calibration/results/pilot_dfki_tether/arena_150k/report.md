# Single-vehicle experiment: arena_dfki

2026-10-03 14:28 · 100 evaluation episodes per controller · 6.7 min

## What was tested

One simulated vehicle must reach a target point and stay within 0.5 m of it for 20 consecutive steps (2 s), within 200 steps (0.1 s each). **The vehicle** is your calibrated physics (`calibrated_parameters.yaml`, `vehicle_spec.yaml`), in still water, with thruster limits: custom layout: 6 thrusters, curve T200 @ ~15 V (BlueROV2 dataset) rescaled to 13.8 V (+43.2 N / -33.8 N); max force 96/96/86 N, max torque 7.4/0.0/25.6 N*m (surge/sway/heave, roll/pitch/yaw); can't push: pitch torque, with no roll/pitch disturbance, with exact sensor readings. Physics engine for the tests: stand-in. Every controller below drives that same vehicle through the same episodes (same targets).

Controllers:

- **trained on your calibration**: PPO trained in the simulator on your calibrated physics, with the same conditions as the test (2 training seed(s), 150,000 steps each)
- **trained on default physics**: the same training, but on the uncalibrated default values from the vehicle description, i.e. what you would have without calibrating
- **hand-written controller**: pushes toward the target in proportion to the distance; no training
- **random commands**: a floor every controller should beat

## Results

| Controller | Engine | Success (95% CI) | Mean time to succeed | Mean final distance | Mean effort* |
|---|---|---|---|---|---|
| trained on your calibration | stand-in | 78% (71%–83%) | 10.0 s | 0.41 m | 567 |
| trained on default physics | stand-in | 100% (98%–100%) | 8.5 s | 0.19 m | 519 |
| hand-written controller | stand-in | 96% (90%–98%) | 15.7 s | 0.39 m | 308 |
| random commands | stand-in | 0% (0%–4%) | n/a | 2.97 m | 803 |

*Effort: the delivered forces (N) and torques (N·m), in absolute value, summed over the episode and multiplied by the 0.1 s step. It's a relative measure of how hard the thrusters work (lower = gentler), not a physical energy.

## Does the calibration matter for this task?

Policies trained on your calibration succeed -22 points (-28 to -18, 95% CI; 200 paired episodes) compared with policies trained on default physics. That difference is clear of zero. Training-seed variation is real in reinforcement learning; per-seed numbers are in `results.json`, and more seeds (`--seeds`) narrow the interval.

Training on your calibration did *worse* here. That usually means the training budget was too short for your vehicle, not that the calibration is wrong. Every policy is tested on your calibrated vehicle, and a heavily damped vehicle is slower to learn on than the lighter default physics. In our tests, tripling `--timesteps` took calibration-trained policies from 10% to 96% (PROJECT_STATUS.md #56). Re-run with a larger `--timesteps` before drawing conclusions.

## What this does and doesn't show

- It is simulation only: the "vehicle" is your calibrated model. How well that model matches the real vehicle is what your calibration report measures; a hardware trial is the final test.
- Success rates are for these episodes and this training budget; a longer `--timesteps` usually helps harder tasks (station keeping, docking).
- Reproduce: `python run_arena_experiment.py --params <calibration output>/calibrated_parameters.yaml --spec <repo>/calibration/vehicle_spec.yaml --task station_keeping --seeds 2 --timesteps 150000 --episodes 100 --workers 7 --compare-generic --thrusters <repo>/calibration/examples/dfki_tether/ingest_dfki_tether.yaml --backend stand_in --algorithm ppo --out <pilot runs>/arena_dfki`
