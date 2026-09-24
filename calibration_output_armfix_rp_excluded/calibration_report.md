# Calibration report: calibration_output_armfix_rp_excluded

2026-09-24

## Bottom line

It is trustworthy for **1 of 6** motion axes: turning rate (yaw). On recording sessions the calibration never saw, it predicts your vehicle's motion 50% more accurately than the default model on average, but that comparison alone can flatter a poor default; each verdict below also checks the model against simply assuming no motion.

## Can I trust it for this motion?

| Motion | Verdict | Why |
|---|---|---|
| forward speed (surge) | ⚠️ Use with care | it relies on forward drag (slow speeds), forward drag (fast speeds), water carried along when accelerating forward which hit their allowed limits or look physically implausible |
| sideways speed (sway) | ⚠️ Use with care | it relies on sideways drag (slow speeds), sideways drag (fast speeds) which hit their allowed limits or look physically implausible |
| up/down speed (heave) | ⚠️ Use with care | it relies on vertical drag (slow speeds), vertical drag (fast speeds), net buoyancy (float vs sink) which hit their allowed limits or look physically implausible |
| roll rate | ❌ Don't rely on it | not calibrated: your logs don't explain this motion, so it was left out of the fit and the simulator uses the reference values from vehicle_spec.yaml |
| pitch rate | ❌ Don't rely on it | not calibrated: your logs don't explain this motion, so it was left out of the fit and the simulator uses the reference values from vehicle_spec.yaml |
| turning rate (yaw) | ✅ Trust it | it predicts this motion well on unseen sessions (skill +0.60; 1 = perfect) and 55% more accurately than the default model |

> ⚠️ The optimizer stopped before fully converging. The result is usable but may improve with a longer run.

## Did it learn the vehicle, or just memorise the training runs?

It generalises: accuracy on unseen sessions is as good as on the training runs. (held-out / training error ratio 0.99; 1.0 is ideal.)

## Be careful with

**Values stuck at a limit (10).** The fit pushed these as far as allowed and stopped, so the true value is probably beyond the limit:

- forward drag (slow speeds) (`Xu` = -50, at its lower limit of [-50.0, -1.0])
- sideways drag (slow speeds) (`Yv` = -80, at its lower limit of [-80.0, -1.6])
- vertical drag (slow speeds) (`Zw` = -100, at its lower limit of [-100.0, -2.0])
- forward drag (fast speeds) (`Xuu` = -100, at its lower limit of [-100.0, -2.0])
- sideways drag (fast speeds) (`Yvv` = -100, at its lower limit of [-100.0, -2.0])
- vertical drag (fast speeds) (`Zww` = -150, at its lower limit of [-150.0, -3.0])
- water carried along when accelerating forward (`Xu_dot` = 1.296e-20, at its lower limit of [0.0, 30.0])
- net buoyancy (float vs sink) (`net_buoyancy_z` = 1, at its upper limit of [-1.0, 1.0])
- surge-pitch coupling (`Xq_dot` = -0.5, at its lower limit of [-0.5, 0.5])
- heave-pitch coupling (`Zq_dot` = -0.5, at its lower limit of [-0.5, 0.5])

## What to do next

1. Your logs don't explain roll rate, pitch rate: that motion isn't caused by the recorded thruster commands (tether pull, waves or contact are common causes). To calibrate it, record tests where the thrusters cause it: tip the vehicle and let it settle (a free-decay test), and command deliberate roll/pitch moves, with the tether slack. Until then, treat the simulator's values for those motions as reference values, not measurements.
2. 10 parameter(s) stopped at the edge of their allowed range. Review those limits in vehicle_spec.yaml with someone who knows the vehicle; if the physics allows, widen them and re-run. Until then treat those values as 'at least this large', not as measurements.
3. Validate on hardware: run the same commands on the real vehicle and in the calibrated simulator and compare. Held-out logs are strong evidence; a live test is the real proof.

## Technical appendix

Held-out RMSE per axis (m/s or rad/s): predicting no motion, default physics, calibrated physics. Skill = 1 - calibrated / no-motion.

| Axis | No motion | Default | Calibrated | Improvement vs default | Skill |
|---|---|---|---|---|---|
| u | 0.4829 | 0.2299 | 0.1456 | +36.7% | +0.70 |
| v | 0.1994 | 0.3397 | 0.1045 | +69.2% | +0.48 |
| w | 0.1860 | 0.4506 | 0.0891 | +80.2% | +0.52 |
| p | 0.1391 | 0.6671 | 0.6397 | +4.1% | -3.60 |
| q | 0.1995 | 1.7845 | 0.8193 | +54.1% | -3.11 |
| r | 0.5803 | 0.5153 | 0.2335 | +54.7% | +0.60 |
