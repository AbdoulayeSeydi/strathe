# Calibration report: calibration_output_dfki_bluerov2

2026-10-03

## Bottom line

It is trustworthy for **1 of 6** motion axes: turning rate (yaw). On recording sessions the calibration never saw, it predicts your vehicle's motion 62% more accurately than the default model on average, but that comparison alone can flatter a poor default; each verdict below also checks the model against simply assuming no motion.

## Can I trust it for this motion?

| Motion | Verdict | Why |
|---|---|---|
| forward speed (surge) | ❌ Don't rely on it | the calibrated model predicts this motion WORSE than the default model |
| sideways speed (sway) | ⚠️ Use with care | it relies on sideways drag (fast speeds) which hit their allowed limits or look physically implausible |
| up/down speed (heave) | ⚠️ Use with care | it relies on vertical drag (fast speeds), net buoyancy (float vs sink) which hit their allowed limits or look physically implausible |
| roll rate | ❌ Don't rely on it | it predicts this motion no better than simply assuming the vehicle isn't moving on this axis (skill -0.26), even though it is 88% better than the default model |
| pitch rate | ❌ Don't rely on it | it predicts this motion no better than simply assuming the vehicle isn't moving on this axis (skill -0.01), even though it is 85% better than the default model |
| turning rate (yaw) | ✅ Trust it | it predicts this motion well on unseen sessions (skill +0.43; 1 = perfect) and 56% more accurately than the default model |

> ⚠️ The optimizer stopped before fully converging. The result is usable but may improve with a longer run.

## Did it learn the vehicle, or just memorise the training runs?

It generalises: accuracy on unseen sessions is as good as on the training runs. (held-out / training error ratio 0.65; 1.0 is ideal.)

## Be careful with

**Values stuck at a limit (11).** The fit pushed these as far as allowed and stopped, so the true value is probably beyond the limit:

- resistance to rolling (inertia + added inertia) (`Ieff_roll` = 0.255, at its lower limit of [0.255, 0.265])
- resistance to pitching (`Ieff_pitch` = 0.235, at its upper limit of [0.225, 0.235])
- forward drag (slow speeds) (`Xu` = -50, at its lower limit of [-50.0, -1.0])
- roll damping (`Kp` = -10, at its lower limit of [-10.0, -0.1])
- pitch damping (`Mq` = -5, at its lower limit of [-5.0, -0.1])
- forward drag (fast speeds) (`Xuu` = -100, at its lower limit of [-100.0, -2.0])
- sideways drag (fast speeds) (`Yvv` = -100, at its lower limit of [-100.0, -2.0])
- vertical drag (fast speeds) (`Zww` = -3, at its upper limit of [-150.0, -3.0])
- roll damping (fast rates) (`Kpp` = -10, at its lower limit of [-10.0, -0.2])
- pitch damping (fast rates) (`Mqq` = -10, at its lower limit of [-10.0, -0.2])
- net buoyancy (float vs sink) (`net_buoyancy_z` = 1, at its upper limit of [-1.0, 1.0])

**Parameters your data can't tell apart.** In these pairs, raising one and lowering the other gives almost the same motion, so the data pins down their combined effect but not each value:

- forward drag (slow speeds) vs forward drag (fast speeds) (correlation -0.99)
- vertical drag (slow speeds) vs vertical drag (fast speeds) (correlation -0.95)
- roll damping vs roll damping (fast rates) (correlation -0.94)
- sideways drag (slow speeds) vs sideways drag (fast speeds) (correlation -0.90)
- turning damping vs turning damping (fast rates) (correlation -0.90)

**Groups of values your data barely pins down.** Each group below can shift together with almost no change in the predicted motion, so treat the individual values as uncertain even where they look precise:

- resistance to pitching, resistance to rolling (inertia + added inertia) (constrained about 20,593x less than the best-constrained combination)
- resistance to rolling (inertia + added inertia), resistance to pitching (constrained about 18,572x less than the best-constrained combination)
- pitch damping (fast rates) (constrained about 1,714x less than the best-constrained combination)

## What to do next

1. Your logs don't explain roll rate, pitch rate: that motion isn't caused by the recorded thruster commands (tether pull, waves or contact are common causes). To calibrate it, record tests where the thrusters cause it: tip the vehicle and let it settle (a free-decay test), and command deliberate roll/pitch moves, with the tether slack. Until then, treat the simulator's values for those motions as reference values, not measurements.
2. Record more manoeuvres that exercise forward speed (surge): short, sharp, varied commands on those axes, in several separate sessions.
3. 11 parameter(s) stopped at the edge of their allowed range. Review those limits in vehicle_spec.yaml with someone who knows the vehicle; if the physics allows, widen them and re-run. Until then treat those values as 'at least this large', not as measurements.
4. Some parameters can trade off against each other (listed above). That's harmless for simulation, but don't report those individual values as vehicle properties. Manoeuvres that move the axes they belong to in different ways (different speeds, and turning while moving) are what separate them.
5. Validate on hardware: run the same commands on the real vehicle and in the calibrated simulator and compare. Held-out logs are strong evidence; a live test is the real proof.

## Technical appendix

Held-out RMSE per axis (m/s or rad/s): predicting no motion, default physics, calibrated physics. Skill = 1 - calibrated / no-motion.

| Axis | No motion | Default | Calibrated | Improvement vs default | Skill |
|---|---|---|---|---|---|
| u | 0.3916 | 0.1480 | 0.1548 | -4.6% | +0.60 |
| v | 0.0858 | 0.2216 | 0.0326 | +85.3% | +0.62 |
| w | 0.1280 | 0.2052 | 0.0776 | +62.2% | +0.39 |
| p | 0.1086 | 1.0947 | 0.1364 | +87.5% | -0.26 |
| q | 0.1643 | 1.0829 | 0.1662 | +84.7% | -0.01 |
| r | 0.3430 | 0.4423 | 0.1957 | +55.8% | +0.43 |

Identifiability: Jacobian condition number 2.06e+04 from 14 training runs, 22 free parameters, same objective as the fit.
