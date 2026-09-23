# Calibration report: calibration_output_10hz_coriolis

2026-09-22

## Bottom line

On recording sessions the calibration never saw, the calibrated simulator predicts your vehicle's motion **65% more accurately** than the default model, averaged over the six motion axes. It is trustworthy for **1 of 6** axes.

## Can I trust it for this motion?

| Motion | Verdict | Why |
|---|---|---|
| forward speed (surge) | ⚠️ Use with care | it improves prediction by 46%, but relies on forward drag (fast speeds), water carried along when accelerating forward which hit their allowed limits or look physically implausible |
| sideways speed (sway) | ⚠️ Use with care | it improves prediction by 56%, but relies on sideways drag (slow speeds) which hit their allowed limits or look physically implausible |
| up/down speed (heave) | ⚠️ Use with care | it improves prediction by 71%, but relies on vertical drag (slow speeds), vertical drag (fast speeds), water carried along vertically which hit their allowed limits or look physically implausible |
| roll rate | ⚠️ Use with care | it improves prediction by 82%, but relies on resistance to rolling (inertia + added inertia), roll damping (fast rates) which hit their allowed limits or look physically implausible |
| pitch rate | ⚠️ Use with care | it improves prediction by 87%, but relies on resistance to pitching, pitch damping, pitch damping (fast rates) which hit their allowed limits or look physically implausible |
| turning rate (yaw) | ✅ Trust it | it predicts this motion 51% more accurately than the default model on unseen sessions |

## Did it learn the vehicle, or just memorise the training runs?

It generalises: accuracy on unseen sessions is as good as on the training runs. (held-out / training error ratio 1.01; 1.0 is ideal.)

## Be careful with

**Values stuck at a limit (9).** The fit pushed these as far as allowed and stopped, so the true value is probably beyond the limit:

- sideways drag (slow speeds) (`Yv` = -80, at its lower limit of [-80.0, -1.6])
- vertical drag (slow speeds) (`Zw` = -100, at its lower limit of [-100.0, -2.0])
- pitch damping (`Mq` = -5, at its lower limit of [-5.0, -0.1])
- forward drag (fast speeds) (`Xuu` = -100, at its lower limit of [-100.0, -2.0])
- vertical drag (fast speeds) (`Zww` = -150, at its lower limit of [-150.0, -3.0])
- roll damping (fast rates) (`Kpp` = -10, at its lower limit of [-10.0, -0.2])
- pitch damping (fast rates) (`Mqq` = -10, at its lower limit of [-10.0, -0.2])
- water carried along when accelerating forward (`Xu_dot` = 30, at its upper limit of [0.0, 30.0])
- water carried along vertically (`Zw_dot` = 1.501e-17, at its lower limit of [0.0, 40.0])

**Physically implausible values.** These disagree strongly with independent measurements:

- resistance to rolling (inertia + added inertia): fitted 5, but BlueROV2 Heavy rigid-body Ixx, public Gazebo model gives 0.26 (19x). The fit is probably using this to compensate for something else.
- resistance to pitching: fitted 5, but BlueROV2 Heavy rigid-body Iyy, public Gazebo model gives 0.23 (22x). The fit is probably using this to compensate for something else.

**Parameters your data can't tell apart.** In these pairs, raising one and lowering the other gives almost the same motion, so the data pins down their combined effect but not each value:

- water carried along when accelerating forward vs water carried along sideways (correlation +0.88)
- sideways drag (slow speeds) vs sideways drag (fast speeds) (correlation -0.84)
- forward drag (slow speeds) vs forward drag (fast speeds) (correlation -0.82)

## What to do next

1. 9 parameter(s) stopped at the edge of their allowed range. Review those limits in vehicle_spec.yaml with someone who knows the vehicle; if the physics allows, widen them and re-run. Until then treat those values as 'at least this large', not as measurements.
2. Fix physically implausible values by holding them at independently measured values (CAD model, pendulum test) and re-calibrating the rest.
3. Some parameters can trade off against each other (listed above). That's harmless for simulation, but don't report those individual values as vehicle properties.
4. Validate on hardware: run the same commands on the real vehicle and in the calibrated simulator and compare. Held-out logs are strong evidence; a live test is the real proof.

## Technical appendix

Held-out RMSE per axis (m/s or rad/s), default vs calibrated physics:

| Axis | Default | Calibrated | Improvement |
|---|---|---|---|
| u | 0.2337 | 0.1258 | +46.2% |
| v | 0.2316 | 0.1030 | +55.5% |
| w | 0.4303 | 0.1233 | +71.3% |
| p | 1.2477 | 0.2299 | +81.6% |
| q | 2.4514 | 0.3227 | +86.8% |
| r | 0.4740 | 0.2320 | +51.1% |

Identifiability: Jacobian condition number 1.19e+04 from 3 training runs.
