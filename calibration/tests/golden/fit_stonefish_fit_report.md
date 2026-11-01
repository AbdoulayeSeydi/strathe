# Stonefish model calibrated from your logs: vehicle.scn

## Bottom line

On recordings it never saw (02-01-2030), the calibrated model's position error after the longest horizon is **0.01 m against 19.90 m** for the original (-100%). Check every horizon below: a calibration can help over a minute and cost over seconds.

## How the models compare on unseen recordings

Windows of 60 s every 10 s; each starts from the logged state and is driven by the logged forces, the model turning and tilting on its own, in still water. Medians.

| Model | drift 10 s | drift 30 s | drift 60 s | velocity error over 10 s, u v w (m/s) | roll, pitch, yaw after 10 s | diverged |
|---|---|---|---|---|---|---|
| Stonefish, original | 10.31 m | 17.91 m | 19.90 m | 0.551, 0.152, 1.092 | 0.0°, 0.0°, 0.0° | 0/4 |
| Stonefish, calibrated by Strathe | 0.00 m | 0.01 m | 0.01 m | 0.001, 0.000, 0.000 | 0.0°, 0.0°, 0.0° | 0/4 |
| no motion | 5.57 m | 8.64 m | 14.12 m | 0.709, 0.145, 0.287 | 64.7°, 18.7°, 85.1° | 0/4 |
| keeps its starting velocity | 8.13 m | 19.26 m | 39.90 m | 0.941, 0.126, 0.127 | 64.7°, 18.7°, 85.1° | 0/4 |

![Drift](drift.png)

"Keeps its starting velocity" is a hard baseline over a closed-loop mission: the logged forces react to the real vehicle, so any open-loop model's error compounds. These numbers rank models; they don't promise minute-ahead prediction.

## What changed in your vehicle file

`vehicle_calibrated.scn` is `vehicle.scn` with only part `Hull` and the buoyancy part `Foam` changed:

- quadratic drag: 0.2 1 1 -> 0.592793 2.52734 2.53126
- viscous drag: 0.02 0.1 0.1 -> 0.0506285 0.299361 0.298659
- added mass: 5 40 40 -> 7.96208 69.9916 69.9119
- buoyant volume of `Foam`: x1.234
- `anisotropic_added_mass="true"`: needs Stonefish with `calibration/stonefish/stonefish_added_mass.patch` (unpatched Stonefish ignores added mass settings and averages it over the three axes)

## How the fit went

Fitted: drag, added_mass, buoyancy (10 values) on 01-01-2030. Cost 104.507 -> 0.000 after 6 evaluations (The maximum number of function evaluations is exceeded.).

Fitting can trade one axis for another, or tailor values to the training sessions: odd values (drag far from the shape's, added mass at a limit) mean the logs don't pin them down. More varied manoeuvres at several speeds help.
