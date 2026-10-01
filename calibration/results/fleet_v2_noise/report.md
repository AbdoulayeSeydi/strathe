# Clean vs noisy fleet simulation, and buoyancy (PROJECT_STATUS.md #58)

> **Correction (2026-10-01, PROJECT_STATUS.md #60).** The numbers below stand, but their explanation doesn't: the cost of `typical` sensor noise is almost all its 0.2 m starting position fix, not the sensors or their drift. With a perfect fix, the `typical` sensors cost 1 point. See [decomposition.md](decomposition.md), and `--noise open_water` for sensor noise measured in open water.

57,120 paired episodes, stand-in physics. Comm L2 @ 1 s and L3a @ 3 s, beliefs calibrated and last_beacon, pooled unless stated.

**Read this first:** `tank` is measured from the reference tank logs; `typical` uses assumed sizes for a DVL + AHRS vehicle, not measured on any open-water vehicle. The costs below are for those assumed sizes (x0.5 and x2 bracket them); for your own fleet, fit your sensors with `strathe fit-noise` and rerun.

## 1. How much does sensor noise cost?

Success for policies trained clean and policies trained with `--mode noisy`, at each noise level. `noisy mode` = typical sensors + the fitted roll/pitch disturbance.

| Level | Trained clean | Noisy − clean level (95% CI) | Trained noisy | Noisy − clean level | Navigation error | Belief error |
|---|---|---|---|---|---|---|
| clean | 83% | — | 84% | — | 0 cm | 42 cm |
| tank | 83% | -0 (-1 to +0) | 84% | +0 (-0 to +1) | 0 cm | 42 cm |
| typical x0.5 | 80% | -3 (-4 to -2) | 81% | -3 (-4 to -1) | 13 cm | 47 cm |
| typical x1 | 73% | -10 (-12 to -8) | 74% | -10 (-11 to -8) | 26 cm | 56 cm |
| typical x2 | 50% | -34 (-36 to -31) | 58% | -26 (-28 to -24) | 52 cm | 78 cm |
| noisy mode | 73% | -10 (-12 to -9) | 74% | -10 (-12 to -8) | 26 cm | 56 cm |

Per condition, clean-trained policies (noise level − clean, points):

| Level | L2@1s `calibrated` | L2@1s `last_beacon` | L3a@3s `calibrated` | L3a@3s `last_beacon` |
|---|---|---|---|---|
| tank | -0 (-0 to +0) | +0 (+0 to +0) | -1 (-4 to +2) | +0 (-1 to +2) |
| typical x0.5 | -2 (-3 to -1) | -2 (-3 to -1) | -5 (-8 to -1) | -3 (-7 to +0) |
| typical x1 | -10 (-12 to -7) | -12 (-14 to -9) | -10 (-14 to -6) | -9 (-14 to -5) |
| typical x2 | -42 (-46 to -38) | -47 (-50 to -43) | -23 (-28 to -18) | -23 (-27 to -18) |
| noisy mode | -8 (-11 to -6) | -12 (-14 to -10) | -10 (-15 to -6) | -10 (-15 to -6) |

## 2. Does training with the noise help?

Trained noisy − trained clean, the same episodes, seeds averaged per episode:

| Level | Difference (95% CI) |
|---|---|
| clean | +1 (-1 to +2) |
| tank | +1 (-0 to +3) |
| typical x0.5 | +1 (-1 to +2) |
| typical x1 | +1 (-1 to +3) |
| typical x2 | +8 (+6 to +11) |
| noisy mode | +1 (-1 to +3) |

## 3. SAC and TD3 under noise

| Policies (trained clean) | Clean | Noisy mode | Noisy − clean (95% CI) |
|---|---|---|---|
| PPO | 83% | 73% | -10 (-12 to -9) |
| SAC | 82% | 69% | -13 (-14 to -11) |
| TD3 | 86% | 64% | -22 (-24 to -20) |

## 4. Buoyancy

Clean-trained PPO, clean sensors. Offset = net buoyancy change from the calibration (− = heavier). `unaware` / `aware`: whether the calibrated teammate model knows the change.

| Offset | calibrated, unaware | vs 0 N | calibrated, aware | vs 0 N | last_beacon | vs 0 N |
|---|---|---|---|---|---|---|
| -10 N | 84% | +1 (-2 to +3) | 84% | +0 (-2 to +2) | 82% | -1 (-3 to +1) |
| -5 N | 84% | +0 (-2 to +2) | 84% | +0 (-2 to +2) | 82% | -1 (-3 to +1) |
| -2 N | 84% | +0 (-2 to +2) | 84% | -0 (-2 to +2) | 83% | +0 (-1 to +2) |
| 0 N | 84% | — | — | — | 83% | — |
| +2 N | 82% | -1 (-3 to +1) | 84% | +0 (-2 to +2) | 82% | -1 (-2 to +0) |
| +5 N | 85% | +1 (-1 to +3) | 84% | +0 (-2 to +2) | 83% | +0 (-2 to +2) |
| +10 N | 88% | +4 (+2 to +7) | 83% | -0 (-3 to +2) | 83% | -1 (-2 to +1) |
