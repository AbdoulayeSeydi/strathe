# Extended analysis: fleet_v2_belief_to_success

Success-rate differences in percentage points, paired over matched episodes, with 95% bootstrap CIs. `*` = the CI excludes zero.

## 1. `calibrated` minus `generic`, per training seed

| Comm | fleet_policy_seed1 | fleet_policy_seed2 | fleet_policy_seed3 |
|---|---|---|---|
| L0@1s | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] |
| L0@3s | +12 [+4, +20] * | +12 [+8, +17] * | +38 [+30, +44] * |
| L1@1s | +0 [+0, +2] | +0 [+0, +0] | +0 [+0, +0] |
| L1@3s | +21 [+12, +29] * | +16 [+10, +22] * | +37 [+30, +44] * |
| L2@1s | +2 [+0, +5] * | +2 [+0, +4] | +0 [+0, +2] |
| L2@3s | +18 [+8, +26] * | +20 [+12, +28] * | +27 [+20, +34] * |
| L3a@1s | +10 [+4, +16] * | +16 [+11, +22] * | +14 [+9, +18] * |
| L3a@3s | +3 [-6, +12] | +25 [+17, +33] * | +20 [+13, +27] * |

## 2. Each belief model minus `last_beacon` (no prediction)

| Comm | `calibrated` | `constant_velocity` | `generic` | `oracle` | `physics_alpha_0.25` | `physics_alpha_0.5` | `physics_alpha_0.75` |
|---|---|---|---|---|---|---|---|
| L0@1s | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] |
| L0@3s | -1 [-3, +1] | -27 [-30, -23] * | -22 [-25, -18] * | +2 [+1, +4] * | -6 [-8, -3] * | -12 [-16, -9] * | -8 [-11, -6] * |
| L1@1s | +0 [+0, +0] | +0 [+0, +0] | -0 [-0, +0] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] |
| L1@3s | -0 [-3, +3] | -16 [-20, -12] * | -25 [-29, -21] * | +6 [+4, +7] * | -6 [-8, -2] * | -8 [-11, -5] * | -10 [-13, -6] * |
| L2@1s | +0 [+0, +0] | +0 [+0, +0] | -2 [-2, -1] * | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | -1 [-2, -0] * |
| L2@3s | +2 [-2, +5] | +0 [-4, +4] | -20 [-24, -16] * | +16 [+13, +19] * | -0 [-4, +4] | +0 [-4, +4] | -9 [-14, -5] * |
| L3a@1s | +1 [-0, +3] | -2 [-4, +1] | -12 [-15, -9] * | +4 [+2, +5] * | +1 [-0, +3] | +1 [-1, +3] | -2 [-4, +0] |
| L3a@3s | +3 [-1, +8] | +1 [-4, +5] | -13 [-18, -8] * | +28 [+24, +31] * | +5 [+0, +9] * | +4 [-1, +8] | -1 [-5, +3] |

## 3. `oracle` (perfect information) minus `calibrated`

If perfect information does WORSE, the policies are not limited by belief accuracy: they are tuned to the error patterns they were trained on.

- L0@1s: +0 [+0, +0]
- L0@3s: +3 [+2, +5] *
- L1@1s: +0 [+0, +0]
- L1@3s: +6 [+4, +8] *
- L2@1s: +0 [+0, +0]
- L2@3s: +14 [+11, +17] *
- L3a@1s: +2 [+1, +4] *
- L3a@3s: +24 [+21, +28] *

