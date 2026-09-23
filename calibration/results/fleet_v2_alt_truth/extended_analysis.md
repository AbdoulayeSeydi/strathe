# Extended analysis: fleet_v2_alt_truth

Success-rate differences in percentage points, paired over matched episodes, with 95% bootstrap CIs. `*` = the CI excludes zero.

## 1. `calibrated` minus `generic`, per training seed

| Comm | fleet_policy_seed1 | fleet_policy_seed2 | fleet_policy_seed3 |
|---|---|---|---|
| L2@1s | -8 [-12, -4] * | +1 [+0, +2] | +4 [-6, +13] |
| L2@3s | +13 [+4, +22] * | +26 [+18, +34] * | +18 [+10, +28] * |
| L3a@1s | +7 [+0, +14] | +20 [+14, +26] * | +14 [+5, +22] * |
| L3a@3s | +15 [+6, +24] * | +19 [+11, +28] * | +22 [+13, +30] * |

## 2. Each belief model minus `last_beacon` (no prediction)

| Comm | `calibrated` | `generic` | `oracle` |
|---|---|---|---|
| L2@1s | -11 [-14, -8] * | -10 [-13, -7] * | -20 [-23, -16] * |
| L2@3s | -6 [-10, -1] * | -25 [-30, -20] * | -10 [-15, -5] * |
| L3a@1s | -4 [-7, -0] * | -17 [-21, -13] * | -16 [-20, -12] * |
| L3a@3s | +3 [-1, +8] | -15 [-20, -10] * | +2 [-2, +8] |

## 3. `oracle` (perfect information) minus `calibrated`

If perfect information does WORSE, the policies are not limited by belief accuracy: they are tuned to the error patterns they were trained on.

- L2@1s: -8 [-12, -5] *
- L2@3s: -4 [-9, +0]
- L3a@1s: -12 [-16, -9] *
- L3a@3s: -1 [-6, +4]

