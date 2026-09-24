# Extended analysis: fleet_v2_alt_truth

Success-rate differences in percentage points, paired over matched episodes, with 95% bootstrap CIs. `*` = the CI excludes zero.

## 1. `calibrated` minus `generic`, per training seed

| Comm | fleet_policy_seed1 | fleet_policy_seed2 | fleet_policy_seed3 |
|---|---|---|---|
| L2@1s | +2 [+0, +4] * | +0 [+0, +2] | -6 [-10, -3] * |
| L2@3s | +18 [+9, +28] * | +22 [+13, +30] * | +23 [+16, +31] * |
| L3a@1s | +17 [+12, +23] * | +14 [+8, +20] * | +7 [+1, +14] * |
| L3a@3s | +5 [-4, +14] | +36 [+28, +44] * | +16 [+8, +24] * |

## 2. Each belief model minus `last_beacon` (no prediction)

| Comm | `calibrated` | `generic` | `oracle` |
|---|---|---|---|
| L2@1s | +27 [+24, +31] * | +28 [+25, +32] * | +13 [+10, +17] * |
| L2@3s | +21 [+16, +26] * | +0 [-5, +5] | +21 [+16, +26] * |
| L3a@1s | +26 [+22, +30] * | +13 [+8, +18] * | +11 [+7, +15] * |
| L3a@3s | +21 [+16, +26] * | +2 [-3, +6] | +31 [+26, +35] * |

## 3. `oracle` (perfect information) minus `calibrated`

If perfect information does WORSE, the policies are not limited by belief accuracy: they are tuned to the error patterns they were trained on.

- L2@1s: -14 [-17, -11] *
- L2@3s: -0 [-5, +5]
- L3a@1s: -14 [-18, -11] *
- L3a@3s: +10 [+5, +14] *

