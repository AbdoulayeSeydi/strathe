# Extended analysis: fleet_v2_mujoco_truth

Success-rate differences in percentage points, paired over matched episodes, with 95% bootstrap CIs. `*` = the CI excludes zero.

## 1. `calibrated` minus `generic`, per training seed

| Comm | fleet_policy_seed1 | fleet_policy_seed2 | fleet_policy_seed3 |
|---|---|---|---|
| L3a@1s | +14 [+8, +19] * | +17 [+12, +23] * | +14 [+10, +19] * |
| L3a@3s | -2 [-10, +8] | +24 [+16, +32] * | +18 [+11, +26] * |

## 2. Each belief model minus `last_beacon` (no prediction)

| Comm | `calibrated` | `generic` |
|---|---|---|
| L3a@1s | +1 [-1, +2] | -14 [-17, -11] * |
| L3a@3s | +2 [-2, +6] | -12 [-17, -7] * |

