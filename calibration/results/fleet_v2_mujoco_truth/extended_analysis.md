# Extended analysis: fleet_v2_mujoco_truth

Success-rate differences in percentage points, paired over matched episodes, with 95% bootstrap CIs. `*` = the CI excludes zero.

## 1. `calibrated` minus `generic`, per training seed

| Comm | fleet_policy_seed1 | fleet_policy_seed2 | fleet_policy_seed3 |
|---|---|---|---|
| L3a@1s | +6 [+0, +12] * | +10 [+5, +14] * | +12 [+8, +16] * |
| L3a@3s | +16 [+8, +25] * | +14 [+6, +22] * | +24 [+17, +32] * |

## 2. Each belief model minus `last_beacon` (no prediction)

| Comm | `calibrated` | `generic` |
|---|---|---|
| L3a@1s | -1 [-2, +0] | -10 [-13, -8] * |
| L3a@3s | -2 [-6, +2] | -21 [-26, -16] * |

