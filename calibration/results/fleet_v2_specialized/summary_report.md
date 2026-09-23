# Fleet v2: belief models with policies trained for them

12800 episodes; training seeds [1, 2]; paired by seed, start state and packet losses. Differences are success-rate points with 95% bootstrap CIs; `*` = CI excludes zero.

## Success rate

| Comm | `calibrated` | `generic` | `last_beacon` | `oracle` |
|---|---|---|---|---|
| L0@1s | 100% | 100% | 100% | 100% |
| L0@3s | 98% | 84% | 100% | 100% |
| L1@1s | 100% | 100% | 100% | 100% |
| L1@3s | 93% | 80% | 100% | 100% |
| L2@1s | 100% | 100% | 100% | 100% |
| L2@3s | 86% | 62% | 94% | 100% |
| L3a@1s | 97% | 86% | 99% | 100% |
| L3a@3s | 70% | 44% | 70% | 98% |

## Paired differences

| Comm | calibrated_vs_generic | calibrated_vs_last_beacon | generic_vs_last_beacon | oracle_vs_calibrated | oracle_vs_last_beacon |
|---|---|---|---|---|---|
| L0@1s | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] |
| L0@3s | +14 [+10, +18] * | -2 [-4, -1] * | -16 [-20, -13] * | +2 [+1, +4] * | +0 [+0, +0] |
| L1@1s | -0 [-1, +0] | -0 [-1, +0] | +0 [+0, +0] | +0 [+0, +1] | +0 [+0, +0] |
| L1@3s | +14 [+9, +18] * | -6 [-9, -4] * | -20 [-24, -16] * | +7 [+4, +9] * | +0 [+0, +1] |
| L2@1s | +0 [-0, +1] | -0 [-1, +0] | -0 [-1, +0] | +0 [+0, +1] | +0 [+0, +0] |
| L2@3s | +24 [+18, +30] * | -7 [-11, -3] * | -31 [-36, -26] * | +14 [+10, +17] * | +6 [+4, +9] * |
| L3a@1s | +11 [+8, +15] * | -2 [-4, -0] * | -13 [-16, -10] * | +3 [+1, +4] * | +1 [+0, +2] |
| L3a@3s | +26 [+20, +32] * | +0 [-5, +5] | -26 [-32, -20] * | +28 [+24, +33] * | +28 [+24, +33] * |
