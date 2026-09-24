# Fleet v2: belief models with policies trained for them

12800 episodes; training seeds [1, 2]; paired by seed, start state and packet losses. Differences are success-rate points with 95% bootstrap CIs; `*` = CI excludes zero.

## Success rate

| Comm | `calibrated` | `generic` | `last_beacon` | `oracle` |
|---|---|---|---|---|
| L0@1s | 100% | 100% | 100% | 100% |
| L0@3s | 40% | 72% | 100% | 100% |
| L1@1s | 100% | 100% | 100% | 100% |
| L1@3s | 58% | 73% | 100% | 100% |
| L2@1s | 100% | 100% | 100% | 100% |
| L2@3s | 74% | 70% | 95% | 100% |
| L3a@1s | 98% | 88% | 99% | 100% |
| L3a@3s | 66% | 57% | 81% | 97% |

## Paired differences

| Comm | calibrated_vs_generic | calibrated_vs_last_beacon | generic_vs_last_beacon | oracle_vs_calibrated | oracle_vs_last_beacon |
|---|---|---|---|---|---|
| L0@1s | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] |
| L0@3s | -32 [-39, -26] * | -60 [-65, -55] * | -28 [-32, -24] * | +60 [+55, +65] * | +0 [+0, +0] |
| L1@1s | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] | +0 [+0, +0] |
| L1@3s | -14 [-20, -8] * | -42 [-46, -36] * | -27 [-32, -23] * | +42 [+36, +46] * | +0 [+0, +0] |
| L2@1s | +0 [+0, +1] | +0 [+0, +0] | -0 [-1, +0] | +0 [+0, +0] | +0 [+0, +0] |
| L2@3s | +4 [-2, +9] | -21 [-26, -16] * | -24 [-29, -20] * | +26 [+22, +31] * | +5 [+3, +8] * |
| L3a@1s | +10 [+6, +13] * | -1 [-2, +0] | -11 [-14, -8] * | +2 [+0, +3] * | +1 [+0, +2] |
| L3a@3s | +9 [+4, +15] * | -15 [-20, -11] * | -24 [-30, -19] * | +31 [+26, +36] * | +16 [+12, +20] * |
