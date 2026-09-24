# Fleet v2: do roll/pitch disturbances matter?

15200 episodes. Disturbance fitted to the real BlueROV2 logs (`bluerov2_rollpitch.yaml`); x1 = as fitted, x2 = twice as rough. Differences are success-rate points with 95% bootstrap CIs over matched episodes; `*` = CI excludes zero.

## Success rate (stand-in engine)

| Policies | Comm | Belief | off | x1 | x2 |
|---|---|---|---|---|---|
| clean | L2@1s | `calibrated` | 100% | 100% | 100% |
| clean | L2@1s | `last_beacon` | 100% | 100% | 100% |
| clean | L3a@3s | `calibrated` | 68% | 68% | 68% |
| clean | L3a@3s | `last_beacon` | 65% | 65% | 65% |
| dist | L2@1s | `calibrated` | 100% | 100% | 100% |
| dist | L2@1s | `last_beacon` | 100% | 100% | 100% |
| dist | L3a@3s | `calibrated` | 75% | 77% | 75% |
| dist | L3a@3s | `last_beacon` | 62% | 62% | 63% |

## Paired differences

**clean: x1 - off**: L2@1s: +0 [+0, +0] (n=1200); L3a@3s: +0 [-2, +2] (n=1200)

**clean: x2 - off**: L2@1s: +0 [+0, +0] (n=1200); L3a@3s: -0 [-3, +2] (n=1200)

**dist: x1 - off**: L2@1s: +0 [+0, +0] (n=800); L3a@3s: +1 [-1, +4] (n=800)

**dist: x2 - off**: L2@1s: +0 [+0, +0] (n=800); L3a@3s: +1 [-1, +3] (n=800)

**trained with disturbance - clean-trained, at x0**: L2@1s: +0 [+0, +0] (n=800); L3a@3s: +6 [+2, +10] * (n=800)

**trained with disturbance - clean-trained, at x1**: L2@1s: +0 [+0, +0] (n=800); L3a@3s: +7 [+3, +11] * (n=800)

**trained with disturbance - clean-trained, at x2**: L2@1s: +0 [+0, +0] (n=800); L3a@3s: +8 [+4, +12] * (n=800)

**MuJoCo - stand-in (seed 1, off and x1)**: L2@1s: +0 [+0, +0] (n=1600); L3a@3s: -1 [-2, +0] (n=1600)

