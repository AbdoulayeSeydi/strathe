# Fleet v2: water current

34800 episodes. Current direction drawn per episode (any heading); both vehicles feel the same water. Differences are success-rate points with 95% bootstrap CIs over matched episodes; `*` = CI excludes zero.

## Success rate, current-water-trained policies (stand-in engine)

| Comm | Teammate model | 0 m/s | 0.1 m/s | 0.2 m/s | 0.3 m/s | 0.2 fluctuating m/s |
|---|---|---|---|---|---|---|
| L2@1s | calibrated, knows current | 100% | 100% | 100% | 100% | 100% |
| L2@1s | calibrated, unaware | 100% | 100% | 100% | 100% | 100% |
| L2@1s | last_beacon | 100% | 100% | 100% | 100% | 100% |
| L3a@3s | calibrated, knows current | 62% | 65% | 64% | 66% | 63% |
| L3a@3s | calibrated, unaware | 62% | 60% | 57% | 53% | 60% |
| L3a@3s | last_beacon | 53% | 56% | 56% | 53% | 54% |

## Success rate, still-water-trained policies (stand-in engine)

| Comm | Teammate model | 0 m/s | 0.1 m/s | 0.2 m/s | 0.3 m/s | 0.2 fluctuating m/s |
|---|---|---|---|---|---|---|
| L2@1s | calibrated, knows current | 100% | 100% | 100% | 100% | 100% |
| L2@1s | calibrated, unaware | 100% | 100% | 100% | 100% | 100% |
| L2@1s | last_beacon | 100% | 100% | 100% | 100% | 100% |
| L3a@3s | calibrated, knows current | 68% | 67% | 69% | 68% | 68% |
| L3a@3s | calibrated, unaware | 68% | 68% | 63% | 60% | 64% |
| L3a@3s | last_beacon | 65% | 66% | 65% | 60% | 69% |

## Paired differences

**current policies: current 0.1 m/s vs still water**: L2@1s: +0 [+0, +0] (n=1200); L3a@3s: +2 [-1, +4] (n=1200)

**current policies: current 0.2 m/s vs still water**: L2@1s: +0 [+0, +0] (n=1200); L3a@3s: +1 [-2, +4] (n=1200)

**current policies: current 0.3 m/s vs still water**: L2@1s: -0 [-0, +0] (n=1200); L3a@3s: -1 [-4, +2] (n=1200)

**current policies: current 0.2 fluctuating m/s vs still water**: L2@1s: -0 [-0, +0] (n=1200); L3a@3s: +0 [-3, +3] (n=1200)

**current policies, current 0.1 m/s: model knows vs unaware of current**: L2@1s: +0 [+0, +0] (n=400); L3a@3s: +6 [+1, +10] * (n=400)

**current policies, current 0.2 m/s: model knows vs unaware of current**: L2@1s: +0 [+0, +0] (n=400); L3a@3s: +7 [+2, +12] * (n=400)

**current policies, current 0.3 m/s: model knows vs unaware of current**: L2@1s: -0 [-1, +0] (n=400); L3a@3s: +13 [+7, +18] * (n=400)

**current policies, current 0.2 fluctuating m/s: model knows vs unaware of current**: L2@1s: +0 [+0, +1] (n=400); L3a@3s: +3 [-3, +8] (n=400)

**still policies: current 0.1 m/s vs still water**: L2@1s: +0 [+0, +0] (n=1800); L3a@3s: -0 [-2, +2] (n=1800)

**still policies: current 0.2 m/s vs still water**: L2@1s: +0 [+0, +0] (n=1800); L3a@3s: -2 [-4, +0] (n=1800)

**still policies: current 0.3 m/s vs still water**: L2@1s: +0 [+0, +0] (n=1800); L3a@3s: -5 [-7, -2] * (n=1800)

**still policies: current 0.2 fluctuating m/s vs still water**: L2@1s: -0 [-0, +0] (n=1800); L3a@3s: -0 [-2, +2] (n=1800)

**still policies, current 0.1 m/s: model knows vs unaware of current**: L2@1s: +0 [+0, +0] (n=600); L3a@3s: -1 [-4, +3] (n=600)

**still policies, current 0.2 m/s: model knows vs unaware of current**: L2@1s: +0 [+0, +0] (n=600); L3a@3s: +6 [+2, +10] * (n=600)

**still policies, current 0.3 m/s: model knows vs unaware of current**: L2@1s: +0 [+0, +0] (n=600); L3a@3s: +8 [+4, +13] * (n=600)

**still policies, current 0.2 fluctuating m/s: model knows vs unaware of current**: L2@1s: -0 [-0, +0] (n=600); L3a@3s: +4 [+0, +8] * (n=600)

**current-trained vs still-trained policies at 0 m/s**: L2@1s: +0 [+0, +0] (n=1200); L3a@3s: -4 [-7, -1] * (n=1200)

**current-trained vs still-trained policies at 0.1 m/s**: L2@1s: +0 [+0, +0] (n=1200); L3a@3s: -2 [-5, +1] (n=1200)

**current-trained vs still-trained policies at 0.2 m/s**: L2@1s: +0 [+0, +0] (n=1200); L3a@3s: -0 [-3, +3] (n=1200)

**current-trained vs still-trained policies at 0.3 m/s**: L2@1s: -0 [-0, +0] (n=1200); L3a@3s: -3 [-6, +0] (n=1200)

**current-trained vs still-trained policies at 0.2 fluctuating m/s**: L2@1s: +0 [-0, +0] (n=1200); L3a@3s: -4 [-7, -1] * (n=1200)

**MuJoCo vs stand-in (seed 1, still water and 0.2 m/s)**: L2@1s: -0 [-0, +0] (n=2400); L3a@3s: -0 [-2, +1] (n=2400)

