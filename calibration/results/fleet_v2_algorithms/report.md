# SAC and TD3 for the fleet, against PPO (PROJECT_STATUS.md #58)

17,600 episodes, clean simulation, stand-in physics. Criteria fixed before any result: [criteria.md](criteria.md).

## Verdict

| Algorithm | Gate 1 (integration) | Learned (≥ floor + half the gap to PPO) | Parity with PPO | Failed seeds | Validated for the fleet |
|---|---|---|---|---|---|
| SAC | pass | yes (87% vs threshold 50%) | yes | none | **yes** |
| TD3 | pass | yes (91% vs threshold 50%) | yes | none | **yes** |

## Success, pooled over the grid

| | Success (95% CI) | Collisions | Per seed |
|---|---|---|---|
| PPO | 87% (86%–88%) | 1% | 83%, 88%, 90% |
| SAC | 87% (86%–88%) | 0% | 90%, 90%, 81% |
| TD3 | 91% (91%–92%) | 0% | 89%, 93%, 93% |
| random commands (the floor) | 12% (11%–14%) | 0% | n/a |
| no commands (reference, not in the criteria) | 8% (6%–9%) | 0% | n/a |

## Paired against PPO (same episodes; seeds averaged per episode)

| Condition | SAC − PPO (95% CI) | TD3 − PPO (95% CI) |
|---|---|---|
| L2@1s | -2 pts (-2 to -1) | +0 pts (+0 to +0) |
| L2@3s | +2 pts (-1 to +4) | +10 pts (+8 to +13) |
| L3a@1s | +1 pts (-0 to +2) | +3 pts (+2 to +4) |
| L3a@3s | -2 pts (-5 to +2) | +5 pts (+1 to +8) |
| **pooled** | **-0 pts** (-1 to +1) | **+4 pts** (+3 to +6) |

## Training cost

| | Steps per seed | Wall-clock per seed |
|---|---|---|
| PPO | 3,000,000 | 65 min, 66 min, 65 min |
| SAC | 1,000,000 | 158 min, 158 min, 158 min |
| TD3 | 1,000,000 | 137 min, 137 min, 137 min |

Gate 1 details are in `metrics.json` (`gate1`). Wall-clock times depend on how many runs shared the machine; PPO's shipped policies were trained with fewer runs in parallel.
