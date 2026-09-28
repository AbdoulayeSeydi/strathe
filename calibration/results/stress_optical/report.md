# Stress test: optical and hybrid links (PROJECT_STATUS.md #59)

2026-09-27 · 5125 episodes · 6.2 min · `python stress_test_optical.py`

## What was run

- **Grid (3480 episodes, stand-in physics):** every link (acoustic, optical, hybrid optical_first, hybrid redundant) x water (clear, coastal, harbour, custom c = 1/m) x terminals (omni, forward_cone, fore_aft_cones, down_cone, and an asymmetric fore/aft pair with a 30 deg beam and 60 deg receiver half-angle) x comm L0/L2/L3a/L3b x beacons every 1 s and 3 s x clean/noisy x policy (PPO, SAC, TD3, random, zero), with buoyancy cycling through none, -5, +5 and -5..+5 N and the belief between calibrated and last_beacon. Optical-only runs have no acoustic channel, so they run once per water, terminals, mode and policy.
- **MuJoCo (61 episodes):** every link configuration once.
- **Exact checks (1464 episodes):** the zero policy (no commands, so the vehicles move the same way whatever the link) on L2@1s and L3a@3s, three starts, two seeds, clean and noisy.
- **Determinism:** 120 grid episodes rerun.
- **Acoustic unchanged:** 96 acoustic cells hashed observation by observation against the reference taken on main (8588809) before this change.

## Checks

| Check | Result | Detail |
|---|---|---|
| Every episode runs to the end with finite observations and consistent bookkeeping | PASS | 5125/5125 episodes |
| Optical deliveries nested by water: clear >= coastal >= custom (1/m) >= harbour, episode by episode | PASS | 360/360 episode sets |
| Forward cones deliver nothing from the line start (clean, zero policy, whole episode) | PASS | 144/144 episodes with no optical delivery |
| Forward cones: the first optical beacons (tick 1) deliver nothing (noisy, zero policy) | PASS | 144/144 episodes |
| Hybrid redundant: acoustic sends, deliveries and draws identical to acoustic only | PASS | 480/480 |
| Hybrid redundant: the teammate belief is never older than with acoustic only, tick by tick | PASS | 480/480 |
| Hybrid optical_first: acoustic draws paired with acoustic only (sent + skipped = sent; same next draw) | PASS | 480/480 |
| Asymmetric terminals: optical_first runs as redundant (no acoustic beacon skipped) | PASS | 96/96 |
| Omni terminals in clear/coastal water: optical_first skips acoustic beacons (the link is up) | PASS | 48/48 |
| Determinism: reruns give identical observations and rewards | PASS | 120/120 reruns identical |
| Acoustic runs unchanged: every observation, reward and delivery identical to main before this change | PASS | 96/96 cells identical |

## What the shipped policies do to pointing (descriptive)

Share of optical beacons delivered across the grid (PPO, SAC, TD3 and random policies; every water):

| Terminals | Optical delivered |
|---|---|
| omni | 89% |
| forward_cone | 0% |
| fore_aft_cones | 4% |
| down_cone | 2% |
| asymmetric | 9% |

With omnidirectional terminals, by water (every policy):

| Water | Optical delivered |
|---|---|
| clear | 99% |
| coastal | 99% |
| custom (c = 1/m) | 97% |
| harbor | 59% |

Hybrid optical_first across the grid skipped 6,025 acoustic beacons; for 3.2% of them the optical copy sent the same tick was lost too, so the teammate got that state on neither link (a vehicle skips only right after an optical message arrived: `link_up_ticks` = 1).


Cone terminals deliver little because the shipped policies were never rewarded for holding attitude: they roll and pitch freely while holding formation (#50), so a narrow cone rarely stays on the teammate. Omni terminals are the fair baseline for these policies.

**11/11 checks passed.**
