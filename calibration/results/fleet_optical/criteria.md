# Optical and hybrid links with the shipped fleet policies: criteria fixed before any result was seen (2026-09-27)

Written after the optical link was implemented and smoke-tested (a handful of
episodes to check the plumbing), before the study below was run. The report
answers these questions as written; nothing here is changed after the results
come in.

## What is evaluated

- **Policies:** the shipped PPO fleet policies `fleet_policy_seed1..3`, trained with
  **acoustic links only**. No policy has ever seen an optical link. This measures what
  a better link gives policies that weren't trained for it, not what a policy trained
  to exploit it could do (that is the next study; see the end).
- **Beliefs:** `calibrated` and `last_beacon`.
- **Acoustic comm:** L2 and L3a, beacons every 1 s and 3 s.
- **Links:** acoustic only; optical only; hybrid `optical_first`; hybrid `redundant`.
- **Water:** clear, coastal, harbour, with omnidirectional terminals.
- **Pointing:** in coastal water, `forward_cone` and `fore_aft_cones` (30 deg half-angles)
  against omni, with optical only and hybrid `optical_first`.
- **Noise:** clean everywhere. A noisy-mode subset reruns acoustic vs hybrid `optical_first`
  in coastal water with omni terminals.
- **Grid:** the standard 8 separations x 5 closing speeds x 5 comm seeds = 200 paired
  episodes per cell. Stand-in physics.

## Gate: integration

Every shipped policy runs every link condition to the end with no failure, and the
bookkeeping holds: nothing is delivered that wasn't sent.

## Questions, and how each is answered

1. **Does the optical link help policies trained without it?** Hybrid `optical_first` minus
   acoustic, paired, for each acoustic condition and water, pooled over policies and beliefs.
   "Helps" = 95% CI above zero; "hurts" = below zero; otherwise "no clear difference".
2. **Freshness:** the median teammate-belief age by link and water. Optical is expected at
   0.1 s (one tick) wherever it is up; the report states whether it is.
3. **Water:** the share of optical beacons delivered, and success, by water. Harbour water
   (50% delivery at 3.9 m) is expected to lose optical at the larger start separations
   (up to 5.5 m).
4. **optical_first vs redundant:** paired difference. Redundant sends a superset of messages,
   so it should never be worse; any difference beyond noise would be a bug.
5. **What pointing costs:** cones minus omni, paired.
6. **Collisions:** collision rate with each link against acoustic only. A link that raises
   collisions is flagged even if success rises.
7. **Noisy mode:** does the hybrid's difference against acoustic hold with sensor noise and
   the roll/pitch disturbance on?

Descriptive only; no threshold beyond the Gate. The optical model is uncalibrated, so
every number is conditional on its assumed water values and delivery curve.
