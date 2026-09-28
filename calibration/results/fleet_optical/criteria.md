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

## Addendum (2026-09-27, after the first run, before the rerun with the 1-tick skip window)

Added after review, not before the first results; it changes no question or verdict
rule above. The rerun uses `optical_first` with a vehicle skipping its acoustic
beacon only if an optical message arrived on the previous tick (`link_up_ticks: 1`;
the first run used 0.3 s).

**What this study can and can't answer.** The shipped policies were trained on
acoustic links only, and nothing in their observation says whether the optical link
is up; the only trace of it is the belief-age input, which drops to one tick while
optical beacons arrive. So this study answers **whether fresher teammate information
helps policies that already exist**. It **cannot** answer the headline question:
whether a fleet *learns* to stay within optical range and pointed at its teammate,
and what that costs in collisions. That needs policies trained with the link:
`strathe train-fleet --link hybrid` (built in this change), cones included, then the
same grid. That is the follow-up study.

**What the beacons carry.** `Beacon.tau` is the wrench delivered on the send tick, not
a planned future command. An optical link at 10 Hz therefore gives *high-rate state
sharing* (the teammate's current state and last thrust, one tick old), not intent
sharing. Intent sharing (planned commands or a trajectory) stays the open lever from #48.

**No latency effect.** Messages are delivered on the 0.1 s tick: any latency above
zero arrives on the next tick, so optical's 1 ms and acoustic's few milliseconds at
these ranges are identical here. Any difference in this study comes from the optical
link's send rate and loss, not its latency.
