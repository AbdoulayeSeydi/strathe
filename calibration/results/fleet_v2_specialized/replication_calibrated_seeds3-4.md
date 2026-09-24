# Replication: policies trained for the calibrated teammate model

The specialized study (seeds 1-2) found calibrated-trained policies failing to settle at 3 s beacons. Two more seeds were trained the same way (`specialized_policies/replication/`, `train_fleet_policy.py --belief calibrated --seed 3|4`) and evaluated on the identical episode grid (200 episodes per cell). Formation success:

| Comm | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|
| L0@1s | 100% | 100% | 100% | 100% |
| L0@3s | 35% | 45% | 98% | 57% |
| L1@1s | 100% | 100% | 100% | 100% |
| L1@3s | 52% | 64% | 98% | 70% |
| L2@1s | 100% | 100% | 100% | 100% |
| L2@3s | 67% | 80% | 88% | 80% |
| L3a@1s | 98% | 98% | 98% | 98% |
| L3a@3s | 63% | 69% | 68% | 72% |

Reading: large seed-to-seed variance (35-98% at L0 with 3 s beacons), not a systematic failure. With this truth vehicle, training a policy for the calibrated teammate model alone is unreliable; policies trained on a mixture of models (the main study) are consistent across seeds. PROJECT_STATUS.md #54.
