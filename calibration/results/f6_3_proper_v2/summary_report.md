# Fleet Experiment Report

**Experiment:** `f6_3_proper_800k`  **Generated:** 2026-09-22 12:08:30  **Runtime:** 330.3s

---

## ⚠️ WHICH AGENT IS WHICH — read before interpreting any result

The environment always contains exactly two agents:
one **calibrated** (uses `calibrated_parameters_a1.yaml`, physics fitted to real BlueROV2 logs) and one **generic** (uses uncalibrated default physics). The policy below controls both.

| Index | Role | Model | Algorithm |
|-------|------|-------|-----------|
| 0 | **external_my_run3** (calibrated — `calibrated_parameters_a1.yaml`) | `formation_hold_ppo_f6_3_proper_v2` | PPO |
| 1 | **generic** (generic — default physics) | `formation_hold_ppo_curriculum_v5` | PPO |

> **Note:** only `vehicles[0]`'s policy is used in the sweep (centralized control). `vehicles[1]` is loaded and validated but does not affect sweep results. The calibrated/generic physics split is set by the environment, not by the policy.

---

## Experiment setup

| Field | Value |
|-------|-------|
| Task | Formation hold — target separation 2.0m, tolerance ±0.2m, hold 20 consecutive steps |
| Comm condition | `L2` |
| Dynamics backend | `stand_in` |
| Episodes per cell | 160 (5 comm seeds × 5 grid seeds × 32 grid points) |
| Comm seeds | [1, 2, 3, 4, 5] |

## Units and conventions

| Quantity | Unit | Convention |
|----------|------|------------|
| Separation, distances | meters (m) | Euclidean distance between agent centroids |
| Velocity components | m/s | SNAME/Fossen body frame: surge (x, forward), sway (y, left), heave (z, up) |
| Angular rates | rad/s | Same body frame: roll (p), pitch (q), yaw (r) |
| Speed of sound | m/s | Used in latency model: latency = distance / speed |
| Belief error | meters | RMS position error — how far the agent's internal model of its teammate's position is from the teammate's true position |
| `delta_physics_*` | meters | calibrated agent value **minus** generic agent value — negative means calibrated is closer to truth |
| `delivery_rate_*` | fraction [0, 1] | Fraction of acoustic beacons actually received by that agent |
| `success_rate` | fraction [0, 1] | Episodes where both agents held target separation (2.0m ± 0.2m) for 20 consecutive steps |
| `collision_rate` | fraction [0, 1] | Episodes ending in collision (separation below safety threshold) |

## Results

> **Key metric:** `delta_physics_belief_error` = calibrated minus generic position belief error. **Negative = calibrated agent has a more accurate model of where its teammate actually is.** Near-zero at L0/L1 is expected (comm is good, physics doesn't matter much). Meaningfully negative at L2/L3a is the Claim C signal.

### f6_3_proper_800k@L2_stand_in (stand_in backend)

| Metric | Mean | Std | Range | Note |
|--------|------|-----|-------|------|
| `success_rate` | 0.5975 | 0.0179 | [0.5750, 0.6250] | noisy at small n — check std |
| `collision_rate` | 0.0125 | 0.0105 | [0.0000, 0.0250] | should rise with severity |
| `delta_physics_off_axis` | 0.3494 | 0.0258 | [0.3089, 0.3838] |  |
| `delta_physics_belief_error` | -0.0083 | 0.0011 | [-0.0101, -0.0071] | **← KEY METRIC** negative = calibrated better |
| `delivery_rate_generic` | 0.6923 | 0.0029 | [0.6889, 0.6958] | should match packet-loss prob |
| `delivery_rate_calibrated` | 0.6929 | 0.0018 | [0.6905, 0.6956] | should match packet-loss prob |

## Honest caveats (from `phase2_config.py` Claim C)

> These are the exact caveats recorded in `phase2_config.py` when Claim C was established. They are reproduced here verbatim, not paraphrased, so the report is honest about what this result does and does not establish.

What Claim C does **not** establish (from `phase2_config.py` Decision 3 addendum, stated explicitly per Decision 3's own discipline):

- `success_rate` and `delta_physics_off_axis` did **not** transfer cleanly between backends — but this is confounded by L0–L3a being four **separately-trained** policies (each reflecting its own training run's idiosyncrasies), not evidence against the backend transfer itself.
- This result claims nothing about whether MuJoCo-based **training** would produce equally-good or better policies. No such training has been run, per Decision 2's training-backend rule. This is purely a transfer-of-an-already-stand-in-trained-policy result.
- Claim C is layered on top of Claims A and B. It presupposes Claim A's own physics-fidelity work (MuJoCo is only a meaningful validator here because its dynamics were already confirmed to match the stand-in), and extends Claim B's single-agent 'trained policy behaves sensibly' result to the multi-agent, cross-backend case.
- Any result reported from Track C must be labeled CLAIM_C explicitly, same discipline as A/B.
- **Missing causal link (identified by F5.4 dogfood review):** this experiment demonstrates one link in the causal chain — calibrated physics → more accurate teammate-position belief under degraded comm (`delta_physics_belief_error`). It does **not** establish the full chain through to fleet task success: lower belief error → better coordination behavior → higher `success_rate` / fewer collisions. The `success_rate` in these results is not broken down calibrated-vs-generic, so the experiment cannot support the claim that calibrated physics improves fleet coordination outcomes. That link requires a separate, controlled experiment.

Additionally:

- `success_rate` is noisy at the episode counts used here — do not read a single cell's success_rate in isolation without checking std.
- No real hardware validation has been done for the fleet work.
- Absolute success rates are proof-of-concept grade, not production-ready.

*Full methodology: `PROJECT_STATUS.md` and `phase2_config.py` `Claim C`.*
