# Unit Economics — Real Compute Cost per Calibration

*Every timing number below is a real, confirmed figure from this
session's actual runs (see `PROJECT_STATUS.md` for the source finding),
not an estimate. Cloud pricing is real, current AWS on-demand pricing
(checked August 2026) — real assumptions and genuine unknowns are
flagged explicitly, not smoothed over.*

## Real timing inputs

| Step | Real, confirmed cost | Source |
|---|---|---|
| Pre-flight check | Seconds | Real, live run this session |
| Stand-in calibration (`--max-nfev 60`, full dataset) | ~6.5 hours (23,427s) | Confirmed real precedent, this session |
| HoloOcean refinement, **per segment** | ~10.9 hours (315 real evaluations, confirmed baseline ~124.2s/eval) | Real segment run, this session |
| Real validation (Claim A / `replay_comparison.py`) | ~85 minutes | Confirmed real, full held-out set, this session |
| Claim B (PPO training) | **Not confirmed this session** — real training happened earlier this project, but no exact wall-clock figure is on record | Flagged gap, not fabricated |
| Optional identifiability report | ~15-17 minutes | Confirmed real, this session |

## The real, honest uncertainty: how many HoloOcean segments does "done" take?

This session needed **at least 5 real segments** to reach the current best validated result (finding #27's segment 5) — and the rotational-axis investigation alone consumed several more beyond that trying different hypotheses. There's no fixed number here; it depends on how much refinement a given vehicle's data needs. Presenting a range, not a false-precision single number:

- **Minimum viable (1 segment)**: a first, real HoloOcean-refined result — good enough to know if the approach works at all
- **This session's actual real usage (5+ segments)**: what it took to reach the best validated result on record

## Real cloud pricing (AWS on-demand, checked August 2026)

| Instance | Specs | Price/hour | Used for |
|---|---|---|---|
| `c5.xlarge` / `c6i.xlarge` | 4 vCPU, 8GB RAM, no GPU | **$0.17/hr** | Stand-in calibration (pure CPU, no HoloOcean) |
| `g4dn.xlarge` | 4 vCPU, 16GB RAM, NVIDIA T4 GPU | **$0.526/hr** | HoloOcean-backed steps (real Unreal Engine rendering) |

**One real, stated assumption**: this assumes a `g4dn.xlarge`-class GPU is *sufficient* for HoloOcean's rendering needs even in the stripped-down, deterministic calibration mode used this session. That has not been directly confirmed against this project's real GPU requirements — if a more powerful GPU instance turns out to be necessary, the HoloOcean-step costs below would scale up accordingly (e.g., a `g5.2xlarge` at $1.212/hr would roughly 2.3x those figures).

## Real cost per full calibration

| Step | Time | Instance | Cost |
|---|---|---|---|
| Pre-flight | ~1 min | CPU | ~$0.00 |
| Stand-in calibration | 6.5 hrs | CPU ($0.17/hr) | **$1.11** |
| HoloOcean refinement (1 segment) | 10.9 hrs | GPU ($0.526/hr) | **$5.73** |
| HoloOcean refinement (5 segments, this session's real usage) | 54.5 hrs | GPU ($0.526/hr) | **$28.67** |
| Claim A validation | 1.42 hrs | GPU ($0.526/hr) | **$0.75** |
| Identifiability report (optional) | ~0.28 hrs | CPU ($0.17/hr) | **$0.05** |

**Real total, minimum-viable path (1 HoloOcean segment): ~$7.60 per vehicle calibration.**
**Real total, this session's actual usage (5 segments): ~$30.55 per vehicle calibration.**

*(Claim B/PPO training cost is not included above — genuinely unconfirmed, not assumed to be zero or negligible.)*

## The real, important takeaway

**Raw compute cost is small — well under $50 even at this session's actual, real usage level.** This is not what would make offering this as a paid product expensive. Worth stating plainly: **the dominant real cost this session was not compute, it was engineer time** — hours spent watching multi-hour runs, diagnosing real bugs live, and making judgment calls about convergence and next steps. Any pricing model built on compute cost alone would badly underestimate what it actually costs to deliver a calibration result *today*, given how much of tonight's process still requires a human in the loop rather than a truly unattended pipeline.

This directly connects to `PRODUCT_VISION.md`'s own priority list: cloud orchestration and a more automated pipeline aren't just about labs' convenience — they're the actual lever that would make the real unit economics (engineer time, not compute) scale at all.

## What's still genuinely unknown, not filled in with a guess

- Claim B's real wall-clock cost
- Whether `g4dn.xlarge`'s GPU is actually sufficient for this project's real HoloOcean rendering needs
- Real storage/data-transfer costs (not computed here — likely small relative to compute, but not confirmed)
- Any pricing model markup, margin, or what a customer would actually be charged — this document is cost, not price