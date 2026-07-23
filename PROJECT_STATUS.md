# Calibrated Digital Twins for Marine Autonomy — Project Status

*Last updated: current session*

## Where we stand against the MVP plan

| Phase 0 acceptance criterion | Status |
|---|---|
| Real logged BlueROV2 dataset in hand, matching input/state format | ✅ Done — 39 unique runs (see dedup note below), oceansystemslab public dataset, validated |
| HoloOcean runs locally, produces a simulated trajectory for a scripted thruster sequence | ✅ **Done** — HoloOcean v2.3.0 installed on Windows, `PierHarbor`/`SimpleUnderwater` scenarios confirmed with `HoveringAUV` (BlueROV2 analog). Ran a scripted thruster command (`[10,10,10,10,0,0,0,0]`) for 200 steps: produced a real, physically sensible simulated trajectory — pure vertical displacement (z: -4.94 → -0.185), orientation unchanged, matching a purely-vertical thruster input with no spurious rotation |

**Phase 0 is now complete.** Getting here took real troubleshooting: a multi-day BYU asset-server outage (documented via GitHub issue #353, resolved after their server migration), a GitHub↔Epic account linking gap (pending invite email), and standard Windows dev-environment setup (Python PATH, Git Bash). Everything below was built as Phase 1 *preparation* work, deliberately done against a placeholder physics model instead of HoloOcean so the calibration pipeline itself could be built and tested without waiting on hardware access — that placeholder era is now over.

---

## Vehicle & dataset (Phase 0)

- **Target vehicle:** BlueROV2 Heavy, 8-thruster configuration — public specs, matches HoloOcean's closest built-in analog (`HoveringAUV`)
- **Dataset:** `oceansystemslab/Bluerov2_trajectory_dataset` (public GitHub repo) — **39 unique real logged runs** (40 files exist, but one — `odom-12-01-2024-run1.csv` — is a byte-identical duplicate present at two paths; see finding #6 below) across 4 collection dates (`12-01-2024`, `14-02-2024`, `16-02-2024`, `19-02-2024`), ~183,555 total samples
- **Maneuver diversity confirmed:** all runs show 6/6 DOFs excited (genuinely coupled, multi-axis motion) — a real stress test, not a soft one
- **Open question, still unresolved:** true sampling rate is unknown (no time column in the dataset, no associated paper). Currently using 50Hz as an explicitly flagged, unconfirmed placeholder throughout

---

## Calibration pipeline (Phase 1 prep) — built and tested

A full calibration pipeline exists in `calibration/`, built against a stand-in analytical physics model (not HoloOcean) so the logic could be tested now:

- **`vehicle_spec.yaml`** (v0.3.0) — known constants vs. calibration-target parameters, every value with unit/source/confidence/bounds-rationale, explicit assumptions section
- **`vehicle_spec_schema.py`** — Pydantic schema for stricter validation (untestable in the dev sandbox, network-restricted; the plain-Python validation in `optimizer.py` is what's actually verified working)
- **`dynamics_model.py`** — simplified 6-DOF underwater vehicle dynamics (diagonal damping/added-mass, simplified buoyancy restoring force, teacher-forced orientation). Explicitly documented as a placeholder for HoloOcean, not a final simulator
- **`data_split.py`** — held-out-by-**date** (not random run) train/validation split, plus deduplicated CSV discovery (`discover_csv_files()`) — so the optimizer is tested against entire unseen collection sessions, and no run gets silently double-counted
- **`objective.py`** — RMSE/residual computation, explicit required sampling rate (no silent defaults), DOF-weighting and run-length-normalization options
- **`optimizer.py`** — `scipy.optimize.least_squares` wrapper: robust loss (`soft_l1`), multi-restart support with configurable initialization strategy, explicit convergence checking, YAML-as-single-source-of-truth parameter schema (no hand-maintained parameter-name lists anywhere)
- **`run_calibration.py`** — full CLI pipeline (discover → split → load → calibrate → evaluate → report → save artifacts), structured logging, exit codes, reproducibility metadata (git commit, timestamps)
- **`diagnose_calibration.py`** — post-hoc diagnostics: active-bounds report, profiling, per-DOF worst-run RMSE, Jacobian-based parameter identifiability/correlation analysis, real-vs-simulated velocity plots, force-term-breakdown plots
- **`check_excitation_confound.py`** — checks whether a run's fit quality is confounded by how much it actually excited the axis being evaluated

All of this has been tested against both synthetic data (fast iteration/bug-catching) and the real dataset (the calibration itself, three times now).

---

## Validation methodology: held-out-by-date, not random split

Every real-data calibration run used `data_split.py`'s `split_by_date()` —
runs grouped by collection date, with entire dates held out, not a random
subset of individual runs. Specifically: fit on `12-01-2024` + `14-02-2024`
(25 unique runs), held out `16-02-2024` + `19-02-2024` (14 runs) entirely.

This matters because runs from the same day likely share conditions —
battery charge, tank/pool state, vehicle trim that day. A random 80/20
split across all runs could still overfit to "how the vehicle behaved
during these four sessions in general" without it showing up in the
numbers, since train and held-out would share day-level conditions. Holding
out whole dates is the harder, more honest test.

**What the result actually shows:** every calibration run had train RMSE ≈
held-out RMSE (generalization gap consistently 0.98–1.00x) — on two ENTIRE
collection dates the optimizer never saw. That's real evidence of no
date-level overfitting.

**What it does NOT show:** that the underlying physics is correct. The
calibration generalizes *consistently* across all 4 dates to the SAME
missing-dynamics limitation (see finding #5 below) — i.e. it's consistently
missing the same things on every date, which is why the split doesn't catch
it. "Generalizes across dates" and "models the vehicle's true physics" are
two different claims; only the first one has been demonstrated so far.

---

## Key findings from the real-data runs

### 1. First calibration run (23 parameters, duplicate file still present)
- Converged cleanly: 78.6% train RMSE improvement, 78.4% held-out — nearly identical, no overfitting signal in this split
- **12 of 23 parameters pinned to a bound** — a red flag investigated rather than papered over with wider bounds
- Runtime: ~40 minutes

### 2. Structural non-identifiability discovered
- Jacobian-based analysis found `Izz` and `Nr_dot` correlated at **exactly -1.000** — not just hard to fit, mathematically impossible to separate, since they only ever appear as a sum in the dynamics
- Confirmed algebraically (not just inferred) that roll (`Ixx+Kp_dot`) and pitch (`Iyy+Mq_dot`) have the identical structure
- **Fix:** reparameterized 6 fields into 3 effective-inertia parameters (`Ieff_roll/pitch/yaw`). 23 params → 20. Zero changes needed to `objective.py`, `optimizer.py`, `run_calibration.py`, or `diagnose_calibration.py` — direct payoff of the YAML-as-single-source-of-truth architecture

### 3. Second calibration run (20 parameters, reparameterized, duplicate still present)
- Much cheaper convergence: 34 evaluations / ~12 minutes vs. 104 evaluations / ~40 minutes — confirms the fix improved conditioning as expected
- RMSE essentially unchanged (78.8%/78.6%) — expected; this fix was about parameter *validity*, not fit quality
- **`Ieff_roll`/`Ieff_pitch` still pinned**, now at a deliberately-widened bound (5.0, vs. old effective 2.4) — ruling out "bounds were too tight" as the explanation
- `Ieff_yaw` did *not* pin — landed at a reasonable interior value, isolating the problem specifically to roll/pitch

### 4. Damping-pair correlations confirmed real, separate from the inertia issue
- Re-ran Jacobian analysis after reparameterization: condition number dropped from `inf` to a finite `~1.48e4` (structural degeneracy resolved), but `Xu/Xuu`, `Yv/Yvv`, `Zw/Zww`, `Kp/Kpp`, `Mq/Mqq`, `Nr/Nrr` all still correlated 0.84–0.93. This is the weaker, separate "practical identifiability" issue (needs wider training speed range), not fixed by the reparameterization

### 5. Rotational dynamics confirmed as the real weak point
- Per-DOF RMSE consistently shows translational DOFs (u/v/w) fitting well (~0.07–0.19) vs. rotational (p/q especially) fitting far worse (~0.24–0.45)
- Real-vs-simulated plots directly confirmed this visually: yaw (`r`) tracks real data's shape reasonably well; pitch (`q`) shows the simulated line barely responding to real input at all; roll (`p`) shows large slow excursions unrelated to real motion
- **Force-breakdown plots ruled out the initial hypothesis** (restoring force dominating) and instead showed **damping** doing the heavy lifting — consistent with the stand-in model's missing Coriolis/cross-coupling terms forcing the optimizer to substitute maximum-strength damping as the only tool available to keep simulated rotational velocity roughly in check

### 6. Duplicate file found and fixed — confirmed to not matter materially
- `odom-12-01-2024-run1.csv` existed as a byte-identical duplicate at two paths (top level and inside a `dataset/` subfolder) — confirmed via `diff`. Every prior calibration run (findings #1–5 above) double-counted this one training run's residuals without anyone noticing
- **Fix:** centralized file discovery into `data_split.discover_csv_files()`, deduplicating by filename. Adopted by `run_calibration.py`, `diagnose_calibration.py`, and `check_excitation_confound.py` — all three had the identical unprotected bug independently
- **Clean re-run result (25 train / 14 heldout, 39 total):** RMSE train 0.2136 / heldout 0.2132, generalization gap 0.99x/1.00x — statistically indistinguishable from the duplicate-inflated numbers. **The duplicate never meaningfully affected any conclusion in findings #1–5.** Parameter values, active-bounds pattern, and damping correlations all unchanged within noise

### 7. Restart bug found and fixed — then a real multi-restart stability check confirmed the optimum is robust
- First attempt: all 5 restarts produced byte-for-byte identical iteration tables — `CalibrationConfig.initialization_strategy` defaulted to `"fixed"` with no CLI override, so `--n-restarts 5` silently ran the identical deterministic optimization 5 times, not 5 different attempts. That "5/5 converged, 100%" result was real but uninformative
- **Fix:** added `--initialization-strategy` (default now `perturbed`, was silently `fixed`) and `--perturbation-frac` CLI flags
- **Re-ran on real data with the fix.** This time the 5 restarts had genuinely different starting costs (`1.01e+05`, `55.1`, `487.3`, `669.1`, `64.0`) and different convergence paths (42, 54, 33, 58, 49 evaluations respectively) — confirmed independent, not duplicated
- **Result: all 5 independently-started restarts converged to the identical final cost (3.6388) and identical parameter values.** This is a real, meaningful stability finding — five different starting points all found their way to the same optimum, which is good evidence the calibrated parameters represent a genuine, robust optimum of this objective (given the current model and bounds), not an artifact of one particular starting guess
- Runtime: 5446.8s (~90.8 min) for all 5 restarts combined (236 total evaluations) — longer than the flawed run (56.7 min, 165 evals) simply because genuinely different starting points took different amounts of work to converge, as expected

### 8. Excitation-confound check: ruled out, on clean (deduplicated) data
- Checked whether the "worst run" used for the force-breakdown analysis (`odom-12-01-2024-run5.csv`) was actually well-excited on the vertical thrusters (pwm5-8, which drive roll/pitch), or whether its high RMSE was partly an artifact of low real rotational excitation
- **Correlation between vertical-thruster excitation and roll/pitch RMSE: +0.793** (strongly positive) — runs with MORE rotational excitation have HIGHER RMSE, the opposite of the confound. This strengthens rather than undermines the missing-physics conclusion: the model fits worst precisely where rotational dynamics matter most
- `run5` specifically: excitation rank 14/39 — solidly mid-pack, not a low-excitation outlier. No caveat needed on the original force-breakdown analysis

### 9. A plausible (not fully confirmed) lead on the runtime mystery
- The Jacobian-only evaluation in `diagnose_calibration.py` costs ~1.6s per `joint_residuals()` call (33.4s / 21 evals) — but the full calibration run costs ~20.6s per evaluation (3404s / 5 identical restarts / 33 evals each), a ~13x gap calling the same function over the same training data
- **Likely explanation, not yet directly confirmed:** `scipy.optimize.least_squares`'s trust-region step involves linear algebra (QR/SVD) on the full Jacobian matrix between function evaluations — with ~5.5M residuals × 20 parameters, that matrix operation plausibly costs several seconds per iteration on its own, on top of the raw function evaluations a standalone profile would capture. Treat as the leading hypothesis, not a settled answer

### 10. Damping-pair correlations: partly data-coverage, partly a missing-physics symptom
- Tested `vehicle_spec.yaml`'s previously-cited (never verified) claim that damping-pair correlations are caused by narrow training speed range — built a second, independent measurement: `corr(v, v·|v|)` using ONLY the real logged velocity distribution, with no model or optimizer involved at all
- **For `Xu/Xuu` and `Yv/Yvv`: the two independent measures agree closely.** This confirms the narrow-speed-range mechanism is real and dominant for these two axes
- **For `Mq/Mqq`: the two measures sharply disagree.** It's the 2nd-worst pair by the fitted-model (Jacobian) measure, but the LEAST severe of all six by pure velocity kinematics. Data coverage alone doesn't explain this
- **Working hypothesis:** `Mq/Mqq`'s correlation is a symptom of the same missing-rotational-physics problem already found (finding #5) — the optimizer using both damping terms jointly as compensation for absent Coriolis/cross-coupling dynamics, not just struggling to tell two similar basis functions apart. `Zw/Zww` and `Kp/Kpp` show milder versions of the same disagreement
- **Concrete follow-up flagged for after the HoloOcean swap:** re-run this same two-measurement comparison once real physics replaces the stand-in. If the hypothesis is right, `Mq/Mqq`'s fitted-model correlation should drop toward its kinematic baseline (0.764) once real Coriolis/cross-coupling terms are present — a testable, falsifiable prediction, not just a plausible story
- `vehicle_spec.yaml`'s `identifiability_note` updated to document both measurements and this per-axis distinction, replacing the earlier blanket "narrow speed range" claim

### 11. The 50Hz sampling rate placeholder was wrong — empirically determined to be ~10Hz instead
- Every calibration result documented above (findings #1–10) used an unverified 50Hz placeholder for the dataset's sampling rate — the dataset has no time column, no associated paper. This was always flagged as provisional, but never actually resolved until now
- **Method:** the dataset logs both body-frame velocity AND world-frame position AND orientation. Velocity integrates to position — pure kinematics, independent of any model or calibration assumption. Forward-integrating logged velocity at a CANDIDATE dt and comparing the reconstruction to the ACTUAL logged position reveals which candidate dt is consistent with the data
- **First attempt was flawed and caught before being trusted:** a naive whole-run integration showed different "optimal" rates per collection date (11Hz, 16Hz, ~20Hz) — investigated further rather than accepted, because real sensor bias compounding over a 40-80+ second integration can masquerade as a rate signal. Built a windowed version (resets to real logged position every 2s, so drift can't accumulate) and validated it against synthetic ground truth — confirmed the whole-run method IS foolable by a small realistic bias (found the wrong rate by 1Hz on synthetic data with a known answer), while the windowed method correctly recovers the true rate even with that same bias present
- **With the corrected windowed method, all 4 real collection dates converged on the same answer (~10Hz)** — the earlier per-date divergence was confirmed to be the drift artifact, not a real difference in logging rate
- **Further validated by:** stability across 3 different window-length choices (1s/2s/5s all agree), a split-half reproducibility check (two independent halves of one date's runs landed on 10Hz within 0.0002m of each other), and per-run outlier screening (37/39 runs cluster tightly; only 2 ordinary outliers, consistent with normal real-world data quality, not evidence against 10Hz)
- **Independently confirmed by a structural check not part of the original plan:** row counts across the dataset cluster overwhelmingly at clean multiples of 1000 (2000/4000/6000 rows), dividing into clean 200s/400s/600s durations at 10Hz exactly — the signature of a real fixed-rate clock recording for round-number session lengths, not a coincidence of curve-fitting. The handful of "short by 1-2 rows" files (1999, 3999, 5999) are explained as early-stopped recordings on top of that same clock, not evidence of rate variability
- **Consequence:** this is a real, substantial correction — a ~5x error means every acceleration estimate and every fitted damping/inertia parameter in findings #1–10 was computed at the wrong timescale. The STRUCTURAL findings (the Izz/Nr_dot degeneracy, the reparameterization fix, the damping-pair correlations, the rotational-dynamics/missing-physics diagnosis) are very likely still qualitatively valid — none of them depend on the absolute value of dt, only on relative patterns. But every specific fitted parameter VALUE and every specific RMSE NUMBER in findings #1–10 should be treated as provisional until recalibrated at 10Hz
- **Fix:** `run_calibration.py`'s `--sampling-rate-hz` now defaults to 10.0 (previously required with no default at all, to force explicit acknowledgment of the uncertainty). `objective.py`'s module docstring updated with the full finding
- **New scripts:** `check_sampling_rate_windowed.py` (the validated method), `check_sampling_rate_per_run.py` (per-run outlier screening)

### 12. Recalibration at the corrected 10Hz rate — structural findings hold, but a real numerical bug and a real dt-mislabeling mechanism were both found along the way

- **First attempt crashed.** Re-running calibration at 10Hz (instead of the old 50Hz placeholder) produced a clean optimizer convergence, but the subsequent evaluation step crashed: `Simulation produced NaN/inf`. Diagnosed rather than worked around: `dynamics_model.py`'s Forward Euler integration had always used the DATA's own sample interval as the physics integration step — fine at 50Hz (dt=0.02s), but a genuine Forward Euler numerical instability at 10Hz (dt=0.1s) specifically for the DEFAULT (uncalibrated, weakly-damped) parameter set. Reproduced directly (same params, same forces, 50Hz stable / 10Hz overflows to NaN) before accepting the diagnosis
- **Fix:** `SimulationConfig` now has a `physics_dt` field (default 0.02s) — the external `dt` is sub-divided into `ceil(dt/physics_dt)` internal sub-steps (holding force/orientation constant across them — zero-order hold), decoupling "how coarsely the data was sampled" from "how fine the physics needs to integrate for stability." At 50Hz this is a no-op (1 sub-step); confirmed via direct regression test that 50Hz results are **byte-for-byte identical** before/after the change. At 10Hz it now takes 5 sub-steps and no longer crashes
- **Recalibration succeeded at 10Hz.** Headline numbers essentially confirm the structural findings: RMSE train 0.2192 / heldout 0.2172 (vs. 0.2136/0.2132 at 50Hz — negligible difference), generalization gap 0.98x/0.99x (same as before, no overfitting signal), and the same rotational-DOF weakness pattern (`p`/`q` still clearly worse than `u`/`v`/`w`)
- **But a real, separate mechanism was found in the added-mass parameters (`Xu_dot`, `Yv_dot`, `Zw_dot`), and it's NOT the same thing sub-stepping fixed.** All three swung dramatically between rates — `Yv_dot` flipped from pinned at its lower bound (0.0) at 50Hz to pinned at its upper bound (40.0) at 10Hz; `Xu_dot` and `Zw_dot` moved from small/interior values to pinned at their upper bounds too. Damping parameters, by contrast, stayed pinned at roughly the same bounds at both rates
- **Root cause, confirmed with a controlled synthetic test, not just inferred from the real numbers:** sub-stepping fixes integration *stability*, but a SEPARATE issue remains if `dt` is mislabeled — the residual comparison implicitly assumes the simulated state after one `dt` of integration should match the real logged state one row later. If the *assumed* `dt` is smaller than the *true* elapsed time between logged rows (exactly the old 50Hz-vs-actual-10Hz bug), the simulator is compared against a real target that moved further than the simulated advance covered. Added mass directly scales acceleration (`v_dot = tau/M_diag`), so it's the parameter most directly forced to shrink to compensate — synthetic test with a KNOWN true added-mass of 15.0 confirmed this exactly: fitting with the correct dt recovered 15.000 (near-exact); fitting with dt mislabeled 5x too small (replicating the old bug) collapsed the fit to 0.000 — pinned at the lower bound, the same failure mode seen in the real `Yv_dot` result at 50Hz
- **Why damping wasn't similarly distorted:** damping was already pinned at its bounds at BOTH rates for the separate, pre-existing linear/quadratic-correlation reason (findings #4, #10) — there was no interior room for a dt-driven shift to become visible. Added mass had genuine interior freedom at 50Hz, which is exactly why it had room to reveal how sensitive it actually is
- **Consequence:** every added-mass value from findings #1–10 (all computed at 50Hz) should be treated as more severely biased than "just the wrong overall scale" — specifically pushed toward the lower bound by this mechanism, not just uniformly off. The 10Hz added-mass values are much more likely to reflect genuine physics, though the usual caveats (stand-in model, missing Coriolis, thruster-curve simplification) still apply
- Full comparison and synthetic validation script available on request — not yet saved as a standalone reusable script, run ad hoc during this investigation

---

## Honest open items

- **Per-evaluation runtime has a plausible lead but isn't confirmed** — see finding #9; likely the trust-region linear algebra cost, not directly profiled to prove it
- **Rotational dynamics gap is a model limitation, not a tuning problem** — further bound-widening or parameter search on the stand-in model is unlikely to resolve it; this is exactly what HoloOcean's real physics (once installed) is expected to fix

---

## What's actually next

1. **~~Re-run the stand-in-model calibration at the corrected 10Hz rate~~ — DONE, see finding #12.**
2. **~~Get HoloOcean installed and running~~ — DONE.** HoloOcean v2.3.0 running on Windows, `HoveringAUV` confirmed producing real scripted trajectories. Phase 0 complete
3. **Swap the stand-in for HoloOcean** in `dynamics_model.simulate_trajectory()` — the pipeline (`objective.py`, `optimizer.py`, `run_calibration.py`, `diagnose_calibration.py`) is built to not need changes when this happens. **This is the actual next step**
4. **Re-run calibration against real HoloOcean physics** — this is when the actual MVP question ("does calibration improve sim-to-real accuracy") gets a real answer, not an approximate one from the stand-in
5. **Re-run `check_damping_identifiability.py`'s comparison against the new HoloOcean-calibrated Jacobian** (finding #10) — specifically check whether `Mq/Mqq`'s fitted correlation drops toward its kinematic baseline (0.764). A testable prediction from this session, worth confirming or refuting once real rotational physics is in the loop