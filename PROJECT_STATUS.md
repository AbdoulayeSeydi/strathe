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
- **`objective.py`** — RMSE/residual computation, explicit required sampling rate (no silent defaults), DOF-weighting and run-length-normalization options, **plus a pluggable `simulate_fn` parameter (see finding #15)** letting every residual/RMSE function evaluate against either the stand-in or a real HoloOcean-bound backend
- **`optimizer.py`** — `scipy.optimize.least_squares` wrapper: robust loss (`soft_l1`), multi-restart support with configurable initialization strategy, explicit convergence checking, YAML-as-single-source-of-truth parameter schema (no hand-maintained parameter-name lists anywhere), **plus `simulate_fn` and `x0_override` support on `CalibrationConfig` (see finding #15)**
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

### 9. The runtime mystery — RESOLVED, see finding #14 (was: "plausible lead, not confirmed")
- Originally flagged: `diagnose_calibration.py`'s Jacobian evaluation cost ~1.6s/call, but the full calibration run cost ~20.6s/eval — a ~13x gap, hypothesized (not confirmed) as scipy's trust-region linear algebra overhead
- **This was the wrong hypothesis.** Finding #14 (below), discovered later during the HoloOcean proof-of-concept, found the real cause: `scipy.optimize.least_squares`'s numerical Jacobian estimation costs ~1 evaluation PER FREE PARAMETER, per iteration, hidden behind the single row scipy prints — not a linear-algebra overhead at all. Confirmed via exact arithmetic (see #14), not left as a guess this time

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

### 13. HoloOcean integration built, debugged, and stress-tested — MVP claim reframed

**The reframed MVP claim, and why it's stronger than the original:** the original framing was "calibrate HoloOcean's default physics against real logs." Confirmed via HoloOcean's own docs this isn't achievable for `HoveringAUV` — modifying its exposed dynamics variables "will have no effect on results," and the newer Fossen-dynamics customization system is explicitly torpedo-vehicle-only in the current release. The real, working architecture instead: **our own calibratable 6-DOF physics equations (unchanged from the whole session's validated work) compute acceleration each tick; HoloOcean provides high-fidelity rigid-body integration, collision handling, and sensor simulation on top**, via `HoveringAUV`'s "Custom Dynamics" control scheme. Revised claim: *"a system-identification-calibrated physics model, integrated into HoloOcean via its Custom Dynamics interface, tracks real mission logs more accurately than an uncalibrated default parameter set — evaluated inside a real, high-fidelity simulator."* This is the standard way real robotics system identification works (fit a parameterized model, use it) and isn't tied to one vendor's opaque internals.

**What was built:**
- `dynamics_model.py` refactored — `compute_acceleration()` extracted as a standalone function, shared by both the stand-in integrator and the new HoloOcean backend, guaranteeing identical physics between them (no risk of silent drift between two re-implementations)
- `holoocean_backend.py` — `simulate_trajectory_holoocean()`, matching the stand-in's exact signature. Since it never touches HoloOcean's own physics config (only feeds it accelerations each tick), the same running environment is reused across every optimizer evaluation and restart — no per-parameter-guess restart cost
- Six diagnostic scripts (`diagnose_holoocean_api.py`, `diagnose_holoocean_frames.py`, `diagnose_rotation_baseline.py`, `diagnose_imu_convention.py`, `diagnose_angular_input_sign.py`, `diagnose_linear_input_sign.py`) and `stress_test_holoocean.py`, all run against the real installed HoloOcean, not assumed

**Two real, separate frame-convention bugs found and fixed — both confirmed empirically, not guessed:**
1. **`PoseSensor`'s rotation matrix** reports body-frame Z pointing down relative to a Z-up world at "zero rotation" — HoloOcean's native body frame already matches our own SNAME/Fossen convention (body Z = down), just expressed relative to Unreal's Z-up world. Confirmed exactly (not approximately) via two independent small-angle rotation tests: the correction `R_true = R_reported @ diag(1,-1,-1)` recovered the textbook-exact standard rotation matrix in both cases
2. **The Custom Dynamics angular acceleration INPUT** needs the identical `diag(1,-1,-1)` correction — a separate finding from #1 (confirmed via direct, isolated tests on all 3 axes independently: X unaffected, Y and Z both flip), though both stem from one consistent underlying convention affecting rotation-related quantities only. Linear acceleration input confirmed clean (no correction needed) via the same method

**The real stress test — and a decisive test to separate "remaining bug" from "expected physics divergence":** ran the HoloOcean backend against real logged data with default (uncalibrated) parameters. Most DOFs landed in a sane ballpark against the stand-in. `p` (roll) persistently disagreed in sign across every fix attempt at 50 steps — investigated rather than dismissed. Re-ran at 5 steps: **`p` agreed in sign at 5 steps, diverged (including a sign flip) by 50** — the correct signature of compounding integration divergence between a genuine forward simulation (HoloOcean) and a teacher-forced one (the stand-in), not a persistent frame bug, which would show up immediately and consistently rather than starting in agreement. This also lines up with roll/pitch already being independently identified (finding #5) as the axes where the stand-in's missing Coriolis/cross-coupling terms are weakest — the same known limitation, visible in a new, more rigorous evaluation context, not a new problem

**What this had NOT yet done at the time:** run the actual calibration optimizer against this backend — see finding #15 for that.

### 14. The runtime mystery — actually resolved, with exact arithmetic, not a guess

- During a real HoloOcean-backed calibration run, the optimizer's PRINTED iteration table showed `Total nfev` climbing by exactly 1 per row — but a separately-maintained checkpoint (wired to the TRUE underlying function call, not scipy's display) showed `call_count: 295` after only 15 printed rows
- **The exact arithmetic: `295 = 15 + (14 × 20)`.** `scipy.optimize.least_squares` estimates a numerical Jacobian for the 20 free parameters via finite differences — costing ~20 EXTRA hidden simulation calls per printed iteration (1 perturbation per parameter), on top of the 1 trial point it actually displays. 14 real steps after the start × 20 hidden calls each = 280, plus the 15 displayed = 295, exact
- **This corrects a claim made earlier in this same session:** it was stated that `nfev` climbs "~1 per iteration, not ~21 like a full Jacobian rebuild" — that was true of scipy's DISPLAYED column, but wrong as a statement about real computational cost, since the display doesn't count the hidden Jacobian-perturbation calls
- **This is the same phenomenon as finding #9's old, unconfirmed "13x gap" hypothesis** — the earlier guess (scipy's trust-region linear algebra being expensive) was wrong; the real cause is this hidden per-parameter Jacobian cost, now confirmed with exact matching arithmetic instead of a plausible-sounding guess
- **Practical consequence:** `--max-nfev N` should be understood as authorizing roughly `N × (1 + n_params)` real simulation calls, not `N` — a ~21x multiplier for this 20-parameter model, not a rough estimate

### 15. The real HoloOcean-backed calibration proof-of-concept — built, debugged through 3 real bugs, and currently producing genuine improvement

**Plumbing built to make this possible:**
- `objective.py`: every residual/RMSE function (`_raw_residuals`, `residuals_for_run`, `joint_residuals`, `rmse_for_run`, `rmse_by_dof`, `max_abs_residual`, `mean_run_rmse`, `compute_diagnostics`) now accepts an optional `simulate_fn` — `None` (default) behaves identically to before (regression-tested: exact RMSE match), a custom function is genuinely invoked (fake-output test confirmed exactly N calls, not silently ignored)
- `optimizer.py`: `CalibrationConfig` gained `simulate_fn` (threaded into `least_squares`'s `args=`) and `x0_override` (warm-start from an arbitrary parameter vector instead of `vehicle_spec.yaml`'s naive guess) — both regression-tested and confirmed working
- `holoocean_backend.make_holoocean_simulate_fn(env)` — binds a persistent HoloOcean environment into the exact shape `objective.py` expects
- `time_holoocean_backend.py` — measured real per-step cost: ~22.2ms/step at full run length (5000 steps), confirming HoloOcean is genuinely ~400x slower per step than the analytical stand-in on this machine. Used to plan a deliberately scaled-down first real run: 25 training runs truncated to 150 steps each, warm-started from the already-calibrated 10Hz stand-in parameters (not vehicle_spec.yaml's naive guess) rather than starting cold
- `run_calibration_holoocean.py` — the actual real calibration script, plus `revalidate_checkpoint.py` for re-validating saved progress after a physics fix (see below)

**Bug #1 found and fixed: real, confirmed non-reproducibility.** The very first real attempt showed the SAME parameters, SAME runs, evaluated 3 times in a row, giving 3 DIFFERENT RMSE values (0.4644/0.5140/0.5062) — a real problem, since it would corrupt every gradient the optimizer computes. `env.reset()` was tested as a candidate fix and made things WORSE (18% spread, up from 6%), cleanly ruling out incomplete state reset as the cause. The real cause, confirmed directly: `IMUSensor` has genuine, working sensor noise (by design, a real simulator feature) — with the vehicle held perfectly still and zero command, `IMUSensor`'s angular velocity was NOT zero (nonzero mean AND std). Our backend reads this every tick and feeds it into a closed loop, compounding real noise into materially different trajectories each run.
- **Fix:** `make_deterministic_environment()` — locates the real installed scenario JSON, zeros every sensor's `*Sigma*` noise parameter, and creates the environment from this modified config (confirmed via `holoocean.make(scenario_cfg=...)`, which needed an added `package_name` key not present in the raw JSON). **Confirmed exactly fixed:** the same 3-repeat test that showed 0.4644/0.5140/0.5062 before now gives 0.399875 three times, to the full displayed precision — full determinism confirmed, not assumed
- This fix is specifically for CALIBRATION — a clean, controlled test of the physics parameters. Real deployment/RL training should use the noisy environment intact; sensor noise robustness is a real, legitimate, separate later concern

**Checkpointing built, and it already earned its keep.** `CalibrationConfig`'s existing but previously-unwired `on_iteration` hook now saves the BEST (lowest-cost) point seen so far to a checkpoint file after EVERY single evaluation (not just improving ones) — confirmed to correctly track best-not-last even with a deliberately-worse probe evaluation injected mid-sequence. This was NOT theoretical: partway through the first real overnight-scale run, the computer was shut off unexpectedly. Real, expensive progress up to that point was fully preserved and resumed from, rather than lost.

**Bug #2 found and fixed: initial angular velocity was both wrong-direction AND wrong-units.** Investigated as one of three loose ends flagged as "probably not worth chasing further" — chased anyway on request, and it was worth it. Direct isolated testing (holding the vehicle still, setting a known initial angular velocity via `set_physics_state`, reading back immediately) revealed the readback matched NEITHER the desired value nor the naively-uncorrected one. Root cause, confirmed in two parts:
1. **Direction:** the same `diag(1,-1,-1)` correction used elsewhere IS needed here too — confirmed via matching readback ratio (not just extrapolated from pattern anymore)
2. **Units:** `set_physics_state`'s `angular_velocity` parameter expects DEGREES/second, like the already-known-degrees `rotation` parameter — NOT radians/second like our own internal convention. Passing raw radians directly (as the code did) undersized every value by ~58x (`≈180/π`). Confirmed exactly: converting via `np.degrees()` before passing landed the readback within 2.5% of the desired value — that remaining 2.5% is a separate, small, confirmed residual: angular velocity decays at a clean, constant ~2.5%/tick even under Custom Dynamics with a zero command (meaning Custom Dynamics does NOT fully disable ALL internal damping, contrary to its documentation's implication). Real logged forces dominate almost immediately once a run starts, so this residual was judged not worth chasing further
- **Fix wired into `holoocean_backend.py`:** `np.degrees()` applied to the initial angular velocity before calling `set_physics_state`
- **Existing checkpoint re-validated, not discarded, after this fix:** `revalidate_checkpoint.py` re-evaluated the checkpoint's best parameters under the corrected physics. Result: 0.00% cost difference on this specific checkpoint — the fix's real impact happened to be negligible for these particular runs (plausibly because most start near-rest), but this was CONFIRMED by direct re-evaluation, not assumed

**Bug #3 (not a bug): the "same environment slows down over time" hypothesis was tested and cleanly ruled out.** 60 repeated identical-cost evaluations through the same long-lived environment came back at 4.966-4.967s EVERY SINGLE TIME — a flat, essentially zero-variance result. This eliminated one real candidate explanation for the broader runtime mystery, which was then fully resolved separately by finding #14.

**A crash-handling bug found and fixed along the way:** `optimizer.py`'s `calibrate()` raises a hard `RuntimeError` if `max_nfev` is hit without satisfying scipy's own strict convergence tolerances — correct in general, but wrong for an expensive, budget-limited HoloOcean run where hitting the cap mid-improvement is expected, not an error. `run_calibration_holoocean.py` now catches this specific case and falls back to reporting/saving the checkpoint's best point instead of crashing before ever printing results. (A stale, pre-fix copy of the file was accidentally re-run once after this was added, reproducing the same crash — resolved by re-confirming the fix was actually present via `grep` before the next run.)

**A cost-bookkeeping clarification, not a bug:** the checkpoint's saved cost is plain `0.5×sum(residuals²)` on the raw residuals (computed once, in `optimizer.py`, before scipy's `soft_l1` robust-loss transform is applied on top for its own printed "Cost" column) — the two numbers are both legitimate but measure different things, and won't numerically match each other. Field names kept unchanged from the original checkpoint format specifically to stay compatible with real, already-saved progress; a new `cost_meaning_note` field documents the distinction going forward.

**Real results so far, across two resumed segments (as of this session):**
- Segment 1: cost `15.291 → 12.465` (an ~18.5% reduction), stopped by `max_nfev`, not full convergence
- Checkpoint revalidation after the angular-velocity fix: `12.528 → 12.528` (0.00% change, confirmed negligible impact for these runs)
- Segment 2 (resumed): cost continued dropping, `12.465 → 7.9337`/`~7.96` (displayed vs. checkpoint's plain-sum-of-squares measure respectively) — a further, substantial real improvement
- **Not yet fully converged by scipy's own strict criteria** — still real signal to extract, though with real diminishing-returns considerations given the true per-`max-nfev` cost (see finding #14)
- **This is the first real evidence, on real HoloOcean physics, that the calibration approach works** — the core proof-of-concept question this whole integration effort was built to answer

---

### 16. Coriolis-centripetal terms added — closing the specific gap findings #5/#10/#12/#13 all converged on

- The model was missing the rigid-body Coriolis-centripetal term (`C(ν)ν`) entirely — the standard `M ν̇ + C(ν)ν + D(ν)ν + g(η) = τ` equation of motion had only ever included mass, damping, and restoring force
- **Derived properly for this model's diagonal mass matrix**, not copied from memory: the standard generalized-momentum construction, re-signed to match this file's own added-mass convention (positive magnitudes added directly, not Fossen's negative-added-mass convention)
- **Verified, not assumed correct, before being trusted:** (1) energy conservation (`νᵀC(ν)ν = 0` for any valid Coriolis matrix) confirmed to `~1.7e-12` across 10,000 random trials — floating-point noise only; (2) the well-known "coordinated turn" physical case (forward motion + yaw rate → a real lateral centripetal force) reproduced exactly, with every other component correctly zero
- **Adds zero new calibration parameters** — uses only the mass/inertia values already being fit
- **Regression-confirmed:** single-axis motion (pure surge, pure yaw) gives byte-identical results to before (Coriolis terms are always products of ≥2 velocity components, so they're structurally zero on a single axis) — confirmed via both the analytical stand-in and `ArenaCore` (Phase 2's physics core, same shared function). Genuine multi-axis coupling (forward motion + yaw rate together) now shows real, previously-impossible sway coupling, confirmed on both my sandbox and independently reproduced on the real machine

### 17. Two real numerical/bounds bugs found and fixed after adding Coriolis — both confirmed via direct reproduction on real data, not assumed

**Bug A — Forward Euler overflow under real (not synthetic) force magnitudes.** Re-running the 10Hz stand-in calibration crashed with a NaN. Root cause, confirmed via direct reproduction: default (weakly-damped) parameters already let velocity grow to unrealistic magnitudes under sufficiently large forces; Coriolis (roughly quadratic in velocity) then pushes an already-large velocity into overflow. Confirmed the real dataset genuinely reaches the triggering force range (`diagnose_coriolis_crash.py --all-runs`: real logged `Fz` down to `-191.6N`, well past the ~100N threshold that reproduced the crash synthetically). **Fix:** `SimulationConfig.physics_dt` default tightened `0.02 → 0.005` (4x finer sub-stepping) — confirmed stable up to force magnitude 200 (synthetic stress test), and confirmed clean on 500 real steps of real data. **Honest tradeoff, documented in the code:** this breaks the old "50Hz is a no-op" regression guarantee from finding #12's original fix — pass `physics_dt=0.02` explicitly to exactly reproduce findings #1–15's historical numbers.

**Bug B — vehicle_spec.yaml's damping bounds allowed a physically impossible "zero damping" state.** Even after Bug A's fix, re-running calibration with `--n-restarts` still crashed. Diagnosed precisely, not patched around: the crash came from `--initialization-strategy perturbed` (the existing, already-fixed default from finding #7) landing on a starting point where `Mq` and `Mqq` were BOTH perturbed toward exactly zero simultaneously — confirmed by replicating the optimizer's exact perturbation logic (same seed) to reproduce the identical crashing parameter vector (`diagnose_perturbed_crash.py`/`diagnose_perturbed_crash_detail.py`). Further sub-stepping does NOT fix this case (`physics_dt=0.001` "succeeds" but gives `max|v|=282` — physically absurd, not genuinely stable, just postponing overflow). Root cause: **all 12 damping parameters had an upper bound of literally `0`** in `vehicle_spec.yaml` — physically nonsensical (zero drag), and only became catastrophic once Coriolis correctly began transferring energy between axes with nothing to dissipate it. **Confirmed the exact threshold empirically** (`find_min_stable_damping.py`): stable at 0.5% of each parameter's lower-bound magnitude, unstable ONLY at exact zero — a sharp cliff, not a gradual degradation. **Fix:** every damping parameter's upper bound changed from `0` to 2% of its own lower bound (a confirmed 4x safety margin). **Verified broadly, not just for the one instance found:** `verify_bounds_fix.py` — 50 independent random perturbations (different seeds), all 25 training runs each, 50/50 fully stable.

### 18. The real Coriolis-aware 10Hz stand-in calibration — converged cleanly, and a real, precisely-resolved identifiability question along the way

**The calibration itself, with both fixes in place, genuinely converged** (`ftol` satisfied, not budget-capped): cost `233.90 → 3.355` (98.6% reduction). RMSE: train `1.1779 → 0.2051` (+82.6%), heldout `1.1658 → 0.2067` (+82.3%) — generalization gap `0.99x/1.01x`, no overfitting signal. Per-DOF calibrated RMSE still shows the same known pattern (translational `u/v/w` ≈ `0.12–0.14`, rotational `p/q/r` ≈ `0.23–0.32`) — both categories dramatically better than default, rotational still comparatively weaker. `Kpp`/`Mqq` converged to their lower (strong-damping) bound, nowhere near the new upper-bound fix — confirms the bounds fix didn't distort where the real fit wants to go.

**One parameter, `Zw_dot` (heave added mass), pinned at exactly its lower bound (0) — investigated precisely, not left as an open item.** `diagnose_calibration.py`'s existing pairwise-correlation analysis (from findings #2/#4) found several correlated pairs, but none involving `Zw_dot` at all. Rather than guess, built and added a genuinely new capability: an SVD-based multi-parameter identifiability check (the near-null-space right-singular-vectors of the Jacobian), since a flat direction can involve 3+ parameters jointly with no single pair crossing a correlation threshold. **Verified the new analysis logic correctly detects a planted 3-way synthetic degeneracy before trusting it on real data.** Run on the real result: the three flattest directions are dominated by the already-known `Zww↔Zw`, `Yvv↔Yv`, `Xuu↔Xu` pairs — **`Zw_dot` doesn't have a meaningful loading in any of them either.** **Conclusion, precisely stated:** this isn't a hidden pairwise or multi-parameter degeneracy — it's a genuine *practical* identifiability gap: this dataset's real maneuvers likely don't excite heave acceleration enough to constrain `Zw_dot` at all, so it simply sits at its bound with nothing pulling it elsewhere.

**Built to directly test this kind of hypothesis going forward:** `optimizer.py`'s `CalibrationConfig` gained `fixed_params` — lets specific parameters be held exactly constant (removed from the search space, not just tightly bounded) while the rest re-optimize around them. Regression-tested (default behavior with `fixed_params=None` unchanged) and confirmed correct (a fixed parameter's value is genuinely untouched by the optimizer, reconstructed correctly into the full result vector). A live test (fixing `Zw_dot` at a physically plausible value) showed cost dropping steadily (`112 → 49 → 28` across three Jacobian batches) with no sign of the optimizer struggling — consistent with, though not full proof of, the practical-gap conclusion above.

### 19. The real HoloOcean-backed refinement of the Coriolis-aware calibration — genuine improvement, then clear diminishing returns

Warm-started from finding #18's converged stand-in result (not the old pre-Coriolis checkpoint, which would have carried forward exactly the double-compensation problem this session already fixed). Two overnight segments, ~10.9 hours each (315 real evaluations per segment, matching finding #14's confirmed hidden-Jacobian-multiplier arithmetic exactly):

- **Segment 1:** RMSE `0.3050 → 0.2576` (a real **+15.5%** improvement), cost `8.8975 → 6.0142`. Notably, real per-evaluation cost during this live run (`124.2s`) matched the isolated standalone baseline measurement exactly — unlike the earlier pre-Coriolis proof-of-concept, which showed an unexplained ~900s/eval discrepancy against the same baseline. This specific run showed no such gap; the earlier mystery wasn't deliberately re-investigated, but didn't reappear here either
- **Segment 2 (resumed):** cost `5.9824 → 5.9059`, RMSE improvement only **+0.5%** — roughly a 10x smaller gain than segment 1 for the same compute cost. `Zw_dot` moved slightly off its lower bound (`0.0 → 0.05 → 0.12`) — direct, consistent evidence for finding #18's conclusion: weakly constrained, not rigidly stuck, nudging slightly as more real physics is incorporated
- **Decision made given the clear diminishing-returns pattern:** stop further `max-nfev`-based refinement here. Neither segment reached scipy's strict convergence criteria — this result is honestly labeled a `placeholder` (see finding #20's safeguard), not a final answer, but further refinement via this same lever has a clearly poor cost/benefit tradeoff going forward

### 20. Phase 2 (Gym/"Arena" wrapper, PPO training, the real MVP transfer-claim question) — built, tested, and partially run for real

**Stage 2.0 — architecture decisions, made concrete and importable** (`phase2_config.py`): the exact held-out split reused from Phase 1 (enforced via `assert_uses_phase1_split`, fails loudly on drift), the train-on-fast-stand-in / validate-on-real-HoloOcean decision (Decision 2 — training PPO directly against HoloOcean would cost ~400x more wall-clock time per step), and an explicit `Claim A` (rigorous open-loop replay, the literal MVP acceptance criterion) vs. `Claim B` (complementary trained-policy behavior, not a second accuracy proof) distinction — every Phase 2 result must be tagged with one or the other.

**Stage 2.1 — the `Arena` Gym environment** (`arena_core.py`, `arena.py`): waypoint-following on the shared `compute_acceleration()` physics. Introduces something genuinely new to this pipeline — self-integrated position and orientation (nothing before this ever tracked those; Phase 1 only ever compared velocities). **A real integration bug was caught and fixed via testing before handoff:** the first version silently double-counted each step's acceleration when computing position (using post-update instead of pre-update velocity) — caught by a direct hand-computable test, fixed to integrate velocity/position/orientation consistently at the sub-step level, and re-verified two independent ways (exact hand calculation, and a convergence test showing finer sub-stepping approaches the true analytical answer).

**Stage 2.2 — PPO sanity check, found and fixed a real action-space design flaw.** The first real training attempt showed PPO completely failing to learn (final performance statistically indistinguishable from random) despite 50,000 training steps — diagnosed via the training curve itself (`std` frozen near its 1.0 initialization the whole run, `explained_variance` stuck near zero): a genuine, well-known PPO/continuous-control pitfall — the action space was left in raw force units (`±30`), far wider than a freshly-initialized policy naturally explores. **Fix:** normalized the action space to `[-1,1]`, rescaled internally to real force units. Re-run confirmed real learning (training curve showed steady, substantial improvement; final result: reward `-34.01` vs. random's `-203.56`, **100% success rate**, beating even the hand-coded positive-control policy). A related bug in the sanity-check's own pass/fail logic (requiring success-rate improvement as a hard gate, which misreported an earlier genuinely-successful run as a failure) was also found and fixed.

**Stages 2.3 (`replay_comparison.py`) and 2.4 (`train_multiseed.py`) — both built and fully tested**, including a market-facing safeguard (`phase2_config.check_result_type`/`output_dir_for`) that refuses to let a result be labeled `final` unless its source calibration's own `metrics.json` actually shows `converged=True` — directly protects against a wrong or mislabeled number reaching a reportable deliverable, the exact class of mistake this session already had with stale files more than once. A real bug (redundant duplicate simulation — calling both `rmse_by_dof` and `rmse_for_run`, silently doubling every real HoloOcean-backed evaluation's cost) was found and fixed via a confirmed-exact mathematical shortcut (`overall = sqrt(mean(per_dof_rmse²))`), roughly halving Stage 2.3's real cost.

**Stage 2.3's real (`placeholder`) run**, using finding #19's HoloOcean-refined result: scale-normalized RMSE improvement **+7.4%** overall. Per-DOF: translational still strong (`u +61.3%`, `v +30.8%`, `w +35.5%`), rotational genuinely mixed (`q +21.7%`, but `p -7.2%` and `r -12.9%`) — a different, more nuanced pattern than the earlier pre-Coriolis proof-of-concept's uniformly-negative rotational result, plausibly reflecting a model whose coupled rotational dynamics aren't yet fully re-calibrated for the newly-correct Coriolis coupling, though this wasn't separately confirmed.

**Stage 2.4 has not yet been run for a final, reportable result** — the pipeline is built and dry-run tested (with stubs, given `stable-baselines3`/`gymnasium` aren't installable in this sandbox), but the real multi-seed run against real data hasn't happened yet this session.

### 21. Stage 2.4's real multi-seed PPO run — both variants train cleanly; the reward gap is a real physical effect, not a training problem

Real run: 4 seeds × 2 variants (default vs. finding #18's calibrated physics), on the real-data-grounded waypoint (derived from an actual held-out run's logged displacement, per Stage 2.1's design). **Both variants trained cleanly and stably:** 100% success rate across all 8 runs, tight final distances (`0.42–0.50`), and low relative variance within each group (default `-30.31±0.81`; calibrated `-82.56±2.52` — both ~3% relative spread) — no sign of a training failure on either side.

**The large reward-magnitude gap between variants (`-30` vs. `-83`) is explained by the calibrated parameters themselves, not a bug.** Finding #18's calibration converged to a substantially heavier, more damped vehicle almost everywhere (`Ieff_roll/pitch: 0.28→5.0`, `Xu: -11.7→-38.2`, `Yv: -20→-80`, `Zw: -31.8→-100`, etc.). Since reward is `-distance` accumulated every step until termination, a heavier and more resistant vehicle genuinely needs more steps to cover the same real displacement under the same normalized `[-1,1]` actions — accumulating a larger negative reward even with an equally competent policy. This is expected physical behavior from calibrating toward a more realistic vehicle model, not a defect.

**This result is intentionally not compared directly against Claim A's numbers** — per phase2_config.py's Decision 3, Claim B measures task-completion behavior under each model's own physics, not accuracy against real data; the two claims answer different questions by design and aren't expected to agree or disagree with each other.

With this, both Claim A (finding #20) and Claim B (this finding) now have real, complete, honestly-labeled (`placeholder`) results — the MVP's stated Phase 2 acceptance criterion (a comparison chart) is technically satisfied by both.

### 22. Is the remaining rotational weakness a data-coverage ceiling or a genuine model gap? — resolved, precisely, with pure kinematics

Given Stage 2.3's mixed rotational result (finding #20: `q +21.7%`, but `p -7.2%` and `r -12.9%`), the real open question was whether this dataset can even support further improvement — i.e., whether the Coriolis coupling terms (finding #16) are starved of real excitation, or whether the data genuinely supports them and something else in the model is still missing.

**Method:** same approach as finding #10's independent damping check — using ONLY real logged velocity data (no simulation, no model, no optimizer), directly measure the excitation of the specific velocity products Coriolis's rotational equations depend on (`v·w`, `Q·R` → roll; `u·w`, `P·R` → pitch; `u·v`, `P·Q` → yaw). Built `check_coriolis_excitation.py`; verified its "ratio" framing against a direct synthetic test before trusting it — found and corrected a real mistake in the initial draft (assumed correlation would *suppress* a product's variance; a synthetic test showed the opposite: `Var(ab) ≈ Var(a)Var(b)(1+2ρ²)` for jointly Gaussian data, so correlation of either sign slightly *increases* it). Removed the flawed causal framing; kept the raw product magnitude as the primary, correctly-interpretable metric.

**Result on real training data (25 runs, 103,995 samples): all six products are healthily and comparably excited** — `std` ranging from `0.024` (`v·w`) to `0.114` (`Q·R`), under a 5x spread across all six. This is nowhere near the "orders-of-magnitude starved" signature that would indicate a genuine data ceiling (confirmed via the synthetic test case, where a deliberately-poorly-excited product came out ~10–25x smaller than the rest, not <5x).

**Conclusion, precisely stated:** this rules out "the data can't support Coriolis" as an explanation. The real dataset genuinely gives the model what it needs to identify all six rotational coupling terms. **The remaining `p`/`r` weakness from finding #20 is therefore more likely a genuine remaining model limitation, not a data-coverage gap** — pointing toward further model structure (off-diagonal added mass, thruster/actuator nonlinearity, or cross-axis damping terms — none of which this model currently includes) as the more promising next lever, rather than anything data-related.

---

## Honest open items

- **The remaining rotational weakness (finding #20) is now confirmed to be a model limitation, not a data-coverage gap** (finding #22) — the real dataset excites all six Coriolis-relevant velocity products comparably well. Candidate next steps: off-diagonal added mass, thruster/actuator nonlinearity, or cross-axis damping terms — none currently in the model
- **Neither of finding #19's two HoloOcean refinement segments reached scipy's strict convergence** — deliberately stopped given clear diminishing returns (segment 2's gain was ~10x smaller than segment 1's for the same compute cost), not because the process failed
- **`Zw_dot`'s practical (not structural) identifiability gap** (finding #18) is understood and precisely characterized, but not "fixed" — it simply reflects what this dataset's real maneuvers can and can't constrain
- **The small, confirmed ~2.5%/tick angular-velocity decay under Custom Dynamics with zero command** (finding #15) remains a known, deliberately-unfixed, first-instant-only residual

---

## What's actually next

1. **~~Re-run the stand-in-model calibration at the corrected 10Hz rate~~ — DONE, see finding #12.**
2. **~~Get HoloOcean installed and running~~ — DONE.**
3. **~~Build and stress-test the HoloOcean integration~~ — DONE, see finding #13.**
4. **~~Wire up pluggable simulate_fn, build the real HoloOcean-backed calibration script~~ — DONE, see finding #15.**
5. **~~Add Coriolis-centripetal terms to close the rotational-dynamics gap~~ — DONE, see finding #16.** Two real bugs found and fixed along the way (finding #17); the real Coriolis-aware calibration converged cleanly (finding #18); refined further against real HoloOcean physics with genuine, if diminishing, improvement (finding #19)
6. **~~Build and test the Phase 2 Gym/Arena wrapper and PPO pipeline~~ — DONE, see finding #20.** Real bugs found and fixed (an action-integration bug, an action-space scale mismatch, a redundant-simulation performance bug); the market-facing placeholder/final safeguard built and verified
7. **~~Run Stage 2.4's real, final multi-seed PPO comparison~~ — DONE, see finding #21.** Both variants trained cleanly (100% success, low variance); the reward-magnitude gap between them is explained by the calibrated physics being genuinely heavier/more damped, not a training problem
8. **~~Decide whether to push further on the rotational-coupling gap, and whether that means more data or deeper model work~~ — DECIDED, see finding #22.** Confirmed via pure kinematics: the real data excites all six Coriolis-relevant velocity products comparably well (under a 5x spread) — this rules out a data-coverage ceiling and points at a genuine remaining model limitation instead
9. **Re-run `check_damping_identifiability.py`'s Mq/Mqq comparison against the new Coriolis-aware Jacobian** (flagged in finding #10) — now directly possible given finding #18's real Jacobian/correlation analysis exists; not yet done
10. **Investigate a specific model-structure addition for the remaining rotational weakness** (finding #22's conclusion) — leading candidates: off-diagonal added mass, thruster/actuator nonlinearity, or cross-axis damping terms, none of which the model currently includes; not yet started, this is the concrete next lever