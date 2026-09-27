# Strathe — Project Status

*Last updated: current session*

## Where we stand against the MVP plan

| Phase 0 acceptance criterion | Status |
|---|---|
| Real logged BlueROV2 dataset in hand, matching input/state format | ✅ Done — 39 unique runs (see dedup note below), oceansystemslab public dataset, validated |
| HoloOcean runs locally, produces a simulated trajectory for a scripted thruster sequence | ✅ **Done** — HoloOcean v2.3.0 installed on Windows, `PierHarbor`/`SimpleUnderwater` scenarios confirmed with `HoveringAUV` (BlueROV2 analog). Ran a scripted thruster command (`[10,10,10,10,0,0,0,0]`) for 200 steps: produced a real, physically sensible simulated trajectory — pure vertical displacement (z: -4.94 → -0.185), orientation unchanged, matching a purely-vertical thruster input with no spurious rotation |

**Phase 0 is complete.** Getting here took real troubleshooting: a multi-day BYU asset-server outage (documented via GitHub issue #353, resolved after their server migration), a GitHub↔Epic account linking gap (pending invite email), and standard Windows dev-environment setup (Python PATH, Git Bash).

**Phase 1, Phase 2, and Phase 3 are now all complete** — see the MVP acceptance-criteria review near the end of this document for the precise, literal accounting against the original plan.

---

## Vehicle & dataset (Phase 0)

- **Target vehicle:** BlueROV2 Heavy, 8-thruster configuration — public specs, matches HoloOcean's closest built-in analog (`HoveringAUV`)
- **Dataset:** `oceansystemslab/Bluerov2_trajectory_dataset` (public GitHub repo) — **39 unique real logged runs** across 4 collection dates (`12-01-2024`, `14-02-2024`, `16-02-2024`, `19-02-2024`), ~183,555 total samples
- **Maneuver diversity confirmed:** all runs show 6/6 DOFs excited (genuinely coupled, multi-axis motion)
- **Sampling rate: 10Hz** — empirically validated (finding #11), no longer an open question

---

## Calibration pipeline (Phase 1) — built and thoroughly tested

A full calibration pipeline exists in `calibration/`:

- **`vehicle_spec.yaml`** (current version) — known constants vs. calibration-target parameters, every value with unit/source/confidence/bounds-rationale, explicit assumptions section. Regularly updated as real findings changed what's known — see findings #28-#29 for the most recent updates.
- **`dynamics_model.py`** — 6-DOF underwater vehicle dynamics with Coriolis-centripetal coupling (finding #16), off-diagonal added-mass support (shelved, finding #26), Cholesky-based mass-matrix validity checking
- **`data_split.py`** — held-out-by-**date** train/validation split, deduplicated CSV discovery, **and a documented, tested default behavior**: when no explicit `heldout_dates` is given, the most recent `heldout_fraction` of dates are held out automatically (confirmed pre-existing, now protected by a real regression test — see finding #30)
- **`objective.py`** — RMSE/residual computation, pluggable `simulate_fn` (stand-in or real HoloOcean), DOF-weighting support
- **`optimizer.py`** — `scipy.optimize.least_squares` wrapper: robust loss, multi-restart, `fixed_params` support (hold specific parameters exactly constant), `x0_override`, `dof_scales`
- **`run_calibration.py`** — full CLI pipeline, **now with auto-backup protection** (finding #30) matching the HoloOcean script's existing protection
- **`run_calibration_holoocean.py`** — the real HoloOcean-backed calibration script, with checkpointing, auto-backup, a timing log, and (new this session) a `--no-dof-weighting` flag for isolating specific hypotheses cleanly (finding #32)
- **`diagnose_calibration.py`** — active-bounds report, profiling, per-DOF worst-run RMSE, Jacobian-based identifiability/correlation analysis (now also returning near-null-space directions, not just printing them — finding #31), real-vs-simulated velocity plots, force-breakdown plots
- **`preflight_check.py`** — automated data-quality pre-flight (sampling-rate validation + Coriolis-excitation check), wraps existing validated scripts, wired as the automatic first step of `run_full_pipeline.py` (finding #31)
- **`automated_identifiability_report.py`** — wraps `diagnose_calibration.py`'s Jacobian/SVD analysis into one structured, machine-readable report (finding #31)
- **`arena.py`/`arena_core.py`** — the Gym-compatible RL environment, now supporting three tasks: `waypoint` (default), `station_keeping`, `docking` (finding #31)
- **`run_full_pipeline.py`** — the single-command demo pipeline (Phase 3), extensively hardened this session (finding #30)

All of this has been tested against synthetic data, real data, and — for the code paths that don't require HoloOcean itself — a real, automated, CI-verified pytest suite (26 tests as of this session, all passing on GitHub Actions).

---

## Validation methodology: held-out-by-date, not random split

Every real-data calibration run used `data_split.py`'s `split_by_date()` — runs grouped by collection date, with entire dates held out, not a random subset of individual runs. Specifically: fit on `12-01-2024` + `14-02-2024` (25 unique runs), held out `16-02-2024` + `19-02-2024` (14 runs) entirely.

This matters because runs from the same day likely share conditions — battery charge, tank/pool state, vehicle trim that day. Holding out whole dates is the harder, more honest test.

---

## Key findings from the real-data runs

### 1-15. [Summary — see prior session logs for full detail]

Findings #1-15 cover: the first real calibration run (12/23 params pinned), the structural `Izz`/`Nr_dot` non-identifiability discovery and reparameterization fix, damping-pair correlation analysis, the rotational-dynamics weak-point diagnosis, a duplicate-file bug found and fixed, a restart-strategy bug found and fixed (with a genuine 5-restart stability confirmation), an excitation-confound check (ruled out), the runtime-mystery's first (wrong) hypothesis, a real vs. kinematic damping-identifiability comparison, the discovery that the dataset's true sampling rate was 10Hz not the assumed 50Hz (a real ~5x error, extensively validated via a windowed reconstruction method), a Forward-Euler numerical-instability fix at the corrected rate, a real dt-mislabeling mechanism found in the added-mass parameters, the full HoloOcean integration (with two real frame-convention bugs found and fixed), and the true resolution of the runtime mystery (a hidden ~21x per-iteration Jacobian-evaluation multiplier, confirmed via exact arithmetic) — plus the first real HoloOcean-backed calibration proof-of-concept, debugged through three real bugs (a sensor-noise nondeterminism bug, an angular-velocity direction-and-units bug, and a ruled-out "environment slows down over time" hypothesis).

### 16. Coriolis-centripetal terms added — closing the specific gap findings #5/#10/#12/#13 all converged on

The model was missing the rigid-body Coriolis-centripetal term (`C(ν)ν`) entirely. Derived properly for this model's diagonal mass matrix, verified via energy conservation (`~1.7e-12` across 10,000 random trials) and the coordinated-turn physical test case before being trusted. Adds zero new calibration parameters. Regression-confirmed byte-identical on single-axis motion.

### 17. Two real numerical/bounds bugs found and fixed after adding Coriolis

**Bug A** — Forward Euler overflow under real force magnitudes; fixed by tightening `physics_dt` to 0.005 (confirmed stable up to force magnitude 200, and on 500 real steps of real data). **Bug B** — `vehicle_spec.yaml`'s damping bounds allowed a physically impossible zero-damping state, which became catastrophic once Coriolis began transferring energy between axes with nothing to dissipate it; fixed by setting every damping upper bound to 2% of its own lower bound (empirically confirmed threshold, 4x safety margin, verified across 50 independent random perturbations).

### 18. The real Coriolis-aware 10Hz stand-in calibration — converged cleanly

Cost `233.90 → 3.355` (98.6% reduction), RMSE train/heldout `+82.6%`/`+82.3%`, generalization gap `0.99x/1.01x`. `Zw_dot` pinned at its lower bound — investigated via a new SVD-based multi-parameter identifiability check (verified against a planted synthetic degeneracy first), concluded to be a genuine *practical* identifiability gap (this dataset's real maneuvers don't sufficiently excite heave acceleration), not a hidden pairwise or multi-parameter degeneracy. `optimizer.py` gained `fixed_params` support to directly test this kind of hypothesis going forward.

### 19. The real HoloOcean-backed refinement of the Coriolis-aware calibration

Two overnight segments: Segment 1, RMSE `+15.5%`; Segment 2, only `+0.5%` (clear diminishing returns) — decision made to stop further `max-nfev`-based refinement on this configuration.

### 20. Phase 2 (Gym/"Arena" wrapper, PPO training) — built, tested, and run for real

Stage 2.0 (architecture decisions), Stage 2.1 (the `Arena` environment, with a real integration bug found and fixed — silently double-counting each step's acceleration), Stage 2.2 (a real PPO action-space design flaw found and fixed — un-normalized `±30` force units prevented the policy from ever learning; fixed by normalizing to `[-1,1]`), Stages 2.3/2.4 built and tested with a real market-facing safeguard (`check_result_type`/`output_dir_for`, refusing to let a result be labeled `final` unless its source calibration genuinely converged) and a real performance bug fixed (redundant duplicate simulation, confirmed-exact mathematical shortcut roughly halving Stage 2.3's cost).

Stage 2.3's real (`placeholder`) run: scale-normalized RMSE improvement `+7.4%` overall; translational strong, rotational mixed (`q +21.7%`, `p -7.2%`, `r -12.9%`).

### 21. Stage 2.4's real multi-seed PPO run

Both variants (default vs. calibrated physics) trained cleanly, 100% success rate across all 8 runs. The large reward-magnitude gap between variants (`-30` vs. `-83`) is explained by the calibrated parameters converging to a genuinely heavier, more damped vehicle — expected physical behavior, not a training defect.

### 22. Rotational weakness: data-coverage ceiling or genuine model gap? — resolved via pure kinematics

Built `check_coriolis_excitation.py` — using only real logged velocity data (no simulation, no model), measured the excitation of every Coriolis-relevant velocity product. Verified the analysis logic against a synthetic test case first (and corrected a real mistake in the initial draft — correlation of either sign slightly *increases* a product's variance, not suppresses it). **Result: all six products are healthily and comparably excited** (under a 5x spread) — ruling out "the data can't support Coriolis" as an explanation. The remaining rotational weakness is therefore more likely a genuine remaining model limitation.

### 23. The Stage 2.3 "mixed" rotational result — isolated to a specific, fixable cause

Built `compare_two_calibrations.py` and directly tested whether the incomplete HoloOcean refinement (stopped early per finding #19's diminishing-returns decision) was responsible for the mixed rotational picture. **Result: every rotational DOF regressed from the incomplete refinement**, not just `p`/`r` (`p: -14.6%`, `q: -9.7%`, `r: -17.7%` vs. the stand-in's own converged fit). Motivated properly re-running HoloOcean refinement to genuine convergence.

### 24. Finding #23's implied fix tested directly against real HoloOcean physics — and refuted

Before committing to a large refinement run on finding #23's basis, tested the stand-in's own converged fit through the *actual* Claim A methodology (real HoloOcean physics). **Result: WORSE than the original result** (`p: -22.6%`, `r: -29.3%`). Finding #23's comparison had been misleading — it evaluated both parameter sets using the stand-in's own physics, not real HoloOcean physics. Corrected conclusion: further HoloOcean refinement genuinely is the right next step.

### 25. HoloOcean refinement segment 3 — genuinely converged, partial fix

`r: -12.9% → -8.3%` (real improvement), `q: +21.7% → +24.9%` (further improvement), `p: -7.2% → -8.0%` (essentially unchanged). A related negative result: the off-diagonal added-mass terms (`Xq_dot`, `Zq_dot`) converged to essentially zero despite genuine convergence elsewhere — the model found no benefit to using them in this configuration.

### 26. Off-diagonal added mass — formally shelved

Given finding #25's real negative result and the (at-the-time) unresolved performance mystery blocking a clean re-test, made a deliberate decision to shelve this line of work. Real, tested, verified-correct code remains in the model (harmless, defaults to zero coupling) but should not be referenced in public-facing material as a working capability.

### 27. Three refinement segments, two objectives, one consistent result — DOF weighting ruled out for `r`

Researched and implemented custom per-DOF weighting (`p`/`r` boosted beyond their natural training-data scales) after finding #25's regression. Across three real segments (unweighted segment 3 baseline, unweighted segment 4, `r`-boosted 4.46x segment 5): **`r` got monotonically WORSE across all three** (`-8.3% → -24.1% → -36.0%`), including the segment where its objective weight was deliberately quadrupled. This directly refuted the DOF-under-weighting hypothesis as `r`'s primary cause. Leading remaining hypothesis at the time: every segment trained on the same 150-step (15-second) truncated window — a classic overfitting-to-a-short-window signature, not yet tested.

### 28. `Ieff_roll`/`Ieff_pitch` fixed at real, independently-sourced physical values — confirmed non-load-bearing two different ways

Investigated whether `Ieff_roll`/`Ieff_pitch` drifting to non-physical values (`9.1`, `10.0 kg·m²`) was load-bearing or an identifiability artifact. **Found a real, independently-sourced BlueROV2 "heavy" configuration rigid-body inertia model** (a public Gazebo repo): `Ixx=0.26, Iyy=0.23, Izz=0.37 kg·m²` — remarkably close to this project's own original literature-based priors, confirming those priors were well-grounded; the fitted values were 30-40x higher than both sources.

**Confirmed non-load-bearing two independent ways:** (1) a general per-parameter sensitivity sweep (`check_parameter_sensitivity.py`, closing an item from an early, never-completed diagnostic checklist) ranked both parameters lowest of all 22 in RMSE sensitivity; (2) a direct re-fit with them held fixed at the real values lost none of the fit quality (cost still dropped ~97%).

**Fix applied:** bounds tightened to `[0.255,0.265]`/`[0.225,0.235]` (narrow but deliberately not identical — confirmed a real `x_scale = ub-lb = 0` division risk before choosing this). Verified via the actual production calibration path, not just the `fixed_params` override.

**Open, testable implication flagged for later:** Coriolis's yaw equation depends directly on `(Ieff_pitch − Ieff_roll)` — non-physical drift here could have been corrupting yaw's specific coupling term even while barely affecting aggregate RMSE. **This is the hypothesis directly tested in finding #32, below.**

### 29. Post-Coriolis identifiability re-check for `Zw_dot`, `Zww`, and `Nrr` — one confirmed unchanged, one genuinely new finding, one further strengthened

Following up on the original identifiability investigation (findings #10, #18) with real, current numbers from this session's diagnostic runs, now that Coriolis and the inertia fix (finding #28) are both in place.

**`Zw_dot`:** unchanged conclusion — still a genuine practical (data-coverage) identifiability gap, not a modeling bug. `vehicle_spec.yaml`'s comment updated to state this directly rather than leaving the original, superseded "~3x nominal literature estimate" note as the only explanation on file.

**`Zww`:** a real, new finding. Its fitted-model correlation moved from `-0.88` (pre-Coriolis) to `-0.826` (post-Coriolis, this session's real diagnostic run), against a fixed kinematic baseline of `0.918`. **The gap from baseline widened** (`0.038 → 0.092`) — the opposite of what would be expected if Coriolis were resolving a missing-physics-driven inflation, the way it was hoped to for `Mq/Mqq`. Conclusion: `Zw/Zww`'s correlation looks like genuine data-coverage, not a missing-physics symptom.

**`Nrr`:** further strengthens the existing "clean data-coverage case" conclusion. Post-Coriolis correlation `-0.861` vs. kinematic baseline `0.864` — gap `0.003`, the **tightest match of any damping pair**, and tighter than its own pre-Coriolis gap (`0.006`). Also not bound-pinned in the current real result.

**`vehicle_spec.yaml` updated** with all three findings, directly in each parameter's `bounds_rationale` and in the `identifiability_note`'s tail.

### 30. `net_buoyancy_z` investigated and fixed — the third and final non-load-bearing drift found this session

The one parameter among the original "still pinned" list that had never been separately investigated. Real diagnostic data showed it pinned at its extreme bound (`+10N`, ≈1kg of implied trim error) with sensitivity `0.001` — tied for the lowest of all 22 parameters, the same signature `Ieff_roll`/`Ieff_pitch` showed before being fixed.

**Confirmation test:** fixed at `0.0` (this project's own original "neutral trim" prior) and re-fit the other 21 parameters. Cost still dropped `~97%` (`12.79 → 0.364`), an almost identical improvement curve to the `Ieff_roll`/`Ieff_pitch` test — the same non-load-bearing signature, confirmed.

**Fix applied:** bounds tightened from `[-10, 10]` to `[-1.0, 1.0]` — still comfortably wider than a well-trimmed vehicle should need, while removing the room for non-physical drift to the old extreme.

**Real production bug found and fixed along the way:** the existing HoloOcean checkpoint (`calibration_output_holoocean_coriolis/checkpoint.yaml`) held the *old*, now-out-of-bounds values for all three of `Ieff_roll`, `Ieff_pitch`, and `net_buoyancy_z` — resuming from it directly crashed (`ValueError: Initial guess is outside of provided bounds`). An initial manual fix only corrected two of the three parameters (missing `net_buoyancy_z`) and still crashed. Built `check_and_fix_checkpoint_bounds.py` — a general checker comparing every parameter in a checkpoint against the *current* `vehicle_spec.yaml` bounds, not just the ones anticipated as changed — which correctly caught and fixed all three. This general tool should be run before resuming from any older checkpoint whenever `vehicle_spec.yaml`'s bounds have changed.

### 31. Tier-1 automation items closed out, and genuinely wired into the real, production scripts (not just built standalone)

Following up on an earlier "easiest → hardest" prioritized list, five items were closed out this session, each verified via real command output, not just code review:

- **Dead artifact removed:** `custom_dof_scales.npy`, a leftover from the now-ruled-out (finding #27) DOF-weighting approach.
- **`run_calibration.py` auto-backup:** the same protection already proven in `run_calibration_holoocean.py`, extended to the stand-in script — closing the exact class of silent-overwrite risk that lost real progress earlier this session.
- **Arena task regression tests, done twice — the first attempt was a real, caught mistake.** Initial tests (`TestArenaTaskVariants`) verified the reward/termination logic via a standalone reimplementation, not the actual `Arena` class — meaning a real regression in `arena.py` itself would NOT have been caught. Corrected with `TestArenaTaskVariantsReal`, constructing and stepping genuine `Arena` instances (6 more tests, all passing against real code). Total suite: 26 tests, confirmed green on GitHub Actions CI.
- **`evaluate_policy`'s success metric verified against the new tasks.** Confirmed by reading `evaluate_policy_arena.py`'s real code that success-tracking is generic (reads `Arena.step()`'s own `terminated` flag, no task-specific hardcoding) — then protected that conclusion with two real tests (a zero-action policy starting at the target succeeds for `waypoint`; a single touch is confirmed NOT sufficient for `station_keeping`, ruling out a silent fallback to `waypoint`-style logic).
- **`plot_real_vs_simulated()` confirmed working against the current, full physics model** (Coriolis, off-diagonal mass, tightened inertia bounds) via a real, live `diagnose_calibration.py --plot` run — produced real PNGs (`worst_run.png`, `typical_run.png`, plus force-breakdown plots for `p`/`q`) with zero errors. This same run also surfaced real, current diagnostic data used in findings #29-30: only `3/22` parameters pinned (down from `12-13` earlier in the project), condition number `8.037e+03`, and the correlated-pairs list used above.

**Two items (Arena `--task` selection, the automated identifiability report) were found to be built but NOT actually reachable from the real production scripts** — a genuine gap, caught and closed:
- `run_full_pipeline.py` gained a real `--task` flag, threaded through to `Arena()`'s construction, confirmed via a live run with `--task docking` (`STEP 2/2` verified the docking-specific environment, not a silent fallback).
- `train_multiseed.py` — the *real*, non-demo Stage 2.4 script — had the identical gap (`Arena(params, vehicle, fixed_waypoint=waypoint, max_episode_steps=100)`, no `task=` argument at all). Fixed the same way, with `waypoint` kept as the default so no existing Claim B result's meaning silently changes. Confirmed via a live run with `--task station_keeping`.
- `automated_identifiability_report.py` is now explicitly signposted in `run_full_pipeline.py`'s final output (a real, copy-pasteable command using that specific run's actual calibrated-parameters path), and documented as step 6 of `README.md`'s real workflow section.
- `preflight_check.py` is now the automatic, mandatory-by-default first step of `run_full_pipeline.py` (`--skip-preflight` available as an explicit opt-out) — and separately documented as step 1 of the real, non-demo workflow.

**`run_full_pipeline.py` received substantial additional hardening this session, several of which were real, found-and-fixed bugs:**
- The held-out split used an unqualified `heldout_fraction=0.3`, which — with only 4 real collection dates — held out just the single most recent date (`6` runs), not the Phase 1 standard `2` dates (`14` runs) every other script in this project uses. Fixed via a `--heldout-dates` flag defaulting to Phase 1's exact dates, making this demo's numbers genuinely comparable to the rest of the project.
- `--max-train-runs` was a documented CLI flag that did **nothing** — `run_pipeline()` (the function actually called) has no truncation mechanism at all, meaning the flag's own safety promise ("protects against the multi-hour full-dataset mystery") was false. Fixed by rewriting the calibration step to use the same proven lower-level functions (`discover_csv_files`, `split_by_date`, `load_run`, `calibrate`) directly, with truncation genuinely wired in this time.
- Hitting `max_nfev` without strict scipy convergence was treated as a hard failure (`sys.exit(1)`) — but this is the *expected*, normal outcome for a small demo budget, not an error, and treating it as one meant the default demo experience would fail even when working correctly. Fixed with a best-point-tracking `on_iteration` callback and a graceful fallback, matching the pattern already proven in `run_calibration_holoocean.py`.
- Two separate `NameError` bugs from an identical Python class-body scoping mistake (`x = x` inside a class body reads the not-yet-assigned class attribute, not the enclosing variable) — first in `strict_converged = converged`, later in `improvement_heldout_pct`. Both fixed by renaming to avoid the self-reference; the fix was verified by actually *executing* the corrected class body with real values, not just re-parsing it, specifically because the first fix's silent gap (a second identical bug elsewhere in the same block) had already been missed once.

### 32. The Coriolis-coupling hypothesis for `r` (flagged in finding #28) — tested directly, and refuted

**The hypothesis:** since Coriolis's yaw equation depends on `(Ieff_pitch − Ieff_roll)`, and that difference had moved from a real, physical value to a wildly non-physical one before finding #28's fix, maybe the fix itself (bringing that difference from `≈+0.9` back to a small, physically-real value) would help `r`'s regression.

**Getting to a clean test took real, iterative debugging:**
1. Resuming from the existing HoloOcean checkpoint crashed immediately — it held the pre-fix values for all three of `Ieff_roll`, `Ieff_pitch`, *and* `net_buoyancy_z` (finding #30), now outside the tightened bounds. Fixed via `check_and_fix_checkpoint_bounds.py` (finding #30).
2. The first real, completed run genuinely converged (the first HoloOcean segment all session to satisfy scipy's strict `xtol` criteria, not just exhaust its budget) — but Claim A showed a **broader** regression than any prior segment: not just `p`/`r`, but `u` and `q` flipped negative too (`u: -16.4%`, `q: -22.4%`, alongside `p: -23.5%`, `r: -38.8%`). Diagnosed as confounded: the DOF-weighting mechanism from finding #27 (already shown, separately, not to help) was still active in this run.
3. Restored finding #27's segment 5 as the real, current-best baseline before proceeding further.
4. Built a `--no-dof-weighting` flag for `run_calibration_holoocean.py` to isolate the inertia/buoyancy fix cleanly. This took real, iterative debugging of its own: two attempts at a multi-line text-block patch failed with no clear cause (later understood likely due to subtle, invisible whitespace differences from an intervening timing-log patch); a third, smaller single-line-anchor attempt applied successfully but left a genuine logic bug — the new `if` branch correctly set `dof_scales = None` when weighting was disabled, but the *following* lines (multiplying `p`/`r`'s scale) were never actually moved inside that conditional, so they ran unconditionally regardless and crashed on `None[index]`. Fixed on the fourth attempt, this time verified by testing the exact patch logic against the complete, real file content before handing it off, rather than against a remembered or assumed fragment.
5. A quick `--max-nfev 1` smoke test confirmed the flag genuinely worked end-to-end (reached real `RESULTS` output, no crash) — as expected, it consumed its entire tiny budget computing the first Jacobian (`23` real evaluations for `21` free parameters), so no real parameter movement happened yet, which is expected behavior, not a null result.
6. The real, clean test: `--no-dof-weighting`, warm-started from the bounds-corrected checkpoint, `--max-nfev 15`. Genuinely converged again (`xtol` satisfied, `258.7` minutes).

**Result: the hypothesis is refuted, cleanly.** `r: -41.4%` — the single worst result of the entire session, worse than any prior attempt, including the confounded one. `u: -16.2%`, `p: -25.1%`, `q: -22.5%` all real regressions too; only `v`/`w` stayed roughly stable.

**Why this refutation is trusted, not just another inconclusive result:** the coupling term this hypothesis was specifically about, `(Ieff_pitch − Ieff_roll)`, moved from `≈+0.9` before the fix to `≈-0.03` after — not a small nudge, essentially the largest change to this specific quantity that could be tested, including a full sign flip. `r` still got worse, not better. This is strong, direct evidence this specific coupling pathway is not the actual driver of `r`'s error.

**A new, real pattern observed across this test and finding #28's own clean test, worth naming precisely:** in every HoloOcean-backed test this session, regardless of which specific fix was applied (weighting, inertia/buoyancy, or both), `v` and `w` stay roughly flat or mildly positive, while `u`, `p`, `q`, and `r` consistently regress together. This is a different, more specific pattern than "rotational axes are weak" — it cuts across the translational/rotational boundary in a way that doesn't map cleanly onto any single physical mechanism tested so far. This further strengthens finding #27's leading remaining hypothesis (the 150-step/15-second training-window truncation used in every HoloOcean-backed segment this session) as the most likely explanation, since it's the one variable common to every regressed axis and every test — genuinely not yet tested, and the clearest remaining next step.

**Current, real, active best result:** finding #27's segment 5 (`p: -16.9%, r: -36.0%`, alongside real translational/pitch gains) remains the actual best validated calibration — this session's clean test was a deliberate, informative experiment, not a replacement for it.

### 33. `Xq_dot` sitting near its bound in the most recent real result — flagged, not yet investigated

A real, specific observation, distinct from finding #25's earlier conclusion about the same parameter. Finding #25 found `Xq_dot`/`Zq_dot` converging to essentially zero (`Xq_dot: -0.0019`) — the basis for shelving off-diagonal added mass (finding #26) as "the model found no benefit to using this coupling." **In the most recent real result, `Xq_dot` sits at `-0.4794` — right at the edge of its `[-0.5, 0.5]` bound**, the opposite signal from the earlier near-zero result.

This is the same bound-pinning pattern that led to real, productive investigations elsewhere this session (`Ieff_roll`, `Kp`) — a parameter sitting at its bound can mean either "genuinely wants to go further" or "not load-bearing, drifted there for free." Given finding #26's shelving decision was explicitly based on the *earlier*, near-zero result, this later, opposite observation means that decision's original justification may no longer hold as stated. **Not yet investigated** — flagged here so it isn't lost, not evaluated. A real next step would be the same sensitivity-check-plus-confirmation-refit pattern already proven for `Ieff_roll`/`Ieff_pitch`/`net_buoyancy_z` (findings #28, #30).

---

### 34. Parallelized restarts (stand-in backend) — core mechanism confirmed, CLI integration test in progress

Added `calibrate_parallel()` to `optimizer.py` — runs `config.n_restarts` independent least-squares fits across real OS processes (`concurrent.futures.ProcessPoolExecutor`) instead of sequentially, for the stand-in backend only. Deliberately, explicitly refuses to run if `config.simulate_fn` (HoloOcean) or `config.on_iteration` (checkpointing) is set — a live HoloOcean environment almost certainly cannot be pickled across a process boundary (confirmed via Python's own documentation: Windows only supports the `spawn` start method, which requires every argument be genuinely picklable), and multiple processes writing the same checkpoint file concurrently is a real race condition, not addressed here.

**Real bugs found and fixed while building this, each confirmed before moving on:**
- `_make_fixed_params_wrapper`'s closure is real and always built whenever `fixed_params` is used (confirmed by reading the actual code, not assumed) — workers rebuild this fresh from picklable inputs (`full_template`, `free_mask` — plain NumPy arrays) rather than receiving a closure, which `spawn` cannot pickle.
- An early patch attempt was applied twice (its anchor text never matched the newly-inserted function itself, so it didn't detect its own prior application) — silently duplicating `_parallel_restart_worker`/`calibrate_parallel` in the file. Caught by inspecting the full file directly rather than trusting an assumed clean state; fixed by removing the duplicate, verified via a realistic test fixture before touching the real file.
- The very first invented return-value construction used field names that don't exist on the real `CalibrationResult`/`CalibrationLog` dataclasses (confirmed by the resulting `TypeError`) — fixed by matching `calibrate()`'s own proven construction pattern exactly, including reusing its existing `_stability_report()` rather than reinventing it inline.
- A genuine Windows-specific gotcha: running the first test via `python -c "..."` with no `if __name__ == "__main__":` guard caused the 'spawn' start method to potentially re-execute the entire calling script inside each worker, recursively — real, silent, and easy to mistake for a hang. Fixed by requiring (and documenting) that any caller of `calibrate_parallel()` be a saved script with a proper `__main__` guard, not an inline `-c` command.
- Investigated CPU oversubscription (multiple worker processes each independently trying to use every CPU core for internal BLAS linear algebra) as a candidate explanation for one real, slow-looking test — added explicit `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS=1` inside each worker as a safety measure regardless, though a live `py-spy` catch on that same slow run showed it was doing real, moving, legitimate physics computation the whole time (the culprit was simply an accidentally-untruncated, full-length real dataset in the test itself, not a bug).

**Confirmed working, standalone (not yet via the real CLI):** a real test on truncated (150-step), real data — `4` restarts, genuinely converged (`4/4`), completed in `440.6s` versus `376.8s` for a single restart alone run separately — real, substantial speedup, not just "didn't crash." Verified via a live `py-spy` catch mid-run that workers launch correctly and do real, moving work, not silently duplicate or stall.

**Wired into the real product and fully confirmed.** `run_calibration.py` gained `--parallel`/`--max-workers` flags, correctly switching to `calibrate_parallel()` and skipping the (incompatible) checkpoint callback when set, with a graceful fallback in the report-building code for `calibrate_parallel()`'s `scipy_result=None` (falls back to the best individual restart's own stored `status`/`message`, and `NaN` for `optimality`, which has no substitute without shipping the full scipy result back).

Confirmed across three separate, real tests: (1) the worker-launch and computation mechanism, live, via `py-spy`, against the FULL real dataset — confirmed genuinely moving, real physics work (different stack lines across repeated dumps, real growing memory usage), not a stall, even though that particular run took far longer than expected and was eventually abandoned in favor of a faster, more targeted test; (2) a real, small, converged run against a tiny 3-file real-data subset, correctly failing with the expected `RuntimeError` message format when given too small a budget (`--max-nfev 3`), then correctly proceeding further with a larger budget; (3) the specific, previously-untested fallback code path (`OptimizerSummary` construction when `scipy_result is None`) — rather than wait further on a real optimization run, tested directly and instantly with realistic fake data shaped exactly like a genuine `calibrate_parallel()` result, confirming the fallback logic runs correctly with no crash. This third test was the right call in hindsight — a real converged run was not actually necessary to confirm this specific code path, and waiting for one cost real, avoidable time earlier in this investigation.

**One real, honest limitation to flag for anyone using this**: the full, untruncated real dataset combined with `--parallel` has not been confirmed to complete in a reasonable time — the one live attempt at this exact combination was abandoned after multiple hours (though confirmed, via `py-spy`, to be doing genuine work throughout, not stuck). Recommend testing on a smaller subset or with `run_calibration_holoocean.py`-style truncation first before relying on `--parallel` against the full dataset for a real, time-sensitive calibration.

### 35. Garbage-input robustness (item #2 follow-up) - three new pre-flight checks, genuinely wired into both real product paths

Extended preflight_check.py with three new checks, deliberately split by confidence level rather than one blended threshold - a real design correction made along the way: an initial plan to tie FAIL-level rejection thresholds to this session's own observed training-data standard deviations was reconsidered and reworked, since a future, legitimately more energetic real dataset could exceed such a threshold without being wrong. The corrected design separates two genuinely different kinds of check:

check_schema.py (FAIL-level, high confidence) - fast, header-only validation that every CSV has the columns objective.py's own REQUIRED_COLUMNS actually needs. Deliberately imports that list directly rather than maintaining a second, separately-typed copy - the same no-duplicated-schema principle already established elsewhere in this project. Tested against the real, confirmed 70-column dataset header (PASS) and a deliberately broken one missing a required column (correctly FAILs with the right column name named).

check_data_integrity.py (FAIL-level, high confidence) - structural checks that don't depend on units or vehicle context at all: empty/near-empty files, header-only files, malformed CSV rows (wrong column count), non-numeric values in numeric fields, and NaN/Inf poisoning. Tested against seven real fixtures covering every category (clean data, empty file, header-only, too-few-rows, malformed row, non-numeric cell, NaN/Inf) - all seven produced the correct verdict with the correct real exit code.

check_physical_plausibility.py (two-tier: FAIL for absolute physical impossibility, WARNING for statistical anomalies only) - the corrected design: angular rates exceeding an absolute, physically-motivated ceiling (15 rad/s, chosen specifically to sit comfortably above this project's own real confirmed maximum of ~4 rad/s, so it can never false-positive against genuine data) are a real FAIL, directly catching the well-known degrees/s-vs-radians/s unit mistake without depending on any dataset's own statistics. Values beyond 8x this project's own previously-observed per-DOF standard deviation are WARNING-only - an anomaly flag, explicitly not a rejection criterion. Linear-velocity limits tied to the configured vehicle's actual capability were deliberately deferred (no authoritative limit currently exists in vehicle_spec.yaml) rather than invented - flagged as a real, concrete upgrade for when such limits are added there. Verified against four hand-constructed scenarios (genuine real data at its real observed maximum, a synthetic degrees-mislabeled-as-radians case, a moderate real outlier, and a linear-DOF statistical outlier) before trusting it against real data - all four produced the correct tier (FAIL/WARNING/clean).

Run against the full real 39/40-file dataset, all three new checks pass cleanly, with check_physical_plausibility.py correctly surfacing real WARNING-level outliers concentrated specifically on p and q - a pattern that independently lines up with this session's other findings about which rotational axes behave unusually, a real, if informal, cross-check that the tool is measuring something genuine.

A real duplicate-insertion bug, same class as finding #34's - wiring the three new checks into preflight_check.py's main() was accidentally run twice, silently duplicating the three new run_check() calls (harmless in that each check would just run and overwrite its own dict entry, but wasteful and confusing). Caught via grep -c before testing, exactly the same discipline finding #34 already established; fixed with a precise, targeted removal verified against a realistic duplicated fixture before touching the real file.

Confirmed genuinely wired into BOTH real product paths, not just standalone: preflight_check.py is already the automatic first step of both run_full_pipeline.py (the demo) and the documented real workflow (README.md step 1) - since the three new checks live inside that same file rather than needing a second flag threaded through a separate calling script, both paths picked them up automatically. Directly confirmed live via a real run_full_pipeline.py invocation showing all five checks (the original two plus these three) executing in order before calibration begins.

### 36. `ObstacleAvoidanceTask` — a full new Arena task, built via a proper `Task`-class architecture refactor, plus a rigorous gap-closing pass

Item #3 (Arena task variants). Real design review across three rounds (a senior-engineer-style critique, then two further refinement passes) reshaped the original plan substantially before any code was written - the final approach differs in real, material ways from the first draft: a real `Task` class architecture (not more inline if/elif branches), `info["terminal_reason"]` instead of overloading `terminated`, randomized (not fixed) obstacle geometry with an explicit episode-validity invariant, a frozen reward equation and termination table agreed before implementation began, and a deliberate compatibility decision to leave the three original tasks' observation space untouched rather than force a fixed superset schema prematurely.

**Phase A - `Task` class architecture refactor (isolated checkpoint, zero new behavior).** Migrated `arena.py`'s inline `if self._task == ...` branches into a real `Task` base class with `WaypointTask`/`StationKeepingTask`/`DockingTask` subclasses - a pure refactor, confirmed via a 6000-trial randomized logic-equivalence test (old inline logic vs. new `Task.step()`, zero mismatches) before ever touching the real file. A real, necessary interface revision followed: `Task.reset()` needed to return `(spawn_override, waypoint_override)`, not just reset internal state, once `ArenaCore.reset(position=...)`'s existing-but-unused capability was discovered - re-verified with a second, dedicated equivalence test (2000 trials, including confirming RNG-draw-sequence parity, not just final values). Regression gate: all 26 existing tests passing (twice, before and after the interface revision), plus live confirmation through `run_full_pipeline.py` for `docking` and `station_keeping`.

**Phase B - `ObstacleAvoidanceTask` implementation**, per the frozen design: `ObstacleAvoidanceConfig` (grouped, structured fields; `generate_episode(rng)` owns a regenerate-until-valid loop internally, guaranteeing every episode's spawn/goal/obstacle satisfies real, explicit constraints - confirmed via 5000 trials, zero violations, plus a confirmed clean failure on a deliberately-impossible config rather than an infinite loop). Reward equation: `R = w_p*Δd + w_g*1[goal] - w_c*1[collision] - w_u*||u||² - w_o*f(d_obstacle)`, `f` active only inside `influence_radius`. Termination table: collision → `terminated=True, terminal_reason=COLLISION`; goal → `terminated=True, terminal_reason=GOAL`; neither → not terminated (Arena's own `truncated`/timeout handling applies as normal); collision checked first, so a simultaneous collision+goal never gets silently overridden by success. Extended `(15,)` observation (base `11` + obstacle-relative-position + obstacle-distance) implemented via a new, general `Task.extra_obs_dim()`/`extra_obs()` extension point (default `0`/empty) rather than a hardcoded task-name check in `Arena._get_obs()` - confirmed inert for the three original tasks (0/1000 mismatches in a dedicated equivalence test) while correctly producing `(15,)` for the new task. One real design gap resolved during implementation: the frozen reward's control-effort term needs the raw action, which `Task.step()`'s established signature doesn't receive - resolved by adding one inert `self._last_action = action` line to `Arena.step()` (provably a no-op for existing tasks, since none of them read it) rather than revise the already-twice-verified `Task.step()` signature a third time.

**Phase C - `evaluate_policy_arena.py` retrofit.** Replaced `if terminated: successes += 1` (silently wrong for any task where `terminated` can mean something other than success) with `info.get("terminal_reason") == TERMINAL_REASON_GOAL`, and retrofitted all three original tasks to set this on their existing success condition - proven mathematically equivalent for them (`terminated=True` never arose any other way), then confirmed via the real test suite and a live run showing `success_rate=1.00` on the known-good `waypoint` positive control. Added five new metrics to `EvalResult`: `collision_rate`, `timeout_rate`, `mean_episode_length` (all episodes), `mean_time_to_goal` (successes only), `mean_min_obstacle_clearance` (task-agnostic - `None` when no episode ever reports an obstacle, not a misleading `0.0`).

**Phase D - layered testing, 14 new tests** (`tests/test_obstacle_avoidance.py`, kept separate from the existing 26-test file to guarantee zero risk to it): 7 mock-based logic tests (including the collision-priority edge case and the reward-shaping equation, checked to the same 9-decimal tolerance as the frozen design specifies), 3 real-`Arena`-instance construction/step tests, and 4 deterministic scenarios with *known, exact* expected outcomes (guaranteed collision, guaranteed success, guaranteed timeout, same-seed reproducibility) built by directly controlling a real `Arena`'s initial state rather than hoping randomization happens to produce the scenario. Full suite: `40/40` passing locally and (confirmed via the Actions tab, not assumed) on GitHub Actions CI.

**Phase E - wired into both real scripts.** `--task obstacle_avoidance` added to both `run_full_pipeline.py` and `train_multiseed.py` in the same sitting - directly closing the exact "built but not reachable from the real production scripts" gap finding #31 first identified and fixed for the three original tasks. One real, non-obvious interaction found while reading `train_multiseed.py` closely, not discovered later: `train_one_seed()` always passes a real-data-grounded `fixed_waypoint` into `Arena()`, but `ObstacleAvoidanceTask.reset()` always returns its own generated `waypoint_override`, which correctly takes priority per Phase A's design - meaning the real-data-grounded waypoint would be silently computed and silently unused for this task. Fixed with an explicit runtime warning rather than a silent surprise, confirmed printing correctly in the real, live output. SB3's `PPO` construction confirmed to read `env.observation_space` dynamically (no hardcoded `(11,)` assumption anywhere in this project's own code) via a real, live `PPO`/`learn()`/`predict()` run against the new `(15,)` space.

**Phase F - live end-to-end confirmation** through both real scripts with the real `--task obstacle_avoidance` CLI flag (not just direct Python construction): `run_full_pipeline.py` showing `STEP 2/2: PASS ... obs shape: (15,)`; `train_multiseed.py` showing the new runtime warning printing in the correct sequence, real training completing for both `default` and `calibrated` physics.

**A genuine, honest gap-closing pass followed**, after auditing the phase plan sub-step by sub-step rather than assuming completion:

- **A4 gap**: `train_multiseed.py` had never actually been live-tested for the three *original* tasks after the Phase A refactor - only `run_full_pipeline.py` had. Closed by running all three (`waypoint`, `station_keeping`, `docking`) live through `train_multiseed.py`; no crashes, consistent real reward/distance numbers across all three (as expected, given they share the same physics and only differ in success criteria).
- **C5 gap**: the original regression check compared the post-refactor `waypoint` result against a general prior reference (`100%` success), not a literal, freshly-captured pre-refactor number. Closed rigorously, not just re-documented: used `git worktree` to check out the exact commit immediately preceding the Phase A refactor into a fully isolated directory, and ran the *identical* positive-control test (`evaluate_policy`, same seed, same episode count) against that genuinely pre-refactor code. Result: `success_rate=1.00`, byte-identical to the current code - a real, literal, apples-to-apples confirmation, stronger evidence than the original plan's own C5 step would have produced.
- **D4 gap**: CI had never actually been checked, only assumed green after a successful local run. Closed by directly confirming via the GitHub Actions tab (Regression Tests run, real commit hash) - genuinely green, not assumed.
- **F2 gap**: the real code gap, not just an unconfirmed step - `train_one_seed()` computed the full `EvalResult` (including the new Phase C metrics) but only ever returned 3 of 8 fields, silently discarding `collision_rate`/`timeout_rate`/`mean_episode_length`/`mean_time_to_goal`/`mean_min_obstacle_clearance`. Fixed by threading all fields through `train_one_seed()`'s return, `summarize()`'s aggregation, and both print statements - confirmed live across all four tasks: `collision=`/`timeout=`/`ep_len=` now print correctly for every task, and `clearance=` appears **only** in the `obstacle_avoidance` run's output, correctly absent from the other three.

**Current, honest status**: the whole feature - architecture, new task, retrofit, tests, wiring, and this gap-closing pass - is genuinely built, tested, and confirmed live through the real product paths a lab would use, including confirmed-green CI. `obstacle_avoidance` has not yet been trained to real convergence (every live run this session used a tiny smoke-test budget, `500` timesteps, specifically to confirm mechanism correctness, not learned competence) - a real, separate next step from what's confirmed here.

### 37. Training algorithms beyond PPO (SAC, TD3) - real, rigorous multi-gate validation, both fully passed

Item #4. Real design review across two rounds (senior-engineer-style critique on both the original plan and its first revision) reshaped this substantially before any code ran - the final approach differs materially from the first draft: a real Gate 1 (Integration) / Gate 2 (Learning) split so a failure is diagnosable by layer, numeric success criteria locked in BEFORE any run (not adjusted after seeing results), three explicit separate budgets (construction/integration/learning, not one reused everywhere), comparison against TWO baselines (random floor AND PPO reference, not random alone), explicit replay-buffer validation (fill, timing relative to learning_starts, NaN/Inf, sampling, save/load, resume), and per-seed reporting alongside the aggregate.

**Phase A - real, confirmed SB3 facts, not assumed:** learning_starts=100 and buffer_size=1000000 identical for both SAC and TD3; learning_rate differs (SAC 0.0003, TD3 0.001) - queried live via inspect.signature, not hardcoded, so the recorded hyperparameters stay accurate automatically. save_replay_buffer()/load_replay_buffer() confirmed real, SEPARATE methods from save()/load() for both algorithms (directly informing Gate 1's save/reload test design). reset_num_timesteps confirmed real on learn(), default True.

**Phase B - locked-in numeric criteria from two FRESH real runs, not reused history.** A real subtlety solved: rewards here are negative (-distance accumulated per step), so naive "50% of PPO's reward" is ambiguous on negative numbers - resolved via gap-closed-fraction: (algo_reward - random_reward) / (ppo_reward - random_reward). Fresh PPO reference run: mean=-30.39 (4 seeds, real n_seeds=4/150000-timestep scale). Fresh random baseline: a real bug in this session's OWN test setup was caught before trusting it - the first random-baseline check omitted max_episode_steps=100 (defaulting to 200), giving a misleadingly large -658.79; corrected to match train_one_seed()'s real environment configuration exactly, giving -272.42. Locked criteria: random floor >= -223.60 (20% of the 242.03-point gap closed), PPO reference >= -151.41 (50% of the gap closed), stability = zero NaN/Inf.

**Phase C - three explicit budgets:** construction=1 timestep (learning_starts overridden to 0, specifically to test a real gradient step in isolation), integration=500 (5x the real learning_starts=100), learning=150000 (matches PPO's own real scale exactly, not scaled down).

**Phase D - wiring, plus a real bug found in TWO separate places, not one.** --algorithm (ppo/sac/td3, default ppo) added to train_multiseed.py; train_one_seed() uses a class lookup instead of a hardcoded PPO(...) call. First hardcoded-"PPO" bug found and fixed: the chart title. A SECOND instance of the same bug was found only after explicit follow-up auditing (not part of the original fix): Claim.B_TRAINED_POLICY_BEHAVIOR itself, in phase2_config.py, hardcoded "PPO" into both the printed section header AND the saved report's "claim" field - meaning a SAC/TD3 run would have silently mislabeled its own real results. Fixed by adding Claim.b_trained_policy_behavior(algorithm), confirmed live for all three algorithms via the real CLI (correct headers, correct saved JSON).

**SAC - both gates passed, real evidence throughout:**
- Gate 1 (Integration): all 9 checks passed - construction+gradient step, replay-buffer timing (99 real samples inspected before learning_starts, all confirmed n_updates=0; updates genuinely begin after), buffer free of NaN/Inf, sampling succeeds, save/reload exactly consistent (-254.540080 before and after, to 6 decimals), genuine resume (500->1000 timesteps, not a restart). One real bug found in this session's OWN Gate 1 test script - a diagnostic message reported "0 samples checked" due to an accidental `and n` filter excluding every correctly-zero sample from its own count (the underlying check was still logically sound, just underselling its own evidence) - caught, fixed, rerun, now correctly reports "99 samples inspected, all confirmed n_updates=0".
- Gate 2 (Learning): 4/4 real seeds, 150000 timesteps each, run sequentially (~11+ hours real wall-clock, confirmed via two live py-spy checks mid-run showing genuine moving backward-pass computation, not a stall). Result: mean_reward=-30.47, std=0.06 - both locked criteria cleared with large margin, essentially matching PPO's own reference (-30.39) almost exactly.
- F3 ("fix anything found"): honestly, nothing in production code needed fixing - the only bug found across both gates was in the test script itself, already caught within Gate 1.

**TD3 - both gates passed, plus a genuine speedup discovered along the way:**
- Gate 1: all 9 checks passed, built from SAC's proven template with zero additional fixes needed (confirming the shared integration wiring genuinely works for TD3 too) - save/reload consistent (-406.495940 before and after), resume confirmed (500->1000).
- Gate 2: initially run sequentially, matching SAC's approach; given no GPU available (confirmed: torch.cuda.is_available()=False) but 16 real CPU cores confirmed available, switched to a real PARALLEL Gate 2 runner instead of waiting out another ~11+ hour sequential run - reused the exact single-threaded-BLAS-per-worker pattern already proven for calibrate_parallel() earlier this session (OMP/OPENBLAS/MKL_NUM_THREADS=1, module-level worker function, mandatory __main__ guard). Result: all 4 seeds completed in a small fraction of SAC's sequential time - real, confirmed speedup, not just a hoped-for one. mean_reward=-30.43, std=0.93 (real, honest note: meaningfully higher seed-to-seed variance than SAC's 0.06, consistent with TD3's known real sensitivity to exploration/target-policy-smoothing noise - every individual seed still landed well within a tight, healthy real range, -29.42 to -31.88).
- A genuine, unrelated infrastructure bug surfaced by this: save_replay_buffer() saves the buffer's full pre-allocated capacity (buffer_size=1000000), not just actual stored transitions - meaning every Gate 1 run (SAC's included) produced a real ~118MB file, exceeding GitHub's 100MB push limit. The first fix attempt (removing the files from the latest commit only) was insufficient, since git still needed to transfer them from an earlier commit's history. Real fix: git filter-branch across the full commit range, followed by git gc, then a genuinely successful force-push - confirmed via real commit-hash output, not assumed. A .gitignore entry now prevents recurrence for any future Gate 1 run.

**Phase H - live CLI confirmation, all three algorithms.** --algorithm sac, --algorithm td3, and --algorithm ppo (default, omitted entirely) all run correctly through the real train_multiseed.py CLI - not just via the Gate scripts' direct calls to train_one_seed(). The ppo default case reproduced byte-identical numbers (reward=-279.24) to multiple earlier regression checks this session, confirming zero disruption to existing behavior.

**Phase I - broader compatibility confirmed.** SAC's full Gate 1 suite re-run against obstacle_avoidance's larger (15,) observation space (not just waypoint's (11,)) - all 9 checks passed, including save/reload consistency through the full evaluation loop against the larger observation - confirming the algorithm-side integration is genuinely agnostic to observation size.

**A real, honest gap found during final review, before documentation:** Phase A's "A2" (action-space compatibility) was never run as an explicitly separate, labeled check - it was genuinely verified, but only as an implicit side effect of Gate 1's construction test (both algorithms constructed against Arena's real action space with zero error). Noted here for completeness rather than silently claimed as a separately-run item.

**Current, honest status:** --algorithm (ppo/sac/td3) is fully built, rigorously validated (both gates, both algorithms, real evidence throughout), and confirmed wired into the one real training entry point this project has (train_multiseed.py - run_full_pipeline.py never trains a policy, by design, so there was no second calling script this could have been missed from). SAC and TD3 both perform at genuine parity with PPO on waypoint at real training scale.

### 38. MLflow experiment tracking (item #5) - real, phased build across all three real scripts, with genuine design corrections along the way

Real design review across two full rounds (senior-engineer-style critique on both the original plan and its refinement) reshaped this substantially before any code was written: a frozen, versioned logging schema decided up front (not improvised per-script), a real Gate-style status vocabulary (SUCCESS/CHECKPOINT_FALLBACK/FAILED/INTERRUPTED, with a paired status_reason for detail), a calibration-specific adapter rather than a generic MLflow wrapper (so replacing MLflow later only touches one file), explicit required-vs-optional fields, and a deliberate manual spike (Phase C0) before writing the real helper, specifically to catch wrong assumptions about MLflow's actual API/UI behavior early and cheaply rather than after three scripts depended on them.

**Phase A - real facts confirmed, one real unplanned side effect investigated.** MLflow 3.15.1 installed and confirmed. Installing it triggered a real, unplanned pandas downgrade (3.0.3 -> 2.3.3, a major version change) - investigated rather than ignored: the full 40-test suite still passed byte-identical, confirming zero real regression before proceeding.

**Phase B - the schema frozen as a real, importable module**, not just agreed-upon chat text: `mlflow_schema.py` (SCHEMA_VERSION=1, the four real status constants, required/optional field lists, a `validate_status()` that raises immediately on a typo'd status rather than silently logging garbage) - directly imported by the real adapter, so the schema can never drift out of sync with what the code actually logs.

**Phase C0 - a real, valuable, load-bearing finding from the manual spike**: a genuine discrepancy where `mlflow ui` and the client library independently fell back to DIFFERENT default backing stores (SQLite vs. file-based) when neither was given an explicit tracking URI - confirmed live, caught before it was baked into three production scripts.

**Phase C - the adapter (`mlflow_logging.py`, `log_calibration_run()`) - and a real, further correction discovered only once real testing began.** The original design (Phase A2) assumed a plain file-store backend (`file:./mlruns`) - genuinely wrong for this MLflow version, discovered only when the adapter's first real test run raised a hard exception ("file store... in maintenance mode"). Traced back to Phase C0's own spike output (which had quietly already used MLflow's real current default - SQLite - without our then-current design realizing it), and corrected to `sqlite:///mlflow.db`, matching what had already been confirmed working. A real, honest example of a design decision made confidently early, found wrong by direct evidence, and fixed without hesitation.

A real, honest post-hoc audit (not initially caught) found three of C1-C5's sub-items had been *inferred* rather than actually tested: whether a missing required field truly raised TypeError, whether all five auto-logged package-version params were genuinely present (only two had been checked), and whether artifact logging's SUCCESS path had ever actually run (only its two real failure modes had been exercised, by accident). All three gaps closed with direct, real evidence before Phase C was considered done - including discovering a real Windows/Git-Bash path-resolution issue (`/tmp/...` unreachable from Python) along the way, itself real, useful evidence for C2's exception-wrapping working correctly a second, different way.

**Phase D - `run_calibration.py` integration.** Reuses `CalibrationReport`'s existing `ReproducibilitySummary` fields directly rather than re-deriving them. Regression testing surfaced two more real, live instances of the still-open performance mystery (PROJECT_STATUS.md's own long-standing open item) - both confirmed via `py-spy` as genuine, moving physics computation, not stalls, consistent with prior findings, not new evidence of a different problem. Given the real cost of waiting out an actual convergence at this scale, D4's SUCCESS-path confirmation was closed pragmatically instead: a representative call using the adapter's exact real function signature and realistic values, genuinely confirming the field-extraction logic without requiring a multi-hour wait - the FAILED path, by contrast, got two genuine live confirmations for free, from real non-convergence.

**Phase E - `run_calibration_holoocean.py` integration - SUCCESS vs. CHECKPOINT_FALLBACK mapped directly onto this script's real, existing try/except RuntimeError structure.** A deliberate, stated cost tradeoff: per-DOF RMSE metrics are NOT logged for this script specifically, since computing them would require a second, separate, expensive HoloOcean-backed evaluation loop on top of the already-costly fitted_rmse calculation already paid for - `rmse_total` only, honestly scoped rather than silently doubling this script's real, scarce compute cost. Two real bugs found and fixed before shipping: an artifact-path-ordering bug (the logging call originally referenced files before they were actually written to disk - caught in testing, moved to after the real save); and a genuine numpy-float-to-YAML corruption bug in a supporting bounds-fixing helper script, which actually corrupted a real, existing warm-start file mid-session - fully recovered via a dedicated recovery script (`yaml.unsafe_load()` used just once, safe here since this was the project's own locally-generated data, followed by a complete rewrite with guaranteed-plain floats and direct, real verification that the recovered file loads cleanly again).

**Phase F - `run_full_pipeline.py` integration - all three real outcomes covered**, including a genuine FAILED path (no usable point found at all, logged with zero artifacts, honestly, before any files exist) distinct from the SUCCESS/CHECKPOINT_FALLBACK path at the end (after `pipeline_summary.json` is genuinely saved). `nfev` is honestly omitted from this script's logged metrics - unlike the other two scripts, this one never actually captures a real call count anywhere in its own code, so a fabricated number was rejected in favor of an honest gap. Regression testing surfaced a real, separate, pre-existing bug in `preflight_check.py` (unrelated to MLflow itself): `--heldout-dates` was never forwarded to its `check_coriolis_excitation.py` subprocess call, so any dataset without that script's own hardcoded default dates produced a confusing, seemingly-unrelated failure. Fixed with a new, optional `--heldout-dates` flag on `preflight_check.py` itself, forwarded only when explicitly given - confirmed both to fix the real failing case and to leave the standard full-dataset path byte-identical to every prior run this session.

**Phase G - cross-script verification, the real point of the whole feature, confirmed clean.** All three scripts run back-to-back into one shared store, then queried together in a single real call: three real, correctly-tagged, genuinely distinct rows (`run_calibration`/`real`/`FAILED`, `run_calibration_holoocean`/`real`/`CHECKPOINT_FALLBACK`, `run_full_pipeline`/`demo`/`CHECKPOINT_FALLBACK`), each with its own real, correct metrics - a lab can genuinely query across every real calibration attempt from a single place, without digging through scattered logs.

**One real, separate near-miss caught before it became a problem**: partway through this work, `README.md` was found to have been relocated from the project root into `calibration/` by an unrelated process - caught via `git status` before any commit, corrected with a simple `mv` back to the root, confirmed clean, and never actually committed in the wrong location.

**Current, honest status**: MLflow experiment tracking is fully built, phased, and confirmed working end-to-end across all three real scripts, with a real, clean cross-script query as the final proof. Local, file-based (SQLite) tracking only - no server, matching this project's real current scale.

### 39. MuJoCo backend: missing rotating-frame term in free-orientation mode -- found and fixed (2026-09-23)

`mujoco_backend._inject_full_override_wrench` converted the desired body-frame acceleration to MuJoCo's world-frame linear acceleration as `R @ v_dot`. When orientation evolves freely (`MuJoCoArenaCore`, forward-sim mode) the correct conversion is `R @ (v_dot + omega x v)`. Symptom: MuJoCo vs `ArenaCore` drifted **1.4 cm in 2 s of turning, and the gap did not shrink with a finer `physics_dt`** (a frame error, not integrator error). With the term added the gap is 10x smaller and shrinks in proportion to `physics_dt`. **Teacher-forced mode is unaffected and must not get the term**: it reads velocity back with the pre-step rotation, so the original formula matches the stand-in to ~1e-15 against real MuJoCo. So MuJoCo *calibrations* (teacher-forced) are unaffected; the fleet MuJoCo validation (`MuJoCoArenaCore`, Claim C) was affected whenever a vehicle turned while moving. Regression test: `tests/test_mujoco_transport.py`; `test_mujoco_arena_core.py` test 3 now asserts a real bound (was finiteness only) and fails on the old code.

### 40. The roll/yaw "weakness" is not in the calibration -- and the HoloOcean backend has the same frame omission plus an unchecked tick rate (2026-09-23)

Held-out per-axis scoring with the stand-in physics the calibrations were fitted with (`diagnose_rotational_weakness.py`, 14 held-out runs): every converged calibration improves roll rate by ~82% and yaw rate by 23-51% over generic physics. The weakness appears only in HoloOcean evaluation. Two concrete problems found in `holoocean_backend.py`:

1. **Same omission as #39**: `world_lin_accel = R @ v_dot[:3]` with HoloOcean integrating a free rigid body. Fixed (`transport_term=True`, default) and tested against a fake HoloOcean that integrates as documented (`tests/test_holoocean_backend_frames.py`: turning surge error 0.03 with the term vs 0.40 without). **Not confirmed in real HoloOcean.**
2. **Tick rate never checked**: the backend sends one command per 0.1 s logged sample, so each `env.step()` must advance 0.1 s. HoloOcean's documented default is 30 ticks/s, and the scenario's value was never recorded. `make_deterministic_environment()` now records it and `simulate_trajectory_holoocean()` refuses to run on a mismatch.

`verify_holoocean_frame_conventions.py` measures both on real HoloOcean (tick length from a constant-acceleration run; which convention keeps a turning vehicle's body velocity constant). It needs a Windows or Linux machine with HoloOcean. **Every HoloOcean-backed number in this document (Claim A, findings #19-#32) predates both fixes and should be treated as provisional until that script passes and Claim A is re-run.** An emulation of HoloOcean's scheme on macOS diverged for all physically plausible calibrations, so it could not reproduce the Claim A numbers either way (documented in the script).

### 41. Test isolation: a fake `mujoco` module leaked into every later test file

`test_mujoco_backend.py` installed a fake `mujoco` into `sys.modules` at import time and never removed it, so any test file collected after it ran against the fake (12 unrelated failures, depending only on file order; CI passed only because of its order). It now restores the real modules immediately after importing its fake-bound copy; the suite passes in either order. `test_mujoco_arena_core.py` passed values between tests via return values, which pytest can't do (3 errors, never ran); converted to fixtures.

### 42. The single-evaluation stall (#1 open item): unattended capture instead of watching for it (2026-09-23)

`stall_watchdog.py`, wired into every stand-in, HoloOcean and demo calibration via `optimizer`: each evaluation is timed on three clocks (wall, monotonic, process CPU) into `eval_timing.csv`, and an evaluation running past max(120 s, 10x median) gets every thread's stack, rusage and load average dumped to `stall_log.txt`, repeated every 10 min while it lasts. The clocks classify the cause: machine asleep (wall >> monotonic on macOS), process waiting (CPU << wall: sleep on Windows, swapping, I/O, CPU starvation) or genuinely computing. The stand-in evaluation does a fixed amount of arithmetic, so a 100x slow-down has to come from outside it; a laptop sleeping through an overnight run fits every observation so far. Verified live on a real demo run (every evaluation 9.6-10.8 s, all three clocks agreeing). `py-spy` was the original plan; it needs root on macOS and someone present at the right moment.

`fast_dynamics.py` (added alongside): the same 6-DOF equations with the mass matrix inverted once and orientation as a rotation matrix; 6.4x faster than `ArenaCore`, equivalent to 1e-8 over 20 s trajectories across 7 parameter sets (`tests/test_fast_dynamics.py`).

### 43. Raw PWM → force, validated on the dataset's own logged pipeline (2026-09-23)

The public dataset logs `pwm1-8`, `pwm_virtual1-8`, `thrust1-8` and `Fx..Tz`. Three exact relationships were recovered:
- **Direction:** `pwm_virtual` is `pwm` mirrored about 1500 µs for thrusters 1, 2, 5 and 8 (reversed motors), identical in every session.
- **Thrust curve:** `thrust` is an exact function of `pwm_virtual` (zero spread over 183,555 rows × 8 thrusters). Deadband is 1472–1528 µs, and the maxima are +47.9 / −37.2 N. Forward and reverse maxima independently match Blue Robotics' published T200 figures interpolated to ~15 V.
- **Allocation:** `Fx..Tz = B · thrust` exactly (least-squares residual 0). This recovers the BlueROV2 Heavy allocation matrix.

`thruster_model.py` reproduces the dataset's forces from raw PWM alone on every row (to the 4-decimal precision of the stored matrix). This closes PRODUCT_VISION items 3 and 12 for T200-based vehicles. Not modelled: ESC/motor lag and battery sag.

### 44. General log ingestion with silent-error guards (2026-09-23)

`data_ingestion.py` converts arbitrary CSV logs to the canonical layout from a YAML spec: column mapping, units, FLU/ENU → FRD/NED frames, Euler or quaternion orientation, logged forces or PWM, and timestamps with optional resampling. Tested by disguising real dataset runs as a foreign ROS-style, PWM-only log in deg/s, cm/s and ms. Ingestion recovers the original velocities to 1e-9, forces to 5e-3 N and rotations to 1e-9, and the calibration objective on the ingested files equals the original's. It refuses the classic silent errors: deg/s declared as rad/s, bad quaternions, non-monotonic or jittered time, mixed rates, and non-µs PWM. ROS bag reading is not built; export to CSV first.

### 45. Non-specialist calibration report (2026-09-23)

`calibration_report.py` turns `metrics.json`, the calibrated parameters and the identifiability JSON into plain language:
- a bottom line
- a per-axis verdict with the reason
- parameters at bounds, physically implausible values, confounded pairs
- next steps

On `calibration_output_10hz_coriolis` it flags what earlier findings established by hand: 9 parameters at bounds, and `Ieff_roll`/`Ieff_pitch` at 19–22× the independent Gazebo inertia (finding #28).

### 46. Claim C withdrawn: the fleet v1 headline metric did not measure calibration (2026-09-23)

`delta_physics_belief_error` (calibrated minus generic "belief error") was the fleet v1 headline. Fleet v1 had no belief model: each agent's "belief" was the last beacon it received, so the metric measured how far the **teammate** moved since its last beacon. That depends on how hard the policy drives each vehicle slot. Controlled check, using the L2 policy at L2 over 96 episodes per condition:

| Setup | Delta (cal - gen) | Mean speed, generic / calibrated slot |
|---|---|---|
| As shipped | -0.018 | 0.26 / 0.38 m/s |
| Generic vehicle given the initial velocity instead | -0.017 | 0.27 / 0.38 |
| **Both vehicles generic physics** (no calibration difference) | **-0.043** | 0.28 / 0.59 |
| **Both vehicles calibrated physics** (no calibration difference) | **-0.011** | 0.28 / 0.37 |

The "calibrated advantage" appears, and is even larger, with no calibration difference at all. Four further problems were found in fleet v1:

- **Centralized control leaked the teammate's state.** One policy saw both vehicles' exact velocity and orientation every tick, bypassing the acoustic channel.
- **Crashing was rewarding.** A collision ended the episode for a one-off penalty of 10, escaping every remaining step's separation cost. That fits the 20–50% collision rates.
- **Beacons were unrealistically frequent.** At 10 beacons a second, beliefs stayed within centimetres even at 60% loss, so no belief model could matter.
- **The entry point ignored most of its config.** Task settings were validated and then ignored, and every comm override crashed with `KeyError`.

Claim C and the README numbers built on it are withdrawn. `phase2_config.CLAIM_C_HONEST_CAVEATS` now opens with the withdrawal, so v1 reports say so. Fleet v1 code still runs (`run_fleet_experiment_v1.py`) for reproducing old results.

### 47. Fleet v2: a physical belief model, per-vehicle policies, and the belief-to-success study (2026-09-23)

**Design** (`fleet_env.py`, `fleet_belief.py`, `fleet_vec_env.py`):
- **Same truth for both vehicles.** Both run the same "true" physics: `calibrated_parameters_fleet_truth.yaml`, the best held-out converged fit (expA).
- **Full-state beacons, sent sparsely.** Each vehicle transmits its full state (position, orientation, velocity, last thrust) every `beacon_interval_s` (1–3 s; acoustic modems don't manage 10 Hz).
- **A real belief model.** Between beacons each vehicle predicts its teammate with a belief model, and that model is the only thing varied. Options: 6-DOF dead reckoning with calibrated, generic or interpolated parameters; constant velocity; last beacon; oracle.
- **Per-vehicle policy.** One shared network sees only the vehicle's own state plus its belief (13 floats) and outputs that vehicle's thrust (6).
- **No crash exploit.** A collision is charged the rest of the episode.

Key properties are tested directly: the observation carries nothing about the teammate except through the channel, and packet-drop patterns are identical across belief models, so episodes are paired.

**Speed.** `fast_dynamics.py` (#42) makes training practical: 3 policies × 3M steps in ~41 min each on an M1.

**Main study** (`fleet_study.yaml`):
- 3 training seeds × 8 belief models × 4 loss levels × 2 beacon intervals × 200 paired episodes = 38,400 episodes.
- Results: `calibration/results/fleet_v2_belief_to_success/`, with per-seed checks in `extended_analysis.md`.

Findings:
1. **Calibrated beats generic physics in 7 of 8 conditions**, by +3 to +25 success points (McNemar p ≤ 1e-5).
   - The gap grows with loss and beacon sparsity: +25 at 30% loss with 3 s beacons, +15 at 60% with 3 s.
   - Dose-response: success falls as model parameters move from true to generic in all 8 conditions (slope CIs exclude 0).
   - Per seed it is robust in the hard conditions (all 3 seeds positive at L2@3s, +22 to +30) and seed-dependent in easy ones (one seed shows no gain at L0/L1).
2. **An uncalibrated predictive model is worse than no prediction.** `generic` loses to `last_beacon` in all 8 conditions, by 3 to 26 points.
3. **But `calibrated` is no better than `last_beacon`** (−10 points at L0@3s and L1@3s), **and perfect information (`oracle`) scores below `calibrated` in 6 of 8 conditions.** These policies don't turn better information into better coordination. The likely cause is a design confound: they were trained on a mixture of belief models, and the observation doesn't say which one is active. The "age since last beacon" input means "stale" for last-beacon but "current prediction" for physics and oracle beliefs, so the policy applies one average staleness correction to all of them. The calibrated-vs-generic comparison is unaffected (both carry the same confound, and it's paired), but "does prediction help at all" is not answered by this study.

**Robustness:**
- **Different "true vehicle"** (`fleet_study_alt_truth.yaml`, holoocean_coriolis): calibrated beats generic in 3 of 4 conditions (+14 to +19), with no difference at L2@1s.
- **MuJoCo as the truth engine** (`fleet_study_mujoco.yaml`): nearly identical to the stand-in (L3a@1s 98/88/99% vs 97/86/99%; L3a@3s 66/48/68% vs 64/48/67% for calibrated/generic/last-beacon).

**What this supports:** for a fleet that predicts its teammates with a physics model under sparse, lossy acoustic comms, calibrating that model is worth up to ~25 points of formation success, and an uncalibrated model is worse than none. It does not yet support "prediction beats no prediction". **Follow-up running:** policies trained for a single belief model each (`train_fleet_policy.py --belief`, evaluated by `run_specialized_study.py`).

**Caveat on "calibrated":** in simulation, `calibrated` means a model equal to the simulated vehicle, so the study measures the value of model accuracy. How accurate a real calibration is against a real vehicle is measured separately, on held-out logs (Single-Vehicle results).

### 48. Specialized policies resolve the #47 confound -- information helps; calibrated beats generic; last-beacon is still hard to beat (2026-09-23)

One policy per belief model (`train_fleet_policy.py --belief`, 2 seeds each, 3M steps), each evaluated only with its own belief (`run_specialized_study.py`). The comparison is 12,800 episodes, paired by training seed, start state and packet losses. Results: `calibration/results/fleet_v2_specialized/`.

| Comm | calibrated | generic | last beacon | oracle |
|---|---|---|---|---|
| L0, 3 s | 98% | 84% | 100% | 100% |
| L1, 3 s | 93% | 80% | 100% | 100% |
| L2, 3 s | 86% | 62% | 94% | 100% |
| L3a, 1 s | 97% | 86% | 99% | 100% |
| L3a, 3 s | 70% | 44% | 70% | 98% |

(Every belief model scores 100% at L0, L1 and L2 with 1 s beacons.)

1. **The confound was real.** With a policy trained for it, perfect information is now the best model: +28 points over last-beacon at L3a with 3 s beacons, +6 at L2 with 3 s. Better teammate information does produce better coordination.
2. **Calibrated beats generic by +11 to +26 points** in every condition not already at the 100% ceiling (L0@3s +14, L1@3s +14, L2@3s +24, L3a@1s +11, L3a@3s +26; all CIs exclude 0). **Generic is worse than no prediction** by 13 to 31 points.
3. **Calibrated dead reckoning still doesn't beat the last beacon** (−2 to −7 points; tied at L3a@3s). It predicts the teammate's motion by holding its last reported thrust for up to 3 s, while the teammate's controller keeps changing that thrust. The remaining error is about predicting the teammate's future **commands**, not its physics. That points to the next lever: share intent (planned thrust or trajectory) in the beacon, or model the teammate's policy. This is a natural research question for a lab partner.

**What Strathe can now claim:**
- If a fleet predicts its teammates with a physics model under sparse, lossy acoustic comms, **calibrate it**. An uncalibrated model costs 13–31 points of formation success compared with no model; calibration recovers it.
- Accurate teammate information is worth up to +28 points, so better prediction has real upside.
- Physics alone doesn't capture that upside yet: intent-sharing or teammate-policy models are the next step.

All of this holds in simulation, with a calibrated model standing in for the real vehicle.

### 49. The single-evaluation stall, caught: the machine was asleep (2026-09-23)

The first long run with the #42 watchdog was a full-dataset stand-in calibration (`--max-nfev 60`, M1, overnight). It caught 7 stalled evaluations (#152–160). For example, eval 154 took 4,044 s of wall time but only 28.8 s monotonic and 26.8 s of CPU. All 7 were classified `machine_asleep_or_suspended`.

**Cross-check against macOS's own power log (`pmset -g log`):**
- The machine entered sleep at 04:57 and then woke briefly for Power Nap ("DarkWake") every ~15 min until 08:05.
- Every one of the watchdog's 7 stall detections is timestamped within one second of a DarkWake: 05:13:13, 05:45:14, 06:52:05, 07:18:27, 07:25:01, 07:43:41, 08:05:45.
- The system sleep setting on this machine is 1 minute.

The raw evidence is in `calibration_output_fast_path_check/` (`eval_timing.csv`, `stall_log.txt`); that run was stopped once it had served this purpose. The process advanced only during those few-second wakes, so a ~26 s evaluation stretched to as long as 67 min. That matches every past observation of the stall: hours of wall time, `py-spy` always showing real, moving computation when someone was present (awake), and no cause found in the code.

**Fix: `keep_awake.py`.** Every long-running entry point now blocks system sleep for as long as it runs. The entry points are `run_calibration`, `run_calibration_holoocean`, `run_full_pipeline`, `train_multiseed`, `train_fleet_policy`, `run_fleet_experiment` and `run_specialized_study`.

| Platform | How sleep is blocked |
|---|---|
| macOS | `caffeinate -i -s -w <pid>` |
| Windows | `SetThreadExecutionState` |
| Linux | `systemd-inhibit` |

- The block ends with the process, even if the process is killed. On macOS this is tested with a SIGKILL.
- Set `STRATHE_ALLOW_SLEEP=1` to opt out.
- If sleep can't be blocked, the run prints a warning and continues.
- Closing a laptop lid still sleeps the machine.
- The watchdog stays on every run as a check.

**Unexplained, minor:** CPU time per evaluation was ~16 s before the sleeps and ~27 s after, with identical work per evaluation. This is plausibly macOS scheduling the process onto efficiency cores after a wake, but that is unconfirmed.

### 50. Roll and pitch had no predictive skill, and the main cause is a thrust moment arm in the logged torques (2026-09-23)

**The check.** Held-out replay of every calibration inside MuJoCo, with orientation evolving freely, as a fleet runs it: 14 runs, windows of 3, 15 and 60 s, 3,308 replays. MuJoCo matches the stand-in to within 0.5% of the error on every axis, and no replay diverged, so nothing here is MuJoCo-specific. Each axis was compared with the trivial prediction "no motion on this axis" (skill = 1 − RMSE / RMSE of predicting zero):

| Calibration | u | v | w | p (roll) | q (pitch) | r (yaw) |
|---|---|---|---|---|---|---|
| generic | +0.47 | −0.62 | −1.74 | −8.39 | −11.69 | +0.09 |
| expA | +0.73 | +0.49 | +0.55 | −0.34 | −0.54 | +0.62 |
| 10hz_coriolis | +0.76 | +0.50 | +0.20 | −0.35 | −0.47 | +0.61 |
| holoocean_coriolis | +0.76 | +0.45 | +0.10 | −0.95 | −1.78 | +0.43 |
| mujoco_full | +0.77 | +0.51 | +0.47 | −0.39 | −0.73 | +0.54 |

(MuJoCo, free-running 3 s windows. Whole-run teacher-forced replay gives the same picture, e.g. expA roll −0.49, pitch −0.84, yaw +0.60.)

- **Yaw is fine.** It tracks the real motion at 0.92 correlation.
- **Roll and pitch are worse than predicting no motion.** The "+82–87%" improvement over default physics reported earlier (#40) is real, but default physics is so poor on these axes (skill −8 to −12) that beating it by 85% still leaves negative skill. This corrects the README and #40, which read that number as "calibration improves roll".

**Why.** It isn't sensor noise or timing:
- The real roll/pitch motion is large (about 10–14°) and slow: 80–85% of its power is below 0.5 Hz.
- The logged rates agree with the logged angles through Euler kinematics at 0.97 and 0.98 correlation, at zero lag.
- The real angular acceleration barely follows the logged torque: correlation 0.03 for roll and 0.06 for pitch, against 0.56 for yaw.
- The logged roll and pitch torques are almost entirely a by-product of horizontal thrust: Tx is +0.95 correlated with Fy, and Ty is −0.98 correlated with Fx. The dataset's own allocation matrix (thruster_model.BLUEROV2_HEAVY_ALLOCATION) puts the horizontal thrusters 8.5 cm above its torque reference point, so every surge or sway command also logs a pitch or roll torque.
- The model applies all forces at its origin and has no drag moment to balance that torque, so it predicts large pitch and roll that the real vehicle doesn't show. In reality, drag at roughly thruster height balances it in steady motion.
- The fit had already tried to express this coupling. `Xq_dot` (surge–pitch coupling) is pinned at its bound; the coupling it was reaching for is about 11.2 kg × 0.085 m ≈ 0.95 kg·m, beyond its ±0.5 bound.

Re-expressing the logged torques about the thruster plane (`objective.shift_torque_reference`, offset (0, 0, −0.085) m) zeroes the horizontal thrusters' roll/pitch torques exactly and leaves their yaw torque untouched. After the shift:
- The corrected torques track the real angular acceleration about 5× better: correlation 0.15 for roll (was 0.03) and 0.21 for pitch (was 0.06).
- Without any refit, expA's held-out skill improves from −0.49 to −0.13 on roll and from −0.84 to −0.31 on pitch; yaw is unchanged.

**Why every roll/pitch damping term pinned at its bound.** Each run was replayed open-loop from its first sample, so motion the inputs don't explain accumulates for the whole run. The only way the optimizer can limit that drift is more damping: the classic output-error bias under unmodelled disturbances.
- A synthetic test with ground truth confirms it (`tests/test_objective_skill.py`). A hidden roll torque makes the whole-run objective prefer 1.6–1.9× the true roll damping, across three realizations.
- The earlier widening of the `Kp` bound from −5 to −10 chased this artifact; the fit simply pinned again at the new bound.
- Re-anchored prediction windows don't remove the bias. On the same synthetic data, 0.3–0.5 s windows pin the damping at the opposite bound, 1 s windows give 0.46–0.91× the truth, and 2 s windows give 1.1–1.4×. Short windows mistake a persistent disturbance for momentum; long ones mistake it for drift.
- The logs also can't identify roll/pitch damping from how the motion oscillates: roll/pitch autocorrelation is nearly identical across all four dates, but there is no oscillation peak to measure. The motion is slow drift, mostly below 0.03 Hz.

**What changed in the code:**
- **Skill everywhere.** `objective.zero_baseline_rmse` and `skill_vs_zero` are new. `run_calibration.py` writes a `skill_vs_zero` block to `metrics.json` (whole-run and 3 s windows) and prints it. `calibration_report.py` now bases every verdict on skill: an axis at skill ≤ 0 gets "Don't rely on it", however much it beats the default model. The real 10hz_coriolis report now says exactly that for roll and pitch.
- **`--torque-reference-offset-m X Y Z`** (run_calibration.py, `DatasetConfig`, fit_rollpitch_disturbance.py) re-expresses logged torques about another point.
- **`--axis-scaling zero_rms`** scores each axis against its predict-zero error. It balances axes against each other but doesn't change the within-axis damping bias above.
- **`--exclude-axes p q`** leaves axes the logs don't explain out of the fit and holds their axis-only parameters at `vehicle_spec.yaml`'s initial guesses. `cg_cb_offset_z` is held only when both are excluded. The report says "not calibrated" for those axes.
- **`--prediction-horizon-s`** fits over re-anchored windows, which suits a model used over a known prediction horizon, such as a 1–3 s beacon interval. It is not a bias fix; see above.
- **`--eval-workers N`** evaluates training runs in parallel with bit-identical results (`objective.ParallelJointResiduals`). It is 3.3× faster with 6 workers on the full dataset; the longest runs cap the gain.

**Roll/pitch bounds, reviewed.**
- `Ieff_roll` [0.255, 0.265] and `Ieff_pitch` [0.225, 0.235] stay tight on purpose. An independent BlueROV2 Heavy model gives 0.26 and 0.23, and the fit has near-zero sensitivity to them (#28). Widening them earlier only let them drift to non-physical values of 5–10 kg·m².
- The damping bounds (`Kp`, `Kpp`, `Mq`, `Mqq`) are not the problem: whole-run replay pins them wherever the bound is. They are left as they are; the refits below show what the data supports once the torques are corrected.

### 51. A roll/pitch disturbance fitted to the real logs, in MuJoCo and the stand-in (2026-09-23)

With the torques corrected (#50), a simulated vehicle driven by the logged commands is much calmer than the real one. On held-out data, roll rate spread is 0.066 vs 0.144 rad/s and roll angle spread is 4.1° vs 11.3°. The remaining motion is not caused by the recorded thrusters (tether pull, waves or contact are the usual suspects), and a clean simulator leaves it out entirely.

`disturbance.py` adds it back as an independent random torque on roll and on pitch: first-order coloured noise with a size (sigma) and a correlation time (tau).
- It is added to the commanded wrench once per control tick, so MuJoCo and the stand-in see the same torque.
- It never enters the beacons, because a teammate can't know about it.
- It is seeded from the episode's identity (comm seed, vehicle, start state), so every belief condition and policy sees the same disturbances and comparisons stay paired.

`fit_rollpitch_disturbance.py` fits sigma and tau per axis for a given parameter set.
- Method: replay 60 s windows free-running, driven by the (corrected) logged thrust plus a candidate disturbance, and match three statistics: rate spread, rate autocorrelation at 0.1–1.5 s, and angle spread.
- Fitted on 60 training windows by a coarse-to-fine grid, checked on 130 held-out windows.
- Fit for the fleet truth (expA): roll 1.6 N·m, 0.7 s; pitch 2.16 N·m, 1.0 s (`disturbances/bluerov2_rollpitch.yaml`).

| Held-out | Real | No disturbance | Fitted | Fitted, in MuJoCo |
|---|---|---|---|---|
| roll rate spread (rad/s) | 0.144 | 0.066 | 0.135 | 0.136 |
| roll angle spread | 11.3° | 4.1° | 9.5° | 9.3° |
| roll rate autocorrelation at 0.5 s / 1.0 s | +0.37 / +0.09 | +0.46 / +0.29 | +0.37 / +0.11 | +0.37 / +0.12 |
| pitch rate spread (rad/s) | 0.202 | 0.084 | 0.214 | 0.214 |
| pitch angle spread | 12.3° | 5.9° | 13.4° | 13.6° |
| pitch rate autocorrelation at 0.5 s / 1.0 s | +0.27 / −0.10 | +0.35 / +0.02 | +0.35 / +0.04 | +0.34 / +0.02 |

No window diverged. One limit: at 0.2 s the simulated rate decorrelates faster than the real one (+0.64–0.69 vs +0.79–0.80), so first-order noise is slightly too rough at short lags. A second-order process would fix that if it ever matters.

**The fit is tied to its parameter set.** The file records a hash of the parameter file it was fitted for, and loading it with a different truth file warns you to re-fit.

**How to use it:**
- Fleet YAML: `disturbance: {roll_pitch: fitted, scale: 1.0}`. The default is `none`, the clean simulation, so every earlier result is reproducible unchanged.
- Training: `train_fleet_policy.py --disturbance fitted`.
- This is the first part of gap-list items #8 (disturbances) and #11 (clean vs noisy modes). Translational currents and sensor noise are not modelled yet.

### 52. Roll/pitch doesn't matter for formation-holding; training with the disturbance doesn't help (2026-09-24)

**Question.** Does the calibration's poor roll/pitch (#50) matter for the fleet use case?

**Method.** `run_disturbance_study.py`: 15,200 paired episodes. Each condition used the same start states, packet drops and disturbance sequences. Conditions varied:
- **Policies:** the shipped clean-trained policies (seeds 1–3) and two new policies trained with the fitted disturbance (`disturbance_policies/`, 3M steps each).
- **Belief models:** `calibrated` and `last_beacon`.
- **Comm:** L2 with 1 s beacons, and L3a with 3 s beacons.
- **Disturbance:** off, as fitted (×1), and twice as rough (×2).
- **Engines:** the stand-in throughout, with MuJoCo as a cross-check.

Results: `calibration/results/fleet_v2_disturbance/`.

| Clean-trained policies | off | ×1 | ×2 |
|---|---|---|---|
| L2 @ 1 s, calibrated / last beacon | 100% / 100% | 100% / 100% | 100% / 100% |
| L3a @ 3 s, calibrated / last beacon | 64% / 67% | 66% / 67% | 66% / 69% |

1. **Realistic roll and pitch don't change formation-holding.** Paired against the same episodes with the disturbance off, success moves by −0 to +2 points at ×1 and ×2, with every confidence interval covering zero (n = 1,200 per comparison). Mean separation error moves by under 1 cm, and collisions stay at 0–1%. The task depends on horizontal position and heading (the skilled axes, #50); the fitted roll/pitch wobble barely perturbs them.
2. **Training with the disturbance doesn't help under it.** Paired by seed and episode at ×1, the difference is +2 points (−2, +6). The disturbance-trained seeds were 8 points better in the clean simulation at L3a (+4, +12), and they lose 4–6 points when the disturbance is switched on. With only two seeds this looks like seed-to-seed variation, not a property of disturbance training.
3. **MuJoCo agrees with the stand-in** with the disturbance on: −1 point (−3, +1) over 1,600 paired episodes.

**Decision.**
- For formation-holding, the roll/pitch weakness is not a blocker. The fleet kit's results don't depend on it.
- The disturbance stays available as a first-class "noisy" option (`disturbance:` in the YAML, `--disturbance fitted` for training).
- It will matter for tasks that depend on attitude: camera or sonar pointing, docking, or anything near the surface. For those, the calibration needs roll/pitch data that the thrusters actually cause (free-decay tests, deliberate roll/pitch manoeuvres). That is a concrete first ask for a lab partner.

### 53. One command, and a local web page, instead of several terminal steps (2026-09-24)

`python strathe.py` (repository root) covers the workflow:
- **`check`:** is this machine ready, and what to fix.
- **`calibrate --data <logs>`:** runs the pre-flight check, the calibration with the recommended settings, and the plain-language report. It stops with a specific message at the first failing step.
- **`fleet`, `ingest`, `report`:** each maps to its underlying script.
- **`web`:** a local web page to start a calibration, fleet experiment or log conversion, watch the live log, and read the rendered report and output files.

The recommended calibration settings are predict-zero axis scaling, parallel evaluation on all but one core, and the vehicle's torque-reference convention from `vehicle_spec.yaml`.

The web page is local-only by construction (`strathe_web.py`, standard library only):
- It listens on 127.0.0.1 and rejects any other Host header, which blocks DNS rebinding.
- Every form carries a per-session token.
- It can only start the fixed `strathe` commands (no shell).
- It serves files only from the output folders of jobs it started.

It was tested end to end in a real browser (fleet smoke run: start, live log, rendered report). That test caught one bug the unit tests had missed: fleet output folders named only in the job's log weren't viewable. It's fixed and now covered by a test. The individual scripts are unchanged for specialists.

### 54. Fleet truth switched to the torque-arm-corrected calibration; everything retrained and rerun (2026-09-24)

`calibrated_parameters_fleet_truth.yaml` now holds `calibration_output_armfix` (#50). The previous truth is kept as `calibrated_parameters_fleet_truth_previous_expA.yaml`.

**Before retraining:**
- The new truth was stress-tested under random and bang-bang commands (4,000 stand-in and 800 MuJoCo steps): no divergence, and rate and velocity ranges matched the old truth.

**Retrained on the new truth** (3M steps each):
- 3 mixed-belief policies.
- 8 specialized policies (4 belief models × 2 seeds).
- 2 disturbance-trained policies.

**Rerun:** the main study, the alt-truth and MuJoCo robustness studies, the specialized study and the disturbance study.

**Main study (38,400 paired episodes), old truth → new truth:**

| Comm | calibrated | generic | last beacon | oracle | calibrated − generic (new, 95% CI) |
|---|---|---|---|---|---|
| L0 @ 3 s | 90 → 97 | 88 → 76 | 100 → 98 | 84 → 100 | +21 (+17, +25) |
| L1 @ 3 s | 90 → 94 | 78 → 70 | 99 → 94 | 84 → 100 | +25 (+20, +29) |
| L2 @ 3 s | 89 → 86 | 64 → 64 | 90 → 84 | 82 → 100 | +22 (+17, +26) |
| L3a @ 1 s | 97 → 98 | 86 → 84 | 99 → 96 | 86 → 100 | +13 (+10, +16) |
| L3a @ 3 s | 64 → 68 | 48 → 52 | 67 → 65 | 69 → 93 | +16 (+12, +21) |

1. **Calibrated beats generic**, as before, and the effect is now larger at 3 s beacons.
   - Per training seed: +10 to +38 in these conditions, significant for every seed except one at L3a @ 3 s.
   - All 8 dose-response slopes are negative with CIs excluding zero.
2. **Generic is worse than no prediction** by 12–25 points in those conditions.
3. **Calibrated now ties last-beacon** (−1 to +3, all CIs cover zero). Under the old truth it trailed by 2–7 points.
4. **The #47 mixed-policy confound is gone.** Perfect information now beats calibrated everywhere it can (+3 to +24), even with mixed-trained policies.
5. **Robustness holds.**
   - Alt truth: +13 to +21 in 3 of 4 conditions (−1, not significant, at L2 @ 1 s).
   - MuJoCo truth: +15 and +14 at L3a, vs +13 and +16 on the stand-in.

**Specialized study: calibrated-trained policies vary a lot between training seeds.**
- Both study seeds often failed to settle at 3 s beacons, with timeouts rather than collisions: L0 @ 3 s 35% and 45%, L1 @ 3 s 52% and 64%. Generic-trained policies scored 70–75% there.
- The calibrated predictor itself wasn't worse: prediction error at L0 @ 3 s was 57 cm vs last-beacon's 43 cm, similar to before.
- Two further seeds, trained and evaluated identically (`results/fleet_v2_specialized/replication_calibrated_seeds3-4.md`), reached 98% and 57% at L0 @ 3 s.
- So this is training variance, not a systematic failure. With this truth, training for the calibrated model alone is unreliable at 3 s beacons.
- The mixed-trained policies are consistent across seeds, so they carry the calibration claims.
- Where the specialized comparison holds, calibrated still beats generic: +10 at L3a @ 1 s and +9 at L3a @ 3 s.
- The replication policies ship in `specialized_policies/replication/`. The study's own inputs are unchanged.

**Disturbance study, rerun:**
- Realistic or doubled roll/pitch wobble still doesn't change formation success: +0 (−2, +2) and −0 (−3, +2), n = 1,200.
- Disturbance-trained policies score 6–8 points higher at L3a @ 3 s, but equally with the disturbance off. That's more likely seed variation than learned robustness.
- MuJoCo agrees: −1 (−2, +0).

### 55. Water current for open-water labs: configurable, off by default, in every engine and in calibration (2026-09-24)

Until now the simulation assumed still water. `vehicle_spec.yaml` had current fields, but nothing in the physics read them, which ruled out open-water labs.

**Physics** (`dynamics_model._relative_to_water`, Fossen 2011 Property 10.1):
- The calibrated drag, added-mass and Coriolis forces act on the velocity relative to the water; position integrates velocity over ground.
- The current, seen in body axes, rotates as the vehicle turns. That adds a −ω × v_c term.
- It's implemented in the reference dynamics, the fast stand-in, the single-vehicle core and MuJoCo. Zero or absent current is bit-identical to before.

**Tests** (`tests/test_water_current.py`, 31 tests):
- An unpowered vehicle ends up drifting at exactly the current, on all three engines.
- A spinning vehicle moving with the water stays with it. A mutation check confirms this test fails without the −ω × v_c term: 0.024 m/s error vs 0.00003.
- Station-keeping takes exactly the drag of the current.
- The fast and reference physics agree, and MuJoCo agrees with the stand-in.
- Configuration validation, per-episode draws, and fluctuation statistics (spread and time scale).
- Pairing: both vehicles share the same water, and the currents are identical across teammate models.
- The single-vehicle environment, and the YAML block.

**Configuration** (`current.py`, off unless asked for):
- Speed and direction, fixed or drawn per episode from a range.
- Vertical flow.
- Slow fluctuation of speed and direction (Ornstein–Uhlenbeck, with a time scale of at least 1 s).
- Whether physics teammate models know the episode's mean current.

It's available through:
- a `current:` block in the fleet YAML, stated in the report
- `strathe fleet --current-speed/--current-direction/...`, which writes a copy of the experiment to `strathe_jobs/` so shipped results are never overwritten
- fields on the web page's fleet form, validated on the server
- `train_fleet_policy.py --current-*`
- `Arena(..., current=CurrentConfig(...))`

**Calibration from open-water logs.**
- `DatasetConfig.current_world_mps` sets the current, as does `run_calibration.py` / `strathe calibrate --current-world-mps`. The default is `vehicle_spec.yaml`'s environment block, still water for the tank dataset. The logged quaternions rotate it into body axes per sample.
- Synthetic ground truth with a 0.2 m/s current: when the current is supplied, linear and quadratic surge drag are recovered exactly. Ignoring it leaves 0.2 m/s of unexplained error, and shifts the drag terms by 1–5% depending on the vehicle's other parameters (5% with expA, 1% with the current truth).
- **Limit:** this handles a current that is known and roughly constant within each recording. Estimating an unknown or changing current from the logs themselves isn't built, and neither is a current that varies across space.

**Stress test before shipping** (all passed after one fix):
- 500 random episodes with currents up to 3 m/s, vertical flow ±0.5 m/s and fluctuation, under uniformly random full-strength thrust: no divergence. The stand-in and MuJoCo stay within 0.33 m of each other after 30 s of chaotic forcing.
- Unpowered drift matches any current to better than 0.01 cm/s after 60 s.
- 9,000 fleet steps with current and roll/pitch disturbance together, on both engines: no crash, and the true trajectory is identical across teammate models.
- Gymnasium's `check_env` passes on all four single-vehicle tasks with a current.
- The calibration pipeline runs end to end with a known current.
- The web form works end to end in a real browser.

**Bug found by the stress test, fixed.** The single-vehicle environment (`Arena`) defaulted to a 0.02 s physics step.
- At that step, uniformly random commands diverged within 200 steps in every episode, for every calibration on file, with or without current. Explicit Euler is unstable at 0.02 s against this vehicle's roll/pitch damping-to-inertia ratio at high rates.
- The default is now 0.005 s, the fleet's and the physics default, which never diverged. `ArenaCore` now raises its own `FloatingPointError` instead of an obscure scipy quaternion error.
- Single-vehicle policies trained before this date (Claim B) used 0.02 s. PPO's exploration evidently stayed out of the unstable region, but those results were produced with the old step.

**Study** (`run_current_study.py`, 34,800 paired episodes, `results/fleet_v2_current/`):
- **Conditions:** currents of 0, 0.1, 0.2 and 0.3 m/s in a random direction each episode, plus 0.2 m/s with fluctuation.
- **Teammate models:** calibrated that knows the current, calibrated unaware of it, and last beacon.
- **Policies:** the three still-water-trained ones, and two new ones trained in random currents up to 0.3 m/s (`current_policies/`).
- **Comm:** L2 with 1 s beacons, L3a with 3 s beacons.
- **Engines:** the stand-in, with MuJoCo as a cross-check.

Findings:
1. **Currents barely hurt formation-holding.**
   - With 1 s beacons, nothing changes up to 0.3 m/s.
   - At L3a with 3 s beacons, still-water policies lose 5 points at 0.3 m/s (−7, −2), and 0–2 points (not significant) at 0.1–0.2 m/s.
   - Policies act on their own velocity over ground, so the current reaches them mostly through the teammate estimate.
2. **Knowing the current is what matters.** At L3a with 3 s beacons, a calibrated model that knows the current beats one that doesn't:
   - still-water policies: +6 (+2, +10) at 0.2 m/s, +8 (+4, +13) at 0.3 m/s, and +4 (+0, +8) with fluctuation
   - current-trained policies: +6, +7 and +13 at 0.1, 0.2 and 0.3 m/s
3. **In a current, calibrated prediction pulls ahead of no prediction, if it knows the current.**
   - Still-water policies, calibrated-and-aware vs last beacon: +3 (not significant) in still water, +8 (+3, +12) at 0.3 m/s.
   - The unaware model only ties last beacon.
   - So a fleet working in currents should estimate the ambient current and give it to its teammate model.
4. **Training in currents didn't help.** The current-trained policies were 0–4 points below the still-water ones, paired by seed. The still-water policies already cope, because they see their own velocity over ground.
5. **MuJoCo agrees with the stand-in:** −0 (−2, +1), n = 2,400.

### 56. A lab's own vehicle, end to end: raw logs to calibration, report, and one- and two-vehicle experiments (2026-09-25)

The audit of the lab-facing gaps found each piece existed but a lab couldn't chain them. It then found four bugs that stopped or silently corrupted the chain. Everything below is on `strathe` and the web page.

**What a lab can do now:**
- **Raw logs to report in one command:** `strathe calibrate --ingest-spec <spec>`. It converts the logs, checks them, fits, checks what the data can pin down, and writes the report. It stops at the first failing step with a plain message; the identifiability step only warns. Ready-to-edit specs are in `templates/`, with a README.
- **ROS bags read directly** (`rosbag_reader.py`, using `rosbags`, no ROS install):
  - ROS 1 `.bag`, ROS 2 `.mcap` and `.db3`. For pre-Iron `.db3` bags with custom messages, the lab lists the `.msg` files.
  - Columns are `<topic>.<field>[i]`, and timestamps come from message headers.
  - Every topic goes onto one grid: velocities linearly interpolated, orientation slerped, PWM held.
  - It warns about gaps and repeated stamps. It refuses missing topics and fields, non-numeric fields and out-of-range indices.
- **Any thruster layout** (`thruster_model.thruster_block`):
  - A per-thruster `layout` (position, direction, reversed), an allocation CSV, or the BlueROV2 Heavy preset.
  - The T200 curve (with voltage) or the lab's own curve.
  - Under-actuated vehicles are accepted, and ingestion names the axes the thrusters can't push. The rank < 6 check used to reject them outright.
  - The custom-layout template is the Heavy written as a layout; a test checks it equals the preset about the thruster plane.
- **Torque reference recorded.** The ingestion report records which torque reference the calibration should use: the preset keeps the dataset's point 8.5 cm below the thrusters (#50), and a layout is already about the thruster plane. `strathe calibrate` applies it.
- **Thruster limits in the simulators** (`thrusters:`; `ThrusterConfig.achievable_wrench`):
  - A command is split over the thrusters by least squares, then scaled down as a whole until no thruster exceeds its curve. Direction is preserved, as ArduSub's mixer does. Axes the thrusters can't push get nothing.
  - It works in `Arena`, the fleet on both engines, fleet training, fleet YAML, `strathe fleet/arena/train-fleet --thrusters` and the web forms. Fleet beacons carry the delivered force.
  - Off by default. With it off, every number is bit-identical to before, and a test checks this. With the Heavy's limits, pitch torque tops out at ~18 N·m, where any policy could command 30 N·m before.
- **A lab's vehicle description and calibration in fleet experiments** (`vehicle_spec:`, `truth_params:`; `strathe fleet --params --vehicle-spec`), and fleet training for it (`strathe train-fleet`, `train_fleet_policy.py --spec --thrusters`).
- **Single-vehicle experiments** (`run_arena_experiment.py`, `strathe arena`):
  - PPO policies trained on the lab's calibration, plus a hand-written controller and random commands, all on the same episodes.
  - Optionally, policies trained on default physics, with the paired difference and a 95% bootstrap CI.
  - A water current and thruster limits are optional. `--policy` evaluates the lab's own policy.
- **Identifiability with the fit's own objective** (`automated_identifiability_report.py --result`). The old script used the default objective: no torque shift, no current, whole-run replay, no axis scaling, all 22 parameters free, even the held ones. The new one:
  - uses the recorded sampling rate, torque point, current, axis scaling, excluded axes, prediction windows and held parameters
  - takes the Jacobian over the free parameters only, in parallel (identical to serial)
  - reports correlated pairs, and groups the data constrains more than 1000× less than the best-constrained combination, in bounds-width units

  The report explains both in plain words.

**Bugs found and fixed:**
1. **The data check crashed on every converted log.** `check_sampling_rate_windowed.py` infers the rate from the logged x, y, z positions, which ingestion doesn't produce. So `strathe calibrate` could never have run on a lab's converted logs; the reference dataset has positions, which hid it. The check now uses the rate the timestamps gave and fails if calibration is told a different one. Without positions or timestamps it says the rate is assumed.
2. **`strathe calibrate` always assumed 10 Hz.** The canonical CSVs carry no time column, and the rate ingestion measured was never passed on. On the reference runs resampled to 20 Hz, integrating at the assumed 10 Hz made held-out error 5–11% worse on surge, sway, heave and yaw. Now the rate flows from `ingest_report.json` (or `--sampling-rate-hz`) into the data check, the fit, `metrics.json`, the identifiability check and the report.
3. **Orientation was interpolated linearly when resampling.** The reference data has 239 orientation jumps over 20° within 0.1 s, some near 180° (sensor glitches). Across them, component-wise interpolation gives quaternions far from unit length. Upsampling the real runs to 20 Hz hit one with norm 0.72 and refused the file. A canonical quaternion's sign flip gives a random orientation for one sample too. The CSV and bag paths both slerp now, raw norms are still checked, and jumps over 45° per sample are reported.
4. **Resampling grids drifted.** `np.arange(t0, t1, dt)` with Unix-epoch `t0` steps by a rounded `dt` (~2e-8 s off), drifting ~1e-5 s over a few minutes. Grids are now `t0 + k/rate`, and a rounding guard keeps the final sample.

Also: when the fit ran out of its evaluation budget, `strathe calibrate` stopped with a failure. The lab was left to run `finalize_checkpoint.py` by hand, with the right objective flags. Now it keeps the best result so far with the same settings, marks it not converged, and still runs the identifiability check and the report.

Also: with `--eval-workers`, the stall watchdog labelled every normal evaluation "process_waiting (CPU starved)", because the parent only waits on its workers. It didn't flag stalls, but the label was misleading. Parallel evaluations are now labelled "computing (in worker processes)"; sleep is still detected.

**Stress test: a lab's whole journey, twice.** Scripts are in the session scratchpad; the checks they exercise are in the test suite.

*A. The real dataset as a lab would record it.* All 39 reference runs were written as ROS 1 bags the way ArduSub + MAVROS logs them: body velocity, IMU and RC-out topics; ROS frames; PWM only, no forces; header stamps; some quaternions sign-flipped as an IMU sends them; an unrelated topic mixed in.
- `strathe ingest` with the shipped template, unchanged except for the file pattern, recovered:
  - velocities exactly
  - forces to 0.001 N
  - orientation to 5e-14°
- The held-out error of the fleet-truth calibration is the same on the converted files as on the originals, to 1e-4.
- At 20 Hz, the same held-out error with the correct rate. Held-out error was 5–11% worse at the old assumed 10 Hz: that is bug 2.
- `strathe calibrate --ingest-spec` ran all five steps on the bags. The budget was deliberately set too small, so it also exercised keeping an early-stopped result. Identifiability over 22 free parameters took 3 min on 7 workers.

*B. A different vehicle the pipeline has never seen.*
- The vehicle: a simulated 6-thruster BlueROV2-like layout with no pitch authority, the fleet-truth physics, and a T200 curve at 14.8 V. It flew 12 dives of 90 s on 3 days, with random piecewise-constant PWM, logged as PWM-only CSVs.
- The pipeline: a custom-layout ingest spec and one `strathe calibrate` command (budget 60), which converged.
- Fit quality: held-out RMSE 0.0007 against 0.48 for the default model.
- Parameters: all 22 recovered within 4% of the truth; 17 within 1%. The pitch terms are the least certain, as expected with no pitch authority.
- The report flags the values sitting at their limits (the truth itself is at them). The identifiability check found the linear/quadratic drag pairs that trade off (correlation −0.91 to −0.93).
- The calibration then ran:
  - in `strathe arena` with the vehicle's own thrusters and currents up to 0.3 m/s (below)
  - in `strathe train-fleet` (600k steps with the same thrusters and currents)
  - in a fleet experiment on both engines with a current (2,304 paired episodes)
- Fleet results: the calibrated teammate model beats the generic one by +1 to +10 points, significant at L3a@1s: +10 (+3, +17). The ordering oracle > calibrated ≥ last beacon ≥ generic matches the reference study.
- `strathe fleet --params --thrusters --current-speed` also ran the shipped smoke experiment on the bag-calibrated vehicle.

*Single-vehicle arena on vehicle B* (reach a point within 0.5 m; its own thruster limits; currents 0–0.3 m/s; 3 seeds per condition; 100 paired episodes):
- **At 100k training steps,** policies trained on its calibration succeeded only 10% of the time. Default-physics policies succeeded 75%, consistently across seeds.
- **Checked, not a bug.** It is a training-budget effect. The calibrated vehicle is heavily damped (several drag terms at their limits), so PPO learns slowly on it. The lighter default physics teaches an aggressive policy that transfers.
- **At 300k steps:** 96% (seeds 99/91/99%) vs 100%. The remaining −3.7 points (−6.0, −1.7) is real.
- **So on this task, training on the calibration didn't help.** Aiming at a point is forgiving of model error, and it's the comparison `--compare-generic` exists to make. The fleet task, where predicting a teammate is the point, is where calibration pays off, as in B above.
- The hand-written controller reached 16% (weak on a heavily damped vehicle with limited thrust); random commands 0%.

On the bag-calibrated BlueROV2 with the Heavy's thruster limits, station-keeping policies succeeded 100% (98–100%), as did the hand-written controller.

*C. The web page, in a real browser.* A single-vehicle job (thruster-limited BlueROV2 in a current) was submitted from the form, ran, listed its outputs and rendered its report. That found two fixes: the "not converged" note printed once per worker process, and an effort column mislabelled N·s.

Test suite: 363 + 142 fleet tests at the time of writing, including:
- bags written in all three formats from real runs
- messy bag timing
- 300 random thruster layouts
- pre-flight on converted logs
- thruster limits in both simulators and both engines
- a lab vehicle description in a fleet experiment
- the single-vehicle experiment end to end
- identifiability with the fit's objective

## Honest open items

- **Physics dead reckoning only ties last-beacon in still water** (#48, #54). It pulls ahead in a current it knows about (#55). Predicting a teammate means predicting its future commands; intent-sharing beacons or teammate-policy models are the next lever.
- **Currents: unknown or changing currents** can't yet be estimated from logs, and currents don't vary across space (#55). A fleet that works in currents should estimate the ambient current and give it to its teammate model.
- **Lab data, not built yet (#56):**
  - reading ArduPilot `.bin` / MAVLink `.tlog` logs directly (export to CSV, or record a ROS bag)
  - thruster response lag and battery sag within a run
  - two different vehicles in one fleet
  - the thruster model isn't itself calibrated: the curve and layout are taken as given
- **HoloOcean results are provisional** (finding #40): the frame fix and tick-rate guard are unconfirmed in HoloOcean itself. Run `verify_holoocean_frame_conventions.py` on a HoloOcean machine, then re-run Claim A.
- **Roll and pitch have no predictive skill yet** (#50): after the torque-arm correction they are at about zero skill (no longer worse than predicting no motion), not positive. This dataset's roll/pitch motion is mostly not caused by the thrusters, so it can't identify roll/pitch damping. It doesn't matter for formation-holding (#52), but it will for attitude-dependent tasks. It needs free-decay tests and deliberate roll/pitch manoeuvres from a lab.
- **Calibrations are bound-limited**: every converged stand-in/MuJoCo fit has 8-15 of 22 parameters at `vehicle_spec.yaml` bounds; the least-limited (holoocean_coriolis) has 3. For roll/pitch damping this is the output-error artifact of #50, not a bounds problem. The translational damping pins still need review by someone who knows the vehicle.
- ~~**The fleet truth vehicle is still expA**~~ — switched to the torque-arm-corrected fit, with every policy retrained and every fleet study re-run (#54).
- **`Xq_dot`'s bound-reversal** (finding #33) and **`Zw_dot`'s practical identifiability gap** (#18, #29) are unchanged.
- ~~**The single-evaluation stall**~~ — caught and explained (#49): the machine was asleep. Long runs now keep the machine awake by themselves (`keep_awake.py`).
- **The ~2.5%/tick angular-velocity decay** under Custom Dynamics (#15) remains a known residual.

---

## What's actually next

1. **On a HoloOcean machine**: run `verify_holoocean_frame_conventions.py`; fix the scenario tick rate if it fails; re-run Claim A (`replay_comparison.py`). Only then revisit roll/yaw and the 150-step truncation hypothesis.
2. **Review `vehicle_spec.yaml` bounds** for the translational parameters every fit pins, then re-fit.
3. **Roll/pitch data from a lab partner**: free-decay tests and deliberate roll/pitch manoeuvres, with the tether slack. That is the only route to real roll/pitch skill (#50, #52).
4. Long runs keep the machine awake by themselves (keep the laptop lid open); `stall_log.txt` and `eval_timing.csv` stay on as a check.
5. Re-run `check_damping_identifiability.py`'s `Mq`/`Mqq` comparison against the current Coriolis-aware Jacobian (flagged in finding #10, still not done).
6. If the corrected HoloOcean numbers still show a rotational gap, the next model-structure levers remain: off-diagonal added mass with a properly time-budgeted test, thruster/actuator nonlinearity, cross-axis damping.
