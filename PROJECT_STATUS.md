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

### 57. Every known limit in one place; MuJoCo and rough water for one vehicle; ArduPilot logs; unknown currents; fleet experiments from flags (2026-09-26)

**`LIMITS.md`** lists every known limit, from a sweep of the code (docstrings, `NotImplementedError`s, assumptions, TODOs), the docs and this log. Each entry says what it means for a lab and what to do. The README, the lab journey and the templates link to it. Lifted limits move to its "Lifted" list.

**Single-vehicle arena: MuJoCo and roll/pitch disturbance.**
- `Arena(..., backend="mujoco", disturbance=...)`, and `strathe arena --backend stand_in|mujoco|both --disturbance fitted|<file> --disturbance-scale S`, also on the web form.
- The disturbance is drawn after the target, so targets are identical with it on or off. It's seeded, so every controller sees the same torques.
- Policies train on the stand-in (with the chosen disturbance, current and thrusters). With `both`, the report tests on each engine and states how closely they agree.
- `MuJoCoArenaCore` now finds its model file from any working directory.

**ArduPilot logs read directly** (`ardupilot_reader.py`, using `pymavlink`):
- DataFlash `.bin`/`.log` and MAVLink `.tlog`, with columns `MESSAGE.field` and `[i]` for multi-instance messages.
- It uses the log's own clock: GPS time, boot time, or the ground station's clock.
- It shares the time-grid code with the bag reader (`series_to_grid`), which now also unwraps angles, so yaw crossing 0/360 isn't interpolated the long way round.
- Corrupt and cut-off records are skipped and counted.
- It says what to fix for a missing message, field or instance, and for a multi-instance message given without an index.
- `body_velocity.linear_frame: world` rotates world-frame velocity into the body frame, for any source.
- Two new templates.

**Unknown water currents.**
- `estimate_current.py` (`strathe estimate-current`) estimates each recording's constant current: the one that makes the vehicle model's 3 s predictions best match the logs.
- It checks each estimate by estimating the recording's two halves separately, and reports how much the vehicle turned. Estimates made with uncalibrated defaults are flagged as rough.
- `currents.csv` (per-recording currents) is used by calibration, the check of what the data can pin down, the report and `finalize_checkpoint.py`. It's picked up automatically from the logs folder.
- `strathe calibrate --estimate-current` alternates: calibrate in still water, estimate, recalibrate with the currents (two rounds by default). There's a web checkbox for it.

**Fleet experiments from flags.** `strathe fleet` flags (and web fields) cover:
- comm levels and beacon interval
- teammate models and the headline pair
- target separation, tolerance, hold time, episode length and collision radius
- starting separations and closing speeds, and sample size
- engine and disturbance, policies and name

They merge into a copy of the chosen YAML, which is validated before anything runs; impossible experiments are refused with the schema's message. `--save-config` writes the file instead of running it.

**Stress test, under deliberately hard conditions** (all passed):
- **Extreme arena conditions, both engines:** 71,625 steps with 5× disturbance, currents to 1.5 m/s (fluctuating, vertical flow), a 6-thruster vehicle with no pitch authority, bang-bang full-scale commands, all four tasks. No NaN or divergence, and pitch torque was never delivered.
- **Engine agreement:** stand-in vs MuJoCo with 2× disturbance and 0.3 m/s current, same seeds and commands. Positions agree within 0.11 cm after 10 s (worst of 30).
- **Gym checks:** Gymnasium's `check_env` passes for all 16 engine × task × disturbance combinations, with thrusters and current.
- **Arena experiment end to end:** both engines, 2× disturbance, current, Heavy thruster limits, station keeping, calibrated vs default training (2 seeds × 150k). The engines agree to within 0.5 points on every controller.
  - Calibration-trained policies scored 11% against 93% for default-physics ones: the slow-learning effect of #56. The report now says so, and tells the lab to raise `--timesteps`.
- **ArduPilot logs, all 39 real runs:**
  - `.bin` with no GPS, junk bytes every 97 records, a cut-off final record and two IMUs: all 183,555 samples recovered; worst per-file median errors 1.5e-5 m/s, 2e-5 N and 0.005° (centidegree attitude).
  - `.tlog` with a corrupt packet every 53: recovered to 1e-8 once aligned. Five files start one sample late because their first message was the corrupt one.
  - `strathe calibrate --ingest-spec` on the `.bin` logs ran all steps, keeping an early-stopped fit.
- **Unknown currents:**
  - **Synthetic:** six dives in currents of 7–30 cm/s. `--estimate-current` recovered every current within ~1 cm/s and 2°. Held-out error was 14× lower than ignoring the currents (0.007 vs 0.097), and the median parameter error was 0.1% (vs 0.6%).
  - **Hard cases:** noisy DVL (2 cm/s noise): 0.2 cm/s error. A current reversing mid-dive: flagged (halves 25 cm/s apart). A vehicle that never turns (5°): flagged. Still water: 0.01 cm/s.
  - **Real tank data** (the true current is zero), with the armfix calibration: the 39 estimates range 0.1–3.8 cm/s (median ~1). A current explains ~0% of the model's error, and one recording is flagged.
- **Fleet flags:** 300 random flag combinations gave 193 valid experiments written and 107 impossible ones refused with the schema's message; none ran. A fleet run built only from flags (both engines, disturbance, current, thrusters; 1,728 episodes) found calibrated vs generic +14 to +25 points where the task isn't nearly solved.
- **Bugs found and fixed:**
  - Fleet plots were silently skipped whenever a condition succeeded every time. The Wilson bound sits 2e-16 below 100%, which made a negative error bar.
  - pymavlink printed a line per damaged spot; those are now counted in the ingestion warnings.
  - The current-reliability rule first judged by "share of error explained". That wrongly rejected good estimates from a calibrated model, whose remaining error is dominated by other things. The rule now flags estimates made with default physics, and currents whose halves disagree.

Tests: 396 + 142 fleet, all passing.

### 58. Clean vs noisy simulation, buoyancy changes, and SAC/TD3 for the fleet (2026-09-27)

**Clean vs noisy as one switch.** `--mode clean|noisy` (and `both` for the fleet) in `strathe arena`, `strathe fleet`, `strathe train-fleet`, the web forms, `train_fleet_policy.py`, `run_arena_experiment.py`, `train_multiseed.py`, and `mode:` in the fleet YAML (`sim_mode.py`).
- `noisy` = sensor noise `typical` + the fitted roll/pitch disturbance (#51), each at scale 1. Explicit `--noise` / `--disturbance` override either part.
- Current, thruster limits and buoyancy belong to the site and the vehicle, so they're set separately in either mode.
- `mode: both` runs every fleet condition clean and noisy on the same episodes. The report adds a paired "Does noise matter?" table.
- In a saved `experiment.yaml`, noisy's defaults are written out, so a rerun is exact.

**Sensor noise** (`sensor_noise.py`), per vehicle and per 0.1 s tick:
- DVL velocity: bias per dive, white noise and dropouts (last reading held).
- Gyro rates, roll/pitch and compass heading: each with a bias and white noise.
- Depth: white noise.
- Position: dead-reckoned from the measured velocity and heading, starting from a fix with error, so it drifts. Or `position: external` (tank motion capture, USBL): white noise, no drift.

What changes and what doesn't:
- The **policy** sees measured values, and targets or teammates relative to where it *believes* it is.
- **Beacons** carry what the sender believes, so a teammate's prediction starts from that.
- **Success, collisions and belief errors** stay judged on the truth. `nav_error_m` reports the drift.
- **Pairing is kept.** The noise depends only on the episode's identity, and every tick draws the same count of random numbers. Clean and noisy runs therefore share start states, targets, packet losses and disturbances; the stress test checks this.

Presets:
- **`tank`** is *measured* from the 39 reference recordings by `fit_sensor_noise.py` (robust second difference): 1.7 mm/s velocity, 0.39°/s gyro, 0.09° roll/pitch, 0.19° heading, 0.8 mm depth, 3 mm external positioning.
- **`typical`** (DVL + AHRS + depth, dead reckoning) uses *stated assumptions*, fixed before any result: 1 cm/s velocity (0.5 cm/s bias, 2% dropouts), 0.57°/s gyro, 0.5° attitude, 2° heading (2° bias), 1 cm depth, 20 cm starting fix.
- **`strathe fit-noise --data <logs>`** writes a lab's own file: white noise, dropouts and positioning. Biases can't be seen in logs, so it writes them as 0 and says so.

**Buoyancy** (`buoyancy.py`):
- `--buoyancy N`, a range drawn per episode, or `buoyancy:` in the YAML. It is added to the calibration's net buoyancy on the true vehicles, in every engine (each physics core gained `set_params`).
- The teammate models learn of it only with `known_to_beliefs`.
- Values beyond ±50 N are refused as a probable units mistake.

**SAC and TD3.**
- `train_fleet_policy.py --algorithm sac|td3` (and `strathe train-fleet`) has SB3-standard settings:
  - 1M steps by default
  - four gradient steps per 16 transitions
  - [256, 256] networks
- `run_arena_experiment.py --algorithm` uses the single-vehicle setup validated in #37.
- Policy files are recognised by their contents (`policy_io.py`): the fleet YAML's `algorithm:` is optional, and a wrong one is refused with the right name.
- Fleet policies can carry a `label`. With several labels (or algorithms), the fleet report adds "Policy groups compared": the same episodes, seeds averaged per episode, a 95% CI per condition. `strathe fleet --policies ppo=... sac=...` does this from the command line.

**Clean simulation is unchanged, bit for bit.** `clean_regression_check.py` reruns clean episodes and compares every field against the same episodes on the code before this change (6c8a2ef; reference in `results/stress_noise_algorithms/`):
- 6,616 fleet episodes: two policy seeds × every comm and belief condition, 216 of them with disturbance, a fluctuating current and thruster limits, some on MuJoCo.
- 16 single-vehicle runs: every task, both engines, with and without those options, with every observation and reward hashed.
- Zero differences.

**SAC and TD3 for the fleet: validated** (`run_algorithm_study.py`, `results/fleet_v2_algorithms/`).
- **Setup:** three seeds each at 1M steps, the same training mix as the shipped PPO policies (3M steps). Judged against criteria written while they were still training (`criteria.md`).
- **Grid:** 17,600 paired episodes; L2 and L3a, 1 s and 3 s beacons, `calibrated` and `last_beacon`.

| | Success (95% CI) | Per seed | vs PPO, paired (95% CI) |
|---|---|---|---|
| PPO | 87% (86–88) | 83, 88, 90% | — |
| SAC | 87% (86–88) | 90, 90, 81% | −0 (−1 to +1) |
| TD3 | 91% (91–92) | 89, 93, 93% | **+4 (+3 to +6)**; +10 at L2 @ 3 s |
| random commands (floor) | 12% | | |
| no commands (reference) | 8% | | |

- **Gate 1 passed for all six policies:** contract, identical actions after save/reload on 1,000 recorded observations, no non-finite action.
- **"Learned":** both clear the floor + ½ gap threshold (50%) by far, and no seed failed.
- **Parity:** both reach it.
- **Cost:** alone on one M1 core, SAC/TD3 run at ~500–650 steps/s vs PPO's ~770, so a 1M-step run is shorter than PPO's 3M. The recorded 2–2.5 h came from 9 trainings sharing 8 cores.

**Clean vs noisy for the fleet** (`run_noise_study.py`, `results/fleet_v2_noise/`).
- **Scale:** 57,120 paired episodes, L2 @ 1 s and L3a @ 3 s, `calibrated` and `last_beacon`.
- **Policies:** the shipped clean-trained PPO, three new PPO seeds trained with `--mode noisy` (`noisy_policies/`), SAC and TD3.

| Level | Trained clean | vs clean level (95% CI) | Trained noisy | Nav error | Belief error |
|---|---|---|---|---|---|
| clean | 83% | — | 84% | 0 | 42 cm |
| tank | 83% | −0 (−1, +0) | 84% | 0.3 cm | 42 cm |
| typical ×0.5 | 80% | −3 (−4, −2) | 81% | 13 cm | 47 cm |
| typical ×1 | 73% | **−10 (−12, −8)** | 74% | 26 cm | 56 cm |
| typical ×2 | 50% | **−34 (−36, −31)** | 58% | 52 cm | 78 cm |
| noisy mode | 73% | −10 (−12, −9) | 74% | 26 cm | 56 cm |

1. **Realistic sensors cost formation-holding 10 points; twice as noisy, 34.** The tank's own sensor noise costs nothing. The cost comes through navigation drift: the believed separation is off by the two vehicles' combined drift, against a ±0.2 m tolerance. *(Corrected in #60: the numbers stand, but almost all of the cost is `typical`'s 0.2 m starting position fix, not drift. With a perfect fix, the `typical` sensors cost 1 point.)*
2. **The roll/pitch disturbance adds nothing on top** (noisy mode = typical ×1), consistent with #52.
3. **Training with the noise helps only when it's severe:**
   - ×1: +1 point (−1 to +3)
   - ×2: +8 points (+6 to +11)
   - The policy can't see its own drift, so there's little to learn.
4. **Algorithms under noise (all trained clean):**
   - PPO −10, SAC −13, TD3 **−22** (−24 to −20).
   - TD3 is best clean (86%) and worst noisy (64%). Its clean-simulation advantage doesn't survive realistic sensors.
   - That makes `--mode both` the right way to compare algorithms.
5. **Buoyancy from −10 to +10 N barely moves formation success:**
   - Changes of −1 to +1 points, whether or not the teammate model knows.
   - The one exception, +4 (+2 to +7), was at +10 N with the model unaware.
   - Both vehicles get the same offset, and the task depends on relative position.

**One vehicle** (`strathe arena`, waypoint, 2 seeds per condition, 100 test episodes each, `results/arena_noise_algorithms/`):
- **Algorithms, clean:**
  - PPO, SAC and TD3 all reach 100%.
  - SAC and TD3 arrive faster: 5.4 s vs 5.9 s.
  - Training took 9.5, 33 and 21 minutes per policy, respectively, under load.
- **Under noise:** clean-trained PPO/SAC/TD3 stay at 97–99%.
- **Training with the noise doesn't help here either:**
  - 97% vs 99% for clean-trained (−2, CI −4 to −0.5, on the stand-in; −1, −2.5 to 0, on MuJoCo).
  - Noise-trained policies used 23% less thruster effort.
- **Buoyancy:**
  - Policies trained at 0 N stay at 100% from −10 to +10 N, 0.1–0.5 s slower.
  - The hand-written controller falls from 65% to 25–32% at ±10 N.

**Stress test** (`stress_test_noise_algorithms.py`, `results/stress_noise_algorithms/`). Every check passed.
- **Two vehicles: 410/410 random combinations ran cleanly.** They varied:
  - comm level L0–L3b and beacons of 0.5–3 s
  - buoyancy of −25 to +25 N, fixed or a range, known or not
  - sensor noise: off, tank, or typical ×1, ×3 or ×10
  - disturbance ×0, ×1 or ×3, and currents to 0.5 m/s, fluctuating
  - thruster limits, and both engines
  - policies: PPO, noise-trained PPO, SAC, TD3, random
  - every belief model, and starts 0.6–8 m apart at ±1 m/s
  - plus the worst corner of all of these on every policy and engine

  Outcomes: 193 formations held, 205 timeouts, 12 collisions. Every observation and error stayed finite, and the largest navigation drift was 10.3 m (typical ×10).
- **One vehicle: 400/400.**
  - Covered all four tasks; PPO, SAC and TD3 policies, hand-written and random controllers; and the same settings.
  - Every step checked that the reported distance is the true one, and that success is never declared from the believed position.
- **Measured against what was configured: 23/23.**
  - The sensor model reproduces its configured sizes: DVL noise 1.00 cm/s, gyro 0.010 rad/s, heading 0.0349 rad, bias spread 0.51 cm/s, dropouts 2.04%.
  - Buoyancy terminal speeds match the calibrated damping balance to 4 digits, from −20 to +20 N, on both engines.
  - Packet losses and true motion are identical with the noise on or off.
  - Noisy rollouts are deterministic on both engines.
  - Clean simulation is bit-identical (above).

**Bugs found and fixed:**
- An unknown `--algorithm` reached a bare `KeyError`; it now gets a plain message.
- A two-vehicle policy given to `strathe arena --policy` crashed inside a worker with an SB3 shape error. It's now refused up front, with what the policy is and where it belongs.
- The web page's range parser couldn't read negative numbers (`-3 to 3` for buoyancy). There's now a signed parser.

Tests: 52 new (`tests/test_noise_buoyancy_algorithms.py`) + 1 web; 449 + 142 fleet, all passing. CI also runs a fleet smoke run with `--mode both --buoyancy -2 2`.

**Caveats, stated where labs see them** (follow-up, same day):
- **`typical` sensor noise is assumed, not measured.** Every description of it now says so: the noise files carry `provenance: assumed|measured`, and reports, `--help` text and the web menu print it. The study reports open with it.
- **SAC/TD3 were validated on the reference BlueROV2 and the formation task only.** This is now stated in the README, CUSTOM_EXPERIMENT section 14 (with the commands to train and compare seeds of each on your own vehicle), LIMITS.md, the web form, `--help`, the algorithm study report, and every fleet report that compares policy groups. That report also warns when a group has a single training seed.
- **Commands without a web form are now listed:** `fit-noise`, `report`, `check` and a standalone `estimate-current`. The README had said every command had a form. The web page explains how to make a noise file in a terminal and then use it on the form.
- Three tests keep the labels in place.

### 59. Optical and hybrid links for the fleet (2026-09-27)

**Why.**
- Finding #48 found that calibrated dead reckoning ties "last beacon": the unknown is the teammate's next commands, not its physics.
- A link fast enough to beacon every tick gives high-rate *state* sharing: the teammate's state and the thrust it delivered on that tick, one tick old. That isn't intent sharing (a planned command or trajectory), which stays the open lever from #48.
- Acoustic links can't do that; optical links can, over short range.
- That is what FAU and UNH are building: hybrid fleets, optical when close and aligned, acoustic otherwise.

**The model** (`comm_channel.py`, still no physics imports):
- **Received power.** Received light power is `exp(-c·d)/d²`. The water's attenuation `c` is:
  - `clear` 0.15/m
  - `coastal` 0.40/m
  - `harbor` 2.2/m
  - `custom`: set by the lab
  - These are textbook figures, marked assumed.
- **Pointing gain.** The power is multiplied by a pointing gain: 1 if the receiver is inside one of the sender's terminal beams and the sender is inside one of the receiver's fields of view.
- **Delivery probability.** A logistic curve in log power turns that into a delivery probability. Its midpoint is set by `rated_range_m`: 50% at 30 m in clear water, which gives ~15 m coastal and ~4 m harbour.
- **Link properties.** Fixed 1 ms latency; a byte budget of `data_rate_bps·dt/8` per tick; one random draw per send, even when blocked. Delivery is quantised to the 0.1 s tick, so optical's 1 ms and acoustic's few ms both arrive on the next tick: no latency advantage is modelled. Optical's edge is send rate and loss.
- **Terminals.** Half-angles everywhere; 180 = omni, and more than 180 is refused as a full angle.
  - Presets: `omni` (default), `forward_cone`, `fore_aft_cones`, `down_cone`, or any list of axes.
- **Shared queue.** Acoustic and optical share one queue/bandwidth base class (`_QueuedChannel`). `send()` takes both vehicles' rotation matrices.
- **Uncalibrated.** Every description says so.

**In the fleet** (`fleet_env.py`, `LinkConfig`):
- **Links.** `link: acoustic | optical | hybrid`. Optical beacons go every `optical.beacon_interval_s` (default every tick). They are judged on the true positions and attitudes and carry the measured beacon.
- **Separate draws.** The optical link draws from its own seeded stream, so acoustic drops are the same with or without it.
- **Hybrid `optical_first`.** A vehicle skips its acoustic beacon only while the link is up: an optical message from its teammate arrived within `link_up_ticks`, default 1 (the previous tick). The skipped beacon still uses its acoustic draw, which keeps the pairing. The report counts skipped beacons whose optical copy was lost too.
  - The first version used "heard since my last acoustic beacon". The smoke test found that one early receipt suppressed a whole 3 s acoustic window after the link had dropped: fore/aft cones went 0/10 against acoustic's 5/10. A 0.3 s timeout fixed that (7/10); review then narrowed it to one tick (below).
- **Asymmetric terminals** (beam ≠ receiver half-angle): hearing your teammate doesn't mean it hears you. `optical_first` runs as `redundant` there, and the report says so.
- **Stale copies.** Beliefs keep the newest send tick whichever link carried it (`fleet_belief.on_tick`, unchanged). A late acoustic copy never rolls one back; a test pins the user's t = 5.0 vs 4.9 case.
- **Observation unchanged.** It is still 13 numbers, so every shipped policy runs with any link.

**Where it's exposed:**
- the experiment YAML: `communication.link` (a list sweeps), `optical:` with `water` (a list sweeps) and `compare_omni`, `hybrid_policy`, `link_up_ticks`; the defaults are written into the saved `experiment.yaml`
- `strathe fleet --link ... --optical-*`
- `strathe train-fleet` / `train_fleet_policy.py --link ...`
- both web forms
- MLflow tags

**The report's "Optical link" section:**
- the settings, with the uncalibrated note, and a start-geometry warning ("0% of vehicle pairs are inside each other's cones")
- per condition: acoustic and optical delivery, acoustic beacons skipped, the share of belief updates by light, and the median belief age
- paired tables: each link against acoustic only, and cones against omni

**What it gives the shipped policies** (`run_optical_study.py`, `results/fleet_optical/`; criteria fixed before the run; 58,800 paired episodes; PPO seeds 1–3, trained on acoustic links only; beliefs calibrated/last_beacon; L2/L3a at 1 s/3 s):
- **Gate:** passed, with 0 bookkeeping failures.
- **Omni terminals, hybrid vs acoustic** (success, pts, 95% CI):

  | | clear | coastal | harbour |
  |---|---|---|---|
  | L2@1s | +0 | +0 | +0 |
  | L2@3s | +15 (+13 to +17) | +15 | +15 |
  | L3a@1s | +3 (+2 to +4) | +3 | +3 |
  | L3a@3s | +33 (+31 to +36) | +33 | +33 |

  - Hybrid is at 100% everywhere.
  - Collisions fall from 0–3% to 0%.
  - The median teammate-belief age falls from 0.7–2.5 s to 0.10 s.
  - Optical only matches this, except −1 point in harbour water at L2@1s.
- **Water barely matters at these ranges:**
  - Harbour water delivers 80–82% of optical beacons, against 97–98% in clear or coastal water.
  - The starts are at 2–5.5 m and the formation is at 2 m, where harbour delivery is ~99%.
- **optical_first vs redundant:** +0 in all 12 conditions.
- **Noisy mode** (hybrid coastal omni vs acoustic):
  - Hybrid is at 94% against acoustic's 90 / 71 / 81 / 56%: +4 to +38 points, all significant.
  - Hybrid stays 6 points below its clean 100%: a fresher link doesn't recover what the noise costs.
- **Pointing (coastal, 30° cones):**
  - **Delivery.** `forward_cone` delivers 0% of optical beacons: the start layout blocks it, and the report warns. `fore_aft_cones` delivers 5%. The shipped policies roll up to ±170° and pitch up to ±60° while holding formation (#50), so a cone rarely stays on the teammate.
  - **Hybrid with cones** is no better than acoustic on success, 3 points or less either way (+4 at L2@3s).
  - **Optical-only with cones** falls to 12% (forward) and 34% (fore/aft).
- **Collisions (flagged by criterion 6):**
  - `fore_aft_cones` raises collisions: hybrid 12% vs 3% at L3a@3s (+9, +7 to +11); optical only 32% vs 0–3%.
  - **Mechanism, measured:** the vehicles start inside each other's cones, so the first optical fix arrives at tick 2. The policy starts closing, rolls out of the cones, and the link goes quiet.
  - All 14 hybrid collisions in an 80-episode check came with a belief 4.6–7.7 s old; acoustic only had none on the same episodes.
  - An intermittent high-rate link is worse than a slow steady one for a policy that never learned to keep it up.

**Stress test** (`stress_test_optical.py`, `results/stress_optical/`): 5,125 episodes, 11/11 checks (after review; below).
- **Coverage:** every link (acoustic, optical, hybrid optical_first/redundant) × water (4) × terminals (omni, 3 cone presets, an asymmetric pair) × L0/L2/L3a/L3b × 1/3 s beacons × clean/noisy × buoyancy (none, ±5 N, a range) × PPO/SAC/TD3/random/zero, plus every link configuration on MuJoCo. All episodes finite and consistent.
- **Exact checks, zero policy:**
  - optical deliveries nested clear ≥ coastal ≥ custom ≥ harbour, episode by episode (360/360)
  - forward cones deliver nothing from the line start (144/144)
  - hybrid redundant has acoustic sends, deliveries and draws identical to acoustic only (480/480), and a teammate belief never older, tick by tick (480/480)
  - optical_first's acoustic draws are paired with acoustic only (480/480)
  - asymmetric terminals skip nothing (96/96)
  - omni in clear/coastal water does skip (48/48)
- **Determinism:** 120/120 reruns identical. The first pass reported 88/120; all 32 mismatches were the harness's cached random policy carrying its stream between episodes, and every other policy was identical. Fixed in the harness, not the simulator.
- **Acoustic unchanged:** 96 acoustic cells (stand-in and MuJoCo, clean and noisy with buoyancy, L0–L3b, 0.5–3 s) are hashed observation by observation against main before the change: identical. `clean_regression_check.py` is also identical: 6,616 fleet episodes, 16 single-vehicle runs.

Tests: 37 new, 41 after review (`tests/test_optical_link.py`): pointing cones, bad settings, range and water, determinism, one draw per send, pairing, the skip rule and its starvation case, stale copies, bit-identical acoustic, the runner, training, the CLI and the web forms; 493 + 142 fleet, all passing. CI also runs a fleet smoke run with `--link acoustic optical hybrid --optical-water harbor --optical-terminals fore_aft_cones --optical-compare-omni`.

**Review follow-ups (2026-09-28).**
- **Skip window narrowed to one tick.** `optical_first` now skips a vehicle's acoustic beacon only if an optical message from its teammate arrived on the previous tick (`link_up_ticks`, default 1). Symmetric terminals make the geometry reciprocal at an instant, but each direction draws its losses independently, so the window matters more than the symmetry check.
- **Skipped beacons are now accounted for.** Reports count skipped acoustic beacons whose optical copy, sent the same tick, was lost too, so the teammate got that state on neither link:
  - study: 0% in clear and coastal water, 4–6% in harbour water
  - stress grid: 3.2% of 6,025 skips
- **Study rerun with the 1-tick window:** every success, collision and pointing number above is unchanged.
- **No latency credit.** Delivery is quantised to the 0.1 s tick, so optical's 1 ms and acoustic's few milliseconds both arrive on the next tick. Optical's advantage in the simulation is send rate and loss. This is stated in the report, LIMITS, CUSTOM_EXPERIMENT and the README, and a test checks both links deliver on the next tick.
- **"Intent sharing" withdrawn.** `Beacon.tau` is the wrench delivered on the send tick, so a 10 Hz optical link is high-rate *state* sharing. Intent sharing (planned commands or a trajectory) stays the open lever from #48, and every doc now says so.
- **Scope of the study stated.** The shipped policies have no input saying whether optical is up; only the belief age shows it. The study therefore answers whether fresher teammate information helps existing policies, not whether a fleet learns to stay in optical range. That is written in `criteria.md` as a dated addendum after the first run (it wasn't there before any result), and in the report. Training with `--link hybrid` is the follow-up.
- **Validation:**
  - `--optical-interval` must be a whole number of ticks; this is now refused at the command line too, not only when the simulation is built.
  - `down_cone` is tested to point at a vehicle directly below. The body frame is Fossen (z down), and a test pins that the simulators' world z is down (a buoyant vehicle rises to −z).
- **Stress check made robust.** Forward cones deliver nothing over the whole episode in clean mode. In noisy mode, where the disturbance rotates the vehicles, only the first optical beacons (tick 1) are checked.
- **Long runs stay awake.** The optical study and stress test now keep the machine awake like the other long runners. The rerun's 643-minute wall clock is mostly the Mac asleep with its lid closed.

**Next.**
- Train with `--link hybrid`, cones included, and ask whether a policy learns to stay pointed and in range, and what that costs in collisions, from clear to harbour water.
- Calibrate the delivery curve from a lab's link logs.

### 60. Open-water sensor noise measured from SOLAQUA; the noise cost is the starting position fix (2026-10-01)

**Why.** The `typical` sensor preset, the default of `--mode noisy`, was an assumption: the most visible one in Strathe, flagged in every report. SINTEF Ocean's public SOLAQUA dataset (a BlueROV2 at a full-scale fish farm in Norway, August 2024; arXiv 2504.01790; CC BY-SA 4.0) logs a Water Linked A50 DVL, ArduSub's attitude and gyro, a depth sensor and a Sonardyne USBL.

**What the data is and isn't** (61 ROS 1 data bags, 173 MB; the 162 GB of video isn't needed):
- **Message definitions:** the bags carry their own, so `rosbags` reads the custom DVL, USBL and attitude types without SINTEF's code.
- **No thruster forces:** `/commanded_thrust` is ArduSub's pilot-input channels (pitch, roll, heave, yaw, forward, lateral as RC PWM), not per-thruster commands. ArduSub's mixer and depth-hold and stabilise controllers sit between them and the thrusters, so **SOLAQUA can't calibrate the physics.**
- **The DVL tracks the net:** it locks onto the fish-farm net at 0.5–2 m (median 1 m), not the seabed. The net moves with the current (0.04–0.2 m/s on the day), while the USBL measures against the earth.
  - Fitting the DVL's own dead-reckoned track to the USBL gives "drift" of 0.05–0.5 m/s per run: the net moving, not a sensor bias.
  - The A50's dead reckoning also resets mid-run, jumping 3–15 m between fixes.
  - **So the biases can't be measured from this data.**

**What was measured** (`fit_solaqua_noise.py`, 58 in-water recordings; the 3 camera-calibration sets on deck are excluded):
- **White noise:** `fit-noise`'s robust second-difference estimator, applied to each sensor **at its own logged rate**: DVL 9 Hz, attitude/gyro 16 Hz, depth 16 Hz. Interpolating onto a common grid would smooth the noise and bias it low, so nothing is resampled.
  - DVL 0.54 cm/s (vs `typical` 1)
  - gyro 0.32°/s (0.6)
  - roll/pitch 0.08° (0.5)
  - heading 0.06° (2)
  - depth 0.24 cm (1)
- **DVL dropouts:** invalid readings plus time in gaps over 3 nominal periods: 6.5% (vs 2%).
- **USBL fix error:** 0.17 m per axis, used as the starting-fix error (vs `typical`'s assumed 0.2 m).
  - **Method:** in 20 s windows, the USBL track is fitted to the DVL's dead reckoning by rotation, scale, offset and drift. Over 20 s the DVL track is good to a few cm, and the drift term absorbs the net's motion.
  - **Check on synthetic tracks:** it recovers known errors of 0.05 / 0.1 / 0.2 m as 0.050 / 0.094 / 0.195.
  - **Spread:** per-run values have an interquartile range of 0.13–0.32 m. The USBL's own reported σ is 0.1 m.
- **Cross-check:** an earlier, uncommitted attempt (left in the working tree, never merged) got the same white noise to within 5%. Its bias estimates (heading 23°, velocity 1.5 cm/s) came from the same DVL-vs-USBL comparison, so they aren't used.
- **Caveats:** a DVL at 0.5–2 m is quieter than one bottom-locked at 10–50 m. Gyro, roll/pitch and heading are ArduSub's filtered estimates.

**The preset.** `--noise open_water` (`sensor_noise/open_water_bluerov2_solaqua.yaml`):
- **Measured:** white noise, dropouts and the starting fix.
- **Assumed:** the four biases are `typical`'s values. The file lists them under `assumed_fields`, and every description says "except velocity bias, gyro bias, attitude bias, heading bias, which are ASSUMED".
- **Licence:** the derived statistics are shared under CC BY-SA 4.0 with attribution to SINTEF Ocean. The raw bags stay out of the repo (`data/raw/` is ignored).
- **Default unchanged:** `typical` stays the default of `--mode noisy`.

**Correction to #58: the noise cost is the starting position fix, not the sensors.** Every mix of the `typical` sensors cost the same 10 points, so the sensor model was taken apart (`run_noise_decomposition.py`, `results/fleet_v2_noise/decomposition.md`).
- **Not pre-registered:** this is an exploratory follow-up, labelled as one.
- **Setup:** 26,400 paired episodes; shipped PPO seeds 1–3; beliefs calibrated and last_beacon; L2 @ 1 s and L3a @ 3 s; the standard grid.

| Variant | vs clean, pts (95% CI) | Navigation error |
|---|---|---|
| typical as shipped (#58) | −10 (−12 to −8) | 26 cm |
| open_water as shipped | −8 (−10 to −6) | 22 cm |
| starting fix only, perfect sensors: 0.05 / 0.1 / 0.2 / 0.3 m | −1 / −3 / −9 / −18 | 6 / 13 / 25 / 38 cm |
| typical sensors, perfect fix | −1 (−2 to −0) | 4 cm |
| typical white noise only / biases only, perfect fix | −0 / −1 | 2 / 3 cm |
| open_water sensors, perfect fix | −1 (−3 to −0) | 3 cm |

- **Why the fix dominates.** Each vehicle's starting fix error is independent, so the teammate's believed relative position is off by about 1.4× the fix error, against a ±0.2 m tolerance. Over a 20 s episode, dead-reckoning drift barely builds up.
- **What was wrong in #58.** #58 said the cost "comes through navigation drift". The numbers stand; the explanation was wrong. Its "×2 costs 34 points" is mostly ×2 doubling the fix to 0.4 m.
- **The assumed 0.2 m fix was about right:** the measured open-water fix is 0.17 m.
- **The lever:** relative positioning between the vehicles (acoustic ranging, a shared USBL), not better sensors.

**The starting fix as its own setting.**
- **Where:** `--start-fix-error M` in `strathe fleet`, `arena`, `train-fleet`, the training scripts, both web forms, and `sensor_noise.start_fix_error_m` in the experiment YAML.
- **Behaviour:** it overrides the preset's starting fix without scaling the sensors. With `mode: noisy` and no preset it uses `typical`; with sensor noise off, or external positioning, it is refused.
- **Bit-identical:** the starting fix is drawn from the same random number as before, so 18 noisy fleet and arena runs (typical, tank, typical ×2; both engines) are bit-identical.
- **Reports:** the fleet report now names the starting fix and its share of the cost.

Tests: 10 new (`tests/test_open_water_noise.py`), one of which refits real SOLAQUA bags when they're present.

### 61. Research leftovers removed before the first lab pilots (2026-10-03)

**Why.** A lab opening the repo saw about 150 one-off scripts and 35 stale output folders next to the product, with no way to tell which ones mattered.

**What was removed:** 402 tracked files (12.5 MB) that nothing uses or mentions.
- **152 one-off scripts:** README patchers, `add_finding*_patch.py`, `diagnose_*`/`check_*`/`fix_*` scripts from findings #1–#50, and the old `test_parallel_restarts_v1`–`v6`.
- **250 stale outputs:** 35 top-level `*_placeholder`/`*_check`/old `calibration_output_*` folders, 14 under `calibration/`, 79 old CSVs (comm curves, monitors, recovery grids) and 12 superseded PPO zips.

**How "unused" was decided** (a script, not by hand):
- Start from the product: `strathe.py`, the CLI and web modules, every test, the CI workflow, and every file the user docs name.
- Follow imports and file names mentioned in code, recursively.
- Keep anything still mentioned by name in any file that stays: every `.md` (this file included), every script and every YAML.

**Kept:** everything in `results/`, every shipped policy, the templates, the sensor-noise presets and the calibration outputs this file cites.

**Getting one back:** tag `research-archive-2026-10` is the last commit with all of them, e.g. `git show research-archive-2026-10:calibration/diagnose_speed.py`.

**Checked:** 503 tests (1 skipped), the 142-test fleet suite and the 10 root tests pass after the removal.

### 62. A second public BlueROV2 dataset, run through the lab flow as a new lab would (2026-10-03)

**Why.** Before the first pilot emails go out, the whole lab flow had to work on data Strathe had never seen, run as a lab would run it: download, prepare, `strathe calibrate`, read the report, then `strathe arena` and `strathe fleet` on the result.

**The data.** Khanmeh, Wehbe, Simetti and Indiveri, Scientific Data (2025), Zenodo 10.5281/zenodo.17360027, CC BY 4.0.
- **Vehicle:** a standard 6-thruster BlueROV2 (ArduSub), with a gripper and ballast.
- **Where:** DFKI's 23 × 19 × 8 m basin.
- **Tracking:** 12-camera Qualisys motion capture.
- **Tether:** its pull is measured at the surface.
- **Runs:** 14 piloted runs, 63 minutes in all. 11 have a limited tether length; 3 are free (circle, serpentine, random).
- **Unlike the reference data in every way that matters to a lab:**
  - one CSV per sensor, at different rates
  - a 6-thruster vehicle
  - world-frame motion-capture velocity
  - no dates

**What it found in Strathe, and what was fixed:**
1. **The BlueROV2 Heavy thruster preset was in ROS axes.**
   - **The cause:** the reference dataset is a ROS recording. Its world z points up (every depth is negative), and its body velocities rotate onto the world track with that same up-pointing z (correlation 0.92–0.97 on every axis). So its body frame is forward-left-up, not forward-right-down. The allocation matrix was recovered from it exactly, so it was in FLU too: the mirror image, in y and z, of ArduSub's own motor definitions.
   - **The effect:** for a lab's ArduSub logs (FRD), the preset's sway, heave, pitch and yaw forces had the wrong sign. Calibration would have fitted those four axes to forces pointing the wrong way.
   - **The evidence on DFKI's ArduSub vehicle:** motors 1–4 are the same in ArduSub's 6- and 8-thruster frames. Through the old preset, yaw torque ran *against* ArduSub's own gyro yaw rate in 14 of 14 runs (median correlation −0.67).
   - **Independent confirmation:** the pilot's sticks (RC_IN) mapped through the old preset come out mirrored: "lateral right" became a push to the left.
   - **The fix:** `bluerov2_heavy` is now FRD in ArduSub's motor numbering. Its 8 columns match ArduSub's VECTORED_6DOF motor factors in sign, and a test checks that. The as-recovered matrix stays available as `bluerov2_heavy(frame="reference")` for reproducing the dataset's own forces, which it still does on all 183,555 rows. The custom-layout template and the allocation CSV template were rewritten the same way, with each motor's place named.
   - **The shipped calibration:** unchanged, because it was fitted to the FLU data as-is and is self-consistent. It is the same vehicle turned half a turn about x. The equations are unchanged by that turn except for two signs: the restoring force's net buoyancy and the surge-pitch coupling `Xq_dot`. So those two fitted values have the opposite sign to the real vehicle's. With `--thrusters bluerov2_heavy`, the vertical thrusters' stronger direction (47.9 vs 37.2 N each) is also reversed relative to gravity in its simulations. Formation-holding and station-keeping are unaffected (LIMITS.md).
   - **What changed in simulation:** only thruster-limited runs, because policies often command more pitch torque than the Heavy can give (30 vs 17.9 N·m), and the command is scaled down by whichever thruster binds first.
     - **Fleet:** in `clean_regression_check.py`'s 216 thruster-limited fleet episodes (with disturbance and current), 192 differ. Success changed in 3 (96.8% → 97.2%), collisions in none, and separation error by a median 9 mm.
     - **Arena:** the 8 single-vehicle runs with thrusters on differ.
     - **Everything else:** the other 6,400 clean episodes and 8 arena runs are identical. The reference was regenerated with this note (`clean_reference_62.json`).
     - **Committed studies:** none use thruster limits, so no published number changes.
2. **`strathe calibrate` only worked on logs that carried the reference dataset's dates.**
   - **The cause:** three steps defaulted to its held-out dates (16-02-2024, 19-02-2024): the data check (`check_coriolis_excitation.py`), the identifiability step and the report.
   - **The effect:** any other lab's calibration stopped at step 2. The message "the data check found problems" pointed at no message at all, because the traceback had been written above the step's header.
   - **Why tests missed it:** every calibrate test replaced the steps with stubs.
   - **The fix:** the three steps now use calibration's own split. The data check flushes its output in order. `tests/test_calibrate_end_to_end.py` runs every step as its own process, on logs dated 2030. Without the fix it fails.
3. **No way back from an interrupted run in one command.**
   - **What happened:** the DFKI fit converged after about 400 evaluations (cost 457.6 → 28.35, flat from there). The laptop then slept for 2.5 hours with the fit still running (`stall_watchdog` caught it: 9,237 s wall against 31 s monotonic), and the run was stopped.
   - **The old recovery:** the documented `finalize_checkpoint.py` command needs the fit's hidden settings re-typed: `--axis-scaling zero_rms`, the torque point and the rate. With the defaults, it would have re-evaluated with the reference dataset's torque point.
   - **The fix:** `strathe calibrate ... --finish` (the same command, plus `--finish`) does it with the run's own settings, then checks identifiability and writes the report.
4. **Smaller:**
   - **One recording session:** a lab with only one was told "Need at least 2 distinct groups". Ingestion now warns, and calibration explains what to do in plain words.
   - **A stray hint:** `calibrate` printed "Next: python strathe.py calibrate ..." in the middle of its own run.
   - **MLflow noise:** MLflow 3's hint for AI agents appeared in `strathe check`.

**What a lab has to do with data like this** (`calibration/examples/dfki_tether/`, a worked example to copy):
- **One CSV per recording:** CSV ingestion reads one table per recording, so the per-sensor CSVs are merged onto ArduSub's ~10 Hz timestamps.
- **Velocity:** taken from the origin marker's track (seen about 100% of the time). The live motion-capture velocity stream misses up to a third of each run, and the two agree to 2 cm/s where both exist.
- **Heading:** ArduSub's yaw is mirrored relative to the basin, offset by about 175° and drifting up to 15° near the basin's steel. The script corrects it by its offset from the markers, tracked wherever the markers are seen.
- **Thrusters:** ArduSub's vectored frame. Which motors are reversed (1, 2 and 5) comes from regressing each motor's output on the pilot's sticks (R² 0.94).
- **Assumed:** where the vertical thrusters sit.
- **Sessions:** with no dates, the limited-tether runs and the free runs are labelled as two sessions.
- **Check:** after ingestion, every force moves with its velocity (surge 0.80, sway 0.57, heave 0.74, yaw 0.65). With a mirrored layout, sway, heave and yaw would be negative.

**The calibration** (`results/pilot_dfki_tether/first_run_62/`; superseded by the rerun in #63; 14 limited-tether training recordings, 8 free-piloting held-out ones; the reference BlueROV2 Heavy `vehicle_spec.yaml`, as a new lab would start):

| Axis | Held-out skill vs "no motion" | vs default physics | Verdict in the report |
|---|---|---|---|
| surge | +0.60 | −4.6% | don't rely on it |
| sway | +0.62 | +85% | use with care (drag at its limit) |
| heave | +0.39 | +62% | use with care |
| roll | −0.26 | +88% | don't rely on it |
| pitch | −0.01 | +85% | don't rely on it |
| yaw | +0.43 | +56% | trust it |

- **Surge is worse than the default model on the free runs.** The training runs had a tether held at 5–15 m, which holds the vehicle back. The fit learned forward drag at its limits (`Xu` −50, `Xuu` −100), which the free runs don't have. The report flags both.
- **Roll and pitch aren't predictable from the thrusters here.** A 6-thruster BlueROV2 can't pitch on command, and the tether pulls on it. The report says so and recommends free-decay tests.
- **Overall:** the report's bottom line ("trustworthy for 1 of 6 axes", 62% better than the default on average) and its next steps are what a lab with this data should hear.

**The rest of the flow on the calibrated vehicle, with its own 6-thruster limits** (`--thrusters` the example spec):
- **`strathe arena`, station-keeping, `--compare-generic`, default budget (150k steps):**
  - **Result:** policies trained on the calibration succeed 78% (71–83%), and policies trained on default physics 100% (−22 points, CI −28 to −18).
  - **The report explains it:** the budget is probably too short for a heavily damped vehicle, so re-run with a larger `--timesteps`.
  - **Following that advice (450k steps, 13.5 min):** both reach 100%, and the hand-written controller 96%.
  - **So the advice holds.**
- **`strathe fleet --params` with the shipped fleet policies:**
  - **The results:** success 46–63%, against 52–100% on the reference vehicle. The belief comparison is mixed: one condition favours calibrated, one generic, two show no difference.
  - **The cause:** those policies learned on the reference Heavy, and nothing in the report said so. A lab would have concluded that calibration doesn't matter for coordination.
  - **New caution:** the report now compares each policy's training vehicle (from its metadata) with the experiment's, and says when they differ and how to train policies for this vehicle.
  - **Following it:** one policy trained on the DFKI vehicle (`strathe train-fleet`, 3M steps, 20.6 min) gives calibrated +12 points over generic at L2@3s, L3a@1s and L3a@3s (CIs +3 to +20, +6 to +17, +3 to +20), and 100% vs 98% at L2@1s.
  - **So the coordination result holds on a second, independently calibrated vehicle,** once the policy is trained for it.

**The fresh install** (a clean clone from GitHub, a new Python 3.13 environment, `pip install -r requirements.txt`, `strathe check`, the reference dataset's download script, both test suites, then `strathe fleet`) works end to end in about 14 minutes on an M1. The only blemish was the MLflow hint, now gone.

Tests: 8 new.
- `tests/test_calibrate_end_to_end.py` (2): every step for real, and the one-session messages.
- `tests/test_dfki_tether_pilot.py` (2): needs the DFKI data, which CI now downloads, pinned by checksum.
- `tests/test_thruster_model.py` (2): the ArduSub motor table, and the reference dataset's frame.
- `--finish` on the command line and on the web form (2).
- The fleet report's training-vehicle caution (1).
- 13 ingestion and thruster tests rewritten: they had encoded the reference dataset as FRD.

### 63. A review of the DFKI run: what not to claim, the tether, drag, battery voltage and automatic checks (2026-10-03)

**Why.** A review of #62 asked: which of its numbers can go in pilot emails, why surge got worse than the default model, and which by-hand discoveries the tool should make on its own.

**What not to claim, and what Strathe itself was claiming:**
- **Averaged improvement.** "N% lower held-out error" averaged over six axes is the trap #50 retracted: default physics is wildly wrong on roll and pitch (about 1.09 rad/s), so those two axes dominate the average while still being no better than predicting no motion. Strathe's own report did this: "62% more accurately than the default model on average" on DFKI, 69% on the reference calibration. **Fixed:** the bottom line now lists each axis that beats "no motion", with its skill.
- **The generalisation check.** It called a held-out/training error ratio below 1 "It generalises", using one error that mixes m/s with rad/s. On DFKI the held-out runs were free and the training runs tethered, so held-out error was lower because those runs were easier. **Fixed:** the median of the per-axis ratios; below 0.85 the report says the unseen sessions were probably easier and that this can't show generalisation. Both reports were regenerated (wording only).
- **Corrections to the review itself:**
  - The mirrored preset affected turning PWM into force for ArduSub logs; no lab had run that yet, and no shipped result used it. In simulation it changed 3 of 216 thruster-limited outcomes, so not "every thruster-limited result was wrong".
  - The clean regressions weren't identical: thruster-limited runs changed, by design (#62).
  - The #62 sleep wasn't `keep_awake` failing. macOS logged "Low Power Sleep ... Using Batt (Charge:1%)": the battery ran out, which no `caffeinate` flag prevents.
  - "Oracle below last-beacon means the policies don't fit the vehicle" would fire on Strathe's own shipped experiment (3 of 4 cells, e.g. 92.7% vs 100% at L2@1s, with policies trained on that vehicle). The training-vehicle caution from #62 is the reliable signal.
  - Roll and pitch inertia are already pinned within ±0.005 of published BlueROV2 values (#30); "at their limits" there means "at the published value".
- **For emails** (from the rerun below, which supersedes #62's numbers): "In simulation, on a second BlueROV2 calibrated from another lab's public data, a teammate model matching the vehicle beat generic physics by 10 to 22 points at 3 of 4 comm levels, and tied at the fourth." That's simulation, one policy seed, and the vehicle as calibrated by us. Quote per-axis skill (surge +0.80, sway +0.61, yaw +0.56, heave +0.45 on held-out free runs), never an average.

**Is the tether being learned as drag? Yes.**
- **Method:** fits on DFKI with per-sample battery voltage. Every fit is scored on the same held-out set (the 4 free "random" recordings), with the same budget (`--max-nfev 30`), so only the training data changes.
- **The tension sensor:** in the limited-tether runs the cable goes taut in bursts: median 0.1–3 N, 95th percentile 43–65 N, above 5 N for 20–40% of the time. The free runs never reach 1 N.

| Fit | Training data | Surge | Sway | Heave | Yaw | Surge vs default | Xu / Xuu |
|---|---|---|---|---|---|---|---|
| A | tethered (14) | +0.59 | +0.69 | +0.46 | +0.42 | +0% | −50 / −100 (both limits) |
| B | tethered + free circle/serpentine | +0.64 | +0.69 | +0.47 | +0.46 | +11% | −41 / −100 |
| C | free only (4 recordings) | **+0.79** | +0.67 | +0.53 | +0.55 | +49% | −11.6 / −84 |
| H | tethered, taut stretches dropped | +0.65 | −0.36 | −0.16 | −0.50 | +14% | −32.7 / −8.6 |
| **G** | **tethered with taut stretches dropped + free** | **+0.75** | +0.67 | **+0.55** | **+0.56** | **+39%** | −50 / −5.1 |

- **Conclusion:** training only on free runs gives the best surge, and dropping the taut stretches from the tethered runs (G) recovers almost all of it while keeping five times the data.
- **H on its own is worse:** the slack stretches are 10-second fragments of gentle motion, too little to pin sway, heave and yaw.
- **A single constant force per recording is the wrong model here:** the pull comes in bursts, so averaging it into one force misrepresents it. **What shipped instead:**
  - `exclude_samples:` in the ingest spec drops stretches where a logged column crosses a threshold (tension, a contact or override flag), splitting the recordings.
  - `examples/dfki_tether/ingest_dfki_tether_slack.yaml` drops the taut stretches.
  - LIMITS.md tells labs to record free sessions with the tether slack.
- **Dropping taut stretches** lifts the surge force/velocity correlation from +0.79 to +0.91.

**Drag: separating linear from quadratic.**

| Fit (data as B) | Surge | Sway | Heave | Roll | Yaw | Notes |
|---|---|---|---|---|---|---|
| B, free drag | +0.64 | +0.69 | +0.47 | −0.30 | +0.46 | Xuu, Yvv −100; Zww −3 (limits) |
| E, `--drag prior` (Wu 2018, weight 0.1) | +0.66 | +0.70 | +0.48 | −0.30 | +0.48 | Yvv −38, Zww −29: off their limits; Xuu still −100 |
| D, linear terms held at Wu (all six) | +0.44 | −2.42 | −0.60 | −11.3 | −0.67 | the replay goes unstable |
| D2, linear terms held at Wu (Xu, Yv, Zw) | +0.62 | −1.20 | −0.59 | −4.82 | +0.15 | the replay goes unstable |

- **The reference values:** Wu (2018), Flinders University, Table 5.3, standard BlueROV2. They are derived from the literature (Sandoy 2016), not measured, and roll and pitch are taken equal to yaw.
- **Bounds:** three of Wu's rotational linear terms (−0.07) are below the spec's bounds and are clamped to them, with a note in the log.
- **The prior** helps a little and moves two quadratic terms off their limits. It ships as an opt-in.
- **On the reference dataset it changes nothing.** That dataset is a BlueROV2 Heavy, and Wu's values are for the standard vehicle.
  - **Method:** free drag against `--drag prior` with the shipped calibration's settings and equal budgets. Both runs were stopped by a time limit at about 440 evaluations and finished with `--finish`.
  - **The prior fit ends at the shipped free-drag solution.** Its held-out skill matches to two decimals: surge +0.72, sway +0.51, heave +0.59, roll −0.03, pitch −0.10, yaw +0.58. The same drag terms sit at their limits. Each recording's residuals are normalised, so this isn't the dataset's size: these logs pull drag to the limits harder than a weight-0.1 prior pulls back.
  - **The free run at the same budget** hadn't reached that point yet (surge +0.50, sway −0.43). That is its optimizer path, not the prior: the shipped calibration is a free-drag fit that did reach it.
  - **So `--drag prior` stays opt-in:** a small gain on DFKI, none on the reference dataset.
- **Holding all six linear terms at Wu's values** makes slow-speed damping too weak, and the open-loop replay diverges. So does holding only Xu, Yv and Zw (fit D2): surge +0.62, sway −1.20, heave −0.59, roll −4.82. Published linear drag is too small for real logged motion, so `--drag linear-fixed` was removed rather than offered.

**Roll and pitch.** Excluding them (`--exclude-axes p q`, fit F, data as B) makes the other axes worse: surge +0.59, sway +0.57, heave +0.39, yaw +0.46, against B's +0.64 / +0.69 / +0.47 / +0.46. The roll and pitch replay is far worse too (−9.0, −2.8), because the held values don't damp them. So it is not made a default for tethered data, and the report doesn't recommend it. The free-decay advice stands.

**The pilot rerun** (`results/pilot_dfki_tether/`; the first run moved to `first_run_62/`). The whole lab flow was run again:
- **What changed:** `ingest_dfki_tether_slack.yaml` (taut stretches dropped) and per-sample battery voltage.
- **What stayed the same:** the split (tethered sessions train, free ones held out), so it's a fair before/after.

| | First run (#62) | Rerun (#63) |
|---|---|---|
| Calibration | stopped by a dead battery, finished with `--finish` | converged in 4.5 min |
| Held-out skill: surge / sway / heave / yaw | +0.60 / +0.62 / +0.39 / +0.43 | **+0.80** / +0.61 / +0.45 / **+0.56** |
| Held-out skill: roll / pitch | −0.26 / −0.01 | −0.25 / +0.01 |
| Surge vs default model | −4.6% | +46.5% |
| Axes the report calls trustworthy | 1 (yaw) | 0 (each better-predicted axis now relies on a drag value at its limit) |
| Arena, station-keeping (calibration / default / hand-written) | 78% / 100% / 96% at 150k; 100% / 100% at 450k | 100% / 100% / 85% at the 450k default |
| Fleet, shipped policies: calibrated − generic (L2@1s, L2@3s, L3a@1s, L3a@3s) | +2, −6, 0, +5 | 0, +8, +7, +6 |
| Fleet, a policy trained on this vehicle | +2, +12, +12, +12 | 0 (100% vs 100%), **+12, +10, +22** |

- **The conclusion holds,** and it is stronger on the better calibration.
- **The "0 of 6 trustworthy" verdict** follows the report's rule (an axis that relies on a value at its limit gets "use with care"). The rule was left as it is.

**Battery voltage.**
- **Effect:** from 15.3 to 13.1 V the T200 loses 17% of its maximum thrust (Blue Robotics' published data).
- **The fix:** `voltage_column:` scales each sample's thrust to its logged voltage, rescaling forward and reverse thrust separately, as `voltage:` does for one value. A steady log gives exactly the single-voltage result.
- **DFKI:** the example now uses it.

**Automatic checks for what #62 found by hand** (ingestion refuses the data; `skip_checks:` is the escape hatch):
- **Force direction:**
  - **The check:** on surge, sway, heave and yaw, force against velocity is pooled over every file. A correlation below −0.2 on an axis the thrusters drive means the forces are reversed: a mirrored layout, wrong `reversed:` flags, the wrong frame, or (with world-frame velocity) a mirrored heading. Every axis's correlation is printed.
  - **Roll and pitch:** reported, not judged. Restoring moments dominate them, and synthetic logs gave −0.22 with correct forces.
  - **On DFKI:** the old ROS-axes layout is refused on sway, heave and yaw.
- **Orientation against position** (with the optional `position:` and body-frame velocity): the body velocity, turned by the orientation, must follow the position track. A mirrored heading and a steady offset are named, with their size; tested on a synthetic circle with a 175° mirror like DFKI's and a 90° compass offset.
  - **What it could not have caught on DFKI:** there, velocity is world-frame motion capture, rotated into the body frame with the very orientation under test, so turning it back reproduces the positions whatever the heading is. The check only works with body-frame velocity (a DVL, or body-frame odometry).
  - **What flags DFKI's mirror:** a test with ArduSub's mirrored yaw showed the force-direction check as first built did *not* catch it. Surge fell from +0.79 to −0.05 and sway from +0.57 to +0.08: near zero, not below the −0.2 refusal threshold.
  - **The fix:** a warning when a driven axis sits between −0.2 and +0.2, which names a wrong heading as the likely cause with world-frame velocity. It is a warning, not a refusal: the data passes, and the lab has to act on it. Correct DFKI data stays clear (lowest driven axis +0.57), and `tests/test_ingest_checks.py` checks both cases.
- **Gaps and velocity source:** gap warnings give total time and share of the recording, and ingestion prints which columns the velocity came from.
  - **Limit:** these are gaps in the timeline ingestion receives. DFKI's live velocity stream was missing up to a third of each run, but inside a raw per-sensor file, before the lab's own merge step, where Strathe never sees it. Labs that merge sensor files themselves have to check each stream's gaps in that step (`examples/dfki_tether/prepare_dfki_tether.py` sidesteps them by taking velocity from the marker track).

**Also in this round:**
- **The arena** trains 450k steps by default (150k was too short on both vehicles).
- **`keep_awake`** warns when a run starts on battery, and again at 20%.
- **Windows output:** UTF-8 everywhere, so `calibrate` doesn't crash when its output goes to a file. The new end-to-end test caught this on the Windows runner; it now runs under an ASCII-only locale on every platform.

### 64. Roll and pitch from free-decay tank tests, and what the reference logs' own decays say (2026-10-04)

**Why.** Mission logs can't calibrate roll and pitch (#50). A BlueROV2 rights itself, pilots rarely command roll or pitch, and most of the logged roll/pitch comes from the tether and the water. The standard method is a free-decay test: tilt the vehicle, let go with the thrusters stopped, record the swing.

**What shipped:**
- **`calibration/FREE_DECAY_PROTOCOL.md`:** the tank test for labs. 8 releases per axis at about 10° and 25°, MANUAL mode with the sticks centred, the tether slack, fully submerged.
- **`templates/ingest_free_decay_ardupilot.yaml`:** converts ArduSub `.bin` tank-test logs.
- **`strathe free-decay`** (`free_decay.py`):
  - **Finds the releases:** thrusters stopped, 8–35° from rest, swinging back. A hold before letting go and a catch mid-swing are cut off.
  - **Fits each axis on its own:** the calibration's own roll/pitch physics, with the inertia held at the spec's value. It fits linear and quadratic damping, the righting arm (`cg_cb_offset_z`) and the rest angle.
  - **Writes:** `free_decay.yaml` and a short report.
- **`strathe calibrate --free-decay free_decay.yaml`** (and the web form): holds those values fixed. The report says they were measured and drops them from "stuck at a limit".
- **An axis is not held** with fewer than 3 releases, a fit explaining under 80% of the swing, or a value at the edge of the fit's range.

**Checked on synthetic tank tests with known answers** (`tests/test_free_decay.py`; the truth is integrated by scipy's adaptive solver, not the fitter's own steps):
- **Recovery:** damping within about 2–8%, righting arm 1.77 against 1.80 cm, rest angle exact, at 10 and 50 Hz, through a real ArduSub `.bin` round trip.
- **Tether tugs of 1–3 N·m and a steady wobble near the swing frequency are refused.** Fitted anyway, they turn a damping term off while the fit can still explain 90–98% of the motion, so fit quality alone is not the gate; the edge check is what catches them.
- **Roll and pitch disagreeing** on the righting arm flags the spec's inertia ratio (a true pitch inertia of 0.40 against the spec's 0.23 is reported as 1.54).
- **A bug found on the way:** a vehicle held tilted for long stretches pulled the rest-angle estimate towards the held angle, and a hand tilting it looked like a release. Rest is now estimated near level only.

**The reference dataset's own free decays.** Its mission logs contain natural releases: moments the thrusters stopped while the vehicle was tilted. Fitted on the training days only (12-01 and 14-02):

| | Roll | Pitch |
|---|---|---|
| Releases | 31 | 35 |
| Period (small swings) | 3.3 s | 3.0 s |
| Damping ratio | 0.24 | 0.10 |
| Fit explains | 83% | 89% |
| Damping, measured: linear / quadratic | −0.24 / −0.50 | −0.10 (at the spec's bound) / −0.50 |
| Damping, shipped calibration | −10 / −10 (at its limits) | −5 / −10 (at its limits) |
| Righting arm, measured (shipped 1.18 cm) | 0.89 cm | 0.96 cm |
| Rests at | +4.1° | +1.9° |

- **Roll and pitch agree** on the righting arm, as one vehicle should.
- **The measured values predict the held-out days' free swings** (16-02 and 19-02: 12 roll, 16 pitch releases): angle R² +0.84 and +0.96, rate skill +0.52 and +0.83. The shipped values do worse than "no motion" on the same swings (angle R² −2.3 and −2.1). So the shipped roll/pitch damping is not a property of the vehicle.

**But holding them makes whole-recording roll/pitch prediction worse.** Held-out skill with the shipped calibration and the measured roll/pitch values swapped in:

| | Surge | Sway | Heave | Roll | Pitch | Yaw |
|---|---|---|---|---|---|---|
| Shipped | +0.72 | +0.51 | +0.59 | −0.03 | −0.10 | +0.58 |
| Measured roll/pitch values swapped in | +0.69 | +0.47 | +0.34 | −3.73 | −3.73 | +0.57 |

- **What it isn't:**
  - the rest angle: re-levelling the logged attitude by +4.1° and +1.9° changes nothing;
  - the surge-pitch and heave-pitch couplings: setting `Xq_dot` and `Zq_dot` to 0 gives −3.92 / −4.11;
  - logged roll/pitch torque that's too big: removing it makes it worse (−9.4 / −10.5).
- **What it is:** both parameter sets predict roll rate with the same timing (correlation 0.43 on a held-out run), but the shipped values predict 5× too little motion and the measured ones 2× too much, mostly below 0.25 Hz. Letting roll and pitch evolve from their own angles, driven by the logged torques, the measured values tip the model far past anything the vehicle did (angle skill −63 and −326). With the vehicle's real, light damping, the roll/pitch the model gets from the thrusters is far too large.
- **So the damping was never the real problem.** The thruster-driven roll/pitch model is. Heavy damping pinned at its limit is how every fit hides it, which is why every fit pins it.
- **A full refit doesn't rescue it.** The measured values were held and everything else refitted, with the shipped calibration's settings (540 evaluations, budget used up, not converged):

  | | Surge | Sway | Heave | Roll | Pitch | Yaw |
  |---|---|---|---|---|---|---|
  | Shipped | +0.72 | +0.51 | +0.59 | −0.03 | −0.10 | +0.58 |
  | Refit, decay values held | +0.58 | +0.35 | −1.13 | −3.04 | −3.62 | +0.27 |

  - **Every axis is worse.** The fit's cost is 17× the shipped fit's (723 against about 43). To compensate, the optimizer pushed yaw inertia to its lower limit (0.15 against 1.89), yaw damping to its limit, and heave linear damping almost to zero.
  - **The result is unstable:** replayed 3 s at a time, the simulation blows up on 9 of the 14 held-out recordings.
  - **So the conclusion stands:** the thruster-driven roll/pitch is what's wrong, and the rest of the model can't make up for it.
- **A crash this exposed, fixed:** the finishing step of `strathe calibrate` died with a traceback when the calibrated simulation blew up on a held-out recording. A lab using `--free-decay` could hit it. Diverged recordings are now counted (`metrics.json` `skill_vs_zero.diverged_heldout`) and left out of the skill, and the report opens with "Unstable: ... Don't simulate with this calibration".

**What it means for a lab:**
- `strathe free-decay` measures the vehicle's unforced roll and pitch correctly.
- `--free-decay` is opt-in. With it, the report's roll/pitch verdict points at the thruster-driven part, not the logs, and the next step is deliberate roll/pitch commands with the tether slack (the protocol's optional section), which Strathe doesn't fit yet.
- Compare with a calibration made without it before simulating attitude-sensitive tasks.

### 65. Validation: position drift, orientation on its own, response and plots, for every calibration (2026-10-06)

**Why.** The calibration report showed one thing: velocity error per axis, with the logged orientation fed in. It didn't show where the simulated vehicle ends up, whether it turns and tilts like the real one when left to itself, how it responds to a command, or any plot. #64's roll/pitch finding only surfaced when the angles were let go by hand.

**What shipped:**
- **`validation.py`**, run by `strathe calibrate` after the fit (a step that warns rather than stops if it fails), and by `strathe validate --result <folder>` for any existing calibration, which then rewrites its report. It works on the held-out recordings and writes `<result>/validation/`: `validation.json` and 7 plots.
- **Position drift:**
  - **How:** predicted velocity is turned into a path with the logged orientation, from the real state at the start of each window (5–60 s). It's compared with the logged velocity integrated the same way. So it is the drift velocity error alone causes.
  - **Baseline:** "the vehicle doesn't move" (the distance it actually travelled).
- **Orientation:**
  - **How:** a free-running replay. The model turns and tilts on its own: the fit's own equations and step (`FastDynamics`, 0.005 s), fed its own attitude, restarted from the real state every 10 s. Roll, pitch and yaw error after 3 and 10 s.
  - **Baseline:** holding the starting attitude.
- **Response:**
  - **The 1 s response:** the velocity change over the next second (the average acceleration), from every moment, against "the velocity stays the same".
  - **Command steps:** where a force or torque jumps and holds, the time to 63% of the velocity change and its size, model against real.
- **Plots:**
  - velocity over whole recordings (best, typical and worst);
  - the path seen from above, and depth;
  - drift against horizon;
  - orientation with the model on its own;
  - step responses.
- **In the report:** a section "How the simulation compares with the real vehicle", with tables, plain verdicts ("worse than simply holding the starting attitude"; "responds faster than the real one" when a majority of axes do) and the plots inline. The web page now shows a report's plots.

**Checked against a known answer** (`tests/test_validation.py`). A recording the model produced itself, integrated independently with the rotation-matrix physics, comes out as:
- **position drift** under 1 cm;
- **orientation error** about 1e-13°;
- **the true response** to command steps.

The baselines are not zero, and a model with four times the slow-speed drag shows up as drift and a smaller step response.
- **A mismatch this caught while building it:** the first free-running replay stepped at 0.02 s while the calibration integrates at 0.005 s, a 4% difference after one sample. It now uses the calibration's own step and acceleration routine.
- **Test scenario:** with default physics the synthetic vehicle tumbled past 50° and any difference grew chaotically, so the test uses the shipped calibration as the "real vehicle".

**On the two datasets** (committed: `calibration_output_armfix/validation/`, `results/pilot_dfki_tether/calibration/validation/`):

| | Reference (BlueROV2 Heavy, 14 unseen recordings) | DFKI (6-thruster BlueROV2, 8 unseen recordings) |
|---|---|---|
| Position drift after 10 / 30 / 60 s, calibrated (median) | 0.84 / 1.33 / 1.88 m | 0.48 / 0.97 / 1.51 m |
| Default physics | 3.23 / 6.64 / 11.06 m | 1.71 / 3.74 / 5.91 m |
| Assuming it doesn't move | 3.06 / 3.60 / 3.21 m | 3.91 / 5.27 / 3.83 m |
| Yaw after 10 s on its own: calibrated / holding the start | 43° / 95° | 26° / 79° |
| Roll after 10 s: calibrated / holding the start | 8.7° / 3.8° | 34.9° / 1.4° |
| Pitch after 10 s: calibrated / holding the start | 10.7° / 6.3° | 19.2° / 8.2° |
| 1 s response skill (surge, sway, heave, roll, pitch, yaw) | +0.18, +0.25, +0.45, +0.21, +0.24, +0.62 | +0.45, +0.45, +0.01, +0.20, +0.36, +0.62 |
| Step response, time to 63%: real / calibrated | surge 0.6 / 0.3 s, sway 0.5 / 0.3 s, heave 0.6 / 0.4 s, yaw 0.5 / 0.5 s | surge 0.9 / 0.8 s, sway 0.9 / 0.6 s, heave 0.3 / 0.8 s, yaw 0.4 / 0.3 s |
| Step response, size calibrated / real | 0.97–1.08 | 0.92–1.01; heave 1.58 |

- **Position and heading:** calibration clearly earns its keep. Drift is a quarter to a third of default physics', and the simulated heading on its own beats holding the start.
- **Roll and pitch on their own are worse than holding the starting attitude on both vehicles.** That is #64, now in every report. Don't trust simulated roll and pitch on their own.
- **On the reference vehicle the model responds about twice as fast as the real one** (0.3 against 0.6 s on surge), at the right size. Thruster spin-up and command lag aren't modelled (LIMITS.md). DFKI is closer, except heave.
- **What drift is measured against:** the logs' own velocity, integrated. It isn't an independent position fix (the reference dataset has none; DFKI's motion capture would be one, not used yet).

### 66. Thruster lag (2026-10-06)

**Why.** Validation (#65) showed the simulated vehicle responding about twice as fast as the real one to a command step: on the reference BlueROV2, 0.3 s to 63% of the velocity change on surge against the real 0.6 s, at the right size. Thrust followed the logged command instantly; real propellers take a fraction of a second to spin up.

**What shipped:**
- **`thruster_time_constant_s`**, a first-order lag on the applied wrench: d(applied)/dt = (commanded - applied) / T. One time constant for every thruster: identical thrusters, and the wrench is linear in their thrusts. Each 0.1 s sample applies the exact average of the lagged wrench over that sample.
- **One discretisation everywhere:**
  - the calibration replay lags a whole recording before cutting it into prediction windows, so each window starts from the thrusters' real state;
  - validation does the same;
  - the arena and fleet cores (`ArenaCore`, `FastCore`, `MuJoCoArenaCore`) step it, starting each episode with the thrusters stopped;
  - the fleet's physics teammate models start at the command in the last beacon.
- **Opt-in, and nothing changes without it:**
  - **In the spec:** a `thruster_dynamics` section in `vehicle_spec.yaml` makes it a calibration target (an example is in the file, commented out).
  - **Without that section** it is held at 0, which is the model before. The same goes for older calibrations and labs' own specs. Zero-width bounds now mean "held", and are recorded as held.
  - **Checked:** every existing result and the bit-identical regression checks are unchanged.

**Checked** (`tests/test_thruster_lag.py`):
- **Exact discretisation:** the sample averages match the analytic step response to 1e-12.
- **Same lag everywhere:** the series form and the step-by-step form agree, and all three cores lag identically (MuJoCo within 3e-6 m/s, its own integration).
- **A known lag is recovered:** a calibration finds 0.25 s from commands and velocities alone.
- **Validation sees it:** a model with the right lag gets the step rise time right, and one without lag responds too fast.

**On real data:**
- **DFKI:** a full calibration with the lag free fits 0.004 s: no lag. Every score is unchanged to two decimals. DFKI's step responses were already close (#65).
- **Reference BlueROV2:** the shipped calibration held, only the lag fitted on the training days: **0.18 s**. On the held-out days:

| | Surge | Sway | Heave | Roll | Pitch | Yaw |
|---|---|---|---|---|---|---|
| Held-out skill, shipped | +0.72 | +0.51 | +0.59 | −0.03 | −0.10 | +0.58 |
| Held-out skill, shipped + 0.18 s lag | +0.73 | +0.52 | +0.60 | **+0.04** | −0.05 | +0.51 |
| Step, time to 63%: real | 0.6 s | 0.5 s | 0.6 s | | | 0.5 s |
| Step, time to 63%: shipped / with lag | 0.3 / **0.6** s | 0.3 / **0.5** s | 0.4 / **0.6** s | | | 0.5 / 0.7 s |
| 1 s response skill, shipped / with lag | +0.18 / +0.19 | +0.25 / +0.25 | +0.45 / +0.47 | +0.21 / +0.25 | +0.24 / +0.27 | +0.62 / +0.55 |

- **What improved:** the step timing on surge, sway and heave now matches the vehicle. Roll skill turns positive for the first time.
- **What got worse:** yaw, because every other parameter was fitted assuming no lag.
- **The full refit with the lag free** (2026-10-06): warm-started from the shipped calibration plus 0.18 s, same settings and days; it converged in 12 evaluations at **0.34 s**.

| | Surge | Sway | Heave | Roll | Pitch | Yaw |
|---|---|---|---|---|---|---|
| Held-out skill, full refit with lag | +0.73 | +0.51 | +0.62 | +0.06 | −0.03 | +0.56 |
| Change from shipped | +0.01 | −0.00 | +0.03 | +0.09 | +0.07 | **−0.025** |
| Step, time to 63% (real / refit) | 0.6 / 0.7 s | 0.5 / 0.6 s | 0.6 / 0.7 s | | 0.4 / 0.4 s | 0.5 / 0.5 s |

  - **What improves:**
    - **Position drift:** 0.77 / 1.24 / 1.74 m after 10 / 30 / 60 s, against the shipped 0.84 / 1.33 / 1.88 m.
    - **Orientation, the model on its own:** roll error 7.5° against 8.7° and pitch 9.9° against 10.7° after 10 s.
    - **Step timing:** close to the vehicle on every axis.
    - **No divergence** on any held-out recording.
  - **Yaw recovers from the lag-only +0.51, but stays 0.025 below the shipped +0.58.** The rule agreed in advance was to make the lag the default only if it helped yaw (no worse than 0.02 below shipped, nothing else worse by more than 0.02). It doesn't, so the lag **stays opt-in**.
  - **The fit isn't physical where it trades lag for inertia.** Yaw inertia went to its lower bound (0.05 against 1.89 shipped), and surge and sway added mass went to zero; a BlueROV2 carries several kilograms of added mass. Lag and inertia both slow the response, and these logs don't separate them. Separating them needs inertia or added mass measured independently (CAD, a pendulum test), or thrusters whose response is measured on a bench.

### 67. Normalized thruster commands (2026-10-06)

**Why.** A lab's ROS bags (MIT, CORL-AUV outreach) log thruster commands normalized to a range, with a lab-supplied command-to-thrust mapping. Ingestion only accepted ESC PWM in microseconds and refused anything outside 900–2100.

**What shipped:**
- **Spec settings:** `command: normalized` and `command_range: [low, high]` in an ingest spec's `thrusters:` block. The default range is [-1, 1]; stopped is the middle of the range, so [0, 1] stops at 0.5.
- **With the lab's mapping:** a `curve_csv` in the logged units (`command,thrust_n`).
- **Without one:** the built-in T200 curve, with the range mapped onto 1100–1900 µs.
- **Reversed thrusters** are mirrored about the stopped value; for PWM that is the same 3000 − pwm as before.
- **Battery-voltage scaling** works as before.
- **Plain messages for mistakes:**
  - normalized data read as PWM says to add `command: normalized`;
  - commands outside the declared range are refused;
  - a curve in the wrong units says so.
- **New templates:** `templates/ingest_rosbag_normalized_thrusters.yaml` (ROS bags with an EKF's `nav_msgs/Odometry`, normalized commands, a custom layout) and `templates/thrust_curve_normalized_template.csv`.
- **Checked** (`tests/test_normalized_commands.py`): commands normalized from PWM give exactly the PWM forces, for [-1, 1] and [0, 1], with and without per-sample voltage, through the whole ingestion.

### 68. Link calibration from message logs (2026-10-06)

**Why.** The acoustic link was a set of assumed loss levels (L0–L3b) with no modem delay, and the optical link was an uncalibrated placeholder. Labs running multi-vehicle acoustic messaging (URI's mvp_acomm) have the logs to measure their own link.

**What shipped:**
- **`strathe fit-link --log <messages.csv> --out <folder>`** (`link_fit.py`; also a card on the web page). The input is one row per message sent: send time, receive time (empty if lost), range when sent, link (`templates/link_log_template.csv` says how to build it from modem logs).
- **The acoustic fit:**
  - **Delivery:** the distance ramp of drop probability, by maximum likelihood. It is kept only if it beats one constant loss by AIC.
  - **Latency:** a fixed modem delay plus range ÷ sound speed, by robust least squares. The sound speed is held at 1500 m/s when the ranges span under 100 m or the fit is implausible.
  - **Clocks:** messages arriving before they were sent flag unsynchronised clocks.
- **The optical fit:**
  - **Delivery:** the rated range and curve width of the delivery S-curve.
  - **The water's clarity is held, not fitted:** coastal by default, a named water, or a transmissometer's measured attenuation.
  - **Why it's held:** delivery against range is one S-curve, pinned by its midpoint and steepness, and the water's attenuation trades off against the modem's rated range and curve width. In six synthetic scenarios over 0.2–80 m in clear, coastal and harbour water, every water type fit within one log-likelihood unit.
  - **What that means:** inside the logged ranges the fitted curve is the same whatever water is held; the water only shapes delivery beyond them. The report and the warning say so.
- **The modem delay in the simulation:** a new `fixed_latency_s` on the acoustic channel (0 by default: identical runs).
- **Using a calibration:** `--link-calibration` on `strathe fleet` and `train-fleet`, `communication.link_calibration` in the experiment YAML, and a web field.
  - **Acoustic:** the measured ramp, sound speed and delay replace the assumed levels, as one level labelled `measured`.
  - **Optical:** the optical link uses the measured curve and latency, and says "Calibrated from your link logs".
  - **The fleet report** says the acoustic losses and latency were measured.

**Checked** (`tests/test_link_fit.py`; logs drawn from Strathe's own channel models with known settings):
- **Acoustic recovery:**
  - loss ramp 59–384 m against the true 50–400 m;
  - drop probabilities 0.04 / 0.57 against 0.05 / 0.60;
  - sound speed 1491 against 1480 m/s;
  - modem delay 2.5 s, exact.
- **Optical recovery:** the 50% range within 5%. Held at the wrong water, the fitted delivery still matches every logged range bin within 8 points.
- **Fallbacks:** a constant loss is chosen when loss doesn't depend on range; narrow ranges hold the sound speed.
- **Wiring:** the measured link reaches the experiment schema (and survives a save and reload), the fleet simulation's channel, training's conditions, the CLI and the web page. A fleet smoke run with a calibration produces the `measured` level and the calibrated optical description.

**Limits:**
- **Clocks:** both vehicles' clocks must be synchronised for latency.
- **Not fitted:** optical pointing (angle). The acoustic model has no multipath, collisions or message priorities.
- **Extrapolation:** beyond the logged ranges, the fits extrapolate.

### 69. Battery voltage with a lab's own thrust table; the direction check on normalized commands (2026-10-07)

**Why.** Both labs lined up for pilots (MIT, URI) use T200 thrusters, and MIT will send its own command-to-thrust mapping. `voltage_column:` (#63) only rescaled Strathe's built-in T200 curve and refused a lab's `curve_csv`, so with their own table the battery sag (17% of thrust from 15.3 to 13.1 V) would have been ignored.

**What shipped:**
- **`curve_voltage:`** in a thrusters block: the battery voltage the lab's `curve_csv` was measured at.
- **What it enables:** `voltage_column:` (per sample) and `voltage:` (one value, for the simulators) rescale that table by Blue Robotics' published T200 ratio, forward and reverse separately. At the table's own voltage it is the table itself.
- **The built-in curve's path is unchanged**, bit for bit.
- **Plain messages:**
  - `voltage_column:` or `voltage:` with a table and no `curve_voltage:` asks for it, saying it's for T200s;
  - `curve_voltage:` without a table, or outside the published 12–20 V, is explained.
- **Templates:** the normalized and custom templates show `curve_voltage:` with `voltage_column:`.

**Checked** (`tests/test_voltage_custom_curve.py`):
- **The ratio:** a table measured at 16 V gives exactly the published ratios at 12 V.
- **Consistency:** the built-in curve given as a lab table, at its matching voltage, scales within 2% of the built-in path.
- **The MIT case end to end** (normalized commands, the lab's table, per-sample voltage): ingestion gives exactly the table's forces at each sample's voltage.
- **The force-direction check (#63) on normalized commands:** a synthetic 6-thruster vehicle's logs, read with its layout mirrored left-right, are refused on sway and yaw; with the right layout, they pass.

**Not addressed:** separating linear from quadratic drag. `--drag prior` (#63) stays opt-in and didn't help on the reference dataset. What separates them is data: runs held at two or three different steady speeds on each axis; ask labs for them.

### 70. URI's AUV run as a lab would run it: atomic checkpoints, a plausibility check from the vehicle's own description, and "describe your vehicle" (2026-10-10)

**Why.** URI sent two ROS 2 bags from its ALPHA RISE AUV (1.5 m torpedo hull, 33 kg, 4 T200 thrusters, untethered):
a Narragansett Bay mission (2025-12-05) and a freshwater pond mission (2026-03-27). We ran them end to end as a lab
would: a fresh clone, only LAB_GUIDE.md, the templates and `strathe` commands, no code changes. The run was 28
calibrations, 14 single-vehicle experiments, 6 trained fleet policies and 11 fleet experiments. It found two bugs and
one gap that made the documented path fail for any vehicle but a BlueROV2.

**What the run showed:**
- **The documented path, literally:** LAB_GUIDE never asked for a vehicle description, so `calibrate` used the
  BlueROV2 Heavy's (11.2 kg). Result: 0 of 6 axes trustworthy, surge worse than default physics, 17 of 22 values at
  their limits, a NaN crash in the first fit, and 70 m of drift after 60 s. The report blamed a tether the AUV doesn't
  have.
- **With an ALPHA description (mass, hull-sized inertia, drag and added-mass guesses and limits):**
  - held-out surge skill +0.83 to +0.89 in both directions (bay to pond, pond to bay);
  - sway and yaw partly predicted;
  - drift 3.5–4.0 m after 60 s, against 26 m for default physics.
- **Roll and pitch stay near zero skill.** Widening the limits the fit hit only moves it to the new limits (pitch
  inertia 200 kg·m², quadratic drag −2000): a torpedo hull's lift and pitching moment at speed aren't modelled.
- **Downstream, against training on default physics** (same episodes, the calibrated vehicle):
  - docking +25 points clean, +36 noisy;
  - station keeping +7.5 clean, +14.5 noisy;
  - two-vehicle formation with ALPHA's own policies: +7 to +9 points at 3 s beacons.
- **Obstacle avoidance trains worse on the calibrated model** (−17.5 points, 4 seeds at 2× budget).
- **The shipped BlueROV2 fleet policies on ALPHA make calibration look harmful** (−20 points).

**What shipped:**
- **Checkpoints are written atomically** (`checkpoint_io.py`).
  - **The bug:** every calibration rewrote `checkpoint.yaml` with `open(path, "w")` after each evaluation, so a run
    stopped at the wrong moment left it empty. One of URI's runs lost its best result that way, and
    `finalize_checkpoint.py` crashed on the empty file. A flat battery or a crash does the same.
  - **The fix:** a temporary file in the same folder, flushed to disk, then `os.replace`, retried briefly on Windows.
    Used by `run_calibration.py`, the MuJoCo and HoloOcean calibrations and `replay_comparison.py`.
  - **Damaged checkpoints:** an empty or unreadable one now gets a plain message in `--finish`,
    `finalize_checkpoint.py` and `--resume-from-checkpoint`.
- **The plausibility check uses the vehicle's own description.** The report compared every vehicle's fitted inertia
  with hardcoded BlueROV2 Gazebo values, calling ALPHA's physically right pitch inertia (29 kg·m²) "127× too big";
  the per-axis verdicts lean on that check. Now any calibration target can carry `reference: {value, source}`, an
  independent measurement of that vehicle (CAD, a pendulum test), and only those are checked. The shipped BlueROV2
  specs carry the Gazebo values, so the reference dataset's reports are unchanged.
- **"Describe your vehicle":**
  - **The guide:** LAB_GUIDE step 4, with a table of first guesses and ranges from mass and size, and `--spec` in
    every command.
  - **The template:** `templates/vehicle_spec_template.yaml`, every value's formula in its comments, worked for a
    torpedo AUV.
  - **Warnings:** `calibrate` warns before fitting when no `--spec` is given and the logs' thrusters aren't a BlueROV2
    Heavy's. The report then says so at the top of "Be careful with", and its first next step is to describe the
    vehicle.
  - **Other docs:** README, LIMITS, the templates README and the lab journey are updated.
  - **Schema:** `measured` is now an allowed confidence label (a weighed mass).

**Checked** (`tests/test_checkpoint_io.py`, `tests/test_vehicle_description.py`):
- **Atomic writes:** a write interrupted mid-dump leaves the previous checkpoint intact and no temporary file. An empty
  or damaged checkpoint gives the plain message in `finalize_checkpoint.py` and in a resumed calibration.
- **The plausibility check:**
  - the BlueROV2 spec still flags a 10 kg·m² roll inertia;
  - a spec without references flags nothing;
  - a spec with its own reference flags against that.
- **The template:** it passes both spec validators.
- **The warning:** it fires for a custom layout with the default spec, and not for the BlueROV2 preset, a given
  `--spec` or logged forces. The report section and next step appear only then.

**Still open, from the same run** (each in the run's friction log): ~~per-thruster thrust curves and limits~~ (fixed, #91); ~~a sensor
mount rotation in ingest specs~~ (fixed, #91); ~~`--finish` for `--estimate-current` runs~~ (fixed, #79); ~~`train-fleet` and arena task geometry
(formation spacing, target distance) for vehicles bigger than a BlueROV2~~ (fixed, #91); ~~`fleet --workers`~~ (fixed, #91); ~~mixed header and
receive timestamps in re-recorded bags; and session dates from header stamps~~ (fixed, #91).

### 71. Torpedo-hull physics, and Strathe against Stonefish on URI's own vehicle model (2026-10-10)

**Why.** The URI run (#70) showed roll/pitch failing on a torpedo hull, and the strategic question: can Strathe beat,
or improve, the simulator a lab already uses? URI simulates ALPHA RISE in Stonefish
(GSO-soslab/world_of_stonefish, vehicles/alpha_rise.scn).

**What shipped (opt-in, like the thruster lag #66):**
- **New terms for slender hulls, one set for both planes (an axisymmetric hull):**
  - hull lift, Zuw (Z += Zuw·u·w, Y += Zuw·u·v);
  - its moment, Muw (M += Muw·u·w, N −= Muw·u·v);
  - speed-dependent pitch/yaw damping, Muq (M += Muq·u·q, N += Muq·u·r);
  - a fore-aft centre-of-gravity offset, `cg_cb_offset_x` (Fossen 2011, eq. 4.6).

  The destabilizing Munk moment was already in the model, through the added-mass Coriolis term.
- **Every physics path:** the reference model, the fast core (calibration, fleet), the rotation-matrix form (arena,
  MuJoCo) and HoloOcean. They agree to 1e-14; off, results are bit-identical to before.
- **Turning them on:** a `hull_lift:` section and `cg_cb_offset_x` in a vehicle spec. They're in the template, with
  slender-body first guesses. Without them they are held at 0.
- **Tests:** `tests/test_torpedo_hull.py` (equivalence, bit-identity when off, physical signs, spec handling, old
  calibrations still load).

**What the URI data said about them: not identifiable from two ordinary missions.**

| Direction | Training error, round 1 | Held-out skill | Notes |
|---|---|---|---|
| bay → pond | 2.16, against 2.41 without the terms | +0.20, against +0.23 | 60 s drift 5.5 m, against 3.5 |
| pond → bay | | −0.10, against +0.18 | yaw breaks |

- **The fitted values contradict each other between directions:** Muw is +40 in one and −60 in the other, Muq −14
  vs −71.
- **Pitch isn't fixed.**
- **What they need:** logs with pitch and heave steps at 2–3 steady speeds. The template now says to calibrate with
  and without them and keep them only if held-out skill improves.

**Strathe against Stonefish.**
- **The set-up:** URI's Stonefish fork, built and run headless, with URI's model unchanged. Same logged wrench,
  logged start state, free-running attitude, still water; 60 s windows every 10 s on the held-out recording.
- **URI's model as it stands:** 27 kg (the real vehicle is 33), a rigid pitch inertia of 64 kg·m², and drag from the
  mesh.

Median drift after 10 / 30 / 60 s:

| | Pond | Bay |
|---|---|---|
| Stonefish, as is | 2.40 / 15.9 / 47.6 m | 1.92 / 11.4 / 37.4 m |
| Strathe's calibrated model | **1.74 / 10.3 / 26.4 m** | **1.47 / 9.1** / 30.9 m |
| Stonefish with Strathe's values translated in (mass, net buoyancy, drag through the hull mesh's areas) | 2.85 / 13.3 / 40.0 m | 1.85 / 9.1 / 28.2 m |
| **Stonefish calibrated directly** (Stonefish's own drag, buoyancy and CG fitted to the training log, 30 s windows, drift in the objective) | 2.99 / 12.3 / 34.9 m | 2.34 / 9.7 / **24.4** m |

- **Strathe's own model wins** in both directions.
- **Strathe also improves URI's own simulator:** up to 35% less drift after 60 s when it calibrates Stonefish directly.
  But 10 s drift gets worse, because Stonefish's added mass can't be set from a scenario.
- **Neither simulator beats "keep the starting velocity"** over a closed-loop cruise (0.4–0.9 m after 10 s). Open-loop
  replay compounds error while the logged forces react to the real vehicle. The scores rank simulators; they don't
  claim minute-ahead prediction.
- **The tools** (the Stonefish replay driver, the comparison and the Stonefish calibration) are outside the product for
  now: `~/strathe_uri_lab/stonefish/replay/`.

### 72. `strathe fit-stonefish`: calibrate a lab's own Stonefish model from its logs, with an added-mass patch (2026-10-10)

**Why.** #71 showed Strathe can improve URI's own simulator: calibrating Stonefish's settings from the logs cut its
held-out drift. That's the "calibration layer under your simulator" product, so it's now a command any Stonefish lab
can run, wired in like the rest. A Stonefish limitation surfaced on the way:
- **Averaged added mass:** Stonefish gives a body one added mass, the mean of its three axes. For ALPHA's hull that's
  about 153 kg in every direction; physically about 10 kg moving forward and over 100 kg sideways.
- **Not settable:** it can't be set from a scenario at all.

**What shipped:**
- **`strathe fit-stonefish`** (`stonefish_fit.py`, CLI, a web card). It takes:
  - the lab's own world scenario, as they run it: ROS-style `$(find PKG)` via `--find`, and `<include>` with `<arg>`;
    sensors are stripped for the headless replay; their files are never changed;
  - the robot, the part to tune, and the logs (`--data` or `--ingest-spec`);
  - the logs' reference point in the robot frame, and the held-out dates.

  It fits Stonefish's own settings on that part:
  - quadratic and viscous drag per axis (or `--axisymmetric`);
  - added mass per axis;
  - net buoyancy, by resizing a named buoyant primitive part;
  - the CG's position.

  The fit is least squares; each Jacobian's Stonefish runs go in parallel (`--workers`). The objective is 30 s
  free-running windows, with velocity error scaled per axis plus end-of-window drift. It then scores the original and
  the calibrated model, and optionally a Strathe calibration (`--compare-params`), on the held-out recordings: 60 s
  windows every 10 s, with the baselines.
- **Outputs:**
  - the lab's vehicle file with only that part (and the buoyancy part) changed;
  - `stonefish_fit.yaml` and a report with a drift plot.
- **`calibration/stonefish/`:**
  - the replay tool: C++, built with CMake against any Stonefish install, with an `--info` mode;
  - `stonefish_added_mass.patch`, about 100 lines. It adds `<hydrodynamics added_mass= added_inertia=
    anisotropic_added_mass=>`, applies the per-axis difference from the mean (and the Munk moment) as forces each step,
    and adds `Compound::getNumOfParts()`. It's off unless asked for, so other scenarios behave exactly as before.
  - **Without the patch,** the command fits drag, buoyancy and the CG only, and says so.
  - **A README** covering the build (including macOS's OpenMP flags) and use.
- **Docs:** LAB_GUIDE (optional extras, a step 4 command, outputs), README, LIMITS (a new section on your own simulator).
- **Tests** (`tests/test_stonefish_fit.py`, plus CLI/web tests): a Python stand-in for the replay tool
  (`tests/fake_stonefish_replay.py`) runs the whole path on CI without Stonefish:
  - scenario preparation ($(find), includes, sensors stripped, density);
  - the calibrated copy keeps comments and sensors;
  - a fit from wrong values cuts held-out drift by over 40% and moves drag toward the truth;
  - an unpatched Stonefish narrows the fit;
  - a missing tool says how to build it;
  - the CLI runs ingestion then the fit;
  - the web form starts exactly the CLI command.

**Re-run of the comparison through the command** (URI's ALPHA RISE: patched Stonefish, all four knobs, axisymmetric,
15 iterations; held-out drift after 10 / 30 / 60 s):

| | Pond (trained on bay) | Bay (trained on pond) |
|---|---|---|
| Stonefish, URI's model | 2.40 / 15.9 / 47.6 m | 1.92 / 11.4 / 37.4 m |
| **Stonefish calibrated by `fit-stonefish`** | 2.49 / **12.5 / 36.7 m** | 2.75 / **10.9 / 23.3 m** |
| Strathe's own model (B0) | 1.74 / 10.3 / 26.4 m | 1.47 / 9.1 / 30.9 m |

- **Training cost** fell 90% and 82%.
- **Attitude after 10 s improved most:**
  - pitch 10.9° → 4.5° (pond) and 8.9° → 2.9° (bay), better than Strathe's own model (10.9°, 9.9°);
  - yaw 32.7° → 16.5° and 19.7° → 8.1°.
- **Known issue: surge.** Its 10 s velocity error worsened (0.17 → 0.28 and 0.15 → 0.32 m/s), and the fitted
  surge added mass went to a nonphysical 172 and 274 kg (Strathe fits about 10). The per-axis scaling (each axis's
  RMS velocity) makes surge errors cheap during a cruise, so the fit trades them for pitch and yaw.

  **Next:** scale by each axis's variation instead, and let labs hold a measured value (e.g. surge added mass).

### 73. `fit-stonefish`: weight each axis by how much it varies, and hold measured values (2026-10-10)

**Why.** #72's fit made surge worse at short horizons and gave ALPHA RISE a nonphysical surge added mass (172–274 kg
against about 10). Each axis's error was divided by its RMS velocity. During a 0.7 m/s cruise surge's RMS is mostly
its mean, so surge errors were cheap and the fit traded them for pitch and yaw.

**What changed:**
- **Each axis's error is divided by its standard deviation** over the training recordings (floored at 0.05 m/s and
  0.02 rad/s). On ALPHA's bay training mission surge's scale went from about 0.7 to 0.09. `--weighting magnitude`
  reproduces #72.
- **`--hold NAME=VALUE`** keeps a setting at a measured value instead of fitting it, e.g. `--hold added_mass_x=3.3`.
  The names:
  - `quadratic_drag_x/y/z`, `viscous_drag_x/y/z` and `added_mass_x/y/z` (`_yz` when axisymmetric);
  - `buoyancy_scale`;
  - `cg_shift_x`, `cg_shift_z`.
- **Plumbing:** in the CLI, the web card (a "hold" field) and the report and YAML (the scales and held values are
  recorded).
- **Tests:** the scales, held settings staying put and leaving the fit, a held value reaching the report, and the CLI
  and web pass-through.

**URI's ALPHA RISE again** (patched Stonefish, axisymmetric, 15 iterations). Held-out drift after 10 / 30 / 60 s, and
surge velocity error over 10 s:

| | Pond | Bay |
|---|---|---|
| Stonefish, URI's model | 2.40 / 15.9 / 47.6 m; surge 0.17 | 1.92 / 11.4 / 37.4 m; surge 0.15 |
| #72 (RMS weighting) | 2.49 / 12.5 / 36.7 m; surge 0.28 | 2.75 / 10.9 / 23.3 m; surge 0.32 |
| **variation weighting** | **2.10** / 11.8 / 38.8 m; surge 0.23 | **2.01** / 8.1 / 24.4 m; surge 0.21 |
| **variation + `--hold added_mass_x=3.3`** (0.1 × mass, no data from either mission) | 2.94 / **9.8 / 22.2 m**; surge 0.36 | 2.17 / **7.8 / 12.6 m**; surge 0.24 |
| Strathe's own model | 1.74 / 10.3 / 26.4 m | 1.47 / 9.1 / 30.9 m |

- **Variation weighting fixes the short horizon:** 10 s drift is better than URI's model in the pond and within 5% in
  the bay.
- **The fitted surge added mass** is still large without the hold: 148 kg (pond) and 81 kg (bay).
- **Holding surge added mass at a physical value** cuts 60 s drift by 53% and 66% against URI's model, and beats
  Strathe's own model in both directions. It generalizes better, too: training cost was higher (158 vs 67) and
  held-out error lower.
- **Still open: surge velocity over 10 s** stays worse than URI's model in every variant. A likely cause (not
  confirmed) is that Stonefish's mesh-face drag couples surge to pitch.

**Recommended for labs:** the default weighting, plus `--hold` for added mass you've measured, surge especially.

### 74. `strathe export-stonefish`: Strathe's values in a lab's Stonefish, and why converting isn't enough (2026-10-11)

**Why.** The strategic test: put Strathe's calibrated values into URI's own Stonefish model and compare it with the
uncalibrated original on URI's logs. If that doesn't improve their simulator, the thesis is in trouble. #71 did it
with lab scripts (no added mass, no inertia); it wasn't in the product.

**What shipped:**
- **`strathe export-stonefish --calibration <folder> --scenario ... --robot ... --part Hull`** (`stonefish_export.py`,
  CLI, a web card). It converts a calibration into the lab's vehicle file, on one part:
  - mass;
  - net buoyancy, by resizing a buoyant part, in the training water;
  - CG − CB;
  - roll/pitch/yaw inertia, rigid plus added: added inertia first, then the part's rigid `<inertia>`, floored at 5%;
  - per-axis added mass (patched Stonefish);
  - per-axis drag.

  Every value is checked by reloading the vehicle in Stonefish, and the original and converted models are scored on
  the calibration's held-out recordings. What Stonefish can't carry is listed: rotational drag, hull lift, thruster
  lag, off-diagonal added mass.
- **`fit-stonefish --from-strathe <folder>`:** fits through Stonefish starting from the converted values; the report
  then scores three Stonefish models.
- **The replay tool's `--info`** now reports, per part:
  - the mesh areas Stonefish's drag uses (projected and tangential, in the part's axes);
  - the rigid inertia;

  and, for the vehicle:
  - the CG and CB;
  - the inertia along the body axes;
  - the water density.
- **Calibrations record `vehicle_mass_kg`** in `calibrated_parameters.yaml`.
- **Fixes:**
  - XML comments are skipped when editing a part, so a commented-out `<inertia>` isn't taken as the real one;
  - an axisymmetric fit starts from the geometric mean of the side and vertical values.
- **Tests:** `tests/test_stonefish_export.py` (11), with the fake replay reporting the new fields.

**Measured: Stonefish's mesh drag is cubic in speed and couples the axes.**
- **From the code:** each face facing the flow adds `v·|v|·(−v_n)·A`.
- **In the replay:** differential probes (two Cd values, same start) give 0.5·ρ·Cd·S·U³ within 1% at 0.25–1 m/s, and
  a linear skin term ρ·Cf·T·U.
- **The coupling:** the force is along the whole flow, so a pitched hull's large side faces add surge drag. One
  held-out window: the converted model pitched about 5° within a second and its surge speed fell from 0.61 to 0.46
  m/s under the logged thrust; Strathe's model held 0.67 and the vehicle 0.64.
- **Also seen:** Stonefish's hydrodynamic forces lag the state by about 0.01 s (2 ticks at 200 Hz). Under 0.01 m/s at
  a window start; not corrected.

**URI's ALPHA RISE** (B0 calibrations, patched Stonefish). Held-out drift after 10 / 30 / 60 s:

| | Pond | Bay |
|---|---|---|
| Stonefish, URI's model | 2.40 / 15.9 / 47.6 m | 1.92 / 11.4 / 37.4 m |
| **Stonefish with Strathe's values (export)** | 4.16 / 14.8 / **34.3** m | 4.48 / 13.9 / **30.9** m |
| fit-stonefish `--from-strathe` (+ hold added_mass_x=3.3) | 4.35 / 14.2 / 32.9 m | 4.14 / 12.8 / 29.0 m |
| fit-stonefish from URI's model (+ hold, #73) | 2.94 / **9.8 / 22.2** m | 2.17 / **7.8 / 12.6** m |
| Strathe's own model | **1.74** / 10.3 / 26.4 m | **1.47** / 9.1 / 30.9 m |

- **Exporting helps over a minute (−28%, −17%) and hurts over seconds (+73%, +133%).** Surge velocity error over
  10 s is 0.42–0.44 m/s against 0.15–0.17 for URI's model: the cross-flow coupling above.
- **Every converted value is exact** (mass, buoyancy, CG − CB, inertia, added mass, and drag at the logged speeds),
  so the gap is Stonefish's drag model, not the conversion.
- **Starting the fit from the converted values ended worse than starting from URI's model.**
  - Bay: the fit barely moved (cost 444.6 → 443.4). The axisymmetric start took the side drag (0.049) for the
    vertical (10.9), now fixed.
  - Pond: it moved the CG straight back to URI's model's (+0.104 m).
  - Not recommended until re-run with the fix.
- **The thesis holds through `fit-stonefish`:** calibrating Stonefish *through* Stonefish cuts URI's 60 s error by
  53–66%. Values from Strathe's model can't simply be pasted in.

**Pitch with the torpedo-hull terms held at physics values.** #71 fitted Zuw, Muw and Muq freely, and the two
directions contradicted each other. Here they're held at slender-body estimates for a 1.5 m × 0.25 m bare hull:
- Zuw = −0.5·ρ·D²·1.2 = −37.5;
- Muw = −8.4 (centre of pressure 0.225 m aft of base_link);
- Muq = Zuw·x_cp² = −1.9.

cg_cb_offset_x is still fitted. Drift after 10 / 60 s, pitch error after 3 / 10 s, and pitch-rate skill:

| | Pond held out | Bay held out |
|---|---|---|
| B0, no hull terms | 0.64 / 3.48 m; 3.2° / 10.2°; −0.03 | 0.72 / 3.98 m; 3.2° / 8.3°; −0.26 |
| T0, hull terms fitted (#71) | 0.86 / 5.50 m; 2.9° / 11.9°; −0.04 | 0.66 / 3.15 m; 2.2° / 7.3°; −0.24 (not converged) |
| T1, hull terms held | 0.66 / 3.56 m; 3.0° / 12.0°; −0.04 | 0.69 / 3.78 m; 3.0° / 11.5°; −0.28 |

- **Pitch still doesn't calibrate, in either direction.** Mq (−100) and Mqq (−150) sit at their bounds in both
  T1 runs, and pitch inertia does in one (60 in the pond run; 42 in the bay run).
- **What it rules out:** the missing lift terms aren't what's wrong. Two missions that hold pitch steady don't
  excite it. The lever stays pitch and heave steps at 2–3 speeds, or free decays (#64).
- **Holding the terms is harmless where fitting them hurt:** drift is level with B0 (5% better at 60 s in the bay),
  and the mean held-out skill is unchanged (+0.18), where T0's fell to −0.10.

### 75. Transplanted values, axis by axis: why Strathe calibrates the customer's simulator instead of translating (2026-10-11)

**Why.** #74 showed that converting Strathe's values into URI's Stonefish helps over a minute and hurts over seconds.
Labs reading an `export-stonefish` report saw the numbers but no verdict per axis, and no direction. The working
explanation ("the values don't balance against Stonefish's added mass and buoyancy") also needed checking: #74
transplants both exactly.

**Axis by axis, after 10 s** (velocity error; attitude error), ALPHA RISE held out:

| | Pond: original → transplanted | Bay: original → transplanted |
|---|---|---|
| surge | 0.17 → 0.44 m/s (worse) | 0.15 → 0.42 m/s (worse) |
| sway | 0.086 → 0.049 m/s (better) | 0.070 → 0.073 m/s (same) |
| heave | 0.25 → 0.07 m/s (better) | 0.24 → 0.09 m/s (better) |
| roll | 4.6° → 10.6° (worse) | 2.4° → 5.1° (worse) |
| pitch | 10.9° → 24.2° (worse) | 8.9° → 11.3° (worse) |
| heading | 32.7° → 14.5° (better) | 19.7° → 14.5° (better) |

**What causes it: Stonefish's drag and righting, not its added mass or buoyancy.**
- **The test:** transplant everything except the centre of gravity, keeping Stonefish's CG − CB of 97 mm instead of
  Strathe's 3–5 mm.
- **Roll returns to the original's:** 4.6° (pond) and 2.5° (bay).
- **Pitch improves in the pond** (24.2° → 15.9°) but stays worse than the original in both.
- **Surge is unchanged** (0.42, 0.39 m/s).
- **60 s drift improves slightly:** 33.3 and 29.6 m.

Three mechanisms:
- **Surge:** Stonefish's mesh drag, cubic in speed and coupled across axes (#74).
- **Roll:** Strathe's near-zero righting arm. It balances Strathe's own roll damping, but not Stonefish's, which comes
  from the mesh.
- **Pitch:** both the righting arm and Stonefish's mesh-derived rotational drag.

**What shipped:** the export report and its printed summary now
- list, per axis, what got better, worse or stayed the same after 10 s (velocity for surge/sway/heave, attitude for
  roll/pitch/heading; within 5% is "same");
- always state the recommendation: Strathe's values aren't a calibrated Stonefish model, and `fit-stonefish`
  calibrates it through Stonefish (53–66% less 60 s error on ALPHA, against 17–28% for the conversion).

Both are in `stonefish_export.yaml` (`axis_verdict_10s`, `recommendation`). Documented in the Stonefish README and
LIMITS. Tests: the verdict on #74's bay numbers, and the report and YAML carrying both.

**Conclusion, now in the product:** Strathe calibrates the customer's simulator directly. It doesn't translate
parameters into it.

### 76. URI full re-run of every variable, and the bugs it found fixed (2026-10-18)

**Why.** After #70–#75, URI's run was repeated on `main` (`748378b`) from a fresh clone, varying every option of
every command, Strathe-only and through Stonefish. The writeup and every table are in
`~/strathe_uri_lab_v2/URI_FULL_RERUN_REPORT.md` (lab machine).
- **Size:** 275 runs, 476 active compute hours, Oct 12–18, on one 8-core laptop.
- **Spread:** B0 repeated 7 times; B8 and B9 4 times each.

**Results:**
- **The original run reproduces.**
  - Docs-literal (A0): −0.66 mean, 70.6 m.
  - B0: surge +0.89 / +0.83, mean +0.23 / +0.18, drift 3.5 / 4.0 m.
  - Ingestion is byte-identical.
- **The fit isn't deterministic.** `run_calibration.py` starts from a randomly perturbed point with no recorded seed.
  - B0 lands at the same place every time (±0.001).
  - B8 (surfacings removed) lands at +0.06 or +0.24 mean skill (10.6 or 4.7 m), and B9 (20 Hz) at +0.18 or +0.23.
  - Both differences from the original run are this, not a code change: the data are byte-identical, and the
    objective at the same parameters is identical to 15 digits under the old and new code.
- **Pitch doesn't calibrate.**
  - Fitted or held hull terms (T0–T2): pitch skill ≤ 0.
  - Held terms with widened limits (T3): 2–4° of pitch error, by freezing pitch (values at the new limits).
- **Stonefish:**
  - `fit-stonefish` with surge added mass held: −55% / −63% at 60 s against URI's model.
  - Also holding sideways added mass at strip theory (F4, pond): better than URI's model at 10, 30 and 60 s.
  - Export: −17% / −28%.
  - `fit-stonefish` is deterministic (1 or 4 workers: identical).
- **Arena:**
  - Clean docking: +12 [+9, +15] with 8 seeds (the original's 2-seed +25 was seed luck).
  - Noisy docking: +32.
  - Calibration-trained seeds range 0.59–1.00 on their own, so read arena results with fewer than 4 seeds as
    provisional.
- **Web:** results identical to the command line; all 9 forms refuse bad input with a plain message.

**Fixed here:**
- **F23, unpatched Stonefish:** the replay tool listed a compound's parts only through the patch, so neither
  command ran. It now walks `Compound::getPart(i)` until it returns no solid, which works in unpatched Stonefish. On
  URI's data with unpatched Stonefish:
  - `fit-stonefish` fits drag, buoyancy and CG: 1.70 / 12.6 / 42.7 m against 2.40 / 15.9 / 47.6 m;
  - `export-stonefish`: 3.88 / 16.4 / 40.7 m.
- **F24, current estimation:** a recording on which the parameters blow up used to stop the whole calibration. It is
  now held at still water and flagged ("no current estimated"), and the others carry on. On X2's real parameters,
  2 of 5 recordings are flagged and the rest estimated.
- **F25, the seed:** `calibrate` draws a seed when none is given, prints it, passes it to every current round, and
  records it in `calibrated_parameters.yaml`, `metrics.json` and the report. `--seed N` repeats a calibration exactly.
- **F9, blow-ups:** a fit that stops because the simulation blew up (NaN/inf) exits with its own code (3). The
  command line names it and points at the limits, instead of blaming the evaluation budget.
- **Fleet config errors:** one plain line per problem ("communication.optical: optical beacon_interval_s must be a
  positive multiple of the 0.1 s tick"), not pydantic's dump.
- **Tests:** `tests/test_rerun_fixes.py` covers the seed (given, drawn, the same for every round, not passed to
  finalize), blow-up vs budget messages, an unstable recording, the report's seed line and the fleet message.

**Still open:**
- ~~F21: `--finish` refuses `--estimate-current` runs.~~ Fixed 2026-10-22 (#79 branch). It finishes the step that was
  running: the final fit from its checkpoint with the last round's currents, or the latest round from its checkpoint
  with that round's input currents. The reason recorded and reported says which, and that the later steps didn't
  run. With `--restart-whole-fit` it keeps finished starts and finishes the interrupted one.
- ~~F1, F2: header stamps; recording-date sessions.~~ Fixed (#91).
- ~~F5: noise from filtered logs.~~ Fixed (#91).
- ~~F15: arena intervals ignore seed variance.~~ Fixed (#91).
- ~~F18: `fleet` has no `--workers`.~~ Fixed (#91).
- F20: shipped policies on another vehicle. The fleet report warns when the policies were trained on another vehicle
  and says to train on yours (`trained_elsewhere_note`); the effect itself is real.
- ~~F11: missing web form fields.~~ Fixed in #78 (calibrate: workers, current rounds, drag, torque offset; Stonefish
  forms: replay tool, iterations, workers).

### 77. `strathe describe-stonefish`: the vehicle description from a lab's own Stonefish model (2026-10-19)

**Why.** Filling in `vehicle_spec_template.yaml` by hand is the hardest step for a lab, and a lab with a Stonefish
model already has most of it. `describe-stonefish` reads the model through the replay tool's `--info` and writes the
description: mass, inertia, added mass, drag, pitch and yaw damping, net buoyancy, the CG's offset and, with
`--hull-lift`, the hull-lift terms. Web card, LAB_GUIDE step, README line and a Stonefish README section.

**The limits are what matter.** Validated on URI's ALPHA RISE against the hand-written description:
- **Limits around the model's values** (5× either way): added mass up to 900 kg. The fit matched the training mission
  and drifted 64 m on the unseen one. ALPHA RISE's model has sway and heave added mass 3× strip theory and nose drag
  20× smaller than the fitted one.
- **Limits sized from the hull, linear damping tight:** too tight to fit.
- **Generous damping limits (V3, what ships):** the pond held out came close; the bay held out blew up (mean skill
  −3.18) from one start.
- **V3 with 4 starting points (#79):** +0.191 on the bay (hand-written +0.183) and +0.205 on the pond (+0.227). So
  every guide, the written file's header, the command's last line and the web card say to calibrate it with
  `--restarts 4`.

### 78. Unstable values don't stop a fit, and the web forms get the missing options (2026-10-19)

**Why.** URI's re-run (#76) left two gaps.
- **X2 (surfacings removed + widened limits) stopped after 3 minutes:** the optimizer tried values inside the limits
  that make the simulation blow up, and the exception ended the fit.
- **The web forms lacked options the command line has (F11):**
  - calibrate: workers, current-estimation rounds, the drag prior, the torque reference point and the seed;
  - fit-stonefish and export-stonefish: the replay tool's location;
  - fit-stonefish: iterations and workers.

**What changed:**
- **`objective.residuals_for_run`** (what the optimizer minimizes) returns, for a recording on which the parameters
  blow up, the residuals of predicting no motion ×10 in the normal shape. The optimizer sees a very bad fit and moves
  away, instead of stopping.
  - Validation and reporting still detect and report divergence as before (`stable_rmse_by_dof`).
  - Other errors still raise.
- **The web forms** have every option above. The replay-tool field is checked: it must exist and be runnable.
- **Tests:**
  - the penalty: same shape as a normal evaluation, with and without prediction windows, and 100× the no-motion cost;
  - other errors still raise;
  - every new form option reaches the command, and bad values are refused.

### 79. `strathe calibrate --restarts N`: several starting points, keep the lowest training cost (2026-10-19)

**Why.** #76 made the random start repeatable (`--seed`), but a hard calibration still depends on it. On URI's re-run,
B8 (surfacings removed) landed at +0.24 or +0.06 mean skill and B9 (20 Hz) at +0.23 or +0.18, depending on the start.
The describe-stonefish variants (#77) all stopped at worse training costs than the hand-written description.

**The evidence that makes the choice safe:** every time repeats landed in different places, the better one on the
unseen mission also had the lower *training* cost:
- B8: +0.24 at a cost of 3.86, against +0.06 at 4.17;
- B9: +0.23 at 2.018, against +0.18 at 2.217.

So the choice never needs the held-out data.

**What changed:**
- **`optimizer.calibrate`** keeps the restart with the lowest cost, converged or not. A best restart that stopped at its
  budget is reported as not converged, instead of being dropped for a worse converged one.
- **`metrics.json`** records each start's cost (`optimizer.restart_costs`). The report lists them and says whether the
  starting point mattered (within 1%) or not.
- **`strathe calibrate --restarts N`** (default 1).
  - With `--estimate-current` it applies to the first round only; later rounds start from its result.
  - The web form has a "Starting points" field.
  - LAB_GUIDE, README and LIMITS say when to use it.
- **Tests (`tests/test_multi_start.py`):**
  - the flag reaches the first fit only;
  - the lowest cost is kept even when it stopped at its budget;
  - one start behaves as before;
  - the report's lines;
  - the web field.

**Validation on URI's data** (4 starts, seed 11). Mean skill on the unseen mission, and drift:

| Run | One start | Four starts | Hand-written description |
|---|---|---|---|
| B9, pond held out | +0.176 | **+0.227**, 3.5 m at 60 s | |
| B8, pond held out | +0.062 | +0.062 | |
| V3 (#77), pond held out | +0.159, pitch 21° | **+0.205**, 5.25 m at 60 s, pitch 12° | +0.227, 3.5 m |
| V3 (#77), bay held out | **−3.18**, 65.7 m at 60 s | **+0.191**, 0.75 / 4.04 m at 10 / 60 s, pitch 7.2° | +0.183, 0.72 / 3.98 m, pitch 8.3° |

- **B9 and V3:** four starts pick the good result in every case. On the bay, V3 goes from a blow-up to the
  hand-written description's result. Two of the four pond starts blew up (costs 170 and 307) and were dropped by cost.
- **B8 doesn't help:** all four round-1 starts agreed (costs 4.76–4.78). Its good/bad split happens in a later round,
  which starts from round 1's result.
- **A gap this found, fixed:** on the bay run, every start used up its evaluation budget, so none "converged". The CLI
  then finished the run from its checkpoint (the lowest cost seen over all four starts), but the report lost each
  start's cost. `optimizer.RestartsFailed` now carries the costs; `run_calibration.py` adds them to the checkpoint, and
  `finalize_checkpoint.py` puts them in `metrics.json`. Tested in `tests/test_multi_start.py`.

**B8, restarting later rounds: tried, didn't help, replaced.**
- **What was tried:** `--restarts-every-round` restarted every current round from the previous round's result plus
  three points around it.
- **The result on B8** (pond held out, seed 11): all four starts in round 2 and in the final fit ended at the same
  cost (4.2372, then 4.1704): +0.062, 10.6 m at 60 s, the bad result. Nothing reachable around round 1's result
  leads elsewhere.
- **Where the split really is:** in round 1, by a hair.

  | Run | Round 1 cost | Round 2 cost | Final cost | Mean skill on the unseen mission |
  |---|---|---|---|---|
  | B8r2, B8r3 (good) | 4.7809 (net buoyancy 25.4 N) | 3.9306 | **3.8599** | +0.236 |
  | B8r4, M_B8, M_B8e (bad) | **4.7598** (16.5 N) | 4.2372 | 4.1704 | +0.062 |

  The two round-1 results lead to different estimated currents, and the start with the *lower* round-1 cost ends
  worse. Choosing the lowest cost after round 1 is too early, and it is what four first-round starts did. The rule
  "the lower training cost is the better result" still holds at the end: 3.86 against 4.17.
- **The replacement, `--restart-whole-fit`.** With `--estimate-current` and `--restarts N`, the whole fit (every
  round) runs N times, with seeds S..S+N−1, each in `startK/`, and the lowest *final* training cost is kept.
  - The check against unseen recordings and the report run once, on the chosen start.
  - `whole_fit_starts.json` records every start's costs round by round, and the report lists them.
  - It takes N times as long.
  - It has a web checkbox and is in LAB_GUIDE, README and LIMITS.
  - `--restarts-every-round` is removed.
  - The optimizer keeps one change from it: a warm start with several starts keeps the warm point, unperturbed, as
    its first.
- **Tests:**
  - three starts × three fits, with the choice made on the final cost, not round 1's;
  - the check and the report run once;
  - the record;
  - the refusals and notes;
  - the report's paragraph;
  - the web checkbox.
- **Validation on URI's data:** B8, pond held out, `--restarts 4 --restart-whole-fit --seed 11` (seeds 11–14). It
  gives **the good result**:

  | B8, pond held out | Mean skill | Drift at 10 / 60 s | Pitch after 10 s | Final training cost |
  |---|---|---|---|---|
  | `--restart-whole-fit`, 4 starts | **+0.237** | 0.68 / 4.69 m | 11.7° | 3.8599 |
  | A good single start (B8r2) | +0.236 | 0.67 / 4.67 m | 11.7° | 3.8599 |
  | 4 starts chosen after round 1, or in every round | +0.062 | 1.68 / 10.56 m | 18.5° | 4.1704 |

  - **All four whole fits ended in the good result** (round 1 4.7809, round 2 3.9306, final 3.8599–3.8604), so this
    run didn't have to choose between the two. That the lower *final* cost is the better result rests on the earlier
    runs (3.86 against 4.17).
  - **What it shows:** with the same seed 11, four starts chosen after round 1 kept the 4.7598 start and ended badly.
    The whole fit, from that seed and three others, ended well every time.
  - **Cost:** 8.5 h on 7 cores, about 2 h of it on battery. Four round-1 starts took 7.4 h on 2 cores.
  - **The report** lists each start's costs by round, and here says they ended within 1% of each other.

### 80. Strathe's damping in Stonefish as it is: the patch's per-axis damping (2026-10-19)

**Why.** `export-stonefish` (#74) helped after a minute but hurt over seconds. Stonefish's drag is a sum over the hull's
mesh: cubic in speed, and coupled across axes (a hull pitched a few degrees picks up surge drag). No Stonefish
coefficients reproduce Strathe's per-axis linear + quadratic damping, so it was converted, and the conversion is what
hurt (#74, #75).

**What changed:**
- **The Stonefish patch** (`calibration/stonefish/stonefish_added_mass.patch`, now about 330 lines) adds optional
  per-axis damping: `<hydrodynamics damping_linear="Xu Yv Zw" damping_linear_rot="Kp Mq Nr" damping_quadratic="..."
  damping_quadratic_rot="..." damping_point="x y z"/>`.
  - Force per axis = (linear + quadratic·|v|)·v, with v relative to the water, at the point (the robot's frame); its
    moment about the CG follows.
  - On a compound's part it replaces the mesh drag of every external part (the values are the whole vehicle's).
  - Off unless a vehicle file sets it. Stonefish without the patch ignores the attributes.
- **The replay tool** reports `axis_damping` in `--info`, and `--features` without a scenario.
- **`export-stonefish`**, when the build has it:
  - writes Strathe's damping unconverted, all six axes, at `--reference-point`;
  - puts the vehicle's geometry where Strathe's model has it. The CG goes to the reference point, through the tuned
    part's `<cg>`. The buoyancy part is moved (`PartSettings.buoyancy_shift`, its `<compound_transform>`) to put the CB
    at the calibrated offset, fore-aft included (zero unless fitted).

  Without the patch, it converts as before, and the report says why.
- **Why the geometry has to move:** Strathe's rigid-body terms and restoring moment are written about the reference
  point. In Stonefish's own geometry, the logged thrust and the damping acted about a CG 11 cm ahead of and 9.6 cm above
  the reference point (the export had raised it to match Strathe's 5 mm CG−CB). That made about 3.4 N·m of pitch moment
  that Strathe's model doesn't have, against a calibrated righting of 1.75 N·m/rad.
  - The damping alone pitched ALPHA RISE 36° (pond) and 23° (bay) after 10 s.
  - Adding Strathe's fore-aft offset brought it to 27° / 14°.
  - Adding the CG at the reference point brought it to 19° / 4°.
- **`strathe check`** says whether the replay tool is built and which of the patch's features its Stonefish has, and
  when to rebuild.

**Results on URI's ALPHA RISE** (B0, position error after 10 / 30 / 60 s on the unseen mission):

| Model | Pond held out | Bay held out |
|---|---|---|
| URI's model | 2.40 / 15.9 / 47.6 m | 1.92 / 11.4 / 37.4 m |
| Drag converted (#74) | 4.16 / 14.8 / 34.3 m | 4.48 / 13.9 / 30.9 m |
| **With the patch's damping** | **1.85 / 10.1 / 36.0 m** | **1.44 / 5.6 / 17.6 m** |
| Strathe's own model | 1.74 / 10.3 / 26.4 m | 1.47 / 9.1 / 30.6 m |
| `fit-stonefish` (F2) | 2.89 / 9.7 / 21.6 m | 2.48 / 7.2 / 13.8 m |

- **Speeds match Strathe's model:**
  - forward-speed error after 10 s: 0.09 and 0.13 m/s, against Strathe's 0.09 and 0.12 (converted: 0.44 and 0.42);
  - sway and heave match too;
  - error after 10 s falls 56–68% against the conversion;
  - better than URI's model at every horizon in both directions.
- **Bay held out:** better than Strathe's own model at every horizon (1.44 / 5.6 / 17.6 m against 1.47 / 9.1 / 30.6 m),
  with pitch 3.8° against 9.9°.
- **Pond held out:** pitch is 19° against Strathe's 11°, which grows the error over a minute (36.0 m).
  - **The cause is water density (found in #88):** this run was scored with `--density-by-date`. The bay-calibrated
    vehicle in fresh water is 2.3% less buoyant in the simulator, while Strathe's own model keeps its calibrated
    buoyancy.
  - **Scored without it,** the export reproduces Strathe's own model: 1.74 / 6.6 / 30.1 m against 1.74 / 10.3 / 26.4 m.
    Velocity errors are identical; pitch 11.2° against 10.9°, yaw 7.2° against 7.2°.
  - So the patched export carries Strathe's model, and the pond's gap was the density setting.
- **So:** export with the patch for a model that behaves like Strathe's; `fit-stonefish` when the best over a minute
  matters on a hard mission (pond: 21.6 m).
- **Tests:** the exact export (values unconverted, the mesh coefficients untouched, attributes written) and the
  `strathe check` line for each kind of build. The patch applies cleanly to a fresh Stonefish checkout (GSO-soslab
  fork, `7d52673`) and builds on macOS.

**Still open:**
- the pond's remaining pitch difference;
- `fit-stonefish` could fit the patch's damping directly, instead of the mesh coefficients.

### 81. One workflow for every simulator: `fit-sim` / `export-sim`, and any simulator through a replay contract (2026-10-19)

**Why.** Labs use Gazebo, MuJoCo, HoloOcean and in-house simulators, not only Stonefish (asked 2026-10-19; plan:
Gazebo, then MuJoCo, then HoloOcean, plus a protocol for any simulator; stress-tested on URI's data). Half of
`fit-stonefish` was already simulator-agnostic. This phase (P0–P1) pulls that half out, and makes any simulator usable
through a documented replay command.

**What changed:**
- **`calibration/simfit/core.py`:** the shared half, moved from `stonefish_fit.py` unchanged:
  - windows, scores, baselines, Strathe's free run;
  - the least-squares fit, which now takes a `simulate(x, i, run, rows, folder)` callback;
  - loading recordings, the table and the plot.

  `stonefish_fit` keeps every name, and `StonefishFitError` is `SimFitError`. **Golden outputs** of `fit-stonefish` and
  `export-stonefish` against the fake simulator were written before the move (`tests/test_golden_stonefish.py`), and
  are byte-identical after it.
- **`simfit/knobs.py`:** `KnobSpec` and `Knobs`, any tunable number in a model:
  - transforms `log_factor`, `log` and `linear`;
  - limits;
  - `--hold`;
  - ties for `--axisymmetric`.
- **`simfit/editors.py`:** change numbers in a lab's own model file, leaving everything else byte for byte:
  - XML by a small XPath (SDF, URDF, MJCF, .scn), using expat byte offsets so comments and formatting stay;
  - JSON and YAML by JSON pointer;
  - regex with one capture group.
- **`simfit/replay_tool.py`:** Stonefish's replay CSV contract, for any command:
  - `--features`, `--info`, `--windows IN --out OUT`, plus `--model-dir`;
  - plain timeout messages;
  - `docker:IMAGE`, which runs the tool in a container with the folders mounted, so there's nothing to build.
- **`simfit/adapters/`:** `FileAdapter` (knob definitions, runnable copies, the calibrated file, export maps).
  `external.py` builds one from a lab's YAML description.
- **`simfit/runner.py`:** `run_fit` and `run_export`, the same steps and report as fit-stonefish for any adapter. The
  outputs are `<model>_calibrated.<ext>`, `<sim>_fit.yaml`, `<sim>_fit_report.md` and `drift.png`.
- **`strathe fit-sim` / `export-sim --sim stonefish|gazebo|mujoco|holoocean|external`:**
  - `--sim stonefish` runs `fit-stonefish` / `export-stonefish` unchanged;
  - the others go to `python -m simfit.cli`;
  - `--ingest-spec` converts raw logs first;
  - relative paths are made absolute;
  - `--help` lists each simulator's options.
- **Docs:**
  - `calibration/simulators/README.md` (the overview, and both ways to run each simulator);
  - `SIMULATOR_PROTOCOL.md` (the contract, frames, the YAML, a checklist);
  - a working example in `calibration/simulators/example/` (`example_replay.py`, `vehicle.json`, `my_sim.yaml`; the
    tests run it);
  - LAB_GUIDE, README and LIMITS updated.

**Tests (`tests/test_simfit_external.py`):**
- fit-sim on a described simulator moves the knobs toward the truth and cuts held-out drift;
- the calibrated file changes only the knobs;
- `--axisymmetric` ties sway and heave;
- export-sim puts Strathe's values in exactly (the true vehicle is reproduced: 0.00 m drift);
- plain messages for a missing key, an incomplete description and missing logs;
- knob transforms, holds and ties;
- the editors change only the number;
- `strathe fit-sim` routing;
- the shipped example runs as documented.

**Next:** Gazebo Harmonic (#82), MuJoCo (#83), HoloOcean (#84), then `setup-sim`, the Docker images, `convert-sim`, the
web cards and the URI stress test of every simulator.

### 82. Gazebo Harmonic: `fit-sim --sim gazebo`, an exact export, and `setup-sim` (2026-10-19)

**What changed:**
- **`calibration/simulators/gazebo/strathe_gz_replay.cpp`:** the replay contract on `gz::sim::TestFixture`.
  - The lab's world runs headless with its own Physics, Buoyancy and Hydrodynamics systems.
  - **Each window:** the pose at `--depth` with the vehicle at rest, then the logged velocity in the link's axes on the
    next step. Done in one step, the body-frame velocity was applied in the previous window's attitude: surge and sway
    were off by 0.18 m/s.
  - **Every step:** the logged wrench at the reference point (`AddWorldWrench`).
  - Velocity is read back at the reference point.
  - Frames are converted NED/FRD ↔ ENU/FLU.
  - It refuses a physics step that doesn't divide the logs' sample time.
  - `--info` reports the link's mass, inertia, centre of mass and primitive collision volumes from the SDF; `--features`
    reports the gz-sim version.
- **`simfit/adapters/gazebo.py`:** finds the vehicle (the model with the Hydrodynamics plugin, in the world or an
  included model folder) and offers:
  - `drag`: the plugin's damping terms present, with their magnitudes tuned;
  - `added_mass`: `<fluid_added_mass>` or `xDotU…`;
  - `buoyancy`: scaling the primitive collisions;
  - `cg`: the inertial pose;
  - `mass`;
  - `--axisymmetric` ties.

  A runnable copy edits only the model file. An included model folder is mirrored with symlinks and the world's
  `<include>` repointed. `--density-by-date` goes into the Buoyancy system.
- **The export is exact** because Gazebo's Hydrodynamics is Strathe's law:
  - damping and hull lift into the plugin;
  - added mass into `<fluid_added_mass>` (DART's own, with Coriolis and Munk terms), with the plugin's legacy explicit
    `xDotU…` terms set to 0, since those turned a test vehicle 0.65 rad/s away from Strathe's model;
  - mass, effective inertia, the centre of mass at the reference point;
  - the collisions resized and moved for the net buoyancy and centre of buoyancy.
- **A reference point away from the link origin:** Gazebo damps at the link origin.
  - Linear damping (6×6 cross terms) and added mass (Jᵀ M J) are moved there exactly.
  - Quadratic damping can't be, so the report says the export is close, not exact.
- **`strathe setup-sim gazebo [--prefix]`** builds the tool against an installed gz-sim 8 (conda-forge, apt or Homebrew)
  and checks it. If Gazebo isn't installed, it prints the install commands for each route.
- **`strathe check`** has a Gazebo line.
- **A Dockerfile** gives the no-compile route (`--replay-tool docker:strathe-gz-replay`, with folders and environment
  passed through). It's untested: there's no Docker on the dev machine.

**Checked on real Gazebo 8.10 (macOS, conda-forge):**
- **Frames:** analytic steady states hit exactly.
  - 20 N forward → 0.737 m/s;
  - 20 N to starboard → +0.434 m/s;
  - 0.5 N·m of yaw → +0.225 rad/s.
- **Export against Strathe's own model,** over a 30 s random-thrust window with a moving, tilted start: at most 0.006 m/s
  or rad/s, neutral or 2 N buoyant.
- **`fit-sim`** recovered drag from logs generated by a "true" world, cutting held-out drift by more than half.
- **Speed:** three 10 s windows replay in about 5 s.

**Tests (`tests/test_simfit_gazebo.py`):**
- Without Gazebo (they run in CI): finding the vehicle and knobs; only tuned numbers change; the export mapping
  (FLU signs, lift, the added-mass matrix, buoyancy, cross terms for an offset reference); plain messages.
- With Gazebo, skipped unless the tool is built: frames and steady states; the export against Strathe's model (neutral
  and buoyant); fit-sim end to end.

### 83. MuJoCo: `fit-sim --sim mujoco`, nothing to build (2026-10-19)

**What changed:**
- **`simfit/adapters/mujoco.py`:** the lab's MJCF vehicle (a body with a free joint, x forward, z up), replayed
  in-process with the `mujoco` package Strathe already installs.
  - **Each window:** the logged pose and velocity are set (the free joint's velocity is the body origin's in world axes;
    its angular velocity is in body axes).
  - **Every timestep:** the logged wrench at the reference point, through `xfrc_applied` moved to the centre of mass.
    The kinematics are refreshed first: after `mj_step` they still describe the pose before the step.
  - **Read-back:** velocity at a site added in memory at the reference point, after a forward pass.
  - It refuses a timestep that doesn't divide the logs' sample time.
- **What can be tuned, found in the file:**
  - `drag`: each named ellipsoid hull geom's blunt, slender and angular coefficients. MJCF's default `fluidcoef` is
    written in first, using a new editor helper, `ensure_attr`. With no ellipsoid geom (the inertia-box model), the
    option's density and viscosity are tuned instead.
  - `lift`: the Kutta and Magnus coefficients.
  - `buoyancy`: `gravcomp`.
  - `cg`: the inertial pos.
  - `mass`.
- **`export-sim` refuses with a plain message.** MuJoCo has no per-axis linear damping or buoyancy force, and doesn't
  apply the part of added mass that resists acceleration. It points to `fit-sim` and to `mujoco_backend.py`, which runs
  Strathe's own model in MuJoCo.
- **`strathe check`** has a MuJoCo line.
- **Guide:** `calibration/simulators/mujoco/README.md`, with an example `example/auv.xml`.

**Tests (`tests/test_simfit_mujoco.py`; MuJoCo is a pip package, so they run in CI):**
- every axis's sign in Strathe's frames;
- the start state round-trips within 0.01 at an offset reference point (no water);
- knobs found, MJCF defaults made explicit, only tuned numbers changed;
- plain messages;
- fit-sim recovers the ellipsoid drag coefficients from logs (held-out drift more than halved);
- export refused.

A 10 s window replays in about 0.3 s.

### 84. HoloOcean: `fit-sim --sim holoocean` through its Fossen dynamics; export as a Fossen class (2026-10-19)

**Why this way.** HoloOcean 2.x drives torpedo vehicles with Fossen's equations (`"fossen_model": "torpedo"`, control
scheme 1), configured by the scenario's `agents[i]["dynamics"]` (docs v2.3.0). Those equations are Python
(`holoocean.fossen_dynamics.torpedo.Torpedo`: `Minv`, `dynamics(eta, nu, u_actual, u_control, h) -> (nu_dot,
u_actual)`). A window can therefore be replayed through the lab's own vehicle class without the Unreal engine, exactly:
- the propeller and fins at rest;
- the logged thrust added as M⁻¹τ, since the equations are linear in τ;
- the velocity and thrust moved between the reference point and Fossen's origin.

**What changed:**
- **`simfit/adapters/holoocean.py`:** reads the scenario (JSON), finds the Fossen agent, and loads the class
  (`--fossen-model`, `--configure`).
  - **What it tunes,** in the `dynamics` keys, with class defaults when a key is missing: the `T_*` time constants,
    `zeta_*`, `K_nomoto`, `Cd`, `e`, `r44`, `r_bg` x/z and `mass`.
  - `--density-by-date` writes `rho`.
  - **Integration:** at `--hz`, attitude by quaternion; frames are already NED/FRD.
- **`export-sim`** writes `strathe_fossen.py`: a self-contained Fossen class (numpy only) with Strathe's model (damping,
  added mass and Coriolis at the reference point, restoring, hull lift), moved to the class's origin
  (M_o = Jᵀ M J, forces Jᵀ f). It has HoloOcean's `dynamics()` interface, for the custom-dynamics loop. The report scores
  it on held-out recordings.
- **`strathe check`** has a HoloOcean line.
- **Guide:** `calibration/simulators/holoocean/README.md`, with an example scenario.

**Tests (`tests/test_simfit_holoocean.py`):** against `tests/fake_holoocean`, a stand-in with HoloOcean's interface,
since HoloOcean doesn't install on macOS or in CI.
- knobs from the scenario and the class;
- the calibrated scenario changes only those keys;
- fit-sim recovers `T_surge` and `Cd` from logs (held-out drift more than halved);
- the exported class replays like Strathe's own model within 0.02 over 30 s, at the origin and at an offset reference
  point, with lift and buoyancy;
- plain messages.

**Still to do, on Windows:**
- the real `holoocean.fossen_dynamics` (the parameter names follow the docs);
- an in-engine run against the Python replay;
- the tick-rate check (#40).

HoloOcean results are provisional until then.

### 85. `strathe convert-sim`: a Stonefish model written for Gazebo, MuJoCo or HoloOcean (2026-10-19)

**Why.** The URI stress test of every simulator needs ALPHA RISE in each of them, and URI only has a Stonefish model.
It's also useful for a lab moving between simulators.

**What it does:** `simfit/convert.py` reads the source through the replay tool's `--info`. Each converted model has its
origin at the logs' reference point.
- **What it reads:**
  - the mass (`--mass`), with the rigid inertia scaled to it;
  - the CG and CB;
  - the hull's length and diameter from the mesh;
  - the displaced volume, grown to keep the model's trim when the weighed mass differs;
  - added mass per axis and inertia;
  - drag: quadratic matched at 0.5 m/s, viscous, rotational by strip theory.
- **Gazebo:** a world with Physics, Buoyancy and Hydrodynamics, `<fluid_added_mass>`, and a cylinder collision at the CB
  with the displaced volume. The hull-lift terms are written at 0 and are now a `lift` knob group (signed, linear) in
  the Gazebo adapter.
- **MuJoCo:** an MJCF with an ellipsoid hull. A body named `buoyancy` at the CB carries the buoyancy (gravcomp on
  0.01 kg). The MuJoCo adapter now tunes that body's gravcomp when it exists (±15%).
- **HoloOcean:** a torpedo scenario with first-order time constants from the damping and added mass.

**Found on ALPHA RISE (URI's model, 33 kg):**
- Keeping the weighed mass with the model's own volume made the Gazebo copy 52 N heavy, hence the trim fix.
- Buoyancy at the centre of mass (plain gravcomp) left MuJoCo with no righting: it pitched 74° under 20 N of thrust.
  With the buoyancy body, 9°.
- A bare converted hull in Gazebo still pitches up under thrust: sway and heave added mass of about 230 kg against
  10 kg in surge make a strong destabilising Munk moment, which ALPHA's fins counter. The conversion has no fin data,
  so the `lift` knobs are there for `fit-sim`.

**Tests (`tests/test_simfit_convert.py`):** the facts relative to the reference point and the trim kept; each converted
model loads in its adapter, and a trimmed MuJoCo copy neither rises nor falls; `strathe convert-sim` passes its options
through.

### 86. The web page: "Calibrate your simulator model" and "Put your Strathe calibration into your simulator" (2026-10-19)

**What changed:** two cards with a simulator selector (Gazebo Harmonic, MuJoCo, HoloOcean, your own description).
- `strathe_web.sim_cli_args` maps each simulator's fields to its options (`SIM_FIELDS`):
  - the model file;
  - Gazebo's included model file;
  - the vehicle's name (Gazebo model, MuJoCo body, HoloOcean agent) and Gazebo's link.
- Shared fields: the logs (an ingestion spec or a folder), the reference point, held-out dates, what to fit, holds,
  `--axisymmetric`, densities, `--from-strathe`, a comparison calibration, the replay tool (a path or `docker:IMAGE`),
  iterations and workers.
- Every field is checked, with a plain line for anything wrong.
- The routes `/jobs/fit_sim` and `/jobs/export_sim` start `strathe fit-sim` / `export-sim`. Stonefish keeps its own cards.

**Tests:**
- every field reaches the command;
- bad input is refused with a plain line;
- the server starts exactly the command;
- the home page has 11 forms, each label pointing at one input. The new cards first reused an existing `fs-` id prefix;
  now `simf-` and `simx-`.

Checked in the browser: the cards render and there are no console errors.

### 87. `setup-sim stonefish`, and Stonefish and Gazebo with nothing to build (Docker) (2026-10-19)

**What changed:**
- **`strathe setup-sim stonefish [--source PATH] [--prefix PATH] [--jobs N]`** does what `calibration/stonefish/README.md`
  sections 1–2 describe:
  1. clone GSO-soslab/stonefish @ 7d52673 (the commit the patch is made for), or use the lab's checkout;
  2. apply `stonefish_added_mass.patch` (it detects when it's already applied);
  3. build and install Stonefish into `~/.strathe/stonefish`;
  4. build the replay tool against it;
  5. finish with the `strathe check` line.

  On macOS it adds Homebrew's OpenMP and the SDK's C++ headers (what the URI lab build needed, #76). Each step that
  fails prints its error.
- **Checked on this Mac, on a clean copy of the source:** patched, built, installed, replay tool built and checked
  (✅ added mass and damping) in 4 min 48 s.
- **`--replay-tool docker:IMAGE` for Stonefish** (`stonefish_fit.Replay` and `find_tool`): the scenario's folder,
  Stonefish's data folder and the work folders are mounted at the same paths. `calibration/stonefish/Dockerfile` builds
  the image (Ubuntu 24.04, the fork at the patch's commit, patched, plus the replay tool). Gazebo's is in #82. **Both
  images are untested:** there's no Docker on the dev machine.
- The golden outputs of fit-stonefish and export-stonefish are unchanged.
- **Tests:** the Docker command line and mounts; the plain "needs Docker" message; setup-sim saying what's missing;
  the macOS flags.

### 88. Stress test of every simulator on URI's ALPHA RISE (2026-10-19/20)

**What was run:** a fresh clone of `feat/simfit` in `~/strathe_uri_lab_v3`, set up from the lab's side:
- `check`;
- `setup-sim gazebo` (48 s) and `setup-sim stonefish` (2 min 43 s);
- `convert-sim` of URI's Stonefish model to Gazebo and MuJoCo (12 s each).

Then 44 runs: Gazebo, MuJoCo and Stonefish through `fit-sim`, the protocol route, and exports of six Strathe
calibrations, with every option varied. The writeup is `~/strathe_uri_lab_v3/URI_SIMULATORS_REPORT.md`, with
`tables/sims.csv` and `NOTES.md`.

**Results** (position error after 10 / 30 / 60 s on the unseen mission; pond, bay):
- **Gazebo, calibrated through Gazebo** from the conversion: 0.81 / 5.1 / 15.3 m and 1.30 / 6.0 / 15.0 m. The
  conversion was 6.10 / 19.2 / 40.1 and 4.76 / 17.9 / 37.7; URI's Stonefish model 2.40 / 15.9 / 47.6 and
  1.92 / 11.4 / 37.4; Strathe's own model 1.74 / 10.3 / 26.4 and 1.47 / 9.1 / 30.6. The best simulator model of ALPHA
  RISE so far.
- **MuJoCo, calibrated:** 0.89 / 6.8 / 27.4 and 1.57 / 8.2 / 19.5 m. Its fluid model can't represent per-axis damping
  or buoyancy at a CB.
- **Exports reproduce Strathe's model** when scored in the calibration's own water: Gazebo B0 on the pond 1.75 / 10.7 /
  27.1 m against Strathe's 1.74 / 10.3 / 26.4, with identical velocity errors and pitch 11.0° against 10.9°; Stonefish
  (patched) 1.74 / 6.6 / 30.1 m, pitch 11.2°.
  - The pond's earlier extra pitch drift (about 19°, #80) was `--density-by-date`: in fresh water the bay-calibrated
    vehicle is 2.3% less buoyant in the simulator, and URI almost certainly re-trimmed it.
  - The guides now say to use the flag only when the ballast was the same in both waters.
- **Cross-checks:**
  - `--sim external` reproduces Gazebo's adapter exactly (same cost, 744.502);
  - `fit-sim --sim stonefish` reproduces `fit-stonefish` (F2) in both directions;
  - workers 1 and 4 give identical results, in Gazebo and MuJoCo;
  - exports re-run give identical results.
- **Options** (the details are in the report):
  - tune drag with buoyancy and the CG, not alone;
  - hull lift and `--axisymmetric` cost over a minute;
  - variation weighting beats magnitude;
  - `--from-strathe` is not the better start (as #74).
- **Multi-start (#79), pond held out:**
  - B9 picks the good answer (+0.227, where the single start gave +0.176);
  - V3 (describe-stonefish) +0.205 / 5.3 m, against the single start's +0.159;
  - B8 still lands at +0.062: its split happens after round 1, the only multi-start round.

**Friction found from the lab's side, all fixed:**
- **F1:** `check` listed HoloOcean twice.
- **F2:** the Gazebo export's rigid inertia broke the triangle inequality (Strathe's effective inertia isn't a rigid
  body's) and Gazebo refused the link. Fix: a valid rigid inertia plus rotational `<fluid_added_mass>`, still exact.
- **F3:** `--from-strathe` on MuJoCo gave export's message.
- **F4:** an incomplete calibration crashed with a TypeError.
- **F5:** a held-out date the logs don't have gave a traceback.

**Ease of use:**
- the "any simulator" guide's example ran as written in the fresh clone;
- 15 common mistakes all give a plain line.

**Not tested:**
- real HoloOcean (needs Windows or Linux; a stand-in with its interface was used);
- the Docker images (no Docker here);
- a public Gazebo model (bluerov2_gz).

### 89. `strathe comms-study`: N vehicles coordinating under acoustic latency, bandwidth and packet loss (2026-10-20)

**Why.** Strathe has to show teams they can test multi-vehicle coordination under the limits of acoustic
communication, not only simulate vehicles. `strathe fleet` is two vehicles flown by trained policies, over a
point-to-point link with a per-tick byte budget. Its only comm knob is the bundled L0–L3b loss levels: latency and
bandwidth couldn't be set on their own, and there was no shared channel.

**What's new:**
- **`acoustic_network.py`, a shared acoustic channel for N vehicles.** Every effect is in seconds:
  - travel time (range ÷ 1500 m/s);
  - time on the air (preamble + bits ÷ bit rate);
  - the modem's decoding delay;
  - TDMA slots or random access, with jitter;
  - half duplex;
  - collisions at each receiver;
  - independent or bursty (Gilbert-Elliott) loss, an optional distance ramp, and range.

  Every packet's fate is counted. It is vehicle-agnostic like `comm_channel.py`, and deterministic per seed.
- **`fleet_coordination.py`, 2–12 vehicles on Strathe's calibrated physics.**
  - **The task:** they follow a leader whose turns they can't predict, in a wedge, line, column or custom
    formation.
  - **What a vehicle knows:** its teammates only through beacons and a belief model (`fleet_belief.py` gains
    `estimate_state`: position, velocity, heading).
  - **Controllers:** a built-in leader-follower PID, or a team's own `module:Class`.
  - **Scoring, always on the truth:** formation error, closest approach and collisions, belief error, beacon age,
    and the channel's statistics.
  - **A formation that turns with the leader** is refused when its outer slot would need to outrun the vehicle,
    with the slowest turn that works.
- **`strathe comms-study` (`comms_study.py`).** One setting at a time, on paired seeds:
  - modem delay, bit rate, loss, burst length, fleet size, beacon interval, spacing;
  - TDMA against random access, teammate prediction, and controller.

  It writes `report.md` (where coordination holds and breaks, with Wilson intervals), `summary.csv`,
  `episodes.csv`, plots and `study.yaml`. Modem classes `slow` / `medium` / `fast` / `ideal` are generic starting
  points.
- **Web card** "Test your fleet's coordination under acoustic limits". **Docs:** `calibration/COMMS_STUDY.md` (with
  how to plug in a controller), README, LAB_GUIDE section 9, and a LIMITS section.
- **Tests:**
  - `tests/test_acoustic_network.py`:
    - airtime and exact latency;
    - TDMA never collides and its cycle is the update interval;
    - random access collides when busy;
    - half duplex;
    - burst statistics;
    - range and distance loss;
    - determinism;
    - no physics imports.
  - `tests/test_fleet_coordination.py`:
    - formations;
    - refusals;
    - a good channel holds and a saturated one doesn't;
    - repeatability and paired leaders;
    - six vehicles;
    - every belief's state;
    - a team's own controller;
    - each setting changes only itself;
    - verdicts;
    - a small study end to end;
    - the CLI pass-through.
  - The web form test.

**Navigation error in `comms-study`** (2026-10-22 review). Vehicles knew their own positions exactly, which LIMITS
listed as not modelled.
- **The model:** `--nav-drift PCT` gives each vehicle a heading-reference error, drawn per vehicle, so its dead reckoning
  drifts by about that % of distance travelled. `--start-fix-error M` gives it a starting fix error.
- **Who uses what:** its controller and its beacons use what it believes; sound travels the true distance; the score is
  the truth.
- **Independence:** the errors come from their own random stream, so everything else in an episode stays the same.
  With both at 0 the results are unchanged.
- **New study settings:** `nav_drift` and `fix_error`, now in the default study. Each report says each vehicle's mean
  position error.
- **On the shipped BlueROV2** (4 vehicles, 1 kbit/s TDMA, 30 paired episodes per value):

  | Setting | Values tested | Success | Formation error |
  |---|---|---|---|
  | Drift | 0 → 5% | 100% throughout | 0.85 → 1.65 m |
  | Starting fix error | 0.5 m | 100% | 1.27 m |
  | | 1 m | 87% | 2.03 m |
  | | 2 m | 23% (12–41%) | 3.75 m |
  | | 5 m | 3% | 9.07 m |

  A follower's slot is off by the difference between its error and the leader's, which beacons can't fix without
  ranging.
- **Web, docs, tests:** web fields and sweep boxes; COMMS_STUDY.md and LIMITS; tests for the model, the settings and
  the web form.

**`strathe fleet` gets the two settings it was missing** (2026-10-22 review). Its two-vehicle experiments had a modem
delay and a byte budget in the experiment YAML, but no flag or web field, so testing latency or bandwidth meant hand-editing
YAML:
- **`--modem-delay S`** sets the acoustic modem's own delay on top of travel time. It's refused together with
  `--link-calibration`, which measures the delay from the lab's logs.
- **`--bandwidth-bytes N`** sets the bytes per vehicle per 0.1 s tick. A beacon is 24 bytes, so this is all or nothing
  in this model, and the help points to `comms-study` for bandwidth shared by several vehicles. It applies on top of one
  packet-loss level, and the error now says to add `--comm L0`.
- **Merging:** the flag merges into the YAML's own loss settings instead of replacing them.
- **Web and report:** both are fields on the web form, and the report states them.
- **Tests:** `test_fleet_modem_delay_and_bandwidth_reach_the_experiment` and the web form test; a smoke experiment ran with
  both.

**Found while building it:**
- A formation that turns rigidly with the leader is unflyable at survey turn rates: a wedge's outer slot moves at
  1.5 m/s in a 3°/s turn, against a vehicle top speed of 0.35 m/s.
- Without integral action the PID sat about 2 m behind its slot: drag at cruise is about 14 N.
- Random access with fixed beacon periods locked two vehicles into colliding forever; random jitter fixed it.

**Results on the shipped BlueROV2** (30 paired episodes per value; success = formation error under 3 m RMS and no
collision; intervals are 95% Wilson). Every run's command, CSVs, plots and report are in `~/strathe_comms_demo/`.

**1. The default study** (`strathe comms-study`, 22 min on 4 cores, 1,170 episodes). 4 vehicles in a 10 m wedge,
a 1 kbit/s modem, TDMA, beacons every 2 s (TDMA cycle 3.4 s), constant-velocity prediction:

| Setting | Holds up to | Breaks at | Formation error, easiest to hardest |
|---|---|---|---|
| Modem delay | 10 s (100%) | 20 s: 77% (59–88%) | 0.84 → 2.77 m |
| Bit rate | 50 bit/s (100%; TDMA cycle 23 s) | – | 0.83 → 2.38 m |
| Independent loss | 70% (100%) | – | 0.85 → 1.75 m |
| The same 30% loss in bursts | bursts of 4 packets (97%) | 8: 53% (36–70%); 16: 23% | 0.93 → 6.68 m |
| Fleet size | 8 vehicles (100%) | – | 0.82 → 0.92 m |
| Beacon interval | 20 s (100%) | – | 0.85 → 1.53 m |
| TDMA against random access | TDMA 100% | random access 0% (0–11%): 1% delivered, channel 151% busy | 0.85 against 24 m |

Burst length matters more than the loss rate: 70% independent loss holds, while 30% loss in bursts of 8 fails half
the time, because a burst leaves a teammate unheard for a whole turn.

**2. A slow modem** (80 bit/s, 4.2 s on the air per beacon: `--modem slow`):
- **Fleet size breaks it.** The TDMA cycle grows with the fleet: 8.5 s for 2 vehicles, 17 s for 4, 26 s for 6,
  35 s for 8.
  - 2–4 vehicles: 100%;
  - 6: 93% (79–98%);
  - 8: 3% (1–17%), with beacons 22 s old.
- **Beacon interval below the cycle changes nothing:** 1–10 s are identical, because the cycle is the real update
  interval. 20 s: 67%.
- **Teammate prediction decides it.** Same 30 episodes:

  | Prediction | Success | Formation error |
  |---|---|---|
  | Perfect knowledge | 100% | 0.99 m |
  | The calibrated physics model | 100% | 1.16 m |
  | Constant velocity | 100% | 1.72 m |
  | **An uncalibrated physics model** (vehicle_spec defaults; drag 4–5× too low) | **43% (27–61%)** | 3.02 m |
  | Last reported position | 0% (0–11%) | 3.16 m |

  A wrong model is worse than none: the uncalibrated one loses to constant velocity. The calibrated row is a best
  case, because the simulated vehicles are that model (LIMITS: "calibrated" in fleet results).

**3. Random access has a sweet spot** (`--mac aloha`, 1 kbit/s, beacon interval varied):

| Beacons every | Success | Delivered | Channel busy |
|---|---|---|---|
| 2 s | 0% | 1% | 151% |
| 5 s | 43% | 33% | 60% |
| 10 s | 70% | 62% | 30% |
| 20 s | 67% | 81% | 15% |
| 40 s | 17% | 89% | 7% |

Too often and packets collide; too rarely and the information is stale. Unlike TDMA, random access also produced
physical collisions between vehicles (1–2 episodes in 30 at 2–10 s and at 40 s). TDMA at the same modem holds at
100%.

**What this shows a team:** with the modem, fleet size, strategy and tolerance as inputs, the study finds the
breaking points before a field trial. It also shows which lever fixes each one:
- turn-taking instead of random access;
- fewer vehicles per channel, or a faster modem, when the TDMA cycle is the limit;
- a calibrated teammate model on slow links;
- interleaving or redundancy against bursty loss.

### 90. A public Gazebo model on public logs: bluerov2_gz, and values a model has at zero (2026-10-21)

**Why.** Everything in #88 was URI's vehicle and models converted from URI's Stonefish model. A lab's first try will
often be a public model, so the Gazebo path was run on public parts end to end:
- **The model:** [bluerov2_gz](https://github.com/clydemcqueen/bluerov2_gz), a BlueROV2 Heavy for Gazebo Harmonic. Its
  licence is MIT (in `package.xml`; #88's plan wrongly said it was unstated).
- **The logs:** oceansystemslab's BlueROV2 Heavy dataset, Strathe's reference data. 39 recordings on 4 days; 16 and
  19 February held out.

**A bug it found, fixed: a value the model has at 0 was never tuned.**
- **What happened:** a `log_factor` knob scales its starting value by up to 30× either way, and a 0 started from a
  1e-9 floor. The comment said "a zero drag still gets tuned"; it didn't.
- **What it hid on bluerov2_gz:** it has quadratic drag only, no linear drag and no added mass, so 9 of its 18 tuned
  values were stuck at zero.
- **The fix:**
  - **Gazebo:** a term at 0 now starts from a value sized from the vehicle, and the report notes it:
    - added mass: 0.3 × the mass;
    - linear drag: the same axis's quadratic drag × a typical speed (0.2 m/s, 0.5 rad/s);
    - quadratic drag: the linear drag ÷ that speed;
    - without either: from the mass or inertia.
  - **The generic adapter (`external`)** refuses a zero with a plain message unless the knob has `if_zero: VALUE`
    (documented in SIMULATOR_PROTOCOL.md).
  - **MuJoCo and HoloOcean had the same bug** (found in the 2026-10-22 review): a zero sat on a 1e-3 or 1e-4 floor.
    - **MuJoCo:** a zero fluid coefficient now starts from MuJoCo's own default (0.5 / 0.25 / 1.5 / 1.0 / 1.0), and a
      zero water density or viscosity from water's.
    - **HoloOcean:** a zero in the scenario starts from the Fossen class's own value, read before the scenario's are
      applied. If the class has 0 too, a plain message asks for a starting value.
    - Tested in `test_simfit_mujoco.py` and `test_simfit_holoocean.py`.
  - **Tests:** `test_a_term_the_model_has_at_zero_still_gets_tuned` and `test_the_generic_adapter_refuses_a_zero_it_cant_scale`.

**Preparing the logs.** The dataset is in ROS axes, so `example_bluerov2_gz/prepare_logs.py` converts it to
forward-right-down exactly as `frame: flu` ingestion does. It applies the 8.5 cm torque shift Strathe's calibration uses
for this dataset, which puts the torques at the horizontal-thruster plane: bluerov2_gz's `base_link` origin, so
`--reference-point 0 0 0`.
- **Check:** each force correlates with motion on its own axis (+0.88 surge, +0.81 sway and heave, +0.68 yaw).
- **Buoyancy:** mean Fz is +1.4 N downward, a slightly buoyant vehicle holding depth.

**Worked example:** `calibration/simulators/gazebo/example_bluerov2_gz/`:
- `world.sdf`: physics, buoyancy and the vehicle, without bluerov2_gz's network includes;
- `prepare_logs.py`;
- the guide.

**Results, the first run** (all 18 values, the default 15 evaluations, 2 cores). Held-out drift falls from
3.51 / 8.11 / 11.99 m to **1.87 / 4.17 / 6.02 m** at 10 / 30 / 60 s, and velocity errors from 0.33 / 0.30 / 0.20 to
0.22 / 0.10 / 0.08 m/s.
- **But attitude got worse:** pitch error 7.8° → 17.3° after 10 s. The fit raised the centre of gravity (1.1 → 2.7
  cm), shortening the righting arm to the buoyancy box.
- **Standing still** beats both models after 30 s, because in a tank the vehicle stays within a few metres.
- **The fit hadn't finished:** its cost was still falling at 15 evaluations.

The full table is in the example's guide.

**The report now warns what drift alone hides** (`simfit/runner.py` `_warnings`, tested):
- the calibrated model turning or tilting worse than the original (an axis 25% and 2° worse after 10 s), with what
  to try;
- standing still beating it at some horizon, which is usual for tank or station-keeping logs.

**Second run: centre of gravity held** (`--fit drag added_mass buoyancy`, converged after 27 evaluations at a training
cost of 874):
- **Better everywhere it was scored:** drift 1.89 / 3.69 / 5.37 m (−55% at 60 s). Roll, pitch and yaw errors are
  3.5° / 6.2° / 41°, slightly better than the published model's.
- **But it blew up in 35 of 712 unseen windows** at the world's 0.01 s step; the medians leave those out.
  - **Rescored at 0.005 s:** 0 of 712, with the same medians (1.90 / 3.73 / 5.46 m).
  - **Why:** the fitted roll and pitch damping (kP 17, mQ 14 N·m·s on 0.26 and 0.23 kg·m²) has an 11 ms time
    constant, and Gazebo applies the Hydrodynamics forces step by step.
- **Two report fixes from it** (both tested):
  - the bottom line says when the calibrated model blew up in unseen windows, and how many;
  - the Gazebo adapter checks, after the fit, each axis's damping time constant against the world's step. It names
    the axes and the step to use: here roll and pitch, 0.005 s. It is silent for the first run and for the published
    model, which never blew up.
- **The guide** says the step must be short enough for the damping.

**From warnings to fixes (2026-10-21).** The lab no longer has to act on the two warnings by hand:
- **A second fit when the attitude gets worse.** When the first fit makes the vehicle turn or tilt worse (an axis 25%
  and 2° worse after 10 s) and the centre of gravity was tuned, `fit-sim` and `fit-stonefish` fit again with it held.
  - **Warm start:** `fit-sim` starts the refit from the first fit's other values; Stonefish starts from the model.
  - **The choice:** the refit is kept if its attitude is no worse than the original's (or fewer axes are), its drift
    is within 10% and it doesn't blow up more often (`simfit.core.prefer_cg_held`).
  - **The files:** the kept model is `<model>_calibrated`; the other is `_cg_tuned` or `_cg_held`. Both are rows in
    the report, which says which was kept and why.
  - **Skipping it:** `--no-cg-rerun`, or a checkbox on both web cards.
  - **On bluerov2_gz** this is exactly the two runs above, as one command.
- **Gazebo: a safe step, used.** When the fitted damping is too stiff for the world's step, the models are scored at a
  safe step (the longest dividing the logs' sample time, under half the shortest damping time constant).
  `calibrated_world/` is written with it: a copy of the lab's world with the calibrated model in it, the model folder
  copied rather than linked, ready for `gz sim`. The report says how to run it.
  - **Moved or shared, it still runs** (found in the 2026-10-22 review): the copy first pointed at its model folder
    by absolute path. It now includes it as `model://`, and `RUN_ME.md` gives the command with the folder first on
    `GZ_SIM_RESOURCE_PATH`. Checked with real `gz sim` on bluerov2_gz after moving the folder: no errors, 0.005 s
    step. Strathe's own replay tool still needs absolute paths, which it writes for itself.
- **The warnings moved to `simfit.core`** (`attitude_worse`, `report_warnings`), so `fit-stonefish`'s report has them
  too.
- **Tests:**
  - `tests/test_simfit_cg_rerun.py`: the choice rule, and the rerun's files, report and skips through `run_fit`;
  - `test_attitude_worse_with_the_cg_tuned_refits_with_it_held` (Stonefish);
  - the safe step and both kinds of world copy, one of them loaded by the real Gazebo tool;
  - the web checkboxes and the CLI pass-through.
  - The Stonefish golden file gains one line, `cg_rerun: null`.
- **Docs:** `calibration/simulators/README.md` ("What the fit does for you"), the Gazebo and Stonefish guides,
  LAB_GUIDE section 6, the README command table and the example's guide.
- **A long simulator fit can be finished from its best point,** as `strathe calibrate --finish` does (2026-10-21).
  - **The checkpoint:** `simfit.core.fit` writes its best point to `<sim>_fit_checkpoint.json` after every improving
    evaluation, atomically, and the result marked complete at the end. The second fit writes
    `..._checkpoint_cg_held.json`.
  - **`--finish`** (`fit-sim`, `fit-stonefish`, a checkbox on both web cards) scores the checkpoint's point instead of
    fitting, and writes the usual outputs. The report marks an unfinished fit as not converged, and says to run
    without `--finish` if the second fit hadn't started.
  - **Refusals:** a checkpoint from other settings (other tuned values, model, logs or held-out dates) is refused, and
    the message says which. A new fit removes old checkpoints.
  - **Tests:** `tests/test_simfit_finish.py`; a Stonefish finish test; the web checkboxes. The Stonefish golden file
    gains `finished_from_checkpoint: []`.


### 91. The URI re-run's open friction items, closed (2026-10-22)

The open list from #76 had F1, F2, F5, F15, F18 and F21 still open. F21 is fixed in #79's branch. The rest:
- **F18: `fleet --workers N`.** The experiment file had `evaluation.workers`, but there was no flag or web field. Now
  there are, and they combine with `--seeds`.
- **F15: arena intervals include seed-to-seed variance.** The paired calibrated-against-generic difference, and each
  controller's success rate, pooled every episode as if independent. But all the episodes of one trained policy share
  its quirks, and the re-run saw a +25-point docking result drop to +12.
  - **The fix:** a two-level bootstrap (`cluster_ci`) that resamples training seeds, then episodes within each.
  - **The report** gives the number of seeds and each seed's difference. With one seed it says the interval covers
    episodes only, and to run `--seeds 3` or more.
  - **Tested:** two seeds that disagree (+50 and 0) now give an interval that includes 0; the old pooled bootstrap
    excluded it.
- **F5: `fit-noise` on smoothed logs.** For white noise, consecutive second differences correlate at −2/3. Smoothing
  pulls that up: an 8-sample moving average of noise gives −0.46, and smooth motion is positive.
  - **The fix:** a channel above −0.55 is flagged as not measurable. The `typical` preset's value is written instead
    of the near-zero reading, with a note naming the cause: filtered by the vehicle, or resampled by interpolation
    at ingestion.
  - **On real logs:** oceansystemslab's BlueROV2 logs (filtered odometry) have velocity and rates at about +0.65.
    URI's ALPHA RISE logs, resampled to 10 Hz, have attitude and heading at +0.75 and velocity at −0.54. Both are
    flagged.
  - **Tested** on raw and smoothed signals.
- **F1, F2: re-recorded ROS bags.** URI's missions (Dec 2025, Mar 2026) were replayed and re-recorded on 7 Oct 2026.
  - **The problem:** topics with header stamps kept the missions' clock; the thruster topic, with none, carried the
    re-recording's. With header stamps (I6) the topics "overlap for only 0.00 s"; with receive times (I10) both
    missions landed on 07-10-2026, one session, so nothing could be held out.
  - **The fix:** the bag reader measures the median (receive time − header stamp) on the topics that have both. When
    it is over a second (re-recorded, not transport delay), it moves the header-less topics onto the header clock and
    says so in the ingest report. With `stamp: receive` on such a bag, it warns that the dates are the
    re-recording's.
  - **On URI's real bags:** I6 now ingests, as 05-12-2025 and 27-03-2026. With `session_date: {from: recording}` the
    missions get those dates too, so held-out-by-date calibration works on them.
  - **Tested** with a bag re-recorded 600 days later (ROS 1 and ROS 2 .mcap): the mission's date and forces line up
    with the motion, and receive times warn. LIMITS has a row.
- **A rotated sensor mount in ingest specs** (open since the first URI run). URI's DVL is mounted rolled 180°, so its
  raw velocity was refused (heave force against motion, F4 of that run).
  - **The fix:** `body_velocity.linear_mount_rpy_deg` / `angular_mount_rpy_deg` (roll, pitch, yaw in degrees, in the
    spec's frame) turn the readings into the vehicle's axes. The ingest report records the mounts.
  - **On URI's raw DVL:** heave goes from −0.48 (refused) to +0.48.
  - **Tested:** a DVL rolled 180° and a gyro yawed 90° come back exactly.
  - **Not handled:** a DVL's lever arm (its velocity includes ω × r at its position).
- **Per-thruster thrust curves and limits** (open since the first URI run). Every thruster shared one curve and its
  maxima.
  - **New per-entry keys in a `layout:`:**
    - `curve_csv`: the thruster's own table;
    - `thrust_scale`: a multiplier on the curve;
    - `max_forward_n` / `max_reverse_n`: caps.
  - **What uses them:** `ThrusterConfig` applies them in thrust, battery-voltage rescaling and `achievable_wrench`
    (what the simulators deliver), and they survive `for_simulation`.
  - **Refused, with a plain message:** battery-voltage rescaling of a thruster with its own table, which would need T200
    data; mixed curve units; unknown or non-positive keys.
  - **Tested** in `tests/test_per_thruster.py`. All existing thruster tests are unchanged.
- **Task geometry for vehicles bigger than a BlueROV2** (open since the first URI run).
  - **The problem:** `train-fleet` always trained a 2 m ± 0.2 m formation with 0.5 m collisions, and the arena always
    used 0.5 m targets up to 3 m away. `strathe fleet` could already *test* other geometries, so a 2 m vehicle could be
    tested on a task it was never trained for.
  - **`train-fleet`** now takes `--target-separation`, `--tolerance`, `--collision-radius`, `--hold-steps`,
    `--separations MIN MAX` and `--episode-steps`. The proximity shaping radius grows with the collision radius, a
    geometry that can't work is refused, and the policy's metadata records it.
  - **`fleet`** warns when the experiment's separation differs from the one a policy was trained for, and gives the
    `train-fleet` command.
  - **The arena** takes `--target-radius` and `--target-range`, and its report states them.
  - **Web forms** have the fields.
  - **Tested:** a short training run with a 6 m geometry, the refusal, the flags passed through, the arena environment
    and text, the caution, and the web forms.
- **F11** (#78) and **F20** (the fleet report's caution about policies trained on another vehicle) were already
  addressed; the list said otherwise.

## Honest open items

- **Physics dead reckoning only ties last-beacon in still water** (#48, #54). It pulls ahead in a current it knows about (#55). Predicting a teammate means predicting its future commands. An optical link gives high-rate state sharing (#59), which isn't intent: planned-command or trajectory beacons, or teammate-policy models, remain the lever.
- **Currents:** one constant current per recording. An unknown one is estimated (#57), but a current that changes within a recording, or varies across space, isn't modelled. A fleet that works in currents should estimate the ambient current and give it to its teammate model.
- **All limits, with what to do about each, are in `LIMITS.md` (#57).**
- **Sensor noise costs formation-holding about 10 points, and almost all of it is the starting position fix (#60, correcting #58).** With a perfect fix, the assumed `typical` or the measured `open_water` sensors cost about 1. The lever is relative positioning between the vehicles (acoustic ranging, a shared USBL). TD3, the best algorithm in clean simulation, loses the most under noise (−22 points): compare algorithms with `--mode both`.
- **Sensor biases are assumed in every open-water preset (#60).** SOLAQUA measured the white noise, dropouts and USBL fix, but its DVL tracks a moving net, so biases need a seabed-locked DVL with an independent reference.
- **The optical link is uncalibrated (#59):** textbook water values, a placeholder delivery curve, a hard cone, no occlusion or sunlight. With omni terminals it lifts the shipped policies to 100% in every loss condition. With 30° cones they lose it as they roll, and fore/aft cones raise collisions (an early fix that goes stale). Whether a policy trained with `--link hybrid` learns to keep the link up is open.
- **Lab data, not built yet (#56, #57):**
  - PX4 `.ulg` and other formats beyond CSV, ROS bags and ArduPilot logs (export to CSV)
  - ~~thruster response lag~~ (opt-in since #66: the full refit trades it against yaw inertia and added mass, #66) and battery sag within a run (handled when voltage is logged, #63)
  - two different vehicles in one fleet
  - the thruster model isn't itself calibrated: the curve and layout are taken as given
- **HoloOcean results are provisional** (finding #40): the frame fix and tick-rate guard are unconfirmed in HoloOcean itself. Run `verify_holoocean_frame_conventions.py` on a HoloOcean machine, then re-run Claim A.
- **Roll and pitch have no predictive skill yet** (#50, #64): about zero skill, not positive. Free decays now measure the unforced swing (`strathe free-decay`), and the reference logs' own decays show the shipped roll/pitch damping isn't the vehicle's. What's wrong is the thruster-driven roll/pitch: with the real, light damping the model rolls and pitches far more than the vehicle does. It doesn't matter for formation-holding (#52), but it will for attitude-dependent tasks. It needs deliberate roll/pitch manoeuvres from a lab, and a fit of the thruster-driven part.
- **Calibrations are bound-limited**: every converged stand-in/MuJoCo fit has 8-15 of 22 parameters at `vehicle_spec.yaml` bounds; the least-limited (holoocean_coriolis) has 3. For roll/pitch damping this is the output-error artifact of #50, not a bounds problem. The translational damping pins still need review by someone who knows the vehicle.
- ~~**The fleet truth vehicle is still expA**~~ — switched to the torque-arm-corrected fit, with every policy retrained and every fleet study re-run (#54).
- **`Xq_dot`'s bound-reversal** (finding #33) and **`Zw_dot`'s practical identifiability gap** (#18, #29) are unchanged.
- ~~**The single-evaluation stall**~~ — caught and explained (#49): the machine was asleep. Long runs now keep the machine awake by themselves (`keep_awake.py`).
- **The ~2.5%/tick angular-velocity decay** under Custom Dynamics (#15) remains a known residual.

---

## What's actually next

0. ~~**Calibrate any simulator a lab uses, not only Stonefish**~~: done in #81–#90 (`fit-sim` / `export-sim` / `convert-sim` for Stonefish, Gazebo, MuJoCo, HoloOcean and any simulator through a replay contract). HoloOcean's in-engine check still waits for a Windows machine (item 1).
1. **On a HoloOcean machine**: run `verify_holoocean_frame_conventions.py`; fix the scenario tick rate if it fails; re-run Claim A (`replay_comparison.py`). Only then revisit roll/yaw and the 150-step truncation hypothesis.
2. **Review `vehicle_spec.yaml` bounds** for the translational parameters every fit pins, then re-fit.
3. **Roll/pitch:** from a lab partner, free-decay tests (FREE_DECAY_PROTOCOL.md) and deliberate roll/pitch manoeuvres with the tether slack, to fit the thruster-driven roll/pitch, which is what's wrong (#50, #52, #64).
4. **Thruster lag against inertia** (#66): the full refit with the lag free trades it against yaw inertia and added mass, which go to their bounds. Pin one of them (a bench measurement of thruster spin-up, or CAD/pendulum inertia), then refit; only then decide on making `thruster_dynamics` part of the default spec.
5. Long runs keep the machine awake by themselves (keep the laptop lid open); `stall_log.txt` and `eval_timing.csv` stay on as a check.
6. Re-run `check_damping_identifiability.py`'s `Mq`/`Mqq` comparison against the current Coriolis-aware Jacobian (flagged in finding #10, still not done).
7. If the corrected HoloOcean numbers still show a rotational gap, the next model-structure levers remain: off-diagonal added mass with a properly time-budgeted test, thruster/actuator nonlinearity, cross-axis damping.
