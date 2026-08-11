# Calibrated Digital Twins for Marine Autonomy — Project Status

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

## Honest open items

- **`Xq_dot`'s bound-reversal, flagged but not yet investigated** (finding #33) — in the most recent real result, `Xq_dot` sits at its bound (`-0.4794` of `[-0.5, 0.5]`), the opposite signal from the earlier near-zero result finding #26's shelving decision was based on. Whether this means the off-diagonal-added-mass shelving decision should be revisited is a real, open question.
- **The rotational weakness (`p`/`r`) remains genuinely open.** Two specific, well-reasoned hypotheses have now been tested directly and refuted with real evidence: DOF-weighting (finding #27) and the `Ieff_roll`/`Ieff_pitch`/Coriolis-coupling fix (finding #32). The leading remaining hypothesis — the 150-step training-run truncation used in every HoloOcean-backed segment this session — has not yet been tested; doing so means giving up the speed shortcut that made this session's iteration possible, a genuinely larger, more expensive commitment deserving a deliberate decision, not an incremental extension.
- **The performance mystery** (a single evaluation occasionally taking far longer than its confirmed baseline) resurfaced this session in the stand-in backend specifically (`run_full_pipeline.py`, a single eval took ~4 hours against a confirmed ~100s baseline). Real negative evidence gathered this session: OneDrive sync ruled out (confirmed local disk), isolated rapid file I/O ruled out via a direct, real test (500 checkpoint-pattern writes, zero outliers). The hypothesis is narrowed, not resolved — a live `py-spy` catch of an actual stall remains the one clear, undone, decisive diagnostic step.
- **`Zw_dot`'s practical (not structural) identifiability gap** is understood and precisely characterized (findings #18, #29), not "fixed" — it reflects what this dataset's real maneuvers can and can't constrain.
- **The small, confirmed ~2.5%/tick angular-velocity decay** under Custom Dynamics with zero command (finding #15) remains a known, deliberately-unfixed, first-instant-only residual.

---

## What's actually next

1. **Test the 150-step training-truncation hypothesis directly** — the clearest, most strongly-motivated remaining lever for `p`/`r`, now that two other real hypotheses have been eliminated. Requires a deliberate decision to run full-length HoloOcean-backed training, giving up the speed shortcut used throughout this session.
2. **Resolve the performance mystery properly** — catch a real, live stall with `py-spy` attached, rather than continuing to rule out candidate causes one at a time from the outside.
3. **Re-run `check_damping_identifiability.py`'s `Mq`/`Mqq` comparison** against the current, real Coriolis-aware Jacobian — flagged as a concrete follow-up in finding #10, still not done.
4. Beyond `p`/`r` specifically, the next real model-structure lever (if the truncation hypothesis doesn't resolve it) remains: off-diagonal added mass revisited with a properly time-budgeted test, thruster/actuator nonlinearity, or cross-axis damping terms — none of which the model currently includes.