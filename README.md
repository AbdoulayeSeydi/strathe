   # Calibrated Digital Twin for BlueROV2 (Marine Autonomy)

   A system-identification pipeline that fits a physics model to real BlueROV2
   mission logs, integrated into [HoloOcean](https://byu-holoocean.github.io/holoocean-docs/v2.0.0/index.html)
   for high-fidelity underwater simulation — plus a Gym-compatible RL training
   environment built on top of the calibrated physics.

   **What this answers:** does calibrating a simulator against real logged
   vehicle data measurably improve how closely it tracks real behavior,
   compared to an uncalibrated default model?

   **Answer so far:** yes, on most axes. Real, held-out-by-date validation
   (never-before-seen collection sessions, not just held-out samples) shows a
   genuine accuracy improvement — see [Results](#results) below for the honest,
   full breakdown, including where it's still a work in progress.

   ## Setup

   ### 1. Python dependencies

   ```bash
   pip install -r requirements.txt
   ```

   ### 2. HoloOcean (the underlying simulator)

   HoloOcean can't be distributed via `pip`/PyPI directly, due to the Unreal
   Engine EULA — it requires a one-time account-linking step first:

   1. **Link your GitHub account to an Epic Games account** and accept the
      Unreal Engine EULA — follow
      [Epic's official steps](https://www.unrealengine.com/en-US/ue-on-github).
      **Accept the email invitation from Unreal Engine** — easy to miss, and
      the next step fails silently without it.
   2. **Clone and install the HoloOcean client library:**
      ```bash
      git clone git@github.com:byu-holoocean/HoloOcean.git holoocean
      cd holoocean/client
      pip install .
      ```
      (If cloning gives a "Page Not Found" error, step 1's account link hasn't
      gone through yet.)
   3. **Install the Ocean world package** (several GB, one-time download):
      ```bash
      python -c "import holoocean; holoocean.install('Ocean')"
      ```

   Full official docs:
   [HoloOcean Installation Guide](https://byu-holoocean.github.io/holoocean-docs/v2.0.0/usage/installation.html).

   ### 3. Use the calibrated model directly

   ```bash
   python -c "
   from calibration.dynamics_model import CalibrationParameters
   import yaml

   with open('calibration_output_holoocean_coriolis/calibrated_parameters.yaml') as f:
      params = CalibrationParameters(**yaml.safe_load(f)['parameters'])
   print(params)
   "
   ```

   *(Confirmed working as-written, run from the project root.)*

   ### 4. Try the fast demo pipeline

   A single, watchable command — pre-flight data-quality checks, a fast
   calibration fit, and a real Gym-environment smoke test, end to end:

   ```bash
   cd calibration
   python run_full_pipeline.py \
      --spec vehicle_spec.yaml \
      --data-dir <path-to-logs> \
      --output-dir <output-dir> \
      --sampling-rate-hz 10.0 \
      --max-train-runs 3 \
      --max-nfev 5
   ```

   Optionally pick a task for the delivered environment with `--task waypoint`
   (default), `--task station_keeping`, `--task docking`, or
   `--task obstacle_avoidance` (this last one has a larger, 15-dim
   observation space rather than 11 - see PROJECT_STATUS.md finding 36).
   This is a fast, truncated demo for confirming the mechanism works end
   to end — not the source of the reportable numbers below. See the real
   workflow below for that.

   ## The real (non-demo) workflow

   The pipeline above is a fast, truncated demo. The real pipeline, run
   against the full dataset and real HoloOcean physics, is a sequence of
   separate scripts, run in order:

   1. **Pre-flight data-quality check** (always run this first):
      ```bash
      cd calibration
      python preflight_check.py --data-dir <path-to-logs> --spec vehicle_spec.yaml
      ```
      Validates sampling rate, Coriolis-relevant excitation, schema
      (required columns), structural data integrity (empty/malformed files,
      NaN/Inf), and physical/units plausibility (e.g. catches a
      degrees-vs-radians unit mistake directly) on the REAL, full dataset
      before committing real compute time to a calibration that might not
      succeed.

   2. **Stand-in calibration** (fast, minutes):
      ```bash
      python run_calibration.py --data-dir <path> --spec vehicle_spec.yaml \
         --sampling-rate-hz 10.0 --heldout-dates 16-02-2024 19-02-2024 \
         --output-dir <output-dir> --max-nfev 60
      ```

      Add --parallel (optionally --max-workers N) to the stand-in step above to run multiple restarts across real CPU cores instead of sequentially - confirmed real speedup (roughly 4 restarts in the time 1 would normally take) on real, smaller data. Honest caveat: checkpointing is disabled in this mode (a real, deliberate tradeoff - see PROJECT_STATUS.md finding 34), and the combination of --parallel with the FULL, untruncated dataset has not yet been confirmed to finish in a reasonable time - recommend testing on a smaller subset first, or preferring the default (sequential) mode for full-dataset runs until that has been verified.

   3. **HoloOcean refinement** (slow, hours — real, confirmed cost ~124s per
      evaluation on this dataset; checkpointed and resumable):
      ```bash
      python run_calibration_holoocean.py --data-dir <path> --spec vehicle_spec.yaml \
         --sampling-rate-hz 10.0 --resume-from-checkpoint <path>/checkpoint.yaml \
         --n-steps-per-run 150 --max-nfev 15 --output-dir <output-dir>
      ```
      If resuming from a checkpoint created before a `vehicle_spec.yaml` bounds
      change, first run `check_and_fix_checkpoint_bounds.py` against it — an
      out-of-date checkpoint can hold values now outside the current bounds,
      which will crash the resume. Add `--no-dof-weighting` to disable the
      custom per-DOF weighting described in the Results section below, for a
      standard, unweighted objective.

   4. **Real validation (Claim A)** against real HoloOcean physics:
      ```bash
      python replay_comparison.py --data-dir <path> --spec vehicle_spec.yaml \
         --sampling-rate-hz 10.0 --calibrated-params <output-dir>/calibrated_parameters.yaml \
         --result-type final --output-dir <output-dir>
      ```

   5. **Real multi-seed PPO training (Claim B)** — optionally pick a task
      (`waypoint` default, `station_keeping`, `docking`, or
      `obstacle_avoidance` - note: obstacle_avoidance generates its own
      randomized goal each episode and will print a warning that the
      real-data-grounded waypoint below is computed but not used for
      this task):
      ```bash
      python train_multiseed.py --data-dir <path> --spec vehicle_spec.yaml \
         --calibrated-params <output-dir>/calibrated_parameters.yaml \
         --result-type final --n-seeds 4 --task waypoint --algorithm ppo --output-dir <output-dir>
      ```

   6. **Optional — deeper diagnostics**, once you have a real calibrated
      result:
      - **Per-DOF real-vs-simulated plots and force-breakdown charts** (fast
      — a few minutes, no Jacobian evaluation):
      ```bash
      python diagnose_calibration.py --data-dir <path> --spec vehicle_spec.yaml \
            --calibrated-params <output-dir>/calibrated_parameters.yaml \
            --sampling-rate-hz 10.0 --heldout-dates 16-02-2024 19-02-2024 \
            --skip-jacobian --plot --plot-output-dir <output-dir>/plots
      ```
      Shows exactly WHERE in a trajectory the fit breaks down, per DOF —
      not just the aggregate RMSE number.
      - **Automated parameter identifiability report** (real cost ~15-17 min
      on the full dataset — condition number, correlated pairs,
      multi-parameter flat directions):
      ```bash
      python automated_identifiability_report.py --data-dir <path> --spec vehicle_spec.yaml \
            --sampling-rate-hz 10.0 --calibrated-params <output-dir>/calibrated_parameters.yaml
      ```

   See `PROJECT_STATUS.md` for the full history of real runs, real bugs found
   and fixed along the way, and the honest per-DOF results.

   ## What's in this repo

   - **`calibration/`** — the full calibration pipeline: data loading, the
   physics model, the optimizer, the HoloOcean integration, the Gym/`Arena`
   RL training environment (four tasks: `waypoint`, `station_keeping`,
   `docking`, `obstacle_avoidance`), three RL training algorithms
   (`--algorithm ppo` (default), `sac`, or `td3` - all real, rigorously
   validated against the same locked criteria, see PROJECT_STATUS.md
   finding 37), diagnostic and reporting tools, and a real, automated
   test suite (`calibration/tests/`, CI-verified on every push).
   - **`calibration_output_*/`** — saved, ready-to-use calibrated parameter
   sets (`calibrated_parameters.yaml` + `metrics.json` in each).
   - **`PROJECT_STATUS.md`** — the full research log: every finding, every bug
   found and fixed, every validation check, in chronological order. This is
   the detailed record for anyone who wants to verify the methodology or
   understand exactly how a specific number was produced. Not required
   reading to just use the model.

   ## Results

   **Methodology:** real BlueROV2 mission logs (public `oceansystemslab`
   dataset), held out by entire collection *date* (not a random split) — so
   validation checks generalization to entire unseen sessions, not just
   unseen samples. Real logged thruster commands replayed open-loop through
   both default and calibrated physics, evaluated inside real HoloOcean
   simulation, compared against real logged trajectories.

   **Headline: ~3–8% scale-normalized RMSE improvement** over uncalibrated
   default physics, on held-out real data (varies by calibration segment —
   see `PROJECT_STATUS.md` for the full history).

   Per-axis breakdown (current best real result):

   | Axis | Improvement |
   |---|---|
   | Surge (u) | +61–63% |
   | Sway (v) | +29–30% |
   | Heave (w) | +27–35% |
   | Roll rate (p) | **known limitation — see below** |
   | Pitch rate (q) | +25–35% |
   | Yaw rate (r) | **known limitation — see below** |

   We're being upfront about roll and yaw rate — they're real, honest results,
   not polished ones. Real, direct evidence rules out a data-coverage
   explanation (the real dataset excites every relevant coupling term
   comparably well). Two specific, well-motivated fixes have since been tested
   directly and refuted with real evidence: a custom per-DOF objective
   weighting, and a fix for a separately-confirmed non-physical parameter
   drift that fed directly into yaw's Coriolis coupling term. Neither helped —
   in fact both isolated tests showed `r` getting **worse**, not better,
   including one test where the specific coupling term this hypothesis was
   about changed by close to the largest amount that could be tested.

   The current leading hypothesis is that every real HoloOcean-backed
   refinement so far has trained on the same short (150-step, ~15 second)
   truncated window of each real run — a deliberate speed shortcut whose
   tradeoff was never revisited. Testing this properly means giving up that
   shortcut for a full-length run, a real, larger compute commitment not yet
   made. See `PROJECT_STATUS.md` findings #20–32 for the complete technical
   investigation, including the two ruled-out hypotheses.

   ## Status

   - ✅ Real-data calibration pipeline, validated (held-out-by-date, multi-restart stability)
   - ✅ HoloOcean integration, frame conventions confirmed
   - ✅ Gym-compatible RL environment (`Arena`), three tasks, PPO training confirmed working
   - ✅ Automated pre-flight data-quality checks, automated identifiability reporting
   - ✅ Real, automated regression test suite (26 tests), CI-verified on every push
   - 🔄 Rotational-axis accuracy (roll, yaw rate) — two hypotheses tested and ruled out; the training-window-truncation hypothesis is the clear next step
   - ⏳ Real hardware validation — not yet done; looking for lab partners with BlueROV2 access

   ## Contact

   Abdoul Seydi — abdoulayeaseydi@gmail.com

   Looking for feedback, collaboration, or lab partners with real BlueROV2
   (or similar hovering-AUV) access for hardware validation.