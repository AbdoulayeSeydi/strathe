# Safe, self-verifying patch: adds the REAL (non-demo) workflow section
# to README.md - this is a REDO of an earlier patch that was never
# actually applied (failed with file-not-found, never retried - a real
# gap only caught now). Includes diagnose_calibration.py --plot as a
# documented step this time too (item 4's real remaining gap).

with open("README.md", encoding="utf-8") as f:
    content = f.read()

anchor = "## What's in this repo"

real_workflow_section = '''## The real (non-demo) workflow

`run_full_pipeline.py` (above) is a fast, truncated DEMO - good for
showing the mechanism works end-to-end, not for producing the final,
reportable numbers. The real pipeline, run against the full dataset and
real HoloOcean physics, is a sequence of separate scripts, run in order:

1. **Pre-flight data-quality check** (always run this first):
```bash
   cd calibration
   python preflight_check.py --data-dir <path-to-logs> --spec vehicle_spec.yaml
```
   Validates sampling rate and Coriolis-relevant excitation on the REAL,
   full dataset before committing real compute time to a calibration that
   might not succeed.

2. **Stand-in calibration** (fast, minutes):
```bash
   python run_calibration.py --data-dir <path> --spec vehicle_spec.yaml \\
       --sampling-rate-hz 10.0 --heldout-dates 16-02-2024 19-02-2024 \\
       --output-dir <output-dir> --max-nfev 60
```

3. **HoloOcean refinement** (slow, hours - real, confirmed cost ~124s per
   evaluation on this dataset; checkpointed and resumable):
```bash
   python run_calibration_holoocean.py --data-dir <path> --spec vehicle_spec.yaml \\
       --sampling-rate-hz 10.0 --resume-from-checkpoint <path>/checkpoint.yaml \\
       --n-steps-per-run 150 --max-nfev 15 --output-dir <output-dir>
```

4. **Real validation (Claim A)** against real HoloOcean physics:
```bash
   python replay_comparison.py --data-dir <path> --spec vehicle_spec.yaml \\
       --sampling-rate-hz 10.0 --calibrated-params <output-dir>/calibrated_parameters.yaml \\
       --result-type final --output-dir <output-dir>
```

5. **Real multi-seed PPO training (Claim B)** - optionally pick a task
   (`waypoint` default, `station_keeping`, or `docking`):
```bash
   python train_multiseed.py --data-dir <path> --spec vehicle_spec.yaml \\
       --calibrated-params <output-dir>/calibrated_parameters.yaml \\
       --result-type final --n-seeds 4 --task waypoint --output-dir <output-dir>
```

6. **Optional - deeper diagnostics**, once you have a real calibrated
   result:
   - **Per-DOF real-vs-simulated plots and force-breakdown charts** (fast
     - a few minutes, no Jacobian evaluation):
```bash
     python diagnose_calibration.py --data-dir <path> --spec vehicle_spec.yaml \\
         --calibrated-params <output-dir>/calibrated_parameters.yaml \\
         --sampling-rate-hz 10.0 --heldout-dates 16-02-2024 19-02-2024 \\
         --skip-jacobian --plot --plot-output-dir <output-dir>/plots
```
     Shows exactly WHERE in a trajectory the fit breaks down, per DOF -
     not just the aggregate RMSE number.
   - **Automated parameter identifiability report** (real cost ~15-17 min
     on the full dataset - condition number, correlated pairs,
     multi-parameter flat directions):
```bash
     python automated_identifiability_report.py --data-dir <path> --spec vehicle_spec.yaml \\
         --sampling-rate-hz 10.0 --calibrated-params <output-dir>/calibrated_parameters.yaml
```

See `PROJECT_STATUS.md` for the full history of real runs, real bugs
found and fixed along the way, and the honest per-DOF results.

'''

occurrences = content.count(anchor)
if occurrences != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence of '{anchor}', found {occurrences}.")
else:
    new_content = content.replace(anchor, real_workflow_section + anchor, 1)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Real workflow section (including diagnostic plots) added to README.md successfully.")
