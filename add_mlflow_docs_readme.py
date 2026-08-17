with open("README.md", encoding="utf-8") as f:
    content = f.read()

anchor = "## Contact"

new_section = '''## Experiment Tracking (MLflow)

Every real calibration run - `run_calibration.py`, `run_calibration_holoocean.py`,
and `run_full_pipeline.py` - automatically logs to a local MLflow experiment
(`marine_sim_calibration`). This is purely observational: if MLflow is
missing or logging fails for any reason, the calibration itself is
completely unaffected - you'll see a warning printed, never a crash.

### Viewing logged runs

```bash
cd calibration
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the URL it prints (default `http://127.0.0.1:5000`) in a
browser. **Always pass `--backend-store-uri sqlite:///mlflow.db`
explicitly** - a real issue found while building this: `mlflow ui` and
the Python client library can each independently fall back to a
*different* default storage location if neither is told explicitly,
silently showing an empty UI even though real data exists. This project
always logs to `sqlite:///mlflow.db` (a local file, `calibration/mlflow.db`
- not tracked in git, regenerated automatically).

### What's tracked

- **Parameters**: which vehicle spec and dataset were used (including a
  dataset "signature" - file count, total size, modification time - an
  *identifier*, not a cryptographic guarantee of uniqueness), the
  optimizer and its settings, sampling rate, and (where applicable)
  algorithm/seed/task, plus real, live-queried package versions
  (numpy, scipy, stable-baselines3, mlflow itself) for reproducibility.
- **Metrics**: RMSE results (aggregate, and per-DOF where the script
  computes them without extra real cost), runtime, and evaluation count.
- **Status**: every run is tagged with one of four honest outcomes -
  `SUCCESS` (the optimizer genuinely converged), `CHECKPOINT_FALLBACK`
  (didn't converge, but a real, usable checkpoint result was
  intentionally substituted instead), `FAILED` (no usable result at
  all), or `INTERRUPTED` - plus a `status_reason` with the real, specific
  detail (e.g. the actual scipy convergence message).
- **Artifacts**: the real `vehicle_spec.yaml` used, the saved calibrated
  parameters, and the saved metrics/report file.

### What's deliberately NOT tracked

Raw trajectory data is never logged - it's large, already lives in your
own dataset directory, and would bloat the tracking store for no real
benefit. Only the dataset's identity (path, file count, size, mtime) is
recorded, so you can tell *which* dataset a run used without duplicating
the data itself.

'''

occurrences = content.count(anchor)
if occurrences != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence of anchor, found {occurrences}.")
else:
    new_content = content.replace(anchor, new_section + anchor, 1)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("MLflow documentation section inserted successfully.")