# Safe, self-verifying patch: adds two things to run_calibration_holoocean.py,
# both anchored on the DOF-weighting insertion from the last patch (exact,
# confirmed text - that patch was already successfully applied):
#
# 1. Auto-backup: before this run touches the output directory at all,
#    back up whatever calibrated_parameters.yaml/checkpoint.yaml ALREADY
#    exist there (e.g. from a prior segment) to timestamped filenames -
#    fixing the exact class of loss that happened to segment 3's result.
# 2. A timing log (timing_log.csv, one line per real evaluation with a
#    real timestamp) - the same capability already built for
#    run_calibration.py, now added here too, since THIS is the backend
#    actually being used, and the performance mystery just resurfaced
#    again in it (649s/eval vs the confirmed 124.2s baseline).

with open("run_calibration_holoocean.py", encoding="utf-8") as f:
    content = f.read()

anchor = '''    # Custom per-DOF weighting - NOT standard scale-normalization.
    # Researched and confirmed (PROJECT_STATUS.md): p is genuinely
    # under-weighted by its small natural scale (0.125, smallest of all
    # 6 DOFs) - standard normalization helps it correctly. But r has the
    # LARGEST natural scale (0.5625) and still regressed across multiple
    # real segments - standard normalization would REDUCE r's weight
    # below even today's unweighted baseline (confirmed: 0.67x vs today's
    # 1.5x), making its regression worse, not better. This applies a
    # deliberate boost to BOTH p and r beyond their natural scales.
    from objective import compute_dof_scales as _compute_dof_scales
    _dof_labels = ["u", "v", "w", "p", "q", "r"]
    dof_scales = _compute_dof_scales(train_runs).copy()
    dof_scales[_dof_labels.index("p")] *= 0.3   # p: strongly boosted - confirmed under-weighted
    dof_scales[_dof_labels.index("r")] *= 0.15  # r: boosted BEYOND its already-large natural scale
    print(f"Custom DOF weighting active - p and r boosted beyond natural scale "
          f"(see PROJECT_STATUS.md for the research behind this): {dict(zip(_dof_labels, dof_scales))}")

    config = CalibrationConfig(
        n_restarts=1,
        seed=42,
        max_nfev=args.max_nfev,
        x0_override=x0_warm_start,
        initialization_strategy="fixed",  # start EXACTLY at the warm-start point, no perturbation
        on_iteration=make_checkpoint_callback(checkpoint_path),
        simulate_fn=simulate_fn,
        dof_scales=dof_scales,
        verbose=2,
    )'''

insertion = '''    # --- Auto-backup: protect whatever already exists in the output dir
    # from a PRIOR segment before this run touches it at all. Fixes the
    # exact class of loss that happened to segment 3's real result
    # (silently overwritten in place, no versioning, nothing recoverable
    # afterward). Cheap - a plain file copy, only runs if something is
    # already there to protect.
    from pathlib import Path as _Path
    import shutil as _shutil
    import time as _time
    _output_dir = _Path(checkpoint_path).parent
    _backup_tag = _time.strftime("%Y%m%d_%H%M%S")
    for _fname in ("calibrated_parameters.yaml", "checkpoint.yaml"):
        _existing = _output_dir / _fname
        if _existing.exists():
            _backup_path = _output_dir / f"{_existing.stem}_backup_{_backup_tag}{_existing.suffix}"
            _shutil.copy2(_existing, _backup_path)
            print(f"Auto-backed up existing {_fname} -> {_backup_path.name} before this run starts")

    # --- Timing log: one line per real evaluation with a real timestamp -
    # same capability already built for run_calibration.py's stand-in
    # backend, now added here too, since the performance mystery just
    # resurfaced again in THIS (HoloOcean) backend specifically
    # (confirmed: 649s/eval vs the established 124.2s baseline, this
    # session). Wraps the existing checkpoint callback rather than
    # modifying it, so the proven checkpointing behavior is untouched.
    _timing_log_path = _output_dir / "timing_log_holoocean.csv"
    _timing_log_is_new = not _timing_log_path.exists()

    def _with_timing_log(inner_callback):
        def wrapped(call_count, param_vec, cost):
            inner_callback(call_count, param_vec, cost)
            with open(_timing_log_path, "a", encoding="utf-8") as f:
                if _timing_log_is_new and call_count == 1:
                    f.write("call_count,unix_timestamp,cost\\n")
                f.write(f"{call_count},{_time.time()},{cost}\\n")
        return wrapped

    # Custom per-DOF weighting - NOT standard scale-normalization.
    # Researched and confirmed (PROJECT_STATUS.md): p is genuinely
    # under-weighted by its small natural scale (0.125, smallest of all
    # 6 DOFs) - standard normalization helps it correctly. But r has the
    # LARGEST natural scale (0.5625) and still regressed across multiple
    # real segments - standard normalization would REDUCE r's weight
    # below even today's unweighted baseline (confirmed: 0.67x vs today's
    # 1.5x), making its regression worse, not better. This applies a
    # deliberate boost to BOTH p and r beyond their natural scales.
    from objective import compute_dof_scales as _compute_dof_scales
    _dof_labels = ["u", "v", "w", "p", "q", "r"]
    dof_scales = _compute_dof_scales(train_runs).copy()
    dof_scales[_dof_labels.index("p")] *= 0.3   # p: strongly boosted - confirmed under-weighted
    dof_scales[_dof_labels.index("r")] *= 0.15  # r: boosted BEYOND its already-large natural scale
    print(f"Custom DOF weighting active - p and r boosted beyond natural scale "
          f"(see PROJECT_STATUS.md for the research behind this): {dict(zip(_dof_labels, dof_scales))}")

    config = CalibrationConfig(
        n_restarts=1,
        seed=42,
        max_nfev=args.max_nfev,
        x0_override=x0_warm_start,
        initialization_strategy="fixed",  # start EXACTLY at the warm-start point, no perturbation
        on_iteration=_with_timing_log(make_checkpoint_callback(checkpoint_path)),
        simulate_fn=simulate_fn,
        dof_scales=dof_scales,
        verbose=2,
    )'''

count = content.count(anchor)
if count != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence, found {count}.")
else:
    content = content.replace(anchor, insertion, 1)
    with open("run_calibration_holoocean.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Auto-backup and timing-log patch applied successfully.")