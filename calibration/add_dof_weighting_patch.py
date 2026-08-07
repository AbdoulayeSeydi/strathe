# Safe, self-verifying patch: wires the researched, custom DOF weighting
# (boosts p and r beyond their natural scales - NOT standard
# scale-normalization, which would help p but make r's already-observed
# regression worse) into run_calibration_holoocean.py's CalibrationConfig.
# Computes the weights INLINE from train_runs (already loaded in this
# script). Uses a LOCAL import for compute_dof_scales, deliberately
# avoiding any edit to the file's existing top-level import section
# (unverified exact format - a real mistake earlier tonight came from
# blindly touching an imports line).

with open("run_calibration_holoocean.py", encoding="utf-8") as f:
    content = f.read()

anchor = '''    config = CalibrationConfig(
        n_restarts=1,
        seed=42,
        max_nfev=args.max_nfev,
        x0_override=x0_warm_start,
        initialization_strategy="fixed",  # start EXACTLY at the warm-start point, no perturbation
        on_iteration=make_checkpoint_callback(checkpoint_path),
        simulate_fn=simulate_fn,
        verbose=2,
    )'''

insertion = '''    # Custom per-DOF weighting - NOT standard scale-normalization.
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

count = content.count(anchor)
if count != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence, found {count}.")
else:
    content = content.replace(anchor, insertion, 1)
    with open("run_calibration_holoocean.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("DOF weighting patch applied successfully.")