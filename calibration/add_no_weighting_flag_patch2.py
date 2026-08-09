with open("run_calibration_holoocean.py", encoding="utf-8") as f:
    content = f.read()

old_weighting = '''    from objective import compute_dof_scales as _compute_dof_scales
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

new_weighting = '''    dof_scales = None
    if not args.no_dof_weighting:
        from objective import compute_dof_scales as _compute_dof_scales
        _dof_labels = ["u", "v", "w", "p", "q", "r"]
        dof_scales = _compute_dof_scales(train_runs).copy()
        dof_scales[_dof_labels.index("p")] *= 0.3
        dof_scales[_dof_labels.index("r")] *= 0.15
        print(f"Custom DOF weighting active - p and r boosted beyond natural scale "
              f"(see PROJECT_STATUS.md for the research behind this): {dict(zip(_dof_labels, dof_scales))}")
    else:
        print("DOF weighting DISABLED (--no-dof-weighting) - standard unweighted objective, "
              "isolating other changes from the separately-tested weighting approach "
              "(PROJECT_STATUS.md finding #27).")
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

old_arg = '''    parser.add_argument("--output-dir", default="../calibration_output_holoocean_poc")
    args = parser.parse_args()'''

new_arg = '''    parser.add_argument("--output-dir", default="../calibration_output_holoocean_poc")
    parser.add_argument("--no-dof-weighting", action="store_true",
                         help="Disable the p/r DOF weighting (finding #27) for a clean, "
                              "isolated test of other changes.")
    args = parser.parse_args()'''

missing = []
if content.count(old_weighting) != 1:
    missing.append("DOF weighting block")
if content.count(old_arg) != 1:
    missing.append("CLI args block")

if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}.")
else:
    content = content.replace(old_weighting, new_weighting, 1)
    content = content.replace(old_arg, new_arg, 1)
    with open("run_calibration_holoocean.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Both changes applied successfully.")
