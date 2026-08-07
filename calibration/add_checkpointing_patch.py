"""
Safe, self-verifying patch: adds real checkpointing to run_calibration.py,
matching the exact pattern already proven in run_calibration_holoocean.py
(save the BEST parameters seen so far to a file after EVERY single
evaluation, resumable via --resume-from-checkpoint). Only touches the
file if BOTH anchor blocks are found EXACTLY as expected — otherwise
refuses and reports which one failed, rather than risk corrupting a
working script (the exact class of mistake that broke this file once
already tonight).
"""
with open("run_calibration.py", encoding="utf-8") as f:
    content = f.read()

# --- Anchor 1: the CLI arg section, right after --workers ---
anchor1 = '''    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser'''

insertion1 = '''    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint-path", default=None,
                         help="Where to save the live checkpoint (overwritten after every "
                              "single evaluation, not just improving ones). Defaults to "
                              "<output-dir>/checkpoint.yaml. Added after a real, confirmed "
                              "latency spike (one evaluation took ~17 minutes vs. the normal "
                              "~9.5s) during a long real run with no prior protection.")
    parser.add_argument("--resume-from-checkpoint", default=None,
                         help="Path to a checkpoint.yaml from a previously-interrupted run — "
                              "resumes from its best-parameters-so-far instead of "
                              "vehicle_spec.yaml's naive initial guess.")
    return parser


def make_checkpoint_callback(checkpoint_path):
    """
    Same proven pattern as run_calibration_holoocean.py's checkpoint —
    saves the BEST (lowest-cost) parameter vector seen so far after EVERY
    evaluation (overwriting one file, not accumulating), so an
    interruption only ever loses the most recent single evaluation's
    progress, not the whole run. Built after run_calibration.py's first
    real 12+ hour attempt showed a genuine, unexplained latency spike
    (one evaluation ~17 minutes vs. the normal ~9.5s) with zero protection
    against losing everything to an interruption.
    """
    best = {"cost": float("inf"), "param_vec": None}

    def callback(call_count, param_vec, cost):
        if cost < best["cost"]:
            best["cost"] = cost
            best["param_vec"] = param_vec.copy()

        current_params = CalibrationParameters.from_vector(param_vec)
        best_params = (CalibrationParameters.from_vector(best["param_vec"])
                        if best["param_vec"] is not None else current_params)

        checkpoint = {
            "call_count": call_count,
            "current_cost": cost,
            "current_parameters": {n: float(getattr(current_params, n))
                                    for n in current_params.param_names},
            "best_cost_so_far": best["cost"],
            "best_parameters_so_far": {n: float(getattr(best_params, n))
                                        for n in best_params.param_names},
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            yaml.dump(checkpoint, f, sort_keys=False)

    return callback


def load_checkpoint_warm_start(checkpoint_path: str) -> np.ndarray:
    with open(checkpoint_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    params = CalibrationParameters(**data["best_parameters_so_far"])
    print(f"Resuming from checkpoint: call_count={data['call_count']}, "
          f"best_cost_so_far={data['best_cost_so_far']:.6f}")
    return params.to_vector()'''

# --- Anchor 2: the CalibrationConfig construction block in main() ---
anchor2 = '''    calibration_config = CalibrationConfig(
        loss=args.loss, n_restarts=args.n_restarts, seed=args.seed, max_nfev=args.max_nfev,
        initialization_strategy=args.initialization_strategy,
        perturbation_frac=args.perturbation_frac,
        verbose=2 if args.verbose else 0,  # scipy's own per-iteration progress
        # (cost, optimality, step norm each iteration) — previously --verbose
        # only affected Python logging.basicConfig, never scipy's own
        # verbose=, so runs gave zero visibility into whether they were
        # progressing or stuck. This is what actually answers that question.
    )'''

insertion2 = '''    checkpoint_path = args.checkpoint_path or str(Path(args.output_dir) / "checkpoint.yaml")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpointing every evaluation to: {checkpoint_path}")

    x0_override = None
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint {args.resume_from_checkpoint}...")
        x0_override = load_checkpoint_warm_start(args.resume_from_checkpoint)

    calibration_config = CalibrationConfig(
        loss=args.loss, n_restarts=args.n_restarts, seed=args.seed, max_nfev=args.max_nfev,
        initialization_strategy=args.initialization_strategy,
        perturbation_frac=args.perturbation_frac,
        verbose=2 if args.verbose else 0,  # scipy's own per-iteration progress
        # (cost, optimality, step norm each iteration) — previously --verbose
        # only affected Python logging.basicConfig, never scipy's own
        # verbose=, so runs gave zero visibility into whether they were
        # progressing or stuck. This is what actually answers that question.
        x0_override=x0_override,
        on_iteration=make_checkpoint_callback(checkpoint_path),
    )'''

missing = []
if anchor1 not in content:
    missing.append("Anchor 1 (CLI args section)")
if anchor2 not in content:
    missing.append("Anchor 2 (CalibrationConfig construction)")

if missing:
    print(f"REFUSING TO PATCH — could not find: {', '.join(missing)}. "
          f"The real file differs from what was expected. Do NOT proceed — "
          f"paste the current content of the relevant section(s) so this can be redone correctly.")
else:
    new_content = content.replace(anchor1, insertion1, 1)
    new_content = new_content.replace(anchor2, insertion2, 1)
    with open("run_calibration.py", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Both anchors found and patched successfully.")