# Safe, self-verifying patch: actually wires item #12 (Arena task choice)
# and item #17 (identifiability report) into run_full_pipeline.py, which
# is the real product entry point a lab would use. Before this patch,
# both existed as real, tested capabilities that nothing in the actual
# demo pipeline exposed or mentioned - fixing that gap.

with open("run_full_pipeline.py", encoding="utf-8") as f:
    content = f.read()

# --- Fix 1: expose --task on the CLI, pass it through to Arena ---
old_args_end = '''    parser.add_argument("--skip-preflight", action="store_true",
                         help="Skip the automated data-quality pre-flight check (item #6 - "
                              "sampling-rate validation, Coriolis-excitation check). On by "
                              "default so real data-quality problems are caught before "
                              "committing time to a calibration that can't succeed. Skip "
                              "only if you've already verified this exact dataset recently.")
    return parser


def smoke_test_gym_environment(calibrated_params_path: str, spec_path: str) -> dict:'''

new_args_end = '''    parser.add_argument("--skip-preflight", action="store_true",
                         help="Skip the automated data-quality pre-flight check (item #6 - "
                              "sampling-rate validation, Coriolis-excitation check). On by "
                              "default so real data-quality problems are caught before "
                              "committing time to a calibration that can't succeed. Skip "
                              "only if you've already verified this exact dataset recently.")
    parser.add_argument("--task", default="waypoint",
                         choices=["waypoint", "station_keeping", "docking"],
                         help="Which Arena task the delivered Gym environment is built for "
                              "(item #12). 'waypoint' (default) - reach a target point once. "
                              "'station_keeping' - stay near the target for sustained steps, "
                              "not just touch it. 'docking' - reach the target aligned "
                              "(orientation) and slow (low speed) simultaneously.")
    return parser


def smoke_test_gym_environment(calibrated_params_path: str, spec_path: str, task: str = "waypoint") -> dict:'''

# --- Fix 2: actually pass task through to Arena() ---
old_arena_call = '''        env = Arena(params=params, vehicle=vehicle)'''
new_arena_call = '''        env = Arena(params=params, vehicle=vehicle, task=task)'''

# --- Fix 3: pass args.task at the call site ---
old_call_site = '''    gym_check = smoke_test_gym_environment(calibrated_params_path, args.spec)'''
new_call_site = '''    gym_check = smoke_test_gym_environment(calibrated_params_path, args.spec, task=args.task)'''

# --- Fix 4: signpost the identifiability report as a real, discoverable next step ---
old_done = '''    print(f"DONE in {total_elapsed:.1f}s. Summary written to {summary_path}")'''
new_done = '''    print(f"DONE in {total_elapsed:.1f}s. Summary written to {summary_path}")
    print()
    print("Optional next step (item #17, NOT run automatically - real cost ~15-17 min on")
    print("the full dataset, would slow down this demo): a deeper, automated parameter")
    print("identifiability report (condition number, correlated pairs, multi-parameter")
    print("flat directions) is available via:")
    print(f"  python automated_identifiability_report.py --data-dir {args.data_dir} \\\\")
    print(f"    --spec {args.spec} --sampling-rate-hz {args.sampling_rate_hz} \\\\")
    print(f"    --calibrated-params {calibrated_params_path}")'''

replacements = [
    ("CLI args + function signature", old_args_end, new_args_end),
    ("Arena() call", old_arena_call, new_arena_call),
    ("smoke_test call site", old_call_site, new_call_site),
    ("DONE message signposting", old_done, new_done),
]

missing = [label for label, old, new in replacements if content.count(old) != 1]
if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}.")
else:
    for label, old, new in replacements:
        content = content.replace(old, new, 1)
    with open("run_full_pipeline.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Both items wired in successfully - --task flag added, identifiability report signposted.")