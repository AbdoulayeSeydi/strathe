# Safe, self-verifying patch: fixes run_full_pipeline.py's held-out split
# to match Phase 1's actual established split (16-02-2024, 19-02-2024 -
# the exact same dates every other script in this project uses) by
# default, instead of an auto-30%-of-4-dates fraction that only held out
# 1 date (6 runs) - making this demo's numbers genuinely comparable to
# the rest of the project, not just "a demo that happens to run."
#
# Added as a --heldout-dates CLI flag (matching run_calibration.py's own
# pattern) rather than hardcoded, so this still works correctly if this
# demo is ever pointed at a different dataset later.

with open("run_full_pipeline.py", encoding="utf-8") as f:
    content = f.read()

old_arg = '''    parser.add_argument("--task", default="waypoint",
                         choices=["waypoint", "station_keeping", "docking"],
                         help="Which Arena task the delivered Gym environment is built for "
                              "(item #12). 'waypoint' (default) - reach a target point once. "
                              "'station_keeping' - stay near the target for sustained steps, "
                              "not just touch it. 'docking' - reach the target aligned "
                              "(orientation) and slow (low speed) simultaneously.")
    return parser'''

new_arg = '''    parser.add_argument("--task", default="waypoint",
                         choices=["waypoint", "station_keeping", "docking"],
                         help="Which Arena task the delivered Gym environment is built for "
                              "(item #12). 'waypoint' (default) - reach a target point once. "
                              "'station_keeping' - stay near the target for sustained steps, "
                              "not just touch it. 'docking' - reach the target aligned "
                              "(orientation) and slow (low speed) simultaneously.")
    parser.add_argument("--heldout-dates", nargs="*",
                         default=["16-02-2024", "19-02-2024"],
                         help="Which collection dates to hold out for validation. Defaults "
                              "to Phase 1's EXACT established split (PROJECT_STATUS.md) - "
                              "the same dates every other script in this project uses - so "
                              "this demo's numbers are genuinely comparable, not just "
                              "'a demo that happens to run.' Override if pointing this at a "
                              "genuinely different dataset.")
    return parser'''

old_split_call = '''    split = split_by_date(csv_paths, heldout_fraction=0.3)'''
new_split_call = '''    split = split_by_date(csv_paths, heldout_dates=args.heldout_dates)'''

missing = []
if content.count(old_arg) != 1:
    missing.append("--task arg block")
if content.count(old_split_call) != 1:
    missing.append("split_by_date call")

if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}.")
else:
    content = content.replace(old_arg, new_arg, 1)
    content = content.replace(old_split_call, new_split_call, 1)
    with open("run_full_pipeline.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Held-out split fix applied successfully - now matches Phase 1's exact dates by default.")