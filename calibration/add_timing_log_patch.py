# Safe, self-verifying patch: extends run_calibration.py's checkpoint
# callback to ALSO append every single evaluation's timestamp to a
# separate log file. Prepared groundwork for item #3 (the performance
# mystery), ready to use next time a stand-in run needs profiling.
# Uses an inline deferred import for time - avoids touching the imports
# section, which would be a second, unverified edit.

with open("run_calibration.py", encoding="utf-8") as f:
    content = f.read()

anchor = '''        checkpoint = {
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

    return callback'''

insertion = '''        checkpoint = {
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

        # Timing log for the still-unexplained performance mystery
        # (PROJECT_STATUS.md) - appends ONE line per real evaluation with
        # a real timestamp, so a future long stand-in run gives a genuine
        # post-hoc distribution of per-call timing instead of another
        # guess. Separate from checkpoint.yaml so it never risks
        # corrupting the resumable state.
        timing_log_path = str(Path(checkpoint_path).with_name("timing_log.csv"))
        is_new = not Path(timing_log_path).exists()
        with open(timing_log_path, "a", encoding="utf-8") as f:
            if is_new:
                f.write("call_count,unix_timestamp,cost\\n")
            f.write(f"{call_count},{__import__('time').time()},{cost}\\n")

    return callback'''

count = content.count(anchor)
if count != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence, found {count}.")
else:
    content = content.replace(anchor, insertion, 1)
    with open("run_calibration.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Timing log capability added successfully (not run - just prepared for later).")