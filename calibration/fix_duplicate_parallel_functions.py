with open("optimizer.py", encoding="utf-8") as f:
    content = f.read()

worker_marker = "def _parallel_restart_worker("
parallel_marker = "def calibrate_parallel("
calibrate_marker = "\ndef calibrate(\n"

n_worker = content.count(worker_marker)
n_parallel = content.count(parallel_marker)
print(f"_parallel_restart_worker occurrences: {n_worker}")
print(f"calibrate_parallel occurrences: {n_parallel}")

if n_worker != 2 or n_parallel != 2:
    print("REFUSING TO PATCH - expected exactly 2 of each, found something different.")
else:
    first_worker_idx = content.index(worker_marker)
    second_worker_idx = content.index(worker_marker, first_worker_idx + 1)
    calibrate_idx = content.index(calibrate_marker)

    if not (first_worker_idx < second_worker_idx < calibrate_idx):
        print("REFUSING TO PATCH - unexpected ordering of markers.")
    else:
        deduplicated = content[:second_worker_idx] + content[calibrate_idx:]

        old_snippet = '''    def wrapped_residual_fn(x_free, *args):
        full_x = full_template.copy()
        full_x[free_mask] = x_free
        return joint_residuals(full_x, *args)

    result = least_squares('''
        new_snippet = '''    import os as _os
    print(f"  Worker starting (PID={_os.getpid()})...", flush=True)

    def wrapped_residual_fn(x_free, *args):
        full_x = full_template.copy()
        full_x[free_mask] = x_free
        return joint_residuals(full_x, *args)

    result = least_squares('''

        if deduplicated.count(old_snippet) != 1:
            print(f"WARNING: visibility snippet not added cleanly ({deduplicated.count(old_snippet)} found) - writing dedup only.")
        else:
            deduplicated = deduplicated.replace(old_snippet, new_snippet, 1)
            print("Also added per-worker startup visibility.")

        with open("optimizer.py", "w", encoding="utf-8") as f:
            f.write(deduplicated)
        print(f"Deduplicated successfully. New file: {len(deduplicated)} chars (was {len(content)}).")
