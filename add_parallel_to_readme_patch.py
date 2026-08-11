with open("README.md", encoding="utf-8") as f:
    content = f.read()

old = '''2. **Stand-in calibration** (fast, minutes):
`````bash
   python run_calibration.py --data-dir <path> --spec vehicle_spec.yaml \\
       --sampling-rate-hz 10.0 --heldout-dates 16-02-2024 19-02-2024 \\
       --output-dir <output-dir> --max-nfev 60
````'''

new = '''2. **Stand-in calibration** (fast, minutes):
```bash
   python run_calibration.py --data-dir <path> --spec vehicle_spec.yaml \\
       --sampling-rate-hz 10.0 --heldout-dates 16-02-2024 19-02-2024 \\
       --output-dir <output-dir> --max-nfev 60
```
   Add `--parallel` (optionally `--max-workers N`) to run multiple restarts
   across real CPU cores instead of sequentially - confirmed real speedup
   (roughly 4 restarts in the time 1 would normally take) on real, smaller
   data. **Honest caveat**: checkpointing is disabled in this mode (a real,
   deliberate tradeoff - see `PROJECT_STATUS.md` finding #34), and the
   combination of `--parallel` with the FULL, untruncated dataset has not
   yet been confirmed to finish in a reasonable time - recommend testing on
   a smaller subset first, or preferring the default (sequential) mode for
   full-dataset runs until that's been verified.'''

count = content.count(old)
print(f"Occurrences: {count}")
if count == 1:
    content = content.replace(old, new, 1)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("--parallel documented in README.md successfully.")
else:
    print("REFUSING - count mismatch, do not proceed blindly.")
