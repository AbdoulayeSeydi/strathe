with open("preflight_check.py", encoding="utf-8") as f:
    content = f.read()

old = '''    results["coriolis_excitation"] = run_check(
        "Coriolis-relevant velocity product excitation",
        [sys.executable, "check_coriolis_excitation.py",
         "--data-dir", args.data_dir, "--sampling-rate-hz", str(args.sampling_rate_hz)],
    )'''

new = '''    results["coriolis_excitation"] = run_check(
        "Coriolis-relevant velocity product excitation",
        [sys.executable, "check_coriolis_excitation.py",
         "--data-dir", args.data_dir, "--sampling-rate-hz", str(args.sampling_rate_hz)],
    )
    results["schema"] = run_check(
        "Schema validation (required columns)",
        [sys.executable, "check_schema.py", "--data-dir", args.data_dir],
    )
    results["data_integrity"] = run_check(
        "Structural data integrity (empty files, malformed rows, NaN/Inf)",
        [sys.executable, "check_data_integrity.py", "--data-dir", args.data_dir],
    )
    results["physical_plausibility"] = run_check(
        "Physical/units plausibility",
        [sys.executable, "check_physical_plausibility.py",
         "--data-dir", args.data_dir, "--sampling-rate-hz", str(args.sampling_rate_hz)],
    )'''

count = content.count(old)
print("Occurrences:", count)
if count == 1:
    content = content.replace(old, new, 1)
    with open("preflight_check.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("All three new checks wired into preflight_check.py successfully.")
else:
    print("REFUSING - count mismatch, do not proceed blindly.")
