with open("preflight_check.py", encoding="utf-8") as f:
    content = f.read()

block = '''    results["schema"] = run_check(
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
    )
'''

count = content.count(block)
print("Occurrences of the block:", count)

if count != 2:
    print("REFUSING TO PATCH - expected exactly 2, found something different.")
else:
    first_idx = content.index(block)
    second_idx = content.index(block, first_idx + len(block))
    deduplicated = content[:second_idx] + content[second_idx + len(block):]

    with open("preflight_check.py", "w", encoding="utf-8") as f:
        f.write(deduplicated)
    print("Deduplicated successfully.")
