with open("preflight_check.py", encoding="utf-8") as f:
    content = f.read()

replacements = []

# 1. Add --heldout-dates to preflight_check.py's own argparse - optional,
# default None, so behavior against the standard full dataset is UNCHANGED
# unless explicitly overridden (needed for smaller/different datasets,
# where check_coriolis_excitation.py's own internal default dates don't exist).
old1 = '''    parser.add_argument("--rate-candidates", type=float, nargs="+", default=[5, 10, 20, 25, 50, 100],
                         help="Candidate sampling rates (Hz) to test on a NEW, unknown dataset — "
                              "the point of this check is determining which one is actually "
                              "correct, not assuming one. Override if you have reason to expect "
                              "a specific rate not in this default list.")
    args = parser.parse_args()'''
new1 = '''    parser.add_argument("--rate-candidates", type=float, nargs="+", default=[5, 10, 20, 25, 50, 100],
                         help="Candidate sampling rates (Hz) to test on a NEW, unknown dataset — "
                              "the point of this check is determining which one is actually "
                              "correct, not assuming one. Override if you have reason to expect "
                              "a specific rate not in this default list.")
    parser.add_argument("--heldout-dates", nargs="*", default=None,
                         help="Forwarded to check_coriolis_excitation.py's own --heldout-dates. "
                              "A real gap found and fixed: without this, that check always fell "
                              "back to ITS OWN internal default dates, which don't exist in a "
                              "smaller/different dataset - producing a confusing, unrelated-"
                              "looking failure. Only forwarded when explicitly given, so behavior "
                              "against the standard full dataset stays exactly unchanged by default.")
    args = parser.parse_args()'''
replacements.append(("add --heldout-dates arg", old1, new1))

# 2. Forward it to the subprocess call ONLY when explicitly provided
old2 = '''    results["coriolis_excitation"] = run_check(
        "Coriolis-relevant velocity product excitation",
        [sys.executable, "check_coriolis_excitation.py",
         "--data-dir", args.data_dir, "--sampling-rate-hz", str(args.sampling_rate_hz)],
    )'''
new2 = '''    _coriolis_cmd = [sys.executable, "check_coriolis_excitation.py",
                      "--data-dir", args.data_dir, "--sampling-rate-hz", str(args.sampling_rate_hz)]
    if args.heldout_dates:
        _coriolis_cmd += ["--heldout-dates", *args.heldout_dates]
    results["coriolis_excitation"] = run_check(
        "Coriolis-relevant velocity product excitation",
        _coriolis_cmd,
    )'''
replacements.append(("forward heldout-dates to subprocess", old2, new2))

missing = [label for label, old, new in replacements if content.count(old) != 1]
if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}")
    for label, old, new in replacements:
        print(f"  {label}: {content.count(old)} occurrence(s)")
else:
    for label, old, new in replacements:
        content = content.replace(old, new, 1)
    with open("preflight_check.py", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"All {len(replacements)} changes applied successfully.")