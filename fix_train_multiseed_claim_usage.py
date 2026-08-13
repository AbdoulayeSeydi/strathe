with open("calibration/train_multiseed.py", encoding="utf-8") as f:
    content = f.read()

replacements = []

old1 = '''    print("\\n" + "=" * 70)
    print(Claim.B_TRAINED_POLICY_BEHAVIOR)
    print("=" * 70)'''
new1 = '''    print("\\n" + "=" * 70)
    print(Claim.b_trained_policy_behavior(args.algorithm))
    print("=" * 70)'''
replacements.append(("printed header", old1, new1))

old2 = '''    report = {
        "claim": Claim.B_TRAINED_POLICY_BEHAVIOR,
        "algorithm": args.algorithm,
        "result_type": args.result_type,'''
new2 = '''    report = {
        "claim": Claim.b_trained_policy_behavior(args.algorithm),
        "algorithm": args.algorithm,
        "result_type": args.result_type,'''
replacements.append(("report claim field", old2, new2))

missing = [label for label, old, new in replacements if content.count(old) != 1]
if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}")
    for label, old, new in replacements:
        print(f"  {label}: {content.count(old)} occurrence(s)")
else:
    for label, old, new in replacements:
        content = content.replace(old, new, 1)
    with open("calibration/train_multiseed.py", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"All {len(replacements)} changes applied successfully.")