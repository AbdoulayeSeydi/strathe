with open("README.md", encoding="utf-8") as f:
    content = f.read()

replacements = []

old1 = "         --result-type final --n-seeds 4 --task waypoint --output-dir <output-dir>"
new1 = "         --result-type final --n-seeds 4 --task waypoint --algorithm ppo --output-dir <output-dir>"
replacements.append(("train_multiseed command --algorithm", old1, new1))

old2 = "   `docking`, `obstacle_avoidance`), diagnostic and reporting tools, and a\n   real, automated test suite (`calibration/tests/`, CI-verified on every\n   push)."
new2 = "   `docking`, `obstacle_avoidance`), three RL training algorithms\n   (`--algorithm ppo` (default), `sac`, or `td3` - all real, rigorously\n   validated against the same locked criteria, see PROJECT_STATUS.md\n   finding 37), diagnostic and reporting tools, and a real, automated\n   test suite (`calibration/tests/`, CI-verified on every push)."
replacements.append(("repo contents algorithm mention", old2, new2))

missing = [label for label, old, new in replacements if content.count(old) != 1]
if missing:
    print("REFUSING TO PATCH - could not find exactly once:", ", ".join(missing))
    for label, old, new in replacements:
        print(f"  {label}: {content.count(old)} occurrence(s)")
else:
    for label, old, new in replacements:
        content = content.replace(old, new, 1)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"All {len(replacements)} changes applied successfully.")