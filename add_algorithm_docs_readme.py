with open("README.md", encoding="utf-8") as f:
    content = f.read()

replacements = []

old1 = "python train_multiseed.py --data-dir <path> --spec vehicle_spec.yaml \\\n      --calibrated-params <output-dir>/calibrated_parameters.yaml \\\n      --result-type final --n-seeds 4 --task waypoint --output-dir <output-dir>"
new1 = "python train_multiseed.py --data-dir <path> --spec vehicle_spec.yaml \\\n      --calibrated-params <output-dir>/calibrated_parameters.yaml \\\n      --result-type final --n-seeds 4 --task waypoint --algorithm ppo --output-dir <output-dir>"
replacements.append(("train_multiseed command --algorithm", old1, new1))

old2 = "docking), diagnostic and reporting tools, and a\n   real, automated test suite (`calibration/tests/`, CI-verified on every\n   push)."
new2 = "docking), diagnostic and reporting tools, three RL training\n   algorithms (`--algorithm ppo` (default), `sac`, or `td3` - all real,\n   rigorously validated against the same locked criteria, see\n   PROJECT_STATUS.md finding 37), and a real, automated test suite\n   (`calibration/tests/`, CI-verified on every push)."
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