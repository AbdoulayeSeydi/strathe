with open("run_calibration_holoocean.py", encoding="utf-8") as f:
    content = f.read()

old_broken = '''    dof_scales = _compute_dof_scales(train_runs).copy()
    if getattr(args, "no_dof_weighting", False):
        dof_scales = None
        print("DOF weighting DISABLED (--no-dof-weighting) - standard unweighted objective, "
              "isolating other changes from the separately-tested weighting approach "
              "(PROJECT_STATUS.md finding #27).")
    dof_scales[_dof_labels.index("p")] *= 0.3   # p: strongly boosted - confirmed under-weighted
    dof_scales[_dof_labels.index("r")] *= 0.15  # r: boosted BEYOND its already-large natural scale
    print(f"Custom DOF weighting active - p and r boosted beyond natural scale "
          f"(see PROJECT_STATUS.md for the research behind this): {dict(zip(_dof_labels, dof_scales))}")'''

new_fixed = '''    dof_scales = _compute_dof_scales(train_runs).copy()
    if getattr(args, "no_dof_weighting", False):
        dof_scales = None
        print("DOF weighting DISABLED (--no-dof-weighting) - standard unweighted objective, "
              "isolating other changes from the separately-tested weighting approach "
              "(PROJECT_STATUS.md finding #27).")
    else:
        dof_scales[_dof_labels.index("p")] *= 0.3   # p: strongly boosted - confirmed under-weighted
        dof_scales[_dof_labels.index("r")] *= 0.15  # r: boosted BEYOND its already-large natural scale
        print(f"Custom DOF weighting active - p and r boosted beyond natural scale "
              f"(see PROJECT_STATUS.md for the research behind this): {dict(zip(_dof_labels, dof_scales))}")'''

count = content.count(old_broken)
print(f"Occurrences of the known-broken block: {count}")
if count != 1:
    print("REFUSING TO PATCH - expected exactly 1, found something different.")
else:
    content = content.replace(old_broken, new_fixed, 1)
    with open("run_calibration_holoocean.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed - p/r multiplication now correctly inside the else branch, only runs when weighting is enabled.")
