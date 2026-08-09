with open("run_calibration_holoocean.py", encoding="utf-8") as f:
    content = f.read()

old_line = "    dof_scales = _compute_dof_scales(train_runs).copy()"
new_line = """    dof_scales = _compute_dof_scales(train_runs).copy()
    if getattr(args, "no_dof_weighting", False):
        dof_scales = None
        print("DOF weighting DISABLED (--no-dof-weighting) - standard unweighted objective, "
              "isolating other changes from the separately-tested weighting approach "
              "(PROJECT_STATUS.md finding #27)."
        )"""

old_arg = '''    parser.add_argument("--output-dir", default="../calibration_output_holoocean_poc")'''
new_arg = '''    parser.add_argument("--output-dir", default="../calibration_output_holoocean_poc")
    parser.add_argument("--no-dof-weighting", action="store_true",
                         help="Disable the p/r DOF weighting (finding #27) for a clean, "
                              "isolated test of other changes.")'''

count_line = content.count(old_line)
count_arg = content.count(old_arg)
print(f"Occurrences - weighting line: {count_line}, arg line: {count_arg}")

if count_line != 1 or count_arg != 1:
    print("REFUSING TO PATCH - one or both anchors not found exactly once.")
else:
    content = content.replace(old_line, new_line, 1)
    content = content.replace(old_arg, new_arg, 1)
    with open("run_calibration_holoocean.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Both changes applied successfully.")
