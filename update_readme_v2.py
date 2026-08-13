with open("README.md", encoding="utf-8") as f:
    content = f.read()

replacements = []

old1 = "   Optionally pick a task for the delivered environment with `--task waypoint`\n   (default), `--task station_keeping`, or `--task docking`. This is a fast,\n   truncated demo for confirming the mechanism works end to end \u2014 not the\n   source of the reportable numbers below. See the real workflow below for that."
new1 = "   Optionally pick a task for the delivered environment with `--task waypoint`\n   (default), `--task station_keeping`, `--task docking`, or\n   `--task obstacle_avoidance` (this last one has a larger, 15-dim\n   observation space rather than 11 - see PROJECT_STATUS.md finding 36).\n   This is a fast, truncated demo for confirming the mechanism works end\n   to end \u2014 not the source of the reportable numbers below. See the real\n   workflow below for that."
replacements.append(("demo --task doc", old1, new1))

old2 = "   5. **Real multi-seed PPO training (Claim B)** \u2014 optionally pick a task\n      (`waypoint` default, `station_keeping`, or `docking`):"
new2 = "   5. **Real multi-seed PPO training (Claim B)** \u2014 optionally pick a task\n      (`waypoint` default, `station_keeping`, `docking`, or\n      `obstacle_avoidance` - note: obstacle_avoidance generates its own\n      randomized goal each episode and will print a warning that the\n      real-data-grounded waypoint below is computed but not used for\n      this task):"
replacements.append(("real workflow --task doc", old2, new2))

old3 = "   RL training environment (three tasks: `waypoint`, `station_keeping`,\n   `docking`), diagnostic and reporting tools, and a real, automated test\n   suite (`calibration/tests/`, CI-verified on every push)."
new3 = "   RL training environment (four tasks: `waypoint`, `station_keeping`,\n   `docking`, `obstacle_avoidance`), diagnostic and reporting tools, and a\n   real, automated test suite (`calibration/tests/`, CI-verified on every\n   push)."
replacements.append(("repo contents task list", old3, new3))

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