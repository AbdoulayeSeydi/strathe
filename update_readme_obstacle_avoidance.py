with open("README.md", encoding="utf-8") as f:
    content = f.read()

replacements = []

# 1. Fast demo pipeline's --task documentation
old1 = """Optionally pick a task for the delivered environment with --task waypoint
   (default), --task station_keeping, or --task docking. This is a fast,
   truncated demo for confirming the mechanism works end to end - not the
   source of the reportable numbers below. See the real workflow below for that."""
new1 = """Optionally pick a task for the delivered environment with --task waypoint
   (default), --task station_keeping, --task docking, or --task
   obstacle_avoidance (this last one has a larger, 15-dim observation
   space rather than 11 - see PROJECT_STATUS.md finding 36). This is a
   fast, truncated demo for confirming the mechanism works end to end -
   not the source of the reportable numbers below. See the real workflow
   below for that."""
replacements.append(("demo --task doc", old1, new1))

# 2. Real workflow step 5's --task documentation
old2 = """Real multi-seed PPO training (Claim B) - optionally pick a task
   (waypoint default, station_keeping, or docking):"""
new2 = """Real multi-seed PPO training (Claim B) - optionally pick a task
   (waypoint default, station_keeping, docking, or obstacle_avoidance -
   note: obstacle_avoidance generates its own randomized goal each
   episode and will print a warning that the real-data-grounded waypoint
   below is computed but not used for this task):"""
replacements.append(("real workflow --task doc", old2, new2))

# 3. "What's in this repo" - task count and list
old3 = """the Gym/Arena
   RL training environment (three tasks: waypoint, station_keeping,
   docking), diagnostic and reporting tools, and a real, automated test
   suite (calibration/tests/, CI-verified on every push)."""
new3 = """the Gym/Arena
   RL training environment (four tasks: waypoint, station_keeping,
   docking, obstacle_avoidance), diagnostic and reporting tools, and a
   real, automated test suite (calibration/tests/, CI-verified on every
   push)."""
replacements.append(("repo contents task list", old3, new3))

missing = [label for label, old, new in replacements if content.count(old) != 1]
if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}.")
    for label, old, new in replacements:
        print(f"  {label}: {content.count(old)} occurrence(s)")
else:
    for label, old, new in replacements:
        content = content.replace(old, new, 1)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
    print(f"All {len(replacements)} changes applied successfully.")