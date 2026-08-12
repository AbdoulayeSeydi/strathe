with open("calibration/run_full_pipeline.py", encoding="utf-8") as f:
    content = f.read()

old1 = '''    parser.add_argument("--task", default="waypoint",
                         choices=["waypoint", "station_keeping", "docking"],
                         help="Which Arena task the delivered Gym environment is built for "
                              "(item #12). 'waypoint' (default) - reach a target point once. "
                              "'station_keeping' - stay near the target for sustained steps, "
                              "not just touch it. 'docking' - reach the target aligned "
                              "(orientation) and slow (low speed) simultaneously.")'''
new1 = '''    parser.add_argument("--task", default="waypoint",
                         choices=["waypoint", "station_keeping", "docking", "obstacle_avoidance"],
                         help="Which Arena task the delivered Gym environment is built for "
                              "(item #12). 'waypoint' (default) - reach a target point once. "
                              "'station_keeping' - stay near the target for sustained steps, "
                              "not just touch it. 'docking' - reach the target aligned "
                              "(orientation) and slow (low speed) simultaneously. "
                              "'obstacle_avoidance' - route around a randomized obstacle to "
                              "reach a randomized goal; this task has a LARGER observation "
                              "space (15 dims, not 11) - see PROJECT_STATUS.md.")'''

count1 = content.count(old1)
print("run_full_pipeline.py occurrences:", count1)
if count1 == 1:
    content = content.replace(old1, new1, 1)
    with open("calibration/run_full_pipeline.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("run_full_pipeline.py updated successfully.")
else:
    print("REFUSING - count mismatch for run_full_pipeline.py")

with open("calibration/train_multiseed.py", encoding="utf-8") as f:
    content2 = f.read()

old2 = '''    parser.add_argument("--task", default="waypoint",
                         choices=["waypoint", "station_keeping", "docking"],
                         help="Which Arena task to train/evaluate on (item #12). Defaults "
                              "to 'waypoint', matching all of this session's existing "
                              "real Claim B results - changing this changes what the "
                              "reported reward numbers actually mean, so be deliberate "
                              "about it, not incidental.")'''
new2 = '''    parser.add_argument("--task", default="waypoint",
                         choices=["waypoint", "station_keeping", "docking", "obstacle_avoidance"],
                         help="Which Arena task to train/evaluate on (item #12). Defaults "
                              "to 'waypoint', matching all of this session's existing "
                              "real Claim B results - changing this changes what the "
                              "reported reward numbers actually mean, so be deliberate "
                              "about it, not incidental. NOTE: 'obstacle_avoidance' "
                              "generates its OWN randomized goal every episode and does "
                              "NOT use the real-data-grounded waypoint below - see the "
                              "printed warning at runtime.")'''

count2 = content2.count(old2)
print("train_multiseed.py (task arg) occurrences:", count2)
if count2 == 1:
    content2 = content2.replace(old2, new2, 1)
else:
    print("REFUSING - count mismatch for train_multiseed.py task arg")

old3 = '''    assert_uses_phase1_split(PHASE1_HELDOUT_DATES)  # sanity: constants haven't drifted
    waypoint = derive_real_waypoint(args.data_dir, PHASE1_HELDOUT_DATES)'''
new3 = '''    assert_uses_phase1_split(PHASE1_HELDOUT_DATES)  # sanity: constants haven't drifted
    waypoint = derive_real_waypoint(args.data_dir, PHASE1_HELDOUT_DATES)
    if args.task == "obstacle_avoidance":
        print("NOTE: --task obstacle_avoidance generates its own randomized goal every "
              "episode (see ObstacleAvoidanceTask.reset()) - the real-data-grounded "
              "waypoint just derived above will be computed but NOT actually used for "
              "this task. This is intentional (obstacle avoidance needs to control "
              "spawn/goal/obstacle together for a valid, solvable episode), not a bug - "
              "flagged here so it's not a silent surprise.")'''

count3 = content2.count(old3)
print("train_multiseed.py (warning insert) occurrences:", count3)
if count3 == 1:
    content2 = content2.replace(old3, new3, 1)
    with open("calibration/train_multiseed.py", "w", encoding="utf-8") as f:
        f.write(content2)
    print("train_multiseed.py updated successfully.")
else:
    print("REFUSING - count mismatch for train_multiseed.py warning insert")
