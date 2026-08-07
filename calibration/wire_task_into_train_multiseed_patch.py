# Safe, self-verifying patch: wires item #12 (Arena task selection) into
# train_multiseed.py - the REAL Stage 2.4 script, not the demo. Before
# this, Arena's task parameter existed but nothing in the actual
# reportable-results pipeline could reach it, same gap run_full_pipeline.py
# had (already fixed there).

with open("train_multiseed.py", encoding="utf-8") as f:
    content = f.read()

old_train_one_seed = '''def train_one_seed(params: CalibrationParameters, vehicle: VehicleConstants,
                    waypoint: np.ndarray, seed: int, timesteps: int,
                    n_eval_episodes: int) -> dict:
    def make_env():
        return Arena(params, vehicle, fixed_waypoint=waypoint, max_episode_steps=100)'''

new_train_one_seed = '''def train_one_seed(params: CalibrationParameters, vehicle: VehicleConstants,
                    waypoint: np.ndarray, seed: int, timesteps: int,
                    n_eval_episodes: int, task: str = "waypoint") -> dict:
    def make_env():
        return Arena(params, vehicle, fixed_waypoint=waypoint, max_episode_steps=100, task=task)'''

old_arg = '''    parser.add_argument("--output-dir", default="train_multiseed")
    args = parser.parse_args()'''

new_arg = '''    parser.add_argument("--output-dir", default="train_multiseed")
    parser.add_argument("--task", default="waypoint",
                         choices=["waypoint", "station_keeping", "docking"],
                         help="Which Arena task to train/evaluate on (item #12). Defaults "
                              "to 'waypoint', matching all of this session's existing "
                              "real Claim B results - changing this changes what the "
                              "reported reward numbers actually mean, so be deliberate "
                              "about it, not incidental.")
    args = parser.parse_args()'''

old_call = '''            r = train_one_seed(params, vehicle, waypoint, seed, args.timesteps, args.n_eval_episodes)'''
new_call = '''            r = train_one_seed(params, vehicle, waypoint, seed, args.timesteps,
                               args.n_eval_episodes, task=args.task)'''

replacements = [
    ("train_one_seed definition", old_train_one_seed, new_train_one_seed),
    ("CLI args block", old_arg, new_arg),
    ("train_one_seed call site", old_call, new_call),
]

missing = [label for label, old, new in replacements if content.count(old) != 1]
if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}.")
else:
    for label, old, new in replacements:
        content = content.replace(old, new, 1)
    with open("train_multiseed.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Task selection wired into train_multiseed.py successfully.")