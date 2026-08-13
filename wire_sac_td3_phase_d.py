with open("calibration/train_multiseed.py", encoding="utf-8") as f:
    content = f.read()

replacements = []

# 1. Import SAC/TD3 alongside PPO
old1 = "from stable_baselines3 import PPO"
new1 = "from stable_baselines3 import PPO, SAC, TD3"
replacements.append(("import SAC/TD3", old1, new1))

# 2. train_one_seed() gains algorithm parameter, uses a lookup instead of
# hardcoded PPO(...)
old2 = '''def train_one_seed(params: CalibrationParameters, vehicle: VehicleConstants,
                    waypoint: np.ndarray, seed: int, timesteps: int,
                    n_eval_episodes: int, task: str = "waypoint") -> dict:
    def make_env():
        return Arena(params, vehicle, fixed_waypoint=waypoint, max_episode_steps=100, task=task)

    train_env = Monitor(make_env())
    model = PPO("MlpPolicy", train_env, verbose=0, seed=seed)
    model.learn(total_timesteps=timesteps)'''
new2 = '''ALGORITHMS = {"ppo": PPO, "sac": SAC, "td3": TD3}


def train_one_seed(params: CalibrationParameters, vehicle: VehicleConstants,
                    waypoint: np.ndarray, seed: int, timesteps: int,
                    n_eval_episodes: int, task: str = "waypoint",
                    algorithm: str = "ppo") -> dict:
    def make_env():
        return Arena(params, vehicle, fixed_waypoint=waypoint, max_episode_steps=100, task=task)

    train_env = Monitor(make_env())
    algo_cls = ALGORITHMS[algorithm]
    model = algo_cls("MlpPolicy", train_env, verbose=0, seed=seed)
    model.learn(total_timesteps=timesteps)'''
replacements.append(("train_one_seed algorithm param", old2, new2))

# 3. --algorithm CLI flag, added right after --task
old3 = '''                              "printed warning at runtime.")
    args = parser.parse_args()'''
new3 = '''                              "printed warning at runtime.")
    parser.add_argument("--algorithm", default="ppo", choices=["ppo", "sac", "td3"],
                         help="Which SB3 algorithm to train with. Defaults to 'ppo', "
                              "matching all of this session's existing real Claim B "
                              "results. 'sac'/'td3' are off-policy algorithms with a "
                              "replay buffer - see PROJECT_STATUS.md for their real, "
                              "confirmed learning_starts/buffer_size defaults and the "
                              "validation gates run before trusting them.")
    args = parser.parse_args()'''
replacements.append(("--algorithm CLI flag", old3, new3))

# 4. Thread algorithm=args.algorithm through the call site
old4 = '''            r = train_one_seed(params, vehicle, waypoint, seed, args.timesteps,
                               args.n_eval_episodes, task=args.task)'''
new4 = '''            r = train_one_seed(params, vehicle, waypoint, seed, args.timesteps,
                               args.n_eval_episodes, task=args.task, algorithm=args.algorithm)'''
replacements.append(("call site algorithm=", old4, new4))

# 5. Fix hardcoded "PPO" in the chart title
old5 = '''    ax.set_title(f"Claim B: PPO reward, default vs. calibrated physics "
                 f"(n={args.n_seeds} seeds each, {args.result_type})")'''
new5 = '''    ax.set_title(f"Claim B: {args.algorithm.upper()} reward, default vs. calibrated physics "
                 f"(n={args.n_seeds} seeds each, {args.result_type})")'''
replacements.append(("chart title algorithm name", old5, new5))

# 6. Add algorithm to the saved report dict, so it's on record, not just implied
old6 = '''    report = {
        "claim": Claim.B_TRAINED_POLICY_BEHAVIOR,
        "result_type": args.result_type,'''
new6 = '''    report = {
        "claim": Claim.B_TRAINED_POLICY_BEHAVIOR,
        "algorithm": args.algorithm,
        "result_type": args.result_type,'''
replacements.append(("report algorithm field", old6, new6))

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