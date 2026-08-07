# Safe, self-verifying patch: adds task selection to Arena (item #12) -
# "waypoint" (existing, UNCHANGED, default), "station_keeping" (new -
# requires staying near the target for hold_steps consecutive steps, not
# just touching it once), "docking" (new - requires position, orientation
# alignment, AND low speed simultaneously, matching a real dock's actual
# requirements). Uses only fields already confirmed to exist
# (self._core.state.nu, self._core.state.orientation.as_euler("xyz")),
# not guessed.

with open("arena.py", encoding="utf-8") as f:
    content = f.read()

old_init = '''    def __init__(
        self,
        params: CalibrationParameters,
        vehicle: VehicleConstants,
        dt: float = 0.1,
        physics_dt: float = 0.02,
        max_episode_steps: int = 200,
        waypoint_radius: float = 0.5,
        fixed_waypoint: Optional[np.ndarray] = None,
        waypoint_range: float = 3.0,
        action_bound: float = DEFAULT_ACTION_BOUND,
    ):
        super().__init__()
        sim_config = SimulationConfig(dt=dt, physics_dt=physics_dt)
        self._core = ArenaCore(params, vehicle, sim_config)
        self.max_episode_steps = max_episode_steps
        self.waypoint_radius = waypoint_radius
        self._fixed_waypoint = None if fixed_waypoint is None else np.array(fixed_waypoint, dtype=float)
        self.waypoint_range = waypoint_range'''

new_init = '''    def __init__(
        self,
        params: CalibrationParameters,
        vehicle: VehicleConstants,
        dt: float = 0.1,
        physics_dt: float = 0.02,
        max_episode_steps: int = 200,
        waypoint_radius: float = 0.5,
        fixed_waypoint: Optional[np.ndarray] = None,
        waypoint_range: float = 3.0,
        action_bound: float = DEFAULT_ACTION_BOUND,
        task: str = "waypoint",
        hold_steps: int = 20,
        docking_orientation_tol_rad: float = 0.2,
        docking_max_speed: float = 0.3,
    ):
        """
        task: "waypoint" (default, ORIGINAL behavior, unchanged) - touch
            the target once, episode ends. "station_keeping" - must STAY
            within waypoint_radius for hold_steps CONSECUTIVE steps before
            counting as success (a vehicle drifting through and continuing
            past should NOT succeed - the whole point of this task).
            "docking" - requires position, orientation alignment
            (roll/pitch within docking_orientation_tol_rad), AND low speed
            (below docking_max_speed) simultaneously, matching what a real
            dock actually requires - not just touching the target point at
            any angle/speed.
        """
        if task not in ("waypoint", "station_keeping", "docking"):
            raise ValueError(
                f"Unknown task {task!r} - expected 'waypoint', 'station_keeping', or 'docking'"
            )
        super().__init__()
        sim_config = SimulationConfig(dt=dt, physics_dt=physics_dt)
        self._core = ArenaCore(params, vehicle, sim_config)
        self.max_episode_steps = max_episode_steps
        self.waypoint_radius = waypoint_radius
        self._fixed_waypoint = None if fixed_waypoint is None else np.array(fixed_waypoint, dtype=float)
        self.waypoint_range = waypoint_range
        self._task = task
        self._hold_steps = hold_steps
        self._docking_orientation_tol_rad = docking_orientation_tol_rad
        self._docking_max_speed = docking_max_speed
        self._consecutive_steps_in_radius = 0  # used only by station_keeping'''

old_reset = '''    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._core.reset()
        self._step_count = 0
        if self._fixed_waypoint is not None:
            self._waypoint = self._fixed_waypoint.copy()
        else:
            self._waypoint = self.np_random.uniform(-self.waypoint_range, self.waypoint_range, size=3)
        return self._get_obs(), {}'''

new_reset = '''    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._core.reset()
        self._step_count = 0
        self._consecutive_steps_in_radius = 0
        if self._fixed_waypoint is not None:
            self._waypoint = self._fixed_waypoint.copy()
        else:
            self._waypoint = self.np_random.uniform(-self.waypoint_range, self.waypoint_range, size=3)
        return self._get_obs(), {}'''

old_step = '''    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32),
                          self.action_space.low, self.action_space.high)
        tau = action * self._action_bound  # rescale [-1,1] -> real force/torque units
        self._core.step(tau)
        self._step_count += 1

        distance = float(np.linalg.norm(self._waypoint - self._core.state.position))
        reward = -distance
        terminated = distance < self.waypoint_radius
        truncated = self._step_count >= self.max_episode_steps
        if terminated:
            reward += 10.0

        info = {"distance": distance, "position": self._core.state.position.copy()}
        return self._get_obs(), reward, terminated, truncated, info'''

new_step = '''    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32),
                          self.action_space.low, self.action_space.high)
        tau = action * self._action_bound  # rescale [-1,1] -> real force/torque units
        self._core.step(tau)
        self._step_count += 1

        distance = float(np.linalg.norm(self._waypoint - self._core.state.position))
        reward = -distance
        info = {"distance": distance, "position": self._core.state.position.copy()}
        truncated = self._step_count >= self.max_episode_steps

        if self._task == "waypoint":
            terminated = distance < self.waypoint_radius
            if terminated:
                reward += 10.0

        elif self._task == "station_keeping":
            if distance < self.waypoint_radius:
                self._consecutive_steps_in_radius += 1
                # Extra shaping while close: penalize speed too, so the
                # policy learns to actually STOP near the target rather
                # than orbit through the radius repeatedly.
                speed = float(np.linalg.norm(self._core.state.nu[:3]))
                reward -= 0.1 * speed
            else:
                self._consecutive_steps_in_radius = 0
            terminated = self._consecutive_steps_in_radius >= self._hold_steps
            if terminated:
                reward += 10.0
            info["consecutive_steps_in_radius"] = self._consecutive_steps_in_radius

        elif self._task == "docking":
            roll, pitch, _yaw = self._core.state.orientation.as_euler("xyz")
            orientation_error = float(np.sqrt(roll ** 2 + pitch ** 2))
            speed = float(np.linalg.norm(self._core.state.nu[:3]))
            reward -= 0.5 * orientation_error  # shape toward alignment throughout, not just at the end
            terminated = (
                distance < self.waypoint_radius
                and orientation_error < self._docking_orientation_tol_rad
                and speed < self._docking_max_speed
            )
            if terminated:
                reward += 10.0
            info["orientation_error"] = orientation_error
            info["speed"] = speed

        return self._get_obs(), reward, terminated, truncated, info'''

missing = []
for label, old in [("__init__", old_init), ("reset", old_reset), ("step", old_step)]:
    if content.count(old) != 1:
        missing.append(label)

if missing:
    print(f"REFUSING TO PATCH - could not find exactly once: {', '.join(missing)}.")
else:
    content = content.replace(old_init, new_init, 1)
    content = content.replace(old_reset, new_reset, 1)
    content = content.replace(old_step, new_step, 1)
    with open("arena.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Arena task support (item #12) applied successfully.")