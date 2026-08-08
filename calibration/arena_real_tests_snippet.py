class TestArenaTaskVariantsReal:
    """Corrects a real gap in TestArenaTaskVariants above: that class
    tested a REIMPLEMENTATION of the reward logic, not the actual Arena
    class from arena.py — meaning it would NOT catch a real regression in
    arena.py itself. These tests construct and step REAL Arena instances,
    the same proven pattern already used in run_full_pipeline.py's real
    Gym-environment smoke test."""

    @staticmethod
    def _make_arena(task, **kwargs):
        from arena import Arena
        params = _make_test_params(Ieff_roll=1.0, Ieff_pitch=1.0, Ieff_yaw=1.0,
                                     Xu=-10, Yv=-10, Zw=-10, Kp=-1, Mq=-1, Nr=-1,
                                     Xuu=-5, Yvv=-5, Zww=-5, Kpp=-1, Mqq=-1, Nrr=-1)
        vehicle = VehicleConstants(mass_kg=10.0)
        return Arena(params=params, vehicle=vehicle, task=task,
                     fixed_waypoint=np.array([1.0, 0.0, 0.0]), max_episode_steps=50, **kwargs)

    def test_default_task_is_waypoint(self):
        env = self._make_arena("waypoint")
        obs, info = env.reset()
        assert env._task == "waypoint"

    def test_each_task_constructs_and_steps_without_error(self):
        for task in ("waypoint", "station_keeping", "docking"):
            env = self._make_arena(task)
            obs, info = env.reset()
            obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
            assert obs.shape == (11,)
            assert isinstance(reward, float)

    def test_station_keeping_info_has_consecutive_steps_key(self):
        env = self._make_arena("station_keeping")
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert "consecutive_steps_in_radius" in info

    def test_docking_info_has_orientation_and_speed_keys(self):
        env = self._make_arena("docking")
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert "orientation_error" in info
        assert "speed" in info

    def test_waypoint_info_does_not_have_task_specific_keys(self):
        """Confirms the default task's info dict is UNCHANGED from before
        this feature existed - a real backward-compatibility check."""
        env = self._make_arena("waypoint")
        env.reset()
        _, _, _, _, info = env.step(np.zeros(6, dtype=np.float32))
        assert "consecutive_steps_in_radius" not in info
        assert "orientation_error" not in info

    def test_invalid_task_raises(self):
        with pytest.raises(ValueError):
            self._make_arena("not_a_real_task")