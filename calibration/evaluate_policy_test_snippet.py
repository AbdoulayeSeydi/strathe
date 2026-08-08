class TestEvaluatePolicySuccessTracksRealTaskTermination:
    """Item 5 follow-up: evaluate_policy_arena.py's success_rate is
    computed generically (just checks Arena.step()'s own `terminated`
    flag) - confirmed correct by reading the code, but never protected
    by a test. This closes that gap: constructs a scenario for EACH task
    where success is/isn't achievable by construction, confirming
    evaluate_policy genuinely reflects that task's real termination
    condition, not a stale/hardcoded assumption."""

    @staticmethod
    def _make_arena(task, **kwargs):
        from arena import Arena
        params = _make_test_params(Ieff_roll=1.0, Ieff_pitch=1.0, Ieff_yaw=1.0,
                                     Xu=-10, Yv=-10, Zw=-10, Kp=-1, Mq=-1, Nr=-1,
                                     Xuu=-5, Yvv=-5, Zww=-5, Kpp=-1, Mqq=-1, Nrr=-1)
        vehicle = VehicleConstants(mass_kg=10.0)
        return Arena(params=params, vehicle=vehicle, task=task,
                     fixed_waypoint=np.array([0.0, 0.0, 0.0]), max_episode_steps=30, **kwargs)

    def test_waypoint_zero_action_at_target_succeeds(self):
        from evaluate_policy_arena import evaluate_policy
        env = self._make_arena("waypoint")
        # Start AT the target (fixed_waypoint=[0,0,0], vehicle starts at
        # origin) - a zero-action policy should immediately terminate.
        result = evaluate_policy(env, policy_fn=lambda obs: np.zeros(6, dtype=np.float32),
                                  n_episodes=3, seed=0)
        assert result.success_rate == 1.0, (
            "waypoint success should be 1.0 when starting AT the target with a zero-action "
            "policy - if this fails, evaluate_policy's success tracking is broken for waypoint"
        )

    def test_station_keeping_single_touch_alone_is_not_enough(self):
        from evaluate_policy_arena import evaluate_policy
        # max_episode_steps=1 - can touch the target once but CANNOT
        # sustain hold_steps (default 20) presence. If evaluate_policy
        # reported success=1.0 here, that would mean it fell back to
        # waypoint-style "touched once" logic instead of genuinely
        # reading station_keeping's real (harder) termination condition.
        env = self._make_arena("station_keeping")
        env.max_episode_steps = 1
        result = evaluate_policy(env, policy_fn=lambda obs: np.zeros(6, dtype=np.float32),
                                  n_episodes=3, seed=0)
        assert result.success_rate == 0.0, (
            "station_keeping should NOT report success from a single touch - if it does, "
            "evaluate_policy is using the wrong (waypoint-style) termination condition"
        )