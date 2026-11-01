"""F15 (URI re-run): arena intervals pooled episodes as if independent, so they ignored how much results change from
one trained policy to the next; a +25-point docking result became +12 on a rerun."""
import numpy as np

import run_arena_experiment as RA


def _rows(diff_by_seed, n=60):
    rows = []
    for seed, p_cal in diff_by_seed.items():
        for e in range(n):
            extra = {"engine": "stand_in", "steps": 100, "collision": 0, "final_distance_m": 1.0, "effort_Ns": 10.0}
            rows.append({"seed": seed, "episode_seed": e, "physics": "calibrated", "controller": "calibrated",
                         "success": int(e < p_cal * n), **extra})
            rows.append({"seed": seed, "episode_seed": e, "physics": "generic", "controller": "generic",
                         "success": 0, **extra})
    return rows


def test_seeds_that_disagree_widen_the_interval_to_include_zero():
    rows = _rows({1: 0.5, 2: 0.0})            # one training run +50 points, the other 0
    c = RA.paired_difference(rows, "calibrated", "generic")
    assert c["diff"] == 0.25 and c["training_seeds"] == 2 and c["per_seed_diff"] == {"1": 0.5, "2": 0.0}
    assert c["ci"][0] <= 0.01 <= 0.49 <= c["ci"][1]          # includes zero and +50
    d = np.array([1.0] * 30 + [0.0] * 90)                   # the old way: the same episodes, pooled as if independent
    naive = np.random.default_rng(0)
    boots = d[naive.integers(0, len(d), (2000, len(d)))].mean(axis=1)
    assert np.percentile(boots, 2.5) > 0.1                   # ... which wrongly excluded zero


def test_seeds_that_agree_keep_a_narrow_interval_and_one_seed_falls_back_to_episodes():
    c = RA.paired_difference(_rows({1: 0.5, 2: 0.5, 3: 0.5}), "calibrated", "generic")
    assert c["ci"][0] > 0.35 and c["ci"][1] < 0.65
    one = RA.paired_difference(_rows({1: 0.5}), "calibrated", "generic")
    assert one["training_seeds"] == 1 and one["ci"][0] > 0.3


def test_each_controllers_success_rate_resamples_seeds_too():
    s = RA.summarize(_rows({1: 1.0, 2: 0.0}), "calibrated")
    assert s["success_rate"] == 0.5 and s["training_seeds"] == 2 and s["ci"][0] <= 0.01 and s["ci"][1] >= 0.99
    s1 = RA.summarize(_rows({1: 0.5}), "calibrated")
    assert s1["training_seeds"] == 1 and 0.3 < s1["ci"][0] < 0.5   # one seed: the Wilson interval, as before
