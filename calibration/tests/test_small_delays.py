"""Small sensor delays (#98): with two or more training runs, state_smoother.fix_delays pools every run's best window,
so a delay each run shows weakly adds up; a single run keeps the scan from before #98; and the run that gains most
can't decide it alone (the leave-one-out check)."""
import numpy as np

import state_smoother as sm
from test_measurements import SPEC, TRUTH, as_run, truth_frame

NOISE = {"usbl": 0.05, "main": 0.002}


def _runs(n_runs, depth_lag=0, pos_lag=0, seed0=0, pos_noise=0.05, lag_runs=None):
    """n_runs synthetic recordings (different seeds), depth and/or fixes logged `lag` samples late in the runs listed in
    lag_runs (all of them by default)."""
    out = []
    for i in range(n_runs):
        df, _, _ = truth_frame(n=900, seed=seed0 + i, dvl=False, pos_every=10, pos_noise=pos_noise)
        df["depth_main"] += 0.002 * np.random.default_rng(100 + seed0 + i).normal(size=len(df))   # a Bar30's noise
        late = lag_runs is None or i in lag_runs
        if depth_lag and late:
            df["depth_main"] = df["depth_main"].shift(depth_lag)
        if pos_lag and late:
            cols = ["pos_usbl_n", "pos_usbl_e", "pos_usbl_d"]
            df[cols] = df[cols].shift(pos_lag)
        out.append(as_run(df, sensors=("usbl", "main")))
    return out


def test_a_small_depth_delay_is_found_when_the_runs_are_pooled():
    found = sm.fix_delays(_runs(4, depth_lag=2), TRUTH, SPEC.vehicle, 0.03, NOISE)["main"]
    assert found["pooled"] and found["runs"] == 4
    assert found["late"] and abs(found["delay_s"] - 0.2) <= 0.1 + 1e-9
    assert found["gain_total"] >= sm.DELAY_POOLED_GAIN and abs(found["loo_delay_s"] - found["delay_s"]) <= 0.2


def test_on_time_sensors_are_not_shifted_when_pooled():
    found = sm.fix_delays(_runs(4), TRUTH, SPEC.vehicle, 0.03, NOISE)
    for name in ("main", "usbl"):
        assert found[name]["pooled"] and not found[name]["late"], (name, found[name])


def test_one_run_keeps_the_scan_from_before():
    """With one run, the same 0.2 s depth delay is under the single-run rule's 0.5 s, so it's said, not taken off."""
    found = sm.fix_delays(_runs(1, depth_lag=2), TRUTH, SPEC.vehicle, 0.03, NOISE)["main"]
    assert not found["pooled"] and found["runs"] == 1 and not found["late"]


def test_one_late_run_cant_decide_it_alone():
    """Fixes 1 s late in one run and on time in the other: the pooled best can follow the late run, but leaving that
    run out, the rest read on time, so nothing is taken off."""
    found = sm.fix_delays(_runs(2, pos_lag=10, lag_runs={0}), TRUTH, SPEC.vehicle, 0.03, NOISE)["usbl"]
    assert found["pooled"] and not found["late"], found


def test_pooling_uses_at_most_the_runs_with_the_most_readings():
    found = sm.fix_delays(_runs(sm.DELAY_MAX_RUNS + 1), TRUTH, SPEC.vehicle, 0.03, NOISE)["main"]
    assert found["runs"] == sm.DELAY_MAX_RUNS


def _report(d):
    from calibration_report import sensors_section
    return "\n".join(sensors_section({"sensor_suite": {"channels": ["usbl.n"], "fix_delays": {"usbl": d}}}))


def test_the_report_says_why_a_pooled_delay_was_or_wasnt_taken_off():
    base = {"kind": "position", "pooled": True, "runs": 8, "fixes": 450, "gain_per_fix": 0.05, "gain_per_s": 0.05,
            "at_the_scan_edge": False, "floor_s": 0.2, "loo_delay_s": 0.3, "applied_s": 0.0}
    took = _report(dict(base, delay_s=0.3, gain_total=24.6, late=True, applied_s=0.3, applied_from="found in the fixes"))
    assert "over 8 recordings, 24.6 better" in took and "the same without the recording that gained most" in took
    assert "0.3 s was taken off" in took
    small = _report(dict(base, delay_s=0.1, gain_total=3.0, late=False))
    assert "under the 0.2 s a position sensor's delay can be told apart at" in small and "wasn't taken off" in small
    weak = _report(dict(base, delay_s=0.3, gain_total=1.2, late=False))
    assert "only 1.2 better in log-likelihood than no delay over 8 recordings; taking it off takes 3" in weak
    one = _report(dict(base, delay_s=1.0, gain_total=53.1, loo_delay_s=2.0, runs=2, late=False))
    assert "one recording decides it: without it, the other 1 read 2.0 s" in one
    assert "s late" not in _report(dict(base, delay_s=0.0, gain_total=0.0, late=False))


def test_a_found_delay_is_scanned_again_with_the_fitted_values():
    """run_calibration.rescan_delays (#98): a delay the first scan found short (here 0.1 s taken off depth that's
    0.3 s late) is scanned again from the sensor's own times with better values, and the readings end up shifted by
    the new delay, not by the old one plus it; a delay the lab gave is left alone."""
    from run_calibration import rescan_delays
    runs = _runs(4, depth_lag=3)
    sm.shift_fixes(runs, "main", 0.1)
    info = {"noise": NOISE, "fix_delays": {
        "main": {"kind": "depth", "delay_s": 0.1, "late": True, "applied_s": 0.1, "applied_from": "found in the fixes"},
        "usbl": {"kind": "position", "delay_s": 0.0, "late": False, "applied_s": 0.5,
                 "applied_from": "the vehicle description's sensors.usbl.delay_s"}}}
    rescan_delays(runs, [], TRUTH, SPEC.vehicle, 0.03, info, 2)
    d = info["fix_delays"]["main"]
    assert abs(d["applied_s"] - 0.3) <= 0.1 + 1e-9 and d["rescanned_round"] == 2 and d["first_scan"]["applied_s"] == 0.1
    assert all(abs(next(c for c in r.channels if c.name == "main").shifted_s - d["applied_s"]) < 1e-9 for r in runs)
    assert info["fix_delays"]["usbl"]["applied_s"] == 0.5 and "rescanned_round" not in info["fix_delays"]["usbl"]


def test_finishing_a_checkpoint_takes_off_the_delay_its_round_used(tmp_path):
    """run_calibration.note_applied_delays / apply_noted_delays (#98): a later round can scan a delay again and change
    it (0.2 s to 0.3 s here), so finishing that round's checkpoint takes off what the round used, not the fresh scan's
    first answer; a checkpoint from before #98 (no record) keeps the fresh scan."""
    from run_calibration import APPLIED_DELAYS_FILE, apply_noted_delays, note_applied_delays
    ck = tmp_path / "checkpoint.yaml"
    note_applied_delays(str(ck), {"main": {"applied_s": 0.3}, "usbl": {"applied_s": 0.0}, "odd": {"note": "few"}})
    assert (tmp_path / APPLIED_DELAYS_FILE).exists()
    runs = _runs(2)
    sm.shift_fixes(runs, "main", 0.2)    # what a fresh scan (the description's values) took off
    delays = {"main": {"applied_s": 0.2, "applied_from": "found in the fixes"}, "usbl": {"applied_s": 0.0}}
    apply_noted_delays(str(ck), runs, [], delays)
    assert delays["main"]["applied_s"] == 0.3 and delays["main"]["applied_from_checkpoint"]
    assert all(abs(next(c for c in r.channels if c.name == "main").shifted_s - 0.3) < 1e-9 for r in runs)
    assert delays["usbl"] == {"applied_s": 0.0}
    old = {"main": {"applied_s": 0.2}}
    apply_noted_delays(str(tmp_path / "elsewhere" / "checkpoint.yaml"), runs, [], old)
    assert old == {"main": {"applied_s": 0.2}}


def test_recordings_that_disagree_have_nothing_taken_off():
    """Depth 0.4 s late in two recordings and on time in two (sensors logged apart and merged, each recording offset
    differently): each fits far better with its own delay than with any one, so none is taken off, and the report
    says the recordings disagree. A consistent delay (the first test) stays under the bar."""
    found = sm.fix_delays(_runs(4, depth_lag=4, lag_runs={0, 1}), TRUTH, SPEC.vehicle, 0.03, NOISE)["main"]
    assert found["disagreement_per_run"] > sm.DELAY_DISAGREE_PER_RUN and not found["late"], found
    assert found["own_delays_s"][0] == 0.0 and found["own_delays_s"][-1] >= 0.3
    text = _report({**found, "applied_s": 0.0})
    assert "the recordings disagree" in text and "wasn't taken off" in text
    same = sm.fix_delays(_runs(4, depth_lag=2), TRUTH, SPEC.vehicle, 0.03, NOISE)["main"]
    assert same["late"] and same["disagreement_per_run"] <= sm.DELAY_DISAGREE_PER_RUN


def test_fixes_too_slow_to_show_a_small_delay_say_to_give_it():
    base = {"kind": "position", "delay_s": 0.0, "late": False, "pooled": True, "runs": 8, "gain_total": 0.0,
            "loo_delay_s": 0.0, "floor_s": 0.2, "disagreement_per_run": 0.0, "fixes": 96, "applied_s": 0.0}
    slow = _report({**base, "fix_interval_s": 5.0})
    assert "every 5.0 s" in slow and "`delay_s:`" in slow
    assert "too far apart" not in _report({**base, "fix_interval_s": 1.0, "fixes": 480})     # 1 Hz: the floor holds
    assert "too far apart" not in _report({**base, "fix_interval_s": 5.0, "late": True, "delay_s": 1.0,
                                           "applied_s": 1.0})                                # found: nothing to add
