"""stall_watchdog.py: timing, stall capture, classification, never-raise."""
import csv
import os
import time

import numpy as np

from optimizer import _wrap_residual_fn
from stall_watchdog import StallWatchdog, classify


def test_classify():
    assert classify(100, 10, 9) == "machine_asleep_or_suspended"
    assert classify(100, 100, 5).startswith("process_waiting")
    assert classify(100, 100, 95) == "computing"
    assert classify(0, 0, 0) == "instant"


def _rows(wd):
    with open(wd.timing_path) as f:
        return list(csv.DictReader(f))


def test_every_evaluation_is_timed_on_three_clocks(tmp_path):
    wd = StallWatchdog(str(tmp_path))
    for _ in range(3):
        with wd.measure("e"):
            sum(range(10000))
    wd.close()
    rows = _rows(wd)
    assert len(rows) == 3
    assert {"wall_s", "monotonic_s", "cpu_s", "classification", "stalled"} <= set(rows[0])
    assert all(r["stalled"] == "0" for r in rows)


def _slow_waiting_evaluation(seconds):
    time.sleep(seconds)


def _slow_computing_evaluation(seconds):
    end = time.process_time() + seconds
    while time.process_time() < end:
        pass


def test_a_stall_is_caught_with_the_stalled_code_in_the_stack_dump(tmp_path):
    wd = StallWatchdog(str(tmp_path), min_threshold_s=0.3, dump_every_s=0.25, poll_s=0.05)
    with wd.measure("residual_evaluation"):
        _slow_waiting_evaluation(1.0)
    wd.close()
    log = open(wd.stall_path).read()
    assert "STALL DETECTED" in log and "still stalled" in log  # repeated dumps during a long stall
    assert "_slow_waiting_evaluation" in log                    # the stalled frame is in the dump
    assert "stall ended" in log and "process_waiting" in log    # sleeping = not computing
    assert _rows(wd)[0]["stalled"] == "1" and wd.n_stalls == 1


def test_a_compute_bound_stall_is_classified_as_computing(tmp_path):
    wd = StallWatchdog(str(tmp_path), min_threshold_s=0.3, poll_s=0.05)
    with wd.measure("e"):
        _slow_computing_evaluation(0.8)
    wd.close()
    assert _rows(wd)[0]["classification"] == "computing"


def test_threshold_adapts_to_normal_evaluation_time(tmp_path):
    wd = StallWatchdog(str(tmp_path), min_threshold_s=0.01, factor=10)
    wd._durations = [2.0, 2.2, 1.8, 2.0]
    assert wd.threshold_s() == 20.0
    wd.close()


def test_optimizer_wrapper_times_evaluations(tmp_path):
    wd = StallWatchdog(str(tmp_path))
    calls = []
    fn = _wrap_residual_fn(lambda x: np.asarray(x) * 2, lambda n, p, c: calls.append(c), wd)
    np.testing.assert_array_equal(fn(np.array([1.0, 2.0])), [2.0, 4.0])
    fn(np.array([0.5]))
    wd.close()
    assert len(_rows(wd)) == 2 and len(calls) == 2


def test_watchdog_failures_never_reach_the_calibration(tmp_path, capsys):
    wd = StallWatchdog(str(tmp_path))
    os.remove(wd.timing_path)
    os.mkdir(wd.timing_path)  # appending to a directory fails
    with wd.measure("e"):
        pass
    with wd.measure("e"):  # disabled after the first failure, still no exception
        pass
    wd.close()
    assert "watchdog disabled" in capsys.readouterr().err
