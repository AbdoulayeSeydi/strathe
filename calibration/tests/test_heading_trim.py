"""A fin vehicle's heading on unseen recordings (#97): the rudder trim set on straight flights only, and the steady sideways water
flow each recording's own straight flight shows."""
from dataclasses import replace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import validation as val
from calibration_report import FLOW_LOOKBACK_S, STRAIGHT_TRIM_DEG, fin_flight_section
from test_fin_pipeline import P
from test_fin_validation import MIXED, _deployment


def _straight(seed, date="02-02-2031", run=1, yaw_trim_deg=0.0, n=1200):
    """A recording flying straight on elevator steps alone, its rudder logged yaw_trim_deg low (re-ballasted), so the model needs that much rudder bias."""
    r = _deployment(seed, date, n=n, run=run)
    ctl = r.controls.copy()
    ctl[:, 0] = 0.0                                   # no rudder: it holds its heading
    r = replace(r, controls=ctl)
    rpy, vel, _ = val.free_run(P, MIXED, r, 0, r.n_samples, full=True)
    logged = ctl.copy()
    logged[:, 0] -= np.radians(yaw_trim_deg)
    return replace(r, real_nu=vel, roll_pitch=rpy[:, :2], orientation_quat=Rotation.from_euler("xyz", rpy).as_quat(), controls=logged)


def test_the_limits_agree_with_the_report():
    assert STRAIGHT_TRIM_DEG == val.STRAIGHT_TRIM_DEG and FLOW_LOOKBACK_S == val.FLOW_LOOKBACK_S


def test_only_flights_that_held_their_heading_are_straight():
    r = _deployment(9)
    n = int(round(val.TRIM_WINDOW_S * r.sample_rate_hz)) + 1
    psi = Rotation.from_quat(r.orientation_quat).as_euler("xyz")[:, 2]
    every = list(range(0, r.n_samples - n + 1, n - 1))
    turned = [a for a in every if abs(val._wrap(psi[a + n - 1] - psi[a])) >= np.radians(val.STRAIGHT_TRIM_DEG)]
    assert turned, "the test recording should turn in some flights"
    assert val._straight_flights(r) == [a for a in every if a not in turned]


def test_the_heading_trim_comes_from_straight_flights():
    """A re-ballasted vehicle needs 1 degree of rudder to fly straight: the straight flights find it."""
    runs = [_straight(9, yaw_trim_deg=1.0), _straight(10, run=2, yaw_trim_deg=1.0)]
    p, used = val.straight_heading_trim(P, MIXED, runs)
    assert used >= val.MIN_TRIM_WINDOWS
    assert np.degrees(p.fin_bias_yaw - P.fin_bias_yaw) == pytest.approx(1.0, abs=.1)
    assert (p.fin_bias_roll, p.fin_bias_pitch) == (P.fin_bias_roll, P.fin_bias_pitch)


def test_too_few_straight_flights_leave_the_heading_trim():
    short = int(val.TRIM_WINDOW_S * 10) * (val.MIN_TRIM_WINDOWS - 1) + 1
    p, used = val.straight_heading_trim(P, MIXED, [_straight(9, n=short, yaw_trim_deg=1.0)])
    assert used == 0 and p == P


def test_the_retrim_sets_heading_from_the_other_recordings_straight_flights():
    runs = [_straight(9, yaw_trim_deg=1.0), _straight(10, run=2, yaw_trim_deg=1.0), _straight(11, run=3, yaw_trim_deg=1.0)]
    per, notes = val.retrim_by_deployment(P, MIXED, runs)
    n = notes["02-02-2031"]
    assert n["retrimmed"] and len(n["heading_from_straight_flights_deg"]) == 3
    assert n["yaw_deg"] == pytest.approx(1.0, abs=.1)
    assert all(np.degrees(q.fin_bias_yaw - P.fin_bias_yaw) == pytest.approx(1.0, abs=.1) for q in per)


def _sliding(r, v_side):
    """r with a steady sideways velocity over ground added, as a cross-current or a turned velocity sensor shows."""
    nu = r.real_nu.copy()
    nu[:, 1] += v_side
    return replace(r, real_nu=nu)


def test_the_sideways_offset_is_the_straight_flights_mean_sideways_velocity():
    r = _sliding(_straight(9), 0.2)
    assert val.sideways_offset(r) == pytest.approx(0.2 + float(np.mean(_straight(9).real_nu[:, 1])), abs=1e-3)


def test_turning_and_slow_samples_dont_set_the_offset():
    r = _sliding(_straight(9), 0.2)
    nu = r.real_nu.copy()
    nu[:600, 0] = 0.3                                  # too slow for the first half: its sideways velocity is ignored
    nu[:600, 1] = 5.0
    q = replace(r, real_nu=nu)
    assert val.sideways_offset(q) == pytest.approx(0.2 + float(np.mean(_straight(9).real_nu[600:, 1])), abs=2e-3)


def test_no_sway_or_no_straight_flight_gives_no_offset():
    r = _straight(9)
    nu = r.real_nu.copy()
    nu[:, 1] = np.nan
    assert val.sideways_offset(replace(r, real_nu=nu)) is None
    nu = r.real_nu.copy()
    nu[:, 0] = 0.0                                     # never moving forward
    assert val.sideways_offset(replace(r, real_nu=nu)) is None


def test_the_offset_is_added_to_the_recordings_current():
    r = _sliding(_straight(9), 0.2)
    known = replace(r, current_body=np.tile([0.1, 0.0, 0.0], (r.n_samples, 1)))
    (a, b), notes = val.with_sideways_offsets([r, known])
    off = val.sideways_offset(r)
    assert np.allclose(a.current_body, [0.0, off, 0.0]) and np.allclose(b.current_body, [0.1, off, 0.0])
    assert notes == {r.filepath: round(off, 3)}


def test_a_model_flown_in_the_sideways_flow_holds_the_recordings_heading():
    """A recording whose vehicle slid sideways at 0.2 m/s: flown in still water from its logged velocity the model turns off its
    heading (the hull weathercocks), flown in the flow it was in it holds it."""
    r = _straight(9)
    drift = r.real_nu.copy()
    drift[:, 1] += 0.2                                 # the velocity sensor sees the flow across the path
    seen = replace(r, real_nu=drift)
    (flown,), _ = val.with_sideways_offsets([seen])
    n = 301
    end = Rotation.from_quat(r.orientation_quat[n - 1]).as_euler("xyz")[2]
    still = val.free_run(P, MIXED, seen, 0, n)
    inflow = val.free_run(P, MIXED, flown, 0, n)
    assert abs(val._wrap(inflow[-1, 2] - end)) < abs(val._wrap(still[-1, 2] - end))


def test_the_report_says_what_was_done_to_heading():
    f = {"windows": 4, "window_s": 30.0, "turning": {"windows": 0}, "diverged": {"calibrated": 0},
         **{m: {k: {"median": 1.0, "p90": 2.0} for k in ("heading_deg", "pitch_deg", "depth_m", "turn_rate_rad_s")}
            for m in ("calibrated", "default", "straight_on")}}
    trim = {"02-02-2031": {"retrimmed": True, "recordings": 2, "trimmed_recordings": 2, "window_s": 10.0, "roll_deg": None,
                           "pitch_deg": 0.5, "yaw_deg": -0.6, "heading_from_straight_flights_deg": {"a.csv": -0.6, "b.csv": -0.6}}}
    said = "\n".join(fin_flight_section({"fin_free_run": f, "fin_retrim": trim, "fin_free_run_as_trained": f,
                                         "sideways_offsets_mps": {"a.csv": -0.21, "b.csv": 0.17}}))
    assert "straight flights alone" in said and "steady sideways water flow" in said and "-0.21 to +0.17 m/s" in said
    assert "As fitted, without re-trimming or the sideways flow" in said
    plain = "\n".join(fin_flight_section({"fin_free_run": f}))
    assert "sideways" not in plain and "straight flights alone" not in plain


# ------------------------------------------------------------------ each flight in the flow just before it, from a steady start
def test_a_flight_uses_nothing_logged_after_its_start():
    r = _sliding(_straight(9), 0.2)
    a = 300
    later = r.real_nu.copy()
    later[a + 1:, 1] += 3.0                            # whatever happens after the start
    q = r.orientation_quat.copy()
    q[a + 1:] = Rotation.from_euler("z", 40, degrees=True).as_quat()
    f1, f2 = val.flight_from(r, a), val.flight_from(replace(r, real_nu=later, orientation_quat=q), a)
    assert np.allclose(f1.current_body[a], f2.current_body[a]) and np.allclose(f1.real_nu[a], f2.real_nu[a])


def test_after_straight_flight_it_starts_moving_with_the_water_and_not_turning():
    r = _sliding(_straight(9), 0.2)
    nu = r.real_nu.copy()
    nu[:, 5] += 0.004                                  # the gyro's wobble, under the straight-flight limit
    r = replace(r, real_nu=nu)
    a = 300
    f = val.flight_from(r, a)
    k = int(val.FLOW_LOOKBACK_S * r.sample_rate_hz)
    flow = f.current_body[a, 1]
    assert flow == pytest.approx(float(np.mean(r.real_nu[a - k:a, 1])), abs=2e-3)   # level and body sideways agree here
    assert f.real_nu[a, 5] == 0.0 and f.real_nu[a, 1] == pytest.approx(flow, abs=2e-3)
    assert np.array_equal(f.real_nu[a + 1:], r.real_nu[a + 1:])                      # only the start changes


def test_after_a_turn_it_starts_from_the_logged_state():
    r = _sliding(_deployment(9), 0.2)
    psi = Rotation.from_quat(r.orientation_quat).as_euler("xyz")[:, 2]
    rate = np.abs(np.diff(psi)) * r.sample_rate_hz
    k = int(val.FLOW_LOOKBACK_S * r.sample_rate_hz)
    a = next(i for i in range(k, r.n_samples) if np.any(rate[i - k:i] > np.radians(val.STEADY_TURN_DEG_S)))
    f = val.flight_from(r, a)
    assert np.array_equal(f.real_nu[a], r.real_nu[a])


def test_too_little_straight_flight_before_it_falls_back_to_the_recordings_offset():
    r = _sliding(_straight(9), 0.2)
    f = val.flight_from(r, 2)                           # 2 s in: not enough flight before it
    assert f.current_body[2, 1] == pytest.approx(val.sideways_offset(r)) and np.array_equal(f.real_nu, r.real_nu)


def test_no_sway_logged_leaves_the_flight_as_it_is():
    r = _straight(9)
    nu = r.real_nu.copy()
    nu[:, 1] = np.nan
    r = replace(r, real_nu=nu)
    assert val.flight_from(r, 300) is r


def test_the_free_run_check_flies_each_window_from_its_own_start():
    r = _sliding(_straight(9), 0.2)
    plain = val.fin_free_run({"calibrated": P}, [r], MIXED, window_s=30.0)
    flown = val.fin_free_run({"calibrated": P}, [r], MIXED, window_s=30.0, flow=True)
    assert flown["windows"] == plain["windows"]
    assert flown["straight_on"] == plain["straight_on"]                          # holding course is scored on the log as it is
    assert flown["calibrated"]["heading_deg"]["median"] < plain["calibrated"]["heading_deg"]["median"]
