"""A fin vehicle flying itself (#97): the check that counts, and a rudder checked with no calibration."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

import fin_check
import validation as val
from calibration_report import fin_flight_section
from test_fin_pipeline import ACT, P, V, recording


def flown():
    """A recording whose attitude comes from the vehicle's own flight, so a free run can match it."""
    r = recording()
    rpy, vel, _ = val.free_run(P, V, r, 0, r.n_samples, full=True)
    return replace(r, real_nu=vel, roll_pitch=rpy[:, :2], orientation_quat=Rotation.from_euler("xyz", rpy).as_quat())


def test_flying_itself_reproduces_its_own_truth_and_beats_holding_course():
    res = val.fin_free_run({"calibrated": P, "default": replace(P, fin_lift_slope_a=1.0)}, [flown()], V, window_s=10.0)
    assert res["windows"] >= 3 and res["diverged"]["calibrated"] == 0
    c, d, s = res["calibrated"], res["default"], res["straight_on"]
    assert s["heading_deg"]["median"] > 1.0 and c["heading_deg"]["median"] < .2 * s["heading_deg"]["median"]
    assert c["heading_deg"]["median"] < d["heading_deg"]["median"]
    assert c["depth_m"]["median"] < .05


def test_depth_in_a_free_run_follows_the_vehicles_own_attitude():
    r = recording()
    rpy, vel, z = val.free_run(P, V, r, 10, 60, full=True)
    assert rpy.shape == (60, 3) and vel.shape == (60, 6) and z.shape == (60,) and z[0] == 0.0
    assert np.isfinite(z).all()


def test_free_run_without_full_is_unchanged():
    r = recording()
    a = val.free_run(P, V, r, 10, 60)
    assert np.array_equal(a, val.free_run(P, V, r, 10, 60, full=True)[0])


def test_a_diverging_window_is_counted():
    r = recording()
    bad = replace(P, Nr=5.0, Nrr=5.0, Mq=5.0, Mqq=5.0, Kp=5.0, Kpp=5.0)
    res = val.fin_free_run({"calibrated": bad}, [r], V, window_s=30.0)
    assert res["windows"] > 0


def test_fin_steps_compare_the_response():
    r = recording()
    s = val.fin_steps({"calibrated": P}, [r], V, ACT)
    assert s["steering"]["steps"] > 0 and s["steering"]["calibrated_size_ratio"] == pytest.approx(1.0, abs=.1)


def _summ(m):
    return {"median": m, "p90": 2 * m, "n": 5}


def test_the_report_opens_with_flying_on_its_own_and_says_when_it_does_no_better():
    f = {"window_s": 30.0, "windows": 5, "diverged": {"calibrated": 1},
         "calibrated": {"heading_deg": _summ(9.), "pitch_deg": _summ(1.), "depth_m": _summ(.2), "turn_rate_rad_s": _summ(.01)},
         "default": {"heading_deg": _summ(20.), "pitch_deg": _summ(5.), "depth_m": _summ(2.), "turn_rate_rad_s": _summ(.05)},
         "straight_on": {"heading_deg": _summ(3.), "pitch_deg": _summ(4.), "depth_m": _summ(1.), "turn_rate_rad_s": _summ(.02)}}
    L = "".join(fin_flight_section({"fin_free_run": f, "fin_steps": {"steering": {"steps": 3, "calibrated_size_ratio": .9}}}))
    assert "Flying on its own" in L and "| Heading | 9.00°" in L and "no better" in L and "heading" in L and "blew up" in L
    assert "3 steering steps" in L and fin_flight_section({}) == []


def _prepared(d, flip=1.0, n=900, moved=True):
    t = np.arange(n) / 10
    a = (.15 * np.sin(t / 4)) if moved else np.zeros(n)
    rate = np.r_[np.zeros(10), a[:-10]] * .4 * flip
    d.mkdir(exist_ok=True)
    pd.DataFrame({"fin_rudder": a, "Br": rate, "Bq": 0.0}).to_csv(d / "r.csv", index=False)
    return d


def test_fin_check_reads_sign_lag_and_gain(tmp_path):
    r = fin_check.check_folder(str(_prepared(tmp_path / "a")), 10.0)["channels"]["rudder"]
    assert r["correlation"] > .9 and r["lag_s"] == pytest.approx(1.0, abs=.15) and r["rate_deg_s_per_deg"] == pytest.approx(.4, rel=.05)
    assert "✅" in " ".join(fin_check.words({"recordings": 1, "channels": {"rudder": {**r, "against": "turn"}}}))


def test_fin_check_catches_a_reversed_rudder(tmp_path):
    res = fin_check.check_folder(str(_prepared(tmp_path / "a", flip=-1.0)), 10.0)
    assert "problem" in res["channels"]["rudder"] and "❌" in " ".join(fin_check.words(res))


def test_fin_check_says_when_a_fin_barely_moved(tmp_path):
    res = fin_check.check_folder(str(_prepared(tmp_path / "a", moved=False)), 10.0)
    assert "note" in res["channels"]["rudder"]


@pytest.mark.parametrize("kind", ["empty", "no_fins", "no_rate"])
def test_fin_check_says_what_it_needs(tmp_path, kind):
    d = tmp_path / "a"
    d.mkdir()
    if kind == "no_fins":
        pd.DataFrame({"Br": [0.0, 1.0]}).to_csv(d / "r.csv", index=False)
    if kind == "no_rate":
        pd.DataFrame({"fin_rudder": [0.0, 1.0]}).to_csv(d / "r.csv", index=False)
    with pytest.raises((FileNotFoundError, ValueError), match="ingest|fin channels|sampling" if kind != "no_rate" else "sampling"):
        fin_check.check_folder(str(d), None if kind == "no_rate" else 10.0)


def test_the_cli_command_exists_and_exits_on_a_problem(tmp_path, capsys):
    import strathe_cli as cli
    p = cli.build_parser()
    a = p.parse_args(["fin-check", "--data", str(_prepared(tmp_path / "a", flip=-1.0)), "--sampling-rate-hz", "10"])
    assert a.fn(a) == 1 and "reversed" in capsys.readouterr().out
