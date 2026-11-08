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


def _prestero():
    """Prestero's published REMUS in Strathe's terms: his lumped coefficients, less what Strathe works out itself (the Munk moment from added mass,
    the fins' share from their flow angle)."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stress"))
    import prestero_remus as PR
    import steering
    from dynamics_model import CalibrationParameters, VehicleConstants
    raw = {"vehicle": {"steering": "fin"}, "fins": {"layout": "cross", "x_m": -.64, "radius_m": .14, "area_m2": 6.65e-3, "span_m": .0857, "stall_deg": 25, "limit_deg": 30}}
    mz = PR.M * PR.ZG
    s = PR.YUUDR / (1025 * 6.65e-3)
    p = CalibrationParameters(Ieff_roll=PR.IXX - PR.KPDOT, Ieff_pitch=PR.IYY - PR.MQDOT, Ieff_yaw=PR.IZZ - PR.NRDOT, Xu=0, Yv=0, Zw=0, Kp=0, Mq=0, Nr=0, Xuu=PR.XUU, Yvv=PR.YVV, Zww=PR.ZWW, Kpp=PR.KPP,
                              Mqq=PR.MQQ, Nrr=PR.NRR, Xu_dot=-PR.XUDOT, Yv_dot=-PR.YVDOT, Zw_dot=-PR.ZWDOT, net_buoyancy_z=PR.B - PR.W, cg_cb_offset_z=PR.ZG, Zuw=PR.ZUW + PR.YUUDR, Muw=-4.4, Muq=0.0,
                              Xq_dot=mz, Yp_dot=-mz, Yr_dot=-PR.YRDOT, Zq_dot=-PR.ZQDOT, fin_lift_slope_a=s, fin_lift_slope_b=s, fin_inflow_a=1.0, fin_inflow_b=1.0)
    return p, VehicleConstants(mass_kg=PR.M, actuators=steering.actuators_from(raw))


def test_stability_finds_prestero_unstable_as_his_own_equations_do():
    """Worked out separately from his equations: +1.04 per second at 1.54 m/s, doubling in 0.67 s."""
    p, v = _prestero()
    s = val.fin_stability(p, v, 1.54)
    assert s["growth_per_s"] == pytest.approx(1.04, rel=.1)
    assert s["doubling_s"] < val.UNSTABLE_DOUBLING_S


def test_strathes_physics_linearises_to_prestero_entry_by_entry():
    """Prestero's equations (stress/prestero_remus.py) were written from his thesis, not from Strathe's model, and lump what Strathe works out
    separately. About straight flight they must still say the same thing. Roll rate is left out: Strathe's fins damp roll, his linear terms don't."""
    import math
    import prestero_remus as PR
    from dynamics_model import compute_acceleration
    p, v = _prestero()
    u, ctl = 1.54, [0.0] * v.actuators.width

    def strathe(x):
        a = compute_acceleration(p, v, np.r_[u, x[:5]], np.zeros(6), x[5], x[6], None, ctl)[0]
        return np.r_[a[1:], x[2] + x[4] * math.tan(x[6]) * math.cos(x[5]), x[3] * math.cos(x[5]) - x[4] * math.sin(x[5])]

    def his(x):
        y = np.zeros(12)
        y[0], y[1:6], y[8], y[9], y[10] = u, x[:5], 30.0, x[5], x[6]
        d = PR.derivative(y, 0.0, 0.0, 1500.0)
        return np.r_[d[1:6], d[9], d[10]]

    def jac(f):
        J = np.zeros((7, 7))
        for i in range(7):
            e = np.zeros(7)
            e[i] = 1e-5
            J[:, i] = (f(e) - f(-e)) / 2e-5
        return J
    A, B = jac(strathe), jac(his)
    A[:, 2] = B[:, 2]
    assert np.abs(A - B).max() < .05 * np.abs(B).max() * .1 + .05   # a worst entry of 0.05, on entries up to 24


def test_a_stable_vehicle_says_so():
    from dataclasses import replace
    p, v = _prestero()
    s = val.fin_stability(replace(p, Muw=-40.0, Muq=-30.0, fin_lift_slope_a=6.0, fin_lift_slope_b=6.0), v, 1.5)
    assert s["doubling_s"] is None


def test_the_report_says_when_a_flight_cant_be_followed():
    from calibration_report import UNSTABLE_DOUBLING_S, fin_flight_section
    assert UNSTABLE_DOUBLING_S == val.UNSTABLE_DOUBLING_S
    m = {"median": 1.0, "p90": 2.0}
    f = {k: m for k in ("heading_deg", "pitch_deg", "depth_m", "turn_rate_rad_s")}
    run = {"window_s": 30.0, "windows": 4, "diverged": {}, "calibrated": f, "default": f, "straight_on": f}
    said = "\n".join(fin_flight_section({"fin_free_run": run, "fin_free_run_short": {**run, "window_s": 5.0}, "fin_stability": {"speed_m_s": 1.5, "doubling_s": .7}}))
    assert "unstable with its fins held" in said and "Flying 5 s" in said
    calm = "\n".join(fin_flight_section({"fin_free_run": run, "fin_stability": {"speed_m_s": 1.5, "doubling_s": None}}))
    assert "unstable" not in calm


# ------------------------------------------------------------------ each deployment flown at its own trim
from fin_force import Actuators, Fin
from objective import RunData

Y3 = Actuators(fins=(Fin("top", (-1.4, 0, -.25), (0, 0, -1), .05, .2, mix=(-1.0, 0.0, 1.0)),
                     Fin("port", (-1.4, -.22, .12), (0, -.87, .5), .05, .2, mix=(-1.0, 1.0, -.5)),
                     Fin("starboard", (-1.4, .22, .12), (0, .87, .5), .05, .2, mix=(-1.0, -1.0, -.5))))


def _held(angles_deg, date="01-01-2030", n=200, turning=False):
    """A run whose fins sit at roll, pitch and yaw angles (degrees, through the y3 mix), with steady rates unless it's turning."""
    A = np.array([f.mix for f in Y3.fins])
    ctl = np.tile(A @ np.radians(angles_deg), (n, 1)) + np.random.default_rng(1).normal(0, 1e-4, (n, 3))
    nu = np.zeros((n, 6))
    nu[:, 0] = 1.5
    if turning:
        nu[:, 5] = .1
    q = np.tile([0, 0, 0, 1.0], (n, 1))
    return RunData(f"f-{date}-run1.csv", np.zeros((n, 6)), np.zeros((n, 2)), nu, nu[0], 1.0, 1.0, date=date, orientation_quat=q, controls=ctl)


def test_a_deployment_is_flown_at_its_own_trim():
    per, notes = val.retrim_by_deployment(P, [_held([0, -3, 0])], [_held([0, -1, 0], "02-02-2031"), _held([.5, -3, .2], "03-03-2032")], Y3)
    a, b = notes["02-02-2031"], notes["03-03-2032"]
    assert a["retrimmed"] and a["pitch_deg"] == pytest.approx(-2, abs=.01) and a["roll_deg"] == pytest.approx(0, abs=.01)
    assert per[0].fin_bias_pitch == pytest.approx(np.radians(-2), abs=1e-4)
    assert b["roll_deg"] == pytest.approx(-.5, abs=.01) and b["pitch_deg"] == pytest.approx(0, abs=.01) and b["yaw_deg"] == pytest.approx(-.2, abs=.01)


def test_the_trim_of_an_axis_comes_only_from_moments_that_axis_holds_steady():
    """A deployment that turns the whole time has no steady yaw: its rudder trim isn't guessed, while its pitch trim still is."""
    _, notes = val.retrim_by_deployment(P, [_held([0, -3, 0])], [_held([0, -1, 2], turning=True)], Y3)
    n = notes["01-01-2030"]
    assert n["yaw_deg"] is None and n["pitch_deg"] == pytest.approx(-2, abs=.01) and n["retrimmed"]


def test_too_few_steady_samples_leave_the_fit_as_it_is():
    per, notes = val.retrim_by_deployment(P, [_held([0, -3, 0])], [_held([0, 5, 0], n=val.MIN_TRIM_SAMPLES - 1)], Y3)
    assert not notes["01-01-2030"]["retrimmed"] and per[0] == P


def test_the_report_says_what_trim_each_deployment_was_flown_at():
    s = {"median": 1.0, "p90": 2.0}
    f = {"window_s": 30, "windows": 5, "diverged": {"calibrated": 0},
         **{m: {k: s for k in ("heading_deg", "pitch_deg", "depth_m", "turn_rate_rad_s")} for m in ("calibrated", "default", "straight_on")}}
    txt = "\n".join(fin_flight_section({"fin_free_run": f, "fin_free_run_as_trained": f,
                                        "fin_retrim": {"02-02-2031": {"retrimmed": True, "roll_deg": .1, "pitch_deg": -2.0, "yaw_deg": None}}}))
    assert "own trim" in txt and "02-02-2031" in txt and "-2.00°" in txt and "Without re-trimming" in txt
