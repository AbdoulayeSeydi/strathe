"""A fin-steered vehicle through the whole calibration (#97): the fit finds fin lift, what it can't see is held, and the report
says how far to trust it. Recovering a vehicle from an independent truth is stress/run_97.py's: this is the plumbing."""
import json
import os
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial.transform import Rotation

import fin_fit
import steering
import vehicle_describe as vd
from dynamics_model import SimulationConfig, TrajectoryInputs, simulate_trajectory
from optimizer import CalibrationConfig, initial_parameters, load_vehicle_spec
from run_calibration import PipelineConfig, run_pipeline

CAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(CAL, "templates", "vehicle_spec_fin_template.yaml")
DATES = ("01-10-2026", "02-10-2026")
SPEC = load_vehicle_spec(TEMPLATE)
TRUTH = replace(initial_parameters(SPEC), fin_lift_slope_a=2.2, fin_lift_slope_b=3.6, cg_cb_offset_z=.02, net_buoyancy_z=0.0)


def recording(seed, n=600, swing=(.15, .08), steps=30):
    rng = np.random.default_rng(seed)
    hold = max(n // steps, 1)
    rud = np.repeat(rng.uniform(-swing[0], swing[0], n // hold + 1), hold)[:n]
    ele = np.repeat(rng.uniform(-swing[1], swing[1], n // hold + 1), hold)[:n]
    df = pd.DataFrame({"fin_rudder": rud, "fin_elevator": ele, "rpm": np.full(n, 1400.0)})
    ctl, _ = steering.controls_from_frame(df, SPEC.vehicle.actuators)
    rp = np.c_[.03 * np.sin(np.arange(n) / 40), .05 * np.sin(np.arange(n) / 60)]
    traj = TrajectoryInputs(nu0=np.array([1.3, 0, 0, 0, 0, 0.]), forces_torques=np.zeros((n, 6)), roll_pitch_series=rp, control_series=ctl)
    nu = simulate_trajectory(TRUTH, SPEC.vehicle, SimulationConfig(dt=.1), traj).nu
    q = Rotation.from_euler("xyz", np.c_[rp, np.cumsum(nu[:, 5]) * .1]).as_quat()
    for i, c in enumerate(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")):
        df[c] = 0.0
    for i, c in enumerate(("Bu", "Bv", "Bw", "Bp", "Bq", "Br")):
        df[c] = nu[:, i]
    for i, c in enumerate(("qx", "qy", "qz", "qw")):
        df[c] = q[:, i]
    return df


def dataset(d, **kw):
    d.mkdir(parents=True, exist_ok=True)
    k = 0
    for date in DATES:
        for r in range(2):
            recording(k + 1, **kw).to_csv(d / f"fin-{date}-run{r + 1}.csv", index=False)
            k += 1
    return d


def calibrate(data, out, nfev=12, spec=TEMPLATE):
    cc = CalibrationConfig(max_nfev=nfev, seed=3, initialization_strategy="fixed", ftol=1e-4, xtol=1e-4, gtol=1e-4,
                           x0_override=initial_parameters(load_vehicle_spec(spec)).to_vector(), horizon_samples=50)
    cfg = PipelineConfig(data_dir=str(data), spec_path=str(spec), output_dir=str(out), sampling_rate_hz=10.0, heldout_dates=[DATES[1]],
                         heldout_fraction=0.5, calibration_config=cc, workers=1, torque_reference_offset_m=(0.0, 0.0, 0.0), fin_flights=False)
    try:
        return run_pipeline(cfg)
    except Exception as e:
        if "converge" not in str(e):
            raise
        return None


@pytest.fixture(scope="module")
def fitted(tmp_path_factory):
    d = tmp_path_factory.mktemp("fin")
    out = d / "out"
    calibrate(dataset(d / "data"), out)
    return out


def test_the_fit_moves_fin_lift_toward_the_truth(fitted):
    p = yaml.safe_load(open(fitted / "calibrated_parameters.yaml"))
    start = initial_parameters(SPEC)
    a, b = p["parameters"]["fin_lift_slope_a"], p["parameters"]["fin_lift_slope_b"]
    assert abs(a - TRUTH.fin_lift_slope_a) < abs(start.fin_lift_slope_a - TRUTH.fin_lift_slope_a)
    assert abs(b - TRUTH.fin_lift_slope_b) < abs(start.fin_lift_slope_b - TRUTH.fin_lift_slope_b)


def test_outputs_say_what_kind_of_vehicle_it_is(fitted):
    assert yaml.safe_load(open(fitted / "calibrated_parameters.yaml"))["vehicle_type"] == "fin"


def test_metrics_record_what_the_logs_covered(fitted):
    m = json.load(open(fitted / "metrics.json"))["fins"]
    env = m["envelope"]
    assert env["speed_mps"][0] > .3 and env["groups"]["steering"]["manoeuvres"] >= fin_fit.MIN_FIN_MANOEUVRES
    assert not m["held"] and not m["few_manoeuvres"]


def test_a_group_of_fins_that_barely_moved_keeps_its_geometry_value(tmp_path):
    out = tmp_path / "o"
    calibrate(dataset(tmp_path / "d", swing=(.15, .005)), out, nfev=6)
    m = json.load(open(out / "metrics.json"))["fins"]
    assert set(m["held"]) == {"fin_lift_slope_b", "fin_inflow_b"}
    p = yaml.safe_load(open(out / "calibrated_parameters.yaml"))["parameters"]
    assert p["fin_lift_slope_b"] == initial_parameters(SPEC).fin_lift_slope_b


def test_few_manoeuvres_are_warned_about(tmp_path):
    out = tmp_path / "o"
    calibrate(dataset(tmp_path / "d", steps=4), out, nfev=4)
    m = json.load(open(out / "metrics.json"))["fins"]
    assert "steering" in m["few_manoeuvres"]


def test_the_report_has_a_fins_section():
    from calibration_report import fins_section
    env = {"speed_mps": [1.2, 1.5], "turn_rate_rad_s": .1, "groups": {"steering": {"swing_deg": 5, "angle_deg": [-9, 9], "manoeuvres": 4}}}
    L = "".join(fins_section({"fins": {"plain": fin_fit.envelope_words(env), "few_manoeuvres": ["steering"], "held_why": {"b": "the pitching fins moved only 1.0°"}, "envelope": env}}))
    assert "Only 4 steering manoeuvres" in L and "aren't supported" in L and "keeps its geometry value" in L and "inertia" in L
    assert fins_section({}) == []


def test_a_fin_vehicle_with_no_speed_source_is_refused_before_the_fit():
    with pytest.raises(steering.SteeringError, match="speed source"):
        steering.check_speed_source(SPEC.raw, {"u": [], "v": [], "w": ["depth.d"]})


def test_manoeuvres_are_counted_swings_not_noise():
    t = np.arange(1000)
    assert fin_fit.manoeuvres(np.sign(np.sin(t / 10))) > 20
    assert fin_fit.manoeuvres(np.r_[np.zeros(500), np.ones(500)]) == 1
    assert fin_fit.manoeuvres(np.zeros(100)) == 0


def test_lift_slope_grows_with_aspect_ratio_and_stays_below_2pi():
    s = [fin_fit.lift_slope(a) for a in (.5, 1, 2, 4, 20)]
    assert s == sorted(s) and s[-1] < 2 * np.pi


def test_held_couplings_follow_the_fins_and_the_centre_of_gravity():
    act = SPEC.vehicle.actuators
    v = fin_fit.held_couplings(act.fins, 1025.0, 30.0, .02)
    assert v["Xq_dot"] == pytest.approx(.6) and v["Yp_dot"] == pytest.approx(-.6)
    assert v["Yr_dot"] < 0 and "Xv_dot" not in v         # fins aft of the centre: sway and yaw oppose
    assert set(v) <= {"Xq_dot", "Xw_dot", "Yp_dot", "Yr_dot", "Zq_dot", "Kr_dot"}


def _args(**kw):
    a = vd.build_parser().parse_args(["--mass", "30", "--length", "1.6", "--diameter", ".19", "--out", kw.pop("out")])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


FIN_ARGS = dict(steering="fin", fins="cross", fin_x=-.7, fin_radius=.12, fin_area=.013, fin_span=.09, propeller_diameter=.14, kt0=.45)


def test_describe_vehicle_writes_a_fin_vehicle_that_loads(tmp_path):
    out = tmp_path / "s.yaml"
    assert vd.run(_args(out=str(out), **FIN_ARGS)) == 0
    s = load_vehicle_spec(str(out))
    assert steering.steering_of(s.raw) == "fin" and len(s.vehicle.actuators.fins) == 4
    t = s.flat_calibration_targets
    assert t["fin_lift_slope_a"]["bounds"][0] < t["fin_lift_slope_a"]["initial_guess"] < t["fin_lift_slope_a"]["bounds"][1]
    assert t["Yr_dot"]["bounds"][0] == t["Yr_dot"]["bounds"][1]       # held


@pytest.mark.parametrize("change,word", [(dict(fins=None), "needs the fins"), (dict(fin_area=None), "--fin-area"), (dict(kt0=None), "--kt0")])
def test_describe_vehicle_says_what_a_fin_vehicle_needs(tmp_path, change, word):
    a = _args(out=str(tmp_path / "s.yaml"), **{**FIN_ARGS, **change})
    with pytest.raises(vd.DescribeError, match=word):
        vd.run(a)


def test_a_hover_description_is_unchanged_by_the_new_options(tmp_path):
    a = tmp_path / "a.yaml"
    vd.run(_args(out=str(a)))
    r = yaml.safe_load(open(a))
    assert "fins" not in r and "steering" not in r["vehicle"]


def test_a_fin_fit_that_runs_out_of_evaluations_can_still_be_finished():
    """#97, J-cli-fin: finishing a fin fit from its checkpoint crashed. A name imported inside finalize() was used before that line."""
    import ast
    import finalize_checkpoint as fc
    tree = ast.parse(open(fc.__file__, encoding="utf-8").read())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "finalize")
    local = {a.asname or a.name for n in ast.walk(fn) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    assert "_jsonable" not in local and callable(fc._jsonable)


def _slocum_log(path, sign=1.0, n=400, dt=4.0):
    """A dbd2asc file: a rudder swinging, a heading that follows it, and a stretch at the surface that must not count."""
    t = 1.65e9 + dt * np.arange(n)
    fin = .4 * np.sin(2 * np.pi * np.arange(n) / 40)
    hdg = (1.0 + np.cumsum(sign * .05 * fin * dt)) % (2 * np.pi)
    depth = np.where(np.arange(n) < 20, .5, 50.0)
    head = ["dbd_label: DBD_ASC(dinkum_binary_data_ascii)file", "encoding_ver: 2", "num_ascii_tags: 3",
            "m_present_time m_fin c_fin m_heading m_depth", "timestamp rad rad rad m", "8 4 4 4 4"]
    body = [f"{a:.3f} {b:.4f} {b:.4f} {c:.4f} {d:.1f}" for a, b, c, d in zip(t, fin, hdg, depth)]
    path.write_text("\n".join(head + body) + "\n")


def test_fin_check_reads_a_slocum_log_as_it_comes(tmp_path):
    import fin_check
    _slocum_log(tmp_path / "a.asc")
    parts, hz = fin_check.read_slocum(str(tmp_path / "a.asc"))
    assert hz == pytest.approx(.25) and sum(len(p) for p in parts) < 385      # the surface stretch is out
    r = fin_check.check_folder(str(tmp_path))["channels"]["rudder"]
    assert r["correlation"] > .9 and "problem" not in r


def test_fin_check_says_when_a_slocum_rudder_runs_the_other_way(tmp_path):
    import fin_check
    _slocum_log(tmp_path / "a.asc", sign=-1.0)
    r = fin_check.check_folder(str(tmp_path))["channels"]["rudder"]
    assert r["correlation"] < -.9 and "reversed" in r["problem"]


def test_fin_check_refuses_a_text_file_that_isnt_a_slocum_log(tmp_path):
    import fin_check
    (tmp_path / "a.asc").write_text("hello\n")
    with pytest.raises(ValueError, match="num_ascii_tags"):
        fin_check.check_folder(str(tmp_path))


def test_a_fin_fit_uses_short_windows_and_a_slow_log_still_gets_a_few_samples():
    import fin_fit
    run = lambda hz: type("R", (), {"sample_rate_hz": hz})()
    assert fin_fit.window_samples([run(10.0)]) == 20
    assert fin_fit.window_samples([run(1.0)]) == fin_fit.FIN_MIN_WINDOW_SAMPLES


def test_a_fin_fit_left_to_its_default_reports_the_window_it_used(tmp_path):
    cc = CalibrationConfig(max_nfev=1, seed=3, initialization_strategy="fixed", x0_override=initial_parameters(load_vehicle_spec(TEMPLATE)).to_vector())
    cfg = PipelineConfig(data_dir=str(dataset(tmp_path / "data")), spec_path=str(TEMPLATE), output_dir=str(tmp_path / "out"), sampling_rate_hz=10.0, heldout_dates=[DATES[1]],
                         heldout_fraction=0.5, calibration_config=cc, workers=1, torque_reference_offset_m=(0.0, 0.0, 0.0), fin_flights=False)
    try:
        run_pipeline(cfg)
    except Exception as e:
        if "converge" not in str(e):
            raise
    assert cc.horizon_samples == 20


def _flight_run(tmp_path, **kw):
    from objective import DatasetConfig, load_run
    spec = load_vehicle_spec(TEMPLATE)
    f = tmp_path / "fin-01-10-2026-run1.csv"
    recording(1, **kw).to_csv(f, index=False)
    dc = DatasetConfig(10.0, (0.0, 0.0, 0.0), None, None, sensors=("velocity",), actuators=spec.vehicle.actuators)
    return load_run(str(f), dc), spec


def test_flights_score_a_wrong_model_and_only_in_the_second_stage(tmp_path):
    import fin_fit
    from objective import residuals_for_run
    run, spec = _flight_run(tmp_path)
    p = TRUTH
    plain = residuals_for_run(p, run, spec.vehicle, horizon_samples=20)
    assert fin_fit.plan_flights([run]) > 0
    both = residuals_for_run(p, run, spec.vehicle, horizon_samples=20)
    assert len(both) > len(plain) and np.array_equal(both[:len(plain)], plain)
    wrong = replace(p, fin_lift_slope_a=p.fin_lift_slope_a * 2)
    worse = residuals_for_run(wrong, run, spec.vehicle, horizon_samples=20)[len(plain):]
    # heading: this recording's roll and pitch are made up, its heading is the model's own
    assert np.sum(worse[2::3] ** 2) > 2 * np.sum(both[len(plain):][2::3] ** 2)
    fin_fit.clear_flights([run])
    assert len(residuals_for_run(p, run, spec.vehicle, horizon_samples=20)) == len(plain)


def test_a_flight_that_diverges_is_a_bad_score_not_a_crash(tmp_path, monkeypatch):
    import fin_fit
    import validation
    run, spec = _flight_run(tmp_path)
    fin_fit.plan_flights([run])
    monkeypatch.setattr(validation, "free_run", lambda *a, **k: None)
    r = fin_fit.flight_residuals(initial_parameters(spec), run, spec.vehicle, run.__dict__["_flights"])
    assert np.isfinite(r).all() and np.abs(r).max() > 0


def test_a_recording_too_short_to_fly_has_no_flights(tmp_path):
    import fin_fit
    run, _ = _flight_run(tmp_path)
    short = replace(run, forces_torques=run.forces_torques[:30], roll_pitch=run.roll_pitch[:30], real_nu=run.real_nu[:30],
                    orientation_quat=run.orientation_quat[:30], controls=run.controls[:30])
    assert fin_fit.plan_flights([short]) == 0 and "_flights" not in short.__dict__


def _stage_config(tmp_path, nfev=2):
    cc = CalibrationConfig(max_nfev=nfev, seed=3, initialization_strategy="fixed", x0_override=initial_parameters(load_vehicle_spec(TEMPLATE)).to_vector())
    return PipelineConfig(data_dir=str(dataset(tmp_path / "data")), spec_path=str(TEMPLATE), output_dir=str(tmp_path / "out"), sampling_rate_hz=10.0, heldout_dates=[DATES[1]],
                          heldout_fraction=0.5, calibration_config=cc, workers=1, torque_reference_offset_m=(0.0, 0.0, 0.0))


def test_the_second_stage_runs_even_when_the_first_ran_out_of_budget(tmp_path, monkeypatch):
    import run_calibration as rc
    seen = {}
    real = rc.fin_flight_stage

    def spy(config, spec, runs, first, info, stopped=None):
        seen.update(first=first, stopped=stopped)
        return real(config, spec, runs, first, info, stopped)
    monkeypatch.setattr(rc, "fin_flight_stage", spy)
    try:
        run_pipeline(_stage_config(tmp_path, nfev=1))
    except Exception as e:
        if "converge" not in str(e):
            raise
    assert "stopped" in seen and (seen["first"] is not None or seen["stopped"] is not None)


def test_the_second_stage_starts_its_checkpoint_afresh(tmp_path, monkeypatch):
    """Its cost has flights in it, so it can't be compared with the first stage's: a shared best would keep the first stage's values."""
    import run_calibration as rc
    made = []
    monkeypatch.setattr(rc, "make_checkpoint_callback", lambda p: made.append(p) or (lambda *a: None))
    monkeypatch.setattr(rc, "calibrate", lambda spec, runs, cc: type("R", (), {"cc": cc, "n_iterations": 1})())
    monkeypatch.setattr("fin_fit.plan_flights", lambda runs: 3)
    cfg = _stage_config(tmp_path)
    cfg.checkpoint_path = str(tmp_path / "checkpoint.yaml")
    cfg.calibration_config.on_iteration = lambda *a: None
    first = type("F", (), {"parameters": initial_parameters(load_vehicle_spec(TEMPLATE))})()
    second = rc.fin_flight_stage(cfg, None, [], first, {"plain": []}).cc
    assert made == [cfg.checkpoint_path] and second.on_iteration is not cfg.calibration_config.on_iteration
    assert second.max_nfev <= fin_fit.FLIGHT_MAX_NFEV and second.n_restarts == 1 and second.ftol == cfg.calibration_config.ftol


def test_a_budget_stopped_first_stage_with_nothing_to_fly_stops_as_it_would_have(tmp_path):
    import run_calibration as rc
    stopped = RuntimeError("budget")
    with pytest.raises(RuntimeError, match="budget"):
        rc.fin_flight_stage(_stage_config(tmp_path), None, [type("R", (), {"sample_rate_hz": 10.0, "orientation_quat": None, "n_samples": 5})()],
                            None, {"plain": []}, stopped)


def test_a_flight_stage_that_uses_its_budget_keeps_the_fit_and_says_so(tmp_path, monkeypatch):
    """A converged first stage mustn't turn into a failed fit because the short second stage stopped at its budget."""
    import run_calibration as rc
    from optimizer import RestartResult, RestartsFailed
    x = initial_parameters(load_vehicle_spec(TEMPLATE)).to_vector()
    stop = RestartResult(x0=x, x=x * 1.0, cost=1.0, success=False, status=0, nfev=25, message="budget", raw_result=None)

    def out_of_budget(spec, runs, cc):
        raise RestartsFailed(1, [stop])
    monkeypatch.setattr(rc, "calibrate", out_of_budget)
    monkeypatch.setattr("fin_fit.plan_flights", lambda runs: 3)
    run, spec = _flight_run(tmp_path)
    first = type("F", (), {"parameters": initial_parameters(spec)})()
    info = {"plain": []}
    got = rc.fin_flight_stage(_stage_config(tmp_path), spec, [run], first, info)
    assert got.cost == 1.0 and info["flight_stage"] == {"converged": False, "nfev": 25}


def test_when_neither_stage_converged_the_fit_finishes_from_its_checkpoint(tmp_path, monkeypatch):
    import run_calibration as rc
    from optimizer import RestartResult, RestartsFailed
    run, spec = _flight_run(tmp_path)
    x = initial_parameters(spec).to_vector()
    stop = RestartResult(x0=x, x=x, cost=1.0, success=False, status=0, nfev=25, message="budget", raw_result=None)

    def out_of_budget(spec, runs, cc):
        raise RestartsFailed(1, [stop])
    monkeypatch.setattr(rc, "calibrate", out_of_budget)
    monkeypatch.setattr("fin_fit.plan_flights", lambda runs: 3)
    with pytest.raises(RestartsFailed):
        rc.fin_flight_stage(_stage_config(tmp_path), spec, [run], None, {"plain": []}, RestartsFailed(1, [stop]))


def test_flights_split_across_workers_give_the_same_residuals(tmp_path):
    import fin_fit
    from objective import ParallelJointResiduals, joint_residuals
    run, spec = _flight_run(tmp_path)
    fin_fit.plan_flights([run])
    x = TRUTH.to_vector()
    serial = joint_residuals(x, [run], spec.vehicle, None, True, None, 20)
    pool = ParallelJointResiduals([run], spec.vehicle, 2)
    try:
        assert np.array_equal(pool(x, [run], spec.vehicle, None, True, None, 20), serial)
    finally:
        pool.close()
