"""Hover or fin: what a lab sees, what a description defines, what's refused (#97). Every refusal here says what to do."""
import copy
import json
import math
import os

import numpy as np
import pandas as pd
import pytest
import yaml

import data_ingestion as di
import steering
from dynamics_model import MASS_COUPLINGS
from fin_force import Actuators
from objective import DatasetConfig, convert_run_dataframe
from optimizer import initial_parameters, load_vehicle_spec, parameter_bounds, held_by_bounds
from steering import SteeringError

CAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIN_SPEC = os.path.join(CAL, "templates", "vehicle_spec_fin_template.yaml")
HOVER_SPEC = os.path.join(CAL, "vehicle_spec.yaml")


def raw(path=FIN_SPEC):
    return yaml.safe_load(open(path, encoding="utf-8"))


def fin_raw(**kw):
    r = raw()
    r["fins"].update(kw)
    return r


def test_the_fin_template_loads_and_is_a_fin_vehicle():
    s = load_vehicle_spec(FIN_SPEC)
    assert steering.steering_of(s.raw) == "fin" and len(s.vehicle.actuators.fins) == 4 and s.vehicle.actuators.propeller
    p = initial_parameters(s)
    assert p.fin_lift_slope_a == 3.0
    assert "fin_lift_slope_a" not in held_by_bounds(*parameter_bounds(s), list(p.param_names))


def test_a_spec_that_doesnt_say_is_hover_and_the_banner_says_so():
    s = load_vehicle_spec(HOVER_SPEC)
    assert s.vehicle.actuators is None
    assert "HOVER" in steering.banner(s.raw) and "taken as hover" in steering.banner(s.raw)


@pytest.mark.parametrize("word", ["fin", "vectored", "hybrid", "hover", "glider"])
def test_every_type_has_its_words(word):
    assert steering.WORDS[word] in steering.banner({"vehicle": {"steering": word}})


def test_a_wrong_type_is_refused_with_the_choices():
    with pytest.raises(SteeringError, match="hover.*fin.*vectored"):
        steering.steering_of({"vehicle": {"steering": "torpedo"}})


@pytest.mark.parametrize("layout,n", [("cross", 4), ("x", 4), ("y3", 3), ("rudder_only", 2)])
def test_each_layout_builds_its_fins(layout, n):
    act = steering.actuators_from(fin_raw(layout=layout))
    assert len(act.fins) == n and all(abs(np.linalg.norm(f.span_axis) - 1) < 1e-12 for f in act.fins)


@pytest.mark.parametrize("layout", ["cross", "x", "y3"])
def test_positive_channels_do_what_their_names_say(layout):
    """Through the real physics: positive rudder yaws to starboard, positive elevator lifts the nose, positive aileron rolls
    to starboard, whatever the layout."""
    from dataclasses import replace
    from dynamics_model import CalibrationParameters
    from fin_force import actuator_wrench, prepare
    act = replace(steering.actuators_from(fin_raw(layout=layout)), propeller=None)
    p = CalibrationParameters(1, 1, 1, *[-1] * 6, *[-1] * 6, 1, 1, 1, 0, 0, fin_lift_slope_a=3.0, fin_lift_slope_b=3.0)
    for name, axis in (("rudder", 5), ("elevator", 4), ("aileron", 3)):
        df = pd.DataFrame({f"fin_{name}": np.full(3, .1)})
        c, cmd = steering.controls_from_frame(df, act)
        w = actuator_wrench(act, prepare(act, p), p, [1.5, 0, 0, 0, 0, 0], list(c[0]), 0, 1, 0, 1)
        assert w[axis] > 0, (layout, name, w)
        assert not cmd


def test_fin_groups_split_steering_from_pitching():
    act = steering.actuators_from(fin_raw(layout="cross"))
    assert {f.name: f.group for f in act.fins} == {"starboard": 1, "port": 1, "top": 0, "bottom": 0}


def test_a_custom_fin_list():
    r = fin_raw(layout="custom", fins=[{"name": "a", "position_m": [-.5, 0, .1], "span_axis": [0, 0, 2], "area_m2": .01, "span_m": .1}])
    f = steering.actuators_from(r).fins[0]
    assert f.span_axis == (0, 0, 1) and f.name == "a"


@pytest.mark.parametrize("change,word", [
    (lambda r: r["fins"].update(layout="star"), "isn't one of"),
    (lambda r: r["fins"].update(area_m2=0), "area_m2"),
    (lambda r: r.pop("fins"), "no fins: section"),
    (lambda r: r["vehicle"].update(steering="hover"), "steering is hover"),
])
def test_a_description_that_cant_be_right_is_refused(change, word):
    r = raw()
    change(r)
    with pytest.raises(SteeringError, match=word):
        steering.actuators_from(r)


def test_a_vectored_vehicle_needs_a_gimballed_propeller():
    r = raw()
    r["vehicle"]["steering"] = "vectored"
    r.pop("fins")
    r.pop("propeller")
    with pytest.raises(SteeringError, match="gimbal"):
        steering.actuators_from(r)


FIN_COLS = ["Fx", "fin_rudder", "rpm"]


@pytest.mark.parametrize("spec,cols,word", [
    ({"vehicle": {}}, FIN_COLS, "vehicle description is hover"),
    ({"vehicle": {"steering": "fin"}}, ["Fx", "Bu"], "no fin or propeller channels"),
    ({"vehicle": {"steering": "glider"}}, FIN_COLS, "fin-check"),
])
def test_a_description_and_logs_that_disagree_are_refused(spec, cols, word):
    with pytest.raises(SteeringError, match=word):
        steering.check_data(spec, cols)


def test_matching_description_and_logs_pass():
    steering.check_data({"vehicle": {"steering": "fin"}}, FIN_COLS)
    steering.check_data({"vehicle": {}}, ["Fx", "Bu"])


def test_a_fin_vehicle_needs_a_speed_source():
    with pytest.raises(SteeringError, match="speed source"):
        steering.check_speed_source({"vehicle": {"steering": "fin"}}, {"u": [], "v": [], "w": ["depth.d"]})
    steering.check_speed_source({"vehicle": {"steering": "fin"}}, {"u": ["dvl.u"]})
    steering.check_speed_source({"vehicle": {}}, {"u": []})


def frame(**cols):
    n = 20
    d = {"Fx": np.zeros(n), "Fy": np.zeros(n), "Fz": np.zeros(n), "Tx": np.zeros(n), "Ty": np.zeros(n), "Tz": np.zeros(n),
         "Bu": np.ones(n), "Bv": np.zeros(n), "Bw": np.zeros(n), "Bp": np.zeros(n), "Bq": np.zeros(n), "Br": np.zeros(n),
         "qx": np.zeros(n), "qy": np.zeros(n), "qz": np.zeros(n), "qw": np.ones(n)}
    d.update({k: np.full(n, v) for k, v in cols.items()})
    return pd.DataFrame(d)


def test_a_run_gets_per_fin_angles_from_rudder_and_elevator():
    act = steering.actuators_from(raw())
    r = convert_run_dataframe(frame(fin_rudder=.1, fin_elevator=.05, rpm=1000.), "a-01-01-2030-run1.csv",
                              DatasetConfig(10.0, actuators=act))
    assert r.controls.shape == (20, 5) and r.servo is None and r.controls[0, -1] == 1000.
    want = [f.mix[1] * .05 + f.mix[2] * .1 for f in act.fins]
    assert np.allclose(r.controls[0, :4], want)


def test_commanded_channels_ask_for_the_servo():
    act = steering.actuators_from(raw())
    r = convert_run_dataframe(frame(fincmd_rudder=.1, rpm=1000.), "a-01-01-2030-run1.csv", DatasetConfig(10.0, actuators=act))
    assert r.servo is act


def test_per_fin_channels_are_used_as_they_are_with_their_sign():
    r = fin_raw(layout="custom", fins=[{"name": "a", "position_m": [-.5, 0, .1], "span_axis": [0, 0, 1], "area_m2": .01, "span_m": .1, "sign": -1}])
    r.pop("propeller")
    c, _ = steering.controls_from_frame(frame(fin_a=.2), steering.actuators_from(r))
    assert c[0, 0] == -.2


def test_ganged_logs_without_an_aileron_say_its_roll_command_is_taken_as_zero(caplog, monkeypatch):
    monkeypatch.setattr(steering, "_said", set())
    act = steering.actuators_from(raw())
    with caplog.at_level("WARNING", logger="steering"):
        steering.controls_from_frame(frame(fin_rudder=.1, fin_elevator=.1, rpm=1.), act)
        steering.controls_from_frame(frame(fin_rudder=.1, fin_elevator=.1, rpm=1.), act)
    said = [r.message for r in caplog.records if "aileron" in r.message]
    assert len(said) == 1 and "taken as zero" in said[0]


@pytest.mark.parametrize("cols", [dict(fin_rudder=.1, fin_elevator=.1, fin_aileron=.0, rpm=1.)])
def test_an_aileron_or_per_fin_angles_need_no_warning(caplog, monkeypatch, cols):
    monkeypatch.setattr(steering, "_said", set())
    with caplog.at_level("WARNING", logger="steering"):
        steering.controls_from_frame(frame(**cols), steering.actuators_from(raw()))
    assert not [r for r in caplog.records if "aileron" in r.message]


@pytest.mark.parametrize("cols,word", [
    (dict(rpm=1000.), "no fin angles"),
    (dict(fin_rudder=.1), "no rpm"),
    (dict(fin_rudder=.1, fincmd_elevator=.1, rpm=1.), "both measured and commanded"),
    (dict(fin_flap=.1, rpm=1.), "aren't rudder"),
    (dict(fin_top=.1, rpm=1.), "but not"),
])
def test_a_recording_missing_what_the_vehicle_needs_is_refused(cols, word):
    with pytest.raises(SteeringError, match=word):
        steering.controls_from_frame(frame(**cols), steering.actuators_from(raw()))


# ---------------------------------------------------------------- ingest

def _ingest_spec(tmp_path, n=300, **top):
    t = np.arange(n) / 10.0
    rud = 10 * np.sin(t / 3)
    r_rate = np.radians(rud) * .5
    df = pd.DataFrame({"t": t, "u": 1.5, "v": 0., "w": 0., "p": 0., "q": 0., "r": np.r_[0, r_rate[:-1]], "roll": 0., "pitch": 0.,
                       "yaw": np.degrees(np.cumsum(r_rate) * .1), "rud": rud, "elev": 2 * np.sin(t), "rpm": 900.})
    d = tmp_path / "raw"
    d.mkdir(exist_ok=True)
    df.to_csv(d / "r.csv", index=False)
    s = {"vehicle": "x", "source": {"files": str(d / "*.csv")}, "time": {"column": "t", "unit": "s"},
         "session_date": {"from": "constant", "value": "2026-01-15"},
         "body_velocity": {"columns": dict(u="u", v="v", w="w", p="p", q="q", r="r"), "linear_unit": "m/s", "angular_unit": "rad/s"},
         "orientation": {"euler": {"roll": "roll", "pitch": "pitch", "yaw": "yaw", "unit": "deg"}},
         "fins": {"source": "measured", "channels": {"rudder": "rud", "elevator": "elev"}, "unit": "deg"},
         "propeller": {"column": "rpm"}}
    s.update(top)
    f = tmp_path / "spec.yaml"
    f.write_text(yaml.safe_dump(s))
    return str(f)


def _run(tmp_path, **top):
    spec = di.load_ingest_spec(_ingest_spec(tmp_path, **top))
    out = tmp_path / "out"
    summary = di.ingest(spec, str(out))
    files = sorted(out.rglob("*.csv"))
    return pd.read_csv(files[0]), summary


def test_fin_logs_are_prepared_in_radians_with_no_wrench_needed(tmp_path):
    df, s = _run(tmp_path)
    assert {"fin_rudder", "fin_elevator", "rpm"} <= set(df.columns)
    assert abs(df["fin_rudder"]).max() == pytest.approx(math.radians(10), rel=.01) and not df["Fx"].any()
    assert s["files"][0]["wrench_source"].startswith("none logged")


def test_the_fin_sign_check_passes_the_right_way_and_refuses_the_wrong_way(tmp_path):
    _, s = _run(tmp_path)
    assert s["files"][0]["sensor_checks"]["fin_direction"]["rudder"]["correlation"] > .5
    f = {"source": "measured", "channels": {"rudder": "rud"}, "unit": "deg", "positive_turns": "port"}
    with pytest.raises(di.IngestError, match="looks reversed.*positive_turns"):
        _run(tmp_path, fins=f)
    _, s = _run(tmp_path, fins=f, skip_checks=["fin_direction"])
    assert "problem" in s["files"][0]["sensor_checks"]["fin_direction"]["rudder"]


def test_a_fin_that_barely_moved_cant_be_checked_and_says_so(tmp_path):
    spec = di.load_ingest_spec(_ingest_spec(tmp_path))
    r = di.fin_direction(np.arange(100) / 10, np.zeros(100), np.zeros(100), np.ones(100, bool), "rudder", "positive_turns")
    assert "note" in r


@pytest.mark.parametrize("unit,extra,scale", [("rad", {}, 1.0), ("percent", {"full_scale_deg": 20.0}, math.radians(.2))])
def test_fin_units(tmp_path, unit, extra, scale):
    f = {"source": "measured", "channels": {"rudder": "elev"}, "unit": unit, **extra}
    if unit == "rad":
        with pytest.raises(di.IngestError, match="check its unit"):
            _run(tmp_path, fins={**f, "channels": {"rudder": "rud"}}, skip_checks=["fin_direction"])
        return
    df, _ = _run(tmp_path, fins=f, skip_checks=["fin_direction"])
    assert abs(df["fin_rudder"]).max() == pytest.approx(2 * scale, rel=.02)


def test_pwm_fin_angles(tmp_path):
    x = di._fin_radians(di.IngestSpec.model_construct(fins=di.FinChannels(source="commanded", channels={"rudder": "a"}, unit="pwm_us", pwm_us_per_deg=10.0)),
                        np.array([1500., 1600.]))
    assert np.allclose(x, [0, math.radians(10)])


@pytest.mark.parametrize("fins,word", [
    ({"source": "measured", "channels": {"rudder": "a"}, "unit": "percent"}, "full_scale_deg"),
    ({"source": "measured", "channels": {"rudder": "a"}, "unit": "pwm_us"}, "pwm_us_per_deg"),
    ({"source": "measured", "channels": {}}, "channels"),
    ({"source": "guessed", "channels": {"rudder": "a"}}, "source"),
])
def test_a_wrong_fins_block_is_refused(tmp_path, fins, word):
    with pytest.raises(di.IngestError, match=word):
        di.load_ingest_spec(_ingest_spec(tmp_path, fins=fins))


def test_commanded_fins_hold_between_commands_when_resampled(tmp_path):
    f = {"source": "commanded", "channels": {"rudder": "rud"}, "unit": "deg"}
    df, _ = _run(tmp_path, fins=f, resample_hz=40.0)
    x = df["fincmd_rudder"].to_numpy()
    assert (np.diff(x) == 0).mean() > .7      # stepped, not interpolated


def test_no_wrench_and_nothing_else_is_refused(tmp_path):
    p = _ingest_spec(tmp_path)
    s = yaml.safe_load(open(p))
    s.pop("fins")
    s.pop("propeller")
    open(p, "w").write(yaml.safe_dump(s))
    with pytest.raises(di.IngestError, match="wrench: is needed"):
        di.load_ingest_spec(p)


def test_a_servo_in_the_thruster_columns_is_refused(tmp_path):
    p = _ingest_spec(tmp_path, wrench={"thrusters": {"pwm_columns": ["motor_1", "rudder_servo"], "preset": "bluerov2_heavy"}})
    with pytest.raises(di.IngestError, match="looks like a fin"):
        di.load_ingest_spec(p)


def test_ardusubs_thrusters_are_not_mistaken_for_fins(tmp_path):
    p = _ingest_spec(tmp_path, wrench={"thrusters": {"pwm_columns": ["SERVO_OUTPUT_RAW.servo1_raw"], "layout": [
        {"position_m": [-.5, 0, 0], "direction": [1, 0, 0]}]}})
    assert di.load_ingest_spec(p).wrench.thrusters.pwm_columns


def test_an_unknown_key_is_refused_not_ignored(tmp_path):
    with pytest.raises(di.IngestError, match="fin_block"):
        di.load_ingest_spec(_ingest_spec(tmp_path, fin_block={}))


@pytest.mark.parametrize("name", ["ingest_fin_csv.yaml", "ingest_fin_moos.yaml", "ingest_fin_rosbag.yaml"])
def test_the_shipped_fin_templates_load(name):
    s = di.load_ingest_spec(os.path.join(CAL, "templates", name))
    assert s.fins is not None and s.propeller is not None and s.wrench is None


def test_other_actuators_are_prepared_in_si(tmp_path):
    t = {"gimbal": {"y": "elev", "z": "rud", "unit": "deg"}, "moving_mass": {"column": "elev", "unit": "mm"},
         "buoyancy_engine": {"column": "elev", "unit": "litre"}}
    df, _ = _run(tmp_path, skip_checks=["fin_direction"], **t)
    assert abs(df["gimbal_z"]).max() == pytest.approx(math.radians(10), rel=.01)
    assert abs(df["moving_mass"]).max() == pytest.approx(.002, rel=.01)
    assert abs(df["buoyancy_change"]).max() == pytest.approx(2 * 1025 * 9.81 / 1000, rel=.001)   # a litre of seawater's lift


def test_the_report_opens_with_the_vehicle_type(tmp_path):
    from calibration_report import write_report
    p = tmp_path / "r.md"
    write_report(str(p), "x", [], None, [], [], None, [], True, vehicle_type="Vehicle type: FIN-STEERED")
    assert "**Vehicle type: FIN-STEERED**" in p.read_text()


def test_the_web_form_makes_labs_choose_and_checks_the_choice(tmp_path):
    import strathe_web as w
    page = w.index_page() if hasattr(w, "index_page") else open(w.__file__, encoding="utf-8").read()
    assert "name=steering required" in page and "Hover: thrusters steer it" in page and "Fin-steered: fins and a propeller" in page
    w.steering_matches({"steering": "fin", "vehicle_spec": FIN_SPEC})
    w.steering_matches({"steering": "hover", "vehicle_spec": ""})
    with pytest.raises(ValueError, match="You chose hover"):
        w.steering_matches({"steering": "hover", "vehicle_spec": FIN_SPEC})
    with pytest.raises(ValueError, match="You chose fin"):
        w.steering_matches({"steering": "fin", "vehicle_spec": ""})


def test_the_web_form_describes_a_fin_vehicle():
    import strathe_web as w
    a = w.describe_vehicle_cli_args({"mass": "30", "length": "1.6", "diameter": ".19", "name": "x", "steering": "fin", "fins": "cross", "fin_x": "-.7",
                                     "fin_radius": ".12", "fin_area": ".013", "fin_span": ".09", "propeller_diameter": ".14", "kt0": ".45"})
    assert a[a.index("--steering") + 1] == "fin" and a[a.index("--fins") + 1] == "cross" and "--kt0" in a
