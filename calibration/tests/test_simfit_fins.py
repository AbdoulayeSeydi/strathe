"""Fins in every simulator path (#97): what's sent, what's refused, what's written. Real engines run in stress/run_97.py."""
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import replace

import numpy as np
import pytest
import yaml

import steering
from fin_force import Actuators, Propeller, other_wrench
from objective import RunData
from simfit import core
from simfit import fin_models as fm
from simfit.adapters.holoocean import _fin_slots, strathe_fossen_source
from simfit.replay_tool import IN_COLUMNS, ReplayTool, write_windows
from test_fin_pipeline import ACT, P, V, recording

CAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = yaml.safe_load(open(os.path.join(CAL, "templates", "vehicle_spec_fin_template.yaml"), encoding="utf-8"))
FULL = steering.actuators_from(RAW)


def run_with(act):
    r = recording()
    n = r.n_samples
    c = np.c_[np.tile(r.controls[:, :1], (1, len(act.fins))), np.full(n, 1200.0)] if act.propeller else r.controls
    return replace(r, controls=c, actuators=act)


def test_hover_windows_are_exactly_as_before():
    r = replace(recording(), actuators=None)
    rows = core.window_rows(r, [(0, 20)])
    assert type(rows) is list and rows[0].shape == (20, 16) and not hasattr(rows, "fins")


def test_fin_windows_carry_each_fin_after_the_wrench():
    r = run_with(FULL)
    rows = core.window_rows(r, [(0, 20), (30, 10)])
    assert rows.fins == tuple(f"fin_{f.name}" for f in FULL.fins) and rows[0].shape == (20, 20)
    assert np.array_equal(rows[1][:, 16:], r.controls[30:40, :4])


def test_the_propeller_is_folded_into_the_wrench():
    r = run_with(FULL)
    rows = core.window_rows(r, [(0, 20)])
    assert rows[0][0, 10] > r.forces_torques[0, 0] + 1.0


def test_other_wrench_is_zero_for_fins_alone():
    r = recording()
    assert not other_wrench(ACT, r.real_nu, r.roll_pitch, r.controls).any()


def test_the_window_file_names_the_fins(tmp_path):
    rows = core.window_rows(run_with(FULL), [(0, 3)])
    p = tmp_path / "w.csv"
    write_windows(str(p), rows)
    head = p.read_text().splitlines()
    assert head[0] == IN_COLUMNS + ",fin_starboard,fin_bottom,fin_port,fin_top"
    assert len(head[1].split(",")) == 22


def test_a_tool_without_fins_is_refused_with_how_to_fix_it(tmp_path):
    from simfit.core import SimFitError
    t = ReplayTool(["unused"], (0, 0, 0))
    t._fins = False
    rows = core.window_rows(run_with(FULL), [(0, 3)])
    with pytest.raises(SimFitError, match="doesn't take fin angles"):
        t.run("m", rows, str(tmp_path))


def test_fin_logs_need_a_description():
    from simfit.core import SimFitError
    with pytest.raises(SimFitError, match="give --spec"):
        core.actuators_of(None, ["Fx", "fin_rudder"])
    assert core.actuators_of(None, ["Fx", "Bu"]) is None


def test_a_description_that_disagrees_with_the_logs_is_refused(tmp_path):
    from simfit.core import SimFitError
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(RAW))
    with pytest.raises(SimFitError, match="no fin or propeller channels"):
        core.actuators_of(str(p), ["Fx", "Bu"])
    assert core.actuators_of(str(p), ["fin_rudder", "rpm"]).fins


def test_the_example_tool_applies_fin_columns(tmp_path):
    tool = os.path.join(CAL, "simulators", "example", "example_replay.py")
    model = tmp_path / "v.json"
    base = json.load(open(os.path.join(CAL, "simulators", "example", "vehicle.json")))
    base["fins"] = [{"name": "top", "position": [-.7, 0, 0], "normal": [0, 1, 0], "k": 3.0}]
    model.write_text(json.dumps(base))
    f = tmp_path / "in.csv"
    lines = [IN_COLUMNS + ",fin_top"] + [",".join(str(x) for x in (0, k, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, .2)) for k in range(11)]
    f.write_text("\n".join(lines) + "\n")
    out = tmp_path / "o.csv"
    subprocess.run([sys.executable, tool, "--model", str(model), "--dt", ".1", "--windows", str(f), "--out", str(out)], check=True)
    last = out.read_text().strip().splitlines()[-1].split(",")
    assert float(last[3]) > 0 and float(last[7]) < 0     # pushed to starboard at the stern: sways out, yaws the nose to port
    assert '"fins": true' in subprocess.run([sys.executable, tool, "--features"], capture_output=True, text=True).stdout


class _Fin:
    def __init__(self, deg):
        self.angle_rad = np.radians(deg)


class _Veh:
    controls = [_Fin(0), _Fin(90), _Fin(180), _Fin(270), object()]


def test_holooceans_fins_are_matched_by_where_they_stand():
    slots = dict(_fin_slots(_Veh(), FULL))
    names = [f.name for f in FULL.fins]
    want = {"starboard": 0, "bottom": 1, "port": 2, "top": 3}
    assert {names[j]: i for i, j in slots.items()} == want


def test_a_fin_with_no_counterpart_is_refused():
    from simfit.core import SimFitError

    class Three:
        controls = [_Fin(0), _Fin(120), _Fin(240), object()]
    with pytest.raises(SimFitError, match="no fin near it"):
        _fin_slots(Three(), FULL)


def test_a_class_with_no_fins_takes_them_in_order():
    assert _fin_slots(object(), ACT) == [(0, 0), (1, 1)]


def test_strathes_fossen_class_carries_the_fins_and_matches_the_physics():
    from dynamics_model import compute_acceleration
    ns = {}
    exec(strathe_fossen_source(P, 30.0, [0, 0, 0], ACT), ns)
    v = ns["StratheFossen"]()
    assert v.dimU == 3 and len(ns["FINS"]) == 2
    nu, eta, c = np.array([1.5, .05, .02, .01, .02, .03]), np.zeros(6), [.1, -.05, 0]
    got = v.dynamics(eta, nu, None, np.array(c), .01)[0]
    assert np.allclose(got, compute_acceleration(P, V, nu, np.zeros(6), 0.0, 0.0, None, c[:2])[0], rtol=1e-8, atol=1e-9)


def test_gazebo_fins_are_valid_and_turn_about_their_span():
    links, plugins = fm.sdf_fins(FULL, "base_link", 1025.0)
    root = ET.fromstring(f"<model>{links}{plugins}</model>")
    assert [j.get("name") for j in root.findall("joint")] == [f"fin_{f.name}" for f in FULL.fins]
    top = next(j for j in root.findall("joint") if j.get("name") == "fin_top")
    assert [float(x) for x in top.find("axis/xyz").text.split()] == pytest.approx([0, 0, 1])    # up in forward-left-up
    assert len(root.findall("plugin")) == 4


def test_gazebos_lift_knob_finds_every_fin():
    from simfit.adapters.gazebo import LIFTDRAG_CLA
    _, plugins = fm.sdf_fins(FULL, "base_link", 1025.0)
    assert len(LIFTDRAG_CLA.findall(plugins)) == 4
    assert LIFTDRAG_CLA.sub(lambda m: m.group(1) + "9" + m.group(3), plugins).count("<cla>9</cla>") == 4


def test_mujoco_fins_are_valid_and_named_for_the_replay():
    root = ET.fromstring(f"<body>{fm.mjcf_fins(FULL)}</body>")
    assert sorted(j.get("name") for j in root.iter("joint")) == sorted(f"fin_{f.name}" for f in FULL.fins)


def test_stonefish_rudders_are_valid_and_tunable():
    from stonefish_fit import PartSettings, RUDDER_LIFT, apply_part_settings
    text = fm.stonefish_fins(FULL, "Hull")
    assert len(ET.fromstring(f"<r>{text}</r>").findall("actuator")) == 4 and len(RUDDER_LIFT.findall(text)) == 1 + 3
    out = apply_part_settings(text, "none", PartSettings(fin_lift=7.5))
    assert out.count('lift_coeff="7.5"') == 4
