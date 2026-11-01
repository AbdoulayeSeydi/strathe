"""`strathe convert-sim` (PROJECT_STATUS.md #85): a Stonefish model's numbers written as a Gazebo world, a MuJoCo MJCF
or a HoloOcean scenario, with the origin at the logs' reference point. Uses an --info dict like the replay tool's."""
import json
import os
import tempfile

import numpy as np
import pytest

from simfit import convert

INFO = {"water_density": 1000.0,
        "base": {"mass": 27.0, "volume": 0.03, "cg": [-0.66, 0.0, -0.005], "cb": [-0.65, 0.0, -0.10],
                 "inertia_body_axes": [0.6, 66.0, 65.5]},
        "parts": [{"name": "r/V/Hull", "external": True, "quadratic_drag": [0.08, 1.0, 1.0], "viscous_drag": [0.008, 0.1, 0.1],
                   "added_mass": [10.0, 75.0, 75.0], "added_inertia": [0.0, 14.0, 14.0],
                   "projected_area": [0.05, 0.375, 0.375], "tangential_area": [1.1, 0.6, 0.6]}]}


def test_the_facts_are_relative_to_the_reference_point_and_keep_the_trim():
    f = convert.facts_from_stonefish(INFO, (-0.76, 0, 0), mass=33.0)
    assert f["cg"] == pytest.approx([0.10, 0.0, -0.005]) and f["cb"] == pytest.approx([0.11, 0.0, -0.10])
    assert f["mass"] == 33.0 and f["volume"] == pytest.approx(0.03 + 6.0 / 1000)  # the model's trim kept
    assert f["net_buoyancy"] == pytest.approx(1000 * 9.81 * 0.03 - 27 * 9.81)
    assert f["inertia_rigid"][1] == pytest.approx((66.0 - 14.0) * 33 / 27)
    assert any("trim" in n for n in f["notes"])


def test_each_converted_model_loads_in_its_adapter(tmp_path, monkeypatch):
    from simfit.adapters.gazebo import GazeboAdapter
    from simfit.adapters.mujoco import MujocoAdapter
    neutral = {**INFO, "base": {**INFO["base"], "volume": 0.027}}  # 27 kg displaced: trimmed neutral
    f = convert.facts_from_stonefish(neutral, (-0.76, 0, 0), mass=33.0)
    g = GazeboAdapter(convert.write_gazebo(f, str(tmp_path)), tool=["unused"])
    assert "lift" in g.groups and "added_mass" in g.groups and g.link == "base_link"
    m = MujocoAdapter(convert.write_mujoco(f, str(tmp_path)))
    assert m.buoyancy_path == "//body[@name='buoyancy']"
    d = tempfile.mkdtemp()
    rows = [np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]] * 51, float)]
    (V, Q), = m.replay(m.write_runnable(None, d), rows, d)
    assert np.isfinite(V).all() and abs(V[-1, 2]) < 0.02  # trimmed neutral: no rise or fall
    monkeypatch.syspath_prepend(os.path.join(os.path.dirname(__file__), "fake_holoocean"))
    from simfit.adapters.holoocean import HolooceanAdapter
    h = HolooceanAdapter(convert.write_holoocean(f, str(tmp_path)))
    dyn = json.load(open(h.model_path))["agents"][0]["dynamics"]
    assert dyn["mass"] == 33.0 and dyn["r_bg"] == pytest.approx([0.10, 0.0, -0.005]) and dyn["T_surge"] > 0


def test_strathe_convert_sim_passes_options_through(tmp_path, monkeypatch):
    import strathe_cli
    monkeypatch.chdir(tmp_path)
    (tmp_path / "w.scn").write_text("<scenario/>")
    calls = []
    assert strathe_cli.main(["convert-sim", "--to", "gazebo", "--scenario", "w.scn", "--robot", "r",
                             "--reference-point", "0", "0", "0", "--out", "o"], runner=lambda c: calls.append(c) or 0) == 0
    assert calls[0][1:3] == ["-m", "simfit.convert"] and str(tmp_path / "w.scn") in calls[0]
