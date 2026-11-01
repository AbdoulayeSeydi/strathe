"""`strathe fit-sim --sim external` / `export-sim` (PROJECT_STATUS.md #81): any simulator, described by the lab in a
YAML, through the replay contract. Uses tests/fake_external_replay.py as the lab's simulator."""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from simfit import cli
from simfit.core import SimFitError
from simfit.editors import open_model
from simfit.knobs import KnobSpec, Knobs
from simfit.replay_tool import ReplayTool, docker_argv, join_command, split_command

FAKE = os.path.join(os.path.dirname(__file__), "..", "simulators", "example", "example_replay.py")
TRUTH = {"mass": 30.0, "added_mass": [6.0, 40.0, 40.0], "inertia": [1.0, 4.0, 4.0],
         "linear_damping": [-5.0, -20.0, -20.0, -1.0, -3.0, -3.0],
         "quadratic_damping": [-40.0, -150.0, -150.0, -2.0, -8.0, -8.0], "water": {"density": 1000.0}}
WRONG = {**TRUTH, "added_mass": [2.0, 15.0, 15.0], "quadratic_damping": [-12.0, -60.0, -60.0, -2.0, -8.0, -8.0]}


def _model(path, d):
    path.write_text(json.dumps(d, indent=2) + "\n")
    return str(path)


def _logs(tmp_path, seconds=120, dates=("01-01-2030", "02-01-2030")):
    truth = _model(tmp_path / "truth.json", TRUTH)
    tool = ReplayTool([sys.executable, FAKE], (0.0, 0.0, 0.0))
    data = tmp_path / "logs"
    data.mkdir()
    rng = np.random.default_rng(0)
    n = int(seconds / 0.1) + 1
    for i, d in enumerate(dates):
        F = np.repeat(rng.normal(0, [25, 20, 20, 0.5, 0.5, 0.5], size=(n // 40 + 1, 6)), 40, axis=0)[:n]
        start = np.concatenate([[0.3, 0, 0, 0, 0, 0], [0, 0, 0, 1], F[0]])
        rows = [np.vstack([np.hstack([start[:10], f]) for f in F])]
        (V, Q), = tool.run(truth, rows, str(tmp_path / f"gen{i}"))
        pd.DataFrame(np.hstack([F, V, Q]), columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp", "Bq",
                                                    "Br", "qx", "qy", "qz", "qw"]).to_csv(data / f"auv-{d}-run1.csv",
                                                                                         index=False)
    return str(data)


def _description(tmp_path, model, knobs=None, export=None):
    d = {"simulator": "My Sim", "replay": f"{sys.executable} {FAKE}", "model": os.path.basename(model),
         "density": "/water/density",
         "knobs": knobs or [{"name": f"quadratic_drag_{a}", "group": "drag", "at": f"/quadratic_damping/{i}",
                             "sign": -1, "axisym": "q_yz" if a in "yz" else None} for i, a in enumerate("xyz")]
         + [{"name": f"added_mass_{a}", "group": "added_mass", "at": f"/added_mass/{i}",
             "axisym": "a_yz" if a in "yz" else None} for i, a in enumerate("xyz")]}
    if export:
        d["export"] = export
    p = tmp_path / "my_sim.yaml"
    p.write_text(yaml.safe_dump(d, sort_keys=False))
    return str(p)


def test_fit_sim_calibrates_any_simulator_from_its_description(tmp_path, capsys):
    data = _logs(tmp_path)
    model = _model(tmp_path / "vehicle.json", WRONG)
    desc = _description(tmp_path, model)
    out = tmp_path / "out"
    rc = cli.main(["fit", "--sim", "external", "--simulator", desc, "--data", data, "--reference-point", "0", "0", "0",
                   "--heldout-dates", "02-01-2030", "--max-iterations", "8", "--workers", "2", "--axisymmetric",
                   "--out", str(out)])
    assert rc == 0, capsys.readouterr().out
    s = yaml.safe_load((out / "my_sim_fit.yaml").read_text())
    held = s["heldout"]
    orig, cal = held["My Sim, original"]["30s"], held["My Sim, calibrated by Strathe"]["30s"]
    assert cal["drift_median_m"] < 0.6 * orig["drift_median_m"]
    after = s["after"]
    assert abs(np.log(after["quadratic_drag_x"] / 40.0)) < abs(np.log(12.0 / 40.0))  # toward the truth
    assert after["quadratic_drag_y"] == after["quadratic_drag_z"]  # --axisymmetric ties sway and heave
    cal_model = json.loads((out / "vehicle_calibrated.json").read_text())
    assert cal_model["quadratic_damping"][0] == pytest.approx(-after["quadratic_drag_x"])
    assert cal_model["linear_damping"] == WRONG["linear_damping"] and cal_model["mass"] == 30.0  # nothing else moved
    report = (out / "my_sim_fit_report.md").read_text()
    assert "# My Sim model calibrated from your logs" in report and "keeps its starting velocity" in report
    assert "quadratic_drag_x" in report and (out / "drift.png").exists()


def test_export_sim_puts_strathes_values_in_exactly(tmp_path):
    data = _logs(tmp_path)
    model = _model(tmp_path / "vehicle.json", WRONG)
    names = ["Xu", "Yv", "Zw", "Kp", "Mq", "Nr"]
    export = [{"param": n, "at": f"/linear_damping/{i}"} for i, n in enumerate(names)]
    export += [{"param": n + n[-1], "at": f"/quadratic_damping/{i}"} for i, n in enumerate(names)]
    export += [{"param": p, "at": f"/added_mass/{i}"} for i, p in enumerate(("Xu_dot", "Yv_dot", "Zw_dot"))]
    export += [{"param": p, "at": f"/inertia/{i}"} for i, p in enumerate(("Ieff_roll", "Ieff_pitch", "Ieff_yaw"))]
    export += [{"param": "mass", "at": "/mass"}]
    desc = _description(tmp_path, model, export=export)
    cal = tmp_path / "cal"
    cal.mkdir()
    params = dict(Ieff_roll=1.0, Ieff_pitch=4.0, Ieff_yaw=4.0, Xu=-5.0, Yv=-20.0, Zw=-20.0, Kp=-1.0, Mq=-3.0, Nr=-3.0,
                  Xuu=-40.0, Yvv=-150.0, Zww=-150.0, Kpp=-2.0, Mqq=-8.0, Nrr=-8.0, Xu_dot=6.0, Yv_dot=40.0, Zw_dot=40.0,
                  Xq_dot=0.0, Zq_dot=0.0, net_buoyancy_z=0.0, cg_cb_offset_z=0.0)
    (cal / "calibrated_parameters.yaml").write_text(yaml.safe_dump({"parameters": params, "vehicle_mass_kg": 30.0}))
    out = tmp_path / "out"
    rc = cli.main(["export", "--sim", "external", "--simulator", desc, "--calibration", str(cal), "--data", data,
                   "--reference-point", "0", "0", "0", "--heldout-dates", "02-01-2030", "--out", str(out)])
    assert rc == 0
    y = yaml.safe_load((out / "my_sim_export.yaml").read_text())
    exp, own = y["heldout"]["My Sim with Strathe's values"], y["heldout"]["Strathe's own model"]
    assert exp["30s"]["drift_median_m"] < y["heldout"]["My Sim, original"]["30s"]["drift_median_m"]
    # the calibration holds the true values and the simulator has Strathe's damping law: the export reproduces the
    # true vehicle (Strathe's own model also has cross-axis terms the fake leaves out, so it isn't the yardstick here)
    assert exp["30s"]["drift_median_m"] < 0.01 and own["30s"]["drift_median_m"] > 0.1
    m = json.loads((out / "vehicle_strathe.json").read_text())
    assert m["quadratic_damping"][1] == -150.0 and m["added_mass"][0] == 6.0
    assert any("net_buoyancy_z" not in n for n in y["notes"]) or not y["notes"]


def test_plain_messages_for_a_lab_mistake(tmp_path, capsys):
    model = _model(tmp_path / "vehicle.json", WRONG)
    desc = _description(tmp_path, model, knobs=[{"name": "d", "group": "drag", "at": "/no/such/key", "sign": -1}])
    data = _logs(tmp_path, seconds=60)
    assert cli.main(["fit", "--sim", "external", "--simulator", desc, "--data", data, "--reference-point", "0", "0", "0",
                     "--heldout-dates", "02-01-2030", "--out", str(tmp_path / "o")]) == 2
    assert "no number at '/no/such/key'" in capsys.readouterr().out
    bad = tmp_path / "bad.yaml"
    bad.write_text("replay: x\n")
    assert cli.main(["fit", "--sim", "external", "--simulator", str(bad), "--data", data, "--reference-point", "0", "0",
                     "0"]) == 2
    assert "needs `model:`" in capsys.readouterr().out
    assert cli.main(["export", "--sim", "external", "--simulator", _description(tmp_path, model), "--calibration",
                     str(tmp_path), "--reference-point", "0", "0", "0"]) == 2


def test_a_windows_command_and_docker_paths_keep_their_backslashes():
    """A Windows lab's replay command keeps its paths, and Docker gets them in the container's terms."""
    py, tool = r"C:\Program Files\Python\python.exe", r"D:\lab\my sim\replay.py"
    line = join_command([py, tool, "--fast"], windows=True)
    assert split_command(line, windows=True) == [py, tool, "--fast"]
    assert split_command(r"C:\Python\python.exe D:\lab\replay.py", windows=True) == [r"C:\Python\python.exe",
                                                                                        r"D:\lab\replay.py"]
    assert split_command(join_command(["/usr/bin/python3", "/lab/my sim/r.py"], windows=False),
                         windows=False) == ["/usr/bin/python3", "/lab/my sim/r.py"]
    argv = docker_argv("img", {r"D:\lab"}, ["--model", r"D:\lab\m.sdf", "--ref", "-0.76", "0", "0"],
                       {"GZ_SIM_RESOURCE_PATH": r"D:\lab;C:\models"}, windows=True)
    assert argv == ["docker", "run", "--rm", "-v", r"D:\lab:/d/lab", "-e", "GZ_SIM_RESOURCE_PATH=/d/lab:/c/models",
                    "img", "--model", "/d/lab/m.sdf", "--ref", "-0.76", "0", "0"]
    assert docker_argv("img", {"/lab"}, ["--model", "/lab/m.sdf"], windows=False) == [
        "docker", "run", "--rm", "-v", "/lab:/lab", "img", "--model", "/lab/m.sdf"]


def test_knobs_map_parameters_and_values():
    k = Knobs([KnobSpec("a", "drag", 10.0), KnobSpec("b", "drag", 20.0, tie="t"), KnobSpec("c", "drag", 30.0, tie="t"),
               KnobSpec("x", "cg", 0.1, "linear", -0.2, 0.2), KnobSpec("v", "buoyancy", 1.0, "log", 0.5, 2.0)],
              held={"a": 12.0})
    assert k.names() == ["b", "x", "v"]  # a held, c tied to b
    v = k.values(np.array([np.log(2.0), 0.05, np.log(1.5)]))
    assert v == pytest.approx({"a": 12.0, "b": 40.0, "c": 40.0, "x": 0.15, "v": 1.5})
    lo, hi = k.bounds()
    assert lo[0] == pytest.approx(-np.log(30)) and hi[1] == 0.2 and hi[2] == pytest.approx(np.log(2.0))
    with pytest.raises(SimFitError, match="can't hold"):
        Knobs([KnobSpec("a", "drag", 1.0)], held={"zz": 1.0})


def test_editors_change_only_the_number():
    xml = '<?xml version="1.0"?>\n<!-- keep -->\n<sdf><model name="m"><link name="base_link"><inertial>' \
          '<mass>10.0</mass><pose>0 0 0.05 0 0 0</pose></inertial></link><plugin name="h"><xUabsU>-33.7</xUabsU>' \
          '</plugin><geom name="hull" fluidcoef="0.5 0.25 1.5 1 1"/></model></sdf>\n'
    f = open_model(xml, "xml")
    assert f.get("//plugin[@name='h']/xUabsU") == -33.7 and f.get("//inertial/pose#2") == 0.05
    f.set("//xUabsU", -40.5)
    f.set("//geom[@name='hull']/@fluidcoef#2", 2.25)
    assert f.text == xml.replace("-33.7", "-40.5").replace("1.5 1 1", "2.25 1 1")
    with pytest.raises(SimFitError, match="nothing at"):
        f.get("//nope")
    j = open_model('{\n    "a": {"b": [1, 2.5]}\n}\n', "json")
    j.set("/a/b/1", 3.0)
    assert json.loads(j.text) == {"a": {"b": [1, 3.0]}} and j.text.startswith('{\n    "a"')
    r = open_model('mass = 12.5 # kg\n', "regex")
    r.set(r"mass = ([\d.]+)", 13)
    assert r.text == "mass = 13 # kg\n"


def test_strathe_fit_sim_routes_every_simulator(tmp_path, monkeypatch):
    import strathe_cli
    (tmp_path / "my_sim.yaml").write_text("x")
    monkeypatch.chdir(tmp_path)
    calls = []
    run = lambda c: calls.append(c) or 0  # noqa: E731
    assert strathe_cli.main(["fit-sim", "--sim", "external", "--simulator", "my_sim.yaml", "--data", str(tmp_path),
                             "--reference-point", "0", "0", "0"], runner=run) == 0
    cmd = calls[-1]
    assert cmd[1:5] == ["-m", "simfit.cli", "fit", "--sim"] and str(tmp_path / "my_sim.yaml") in cmd  # made absolute
    assert strathe_cli.main(["fit-sim", "--data", str(tmp_path)], runner=run) == 2  # no --sim
    assert strathe_cli.main(["fit-sim", "--sim", "external", "--simulator", "my_sim.yaml"], runner=run) == 2  # no data
    assert strathe_cli.main(["export-sim", "--sim", "external", "--ingest-spec", "x.yaml"], runner=run) == 2
    calls.clear()
    (tmp_path / "w.scn").write_text("<scenario/>")
    assert strathe_cli.main(["fit-sim", "--sim", "stonefish", "--scenario", "w.scn", "--robot", "r", "--part", "Hull",
                             "--data", str(tmp_path), "--reference-point", "0", "0", "0"], runner=run) == 0
    assert calls[-1][1] == "stonefish_fit.py"  # Stonefish: fit-stonefish, unchanged


def test_the_documented_example_runs_as_shipped(tmp_path):
    """calibration/simulators/example/: the description a lab starts from fits and exports as the guide says."""
    import shutil
    ex = tmp_path / "example"
    shutil.copytree(os.path.join(os.path.dirname(FAKE)), ex)
    data = _logs(tmp_path, seconds=90)
    tool = f"{sys.executable} {ex / 'example_replay.py'}"  # the guide says `python`; the tests use this Python
    args = ["--sim", "external", "--simulator", str(ex / "my_sim.yaml"), "--replay-tool", tool, "--data", data,
            "--reference-point", "0", "0", "0", "--heldout-dates", "02-01-2030"]
    assert cli.main(["fit", *args, "--max-iterations", "3", "--workers", "2", "--axisymmetric",
                     "--out", str(tmp_path / "fit")]) == 0
    assert (tmp_path / "fit" / "example_sim_fit_report.md").exists()
    assert (tmp_path / "fit" / "vehicle_calibrated.json").exists()
    d = yaml.safe_load((ex / "my_sim.yaml").read_text())
    assert {e["param"] for e in d["export"]} >= {"Xuu", "Yv_dot", "Ieff_pitch", "mass"}


def test_the_web_cards_build_fit_sim_and_export_sim(tmp_path):
    import strathe_web as web
    world = tmp_path / "w.sdf"
    world.write_text("<sdf/>")
    cal = tmp_path / "cal"
    cal.mkdir()
    (cal / "calibrated_parameters.yaml").write_text("parameters: {}\n")
    form = {"sim": "gazebo", "model": str(world), "vehicle": "auv", "link": "base_link", "reference_point": "0 0 0",
            "data": str(tmp_path), "heldout": "02-01-2030", "fit": "drag added_mass", "hold": "xUabsU=30",
            "axisymmetric": "on", "replay_tool": "docker:strathe-gz-replay", "max_iterations": "5", "workers": "2"}
    a = web.sim_cli_args(form, "fit")
    assert a[:4] == ["--sim", "gazebo", "--world", str(world)] and a[a.index("--model-name") + 1] == "auv"
    for flag, val in (("--fit", "drag"), ("--hold", "xUabsU=30"), ("--replay-tool", "docker:strathe-gz-replay"),
                      ("--max-iterations", "5"), ("--heldout-dates", "02-01-2030")):
        assert a[a.index(flag) + 1] == val
    assert "--no-cg-rerun" not in a and "--no-cg-rerun" in web.sim_cli_args({**form, "no_cg_rerun": "1"}, "fit")
    assert "--finish" not in a and "--finish" in web.sim_cli_args({**form, "finish": "1"}, "fit")
    assert "--axisymmetric" in a
    mjcf = tmp_path / "w.xml"
    mjcf.write_text("<mujoco/>")
    x = web.sim_cli_args({"sim": "mujoco", "model": str(mjcf), "reference_point": "0 0 0", "calibration": str(cal)},
                         "export")
    assert x[:3] == ["--sim", "mujoco", "--model"] and x[x.index("--calibration") + 1] == str(cal)
    for bad, msg in (({**form, "sim": "nope"}, "Choose the simulator"), ({**form, "reference_point": "1 2"}, "three numbers"),
                     ({**form, "heldout": "2030"}, "27-03-2026"), ({**form, "hold": "x"}, "NAME=VALUE"),
                     ({**form, "data": "", "ingest_spec": ""}, "Give the logs")):
        with pytest.raises(ValueError, match=msg):
            web.sim_cli_args(bad, "fit")


def test_the_bottom_line_warns_what_drift_alone_hides():
    """#90, bluerov2_gz: the calibrated model halved drift but pitched twice as badly, and standing still beat it
    after 30 s in a tank. Both are now said up front."""
    from simfit.runner import _bottom_line

    def r(d10, d30, d60, att):
        return {"10s": {"drift_median_m": d10, "attitude_error_median_deg": att}, "30s": {"drift_median_m": d30},
                "60s": {"drift_median_m": d60}}
    res = {"orig": r(3.51, 8.11, 11.99, [3.9, 7.8, 41.4]), "cal": r(1.87, 4.17, 6.02, [7.3, 17.3, 49.6]),
           "no motion": r(3.06, 3.58, 3.29, [3.9, 6.3, 95.1])}
    text = "".join(_bottom_line(res, "orig", "cal", ["16-02-2024"], "the calibrated model"))
    assert "6.02 m" in text and "-50%" in text
    assert "roll 3.9° -> 7.3°, pitch 7.8° -> 17.3°" in text and "yaw" not in text.split("Be careful")[1].split("The fit")[0]
    assert "Standing still beats the calibrated model after 30 s" in text
    res["cal"].update(diverged=35, windows=712)
    text = "".join(_bottom_line(res, "orig", "cal", ["16-02-2024"], "the calibrated model"))
    assert "blew up in 35 of 712 unseen windows" in text
    res["cal"] = r(1.0, 2.0, 3.0, [4.0, 8.0, 42.0])
    text = "".join(_bottom_line(res, "orig", "cal", ["16-02-2024"], "the calibrated model"))
    assert "Be careful" not in text and "Standing still" not in text
