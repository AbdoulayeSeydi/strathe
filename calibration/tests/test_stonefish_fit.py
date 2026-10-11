"""
`strathe fit-stonefish` (stonefish_fit.py, PROJECT_STATUS.md #72): calibrating a lab's own Stonefish vehicle model from
its logs. Stonefish itself isn't needed: tests/fake_stonefish_replay.py speaks the replay tool's interface with a simple
vehicle whose motion follows the scenario's settings, so the whole path is checked -- the lab's scenario as written
(ROS-style includes, sensors), the fit, the held-out scores, the files written, the command line and the web form.
"""
import os
import sys
import types

import numpy as np
import pytest
import yaml

import stonefish_fit as SF
import strathe_cli as cli

HERE = os.path.dirname(os.path.abspath(__file__))
FAKE = [sys.executable, os.path.join(HERE, "fake_stonefish_replay.py")]

VEHICLE = """<?xml version="1.0"?>
<!-- the lab's vehicle file: comments and sensors must survive in the calibrated copy -->
<scenario>
    <robot name="$(arg robot_name)" fixed="false" self_collisions="false">
        <base_link name="Vehicle" type="compound" physics="submerged">
            <external_part name="Hull" type="model" physics="submerged" buoyant="false">
                <mass value="30.0"/>
                {hydro}
                <compound_transform rpy="0 0 0" xyz="0 0 0"/>
                <cg xyz="-0.5 0.0 0.0" rpy="0 0 0"/>
            </external_part>
            <internal_part name="Foam" type="cylinder" physics="floating" buoyant="true">
                <dimensions radius="{r}" height="1.0"/>
                <mass value="0.5"/>
            </internal_part>
        </base_link>
        <sensor name="dvl" type="dvl" rate="5.0"><link name="Base"/></sensor>
        <world_transform xyz="0 0 0" rpy="0 0 0"/>
    </robot>
</scenario>
"""
WORLD = """<?xml version="1.0"?>
<scenario>
    <environment><ocean><water density="1000.0" jerlov="0.2"/></ocean></environment>
    <include file="$(find lab_pkg)/vehicles/vehicle.scn"><arg name="robot_name" value="auv"/></include>
</scenario>
"""
TRUTH = dict(hydro='<hydrodynamics quadratic_drag="0.6 2.5 2.5" viscous_drag="0.05 0.3 0.3" added_mass="8 70 70" '
                   'added_inertia="0.1 2 2" anisotropic_added_mass="true"/>', r="0.1")


def _lab(tmp_path, hydro="", r="0.09"):
    pkg = tmp_path / "lab_pkg"
    (pkg / "vehicles").mkdir(parents=True)
    (pkg / "vehicles" / "vehicle.scn").write_text(VEHICLE.format(hydro=hydro, r=r))
    (tmp_path / "world.scn").write_text(WORLD)
    return SF.Scenario(str(tmp_path / "world.scn"), {"lab_pkg": str(pkg)}, "Hull", "Foam")


def _logs(tmp_path, seconds=90, dates=("01-01-2030", "02-01-2030")):
    """Recordings from the 'true' vehicle, driven by piecewise-constant forces, in Strathe's CSV layout."""
    truth = _lab(tmp_path / "truth", **TRUTH)
    truth.discover()
    rep = SF.Replay(FAKE, str(tmp_path), "auv", (0.0, 0.0, 0.0), hz=50.0)
    scn = truth.write_runnable(str(tmp_path / "truth_run"))
    data = tmp_path / "logs"
    data.mkdir()
    rng = np.random.default_rng(0)
    n = int(seconds / 0.1) + 1
    for i, d in enumerate(dates):
        F = np.repeat(rng.normal(0, [20, 15, 15, 0.5, 0.5, 0.5], size=(n // 50 + 1, 6)), 50, axis=0)[:n]
        start = np.concatenate([[0.3, 0, 0, 0, 0, 0], [0, 0, 0, 1], F[0]])
        rows = [np.vstack([np.hstack([start[:10], f]) for f in F])]
        (V, Q), = rep.run(scn, rows, str(tmp_path / f"gen{i}"))
        import pandas as pd
        df = pd.DataFrame(np.hstack([F, V, Q]), columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp",
                                                          "Bq", "Br", "qx", "qy", "qz", "qw"])
        df.to_csv(data / f"auv-{d}-run1.csv", index=False)
    return str(data)


def test_the_lab_scenario_is_prepared_for_a_headless_run(tmp_path):
    sc = _lab(tmp_path)
    sc.discover()
    assert sc.vehicle_file.endswith("vehicle.scn")
    root = sc.write_runnable(str(tmp_path / "run"), SF.PartSettings(quadratic_drag=[1, 2, 3], buoyancy_scale=4.0), 1023)
    txt = "".join(open(os.path.join(tmp_path, "run", f)).read() for f in os.listdir(tmp_path / "run"))
    assert "$(find" not in txt and "<sensor" not in txt and 'density="1023"' in txt
    assert 'quadratic_drag="1 2 3"' in txt and 'radius="0.18"' in txt  # a cylinder's volume x4: radius x2
    assert os.path.basename(root).endswith("world.scn")
    with pytest.raises(SF.StonefishFitError, match=r"--find lab_pkg="):
        SF.Scenario(str(tmp_path / "world.scn"), {}, "Hull").files()
    with pytest.raises(SF.StonefishFitError, match="no part named 'Keel'"):
        SF.Scenario(str(tmp_path / "world.scn"), {"lab_pkg": str(tmp_path / "lab_pkg")}, "Keel").discover()


def test_the_calibrated_vehicle_file_keeps_everything_else(tmp_path):
    sc = _lab(tmp_path, hydro='<hydrodynamics quadratic_drag="1 1 1"/>')
    sc.discover()
    out = tmp_path / "calibrated.scn"
    sc.write_calibrated_vehicle(str(out), SF.PartSettings(quadratic_drag=[0.5, 3, 3], added_mass=[9, 80, 80],
                                                          added_inertia=[0.1, 2, 2], anisotropic=True, cg_shift=[0.02, 0, 0]))
    t = out.read_text()
    assert "comments and sensors must survive" in t and '<sensor name="dvl"' in t
    assert 'quadratic_drag="0.5 3 3"' in t and 'added_mass="9 80 80"' in t and 'anisotropic_added_mass="true"' in t
    assert '<cg xyz="-0.48 0 0"' in t and t.count("<hydrodynamics") == 1


def _args(tmp_path, data, **kw):
    a = SF.build_parser().parse_args(["--scenario", str(tmp_path / "lab" / "world.scn"), "--robot", "auv", "--part", "Hull",
                                      "--buoyancy-part", "Foam", "--data", data, "--reference-point", "0", "0", "0",
                                      "--heldout-dates", "02-01-2030", "--find", f"lab_pkg={tmp_path / 'lab' / 'lab_pkg'}",
                                      "--fit", "drag", "added_mass", "buoyancy", "--max-iterations", "6",
                                      "--workers", "4", "--physics-hz", "50", "--out", str(tmp_path / "out")])
    a.replay_tool_argv = FAKE
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_fit_stonefish_improves_the_lab_model_on_unseen_logs(tmp_path):
    data = _logs(tmp_path)
    _lab(tmp_path / "lab")
    assert SF.run_fit_stonefish(_args(tmp_path, data)) == 0
    out = tmp_path / "out"
    s = yaml.safe_load(open(out / "stonefish_fit.yaml"))
    held = s["heldout"]
    orig, cal = held["Stonefish, original"]["30s"], held["Stonefish, calibrated by Strathe"]["30s"]
    assert cal["drift_median_m"] < 0.6 * orig["drift_median_m"]
    assert s["fit"]["cost"] < 0.5 * s["fit"]["start_cost"] and s["heldout_dates"] == ["02-01-2030"]
    before, after = np.array(s["before"]["quadratic_drag"]), np.array(s["after"]["quadratic_drag"])
    truth = np.array([0.6, 2.5, 2.5])
    assert np.abs(np.log(after / truth)).mean() < np.abs(np.log(before / truth)).mean()  # toward the truth
    cal_file = out / "vehicle_calibrated.scn"
    assert cal_file.exists() and 'anisotropic_added_mass="true"' in cal_file.read_text()
    report = (out / "stonefish_fit_report.md").read_text()
    assert "On recordings it never saw (02-01-2030)" in report and "keeps its starting velocity" in report
    assert (out / "drift.png").exists()


def test_unpatched_stonefish_and_no_buoyancy_part_narrow_the_fit(tmp_path, monkeypatch):
    data = _logs(tmp_path, seconds=70)
    _lab(tmp_path / "lab")
    real_info = SF.Replay.info
    monkeypatch.setattr(SF.Replay, "info", lambda self, scn: {**real_info(self, scn), "patched": False})
    a = _args(tmp_path, data, buoyancy_part=None, max_iterations=1)
    assert SF.run_fit_stonefish(a) == 0
    s = yaml.safe_load(open(tmp_path / "out" / "stonefish_fit.yaml"))
    assert s["fitted"] == ["drag"] and any("stonefish_added_mass.patch" in n for n in s["notes"])
    assert any("--buoyancy-part" in n for n in s["notes"])


def test_a_missing_replay_tool_says_how_to_build_it(tmp_path, capsys):
    rc = SF.main(["--scenario", "x.scn", "--robot", "r", "--part", "p", "--data", str(tmp_path),
                  "--reference-point", "0", "0", "0", "--replay-tool", str(tmp_path / "nope")])
    assert rc == 2 and "calibration/stonefish/README.md" in capsys.readouterr().out


def test_the_command_line_runs_the_fit_after_converting_the_logs(tmp_path):
    (tmp_path / "w.scn").write_text("<scenario/>")
    (tmp_path / "spec.yaml").write_text("vehicle: x\n")
    calls = []
    rc = cli.main(["fit-stonefish", "--scenario", str(tmp_path / "w.scn"), "--robot", "auv", "--part", "Hull",
                   "--ingest-spec", str(tmp_path / "spec.yaml"), "--reference-point", "-0.76", "0", "0",
                   "--axisymmetric", "--buoyancy-part", "Foam", "--find", "pkg=/p", "--density-by-date", "1-1=1000",
                   "--out", str(tmp_path / "o")], runner=lambda c: calls.append(c) or 0)
    assert rc == 0 and len(calls) == 2
    assert "data_ingestion.py" in " ".join(calls[0])
    fit = calls[1]
    assert os.path.basename(fit[1]) == "stonefish_fit.py"
    assert fit[fit.index("--data") + 1] == str(tmp_path / "o" / "ingested_logs")
    assert fit[fit.index("--reference-point") + 1:fit.index("--reference-point") + 4] == ["-0.76", "0.0", "0.0"]
    assert "--axisymmetric" in fit and fit[fit.index("--buoyancy-part") + 1] == "Foam"
    assert cli.main(["fit-stonefish", "--scenario", str(tmp_path / "w.scn"), "--robot", "a", "--part", "H",
                     "--reference-point", "0", "0", "0"], runner=lambda c: 0) == 2  # no logs given


def test_web_form_builds_the_command(tmp_path):
    import strathe_web as web
    (tmp_path / "w.scn").write_text("<scenario/>")
    (tmp_path / "logs").mkdir()
    form = {"scenario": str(tmp_path / "w.scn"), "robot": "auv", "part": "Hull", "reference_point": "-0.76 0 0",
            "data": str(tmp_path / "logs"), "heldout": "27-03-2026", "buoyancy_part": "Foam", "find": "pkg=/x",
            "density_by_date": "27-03-2026=1000", "axisymmetric": "on"}
    args = web.stonefish_cli_args(form)
    assert args[:6] == ["--scenario", str(tmp_path / "w.scn"), "--robot", "auv", "--part", "Hull"]
    assert "--axisymmetric" in args and args[args.index("--density-by-date") + 1] == "27-03-2026=1000"
    for bad, match in (({"reference_point": "1 2"}, "three numbers"), ({"data": ""}, "Give the logs"),
                       ({"density_by_date": "fresh"}, "DATE=RHO"), ({"find": "pkg"}, "PKG=/path")):
        with pytest.raises(ValueError, match=match):
            web.stonefish_cli_args({**form, **bad})
    assert "id=sf-scn" in open(web.__file__, encoding="utf-8").read()  # the card is on the page
