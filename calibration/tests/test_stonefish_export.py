"""
`strathe export-stonefish` (stonefish_export.py, PROJECT_STATUS.md #74) and `fit-stonefish --from-strathe`: a Strathe
calibration's values put into a lab's own Stonefish vehicle. Stonefish itself isn't needed: tests/fake_stonefish_replay.py
reports the same --info fields (mesh areas, inertia, CG and CB, water density) from the scenario's settings.
"""
import json
import os
import sys

import numpy as np
import pytest
import yaml

import stonefish_export as EX
import stonefish_fit as SF
import strathe_cli as cli
from dynamics_model import CalibrationParameters

from test_stonefish_fit import FAKE, _lab, _logs

PARAMS = dict(Ieff_roll=0.8, Ieff_pitch=6.0, Ieff_yaw=7.0, Xu=-8.0, Yv=-40.0, Zw=-45.0, Kp=-2.0, Mq=-10.0, Nr=-10.0,
              Xuu=-50.0, Yvv=-200.0, Zww=-220.0, Kpp=-1.0, Mqq=-5.0, Nrr=-5.0, Xu_dot=6.0, Yv_dot=60.0, Zw_dot=65.0,
              net_buoyancy_z=4.0, cg_cb_offset_z=0.02)


def _calibration(tmp_path, data, mass=32.0):
    """A calibration's output folder as `strathe calibrate` writes it."""
    cal = tmp_path / "cal"
    cal.mkdir()
    doc = {"vehicle_mass_kg": mass, "parameters": dict(PARAMS)}
    if mass is None:
        doc.pop("vehicle_mass_kg")
    (cal / "calibrated_parameters.yaml").write_text(yaml.safe_dump(doc))
    os.symlink(data, cal / "ingested_logs")
    (cal / "metrics.json").write_text(json.dumps({"dataset": {"train_dates": ["01-01-2030"],
                                                              "heldout_dates": ["02-01-2030"]}}))
    return str(cal)


def _measure(sc, tmp_path):
    rep = SF.Replay(FAKE, str(tmp_path), "auv", (0.0, 0.0, 0.0), hz=50.0)
    n = {"i": 0}

    def measure(settings):
        n["i"] += 1
        return rep.info(sc.write_runnable(str(tmp_path / f"m{n['i']}"), settings))
    return measure


def test_stonefish_drag_is_matched_over_the_logged_speeds():
    speeds = np.linspace(0.5, 0.8, 50)
    cd, cf = EX.drag_coefficients(-8.0, -50.0, speeds, 1000.0, 0.05, 1.1, lambda U: 0.0 * U)
    assert cd > 0 and cf > 0
    strathe = 8.0 * speeds + 50.0 * speeds ** 2
    stonefish = 0.5 * 1000 * cd * 0.05 * speeds ** 3 + 1000 * cf * 1.1 * speeds
    assert np.max(np.abs(stonefish - strathe) / strathe) < 0.03
    # what the other parts already give is taken off; nothing left -> no drag on the part
    assert EX.drag_coefficients(-8.0, -50.0, speeds, 1000.0, 0.05, 1.1,
                                lambda U: 8.0 * U + 50.0 * U ** 2) == (0.0, 0.0)


def test_a_still_axis_is_matched_over_a_short_grid():
    class R:
        real_nu = np.zeros((500, 6))
    assert EX.axis_speeds([R()], 1).max() == pytest.approx(0.2)


def test_the_export_reaches_strathes_values_in_stonefish(tmp_path):
    sc = _lab(tmp_path / "lab", hydro='<hydrodynamics quadratic_drag="1 1 1"/>', r="0.1")
    sc.discover()
    p = CalibrationParameters(**PARAMS)
    e = EX.export_settings(p, 32.0, _measure(sc, tmp_path), "Hull", "Foam", 1000.0,
                           [np.linspace(0.3, 0.8, 40), np.linspace(0.0, 0.2, 40), np.linspace(0.0, 0.2, 40)])
    after = {r[0]: (r[1], r[3]) for r in e.rows}
    for q in ("mass", "net buoyancy (water 1000 kg/m³)", "CG − CB, z", "roll inertia (rigid + added)",
              "pitch inertia (rigid + added)", "yaw inertia (rigid + added)", "added mass y"):
        strathe, stonefish = after[q]
        assert stonefish == pytest.approx(strathe, rel=1e-3, abs=1e-6), q
    assert e.settings.anisotropic and e.settings.added_mass[0] == pytest.approx(6.0)
    for d in e.drag.values():
        assert np.allclose(d["stonefish_force_n"], d["strathe_force_n"], rtol=0.1, atol=0.5)
    assert any("Drag converted" in n for n in e.notes)  # without the damping patch, drag is matched, not carried


def test_with_the_damping_patch_strathes_damping_goes_in_as_it_is(tmp_path):
    """#80: Stonefish's mesh drag can't take Strathe's damping (cubic in speed, coupled axes); the patch's per-axis
    damping is the same law, so the values go in unconverted, all six axes, at the logs' reference point."""
    sc = _lab(tmp_path / "lab", hydro='<hydrodynamics quadratic_drag="1 1 1"/>', r="0.1")
    sc.discover()
    p = CalibrationParameters(**PARAMS)
    measure = _measure(sc, tmp_path)
    e = EX.export_settings(p, 32.0, lambda s: {**measure(s), "axis_damping": True}, "Hull", "Foam", 1000.0,
                           [np.linspace(0.3, 0.8, 40)] * 3, (-0.76, 0.0, 0.0))
    d = e.settings.axis_damping
    assert d["linear"] == [p.Xu, p.Yv, p.Zw] and d["linear_rot"] == [p.Kp, p.Mq, p.Nr]
    assert d["quadratic"] == [p.Xuu, p.Yvv, p.Zww] and d["quadratic_rot"] == [p.Kpp, p.Mqq, p.Nrr]
    assert d["point"] == [-0.76, 0.0, 0.0]
    assert e.settings.quadratic_drag is None and e.drag == {}  # the mesh coefficients are left alone
    assert not any("Drag converted" in n for n in e.notes)
    out = tmp_path / "v.scn"
    sc.write_calibrated_vehicle(str(out), e.settings)
    t = out.read_text()
    assert f'damping_linear="{SF._fmt([p.Xu, p.Yv, p.Zw])}"' in t and 'damping_point="-0.76 0 0"' in t
    assert 'damping_quadratic_rot=' in t and 'quadratic_drag="1 1 1"' in t
    # Strathe's model is written about the reference point: the CG goes there, the CB to the calibrated offset from it
    info = measure(e.settings)
    assert info["base"]["cg"] == pytest.approx([-0.76, 0.0, 0.0], abs=1e-6)
    assert info["base"]["cb"] == pytest.approx([-0.76 - p.cg_cb_offset_x, 0.0, -p.cg_cb_offset_z], abs=1e-6)
    assert "<compound_transform rpy=\"0 0 0\" xyz=\"" in t and any("reference point" in n for n in e.notes)


def test_inertia_below_what_stonefish_allows_is_floored_and_said(tmp_path):
    sc = _lab(tmp_path / "lab", r="0.1")
    sc.discover()
    p = CalibrationParameters(**{**PARAMS, "Ieff_pitch": 0.001})
    e = EX.export_settings(p, 32.0, _measure(sc, tmp_path), "Hull", "Foam", 1000.0, [np.linspace(0.1, 0.5, 20)] * 3)
    assert e.settings.inertia[1] == pytest.approx(EX.INERTIA_FLOOR * 3.0)
    assert any(n.startswith("Pitch inertia") for n in e.notes)


def test_the_vehicle_file_gets_mass_and_inertia_and_keeps_comments(tmp_path):
    sc = _lab(tmp_path / "lab")
    sc.discover()
    t = open(sc.vehicle_file).read().replace('<mass value="30.0"/>', '<mass value="30.0"/>\n                '
                                                                       '<!-- <inertia xyz="1 8 8"/> -->')
    open(sc.vehicle_file, "w").write(t)
    out = tmp_path / "v.scn"
    sc.write_calibrated_vehicle(str(out), SF.PartSettings(mass=31.5, inertia=[0.5, 4, 4]))
    t = out.read_text()
    assert '<mass value="31.5"/>' in t and '<inertia xyz="0.5 4 4"/>' in t and '<!-- <inertia xyz="1 8 8"/> -->' in t


def test_settings_merge_on_top_of_the_export():
    base = SF.PartSettings(quadratic_drag=[1, 2, 2], mass=31.0, inertia=[1, 5, 5], cg_shift=[0.01, 0, -0.02],
                           buoyancy_scale=1.2, added_mass=[5, 60, 60], added_inertia=[0, 1, 1], anisotropic=True)
    top = SF.PartSettings(quadratic_drag=[2, 3, 3], cg_shift=[0.01, 0, 0.01], buoyancy_scale=1.5)
    m = SF.merge_settings(base, top)
    assert m.quadratic_drag == [2, 3, 3] and m.mass == 31.0 and m.inertia == [1, 5, 5] and m.added_mass == [5, 60, 60]
    assert m.cg_shift == pytest.approx([0.02, 0, -0.01]) and m.buoyancy_scale == pytest.approx(1.8)
    assert SF.merge_settings(None, top) is top and SF.merge_settings(base, None) is base


def test_export_stonefish_writes_the_vehicle_and_scores_it(tmp_path):
    data = _logs(tmp_path, seconds=90)
    _lab(tmp_path / "lab", r="0.1")
    cal = _calibration(tmp_path, data)
    out = tmp_path / "out"
    a = EX.build_parser().parse_args(["--calibration", cal, "--scenario", str(tmp_path / "lab" / "world.scn"),
                                      "--robot", "auv", "--part", "Hull", "--buoyancy-part", "Foam",
                                      "--reference-point", "0", "0", "0", "--find",
                                      f"lab_pkg={tmp_path / 'lab' / 'lab_pkg'}", "--physics-hz", "50",
                                      "--out", str(out)])
    a.replay_tool_argv = FAKE
    assert EX.run_export(a) == 0
    y = yaml.safe_load((out / "stonefish_export.yaml").read_text())
    assert y["heldout_dates"] == ["02-01-2030"] and y["mass_kg"] == 32.0
    assert {"Stonefish, original", "Stonefish with Strathe's values", "Strathe's own model"} <= set(y["heldout"])
    report = (out / "stonefish_export_report.md").read_text()
    assert "Not carried, and why" in report and "| mass | 32 kg |" in report
    assert "After 10 s, by axis" in report and "strathe fit-stonefish" in report
    assert "fit-stonefish" in y["recommendation"] and set(y["axis_verdict_10s"]) == {"better", "worse", "same"}
    assert '<mass value="' in (out / "vehicle_strathe.scn").read_text()


def test_a_calibration_without_its_mass_asks_for_it(tmp_path):
    data = _logs(tmp_path, seconds=20)
    cal = _calibration(tmp_path, data, mass=None)
    with pytest.raises(SF.StonefishFitError, match="give --mass"):
        EX.load_calibration(cal)
    assert EX.load_calibration(cal, 30.0)[1] == 30.0


def test_fit_stonefish_can_start_from_the_strathe_calibration(tmp_path):
    from test_stonefish_fit import _args
    data = _logs(tmp_path, seconds=40)
    _lab(tmp_path / "lab", r="0.1")
    cal = _calibration(tmp_path, data)
    a = _args(tmp_path, data, from_strathe=cal, max_iterations=1, compare_params=None, compare_mass=None)
    assert SF.run_fit_stonefish(a) == 0
    y = yaml.safe_load((tmp_path / "out" / "stonefish_fit.yaml").read_text())
    assert {"Stonefish, original", "Stonefish with Strathe's values", "Stonefish, calibrated by Strathe",
            "Strathe's own model"} <= set(y["heldout"])
    assert y["from_strathe"]["exported"]["mass"] == pytest.approx(32.0)
    assert '<mass value="32"/>' in (tmp_path / "out" / "vehicle_calibrated.scn").read_text()


def test_the_command_line_and_web_form_build_the_export(tmp_path):
    data = _logs(tmp_path, seconds=20)
    _lab(tmp_path / "lab", r="0.1")
    cal = _calibration(tmp_path, data)
    calls = []
    rc = cli.main(["export-stonefish", "--calibration", cal, "--scenario", str(tmp_path / "lab" / "world.scn"),
                   "--robot", "auv", "--part", "Hull", "--reference-point", "-0.76", "0", "0", "--buoyancy-part", "Foam",
                   "--density-by-date", "02-01-2030=1023", "--mass", "33", "--out", str(tmp_path / "x")],
                  runner=lambda argv: calls.append(argv) or 0)
    assert rc == 0
    argv = calls[0]
    assert os.path.basename(argv[1]) == "stonefish_export.py" and argv[argv.index("--calibration") + 1] == cal
    assert argv[argv.index("--mass") + 1] == "33.0" and "--density-by-date" in argv and "--buoyancy-part" in argv
    assert cli.main(["export-stonefish", "--calibration", str(tmp_path), "--scenario", "x.scn", "--robot", "auv",
                     "--part", "Hull", "--reference-point", "0", "0", "0"], runner=lambda argv: 0) == 2
    calls.clear()
    cli.main(["fit-stonefish", "--scenario", str(tmp_path / "lab" / "world.scn"), "--robot", "auv", "--part", "Hull",
              "--data", data, "--reference-point", "0", "0", "0", "--from-strathe", cal],
             runner=lambda argv: calls.append(argv) or 0)
    assert calls[0][calls[0].index("--from-strathe") + 1] == cal

    import strathe_web as web
    form = {"calibration": cal, "scenario": str(tmp_path / "lab" / "world.scn"), "robot": "auv", "part": "Hull",
            "reference_point": "-0.76 0 0", "buoyancy_part": "Foam", "mass": "33", "density_by_date": "02-01-2030=1023"}
    args = web.stonefish_export_cli_args(form)
    assert args[:2] == ["--calibration", cal] and "--mass" in args and "--buoyancy-part" in args
    with pytest.raises(ValueError, match="calibrated_parameters.yaml"):
        web.stonefish_export_cli_args({**form, "calibration": str(tmp_path)})
    fit = web.stonefish_cli_args({**form, "data": data, "from_strathe": cal})
    assert fit[fit.index("--from-strathe") + 1] == cal


def test_an_axisymmetric_fit_starts_between_the_side_and_vertical_values():
    k = SF.Knobs(("drag",), {"quadratic_drag": [1.0, 0.04, 10.0], "viscous_drag": [0.02, 0.3, 0.3],
                             "added_mass": [1, 1, 1], "added_inertia": [0, 0, 0]}, axisymmetric=True)
    v = k.values(k.x0())
    assert v["quadratic_drag_yz"] == pytest.approx(np.sqrt(0.4)) and v["viscous_drag_yz"] == pytest.approx(0.3)


def test_the_report_says_which_axes_got_worse_and_to_calibrate_directly():
    def r(vel, att):
        return {"10s": {"velocity_rmse": vel, "attitude_error_median_deg": att}}
    # URI's ALPHA RISE, bay held out (#74)
    v = EX.axis_verdict(r([0.150, 0.070, 0.239, 0, 0, 0], [2.41, 8.91, 19.74]),
                        r([0.424, 0.073, 0.093, 0, 0, 0], [5.14, 11.26, 14.46]))
    assert v["better"] == ["up/down speed (heave)", "heading (yaw)"]
    assert v["worse"] == ["forward speed (surge)", "roll", "pitch"] and v["same"] == ["sideways speed (sway)"]
    assert "fit-stonefish" in EX.recommendation(SF.PartSettings())
    assert "Strathe's own model" in EX.recommendation(SF.PartSettings(axis_damping={"linear": [0, 0, 0]}))
    assert "fit-stonefish" in EX.RECOMMENDATION and EX.axis_verdict({}, {}) == {"better": [], "worse": [], "same": []}


def test_strathe_check_says_what_the_stonefish_build_supports(tmp_path):
    import strathe_cli as cli
    assert "not built" in cli.stonefish_status(str(tmp_path / "missing"))
    for out, want in (('{"patched": true, "axis_damping": true}', "Strathe's damping"),
                      ('{"patched": true, "axis_damping": false}', "older patch"),
                      ('{"patched": false, "axis_damping": false}', "without the patch"),
                      ("usage: ...", "older than this version")):
        if sys.platform == "win32":  # Windows can't run a shell script; a batch file stands in for the tool
            tool = tmp_path / "replay.bat"
            tool.write_text(f"@echo {out}\n")
        else:
            tool = tmp_path / "replay"
            tool.write_text(f"#!/bin/sh\necho '{out}'\n")
            tool.chmod(0o755)
        assert want in cli.stonefish_status(str(tool)), out


def test_stonefish_runs_in_docker_with_its_folders_mounted(monkeypatch, tmp_path):
    """--replay-tool docker:IMAGE: nothing to build (PROJECT_STATUS.md #87)."""
    import shutil
    import sim_docker
    data, world, work = str(tmp_path), *(os.path.abspath(p) for p in ("/w/world.scn", "/work"))
    r = SF.Replay(["docker:strathe-stonefish-replay"], data, "auv", (-0.76, 0.0, 0.0))
    argv = r._argv(world, (work,))
    assert argv[:3] == ["docker", "run", "--rm"]
    for d in (data, os.path.dirname(world), work):  # on Windows D:\data is mounted at /d/data
        assert f"{d}:{sim_docker.inside(d)}" in argv
    assert argv[argv.index("--data") + 1] == sim_docker.inside(data)
    assert argv[argv.index("--scenario") + 1] == sim_docker.inside(world)
    if sys.platform == "win32":
        assert sim_docker.inside(data).startswith("/") and "\\" not in sim_docker.inside(world)
    assert argv[argv.index("strathe-stonefish-replay") + 1] == "--scenario"
    monkeypatch.setattr(shutil, "which", lambda x: None)
    with pytest.raises(SF.StonefishFitError, match="Docker isn't installed"):
        SF.find_tool("docker:strathe-stonefish-replay")


def test_setup_sim_stonefish_says_what_is_missing(monkeypatch, capsys):
    import shutil
    import strathe_cli as cli
    monkeypatch.setattr(shutil, "which", lambda x: None)
    assert cli.main(["setup-sim", "stonefish"]) == 2
    assert "needs git and cmake" in capsys.readouterr().out
    flags = cli._macos_cmake_flags()
    assert all(f.startswith("-D") for f in flags)
