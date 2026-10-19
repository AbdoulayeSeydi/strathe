"""`strathe describe-stonefish` (stonefish_describe.py, PROJECT_STATUS.md #77): a vehicle description written from the
lab's own Stonefish model. Stonefish itself isn't needed: the info dict is the replay tool's --info, and the end-to-end
test uses tests/fake_stonefish_replay.py."""
import math
import os

import pytest

import stonefish_describe as SD
import stonefish_fit as SF
import strathe_cli as cli
from optimizer import initial_parameters, load_vehicle_spec

from test_stonefish_fit import FAKE, _lab


def _info(model_mass=27.0, slender=True):
    """A torpedo-like vehicle as the replay tool reports it: a hull and a small external part, plus internal foam."""
    S_hull = [0.05, 0.375, 0.375] if slender else [0.3, 0.35, 0.4]
    return {"patched": True, "water_density": 1000.0,
            "base": {"mass": model_mass, "volume": 0.03, "cg": [-0.66, 0.0, -0.005], "cb": [-0.65, 0.0, -0.10],
                     "inertia_body_axes": [0.6, 66.0, 65.5]},
            "parts": [{"name": "r/V/Hull", "mass": 26.0, "external": True, "quadratic_drag": [0.08, 1.0, 1.0],
                       "viscous_drag": [0.008, 0.1, 0.1], "added_mass": [10.0, 220.0, 220.0], "added_inertia": [0.0, 54.0, 54.0],
                       "inertia": [0.5, 10.0, 10.0], "projected_area": S_hull, "tangential_area": [1.1, 0.6, 0.6]},
                      {"name": "r/V/Mast", "mass": 0.3, "external": True, "quadratic_drag": [0.3, 1.0, 0.2],
                       "viscous_drag": [0.03, 0.1, 0.02], "added_mass": [0.1, 0.5, 5.0], "added_inertia": [0.0, 0.0, 0.0],
                       "inertia": [0.0, 0.0, 0.0], "projected_area": [0.005, 0.015, 0.003], "tangential_area": [0.03, 0.01, 0.03]},
                      {"name": "r/V/Foam", "mass": 0.5, "external": False, "quadratic_drag": [0.5, 0.5, 1.0],
                       "viscous_drag": [0.05, 0.05, 0.1], "added_mass": [27.0, 27.0, 40.0], "added_inertia": [0.2, 0.2, 0.0],
                       "inertia": [0.02, 0.02, 0.003], "projected_area": [0.15, 0.15, 0.04], "tangential_area": [0.3, 0.3, 0.5]}]}


def _targets(spec):
    return {k: v for g in spec["calibration_targets"].values() for k, v in g.items() if isinstance(v, dict) and "bounds" in v}


def test_the_description_takes_the_models_values():
    spec, notes = SD.describe(_info(), cruise_speed=0.5, reference_point=(-0.76, 0, 0))
    t = _targets(spec)
    assert spec["fixed_parameters"]["mass_kg"]["value"] == 27.0
    assert t["Xu_dot"]["initial_guess"] == pytest.approx(10.1)  # external parts only (the foam is inside)
    rho = 1000.0
    assert t["Yvv"]["initial_guess"] == pytest.approx(-0.5 * rho * (1.0 * 0.375 + 1.0 * 0.015) * 0.5, rel=1e-3)
    assert t["Xu"]["initial_guess"] == pytest.approx(-rho * (0.008 * 1.1 + 0.03 * 0.03), rel=1e-3)
    added_I = 54.0 + 0.2
    assert t["Ieff_pitch"]["initial_guess"] == pytest.approx((66.0 - added_I) + added_I + 27.0 * (0.1 ** 2 + 0.005 ** 2), rel=1e-3)
    assert t["net_buoyancy_z"]["initial_guess"] == pytest.approx(rho * 9.81 * 0.03 - 27.0 * 9.81, rel=1e-3)
    assert t["cg_cb_offset_z"]["initial_guess"] == pytest.approx(0.095, rel=1e-3)  # CG 9.5 cm below CB
    assert all("Stonefish" in t[k]["source"] for k in ("Ieff_pitch", "Xu_dot", "Yvv", "net_buoyancy_z"))


def test_limits_come_from_the_hulls_size_not_the_models_values():
    """Stonefish's geometric added mass can be ~3x strip theory (ALPHA RISE's is). Limits sized from those values let the
    fit wander: on ALPHA RISE it fitted the training mission and drifted 64 m on the unseen one, against 3.5 m with
    limits sized from the hull (#77)."""
    spec, notes = SD.describe(_info())
    t = _targets(spec)
    D = math.sqrt(4 * 0.055 / math.pi)
    L = 0.39 / D
    strip = 1000.0 * 0.055 * L
    assert t["Yv_dot"]["bounds"] == [0.0, pytest.approx(2 * strip, rel=1e-3)]
    assert t["Yv_dot"]["initial_guess"] < 2 * strip < 220.5  # the model's 220.5 kg starts just inside
    assert any("Yv_dot 220" in n for n in notes)
    assert t["Xuu"]["bounds"][0] <= -200.0  # a drag coefficient up to 8: the fitted ALPHA RISE value (-47 to -64) is inside
    for k, v in t.items():
        lo, hi = v["bounds"]
        assert lo <= v["initial_guess"] <= hi, k
    info = _info()
    for p in info["parts"]:
        p["added_mass"] = [10 * a for a in p["added_mass"]]
        p["quadratic_drag"] = [10 * c for c in p["quadratic_drag"]]
    t10 = _targets(SD.describe(info)[0])
    assert all(t10[k]["bounds"] == t[k]["bounds"] for k in ("Xu_dot", "Yv_dot", "Zw_dot", "Xuu", "Yvv", "Zww"))


def test_a_weighed_mass_scales_the_rigid_inertia_and_keeps_the_trim():
    spec, notes = SD.describe(_info(model_mass=27.0), mass=33.0, reference_point=(-0.66, 0.0, -0.005))  # at the CG
    t = _targets(spec)
    assert spec["fixed_parameters"]["mass_kg"]["value"] == 33.0
    assert spec["fixed_parameters"]["mass_kg"]["confidence"] == "measured"
    rigid = 66.0 - 54.2
    assert t["Ieff_pitch"]["initial_guess"] == pytest.approx(rigid * 33 / 27 + 54.2, rel=1e-3)
    assert t["net_buoyancy_z"]["initial_guess"] == pytest.approx(1000 * 9.81 * 0.03 - 27 * 9.81, rel=1e-3)
    assert any("weighs 27.0 kg" in n for n in notes)


def test_hull_lift_only_when_asked_and_only_for_a_slender_hull():
    off, notes = SD.describe(_info())
    assert "hull_lift" not in off["calibration_targets"] and "cg_cb_offset_x" not in off["calibration_targets"]["buoyancy"]
    assert any("--hull-lift" in n for n in notes)
    on, _ = SD.describe(_info(), hull_lift=True)
    D = math.sqrt(4 * 0.055 / math.pi)
    assert on["calibration_targets"]["hull_lift"]["Zuw"]["initial_guess"] == pytest.approx(-0.5 * 1000 * D ** 2 * 1.2, rel=1e-3)
    assert on["calibration_targets"]["buoyancy"]["cg_cb_offset_x"]["initial_guess"] == pytest.approx(-0.01)
    box, _ = SD.describe(_info(slender=False), hull_lift=True)
    assert "hull_lift" not in box["calibration_targets"]


def test_an_old_replay_tool_says_to_rebuild():
    info = _info()
    del info["base"]["cg"]
    with pytest.raises(SF.StonefishFitError, match="rebuild"):
        SD.describe(info)


def test_describe_stonefish_writes_a_description_calibrate_can_load(tmp_path, capsys):
    _lab(tmp_path / "lab", r="0.1")
    out = tmp_path / "v.yaml"
    a = SD.build_parser().parse_args(["--scenario", str(tmp_path / "lab" / "world.scn"), "--robot", "auv",
                                      "--find", f"lab_pkg={tmp_path / 'lab' / 'lab_pkg'}", "--mass", "32",
                                      "--name", "auv", "--out", str(out)])
    a.replay_tool_argv = FAKE
    assert SD.run(a) == 0
    assert "describe-stonefish" in out.read_text().splitlines()[0]
    params = initial_parameters(load_vehicle_spec(str(out)))
    assert params.Ieff_pitch > 0 and params.Xu < 0
    assert "Wrote" in capsys.readouterr().out


def test_the_command_line_and_web_form_build_it(tmp_path):
    (tmp_path / "w.scn").write_text("<scenario/>")
    calls = []
    rc = cli.main(["describe-stonefish", "--scenario", str(tmp_path / "w.scn"), "--robot", "auv", "--reference-point",
                   "-0.76", "0", "0", "--mass", "33", "--hull-lift", "--out", str(tmp_path / "v.yaml")],
                  runner=lambda c: calls.append(c) or 0)
    assert rc == 0
    argv = calls[0]
    assert os.path.basename(argv[1]) == "stonefish_describe.py"
    assert argv[argv.index("--mass") + 1] == "33.0" and "--hull-lift" in argv
    assert argv[argv.index("--reference-point") + 1:argv.index("--reference-point") + 4] == ["-0.76", "0.0", "0.0"]
    assert cli.main(["describe-stonefish", "--scenario", str(tmp_path / "nope.scn"), "--robot", "auv"],
                    runner=lambda c: 0) == 2
    import strathe_web as web
    args = web.describe_stonefish_cli_args({"scenario": str(tmp_path / "w.scn"), "robot": "auv", "reference_point": "-0.76 0 0",
                                            "mass": "33", "hull_lift": "on"})
    assert args[:4] == ["--scenario", str(tmp_path / "w.scn"), "--robot", "auv"] and "--hull-lift" in args
    tool = tmp_path / "replay"
    tool.write_text("#!/bin/sh\n"); tool.chmod(0o755)
    args = web.describe_stonefish_cli_args({"scenario": str(tmp_path / "w.scn"), "robot": "auv", "replay_tool": str(tool)})
    assert args[args.index("--replay-tool") + 1] == str(tool)
    with pytest.raises(ValueError, match="three numbers"):
        web.describe_stonefish_cli_args({"scenario": str(tmp_path / "w.scn"), "robot": "auv", "reference_point": "1 2"})


def test_a_replay_tool_that_hangs_is_a_plain_error(monkeypatch, tmp_path):
    import subprocess

    def hang(*a, **k):
        raise subprocess.TimeoutExpired(cmd="replay", timeout=600)
    monkeypatch.setattr(subprocess, "run", hang)
    rp = SF.Replay(["replay"], str(tmp_path), "auv", (0.0, 0.0, 0.0))
    with pytest.raises(SF.StonefishFitError, match="within 10 minutes"):
        rp.info("w.scn")
    with pytest.raises(SF.StonefishFitError, match="within 24 hours"):
        rp.run("w.scn", [], str(tmp_path / "r"))
