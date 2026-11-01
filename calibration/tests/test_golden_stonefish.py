"""Golden outputs for fit-stonefish and export-stonefish (PROJECT_STATUS.md #81): the multi-simulator refactor moves the
shared fitting and scoring into calibration/simfit/, and Stonefish's outputs must stay the same: byte for byte on the
platform that wrote the golden files, and to 10 significant figures on another. The golden files were written by the
code before the refactor; STRATHE_WRITE_GOLDEN=1 rewrites them (only for an intended change, said in the commit)."""
import os
import re

import stonefish_export as EX
import stonefish_fit as SF

from test_stonefish_export import _calibration
from test_stonefish_fit import FAKE, _args, _lab, _logs

GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
WRITE = os.environ.get("STRATHE_WRITE_GOLDEN") == "1"


def _same_anywhere(text):
    """What must match on every platform: paths with / (Windows writes \\), and numbers to 10 significant figures
    (another platform's maths library can differ in the last digits of a fitted value)."""
    text = re.sub(r"<tmp>[^\s'\"]*", lambda m: m.group().replace("\\", "/"), text)
    return re.sub(r"-?\d+\.\d+(?:[eE][-+]?\d+)?", lambda m: f"{float(m.group()):.10g}", text)


def _check(tmp_path, out, names, prefix):
    for name in names:
        text = (out / name).read_text().replace(str(tmp_path), "<tmp>")
        golden = os.path.join(GOLDEN, f"{prefix}_{name}")
        if WRITE:
            open(golden, "w", encoding="utf-8").write(text)
        assert _same_anywhere(text) == _same_anywhere(open(golden, encoding="utf-8").read()), \
            f"{name} differs from {golden}"


def test_fit_stonefish_output_is_unchanged(tmp_path):
    data = _logs(tmp_path)
    _lab(tmp_path / "lab")
    assert SF.run_fit_stonefish(_args(tmp_path, data, workers=1)) == 0
    _check(tmp_path, tmp_path / "out", ["stonefish_fit.yaml", "stonefish_fit_report.md", "vehicle_calibrated.scn"], "fit")


def test_export_stonefish_output_is_unchanged(tmp_path):
    data = _logs(tmp_path, seconds=90)
    _lab(tmp_path / "lab", r="0.1")
    cal = _calibration(tmp_path, data)
    a = EX.build_parser().parse_args(["--calibration", cal, "--scenario", str(tmp_path / "lab" / "world.scn"),
                                      "--robot", "auv", "--part", "Hull", "--buoyancy-part", "Foam",
                                      "--reference-point", "0", "0", "0", "--find",
                                      f"lab_pkg={tmp_path / 'lab' / 'lab_pkg'}", "--physics-hz", "50",
                                      "--out", str(tmp_path / "out")])
    a.replay_tool_argv = FAKE
    assert EX.run_export(a) == 0
    _check(tmp_path, tmp_path / "out", ["stonefish_export.yaml", "stonefish_export_report.md", "vehicle_strathe.scn"],
           "export")
