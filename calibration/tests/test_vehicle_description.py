"""
A lab's own vehicle description (PROJECT_STATUS.md #70): the template, the warning when the shipped BlueROV2 Heavy
description is used for other thrusters, and the plausibility check against the vehicle's own `reference:` values
(it used to compare every vehicle with hardcoded BlueROV2 numbers: URI's 33 kg AUV's pitch inertia was "127x too big").
"""
import json
import os

import pytest
import yaml

import strathe_cli as cli
from calibration_report import build_report, implausible_parameters, vehicle_description_note
from optimizer import initial_parameters, load_vehicle_spec
from vehicle_spec_schema import load_and_validate

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
TEMPLATE = os.path.join(CAL, "templates", "vehicle_spec_template.yaml")
DEFAULT = os.path.join(CAL, "vehicle_spec.yaml")
CUSTOM = "custom layout: 4 thrusters, curve t200 (+51.5 N / -40.0 N); max force 51/51/103 N"
HEAVY = "BlueROV2 Heavy: 8 thrusters, curve T200 @ ~15 V (BlueROV2 dataset) (+51.5 N / -40.0 N)"


def _params(spec, **values):
    p = initial_parameters(load_vehicle_spec(spec))
    for k, v in values.items():
        setattr(p, k, v)
    return p


def test_the_template_is_a_valid_vehicle_description():
    spec = load_vehicle_spec(TEMPLATE)
    assert spec.vehicle.mass_kg == 33.0
    load_and_validate(TEMPLATE)  # the stricter schema too
    initial_parameters(spec)


def test_the_bluerov2_description_still_flags_its_own_implausible_inertia():
    flagged = implausible_parameters(_params(DEFAULT, Ieff_roll=10.0), load_vehicle_spec(DEFAULT))
    assert [p["name"] for p in flagged] == ["Ieff_roll"] and "BlueROV2" in flagged[0]["source"]


def test_another_vehicle_is_checked_against_its_own_references_only(tmp_path):
    # URI's AUV: a pitch inertia of ~29 kg*m^2 is physically right for a 33 kg, 1.5 m hull
    assert implausible_parameters(_params(TEMPLATE, Ieff_pitch=29.2), load_vehicle_spec(TEMPLATE)) == []
    raw = yaml.safe_load(open(TEMPLATE))
    raw["calibration_targets"]["effective_inertia"]["Ieff_pitch"]["reference"] = {"value": 20.0, "source": "CAD model"}
    path = tmp_path / "with_cad.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))
    spec = load_vehicle_spec(str(path))
    assert implausible_parameters(_params(str(path), Ieff_pitch=29.2), spec) == []
    flagged = implausible_parameters(_params(str(path), Ieff_pitch=150.0), spec)
    assert flagged[0]["name"] == "Ieff_pitch" and flagged[0]["source"] == "CAD model"
    load_and_validate(str(path))


def test_a_reference_needs_a_value_and_a_source(tmp_path):
    raw = yaml.safe_load(open(TEMPLATE))
    raw["calibration_targets"]["effective_inertia"]["Ieff_roll"]["reference"] = {"value": 0, "source": "x"}
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="nonzero value and a source"):
        load_vehicle_spec(str(path))


def _ingest(tmp_path, thrusters):
    p = tmp_path / "ingest.yaml"
    p.write_text(yaml.safe_dump({"vehicle": "x", "wrench": thrusters}))
    return str(p)


def test_calibrate_warns_when_the_default_description_meets_other_thrusters(tmp_path):
    layout = _ingest(tmp_path, {"thrusters": {"pwm_columns": ["a", "b", "c", "d"], "layout": [{}, {}, {}, {}]}})
    w = cli.default_spec_warning(cli.DEFAULT_SPEC, layout, None)
    assert "BlueROV2 Heavy (11.2 kg" in w and "4 thrusters" in w and "vehicle_spec_template.yaml" in w
    assert cli.default_spec_warning("my_vehicle_spec.yaml", layout, None) is None  # their own description
    heavy = _ingest(tmp_path, {"thrusters": {"pwm_columns": ["a"], "preset": "bluerov2_heavy"}})
    assert cli.default_spec_warning(cli.DEFAULT_SPEC, heavy, None) is None
    forces = _ingest(tmp_path, {"columns": {k: k for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}})
    assert cli.default_spec_warning(cli.DEFAULT_SPEC, forces, None) is None  # can't tell which vehicle
    (tmp_path / "ingest_report.json").write_text(json.dumps({"thrusters": CUSTOM}))
    assert "custom layout" in cli.default_spec_warning(cli.DEFAULT_SPEC, None, str(tmp_path))


def _result(tmp_path, thrusters):
    d = tmp_path / "result"
    d.mkdir()
    p = initial_parameters(load_vehicle_spec(DEFAULT))
    (d / "calibrated_parameters.yaml").write_text(yaml.safe_dump(
        {"parameters": {n: float(v) for n, v in zip(p.param_names, p.to_vector())}}))
    (d / "metrics.json").write_text(json.dumps({"per_dof": {"default_heldout": [0.4] * 6, "calibrated_heldout": [0.5] * 6},
                                                "skill_vs_zero": {"zero_heldout": [0.3] * 6}}))
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "ingest_report.json").write_text(json.dumps({"thrusters": thrusters}))
    return str(d), str(logs)


def test_the_report_says_the_description_isnt_the_vehicles(tmp_path):
    result, logs = _result(tmp_path, CUSTOM)
    r = build_report(result, DEFAULT, data_dir=logs)
    assert r["vehicle_note"] and r["next_steps"][0].startswith("Describe your own vehicle")
    text = open(r["report"], encoding="utf-8").read()
    assert "The vehicle description isn't your vehicle's" in text and "custom layout: 4 thrusters" in text


@pytest.mark.parametrize("spec,thrusters", [(DEFAULT, HEAVY), (TEMPLATE, CUSTOM)])
def test_no_note_for_a_bluerov2_or_a_lab_description(tmp_path, spec, thrusters):
    assert vehicle_description_note(load_vehicle_spec(spec), _result(tmp_path, thrusters)[1]) is None
