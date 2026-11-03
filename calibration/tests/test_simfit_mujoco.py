"""`strathe fit-sim --sim mujoco` (PROJECT_STATUS.md #83): a lab's own MJCF, calibrated through MuJoCo's fluid model.
MuJoCo is a pip package, so these run everywhere, CI included."""
import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import yaml

from simfit import cli
from simfit.adapters.mujoco import MujocoAdapter
from simfit.core import SimFitError
from simfit.knobs import Knobs
from simfit.editors import XmlFile

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "simulators", "mujoco", "example", "auv.xml")


def _rows(F, start=(0, 0, 0, 0, 0, 0), q=(0, 0, 0, 1), n=101):
    F = np.asarray(F, float)
    F = F if F.ndim == 2 else np.repeat(F[None], n, axis=0)
    rows = np.zeros((len(F), 16))
    rows[:, 10:] = F
    rows[0, :6], rows[0, 6:10] = start, np.array(q) / np.linalg.norm(q)
    return rows


def test_frames_match_strathes():
    ad = MujocoAdapter(MODEL)
    d = tempfile.mkdtemp()
    model = ad.write_runnable(None, d)
    for i, F in enumerate(np.eye(6) * [20, 20, 20, 0.3, 0.3, 0.5]):
        (V, Q), = ad.replay(model, [_rows(F)], d)
        assert V[-1, i] > 0.1 and np.abs(np.delete(V[-1], i)).max() < 1e-6, (F, V[-1])  # pushes to starboard -> +v
    (V, Q), = ad.replay(model, [_rows((0, 0, 0, 0, 0, 0.5))], d)
    assert Q[-1, 2] > 0.1  # a positive yaw rate turns clockwise from above (north-east-down)


def test_the_start_state_round_trips_at_an_offset_reference_point(tmp_path):
    """No water (density and viscosity 0): over 0.1 s the vehicle keeps the logged velocity at the reference point."""
    m = tmp_path / "dry.xml"
    m.write_text(open(MODEL).read().replace('density="1000" viscosity="0.0009"', 'density="0" viscosity="0"'))
    ad = MujocoAdapter(str(m), ref=(-0.3, 0.1, 0.05))
    d = str(tmp_path / "r")
    start = (0.4, -0.1, 0.05, 0.02, -0.03, 0.05)
    (V, Q), = ad.replay(ad.write_runnable(None, d), [_rows(np.zeros(6), start, (0.05, -0.02, 0.3, 1), n=2)], d)
    assert V[1] == pytest.approx(np.array(start), abs=0.01)
    assert Q[1] == pytest.approx(np.array((0.05, -0.02, 0.3, 1)) / np.linalg.norm((0.05, -0.02, 0.3, 1)), abs=0.01)


def test_knobs_come_from_the_file_and_only_they_change(tmp_path):
    m = tmp_path / "auv.xml"
    m.write_text(open(MODEL).read().replace(' fluidcoef="0.5 0.25 1.5 1.0 1.0"', ""))  # MJCF default, made explicit
    ad = MujocoAdapter(str(m))
    names = [n for n, *_ in ad._knob_defs()]
    assert {"hull_blunt_drag", "hull_kutta_lift", "gravcomp", "cg_x", "mass"} <= set(names)
    assert ad.groups == ("drag", "lift", "buoyancy", "cg", "mass")
    out = ad.write_calibrated({"hull_blunt_drag": 0.8, "gravcomp": 1.02}, str(tmp_path), "calibrated")
    t = open(out).read()
    assert 'fluidcoef="0.8 0.25 1.5 1.0 1.0"' in t and 'gravcomp="1.02"' in t
    assert "An example underwater vehicle" in t
    assert XmlFile(t).get("//inertial/@mass") == 10.0
    with pytest.raises(SimFitError, match="can't take Strathe's model"):
        ad.export_values(None, 10.0, {}, [], 1000.0)


def test_plain_messages(tmp_path):
    bad = tmp_path / "x.xml"
    bad.write_text("<mujoco><worldbody><body name='a'/></worldbody></mujoco>")
    with pytest.raises(SimFitError, match="free joint"):
        MujocoAdapter(str(bad))
    off = tmp_path / "step.xml"
    off.write_text(open(MODEL).read().replace('timestep="0.002"', 'timestep="0.003"'))
    ad = MujocoAdapter(str(off))
    with pytest.raises(SimFitError, match="whole number of the model's timestep"):
        ad.replay(ad.write_runnable(None, str(tmp_path / "s")), [_rows(np.zeros(6), n=3)], str(tmp_path / "s"))


def test_fit_sim_mujoco_recovers_the_fluid_coefficients(tmp_path):
    truth = MujocoAdapter(MODEL)
    tv = {"hull_blunt_drag": 0.9, "hull_slender_drag": 0.5}
    data = tmp_path / "logs"
    data.mkdir()
    rng = np.random.default_rng(0)
    for i, date in enumerate(("01-01-2030", "02-01-2030")):
        F = np.repeat(rng.normal(0, [15, 12, 12, 0.2, 0.2, 0.2], size=(30, 6)), 30, axis=0)[:901]
        d = str(tmp_path / f"g{i}")
        (V, Q), = truth.replay(truth.write_runnable(tv, d), [_rows(F, (0.2, 0, 0, 0, 0, 0))], d)
        pd.DataFrame(np.hstack([F, V, Q]), columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp", "Bq",
                                                    "Br", "qx", "qy", "qz", "qw"]).to_csv(data / f"auv-{date}-run1.csv",
                                                                                         index=False)
    out = tmp_path / "out"
    rc = cli.main(["fit", "--sim", "mujoco", "--model", MODEL, "--data", str(data), "--reference-point", "0", "0", "0",
                   "--heldout-dates", "02-01-2030", "--fit", "drag", "--max-iterations", "8", "--workers", "4",
                   "--out", str(out)])
    assert rc == 0
    s = yaml.safe_load((out / "mujoco_fit.yaml").read_text())
    held = s["heldout"]
    assert held["MuJoCo, calibrated by Strathe"]["30s"]["drift_median_m"] < 0.5 * held["MuJoCo, original"]["30s"]["drift_median_m"]
    assert abs(np.log(s["after"]["hull_blunt_drag"] / 0.9)) < abs(np.log(0.5 / 0.9))
    assert (out / "auv_calibrated.xml").exists() and (out / "mujoco_fit_report.md").exists()
    rc = cli.main(["export", "--sim", "mujoco", "--model", MODEL, "--calibration", str(tmp_path), "--data", str(data),
                   "--reference-point", "0", "0", "0", "--out", str(tmp_path / "e")])
    assert rc == 2


def test_from_strathe_on_mujoco_says_to_drop_it(tmp_path, capsys):
    """#88 F3: the refusal names --from-strathe, not export-sim."""
    import yaml as _y
    from simfit import cli
    cal = tmp_path / "cal"
    cal.mkdir()
    params = dict(Ieff_roll=0.3, Ieff_pitch=0.5, Ieff_yaw=0.6, Xu=-6.0, Yv=-18.0, Zw=-22.0, Kp=-1.5, Mq=-2.5, Nr=-2.0,
                  Xuu=-35.0, Yvv=-70.0, Zww=-80.0, Kpp=-1.0, Mqq=-1.2, Nrr=-0.8, Xu_dot=3.0, Yv_dot=8.0, Zw_dot=9.0,
                  net_buoyancy_z=0.0, cg_cb_offset_z=0.02)
    (cal / "calibrated_parameters.yaml").write_text(_y.safe_dump({"parameters": params, "vehicle_mass_kg": 10.0}))
    data = tmp_path / "logs"
    data.mkdir()
    F = np.zeros((40, 6))
    for date in ("01-01-2030", "02-01-2030"):
        pd.DataFrame(np.hstack([F, np.zeros((40, 6)), np.tile([0, 0, 0, 1.0], (40, 1))]),
                     columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp", "Bq", "Br", "qx", "qy", "qz",
                              "qw"]).to_csv(data / f"auv-{date}-run1.csv", index=False)
    rc = cli.main(["fit", "--sim", "mujoco", "--model", MODEL, "--data", str(data), "--reference-point", "0", "0", "0",
                   "--heldout-dates", "02-01-2030", "--from-strathe", str(cal), "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert rc == 2 and "--from-strathe needs" in out and "without --from-strathe" in out


def test_an_incomplete_calibration_is_a_plain_message(tmp_path):
    from stonefish_export import load_calibration
    from simfit.core import SimFitError
    (tmp_path / "calibrated_parameters.yaml").write_text("parameters: {Xu: -1}\nvehicle_mass_kg: 10\n")
    with pytest.raises(SimFitError, match="isn't a complete calibration"):
        load_calibration(str(tmp_path))


def test_a_held_out_date_the_logs_dont_have_is_a_plain_message(tmp_path):
    from simfit import core
    from simfit.core import SimFitError
    for date in ("01-01-2030", "02-01-2030"):
        pd.DataFrame(np.zeros((5, 16)), columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp", "Bq", "Br",
                                                 "qx", "qy", "qz", "qw"]).to_csv(tmp_path / f"auv-{date}-run1.csv", index=False)
    with pytest.raises(SimFitError, match="02-01-2030"):
        core.load_runs(str(tmp_path), ["01-01-1999"])


def test_a_coefficient_the_model_has_at_zero_still_gets_tuned(tmp_path):
    """#90 (bluerov2_gz in Gazebo first): a factor of 0 stays 0, so a zero fluid coefficient starts from MuJoCo's own
    default."""
    m = tmp_path / "auv.xml"
    m.write_text(open(MODEL).read().replace('fluidcoef="0.5 0.25 1.5 1.0 1.0"', 'fluidcoef="0 0.25 1.5 0 1.0"'))
    specs = {s.name: s for s in MujocoAdapter(str(m)).knob_specs({}, ("drag", "lift"), False)}
    blunt = next(s for n, s in specs.items() if n.endswith("blunt_drag"))
    kutta = next(s for n, s in specs.items() if n.endswith("kutta_lift"))
    assert blunt.if_zero == 0.5 and kutta.if_zero == 1.0 and "0 in your model" in blunt.note
    assert next(s for n, s in specs.items() if n.endswith("slender_drag")).if_zero is None
    k = Knobs([blunt])
    assert k.values(k.x0())[blunt.name] == 0.5


def _sensor_logs(tmp_path):
    """Two days of a vehicle with no DVL: thrust, gyro, attitude, 1 Hz position fixes and depth (#96)."""
    from test_measurements import truth_frame
    data = tmp_path / "logs"
    data.mkdir()
    for k, date in enumerate(("01-01-2030", "02-01-2030")):
        df, _, _ = truth_frame(n=700, seed=k, dvl=False, pos_every=10, pos_noise=0.05)
        df.to_csv(data / f"auv-{date}-run1.csv", index=False)
    return data


def _calibration(tmp_path):
    from optimizer import initial_parameters, load_vehicle_spec
    p = initial_parameters(load_vehicle_spec(os.path.join(HERE, "..", "vehicle_spec.yaml")))
    cal = tmp_path / "cal"
    cal.mkdir()
    (cal / "calibrated_parameters.yaml").write_text(yaml.safe_dump(
        {"parameters": {k: float(getattr(p, k)) for k in p.param_names}, "vehicle_mass_kg": 11.2}))
    return cal


def test_starts_from_gives_mujoco_the_physics_aided_starts_alone(tmp_path, capsys):
    """--starts-from (#96): with no DVL, each window starts from the smoother with a Strathe calibration's physics,
    which --from-strathe can't give MuJoCo (its values don't go into a MuJoCo model); the lab's own model is fitted."""
    data, cal = _sensor_logs(tmp_path), _calibration(tmp_path)
    rc = cli.main(["fit", "--sim", "mujoco", "--model", MODEL, "--data", str(data), "--reference-point", "0", "0", "0",
                   "--heldout-dates", "02-01-2030", "--starts-from", str(cal), "--fit", "drag", "--max-iterations", "1",
                   "--out", str(tmp_path / "o")])
    out = capsys.readouterr().out
    assert rc == 0 and "Window starts from the smoother with your Strathe calibration's physics" in out
    assert "Starting from" not in out     # the model's own values, not Strathe's


def test_fit_sim_takes_off_the_delay_the_calibration_took_off(tmp_path, capsys):
    """A late USBL (#96): the simulator is fitted to the fixes the Strathe calibration was fitted to."""
    data, cal = _sensor_logs(tmp_path), _calibration(tmp_path)
    (cal / "metrics.json").write_text(json.dumps({"sensor_suite": {"fix_delays": {"usbl": {
        "delay_s": 1.0, "late": True, "applied_s": 1.0, "applied_from": "found in the fixes"}}}}))
    rc = cli.main(["fit", "--sim", "mujoco", "--model", MODEL, "--data", str(data), "--reference-point", "0", "0", "0",
                   "--heldout-dates", "02-01-2030", "--starts-from", str(cal), "--fit", "drag", "--max-iterations", "1",
                   "--out", str(tmp_path / "o")])
    assert rc == 0 and "usbl's fixes: 1 s of delay taken off, as the calibration did." in capsys.readouterr().out


def test_starts_from_and_from_strathe_together_is_refused(tmp_path, capsys):
    data, cal = _sensor_logs(tmp_path), _calibration(tmp_path)
    rc = cli.main(["fit", "--sim", "mujoco", "--model", MODEL, "--data", str(data), "--reference-point", "0", "0", "0",
                   "--heldout-dates", "02-01-2030", "--starts-from", str(cal), "--from-strathe", str(cal),
                   "--out", str(tmp_path / "o")])
    assert rc == 2 and "not both" in capsys.readouterr().out
