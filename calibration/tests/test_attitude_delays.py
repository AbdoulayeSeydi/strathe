"""A gyro's and an orientation's latency, given in the ingest spec (#98): taken off at ingest, so everything after
reads them on time. Calibration doesn't look for them (B0: the logs can't tell a late gyro from a late orientation, and
a thruster spin-up looks like a late gyro), so ingest only says it sees one."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from test_ingest_sensors import raw, run, spec_dict, synth_vehicle, write_csv  # noqa: F401 (raw is a fixture)

QCOLS = ["qx", "qy", "qz", "qw"]


def _late_rows(df, cols, k):
    """These columns logged k rows late: each row holds what the sensor saw k rows earlier."""
    df[cols] = df[cols].shift(k).bfill()
    return df


def _file(summary):
    return summary["files"][0]


def test_a_late_gyro_given_its_delay_is_put_back_on_time(raw, tmp_path):
    veh = synth_vehicle()
    df = write_csv(raw / "r.csv", veh)
    _late_rows(df, ["gp", "gq", "gr"], 3).to_csv(raw / "r.csv", index=False)       # the gyro 0.3 s late
    bv = {"columns": {"p": "gp", "q": "gq", "r": "gr"}}
    s, (out,) = run(tmp_path, spec_dict(raw, body_velocity=bv), name="said_nothing")
    lag = _file(s)["sensor_checks"]["gyro_vs_orientation"]
    assert max(v for v in lag["lag_s"].values() if v is not None) == pytest.approx(0.3, abs=0.11)   # yaw turns too slowly to time
    assert "body_velocity.angular_delay_s" in lag["warning"]                         # it says how to give it
    s, (out,) = run(tmp_path, spec_dict(raw, body_velocity={**bv, "angular_delay_s": 0.3}), name="given")
    assert _file(s)["gyro_delay_s"] == 0.3
    got = out[["Bp", "Bq", "Br"]].to_numpy()
    assert np.allclose(got[:-5], veh["rates"][:-5], atol=1e-6)                       # on time again (CSV rounding)
    check = _file(s)["sensor_checks"]["gyro_vs_orientation"]
    assert all(abs(v) < 0.05 for v in check["lag_s"].values() if v is not None) and "warning" not in check


def test_a_late_orientation_given_its_delay_is_put_back_on_time(raw, tmp_path):
    veh = synth_vehicle()
    df = write_csv(raw / "r.csv", veh)
    _late_rows(df, QCOLS, 2).to_csv(raw / "r.csv", index=False)                       # the attitude 0.2 s late
    o = {"quaternion": {"x": "qx", "y": "qy", "z": "qz", "w": "qw"}, "delay_s": 0.2}
    s, (out,) = run(tmp_path, spec_dict(raw, orientation=o))
    assert _file(s)["orientation_delay_s"] == 0.2
    R = Rotation.from_quat(out[QCOLS].to_numpy()).as_matrix()
    err = np.degrees(np.linalg.norm(Rotation.from_matrix(np.einsum("nji,njk->nik", R[:-5], veh["R"][:-5])).as_rotvec(), axis=1))
    assert err.max() < 1e-4                                                           # degrees: the CSV's rounding


def test_a_delay_under_one_sample_is_interpolated(raw, tmp_path):
    veh = synth_vehicle()
    df = write_csv(raw / "r.csv", veh)
    t = veh["t"]
    for k, c in enumerate(("gp", "gq", "gr")):                                        # 0.05 s late: half a sample
        df[c] = np.interp(t - 0.05, t, veh["rates"][:, k])
    df.to_csv(raw / "r.csv", index=False)
    bv = {"columns": {"p": "gp", "q": "gq", "r": "gr"}, "angular_delay_s": 0.05}
    s, (out,) = run(tmp_path, spec_dict(raw, body_velocity=bv))
    got = out[["Bp", "Bq", "Br"]].to_numpy()[2:-2]
    late = np.stack([np.interp(t - 0.05, t, veh["rates"][:, k]) for k in range(3)], 1)[2:-2]
    rms = lambda x: float(np.sqrt(np.mean((x - veh["rates"][2:-2]) ** 2)))   # noqa: E731
    assert rms(got) < 0.5 * rms(late)                                                 # most of the lag taken off


def test_zero_is_said_and_left_out_is_not(raw, tmp_path):
    veh = synth_vehicle()
    write_csv(raw / "r.csv", veh)
    bv = {"columns": {"p": "gp", "q": "gq", "r": "gr"}, "angular_delay_s": 0}
    o = {"quaternion": {"x": "qx", "y": "qy", "z": "qz", "w": "qw"}, "delay_s": 0}
    s, (said,) = run(tmp_path, spec_dict(raw, body_velocity=bv, orientation=o), name="zero")
    assert _file(s)["gyro_delay_s"] == 0 and _file(s)["orientation_delay_s"] == 0
    s, (plain,) = run(tmp_path, spec_dict(raw), name="plain")
    assert "gyro_delay_s" not in _file(s) and "orientation_delay_s" not in _file(s)
    assert said.equals(plain)                                                         # 0 changes nothing


@pytest.mark.parametrize("block, message", [
    ({"body_velocity": {"columns": {"p": "gp", "q": "gq", "r": "gr"}, "angular_delay_s": -0.1}}, "0 s or more"),
    ({"body_velocity": {"columns": {"u": "gp", "v": "gq", "w": "gr"}, "angular_delay_s": 0.1}}, "no gyro"),
    ({"orientation": {"quaternion": {"x": "qx", "y": "qy", "z": "qz", "w": "qw"}, "delay_s": float("nan")}},
     "0 s or more"),
])
def test_a_wrong_delay_is_refused_with_what_to_do(raw, tmp_path, block, message):
    veh = synth_vehicle(n=200)
    write_csv(raw / "r.csv", veh)
    with pytest.raises(Exception, match=message):
        run(tmp_path, spec_dict(raw, **block))
