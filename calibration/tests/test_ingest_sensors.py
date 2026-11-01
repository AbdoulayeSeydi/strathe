"""The sensor suite's ingestion (PROJECT_STATUS.md #96): a vehicle's own sensors (depth, pressure, position fixes,
GPS, an accelerometer, a DVL with gaps) read as readings of their own, converted, placed where they happened and
checked for the silent errors. Synthetic vehicles with a known true motion, written as a CSV and as a ROS bag."""
import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial.transform import Rotation

import sensor_ingest as si
from data_ingestion import ENU_TO_NED, FLU_TO_FRD, IngestError, ingest, load_ingest_spec

RATE = 10.0


# ------------------------------------------------------------------ a vehicle with a known motion

def synth_vehicle(n=1200, seed=0, dt=1.0 / RATE):
    """A box vehicle driven by smooth random forces through first-order dynamics (so thrust pushes the way it moves),
    turning in yaw, rolling and pitching a little. Everything in FRD body / NED world, SI units."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) * dt
    m, d = np.array([25.0, 30.0, 35.0]), np.array([20.0, 25.0, 30.0])

    def smooth(scale, tau):
        x = np.cumsum(rng.normal(0, 1, n))
        k = int(tau / dt)
        x = np.convolve(x - x.mean(), np.ones(k) / k, mode="same")
        return scale * x / (np.std(x) + 1e-9)
    F = np.stack([smooth(15, 8), smooth(8, 6), smooth(10, 7)], 1)
    v = np.zeros((n, 3))
    for k in range(1, n):
        v[k] = v[k - 1] + dt * (F[k - 1] - d * v[k - 1]) / m
    yaw = np.cumsum(smooth(0.3, 10)) * dt
    roll, pitch = smooth(0.05, 5), smooth(0.04, 5)
    R = Rotation.from_euler("ZYX", np.stack([yaw, pitch, roll], 1)).as_matrix()
    rates = si.rates_from_orientation(t, R)
    pos = np.cumsum(np.einsum("nij,nj->ni", R, v), 0) * dt + np.array([3.0, -2.0, 4.0])
    vdot = np.gradient(v, dt, axis=0)
    acc = vdot + np.cross(rates, v) + np.einsum("nji,j->ni", R, np.array([0.0, 0.0, -si.G]))
    Tz = 6.0 * rates[:, 2]
    return {"t": t, "R": R, "v": v, "rates": rates, "pos": pos, "acc": acc,
            "wrench": np.hstack([F, np.zeros((n, 2)), Tz[:, None]])}


def write_csv(path, veh, every=10, noise=0.0, seed=1, gps_latlon=False, drop_rows=(), delay_rows=0,
              depth_scale=1.0, acc_scale=1.0, gyro_scale=1.0, hold_fixes=False):
    """The vehicle as one CSV per recording, FRD/NED: forces, gyro, quaternion every row; a position fix every
    `every` rows (empty cells between), depth every 2nd row, the accelerometer every row."""
    rng = np.random.default_rng(seed)
    n = len(veh["t"])
    q = Rotation.from_matrix(veh["R"]).as_quat()
    df = pd.DataFrame({"time": 1.7e9 + veh["t"]})
    for k, c in enumerate(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")):
        df[c] = veh["wrench"][:, k]
    for k, c in enumerate("pqr"):
        df["g" + c] = veh["rates"][:, k] * gyro_scale
    for k, c in enumerate("xyzw"):
        df["q" + c] = q[:, k]
    fix = np.full((n, 3), np.nan)
    rows = np.arange(0, n, every)
    src = np.clip(rows - delay_rows, 0, n - 1)     # a delayed sensor reports where the vehicle was earlier
    fix[rows] = veh["pos"][src] + noise * rng.normal(size=(len(rows), 3))
    if hold_fixes:   # a logger that repeats the last fix on every row until the next one
        fix = pd.DataFrame(fix).ffill().to_numpy()
    for k in drop_rows:
        fix[k] = np.nan
    if gps_latlon:
        lat0, lon0 = 41.4436494, -71.42388341
        df["lat"] = lat0 + np.degrees(fix[:, 0] / 6378137.0)
        df["lon"] = lon0 + np.degrees(fix[:, 1] / (6378137.0 * np.cos(np.radians(lat0))))
    else:
        df["px"], df["py"], df["pz"] = fix[:, 0], fix[:, 1], fix[:, 2]
    depth = np.full(n, np.nan)
    depth[::2] = veh["pos"][::2, 2] * depth_scale
    df["depth"] = depth
    for k, c in enumerate("xyz"):
        df["a" + c] = veh["acc"][:, k] * acc_scale
    df.to_csv(path, index=False)
    return df


def spec_dict(raw, **over):
    s = {"vehicle": "synth", "source": {"files": str(raw / "*.csv")}, "time": {"column": "time", "unit": "s"},
         "session_date": {"from": "constant", "value": "2026-11-01"}, "frame": "frd",
         "body_velocity": {"columns": {"p": "gp", "q": "gq", "r": "gr"}},
         "orientation": {"quaternion": {"x": "qx", "y": "qy", "z": "qz", "w": "qw"}},
         "wrench": {"columns": {c: c for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}},
         "position": {"use": "calibrate", "source": "usbl", "x": "px", "y": "py", "z": "pz"},
         "depth": {"column": "depth"},
         "accelerometer": {"x": "ax", "y": "ay", "z": "az"}}
    for k, v in over.items():
        if v is None:
            s.pop(k, None)
        else:
            s[k] = v
    return s


def run(tmp_path, spec, name="s"):
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(spec))
    out = tmp_path / f"out_{name}"
    summary = ingest(load_ingest_spec(str(p)), str(out))
    frames = [pd.read_csv(out / f["output"]) for f in summary["files"] if f.get("output")]
    return summary, frames


@pytest.fixture
def raw(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    return d


# ------------------------------------------------------------------ conversions

def test_latlon_to_ned_matches_the_ellipsoid():
    # one thousandth of a degree at the equator: 110.574 m of latitude, 111.320 m of longitude (WGS84)
    ne = si.latlon_to_ned(np.array([1e-3, 0.0]), np.array([0.0, 1e-3]), 0.0, 0.0)
    assert ne[0, 0] == pytest.approx(110.574, abs=0.01) and abs(ne[0, 1]) < 1e-6
    assert ne[1, 1] == pytest.approx(111.320, abs=0.01) and abs(ne[1, 0]) < 1e-3
    # at 60 N a degree of longitude is half as long (to the ellipsoid's ~0.3%)
    e60 = si.latlon_to_ned(np.array([60.0]), np.array([1e-3]), 60.0, 0.0)[0, 1]
    assert e60 == pytest.approx(55.800, abs=0.02)
    # and it inverts what write_csv does (a small-angle sphere) to well under a centimetre over 100 m
    lat0 = 41.4436494
    n, e = 73.0, -41.0
    back = si.latlon_to_ned(np.array([lat0 + np.degrees(n / 6378137.0)]),
                            np.array([np.degrees(e / (6378137.0 * np.cos(np.radians(lat0))))]), lat0, 0.0)[0]
    assert np.allclose(back, [n, e], atol=0.5)   # the sphere vs the ellipsoid: under 0.7%


def test_pressure_to_depth_fresh_salt_and_surface():
    p0 = 101325.0
    p = p0 + np.array([0.0, 997 * si.G * 10, 1025 * si.G * 10])
    d, how = si.pressure_to_depth(p, "Pa", "fresh", p0)
    assert d[:2] == pytest.approx([0.0, 10.0]) and how["surface"] == "given"
    d, _ = si.pressure_to_depth(p / 100, "mbar", "salt", p0 / 100)
    assert d[2] == pytest.approx(10.0)
    d, how = si.pressure_to_depth(np.r_[np.full(50, p0), p0 + 997 * si.G * np.linspace(0, 5, 950)], "Pa", "fresh",
                                  "auto")
    assert d.min() == pytest.approx(0.0, abs=1e-9) and d.max() == pytest.approx(5.0, abs=1e-6)
    assert "auto" in how["surface"]


def test_readings_are_placed_where_they_happened_never_interpolated():
    grid = np.arange(0, 10, 0.1)
    t = np.array([0.02, 1.0, 1.04, 5.55, 20.0])     # 20 s: outside the recording
    v = np.array([[1.0], [2.0], [4.0], [7.0], [9.0]])
    g = si.place_on_grid(t, v, grid)
    assert g[0, 0] == 1.0 and g[10, 0] == 3.0        # two readings in one sample: their mean
    assert g[55, 0] == 7.0 or g[56, 0] == 7.0         # 5.55 s is half-way: one of its two neighbours, not both
    assert np.isfinite(g[:, 0]).sum() == 3           # nothing invented between them, the outside one dropped
    # a non-uniform grid works the same
    g2 = si.place_on_grid(np.array([0.31]), np.array([[5.0]]), np.array([0.0, 0.1, 0.3, 0.6]))
    assert np.flatnonzero(np.isfinite(g2[:, 0])).tolist() == [2]


def test_held_fixes_collapse_and_impossible_fixes_are_dropped():
    t = np.arange(10.0)
    v = np.repeat(np.arange(5.0), 2)[:, None] * np.ones((1, 2))
    t2, v2, n = si.collapse_repeats(t, v)
    assert n == 5 and t2.tolist() == [0, 2, 4, 6, 8]
    # a 30 m jump at 1 Hz from a vehicle that does 0.5 m/s: dropped; honest 0.2 m noise isn't
    rng = np.random.default_rng(0)
    t = np.arange(100.0)
    v = np.c_[0.5 * t, np.zeros(100)] + rng.normal(0, 0.2, (100, 2))
    v[40] += 30.0
    t3, v3, bad = si.drop_unreachable_fixes(t, v, max_speed_mps=1.0, allowance_m=3 * np.sqrt(2) * 0.2)
    assert bad == 1 and 40.0 not in t3
    # a bad FIRST fix doesn't condemn the rest
    v[0] += 50.0
    _, _, bad = si.drop_unreachable_fixes(t, v, max_speed_mps=1.0, allowance_m=3 * np.sqrt(2) * 0.2)
    assert bad == 2


def test_rates_from_orientation_recover_a_known_turn():
    t = np.arange(0, 20, 0.1)
    R = Rotation.from_euler("ZYX", np.c_[0.3 * t, 0.1 * np.sin(t), np.zeros_like(t)]).as_matrix()
    w = si.rates_from_orientation(t, R)
    # body rates of a yaw-then-pitch sequence: r = yaw_rate*cos(pitch) when roll is 0
    expect_r = 0.3 * np.cos(0.1 * np.sin(t))
    assert np.allclose(w[2:-2, 2], expect_r[2:-2], atol=2e-3)


# ------------------------------------------------------------------ the checks, on planted errors

def test_gyro_check_catches_units_sign_and_reports_a_clock_offset():
    veh = synth_vehicle()
    R, rates, t = veh["R"], veh["rates"], veh["t"]
    assert "problem" not in si.gyro_vs_orientation(t, R, rates)
    deg = si.gyro_vs_orientation(t, R, np.degrees(rates))       # deg/s logged, rad/s declared
    assert "57: logged in deg/s" in deg["problem"]
    rad = si.gyro_vs_orientation(t, R, np.radians(rates))
    assert "1/57" in rad["problem"]
    flipped = si.gyro_vs_orientation(t, R, rates * np.array([1, 1, -1]))
    assert "r: the gyro turns the opposite way" in flipped["problem"]
    late = si.gyro_vs_orientation(t, R, np.roll(rates, 10, axis=0))  # the gyro 1 s late
    assert late["lag_s"]["r"] == pytest.approx(1.0, abs=0.11)
    assert "different clocks" in late.get("warning", "")
    noise = si.gyro_vs_orientation(t, R, np.random.default_rng(0).normal(0, 0.2, rates.shape))
    assert "problem" not in noise and "weren't checked" in noise["warning"]   # unrelated: said, not refused


def test_accelerometer_check_catches_g_units_and_a_frame_error():
    veh = synth_vehicle()
    assert "problem" not in si.accelerometer_gravity(veh["R"], veh["acc"], veh["rates"])
    assert "logged in g" in si.accelerometer_gravity(veh["R"], veh["acc"] / si.G, veh["rates"])["problem"]
    flu = veh["acc"] @ FLU_TO_FRD    # an unconverted ROS IMU
    assert "frame or mount" in si.accelerometer_gravity(veh["R"], flu, veh["rates"])["problem"]


def test_depth_check_catches_a_flipped_sign_and_wrong_units():
    z = np.sin(np.linspace(0, 10, 500)) * 3 + 5
    assert "problem" not in si.depth_vs_position(z + 0.4, z)
    assert "sign flipped" in si.depth_vs_position(-z, z)["problem"]
    assert "units" in si.depth_vs_position(z * 100, z)["problem"]


def test_heading_check_against_fixes_catches_a_mirrored_heading():
    veh = synth_vehicle(n=3000)
    pos = np.full_like(veh["pos"], np.nan)
    pos[::10] = veh["pos"][::10]
    ok = si.heading_vs_fixes(veh["t"], pos, veh["R"], veh["v"], "body velocity")
    assert "problem" not in ok and ok["match_r2"] > 0.9
    yaw_m = Rotation.from_matrix(veh["R"]).as_euler("ZYX")
    yaw_m[:, 0] *= -1
    Rm = Rotation.from_euler("ZYX", yaw_m).as_matrix()
    assert "mirrored" in si.heading_vs_fixes(veh["t"], pos, Rm, veh["v"], "body velocity")["problem"]


# ------------------------------------------------------------------ the spec

@pytest.mark.parametrize("block, message", [
    ({"position": {"use": "calibrate", "source": "odometry", "x": "a", "y": "b"}}, "can't be used to calibrate"),
    ({"position": {"x": "a"}}, "needs x and y"),
    ({"position": {"latlon": {"lat": "a", "lon": "b"}, "z": "c"}}, "takes no z"),
    ({"position": {"latlon": {"lat": "a", "lon": "b"}, "x": "c", "y": "d"}}, "not both"),
    ({"body_velocity": {"columns": {"u": "a", "v": "b", "p": "c", "q": "d", "r": "e"}}}, "give all three"),
    ({"depth": {"column": "a", "pressure": {"column": "b", "unit": "Pa"}}}, "exactly one of"),
    ({"depth": [{"column": "a"}, {"column": "b"}]}, "named"),
    ({"accelerometer": {"x": "a", "y": "b", "z": "c", "mount_rpy_deg": [0, 0]}}, "three angles"),
])
def test_spec_mistakes_are_refused_with_what_to_do(raw, tmp_path, block, message):
    write_csv(raw / "r.csv", synth_vehicle(n=200))
    s = spec_dict(raw, **block)
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(s))
    with pytest.raises(IngestError, match=message):
        load_ingest_spec(str(tmp_path / "bad.yaml"))


# ------------------------------------------------------------------ end to end, CSV

def test_a_vehicle_with_no_dvl_ingests_its_own_sensors(raw, tmp_path):
    veh = synth_vehicle()
    write_csv(raw / "r.csv", veh)
    s, (df,) = run(tmp_path, spec_dict(raw))
    # no velocity column at all: nothing for a calibration to peek at
    assert not {"Bu", "Bv", "Bw"} & set(df.columns)
    assert {"pos_usbl_n", "pos_usbl_e", "pos_usbl_d", "depth_main", "acc_x", "acc_y", "acc_z"} <= set(df.columns)
    # every fix where it happened, the samples between empty
    fixes = np.flatnonzero(df["pos_usbl_n"].notna())
    assert len(fixes) == 120 and np.all(np.diff(fixes) == 10)
    assert np.allclose(df.loc[fixes, ["pos_usbl_n", "pos_usbl_e", "pos_usbl_d"]].values, veh["pos"][::10], atol=1e-9)
    assert df["depth_main"].notna().sum() == 600
    assert np.allclose(df["acc_z"], veh["acc"][:, 2])
    sens = s["sensors"]
    assert sens["axes_not_seen"] == []
    assert sens["axes_seen_by"]["surge"] == ["horizontal position"]
    assert sens["axes_seen_by"]["heave"] == ["depth", "vertical position"]
    assert sens["sensors"]["usbl"]["rate_hz"] == pytest.approx(1.0) and sens["sensors"]["main"]["rate_hz"] == 5.0
    checks = s["files"][0]["sensor_checks"]
    assert "problem" not in checks["gyro_vs_orientation"] and "problem" not in checks["accelerometer"]
    assert "problem" not in checks["heading_vs_position"]["usbl"]
    # the force-direction check still runs, from the fixes and depth
    fd = s["force_direction"]
    assert fd["velocity_from"].startswith("position fixes") and fd["correlation_force_vs_velocity"]["surge"] > 0.2
    assert fd["reversed_axes"] == []


def test_only_depth_and_gyro_says_what_it_cannot_see(raw, tmp_path):
    write_csv(raw / "r.csv", synth_vehicle())
    s, (df,) = run(tmp_path, spec_dict(raw, position=None, accelerometer=None))
    assert s["sensors"]["axes_not_seen"] == ["surge", "sway"]
    assert any("no sensor sees surge, sway motion" in w for w in s["warnings"])
    assert s["force_direction"]["correlation_force_vs_velocity"]["surge"] is None


def test_noise_is_measured_on_the_sensors_own_readings(raw, tmp_path):
    write_csv(raw / "r.csv", synth_vehicle(n=4000), noise=0.17)
    s, _ = run(tmp_path, spec_dict(raw))
    noise = s["sensors"]["sensors"]["usbl"]["noise_std"]
    # the SOLAQUA USBL's 0.17 m, measured from 1 Hz fixes among 10 Hz rows (the motion between fixes adds a little)
    assert all(0.15 < x < 0.21 for x in noise), noise


def test_gps_latlon_with_no_fix_underwater(raw, tmp_path):
    veh = synth_vehicle()
    df = write_csv(raw / "r.csv", veh, gps_latlon=True)
    df["fix_status"] = np.where(np.arange(len(df)) < 600, 0, -1)   # a fix for the first minute only
    df.loc[650, ["lat", "lon"]] = 0.0                              # a receiver reporting 0, 0 with no fix
    df.to_csv(raw / "r.csv", index=False)
    pos = {"use": "calibrate", "source": "gps", "latlon": {"lat": "lat", "lon": "lon"},
           "drop_when": [{"column": "fix_status", "below": 0}]}
    s, (out,) = run(tmp_path, spec_dict(raw, position=pos))
    fixes = np.flatnonzero(out["pos_gps_n"].notna())
    assert fixes.max() < 600 and len(fixes) == 60
    assert "pos_gps_d" not in out.columns
    ch = s["files"][0]["sensors"]["gps"]
    assert ch["dropped"]["by_drop_when"] == 59 and ch["dropped"]["no_fix_zero"] == 1
    # north/east relative to the first fix, to the sphere-vs-ellipsoid difference
    truth = veh["pos"][fixes, :2] - veh["pos"][0, :2]
    assert np.allclose(out.loc[fixes, ["pos_gps_n", "pos_gps_e"]].values, truth, atol=0.02 * 60 + 0.05)


def test_pressure_in_mbar_becomes_depth(raw, tmp_path):
    veh = synth_vehicle()
    df = write_csv(raw / "r.csv", veh)
    df["press"] = (101325.0 + 1025 * si.G * df["depth"]) / 100.0
    df.to_csv(raw / "r.csv", index=False)
    dep = {"name": "bar30", "pressure": {"column": "press", "unit": "mbar", "water": "salt", "surface": 1013.25}}
    s, (out,) = run(tmp_path, spec_dict(raw, depth=dep))
    ok = out["depth_bar30"].notna()
    assert np.allclose(out.loc[ok, "depth_bar30"], veh["pos"][::2, 2], atol=1e-6)


def test_a_delayed_sensor_is_put_back_in_time(raw, tmp_path):
    veh = synth_vehicle()
    write_csv(raw / "r.csv", veh, delay_rows=3)               # the USBL reports 0.3 s late
    pos = {"use": "calibrate", "source": "usbl", "x": "px", "y": "py", "z": "pz", "delay_s": 0.3}
    _, (out,) = run(tmp_path, spec_dict(raw, position=pos))
    fixes = np.flatnonzero(out["pos_usbl_n"].notna())
    assert np.allclose(out.loc[fixes, ["pos_usbl_n", "pos_usbl_e"]].values, veh["pos"][fixes, :2], atol=1e-9)


def test_held_fixes_in_a_csv_are_one_reading_each(raw, tmp_path):
    veh = synth_vehicle()
    write_csv(raw / "r.csv", veh, hold_fixes=True)
    s, (out,) = run(tmp_path, spec_dict(raw))
    assert out["pos_usbl_n"].notna().sum() == 120
    assert s["files"][0]["sensors"]["usbl"]["dropped"]["held_repeats"] == 1080


def test_resampling_a_csv_never_interpolates_the_fixes(raw, tmp_path):
    veh = synth_vehicle()
    write_csv(raw / "r.csv", veh)
    _, (out,) = run(tmp_path, spec_dict(raw, resample_hz=20.0))
    assert len(out) >= 2 * (len(veh["t"]) - 1)        # the 20 Hz grid (its length is the CSV resampler's)
    assert out["pos_usbl_n"].notna().sum() == 120 and out["depth_main"].notna().sum() == 600


@pytest.mark.parametrize("planted, kw, message", [
    ("gyro in deg/s", {"gyro_scale": np.degrees(1.0)}, "logged in deg/s"),
    ("accelerometer in g", {"acc_scale": 1 / si.G}, "logged in g"),
    ("depth flipped", {"depth_scale": -1.0}, "sign flipped"),
    ("depth in cm", {"depth_scale": 100.0}, "unit"),
])
def test_planted_sensor_errors_are_refused(raw, tmp_path, planted, kw, message):
    write_csv(raw / "r.csv", synth_vehicle(), **kw)
    with pytest.raises(IngestError, match=message):
        run(tmp_path, spec_dict(raw))


def test_a_check_can_be_skipped_and_says_so(raw, tmp_path):
    write_csv(raw / "r.csv", synth_vehicle(), acc_scale=1 / si.G)
    s, _ = run(tmp_path, spec_dict(raw, skip_checks=["accelerometer"]))
    assert any("check skipped" in w for w in s["files"][0]["warnings"])


def test_impossible_position_units_are_refused(raw, tmp_path):
    veh = synth_vehicle()
    df = write_csv(raw / "r.csv", veh, noise=0.05)
    bad = df["px"].notna() & (np.arange(len(df)) % 20 == 0)
    df.loc[bad, "px"] += 500.0       # every other fix 500 m off: two systems' x, or millimetres in metres
    df.to_csv(raw / "r.csv", index=False)
    with pytest.raises(IngestError, match="than the vehicle could move"):
        run(tmp_path, spec_dict(raw))


def test_no_gyro_logged_derives_rates_with_a_warning(raw, tmp_path):
    veh = synth_vehicle()
    write_csv(raw / "r.csv", veh)
    s, (out,) = run(tmp_path, spec_dict(raw, body_velocity=None))
    assert np.allclose(out[["Bp", "Bq", "Br"]].values[2:-2], veh["rates"][2:-2], atol=1e-6)
    assert s["files"][0]["angular_rates_from"].startswith("the orientation")
    assert any("no gyro rates logged" in w for w in s["files"][0]["warnings"])


def test_a_dvl_with_gaps_kept(raw, tmp_path):
    veh = synth_vehicle()
    df = write_csv(raw / "r.csv", veh)
    dvl = veh["v"].copy()
    lost = (np.arange(len(df)) // 50) % 3 == 0      # bottom lock lost a third of the time
    dvl[lost] = np.nan
    df["du"], df["dv"], df["dw"] = dvl[:, 0], dvl[:, 1], dvl[:, 2]
    df.to_csv(raw / "r.csv", index=False)
    vel = {"columns": {"u": "du", "v": "dv", "w": "dw", "p": "gp", "q": "gq", "r": "gr"}, "linear_gaps": "keep"}
    s, (out,) = run(tmp_path, spec_dict(raw, body_velocity=vel))
    assert out["Bu"].isna().sum() == lost.sum() and len(out) == len(df)
    assert np.allclose(out.loc[~lost, ["Bu", "Bv", "Bw"]].values, veh["v"][~lost])
    assert s["files"][0]["sensors"]["dvl"]["coverage_pct"] == pytest.approx(100 * (~lost).mean(), abs=0.1)
    assert "dvl" in s["sensors"]["axes_seen_by"]["surge"]


def test_a_ros_flu_vehicle_is_converted(raw, tmp_path):
    """The same vehicle logged the ROS way (FLU body, ENU world): the canonical output is identical."""
    veh = synth_vehicle()
    df = write_csv(raw / "r.csv", veh)
    _, (frd,) = run(tmp_path, spec_dict(raw), "frd")
    df[["Fy", "Fz", "Ty", "Tz"]] *= -1
    df[["gq", "gr"]] *= -1
    q = Rotation.from_matrix(ENU_TO_NED @ veh["R"] @ FLU_TO_FRD).as_quat()
    df[["qx", "qy", "qz", "qw"]] = q
    enu = df[["px", "py", "pz"]].values @ ENU_TO_NED      # NED -> ENU (the matrix is its own inverse)
    df["px"], df["py"], df["pz"] = enu[:, 0], enu[:, 1], enu[:, 2]
    df[["ay", "az"]] *= -1
    df.to_csv(raw / "r.csv", index=False)
    _, (flu,) = run(tmp_path, spec_dict(raw, frame="flu"), "flu")
    cols = [c for c in frd.columns if c not in ("qx", "qy", "qz", "qw")]
    assert np.allclose(frd[cols].fillna(0).values, flu[cols].fillna(0).values, atol=1e-9)
    assert np.allclose(np.abs((Rotation.from_quat(frd[["qx", "qy", "qz", "qw"]].values).inv()
                               * Rotation.from_quat(flu[["qx", "qy", "qz", "qw"]].values)).magnitude()), 0, atol=1e-9)


def test_old_specs_still_mean_what_they_meant(raw, tmp_path):
    """A spec from before #96 (six velocity columns, position for the check) gives the same columns as before."""
    veh = synth_vehicle()
    df = write_csv(raw / "r.csv", veh)
    df["u"], df["v"], df["w"] = veh["v"][:, 0], veh["v"][:, 1], veh["v"][:, 2]
    df[["px", "py", "pz"]] = veh["pos"]
    df.to_csv(raw / "r.csv", index=False)
    old = spec_dict(raw, body_velocity={"columns": {"u": "u", "v": "v", "w": "w", "p": "gp", "q": "gq", "r": "gr"}},
                    position={"x": "px", "y": "py", "z": "pz"}, depth=None, accelerometer=None)
    s, (out,) = run(tmp_path, old)
    assert list(out.columns) == ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp", "Bq", "Br",
                                 "qx", "qy", "qz", "qw"]
    assert "sensors" not in s and "problem" not in s["files"][0]["orientation_vs_position"]


# ------------------------------------------------------------------ end to end, a ROS bag with real message types

def _bag_with_sensors(path, veh, gps_until_s=60.0):
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore
    ts = get_typestore(Stores.ROS1_NOETIC)
    g = ts.types
    T0 = 1.7e9
    flip = np.array([1.0, -1.0, -1.0])
    q_enu = Rotation.from_matrix(ENU_TO_NED @ veh["R"] @ FLU_TO_FRD).as_quat()

    def header(t):
        return g["std_msgs/msg/Header"](seq=0, stamp=g["builtin_interfaces/msg/Time"](
            sec=int(t), nanosec=int(round((t - int(t)) * 1e9))), frame_id="")
    msgs = []
    V3, Q = g["geometry_msgs/msg/Vector3"], g["geometry_msgs/msg/Quaternion"]
    for k, t in enumerate(veh["t"]):
        a = veh["acc"][k] * flip
        w = veh["rates"][k] * flip
        imu = g["sensor_msgs/msg/Imu"](header=header(T0 + t), orientation=Q(x=q_enu[k, 0], y=q_enu[k, 1], z=q_enu[k, 2],
                                                                           w=q_enu[k, 3]),
                                       orientation_covariance=np.zeros(9), angular_velocity=V3(x=w[0], y=w[1], z=w[2]),
                                       angular_velocity_covariance=np.zeros(9),
                                       linear_acceleration=V3(x=a[0], y=a[1], z=a[2]),
                                       linear_acceleration_covariance=np.zeros(9))
        msgs.append((T0 + t, "/imu", "sensor_msgs/msg/Imu", imu))
        F = veh["wrench"][k] * np.r_[flip, flip]
        wr = g["geometry_msgs/msg/WrenchStamped"](header=header(T0 + t), wrench=g["geometry_msgs/msg/Wrench"](
            force=V3(x=F[0], y=F[1], z=F[2]), torque=V3(x=F[3], y=F[4], z=F[5])))
        msgs.append((T0 + t, "/wrench", "geometry_msgs/msg/WrenchStamped", wr))
        if k % 4 == 0:   # a pressure sensor at 2.5 Hz, Pa
            p = g["sensor_msgs/msg/FluidPressure"](header=header(T0 + t + 0.013),
                                                   fluid_pressure=101325.0 + 997 * si.G * veh["pos"][k, 2], variance=0.0)
            msgs.append((T0 + t + 0.013, "/pressure", "sensor_msgs/msg/FluidPressure", p))
        if k % 10 == 3:  # GPS at 1 Hz, with a fix only while surfaced (the first minute)
            n_, e_ = veh["pos"][k, 0], veh["pos"][k, 1]
            st = g["sensor_msgs/msg/NavSatStatus"](status=0 if t < gps_until_s else -1, service=1)
            fix = g["sensor_msgs/msg/NavSatFix"](header=header(T0 + t), status=st,
                                                 latitude=41.0 + np.degrees(n_ / 6378137.0),
                                                 longitude=-71.0 + np.degrees(e_ / (6378137.0 * np.cos(np.radians(41)))),
                                                 altitude=0.0, position_covariance=np.zeros(9),
                                                 position_covariance_type=0)
            msgs.append((T0 + t, "/fix", "sensor_msgs/msg/NavSatFix", fix))
    msgs.sort(key=lambda m: m[0])
    with Writer(path) as wtr:
        conns = {}
        for t, tp, mt, m in msgs:
            if tp not in conns:
                conns[tp] = wtr.add_connection(tp, mt, typestore=ts)
            wtr.write(conns[tp], int(t * 1e9), ts.serialize_ros1(m, mt))


def test_a_ros_bag_with_imu_pressure_and_gps(raw, tmp_path):
    veh = synth_vehicle()
    _bag_with_sensors(str(raw / "dive.bag"), veh)
    spec = {"vehicle": "synth", "source": {"files": str(raw / "*.bag"), "format": "rosbag"}, "resample_hz": 10,
            "session_date": {"from": "recording"}, "frame": "flu",
            "body_velocity": {"columns": {k: f"/imu.angular_velocity.{a}" for k, a in zip("pqr", "xyz")}},
            "orientation": {"quaternion": {a: f"/imu.orientation.{a}" for a in "xyzw"}},
            "wrench": {"columns": {"Fx": "/wrench.wrench.force.x", "Fy": "/wrench.wrench.force.y",
                                   "Fz": "/wrench.wrench.force.z", "Tx": "/wrench.wrench.torque.x",
                                   "Ty": "/wrench.wrench.torque.y", "Tz": "/wrench.wrench.torque.z"}},
            "depth": {"name": "baro", "pressure": {"column": "/pressure.fluid_pressure", "unit": "Pa",
                                                   "surface": 101325.0}},
            "position": {"use": "calibrate", "source": "gps",
                         "latlon": {"lat": "/fix.latitude", "lon": "/fix.longitude"},
                         "drop_when": [{"column": "/fix.status.status", "below": 0}]},
            "accelerometer": {k: f"/imu.linear_acceleration.{k}" for k in "xyz"}}
    s, (out,) = run(tmp_path, spec)
    # the GPS has fixes for one minute of two, yet the recording isn't cut to that minute
    assert len(out) >= len(veh["t"]) - 2
    fixes = np.flatnonzero(out["pos_gps_n"].notna())
    assert 55 <= len(fixes) <= 60 and out.index[fixes].max() < 610
    dep = out["depth_baro"].notna()
    assert 290 <= dep.sum() <= 301
    # the pressure readings came 13 ms after the grid's samples: each sits in its own sample, converted
    k = np.flatnonzero(dep)
    assert np.allclose(out.loc[dep, "depth_baro"], veh["pos"][k, 2], atol=0.02)
    assert np.allclose(out["acc_z"], veh["acc"][: len(out), 2], atol=1e-6)
    assert s["sensors"]["axes_not_seen"] == []
    json.dumps(s)   # nothing un-serialisable leaked into the report
    assert os.path.isfile(tmp_path / "out_s" / "ingest_report.json")
