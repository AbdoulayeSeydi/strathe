"""
Lab-data tests for ROS bags, custom thruster layouts, the ingest templates and
thrust limiting. Where the reference dataset is available, real runs are
written into ROS 1 / ROS 2 bags the way a BlueROV2 with MAVROS would record
them (ROS FLU/ENU frames, header stamps, thruster PWM as an RCOut array) and
ingestion must recover the original canonical data.
"""
import glob
import json
import os

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.spatial.transform import Rotation

import thruster_model as tm
from data_ingestion import (ENU_TO_NED, FLU_TO_FRD, IngestError, ingest, load_ingest_spec, reference_run_in_frd,
                            torque_reference_for)

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(HERE, "..", "templates")
DATA = os.path.join(HERE, "..", "..", "data", "raw", "Bluerov2_trajectory_dataset", "dataset")
REAL = sorted(glob.glob(os.path.join(DATA, "*.csv")))
needs_data = pytest.mark.skipif(not REAL, reason="BlueROV2 dataset not downloaded")
rosbags = pytest.importorskip("rosbags")
from rosbags.rosbag1 import Writer as Writer1  # noqa: E402
from rosbags.rosbag2 import StoragePlugin, Writer as Writer2  # noqa: E402
from rosbags.typesys import Stores, get_types_from_msg, get_typestore  # noqa: E402

RCOUT_MSG = "std_msgs/Header header\nuint16[] channels\n"
T0 = 1_707_000_000.0  # 2024-02-03 22:40 UTC


# ------------------------------------------------------------------ thrusters

def test_every_template_loads_and_builds_its_thrusters(tmp_path):
    names = sorted(glob.glob(os.path.join(TEMPLATES, "ingest_*.yaml")))
    assert len(names) == 15   # + PX4, MOOS-IvP, LCM (#94); + ArduSub no-DVL, ROS sensors, CSV with a USBL (#96)
    for path in names:
        spec = load_ingest_spec(path)
        if spec.wrench.thrusters is not None:
            assert spec._thruster_cfg.n_thrusters == len(spec.wrench.thrusters.pwm_columns)


def test_custom_layout_template_is_exactly_the_bluerov2_heavy():
    layout_spec = load_ingest_spec(os.path.join(TEMPLATES, "ingest_csv_pwm_custom_thrusters.yaml"))
    preset_spec = load_ingest_spec(os.path.join(TEMPLATES, "ingest_csv_pwm_bluerov2_heavy.yaml"))
    lay, pre = layout_spec._thruster_cfg, preset_spec._thruster_cfg
    np.testing.assert_allclose(lay.allocation, pre.for_simulation().allocation, atol=1e-4)
    np.testing.assert_array_equal(lay.directions, pre.directions)
    rng = np.random.default_rng(0)
    pwm = rng.uniform(1100, 1900, (500, 8))
    np.testing.assert_allclose(lay.wrench(pwm), tm.ThrusterConfig(pre.for_simulation().allocation, pre.directions,
                                                                   pre.curve).wrench(pwm), atol=0.02)
    # the two specs tell calibration how to treat their torques: the preset keeps the dataset's reference
    # point (shift 8.5 cm, z down in FRD); a layout is already about the thruster plane
    assert torque_reference_for(preset_spec) == [0.0, 0.0, 0.085]
    assert torque_reference_for(layout_spec) == [0.0, 0.0, 0.0]


def test_allocation_template_matches_the_layout():
    cfg = tm.thruster_block({"allocation_csv": "allocation_template.csv", "directions": [-1, -1, 1, 1, -1, 1, 1, -1],
                             "curve_csv": "thrust_curve_template.csv"}, TEMPLATES)
    np.testing.assert_allclose(cfg.allocation, tm.bluerov2_heavy().for_simulation().allocation, atol=1e-4)
    full = tm.ThrustCurve.t200_bluerov2_dataset()
    for p in (1100, 1300, 1500, 1700, 1900):  # the template samples the real curve
        assert cfg.curve(p) == pytest.approx(float(full(p)), abs=0.01)


def test_six_thruster_vehicle_without_pitch_authority():
    """A 6-thruster BlueROV2 (ArduSub's vectored frame, FRD): 4 vectored horizontal + 2 vertical on the
    centreline."""
    layout = [{"position_m": [0.1335, 0.1335, 0], "direction": [-1, 1, 0], "reversed": True},
              {"position_m": [0.1335, -0.1335, 0], "direction": [-1, -1, 0], "reversed": True},
              {"position_m": [-0.1335, 0.1335, 0], "direction": [1, 1, 0]},
              {"position_m": [-0.1335, -0.1335, 0], "direction": [1, -1, 0]},
              {"position_m": [0, 0.11, 0], "direction": [0, 0, 1], "reversed": True},
              {"position_m": [0, -0.11, 0], "direction": [0, 0, 1]}]
    cfg = tm.thruster_block({"layout": layout, "curve": "t200", "voltage": 16})
    assert cfg.unreachable_axes() == ["pitch torque"]
    w = cfg.achievable_wrench([10, 0, 0, 0, 5, 0])
    assert w[4] == pytest.approx(0, abs=1e-9) and w[0] == pytest.approx(10)
    assert "can't push: pitch torque" in tm.describe(cfg)


def test_achievable_wrench_respects_every_thruster_limit_and_keeps_direction():
    cfg = tm.bluerov2_heavy().for_simulation()
    B, pinv = cfg.allocation, np.linalg.pinv(cfg.allocation)
    rng = np.random.default_rng(1)
    taus = rng.uniform(-1, 1, (5000, 6)) * rng.choice([5.0, 30.0, 200.0], (5000, 1))
    out = cfg.achievable_wrench(taus)
    thrust = out @ pinv.T
    assert thrust.max() <= cfg.curve.max_forward_n + 1e-6 and -thrust.min() <= cfg.curve.max_reverse_n + 1e-6
    # direction kept: the achieved wrench is a non-negative multiple (<= 1) of the command
    scale = np.einsum("ij,ij->i", out, taus) / np.einsum("ij,ij->i", taus, taus)
    np.testing.assert_allclose(out, taus * scale[:, None], atol=1e-8)
    assert np.all((scale > 0) & (scale <= 1 + 1e-12))
    small = np.array([5.0, -3, 2, 0.5, -0.3, 0.4])
    np.testing.assert_allclose(cfg.achievable_wrench(small), small, atol=1e-9)  # within limits: untouched
    mw = cfg.max_wrench()
    assert mw[0] == pytest.approx(105, abs=2) and mw[4] == pytest.approx(17.9, abs=0.5)


def test_simulation_view_moves_torques_to_the_thruster_plane():
    """The preset reproduces the dataset's torque point; the simulators need torques about the model's
    origin, which is the same 8.5 cm shift calibration applies (vehicle_spec.yaml data_conventions)."""
    from objective import shift_torque_reference
    pre = tm.bluerov2_heavy()
    pwm = np.random.default_rng(2).uniform(1100, 1900, (200, 8))
    np.testing.assert_allclose(pre.for_simulation().wrench(pwm),
                               shift_torque_reference(pre.wrench(pwm), pre.torque_reference_offset_m), atol=1e-9)
    assert np.abs(pre.for_simulation().allocation[3:5, :4]).max() < 1e-5  # horizontals: no roll/pitch torque


@pytest.mark.parametrize("block,match", [
    ({"preset": "bluerov2_heavy", "layout": [{"position_m": [0, 0, 0], "direction": [1, 0, 0]}]}, "exactly one of"),
    ({"preset": "bluerov2"}, "only preset is bluerov2_heavy"),
    ({"preset": "bluerov2_heavy", "curve_csv": "x.csv"}, "fixes the layout and curve"),
    ({"layout": [{"position_m": [0, 0, 0], "direction": [1, 0, 0]}], "curve_csv": "thrust_curve_template.csv",
      "voltage": 15}, "give curve_voltage"),
    ({"layout": [{"position_m": [0, 0], "direction": [1, 0, 0]}]}, "3 numbers"),
    ({"layout": [{"position_m": [0, 0, 0], "direction": [0, 0, 0]}]}, "all zeros"),
    ({"layout": [{"position_m": [0, 0, 0], "dir": [1, 0, 0]}]}, "needs position_m and direction"),
    ({"layout": [{"position_m": [0, 0, 0], "direction": [1, 0, 0]}], "directions": [1]}, "reversed: true"),
    ({"allocation_csv": "allocation_template.csv"}, "needs directions"),
    ({"preset": "bluerov2_heavy", "thrust": 3}, "unknown thrusters key"),
    ({"layout": [{"position_m": [0, 0, 0], "direction": [1, 0, 0]}], "curve": "t300"}, "unknown curve"),
])
def test_bad_thruster_blocks_are_refused(block, match):
    with pytest.raises(ValueError, match=match):
        tm.thruster_block(block, TEMPLATES)


def test_thrusters_can_come_from_an_ingest_spec_file():
    cfg = tm.thruster_block(os.path.join(TEMPLATES, "ingest_csv_pwm_custom_thrusters.yaml"))
    assert cfg.n_thrusters == 8 and cfg.curve.max_forward_n > 40


# ------------------------------------------------------------------ CSV edge cases

def test_resampling_across_a_quaternion_sign_flip(tmp_path):
    """Yaw sweeping through 180 deg flips the sign of a canonical (w >= 0) quaternion; interpolating
    across the flip used to produce a near-zero quaternion, i.e. a random orientation for one sample."""
    raw = tmp_path / "raw"
    raw.mkdir()
    n = 200
    rng = np.random.default_rng(3)
    t = np.arange(n) * 0.1 + rng.uniform(-0.02, 0.02, n)
    yaw = np.linspace(170, 190, n)
    q = Rotation.from_euler("z", yaw[:, None], degrees=True).as_quat()
    q[q[:, 3] < 0] *= -1  # canonical form: sign flips at yaw = 180
    pd.DataFrame({"t": t, "u": 0.2, "v": 0.0, "w": 0.0, "p": 0.0, "q": 0.0, "r": 0.1,
                  "qx": q[:, 0], "qy": q[:, 1], "qz": q[:, 2], "qw": q[:, 3],
                  **{k: 1.0 for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}}).to_csv(raw / "2024-01-12_a.csv", index=False)
    spec = {"vehicle": "flip", "source": {"files": "raw/*.csv"}, "time": {"column": "t"}, "resample_hz": 10,
            "session_date": {"from": "filename", "pattern": r"(\d{4}-\d{2}-\d{2})", "format": "%Y-%m-%d"},
            "body_velocity": {"columns": {k: k for k in "uvwpqr"}},
            "orientation": {"quaternion": {k: f"q{k}" for k in "xyzw"}},
            "wrench": {"columns": {k: k for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}}}
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(spec))
    ingest(load_ingest_spec(str(tmp_path / "s.yaml")), str(tmp_path / "out"))
    got = pd.read_csv(tmp_path / "out" / "flip-12-01-2024-run1.csv")
    yaw_got = np.degrees(np.unwrap(Rotation.from_quat(got[["qx", "qy", "qz", "qw"]].values).as_euler("xyz")[:, 2]))
    assert np.abs(np.diff(yaw_got)).max() < 1.0  # smooth: no sample jumps to a random heading


def test_session_date_from_recording_needs_clock_time(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    n = 50
    df = pd.DataFrame({"t": T0 + np.arange(n) * 0.1, **{k: 0.1 for k in "uvwpqr"}, "qx": 0.0, "qy": 0.0, "qz": 0.0,
                       "qw": 1.0, **{k: 1.0 for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}})
    df.to_csv(raw / "log.csv", index=False)
    spec = {"vehicle": "rec", "source": {"files": "raw/*.csv"}, "time": {"column": "t"},
            "session_date": {"from": "recording"}, "body_velocity": {"columns": {k: k for k in "uvwpqr"}},
            "orientation": {"quaternion": {k: f"q{k}" for k in "xyzw"}},
            "wrench": {"columns": {k: k for k in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}}}
    (tmp_path / "s.yaml").write_text(yaml.safe_dump(spec))
    assert ingest(load_ingest_spec(str(tmp_path / "s.yaml")), str(tmp_path / "o"))["sessions"] == {"03-02-2024": 1}
    df.assign(t=np.arange(n) * 0.1).to_csv(raw / "log.csv", index=False)
    with pytest.raises(IngestError, match="real clock timestamps"):
        ingest(load_ingest_spec(str(tmp_path / "s.yaml")), str(tmp_path / "o2"))


# ------------------------------------------------------------------ ROS bags

def _typestore(ros1: bool):
    ts = get_typestore(Stores.ROS1_NOETIC if ros1 else Stores.ROS2_HUMBLE)
    ts.register(get_types_from_msg(RCOUT_MSG, "mavros_msgs/msg/RCOut"))
    return ts


def _header(ts, t, frame=""):
    T = ts.types["builtin_interfaces/msg/Time"]
    H = ts.types["std_msgs/msg/Header"]
    stamp = T(sec=int(np.floor(t)), nanosec=int(round((t - np.floor(t)) * 1e9)))
    return H(seq=0, stamp=stamp, frame_id=frame) if "seq" in H.__dataclass_fields__ else H(stamp=stamp, frame_id=frame)


def _odom(ts, t, lin, ang, q):
    g = ts.types
    V3, Q, P = g["geometry_msgs/msg/Vector3"], g["geometry_msgs/msg/Quaternion"], g["geometry_msgs/msg/Point"]
    pose = g["geometry_msgs/msg/PoseWithCovariance"](
        pose=g["geometry_msgs/msg/Pose"](position=P(x=0.0, y=0.0, z=0.0), orientation=Q(x=q[0], y=q[1], z=q[2], w=q[3])),
        covariance=np.zeros(36))
    twist = g["geometry_msgs/msg/TwistWithCovariance"](
        twist=g["geometry_msgs/msg/Twist"](linear=V3(x=lin[0], y=lin[1], z=lin[2]), angular=V3(x=ang[0], y=ang[1], z=ang[2])),
        covariance=np.zeros(36))
    return g["nav_msgs/msg/Odometry"](header=_header(ts, t, "odom"), child_frame_id="base_link", pose=pose, twist=twist)


def _ros_frames(d):
    """Canonical FRD/NED run -> ROS FLU body velocities and ENU<-FLU quaternions."""
    flip = np.array([1.0, -1.0, -1.0])
    lin = d[["Bu", "Bv", "Bw"]].values * flip
    ang = d[["Bp", "Bq", "Br"]].values * flip
    R = ENU_TO_NED @ Rotation.from_quat(d[["qx", "qy", "qz", "qw"]].values).as_matrix() @ FLU_TO_FRD
    return lin, ang, Rotation.from_matrix(R).as_quat()


def write_bag(path, d, kind="ros1", odom_dt=0.1, rc_dt=0.1, rc_offset=0.0, odom_offset=0.0, recv_delay=0.004,
              rc_header=True):
    """Write run `d` as /odom (Odometry) + /mavros/rc/out (RCOut) messages. kind: ros1 | mcap | sqlite3.
    recv_delay: receive time - header stamp (days for a re-recorded bag); rc_header=False: RCOut stamps are 0."""
    ts = _typestore(kind == "ros1")
    lin, ang, q = _ros_frames(d)
    pwm = np.ascontiguousarray(d[[f"pwm{i}" for i in range(1, 9)]].values.astype(np.uint16))
    n = len(d)
    t_data = T0 + np.arange(n) * 0.1
    msgs = []
    for tt in np.arange(0, (n - 1) * 0.1 + 1e-9, odom_dt) + odom_offset:
        k = min(int(round((tt - odom_offset) / 0.1)), n - 1)
        msgs.append((T0 + tt, "/odom", "nav_msgs/msg/Odometry", _odom(ts, T0 + tt, lin[k], ang[k], q[k])))
    for tt in np.arange(0, (n - 1) * 0.1 + 1e-9, rc_dt) + rc_offset:
        k = min(int(np.floor((tt - rc_offset) / 0.1 + 1e-9)), n - 1)
        msgs.append((T0 + tt, "/mavros/rc/out", "mavros_msgs/msg/RCOut",
                     ts.types["mavros_msgs/msg/RCOut"](header=_header(ts, T0 + tt if rc_header else 0.0),
                                                       channels=np.ascontiguousarray(pwm[k]))))
    msgs.sort(key=lambda m: m[0])
    if kind == "ros1":
        with Writer1(path) as w:
            conns = {tp: w.add_connection(tp, mt, typestore=ts) for tp, mt in
                     (("/odom", "nav_msgs/msg/Odometry"), ("/mavros/rc/out", "mavros_msgs/msg/RCOut"))}
            for t, tp, mt, m in msgs:
                w.write(conns[tp], int((t + recv_delay) * 1e9), ts.serialize_ros1(m, mt))  # received recv_delay later
    else:
        plugin = StoragePlugin.MCAP if kind == "mcap" else StoragePlugin.SQLITE3
        with Writer2(path, version=9, storage_plugin=plugin) as w:
            conns = {tp: w.add_connection(tp, mt, typestore=ts) for tp, mt in
                     (("/odom", "nav_msgs/msg/Odometry"), ("/mavros/rc/out", "mavros_msgs/msg/RCOut"))}
            for t, tp, mt, m in msgs:
                w.write(conns[tp], int((t + recv_delay) * 1e9), ts.serialize_cdr(m, mt))
    return t_data


def strip_definitions(bag_dir):
    """Make a ROS 2 .db3 bag look like one recorded before ROS 2 Iron: no stored message definitions."""
    import sqlite3
    for db in glob.glob(os.path.join(str(bag_dir), "*.db3")):
        con = sqlite3.connect(db)
        con.execute("DELETE FROM message_definitions")
        con.commit()
        con.close()


def bag_spec(tmp_path, files, **over):
    odom = "/odom"
    spec = {"vehicle": "bagrov", "source": {"files": files, "format": "rosbag"}, "resample_hz": 10,
            "session_date": {"from": "recording"}, "frame": "flu",
            "body_velocity": {"columns": {"u": f"{odom}.twist.twist.linear.x", "v": f"{odom}.twist.twist.linear.y",
                                          "w": f"{odom}.twist.twist.linear.z", "p": f"{odom}.twist.twist.angular.x",
                                          "q": f"{odom}.twist.twist.angular.y", "r": f"{odom}.twist.twist.angular.z"}},
            "orientation": {"quaternion": {k: f"{odom}.pose.pose.orientation.{k}" for k in "xyzw"}},
            "wrench": {"thrusters": {"pwm_columns": [f"/mavros/rc/out.channels[{i}]" for i in range(8)],
                                     "preset": "bluerov2_heavy"}}}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(spec.get(k), dict):
            spec[k] = {**spec[k], **v}
        else:
            spec[k] = v
    p = tmp_path / "bag_spec.yaml"
    p.write_text(yaml.safe_dump(spec))
    return str(p)


def _check_matches(got_csv, d, atol_v=1e-6, atol_f=5e-3):
    got = pd.read_csv(got_csv)
    n = min(len(got), len(d))
    for c in ("Bu", "Bv", "Bw", "Bp", "Bq", "Br"):
        np.testing.assert_allclose(got[c].values[:n], d[c].values[:n], atol=atol_v, err_msg=c)
    for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz"):
        np.testing.assert_allclose(got[c].values[:n], d[c].values[:n], atol=atol_f, err_msg=c)
    Rg = Rotation.from_quat(got[["qx", "qy", "qz", "qw"]].values[:n])
    Ro = Rotation.from_quat(d[["qx", "qy", "qz", "qw"]].values[:n])
    assert np.degrees((Rg.inv() * Ro).magnitude()).max() < 1e-4
    return got


@needs_data
@pytest.mark.parametrize("kind", ["ros1", "mcap", "sqlite3"])
def test_real_runs_round_trip_through_ros_bags(tmp_path, kind):
    d = reference_run_in_frd(pd.read_csv(REAL[0])).iloc[:600].reset_index(drop=True)
    target = tmp_path / "bags" / ("run.bag" if kind == "ros1" else "run")
    target.parent.mkdir()
    write_bag(target, d, kind)
    over = {}
    if kind == "sqlite3":  # older .db3 bags don't store message definitions: the lab supplies the .msg file
        strip_definitions(target)
        msg = tmp_path / "msgs" / "mavros_msgs" / "msg" / "RCOut.msg"
        msg.parent.mkdir(parents=True)
        msg.write_text(RCOUT_MSG)
        over["source"] = {"message_types": [str(msg)]}
    summary = ingest(load_ingest_spec(bag_spec(tmp_path, "bags/*", **over)), str(tmp_path / "out"))
    assert summary["sessions"] == {"03-02-2024": 1} and summary["sampling_rate_hz"] == pytest.approx(10.0)
    got = _check_matches(tmp_path / "out" / "bagrov-03-02-2024-run1.csv", d)
    assert len(got) >= 598
    rep = summary["files"][0]["bag"]
    assert rep["topics"]["/odom"]["timestamps"] == "header.stamp"
    assert summary["torque_reference_offset_m"] == [0.0, 0.0, 0.085]


@needs_data
def test_sqlite3_bag_with_custom_messages_and_no_definitions_says_what_to_do(tmp_path):
    d = reference_run_in_frd(pd.read_csv(REAL[0])).iloc[:100]
    (tmp_path / "bags").mkdir()
    write_bag(tmp_path / "bags" / "run", d, "sqlite3")
    strip_definitions(tmp_path / "bags" / "run")
    with pytest.raises(IngestError, match="message_types"):
        ingest(load_ingest_spec(bag_spec(tmp_path, "bags/*")), str(tmp_path / "out"))


@needs_data
def test_topics_at_different_rates_and_times(tmp_path):
    """Odometry at 50 Hz starting 13 ms late, PWM at 5 Hz: velocities are interpolated, PWM held."""
    d = reference_run_in_frd(pd.read_csv(REAL[1])).iloc[:600].reset_index(drop=True)
    d_fine = d.copy()
    (tmp_path / "bags").mkdir()
    write_bag(tmp_path / "bags" / "run.bag", d_fine, "ros1", odom_dt=0.02, odom_offset=0.013, rc_dt=0.2)
    summary = ingest(load_ingest_spec(bag_spec(tmp_path, "bags/*.bag")), str(tmp_path / "out"))
    topics = summary["files"][0]["bag"]["topics"]
    assert topics["/odom"]["rate_hz"] == pytest.approx(50, rel=0.01)
    assert topics["/mavros/rc/out"]["rate_hz"] == pytest.approx(5, rel=0.01)
    got = pd.read_csv(tmp_path / "out" / "bagrov-03-02-2024-run1.csv")
    assert np.all(np.isfinite(got.values)) and len(got) > 590
    # PWM held at 5 Hz: forces change at most every 2 grid samples
    changes = np.flatnonzero(np.any(np.diff(got[["Fx", "Fy", "Fz"]].values, axis=0) != 0, axis=1))
    assert np.all(np.diff(changes) >= 2)
    # velocities: grid points fall 13 ms from the (step-constant) source samples -> close to the original
    err = np.abs(got["Bu"].values[5:500] - d["Bu"].values[5:500])
    assert np.median(err) < 0.01


@needs_data
def test_receive_time_option_and_bag_errors(tmp_path):
    d = reference_run_in_frd(pd.read_csv(REAL[0])).iloc[:120]
    (tmp_path / "bags").mkdir()
    write_bag(tmp_path / "bags" / "run.bag", d, "ros1")
    s = ingest(load_ingest_spec(bag_spec(tmp_path, "bags/*.bag", source={"stamp": "receive"})), str(tmp_path / "o1"))
    assert s["files"][0]["bag"]["topics"]["/odom"]["timestamps"] == "bag receive time"
    bad_topic = {"columns": {"u": "/nope.twist.twist.linear.x", "v": "/odom.twist.twist.linear.y",
                             "w": "/odom.twist.twist.linear.z", "p": "/odom.twist.twist.angular.x",
                             "q": "/odom.twist.twist.angular.y", "r": "/odom.twist.twist.angular.z"}}
    with pytest.raises(IngestError, match=r"topic\(s\) \['/nope'\] not in the bag"):
        ingest(load_ingest_spec(bag_spec(tmp_path, "bags/*.bag", body_velocity=bad_topic)), str(tmp_path / "o2"))
    bad_field = dict(bad_topic, columns={**bad_topic["columns"], "u": "/odom.twist.twist.linear.speed"})
    with pytest.raises(IngestError, match="no field 'speed'"):
        ingest(load_ingest_spec(bag_spec(tmp_path, "bags/*.bag", body_velocity=bad_field)), str(tmp_path / "o3"))
    not_number = dict(bad_topic, columns={**bad_topic["columns"], "u": "/odom.twist.twist.linear"})
    with pytest.raises(IngestError, match="not a number"):
        ingest(load_ingest_spec(bag_spec(tmp_path, "bags/*.bag", body_velocity=not_number)), str(tmp_path / "o4"))
    too_far = {"pwm_columns": [f"/mavros/rc/out.channels[{i}]" for i in range(1, 9)], "preset": "bluerov2_heavy"}
    with pytest.raises(IngestError, match="index 8 but the array has 8"):
        ingest(load_ingest_spec(bag_spec(tmp_path, "bags/*.bag", wrench={"thrusters": too_far})), str(tmp_path / "o5"))


@pytest.mark.parametrize("over,match", [
    ({"time": {"column": "t"}}, "remove the time: block"),
    ({"resample_hz": None}, "needs resample_hz"),
])
def test_bag_spec_rules(tmp_path, over, match):
    with pytest.raises(IngestError, match=match):
        load_ingest_spec(bag_spec(tmp_path, "bags/*.bag", **over))


# ------------------------------------------------------------------ pre-flight on converted logs

@needs_data
def test_preflight_passes_on_converted_logs_and_catches_a_wrong_rate(tmp_path):
    """Converted logs have no position columns, which the old sampling-rate inference needed: the
    pre-flight check crashed on every lab's converted logs. It now uses the rate the timestamps gave."""
    import subprocess
    import sys
    (tmp_path / "bags").mkdir()
    global T0
    for i, src in enumerate((REAL[0], REAL[-1])):
        old, T0 = T0, 1_707_000_000.0 + 86400.0 * 3 * i  # two different days
        try:
            write_bag(tmp_path / "bags" / f"run{i}.bag", reference_run_in_frd(pd.read_csv(src)).iloc[:400], "ros1")
        finally:
            T0 = old
    out = tmp_path / "out"
    ingest(load_ingest_spec(bag_spec(tmp_path, "bags/*.bag")), str(out))
    cal = os.path.dirname(HERE)

    def preflight(rate):
        return subprocess.run([sys.executable, "preflight_check.py", "--data-dir", str(out), "--spec",
                               "vehicle_spec.yaml", "--sampling-rate-hz", str(rate), "--heldout-dates", "06-02-2024"],
                              cwd=cal, capture_output=True, text=True, encoding="utf-8", errors="replace")
    ok = preflight(10)
    assert ok.returncode == 0, ok.stdout[-3000:] + ok.stderr[-3000:]
    assert "timestamps give 10 Hz" in ok.stdout
    wrong = preflight(20)
    assert wrong.returncode == 1 and "These disagree" in wrong.stdout


def test_preflight_rate_source_without_positions_or_timestamps(tmp_path):
    from preflight_check import sampling_rate_source
    pd.DataFrame({c: [0.0] * 5 for c in ("Fx", "Bu", "qw")}).to_csv(tmp_path / "a-12-01-2024-run1.csv", index=False)
    how, msg, ok = sampling_rate_source(str(tmp_path), 10.0)
    assert how == "no_positions" and ok and "assume 10 Hz" in msg
    pd.DataFrame({c: [0.0] * 5 for c in ("x", "y", "z", "Bu")}).to_csv(tmp_path / "a-12-01-2024-run1.csv", index=False)
    assert sampling_rate_source(str(tmp_path), 10.0)[0] == "infer"


# ------------------------------------------------------------------ randomized robustness

def test_random_thruster_layouts_never_exceed_a_thruster_and_report_what_they_cant_push():
    rng = np.random.default_rng(7)
    for _ in range(300):
        n = int(rng.integers(2, 11))
        layout = [{"position_m": list(rng.uniform(-0.4, 0.4, 3)), "direction": list(rng.normal(size=3)),
                   "reversed": bool(rng.random() < 0.5)} for _ in range(n)]
        cfg = tm.thruster_block({"layout": layout, "curve": "t200", "voltage": float(rng.uniform(12, 20))})
        rank = np.linalg.matrix_rank(cfg.allocation)
        assert (len(cfg.unreachable_axes()) == 0) == (rank == 6)
        taus = rng.normal(size=(50, 6)) * rng.choice([1.0, 30.0, 500.0])
        out = cfg.achievable_wrench(taus)
        t = out @ np.linalg.pinv(cfg.allocation).T
        assert np.all(np.isfinite(out))
        assert t.max() <= cfg.curve.max_forward_n + 1e-6 and -t.min() <= cfg.curve.max_reverse_n + 1e-6
        # what comes out is always something the thrusters can actually make
        P = cfg.allocation @ np.linalg.pinv(cfg.allocation)
        np.testing.assert_allclose(out @ P.T, out, atol=1e-6 * max(1.0, np.abs(out).max()))
        pwm = rng.uniform(1100, 1900, (20, n))
        assert np.all(np.isfinite(cfg.wrench(pwm)))


@needs_data
def test_bags_with_messy_timing(tmp_path):
    """Messages written out of order, duplicated stamps, jittered stamps and a gap: ingestion sorts,
    drops duplicates, interpolates, reports the gap, and the result stays close to the original."""
    d = reference_run_in_frd(pd.read_csv(REAL[2])).iloc[:500].reset_index(drop=True)
    ts = _typestore(True)
    lin, ang, q = _ros_frames(d)
    pwm = np.ascontiguousarray(d[[f"pwm{i}" for i in range(1, 9)]].values.astype(np.uint16))
    rng = np.random.default_rng(3)
    msgs = []
    for k in range(len(d)):
        if 200 <= k < 215:  # a 1.5 s gap in odometry
            continue
        t = T0 + 0.1 * k + rng.uniform(-0.004, 0.004)
        msgs.append((t, "/odom", "nav_msgs/msg/Odometry", _odom(ts, t, lin[k], ang[k], q[k] * (-1 if k % 5 == 0 else 1))))
        if k % 50 == 0:
            msgs.append((t, "/odom", "nav_msgs/msg/Odometry", _odom(ts, t, lin[k], ang[k], q[k])))  # duplicate stamp
        msgs.append((T0 + 0.1 * k, "/mavros/rc/out", "mavros_msgs/msg/RCOut",
                     ts.types["mavros_msgs/msg/RCOut"](header=_header(ts, T0 + 0.1 * k), channels=np.ascontiguousarray(pwm[k]))))
    order = rng.permutation(len(msgs))  # recorded out of order
    (tmp_path / "bags").mkdir()
    with Writer1(tmp_path / "bags" / "messy.bag") as w:
        conns = {tp: w.add_connection(tp, mt, typestore=ts) for tp, mt in
                 (("/odom", "nav_msgs/msg/Odometry"), ("/mavros/rc/out", "mavros_msgs/msg/RCOut"))}
        for i in order:
            t, tp, mt, m = msgs[i]
            w.write(conns[tp], int(T0 * 1e9) + int(i), ts.serialize_ros1(m, mt))  # receive times meaningless
    s = ingest(load_ingest_spec(bag_spec(tmp_path, "bags/*.bag")), str(tmp_path / "out"))
    rep = s["files"][0]
    assert rep["bag"]["topics"]["/odom"]["gaps"] == 1
    assert any("repeated timestamp" in w for w in rep["warnings"]) and any("gap" in w for w in rep["warnings"])
    got = pd.read_csv(tmp_path / "out" / s["files"][0]["output"])
    assert np.all(np.isfinite(got.values)) and np.allclose(np.linalg.norm(got[["qx", "qy", "qz", "qw"]].values, axis=1), 1)
    ok = np.r_[np.arange(5, 195), np.arange(220, 495)]
    assert np.median(np.abs(got["Bu"].values[ok] - d["Bu"].values[ok])) < 0.01
    Rg = Rotation.from_quat(got[["qx", "qy", "qz", "qw"]].values[ok])
    Ro = Rotation.from_quat(d[["qx", "qy", "qz", "qw"]].values[ok])
    assert np.median(np.degrees((Rg.inv() * Ro).magnitude())) < 1.0


@needs_data
@pytest.mark.parametrize("kind", ["ros1", "mcap"])
def test_a_re_recorded_bag_lines_up_on_the_mission_clock_and_date(tmp_path, kind):
    """F1, F2 (URI re-run): a mission replayed and re-recorded 600 days later. /odom has header stamps from the
    mission; the thruster topic has none, so it carried the re-recording's time: no overlap ("the topics overlap
    for only 0.00 s"), and the session date was the re-recording's."""
    d = reference_run_in_frd(pd.read_csv(REAL[1])).iloc[:300].reset_index(drop=True)
    (tmp_path / "bags").mkdir()
    name = "run.bag" if kind == "ros1" else "run"
    write_bag(tmp_path / "bags" / name, d, kind, recv_delay=600 * 86400.0, rc_header=False)
    summary = ingest(load_ingest_spec(bag_spec(tmp_path, "bags/*.bag" if kind == "ros1" else "bags/run")),
                     str(tmp_path / "out"))
    rep = summary["files"][0]
    assert rep["session_date"] == "03-02-2024"                           # the mission's date, from its header clock
    assert "moved" in rep["bag"]["topics"]["/mavros/rc/out"]["timestamps"]
    assert any("re-recorded bag" in w for w in rep["warnings"])
    _check_matches(tmp_path / "out" / "bagrov-03-02-2024-run1.csv", d)  # forces line up with the motion again
    # the same bag read with receive times: the re-recording's clock, and a warning that says so
    s2 = ingest(load_ingest_spec(bag_spec(tmp_path, "bags/*.bag" if kind == "ros1" else "bags/run",
                                          source={"stamp": "receive"})), str(tmp_path / "out2"))
    assert s2["files"][0]["session_date"] != "03-02-2024"
    assert any("stamp: header (the default) uses the mission's own" in w for w in s2["files"][0]["warnings"])
