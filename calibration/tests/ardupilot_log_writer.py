"""
Writes ArduPilot DataFlash (.bin) and MAVLink telemetry (.tlog) logs from a
canonical run, the way ArduSub records them, for the ingestion tests and the
stress test. Not part of the product.

.bin: FMT-described binary records (0xA3 0x95 <id> <payload>), with
  RCOU (servo PWM), ATT (roll/pitch in centidegrees, yaw 0-36000),
  IMU[0] and IMU[1] (body rates, rad/s), XKF1[0] (NED velocity), and
  optionally GPS week/ms so the log carries real clock time.
.tlog: 8-byte big-endian microsecond timestamp + raw MAVLink packet, with
  SERVO_OUTPUT_RAW, ATTITUDE (rad) and LOCAL_POSITION_NED (NED velocity).
"""
import struct

import numpy as np
from scipy.spatial.transform import Rotation

HEAD = b"\xa3\x95"
DF_STRUCT = {"Q": "Q", "q": "q", "f": "f", "d": "d", "B": "B", "b": "b", "H": "H", "h": "h", "I": "I", "i": "i",
             "c": "h", "C": "H", "e": "i", "E": "I", "L": "i", "M": "B", "n": "4s", "N": "16s", "Z": "64s"}
GPS_EPOCH_UNIX = 315964800  # 1980-01-06
LEAP_S = 18


def _fmt_record(type_id, name, fmt, cols):
    length = 3 + struct.calcsize("<" + "".join(DF_STRUCT[c] for c in fmt))
    return HEAD + bytes([128]) + struct.pack("<BB4s16s64s", type_id, length, name.encode(), fmt.encode(),
                                             cols.encode())


def _rec(type_id, fmt, *vals):
    return HEAD + bytes([type_id]) + struct.pack("<" + "".join(DF_STRUCT[c] for c in fmt), *vals)


def euler_deg(d):
    """Canonical FRD/NED quaternions -> ArduPilot roll, pitch, yaw in degrees."""
    return np.degrees(Rotation.from_quat(d[["qx", "qy", "qz", "qw"]].values).as_euler("xyz"))


def ned_velocity(d):
    R = Rotation.from_quat(d[["qx", "qy", "qz", "qw"]].values)
    return R.apply(d[["Bu", "Bv", "Bw"]].values)


def write_bin(path, d, t0_unix=None, dt=0.1, offsets=None, corrupt_tail=False, garbage_every=0, drop_type=None):
    """d: canonical run with pwm1..pwm8. t0_unix: give the log GPS time (else boot time only).
    offsets: per-type time offsets in seconds. garbage_every: insert junk bytes every N records."""
    offsets = offsets or {}
    eul = euler_deg(d)
    vel = ned_velocity(d)
    gyro = d[["Bp", "Bq", "Br"]].values
    pwm = d[[f"pwm{i}" for i in range(1, 9)]].values
    out = bytearray()
    out += _fmt_record(128, "FMT", "BBnNZ", "Type,Length,Name,Format,Columns")
    out += _fmt_record(60, "RCOU", "QHHHHHHHH", "TimeUS,C1,C2,C3,C4,C5,C6,C7,C8")
    out += _fmt_record(61, "ATT", "QccccCC", "TimeUS,DesRoll,Roll,DesPitch,Pitch,DesYaw,Yaw")
    out += _fmt_record(62, "IMU", "QBfff", "TimeUS,I,GyrX,GyrY,GyrZ")
    out += _fmt_record(63, "XKF1", "QBfff", "TimeUS,C,VN,VE,VD")
    out += _fmt_record(64, "GPS", "QBIH", "TimeUS,Status,GMS,GWk")
    boot_us = 5_000_000
    n = 0
    for k in range(len(d)):
        base = boot_us + int(round(k * dt * 1e6))
        us = {m: base + int(round(offsets.get(m, 0.0) * 1e6)) for m in ("RCOU", "ATT", "IMU", "XKF1", "GPS")}
        recs = []
        if t0_unix is not None and k % 2 == 0:
            gps_s = t0_unix - GPS_EPOCH_UNIX + LEAP_S + (us["GPS"] - boot_us) * 1e-6
            wk, ms = int(gps_s // 604800), int(round((gps_s % 604800) * 1000))
            recs.append(_rec(64, "QBIH", us["GPS"], 3, ms, wk))
        if drop_type != "RCOU":
            recs.append(_rec(60, "QHHHHHHHH", us["RCOU"], *[int(x) for x in pwm[k]]))
        yaw = eul[k, 2] % 360.0
        recs.append(_rec(61, "QccccCC", us["ATT"], 0, int(round(eul[k, 0] * 100)), 0, int(round(eul[k, 1] * 100)),
                         0, int(round(yaw * 100)) % 36000))
        recs.append(_rec(62, "QBfff", us["IMU"], 0, *gyro[k]))
        recs.append(_rec(62, "QBfff", us["IMU"] + 700, 1, *(gyro[k] + 0.3)))  # a second, biased IMU
        recs.append(_rec(63, "QBfff", us["XKF1"], 0, *vel[k]))
        for r in recs:
            out += r
            n += 1
            if garbage_every and n % garbage_every == 0:
                out += b"\x00\xa3\x07\xff\x13garbage"
    if corrupt_tail:
        out += HEAD + bytes([60]) + b"\x01\x02\x03"  # a record cut off at power-down
    with open(path, "wb") as f:
        f.write(bytes(out))


def write_tlog(path, d, t0_unix=1_707_000_000.0, dt=0.1, offsets=None, corrupt_every=0):
    from pymavlink.dialects.v20 import ardupilotmega as mav
    offsets = offsets or {}
    m = mav.MAVLink(None, srcSystem=1, srcComponent=1)
    eul = np.radians(euler_deg(d))
    vel = ned_velocity(d)
    gyro = d[["Bp", "Bq", "Br"]].values
    pwm = d[[f"pwm{i}" for i in range(1, 9)]].values
    out = bytearray()
    n = 0
    for k in range(len(d)):
        t = t0_unix + k * dt
        msgs = [(t + offsets.get("SERVO_OUTPUT_RAW", 0.0),
                 m.servo_output_raw_encode(int(k * dt * 1e6), 0, *[int(x) for x in pwm[k]])),
                (t + offsets.get("ATTITUDE", 0.0),
                 m.attitude_encode(int(k * dt * 1e3), *[float(x) for x in eul[k]], *[float(x) for x in gyro[k]])),
                (t + offsets.get("LOCAL_POSITION_NED", 0.0),
                 m.local_position_ned_encode(int(k * dt * 1e3), 0.0, 0.0, 0.0, *[float(x) for x in vel[k]]))]
        for tt, msg in msgs:
            out += struct.pack(">Q", int(round(tt * 1e6))) + msg.pack(m)
            n += 1
            if corrupt_every and n % corrupt_every == 0:
                out += struct.pack(">Q", int(round(tt * 1e6))) + b"\xfd\x09\x00\x00\x00junkjunk"
    with open(path, "wb") as f:
        f.write(bytes(out))
