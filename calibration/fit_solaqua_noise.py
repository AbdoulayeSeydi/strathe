#!/usr/bin/env python3
"""
fit_solaqua_noise.py -- measure an open-water BlueROV2's sensor noise from the
SOLAQUA dataset and write the `open_water` sensor-noise preset.
PROJECT_STATUS.md #60.

    python fit_solaqua_noise.py --bags ../data/raw/solaqua \
        [--out sensor_noise/open_water_bluerov2_solaqua.yaml] [--report results/solaqua_noise]

The data: SINTEF Ocean's SOLAQUA (https://data.sintef.no/feature/fe-a8f86232-5107-495e-a3dd-a86460eebef6,
arXiv 2504.01790, CC BY-SA 4.0). A BlueROV2 at a full-scale fish farm in Norway, August 2024: a Water
Linked A50 DVL, ArduSub's attitude and gyro estimates, a depth sensor and a Sonardyne USBL. Only the
`*_data.bag` files are needed (about 173 MB for all 61; the video bags are not). Download them from the
page above; ROS 1 bags carry their own message definitions, so nothing else is needed to read them.

What is measured, and how:
  - White noise of DVL velocity, gyro rates, roll/pitch, heading and depth: the robust second-difference
    estimator of fit_sensor_noise.py, applied to each sensor at its OWN logged rate, straight from the
    bags. Not resampled: interpolating onto a common grid smooths the noise and biases it low.
  - DVL dropouts: invalid readings plus time spent in gaps longer than 3 nominal periods.
  - The USBL fix error (used as the starting-fix error for dead reckoning): in sliding windows of 10
    fixes (about 20 s), the USBL track is fitted to the DVL's own dead-reckoned track by a rotation,
    scale, offset and constant drift velocity. Over 20 s the DVL track is accurate to a few cm, and the
    drift term absorbs the net and water moving, so the residual scatter is the USBL's own error.

What can NOT be measured from this data, and why:
  - Biases (DVL velocity offset, gyro, attitude and heading offsets) and dead-reckoning drift. The DVL
    locks onto the fish-farm net at 0.5-2 m range, not the seabed, and the net moves with the current
    (0.04-0.2 m/s on the day), while the USBL measures against the earth. Their disagreement is mostly
    the net's motion, not a sensor bias. The preset keeps the `typical` preset's assumed biases and
    labels them assumed.
  - Thruster forces: /commanded_thrust holds ArduSub pilot-input channels, not per-thruster commands,
    so this data can't calibrate the physics.
Caveats that travel with the numbers: a DVL at 0.5-2 m range is quieter than one bottom-locked at
10-50 m; gyro, attitude and heading are ArduSub's filtered estimates (what a policy reading ArduSub
sees), not raw IMU samples.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np
import yaml

from fit_sensor_noise import white_noise_std

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "sensor_noise", "open_water_bluerov2_solaqua.yaml")
DEFAULT_REPORT = os.path.join(HERE, "results", "solaqua_noise")
SOURCE_URL = "https://data.sintef.no/feature/fe-a8f86232-5107-495e-a3dd-a86460eebef6"
TOPICS = ("/sensor/dvl_velocity", "/sensor/dvl_position", "/sensor/attitude", "/sensor/depth_temperature",
          "/sensor/usbl")
# the first three recordings calibrate the stereo camera on deck; everything from 13:55 on is in the water
FIRST_IN_WATER = "2024-08-20_13-55"
USBL_WINDOW = 10          # fixes per window (~20 s at 0.5 Hz)
USBL_STEP = 5
DR_JUMP_M = 1.0           # a dead-reckoning jump this large between fixes is a reset: skip that window
# Biases this data can't measure (see the docstring): taken from the typical preset, labelled assumed.
ASSUMED_FIELDS = ("velocity_bias_mps", "gyro_bias_rps", "attitude_bias_rad", "heading_bias_rad")


def read_bag(path: str) -> dict:
    """The five sensor streams of one SOLAQUA data bag, as arrays with the header time in column 0."""
    from rosbags.highlevel import AnyReader
    s = {t: [] for t in TOPICS}
    with AnyReader([Path(path)]) as r:
        conns = [c for c in r.connections if c.topic in TOPICS]
        for c, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            ts = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if c.topic == "/sensor/dvl_velocity":
                s[c.topic].append((ts, m.velocity.x, m.velocity.y, m.velocity.z, float(m.velocity_valid),
                                   m.altitude))
            elif c.topic == "/sensor/dvl_position":
                s[c.topic].append((ts, m.x, m.y))
            elif c.topic == "/sensor/attitude":
                s[c.topic].append((ts, m.roll, m.pitch, m.yaw, m.rollspeed, m.pitchspeed, m.yawspeed))
            elif c.topic == "/sensor/depth_temperature":
                s[c.topic].append((ts, m.depth))
            elif c.topic == "/sensor/usbl":
                s[c.topic].append((ts, m.east, m.north, m.sigma))
    return {k: np.array(v, dtype=float).reshape(-1, {"/sensor/dvl_velocity": 6, "/sensor/dvl_position": 3,
                                                     "/sensor/attitude": 7, "/sensor/depth_temperature": 2,
                                                     "/sensor/usbl": 4}[k]) for k, v in s.items()}


def dvl_dropout(dv: np.ndarray) -> tuple:
    """(dropout fraction, nominal rate Hz): invalid readings plus time in gaps longer than 3 nominal periods."""
    if len(dv) < 20:
        return float("nan"), float("nan")
    dt = np.diff(dv[:, 0])
    period = float(np.median(dt))
    dur = dv[-1, 0] - dv[0, 0]
    gap_time = float(np.sum(dt[dt > 3 * period] - period))
    invalid = float(np.mean(dv[:, 4] < 0.5))
    return invalid + gap_time / dur, 1.0 / period


def usbl_fix_error(us: np.ndarray, dp: np.ndarray) -> tuple:
    """(per-axis USBL error std in m, windows used): residual scatter after fitting short windows of the
    USBL track to the DVL's dead-reckoned track (rotation + scale + offset + drift velocity)."""
    if len(us) < USBL_WINDOW or len(dp) < 20:
        return float("nan"), 0
    inside = (us[:, 0] >= dp[0, 0]) & (us[:, 0] <= dp[-1, 0])
    u = us[inside]
    if len(u) < USBL_WINDOW:
        return float("nan"), 0
    dx = np.interp(u[:, 0], dp[:, 0], dp[:, 1])
    dy = np.interp(u[:, 0], dp[:, 0], dp[:, 2])
    variances = []
    for a in range(0, len(u) - USBL_WINDOW + 1, USBL_STEP):
        sl = slice(a, a + USBL_WINDOW)
        x, y, t = dx[sl], dy[sl], u[sl, 0] - u[a, 0]
        if np.max(np.hypot(np.diff(x), np.diff(y))) > DR_JUMP_M:
            continue                                   # a DVL dead-reckoning reset inside the window
        best = None
        for flip in (1.0, -1.0):                       # the DVL frame's handedness against east/north
            yy = flip * y
            one, zero = np.ones_like(t), np.zeros_like(t)
            # E = a x - b y + c + ve t ;  N = b x + a y + d + vn t   (similarity transform + drift)
            A = np.vstack([np.column_stack([x, -yy, one, zero, t, zero]),
                           np.column_stack([yy, x, zero, one, zero, t])])
            obs = np.concatenate([u[sl, 1], u[sl, 2]])
            coef, *_ = np.linalg.lstsq(A, obs, rcond=None)
            res = obs - A @ coef
            dof = len(obs) - A.shape[1]
            var = float(res @ res / dof)               # per-axis variance (E and N pooled)
            if best is None or var < best:
                best = var
        variances.append(best)
    if not variances:
        return float("nan"), 0
    return float(np.sqrt(np.median(variances))), len(variances)


def fit_run(path: str) -> dict:
    d = read_bag(path)
    r = {"run": Path(path).name[:19]}
    dv = d["/sensor/dvl_velocity"]
    if len(dv) >= 20:
        ok = dv[:, 4] > 0.5
        r["dvl_rate_hz"] = round(1.0 / float(np.median(np.diff(dv[:, 0]))), 2)
        r["velocity_std"] = [white_noise_std(dv[ok, i]) for i in (1, 2, 3)]
        r["velocity_dropout"], _ = dvl_dropout(dv)
        r["dvl_range_m"] = float(np.median(dv[ok, 5])) if ok.any() else float("nan")
    at = d["/sensor/attitude"]
    if len(at) >= 20:
        r["attitude_rate_hz"] = round(1.0 / float(np.median(np.diff(at[:, 0]))), 2)
        r["roll_pitch_std"] = [white_noise_std(at[:, 1]), white_noise_std(at[:, 2])]
        r["heading_std"] = white_noise_std(np.unwrap(at[:, 3]))
        r["gyro_std"] = [white_noise_std(at[:, i]) for i in (4, 5, 6)]
    dp = d["/sensor/depth_temperature"]
    if len(dp) >= 20:
        r["depth_rate_hz"] = round(1.0 / float(np.median(np.diff(dp[:, 0]))), 2)
        r["depth_std"] = white_noise_std(dp[:, 1])
        r["depth_mean_m"] = float(np.mean(dp[:, 1]))
    us = d["/sensor/usbl"]
    if len(us) >= 2:
        r["usbl_rate_hz"] = round(1.0 / float(np.median(np.diff(us[:, 0]))), 2)
        r["usbl_reported_sigma_m"] = float(np.median(us[:, 3]))
    r["usbl_fix_error"], r["usbl_windows"] = usbl_fix_error(us, d["/sensor/dvl_position"])
    dur = [a[-1, 0] - a[0, 0] for a in d.values() if len(a) > 1]
    r["duration_s"] = round(max(dur), 1) if dur else 0.0
    return r


def _med(rows, key, idx=None):
    vals = []
    for r in rows:
        v = r.get(key)
        if v is None:
            continue
        v = v[idx] if idx is not None else v
        if np.isfinite(v):
            vals.append(v)
    return float(np.median(vals)) if vals else float("nan"), len(vals)


def summarise(rows: list) -> dict:
    """Median over runs of each per-run value, combined the way fit_sensor_noise.py combines axes."""
    m = lambda k, i=None: _med(rows, k, i)[0]
    usbl, n_usbl = _med(rows, "usbl_fix_error")
    from sensor_noise import load_sensor_noise
    typical = load_sensor_noise("typical")
    sensors = {
        "velocity_std_mps": float(np.mean([m("velocity_std", i) for i in range(3)])),
        "velocity_bias_mps": typical.velocity_bias_mps,
        "velocity_dropout_prob": m("velocity_dropout"),
        "gyro_std_rps": float(np.mean([m("gyro_std", i) for i in range(3)])),
        "gyro_bias_rps": typical.gyro_bias_rps,
        "attitude_std_rad": float(np.mean([m("roll_pitch_std", i) for i in range(2)])),
        "attitude_bias_rad": typical.attitude_bias_rad,
        "heading_std_rad": m("heading_std"),
        "heading_bias_rad": typical.heading_bias_rad,
        "depth_std_m": m("depth_std"),
        "position": "dead_reckoning",
        "position_std_m": usbl,
    }
    for k, v in sensors.items():
        if isinstance(v, float):
            sensors[k] = round(v, 6)
    detail = {
        "velocity_std_by_axis_mps": [round(m("velocity_std", i), 6) for i in range(3)],
        "gyro_std_by_axis_rps": [round(m("gyro_std", i), 6) for i in range(3)],
        "roll_pitch_std_rad": [round(m("roll_pitch_std", i), 6) for i in range(2)],
        "usbl_fix_error_runs": n_usbl,
        "usbl_reported_sigma_m": round(m("usbl_reported_sigma_m"), 3),
        "dvl_range_m_median": round(m("dvl_range_m"), 2),
        "native_rates_hz": {"velocity": round(m("dvl_rate_hz"), 1), "attitude_and_gyro": round(m("attitude_rate_hz"), 1),
                            "depth": round(m("depth_rate_hz"), 1), "usbl": round(m("usbl_rate_hz"), 2)},
    }
    return {"sensors": sensors, "detail": detail}


def _plain(v):
    """numpy scalars and lists of them -> plain Python numbers (rounded), for YAML and JSON."""
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 6) if np.isfinite(v) else None
    return v


def write_preset(path: str, summary: dict, rows: list) -> None:
    s, d = summary["sensors"], summary["detail"]
    doc = {
        "sensors": s,
        "provenance": "measured",
        "assumed_fields": list(ASSUMED_FIELDS),
        "fit": {
            "date": date.today().isoformat(),
            "method": "fit_solaqua_noise.py: robust second difference per sensor at its own logged rate; "
                      "USBL fix error from short windows fitted to DVL dead reckoning",
            "logs": "SOLAQUA (SINTEF Ocean), BlueROV2 at a fish farm, 2024-08-20/22",
            "recordings": len(rows),
            **d,
            "notes": [
                f"White noise, dropouts and the starting-fix error are MEASURED from {len(rows)} in-water "
                f"recordings. The four biases are NOT measurable from this data (the DVL tracks the moving "
                f"net, not the seabed) and are the typical preset's ASSUMED values.",
                f"The DVL tracked the net at about {d['dvl_range_m_median']:g} m: a DVL bottom-locked at "
                f"10-50 m is noisier than this.",
                "Gyro, roll/pitch and heading are ArduSub's filtered estimates, not raw IMU samples.",
                "position_std_m is the USBL fix error (per axis), used as the starting-fix error for dead "
                "reckoning; the USBL reported sigma is listed for comparison.",
            ],
        },
        "source_data": {"name": "SOLAQUA - SINTEF Ocean Large AQUAculture robotics data set",
                        "url": SOURCE_URL, "paper": "arXiv:2504.01790", "license": "CC BY-SA 4.0",
                        "attribution": "SINTEF Ocean"},
        "per_recording": [{k: _plain(v) for k, v in r.items()} for r in rows],
    }
    header = (
        "# Open-water sensor noise for a BlueROV2 (Water Linked A50 DVL, ArduSub attitude/gyro, depth sensor,\n"
        "# Sonardyne USBL), MEASURED from the SOLAQUA dataset by fit_solaqua_noise.py (PROJECT_STATUS.md #60).\n"
        "# White noise, DVL dropouts and the starting-fix error are measured; the four biases are the typical\n"
        "# preset's ASSUMED values (this data can't measure them): see assumed_fields and fit.notes.\n"
        "# Derived from SOLAQUA by SINTEF Ocean (" + SOURCE_URL + ", CC BY-SA 4.0);\n"
        "# this file is shared under the same licence, CC BY-SA 4.0, with attribution to SINTEF Ocean.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.safe_dump(doc, f, sort_keys=False)


def write_report(out_dir: str, summary: dict, rows: list, preset_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sensor_noise import load_sensor_noise
    os.makedirs(out_dir, exist_ok=True)
    s = summary["sensors"]
    typ = load_sensor_noise("typical")
    tank = load_sensor_noise("tank")
    names = [("velocity_std_mps", "DVL velocity", "cm/s", 100), ("velocity_dropout_prob", "DVL dropouts", "%", 100),
             ("gyro_std_rps", "Gyro", "deg/s", np.degrees(1)), ("attitude_std_rad", "Roll/pitch", "deg", np.degrees(1)),
             ("heading_std_rad", "Heading", "deg", np.degrees(1)), ("depth_std_m", "Depth", "cm", 100),
             ("position_std_m", "Position fix (USBL; tank: motion capture)", "m", 1)]
    L = ["# Open-water sensor noise from SOLAQUA (PROJECT_STATUS.md #60)\n",
         f"{date.today().isoformat()} · {len(rows)} in-water recordings · `python fit_solaqua_noise.py` · "
         f"preset: `{os.path.relpath(preset_path, HERE)}` (`--noise open_water`)\n",
         "Data: SOLAQUA by SINTEF Ocean, a BlueROV2 at a full-scale fish farm in Norway "
         f"([{SOURCE_URL}]({SOURCE_URL}), CC BY-SA 4.0). Derived statistics shared under the same licence.\n",
         "## Measured against what Strathe assumed\n",
         "| Sensor | SOLAQUA (measured) | `typical` (assumed) | `tank` (measured, reference tank) |",
         "|---|---|---|---|"]
    for k, label, unit, f in names:
        L.append(f"| {label} | {s[k] * f:.3g} {unit} | {getattr(typ, k) * f:.3g} {unit} | {getattr(tank, k) * f:.3g} {unit} |")
    d = summary["detail"]
    L += ["",
          f"- Per axis: DVL {', '.join(f'{100 * v:.2g}' for v in d['velocity_std_by_axis_mps'])} cm/s (u, v, w); "
          f"gyro {', '.join(f'{np.degrees(v):.2g}' for v in d['gyro_std_by_axis_rps'])} deg/s; roll, pitch "
          f"{', '.join(f'{np.degrees(v):.2g}' for v in d['roll_pitch_std_rad'])} deg.",
          f"- Native rates: DVL {d['native_rates_hz']['velocity']} Hz, attitude/gyro "
          f"{d['native_rates_hz']['attitude_and_gyro']} Hz, depth {d['native_rates_hz']['depth']} Hz, USBL "
          f"{d['native_rates_hz']['usbl']} Hz. Each was measured at its own rate (not resampled).",
          f"- USBL fix error {s['position_std_m']:.2g} m per axis, from {d['usbl_fix_error_runs']} recordings "
          f"(short windows fitted to DVL dead reckoning); the USBL's own reported sigma is "
          f"{d['usbl_reported_sigma_m']:g} m.\n",
          "## What this data can't tell, and what the preset does instead\n",
          "- **Biases and dead-reckoning drift.** The DVL tracked the fish-farm net at about "
          f"{d['dvl_range_m_median']:g} m, and the net moves with the current (0.04-0.2 m/s on the day), while "
          "the USBL measures against the earth. Their disagreement is mostly the net moving, not a sensor bias, "
          "so the preset keeps `typical`'s ASSUMED biases (`assumed_fields` in the file says which).",
          "- **Short-range DVL.** A DVL at 0.5-2 m is quieter than one bottom-locked at 10-50 m.",
          "- **Filtered attitude.** Gyro, roll/pitch and heading are ArduSub's estimates, which is what a "
          "policy reading ArduSub sees, not raw IMU noise.",
          "- **No thruster forces.** `/commanded_thrust` holds ArduSub pilot-input channels, so SOLAQUA can't "
          "calibrate the physics.\n",
          "![Measured vs assumed](noise_measured_vs_assumed.png)\n"]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    with open(os.path.join(out_dir, "per_recording.json"), "w") as f:
        json.dump([{k: _plain(v) for k, v in r.items()} for r in rows], f, indent=1)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    labels = [n[1] for n in names]
    xs = np.arange(len(names))
    for j, (cfg, lab) in enumerate(((None, "SOLAQUA, measured"), (typ, "typical, assumed"), (tank, "tank, measured"))):
        vals = [(s[k] if cfg is None else getattr(cfg, k)) * f for k, _, _, f in names]
        ax.bar(xs + (j - 1) * 0.27, vals, 0.27, label=lab)
    ax.set_yscale("log")
    ax.set_xticks(xs, [f"{l}\n({u})" for _, l, u, _ in names], fontsize=8)
    ax.set_ylabel("one standard deviation (log scale)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "noise_measured_vs_assumed.png"), dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bags", required=True, help="folder of SOLAQUA *_data.bag files")
    ap.add_argument("--out", default=DEFAULT_OUT, help="sensor-noise YAML to write (default: the open_water preset)")
    ap.add_argument("--report", default=DEFAULT_REPORT, help="folder for report.md and the chart")
    args = ap.parse_args(argv)
    bags = sorted(p for p in glob.glob(os.path.join(args.bags, "*_data.bag"))
                  if Path(p).name[:16] >= FIRST_IN_WATER)
    if not bags:
        print(f"ERROR: no SOLAQUA *_data.bag files in {args.bags!r} (see this script's docstring for the download)",
              file=sys.stderr)
        return 2
    rows = []
    for p in bags:
        rows.append(fit_run(p))
        print(f"  {rows[-1]['run']}: {rows[-1]['duration_s']:.0f} s", flush=True)
    summary = summarise(rows)
    write_preset(args.out, summary, rows)
    from sensor_noise import load_sensor_noise
    cfg = load_sensor_noise(args.out)                # the written file must load
    write_report(args.report, summary, rows, args.out)
    print(f"\nWrote {args.out} from {len(rows)} recordings:\n  {cfg.describe()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
