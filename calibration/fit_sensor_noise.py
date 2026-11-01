#!/usr/bin/env python3
"""
fit_sensor_noise.py -- measure how noisy a vehicle's own sensor readings are,
from its logs, and write a sensor-noise file the simulators can use.

    python fit_sensor_noise.py --data-dir <logs in Strathe's CSV layout> --out my_rov_noise.yaml \
        [--position dead_reckoning|external] [--sampling-rate-hz 10]
    (or: python strathe.py fit-noise --data <logs> --out my_rov_noise.yaml)

Then: `strathe fleet --noise my_rov_noise.yaml`, `strathe arena --noise ...`,
`strathe train-fleet --noise ...`, or `sensor_noise: {preset: my_rov_noise.yaml}`
in a fleet experiment YAML.

Method: a logged signal is the true motion plus sensor noise. The vehicle's
motion is smooth from one sample to the next, the noise isn't, so the second
difference x[k+1] - 2 x[k] + x[k-1] is almost all noise; for white noise of
standard deviation s its spread is sqrt(6) s. The spread is taken robustly
(1.4826 x the median absolute deviation) so manoeuvres and glitches don't
inflate it, per recording, and the median over recordings is reported.
Measured per logged sample: the simulator applies the numbers per 0.1 s tick,
so logs at a very different rate give a rough value (it says so).

Filtered logs (F5): many autopilots log a smoothed velocity or attitude, and
then the second difference is the smoothed motion, not noise, so it would
report almost no noise. For white noise, consecutive second differences
correlate at exactly -2/3 (about +-0.04 over 800 samples); smoothing pulls
that up (an 8-sample moving average of noise: -0.46; smooth motion: positive).
A channel above -0.55 is reported as not measurable, and the `typical`
preset's value is written for it, with a note, instead of ~0.

What logs can and can't tell:
  - white noise of velocity, rates, roll/pitch, heading, depth and (if the
    logs have positions) position: measured
  - DVL dropouts: the fraction of samples where the logged velocity repeats
    exactly (a held reading) or is missing: measured
  - biases (a constant offset per dive) and navigation drift: NOT measurable
    without an independent reference; they are written as 0 with a note.
    Put your sensors' datasheet values in the file if you know them.

--position: how the vehicle knows where it is. external (a tank's motion
capture, a USBL fix) = white noise, no drift; dead_reckoning (default when the
logs have no positions) = integrated from the DVL and compass, so it drifts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import yaml

SQRT6 = np.sqrt(6.0)


def white_noise_std(x: np.ndarray) -> float:
    """Per-sample white-noise standard deviation of a smooth signal plus noise (robust second difference)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return float("nan")
    d2 = np.diff(x, 2)
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(1.4826 * mad / SQRT6)


# white noise gives -2/3 (about +-0.04 over 800 samples); an 8-sample moving average of noise gives -0.46 and smooth
# motion is positive: above -0.55 (about 3 standard errors from -2/3), the channel is smoothed
FILTERED_ABOVE = -0.55


def noise_correlation(x: np.ndarray) -> float:
    """Lag-1 correlation of the second difference: -2/3 for white noise, positive when the logs are smoothed."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return float("nan")
    d2 = np.diff(x, 2)
    if np.std(d2[1:]) == 0 or np.std(d2[:-1]) == 0:
        return float("nan")
    return float(np.corrcoef(d2[1:], d2[:-1])[0, 1])


def held_fraction(block: np.ndarray) -> float:
    """Fraction of samples whose (u, v, w) repeat the previous sample exactly, or are missing."""
    block = np.asarray(block, dtype=float)
    if len(block) < 2:
        return 0.0
    missing = ~np.all(np.isfinite(block), axis=1)
    same = np.r_[False, np.all(block[1:] == block[:-1], axis=1)]
    return float(np.mean(missing | same))


def _sensor_columns(df: pd.DataFrame) -> dict:
    """The sensor suite's channels (#96) measured here: the first position fix's n/e as x/y, the first depth sensor as
    z (a logged x/y/z, the reference dataset's motion capture, is used when there's no such channel)."""
    out = {}
    pos = sorted(c[:-2] for c in df.columns if c.startswith("pos_") and c.endswith("_n"))
    if pos:
        out["x"], out["y"] = pos[0] + "_n", pos[0] + "_e"
    depth = sorted(c for c in df.columns if c.startswith("depth_"))
    if depth:
        out["z"] = depth[0]
    return out


def fit_recording(df: pd.DataFrame) -> dict:
    from scipy.spatial.transform import Rotation
    out = {}
    channel = {c: c for c in ("Bu", "Bv", "Bw", "Bp", "Bq", "Br", "x", "y", "z")}
    channel.update(_sensor_columns(df))
    for key, col in channel.items():
        if col in df:
            # a sensor's own readings only: a column with gaps (a DVL whose dropouts were kept, a 1 Hz fix) is NaN
            # between readings, and the second difference is taken over consecutive readings
            x = df[col].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            if len(x) < 5:
                continue
            out[key] = white_noise_std(x)
            out[f"{key}_corr"] = noise_correlation(x)
    if all(c in df for c in ("qx", "qy", "qz", "qw")):
        q = df[["qx", "qy", "qz", "qw"]].values
        ok = np.all(np.isfinite(q), axis=1) & (np.linalg.norm(q, axis=1) > 0.5)
        if ok.sum() >= 5:
            e = np.unwrap(Rotation.from_quat(q[ok]).as_euler("xyz"), axis=0)
            out["roll"], out["pitch"], out["yaw"] = (white_noise_std(e[:, i]) for i in range(3))
            out["roll_corr"], out["pitch_corr"], out["yaw_corr"] = (noise_correlation(e[:, i]) for i in range(3))
    if all(c in df for c in ("Bu", "Bv", "Bw")):
        out["held"] = held_fraction(df[["Bu", "Bv", "Bw"]].values)
    out["samples"] = len(df)
    return out


def _typical_preset() -> dict:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_noise", "typical_dvl_ahrs.yaml"),
              encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("sensors", {})


def _median(rows, key):
    vals = [r[key] for r in rows if key in r and np.isfinite(r[key])]
    return float(np.median(vals)) if vals else None


def fit(files: list, position: str = None) -> tuple:
    rows = []
    for f in files:
        df = pd.read_csv(f)
        r = fit_recording(df)
        r["file"] = os.path.basename(f)
        rows.append(r)
    have_pos = any("x" in r for r in rows)
    position = position or ("external" if have_pos else "dead_reckoning")
    mean = lambda keys: (float(np.mean([m for m in (_median(rows, k) for k in keys) if m is not None]))
                         if any(_median(rows, k) is not None for k in keys) else 0.0)
    sensors = {
        "velocity_std_mps": round(mean(("Bu", "Bv", "Bw")), 6),
        "velocity_bias_mps": 0.0,
        "velocity_dropout_prob": round(min(0.5, _median(rows, "held") or 0.0), 4),
        "gyro_std_rps": round(mean(("Bp", "Bq", "Br")), 6),
        "gyro_bias_rps": 0.0,
        "attitude_std_rad": round(mean(("roll", "pitch")), 6),
        "attitude_bias_rad": 0.0,
        "heading_std_rad": round(_median(rows, "yaw") or 0.0, 6),
        "heading_bias_rad": 0.0,
        "depth_std_m": round(_median(rows, "z") or 0.0, 6),
        "position": position,
        "position_std_m": round(mean(("x", "y")) if (position == "external" and have_pos) else 0.0, 6),
    }
    # F5: a channel whose second differences aren't noise (smoothed logs) can't be measured: say so, use the preset
    groups = {"velocity_std_mps": ("Bu", "Bv", "Bw"), "gyro_std_rps": ("Bp", "Bq", "Br"),
              "attitude_std_rad": ("roll", "pitch"), "heading_std_rad": ("yaw",), "depth_std_m": ("z",)}
    if sensors["position"] == "external" and have_pos:
        groups["position_std_m"] = ("x", "y")
    filtered = {}
    preset = None
    for key, cols in groups.items():
        corr = [m for m in (_median(rows, f"{c}_corr") for c in cols) if m is not None]
        if corr and float(np.median(corr)) > FILTERED_ABOVE:
            if preset is None:
                preset = _typical_preset()
            filtered[key] = {"measured": sensors[key], "second_difference_correlation": round(float(np.median(corr)), 3),
                             "written": preset.get(key, sensors[key])}
            sensors[key] = preset.get(key, sensors[key])
    # a sensor the vehicle doesn't have (#96: no DVL, no depth sensor) isn't measured as perfect: the preset's value
    # is written, and the note says it wasn't measured
    absent = {}
    for key, cols, what in (("velocity_std_mps", ("Bu", "Bv", "Bw"), "no DVL"), ("depth_std_m", ("z",), "no depth")):
        if not any(_median(rows, c) is not None for c in cols):
            preset = preset or _typical_preset()
            absent[key] = {"why": f"{what} in these logs", "written": preset.get(key, sensors[key])}
            sensors[key] = preset.get(key, sensors[key])
    if absent:
        filtered.update({k: {**v, "second_difference_correlation": None, "measured": None} for k, v in absent.items()})
    return sensors, rows, filtered


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="logs in Strathe's CSV layout (see `strathe ingest`)")
    ap.add_argument("--out", required=True, help="sensor-noise YAML to write")
    ap.add_argument("--position", choices=["dead_reckoning", "external"],
                    help="how the vehicle knows its position (default: external if the logs have x/y, "
                         "else dead_reckoning)")
    ap.add_argument("--sampling-rate-hz", type=float, help="default: ingest_report.json, else 10")
    args = ap.parse_args(argv)
    from data_split import discover_csv_files
    files, _ = discover_csv_files(args.data_dir)
    if not files:
        print(f"ERROR: no CSV logs in {args.data_dir}", file=sys.stderr)
        return 2
    rate = args.sampling_rate_hz
    ip = os.path.join(args.data_dir, "ingest_report.json")
    if rate is None and os.path.exists(ip):
        with open(ip, encoding="utf-8") as f:
            rate = json.load(f).get("sampling_rate_hz")
    rate = float(rate or 10.0)
    try:
        sensors, rows, filtered = fit(files, args.position)
    except (KeyError, ValueError, pd.errors.ParserError) as e:
        print(f"ERROR: couldn't read the logs: {e}", file=sys.stderr)
        return 2
    notes = ["White noise measured per logged sample by the robust second-difference method; the simulator "
             "applies it per 0.1 s tick."]
    if abs(rate - 10.0) > 2.5:
        notes.append(f"The logs are at {rate:g} Hz, not 10 Hz: per-sample noise at another rate is only a rough "
                     f"guide to per-tick noise in the simulator.")
    notes.append("Biases (a constant offset per dive) can't be measured from logs without an independent reference; "
                 "they are 0 here. Put your sensors' datasheet values in if you know them.")
    absent = {k: v for k, v in filtered.items() if v.get("why")}
    filtered = {k: v for k, v in filtered.items() if not v.get("why")}
    if absent:
        notes.append("Not measured: " + ", ".join(f"{k} ({v['why']})" for k, v in absent.items())
                     + ". The `typical` preset's values are written for those; the simulators model a DVL and a depth "
                       "sensor on every vehicle (calibration/SENSORS.md).")
    if filtered:
        notes.append("These logs look smoothed (filtered by the vehicle, or resampled by interpolating between "
                     "samples) for " + ", ".join(
            f"{k} (second differences correlate at {v['second_difference_correlation']:+.2f}; white noise gives "
            f"-0.67)" for k, v in filtered.items())
            + ": the noise can't be measured from them, and measuring would have given almost none. The `typical` "
              "preset's values are written for those instead. Log the raw sensor readings at their own rate (ingest "
              "without resample_hz), or put your datasheet values in.")
    if sensors["position"] == "dead_reckoning":
        notes.append("position_std_m is the error of the starting position fix; set it to your fix's accuracy.")
    doc = {"sensors": sensors, "provenance": "measured",
           "fit": {"date": date.today().isoformat(), "method": "fit_sensor_noise.py (robust second difference)",
                   "logs": os.path.basename(os.path.abspath(args.data_dir)), "recordings": len(rows),
                   "sampling_rate_hz": rate, "notes": notes, "not_measurable_filtered": filtered or None,
                   **({"not_measured_absent": absent} if absent else {})},
           "per_recording": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()} for r in rows]}
    from sensor_noise import load_sensor_noise
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Sensor noise measured from a vehicle's logs. See sensor_noise.py.\n")
        yaml.safe_dump(doc, f, sort_keys=False)
    cfg = load_sensor_noise(args.out)  # the written file must load
    print(f"Wrote {args.out} from {len(rows)} recording(s):\n  {cfg.describe()}")
    for n in notes:
        print(f"  note: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
