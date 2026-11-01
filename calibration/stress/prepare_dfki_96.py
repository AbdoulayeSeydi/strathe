"""
stress/prepare_dfki_96.py -- DFKI's BlueROV2 as a lab with no DVL (#96): its own sensors, the motion capture's
velocity hidden.

    python stress/prepare_dfki_96.py <out folder>

The DFKI ROV-tether dataset (examples/dfki_tether) logs what a BlueROV2 lab usually has: ArduSub's attitude and gyro,
a raw IMU, two pressure sensors (the Bar30 outside, one inside the hull), thruster PWM and battery voltage, plus a
12-camera Qualisys motion capture. The worked example's prepared recordings (data/raw/dfki_rov_tether/strathe_csv)
carry the motion-capture velocity, which Strathe calibrates from today. This adds each sensor's own readings to
those recordings, where they happened (sensor_ingest.place_on_grid: nothing interpolated), and ingests them through
the slack-tether spec six ways:

    v     the example's velocity (the reference fit, V; also the hidden truth everything else is scored against)
    d1    motion-capture positions (the origin marker) and the Bar30: no velocity
    d2    the positions as a USBL would give them (1 Hz, 0.17 m noise, 6.5% dropouts: SOLAQUA's) and the Bar30
    d3    the Bar30, ArduSub's gyro and attitude, nothing else: the BlueROV2 lab with no DVL and no positioning
    d4    d3 and the raw IMU's accelerometer
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, CAL)
RAW = os.path.join(os.path.dirname(CAL), "data", "raw", "dfki_rov_tether")
PREPARED = os.path.join(RAW, "strathe_csv")
RUNS = os.path.join(RAW, "rov_tether_dataset_v1.2")
SPEC = os.path.join(CAL, "examples", "dfki_tether", "ingest_dfki_tether_slack.yaml")


def run_folder(name: str) -> str:
    """2025-01-01_limited_10m(_partN).csv -> 'limited cable/10m'; 2025-01-02_unlimited_circle -> 'unlimited cable/circle'."""
    stem = os.path.splitext(name)[0].split("_", 1)[1]
    kind, rest = stem.split("_", 1)
    rest = rest.split("_part")[0]
    return os.path.join(RUNS, f"{kind} cable", rest)


def augment(out_dir: str, seed: int = 96) -> list:
    import sensor_ingest as si
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for f in sorted(glob.glob(os.path.join(PREPARED, "*.csv"))):
        d = pd.read_csv(f)
        t = d["time_s"].to_numpy()
        rd = run_folder(os.path.basename(f))
        qtm = pd.read_csv(os.path.join(rd, "qtm_rov.csv"))
        o = qtm[["rov_origin_x", "rov_origin_y", "rov_origin_z"]].to_numpy()
        ok = np.isfinite(o).all(1)
        # basin frame (z up) -> the prepared files' NED-like world (x, -y, -z), as prepare_dfki_tether.py does
        pos = np.c_[o[ok, 0], -o[ok, 1], -o[ok, 2]]
        g = si.place_on_grid(qtm["time"].to_numpy()[ok], pos, t)
        d["mocap_n"], d["mocap_e"], d["mocap_d"] = g[:, 0], g[:, 1], g[:, 2]
        fix = np.zeros(len(t), bool)
        fix[::10] = True
        fix &= np.isfinite(g).all(1) & (rng.random(len(t)) >= 0.065)
        noisy = g + 0.17 * rng.normal(size=g.shape)
        for k, c in enumerate("ned"):
            d[f"usbl_{c}"] = np.where(fix, noisy[:, k], np.nan)
        pr = pd.read_csv(os.path.join(rd, "pressure2.csv"))   # the Bar30, outside the hull (Pa)
        d["bar30_pa"] = si.place_on_grid(pr["time"].to_numpy(), pr[["press_abs"]].to_numpy(), t)[:, 0]
        imu = pd.read_csv(os.path.join(rd, "imu_scaled.csv"))  # in g, FRD
        a = si.place_on_grid(imu["time"].to_numpy(), imu[["xacc", "yacc", "zacc"]].to_numpy(), t)
        d["acc_x_g"], d["acc_y_g"], d["acc_z_g"] = a[:, 0], a[:, 1], a[:, 2]
        p = os.path.join(out_dir, os.path.basename(f))
        d.to_csv(p, index=False)
        written.append(p)
    return written


def specs(raw_dir: str) -> dict:
    base = yaml.safe_load(open(SPEC, encoding="utf-8"))
    base["source"]["files"] = os.path.join(raw_dir, "*.csv")
    gyro = {"columns": {"p": "p_rps", "q": "q_rps", "r": "r_rps"}, "angular_unit": "rad/s"}
    bar30 = {"name": "bar30", "pressure": {"column": "bar30_pa", "unit": "Pa", "water": "fresh", "surface": "auto"}}
    out = {"v": dict(base)}
    out["d1"] = {**base, "body_velocity": gyro, "depth": bar30,
                 "position": {"use": "calibrate", "source": "mocap", "name": "mocap", "x": "mocap_n", "y": "mocap_e",
                              "z": "mocap_d"}}
    out["d2"] = {**base, "body_velocity": gyro, "depth": bar30,
                 "position": {"use": "calibrate", "source": "usbl", "name": "usbl", "x": "usbl_n", "y": "usbl_e",
                              "z": "usbl_d"}}
    out["d3"] = {**base, "body_velocity": gyro, "depth": bar30}
    out["d4"] = {**base, "body_velocity": gyro, "depth": bar30,
                 "accelerometer": {"x": "acc_x_g", "y": "acc_y_g", "z": "acc_z_g", "unit": "g"}}
    return out


def main(out: str) -> int:
    import data_ingestion as di
    raw = os.path.join(out, "raw")
    augment(raw)
    lens = {}
    for tag, sp in specs(raw).items():
        p = os.path.join(out, f"{tag}.yaml")
        yaml.safe_dump(sp, open(p, "w", encoding="utf-8"), sort_keys=False)
        s = di.ingest(di.load_ingest_spec(p), os.path.join(out, tag))
        lens[tag] = {f["output"]: f["rows_out"] for f in s["files"] if f.get("output")}
        print(tag, len(lens[tag]), "recordings;", (s.get("sensors") or {}).get("axes_seen_by"), flush=True)
    for tag in lens:   # every version on the same rows: the truth lines up sample for sample
        assert lens[tag] == lens["v"], f"{tag} isn't on the reference's rows"
    truth = os.path.join(out, "truth")
    os.makedirs(truth, exist_ok=True)
    for f in lens["v"]:
        pd.read_csv(os.path.join(out, "v", f))[["Bu", "Bv", "Bw", "Bp", "Bq", "Br"]].to_csv(os.path.join(truth, f),
                                                                                            index=False)
    print("truth written for", len(lens["v"]), "recordings")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
