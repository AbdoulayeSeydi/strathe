"""
stress/prepare_marinarium_96.py -- KTH's Marinarium BlueROV2 (PX4) as a lab with no DVL (#96): the motion capture's
velocity hidden.

    python stress/prepare_marinarium_96.py <folder with bags/ and px4_msgs/> <out folder>

The bags (github.com/ViktorNfa/bluerov2_dynamics, MIT; examples/marinarium/README.md) carry what a PX4 BlueROV2 lab
logs: PX4's normalized motor commands, its raw IMU (sensor_combined: gyro and accelerometer), and the Marinarium's
motion capture (pose and body velocity). There is no pressure sensor in them, so depth is the motion capture's z, as
for the reference tank (R-*). They are ingested once through examples/marinarium/ingest_marinarium.yaml, and the
recordings are then cut down to each sensor set:

    v     the motion capture's velocity and PX4's gyro (the reference fit, V; the velocity is the hidden truth)
    k1    motion-capture positions and depth: no velocity
    k2    the positions as a USBL would give them (1 Hz, 0.17 m noise, 6.5% dropouts: SOLAQUA's) and depth with a
          Bar30's noise (2.4 mm)
    k3    depth (with that noise), PX4's gyro and the attitude, nothing else
    k4    k3 and PX4's accelerometer
"""
from __future__ import annotations

import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, CAL)
SPEC = os.path.join(CAL, "examples", "marinarium", "ingest_marinarium.yaml")
VEL, POS, DEPTH = ["Bu", "Bv", "Bw"], ["pos_mocap_n", "pos_mocap_e", "pos_mocap_d"], "depth_main"
ACC = ["acc_x", "acc_y", "acc_z"]


def ingest(src: str, out: str) -> str:
    """The bags through the example's spec, as a lab would run it: everything in one ingest."""
    import data_ingestion as di
    spec = yaml.safe_load(open(SPEC, encoding="utf-8"))
    spec["source"]["files"] = os.path.join(src, "bags", "rosbag2_*")
    spec["source"]["message_types"] = [os.path.join(src, p) for p in spec["source"]["message_types"]]
    spec["position"]["use"] = "calibrate"   # written out as columns, for k1 and k2 (the example only checks them)
    sp = os.path.join(out, "ingest_marinarium.yaml")
    os.makedirs(out, exist_ok=True)
    yaml.safe_dump(spec, open(sp, "w", encoding="utf-8"), sort_keys=False)
    di.ingest(di.load_ingest_spec(sp), os.path.join(out, "all"))
    return os.path.join(out, "all")


def variants(all_dir: str, out: str, seed: int = 96) -> dict:
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(os.path.join(all_dir, "*.csv")))
    for tag in ("v", "k1", "k2", "k3", "k4", "truth"):
        shutil.rmtree(os.path.join(out, tag), ignore_errors=True)
        os.makedirs(os.path.join(out, tag))
    rows = {}
    for f in files:
        d = pd.read_csv(f)
        name = os.path.basename(f)
        rows[name] = len(d)
        d[VEL + ["Bp", "Bq", "Br"]].to_csv(os.path.join(out, "truth", name), index=False)
        base = d.drop(columns=VEL + POS + ACC)
        d.drop(columns=POS + ACC).to_csv(os.path.join(out, "v", name), index=False)
        k1 = base.copy()
        k1[POS] = d[POS]
        k1[DEPTH] = d["pos_mocap_d"]
        k1.to_csv(os.path.join(out, "k1", name), index=False)
        fix = np.zeros(len(d), bool)
        fix[::10] = True
        fix &= np.isfinite(d[POS].to_numpy()).all(1) & (rng.random(len(d)) >= 0.065)
        noisy = d[POS].to_numpy() + 0.17 * rng.normal(size=(len(d), 3))
        depth = d["pos_mocap_d"].to_numpy() + 0.0024 * rng.normal(size=len(d))
        k2 = base.copy()
        for k, c in enumerate("ned"):
            k2[f"pos_usbl_{c}"] = np.where(fix, noisy[:, k], np.nan)
        k2[DEPTH] = depth
        k2.to_csv(os.path.join(out, "k2", name), index=False)
        k3 = base.copy()
        k3[DEPTH] = depth
        k3.to_csv(os.path.join(out, "k3", name), index=False)
        k4 = k3.copy()
        k4[ACC] = d[ACC]
        k4.to_csv(os.path.join(out, "k4", name), index=False)
    for tag in ("v", "k1", "k2", "k3", "k4"):   # its rate and torque reference (the layout's: no shift), and noise
        shutil.copy(os.path.join(all_dir, "ingest_report.json"), os.path.join(out, tag))
    return rows


def main(src: str, out: str) -> int:
    rows = variants(ingest(src, out), out)
    print(f"{len(rows)} recordings ({sum(rows.values())} samples at 10 Hz): v, k1-k4 and truth in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
