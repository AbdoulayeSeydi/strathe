"""The public oceansystemslab BlueROV2 Heavy logs (data/download_dataset.sh) in Strathe's layout for `fit-sim`:
forward-right-down body, north-east-down world, as `frame: flu` ingestion converts ROS logs.

The torques get the same 8.5 cm shift Strathe's calibration applies to this dataset (vehicle_spec.yaml,
data_conventions.torque_reference_offset_m, in the dataset's own axes): afterwards they are about the
horizontal-thruster plane, which is bluerov2_gz's base_link origin. So `--reference-point 0 0 0`.

    python prepare_logs.py OUT_FOLDER [DATASET_FOLDER]
"""
import glob
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, CAL)
from data_ingestion import reference_run_in_frd  # noqa: E402
from objective import shift_torque_reference  # noqa: E402

out = os.path.abspath(sys.argv[1])
src = sys.argv[2] if len(sys.argv) > 2 else os.path.join(CAL, "..", "data", "raw", "Bluerov2_trajectory_dataset",
                                                         "dataset")
files = sorted(glob.glob(os.path.join(src, "odom-*.csv")))
if not files:
    sys.exit(f"❌ no odom-*.csv in {src}: run data/download_dataset.sh first")
os.makedirs(out, exist_ok=True)
wrench = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
for f in files:
    df = pd.read_csv(f, index_col=0)
    df[wrench] = shift_torque_reference(df[wrench].values, (0.0, 0.0, -0.085))
    reference_run_in_frd(df).to_csv(os.path.join(out, os.path.basename(f)), index=False)
with open(os.path.join(out, "ingest_report.json"), "w", encoding="utf-8") as fh:
    json.dump({"sampling_rate_hz": 10.0, "frame": "frd, converted from the dataset's ROS FLU/ENU",
               "torque_reference_offset_m": [0.0, 0.0, -0.085]}, fh, indent=2)
print(f"✅ {len(files)} recordings in {out}")
