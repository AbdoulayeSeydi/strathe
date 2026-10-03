"""
The second public BlueROV2 dataset (DFKI, Scientific Data 2025, Zenodo 10.5281/zenodo.17360027,
CC BY 4.0): an ArduSub vehicle with motion capture, so thruster conventions can be checked against
measured motion. PROJECT_STATUS.md #62. Skipped unless the dataset is unzipped under data/raw/.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import thruster_model as tm

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
DFKI = os.path.join(os.path.dirname(CAL), "data", "raw", "dfki_rov_tether", "rov_tether_dataset_v1.2")
RUNS = sorted(glob.glob(os.path.join(DFKI, "*", "*", "rc-out.csv")))
EXAMPLE = os.path.join(CAL, "examples", "dfki_tether")
pytestmark = pytest.mark.skipif(not RUNS, reason="DFKI tether dataset not downloaded (see examples/dfki_tether)")


def test_preset_yaw_torque_turns_the_vehicle_the_way_its_own_gyro_says():
    """Motors 1-4 are the same in ArduSub's 6- and 8-thruster frames, so the Heavy preset's horizontal
    columns apply to this vehicle. In FRD the yaw torque they give follows ArduSub's measured yaw rate in
    every run; the reference dataset's FLU version runs against it."""
    frd, flu = tm.bluerov2_heavy(), tm.bluerov2_heavy(frame="reference")
    for rc_path in RUNS:
        rc = pd.read_csv(rc_path)
        att = pd.read_csv(os.path.join(os.path.dirname(rc_path), "attitude.csv"))
        pwm = np.column_stack([rc[[f"channel{i}" for i in range(1, 5)]].to_numpy(), np.full((len(rc), 4), 1500.0)])
        r = np.interp(rc["time"], att["time"], att["yaw_speed"])
        assert np.corrcoef(frd.wrench(pwm)[:, 5], r)[0, 1] > 0.2, rc_path
        assert np.corrcoef(flu.wrench(pwm)[:, 5], r)[0, 1] < -0.2, rc_path


def test_prepared_and_ingested_forces_push_the_way_the_vehicle_moves(tmp_path):
    """The worked example end to end up to calibration: prepare_dfki_tether.py, then ingestion with the
    example spec. Each axis's force and velocity move together (they'd be negative with a mirrored
    layout)."""
    sys.path.insert(0, EXAMPLE)
    try:
        import prepare_dfki_tether as prep
    finally:
        sys.path.remove(EXAMPLE)
    prepared = tmp_path / "prepared"
    assert prep.main([DFKI, str(prepared)]) == 0
    files = sorted(prepared.glob("*.csv"))
    assert len(files) >= 20
    assert sum(len(pd.read_csv(f)) for f in files) > 30000  # ~55 min of the ~63 min recorded

    from data_ingestion import ingest, load_ingest_spec
    spec = yaml.safe_load(open(os.path.join(EXAMPLE, "ingest_dfki_tether.yaml"), encoding="utf-8"))
    spec["source"]["files"] = str(prepared / "*.csv")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    s = ingest(load_ingest_spec(str(tmp_path / "spec.yaml")), str(tmp_path / "out"))
    assert s["sessions"] == {"01-01-2025": len([f for f in files if "limited" in f.name and "unlimited" not in f.name]),
                             "02-01-2025": len([f for f in files if "unlimited" in f.name])}
    assert any("pitch torque" in w for w in s["warnings"])  # a 6-thruster BlueROV2 can't pitch on command
    d = pd.concat([pd.read_csv(f) for f in sorted((tmp_path / "out").glob("*.csv"))])
    for force, vel in (("Fx", "Bu"), ("Fy", "Bv"), ("Fz", "Bw"), ("Tz", "Br")):
        assert np.corrcoef(d[force], d[vel])[0, 1] > 0.4, (force, vel)
    assert 0.1 < d["Bu"].mean() < 0.4  # mostly driving forwards
