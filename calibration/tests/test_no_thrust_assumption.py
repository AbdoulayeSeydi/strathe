"""Nothing `strathe` or the web page runs may assume a vehicle's forces are in its logged wrench (PROJECT_STATUS.md #97).

A fin's force isn't logged: it's worked out at each step from the logged angle. Every lab-facing module that reads the logged wrench is
listed here with how a fin vehicle's controls travel with it. A new one fails this test until it's checked and listed."""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CAL, "stress"))

AUDITED = {
    "data_ingestion.py": "the wrench is optional; fins, rpm, gimbal, moving mass and buoyancy are their own channels",
    "diagnose_calibration.py": "only identifiability_analysis is reachable, which runs no trajectory; its plots are a dev tool",
    "dynamics_model.py": "control_series travels beside the wrench; a fin vehicle without it raises",
    "estimate_current.py": "windows slice controls with the wrench",
    "free_decay.py": "decay tests log no forces and no fins move: the wrench columns are zero by definition",
    "measurements.py": "slice_run and the anchor's prediction carry controls",
    "mujoco_backend.py": "experiments: a fin vehicle is refused here (test_fin_experiments.py), and goes through FastCore",
    "objective.py": "horizon windows slice controls; thruster lag filters the wrench only, never a fin's force",
    "simfit/core.py": "window_rows adds the propeller's force and passes fin angles to the engine; strathe_free_run uses control_at",
    "state_smoother.py": "the smoother's physics takes each sample's controls",
    "validation.py": "free runs use control_at; step detection on the wrench is hover's, fin_steps is a fin vehicle's",
}


def test_every_wrench_reader_is_audited():
    import inventory
    found = set()
    for rel in inventory.reachable():
        try:
            text = open(os.path.join(CAL, rel), encoding="utf-8").read()
        except OSError:
            continue
        if re.search(r"\.forces_torques\b", text) or re.search(r"[\"']Fx[\"']", text):
            found.add(rel)
    new = sorted(found - set(AUDITED))
    assert not new, f"{new} read the logged wrench and aren't audited for fin vehicles: check them and add them to AUDITED with how they handle it"
    gone = sorted(set(AUDITED) - found)
    assert not gone, f"{gone} no longer read the wrench: drop them from AUDITED"
