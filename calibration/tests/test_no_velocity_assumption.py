"""Nothing `strathe` or the web page runs may assume a vehicle logs its body velocity (PROJECT_STATUS.md #96).

Every lab-facing module that reads the logged velocity (RunData.real_nu, or the Bu/Bv/Bw columns) is listed here with
how it handles a recording without a DVL: NaN-aware, a sensor-suite branch, or a refusal with a clear message. A new
module that reads velocity fails this test until it's checked and listed: the assumption can't come back unnoticed."""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(CAL, "stress"))

AUDITED = {
    "fin_fit.py": "speed from a DVL if it read at most samples, else from the propeller's rpm; gyro rates otherwise",
    "objective.py": "the sensor-suite path (run.channels) for every residual and RMSE; NaN-aware baselines",
    "measurements.py": "u, v, w only where a DVL read (the DVL channel); never the hidden velocity",
    "state_smoother.py": "the gyro rates only (real_nu[:, 3:6])",
    "data_ingestion.py": "body velocity is optional; DVL gaps kept; force check from fixes and depth without it",
    "validation.py": "start_velocity / drift against fixes; NaN-aware response, steps and run picking",
    "estimate_current.py": "sensor-suite setup and slicing (measurements.slice_run)",
    "fit_sensor_noise.py": "each sensor's own readings; an absent DVL is written as not measured",
    "check_coriolis_excitation.py": "products over measured samples; 'not measured (no DVL)'",
    "check_data_integrity.py": "a sensor's empty cell is no reading",
    "check_schema.py": "no DVL is a sensor suite, said, not a missing column",
    "check_physical_plausibility.py": "measured values only",
    "check_sampling_rate_windowed.py": "refuses with --sampling-rate-hz advice when velocity or positions are missing",
    "stonefish_export.py": "axis_speeds: unmeasured speeds fall back to its short speed grid",
    "diagnose_calibration.py": "only identifiability_analysis is reachable (no velocity); its plots are a dev tool",
    "simfit/sensors.py": "gyro rates (real_nu[:, 3:6]) and NaN-aware velocity errors where a DVL read",
    "simfit/core.py": "fit-sim: windows start from the estimated velocity, scored against the sensors (simfit/sensors.py)",
}


def test_every_velocity_reader_is_audited():
    import inventory
    found = set()
    for rel in inventory.reachable():
        try:
            text = open(os.path.join(CAL, rel), encoding="utf-8").read()
        except OSError:
            continue
        if re.search(r"\.real_nu\b", text) or re.search(r"[\"']B[uvw][\"']", text):
            found.add(rel)
    new = sorted(found - set(AUDITED))
    assert not new, (f"{new} read the logged velocity and aren't audited for vehicles without a DVL: check them and "
                     f"add them to AUDITED with how they handle it")
    gone = sorted(set(AUDITED) - found)
    assert not gone, f"{gone} no longer read velocity: drop them from AUDITED"
