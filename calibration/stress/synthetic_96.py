"""
stress/synthetic_96.py -- vehicles whose physics is known exactly, measured by any mix of sensors (#96).

A truth vehicle (a parameter file, e.g. calibrated_parameters_fleet_truth.yaml) replays real recordings' logged thrust
and orientation through Strathe's physics. Its sensors then read that motion with chosen rates, noise, dropouts and
latency, and are written as ingested recordings (the canonical CSVs calibration reads). The truth's own body velocity
goes to a separate folder the calibration is never pointed at: it's what a fit made without a DVL is scored against.

    truth_dataset(out, sources, truth_params, spec, sensors) -> writes out/data/*.csv and out/truth/*.csv
    score(cal_params, truth_params, spec, truth_dir, heldout) -> how close the fit got to the truth

Sensors, as a dict:
    {"dvl": {"rate_hz": 10, "noise": 0.0, "dropout": 0.0},
     "position": {"name": "usbl", "rate_hz": 1.0, "noise": 0.17, "dropout": 0.065, "delay_s": 0.0, "axes": "ned"},
     "depth": {"name": "main", "rate_hz": 16, "noise": 0.0024},
     "gyro": {"noise": 0.0, "bias": 0.0}}
Leaving a sensor out leaves it out of the vehicle.
"""
from __future__ import annotations

import glob
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
REPO = os.path.dirname(CAL)
if CAL not in sys.path:
    sys.path.insert(0, CAL)

REFERENCE = os.path.join(REPO, "data", "raw", "Bluerov2_trajectory_dataset", "dataset")
TRUTH_BLUEROV2 = os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml")
SPEC_BLUEROV2 = os.path.join(CAL, "vehicle_spec.yaml")


def load_params(path: str):
    from dynamics_model import CalibrationParameters
    with open(path, encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return CalibrationParameters(**(d.get("parameters") or d))


def _sample_mask(n: int, every: int, dropout: float, rng, offset: int = 0) -> np.ndarray:
    m = np.zeros(n, dtype=bool)
    m[offset::max(1, every)] = True
    if dropout > 0:
        m &= rng.random(n) >= dropout
    return m


def truth_dataset(out: str, sources: list, truth_path: str = TRUTH_BLUEROV2, spec_path: str = SPEC_BLUEROV2,
                  sensors: Optional[dict] = None, seed: int = 0, rate_hz: float = 10.0,
                  current_world: Optional[tuple] = None) -> dict:
    """Write the truth vehicle's sensor readings for each source recording (see module docstring)."""
    from dynamics_model import SimulationConfig, TrajectoryInputs, simulate_trajectory
    from objective import DatasetConfig, load_run, run_with_current
    from optimizer import load_vehicle_spec
    from run_calibration import spec_torque_reference_offset
    from scipy.spatial.transform import Rotation
    sensors = sensors or {}
    rng = np.random.default_rng(seed)
    spec = load_vehicle_spec(spec_path)
    truth = load_params(truth_path)
    dc = DatasetConfig(sampling_rate_hz=rate_hz, torque_reference_offset_m=spec_torque_reference_offset(spec))
    data_dir, truth_dir = os.path.join(out, "data"), os.path.join(out, "truth")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(truth_dir, exist_ok=True)
    every = lambda hz: max(1, int(round(rate_hz / hz)))
    written = []
    for src in sources:
        raw = pd.read_csv(src)
        run = load_run(src, dc)
        if current_world is not None:
            run = run_with_current(run, current_world)
        n = run.n_samples
        traj = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques, roll_pitch_series=run.roll_pitch,
                                current_body_series=run.current_body)
        nu = simulate_trajectory(truth, spec.vehicle, SimulationConfig(dt=run.dt), traj).nu
        R = Rotation.from_quat(run.orientation_quat).as_matrix()
        vw = np.einsum("nij,nj->ni", R, nu[:, :3])
        path = np.vstack([np.zeros(3), np.cumsum(vw[:-1] * run.dt, axis=0)]) + np.array([2.0, -1.0, 3.0])
        df = pd.DataFrame({c: raw[c].to_numpy(dtype=float) for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")})
        g = sensors.get("gyro", {})
        ang = nu[:, 3:6] + g.get("bias", 0.0) + g.get("noise", 0.0) * rng.normal(size=(n, 3))
        df["Bp"], df["Bq"], df["Br"] = ang[:, 0], ang[:, 1], ang[:, 2]
        for k, c in enumerate(("qx", "qy", "qz", "qw")):
            df[c] = run.orientation_quat[:, k]
        if "dvl" in sensors:
            d = sensors["dvl"]
            m = _sample_mask(n, every(d.get("rate_hz", rate_hz)), d.get("dropout", 0.0), rng)
            lin = nu[:, :3] + d.get("noise", 0.0) * rng.normal(size=(n, 3))
            lin[~m] = np.nan
            df["Bu"], df["Bv"], df["Bw"] = lin[:, 0], lin[:, 1], lin[:, 2]
        positions = sensors.get("position")
        for p in ([positions] if isinstance(positions, dict) else positions or []):
            name, axes = p.get("name", "usbl"), p.get("axes", "ned")
            m = _sample_mask(n, every(p.get("rate_hz", 1.0)), p.get("dropout", 0.0), rng, p.get("offset", 0))
            if p.get("surface_only_below_m") is not None:
                m &= path[:, 2] - path[:, 2].min() < p["surface_only_below_m"]
            lag = int(round(p.get("delay_s", 0.0) * rate_hz))
            src_idx = np.clip(np.arange(n) - lag, 0, n - 1)
            vals = path[src_idx] + p.get("noise", 0.0) * rng.normal(size=(n, 3))
            vals[~m] = np.nan
            for k, a in enumerate("ned"):
                if a in axes:
                    df[f"pos_{name}_{a}"] = vals[:, k]
        dep = sensors.get("depth")
        for d in ([dep] if isinstance(dep, dict) else dep or []):
            m = _sample_mask(n, every(d.get("rate_hz", rate_hz)), d.get("dropout", 0.0), rng)
            vals = path[:, 2] + d.get("noise", 0.0) * rng.normal(size=n)
            vals[~m] = np.nan
            df[f"depth_{d.get('name', 'main')}"] = vals
        if "accelerometer" in sensors:
            a = sensors["accelerometer"]
            vdot = np.gradient(nu[:, :3], run.dt, axis=0)
            f = vdot + np.cross(nu[:, 3:6], nu[:, :3]) + np.einsum("nji,j->ni", R, np.array([0.0, 0.0, -9.80665]))
            f = f + a.get("bias", 0.0) + a.get("noise", 0.0) * rng.normal(size=(n, 3))
            df["acc_x"], df["acc_y"], df["acc_z"] = f[:, 0], f[:, 1], f[:, 2]
        name = os.path.basename(src)
        df.to_csv(os.path.join(data_dir, name), index=False)
        pd.DataFrame({"Bu": nu[:, 0], "Bv": nu[:, 1], "Bw": nu[:, 2], "Bp": nu[:, 3], "Bq": nu[:, 4],
                      "Br": nu[:, 5]}).to_csv(os.path.join(truth_dir, name), index=False)
        written.append(name)
    return {"data": data_dir, "truth": truth_dir, "files": written}


def range_error(cal, truth, spec_path: str = SPEC_BLUEROV2, names: Optional[list] = None) -> dict:
    """|calibrated - truth| as a fraction of each parameter's allowed range (vehicle_spec bounds), per parameter."""
    from optimizer import load_vehicle_spec, parameter_bounds
    spec = load_vehicle_spec(spec_path)
    lb, ub = parameter_bounds(spec)
    all_names = list(truth.param_names)
    out = {}
    for i, k in enumerate(all_names):
        if names is not None and k not in names:
            continue
        w = ub[i] - lb[i]
        if w <= 0:
            continue
        out[k] = abs(float(getattr(cal, k)) - float(getattr(truth, k))) / w
    return out


def heldout_truth_skill(cal, spec_path: str, data_dir: str, truth_dir: str, files: list, horizon_s: float = 3.0,
                        rate_hz: float = 10.0) -> dict:
    """Held-out skill per axis against the hidden truth velocity: windows started from the truth (as a DVL would),
    the calibrated values' simulation against the truth. The same measure as the velocity fit's skill_heldout_short,
    so a fit without a DVL and one with it are compared like for like."""
    from objective import DatasetConfig, load_run, rmse_by_dof, zero_baseline_rmse, skill_vs_zero
    from optimizer import load_vehicle_spec
    from run_calibration import spec_torque_reference_offset
    spec = load_vehicle_spec(spec_path)
    dc = DatasetConfig(sampling_rate_hz=rate_hz, torque_reference_offset_m=spec_torque_reference_offset(spec),
                       sensors=("velocity",))
    runs = []
    for f in files:
        d = pd.read_csv(os.path.join(data_dir, f))
        t = pd.read_csv(os.path.join(truth_dir, f))
        for c in ("Bu", "Bv", "Bw", "Bp", "Bq", "Br"):
            d[c] = t[c]
        d = d[[c for c in d.columns if not c.startswith(("pos_", "depth_", "acc_"))]]
        tmp = os.path.join(truth_dir, "_score_" + f)
        d.to_csv(tmp, index=False)
        runs.append(load_run(tmp, dc))
        os.remove(tmp)
    H = max(2, int(round(horizon_s * rate_hz)))
    rm = np.mean([rmse_by_dof(cal, r, spec.vehicle, horizon_samples=H) for r in runs], axis=0)
    zero = zero_baseline_rmse(runs)
    sk = skill_vs_zero(rm, zero)
    return {"rmse": [round(float(x), 5) for x in rm], "skill": [round(float(x), 4) for x in sk],
            "mean_skill": round(float(np.mean(sk)), 4)}


def reference_sources(dates=("12-01-2024",), limit: Optional[int] = None) -> list:
    files = []
    for d in dates:
        files += sorted(glob.glob(os.path.join(REFERENCE, f"odom-{d}-run*.csv")),
                        key=lambda p: int(p.rsplit("run", 1)[1].split(".")[0]))
    return files[:limit] if limit else files
