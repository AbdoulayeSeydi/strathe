"""
Turns calibration parameters + real logged runs into residuals (for the
optimizer) or RMSE numbers (for reporting/diagnostics).

## Pipeline
    read_run_csv() -> validate_run_dataframe() -> convert_run_dataframe() -> RunData
    RunData -> residuals_for_run() -> joint_residuals() -> scipy.optimize.least_squares

Nothing here is coupled to scipy, HoloOcean, or any specific optimizer —
this module only produces residual vectors and RMSE numbers; optimizer.py
decides what to do with them.

## Sampling rate — empirically determined, not guessed
There is no silent/implicit default in THIS file — DatasetConfig always
requires sampling_rate_hz explicitly (run_calibration.py's CLI supplies a
default of 10.0 Hz, but that's a caller decision, not something baked in
here). The dataset has no explicit time column, so this was originally an
unverified 50Hz placeholder used throughout early calibration runs.

That 50Hz placeholder was WRONG. A dedicated investigation (see
check_sampling_rate_windowed.py, check_sampling_rate_per_run.py, and
PROJECT_STATUS.md for the full writeup) found the true rate is ~10Hz — a
5x error, meaning every acceleration estimate and every fitted damping/
inertia parameter from every calibration run before that investigation was
computed at the wrong timescale. The 10Hz finding itself is well-validated:
confirmed against synthetic ground truth, shown to survive a deliberate
sensor-bias drift stress test, consistent across all 4 real collection
dates independently, stable across multiple window-length choices, and
confirmed by row counts clustering at clean round-number durations (200s/
400s/600s at 10Hz) — a structural signature of a real fixed clock, not an
artifact of curve-fitting.

Still worth getting this right for any NEW dataset rather than assuming
10Hz transfers: if the real rate isn't known for a different dataset, that
has to be resolved the same way (or better) before trusting calibration
results, not guessed and moved past.

## This is system identification, not full forward prediction
Initial velocity (RunData.nu0) is taken directly from the logged data at
t=0 — i.e. assumed exactly known, no estimation error. Orientation
(RunData.roll_pitch) is the REAL LOGGED trajectory, fed into the dynamics
model at every timestep rather than self-integrated (see
dynamics_model.py's TEACHER_FORCED_ORIENTATION). Both are standard and
correct for offline system identification — the question being asked is
"do these parameters explain the observed trajectory," not "can this model
run open-loop from just an initial condition." But it means these RMSE
numbers describe how well the parameters explain observed data, not how a
forward-simulating deployment (like HoloOcean) would perform standalone.
Don't conflate the two when these numbers get used anywhere else in the repo.

## Known scientific limitation (not fixed here, deliberately)
Residuals are VELOCITY error only — no position, orientation, or
acceleration terms. This means multiple different parameter combinations
can plausibly explain similar velocity trajectories (weaker identifiability
than a residual that also penalized position/orientation drift). Fine for
this MVP; worth revisiting if the calibrated parameters turn out to be
unstable/inconsistent across repeated fits.

## Objective function abstraction
Only least-squares (RMSE) is implemented. joint_residuals() returns a raw
residual vector, which is already the right shape to hand to any of
scipy.optimize.least_squares' built-in loss= options (e.g. loss="huber")
without restructuring this file — so a full pluggable-objective abstraction
isn't needed yet; scipy already provides it at the residual-vector boundary.

## Performance note
load_run() does real file I/O and CSV parsing — call it ONCE per run before
optimization starts, not per optimizer iteration. optimizer.py is expected
to load all training/held-out RunData up front and pass the already-loaded
objects into repeated joint_residuals() calls. Likewise, orientation
(quaternion -> roll/pitch) is converted once at load time, not recomputed
inside the optimizer's hot loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from dynamics_model import (
    CalibrationParameters,
    VehicleConstants,
    SimulationConfig,
    TrajectoryInputs,
    simulate_trajectory,
    quaternion_to_roll_pitch,
)
from data_split import extract_date

REQUIRED_COLUMNS = [
    "Fx", "Fy", "Fz", "Tx", "Ty", "Tz",
    "Bu", "Bv", "Bw", "Bp", "Bq", "Br",
    "qx", "qy", "qz", "qw",
]


@dataclass
class DatasetConfig:
    """
    Sampling rate must be explicitly configured — see module docstring.
    No default is provided on purpose.
    """
    sampling_rate_hz: float

    def validate(self) -> None:
        if self.sampling_rate_hz <= 0:
            raise ValueError(f"sampling_rate_hz must be positive, got {self.sampling_rate_hz}")

    @property
    def dt(self) -> float:
        return 1.0 / self.sampling_rate_hz


@dataclass
class RunData:
    """One run's worth of data, ready for simulation. Named fields, not a
    dict — no run["nu0"]-style typo risk."""
    filepath: str
    forces_torques: np.ndarray      # (N,6) logged Fx,Fy,Fz,Tx,Ty,Tz
    roll_pitch: np.ndarray          # (N,2) REAL LOGGED roll,pitch (radians) — teacher forcing input
    real_nu: np.ndarray             # (N,6) ground truth [u,v,w,p,q,r]
    nu0: np.ndarray                 # (6,) — ASSUMPTION: taken as exactly known, see module docstring
    dt: float
    sample_rate_hz: float
    date: Optional[str] = None      # best-effort from filename; None if unparseable
    n_samples: int = field(init=False)

    def __post_init__(self):
        self.n_samples = self.forces_torques.shape[0]


# ---------------------------------------------------------------------------
# Loading pipeline — split into named, independently-testable steps.
# ---------------------------------------------------------------------------

def read_run_csv(filepath: str) -> pd.DataFrame:
    """Step 1: raw read, no interpretation."""
    return pd.read_csv(filepath)


def validate_run_dataframe(df: pd.DataFrame, filepath: str) -> None:
    """Step 2: fail loudly and specifically, before any conversion happens."""
    if df.empty:
        raise ValueError(f"Empty CSV, no rows: {filepath}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{filepath} is missing required column(s): {missing}")

    nan_cols = [c for c in REQUIRED_COLUMNS if df[c].isna().any()]
    if nan_cols:
        raise ValueError(f"{filepath} contains NaN in required column(s): {nan_cols}")


def convert_run_dataframe(df: pd.DataFrame, filepath: str, dataset_config: DatasetConfig) -> RunData:
    """Step 3: DataFrame -> RunData. Orientation is converted ONCE here,
    not recomputed on every optimizer iteration (see Performance note)."""
    dataset_config.validate()

    forces_torques = df[["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]].to_numpy()
    real_nu = df[["Bu", "Bv", "Bw", "Bp", "Bq", "Br"]].to_numpy()
    roll, pitch = quaternion_to_roll_pitch(
        df["qx"].to_numpy(), df["qy"].to_numpy(),
        df["qz"].to_numpy(), df["qw"].to_numpy(),
    )
    roll_pitch = np.stack([roll, pitch], axis=1)

    try:
        date = extract_date(filepath)
    except ValueError:
        date = None  # best-effort metadata — not fatal if the filename doesn't match

    return RunData(
        filepath=filepath,
        forces_torques=forces_torques,
        roll_pitch=roll_pitch,
        real_nu=real_nu,
        nu0=real_nu[0].copy(),
        dt=dataset_config.dt,
        sample_rate_hz=dataset_config.sampling_rate_hz,
        date=date,
    )


def load_run(filepath: str, dataset_config: DatasetConfig) -> RunData:
    """Orchestrates read -> validate -> convert. Call once per run, before
    optimization starts (see Performance note in module docstring)."""
    df = read_run_csv(filepath)
    validate_run_dataframe(df, filepath)
    return convert_run_dataframe(df, filepath, dataset_config)


# ---------------------------------------------------------------------------
# Residuals / RMSE
# ---------------------------------------------------------------------------

ParamsLike = Union[np.ndarray, CalibrationParameters]


def _coerce_params(params: ParamsLike) -> CalibrationParameters:
    return params if isinstance(params, CalibrationParameters) else CalibrationParameters.from_vector(params)


def _raw_residuals(params: CalibrationParameters, run: RunData, vehicle: VehicleConstants) -> np.ndarray:
    """(simulated - real) velocity residuals, shape (N,6), in each DOF's
    natural physical units (m/s for u,v,w; rad/s for p,q,r) — unweighted,
    unnormalized. This is the basis for both RMSE reporting and the
    optimizer-facing residual vector."""
    sim_config = SimulationConfig(dt=run.dt)
    trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                    roll_pitch_series=run.roll_pitch)
    result = simulate_trajectory(params, vehicle, sim_config, trajectory)
    assert result.used_teacher_forced_orientation, (
        "Expected teacher-forced orientation from this backend — if this "
        "assertion fires, simulate_trajectory's implementation changed "
        "without updating this check (see module docstring)."
    )
    residuals = result.nu - run.real_nu

    if not np.all(np.isfinite(residuals)):
        raise ValueError(
            f"Non-finite residuals for run {run.filepath!r} — check dt, "
            f"parameter bounds, or whether this parameter combination made "
            f"the simulation unstable (see dynamics_model.py's numerical "
            f"stability note)."
        )
    return residuals


def compute_dof_scales(runs: Sequence[RunData], epsilon: float = 1e-3) -> np.ndarray:
    """
    Per-DOF scale factors (shape (6,)) for residual weighting: the standard
    deviation of each DOF's REAL logged velocity across the given runs.
    Dividing residuals by these scales puts angular (rad/s) and linear
    (m/s) DOFs on comparable footing before they're summed into one
    least-squares objective — without this, whichever DOF naturally has
    larger magnitude dominates the fit regardless of physical importance.

    Compute this from TRAINING runs only and reuse the same scales for
    held-out evaluation — computing it from held-out data would leak
    held-out statistics into what's supposed to be an out-of-sample check.
    """
    all_nu = np.concatenate([run.real_nu for run in runs], axis=0)
    scales = np.std(all_nu, axis=0)
    return np.maximum(scales, epsilon)  # avoid divide-by-zero on a near-constant DOF


def residuals_for_run(
    params: ParamsLike,
    run: RunData,
    vehicle: VehicleConstants,
    dof_scales: Optional[np.ndarray] = None,
    normalize_by_run_length: bool = True,
) -> np.ndarray:
    """
    Flattened residuals for the OPTIMIZER (not for human-readable RMSE —
    see rmse_for_run for that). Two optional adjustments, both documented
    here since they change what the optimizer actually minimizes:

    dof_scales: if given (see compute_dof_scales), residuals are divided
        per-DOF before flattening, preventing large-magnitude DOFs from
        dominating purely due to units. None = unweighted (not
        recommended without a specific reason — see module docstring's
        scientific limitations).

    normalize_by_run_length: if True (default), each run's residuals are
        divided by sqrt(n_samples) so a 6000-sample run doesn't
        contribute 6x the squared-residual weight of a 1000-sample run
        purely because it's longer. Disable only if unequal run-length
        weighting is actually intended.

    Flatten order: row-major (C order) — for each timestep, all 6 DOFs
    appear consecutively, then the next timestep. DOF varies fastest
    within a timestep block; time is the outer, slower-varying index.
    """
    params = _coerce_params(params)
    residuals = _raw_residuals(params, run, vehicle)

    if dof_scales is not None:
        residuals = residuals / dof_scales

    if normalize_by_run_length:
        residuals = residuals / np.sqrt(run.n_samples)

    return residuals.ravel()


def joint_residuals(
    param_vec: np.ndarray,
    runs: Sequence[RunData],
    vehicle: VehicleConstants,
    dof_scales: Optional[np.ndarray] = None,
    normalize_by_run_length: bool = True,
) -> np.ndarray:
    """
    Concatenated residuals across MULTIPLE runs — what
    scipy.optimize.least_squares actually minimizes, so no single run can
    dominate the fit (see the earlier held-out-by-date discussion: fitting
    one run at a time risks absorbing that run's specific quirks as if
    they were vehicle properties).
    """
    params = CalibrationParameters.from_vector(param_vec)
    all_residuals = [
        residuals_for_run(params, run, vehicle, dof_scales, normalize_by_run_length)
        for run in runs
    ]
    return np.concatenate(all_residuals)


def rmse_for_run(params: ParamsLike, run: RunData, vehicle: VehicleConstants) -> float:
    """Human-readable, UNWEIGHTED RMSE across all 6 DOFs, natural units.
    For reporting/comparison — NOT what the optimizer minimizes (that's
    joint_residuals, which may be weighted/normalized)."""
    params = _coerce_params(params)
    residuals = _raw_residuals(params, run, vehicle)
    return float(np.sqrt(np.mean(residuals ** 2)))


def rmse_by_dof(params: ParamsLike, run: RunData, vehicle: VehicleConstants) -> np.ndarray:
    """Per-DOF RMSE, shape (6,) = [u,v,w,p,q,r]. Diagnostic: reveals which
    axis is actually driving the fit error, rather than one averaged number."""
    params = _coerce_params(params)
    residuals = _raw_residuals(params, run, vehicle)
    return np.sqrt(np.mean(residuals ** 2, axis=0))


def max_abs_residual(params: ParamsLike, run: RunData, vehicle: VehicleConstants) -> float:
    params = _coerce_params(params)
    residuals = _raw_residuals(params, run, vehicle)
    return float(np.max(np.abs(residuals)))


def mean_run_rmse(params: ParamsLike, runs: Sequence[RunData], vehicle: VehicleConstants) -> float:
    """Average per-run RMSE — the human-readable summary metric (renamed
    from mean_rmse for clarity: this averages RMSE ACROSS RUNS, each of
    which is itself already an RMSE across timesteps/DOFs)."""
    params = _coerce_params(params)
    return float(np.mean([rmse_for_run(params, run, vehicle) for run in runs]))


@dataclass
class Diagnostics:
    per_run_rmse: dict           # {filepath: rmse}
    per_dof_rmse: np.ndarray     # (6,) averaged across runs — [u,v,w,p,q,r]
    max_abs_residual: float
    n_runs: int


def compute_diagnostics(params: ParamsLike, runs: Sequence[RunData], vehicle: VehicleConstants) -> Diagnostics:
    """RMSE by run, RMSE by DOF, and the single worst residual — for
    inspecting a fit rather than just reading one averaged number."""
    params = _coerce_params(params)
    per_run = {run.filepath: rmse_for_run(params, run, vehicle) for run in runs}
    per_dof = np.mean([rmse_by_dof(params, run, vehicle) for run in runs], axis=0)
    worst = max(max_abs_residual(params, run, vehicle) for run in runs)
    return Diagnostics(per_run_rmse=per_run, per_dof_rmse=per_dof,
                        max_abs_residual=worst, n_runs=len(runs))