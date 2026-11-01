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

## Prediction horizon (PROJECT_STATUS.md #50)
By default each run is replayed from its first sample to its last with
velocity integrated open-loop (only orientation is teacher-forced). Motion
the logged inputs don't cause -- this dataset's roll and pitch, see #50 --
then accumulates for the whole run, and the only way the optimizer can
limit it is to push that axis's damping to its bound: the classic
output-error bias under unmodelled disturbances. horizon_samples=H instead
re-anchors the simulation to the real logged velocity every H samples
and scores only the H-1 predicted samples of each window (the anchor
sample is exact by construction, so it is dropped). Disturbances then
act like noise over a short window instead of an ever-growing drift.

## Skill against "predict zero" (PROJECT_STATUS.md #50)
An RMSE is only meaningful next to what a trivial predictor achieves.
zero_baseline_rmse() is the error of predicting "no motion on this axis";
skill = 1 - RMSE_model / RMSE_zero. Skill <= 0 means the model is no better
than assuming the vehicle isn't moving on that axis, however much it beats
default physics. compute_zero_baseline_scales() uses the same quantity to
put each axis on that footing inside the objective.

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

from dataclasses import dataclass, field, replace
from typing import Callable, Optional, Sequence, Union

import os

import numpy as np
import pandas as pd

from dynamics_model import (
    CalibrationParameters,
    VehicleConstants,
    SimulationConfig,
    TrajectoryInputs,
    SimulationResult,
    simulate_trajectory,
    quaternion_to_roll_pitch,
)
from data_split import extract_date

# A pluggable simulation backend — anything matching this signature can be
# passed as simulate_fn to any function below. Defaults to the analytical
# stand-in (dynamics_model.simulate_trajectory) everywhere; pass a
# HoloOcean-bound callable (see holoocean_backend.py) to evaluate the SAME
# calibration parameters inside real HoloOcean physics instead — nothing
# else about these functions changes. Building this callable (e.g. binding
# a persistent HoloOcean env instance via functools.partial) is the
# caller's responsibility, not this module's.
SimulateFn = Callable[
    [CalibrationParameters, VehicleConstants, SimulationConfig, TrajectoryInputs],
    SimulationResult,
]

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
    # Move the point the logged torques are taken about (see
    # shift_torque_reference and PROJECT_STATUS.md #50). (0, 0, 0) = use the
    # log's torques unchanged (the original behaviour).
    torque_reference_offset_m: tuple = (0.0, 0.0, 0.0)
    # Known water current during the recordings, in the LOG's world frame
    # (the frame its orientation quaternions are expressed in), m/s. None =
    # still water (a tank). Open-water logs taken in a current the model
    # doesn't know about bias every drag estimate (PROJECT_STATUS.md #55).
    current_world_mps: Optional[tuple] = None
    # A different current for each recording: ((file basename, (vx, vy, vz)), ...), e.g. from
    # currents.csv written by estimate_current.py. A recording listed here uses its own current
    # instead of current_world_mps.
    current_by_file: Optional[tuple] = None

    def current_for(self, filepath: str) -> Optional[tuple]:
        if self.current_by_file:
            from data_split import recording_path
            name = os.path.basename(recording_path(filepath))
            for f, c in self.current_by_file:
                if f == name:
                    return tuple(c)
        return self.current_world_mps

    def validate(self) -> None:
        if self.sampling_rate_hz <= 0:
            raise ValueError(f"sampling_rate_hz must be positive, got {self.sampling_rate_hz}")
        if len(self.torque_reference_offset_m) != 3 or not np.all(np.isfinite(self.torque_reference_offset_m)):
            raise ValueError(f"torque_reference_offset_m must be 3 finite numbers (m, body FRD), "
                             f"got {self.torque_reference_offset_m}")
        if self.current_world_mps is not None and (len(self.current_world_mps) != 3
                                                   or not np.all(np.isfinite(self.current_world_mps))):
            raise ValueError(f"current_world_mps must be 3 finite numbers (m/s), got {self.current_world_mps}")

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
    current_body: Optional[np.ndarray] = None  # (N,3) known water current in body axes per sample
    orientation_quat: Optional[np.ndarray] = None  # (N,4) logged body->world quaternions (x, y, z, w)
    n_samples: int = field(init=False)

    def __post_init__(self):
        self.n_samples = self.forces_torques.shape[0]


# ---------------------------------------------------------------------------
# Loading pipeline — split into named, independently-testable steps.
# ---------------------------------------------------------------------------

def read_run_csv(filepath: str) -> pd.DataFrame:
    """Step 1: raw read, no interpretation. `path::rows=A-B` reads rows A..B-1 of the recording: the part of one
    recording a one-session split trains or scores on (data_split.py, #93)."""
    from data_split import parse_ref
    path, a, b = parse_ref(filepath)
    df = pd.read_csv(path)
    return df if a is None else df.iloc[a:b].reset_index(drop=True)


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


def shift_torque_reference(forces_torques: np.ndarray, offset_m) -> np.ndarray:
    """Re-express (N,6) body wrenches [F; T] about a point at `offset_m`
    (body FRD, metres) from the original reference: T' = T - r x F.

    Why (PROJECT_STATUS.md #50): the reference BlueROV2 dataset computes
    torques about a point 8.5 cm below the horizontal thrusters, so every
    surge/sway command also logs a pitch/roll torque (logged Ty = -0.98
    correlated with Fx, Tx = +0.95 with Fy). The dynamics model applies all
    forces at its origin and has no drag moment to balance that torque, so it
    predicts large pitch/roll the real vehicle never shows. offset
    (0, 0, -0.085) moves the reference up to the thruster plane, which is
    where the vehicle's drag acts in steady motion."""
    r = np.asarray(offset_m, dtype=float)
    if not np.any(r):
        return forces_torques
    out = np.array(forces_torques, dtype=float, copy=True)
    out[:, 3:6] = out[:, 3:6] - np.cross(r, out[:, 0:3])
    return out


def convert_run_dataframe(df: pd.DataFrame, filepath: str, dataset_config: DatasetConfig) -> RunData:
    """Step 3: DataFrame -> RunData. Orientation is converted ONCE here,
    not recomputed on every optimizer iteration (see Performance note)."""
    dataset_config.validate()

    forces_torques = shift_torque_reference(df[["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]].to_numpy(),
                                            dataset_config.torque_reference_offset_m)
    real_nu = df[["Bu", "Bv", "Bw", "Bp", "Bq", "Br"]].to_numpy()
    roll, pitch = quaternion_to_roll_pitch(
        df["qx"].to_numpy(), df["qy"].to_numpy(),
        df["qz"].to_numpy(), df["qw"].to_numpy(),
    )
    roll_pitch = np.stack([roll, pitch], axis=1)
    quats = df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=float)
    current = dataset_config.current_for(filepath)
    current_body = None
    if current is not None and np.any(current):
        from scipy.spatial.transform import Rotation
        current_body = Rotation.from_quat(quats).inv().apply(np.asarray(current, dtype=float))

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
        current_body=current_body,
        orientation_quat=quats,
    )


def run_with_current(run: RunData, current_world) -> RunData:
    """The same run, assumed recorded in a constant world-frame current (m/s); None or zero = still water."""
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    c = None if current_world is None else np.asarray(current_world, dtype=float)
    if c is None or not np.any(c):
        return replace(run, current_body=None)
    return replace(run, current_body=Rotation.from_quat(run.orientation_quat).inv().apply(c))


def read_currents_file(path: str) -> tuple:
    """currents.csv (file, vx_mps, vy_mps, vz_mps; one row per recording) -> DatasetConfig.current_by_file."""
    df = pd.read_csv(path)
    need = {"file", "vx_mps", "vy_mps", "vz_mps"}
    if not need <= set(df.columns):
        raise ValueError(f"{path} needs columns {sorted(need)} (one row per recording), has {list(df.columns)}")
    out = []
    for _, r in df.iterrows():
        c = (float(r["vx_mps"]), float(r["vy_mps"]), float(r["vz_mps"]))
        if not np.all(np.isfinite(c)):
            raise ValueError(f"{path}: non-numeric current for {r['file']}")
        out.append((os.path.basename(str(r["file"])), c))
    return tuple(out)


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


def horizon_windows(n_samples: int, horizon_samples: Optional[int]) -> list:
    """[(start, end)] prediction windows covering a run. None = one window
    (the whole run). The last window is kept even if shorter than the
    horizon, as long as it predicts at least one sample."""
    if horizon_samples is None or horizon_samples >= n_samples:
        return [(0, n_samples)]
    if horizon_samples < 2:
        raise ValueError(f"horizon_samples must be >= 2 (one anchor + one prediction), got {horizon_samples}")
    return [(a, min(a + horizon_samples, n_samples)) for a in range(0, n_samples - 1, horizon_samples)
            if min(a + horizon_samples, n_samples) - a >= 2]


def with_thruster_lag(run: RunData, params: CalibrationParameters) -> RunData:
    """The run with its logged (commanded) wrench replaced by what lagging thrusters apply over each sample
    (dynamics_model.apply_thruster_lag, params.thruster_time_constant_s). The run itself when there is no lag.
    Applied to a whole recording before it's cut into prediction windows, so each window starts from the
    thrusters' real state."""
    T = float(getattr(params, "thruster_time_constant_s", 0.0) or 0.0)
    if T <= 0:
        return run
    from dynamics_model import apply_thruster_lag
    return replace(run, forces_torques=apply_thruster_lag(run.forces_torques, T, run.dt))


# A blown-up evaluation's residuals: this many times those of predicting no motion (residuals_for_run).
DIVERGED_PENALTY = 10.0


def _zero_prediction_rows(run: RunData, horizon_samples: Optional[int] = None) -> np.ndarray:
    """The logged velocity in the rows a normal evaluation's residuals cover (whole run, or the prediction windows
    after each re-anchored start), i.e. the residuals, up to sign, of predicting no motion."""
    if horizon_samples is not None and horizon_samples < run.n_samples:
        return np.concatenate([run.real_nu[a:b][1:] for a, b in horizon_windows(run.n_samples, horizon_samples)], axis=0)
    return np.asarray(run.real_nu, dtype=float)


def _raw_residuals(
    params: CalibrationParameters,
    run: RunData,
    vehicle: VehicleConstants,
    simulate_fn: Optional[SimulateFn] = None,
    horizon_samples: Optional[int] = None,
) -> np.ndarray:
    """(simulated - real) velocity residuals, with the thruster lag applied once to the whole recording (see
    with_thruster_lag, and _lagged_raw_residuals for the rest)."""
    return _lagged_raw_residuals(params, with_thruster_lag(run, params), vehicle, simulate_fn, horizon_samples)


def _lagged_raw_residuals(
    params: CalibrationParameters,
    run: RunData,
    vehicle: VehicleConstants,
    simulate_fn: Optional[SimulateFn] = None,
    horizon_samples: Optional[int] = None,
) -> np.ndarray:
    """(simulated - real) velocity residuals, shape (N,6), in each DOF's
    natural physical units (m/s for u,v,w; rad/s for p,q,r) — unweighted,
    unnormalized. This is the basis for both RMSE reporting and the
    optimizer-facing residual vector.

    simulate_fn: defaults to the analytical stand-in
        (dynamics_model.simulate_trajectory). Pass a HoloOcean-bound
        callable (see holoocean_backend.py) to evaluate the SAME
        parameters/run against real HoloOcean physics instead — every
        caller of this function (residuals_for_run, joint_residuals,
        rmse_for_run, etc.) accepts and forwards this same parameter.
    """
    if horizon_samples is not None and horizon_samples < run.n_samples:
        # re-anchored prediction windows (see "Prediction horizon" above)
        parts = []
        for a, b in horizon_windows(run.n_samples, horizon_samples):
            sub = replace(run, forces_torques=run.forces_torques[a:b], roll_pitch=run.roll_pitch[a:b],
                          real_nu=run.real_nu[a:b], nu0=run.real_nu[a].copy(),
                          current_body=None if run.current_body is None else run.current_body[a:b],
                          orientation_quat=None if run.orientation_quat is None else run.orientation_quat[a:b])
            parts.append(_lagged_raw_residuals(params, sub, vehicle, simulate_fn, None)[1:])
        return np.concatenate(parts, axis=0)

    simulate = simulate_fn if simulate_fn is not None else simulate_trajectory
    sim_config = SimulationConfig(dt=run.dt)
    trajectory = TrajectoryInputs(nu0=run.nu0, forces_torques=run.forces_torques,
                                    roll_pitch_series=run.roll_pitch, current_body_series=run.current_body)
    result = simulate(params, vehicle, sim_config, trajectory)

    if simulate_fn is None:
        # Only enforced for the DEFAULT stand-in backend — this guards
        # against dynamics_model.py's own implementation silently changing
        # to non-teacher-forced behavior without updating this check. A
        # custom simulate_fn (e.g. the HoloOcean backend) is EXPECTED to
        # do genuine forward simulation (used_teacher_forced_orientation=
        # False) — that's by design, not an error condition here.
        assert result.used_teacher_forced_orientation, (
            "Expected teacher-forced orientation from the default stand-in "
            "backend — if this assertion fires, simulate_trajectory's "
            "implementation changed without updating this check (see "
            "module docstring)."
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


def zero_baseline_rmse(runs: Sequence[RunData]) -> np.ndarray:
    """Per-DOF RMSE of predicting zero velocity, shape (6,): the trivial
    baseline every per-axis accuracy number should be read against (see
    "Skill against predict zero" above). Averaged per run, like
    rmse_by_dof results usually are."""
    return np.mean([np.sqrt(np.mean(run.real_nu ** 2, axis=0)) for run in runs], axis=0)


def compute_zero_baseline_scales(runs: Sequence[RunData], epsilon: float = 1e-3) -> np.ndarray:
    """Per-DOF residual scales for the objective: the "predict zero" error
    on the given (TRAINING) runs, so each axis is scored relative to the
    trivial baseline for that axis rather than in its raw units. Pooled
    over all samples (not averaged per run) so short runs don't distort it."""
    all_nu = np.concatenate([run.real_nu for run in runs], axis=0)
    return np.maximum(np.sqrt(np.mean(all_nu ** 2, axis=0)), epsilon)


def skill_vs_zero(model_rmse: np.ndarray, zero_rmse: np.ndarray) -> np.ndarray:
    """1 - RMSE_model / RMSE_zero per axis. 1 = perfect, 0 = no better than
    predicting "no motion", negative = worse than that."""
    return 1.0 - np.asarray(model_rmse, dtype=float) / np.asarray(zero_rmse, dtype=float)


def residuals_for_run(
    params: ParamsLike,
    run: RunData,
    vehicle: VehicleConstants,
    dof_scales: Optional[np.ndarray] = None,
    normalize_by_run_length: bool = True,
    simulate_fn: Optional[SimulateFn] = None,
    horizon_samples: Optional[int] = None,
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

    simulate_fn: see _raw_residuals — defaults to the analytical stand-in.

    horizon_samples: None = replay the whole run open-loop (the original
        objective); H = re-anchored H-sample prediction windows (see the
        module docstring's "Prediction horizon" section).

    Flatten order: row-major (C order) — for each timestep, all 6 DOFs
    appear consecutively, then the next timestep. DOF varies fastest
    within a timestep block; time is the outer, slower-varying index.
    """
    params = _coerce_params(params)
    try:
        residuals = _raw_residuals(params, run, vehicle, simulate_fn, horizon_samples)
    except (FloatingPointError, ValueError) as e:
        if isinstance(e, ValueError) and "Non-finite" not in str(e):
            raise
        # These parameters make the simulation blow up on this recording. Tell the optimizer it's a very bad fit,
        # rather than stopping the whole calibration (PROJECT_STATUS.md #77; URI's X2 crashed this way): the
        # residuals of predicting no motion, x DIVERGED_PENALTY, in the shape a normal evaluation has.
        residuals = -DIVERGED_PENALTY * _zero_prediction_rows(run, horizon_samples)

    if dof_scales is not None:
        residuals = residuals / dof_scales

    if normalize_by_run_length:
        residuals = residuals / np.sqrt(residuals.shape[0])

    return residuals.ravel()


def joint_residuals(
    param_vec: np.ndarray,
    runs: Sequence[RunData],
    vehicle: VehicleConstants,
    dof_scales: Optional[np.ndarray] = None,
    normalize_by_run_length: bool = True,
    simulate_fn: Optional[SimulateFn] = None,
    horizon_samples: Optional[int] = None,
) -> np.ndarray:
    """
    Concatenated residuals across MULTIPLE runs — what
    scipy.optimize.least_squares actually minimizes, so no single run can
    dominate the fit (see the earlier held-out-by-date discussion: fitting
    one run at a time risks absorbing that run's specific quirks as if
    they were vehicle properties).

    simulate_fn: see _raw_residuals — pass a HoloOcean-bound callable to
    run the optimizer against real HoloOcean physics instead of the
    stand-in. Every run uses the SAME simulate_fn (and, if it wraps a
    persistent HoloOcean environment, the SAME running instance) — no
    per-run or per-evaluation restart cost.
    """
    params = CalibrationParameters.from_vector(param_vec)
    all_residuals = [
        residuals_for_run(params, run, vehicle, dof_scales, normalize_by_run_length, simulate_fn,
                          horizon_samples)
        for run in runs
    ]
    return np.concatenate(all_residuals)


# --- parallel evaluation ----------------------------------------------------
# One full-dataset stand-in evaluation is ~16 s of pure-Python integration on
# an M1, and a fit needs several hundred. Runs are independent, so they are
# spread over worker processes that each hold their own copy of the runs
# (sent once, at pool start-up, not on every evaluation).
_POOL_RUNS = None
_POOL_VEHICLE = None


def _pool_init(runs, vehicle):
    import os as _os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[var] = "1"
    global _POOL_RUNS, _POOL_VEHICLE
    _POOL_RUNS, _POOL_VEHICLE = runs, vehicle


def _pool_residuals(job):
    i, param_vec, dof_scales, normalize_by_run_length, horizon_samples = job
    return residuals_for_run(CalibrationParameters.from_vector(param_vec), _POOL_RUNS[i], _POOL_VEHICLE,
                             dof_scales, normalize_by_run_length, None, horizon_samples)


class ParallelJointResiduals:
    """Drop-in for joint_residuals (same call signature) that evaluates runs
    in worker processes. Results are identical to joint_residuals, in the
    same order. Stand-in physics only: a simulate_fn bound to a live engine
    (HoloOcean, MuJoCo model/data) can't be shared across processes.
    Use as a context manager, or call close()."""

    def __init__(self, runs: Sequence[RunData], vehicle: VehicleConstants, workers: int):
        from concurrent.futures import ProcessPoolExecutor
        if workers < 2:
            raise ValueError("ParallelJointResiduals needs workers >= 2; use joint_residuals otherwise")
        self._runs = list(runs)
        self._vehicle = vehicle
        # longest runs first so no worker is left with a long run at the end
        self._order = sorted(range(len(self._runs)), key=lambda i: -self._runs[i].n_samples)
        self._pool = ProcessPoolExecutor(max_workers=workers, initializer=_pool_init,
                                         initargs=(self._runs, vehicle))

    def __call__(self, param_vec, runs, vehicle, dof_scales=None, normalize_by_run_length=True,
                 simulate_fn=None, horizon_samples=None) -> np.ndarray:
        if simulate_fn is not None:
            raise ValueError("ParallelJointResiduals only supports the stand-in physics (simulate_fn=None)")
        if len(runs) != len(self._runs) or any(a is not b for a, b in zip(runs, self._runs)):
            raise ValueError("ParallelJointResiduals was built for a different set of runs")
        vec = np.asarray(param_vec, dtype=float)
        jobs = [(i, vec, dof_scales, normalize_by_run_length, horizon_samples) for i in self._order]
        parts = dict(zip(self._order, self._pool.map(_pool_residuals, jobs)))
        return np.concatenate([parts[i] for i in range(len(self._runs))])

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def rmse_for_run(
    params: ParamsLike, run: RunData, vehicle: VehicleConstants,
    simulate_fn: Optional[SimulateFn] = None,
) -> float:
    """Human-readable, UNWEIGHTED RMSE across all 6 DOFs, natural units.
    For reporting/comparison — NOT what the optimizer minimizes (that's
    joint_residuals, which may be weighted/normalized).
    simulate_fn: see _raw_residuals."""
    params = _coerce_params(params)
    residuals = _raw_residuals(params, run, vehicle, simulate_fn)
    return float(np.sqrt(np.mean(residuals ** 2)))


def rmse_by_dof(
    params: ParamsLike, run: RunData, vehicle: VehicleConstants,
    simulate_fn: Optional[SimulateFn] = None,
    horizon_samples: Optional[int] = None,
) -> np.ndarray:
    """Per-DOF RMSE, shape (6,) = [u,v,w,p,q,r]. Diagnostic: reveals which
    axis is actually driving the fit error, rather than one averaged number.
    simulate_fn: see _raw_residuals. horizon_samples: see residuals_for_run."""
    params = _coerce_params(params)
    residuals = _raw_residuals(params, run, vehicle, simulate_fn, horizon_samples)
    return np.sqrt(np.mean(residuals ** 2, axis=0))


def max_abs_residual(
    params: ParamsLike, run: RunData, vehicle: VehicleConstants,
    simulate_fn: Optional[SimulateFn] = None,
) -> float:
    params = _coerce_params(params)
    residuals = _raw_residuals(params, run, vehicle, simulate_fn)
    return float(np.max(np.abs(residuals)))


def mean_run_rmse(
    params: ParamsLike, runs: Sequence[RunData], vehicle: VehicleConstants,
    simulate_fn: Optional[SimulateFn] = None,
) -> float:
    """Average per-run RMSE — the human-readable summary metric (renamed
    from mean_rmse for clarity: this averages RMSE ACROSS RUNS, each of
    which is itself already an RMSE across timesteps/DOFs).
    simulate_fn: see _raw_residuals."""
    params = _coerce_params(params)
    return float(np.mean([rmse_for_run(params, run, vehicle, simulate_fn) for run in runs]))


def stable_rmse_by_dof(
    params: ParamsLike, runs: Sequence[RunData], vehicle: VehicleConstants,
    horizon_samples: Optional[int] = None, simulate_fn: Optional[SimulateFn] = None,
) -> tuple:
    """(per-DOF RMSE averaged over the runs the simulation survives, number of runs where it diverged). For
    evaluating a finished calibration: a parameter set that blows up on some recordings is reported, not a crash
    (PROJECT_STATUS.md #64: roll/pitch damping held from free-decay tests diverged on 9 of 14 held-out runs' 3 s
    windows). All runs diverging gives NaN."""
    params = _coerce_params(params)
    rows, diverged = [], 0
    for run in runs:
        try:
            rows.append(rmse_by_dof(params, run, vehicle, simulate_fn, horizon_samples))
        except FloatingPointError:
            diverged += 1
    return (np.mean(rows, axis=0) if rows else np.full(6, np.nan)), diverged


def stable_mean_run_rmse(
    params: ParamsLike, runs: Sequence[RunData], vehicle: VehicleConstants,
    simulate_fn: Optional[SimulateFn] = None,
) -> tuple:
    """(mean_run_rmse over the runs the simulation survives, number of runs where it diverged)."""
    params = _coerce_params(params)
    vals, diverged = [], 0
    for run in runs:
        try:
            vals.append(rmse_for_run(params, run, vehicle, simulate_fn))
        except FloatingPointError:
            diverged += 1
    return (float(np.mean(vals)) if vals else float("nan")), diverged


@dataclass
class Diagnostics:
    per_run_rmse: dict           # {filepath: rmse}
    per_dof_rmse: np.ndarray     # (6,) averaged across runs — [u,v,w,p,q,r]
    max_abs_residual: float
    n_runs: int


def compute_diagnostics(
    params: ParamsLike, runs: Sequence[RunData], vehicle: VehicleConstants,
    simulate_fn: Optional[SimulateFn] = None,
) -> Diagnostics:
    """RMSE by run, RMSE by DOF, and the single worst residual — for
    inspecting a fit rather than just reading one averaged number.
    simulate_fn: see _raw_residuals."""
    params = _coerce_params(params)
    per_run = {run.filepath: rmse_for_run(params, run, vehicle, simulate_fn) for run in runs}
    per_dof = np.mean([rmse_by_dof(params, run, vehicle, simulate_fn) for run in runs], axis=0)
    worst = max(max_abs_residual(params, run, vehicle, simulate_fn) for run in runs)
    return Diagnostics(per_run_rmse=per_run, per_dof_rmse=per_dof,
                        max_abs_residual=worst, n_runs=len(runs))