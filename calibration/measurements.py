"""
measurements.py -- the calibration's measurement model: what each of a vehicle's sensors would have read if the
vehicle had the simulated motion (PROJECT_STATUS.md #96, SENSORS.md).

Before #96 the objective compared simulated body velocity with logged body velocity, so a vehicle had to log body
velocity (a DVL, or motion capture). Now each sensor is a channel, compared with what it actually read:

    gyro         p, q, r            the simulated body rates                          (always present)
    dvl          u, v, w            the simulated body velocity, where the DVL read   (optional, gaps allowed)
    position     n, e (, d)         the simulated path, integrated from the velocity  (USBL, LBL, mocap, GPS)
                                    with the logged orientation
    depth        d                  the simulated path's depth                       (a depth sensor)
    accelerometer ax, ay, az        the simulated specific force (opt-in, #96 measures whether it helps)

Every recording is cut into prediction windows (objective.horizon_windows). Each window starts from the logged gyro
rates and a linear velocity: the DVL's reading there, or, with no DVL reading, the state smoother's estimate from every
sensor (state_smoother.py). The simulation then runs open-loop over the window.

A position or depth fix says where the vehicle was, but nobody knows where the window started to the same accuracy.
That unknown start is removed exactly: position is linear in it, so the residuals of each window are taken about
their mean over the window's fixes (variable projection). A window needs at least two fixes of a channel for that
channel to count. The accelerometer's unknown bias is removed the same way.

Scaling: each channel-axis is divided by what predicting no motion gets wrong on the training recordings (its "still"
RMS: velocity zero, position and depth standing still at their window mean), and by the square root of its number of
readings. Each channel-axis then weighs one unit in the objective, whatever its units, rate or coverage: a 1 Hz GPS on
the surface counts as much as a 50 Hz gyro, and neither swamps the other. Skill per channel-axis is
1 - RMSE_model / RMSE_still, the sensor version of objective.skill_vs_zero.

A run with channels (RunData.channels) takes this path everywhere objective.py evaluates a run; a run without them
(a DVL at every sample, the original case) takes the original path, bit for bit.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

import numpy as np

G = 9.80665
MIN_FIXES_PER_WINDOW = 2
SCALE_EPSILON = {"gyro": 1e-3, "dvl": 1e-3, "position": 1e-3, "depth": 1e-3, "accelerometer": 1e-3}


@dataclass
class Channel:
    """One sensor's readings on the run's grid: values (N, k), NaN where it didn't read."""
    kind: str                       # gyro | dvl | position | depth | accelerometer
    name: str                       # gyro, dvl, the position/depth sensor's name, acc
    axes: tuple                     # ("p","q","r") | ("u","v","w") | ("n","e"[,"d"]) | ("d",) | ("ax","ay","az")
    values: np.ndarray
    lever_arm_m: Optional[np.ndarray] = None   # the sensor's position in the body, FRD, from the reference point
    track: str = "bottom"                      # a DVL: bottom (over ground) or water (through the water)
    shifted_s: float = 0.0                     # a late position sensor's delay already taken off (state_smoother)

    @property
    def labels(self) -> list:
        return [f"{self.name}.{a}" for a in self.axes]


@dataclass
class Plan:
    """Per run and horizon: the windows, and for each window and channel the sample indices that count. Fixed for a
    run, so the residual vector has the same length at every evaluation."""
    windows: list
    picks: list                     # per window: {channel index: sample indices (absolute)}
    counts: np.ndarray              # (n channel-axes,) readings that count, over the run
    still: list                     # per channel-axis: residuals of predicting no motion (for scales and skill)
    R: np.ndarray                   # (N, 3, 3) logged body->world rotation


# ---------------------------------------------------------------------------- channels from a run's columns

GRADES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_grades.yaml")
NOISE_KINDS = ("dvl", "position", "depth")   # the sensors whose noise the state smoother uses


def sensor_grades() -> dict:
    """sensor_grades.yaml's presets: {grade: {kind, noise | noise_fraction_of_range, source}}."""
    import yaml
    with open(GRADES_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)["grades"]


def _positive(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x) and x > 0


def sensor_mounts(raw_spec: dict) -> dict:
    """The vehicle description's sensors: section -> {name: {"position_m": (3,), "track": "bottom"|"water", "grade",
    "noise", "range_m"}}. Refused, with what's wrong, if malformed."""
    out = {}
    grades = None
    for name, m in ((raw_spec or {}).get("sensors") or {}).items():
        m = m or {}
        pos = m.get("position_m", [0.0, 0.0, 0.0])
        if not (isinstance(pos, (list, tuple)) and len(pos) == 3 and all(isinstance(x, (int, float)) for x in pos)):
            raise ValueError(f"vehicle description sensors.{name}.position_m: three numbers (m, body FRD from the "
                             f"reference point), got {pos!r}")
        track = m.get("track") or "bottom"
        if track not in ("bottom", "water"):
            raise ValueError(f"vehicle description sensors.{name}.track: bottom or water, got {track!r}")
        if track == "water" and name != "dvl":
            raise ValueError(f"vehicle description sensors.{name}.track: only a DVL has a track (bottom or water)")
        noise, grade, rng = m.get("noise"), m.get("grade"), m.get("range_m")
        if noise is not None and not _positive(noise):
            raise ValueError(f"vehicle description sensors.{name}.noise: one reading's standard deviation, a positive "
                             f"number (m for position and depth, m/s for a DVL), got {noise!r}")
        if rng is not None and not _positive(rng):
            raise ValueError(f"vehicle description sensors.{name}.range_m: the typical range to the positioning "
                             f"system's transceiver, a positive number of metres, got {rng!r}")
        if grade is not None:
            grades = grades or sensor_grades()
            if grade not in grades:
                raise ValueError(f"vehicle description sensors.{name}.grade: {grade!r} isn't a sensor grade. Known: "
                                 f"{', '.join(sorted(grades))} (calibration/sensor_grades.yaml). For another sensor, "
                                 f"give its noise: (one reading's standard deviation) instead")
            if "noise_fraction_of_range" in grades[grade] and rng is None and noise is None:
                raise ValueError(f"vehicle description sensors.{name}: grade {grade}'s accuracy is a share of the "
                                 f"range, so give range_m: (the typical distance to its transceiver, m)")
        delay = m.get("delay_s")
        if delay is not None and not (isinstance(delay, (int, float)) and not isinstance(delay, bool)
                                      and np.isfinite(delay) and delay >= 0):
            raise ValueError(f"vehicle description sensors.{name}.delay_s: how long after the vehicle was there the "
                             f"sensor reports, a number of seconds (0 or more), got {delay!r}")
        out[name] = {"position_m": np.asarray(pos, dtype=float), "track": track, "grade": grade, "noise": noise,
                     "range_m": rng, "delay_s": None if delay is None else float(delay)}
    return out


def given_delays(data_dir: Optional[str], mounts: Optional[dict]) -> dict:
    """{position sensor: (delay in s, where it was said)}: the delays the lab gave, which calibration takes as said
    instead of looking for one. The vehicle description's sensors.<name>.delay_s is taken off the fixes here; the
    ingest spec's position.delay_s was taken off at ingest, so it counts as 0 more."""
    import json
    out = {}
    path = os.path.join(data_dir or "", "ingest_report.json")
    if data_dir and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                sens = (json.load(f).get("sensors") or {}).get("sensors") or {}
        except (OSError, ValueError):
            sens = {}
        out.update({name: (0.0, f"the ingest spec's delay_s ({ch['delay_s']:g} s, already taken off)")
                    for name, ch in sens.items() if ch.get("kind") == "position" and "delay_s" in ch})
    for name, m in (mounts or {}).items():
        if m.get("delay_s") is not None:
            if name in out:
                raise ValueError(f"{name}'s delay_s is given in both the ingest spec and the vehicle description: "
                                 f"give it once (the ingest spec's is already taken off the fixes)")
            out[name] = (m["delay_s"], f"the vehicle description's sensors.{name}.delay_s")
    return out


def measured_noise(data_dir: Optional[str]) -> dict:
    """{sensor name: reading std} from the ingest report's own measurement, for sensors whose readings weren't smoothed
    by the logging system (a smoothed sensor's measured noise is too low to trust)."""
    import json
    path = os.path.join(data_dir or "", "ingest_report.json")
    if not data_dir or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            sens = (json.load(f).get("sensors") or {}).get("sensors") or {}
    except (OSError, ValueError):
        return {}
    return {name: float(max(ch["noise_std"])) for name, ch in sens.items()
            if ch.get("noise_std") and not ch.get("filtered")}


def smoother_noise(data_dir: Optional[str], mounts: Optional[dict], runs: Sequence) -> tuple:
    """Each sensor's reading noise for the state smoother, and where it came from: the description's noise:, the
    noise ingestion measured (unsmoothed readings), the description's grade: (sensor_grades.yaml), else the smoother's
    default for its kind. Returns ({name: std}, {name: source}). A grade of the wrong kind is refused."""
    kinds = {}
    for r in runs:
        for ch in (getattr(r, "channels", None) or []):
            if ch.kind in NOISE_KINDS:
                kinds.setdefault(ch.name, ch.kind)
    noise, source = {}, {}
    grades = None
    for name, m in (mounts or {}).items():
        if not isinstance(m, dict) or m.get("noise") is not None or not m.get("grade"):
            continue
        grades = grades or sensor_grades()
        g = grades[m["grade"]]
        if name in kinds and kinds[name] != g["kind"]:
            raise ValueError(f"vehicle description sensors.{name}.grade: {m['grade']} is a {g['kind']} sensor, but "
                             f"{name} in your logs is a {kinds[name]} sensor")
        noise[name] = float(g["noise"]) if "noise" in g else float(g["noise_fraction_of_range"]) * float(m["range_m"])
        source[name] = f"grade {m['grade']}"
    for name, v in measured_noise(data_dir).items():
        noise[name], source[name] = v, "measured from your logs"
    for name, m in (mounts or {}).items():
        if isinstance(m, dict) and m.get("noise") is not None:
            noise[name], source[name] = float(m["noise"]), "your description"
    for name in kinds:
        source.setdefault(name, "default")
    return noise, source


def channels_from_frame(df, use: Sequence[str] = ("auto",), use_accelerometer: bool = False,
                        lever_arms: Optional[dict] = None) -> list:
    """A canonical run's sensor channels. use: "auto"/"all" (every sensor in the file), or names to keep
    (dvl, a position or depth sensor's name, acc); the gyro is always kept. The accelerometer only with
    use_accelerometer (or named). lever_arms: {name: (3,)} or sensor_mounts()' {name: {position_m, track}}."""
    lever_arms = lever_arms or {}
    tracks = {k: v.get("track", "bottom") for k, v in lever_arms.items() if isinstance(v, dict)}
    lever_arms = {k: (v["position_m"] if isinstance(v, dict) else v) for k, v in lever_arms.items()}
    pick = set(use)
    everything = bool(pick & {"auto", "all"})

    def wanted(name):
        return everything or name in pick

    out = [Channel("gyro", "gyro", ("p", "q", "r"), df[["Bp", "Bq", "Br"]].to_numpy(dtype=float))]
    if {"Bu", "Bv", "Bw"} <= set(df.columns) and wanted("dvl"):
        v = df[["Bu", "Bv", "Bw"]].to_numpy(dtype=float)
        if np.isfinite(v).any():
            out.append(Channel("dvl", "dvl", ("u", "v", "w"), v, _arm(lever_arms.get("dvl")),
                               tracks.get("dvl", "bottom")))
    for c in sorted(df.columns):
        if c.startswith("pos_") and c.endswith("_n"):
            name = c[4:-2]
            if not wanted(name):
                continue
            cols = [f"pos_{name}_n", f"pos_{name}_e"] + ([f"pos_{name}_d"] if f"pos_{name}_d" in df.columns else [])
            out.append(Channel("position", name, tuple(x[-1] for x in cols), df[cols].to_numpy(dtype=float),
                               _arm(lever_arms.get(name))))
        elif c.startswith("depth_"):
            name = c[6:]
            if wanted(name):
                out.append(Channel("depth", name, ("d",), df[[c]].to_numpy(dtype=float), _arm(lever_arms.get(name))))
    if {"acc_x", "acc_y", "acc_z"} <= set(df.columns) and (use_accelerometer or "acc" in pick):
        out.append(Channel("accelerometer", "acc", ("ax", "ay", "az"),
                           df[["acc_x", "acc_y", "acc_z"]].to_numpy(dtype=float), _arm(lever_arms.get("acc"))))
    return out


def _arm(x):
    if x is None:
        return None
    a = np.asarray(x, dtype=float).reshape(3)
    return a if np.any(a) else None


def channel_labels(channels: Sequence[Channel]) -> list:
    return [lab for ch in channels for lab in ch.labels]


def slice_run(run, a: int, b: int):
    """Samples a..b-1 of a run, its sensor channels and window starts with it."""
    from dataclasses import replace
    out = replace(run, forces_torques=run.forces_torques[a:b], roll_pitch=run.roll_pitch[a:b],
                  real_nu=run.real_nu[a:b], nu0=np.nan_to_num(run.real_nu[a]).copy(),
                  current_body=None if run.current_body is None else run.current_body[a:b],
                  orientation_quat=None if run.orientation_quat is None else run.orientation_quat[a:b],
                  channels=None if run.channels is None else [replace(ch, values=ch.values[a:b]) for ch in run.channels],
                  anchor_lin=None if run.anchor_lin is None else run.anchor_lin[a:b])
    return out


# ---------------------------------------------------------------------------- the plan (fixed per run)

def build_plan(run, horizon_samples: Optional[int]) -> Plan:
    from objective import horizon_windows
    from scipy.spatial.transform import Rotation
    n = run.n_samples
    windows = horizon_windows(n, horizon_samples)
    R = Rotation.from_quat(run.orientation_quat).as_matrix()
    picks, still_parts = [], [[] for _ in channel_labels(run.channels)]
    counts = np.zeros(len(still_parts), dtype=int)
    for a, b in windows:
        pw = {}
        col = 0
        for ci, ch in enumerate(run.channels):
            k = len(ch.axes)
            seg = ch.values[a:b]
            valid = np.all(np.isfinite(seg), axis=1)
            if ch.kind == "accelerometer" and len(valid):
                # the predicted acceleration is a centred difference of the simulated velocity: the window's first and
                # last samples have only one side
                valid[0] = valid[-1] = False
            if ch.kind in ("gyro", "dvl"):
                valid[0] = False          # the window's start is the anchor itself
                idx = np.flatnonzero(valid) + a
                if len(idx):
                    pw[ci] = idx
                    for j in range(k):
                        still_parts[col + j].append(-ch.values[idx, j])
            else:
                idx = np.flatnonzero(valid) + a
                if len(idx) >= MIN_FIXES_PER_WINDOW:
                    pw[ci] = idx
                    vals = ch.values[idx]
                    for j in range(k):
                        still_parts[col + j].append(-(vals[:, j] - vals[:, j].mean()))
            if ci in pw:
                counts[col:col + k] += len(pw[ci])
            col += k
        picks.append(pw)
    still = [np.concatenate(p) if p else np.empty(0) for p in still_parts]
    return Plan(windows=windows, picks=picks, counts=counts, still=still, R=R)


def plan_for(run, horizon_samples: Optional[int]) -> Plan:
    cache = run.__dict__.setdefault("_plans", {})
    if horizon_samples not in cache:
        cache[horizon_samples] = build_plan(run, horizon_samples)
    return cache[horizon_samples]


# ---------------------------------------------------------------------------- predicted readings

def _window_nu(params, run, vehicle, a, b, simulate_fn):
    from dynamics_model import SimulationConfig, TrajectoryInputs, simulate_trajectory
    nu0 = np.r_[run.anchor_lin[a], run.real_nu[a, 3:6]]
    traj = TrajectoryInputs(nu0=nu0, forces_torques=run.forces_torques[a:b], roll_pitch_series=run.roll_pitch[a:b],
                            current_body_series=None if run.current_body is None else run.current_body[a:b])
    sim = simulate_fn if simulate_fn is not None else simulate_trajectory
    nu = sim(params, vehicle, SimulationConfig(dt=run.dt), traj).nu
    if not np.all(np.isfinite(nu)):
        raise FloatingPointError("non-finite simulated velocity")
    return nu


def path_in_window(nu: np.ndarray, R: np.ndarray, dt: float) -> np.ndarray:
    """World displacement from the window's start, (n, 3): the body velocity turned by the logged orientation, summed
    sample by sample (validation.path_from_velocity's rule)."""
    vw = np.einsum("nij,nj->ni", R, nu[:, :3])
    return np.vstack([np.zeros(3), np.cumsum(vw[:-1] * dt, axis=0)])


def predicted(ch: Channel, nu: np.ndarray, Rw: np.ndarray, path: np.ndarray, dt: float, rel: np.ndarray,
              current_body: Optional[np.ndarray] = None) -> np.ndarray:
    """What channel `ch` reads at window-relative samples `rel`, for simulated velocity nu over the window (velocity
    over ground, dynamics_model._relative_to_water). current_body: the window's water current in body axes, for a
    water-tracking DVL."""
    if ch.kind == "gyro":
        return nu[rel, 3:6]
    if ch.kind == "dvl":
        v = nu[rel, :3]
        if ch.lever_arm_m is not None:
            v = v + np.cross(nu[rel, 3:6], ch.lever_arm_m)
        if ch.track == "water" and current_body is not None:
            v = v - current_body[rel]   # through the water: over ground minus the current
        return v
    if ch.kind in ("position", "depth"):
        p = path[rel]
        if ch.lever_arm_m is not None:
            p = p + np.einsum("nij,j->ni", Rw[rel], ch.lever_arm_m)
        return p[:, 2:3] if ch.kind == "depth" else p[:, :len(ch.axes)]
    if ch.kind == "accelerometer":
        vdot = np.gradient(nu[:, :3], dt, axis=0)
        f = vdot + np.cross(nu[:, 3:6], nu[:, :3]) + np.einsum("nji,j->ni", Rw, np.array([0.0, 0.0, -G]))
        return f[rel]
    raise ValueError(f"unknown channel kind {ch.kind!r}")


def raw_channel_residuals(params, run, vehicle, horizon_samples: Optional[int] = None, simulate_fn=None) -> list:
    """Per channel-axis, (predicted - measured) at every reading that counts, in the channel's units; position, depth
    and accelerometer residuals taken about their window mean. Raises FloatingPointError if the simulation blows up."""
    from objective import with_thruster_lag
    run_l = with_thruster_lag(run, params)
    plan = plan_for(run, horizon_samples)
    labels = channel_labels(run.channels)
    parts = [[] for _ in labels]
    for (a, b), pw in zip(plan.windows, plan.picks):
        if not pw:
            continue
        nu = _window_nu(params, run_l, vehicle, a, b, simulate_fn)
        Rw = plan.R[a:b]
        path = path_in_window(nu, Rw, run.dt) if any(run.channels[ci].kind in ("position", "depth")
                                                      for ci in pw) else None
        col = 0
        for ci, ch in enumerate(run.channels):
            k = len(ch.axes)
            if ci in pw:
                idx = pw[ci]
                cb = None if run.current_body is None else run.current_body[a:b]
                r = predicted(ch, nu, Rw, path, run.dt, idx - a, cb) - ch.values[idx]
                if ch.kind in ("position", "depth", "accelerometer"):
                    r = r - r.mean(axis=0)
                for j in range(k):
                    parts[col + j].append(r[:, j])
            col += k
    return [np.concatenate(p) if p else np.empty(0) for p in parts]


# ---------------------------------------------------------------------------- scales, the objective's vector, skill

def still_scales(runs: Sequence, horizon_samples: Optional[int]) -> np.ndarray:
    """Per channel-axis: the RMS error of predicting no motion, pooled over the (training) runs, by label. Runs may
    carry different channels (a GPS only in some recordings): each label pools the runs that have it."""
    pooled: dict = {}
    for run in runs:
        plan = plan_for(run, horizon_samples)
        for lab, s in zip(channel_labels(run.channels), plan.still):
            pooled.setdefault(lab, []).append(s)
    return {lab: max(float(np.sqrt(np.mean(np.concatenate(v) ** 2))) if sum(len(x) for x in v) else 0.0,
                     SCALE_EPSILON[_kind_of(lab, runs)]) for lab, v in pooled.items()}


def _kind_of(label: str, runs) -> str:
    for run in runs:
        for ch in run.channels:
            if label in ch.labels:
                return ch.kind
    return "gyro"


def residual_vector(params, run, vehicle, horizon_samples: Optional[int] = None, simulate_fn=None,
                    zero_weight: Sequence[str] = ()) -> np.ndarray:
    """The optimizer's residuals for one run: each channel-axis divided by its still RMS (run.sensor_scales) and the
    square root of its number of readings. zero_weight: labels left out (an excluded gyro axis, e.g. gyro.p). If the
    simulation blows up, the still residuals x objective.DIVERGED_PENALTY in the same shape (a very bad fit, not a
    crash, as for the velocity objective)."""
    from objective import DIVERGED_PENALTY
    plan = plan_for(run, horizon_samples)
    labels = channel_labels(run.channels)
    try:
        parts = raw_channel_residuals(params, run, vehicle, horizon_samples, simulate_fn)
    except (FloatingPointError, ValueError) as e:
        if isinstance(e, ValueError) and "Non-finite" not in str(e) and "non-finite" not in str(e):
            raise
        parts = [DIVERGED_PENALTY * s for s in plan.still]
    out = []
    for lab, r in zip(labels, parts):
        if lab in zero_weight or not len(r):
            continue
        out.append(r / run.sensor_scales[lab] / np.sqrt(len(r)))
    return np.concatenate(out) if out else np.zeros(0)


def channel_rmse(params, run, vehicle, horizon_samples: Optional[int] = None, simulate_fn=None) -> dict:
    """{label: RMSE in the channel's units} for one run (NaN for a channel-axis with no reading that counts)."""
    parts = raw_channel_residuals(params, run, vehicle, horizon_samples, simulate_fn)
    return {lab: (float(np.sqrt(np.mean(r ** 2))) if len(r) else float("nan"))
            for lab, r in zip(channel_labels(run.channels), parts)}


def still_rmse(run, horizon_samples: Optional[int] = None) -> dict:
    plan = plan_for(run, horizon_samples)
    return {lab: (float(np.sqrt(np.mean(s ** 2))) if len(s) else float("nan"))
            for lab, s in zip(channel_labels(run.channels), plan.still)}


def velocity_view(params, run, vehicle, horizon_samples: Optional[int] = None, simulate_fn=None) -> np.ndarray:
    """The original (simulated - logged) body-velocity residuals, (M, 6), for the samples the gyro channel counts:
    u, v, w from the DVL (NaN where it didn't read, or with no DVL), p, q, r from the gyro. What objective.py's
    per-axis RMSE reports read for a run with channels."""
    from objective import with_thruster_lag
    run_l = with_thruster_lag(run, params)
    plan = plan_for(run, horizon_samples)
    rows = []
    for a, b in plan.windows:
        if b - a < 2:
            continue
        nu = _window_nu(params, run_l, vehicle, a, b, simulate_fn)
        lin = run.real_nu[a + 1:b, :3]
        dvl = next((ch for ch in run.channels if ch.kind == "dvl"), None)
        sim_lin = nu[1:, :3] if dvl is None else predicted(
            dvl, nu, plan.R[a:b], None, run.dt, np.arange(1, b - a),
            None if run.current_body is None else run.current_body[a:b])   # as the DVL would read it
        res = np.hstack([sim_lin - lin, nu[1:, 3:6] - run.real_nu[a + 1:b, 3:6]])
        rows.append(res)
    return np.vstack(rows) if rows else np.zeros((0, 6))


def summarise(per_run: Sequence[dict]) -> dict:
    """Average per-run {label: value} dicts over the runs that have each label (NaN-aware)."""
    labels = list(dict.fromkeys(lab for d in per_run for lab in d))
    out = {}
    for lab in labels:
        vals = [d[lab] for d in per_run if lab in d and np.isfinite(d[lab])]
        out[lab] = float(np.mean(vals)) if vals else float("nan")
    return out


# ---------------------------------------------------------------------------- what the sensors can calibrate

LINEAR_AXIS_SEEN_BY = {"u": ("dvl.u", "position.n", "position.e"), "v": ("dvl.v", "position.n", "position.e"),
                       "w": ("dvl.w", "position.d", "depth.d")}
MIN_WINDOWS_TO_SEE = 3       # a channel sees an axis with readings that count in at least this many windows...
MIN_READINGS_TO_SEE = 30     # ...and this many readings in all

# parameters that only matter through an axis's own motion: held at the vehicle description's values when no sensor
# sees that axis (run_calibration.EXCLUDABLE_AXIS_PARAMS does the same for roll and pitch with --exclude-axes)
LINEAR_AXIS_PARAMS = {"u": ["Xu", "Xuu", "Xu_dot", "Xq_dot", "Muq"],
                      "v": ["Yv", "Yvv", "Yv_dot"],
                      "w": ["Zw", "Zww", "Zw_dot", "net_buoyancy_z", "Zq_dot"]}
COUPLED_PARAMS = {("u", "w"): ["Zuw", "Muw"]}


def axes_seen(runs: Sequence, horizon_samples: Optional[int]) -> dict:
    """{linear axis: [the channels that see it]} over the (training) runs. A horizontal fix sees surge and sway;
    a channel counts once it has at least MIN_WINDOWS_TO_SEE windows with the readings it needs."""
    windows: dict = {}
    readings: dict = {}
    for run in runs:
        plan = plan_for(run, horizon_samples)
        for pw in plan.picks:
            for ci, idx in pw.items():
                ch = run.channels[ci]
                for a in ch.axes:
                    windows[(ch.kind, a)] = windows.get((ch.kind, a), 0) + 1
                    readings[(ch.kind, a)] = readings.get((ch.kind, a), 0) + len(idx)
    seen = {}
    for axis, by in LINEAR_AXIS_SEEN_BY.items():
        seen[axis] = sorted({lab.split(".")[0] for lab in by
                             if windows.get(tuple(lab.split(".")), 0) >= MIN_WINDOWS_TO_SEE
                             and readings.get(tuple(lab.split(".")), 0) >= MIN_READINGS_TO_SEE})
    return seen


def unobservable_holds(seen: dict, default_params) -> dict:
    """{parameter: held value} for the linear axes no sensor sees."""
    blind = {a for a, by in seen.items() if not by}
    held = {}
    for a in sorted(blind):
        for name in LINEAR_AXIS_PARAMS[a]:
            if hasattr(default_params, name):
                held[name] = float(getattr(default_params, name))
    for axes, names in COUPLED_PARAMS.items():
        if blind & set(axes):
            for name in names:
                if hasattr(default_params, name):
                    held[name] = float(getattr(default_params, name))
    return held


AXIS_WORDS = {"u": "surge", "v": "sway", "w": "heave"}
SENSOR_WORDS = {"dvl": "the DVL", "position": "position fixes", "depth": "depth"}


def seen_words(seen: dict) -> list:
    """Plain sentences: which linear axes were calibrated and from what, which weren't."""
    out = []
    for a, by in seen.items():
        if by:
            out.append(f"{AXIS_WORDS[a].capitalize()}: calibrated from {' and '.join(SENSOR_WORDS[b] for b in by)}.")
        else:
            out.append(f"{AXIS_WORDS[a].capitalize()}: not calibrated. Your logs have no "
                       f"{'DVL or horizontal position fixes' if a in 'uv' else 'DVL, depth sensor or vertical position'}"
                       f" for it, so its values are your vehicle description's.")
    return out
