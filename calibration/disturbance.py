"""
disturbance.py -- random roll/pitch torques shaped like the real motion the
logged thrusters don't explain (PROJECT_STATUS.md #50-51).

Why: in the reference BlueROV2 dataset, roll and pitch swing by ~10-14
degrees, but the logged thruster torques barely correlate with that motion
(|corr| <= 0.11; yaw, by contrast, tracks its torque at 0.69). Something
unlogged -- tether pull, waves, contact -- drives it. A simulator without it
has a vehicle that sits unrealistically level, and policies trained there
never see the wobble a real vehicle has.

Model: each axis gets an independent Ornstein-Uhlenbeck torque (first-order
coloured noise) with stationary standard deviation sigma_nm and correlation
time tau_s, added to the commanded roll (Tx) and pitch (Ty) torque:

    d_{k+1} = a d_k + sigma * sqrt(1 - a^2) * xi_k,   a = exp(-dt / tau)

sampled once per control tick and held over the physics sub-steps, exactly
like the commanded thrust (zero-order hold), so the stand-in and MuJoCo
engines see the same torque. sigma and tau are fitted per parameter set by
fit_rollpitch_disturbance.py so that the SIMULATED roll/pitch statistics
(rate spread, rate autocorrelation, angle spread) match the real logs; the
fitted values are only meaningful together with the damping/restoring
parameters they were fitted for, and the YAML records which those were.

Seeding: disturbances never depend on the vehicles' actions and are seeded
from the episode's identity, so paired comparisons (the same start state and
comm seed under different belief models) see identical disturbances.
"""
from __future__ import annotations

import hashlib
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import yaml

DEFAULT_FIT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "disturbances",
                                "bluerov2_rollpitch.yaml")
AXIS_INDEX = {"roll": 3, "pitch": 4}   # positions in the body wrench [Fx, Fy, Fz, Tx, Ty, Tz]


@dataclass
class AxisDisturbance:
    sigma_nm: float     # stationary standard deviation of the torque (N*m)
    tau_s: float        # correlation time (s)

    def validate(self, name: str) -> None:
        if not (self.sigma_nm >= 0 and math.isfinite(self.sigma_nm)):
            raise ValueError(f"{name}.sigma_nm must be finite and >= 0, got {self.sigma_nm}")
        if not (self.tau_s > 0 and math.isfinite(self.tau_s)):
            raise ValueError(f"{name}.tau_s must be finite and > 0, got {self.tau_s}")


@dataclass
class DisturbanceConfig:
    """Roll/pitch disturbance for the fleet simulation. scale multiplies both
    sigmas (0 = off, 1 = as fitted, 2 = twice as rough)."""
    roll: AxisDisturbance
    pitch: AxisDisturbance
    scale: float = 1.0
    source: str = ""                         # where the numbers came from (for reports)
    fitted_for_params_sha: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        self.roll.validate("roll")
        self.pitch.validate("pitch")
        if not (self.scale >= 0 and math.isfinite(self.scale)):
            raise ValueError(f"disturbance scale must be finite and >= 0, got {self.scale}")

    @property
    def active(self) -> bool:
        return self.scale > 0 and (self.roll.sigma_nm > 0 or self.pitch.sigma_nm > 0)


def params_sha(params_path: str) -> str:
    """Short content hash of a calibrated_parameters YAML's parameter values."""
    with open(params_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    p = raw.get("parameters", raw)
    canon = ",".join(f"{k}={float(p[k]):.10g}" for k in sorted(p))
    return hashlib.sha256(canon.encode()).hexdigest()[:12]


def load_disturbance(path: str = DEFAULT_FIT_PATH, scale: float = 1.0,
                     truth_params_path: Optional[str] = None) -> DisturbanceConfig:
    """Load a fitted disturbance YAML. Warns if it was fitted for a different
    parameter set than truth_params_path (its sigmas then won't reproduce the
    real motion statistics; re-run fit_rollpitch_disturbance.py)."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    try:
        cfg = DisturbanceConfig(roll=AxisDisturbance(**raw["roll"]), pitch=AxisDisturbance(**raw["pitch"]),
                                scale=scale, source=os.path.basename(path),
                                fitted_for_params_sha=raw.get("fitted_for", {}).get("params_sha"),
                                extra={k: v for k, v in raw.items() if k not in ("roll", "pitch")})
    except (KeyError, TypeError) as e:
        raise ValueError(f"{path!r} is not a roll/pitch disturbance file (needs roll/pitch sigma_nm and tau_s): {e}")
    cfg.validate()
    if truth_params_path and cfg.fitted_for_params_sha and params_sha(truth_params_path) != cfg.fitted_for_params_sha:
        warnings.warn(f"{os.path.basename(path)} was fitted for a different parameter set than "
                      f"{os.path.basename(truth_params_path)}; its roll/pitch statistics won't match the real logs. "
                      f"Re-fit with fit_rollpitch_disturbance.py --params {truth_params_path}", stacklevel=2)
    return cfg


class RollPitchDisturbance:
    """Per-vehicle torque generator. next() returns the (6,) body wrench to add
    for the coming control tick."""

    def __init__(self, cfg: DisturbanceConfig, dt: float, seed):
        cfg.validate()
        self.cfg, self.dt = cfg, dt
        self._rng = np.random.default_rng(seed)
        self._a = {ax: math.exp(-dt / getattr(cfg, ax).tau_s) for ax in AXIS_INDEX}
        self._sig = {ax: getattr(cfg, ax).sigma_nm * cfg.scale for ax in AXIS_INDEX}
        # start from the stationary distribution: no warm-up transient at t=0
        self._d = {ax: self._sig[ax] * float(self._rng.standard_normal()) for ax in AXIS_INDEX}

    def next(self) -> np.ndarray:
        w = np.zeros(6)
        for ax, idx in AXIS_INDEX.items():
            w[idx] = self._d[ax]
            a = self._a[ax]
            self._d[ax] = a * self._d[ax] + self._sig[ax] * math.sqrt(1.0 - a * a) * float(self._rng.standard_normal())
        return w


def episode_seed(comm_seed: int, agent: int, initial_separation: float, closing_speed: float,
                 salt: Sequence[int] = ()) -> np.random.SeedSequence:
    """Deterministic per-episode, per-vehicle seed that doesn't depend on the
    belief model or the policy, so paired conditions share disturbances."""
    return np.random.SeedSequence([int(comm_seed) & 0xFFFFFFFF, int(agent), 0x5EED,
                                   int(round(initial_separation * 1e6)) & 0xFFFFFFFF,
                                   int(round((closing_speed + 100.0) * 1e6)) & 0xFFFFFFFF, *salt])
