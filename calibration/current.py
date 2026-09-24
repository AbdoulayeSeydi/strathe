"""
current.py -- configurable water current for open-water simulation
(PROJECT_STATUS.md #55).

Strathe's reference data comes from a tank, so the simulation used to assume
still water. Open-water labs need a current, and how much of one is their
call, so everything here is optional and configurable:

    current:                       # in a fleet experiment YAML (default: none = still water)
      speed_mps: 0.2               # horizontal speed, or [0.0, 0.3] to draw one per episode
      direction_deg: 90            # direction the water flows TOWARD, degrees from the world
                                   # x axis toward +y (NED: 0 = north, 90 = east), or [0, 360]
      vertical_mps: 0.0            # + = downward flow (NED)
      variability:                 # optional slow fluctuation around the episode's mean
        speed_std_mps: 0.05
        direction_std_deg: 10
        time_scale_s: 30
      known_to_beliefs: true       # physics teammate models account for the episode's mean current

Physics: the calibrated hydrodynamic forces act on the velocity relative to
the water (dynamics_model._relative_to_water, Fossen 2011 Property 10.1);
position still integrates velocity over ground. The current is uniform in
space (both vehicles feel the same water), and is updated once per control
tick (zero-order hold), like commanded thrust.

The fluctuation is an Ornstein-Uhlenbeck process on speed and on direction,
started from its stationary distribution. Its derivative is not fed into the
dynamics (the physics assumes a locally constant current), which is accurate
for fluctuations slower than a few seconds; keep time_scale_s at several
seconds or more.

Seeding: a current depends only on the episode's identity (comm seed and
start state), never on actions or the belief model, so paired comparisons
stay paired.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple, Union

import numpy as np

Range = Union[float, Tuple[float, float]]


def _as_range(x, name: str) -> Tuple[float, float]:
    if isinstance(x, (int, float)):
        return float(x), float(x)
    if isinstance(x, (list, tuple)) and len(x) == 2:
        lo, hi = float(x[0]), float(x[1])
        if lo > hi:
            raise ValueError(f"{name} range must be [min, max], got {list(x)}")
        return lo, hi
    raise ValueError(f"{name} must be a number or a [min, max] pair, got {x!r}")


@dataclass
class CurrentConfig:
    speed_mps: Range = 0.0
    direction_deg: Range = 0.0
    vertical_mps: float = 0.0
    speed_std_mps: float = 0.0
    direction_std_deg: float = 0.0
    time_scale_s: float = 30.0
    known_to_beliefs: bool = True

    def validate(self) -> None:
        lo, hi = _as_range(self.speed_mps, "speed_mps")
        if lo < 0:
            raise ValueError(f"speed_mps must be >= 0, got {self.speed_mps}")
        if hi > 3.0:
            raise ValueError(f"speed_mps above 3 m/s ({hi}) is beyond anything a small hovering AUV can work in; "
                             f"check the units")
        _as_range(self.direction_deg, "direction_deg")
        for name in ("vertical_mps", "speed_std_mps", "direction_std_deg", "time_scale_s"):
            v = getattr(self, name)
            if not math.isfinite(v):
                raise ValueError(f"{name} must be finite, got {v}")
        if abs(self.vertical_mps) > 1.0:
            raise ValueError(f"vertical_mps above 1 m/s ({self.vertical_mps}); check the units")
        if self.speed_std_mps < 0 or self.direction_std_deg < 0:
            raise ValueError("variability standard deviations must be >= 0")
        if self.time_scale_s < 1.0:
            raise ValueError(f"time_scale_s must be >= 1 s (the physics assumes a slowly varying current), "
                             f"got {self.time_scale_s}")

    @property
    def active(self) -> bool:
        return (_as_range(self.speed_mps, "speed_mps")[1] > 0 or self.vertical_mps != 0 or self.speed_std_mps > 0)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "CurrentConfig":
        d = dict(d or {})
        var = d.pop("variability", None) or {}
        unknown = set(var) - {"speed_std_mps", "direction_std_deg", "time_scale_s"}
        if unknown:
            raise ValueError(f"unknown current.variability keys: {sorted(unknown)}")
        cfg = cls(**d, **var)
        cfg.validate()
        return cfg

    def describe(self) -> str:
        s_lo, s_hi = _as_range(self.speed_mps, "speed_mps")
        d_lo, d_hi = _as_range(self.direction_deg, "direction_deg")
        speed = f"{s_lo:g} m/s" if s_lo == s_hi else f"{s_lo:g}-{s_hi:g} m/s (drawn per episode)"
        heading = f"toward {d_lo:g} deg" if d_lo == d_hi else f"toward {d_lo:g}-{d_hi:g} deg (drawn per episode)"
        parts = [f"{speed} {heading}"]
        if self.vertical_mps:
            parts.append(f"vertical {self.vertical_mps:+g} m/s")
        if self.speed_std_mps or self.direction_std_deg:
            parts.append(f"fluctuating by {self.speed_std_mps:g} m/s and {self.direction_std_deg:g} deg over "
                         f"~{self.time_scale_s:g} s")
        parts.append("known to physics teammate models" if self.known_to_beliefs
                     else "unknown to teammate models")
        return ", ".join(parts)


class WaterCurrent:
    """One episode's current. mean_world is fixed for the episode; next()
    returns the (fluctuating) world-frame current for the coming tick."""

    def __init__(self, cfg: CurrentConfig, dt: float, seed):
        cfg.validate()
        self.cfg, self.dt = cfg, dt
        rng = np.random.default_rng(seed)
        self._rng = rng
        s_lo, s_hi = _as_range(cfg.speed_mps, "speed_mps")
        d_lo, d_hi = _as_range(cfg.direction_deg, "direction_deg")
        self.mean_speed = float(rng.uniform(s_lo, s_hi)) if s_hi > s_lo else s_lo
        self.mean_direction_deg = float(rng.uniform(d_lo, d_hi)) if d_hi > d_lo else d_lo
        self._a = math.exp(-dt / cfg.time_scale_s)
        # fluctuations start from their stationary distribution (no warm-up transient)
        self._ds = cfg.speed_std_mps * float(rng.standard_normal())
        self._dd = cfg.direction_std_deg * float(rng.standard_normal())

    @staticmethod
    def _vector(speed: float, direction_deg: float, vertical: float) -> np.ndarray:
        th = math.radians(direction_deg)
        return np.array([speed * math.cos(th), speed * math.sin(th), vertical])

    @property
    def mean_world(self) -> np.ndarray:
        return self._vector(self.mean_speed, self.mean_direction_deg, self.cfg.vertical_mps)

    def next(self) -> np.ndarray:
        cfg = self.cfg
        out = self._vector(max(0.0, self.mean_speed + self._ds), self.mean_direction_deg + self._dd,
                           cfg.vertical_mps)
        a, k = self._a, math.sqrt(1.0 - self._a * self._a)
        if cfg.speed_std_mps:
            self._ds = a * self._ds + cfg.speed_std_mps * k * float(self._rng.standard_normal())
        if cfg.direction_std_deg:
            self._dd = a * self._dd + cfg.direction_std_deg * k * float(self._rng.standard_normal())
        return out


def current_from_spec(loaded_spec) -> Optional[np.ndarray]:
    """vehicle_spec.yaml's environment block as a world-frame current vector
    for the CALIBRATION logs, or None for still water (the tank default)."""
    env = (loaded_spec.raw.get("environment") or {})
    speed = float(env.get("current_velocity_mps", 0.0) or 0.0)
    vertical = float(env.get("current_vertical_mps", 0.0) or 0.0)
    if speed == 0 and vertical == 0:
        return None
    return WaterCurrent._vector(speed, float(env.get("current_direction_deg", 0.0) or 0.0), vertical)
