"""
sensor_noise.py -- what each vehicle's own sensors report, instead of the
simulator's exact state (the "noisy" half of clean vs noisy simulation).

Why: a policy trained on exact readings never sees what a real vehicle's
sensors do: a DVL velocity that jitters, drops out and is slightly biased, a
gyro with an offset, an attitude and compass heading that are a little off, a
depth sensor with noise, and a position that is dead-reckoned from all of
that and so drifts. Without it, a simulator result can be better than any
real vehicle could manage.

What changes when sensor noise is on (it is off by default):
  - The POLICY sees measured values: its own velocity (DVL), body rates
    (gyro), roll and pitch (attitude), and target or teammate positions
    relative to where it THINKS it is, in the axes it thinks it points along.
  - BEACONS (fleet) carry what the sender measured and believes, not the
    truth; a teammate's prediction starts from that.
  - The TASK is still judged on the truth: success, collisions and belief
    errors are measured against the real positions.

Model, per vehicle and per 0.1 s tick (every size is the standard deviation
of one sample; `scale` multiplies them all):
    velocity (u, v, w)  true + per-episode bias + white noise; with
                        probability velocity_dropout_prob the DVL gives no
                        reading that tick and the last one is held
    rates (p, q, r)     true + per-episode bias + white noise
    roll, pitch         true + per-episode bias + white noise
    heading             true + per-episode bias + white noise
    depth               true + white noise
    position            dead_reckoning: horizontal position integrated from
                        the measured velocity and heading, starting from a fix
                        with error position_std_m, so velocity and heading
                        errors accumulate into drift; depth from the depth
                        sensor. external: an external positioning system (a
                        tank's motion capture, USBL) -- white noise of
                        position_std_m, no drift.

Presets (sensor_noise/*.yaml):
    tank     fitted to the reference BlueROV2 tank logs (fit_sensor_noise.py):
             the high-frequency noise those logs actually contain. The tank's
             positioning is external, so there is no drift. Small.
    typical  a DVL + AHRS + depth sensor vehicle navigating by dead
             reckoning. These sizes are stated ASSUMPTIONS of common
             magnitude, not a fit; fit your own vehicle's logs with
             fit_sensor_noise.py and pass that file instead.

Seeding: the noise depends only on the episode's identity, never on the
policy's actions or the belief model, and every tick draws the same number
of random numbers, so paired comparisons (clean vs noisy, calibrated vs
generic) stay paired.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, fields
from typing import Optional

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
PRESET_DIR = os.path.join(HERE, "sensor_noise")
PRESETS = {"tank": os.path.join(PRESET_DIR, "bluerov2_tank.yaml"),
           "typical": os.path.join(PRESET_DIR, "typical_dvl_ahrs.yaml")}
POSITION_MODES = ("dead_reckoning", "external")
SIZE_FIELDS = ("velocity_std_mps", "velocity_bias_mps", "gyro_std_rps", "gyro_bias_rps", "attitude_std_rad",
               "attitude_bias_rad", "heading_std_rad", "heading_bias_rad", "depth_std_m", "position_std_m")
# Sanity ceilings: well past any working sensor; a value above one is almost always a units mistake
# (degrees for radians, cm for m).
CEILINGS = {"velocity_std_mps": 0.5, "velocity_bias_mps": 0.5, "gyro_std_rps": 0.5, "gyro_bias_rps": 0.5,
            "attitude_std_rad": 0.5, "attitude_bias_rad": 0.5, "heading_std_rad": 1.0, "heading_bias_rad": 1.0,
            "depth_std_m": 1.0, "position_std_m": 20.0}
MAX_DROPOUT = 0.5


@dataclass
class SensorNoiseConfig:
    velocity_std_mps: float = 0.0
    velocity_bias_mps: float = 0.0
    velocity_dropout_prob: float = 0.0
    gyro_std_rps: float = 0.0
    gyro_bias_rps: float = 0.0
    attitude_std_rad: float = 0.0
    attitude_bias_rad: float = 0.0
    heading_std_rad: float = 0.0
    heading_bias_rad: float = 0.0
    depth_std_m: float = 0.0
    position: str = "dead_reckoning"
    position_std_m: float = 0.0
    scale: float = 1.0
    source: str = ""                       # where the numbers came from (for reports)
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        for name in SIZE_FIELDS:
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and v >= 0):
                raise ValueError(f"sensor noise {name} must be a finite number >= 0, got {v!r}")
            if v > CEILINGS[name]:
                raise ValueError(f"sensor noise {name} = {v} is beyond any working sensor (limit {CEILINGS[name]}); "
                                 f"check the units (radians, metres, m/s)")
        p = self.velocity_dropout_prob
        if not (isinstance(p, (int, float)) and 0.0 <= p <= MAX_DROPOUT):
            raise ValueError(f"velocity_dropout_prob must be between 0 and {MAX_DROPOUT}, got {p!r}")
        if self.position not in POSITION_MODES:
            raise ValueError(f"sensor noise position must be one of {POSITION_MODES}, got {self.position!r}")
        if not (isinstance(self.scale, (int, float)) and math.isfinite(self.scale) and 0 <= self.scale <= 10):
            raise ValueError(f"sensor noise scale must be between 0 and 10, got {self.scale!r}")

    @property
    def active(self) -> bool:
        return self.scale > 0 and (any(getattr(self, n) > 0 for n in SIZE_FIELDS) or self.velocity_dropout_prob > 0)

    def scaled(self, name: str) -> float:
        return getattr(self, name) * self.scale

    @property
    def dropout(self) -> float:
        return min(MAX_DROPOUT, self.velocity_dropout_prob * self.scale)

    def describe(self) -> str:
        s = self.scale
        nav = (f"position dead-reckoned from the DVL and compass (start fix ±{self.scaled('position_std_m'):.2g} m)"
               if self.position == "dead_reckoning"
               else f"external positioning ±{100 * self.scaled('position_std_m'):.2g} cm")
        return (f"`{self.source or 'custom'}` x{s:g}: DVL ±{100 * self.scaled('velocity_std_mps'):.2g} cm/s "
                f"(bias ±{100 * self.scaled('velocity_bias_mps'):.2g}, dropouts {100 * self.dropout:.0f}%), "
                f"gyro ±{math.degrees(self.scaled('gyro_std_rps')):.2g} deg/s "
                f"(bias ±{math.degrees(self.scaled('gyro_bias_rps')):.2g}), "
                f"roll/pitch ±{math.degrees(self.scaled('attitude_std_rad')):.2g} deg "
                f"(bias ±{math.degrees(self.scaled('attitude_bias_rad')):.2g}), "
                f"heading ±{math.degrees(self.scaled('heading_std_rad')):.2g} deg "
                f"(bias ±{math.degrees(self.scaled('heading_bias_rad')):.2g}), "
                f"depth ±{100 * self.scaled('depth_std_m'):.2g} cm, {nav}")

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name not in ("extra",)}


def _known_keys() -> set:
    return {f.name for f in fields(SensorNoiseConfig)} - {"scale", "source", "extra"}


def load_sensor_noise(which: str, scale: float = 1.0) -> SensorNoiseConfig:
    """which: 'tank', 'typical', or a path to a sensor-noise YAML (fit_sensor_noise.py writes one)."""
    path = PRESETS.get(which, which)
    if not os.path.exists(path):
        raise ValueError(f"sensor noise must be none, tank, typical, or an existing YAML file; got {which!r}")
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"{path!r} isn't valid YAML: {e}") from e
    if not isinstance(raw, dict) or not isinstance(raw.get("sensors"), dict):
        raise ValueError(f"{path!r} is not a sensor-noise file (needs a 'sensors:' mapping; see "
                         f"sensor_noise/typical_dvl_ahrs.yaml)")
    sensors = raw["sensors"]
    unknown = set(sensors) - _known_keys()
    if unknown:
        raise ValueError(f"{path!r}: unknown sensor-noise keys {sorted(unknown)} (known: {sorted(_known_keys())})")
    try:
        cfg = SensorNoiseConfig(**{k: (float(v) if k != "position" else v) for k, v in sensors.items()},
                                scale=float(scale), source=os.path.basename(path),
                                extra={k: v for k, v in raw.items() if k != "sensors"})
    except (TypeError, ValueError) as e:
        raise ValueError(f"{path!r}: {e}") from e
    cfg.validate()
    return cfg


def resolve_sensor_noise(which: Optional[str], scale: float = 1.0) -> Optional[SensorNoiseConfig]:
    """None / 'none' / scale 0 -> None (exact readings, the original behaviour)."""
    if which in (None, "", "none") or scale == 0:
        return None
    cfg = load_sensor_noise(which, scale)
    return cfg if cfg.active else None


# ----------------------------------------------------------------- rotation helpers (scipy 'xyz' convention)

def euler_from_matrix(R) -> tuple:
    """(roll, pitch, yaw) with R = Rz(yaw) Ry(pitch) Rx(roll), as scipy's as_euler('xyz')."""
    pitch = -math.asin(max(-1.0, min(1.0, R[2][0])))
    return math.atan2(R[2][1], R[2][2]), pitch, math.atan2(R[1][0], R[0][0])


def matrix_from_euler(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch),
                              math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


@dataclass
class Measurement:
    """What one vehicle believes about itself this tick."""
    position: np.ndarray       # (3,) world
    R: np.ndarray              # (3,3) body -> world
    roll: float
    pitch: float
    nu: np.ndarray             # (6,) body velocity and rates


def exact(position, R, nu) -> Measurement:
    R = np.asarray(R, dtype=float)
    roll, pitch, _ = euler_from_matrix(R)
    return Measurement(np.asarray(position, dtype=float).copy(), R, roll, pitch, np.asarray(nu, dtype=float).copy())


class SensorSuite:
    """One vehicle's sensors for one episode. start() at reset, update() once per tick after the physics."""
    DRAWS_PER_TICK = 16  # fixed, so the random stream never depends on the configuration or the actions

    def __init__(self, cfg: SensorNoiseConfig, dt: float, seed):
        cfg.validate()
        self.cfg, self.dt = cfg, dt
        self._rng = np.random.default_rng(seed)
        c, r = cfg, self._rng
        self.bias_v = r.standard_normal(3) * c.scaled("velocity_bias_mps")
        self.bias_g = r.standard_normal(3) * c.scaled("gyro_bias_rps")
        self.bias_att = r.standard_normal(2) * c.scaled("attitude_bias_rad")
        self.bias_h = float(r.standard_normal()) * c.scaled("heading_bias_rad")
        fix = r.standard_normal(2) * c.scaled("position_std_m")
        self._fix = np.array([fix[0], fix[1], 0.0])
        self._s = {n: c.scaled(n) for n in ("velocity_std_mps", "gyro_std_rps", "attitude_std_rad",
                                            "heading_std_rad", "depth_std_m", "position_std_m")}
        self._nav = None
        self._prev_true = None
        self._v_held = None

    def start(self, position, R, nu) -> Measurement:
        position = np.asarray(position, dtype=float)
        self._nav = position + (self._fix if self.cfg.position == "dead_reckoning" else 0.0)
        self._prev_true = position.copy()
        self._v_held = None
        self._last = self._measure(position, np.asarray(R, dtype=float), np.asarray(nu, dtype=float), first=True)
        return self._last

    def update(self, position, R, nu) -> Measurement:
        self._last = self._measure(np.asarray(position, dtype=float), np.asarray(R, dtype=float),
                                   np.asarray(nu, dtype=float), first=False)
        return self._last

    @property
    def last(self) -> Measurement:
        return self._last

    def _measure(self, pos, R, nu, first: bool) -> Measurement:
        s = self._s
        z = self._rng.standard_normal(self.DRAWS_PER_TICK)
        u = float(self._rng.random())
        v_lin = nu[:3] + self.bias_v + z[0:3] * s["velocity_std_mps"]
        if self._v_held is not None and u < self.cfg.dropout:
            v_lin = self._v_held          # no DVL reading this tick: the last one is held
        self._v_held = v_lin
        rates = nu[3:] + self.bias_g + z[3:6] * s["gyro_std_rps"]
        roll_t, pitch_t, yaw_t = euler_from_matrix(R)
        roll = roll_t + self.bias_att[0] + z[6] * s["attitude_std_rad"]
        pitch = pitch_t + self.bias_att[1] + z[7] * s["attitude_std_rad"]
        yaw = yaw_t + self.bias_h + z[8] * s["heading_std_rad"]
        R_m = matrix_from_euler(roll, pitch, yaw)
        depth = pos[2] + z[9] * s["depth_std_m"]
        if self.cfg.position == "external":
            nav = np.array([pos[0] + z[10] * s["position_std_m"], pos[1] + z[11] * s["position_std_m"], depth])
        else:
            if not first:
                # the true displacement plus the error the navigation adds by integrating what it measured
                # instead of the truth: exact when every size is zero, drifts with biases and dropouts
                err = (R_m @ v_lin - R @ nu[:3]) * self.dt
                self._nav = self._nav + (pos - self._prev_true) + err
            nav = np.array([self._nav[0], self._nav[1], depth])
            self._nav = nav.copy()
        self._prev_true = pos.copy()
        return Measurement(nav, R_m, roll, pitch, np.concatenate([v_lin, rates]))
