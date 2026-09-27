"""
buoyancy.py -- change the simulated vehicle's net buoyancy from what it was
calibrated with: added payload, a different trim, fresh vs salt water.

    buoyancy:                    # in a fleet experiment YAML (default: none = as calibrated)
      offset_n: -3.0             # newtons added to the calibration's net buoyancy;
                                 # + = more buoyant (floats up), - = heavier (sinks);
                                 # or [-5, 5] to draw one per episode (training on a range)
      known_to_beliefs: false    # physics teammate models were calibrated before the change

Physics: the calibration's net_buoyancy_z (buoyancy minus weight, N) is the
vertical restoring force in every engine (dynamics_model._restoring_forces,
fast_dynamics, MuJoCo); the offset is added to it for the true vehicle. The
righting moment (cg_cb_offset_z) is unchanged.

Beliefs: by default the physics teammate models keep the calibrated value,
because a calibration made before a payload change doesn't know about it;
known_to_beliefs: true gives them the new value too. Both vehicles get the
same offset (same vehicles, same payload).

Seeding: an offset drawn from a range depends only on the episode's
identity, so paired comparisons stay paired.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np

Range = Union[float, Tuple[float, float]]
# The thrusters' vertical authority in the simulators is 30 N by default (action_bound); an offset beyond this
# can't be held at all, and beyond ~50 N it's almost certainly a units mistake (kg for N).
MAX_ABS_OFFSET_N = 50.0


def _as_range(x) -> Tuple[float, float]:
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x), float(x)
    if isinstance(x, (list, tuple)) and len(x) == 2:
        lo, hi = float(x[0]), float(x[1])
        if lo > hi:
            raise ValueError(f"buoyancy offset range must be [min, max], got {list(x)}")
        return lo, hi
    raise ValueError(f"buoyancy offset_n must be a number or a [min, max] pair, got {x!r}")


@dataclass
class BuoyancyConfig:
    offset_n: Range = 0.0
    known_to_beliefs: bool = False

    def validate(self) -> None:
        lo, hi = _as_range(self.offset_n)
        for v in (lo, hi):
            if not math.isfinite(v):
                raise ValueError(f"buoyancy offset_n must be finite, got {self.offset_n}")
            if abs(v) > MAX_ABS_OFFSET_N:
                raise ValueError(f"buoyancy offset {v:g} N is beyond +/-{MAX_ABS_OFFSET_N:g} N; check the units "
                                 f"(newtons, not kg: 1 kg of payload is about -9.8 N)")

    @property
    def active(self) -> bool:
        lo, hi = _as_range(self.offset_n)
        return lo != 0.0 or hi != 0.0

    def draw(self, rng: np.random.Generator) -> float:
        lo, hi = _as_range(self.offset_n)
        return float(rng.uniform(lo, hi)) if hi > lo else lo

    def describe(self, teammates: bool = True) -> str:
        lo, hi = _as_range(self.offset_n)
        size = f"{lo:+g} N" if lo == hi else f"{lo:+g} to {hi:+g} N (drawn per episode)"
        s = f"net buoyancy {size} from the calibration (+ floats up, - sinks)"
        if teammates:
            s += ", known to physics teammate models" if self.known_to_beliefs else ", unknown to teammate models"
        return s

    @classmethod
    def from_dict(cls, d) -> "BuoyancyConfig":
        d = dict(d or {})
        unknown = set(d) - {"offset_n", "known_to_beliefs"}
        if unknown:
            raise ValueError(f"unknown buoyancy keys: {sorted(unknown)} (known: offset_n, known_to_beliefs)")
        off = d.get("offset_n", 0.0)
        cfg = cls(offset_n=tuple(off) if isinstance(off, list) else off,
                  known_to_beliefs=bool(d.get("known_to_beliefs", False)))
        cfg.validate()
        return cfg

    def to_dict(self) -> dict:
        off = self.offset_n
        return {"offset_n": list(off) if isinstance(off, tuple) else off, "known_to_beliefs": self.known_to_beliefs}


def with_buoyancy_offset(params, offset_n: float):
    """A copy of the calibration with offset_n added to its net buoyancy (the same object when 0)."""
    if not offset_n:
        return params
    return dataclasses.replace(params, net_buoyancy_z=float(params.net_buoyancy_z) + float(offset_n))
