"""
simfit/knobs.py -- the settings a fit tunes in a simulator's model, for any simulator (PROJECT_STATUS.md #81).

A knob is one number in the lab's model file (a drag coefficient, a mass, a position). The fit works on a parameter per
knob, so very different scales look alike to the optimizer:
  log_factor  parameter = log(value / start); limits are a factor either way (drag, added mass: x1/30 to x30)
  log         parameter = log(value); limits in the value's units (a volume factor)
  linear      parameter = value - start; limits in the value's units, around the start (a position, metres)
A knob can be held at a value the lab measured (`--hold NAME=VALUE`) instead of fitted. Knobs that share a `tie` move
together (an axisymmetric hull: sway = heave): only the first is fitted, the others take its value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from simfit.core import SimFitError

TRANSFORMS = ("log_factor", "log", "linear")


@dataclass
class KnobSpec:
    name: str                      # what the lab types in --hold, and what the report shows
    group: str                     # drag, added_mass, buoyancy, cg, mass, inertia, ... (--fit chooses groups)
    start: float                   # the model's current value
    transform: str = "log_factor"
    lo: Optional[float] = None     # limits: a factor for log_factor (default 30), else the value's (or offset's) units
    hi: Optional[float] = None
    tie: Optional[str] = None      # knobs with the same tie take one value (the first's)
    floor: float = 1e-9            # log transforms: a start at or below this counts as zero
    if_zero: Optional[float] = None  # log transforms: where to start when the model has 0 (a factor of 0 stays 0)
    note: str = ""                 # a line for the report (what the value is, in the lab's file)

    def __post_init__(self):
        if self.transform not in TRANSFORMS:
            raise SimFitError(f"knob {self.name}: transform must be one of {', '.join(TRANSFORMS)}, got {self.transform!r}")
        if self.transform != "linear" and self.start < 0:
            raise SimFitError(f"knob {self.name}: a {self.transform} knob needs a positive value, the model has "
                              f"{self.start:g} (use transform: linear, or give the magnitude)")


@dataclass
class Knobs:
    """The parameter vector <-> {knob: value} mapping that simfit.core.fit works with."""
    specs: list
    held: dict = field(default_factory=dict)   # {knob name: value}

    def __post_init__(self):
        names = [s.name for s in self.specs]
        if len(set(names)) != len(names):
            raise SimFitError(f"two knobs share a name: {', '.join(sorted(n for n in set(names) if names.count(n) > 1))}")
        unknown = set(self.held) - set(names)
        if unknown:
            raise SimFitError(f"can't hold {', '.join(sorted(unknown))}: the settings being fitted are "
                              f"{', '.join(names) or 'none'}")

    def _leaders(self) -> list:
        """One spec per tie (the first), plus every untied spec, in order."""
        seen, out = set(), []
        for s in self.specs:
            key = s.tie or s.name
            if key not in seen:
                seen.add(key)
                out.append(s)
        return out

    def all_names(self) -> list:
        return [s.name for s in self.specs]

    def names(self) -> list:
        """The fitted knobs (the parameter vector's entries): tie leaders that aren't held."""
        held_ties = {s.tie for s in self.specs if s.name in self.held and s.tie}
        return [s.name for s in self._leaders() if s.name not in self.held and (s.tie is None or s.tie not in held_ties)]

    def _spec(self, name: str) -> KnobSpec:
        return next(s for s in self.specs if s.name == name)

    @staticmethod
    def _base(s: KnobSpec) -> float:
        if s.transform == "linear":
            return s.start
        if s.start <= s.floor and s.if_zero:
            return s.if_zero
        return max(s.start, s.floor)

    def _value(self, s: KnobSpec, p: float) -> float:
        if s.transform == "log_factor":
            return self._base(s) * math.exp(p)
        if s.transform == "log":
            return math.exp(p)
        return s.start + p

    def x0(self) -> np.ndarray:
        out = []
        for n in self.names():
            s = self._spec(n)
            out.append(math.log(self._base(s)) if s.transform == "log" else 0.0)
        return np.array(out)

    def bounds(self):
        lo, hi = [], []
        for n in self.names():
            s = self._spec(n)
            if s.transform == "log_factor":
                lo.append(-math.log(s.lo or 30.0)); hi.append(math.log(s.hi or 30.0))
            elif s.transform == "log":
                lo.append(math.log(s.lo if s.lo else self._base(s) / 30)); hi.append(math.log(s.hi if s.hi else self._base(s) * 30))
            else:
                lo.append(s.lo if s.lo is not None else -abs(s.start) - 1.0)
                hi.append(s.hi if s.hi is not None else abs(s.start) + 1.0)
        lo, hi = np.array(lo, dtype=float), np.array(hi, dtype=float)
        x0 = self.x0()
        if np.any(x0 < lo) or np.any(x0 > hi):
            bad = [n for n, a, b, c in zip(self.names(), lo, x0, hi) if not a <= b <= c]
            raise SimFitError(f"the model's current value of {', '.join(bad)} is outside its limits: widen them")
        return lo, hi

    def values(self, x) -> dict:
        """{knob: value} for a parameter vector, held and tied knobs included, in spec order."""
        v = {n: self._value(self._spec(n), float(p)) for n, p in zip(self.names(), x)}
        v.update({n: float(h) for n, h in self.held.items()})
        lead = {}
        for s in self.specs:
            if s.tie and s.name in v:
                lead.setdefault(s.tie, v[s.name])
        out = {}
        for s in self.specs:
            if s.name in v:
                out[s.name] = v[s.name]
            elif s.tie in lead:
                out[s.name] = lead[s.tie]
            else:
                out[s.name] = s.start
        return out


def parse_holds(items) -> dict:
    """--hold NAME=VALUE ... -> {name: value}"""
    out = {}
    for kv in items or []:
        name, eq, value = kv.partition("=")
        try:
            out[name.strip()] = float(value)
        except ValueError:
            raise SimFitError(f"--hold takes NAME=VALUE (e.g. added_mass_x=10.6), got {kv!r}")
    return out
