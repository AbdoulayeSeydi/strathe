"""
fleet_belief.py -- how each vehicle estimates where its teammate is
between acoustic beacons (fleet v2).

## Why this exists
Fleet v1 had no belief model: each agent's "belief" was simply the last
beacon it received, and delta_physics_belief_error measured how far the
TEAMMATE had moved since then. That number depends on how fast the policy
drives each vehicle slot, not on either agent's physics -- a controlled
check showed the same negative "calibrated advantage" with no calibration
difference at all (both vehicles on generic physics: -0.043; both on
calibrated physics: -0.011).

Fleet v2 makes the physics matter where it physically would: an agent
dead-reckons its teammate forward from the last beacon using a dynamics
model of that teammate. A better model predicts the teammate better while
beacons are being lost. Which model the agent carries is the experimental
variable; the true physics of both vehicles is held fixed.

## Beacon contents
A beacon carries the sender's state at send time: world position, body
orientation, body velocity (nu), and the thrust command (tau) it applied
on its most recent tick. Real acoustic modems send small compressed
state packets; the channel's payload_size_bytes stays a nominal 24 bytes
so the named comm severities (including L3b's "1 byte short of a beacon")
keep their existing meaning.

## Belief conditions
    last_beacon        teammate assumed to sit where the last beacon said
    constant_velocity  last beacon position + world velocity * age
    physics            full 6-DOF dead reckoning with a chosen parameter
                       set, holding the teammate's last reported thrust
                       (zero-order hold) -- the only condition where
                       calibration can matter
    oracle             true teammate position (upper bound; not physical)

Timing: a beacon sent at tick s (describing the state at the END of tick s)
arrives at tick k >= s+1. On arrival the physics belief resets to the
beacon state and is stepped (k - s) ticks to "now"; after that it steps
once per tick until the next beacon.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from dynamics_model import CalibrationParameters, SimulationConfig, VehicleConstants
from fast_dynamics import FastCore

BELIEF_KINDS = ("last_beacon", "constant_velocity", "physics", "oracle")


@dataclass
class Beacon:
    send_tick: int
    position: np.ndarray        # (3,) world
    orientation: np.ndarray     # (3,3) body->world rotation matrix
    nu: np.ndarray              # (6,) body-frame velocity
    tau: np.ndarray             # (6,) body-frame thrust applied on send_tick


class TeammateBelief:
    """Base class. Subclasses implement _estimate()."""
    kind = "base"

    def __init__(self, dt: float):
        self.dt = dt
        self.reset()

    def reset(self) -> None:
        self._beacon: Optional[Beacon] = None
        self._now_tick = 0

    def on_tick(self, now_tick: int, delivered: list) -> None:
        """Called once per tick with every beacon that arrived this tick
        (possibly none). Only the most recently SENT beacon matters."""
        self._now_tick = now_tick
        newest = max(delivered, key=lambda b: b.send_tick) if delivered else None
        if newest is not None and (self._beacon is None or newest.send_tick > self._beacon.send_tick):
            self._beacon = newest
            self._on_new_beacon(newest, now_tick)
        elif self._beacon is not None:
            self._advance_one_tick()

    def _on_new_beacon(self, beacon: Beacon, now_tick: int) -> None:
        pass

    def _advance_one_tick(self) -> None:
        pass

    @property
    def has_estimate(self) -> bool:
        return self._beacon is not None

    @property
    def age_s(self) -> float:
        """Seconds since the state the newest beacon describes."""
        if self._beacon is None:
            return float("inf")
        return (self._now_tick - self._beacon.send_tick) * self.dt

    def estimate(self, true_teammate_position: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """World-frame teammate position estimate, or None if nothing has
        ever been received. true_teammate_position is only read by the
        oracle belief."""
        if self._beacon is None:
            return None
        return self._estimate()

    def _estimate(self) -> np.ndarray:
        raise NotImplementedError


class LastBeaconBelief(TeammateBelief):
    kind = "last_beacon"

    def _estimate(self):
        return self._beacon.position.copy()


class ConstantVelocityBelief(TeammateBelief):
    kind = "constant_velocity"

    def _estimate(self):
        b = self._beacon
        return b.position + (b.orientation @ b.nu[:3]) * self.age_s


class PhysicsBelief(TeammateBelief):
    kind = "physics"

    def __init__(self, dt: float, params: CalibrationParameters, vehicle: VehicleConstants,
                 physics_dt: float = 0.005):
        self._core = FastCore(params, vehicle, SimulationConfig(dt=dt, physics_dt=physics_dt))
        super().__init__(dt)

    def _on_new_beacon(self, beacon, now_tick):
        self._core.reset(position=beacon.position, nu=beacon.nu,
                         orientation=Rotation.from_matrix(beacon.orientation))
        for _ in range(max(0, now_tick - beacon.send_tick)):
            self._step()

    def _advance_one_tick(self):
        self._step()

    def _step(self):
        try:
            self._core.step(self._beacon.tau)
        except FloatingPointError:
            # A badly-mismatched model can diverge while coasting on a stale
            # command; fall back to the last beacon rather than crash an episode.
            b = self._beacon
            self._core.reset(position=b.position, nu=np.zeros(6),
                             orientation=Rotation.from_matrix(b.orientation))

    def _estimate(self):
        return self._core.position_array()


class OracleBelief(TeammateBelief):
    """Upper bound: perfect knowledge of the teammate's true position,
    available as soon as any beacon has ever arrived (so its observation
    validity pattern matches the other conditions)."""
    kind = "oracle"

    def estimate(self, true_teammate_position=None):
        if self._beacon is None:
            return None
        if true_teammate_position is None:
            raise ValueError("OracleBelief.estimate() needs the true teammate position")
        return np.asarray(true_teammate_position, dtype=float).copy()


def interpolate_params(truth: CalibrationParameters, other: CalibrationParameters,
                       alpha: float) -> CalibrationParameters:
    """(1-alpha)*truth + alpha*other, parameter by parameter. alpha=0 is the
    true model, alpha=1 is `other`. Stays physical: every damping term keeps
    its sign and the mass matrix is a convex combination of two
    positive-definite matrices, so it stays positive-definite."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    vec = (1.0 - alpha) * truth.to_vector() + alpha * other.to_vector()
    return CalibrationParameters.from_vector(vec)


@dataclass(frozen=True)
class BeliefSpec:
    """A picklable description of one belief condition (what an experiment
    config resolves to). params is required for kind='physics'."""
    kind: str
    label: str
    params: Optional[CalibrationParameters] = None

    def __post_init__(self):
        if self.kind not in BELIEF_KINDS:
            raise ValueError(f"belief kind must be one of {BELIEF_KINDS}, got {self.kind!r}")
        if self.kind == "physics" and self.params is None:
            raise ValueError("a 'physics' belief needs a parameter set")

    def build(self, dt: float, vehicle: VehicleConstants) -> TeammateBelief:
        if self.kind == "last_beacon":
            return LastBeaconBelief(dt)
        if self.kind == "constant_velocity":
            return ConstantVelocityBelief(dt)
        if self.kind == "oracle":
            return OracleBelief(dt)
        return PhysicsBelief(dt, self.params, vehicle)
