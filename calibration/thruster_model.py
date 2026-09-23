"""
thruster_model.py -- turn raw thruster PWM commands into the body-frame
force/torque (wrench) the calibration pipeline needs.

Most real vehicle logs record what the autopilot sent to each ESC (PWM, in
microseconds), not force. The calibration pipeline needs force. This module
bridges the two in three explicit, separately checkable steps:

    1. direction   reversed thrusters get their PWM mirrored about 1500 us
                   (ArduSub's MOT_n_DIRECTION = -1): pwm' = 3000 - pwm
    2. curve       pwm' -> thrust (N) per thruster, from a thrust-curve table
    3. allocation  wrench (6,) = B (6 x n_thrusters) @ thrust

## The BlueROV2 Heavy preset, and how it was validated
The public oceansystemslab dataset logs all three stages (pwm_i,
pwm_virtual_i, thrust_i, and Fx..Tz). From it:
  - thrust_i is an exact function of pwm_virtual_i (zero spread, 183,555
    rows x 8 thrusters), giving the T200 curve table in
    thrusters/t200_bluerov2_dataset_curve.csv; its maxima (+47.9 N / -37.2 N)
    match Blue Robotics' published T200 maxima interpolated to ~15 V, i.e.
    a 4S battery -- forward and reverse independently give the same voltage
  - thrusters 1, 2, 5, 8 are mirrored in every session, 3, 4, 6, 7 never
  - Fx..Tz = B @ thrust exactly (least-squares residual 0), which recovers
    the allocation matrix below: four horizontal thrusters at 45 degrees,
    four vertical ones, with their moment arms
tests/test_thruster_model.py reproduces the dataset's Fx..Tz from raw PWM
alone to floating-point precision on every row.

## Other voltages
The T200's thrust grows with supply voltage. at_voltage() rescales the
reference curve so its forward and reverse maxima match Blue Robotics'
published maxima (12/16/20 V, linearly interpolated). Assumption: the
curve's SHAPE doesn't change with voltage, only its amplitude.

## Not modelled
ESC/motor response lag (the dataset's thrust is an instantaneous function
of PWM, so this model is too), battery sag within a run, thruster-thruster
and thruster-hull interaction.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
T200_DATASET_CURVE = os.path.join(HERE, "thrusters", "t200_bluerov2_dataset_curve.csv")
KGF = 9.80665

# Blue Robotics T200 (R2) published maximum thrust, kgf (forward, reverse),
# from the product page's technical details.
T200_PUBLISHED_MAX_KGF = {12.0: (3.71, 2.92), 16.0: (5.25, 4.10), 20.0: (6.70, 5.05)}

# BlueROV2 Heavy allocation (rows Fx, Fy, Fz, Tx, Ty, Tz; columns thrusters 1-8),
# recovered exactly from the dataset (see module docstring).
BLUEROV2_HEAVY_ALLOCATION = np.array([
    [-0.7071, -0.7071, 0.7071, 0.7071, 0.0, 0.0, 0.0, 0.0],
    [-0.7071, 0.7071, -0.7071, 0.7071, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
    [-0.0601, 0.0601, -0.0601, 0.0601, 0.218, -0.218, 0.218, -0.218],
    [0.0601, 0.0601, -0.0601, -0.0601, 0.12, 0.12, -0.12, -0.12],
    [-0.1888, 0.1888, 0.1888, -0.1888, 0.0, 0.0, 0.0, 0.0],
])
BLUEROV2_HEAVY_DIRECTIONS = np.array([-1, -1, 1, 1, -1, 1, 1, -1])


def _published_max_n(voltage: float) -> tuple:
    vs = sorted(T200_PUBLISHED_MAX_KGF)
    if not vs[0] <= voltage <= vs[-1]:
        raise ValueError(f"T200 published data covers {vs[0]}-{vs[-1]} V, got {voltage} V")
    fwd = np.interp(voltage, vs, [T200_PUBLISHED_MAX_KGF[v][0] for v in vs])
    rev = np.interp(voltage, vs, [T200_PUBLISHED_MAX_KGF[v][1] for v in vs])
    return fwd * KGF, rev * KGF


def voltage_matching(max_forward_n: float) -> float:
    """Supply voltage at which the published T200 forward maximum equals max_forward_n."""
    vs = sorted(T200_PUBLISHED_MAX_KGF)
    return float(np.interp(max_forward_n / KGF, [T200_PUBLISHED_MAX_KGF[v][0] for v in vs], vs))


@dataclass(frozen=True)
class ThrustCurve:
    """PWM (us) -> thrust (N), piecewise-linear through a table. Commands
    outside the table are clipped to its ends (the ESC saturates)."""
    pwm_us: np.ndarray
    thrust_n: np.ndarray
    label: str = "custom"

    def __post_init__(self):
        if len(self.pwm_us) < 2 or np.any(np.diff(self.pwm_us) <= 0):
            raise ValueError("thrust curve PWM values must be strictly increasing (at least 2 points)")

    @classmethod
    def from_csv(cls, path: str, label: Optional[str] = None) -> "ThrustCurve":
        data = np.loadtxt(path, delimiter=",", comments="#", skiprows=0, dtype=str)
        rows = [r for r in data if r[0] != "pwm_us"]
        arr = np.array(rows, dtype=float)
        return cls(arr[:, 0], arr[:, 1], label or os.path.basename(path))

    @classmethod
    def t200_bluerov2_dataset(cls) -> "ThrustCurve":
        return cls.from_csv(T200_DATASET_CURVE, "T200 @ ~15 V (BlueROV2 dataset)")

    def __call__(self, pwm) -> np.ndarray:
        return np.interp(np.asarray(pwm, dtype=float), self.pwm_us, self.thrust_n)

    @property
    def max_forward_n(self) -> float:
        return float(self.thrust_n.max())

    @property
    def max_reverse_n(self) -> float:
        return float(-self.thrust_n.min())

    def at_voltage(self, voltage: float) -> "ThrustCurve":
        """Rescale positive and negative thrust separately so the maxima match
        the published T200 maxima at `voltage` (shape assumed unchanged)."""
        fwd, rev = _published_max_n(voltage)
        t = np.where(self.thrust_n >= 0, self.thrust_n * fwd / self.max_forward_n,
                     self.thrust_n * rev / self.max_reverse_n)
        return ThrustCurve(self.pwm_us.copy(), t, f"{self.label} rescaled to {voltage:g} V")


@dataclass(frozen=True)
class ThrusterConfig:
    allocation: np.ndarray            # (6, n) wrench per unit thrust
    directions: np.ndarray            # (n,) +1 normal, -1 reversed (PWM mirrored about 1500)
    curve: ThrustCurve
    label: str = "custom"

    def __post_init__(self):
        B = np.asarray(self.allocation, dtype=float)
        if B.ndim != 2 or B.shape[0] != 6:
            raise ValueError(f"allocation must be 6 x n_thrusters, got {B.shape}")
        if len(self.directions) != B.shape[1]:
            raise ValueError(f"{B.shape[1]} thrusters in allocation but {len(self.directions)} directions")
        if not set(np.unique(self.directions)) <= {-1, 1}:
            raise ValueError("directions must be +1 or -1")
        if np.linalg.matrix_rank(B) < 6:
            raise ValueError("allocation matrix can't produce all 6 force/torque axes (rank < 6)")

    @property
    def n_thrusters(self) -> int:
        return self.allocation.shape[1]

    def effective_pwm(self, pwm) -> np.ndarray:
        pwm = np.asarray(pwm, dtype=float)
        return np.where(self.directions > 0, pwm, 3000.0 - pwm)

    def thrusts(self, pwm) -> np.ndarray:
        """(..., n) PWM -> (..., n) thrust in newtons."""
        pwm = np.asarray(pwm, dtype=float)
        if pwm.shape[-1] != self.n_thrusters:
            raise ValueError(f"expected {self.n_thrusters} PWM values per sample, got {pwm.shape[-1]}")
        return self.curve(self.effective_pwm(pwm))

    def wrench(self, pwm) -> np.ndarray:
        """(..., n) PWM -> (..., 6) body-frame [Fx, Fy, Fz, Tx, Ty, Tz]."""
        return self.thrusts(pwm) @ np.asarray(self.allocation).T


def bluerov2_heavy(voltage: Optional[float] = None) -> ThrusterConfig:
    """BlueROV2 Heavy (8 x T200), validated against the oceansystemslab dataset.
    voltage=None uses the dataset's own curve (~15 V)."""
    curve = ThrustCurve.t200_bluerov2_dataset()
    if voltage is not None:
        curve = curve.at_voltage(voltage)
    return ThrusterConfig(BLUEROV2_HEAVY_ALLOCATION, BLUEROV2_HEAVY_DIRECTIONS, curve, "BlueROV2 Heavy")


def allocation_from_geometry(positions_m: Sequence, directions_unit: Sequence) -> np.ndarray:
    """Build B from thruster mounting positions (body frame, m, relative to
    the centre of mass) and thrust directions: column i = [d_i ; r_i x d_i]."""
    r = np.asarray(positions_m, dtype=float)
    d = np.asarray(directions_unit, dtype=float)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return np.vstack([d.T, np.cross(r, d).T])
