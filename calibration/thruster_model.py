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

## Frames: the dataset is ROS FLU/ENU; the preset is FRD (PROJECT_STATUS.md #62)
The reference dataset is a ROS recording: its world z points up (depths are
negative) and its body frame is forward-left-up. The allocation recovered
from it (BLUEROV2_HEAVY_ALLOCATION_REFERENCE_FRAME) is therefore in FLU, the
mirror image in y and z of ArduSub's own motor definitions. The preset,
bluerov2_heavy(), is the same vehicle in FRD (forward-right-down), matching
ArduSub's motor numbering and every lab log Strathe converts to FRD. Checked
against ArduSub's VECTORED_6DOF motor factors (tests/test_thruster_model.py)
and against a second, independent ArduSub BlueROV2 with motion capture
(DFKI, tests/test_dfki_tether_pilot.py): in the FLU version, yaw torque runs
opposite to the vehicle's own measured yaw rate in 14 of 14 runs.
bluerov2_heavy(frame="reference") keeps the as-recovered version for
reproducing the dataset's own logged forces.

## Other voltages
The T200's thrust grows with supply voltage. at_voltage() rescales the
reference curve so its forward and reverse maxima match Blue Robotics'
published maxima (12/16/20 V, linearly interpolated). Assumption: the
curve's SHAPE doesn't change with voltage, only its amplitude.

## Your own vehicle
thruster_block() reads a `thrusters:` YAML block (the same one in an ingest
spec, a fleet experiment YAML or `strathe arena`): the BlueROV2 Heavy
preset, a per-thruster `layout` (mounting position + thrust direction +
reversed flag, turned into the allocation matrix here), or a 6 x n
allocation CSV. The curve is the built-in T200 one (optionally rescaled to a
voltage) or your own CSV. Vehicles whose thrusters can't push every axis
(e.g. a 6-thruster BlueROV2 has no pitch authority) are accepted;
unreachable_axes() names what they can't push.

## In the simulators
achievable_wrench() limits a commanded body force/torque to what the
thrusters can actually deliver: it distributes the command over the
thrusters (least squares), and if any thruster would exceed its curve's
maximum, scales the whole command down until none does (the direction of
the command is kept, as ArduSub's own mixer does). Axes the thrusters can't
push get nothing.

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
# recovered exactly from the dataset (see module docstring), in the dataset's own ROS FLU frame.
BLUEROV2_HEAVY_ALLOCATION_REFERENCE_FRAME = np.array([
    [-0.7071, -0.7071, 0.7071, 0.7071, 0.0, 0.0, 0.0, 0.0],
    [-0.7071, 0.7071, -0.7071, 0.7071, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0],
    [-0.0601, 0.0601, -0.0601, 0.0601, 0.218, -0.218, 0.218, -0.218],
    [0.0601, 0.0601, -0.0601, -0.0601, 0.12, 0.12, -0.12, -0.12],
    [-0.1888, 0.1888, 0.1888, -0.1888, 0.0, 0.0, 0.0, 0.0],
])
BLUEROV2_HEAVY_DIRECTIONS = np.array([-1, -1, 1, 1, -1, 1, 1, -1])
# the dataset's torques are about a point 8.5 cm below the horizontal thrusters (z is up in its FLU
# frame); this moves them to the thruster plane (same value as vehicle_spec.yaml data_conventions,
# PROJECT_STATUS.md #50)
BLUEROV2_DATASET_TORQUE_OFFSET_M = (0.0, 0.0, -0.085)

# FLU <-> FRD for a wrench: a half turn about x flips y and z of the force and of the torque
FLU_TO_FRD_WRENCH = np.diag([1.0, -1.0, -1.0, 1.0, -1.0, -1.0])
# The same vehicle in FRD, ArduSub's own motor numbering: what lab logs need (PROJECT_STATUS.md #62)
BLUEROV2_HEAVY_ALLOCATION = FLU_TO_FRD_WRENCH @ BLUEROV2_HEAVY_ALLOCATION_REFERENCE_FRAME
BLUEROV2_HEAVY_TORQUE_OFFSET_M = (0.0, 0.0, 0.085)  # the same point, 8.5 cm below, with z down


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
    """Command -> thrust (N), piecewise-linear through a table. Commands
    outside the table are clipped to its ends (the ESC saturates).
    unit: "pwm_us" (ESC microseconds; the column is called pwm_us for that
    reason) or "command" (a normalized command, in the units the vehicle logs
    it: thruster_block's command: normalized)."""
    pwm_us: np.ndarray
    thrust_n: np.ndarray
    label: str = "custom"
    unit: str = "pwm_us"
    # the battery voltage a lab's own T200 table was measured at (thruster block's curve_voltage:), which lets it be
    # rescaled to other voltages; None for the built-in dataset curve, which is rescaled from its own maxima
    reference_voltage: Optional[float] = None

    def __post_init__(self):
        if len(self.pwm_us) < 2 or np.any(np.diff(self.pwm_us) <= 0):
            raise ValueError("thrust curve PWM values must be strictly increasing (at least 2 points)")

    @classmethod
    def from_csv(cls, path: str, label: Optional[str] = None) -> "ThrustCurve":
        """Two columns: pwm_us,thrust_n (ESC microseconds) or command,thrust_n (a normalized command, in the
        units it is logged in)."""
        data = np.loadtxt(path, delimiter=",", comments="#", skiprows=0, dtype=str, ndmin=2)
        unit = "pwm_us"
        if data[0][0].strip() in ("pwm_us", "command"):
            unit = "pwm_us" if data[0][0].strip() == "pwm_us" else "command"
            data = data[1:]
        arr = np.array(data, dtype=float)
        return cls(arr[:, 0], arr[:, 1], label or os.path.basename(path), unit)

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

    def voltage_scales(self, volts):
        """(forward, reverse) thrust scale factors at these supply voltages, from Blue Robotics' published T200
        maxima (12-20 V). A lab's own T200 table (reference_voltage set): published(V) / published(reference), so
        the table keeps its shape and its own level at the voltage it was measured at (#69). The built-in dataset
        curve (no reference_voltage): published(V) / its own maxima, as before."""
        fwd, rev = np.vectorize(_published_max_n)(np.asarray(volts, dtype=float))
        if self.reference_voltage is None:
            return fwd / self.max_forward_n, rev / self.max_reverse_n
        f0, r0 = _published_max_n(self.reference_voltage)
        return fwd / f0, rev / r0

    def at_voltage(self, voltage: float) -> "ThrustCurve":
        """Rescale positive and negative thrust separately to `voltage` (shape assumed unchanged): to the
        published T200 maxima for the built-in curve, by the published ratio for a lab's own table."""
        fs, rs = self.voltage_scales(voltage)
        t = np.where(self.thrust_n >= 0, self.thrust_n * fs, self.thrust_n * rs)
        return ThrustCurve(self.pwm_us.copy(), t, f"{self.label} rescaled to {voltage:g} V", self.unit,
                           None if self.reference_voltage is None else float(voltage))

    def for_normalized_commands(self, lo: float, hi: float) -> "ThrustCurve":
        """This PWM curve for commands logged on [lo, hi] instead: lo..hi maps linearly onto the ESC's
        1100..1900 us, so the middle of the range is stopped (1500 us)."""
        if self.unit != "pwm_us":
            return self
        x = lo + (self.pwm_us - 1100.0) / 800.0 * (hi - lo)
        return ThrustCurve(x, self.thrust_n.copy(), f"{self.label}, commands {lo:g}..{hi:g} mapped to 1100-1900 us",
                           "command", self.reference_voltage)


@dataclass(frozen=True)
class ThrusterConfig:
    allocation: np.ndarray            # (6, n) wrench per unit thrust
    directions: np.ndarray            # (n,) +1 normal, -1 reversed (PWM mirrored about 1500)
    curve: ThrustCurve
    label: str = "custom"
    # the offset calibration should re-express this config's torques by (vehicle_spec.yaml
    # data_conventions.torque_reference_offset_m); nonzero only for the BlueROV2 preset, which
    # reproduces the dataset's own torque reference point (PROJECT_STATUS.md #50)
    torque_reference_offset_m: tuple = (0.0, 0.0, 0.0)
    # what the logged commands are: ESC PWM in microseconds (command_range None, stopped at 1500), or normalized
    # commands on command_range (stopped in its middle). A reversed thruster's command is mirrored about the
    # stopped value (PROJECT_STATUS.md #67)
    command_range: Optional[tuple] = None
    command_neutral: float = 1500.0
    # per-thruster differences (#91; a layout entry's curve_csv, thrust_scale, max_forward_n / max_reverse_n): a
    # thruster with its own table, one rated or set to a fraction of the curve, or capped. None: all the same
    curves: Optional[tuple] = None          # one ThrustCurve per thruster (None entries: the shared curve)
    thrust_scales: Optional[tuple] = None   # multiplier per thruster
    max_forward: Optional[tuple] = None     # cap per thruster, N (None: the curve's own)
    max_reverse: Optional[tuple] = None

    def __post_init__(self):
        B = np.asarray(self.allocation, dtype=float)
        if B.ndim != 2 or B.shape[0] != 6:
            raise ValueError(f"allocation must be 6 x n_thrusters, got {B.shape}")
        if len(self.directions) != B.shape[1]:
            raise ValueError(f"{B.shape[1]} thrusters in allocation but {len(self.directions)} directions")
        if not set(np.unique(self.directions)) <= {-1, 1}:
            raise ValueError("directions must be +1 or -1")
        if np.linalg.matrix_rank(B) == 0:
            raise ValueError("allocation matrix is all zeros (rank 0): the thrusters can't push the vehicle at all")
        for name in ("curves", "thrust_scales", "max_forward", "max_reverse"):
            v = getattr(self, name)
            if v is not None and len(v) != B.shape[1]:
                raise ValueError(f"{name}: {len(v)} entries for {B.shape[1]} thrusters")

    @property
    def per_thruster(self) -> bool:
        return any(getattr(self, n) is not None for n in ("curves", "thrust_scales", "max_forward", "max_reverse"))

    def curve_of(self, i: int) -> "ThrustCurve":
        c = self.curves[i] if self.curves is not None else None
        return c if c is not None else self.curve

    def _raw_thrusts(self, pwm) -> np.ndarray:
        ep = self.effective_pwm(pwm)
        if self.curves is None:
            return self.curve(ep)
        return np.stack([self.curve_of(i)(ep[..., i]) for i in range(self.n_thrusters)], axis=-1)

    def _finish(self, t) -> np.ndarray:
        """Per-thruster scale, then per-thruster caps."""
        if self.thrust_scales is not None:
            t = t * np.asarray(self.thrust_scales, dtype=float)
        if self.max_forward is not None or self.max_reverse is not None:
            t = np.clip(t, -self.reverse_limits(), self.forward_limits())
        return t

    def forward_limits(self) -> np.ndarray:
        """(n,) the most each thruster can push forward, N."""
        sc = np.ones(self.n_thrusters) if self.thrust_scales is None else np.asarray(self.thrust_scales, dtype=float)
        f = np.array([self.curve_of(i).max_forward_n for i in range(self.n_thrusters)]) * sc
        if self.max_forward is not None:
            f = np.minimum(f, [np.inf if m is None else m for m in self.max_forward])
        return f

    def reverse_limits(self) -> np.ndarray:
        sc = np.ones(self.n_thrusters) if self.thrust_scales is None else np.asarray(self.thrust_scales, dtype=float)
        r = np.array([self.curve_of(i).max_reverse_n for i in range(self.n_thrusters)]) * sc
        if self.max_reverse is not None:
            r = np.minimum(r, [np.inf if m is None else m for m in self.max_reverse])
        return r

    @property
    def n_thrusters(self) -> int:
        return self.allocation.shape[1]

    def effective_pwm(self, pwm) -> np.ndarray:
        """Commands with reversed thrusters mirrored about the stopped value (3000 - pwm for PWM)."""
        pwm = np.asarray(pwm, dtype=float)
        return np.where(self.directions > 0, pwm, 2.0 * self.command_neutral - pwm)

    def thrusts(self, pwm) -> np.ndarray:
        """(..., n) PWM -> (..., n) thrust in newtons."""
        pwm = np.asarray(pwm, dtype=float)
        if pwm.shape[-1] != self.n_thrusters:
            raise ValueError(f"expected {self.n_thrusters} PWM values per sample, got {pwm.shape[-1]}")
        return self._finish(self._raw_thrusts(pwm))

    def wrench(self, pwm) -> np.ndarray:
        """(..., n) PWM -> (..., 6) body-frame [Fx, Fy, Fz, Tx, Ty, Tz]."""
        return self.thrusts(pwm) @ np.asarray(self.allocation).T

    def wrench_at_voltage(self, pwm, volts) -> np.ndarray:
        """(N, n) commands and (N,) supply voltage per sample -> (N, 6). Each sample's thrust is rescaled as
        ThrustCurve.at_voltage does for one voltage, forward and reverse separately (ThrustCurve.voltage_scales;
        12-20 V, callers clip). The base curve must not be one already rescaled: the built-in T200 dataset curve
        (PROJECT_STATUS.md #63), or a lab's own T200 table with its curve_voltage (#69)."""
        volts = np.asarray(volts, dtype=float)
        if self.curves is not None and any(c is not None for c in self.curves):
            raise ValueError("a thruster with its own curve_csv can't be rescaled by battery voltage (that uses "
                             "Blue Robotics' published T200 data): drop voltage_column, or give every thruster the "
                             "shared curve")
        pwm = np.asarray(pwm, dtype=float)
        if pwm.shape[-1] != self.n_thrusters:
            raise ValueError(f"expected {self.n_thrusters} PWM values per sample, got {pwm.shape[-1]}")
        t = self._raw_thrusts(pwm)
        fs, rs = self.curve.voltage_scales(volts)
        t = np.where(t >= 0, t * np.asarray(fs)[:, None], t * np.asarray(rs)[:, None])
        return self._finish(t) @ np.asarray(self.allocation).T

    def for_simulation(self) -> "ThrusterConfig":
        """The same thrusters with torques taken about the physics model's reference point
        (applies torque_reference_offset_m to the allocation), as the simulators need."""
        r = np.asarray(self.torque_reference_offset_m, dtype=float)
        if not np.any(r):
            return self
        from dataclasses import replace
        B = np.array(self.allocation, dtype=float, copy=True)
        B[3:6] = B[3:6] - np.cross(r, B[0:3].T).T
        return replace(self, allocation=B, torque_reference_offset_m=(0.0, 0.0, 0.0))

    def unreachable_axes(self, tol: float = 1e-6) -> list:
        """Names of the body axes no combination of thrusts can push on its own."""
        B = np.asarray(self.allocation, dtype=float)
        P = B @ np.linalg.pinv(B)  # projector onto what the thrusters can produce
        return [AXIS_NAMES[j] for j in range(6) if np.linalg.norm(P[:, j] - np.eye(6)[:, j]) > tol]

    def achievable_wrench(self, tau) -> np.ndarray:
        """(..., 6) commanded wrench -> (..., 6) the part the thrusters can deliver.

        Least-squares thrust split, then one common scale factor so no thruster
        exceeds its curve's forward/reverse maximum (direction preserved)."""
        tau = np.asarray(tau, dtype=float)
        B = np.asarray(self.allocation, dtype=float)
        t = tau @ np.linalg.pinv(B).T
        fwd, rev = ((self.forward_limits(), self.reverse_limits()) if self.per_thruster
                    else (self.curve.max_forward_n, self.curve.max_reverse_n))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(t > 0, t / fwd, np.where(t < 0, -t / rev, 0.0))
        worst = np.max(ratio, axis=-1, keepdims=True)
        scale = np.where(worst > 1.0, 1.0 / np.where(worst > 1.0, worst, 1.0), 1.0)
        return (t * scale) @ B.T

    def max_wrench(self) -> np.ndarray:
        """(6,) the largest force/torque the thrusters can produce on each axis alone."""
        return np.array([self.achievable_wrench(1e6 * np.eye(6)[j])[j] for j in range(6)])


def bluerov2_heavy(voltage: Optional[float] = None, frame: str = "frd") -> ThrusterConfig:
    """BlueROV2 Heavy (8 x T200), validated against the oceansystemslab dataset.
    voltage=None uses the dataset's own curve (~15 V). frame="frd" (the preset everywhere: ArduSub's
    motor numbering in forward-right-down); frame="reference" gives the dataset's own FLU version,
    only for reproducing its logged forces (module docstring)."""
    curve = ThrustCurve.t200_bluerov2_dataset()
    if voltage is not None:
        curve = curve.at_voltage(voltage)
    if frame == "frd":
        return ThrusterConfig(BLUEROV2_HEAVY_ALLOCATION, BLUEROV2_HEAVY_DIRECTIONS, curve, "BlueROV2 Heavy",
                              BLUEROV2_HEAVY_TORQUE_OFFSET_M)
    if frame == "reference":
        return ThrusterConfig(BLUEROV2_HEAVY_ALLOCATION_REFERENCE_FRAME, BLUEROV2_HEAVY_DIRECTIONS, curve,
                              "BlueROV2 Heavy (reference dataset's FLU frame)", BLUEROV2_DATASET_TORQUE_OFFSET_M)
    raise ValueError(f"frame must be 'frd' or 'reference', got {frame!r}")


AXIS_NAMES = ("surge force", "sway force", "heave force", "roll torque", "pitch torque", "yaw torque")


def allocation_from_geometry(positions_m: Sequence, directions_unit: Sequence) -> np.ndarray:
    """Build B from thruster mounting positions (body frame, m, relative to
    the centre of mass) and thrust directions: column i = [d_i ; r_i x d_i]."""
    r = np.asarray(positions_m, dtype=float)
    d = np.asarray(directions_unit, dtype=float)
    d = d / np.linalg.norm(d, axis=1, keepdims=True)
    return np.vstack([d.T, np.cross(r, d).T])


def thruster_config_from_layout(layout: Sequence[dict], curve: ThrustCurve, label: str = "custom layout") -> ThrusterConfig:
    """layout: one dict per thruster (in PWM-column order) with position_m [x, y, z] (body FRD, m,
    relative to the point torques are taken about), direction [dx, dy, dz] (the way it pushes the
    vehicle for thrust above zero) and optional reversed (ArduSub MOT_n_DIRECTION = -1)."""
    B = allocation_from_geometry([th["position_m"] for th in layout], [th["direction"] for th in layout])
    directions = np.array([-1 if th.get("reversed", False) else 1 for th in layout])
    return ThrusterConfig(B, directions, curve, label)


# ----------------------------------------------------------------- YAML block

THRUSTER_BLOCK_KEYS = {"preset", "layout", "allocation_csv", "directions", "curve", "curve_csv", "curve_voltage",
                       "voltage", "command", "command_range"}
COMMAND_UNITS = ("pwm_us", "normalized")


def with_command_units(cfg: ThrusterConfig, command: str = "pwm_us", command_range=None) -> ThrusterConfig:
    """The thrusters for logged commands in these units (PROJECT_STATUS.md #67).
    pwm_us: ESC microseconds, the default. normalized: commands on command_range (default [-1, 1]), stopped in
    its middle; a curve_csv must then be in those units (command,thrust_n), and the built-in T200 curve maps the
    range onto 1100..1900 us."""
    from dataclasses import replace
    if command not in COMMAND_UNITS:
        raise ValueError(f"thrusters command must be one of {COMMAND_UNITS}, got {command!r}")
    if command == "pwm_us":
        if command_range is not None:
            raise ValueError("command_range is for command: normalized")
        if cfg.curve.unit != "pwm_us":
            raise ValueError("this curve_csv is in normalized command units (command,thrust_n); add "
                             "command: normalized")
        return cfg
    lo, hi = (-1.0, 1.0) if command_range is None else (float(command_range[0]), float(command_range[1]))
    if not lo < hi:
        raise ValueError(f"command_range must be [low, high] with low < high, got {command_range}")
    curve = cfg.curve
    if curve.unit == "pwm_us" and curve.label.startswith("T200"):
        curve = curve.for_normalized_commands(lo, hi)
    elif curve.unit == "pwm_us":
        raise ValueError("with command: normalized, give curve_csv in your command's units (header "
                         "command,thrust_n) or use the built-in T200 curve")
    curves = cfg.curves
    if curves is not None:
        mapped = []
        for c in curves:
            if c is not None and c.unit == "pwm_us":
                if not c.label.startswith("T200"):
                    raise ValueError("with command: normalized, every thruster's curve_csv must be in your command's "
                                     "units (header command,thrust_n)")
                c = c.for_normalized_commands(lo, hi)
            mapped.append(c)
        curves = tuple(mapped)
    return replace(cfg, curve=curve, command_range=(lo, hi), command_neutral=0.5 * (lo + hi), curves=curves)


def thruster_block(block, base_dir: str = ".") -> ThrusterConfig:
    """Build a ThrusterConfig from a `thrusters:` YAML block, or from a path to a YAML file that
    contains one (at the top level, or under wrench: as in an ingest spec).

        {preset: bluerov2_heavy, voltage: 15}
        {layout: [{position_m: [...], direction: [...], reversed: false}, ...], curve: t200, voltage: 16}
        {allocation_csv: B.csv, directions: [1, -1, ...], curve_csv: my_curve.csv}

    Relative file paths resolve against base_dir (the YAML's own folder)."""
    import yaml
    if isinstance(block, str):
        path = block if os.path.isabs(block) else os.path.join(base_dir, block)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        inner = (raw.get("wrench") or {}).get("thrusters") or raw.get("thrusters")
        if inner is None:
            if not isinstance(raw, dict) or not set(raw) <= THRUSTER_BLOCK_KEYS | {"pwm_columns", "voltage_column"}:
                raise ValueError(f"{path} has no thrusters: block (top level, or under wrench: as in an ingest "
                                 f"spec)")
            inner = raw
        return thruster_block(inner, os.path.dirname(os.path.abspath(path)))
    if not isinstance(block, dict):
        raise ValueError(f"thrusters must be a mapping or a path to a YAML file, got {type(block).__name__}")
    # ingestion-only keys: which log columns hold the PWM and the battery voltage (simulators have no logs; with
    # voltage_column they use voltage:, the nominal voltage, or the dataset curve's ~15 V)
    block = {k: v for k, v in block.items() if k not in ("pwm_columns", "voltage_column")}
    unknown = set(block) - THRUSTER_BLOCK_KEYS
    if unknown:
        raise ValueError(f"unknown thrusters key(s) {sorted(unknown)}; allowed: {sorted(THRUSTER_BLOCK_KEYS)}")
    command, command_range = block.pop("command", None) or "pwm_us", block.pop("command_range", None)
    return with_command_units(_thruster_block_body(block, base_dir), command, command_range)


def _thruster_block_body(block: dict, base_dir: str) -> ThrusterConfig:

    def _path(p):
        return p if os.path.isabs(p) else os.path.join(base_dir, p)

    sources = [k for k in ("preset", "layout", "allocation_csv") if block.get(k) is not None]
    if len(sources) != 1:
        raise ValueError("thrusters needs exactly one of: preset, layout, allocation_csv "
                         f"(got {sources or 'none'})")
    voltage = block.get("voltage")
    if sources[0] == "preset":
        if block["preset"] != "bluerov2_heavy":
            raise ValueError(f"unknown thrusters preset {block['preset']!r}; the only preset is bluerov2_heavy "
                             "(use layout: for any other vehicle)")
        extra = {"layout", "allocation_csv", "directions", "curve", "curve_csv", "curve_voltage"} & {
            k for k, v in block.items() if v is not None}
        if extra:
            raise ValueError(f"a thrusters preset fixes the layout and curve; remove {sorted(extra)}")
        return bluerov2_heavy(voltage)

    if block.get("curve_csv") is not None and block.get("curve") not in (None, "custom"):
        raise ValueError("give curve: t200 or curve_csv, not both")
    curve_voltage = block.get("curve_voltage")
    if curve_voltage is not None and block.get("curve_csv") is None:
        raise ValueError("curve_voltage is the battery voltage your own curve_csv was measured at; the built-in T200 "
                         "curve knows its own")
    if block.get("curve_csv") is not None:
        from dataclasses import replace
        curve = ThrustCurve.from_csv(_path(block["curve_csv"]))
        if curve_voltage is not None:
            cv = float(curve_voltage)
            _published_max_n(cv)  # raises outside the published 12-20 V
            curve = replace(curve, reference_voltage=cv, label=f"{curve.label} (measured at {cv:g} V)")
        if voltage is not None:
            if curve_voltage is None:
                raise ValueError("voltage rescaling uses Blue Robotics' published T200 data: for your own curve_csv "
                                 "of T200 thrusters, give curve_voltage (the battery voltage it was measured at); "
                                 "for other thrusters, put the curve at the right voltage in curve_csv instead")
            curve = curve.at_voltage(voltage)
    else:
        if block.get("curve", "t200") != "t200":
            raise ValueError(f"unknown curve {block.get('curve')!r}: use t200 or curve_csv")
        curve = ThrustCurve.t200_bluerov2_dataset()
        if voltage is not None:
            curve = curve.at_voltage(voltage)

    if sources[0] == "layout":
        layout = block["layout"]
        if not isinstance(layout, list) or not layout:
            raise ValueError("thrusters layout must be a non-empty list, one entry per thruster")
        for i, th in enumerate(layout, 1):
            if not isinstance(th, dict) or {"position_m", "direction"} - set(th):
                raise ValueError(f"layout thruster {i} needs position_m and direction")
            allowed = {"position_m", "direction", "reversed", "curve_csv", "thrust_scale", "max_forward_n",
                       "max_reverse_n"}
            if set(th) - allowed:
                raise ValueError(f"layout thruster {i}: unknown key(s) {sorted(set(th) - allowed)}; allowed: "
                                 f"{sorted(allowed)}")
            for k in ("thrust_scale", "max_forward_n", "max_reverse_n"):
                if th.get(k) is not None and not float(th[k]) > 0:
                    raise ValueError(f"layout thruster {i}: {k} must be positive, got {th[k]}")
            if len(th["position_m"]) != 3 or len(th["direction"]) != 3:
                raise ValueError(f"layout thruster {i}: position_m and direction need 3 numbers each")
            if np.linalg.norm(np.asarray(th["direction"], dtype=float)) < 1e-9:
                raise ValueError(f"layout thruster {i}: direction is all zeros")
        if block.get("directions") is not None:
            raise ValueError("with layout, mark reversed thrusters with reversed: true (directions is for "
                             "allocation_csv)")
        cfg = thruster_config_from_layout(layout, curve)
        if any(k in th for th in layout for k in ("curve_csv", "thrust_scale", "max_forward_n", "max_reverse_n")):
            from dataclasses import replace
            own = [ThrustCurve.from_csv(_path(th["curve_csv"])) if th.get("curve_csv") else None for th in layout]
            units = {c.unit for c in own if c is not None} | {curve.unit}
            if len(units) > 1:
                raise ValueError("every thruster's curve must be in the same units as the logged commands (pwm_us "
                                 "or command)")
            cfg = replace(cfg,
                          curves=tuple(own) if any(c is not None for c in own) else None,
                          thrust_scales=(tuple(float(th.get("thrust_scale", 1.0)) for th in layout)
                                         if any("thrust_scale" in th for th in layout) else None),
                          max_forward=(tuple(th.get("max_forward_n") for th in layout)
                                       if any("max_forward_n" in th for th in layout) else None),
                          max_reverse=(tuple(th.get("max_reverse_n") for th in layout)
                                       if any("max_reverse_n" in th for th in layout) else None),
                          label="custom layout, per-thruster curves/limits")
        return cfg

    if block.get("directions") is None:
        raise ValueError("allocation_csv needs directions: [1 or -1 per thruster]")
    B = np.loadtxt(_path(block["allocation_csv"]), delimiter=",", comments="#", ndmin=2)
    return ThrusterConfig(B, np.array(block["directions"]), curve, os.path.basename(block["allocation_csv"]))


def describe(cfg: ThrusterConfig) -> str:
    unreach = cfg.unreachable_axes()
    mw = cfg.max_wrench()
    per = ""
    if cfg.per_thruster:
        f, r = cfg.forward_limits(), cfg.reverse_limits()
        per = " per thruster: " + ", ".join(f"{i + 1}: +{a:.0f}/-{b:.0f} N" for i, (a, b) in enumerate(zip(f, r))) + ";"
    return (f"{cfg.label}: {cfg.n_thrusters} thrusters, curve {cfg.curve.label} (+{cfg.curve.max_forward_n:.1f} N / "
            f"-{cfg.curve.max_reverse_n:.1f} N);{per} max force {mw[0]:.0f}/{mw[1]:.0f}/{mw[2]:.0f} N, max torque "
            f"{mw[3]:.1f}/{mw[4]:.1f}/{mw[5]:.1f} N*m (surge/sway/heave, roll/pitch/yaw)"
            + (f"; can't push: {', '.join(unreach)}" if unreach else ""))
