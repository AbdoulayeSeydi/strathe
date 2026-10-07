"""
Comm channel — Track A0. A standalone, vehicle-agnostic underwater
acoustic communication model.

## Vehicle-agnostic by construction
This module imports nothing from dynamics_model.py, arena_core.py, or
arena.py, and never will — see A0.9's zero-physics-dependency test. It
only ever consumes position arrays (plain numpy-compatible sequences) and
produces/delivers Message objects. Any vehicle, calibrated or not, real
physics or a mock, can sit on either end of this channel as long as it can
report a position each tick.

## Physical model (formula mode — see the Strathe product vision doc)
- Latency = distance / speed_of_sound_mps (1500 m/s default, real seawater
  value). Fixed at SEND time, not delivery time — matches how sound
  actually propagates: a message's travel time depends on the distance
  between sender and receiver at the moment it left, not on where either
  agent has moved to by the time it arrives.
- Packet loss: a Bernoulli draw at send time, with drop probability
  increasing with distance via a simple linear ramp between a near-field
  floor and a far-field ceiling. No obstruction/line-of-sight geometry in
  this MVP — distance-only, per A0.5.
- Bandwidth: a per-agent, per-tick byte budget. Messages arriving in a
  given tick are accepted in arrival-time order until the budget is
  exhausted; anything beyond it is dropped OUTRIGHT for that tick, not
  deferred to the next one (A0.4's explicit MVP scope decision).

## Optical and hybrid links (OpticalChannel, LinkConfig)
An optional second link: short range, near-zero latency, Mbps. Received
light power falls as exp(-c*d)/d^2 (c = the water's attenuation per metre),
times a pointing gain (1 if the receiver is inside a sender terminal's beam
and the sender inside a receiver terminal's field of view, else 0), turned
into a delivery probability by a smooth S-curve. Pointing needs orientation,
so send() takes the sender's and receiver's rotation matrices (plain 3x3
arrays -- still no physics imports). UNCALIBRATED: the water values are
rough textbook figures and the S-curve is a placeholder a lab's link logs
(success against range and angle) would fit. Angles are HALF-angles
everywhere: 180 means every direction.

## What this file deliberately does NOT do yet
No obstruction modeling, no empirically-calibrated parameter fitting
(formula mode only), no message prioritization beyond arrival order. All
flagged as real, separate, later extensions in the product vision doc —
not oversights here.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

SPEED_OF_SOUND_MPS = 1500.0

MESSAGE_TYPE_POSITION_BEACON = "position_beacon"
MESSAGE_TYPE_COORDINATION_COMMAND = "coordination_command"
MESSAGE_TYPE_FULL_STATE_SYNC = "full_state_sync"

_KNOWN_MESSAGE_TYPES = (
    MESSAGE_TYPE_POSITION_BEACON,
    MESSAGE_TYPE_COORDINATION_COMMAND,
    MESSAGE_TYPE_FULL_STATE_SYNC,
)


def latency_seconds(pos_a, pos_b, speed_of_sound_mps: float = SPEED_OF_SOUND_MPS) -> float:
    """Pure function, A0.2. Takes two position-like arrays, returns a
    float. Nothing else — no state, no side effects, trivially testable
    and trivially reusable from anywhere (including, later, a MuJoCo-scene
    position array — this function doesn't care where the positions came
    from)."""
    a = np.asarray(pos_a, dtype=float)
    b = np.asarray(pos_b, dtype=float)
    distance = float(np.linalg.norm(a - b))
    return distance / speed_of_sound_mps


@dataclass
class PacketLossConfig:
    """Distance-conditioned drop probability, linear ramp between a
    near-field floor and a far-field ceiling. A constant-probability
    channel (e.g. for isolated bandwidth/latency testing) is just
    near_field_prob == far_field_prob."""
    near_distance_m: float
    far_distance_m: float
    near_field_prob: float
    far_field_prob: float

    def validate(self) -> None:
        if self.near_distance_m < 0 or self.far_distance_m < 0:
            raise ValueError("distances must be non-negative")
        if self.far_distance_m < self.near_distance_m:
            raise ValueError(
                f"far_distance_m ({self.far_distance_m}) must be >= "
                f"near_distance_m ({self.near_distance_m})"
            )
        for name, p in (("near_field_prob", self.near_field_prob),
                        ("far_field_prob", self.far_field_prob)):
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"{name} must be in [0,1], got {p}")

    def drop_probability(self, distance_m: float) -> float:
        if distance_m <= self.near_distance_m:
            return self.near_field_prob
        if distance_m >= self.far_distance_m:
            return self.far_field_prob
        span = self.far_distance_m - self.near_distance_m
        frac = (distance_m - self.near_distance_m) / span
        return self.near_field_prob + frac * (self.far_field_prob - self.near_field_prob)


@dataclass
class BandwidthConfig:
    """A0.4 — a flat per-agent, per-tick byte budget. No per-message-type
    prioritization in the MVP (e.g. beacons vs. full-state-sync competing
    for the same budget equally) — a real, separate future refinement."""
    bytes_per_tick: float

    def validate(self) -> None:
        if self.bytes_per_tick < 0:
            raise ValueError(f"bytes_per_tick must be non-negative, got {self.bytes_per_tick}")


@dataclass
class Message:
    """A0.1 — no physics fields. sender_id/receiver_id are opaque strings;
    this module never interprets them beyond using them as dict keys."""
    sender_id: str
    receiver_id: str
    message_type: str
    send_time: float
    arrival_time: float
    payload: Any = None
    payload_size_bytes: int = 0
    link: str = "acoustic"

    def validate(self) -> None:
        if self.message_type not in _KNOWN_MESSAGE_TYPES:
            raise ValueError(
                f"Unknown message_type {self.message_type!r} — expected one of "
                f"{_KNOWN_MESSAGE_TYPES}"
            )
        if self.payload_size_bytes < 0:
            raise ValueError(f"payload_size_bytes must be non-negative, got {self.payload_size_bytes}")
        if self.arrival_time < self.send_time:
            raise ValueError(
                f"arrival_time ({self.arrival_time}) precedes send_time "
                f"({self.send_time}) — latency cannot be negative"
            )


class _QueuedChannel:
    """Per-receiver delay queues (min-heaps on arrival_time), a per-tick
    delivered buffer, and the bandwidth budget -- shared by the acoustic and
    optical links. Subclasses decide in send() whether a message survives and
    when it arrives, then call _enqueue()."""

    def __init__(self, bandwidth_config: BandwidthConfig):
        bandwidth_config.validate()
        self.bandwidth_config = bandwidth_config
        # receiver_id -> heap of (arrival_time, tie_breaker, Message)
        self._queues: dict[str, list] = {}
        # receiver_id -> list[Message] delivered THIS tick, cleared on receive()
        self._delivered: dict[str, list] = {}
        self._counter = 0  # heap tie-breaker so Message never needs to be orderable

    def reset(self) -> None:
        """Clears all in-flight and delivered messages — call between
        episodes so a previous episode's tail messages never leak into
        the next one."""
        self._queues.clear()
        self._delivered.clear()
        self._counter = 0

    def _enqueue(self, msg: Message) -> Message:
        msg.validate()
        heapq.heappush(
            self._queues.setdefault(msg.receiver_id, []),
            (msg.arrival_time, self._counter, msg),
        )
        self._counter += 1
        return msg

    def step(self, current_time: float) -> None:
        """A0.6 + A0.4 composed: for every receiver, pops every message
        whose arrival_time has come due, then applies the bandwidth budget
        — messages are accepted in arrival-time order (the heap already
        guarantees this) until the budget is exhausted; anything beyond it
        is dropped outright for this tick, not deferred (A0.4)."""
        for receiver_id, queue in self._queues.items():
            due: list[Message] = []
            while queue and queue[0][0] <= current_time:
                _, _, msg = heapq.heappop(queue)
                due.append(msg)

            budget = self.bandwidth_config.bytes_per_tick
            used = 0.0
            accepted: list[Message] = []
            for msg in due:
                if used + msg.payload_size_bytes <= budget:
                    accepted.append(msg)
                    used += msg.payload_size_bytes
                # else: dropped outright — A0.4's explicit MVP scope decision

            if accepted:
                self._delivered.setdefault(receiver_id, []).extend(accepted)

    def receive(self, receiver_id: str) -> list:
        """A0.7 — returns and clears whatever arrived for this agent this
        tick. An empty list is a legitimate, expected outcome (nothing
        arrived, or everything in flight was dropped/delayed/over-budget),
        not an error condition."""
        msgs = self._delivered.get(receiver_id, [])
        self._delivered[receiver_id] = []
        return msgs

    def pending_count(self, receiver_id: str) -> int:
        """Diagnostic helper — how many messages are still in flight
        (queued but not yet due) for a given receiver. Not used by the
        physical model itself; useful for tests and later debugging."""
        return len(self._queues.get(receiver_id, []))


class CommChannel(_QueuedChannel):
    """
    A0.3/A0.6/A0.7 — the acoustic channel. The three physical effects
    (latency, packet loss, bandwidth) composed together in send()/step().

    Usage per tick:
        for each agent pair that wants to communicate this tick:
            channel.send(sender_id, receiver_id, current_time,
                         sender_pos, receiver_pos, message_type, ...)
        channel.step(current_time)   # advances delivery, applies bandwidth
        for each agent:
            msgs = channel.receive(agent_id)   # may be [] — not an error
    """

    link = "acoustic"

    def __init__(self, packet_loss_config: PacketLossConfig,
                 bandwidth_config: BandwidthConfig,
                 speed_of_sound_mps: float = SPEED_OF_SOUND_MPS,
                 seed: Optional[int] = None,
                 fixed_latency_s: float = 0.0):
        """fixed_latency_s: the modem's own delay on top of travel time (processing and packet airtime; often
        seconds for acoustic modems). 0 = travel time only, the model before link calibration (#68)."""
        packet_loss_config.validate()
        if not (0.0 <= fixed_latency_s <= 60.0):
            raise ValueError(f"fixed_latency_s must be in [0, 60] s, got {fixed_latency_s}")
        super().__init__(bandwidth_config)
        self.packet_loss_config = packet_loss_config
        self.speed_of_sound_mps = speed_of_sound_mps
        self.fixed_latency_s = float(fixed_latency_s)
        self._rng = np.random.default_rng(seed)

    def send(self, sender_id: str, receiver_id: str, send_time: float,
              sender_pos, receiver_pos, message_type: str,
              payload: Any = None, payload_size_bytes: int = 0,
              sender_R=None, receiver_R=None) -> Optional[Message]:
        """A0.2 + A0.5 composed: draws packet loss first (at the real
        physical distance, at send time); if the message survives,
        computes latency and enqueues it. Returns the Message if it was
        sent into the channel, or None if it was dropped outright at send
        time — the caller does not need to distinguish "lost in transit"
        from "still in flight," since receive() is the only place delivery
        is ever observed. sender_R/receiver_R are accepted so both links
        share one call; sound doesn't care which way a vehicle faces."""
        distance = float(np.linalg.norm(np.asarray(sender_pos, dtype=float)
                                         - np.asarray(receiver_pos, dtype=float)))
        drop_prob = self.packet_loss_config.drop_probability(distance)
        if self._rng.random() < drop_prob:
            return None

        latency = self.fixed_latency_s + distance / self.speed_of_sound_mps
        return self._enqueue(Message(
            sender_id=sender_id, receiver_id=receiver_id, message_type=message_type,
            send_time=send_time, arrival_time=send_time + latency,
            payload=payload, payload_size_bytes=payload_size_bytes, link=self.link,
        ))

    def skip_draw(self) -> None:
        """Consume the random draw a send would have used, without sending.
        A hybrid link that skips an acoustic beacon calls this, so the
        acoustic drop pattern stays identical to an acoustic-only run of the
        same episode (paired comparison)."""
        self._rng.random()


# ============================================================ optical link

# Beam attenuation coefficient c (1/m) by water type. Rough textbook figures
# (Petzold's classic clear-ocean / coastal / turbid-harbour measurements, as
# usually quoted in the underwater optical communication literature):
# ASSUMED, not measured for any lab's site.
WATER_ATTENUATION_PER_M = {"clear": 0.15, "coastal": 0.40, "harbor": 2.2}
WATER_TYPES = (*WATER_ATTENUATION_PER_M, "custom")
TERMINAL_PRESETS = ("omni", "forward_cone", "fore_aft_cones", "down_cone")
LINKS = ("acoustic", "optical", "hybrid")
HYBRID_POLICIES = ("optical_first", "redundant")
_PRESET_AXES = {"forward_cone": [(1.0, 0.0, 0.0)],
                "fore_aft_cones": [(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)],
                "down_cone": [(0.0, 0.0, 1.0)]}   # body z points down: Fossen body frame, and the
                                                  # simulators' world z is down too (a buoyant vehicle rises to -z)
_REFERENCE_WATER = "clear"   # rated_range_m is quoted in clear water, as vendors do
_MIN_RANGE_M = 0.1           # received power is capped below this range


def _check_half_angle(name: str, value: float) -> None:
    if not (0.0 < float(value) <= 180.0):
        hint = (" -- it is a HALF-angle: 360 is a full angle; use 180 for every direction"
                if float(value) > 180.0 else "")
        raise ValueError(f"{name} must be a half-angle in (0, 180] degrees, got {value}{hint}")


@dataclass
class OpticalTerminal:
    """One optical transceiver on a vehicle: the body-frame direction it faces
    and the HALF-angles of its transmit beam and receive field of view."""
    axis_body: tuple = (1.0, 0.0, 0.0)
    beam_half_angle_deg: float = 180.0
    receiver_half_fov_deg: float = 180.0

    def validate(self) -> None:
        _check_half_angle("beam_half_angle_deg", self.beam_half_angle_deg)
        _check_half_angle("receiver_half_fov_deg", self.receiver_half_fov_deg)
        a = np.asarray(self.axis_body, dtype=float)
        if a.shape != (3,) or not np.all(np.isfinite(a)) or np.linalg.norm(a) < 1e-9:
            raise ValueError(f"axis_body must be a non-zero 3-vector, got {self.axis_body}")

    @property
    def omni(self) -> bool:
        return self.beam_half_angle_deg >= 180.0 and self.receiver_half_fov_deg >= 180.0


def optical_terminals(preset: str = "omni", beam_half_angle_deg: Optional[float] = None,
                      receiver_half_fov_deg: Optional[float] = None) -> list:
    """The terminals a preset puts on each vehicle. Cone presets default to a
    30 deg beam and a 30 deg receiver half-angle (60 deg full field of view)."""
    if preset == "omni":
        return [OpticalTerminal((1.0, 0.0, 0.0), 180.0, 180.0)]
    if preset not in _PRESET_AXES:
        raise ValueError(f"optical terminals must be one of {TERMINAL_PRESETS}, got {preset!r}")
    beam = 30.0 if beam_half_angle_deg is None else float(beam_half_angle_deg)
    fov = 30.0 if receiver_half_fov_deg is None else float(receiver_half_fov_deg)
    return [OpticalTerminal(axis, beam, fov) for axis in _PRESET_AXES[preset]]


def log_received_power(distance_m: float, attenuation_per_m: float) -> float:
    """ln of the relative received light power exp(-c*d)/d^2 (d floored at 0.1 m).
    Kept in log form so long ranges in murky water don't underflow to zero."""
    d = max(float(distance_m), _MIN_RANGE_M)
    return float(-attenuation_per_m * d - 2.0 * np.log(d))


def received_power(distance_m: float, attenuation_per_m: float) -> float:
    """Relative received light power exp(-c*d)/d^2 (d floored at 0.1 m)."""
    return float(np.exp(log_received_power(distance_m, attenuation_per_m)))


def _within(axis_world, direction, half_angle_deg: float) -> bool:
    if half_angle_deg >= 180.0:
        return True
    a = np.asarray(axis_world, dtype=float)
    a = a / np.linalg.norm(a)
    cosang = float(np.clip(np.dot(a, direction), -1.0, 1.0))
    return cosang >= np.cos(np.radians(half_angle_deg)) - 1e-12


def beam_covers(sender_pos, sender_R, target_pos, terminals) -> bool:
    """True if some terminal's transmit beam covers target_pos (the sender half of pointing_gain)."""
    v = np.asarray(target_pos, dtype=float) - np.asarray(sender_pos, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return True
    R = np.eye(3) if sender_R is None else np.asarray(sender_R, dtype=float)
    return any(_within(R @ np.asarray(t.axis_body, dtype=float), v / n, t.beam_half_angle_deg) for t in terminals)


def pointing_gain(sender_pos, sender_R, receiver_pos, receiver_R, terminals) -> float:
    """1.0 if some sender terminal's beam covers the receiver AND some receiver
    terminal's field of view covers the sender, else 0.0. Both vehicles carry
    the same terminals. R: body-to-world rotation matrix (None = identity)."""
    s = np.asarray(sender_pos, dtype=float)
    r = np.asarray(receiver_pos, dtype=float)
    v = r - s
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return 1.0
    to_receiver = v / n
    Rs = np.eye(3) if sender_R is None else np.asarray(sender_R, dtype=float)
    Rr = np.eye(3) if receiver_R is None else np.asarray(receiver_R, dtype=float)
    tx = any(_within(Rs @ np.asarray(t.axis_body, dtype=float), to_receiver, t.beam_half_angle_deg)
             for t in terminals)
    if not tx:
        return 0.0
    rx = any(_within(Rr @ np.asarray(t.axis_body, dtype=float), -to_receiver, t.receiver_half_fov_deg)
             for t in terminals)
    return 1.0 if rx else 0.0


@dataclass
class OpticalLinkConfig:
    """An optical modem pair. rated_range_m is where a perfectly pointed link
    in CLEAR water delivers half its messages (the figure vendors quote); the
    same received-power threshold then gives ~15 m in coastal and ~4 m in
    harbour water at the defaults. curve_width is the S-curve's width in
    natural-log units of received power. Everything here is uncalibrated."""
    water: str = "coastal"
    attenuation_per_m: Optional[float] = None     # required for water: custom
    terminals: Any = "omni"                       # preset name, or a list of OpticalTerminal / dicts
    beam_half_angle_deg: Optional[float] = None   # cone presets only (default 30)
    receiver_half_fov_deg: Optional[float] = None
    data_rate_bps: float = 1.0e6
    latency_s: float = 0.001                      # the modem's processing delay
    rated_range_m: float = 30.0
    curve_width: float = 1.0
    beacon_interval_s: float = 0.1                # every tick at the fleet's 0.1 s tick
    calibrated_from: Optional[str] = None         # link_calibration.yaml the water, curve and latency came from

    @property
    def attenuation(self) -> float:
        if self.water == "custom":
            return float(self.attenuation_per_m)
        return WATER_ATTENUATION_PER_M[self.water]

    @property
    def terminal_list(self) -> list:
        if isinstance(self.terminals, str):
            return optical_terminals(self.terminals, self.beam_half_angle_deg, self.receiver_half_fov_deg)
        return [t if isinstance(t, OpticalTerminal) else OpticalTerminal(**t) for t in self.terminals]

    @property
    def pointing_label(self) -> str:
        if isinstance(self.terminals, str):
            if self.terminals == "omni" or self.beam_half_angle_deg is None and self.receiver_half_fov_deg is None:
                return self.terminals
            return (f"{self.terminals}({self.beam_half_angle_deg if self.beam_half_angle_deg is not None else 30:g}/"
                    f"{self.receiver_half_fov_deg if self.receiver_half_fov_deg is not None else 30:g})")
        return "custom_terminals"

    @property
    def omni(self) -> bool:
        return all(t.omni for t in self.terminal_list)

    @property
    def symmetric(self) -> bool:
        """True when every terminal's beam and receiver half-angles are equal:
        then 'my teammate's light reaches me' implies 'mine reaches it'
        (both vehicles carry the same terminals), up to the random draw."""
        return all(abs(t.beam_half_angle_deg - t.receiver_half_fov_deg) < 1e-9 for t in self.terminal_list)

    def validate(self) -> None:
        if self.water not in WATER_TYPES:
            raise ValueError(f"optical water must be one of {WATER_TYPES}, got {self.water!r}")
        if self.water == "custom":
            if self.attenuation_per_m is None:
                raise ValueError("optical water: custom needs attenuation_per_m (the water's attenuation "
                                 "coefficient c, per metre; ~0.15 clear ocean, ~0.4 coastal, 2+ harbour)")
        if self.attenuation_per_m is not None and not (0.0 < float(self.attenuation_per_m) <= 20.0):
            raise ValueError(f"attenuation_per_m must be in (0, 20] per metre, got {self.attenuation_per_m}")
        if self.attenuation_per_m is not None and self.water != "custom":
            raise ValueError(f"attenuation_per_m is only used with water: custom (got water: {self.water})")
        if isinstance(self.terminals, str) and self.terminals not in TERMINAL_PRESETS:
            raise ValueError(f"optical terminals must be one of {TERMINAL_PRESETS} or a list, got {self.terminals!r}")
        if isinstance(self.terminals, str) and self.terminals == "omni" and (
                self.beam_half_angle_deg is not None or self.receiver_half_fov_deg is not None):
            if (self.beam_half_angle_deg or 180.0) != 180.0 or (self.receiver_half_fov_deg or 180.0) != 180.0:
                raise ValueError("omni terminals see every direction; set terminals to forward_cone, "
                                 "fore_aft_cones or down_cone to use a beam or field-of-view half-angle")
        for name in ("beam_half_angle_deg", "receiver_half_fov_deg"):
            v = getattr(self, name)
            if v is not None:
                _check_half_angle(name, v)
        terms = self.terminal_list
        if not terms:
            raise ValueError("optical terminals must list at least one terminal")
        for t in terms:
            t.validate()
        if not (self.data_rate_bps > 0):
            raise ValueError(f"data_rate_bps must be positive, got {self.data_rate_bps}")
        if not (0.0 <= self.latency_s <= 1.0):
            raise ValueError(f"optical latency_s must be in [0, 1] s, got {self.latency_s}")
        if not (0.0 < self.rated_range_m <= 500.0):
            raise ValueError(f"rated_range_m must be in (0, 500] m, got {self.rated_range_m}")
        if not (self.curve_width > 0):
            raise ValueError(f"curve_width must be positive, got {self.curve_width}")
        if not (self.beacon_interval_s > 0):
            raise ValueError(f"optical beacon_interval_s must be positive, got {self.beacon_interval_s}")

    def success_probability(self, distance_m: float, gain: float = 1.0) -> float:
        if gain <= 0.0:
            return 0.0
        ref = log_received_power(self.rated_range_m, WATER_ATTENUATION_PER_M[_REFERENCE_WATER])
        margin = (log_received_power(distance_m, self.attenuation) - ref) / self.curve_width
        return float(gain / (1.0 + np.exp(-np.clip(margin, -500.0, 500.0))))

    def range_at(self, probability: float = 0.5) -> float:
        """Distance (m) at which a perfectly pointed link delivers `probability` of its messages."""
        lo, hi = _MIN_RANGE_M, 1000.0
        if self.success_probability(lo) < probability:
            return 0.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if self.success_probability(mid) >= probability else (lo, mid)
        return lo

    def describe(self) -> str:
        water = (f"custom water (c = {self.attenuation:g}/m)" if self.water == "custom"
                 else f"{self.water} water (c = {self.attenuation:g}/m, assumed)")
        terms = self.terminal_list
        if self.omni:
            aim = "omnidirectional terminals"
        else:
            aim = (f"{self.pointing_label} terminals (half-angles: beam "
                   f"{', '.join(f'{t.beam_half_angle_deg:g}' for t in terms)} deg, receiver "
                   f"{', '.join(f'{t.receiver_half_fov_deg:g}' for t in terms)} deg)")
        base = (f"optical link, {water}, {aim}; 50% delivery at {self.range_at(0.5):.1f} m when pointed, "
                f"{self.data_rate_bps:g} bit/s, {1000 * self.latency_s:g} ms latency, a beacon every "
                f"{self.beacon_interval_s:g} s.")
        if self.calibrated_from:
            return (f"{base} Calibrated from your link logs ({self.calibrated_from.replace(chr(92), '/').rsplit('/', 1)[-1]}): the water's "
                    f"attenuation, the delivery curve and the latency were fitted to logged messages; the "
                    f"terminals' pointing is still the model's")
        return (f"{base} Uncalibrated: the water values and the delivery curve are rough placeholders; check a "
                f"vendor's specs and fit to your own link logs (strathe fit-link) before quoting")

    def to_dict(self) -> dict:
        d = {"water": self.water, "terminals": self.terminals if isinstance(self.terminals, str)
             else [{"axis_body": list(t.axis_body), "beam_half_angle_deg": t.beam_half_angle_deg,
                    "receiver_half_fov_deg": t.receiver_half_fov_deg} for t in self.terminal_list],
             "data_rate_bps": self.data_rate_bps, "latency_s": self.latency_s, "rated_range_m": self.rated_range_m,
             "curve_width": self.curve_width, "beacon_interval_s": self.beacon_interval_s}
        for k in ("attenuation_per_m", "beam_half_angle_deg", "receiver_half_fov_deg", "calibrated_from"):
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OpticalLinkConfig":
        cfg = cls(**d)
        cfg.validate()
        return cfg


class OpticalChannel(_QueuedChannel):
    """The optical link: a fixed small latency; delivery drawn once per send
    from success_probability(distance, pointing); a per-tick byte budget of
    data_rate_bps * tick_s / 8. Exactly one random draw per send, even when
    the pointing gain is zero, so the draw sequence depends only on when
    messages are sent -- conditions stay paired."""

    link = "optical"

    def __init__(self, config: OpticalLinkConfig, tick_s: float, seed: Optional[int] = None):
        config.validate()
        super().__init__(BandwidthConfig(bytes_per_tick=config.data_rate_bps * tick_s / 8.0))
        self.config = config
        self._terminals = config.terminal_list
        self._omni = all(t.omni for t in self._terminals)
        self._rng = np.random.default_rng(seed)

    def send(self, sender_id: str, receiver_id: str, send_time: float,
             sender_pos, receiver_pos, message_type: str,
             payload: Any = None, payload_size_bytes: int = 0,
             sender_R=None, receiver_R=None) -> Optional[Message]:
        if not self._omni and (sender_R is None or receiver_R is None):
            raise ValueError("an optical link with pointing cones needs sender_R and receiver_R "
                             "(which way each vehicle faces)")
        distance = float(np.linalg.norm(np.asarray(sender_pos, dtype=float) - np.asarray(receiver_pos, dtype=float)))
        gain = 1.0 if self._omni else pointing_gain(sender_pos, sender_R, receiver_pos, receiver_R, self._terminals)
        u = self._rng.random()
        if u >= self.config.success_probability(distance, gain):
            return None
        return self._enqueue(Message(
            sender_id=sender_id, receiver_id=receiver_id, message_type=message_type,
            send_time=send_time, arrival_time=send_time + self.config.latency_s,
            payload=payload, payload_size_bytes=payload_size_bytes, link=self.link,
        ))


@dataclass
class LinkConfig:
    """Which link(s) a fleet uses. acoustic = today's channel only (default);
    optical = the optical link only; hybrid = both. Hybrid policies:
      optical_first  optical beacons at their own rate; a vehicle skips its
                     acoustic beacon only while the optical link is up: an
                     optical message from its teammate arrived within the
                     last link_up_ticks ticks (default 1: on the previous
                     tick). Needs symmetric terminals; otherwise it runs as
                     redundant. Symmetric terminals make the geometry
                     reciprocal at an instant; the two directions still
                     draw their losses independently, which is why the
                     window is kept short.
      redundant      every beacon on both links.
    Acoustic energy and airtime aren't modelled, so redundant is never worse
    in this simulation; optical_first's value (saved energy and acoustic
    airtime) matters once 3+ vehicles share one acoustic channel."""
    link: str = "acoustic"
    optical: Optional[OpticalLinkConfig] = None
    hybrid_policy: str = "optical_first"
    link_up_ticks: int = 1                      # "up" = heard optically within this many ticks

    @property
    def uses_acoustic(self) -> bool:
        return self.link in ("acoustic", "hybrid")

    @property
    def uses_optical(self) -> bool:
        return self.link in ("optical", "hybrid")

    def validate(self) -> None:
        if self.link not in LINKS:
            raise ValueError(f"link must be one of {LINKS}, got {self.link!r}")
        if self.hybrid_policy not in HYBRID_POLICIES:
            raise ValueError(f"hybrid_policy must be one of {HYBRID_POLICIES}, got {self.hybrid_policy!r}")
        if self.uses_optical:
            if self.optical is None:
                raise ValueError(f"link: {self.link} needs optical settings")
            self.optical.validate()
        if not (isinstance(self.link_up_ticks, (int, np.integer)) and 1 <= self.link_up_ticks <= 600):
            raise ValueError(f"link_up_ticks must be a whole number of ticks from 1 to 600, got {self.link_up_ticks}")

    def effective_hybrid_policy(self) -> tuple:
        """(policy actually used, reason or None). optical_first with
        asymmetric terminals would let one vehicle hear the other while its
        own light never gets back, skip its acoustic beacon, and starve its
        teammate: it runs as redundant instead."""
        if self.link != "hybrid":
            return None, None
        if self.hybrid_policy == "optical_first" and not self.optical.symmetric:
            return "redundant", ("optical_first needs symmetric terminals (beam half-angle = receiver "
                                 "half-angle); with these terminals hearing the teammate doesn't mean it hears "
                                 "you, so this run uses redundant (every beacon on both links)")
        return self.hybrid_policy, None

    @property
    def label(self) -> str:
        if self.link == "acoustic":
            return "acoustic"
        o = self.optical
        water = f"custom{o.attenuation:g}" if o.water == "custom" else o.water
        base = f"{self.link}/{water}"
        if self.link == "hybrid" and self.hybrid_policy != "optical_first":
            base += f"/{self.hybrid_policy}"
        return base if o.omni else f"{base}/{o.pointing_label}"

    def describe(self) -> str:
        if self.link == "acoustic":
            return "acoustic link only"
        text = self.optical.describe()
        if self.link == "optical":
            return f"optical only: {text}"
        pol, why = self.effective_hybrid_policy()
        return (f"hybrid ({pol}{'; ' + why if why else ''}): acoustic as configured plus an {text}")

    def start_geometry(self, separations) -> dict:
        """The task's start layout (vehicle 0 at the origin, vehicle 1 at
        (d0, 0, 0), both level and facing +x): is each direction inside the
        cones, and the delivery probability at each start separation."""
        o = self.optical
        terms = o.terminal_list
        g01 = pointing_gain((0, 0, 0), None, (1, 0, 0), None, terms)
        g10 = pointing_gain((1, 0, 0), None, (0, 0, 0), None, terms)
        seps = sorted(float(d) for d in separations)
        return {"0_to_1": bool(g01), "1_to_0": bool(g10),
                "both_ways_fraction": 1.0 if (g01 and g10) else 0.0,
                "probability": {d: min(o.success_probability(d, g01), o.success_probability(d, g10)) for d in seps},
                "pointed_probability": {d: o.success_probability(d, 1.0) for d in seps}}

    def start_geometry_note(self, separations) -> Optional[str]:
        """A plain warning when the optical link can't (or barely can) connect at the start, else None."""
        if not self.uses_optical:
            return None
        g = self.start_geometry(separations)
        if not (g["0_to_1"] and g["1_to_0"]):
            dirs = [n for n, ok in (("rear->front", g["0_to_1"]), ("front->rear", g["1_to_0"])) if not ok]
            return (f"At the start, 0% of vehicle pairs are inside each other's cones ({' and '.join(dirs)} "
                    f"blocked: both vehicles start in a line facing +x). The optical link carries nothing until "
                    f"the vehicles turn, so a flat optical result may just reflect the start layout.")
        probs = g["probability"]
        lo, hi = min(probs.values()), max(probs.values())
        if hi < 0.5:
            return (f"At the start separations ({min(probs):g}-{max(probs):g} m) the optical link delivers only "
                    f"{100 * lo:.0f}-{100 * hi:.0f}% of messages in this water; it matters only once the vehicles "
                    f"close in.")
        return None

    def to_dict(self) -> dict:
        d = {"link": self.link}
        if self.uses_optical:
            d["optical"] = self.optical.to_dict()
        if self.link == "hybrid":
            d["hybrid_policy"] = self.hybrid_policy
            if self.link_up_ticks != 1:
                d["link_up_ticks"] = self.link_up_ticks
        return d
