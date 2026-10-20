"""
acoustic_network.py -- a shared underwater acoustic channel for a fleet of N vehicles (PROJECT_STATUS.md #89).

comm_channel.py models one point-to-point link per vehicle pair with a per-tick byte budget. Real acoustic networks
behave differently, and the differences are what break multi-vehicle coordination:
- **One shared medium.** Every vehicle hears every transmission (broadcast), and two packets that overlap at a
  receiver destroy each other (a collision).
- **Half duplex.** A modem can't hear while it transmits.
- **Airtime.** A packet occupies the water for `packet_overhead_s + 8 * bytes / bit_rate_bps`: a 32-byte state
  beacon at 80 bit/s takes 3.2 s plus the preamble. Bandwidth is time on the air, not bytes per tick.
- **Latency** is travel time (range / speed of sound) + the time on the air + the modem's own decoding delay.
- **Losses come in bursts** (multipath fades, noise events), not as independent coin flips.

## The model
- **Times are continuous** (seconds), and the fleet simulation calls `advance(now, ...)` once per tick. A
  transmission that starts inside a tick carries the sender's state at the end of that tick.
- **Medium access (`mac`):**
  - `tdma`: vehicle i owns slot i of a repeating cycle. slot = airtime + guard, guard = the longest travel time in
    the fleet + a margin, so with the guard set right nothing collides. A vehicle sends at its slot if a beacon is
    due (`beacon_interval_s` since its last one); a beacon that falls due while the vehicle waits for its slot is
    sent at the slot, with the state at the slot. When `beacon_interval_s` is shorter than the cycle, the cycle is
    the real update interval, and the beacons that couldn't go are counted (`no_slot`).
  - `aloha`: each vehicle sends whenever a beacon is due (each vehicle's schedule starts at a random phase, and each
    beacon goes a random 0 to `aloha_jitter` x the interval after it falls due, so two vehicles that start in step
    don't collide forever). A
    beacon due while the modem is still transmitting waits, and a newer one replaces it (only the newest state
    matters); replaced beacons are counted (`superseded`). Packets that overlap at a receiver are lost
    (`collision`), as are packets arriving while the receiver transmits (`half_duplex`). No carrier sensing.
- **Loss (`LossModel`)**, per receiver, independent of collisions: a mean loss probability with independent losses,
  or a Gilbert-Elliott two-state chain with a mean burst length; times an optional distance ramp (comm_channel's
  PacketLossConfig). Each (sender, receiver) pair has its own chain.
- **Range:** a packet to a receiver farther than `max_range_m` is lost (`out_of_range`).
- **Determinism:** one RNG per network, drawn in a fixed order (per transmission, receivers in id order), so two
  runs with the same seed and the same transmissions lose the same packets.

## Not modelled (LIMITS.md)
Multipath geometry, Doppler, the capture effect (the stronger of two colliding packets surviving), clock drift
between TDMA slots, carrier sensing, acknowledgements and retransmission, and multi-hop relaying.

Like comm_channel.py, this file imports nothing from the physics: it consumes positions and opaque payloads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional

import numpy as np

from comm_channel import PacketLossConfig, SPEED_OF_SOUND_MPS

MAC_KINDS = ("tdma", "aloha")

# Starting points when a lab hasn't measured its modem (calibration/COMMS_STUDY.md). Generic classes, not specific
# products: check your modem's data sheet, or measure the link with `strathe fit-link`.
MODEM_PRESETS = {
    "ideal": dict(bit_rate_bps=math.inf, packet_overhead_s=0.0, processing_latency_s=0.0,
                  note="no airtime, no modem delay: travel time only"),
    "slow": dict(bit_rate_bps=80.0, packet_overhead_s=1.0, processing_latency_s=0.5,
                 note="robust low-rate FSK class: tens of bit/s, about a second of preamble"),
    "medium": dict(bit_rate_bps=1000.0, packet_overhead_s=0.5, processing_latency_s=0.3,
                   note="PSK class around 1 kbit/s"),
    "fast": dict(bit_rate_bps=5000.0, packet_overhead_s=0.25, processing_latency_s=0.2,
                 note="high-rate PSK class, several kbit/s, short range in practice"),
}


@dataclass
class ModemConfig:
    bit_rate_bps: float = math.inf          # inf = no airtime for the payload
    packet_overhead_s: float = 0.0          # preamble/acquisition on the air for every packet
    processing_latency_s: float = 0.0       # decoding delay after the packet has arrived
    speed_of_sound_mps: float = SPEED_OF_SOUND_MPS
    max_range_m: float = 3000.0

    @classmethod
    def preset(cls, name: str, **overrides) -> "ModemConfig":
        if name not in MODEM_PRESETS:
            raise ValueError(f"unknown modem preset {name!r}; choose from {', '.join(MODEM_PRESETS)}")
        values = {k: v for k, v in MODEM_PRESETS[name].items() if k != "note"}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def validate(self) -> None:
        if not self.bit_rate_bps > 0:
            raise ValueError(f"bit rate must be positive, got {self.bit_rate_bps}")
        for name in ("packet_overhead_s", "processing_latency_s"):
            v = getattr(self, name)
            if not 0 <= v <= 600:
                raise ValueError(f"{name} must be between 0 and 600 s, got {v}")
        if not self.speed_of_sound_mps > 0:
            raise ValueError("speed of sound must be positive")
        if not self.max_range_m > 0:
            raise ValueError("max range must be positive")

    def airtime_s(self, payload_bytes: float) -> float:
        bits = 8.0 * payload_bytes
        return self.packet_overhead_s + (0.0 if math.isinf(self.bit_rate_bps) else bits / self.bit_rate_bps)


@dataclass
class LossModel:
    """loss: mean probability that a packet is lost at a receiver (before collisions). burst_length: None for
    independent losses, else the mean number of consecutive losses (a Gilbert-Elliott chain; independent losses at
    30% already come in runs of 1/(1-0.3) = 1.4 on average). distance: an optional extra distance ramp."""
    loss: float = 0.0
    burst_length: Optional[float] = None
    distance: Optional[PacketLossConfig] = None

    def validate(self) -> None:
        if not 0.0 <= self.loss <= 1.0:
            raise ValueError(f"loss must be between 0 and 1, got {self.loss}")
        if self.burst_length is not None:
            if self.burst_length < 1.0:
                raise ValueError(f"burst length must be at least 1 packet, got {self.burst_length}")
            if self.loss >= 1.0:
                raise ValueError("bursty losses need a loss below 1")
            if self.p_good_to_bad > 1.0:
                raise ValueError(f"a {self.loss:.0%} loss can't have bursts as short as {self.burst_length:g} packets "
                                 f"(at least {self.loss / (1 - self.loss):.2f})")
        if self.distance is not None:
            self.distance.validate()

    @property
    def p_bad_to_good(self) -> float:
        return 1.0 / self.burst_length

    @property
    def p_good_to_bad(self) -> float:
        # stationary P(bad) = g/(g+b) = loss  ->  g = loss * b / (1 - loss)
        return self.loss * self.p_bad_to_good / (1.0 - self.loss)

    def describe(self) -> str:
        s = f"{self.loss:.0%} loss" + ("" if self.burst_length is None else f" in bursts of {self.burst_length:g}")
        if self.distance is not None:
            s += (f", plus {self.distance.near_field_prob:.0%}-{self.distance.far_field_prob:.0%} from "
                  f"{self.distance.near_distance_m:g} to {self.distance.far_distance_m:g} m")
        return s


@dataclass
class NetworkConfig:
    modem: ModemConfig = field(default_factory=ModemConfig)
    loss: LossModel = field(default_factory=LossModel)
    mac: str = "tdma"
    beacon_interval_s: float = 2.0
    beacon_bytes: int = 32
    guard_s: Optional[float] = None         # TDMA guard; None = set from the fleet's size (tdma_guard_s)
    half_duplex: bool = True
    aloha_jitter: float = 0.25              # aloha: each beacon waits a random 0..this x beacon_interval_s

    def validate(self) -> None:
        self.modem.validate()
        self.loss.validate()
        if self.mac not in MAC_KINDS:
            raise ValueError(f"mac must be one of {MAC_KINDS}, got {self.mac!r}")
        if not self.beacon_interval_s > 0:
            raise ValueError(f"beacon interval must be positive, got {self.beacon_interval_s}")
        if not 1 <= self.beacon_bytes <= 100000:
            raise ValueError(f"beacon size must be 1 to 100000 bytes, got {self.beacon_bytes}")
        if self.guard_s is not None and self.guard_s < 0:
            raise ValueError("guard time must be non-negative")
        if not 0 <= self.aloha_jitter <= 1:
            raise ValueError(f"aloha jitter must be 0 to 1 (a fraction of the beacon interval), got {self.aloha_jitter}")

    @property
    def airtime_s(self) -> float:
        return self.modem.airtime_s(self.beacon_bytes)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["modem"]["bit_rate_bps"] = None if math.isinf(self.modem.bit_rate_bps) else self.modem.bit_rate_bps
        return d


def tdma_guard_s(max_distance_m: float, speed_of_sound_mps: float = SPEED_OF_SOUND_MPS, margin_s: float = 0.05) -> float:
    """A guard long enough for a packet to reach the farthest teammate before the next slot starts."""
    return max_distance_m / speed_of_sound_mps + margin_s


@dataclass
class Packet:
    sender: int
    receiver: int
    payload: Any
    tx_start: float
    arrive_start: float
    arrive_end: float
    deliver_at: float
    fate: str = "pending"   # pending, delivered, lost, collision, half_duplex, out_of_range


COUNTERS = ("generated", "sent", "delivered", "lost", "collision", "half_duplex", "out_of_range", "superseded",
            "no_slot")


class AcousticNetwork:
    """One episode's channel. `advance(now, positions, make_payload)` once per tick returns, per vehicle, the
    payloads delivered to it this tick (oldest first)."""

    def __init__(self, n: int, config: NetworkConfig, seed: int, max_distance_m: float = 0.0):
        config.validate()
        if n < 1:
            raise ValueError("a network needs at least one vehicle")
        self.n, self.cfg = n, config
        self.rng = np.random.default_rng(seed)
        self.airtime = config.airtime_s
        self.guard = config.guard_s if config.guard_s is not None else tdma_guard_s(max_distance_m,
                                                                                    config.modem.speed_of_sound_mps)
        self.slot = self.airtime + self.guard
        self.cycle = self.n * self.slot
        self.counts = {k: 0 for k in COUNTERS}
        self.now = 0.0
        self._last_due = [-math.inf] * n        # when each vehicle's newest beacon fell due
        self._last_sent_due = [-math.inf] * n   # the due time of the beacon it last sent
        self._busy_until = [-math.inf] * n      # end of each vehicle's current transmission
        self._tx = [[] for _ in range(n)]       # each vehicle's transmissions: (start, end), recent ones kept
        self._arrivals = [[] for _ in range(n)]  # packets heading to each receiver, not yet decided
        self._state = {}                         # (sender, receiver) -> Gilbert-Elliott "bad" state
        # aloha: each vehicle's beacon schedule starts at a random phase (no two modems power up in step)
        self._phase = [float(self.rng.uniform(0, config.beacon_interval_s)) for _ in range(n)]
        self._pending = [None] * n               # aloha: a due beacon waiting for the modem to finish
        self.airtime_used = 0.0

    # --- schedule -------------------------------------------------------------------------------------------------
    def _due_times(self, i: int, t0: float, t1: float) -> list:
        """Beacon due times for vehicle i in (t0, t1]."""
        T = self.cfg.beacon_interval_s
        ph = self._phase[i] if self.cfg.mac == "aloha" else 0.0
        k0 = math.floor((t0 - ph) / T + 1e-9) + 1
        out = []
        k = max(k0, 0)
        while ph + k * T <= t1 + 1e-9:
            out.append(ph + k * T)
            k += 1
        return out

    def _slot_starts(self, i: int, t0: float, t1: float) -> list:
        k = max(0, math.floor((t0 - i * self.slot) / self.cycle + 1e-9) + 1)
        out = []
        while k * self.cycle + i * self.slot <= t1 + 1e-9:
            out.append(k * self.cycle + i * self.slot)
            k += 1
        return out

    # --- the channel ----------------------------------------------------------------------------------------------
    def _lost(self, sender: int, receiver: int, distance: float) -> bool:
        L = self.cfg.loss
        r = self.rng.random()
        if L.burst_length is None:
            lost = r < L.loss
        else:
            key = (sender, receiver)
            bad = self._state.get(key)
            if bad is None:
                bad = r < L.loss                       # first packet: the chain's stationary distribution
            else:
                bad = (r >= L.p_bad_to_good) if bad else (r < L.p_good_to_bad)
            self._state[key] = bad
            lost = bad
        r2 = self.rng.random()                         # always drawn, so the draws stay paired across settings
        if L.distance is not None and not lost:
            lost = r2 < L.distance.drop_probability(distance)
        return lost

    def _transmit(self, i: int, start: float, positions, payload) -> None:
        end = start + self.airtime
        self._tx[i].append((start, end))
        self._busy_until[i] = end
        self.counts["sent"] += 1
        self.airtime_used += self.airtime
        m = self.cfg.modem
        for j in range(self.n):
            if j == i:
                continue
            d = float(np.linalg.norm(np.asarray(positions[i], float) - np.asarray(positions[j], float)))
            lost = self._lost(i, j, d)
            travel = d / m.speed_of_sound_mps
            p = Packet(i, j, payload, start, start + travel, end + travel, end + travel + m.processing_latency_s)
            if d > m.max_range_m:
                p.fate = "out_of_range"
            elif lost:
                p.fate = "lost"
            else:  # a packet that reaches the receiver destroys, and is destroyed by, any other one overlapping it
                for q in self._arrivals[j]:
                    if q.fate in ("pending", "collision") and q.arrive_start < p.arrive_end \
                            and p.arrive_start < q.arrive_end:
                        p.fate = q.fate = "collision"
            self._arrivals[j].append(p)

    def advance(self, now: float, positions, make_payload: Callable[[int], Any]) -> list:
        """Run the channel from the previous call to `now`. make_payload(i) is vehicle i's beacon (its state at the end
        of this tick), built only when it transmits. Returns, per vehicle, the payloads delivered to it."""
        t0, self.now = self.now, now
        starts = []
        for i in range(self.n):
            dues = self._due_times(i, t0, now)
            self.counts["generated"] += len(dues)
            if self.cfg.mac == "tdma":
                # beacons and this vehicle's slots in time order (a beacon due at a slot's start makes that slot)
                events = sorted([(d, 0) for d in dues] + [(s, 1) for s in self._slot_starts(i, t0, now)])
                for t, kind in events:
                    if kind == 0:
                        if self._last_due[i] > self._last_sent_due[i]:
                            self.counts["no_slot"] += 1     # replaced by a newer one before its slot came
                        self._last_due[i] = t
                    elif self._last_due[i] > self._last_sent_due[i]:
                        starts.append((t, i))
                        self._last_sent_due[i] = self._last_due[i]
            else:
                for d in dues:
                    if self._pending[i] is not None:
                        self.counts["superseded"] += 1
                    self._pending[i] = d + float(self.rng.uniform(0, self.cfg.aloha_jitter)) * self.cfg.beacon_interval_s
                if self._pending[i] is not None:
                    s = max(self._pending[i], self._busy_until[i])
                    if s <= now + 1e-9:
                        starts.append((s, i))
                        self._pending[i] = None
        for s, i in sorted(starts):
            self._transmit(i, s, positions, make_payload(i))
        out = [[] for _ in range(self.n)]
        for j in range(self.n):
            keep = []
            for p in self._arrivals[j]:
                if p.deliver_at > now + 1e-9:
                    keep.append(p)
                    continue
                if p.fate == "pending" and self.cfg.half_duplex and any(
                        a < p.arrive_end and p.arrive_start < b for a, b in self._tx[j]):
                    p.fate = "half_duplex"
                if p.fate == "pending":
                    p.fate = "delivered"
                    out[j].append(p.payload)
                self.counts[p.fate] += 1
            self._arrivals[j] = keep
        horizon = now - 2 * (self.airtime + self.cfg.modem.processing_latency_s) - self.cfg.modem.max_range_m / \
            self.cfg.modem.speed_of_sound_mps - 1.0
        for i in range(self.n):
            self._tx[i] = [(a, b) for a, b in self._tx[i] if b >= horizon]
        return out

    # --- reporting ------------------------------------------------------------------------------------------------
    def utilisation(self) -> float:
        """Fraction of the elapsed time the water carried a transmission (summed over vehicles; above 1 = overlap)."""
        return self.airtime_used / self.now if self.now > 0 else 0.0

    def delivery_ratio(self) -> float:
        """Delivered / (sent x receivers) over the packets decided so far."""
        decided = sum(self.counts[k] for k in ("delivered", "lost", "collision", "half_duplex", "out_of_range"))
        return self.counts["delivered"] / decided if decided else float("nan")

    def describe(self) -> str:
        c, m = self.cfg, self.cfg.modem
        rate = "unlimited" if math.isinf(m.bit_rate_bps) else f"{m.bit_rate_bps:g} bit/s"
        s = (f"{c.mac.upper()}, {rate}, {c.beacon_bytes}-byte beacons every {c.beacon_interval_s:g} s "
             f"({self.airtime:.2f} s on the air each), modem delay {m.processing_latency_s:g} s, {c.loss.describe()}")
        if c.mac == "tdma":
            s += f"; cycle {self.cycle:.2f} s for {self.n} vehicles"
        return s
