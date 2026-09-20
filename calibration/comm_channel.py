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

## Physical model (formula mode — see the Mera product vision doc)
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


class CommChannel:
    """
    A0.3/A0.6/A0.7 — the channel itself. Per-receiver delay queues (min-
    heaps on arrival_time), a per-tick delivered buffer, and the three
    physical effects (latency, packet loss, bandwidth) composed together
    in send()/step().

    Usage per tick:
        for each agent pair that wants to communicate this tick:
            channel.send(sender_id, receiver_id, current_time,
                         sender_pos, receiver_pos, message_type, ...)
        channel.step(current_time)   # advances delivery, applies bandwidth
        for each agent:
            msgs = channel.receive(agent_id)   # may be [] — not an error
    """

    def __init__(self, packet_loss_config: PacketLossConfig,
                 bandwidth_config: BandwidthConfig,
                 speed_of_sound_mps: float = SPEED_OF_SOUND_MPS,
                 seed: Optional[int] = None):
        packet_loss_config.validate()
        bandwidth_config.validate()
        self.packet_loss_config = packet_loss_config
        self.bandwidth_config = bandwidth_config
        self.speed_of_sound_mps = speed_of_sound_mps
        self._rng = np.random.default_rng(seed)

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

    def send(self, sender_id: str, receiver_id: str, send_time: float,
              sender_pos, receiver_pos, message_type: str,
              payload: Any = None, payload_size_bytes: int = 0) -> Optional[Message]:
        """A0.2 + A0.5 composed: draws packet loss first (at the real
        physical distance, at send time); if the message survives,
        computes latency and enqueues it. Returns the Message if it was
        sent into the channel, or None if it was dropped outright at send
        time — the caller does not need to distinguish "lost in transit"
        from "still in flight," since receive() is the only place delivery
        is ever observed."""
        distance = float(np.linalg.norm(np.asarray(sender_pos, dtype=float)
                                         - np.asarray(receiver_pos, dtype=float)))
        drop_prob = self.packet_loss_config.drop_probability(distance)
        if self._rng.random() < drop_prob:
            return None

        latency = distance / self.speed_of_sound_mps
        msg = Message(
            sender_id=sender_id, receiver_id=receiver_id, message_type=message_type,
            send_time=send_time, arrival_time=send_time + latency,
            payload=payload, payload_size_bytes=payload_size_bytes,
        )
        msg.validate()

        heapq.heappush(
            self._queues.setdefault(receiver_id, []),
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