"""
A0.8 / A0.9 — Track A0's test suite, run directly (no pytest available in
this sandbox — confirmed, not assumed: `pip install pytest` failed outright,
no network access here). Plain assert-based functions, run via __main__
at the bottom. Every test constructs positions as hand-written literal
arrays only (A0.9) — nothing here ever imports or references a real
vehicle model.
"""
import numpy as np

from comm_channel import (
    CommChannel, PacketLossConfig, BandwidthConfig,
    latency_seconds, MESSAGE_TYPE_POSITION_BEACON,
)


def test_zero_distance_latency():
    lat = latency_seconds([0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    assert lat == 0.0, f"expected exactly 0.0, got {lat}"


def test_known_distance_latency():
    # distance = 1500 m along x -> latency should be exactly 1.0s
    lat = latency_seconds([0.0, 0.0, 0.0], [1500.0, 0.0, 0.0])
    assert lat == 1.0, f"expected exactly 1.0, got {lat}"

    # a second, non-round-number case, checked against the formula directly
    pos_a = [10.0, -5.0, 2.0]
    pos_b = [37.0, 12.0, -8.0]
    expected = float(np.linalg.norm(np.array(pos_b) - np.array(pos_a))) / 1500.0
    lat2 = latency_seconds(pos_a, pos_b)
    assert lat2 == expected, f"expected {expected}, got {lat2}"


def test_bandwidth_drop_and_reset():
    # Zero packet loss so this test isolates bandwidth behavior only.
    no_loss = PacketLossConfig(near_distance_m=0.0, far_distance_m=1000.0,
                                near_field_prob=0.0, far_field_prob=0.0)
    tiny_budget = BandwidthConfig(bytes_per_tick=100.0)
    channel = CommChannel(no_loss, tiny_budget, seed=0)

    pos_a, pos_b = [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]

    # Tick 1: a payload bigger than the whole budget -> must be dropped.
    channel.send("a", "b", send_time=0.0, sender_pos=pos_a, receiver_pos=pos_b,
                 message_type=MESSAGE_TYPE_POSITION_BEACON,
                 payload="too_big", payload_size_bytes=500)
    channel.step(current_time=1.0)  # latency here is tiny (10/1500s), so it's due by t=1.0
    delivered_tick1 = channel.receive("b")
    assert delivered_tick1 == [], f"expected the oversized message to be dropped, got {delivered_tick1}"

    # Tick 2: confirm the budget genuinely resets — a small payload, well
    # within budget, must go through cleanly on the NEXT tick.
    channel.send("a", "b", send_time=1.0, sender_pos=pos_a, receiver_pos=pos_b,
                 message_type=MESSAGE_TYPE_POSITION_BEACON,
                 payload="fits", payload_size_bytes=50)
    channel.step(current_time=2.0)
    delivered_tick2 = channel.receive("b")
    assert len(delivered_tick2) == 1, f"expected exactly 1 delivered message, got {delivered_tick2}"
    assert delivered_tick2[0].payload == "fits"


def test_packet_loss_empirical_rate():
    # Constant drop probability (near==far) so distance doesn't confound this.
    const_loss = PacketLossConfig(near_distance_m=0.0, far_distance_m=1000.0,
                                   near_field_prob=0.3, far_field_prob=0.3)
    generous_bandwidth = BandwidthConfig(bytes_per_tick=1e9)
    channel = CommChannel(const_loss, generous_bandwidth, seed=42)

    pos_a, pos_b = [0.0, 0.0, 0.0], [100.0, 0.0, 0.0]
    n = 5000
    n_sent_into_channel = 0
    for i in range(n):
        result = channel.send("a", "b", send_time=float(i), sender_pos=pos_a,
                               receiver_pos=pos_b, message_type=MESSAGE_TYPE_POSITION_BEACON,
                               payload=i, payload_size_bytes=1)
        if result is not None:
            n_sent_into_channel += 1

    observed_drop_rate = 1.0 - (n_sent_into_channel / n)
    assert abs(observed_drop_rate - 0.3) < 0.02, (
        f"expected drop rate near 0.30, observed {observed_drop_rate:.4f} over {n} trials"
    )


def test_ordering_by_arrival_not_send():
    no_loss = PacketLossConfig(near_distance_m=0.0, far_distance_m=10000.0,
                                near_field_prob=0.0, far_field_prob=0.0)
    generous_bandwidth = BandwidthConfig(bytes_per_tick=1e9)
    channel = CommChannel(no_loss, generous_bandwidth, seed=0)

    origin = [0.0, 0.0, 0.0]
    far_point = [3000.0, 0.0, 0.0]   # latency = 3000/1500 = 2.0s
    near_point = [150.0, 0.0, 0.0]   # latency = 150/1500 = 0.1s

    # Sent FIRST, arrives LATER (arrival_time = 0.0 + 2.0 = 2.0).
    msg_far = channel.send("a", "b", send_time=0.0, sender_pos=origin,
                            receiver_pos=far_point, message_type=MESSAGE_TYPE_POSITION_BEACON,
                            payload="sent_first_arrives_second")
    # Sent SECOND, arrives EARLIER (arrival_time = 0.5 + 0.1 = 0.6).
    msg_near = channel.send("a", "b", send_time=0.5, sender_pos=origin,
                             receiver_pos=near_point, message_type=MESSAGE_TYPE_POSITION_BEACON,
                             payload="sent_second_arrives_first")

    assert msg_far.arrival_time == 2.0
    assert msg_near.arrival_time == 0.6

    # At t=0.6, only the second-sent (near) message should be due.
    channel.step(current_time=0.6)
    delivered_early = channel.receive("b")
    assert len(delivered_early) == 1
    assert delivered_early[0].payload == "sent_second_arrives_first"

    # At t=2.0, the first-sent (far) message finally arrives.
    channel.step(current_time=2.0)
    delivered_late = channel.receive("b")
    assert len(delivered_late) == 1
    assert delivered_late[0].payload == "sent_first_arrives_second"


def test_empty_receive_is_not_an_error():
    no_loss = PacketLossConfig(0.0, 1000.0, 0.0, 0.0)
    bw = BandwidthConfig(bytes_per_tick=1000.0)
    channel = CommChannel(no_loss, bw, seed=0)
    # Nothing was ever sent to "ghost" — receive() must return [], not raise.
    result = channel.receive("ghost")
    assert result == []


def test_zero_physics_dependency():
    """A0.9 — structural confirmation that this module is genuinely
    vehicle-agnostic: it must never IMPORT anything from the physics
    layer. Checked via the actual parsed AST import statements, not a
    crude substring search — a first attempt at this check used
    substring matching and produced a false positive, flagging the
    module's own docstring (which names dynamics_model.py/arena_core.py
    specifically to explain why they're NOT imported) as if it were a
    real import. Caught by running the test, not assumed correct because
    it looked reasonable — fixed by checking the parsed import
    statements themselves, which is what A0.9 actually cares about."""
    import ast
    import inspect
    import comm_channel as cc

    src = inspect.getsource(cc)
    tree = ast.parse(src)

    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_modules.add(node.module.split(".")[0])

    forbidden_modules = {"dynamics_model", "arena_core", "arena"}
    violations = imported_modules & forbidden_modules
    assert not violations, f"comm_channel.py must never IMPORT {violations} — found in actual import statements"


if __name__ == "__main__":
    tests = [
        test_zero_distance_latency,
        test_known_distance_latency,
        test_bandwidth_drop_and_reset,
        test_packet_loss_empirical_rate,
        test_ordering_by_arrival_not_send,
        test_empty_receive_is_not_an_error,
        test_zero_physics_dependency,
    ]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        raise SystemExit(1)