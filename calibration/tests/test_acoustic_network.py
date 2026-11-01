"""The shared acoustic channel (acoustic_network.py, PROJECT_STATUS.md #89)."""
import math

import numpy as np
import pytest

from acoustic_network import AcousticNetwork, LossModel, ModemConfig, NetworkConfig, tdma_guard_s
from comm_channel import PacketLossConfig


def _run(net, positions, seconds, dt=0.1):
    got = [[] for _ in range(net.n)]
    for k in range(1, int(round(seconds / dt)) + 1):
        t = k * dt
        out = net.advance(t, positions, lambda i, _t=t: (i, _t))
        for j in range(net.n):
            got[j] += [(t, p) for p in out[j]]
    return got


def test_airtime_is_the_preamble_plus_the_bits():
    m = ModemConfig.preset("slow")
    assert m.airtime_s(32) == pytest.approx(1.0 + 256 / 80)
    assert ModemConfig.preset("ideal").airtime_s(32) == 0.0
    with pytest.raises(ValueError, match="unknown modem preset"):
        ModemConfig.preset("whoi")


def test_latency_is_airtime_plus_travel_time_plus_the_modems_delay():
    m = ModemConfig(bit_rate_bps=256.0, packet_overhead_s=0.5, processing_latency_s=2.0)
    net = AcousticNetwork(2, NetworkConfig(modem=m, beacon_interval_s=10.0), seed=0, max_distance_m=1500)
    pos = [np.zeros(3), np.array([1500.0, 0, 0])]
    got = _run(net, pos, 40, dt=0.01)
    t_rx, (sender, t_tx) = got[1][0]
    assert sender == 0
    # vehicle 0 sends at its first slot after its first beacon falls due (t = 10 s)
    expected = 10.0 + 1.5 + 1.0 + 2.0   # airtime (0.5 + 256 bits / 256 bit/s) + 1500 m / 1500 m/s + decoding
    assert expected <= t_rx < expected + 0.011 + net.cycle


def test_tdma_never_collides_and_its_cycle_is_the_update_interval():
    cfg = NetworkConfig(modem=ModemConfig.preset("slow"), mac="tdma", beacon_interval_s=2.0)
    pos = [np.array([20.0 * i, 5.0 * (i % 2), 0]) for i in range(6)]
    net = AcousticNetwork(6, cfg, seed=3, max_distance_m=200)
    got = _run(net, pos, 600)
    assert net.counts["collision"] == net.counts["half_duplex"] == 0
    assert net.delivery_ratio() == 1.0
    assert net.cycle == pytest.approx(6 * (cfg.airtime_s + tdma_guard_s(200)))
    times = [t for t, (s, _) in got[0] if s == 1]
    assert np.diff(times).mean() == pytest.approx(net.cycle, rel=0.02)   # one beacon per cycle, not per 2 s
    assert net.counts["no_slot"] > 0                                      # the beacons that had no slot are counted


def test_random_access_collides_when_the_channel_is_busy():
    pos = [np.array([10.0 * i, 0, 0]) for i in range(4)]
    busy = AcousticNetwork(4, NetworkConfig(modem=ModemConfig.preset("slow"), mac="aloha", beacon_interval_s=2.0),
                           seed=1, max_distance_m=50)
    _run(busy, pos, 300)
    assert busy.counts["collision"] > 0 and busy.counts["superseded"] > 0 and busy.delivery_ratio() < 0.2
    light = AcousticNetwork(4, NetworkConfig(modem=ModemConfig.preset("fast"), mac="aloha", beacon_interval_s=10.0),
                            seed=1, max_distance_m=50)
    _run(light, pos, 300)
    assert light.delivery_ratio() > 0.8


def test_a_modem_cant_hear_while_it_transmits():
    m = ModemConfig(bit_rate_bps=math.inf, packet_overhead_s=1.0)
    net = AcousticNetwork(2, NetworkConfig(modem=m, mac="aloha", beacon_interval_s=10.0, aloha_jitter=0.0), seed=0)
    net._phase = [5.0, 5.05]          # both transmit within 0.05 s of each other, every 10 s
    _run(net, [np.zeros(3), np.array([1.0, 0, 0])], 34)
    assert net.counts["half_duplex"] == net.counts["sent"] and net.counts["delivered"] == 0


def test_burst_losses_have_the_asked_mean_and_burst_length():
    def runs(model, n=40000):
        net = AcousticNetwork(2, NetworkConfig(loss=model), seed=5)
        lost = [net._lost(0, 1, 10.0) for _ in range(n)]
        lengths, k = [], 0
        for x in lost:
            if x:
                k += 1
            elif k:
                lengths.append(k)
                k = 0
        return np.mean(lost), np.mean(lengths)
    mean, burst = runs(LossModel(0.3, 8.0))
    assert mean == pytest.approx(0.3, abs=0.03) and burst == pytest.approx(8.0, rel=0.15)
    mean, burst = runs(LossModel(0.3))
    assert mean == pytest.approx(0.3, abs=0.01) and burst == pytest.approx(1 / 0.7, rel=0.05)
    with pytest.raises(ValueError, match="can't have bursts as short"):
        LossModel(0.9, 2.0).validate()


def test_range_and_distance_loss():
    m = ModemConfig(max_range_m=100.0)
    net = AcousticNetwork(2, NetworkConfig(modem=m, beacon_interval_s=1.0), seed=0, max_distance_m=200)
    _run(net, [np.zeros(3), np.array([150.0, 0, 0])], 20)
    assert net.counts["out_of_range"] > 0 and net.counts["delivered"] == 0
    ramp = LossModel(0.0, distance=PacketLossConfig(10.0, 100.0, 0.0, 1.0))
    net = AcousticNetwork(2, NetworkConfig(loss=ramp, beacon_interval_s=1.0), seed=0, max_distance_m=200)
    _run(net, [np.zeros(3), np.array([55.0, 0, 0])], 400)
    assert net.delivery_ratio() == pytest.approx(0.5, abs=0.08)


def test_the_same_seed_loses_the_same_packets():
    cfg = NetworkConfig(modem=ModemConfig.preset("medium"), mac="aloha", loss=LossModel(0.2, 3.0))
    pos = [np.array([15.0 * i, 0, 0]) for i in range(3)]
    a, b = AcousticNetwork(3, cfg, seed=9, max_distance_m=50), AcousticNetwork(3, cfg, seed=9, max_distance_m=50)
    assert _run(a, pos, 120) == _run(b, pos, 120) and a.counts == b.counts


def test_the_channel_knows_nothing_about_physics():
    import acoustic_network
    src = open(acoustic_network.__file__, encoding="utf-8").read()
    for name in ("dynamics_model", "fast_dynamics", "arena", "mujoco"):
        assert f"import {name}" not in src and f"from {name}" not in src
