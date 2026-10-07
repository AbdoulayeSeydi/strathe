"""Synthetic link logs drawn from Strathe's own channel models with known settings, for test_link_fit.py."""
import numpy as np
import pandas as pd

from comm_channel import BandwidthConfig, CommChannel, OpticalChannel, OpticalLinkConfig, PacketLossConfig


def acoustic_log(n=1500, plc=PacketLossConfig(50.0, 400.0, 0.05, 0.6), speed=1480.0, fixed=2.5,
                 ranges=(5.0, 600.0), seed=0, jitter_s=0.05):
    rng = np.random.default_rng(seed)
    ch = CommChannel(plc, BandwidthConfig(1e12), speed_of_sound_mps=speed, seed=seed + 1, fixed_latency_s=fixed)
    rows = []
    for i in range(n):
        d = float(rng.uniform(*ranges))
        t = 10.0 * i
        m = ch.send("a", "b", t, (0, 0, 0), (d, 0, 0), "position_beacon")
        rows.append({"send_time_s": t, "receive_time_s": (m.arrival_time + rng.normal(0, jitter_s)) if m else np.nan,
                     "range_m": d, "link": "acoustic"})
    return pd.DataFrame(rows)


def optical_log(n=1500, water="coastal", rated=30.0, width=1.0, latency=0.002, ranges=(1.0, 40.0), seed=0,
                attenuation=None):
    rng = np.random.default_rng(seed)
    cfg = OpticalLinkConfig(water=water, attenuation_per_m=attenuation, rated_range_m=rated, curve_width=width,
                            latency_s=latency)
    ch = OpticalChannel(cfg, 0.1, seed=seed + 1)
    rows = []
    for i in range(n):
        d = float(rng.uniform(*ranges))
        t = 0.1 * i
        m = ch.send("a", "b", t, (0, 0, 0), (d, 0, 0), "position_beacon")
        rows.append({"send_time_s": t, "receive_time_s": m.arrival_time if m else np.nan, "range_m": d,
                     "link": "optical"})
    return pd.DataFrame(rows), cfg
