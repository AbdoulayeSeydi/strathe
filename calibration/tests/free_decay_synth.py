"""Synthetic free-decay tank tests with known answers, for test_free_decay.py. The truth is integrated with scipy's
adaptive solver at tight tolerances, independently of free_decay.py's own fixed-step integrator."""
import math

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation

G = 9.81
TRUTH = {"mass_kg": 11.2, "Ieff_roll": 0.26, "Ieff_pitch": 0.23,
         "Kp": -0.6, "Kpp": -1.2, "Mq": -0.9, "Mqq": -0.8, "cg_cb_offset_z": 0.018,
         "rest_roll_deg": 1.5, "rest_pitch_deg": -2.0}


def _decay(t, angle0, I, K, Kq, zg, rest, W, tugs=()):
    """tugs: (start s, duration s, torque N*m) kicks from outside (a tether), unknown to the fit."""
    def f(_t, y):
        a, w = y
        ext = sum(m for t0, d, m in tugs if t0 <= _t < t0 + d)
        return [w, (K * w + Kq * abs(w) * w - zg * W * math.sin(a - rest) + ext) / I]
    sol = solve_ivp(f, (0, t[-1]), [angle0, 0.0], t_eval=t, rtol=1e-10, atol=1e-12, method="DOP853",
                    max_step=0.02 if tugs else np.inf)
    return sol.y[0], sol.y[1]


def tank_test(axis="roll", starts_deg=(10, -15, 25, 18), rate_hz=10.0, noise_deg=0.15, gyro_noise=0.01,
              truth=TRUTH, seed=0, regrab=True, driving_s=5.0, hold_s=1.5, decay_s=12.0, wobble_deg=0.0,
              inertia=None, tug_nm=0.0):
    """One recording: driving with thrust, then for each start angle: (a hand tilts it over 1 s, holds it,) lets
    go, and it swings back with the thrusters stopped. wobble_deg adds motion the vehicle's own physics doesn't
    explain; tug_nm adds random torque kicks of about that size during the swings (a tether tugging); inertia overrides the true inertia (the spec's stays as it is). The sample index of
    each release is in df.attrs["releases"]."""
    rng = np.random.default_rng(seed)
    W = truth["mass_kg"] * G
    if axis == "roll":
        I, K, Kq, rest = truth["Ieff_roll"], truth["Kp"], truth["Kpp"], math.radians(truth["rest_roll_deg"])
    else:
        I, K, Kq, rest = truth["Ieff_pitch"], truth["Mq"], truth["Mqq"], math.radians(truth["rest_pitch_deg"])
    if inertia is not None:
        I = inertia
    dt = 1 / rate_hz
    ang, rate, wrench, releases = [], [], [], []
    n = int(driving_s * rate_hz)  # driving: thrust on, a small wobble
    ang += list(rest + 0.02 * np.sin(np.arange(n) * dt * 2)); rate += list(0.04 * np.cos(np.arange(n) * dt * 2))
    wrench += [rng.normal(0, 8, 6) for _ in range(n)]
    cur = rest
    for s in starts_deg:
        target = rest + math.radians(s)
        if regrab:  # a hand tilts it over 1 s (thrusters stopped), then holds it
            m = int(1.0 * rate_hz)
            ramp = cur + (target - cur) * (1 - np.cos(np.linspace(0, math.pi, m))) / 2
            ang += list(ramp); rate += list(np.gradient(ramp, dt)); wrench += [np.zeros(6)] * m
        h = int(hold_s * rate_hz)
        ang += [target] * h; rate += [0.0] * h; wrench += [np.zeros(6)] * h
        releases.append(len(ang))
        t = np.arange(int(decay_s * rate_hz)) * dt
        tugs = []
        if tug_nm:
            for t0 in np.arange(0.3, t[-1], 0.8):
                if rng.random() < 0.6:
                    tugs.append((t0, 0.3, tug_nm * rng.uniform(-1, 1)))
        a, w = _decay(t, target, I, K, Kq, truth["cg_cb_offset_z"], rest, W, tugs)
        ang += list(a); rate += list(w); wrench += [np.zeros(6)] * len(t)
        cur = a[-1]
    ang, rate = np.array(ang), np.array(rate)
    if wobble_deg:
        tt = np.arange(len(ang)) * dt
        ang = ang + math.radians(wobble_deg) * np.sin(2 * math.pi * 0.7 * tt)
        rate = rate + math.radians(wobble_deg) * 2 * math.pi * 0.7 * np.cos(2 * math.pi * 0.7 * tt)
    ang = ang + rng.normal(0, math.radians(noise_deg), len(ang))
    rate = rate + rng.normal(0, gyro_noise, len(rate))
    roll = ang if axis == "roll" else np.zeros(len(ang))
    pitch = ang if axis == "pitch" else np.zeros(len(ang))
    q = Rotation.from_euler("xyz", np.column_stack([roll, pitch, np.full(len(ang), 0.7)])).as_quat()
    df = pd.DataFrame(np.array(wrench), columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
    for c in ("Bu", "Bv", "Bw", "Br"):
        df[c] = rng.normal(0, 0.005, len(df))
    df["Bp"] = rate if axis == "roll" else rng.normal(0, gyro_noise, len(df))
    df["Bq"] = rate if axis == "pitch" else rng.normal(0, gyro_noise, len(df))
    df[["qx", "qy", "qz", "qw"]] = q
    df = df[["Fx", "Fy", "Fz", "Tx", "Ty", "Tz", "Bu", "Bv", "Bw", "Bp", "Bq", "Br", "qx", "qy", "qz", "qw"]].copy()
    df.attrs["releases"] = releases
    return df
