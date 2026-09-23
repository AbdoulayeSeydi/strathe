#!/usr/bin/env python3
"""
verify_holoocean_frame_conventions.py -- two measurements to run on a machine
with HoloOcean installed (Windows or Linux) BEFORE trusting any new
HoloOcean-backed calibration or Claim A result.

    python verify_holoocean_frame_conventions.py

1. Tick length. holoocean_backend sends one acceleration command per 0.1 s
   logged sample, so each env.step() must advance 0.1 s of simulated time.
   HoloOcean's default is 30 ticks/s. Measured here from rest: command a
   constant world-frame acceleration a for N steps; speed = a * N * tick.

2. Rotating-frame transport term. A vehicle moving forward at u while
   yawing at r, with ZERO body-frame acceleration, must keep u constant in
   its own frame. That needs a world-frame command R @ (omega x v); a zero
   command lets the world-frame velocity stay put while the body turns, so
   surge leaks into sway. Both commands are applied; the one that keeps the
   body velocity constant is the correct convention, and holoocean_backend's
   transport_term default (True) should match it.

Prints PASS/FAIL for each and exits non-zero on any FAIL.
"""
from __future__ import annotations

import sys

import numpy as np

import holoocean_backend as hb

DATA_DT = 0.1


def measure_tick(env, agent, n=30, a=1.0) -> float:
    agent.set_physics_state(location=[0, 0, -5], rotation=[0, 0, 0], velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
    state = None
    for _ in range(n):
        state = env.step(np.array([a, 0, 0, 0, 0, 0], dtype=float))
    speed = float(np.asarray(state["VelocitySensor"])[0])
    return speed / (a * n)


def turning_drift(env, agent, transport: bool, n=30, u=0.5, r=0.5) -> np.ndarray:
    """Max |body velocity - initial| over n steps, with zero body acceleration commanded."""
    agent.set_physics_state(location=[0, 0, -5], rotation=[0, 0, 0], velocity=[u, 0, 0],
                            angular_velocity=hb._BODY_FRAME_CORRECTION @ np.array([0, 0, r]))
    v = np.array([u, 0, 0, 0, 0, r])
    R = np.eye(3)
    worst = np.zeros(6)
    for _ in range(n):
        lin = np.cross(v[3:], v[:3]) if transport else np.zeros(3)
        state = env.step(np.concatenate([R @ lin, np.zeros(3)]))
        R, _ = hb._pose_to_rotation_and_position(state["PoseSensor"])
        v = hb._extract_body_velocity(state, R)
        worst = np.maximum(worst, np.abs(v - np.array([u, 0, 0, 0, 0, r])))
    return worst


def main() -> int:
    env = hb.make_deterministic_environment()
    agent = env.agents["auv0"]
    agent.set_control_scheme(hb.CUSTOM_DYNAMICS_SCHEME)
    failures = 0

    tick = measure_tick(env, agent)
    ok = abs(tick - DATA_DT) < 0.05 * DATA_DT
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] one env.step() = {tick:.4f} s simulated "
          f"({1 / tick:.1f} ticks/s); needs {DATA_DT} s to match the 10 Hz logs")

    with_term = turning_drift(env, agent, transport=True)
    without = turning_drift(env, agent, transport=False)
    print(f"        body-velocity drift over 3 s of turning, WITH R(w x v):    {np.round(with_term, 4)}")
    print(f"        body-velocity drift over 3 s of turning, WITHOUT the term: {np.round(without, 4)}")
    ok = with_term[:3].max() < 0.2 * without[:3].max()
    failures += not ok
    print(f"[{'PASS' if ok else 'FAIL'}] transport term {'keeps' if ok else 'does NOT keep'} a turning vehicle's "
          f"body velocity constant -- holoocean_backend's transport_term=True is "
          f"{'correct' if ok else 'WRONG for this HoloOcean build; investigate before calibrating'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
