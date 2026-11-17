"""
escort_remus.py -- the REMUS 100 a BlueROV2 escorts (heterogeneous escort study).

## The truth vehicle
Prestero's REMUS 100 (stress/prestero_remus.py: his 2001 MIT thesis, written from his published coefficients, not from
Strathe's model), flown by the autopilot every #97 Prestero mission flies under (stress/fin_truth.prestero_autopilot),
integrated with fourth-order Runge-Kutta at 100 Hz. Two changes, both written down here:

1. Trim. Prestero's vehicle is 7 N positive (B = 306 N, W = 299 N). At the speeds a BlueROV2 can keep up with
   (~0.7 m/s) that buoyancy beats the fins and the vehicle surfaces, so the escort vehicle is re-ballasted
   BUOYANCY_CHANGE_N lighter (1 N positive, a usual survey trim). The change enters exactly as a smaller B in his
   equations: buoyancy acts at his origin, so it adds force and no moment.
2. Depth. His autopilot takes a pitch order; a survey holds depth, so an outer loop turns depth error into the pitch
   order (DEPTH_GAIN, PITCH_ORDER_LIMIT_RAD).

## Missions
Survey legs at constant rpm and depth with a heading change of +90 or -90 degrees every 35-55 s (the first one 5-25 s
in, so most episode windows have a turn). At 700 rpm a 90 degree turn takes about 15 s; the vehicle slows from ~0.70
to ~0.5 m/s in it, holds depth within ~0.4 m and never saturates its elevator. A mission is a function of its seed only. The REMUS never reacts to the
ROV, so its whole trajectory is precomputed (MissionLibrary), which pairs every belief and policy on the same motion
and keeps training fast.

Stored per 0.1 s tick k (the state at the END of tick k; k = 0 is the start): position (NED, m), body->world rotation,
body velocity, and the fin and propeller command applied during tick k in Prestero's convention (delta_r, delta_s,
rpm). A beacon carries that command in Strathe's own control layout (controls_for), the zero-order-hold input a
receiver's model is driven with.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
_STRESS = os.path.join(HERE, "stress")
if _STRESS not in sys.path:
    sys.path.insert(0, _STRESS)
import prestero_remus as PR  # noqa: E402

BUOYANCY_CHANGE_N = -6.0        # re-ballasted from Prestero's +7 N to +1 N
DEPTH_GAIN = 0.2                # pitch order (rad) per metre of depth error
PITCH_ORDER_LIMIT_RAD = 0.3
AUTOPILOT_H = 0.01              # 100 Hz, as every #97 Prestero mission
FIN_LIMIT_RAD = 15 * math.pi / 180   # fin_truth.LIMIT
TURNING_RATE_RAD_S = 0.05       # |r| above this counts as turning in the straight/turn split


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def autopilot(pitch, q, yaw, r, psi_d, th_d, ii, h):
    """stress/fin_truth.prestero_autopilot, the same gains and limits (kept identical by test_escort.py)."""
    e_th = pitch - th_d
    ii = float(np.clip(ii + e_th * h, -.5, .5))
    ds = float(np.clip(1.5 * e_th + 1.2 * q + .4 * ii, -FIN_LIMIT_RAD, FIN_LIMIT_RAD))
    dr = float(np.clip(-1.2 * _wrap(psi_d - yaw) + 1.2 * r, -FIN_LIMIT_RAD, FIN_LIMIT_RAD))
    return ds, dr, ii


def derivative(x, delta_s, delta_r, rpm, buoyancy_change_n: float = BUOYANCY_CHANGE_N) -> np.ndarray:
    """Prestero's state derivative with buoyancy B + buoyancy_change_n."""
    d = PR.derivative(x, delta_s, delta_r, rpm)
    if buoyancy_change_n:
        k = -buoyancy_change_n             # his (W - B) grows by this much
        phi, th = x[9], x[10]
        f = np.array([-k * math.sin(th), k * math.cos(th) * math.sin(phi), k * math.cos(th) * math.cos(phi), 0, 0, 0])
        d[:6] += PR.MINV @ f
    return d


def rk4_step(x, delta_s, delta_r, rpm, h, buoyancy_change_n: float = BUOYANCY_CHANGE_N) -> np.ndarray:
    f = lambda y: derivative(y, delta_s, delta_r, rpm, buoyancy_change_n)
    k1 = f(x)
    k2 = f(x + .5 * h * k1)
    k3 = f(x + .5 * h * k2)
    k4 = f(x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


@dataclass(frozen=True)
class MissionConfig:
    rpm: float = 700.0                       # ~0.70 m/s on a straight leg, ~0.5 m/s in a turn, at the escort trim
    depth_m: float = 10.0
    duration_s: float = 120.0
    leg_s: tuple = (35.0, 55.0)              # a 90 degree turn takes ~15 s at this speed, so >= 20 s of straight leg
    first_turn_s: tuple = (5.0, 25.0)
    turns_deg: tuple = (90.0, -90.0)
    buoyancy_change_n: float = BUOYANCY_CHANGE_N
    dt: float = 0.1

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


@dataclass
class Mission:
    seed: int
    t: np.ndarray            # (N,)
    pos: np.ndarray          # (N, 3) NED
    R: np.ndarray            # (N, 3, 3) body -> world
    nu: np.ndarray           # (N, 6) body velocity
    fins: np.ndarray         # (N, 3) Prestero delta_r, delta_s (rad), rpm applied during tick k
    heading_order: np.ndarray  # (N,) rad

    def __len__(self):
        return len(self.t)

    def world_velocity(self, k: int) -> np.ndarray:
        return self.R[k] @ self.nu[k, :3]

    def turning(self, k: int) -> bool:
        return abs(float(self.nu[k, 5])) > TURNING_RATE_RAD_S


def fly_mission(seed: int, cfg: MissionConfig = MissionConfig()) -> Mission:
    rng = np.random.default_rng(seed)
    h = AUTOPILOT_H
    every = int(round(cfg.dt / h))
    u0 = 1.54 * cfg.rpm / 1500.0     # his propeller term's steady speed in still water (Xuu u^2 = thrust)
    x = np.zeros(12)
    x[0], x[8] = u0, cfg.depth_m
    psi = float(rng.uniform(-math.pi, math.pi))
    x[11] = psi
    nxt = float(rng.uniform(*cfg.first_turn_s))
    ii, ds, dr = 0.0, 0.0, 0.0
    n_ticks = int(round(cfg.duration_s / cfg.dt)) + 1
    T, P, Rm, NU, F, H = [], [], [], [], [], []

    def record(k):
        T.append(k * cfg.dt)
        P.append(x[6:9].copy())
        Rm.append(Rotation.from_euler("ZYX", [x[11], x[10], x[9]]).as_matrix())
        NU.append(x[:6].copy())
        F.append((dr, ds, cfg.rpm))
        H.append(psi)

    record(0)
    for k in range(1, n_ticks):
        for _ in range(every):
            t = ((k - 1) * every + _) * h
            if t >= nxt:
                psi = _wrap(psi + math.radians(float(rng.choice(cfg.turns_deg))))
                nxt = t + float(rng.uniform(*cfg.leg_s))
            th_d = float(np.clip(DEPTH_GAIN * (x[8] - cfg.depth_m), -PITCH_ORDER_LIMIT_RAD, PITCH_ORDER_LIMIT_RAD))
            ds, dr, ii = autopilot(x[10], x[4], x[11], x[5], psi, th_d, ii, h)
            x = rk4_step(x, ds, dr, cfg.rpm, h, cfg.buoyancy_change_n)
        if not np.all(np.isfinite(x)):
            raise FloatingPointError(f"REMUS mission {seed} diverged at tick {k}")
        record(k)
    return Mission(seed, np.array(T), np.array(P), np.array(Rm), np.array(NU), np.array(F), np.array(H))


def controls_basis(act) -> tuple:
    """Strathe's controls for unit (rudder, elevator, rpm) in Strathe's signs: controls are linear in the channels,
    so the mix is read off once (as stress/fin_truth.fly_model_on_autopilot does)."""
    import logging
    import pandas as pd
    import steering
    logging.getLogger(steering.__name__).setLevel(logging.ERROR)   # the no-aileron note is expected: REMUS logs none
    return tuple(steering.controls_from_frame(pd.DataFrame({"fin_rudder": [a], "fin_elevator": [b], "rpm": [c]}), act)[0][0]
                 for a, b, c in ((1, 0, 0), (0, 1, 0), (0, 0, 1)))


def controls_for(basis, dr: float, ds: float, rpm: float) -> np.ndarray:
    """Prestero's (delta_r, delta_s, rpm) as Strathe controls: his +delta_r yaws to port and +delta_s pushes the nose
    down, the opposite of Strathe's channels (fin_truth._frame's signs)."""
    br, be, bn = (np.asarray(b, dtype=float) for b in basis)
    return -dr * br - ds * be + rpm * bn


class MissionLibrary:
    """Precomputed missions, cached as one .npz per (config, seeds)."""

    def __init__(self, missions: Sequence[Mission], cfg: MissionConfig):
        self.missions = list(missions)
        self.cfg = cfg

    def __len__(self):
        return len(self.missions)

    def __getitem__(self, i) -> Mission:
        return self.missions[i]

    @staticmethod
    def cache_path(cache_dir: str, cfg: MissionConfig, seeds: Sequence[int]) -> str:
        h = hashlib.sha256((cfg.key() + ",".join(map(str, seeds))).encode()).hexdigest()[:12]
        return os.path.join(cache_dir, f"remus_missions_{h}.npz")

    @classmethod
    def build(cls, seeds: Sequence[int], cfg: MissionConfig = MissionConfig(), cache_dir: Optional[str] = None,
              workers: int = 1) -> "MissionLibrary":
        seeds = [int(s) for s in seeds]
        path = cls.cache_path(cache_dir, cfg, seeds) if cache_dir else None
        if path and os.path.exists(path):
            return cls.load(path, cfg)
        if workers > 1:
            from multiprocessing import Pool
            with Pool(workers) as pool:
                missions = pool.starmap(fly_mission, [(s, cfg) for s in seeds])
        else:
            missions = [fly_mission(s, cfg) for s in seeds]
        lib = cls(missions, cfg)
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            lib.save(path)
        return lib

    def save(self, path: str) -> None:
        m = self.missions
        np.savez_compressed(path, seeds=np.array([x.seed for x in m]), t=np.stack([x.t for x in m]),
                            pos=np.stack([x.pos for x in m]), R=np.stack([x.R for x in m]), nu=np.stack([x.nu for x in m]),
                            fins=np.stack([x.fins for x in m]), heading_order=np.stack([x.heading_order for x in m]),
                            config=json.dumps(asdict(self.cfg)))

    @classmethod
    def load(cls, path: str, cfg: Optional[MissionConfig] = None) -> "MissionLibrary":
        d = np.load(path)
        stored = MissionConfig(**{k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(str(d["config"])).items()})
        if cfg is not None and stored != cfg:
            raise ValueError(f"{path} was built for {stored}, not {cfg}")
        # each d[key] decompresses the whole stacked array, so read every array once: indexing d[key] per mission
        # kept one full copy alive per mission (400 missions held ~35 GB, not 88 MB)
        a = {k: d[k] for k in ("seeds", "t", "pos", "R", "nu", "fins", "heading_order")}
        missions = [Mission(int(s), a["t"][i], a["pos"][i], a["R"][i], a["nu"][i], a["fins"][i], a["heading_order"][i])
                    for i, s in enumerate(a["seeds"])]
        return cls(missions, stored)


TRAIN_SEEDS = range(0, 400)          # training missions
EVAL_SEEDS = range(100_000, 100_100)  # held-out missions: never trained on
