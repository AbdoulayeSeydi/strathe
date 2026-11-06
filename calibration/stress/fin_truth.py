"""
stress/fin_truth.py -- recordings from vehicles Strathe wasn't written from, for the fin proof (PROJECT_STATUS.md #97).

Each engine flies closed-loop missions, as a real vehicle does (Prestero's REMUS is unstable left alone), and the commands and motion go
into Strathe's prepared layout. Nothing else about the engine reaches the fit: it gets the vehicle's size and fins, as a lab would.

    prestero   stress/prestero_remus.py, from Prestero's 2001 thesis: lumped lift, off-diagonal mass, flow-driven fin force
    fossen     Fossen's remus100, run unmodified (MIT; pinned commit, FOSSEN_SRC). Its fin force has no flow angle
    holoocean  HoloOcean's Torpedo class, run unmodified (HOLOOCEAN_SRC). Four independent fins

Missions: rich (steps of heading, pitch and speed), survey (straight lines, a turn every 150 s, a pitch sawtooth), or mixed.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import prestero_remus as PR  # noqa: E402

D2R = math.pi / 180
LIMIT = 15 * D2R
H = os.path.expanduser("~")
FOSSEN_SRC = os.environ.get("FOSSEN_SRC", f"{H}/strathe_fin_data/fossen_src/PythonVehicleSimulator-a93d1938ba1f78f2580c24c73f003091b2f03cd0/src")
HOLOOCEAN_SRC = os.environ.get("HOLOOCEAN_SRC", f"{H}/strathe_fin_data/holoocean_src")
TRAIN_DATE, HELD_DATE = "01-10-2026", "02-10-2026"


def available(engine: str) -> bool:
    return engine == "prestero" or os.path.isdir({"fossen": FOSSEN_SRC, "holoocean": HOLOOCEAN_SRC}[engine])


class Plan:
    """What a mission asks for at time t: heading, pitch and rpm."""
    def __init__(self, kind: str, seed: int, rpm: float = 1300.0):
        self.kind, self.rng, self.rpm = kind, np.random.default_rng(seed), rpm
        self.next, self.psi, self.theta, self.n = 0.0, 0.0, 0.0, rpm

    def __call__(self, t: float) -> tuple:
        k = self.kind if self.kind != "mixed" else ("rich" if (t // 300) % 2 else "survey")
        if k == "rich":
            if t >= self.next:
                self.psi += self.rng.uniform(-70, 70) * D2R
                self.theta = self.rng.uniform(-12, 12) * D2R
                self.n = self.rpm * self.rng.uniform(.75, 1.15)
                self.next = t + self.rng.uniform(6, 16)
        else:
            if t >= self.next:
                self.psi += self.rng.choice([90, 180, -90]) * D2R
                self.next = t + 150
            self.theta = (8 if (t % 120) < 60 else -8) * D2R
            self.n = self.rpm
        return self.psi, self.theta, self.n


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _frame(rows: list, rudder_sign: float, stern_sign: float) -> pd.DataFrame:
    """rows: roll pitch yaw depth u v w p q r rudder stern rpm. Rudder and elevator signed as Strathe's: positive turns to starboard,
    positive lifts the nose."""
    a = np.array(rows)
    d = {c: np.zeros(len(a)) for c in ("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")}
    for i, c in enumerate(("Bu", "Bv", "Bw", "Bp", "Bq", "Br")):
        d[c] = a[:, 4 + i]
    q = Rotation.from_euler("ZYX", np.c_[a[:, 2], a[:, 1], a[:, 0]]).as_quat()
    for i, c in enumerate(("qx", "qy", "qz", "qw")):
        d[c] = q[:, i]
    d["fin_rudder"] = rudder_sign * a[:, 10]
    d["fin_elevator"] = stern_sign * a[:, 11]
    d["rpm"] = a[:, 12]
    d["depth_true"] = a[:, 3]
    return pd.DataFrame(d)


def prestero_autopilot(pitch, q, yaw, r, psi_d, th_d, ii, h) -> tuple:
    """The autopilot every Prestero mission flies under: (his delta_s, his delta_r, the integrator)."""
    e_th = pitch - th_d
    ii = float(np.clip(ii + e_th * h, -.5, .5))
    ds = float(np.clip(1.5 * e_th + 1.2 * q + .4 * ii, -LIMIT, LIMIT))    # his +delta_s pushes the nose down
    dr = float(np.clip(-1.2 * _wrap(psi_d - yaw) + 1.2 * r, -LIMIT, LIMIT))   # his +delta_r yaws to port
    return ds, dr, ii


def fly_prestero(kind, seed, T, rate=10.0, current=(0, 0, 0), rpm=1500.0) -> pd.DataFrame:
    plan, h = Plan(kind, seed, rpm), .01
    x = np.zeros(12)
    x[0], x[8] = 1.4, 30.0
    every, rows, ii = int(round(1 / rate / h)), [], 0.0
    for i in range(int(T / h)):
        psi_d, th_d, n = plan(i * h)
        ds, dr, ii = prestero_autopilot(x[10], x[4], x[11], x[5], psi_d, th_d, ii, h)
        if i % every == 0:
            rows.append([x[9], x[10], x[11], x[8], *x[:6], dr, ds, n])
        x = PR.step(x, ds, dr, n, h, current)
        x[8] = max(x[8], 2.0)
    return _frame(rows, -1.0, -1.0)


def fly_truth_on_autopilot(x0, orders, h=.01) -> np.ndarray:
    """Prestero's vehicle from state x0 under its autopilot, on orders [(until, heading, pitch, rpm)]: (n, 3) heading, pitch, depth, once a second."""
    x, ii, out, t = np.array(x0, dtype=float), 0.0, [], 0.0
    every = int(round(1 / h))
    for psi_d, th_d, n, steps in _order_steps(orders, h):
        for _ in range(steps):
            if int(round(t / h)) % every == 0:
                out.append((x[11], x[10], x[8]))
            ds, dr, ii = prestero_autopilot(x[10], x[4], x[11], x[5], psi_d, th_d, ii, h)
            x = PR.step(x, ds, dr, n, h)
            t += h
    return np.array(out)


def _order_steps(orders, h) -> list:
    done, out = 0.0, []
    for until, psi_d, th_d, n in orders:
        out.append((psi_d, th_d, n, int(round((until - done) / h))))
        done = until
    return out


def fly_model_on_autopilot(params, vehicle, x0, orders, h=.01) -> Optional[np.ndarray]:
    """A calibrated model from the same state under the same autopilot and orders, through Strathe's own physics. None if it diverges. It sees its own
    motion: this is a lab's simulator with the vehicle's controller on it."""
    from fast_dynamics import FastDynamics
    import steering
    act = vehicle.actuators
    acc = FastDynamics(params, vehicle).acceleration_trig
    # his signs to Strathe's, as _frame does; controls are linear in the channels, so read the mix off once
    basis = [steering.controls_from_frame(pd.DataFrame({"fin_rudder": [a], "fin_elevator": [b], "rpm": [c]}), act)[0][0] for a, b, c in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    zero = [0.0] * 6
    v = [float(a) for a in x0[:6]]
    z = float(x0[8])
    qx, qy, qz, qw = Rotation.from_euler("ZYX", [x0[11], x0[10], x0[9]]).as_quat()
    every, out, ii, t = int(round(1 / h)), [], 0.0, 0.0
    for psi_d, th_d, n, steps in _order_steps(orders, h):
        for _ in range(steps):
            r20, r21, r22 = 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)
            pitch = -math.asin(max(-1.0, min(1.0, r20)))
            yaw = math.atan2(2 * (qx * qy + qw * qz), 1 - 2 * (qy * qy + qz * qz))
            if int(round(t / h)) % every == 0:
                out.append((yaw, pitch, z))
            ds, dr, ii = prestero_autopilot(pitch, v[4], yaw, v[5], psi_d, th_d, ii, h)
            ctl = [-dr * a - ds * b + n * c for a, b, c in zip(*basis)]
            vd = acc(v, zero, r21, r22, -r20, 1.0, None, ctl)
            v = [v[j] + vd[j] * h for j in range(6)]
            if not all(math.isfinite(a) and abs(a) < 1e3 for a in v):
                return None
            z += (r20 * v[0] + r21 * v[1] + r22 * v[2]) * h
            wx, wy, wz = v[3] * h, v[4] * h, v[5] * h
            th = math.sqrt(wx * wx + wy * wy + wz * wz)
            if th > 1e-12:
                c, s = math.cos(th / 2), math.sin(th / 2) / th
                dx, dy, dz = wx * s, wy * s, wz * s
                qw, qx, qy, qz = (qw * c - qx * dx - qy * dy - qz * dz, qw * dx + qx * c + qy * dz - qz * dy,
                                  qw * dy - qx * dz + qy * c + qz * dx, qw * dz + qx * dy - qy * dx + qz * c)
            m = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            qx, qy, qz, qw = qx / m, qy / m, qz / m, qw / m
            t += h
    return np.array(out)


def truth_against_itself(path: str, window_s: float, every: int = 10, h: float = .01) -> float:
    """Prestero's own equations replaying a recording's logged fins from its real start: median heading error, degrees, at the window's end. The best any
    model of this vehicle could do from these logs, because the logged fins are 10 Hz samples of a 100 Hz loop."""
    d = pd.read_csv(path)
    e = Rotation.from_quat(d[["qx", "qy", "qz", "qw"]].to_numpy()).as_euler("ZYX")[:, ::-1]
    n = int(round(window_s * 10)) + 1
    err = []
    for a in range(0, len(d) - n, n - 1):
        x = np.zeros(12)
        x[:6] = d.loc[a, ["Bu", "Bv", "Bw", "Bp", "Bq", "Br"]].to_numpy()
        x[8], x[9:12] = 30.0, e[a]
        for i in range(a, a + n - 1):
            dr, ds, rpm = -d.fin_rudder.iloc[i], -d.fin_elevator.iloc[i], d.rpm.iloc[i]
            for _ in range(every):
                x = PR.step(x, ds, dr, rpm, h)
        err.append(abs(_wrap(x[11] - e[a + n - 1, 2])))
    return float(np.degrees(np.median(err)))


def fly_fossen(kind, seed, T, rate=10.0, current=(0, 0), rpm=1100.0) -> pd.DataFrame:
    if FOSSEN_SRC not in sys.path:
        sys.path.insert(0, FOSSEN_SRC)
    from python_vehicle_simulator.lib.gnc import attitudeEuler
    from python_vehicle_simulator.vehicles.remus100 import remus100
    veh = remus100(V_current=current[0], beta_current=current[1])
    plan, h = Plan(kind, seed, rpm), .02
    eta = np.array([0, 0, 30.0, 0, 0, 0])
    nu = np.array([1.4, 0, 0, 0, 0, 0.])
    ua = np.array([0, 0, rpm])
    every, rows = int(round(1 / rate / h)), []
    for i in range(int(T / h)):
        psi_d, th_d, n = plan(i * h)
        dr = float(np.clip(1.5 * _wrap(psi_d - eta[5]) - 1.0 * nu[5], -LIMIT, LIMIT))
        ds = float(np.clip(1.5 * (eta[4] - th_d) + 1.0 * nu[4], -LIMIT, LIMIT))
        if i % every == 0:
            rows.append([eta[3], eta[4], eta[5], eta[2], *nu, ua[0], ua[1], ua[2]])
        nu, ua = veh.dynamics(eta, nu.copy(), ua, np.array([dr, ds, n], float), h)
        eta = attitudeEuler(eta, nu, h)
    return _frame(rows, 1.0, -1.0)


def fly_holoocean(kind, seed, T, rate=10.0, rpm=1100.0) -> pd.DataFrame:
    if HOLOOCEAN_SRC not in sys.path:
        sys.path.insert(0, HOLOOCEAN_SRC)
    from holoocean.fossen_dynamics.torpedo import Torpedo
    veh = Torpedo()
    plan, h = Plan(kind, seed, rpm), .02
    eta = np.array([0, 0, 30.0, 0, 0, 0])
    nu = np.array([1.4, 0, 0, 0, 0, 0.])
    every, rows = int(round(1 / rate / h)), []
    for i in range(int(T / h)):
        psi_d, th_d, n = plan(i * h)
        # its controls, probed fin by fin: starboard stern, bottom rudder, port stern, top rudder, thruster. Opposite fins are defined
        # with opposite senses, so a plane moves them with opposite signs
        dr = float(np.clip(-1.5 * _wrap(psi_d - eta[5]) + 1.0 * nu[5], -LIMIT, LIMIT))
        ds = float(np.clip(1.5 * (eta[4] - th_d) + 1.0 * nu[4], -LIMIT, LIMIT))
        u = np.array([ds, dr, -ds, -dr, n], float)
        nd, _ = veh.dynamics(eta, nu, None, u, h)
        if i % every == 0:
            rows.append([eta[3], eta[4], eta[5], eta[2], *nu, dr, ds, n])
        nu = nu + h * nd
        c1, s1, c2, s2, c3, s3 = (math.cos(eta[3]), math.sin(eta[3]), math.cos(eta[4]), math.sin(eta[4]), math.cos(eta[5]), math.sin(eta[5]))
        zd = -s2 * nu[0] + c2 * s1 * nu[1] + c1 * c2 * nu[2]
        t2 = s2 / c2
        eta = eta + h * np.array([0, 0, zd, nu[3] + s1 * t2 * nu[4] + c1 * t2 * nu[5], c1 * nu[4] - s1 * nu[5], s1 / c2 * nu[4] + c1 / c2 * nu[5]])
    return _frame(rows, -1.0, -1.0)


FLY = {"prestero": fly_prestero, "fossen": fly_fossen, "holoocean": fly_holoocean}
# a lab's tape measure and propeller sheet: nothing here is a fitted value
SIZE = {"prestero": dict(mass=30.48, length=1.33, diameter=.191, fin_x=-.64, fin_radius=.14, fin_area=6.65e-3, fin_span=.0857,
                         prop=dict(diameter_m=.14, kt0=.0157, kq0=.0158, spin=-1.0, position_m=[-.7, 0, 0])),
        "fossen": dict(mass=30.5, length=1.6, diameter=.19, fin_x=-.8, fin_radius=.14, fin_area=.0133, fin_span=.0857,
                       prop=dict(diameter_m=.14, kt0=.4566, kt1=-.4174, kq0=.007, kq1=-.0058, wake=.944, thrust_deduction=.1,
                                 position_m=[-.85, 0, 0])),
        "holoocean": dict(mass=30.5, length=1.6, diameter=.19, fin_x=-.8, fin_radius=.1, fin_area=.00665, fin_span=.0857,
                          prop=dict(diameter_m=.14, kt0=.4566, kt1=-.4174, kq0=.007, kq1=-.0058, wake=.944, thrust_deduction=.1,
                                    position_m=[-.85, 0, 0]))}


def vehicle_spec(engine: str, path: str, inertia_scale: float = 1.0, thrust_scale_free: bool = False, kt_scale: float = 1.0) -> str:
    """What `describe-vehicle --steering fin` writes for this engine's size."""
    import yaml
    import vehicle_describe as vd
    s = SIZE[engine]
    prop = {**s["prop"]}
    prop["kt0"] = prop["kt0"] * kt_scale
    spec, _ = vd.describe_hull(s["mass"], s["length"], diameter=s["diameter"], shape="torpedo", hull_lift=True, name=f"truth_{engine}",
                               steering="fin", propeller=prop,
                               fins={"layout": "cross", "x_m": s["fin_x"], "radius_m": s["fin_radius"], "area_m2": s["fin_area"],
                                     "span_m": s["fin_span"], "limit_deg": 30.0, "stall_deg": 25.0})
    t = spec["calibration_targets"]
    if inertia_scale != 1.0:
        for k in ("Ieff_pitch", "Ieff_yaw"):
            e = t["effective_inertia"][k]
            e["initial_guess"] *= inertia_scale
            e["bounds"] = [e["initial_guess"] * .999, e["initial_guess"] * 1.001]
    if thrust_scale_free:
        t["fins"]["thrust_scale"] = {"unit": "-", "initial_guess": 1.0, "bounds": [.3, 3.0], "bounds_rationale": "unknown propeller",
                                     "source": "test", "confidence": "assumed"}
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False)
    return path


def write_set(engine: str, out: str, kind: str = "rich", train_s: float = 900.0, held_s: float = 600.0, rate: float = 10.0, seed: int = 11,
              **fly) -> str:
    """Training recordings of `kind`, and held-out rich and survey recordings on another day."""
    os.makedirs(out, exist_ok=True)
    f = FLY[engine]
    keep = lambda d: d.drop(columns=["depth_true"])
    keep(f(kind, seed, train_s, rate, **fly)).to_csv(f"{out}/remus-{TRAIN_DATE}-run1.csv", index=False)
    keep(f("rich", seed + 10, held_s, rate, **fly)).to_csv(f"{out}/remus-{HELD_DATE}-run1.csv", index=False)
    keep(f("survey", seed + 11, held_s, rate, **fly)).to_csv(f"{out}/remus-{HELD_DATE}-run2.csv", index=False)
    return out
