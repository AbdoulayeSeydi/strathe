"""
simfit/adapters/holoocean.py -- HoloOcean 2.x Fossen dynamics for `strathe fit-sim --sim holoocean` / `export-sim`
(PROJECT_STATUS.md #84). Guide: calibration/simulators/holoocean/README.md.

HoloOcean drives a torpedo vehicle (`"fossen_model": "torpedo"`, control scheme 1) with Fossen's equations, configured
by the scenario's agents[i]["dynamics"] dict. Those equations run in Python (holoocean.fossen_dynamics), so windows are
replayed through the lab's own Fossen class directly: from the logged state, with the propeller and fins at rest, the
logged thrust added as M^-1 tau each step (exact: the equations are linear in tau), in Strathe's frames already (NED,
forward-right-down). What's tuned is in the dynamics dict (a missing key takes the class's own value):
  drag        T_surge, T_sway, T_yaw (time constants), zeta_roll, zeta_pitch, K_nomoto, Cd
  lift        e (Oswald efficiency)
  added_mass  r44 (added roll inertia)
  cg          r_bg x and z (centre of gravity)
  mass        mass
export-sim writes strathe_fossen.py: a Fossen class with Strathe's own model and HoloOcean's dynamics() interface, for
the custom-dynamics control scheme, and scores it on the held-out logs.
"""
from __future__ import annotations

import copy
import importlib
import json
import math
import os

import numpy as np

from simfit.core import SimFitError
from simfit.knobs import KnobSpec

DEFAULT_MODEL = "holoocean.fossen_dynamics.torpedo:Torpedo"
DEFAULT_CONFIGURE = "holoocean.fossen_dynamics.torpedo:configure_torpedo_from_scenario"
KNOBS = (("T_surge", "drag"), ("T_sway", "drag"), ("T_yaw", "drag"), ("zeta_roll", "drag"), ("zeta_pitch", "drag"),
         ("K_nomoto", "drag"), ("Cd", "drag"), ("e", "lift"), ("r44", "added_mass"), ("r_bg/0", "cg"),
         ("r_bg/2", "cg"), ("mass", "mass"))
ATTR = {"mass": ("m", "mass"), "T_surge": ("T_surge",), "T_sway": ("T_sway",), "T_yaw": ("T_yaw",),
        "zeta_roll": ("zeta_roll",), "zeta_pitch": ("zeta_pitch",), "K_nomoto": ("K_nomoto",), "Cd": ("Cd",),
        "e": ("e",), "r44": ("r44",), "r_bg": ("r_bg",)}


def _load(spec: str, what: str):
    mod, _, name = spec.partition(":")
    try:
        return getattr(importlib.import_module(mod), name)
    except ImportError:
        raise SimFitError(f"{mod} can't be imported: HoloOcean's Fossen code comes with HoloOcean's Python client "
                          f"(Windows or Linux; README.md 'Setup' step 2). Or give --fossen-model module:Class for your "
                          f"own Fossen class")
    except AttributeError:
        raise SimFitError(f"{mod} has no {name} ({what})")


def _R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _euler(q):
    R = _R(q)
    return math.atan2(R[2, 1], R[2, 2]), -math.asin(max(-1.0, min(1.0, R[2, 0]))), math.atan2(R[1, 0], R[0, 0])


class HolooceanAdapter:
    sim, label = "holoocean", "HoloOcean"
    default_groups = ("drag", "added_mass", "cg")

    def __init__(self, scenario: str, agent=None, model=DEFAULT_MODEL, configure=DEFAULT_CONFIGURE, ref=(0, 0, 0),
                 depth: float = 3.0, hz: float = 100.0):
        if not os.path.exists(scenario):
            raise SimFitError(f"the scenario {scenario} doesn't exist (save your scenario dict as JSON)")
        self.model_path = os.path.abspath(scenario)
        try:
            self.scenario = json.load(open(self.model_path, encoding="utf-8"))
        except ValueError as e:
            raise SimFitError(f"{scenario} isn't valid JSON: {e} (save your scenario dict with json.dump)")
        agents = self.scenario.get("agents", [])
        fossen = [a for a in agents if a.get("fossen_model")]
        if agent:
            fossen = [a for a in agents if a.get("agent_name") == agent]
        if len(fossen) != 1:
            raise SimFitError(f"the scenario has {len(fossen)} Fossen agents" + (f" named {agent!r}" if agent else "")
                              + ": give --agent")
        self.agent = fossen[0]["agent_name"]
        self.index = agents.index(fossen[0])
        self.cls, self.configure = _load(model, "the Fossen class"), _load(configure, "its configure function") \
            if configure else None
        self.ref, self.depth, self.hz = np.array(ref, dtype=float), float(depth), float(hz)
        self.notes, self.base, self.dt = [], None, 0.1
        self._defaults = self._vehicle(self.scenario)
        bare = json.loads(json.dumps(self.scenario))
        bare["agents"][self.index]["dynamics"] = {}
        self._class_defaults = self._vehicle(bare)   # the class's own values, before the scenario's (zero starts)

    # --- the model -------------------------------------------------------------------------------------------------
    def _vehicle(self, scenario):
        try:
            v = self.cls()
        except TypeError:
            v = self.cls("stepInput")
        if self.configure:
            self.configure(scenario, self.agent, v)
        if not hasattr(v, "Minv") or not hasattr(v, "dynamics"):
            raise SimFitError(f"{self.cls.__name__} has no Minv / dynamics(): not a Fossen vehicle class")
        return v

    def _dyn(self, scenario):
        return scenario["agents"][self.index].setdefault("dynamics", {})

    def _get(self, scenario, key):
        name, _, comp = key.partition("/")
        dyn = self._dyn(scenario)
        if name in dyn:
            val = dyn[name]
        else:
            val = next((getattr(self._defaults, a) for a in ATTR[name] if hasattr(self._defaults, a)), None)
            if val is None:
                return None
        return float(np.asarray(val, dtype=float).ravel()[int(comp)]) if comp else float(val)

    def _set(self, scenario, key, value):
        name, _, comp = key.partition("/")
        dyn = self._dyn(scenario)
        if comp:
            cur = dyn.get(name)
            if cur is None:
                cur = [float(x) for x in np.asarray(next(getattr(self._defaults, a) for a in ATTR[name]
                                                         if hasattr(self._defaults, a)), dtype=float).ravel()]
            cur = list(cur)
            cur[int(comp)] = float(value)
            dyn[name] = cur
        else:
            dyn[name] = float(value)

    def _edited(self, values, rho=None):
        sc = copy.deepcopy(self.scenario)
        for layer in (self.base, values):
            for k, v in (layer or {}).items():
                if k == "__fossen_class__":
                    continue
                self._set(sc, k, v)
        if rho is not None:
            self._dyn(sc)["rho"] = float(rho)
        return sc

    @property
    def groups(self) -> tuple:
        g = []
        for k, grp in KNOBS:
            if grp not in g and self._get(self.scenario, k) is not None:
                g.append(grp)
        return tuple(g)

    def set_base(self, values):
        self.base = dict(values) if values else None

    def knob_specs(self, info, groups, axisymmetric) -> list:
        sc = self._edited(None)
        specs = []
        for k, grp in KNOBS:
            if grp not in groups:
                continue
            v = self._get(sc, k)
            if v is None:
                continue
            if grp == "cg":
                specs.append(KnobSpec(k, grp, v, "linear", v - 0.1, v + 0.1))
            elif grp == "mass":
                specs.append(KnobSpec(k, grp, v, "log_factor", 1.25, 1.25))
            else:
                if_zero = None
                if abs(v) <= 1e-4:   # a factor of 0 stays 0 (#90): start from the Fossen class's own value
                    d = next((getattr(self._class_defaults, a) for a in ATTR.get(k.split("/")[0], ()) if
                              hasattr(self._class_defaults, a)), None)
                    d = abs(float(np.asarray(d, dtype=float).ravel()[0])) if d is not None else 0.0
                    if d <= 1e-4:
                        raise SimFitError(f"{k} is 0 in your scenario and in the Fossen class, and a factor of 0 stays "
                                          f"0: give it a starting value in the scenario's dynamics, or leave "
                                          f"{grp} out of --fit")
                    if_zero = d
                specs.append(KnobSpec(k, grp, abs(v), "log_factor", floor=1e-4, if_zero=if_zero,
                                      note=f"0 in your scenario; tuned from the class's {if_zero:g}" if if_zero else ""))
        return specs

    def write_runnable(self, values, folder, rho=None) -> str:
        os.makedirs(folder, exist_ok=True)
        p = os.path.join(folder, os.path.basename(self.model_path))
        sc = self._edited(values, rho)
        if values and "__fossen_class__" in values:  # an export: the replay runs Strathe's own Fossen class
            sc["__strathe_fossen__"] = values["__fossen_class__"]
        json.dump(sc, open(p, "w", encoding="utf-8"), indent=2)
        return p

    def write_calibrated(self, values, out_dir, suffix) -> str:
        stem, ext = os.path.splitext(os.path.basename(self.model_path))
        p = os.path.join(out_dir, f"{stem}_{suffix}{ext}")
        json.dump(self._edited(values), open(p, "w", encoding="utf-8"), indent=2)
        if values and "__fossen_class__" in values:
            open(os.path.join(out_dir, "strathe_fossen.py"), "w", encoding="utf-8").write(values["__fossen_class__"])
        return p

    def changes(self, values) -> list:
        L = []
        for k, v in values.items():
            if k == "__fossen_class__":
                L.append("- `strathe_fossen.py`: a Fossen class with Strathe's own model (see the guide to use it)")
                continue
            old = self._get(self.scenario, k)
            if old is None or abs(old - float(v)) > 1e-12 * max(1.0, abs(old)):
                L.append(f"- dynamics {k}: {old if old is not None else '(class default)'} -> {float(v):.6g}")
        return L or ["- nothing (the fit kept every value)"]

    def info(self) -> dict:
        v = self._defaults
        return {"simulator": "holoocean", "fossen_class": f"{self.cls.__module__}.{self.cls.__name__}",
                "agent": self.agent, "mass": float(getattr(v, "m", getattr(v, "mass", float("nan")))),
                "water_density": float(self._dyn(self.scenario).get("rho", getattr(v, "rho", 1025.0)))}

    # --- the replay ------------------------------------------------------------------------------------------------
    def replay(self, model_file, rows_by_window, folder) -> list:
        sc = json.load(open(model_file, encoding="utf-8"))
        cls_src = sc.get("__strathe_fossen__")
        h = 1.0 / self.hz
        sub = int(round(self.dt / h))
        if abs(sub * h - self.dt) > 1e-9:
            raise SimFitError(f"--hz {self.hz:g} doesn't divide the logs' sample time ({self.dt:g} s)")
        r = self.ref
        out = []
        for rows in rows_by_window:
            rows = np.asarray(rows, dtype=float)
            veh = self._strathe_vehicle(cls_src) if cls_src else self._vehicle(sc)
            n = len(rows)
            V, Q = np.full((n, 6), np.nan), np.full((n, 4), np.nan)
            V[0], Q[0] = rows[0, :6], rows[0, 6:10]
            w0 = rows[0, 3:6]
            nu = np.concatenate([rows[0, :3] - np.cross(w0, r), w0])  # at the vehicle's origin (CO)
            q = rows[0, 6:10] / np.linalg.norm(rows[0, 6:10])
            dimU = int(getattr(veh, "dimU", 3))
            u_actual, u_control = np.zeros(dimU), np.zeros(dimU)
            Minv = np.asarray(veh.Minv, dtype=float)
            pos = np.array([0.0, 0.0, self.depth])
            for k in range(n - 1):
                F, T = rows[k, 10:13], rows[k, 13:16]
                tau = np.concatenate([F, T + np.cross(r, F)])  # about the origin
                for _ in range(sub):
                    eta = np.concatenate([pos, _euler(q)])
                    res = veh.dynamics(eta, nu, u_actual, u_control, h)
                    nu_dot, u_actual = np.asarray(res[0], dtype=float), res[1]
                    nu = nu + h * (nu_dot + Minv @ tau)
                    pos = pos + h * (_R(q) @ nu[:3])
                    wx, wy, wz = nu[3:] * h
                    th = math.sqrt(wx * wx + wy * wy + wz * wz)
                    if th > 1e-12:
                        c, s = math.cos(th / 2), math.sin(th / 2) / th
                        dx, dy, dz = wx * s, wy * s, wz * s
                        x_, y_, z_, w_ = q
                        q = np.array([w_ * dx + x_ * c + y_ * dz - z_ * dy, w_ * dy - x_ * dz + y_ * c + z_ * dx,
                                      w_ * dz + x_ * dy - y_ * dx + z_ * c, w_ * c - x_ * dx - y_ * dy - z_ * dz])
                        q /= np.linalg.norm(q)
                if not (np.all(np.isfinite(nu)) and np.abs(nu).max() < 1e3):
                    break
                V[k + 1] = np.concatenate([nu[:3] + np.cross(nu[3:], r), nu[3:]])
                Q[k + 1] = q
            out.append((V, Q))
        return out

    # --- export: Strathe's model as a Fossen class ------------------------------------------------------------------
    def _strathe_vehicle(self, src):
        ns = {}
        exec(compile(src, "strathe_fossen.py", "exec"), ns)  # our own generated source (export_values)
        return ns["StratheFossen"]()

    def export_values(self, params, mass, info, train, rho):
        src = strathe_fossen_source(params, mass, self.ref)
        notes = ["HoloOcean's torpedo model has time constants and drag coefficients, not per-axis damping, so Strathe's "
                 "model goes in as its own Fossen class (strathe_fossen.py, same dynamics() interface), used with the "
                 "custom-dynamics control scheme (calibration/simulators/holoocean/README.md). Its centre of gravity is "
                 "at the logs' reference point, as in Strathe's model."]
        if params.thruster_time_constant_s:
            notes.append(f"Thruster lag ({params.thruster_time_constant_s:.2f} s) not carried.")
        return {"__fossen_class__": src}, notes



def strathe_fossen_source(params, mass: float, ref) -> str:
    """A self-contained Fossen vehicle class with Strathe's calibrated model (no Strathe import needed)."""
    p = {k: float(getattr(params, k)) for k in params.__dataclass_fields__}
    r = [float(x) for x in ref]
    return f'''"""StratheFossen: a vehicle calibrated by Strathe (strathe export-sim --sim holoocean), as a Fossen class with
HoloOcean's interface: dynamics(eta, nu, u_actual, u_control, sampleTime) -> (nu_dot, u_actual), with nu at the
logs' reference point offset {r} m from this model's origin. Add your thrust as Minv @ tau (forward-right-down,
about the origin). Generated; edit the numbers in PARAMS only if you know why."""
import math

import numpy as np

PARAMS = {json.dumps(p, indent=4)}
MASS = {float(mass)!r}
REF = np.array({r})
G = 9.81


def _skew(x):
    return np.array([[0.0, -x[2], x[1]], [x[2], 0.0, -x[0]], [-x[1], x[0], 0.0]])


class StratheFossen:
    dimU = 3

    def __init__(self, *args, **kwargs):
        P = PARAMS
        M = np.diag([MASS + P["Xu_dot"], MASS + P["Yv_dot"], MASS + P["Zw_dot"], P["Ieff_roll"], P["Ieff_pitch"],
                     P["Ieff_yaw"]])
        M[0, 4] = M[4, 0] = P["Xq_dot"]
        M[2, 4] = M[4, 2] = P["Zq_dot"]
        self.M_ref = M
        J = np.block([[np.eye(3), -_skew(REF)], [np.zeros((3, 3)), np.eye(3)]])  # nu_ref = J nu_origin
        self.J = J
        self.M = J.T @ M @ J
        self.Minv = np.linalg.inv(self.M)

    def _accel_ref(self, v, phi, theta):
        P = PARAMS
        lin = np.array([P["Xu"], P["Yv"], P["Zw"], P["Kp"], P["Mq"], P["Nr"]])
        quad = np.array([P["Xuu"], P["Yvv"], P["Zww"], P["Kpp"], P["Mqq"], P["Nrr"]])
        D = lin * v + quad * np.abs(v) * v
        if P.get("Zuw") or P.get("Muw") or P.get("Muq"):
            u, vv, w, _, q, r = v
            D = D + np.array([0.0, P["Zuw"] * u * vv, P["Zuw"] * u * w, 0.0, P["Muw"] * u * w + P["Muq"] * u * q,
                              -P["Muw"] * u * vv + P["Muq"] * u * r])
        W = MASS * G
        diff = -P["net_buoyancy_z"]
        g = np.array([diff * math.sin(theta), -diff * math.cos(theta) * math.sin(phi),
                      -diff * math.cos(theta) * math.cos(phi),
                      P["cg_cb_offset_z"] * W * math.cos(theta) * math.sin(phi), P["cg_cb_offset_z"] * W * math.sin(theta),
                      0.0])
        if P.get("cg_cb_offset_x"):
            g[4] += P["cg_cb_offset_x"] * W * math.cos(theta) * math.cos(phi)
            g[5] -= P["cg_cb_offset_x"] * W * math.cos(theta) * math.sin(phi)
        pm = self.M_ref @ v
        C = np.zeros((6, 6))
        C[:3, 3:] = -_skew(pm[:3])
        C[3:, :3] = -_skew(pm[:3])
        C[3:, 3:] = -_skew(pm[3:])
        return D - g - C @ v  # the generalized force at the reference point, without thrust

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        """nu: velocity at this model's origin (forward-right-down). Returns the acceleration at the origin without
        thrust, and u_actual unchanged."""
        v = self.J @ np.asarray(nu, dtype=float)
        f = self._accel_ref(v, eta[3], eta[4])
        # forces at the reference point -> generalized forces at the origin (J^T), solved with the origin's mass matrix
        return self.Minv @ (self.J.T @ f), u_actual
'''


def add_arguments(ap) -> None:
    ap.add_argument("--scenario", required=True, help="your HoloOcean scenario, saved as JSON (json.dump(scenario))")
    ap.add_argument("--agent", help="the Fossen agent's name (default: the one with a fossen_model)")
    ap.add_argument("--fossen-model", default=DEFAULT_MODEL, help=f"module:Class of the Fossen vehicle (default: "
                                                                  f"{DEFAULT_MODEL})")
    ap.add_argument("--configure", default=DEFAULT_CONFIGURE,
                    help="module:function that configures it from the scenario (default: HoloOcean's torpedo one; "
                         "'' for none)")
    ap.add_argument("--hz", type=float, default=100.0, help="integration rate, Hz (default 100; divides 10 Hz logs)")
    ap.add_argument("--start-depth", type=float, default=3.0, help="depth the windows are replayed at, m")


def adapter_from_args(a) -> HolooceanAdapter:
    return HolooceanAdapter(a.scenario, a.agent, a.fossen_model, a.configure or None, a.reference_point,
                            a.start_depth, a.hz)
