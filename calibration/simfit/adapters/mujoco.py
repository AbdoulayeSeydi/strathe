"""
simfit/adapters/mujoco.py -- MuJoCo for `strathe fit-sim --sim mujoco` (PROJECT_STATUS.md #83). Guide:
calibration/simulators/mujoco/README.md. Runs in Python (the `mujoco` package Strathe already installs): nothing to
build.

The vehicle is the lab's MJCF body with a free joint (--body), x forward and z up (the ROS convention). MuJoCo has no
water: its fluid model (option density/viscosity, and per geom fluidshape="ellipsoid" with five coefficients) gives
drag, lift and a velocity-dependent share of added mass, and buoyancy is usually `gravcomp` on the body. What can be
tuned, found in the file:
  drag        each ellipsoid hull geom's blunt, slender and angular drag coefficients (fluidcoef 1-3), or -- with no
              ellipsoid geom (MuJoCo's inertia-box model) -- the option's density and viscosity
  lift        each ellipsoid geom's Kutta and Magnus lift coefficients (fluidcoef 4-5)
  buoyancy    the body's gravcomp (1 = neutral; above 1 floats; acts at the centre of mass: no righting moment)
  cg          the body's <inertial pos> x and z
  mass        its <inertial mass>
export-sim isn't possible: MuJoCo has no per-axis linear damping, no buoyancy force and doesn't apply the part of added
mass that resists acceleration, so Strathe's model can't be written into it. For Strathe's own model inside MuJoCo,
see calibration/mujoco_backend.py.
"""
from __future__ import annotations

import os

import numpy as np

from simfit.core import SimFitError
from simfit.editors import XmlFile, _parse_xml, fmt_number
from simfit.knobs import KnobSpec

T_NED_ENU = np.array([[0.0, 1, 0], [1, 0, 0], [0, 0, -1]])
T_FRD_FLU = np.diag([1.0, -1, -1])
DEFAULT_FLUIDCOEF = "0.5 0.25 1.5 1.0 1.0"
COEF = (("blunt_drag", "drag"), ("slender_drag", "drag"), ("angular_drag", "drag"), ("kutta_lift", "lift"),
        ("magnus_lift", "lift"))


def _descend(n):
    for c in n.children:
        yield c
        yield from _descend(c)


def _quat_xyzw_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _R_to_quat_wxyz(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return q


class MujocoAdapter:
    sim, label = "mujoco", "MuJoCo"
    default_groups = ("drag", "buoyancy", "cg")

    def __init__(self, model: str, body=None, ref=(0.0, 0.0, 0.0), depth: float = 3.0):
        if not os.path.exists(model):
            raise SimFitError(f"the MJCF file {model} doesn't exist")
        try:
            import mujoco  # noqa: F401
        except ImportError:
            raise SimFitError("MuJoCo isn't installed: pip install -r requirements.txt")
        self.model_path = os.path.abspath(model)
        self.text = open(self.model_path, encoding="utf-8").read()
        self.ref, self.depth = np.array(ref, dtype=float), float(depth)
        self.notes, self.base, self.dt = [], None, 0.1
        self._discover(body)

    def _discover(self, body):
        root = _parse_xml(self.text.encode("utf-8"))
        bodies = [n for n in _descend(root) if n.tag == "body" and
                  any(c.tag == "freejoint" or (c.tag == "joint" and c.attrs.get("type") == "free") for c in n.children)]
        if body:
            bodies = [b for b in bodies if b.attrs.get("name") == body]
        if len(bodies) != 1 or not bodies[0].attrs.get("name"):
            raise SimFitError(f"{os.path.basename(self.model_path)} has {len(bodies)} named bodies with a free joint"
                              + (f" called {body!r}" if body else "") + ": give --body (the vehicle)")
        b = bodies[0]
        self.body = b.attrs["name"]
        self.body_path = f"//body[@name='{self.body}']"
        self.ellipsoids = [g.attrs.get("name") for g in b.children if g.tag == "geom"
                           and g.attrs.get("fluidshape") == "ellipsoid"]
        if any(not g for g in self.ellipsoids):
            raise SimFitError("every geom with fluidshape=\"ellipsoid\" on the vehicle needs a name, so it can be tuned")
        self.inertial = next((c for c in b.children if c.tag == "inertial"), None)
        # buoyancy: a child body named "buoyancy" at the centre of buoyancy carries it (righting moment), else the body
        floats = [c for c in b.children if c.tag == "body" and c.attrs.get("name") == "buoyancy"]
        self.buoyancy_path = "//body[@name='buoyancy']" if floats else self.body_path
        opts = [n for n in _descend(root) if n.tag == "option"]
        self.option = opts[0] if opts else None
        if not self.ellipsoids:
            self.notes.append("No geom with fluidshape=\"ellipsoid\": MuJoCo's inertia-box fluid model, so the drag "
                              "tuned is the option's density and viscosity (the water density per recording can't be "
                              "set separately).")

    def _prepared(self, text: str) -> XmlFile:
        """The file with every attribute a knob needs present (MJCF defaults made explicit)."""
        ed = XmlFile(text)
        for g in self.ellipsoids:
            ed.ensure_attr(f"{self.body_path}/geom[@name='{g}']", "fluidcoef", DEFAULT_FLUIDCOEF)
        ed.ensure_attr(self.buoyancy_path, "gravcomp", "0")
        return ed

    def _knob_defs(self):
        out = []
        for g in self.ellipsoids:
            for i, (name, group) in enumerate(COEF):
                out.append((f"{g}_{name}", group, f"{self.body_path}/geom[@name='{g}']/@fluidcoef#{i}"))
        if not self.ellipsoids and self.option is not None:
            out += [("water_density", "drag", "//option/@density"), ("water_viscosity", "drag", "//option/@viscosity")]
        out.append(("gravcomp", "buoyancy", f"{self.buoyancy_path}/@gravcomp"))
        if self.inertial is not None:
            if "pos" in self.inertial.attrs:
                out += [("cg_x", "cg", f"{self.body_path}/inertial/@pos#0"), ("cg_z", "cg", f"{self.body_path}/inertial/@pos#2")]
            if "mass" in self.inertial.attrs:
                out.append(("mass", "mass", f"{self.body_path}/inertial/@mass"))
        return out

    @property
    def groups(self) -> tuple:
        g = []
        for _n, grp, _p in self._knob_defs():
            if grp not in g:
                g.append(grp)
        return tuple(g)

    def set_base(self, values):
        self.base = dict(values) if values else None

    def _edit(self, values, rho=None) -> str:
        ed = self._prepared(self.text)
        defs = {n: p for n, _g, p in self._knob_defs()}
        for layer in (self.base, values):
            for k, v in (layer or {}).items():
                if k not in defs:
                    raise SimFitError(f"unknown MuJoCo setting {k!r}")
                ed.set(defs[k], float(v))
        if rho is not None and self.ellipsoids:
            if self.option is None or "density" not in self.option.attrs:
                raise SimFitError("--density-by-date: the MJCF needs <option density=\"...\"/>")
            ed.set("//option/@density", rho)
        return ed.text

    def knob_specs(self, info: dict, groups: tuple, axisymmetric: bool) -> list:
        ed = XmlFile(self._edit(None))
        specs = []
        for name, group, path in self._knob_defs():
            if group not in groups:
                continue
            v = ed.get(path)
            if group == "cg" or (group == "buoyancy" and self.buoyancy_path == self.body_path):
                specs.append(KnobSpec(name, group, v, "linear", v - 0.15, v + 0.15))
            elif group == "buoyancy":  # a buoyancy body's gravcomp: +-15% of the buoyancy
                specs.append(KnobSpec(name, group, v, "log_factor", 1.15, 1.15))
            elif group == "mass":
                specs.append(KnobSpec(name, group, v, "log_factor", 1.25, 1.25))
            else:
                specs.append(KnobSpec(name, group, v, "log_factor", floor=1e-3))
        return specs

    def write_runnable(self, values, folder: str, rho=None) -> str:
        os.makedirs(folder, exist_ok=True)
        p = os.path.join(folder, os.path.basename(self.model_path))
        open(p, "w", encoding="utf-8").write(self._edit(values, rho))
        return p

    def write_calibrated(self, values, out_dir: str, suffix: str) -> str:
        stem, ext = os.path.splitext(os.path.basename(self.model_path))
        p = os.path.join(out_dir, f"{stem}_{suffix}{ext}")
        open(p, "w", encoding="utf-8").write(self._edit(values))
        return p

    def changes(self, values) -> list:
        orig = self._prepared(self.text)
        defs = {n: p for n, _g, p in self._knob_defs()}
        L = []
        for k, v in values.items():
            old = orig.get(defs[k])
            if abs(old - float(v)) > 1e-12 * max(1.0, abs(old)):
                L.append(f"- {k}: {fmt_number(old)} -> {fmt_number(v)} (`{defs[k]}`)")
        return L or ["- nothing (the fit kept every value)"]

    def _load(self, model_file: str):
        import mujoco
        spec = mujoco.MjSpec.from_file(model_file)
        b = spec.body(self.body)
        site = b.add_site()
        site.name = "strathe_reference_point"
        site.pos = [self.ref[0], -self.ref[1], -self.ref[2]]  # forward-left-up
        m = spec.compile()
        return m, mujoco.MjData(m)

    def info(self) -> dict:
        import mujoco
        m, d = self._load(self.write_runnable(None, __import__("tempfile").mkdtemp(prefix="strathe_mj_info_")))
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.body)
        return {"simulator": "mujoco", "mujoco": mujoco.__version__, "body": self.body, "mass": float(m.body_mass[bid]),
                "inertia": m.body_inertia[bid].tolist(), "timestep": float(m.opt.timestep),
                "water_density": float(m.opt.density), "ellipsoid_geoms": self.ellipsoids}

    def replay(self, model_file: str, rows_by_window: list, folder: str) -> list:
        import mujoco
        m, d = self._load(model_file)
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.body)
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "strathe_reference_point")
        jnt = m.body_jntadr[bid]
        qa, va = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]
        sub = int(round(self.dt / m.opt.timestep))
        if sub < 1 or abs(sub * m.opt.timestep - self.dt) > 1e-9 * max(1.0, self.dt):
            raise SimFitError(f"the logs' sample time ({self.dt:g} s) isn't a whole number of the model's timestep "
                              f"({m.opt.timestep:g} s): set <option timestep> so it divides it")
        r = self.ref
        out = []
        res = np.zeros(6)
        for rows in rows_by_window:
            rows = np.asarray(rows, dtype=float)
            n = len(rows)
            V, Q = np.full((n, 6), np.nan), np.full((n, 4), np.nan)
            mujoco.mj_resetData(m, d)
            v0, w0, q0 = rows[0, :3], rows[0, 3:6], rows[0, 6:10]
            Re = T_NED_ENU @ _quat_xyzw_to_R(q0) @ T_FRD_FLU
            d.qpos[qa:qa + 3] = [0.0, 0.0, -self.depth]
            d.qpos[qa + 3:qa + 7] = _R_to_quat_wxyz(Re)
            vo = v0 - np.cross(w0, r)  # the body origin's velocity, FRD
            d.qvel[va:va + 3] = Re @ (T_FRD_FLU @ vo)
            d.qvel[va + 3:va + 6] = T_FRD_FLU @ w0
            mujoco.mj_forward(m, d)
            V[0], Q[0] = rows[0, :6], rows[0, 6:10]
            for k in range(n - 1):
                F, Tq = T_FRD_FLU @ rows[k, 10:13], T_FRD_FLU @ rows[k, 13:16]
                for _ in range(sub):
                    mujoco.mj_kinematics(m, d)  # the pose now (after mj_step it's the pose before the step)
                    R = d.xmat[bid].reshape(3, 3)
                    Fw, Tw = R @ F, R @ Tq
                    d.xfrc_applied[bid, :3] = Fw
                    d.xfrc_applied[bid, 3:] = Tw + np.cross(d.site_xpos[sid] - d.xipos[bid], Fw)
                    mujoco.mj_step(m, d)
                mujoco.mj_forward(m, d)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, res, 1)
                vb, wb = T_FRD_FLU @ res[3:], T_FRD_FLU @ res[:3]
                if not (np.all(np.isfinite(vb)) and np.abs(np.concatenate([vb, wb])).max() < 1e3):
                    break
                Rn = T_NED_ENU @ d.xmat[bid].reshape(3, 3) @ T_FRD_FLU
                qw = _R_to_quat_wxyz(Rn)
                V[k + 1] = np.concatenate([vb, wb])
                Q[k + 1] = [qw[1], qw[2], qw[3], qw[0]]
            out.append((V, Q))
        return out

    def export_values(self, params, mass, info, train, rho):
        raise SimFitError("MuJoCo can't take Strathe's model as it is: it has no per-axis linear damping, no buoyancy "
                          "force (gravcomp acts at the centre of mass, with no righting moment) and doesn't apply the "
                          "part of added mass that resists acceleration. Calibrate your MuJoCo model through MuJoCo "
                          "instead (strathe fit-sim --sim mujoco), or run Strathe's own model inside MuJoCo "
                          "(calibration/mujoco_backend.py).")


def add_arguments(ap) -> None:
    ap.add_argument("--model", required=True, help="your MJCF (.xml) with the vehicle in it")
    ap.add_argument("--body", help="the vehicle's body (default: the one body with a free joint)")
    ap.add_argument("--start-depth", type=float, default=3.0, help="depth the windows are replayed at, m")


def adapter_from_args(a) -> MujocoAdapter:
    return MujocoAdapter(a.model, a.body, a.reference_point, a.start_depth)
