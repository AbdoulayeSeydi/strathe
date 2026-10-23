"""
simfit/adapters/gazebo.py -- Gazebo Harmonic (gz-sim 8) for `strathe fit-sim --sim gazebo` / `export-sim`
(PROJECT_STATUS.md #82). Guide: calibration/simulators/gazebo/README.md.

The lab's world runs headless in calibration/simulators/gazebo/strathe_gz_replay (or --replay-tool docker:IMAGE) with
its own Physics, Buoyancy and Hydrodynamics systems. The vehicle is the model with the Hydrodynamics plugin, in the
world file or in a model folder the world includes (--model-file). What can be tuned, found in the files:
  drag        the plugin's damping: xU..nR (linear) and xUabsU..nRabsR (quadratic), as present (negative: their
              magnitudes are tuned)
  added_mass  the plugin's xDotU, yDotV, zDotW (negative), or the link's <fluid_added_mass> xx, yy, zz
  buoyancy    the size of the link's box/cylinder/sphere collisions (what the Buoyancy system measures)
  cg          the link's centre of mass, <inertial><pose> x and z
  mass        the link's <inertial><mass> (not tuned unless asked: weigh the vehicle)
Gazebo's Hydrodynamics is Strathe's damping law (Fossen), so export-sim puts a Strathe calibration in exactly: damping
and lift as they are, added mass, inertia, mass, the centre of mass at the logs' reference point and the collisions
moved and resized for the centre of buoyancy and net buoyancy. Gazebo's frames are forward-left-up / east-north-up;
the diagonal terms don't change sign between frames, the replay tool converts the rest.
"""
from __future__ import annotations

import os
import re
import tempfile

import numpy as np

from simfit.core import SimFitError
from simfit.editors import XmlFile, _parse_xml, fmt_number
from simfit.knobs import KnobSpec
from simfit.replay_tool import ReplayTool, find

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(os.path.dirname(HERE))
DEFAULT_TOOL = os.path.join(CAL, "simulators", "gazebo", "build", "strathe_gz_replay")
AX = "xyzkmn"
VEL = "UVWPQR"
AXIS_NAMES = {"x": "surge", "y": "sway", "z": "heave", "k": "roll", "m": "pitch", "n": "yaw"}
STRATHE_LIN = ("Xu", "Yv", "Zw", "Kp", "Mq", "Nr")
STRATHE_QUAD = ("Xuu", "Yvv", "Zww", "Kpp", "Mqq", "Nrr")
TIES = {"y": "yz", "z": "yz", "m": "mn", "n": "mn"}  # --axisymmetric: sway = heave, pitch = yaw


def _descend(n):
    for c in n.children:
        yield c
        yield from _descend(c)


class GazeboAdapter:
    sim, label = "gazebo", "Gazebo"
    default_groups = ("drag", "added_mass", "buoyancy", "cg")

    def __init__(self, world: str, model_file=None, model_name=None, link=None, tool=None, ref=(0.0, 0.0, 0.0),
                 depth: float = 3.0, resource_path=None):
        for f in (world, model_file):
            if f and not os.path.exists(f):
                raise SimFitError(f"{f} doesn't exist")
        self.world_path = os.path.abspath(world)
        self.model_path = os.path.abspath(model_file or world)
        self.separate = self.model_path != self.world_path
        self.world_text = open(self.world_path, encoding="utf-8").read()
        self.model_text = open(self.model_path, encoding="utf-8").read() if self.separate else self.world_text
        self.model_name, self.ref = model_name, tuple(ref)
        self.notes, self.base, self.dt = [], None, 0.1
        self.step_override = None   # a physics step to run at instead of the world's (safe_step, #90)
        self._discover(link)
        extra = ["--depth", f"{depth:g}"] + (["--model-name", model_name] if model_name else []) + ["--link", self.link]
        self.tool = ReplayTool(tool or [DEFAULT_TOOL], tuple(ref), extra=extra, name="Gazebo")
        paths = [os.path.dirname(os.path.dirname(self.model_path))] if self.separate else []
        paths += [p for p in (resource_path or "").split(os.pathsep) if p]
        old = os.environ.get("GZ_SIM_RESOURCE_PATH", "")
        self.env = {"GZ_IP": "127.0.0.1", "GZ_SIM_RESOURCE_PATH": os.pathsep.join(paths + ([old] if old else []))}

    # --- what's in the files -------------------------------------------------------------------------------------
    def _discover(self, link):
        root = _parse_xml(self.model_text.encode("utf-8"))
        plugins = [n for n in _descend(root) if n.tag == "plugin" and "hydrodynamics" in
                   (n.attrs.get("filename", "") + n.attrs.get("name", "")).lower()]
        if self.model_name:
            plugins = [p for p in plugins if self._model_of(p) == self.model_name] or plugins
        if len(plugins) != 1:
            raise SimFitError(f"{os.path.basename(self.model_path)} has {len(plugins)} Hydrodynamics plugins; the vehicle "
                              f"needs exactly one (give --model-file for the model that has it, or --model-name)")
        p = plugins[0]
        attr = "name" if p.attrs.get("name") else "filename"
        self.plugin = f"//plugin[@{attr}='{p.attrs[attr]}']"
        data = self.model_text.encode("utf-8")
        ln = [c for c in p.children if c.tag == "link_name"]
        self.link = link or (data[ln[0].head_end:ln[0].end].decode().strip() if ln else None)
        if not self.link:
            raise SimFitError("the Hydrodynamics plugin has no <link_name>; give --link")
        self.link_path = f"//link[@name='{self.link}']"
        links = [n for n in _descend(root) if n.tag == "link" and n.attrs.get("name") == self.link]
        if len(links) != 1:
            raise SimFitError(f"{os.path.basename(self.model_path)} has {len(links)} links named {self.link!r}, not one "
                              f"(--model-file: the file with the vehicle's model)")
        self.params = {c.tag: c for c in p.children}
        lk = links[0]
        inert = [c for c in lk.children if c.tag == "inertial"]
        self.inertial = inert[0] if inert else None
        self.fluid_added = bool(self.inertial and any(c.tag == "fluid_added_mass" for c in self.inertial.children))
        self.collisions = []
        for c in lk.children:
            if c.tag != "collision":
                continue
            geo = [g for g in c.children if g.tag == "geometry"]
            shape = next((s for s in (geo[0].children if geo else []) if s.tag in ("box", "cylinder", "sphere")), None)
            self.collisions.append((c.attrs.get("name", ""), shape.tag if shape else None,
                                    any(x.tag == "pose" for x in c.children)))

    def _model_of(self, node):
        n = node.parent
        while n is not None and n.tag != "model":
            n = n.parent
        return n.attrs.get("name") if n is not None else None

    def _knob_defs(self):
        """[(name, group, path, sign, axisym key)] for what the files have."""
        out = []
        for a, v in zip(AX, VEL):
            for tag, group in ((f"{a}{v}", "drag"), (f"{a}{v}abs{v}", "drag")):
                if tag in self.params:
                    tie = f"{TIES[a]}_{'quadratic' if 'abs' in tag else 'linear'}" if a in TIES else None
                    out.append((tag, group, f"{self.plugin}/{tag}", -1.0, tie))
        for tag in ("zUW", "yUV", "mUW", "nUV", "mUQ", "nUR"):  # hull lift (cross terms), when the model has them
            if tag in self.params:
                out.append((tag, "lift", f"{self.plugin}/{tag}", 1.0, None))
        if self.fluid_added:
            for a, k in (("x", "xx"), ("y", "yy"), ("z", "zz")):
                out.append((f"added_mass_{a}", "added_mass", f"{self.link_path}/inertial/fluid_added_mass/{k}", 1.0,
                            "added_mass_yz" if a in "yz" else None))
        else:
            for a, v in zip("xyz", "UVW"):
                tag = f"{a}Dot{v}"
                if tag in self.params:
                    out.append((tag, "added_mass", f"{self.plugin}/{tag}", -1.0, "Dot_yz" if a in "yz" else None))
        if self.inertial is not None:
            if any(c.tag == "pose" for c in self.inertial.children):
                out.append(("cg_x", "cg", f"{self.link_path}/inertial/pose#0", 1.0, None))
                out.append(("cg_z", "cg", f"{self.link_path}/inertial/pose#2", 1.0, None))
            if any(c.tag == "mass" for c in self.inertial.children):
                out.append(("mass", "mass", f"{self.link_path}/inertial/mass", 1.0, None))
        return out

    @property
    def groups(self) -> tuple:
        g = []
        for _n, grp, *_ in self._knob_defs():
            if grp not in g:
                g.append(grp)
        if any(t for _n, t, _p in self.collisions):
            g.insert(min(2, len(g)), "buoyancy")
        return tuple(g)

    # --- applying values ---------------------------------------------------------------------------------------------
    def set_base(self, values):
        self.base = dict(values) if values else None

    def _edit(self, values):
        """The model file's text with the base and these values applied."""
        ed = XmlFile(self.model_text)
        defs = {n: (p, s) for n, _g, p, s, _t in self._knob_defs()}
        for layer in (self.base, values):
            for key, v in (layer or {}).items():
                if key in defs:
                    ed.set(defs[key][0], float(v) * defs[key][1])
                elif key == "buoyancy_scale":
                    self._scale_collisions(ed, float(v))
                elif key == "collision_shift":
                    self._shift_collisions(ed, v)
                elif key.startswith("@"):
                    ed.set(key[1:], float(v))
                elif key.startswith("+"):
                    parent, tag = key[1:].rsplit("|", 1)
                    ed.set_or_add(parent, tag, v if isinstance(v, str) else fmt_number(v))
                else:
                    raise SimFitError(f"unknown Gazebo setting {key!r}")
        return ed.text

    def _scale_collisions(self, ed, k: float):
        f = k ** (1 / 3)
        n = 0
        for name, shape, _ in self.collisions:
            c = f"{self.link_path}/collision[@name='{name}']/geometry/{shape}"
            if shape == "box":
                for i in range(3):
                    ed.set(f"{c}/size#{i}", ed.get(f"{c}/size#{i}") * f)
            elif shape == "cylinder":
                ed.set(f"{c}/radius", ed.get(f"{c}/radius") * f)
                ed.set(f"{c}/length", ed.get(f"{c}/length") * f)
            elif shape == "sphere":
                ed.set(f"{c}/radius", ed.get(f"{c}/radius") * f)
            else:
                continue
            n += 1
        if not n:
            raise SimFitError("buoyancy: the link has no box, cylinder or sphere collision to resize")

    def _shift_collisions(self, ed, d):
        for name, shape, has_pose in self.collisions:
            c = f"{self.link_path}/collision[@name='{name}']"
            if not has_pose:
                ed.set_or_add(c, "pose", "0 0 0 0 0 0")
            for i in range(3):
                ed.set(f"{c}/pose#{i}", ed.get(f"{c}/pose#{i}") + float(d[i]))

    def _world_for(self, model_text: str, folder: str, rho):
        """The world to run: with the edited model (inline, or a copy of the model folder the world includes), and the
        water density for this recording."""
        if self.separate:
            src = os.path.dirname(self.model_path)
            dst = os.path.join(folder, os.path.basename(src))
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(src):
                t = os.path.join(dst, f)
                if os.path.basename(self.model_path) == f:
                    open(t, "w", encoding="utf-8").write(model_text)
                elif not os.path.exists(t):
                    os.symlink(os.path.join(src, f), t)
            world = self._include(self.world_text, src, dst)
        else:
            world = model_text
        if rho is not None:
            w = XmlFile(world)
            for p in ("uniform_fluid_density", "graded_buoyancy/default_density"):
                bp = f"//plugin[@name='gz::sim::systems::Buoyancy']/{p}"
                if w.exists(bp):
                    w.set(bp, rho)
                    break
            world = w.text
        return world

    def _include(self, world: str, src: str, dst: str) -> str:
        name = os.path.basename(src)
        out, n = re.subn(r"(<include\b[^>]*>.*?<uri>\s*)(model://%s|file://%s|%s)(\s*</uri>)" %
                         (re.escape(name), re.escape(src), re.escape(src)), lambda m: m.group(1) + dst + m.group(3),
                         world, flags=re.S)
        if not n:
            raise SimFitError(f"the world doesn't <include> the model folder {name} (model://{name}); give the world that "
                              f"includes it, or put the model in the world")
        return out

    def write_runnable(self, values, folder: str, rho=None) -> str:
        os.makedirs(folder, exist_ok=True)
        p = os.path.join(folder, os.path.basename(self.world_path))
        open(p, "w", encoding="utf-8").write(self._with_step(self._world_for(self._edit(values), folder, rho)))
        return p

    def replay(self, model: str, rows_by_window: list, folder: str) -> list:
        self.tool.dt, self.tool.env = self.dt, self.env
        return self.tool.run(model, rows_by_window, folder)

    def info(self) -> dict:
        self.tool.dt, self.tool.env = self.dt, self.env
        i = self.tool.info(self.write_runnable(None, tempfile.mkdtemp(prefix="strathe_gz_info_")))
        w = XmlFile(self.world_text)
        for p in ("uniform_fluid_density", "graded_buoyancy/default_density"):
            bp = f"//plugin[@name='gz::sim::systems::Buoyancy']/{p}"
            if w.exists(bp):
                i["water_density"] = w.get(bp)
                break
        return i

    def knob_specs(self, info: dict, groups: tuple, axisymmetric: bool) -> list:
        ed = XmlFile(self._edit(None))
        specs = []
        for name, group, path, sign, tie in self._knob_defs():
            if group not in groups:
                continue
            v = ed.get(path) * sign
            if group == "cg":
                specs.append(KnobSpec(name, group, v, "linear", v - 0.15, v + 0.15))
            elif group == "lift":  # signed, can start at 0: tuned by adding to it
                specs.append(KnobSpec(name, group, v, "linear", v - 200.0, v + 200.0))
            elif group == "mass":
                specs.append(KnobSpec(name, group, v, "log_factor", 1.25, 1.25))
            else:
                start = max(v, 0.0)
                floor = 1e-3 if group == "drag" else 1e-2
                if_zero = self._zero_start(name, ed) if start <= floor else None
                note = (f"0 in your model; tuned from {fmt_number(if_zero)}, sized from the vehicle" if if_zero else "")
                specs.append(KnobSpec(name, group, start, "log_factor", floor=floor, if_zero=if_zero,
                                      tie=tie if axisymmetric else None, note=note))
        if "buoyancy" in groups and any(t for _n, t, _p in self.collisions):
            specs.append(KnobSpec("buoyancy_scale", "buoyancy", 1.0, "log", 0.8, 1.25))
        return specs

    TYPICAL_SPEED = {"x": 0.2, "y": 0.2, "z": 0.2, "k": 0.5, "m": 0.5, "n": 0.5}   # m/s and rad/s, ROV scale

    def _zero_start(self, name: str, ed) -> float:
        """A starting value for a drag or added-mass term the model has at 0 (#89 follow-up: bluerov2_gz has no
        linear drag and no added mass). Added mass: 0.3 x the vehicle's mass. Linear drag: the same axis's
        quadratic drag x a typical speed; quadratic: linear / that speed; without either, sized from the mass (or
        the inertia for rotation). The fit moves it up to x30 either way."""
        def get(path, default):
            try:
                return float(ed.get(path))
            except Exception:
                return default
        mass = get(f"{self.link_path}/inertial/mass", 10.0)
        if name.startswith("added_mass") or "Dot" in name:
            return 0.3 * mass
        a = name[0]
        rot = a in "kmn"
        inertia = get(f"{self.link_path}/inertial/inertia/{ {'k': 'ixx', 'm': 'iyy', 'n': 'izz'}.get(a, 'ixx') }", 0.1 * mass)
        s = self.TYPICAL_SPEED.get(a, 0.2)
        v = name[1]
        if "abs" in name:   # quadratic
            lin = abs(get(f"{self.plugin}/{a}{v}", 0.0))
            return lin / s if lin > 1e-3 else (5.0 * inertia if rot else 2.0 * mass)
        quad = abs(get(f"{self.plugin}/{a}{v}abs{v}", 0.0))
        return quad * s if quad > 1e-3 else (2.0 * inertia if rot else 1.0 * mass)

    def world_step(self) -> float:
        try:
            return float(XmlFile(self.world_text).get("//physics/max_step_size"))
        except Exception:
            return 0.001   # SDF's default

    def stiff_axes(self, values, step: float) -> tuple:
        """([axis names], shortest time constant s) for the axes whose damping is too stiff for `step`.

        #90, bluerov2_gz: the fit found roll and pitch damping so stiff for a light vehicle that Gazebo's 0.01 s step
        blew up in 35 of 712 unseen windows (0 at 0.005 s). Gazebo applies the Hydrodynamics forces step by step, so a
        step longer than about half an axis's damping time constant (inertia / damping at a typical speed) is unsafe."""
        ed = XmlFile(self._edit(values))

        def get(path, default=0.0):
            try:
                return float(ed.get(path))
            except Exception:
                return default
        mass = get(f"{self.link_path}/inertial/mass", 10.0)
        added = {a: (get(f"{self.link_path}/inertial/fluid_added_mass/{a}{a}") if self.fluid_added
                     else abs(get(f"{self.plugin}/{a}Dot{v}"))) for a, v in zip("xyz", "UVW")}
        inertia = {"k": "ixx", "m": "iyy", "n": "izz"}
        stiff = []
        for a, v in zip(AX, VEL):
            k = abs(get(f"{self.plugin}/{a}{v}")) + abs(get(f"{self.plugin}/{a}{v}abs{v}")) * self.TYPICAL_SPEED[a]
            if k <= 0:
                continue
            m = get(f"{self.link_path}/inertial/inertia/{inertia[a]}", 0.1) if a in inertia else mass + added[a]
            if step > 0.5 * (m / k):
                stiff.append((m / k, AXIS_NAMES.get(a, a)))
        if not stiff:
            return [], None
        return [n for _, n in sorted(stiff)], min(t for t, _ in stiff)

    def safe_step(self, values, step=None):
        """The physics step to run these values at, or None when `step` (default: the one in use) is safe. It divides
        the logs' sample time."""
        axes, tau = self.stiff_axes(values, step or self.step_override or self.world_step())
        if not axes:
            return None
        ok = [st for st in (0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001)
              if st <= 0.5 * tau and abs(self.dt / st - round(self.dt / st)) < 1e-9]
        return ok[0] if ok else 0.0001

    def after_fit_notes(self, values) -> list:
        step = self.world_step()
        axes, tau = self.stiff_axes(values, step)
        if not axes:
            return []
        safe = self.safe_step(values, step)
        return [f"Your world's physics step ({step:g} s) is too long for the {', '.join(axes)} damping now in the "
                f"model (time constant {tau * 1000:.0f} ms): Gazebo applies it step by step, and it can blow up. Strathe "
                f"scored the models at {safe:g} s, and `calibrated_world/` runs at it "
                f"(`<max_step_size>{safe:g}</max_step_size>`). To keep your own step, `--hold` those damping values."]

    def write_deliverable(self, values, out_dir: str) -> str:
        """`calibrated_world/` in out_dir: a copy of the lab's world with the calibrated model in it (copied, not
        linked, and included as model:// so it runs when moved), at the safe physics step when the world's is too long.
        calibrated_world/RUN_ME.md says how to run it (run_instructions)."""
        import shutil
        dest = os.path.join(out_dir, "calibrated_world")
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.makedirs(dest)
        model_text = self._edit(values)
        name = os.path.basename(self.world_path)
        if self.separate:
            src = os.path.dirname(self.model_path)
            dst = os.path.join(dest, os.path.basename(src))
            shutil.copytree(src, dst, symlinks=False)
            open(os.path.join(dst, os.path.basename(self.model_path)), "w", encoding="utf-8").write(model_text)
            # model://, not an absolute path, so the folder still runs when it's moved or shared (gz sim finds it
            # through GZ_SIM_RESOURCE_PATH, with this folder first so the calibrated copy wins over your original)
            world = self._include(self.world_text, src, f"model://{os.path.basename(src)}")
        else:
            world = model_text
        p = os.path.join(dest, name)
        open(p, "w", encoding="utf-8").write(self._with_step(world))
        open(os.path.join(dest, "RUN_ME.md"), "w", encoding="utf-8").write(self.run_instructions(name) + "\n")
        return p

    def run_command(self, world_name: str, folder: str = "calibrated_world") -> str:
        """The macOS / Linux command that runs the world copy, from the folder that contains it."""
        if not self.separate:
            return f"gz sim {folder}/{world_name}"
        return f"GZ_SIM_RESOURCE_PATH={folder}:$GZ_SIM_RESOURCE_PATH gz sim {folder}/{world_name}"

    def run_instructions(self, world_name: str, folder: str = "calibrated_world") -> str:
        """How to run the world copy, from the folder that contains it."""
        return self._run_instructions(world_name, folder) + (
            "\n\nThis fit ran Gazebo in Docker, which has no window: `gz sim` needs Gazebo Harmonic installed on this "
            "machine (calibration/simulators/gazebo/README.md, route A)." if self.tool and self.tool.tool
            and str(self.tool.tool[0]).startswith("docker:") else "")

    def _run_instructions(self, world_name: str, folder: str) -> str:
        if not self.separate:
            return (f"Run it from the folder that contains `{folder}/`:\n\n    {self.run_command(world_name, folder)}\n\n"
                    f"(`-s` runs it without the window.)")
        return (f"Run it from the folder that contains `{folder}/` (the model folder in it must come first on Gazebo's "
                f"resource path, so the calibrated copy is used and not your original):\n\n"
                f"macOS / Linux:\n\n    {self.run_command(world_name, folder)}\n\n"
                f"Windows (PowerShell):\n\n    $env:GZ_SIM_RESOURCE_PATH=\"{folder};$env:GZ_SIM_RESOURCE_PATH\"; "
                f"gz sim {folder}/{world_name}\n\n"
                f"(`-s` runs it without the window.) The folder can be moved or shared: nothing in it points elsewhere.")

    def _with_step(self, world: str) -> str:
        if not self.step_override:
            return world
        w = XmlFile(world)
        if w.exists("//physics/max_step_size"):
            w.set("//physics/max_step_size", self.step_override)
            return w.text
        return world

    def write_calibrated(self, values, out_dir: str, suffix: str) -> str:
        stem, ext = os.path.splitext(os.path.basename(self.model_path))
        p = os.path.join(out_dir, f"{stem}_{suffix}{ext}")
        open(p, "w", encoding="utf-8").write(self._edit(values))
        return p

    def changes(self, values) -> list:
        L = []
        orig = XmlFile(self.model_text)
        defs = {n: (p, s) for n, _g, p, s, _t in self._knob_defs()}
        for key, v in values.items():
            if key in defs:
                old = orig.get(defs[key][0])
                new = float(v) * defs[key][1]
                if abs(old - new) > 1e-12 * max(1.0, abs(old)):
                    L.append(f"- {key}: {fmt_number(old)} -> {fmt_number(new)} (`{defs[key][0]}`)")
            elif key == "buoyancy_scale" and abs(float(v) - 1) > 1e-9:
                L.append(f"- buoyant volume (the link's collisions): x{float(v):.3f}")
            elif key == "collision_shift":
                L.append(f"- the link's collisions moved by {', '.join(f'{x:+.3f}' for x in v)} m (x, y, z, forward-left-up)")
            elif key.startswith(("@", "+")):
                L.append(f"- `{key[1:].replace('|', '/')}` = {v if isinstance(v, str) else fmt_number(v)}")
        return L or ["- nothing (the fit kept every value)"]

    # --- export: Strathe's model in Gazebo ---------------------------------------------------------------------------
    def export_values(self, params, mass: float, info: dict, train: list, rho: float):
        notes, v = [], {}
        add = lambda parent, tag, x: v.__setitem__(f"+{parent}|{tag}", fmt_number(x))  # noqa: E731
        # Gazebo's plugin damps at the link's origin; Strathe's model at the logs' reference point. The linear part moves
        # there exactly (cross terms of the 6x6 matrix); the quadratic part can't be expressed at another point and stays
        # diagonal at the origin (exact when the reference point is the link origin, the usual base_link set-up).
        r = np.array([self.ref[0], -self.ref[1], -self.ref[2]])  # forward-left-up, from the link origin
        S = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        Dl = np.diag([getattr(params, n) for n in STRATHE_LIN])
        A = np.block([[np.eye(3), np.zeros((3, 3))], [S, np.eye(3)]])
        Bm = np.block([[np.eye(3), -S], [np.zeros((3, 3)), np.eye(3)]])
        D6 = A @ Dl @ Bm
        for i, a in enumerate(AX):
            for j, u in enumerate(VEL):
                if i == j or abs(D6[i, j]) > 1e-12 or f"{a}{u}" in self.params:
                    add(self.plugin, f"{a}{u}", D6[i, j])
            add(self.plugin, f"{a}{VEL[i]}abs{VEL[i]}", getattr(params, STRATHE_QUAD[i]))
        if np.linalg.norm(r) > 1e-9:
            notes.append(f"The logs' reference point is {np.linalg.norm(r):.3f} m from link '{self.link}''s origin, where "
                         f"Gazebo applies damping: the linear damping is moved there exactly (cross terms), the "
                         f"quadratic damping isn't (Gazebo has no term for it), so the model is close to Strathe's but "
                         f"not exact. For an exact export, log velocity and thrust at the link's origin (base_link).")
        lift = {"yUV": params.Zuw, "zUW": params.Zuw, "mUW": params.Muw, "nUV": -params.Muw, "mUQ": params.Muq,
                "nUR": params.Muq}
        if any(lift.values()):
            for tag, x in lift.items():
                add(self.plugin, tag, x)
        # added mass through SDF's <fluid_added_mass> (the physics engine's own, with its Coriolis and Munk terms): the
        # plugin's legacy xDotU.. terms estimate accelerations explicitly and mis-turned ALPHA-like vehicles (#82)
        if self.inertial is None:
            raise SimFitError(f"link {self.link!r} has no <inertial> to put the mass and inertia in")
        fa = f"{self.link_path}/inertial/fluid_added_mass"
        if not self.fluid_added:
            v[f"+{self.link_path}/inertial|fluid_added_mass"] = ""
        # Strathe's added mass is about the reference point; Gazebo's about the link origin: moved there exactly
        # (M_origin = J^T M_ref J, with v_ref = v_origin - r x w)
        # Strathe's effective inertia (rigid + added) needn't be a rigid body's: B0 on ALPHA RISE has roll 3, pitch 60,
        # yaw 35.5 kg m2, which breaks the triangle inequality and Gazebo refuses the link (#88). So the rigid part stays
        # a valid rigid inertia (the model's own, capped at Strathe's) and the rest is rotational added mass.
        Ieff = np.array([params.Ieff_roll, params.Ieff_pitch, params.Ieff_yaw])
        rigid = rigid_inertia_within(np.array(info.get("inertia") or Ieff, dtype=float), Ieff)
        Ma = np.zeros((6, 6))
        Ma[0, 0], Ma[1, 1], Ma[2, 2] = params.Xu_dot, params.Yv_dot, params.Zw_dot
        Ma[3, 3], Ma[4, 4], Ma[5, 5] = Ieff - rigid
        Ma[0, 4] = Ma[4, 0] = -params.Xq_dot  # forward-left-up signs
        Ma[2, 4] = Ma[4, 2] = params.Zq_dot
        J = np.block([[np.eye(3), -S], [np.zeros((3, 3)), np.eye(3)]])
        Mo = J.T @ Ma @ J
        names = "xyzpqr"
        for i in range(6):
            for j in range(i, 6):
                add(fa, names[i] + names[j], Mo[i, j])
        for tag in self.params:
            if re.fullmatch(r"[xyzkmn]Dot[UVWPQR]", tag):
                add(self.plugin, tag, 0.0)
        ip = f"{self.link_path}/inertial"
        add(ip, "mass", mass)
        refl = np.array([self.ref[0], -self.ref[1], -self.ref[2]])  # forward-left-up
        v[f"+{ip}|pose"] = f"{fmt_number(refl[0])} {fmt_number(refl[1])} {fmt_number(refl[2])} 0 0 0"
        ii = f"{ip}/inertia"
        for tag, x in (("ixx", rigid[0]), ("iyy", rigid[1]), ("izz", rigid[2]), ("ixy", 0.0), ("ixz", 0.0), ("iyz", 0.0)):
            add(ii, tag, x)
        notes.append(f"Inertia: Strathe's effective inertia ({', '.join(f'{x:.3g}' for x in Ieff)} kg·m², rigid + added) "
                     f"as the link's rigid inertia ({', '.join(f'{x:.3g}' for x in rigid)}) plus rotational added mass "
                     f"(<fluid_added_mass> pp, qq, rr), so the total is Strathe's and the link is a valid rigid body.")
        # buoyancy: the collisions' volume and centre, from what the replay tool measured
        cols = [c for c in info.get("collisions", []) if c.get("volume", 0) > 0]
        if len(cols) != len(info.get("collisions", [])) or not cols:
            notes.append("Buoyancy and the centre of buoyancy not carried: the link's collisions aren't all boxes, "
                         "cylinders or spheres, so their volume isn't known.")
        else:
            vol = sum(c["volume"] for c in cols)
            cob = sum(np.array(c["center_flu"]) * c["volume"] for c in cols) / vol
            need = (mass * 9.81 + params.net_buoyancy_z) / (rho * 9.81)
            v["buoyancy_scale"] = need / vol
            target = refl + np.array([-params.cg_cb_offset_x, 0.0, params.cg_cb_offset_z])
            v["collision_shift"] = [float(x) for x in target - cob]
            notes.append(f"Net buoyancy {params.net_buoyancy_z:+.1f} N in water of {rho:g} kg/m³: the collisions "
                         f"resized x{need / vol:.3f} and moved so the centre of buoyancy sits at the calibrated offset "
                         f"from the centre of mass (at the logs' reference point, as in Strathe's model).")
        if params.thruster_time_constant_s:
            notes.append(f"Thruster lag ({params.thruster_time_constant_s:.2f} s) not carried: set it on your thrusters.")
        return v, notes


def rigid_inertia_within(model: np.ndarray, Ieff: np.ndarray) -> np.ndarray:
    """A rigid inertia (principal moments) no larger than Strathe's effective inertia on any axis and satisfying the
    triangle inequality: the model's own, capped at Strathe's; else isotropic at half the smallest."""
    r = np.minimum(np.maximum(model, 1e-6), Ieff)
    if all(r[i] <= r[(i + 1) % 3] + r[(i + 2) % 3] + 1e-12 for i in range(3)):
        return r
    return np.full(3, 0.5 * float(Ieff.min()))


def add_arguments(ap) -> None:
    ap.add_argument("--world", required=True, help="your Gazebo world (.sdf) with the vehicle in it, or including it")
    ap.add_argument("--model-file", help="the vehicle's model.sdf, when the world <include>s it from a model folder "
                                         "(default: the vehicle is in the world file)")
    ap.add_argument("--model-name", help="the vehicle's model name (default: the one with the Hydrodynamics plugin)")
    ap.add_argument("--link", help="the vehicle's link (default: the Hydrodynamics plugin's <link_name>)")
    ap.add_argument("--start-depth", type=float, default=3.0, help="depth the windows are replayed at, m")
    ap.add_argument("--resource-path", help="extra folders for model:// URIs (as GZ_SIM_RESOURCE_PATH)")
    ap.add_argument("--replay-tool", help="the built strathe_gz_replay; or docker: Strathe's Gazebo image (setup-sim "
                                          "gazebo --docker, once); or docker:IMAGE (default: calibration/simulators/"
                                          "gazebo/build/, else the Docker image if it's built)")


def adapter_from_args(a) -> GazeboAdapter:
    tool = find(a.replay_tool, "STRATHE_GZ_REPLAY", DEFAULT_TOOL, "Gazebo",
                "Build it: python strathe.py setup-sim gazebo (calibration/simulators/gazebo/README.md)", sim="gazebo")
    return GazeboAdapter(a.world, a.model_file, a.model_name, a.link, tool, a.reference_point, a.start_depth,
                         a.resource_path)
