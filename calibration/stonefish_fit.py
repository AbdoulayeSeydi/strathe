"""
stonefish_fit.py -- calibrate a lab's own Stonefish vehicle model from its logs (`strathe fit-stonefish`,
PROJECT_STATUS.md #72).

Strathe's calibration drives the simulator the lab already uses. It fits Stonefish's own settings on one part of the
vehicle -- quadratic and viscous drag per axis, added mass per axis (needs calibration/stonefish/
stonefish_added_mass.patch), the buoyancy (by resizing a buoyant primitive part) and the centre of gravity's position
-- to the training recordings. It then scores the original and the calibrated model on recordings it never saw, with
the same windows, logged starting states and logged wrench (free-running motion and attitude, still water).

Stonefish runs through calibration/stonefish/strathe_stonefish_replay (build it with that folder's CMakeLists.txt);
any command that speaks its interface can stand in (the tests use a small Python one).

Outputs, in the output folder:
  <vehicle file>_calibrated.scn  the lab's own vehicle file with only the tuned part (and buoyancy part) changed
  stonefish_fit.yaml             the settings before and after, and how the fit went
  stonefish_fit_report.md        held-out drift / velocity / attitude: original vs calibrated (vs Strathe, if given)
  drift.png                      the same, as a plot
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOOL = os.path.join(HERE, "stonefish", "build", "strathe_stonefish_replay")
KNOBS = ("drag", "added_mass", "buoyancy", "cg")
FIT_WINDOW_S, FIT_STRIDE_S = 30.0, 30.0
EVAL_WINDOW_S, EVAL_STRIDE_S, HORIZONS_S = 60.0, 10.0, (10, 30, 60)


class StonefishFitError(RuntimeError):
    """A plain message for the lab: what is wrong and what to do."""


# ---------------------------------------------------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------------------------------------------------

_STRIP = [re.compile(r"<!--.*?-->", re.S),
          re.compile(r"<sensor\b.*?</sensor>", re.S), re.compile(r"<sensor\b[^>]*/>"),
          re.compile(r"<comm\b.*?</comm>", re.S), re.compile(r"<comm\b[^>]*/>"),
          re.compile(r"<ros_(publisher|subscriber)\b[^>]*/>"),
          re.compile(r"<(light|camera|vision_sensor)\b.*?</\1>", re.S)]


def _resolve_find(text: str, find: dict, where: str) -> str:
    def rep(m):
        pkg = m.group(1).strip()
        if pkg not in find:
            raise StonefishFitError(f"{where} uses $(find {pkg}): give its folder with --find {pkg}=/path/to/{pkg}")
        return find[pkg].rstrip("/")
    return re.sub(r"\$\(find\s+([^)]+)\)", rep, text)


def _part_span(text: str, part: str) -> Optional[tuple]:
    """(start, end) of the element (external_part, internal_part, link or base_link) named `part`."""
    m = re.search(r'<(external_part|internal_part|base_link|link)\b[^>]*\bname="%s"[^>]*>' % re.escape(part), text)
    if not m:
        return None
    tag = m.group(1)
    if m.group(0).endswith("/>"):
        return m.start(), m.end()
    depth, i = 1, m.end()
    pat = re.compile(r"<(/?)%s\b[^>]*?(/?)>" % tag)
    while depth:
        n = pat.search(text, i)
        if not n:
            raise StonefishFitError(f"the element of part '{part}' isn't closed")
        if n.group(1):
            depth -= 1
        elif not n.group(2):
            depth += 1
        i = n.end()
    return m.start(), i


def _set_attr(elem: str, name: str, value: str) -> str:
    if re.search(r'\b%s="[^"]*"' % name, elem):
        return re.sub(r'\b%s="[^"]*"' % name, f'{name}="{value}"', elem, count=1)
    return re.sub(r"\s*(/?>)$", f' {name}="{value}"\\1', elem, count=1)


@dataclass
class PartSettings:
    """What fit-stonefish (or export-stonefish) writes into the tuned part (None = leave the scenario's own)."""
    quadratic_drag: Optional[list] = None
    viscous_drag: Optional[list] = None
    added_mass: Optional[list] = None
    added_inertia: Optional[list] = None
    anisotropic: bool = False
    cg_shift: Optional[list] = None          # metres added to the part's <cg xyz> (x, y, z)
    buoyancy_scale: Optional[float] = None   # volume factor for the buoyancy part
    mass: Optional[float] = None             # the part's <mass value>, kg
    inertia: Optional[list] = None           # the part's <inertia xyz>, kg m2 (rigid, about its CG)


def _search(pattern: str, text: str):
    """re.search that skips XML comments (a commented-out <inertia .../> isn't the part's inertia)."""
    masked = re.sub(r"<!--.*?-->", lambda m: " " * len(m.group(0)), text, flags=re.S)
    return re.search(pattern, masked)


def _replace_or_add(body: str, tag: str, attr: str, value: str, part: str) -> str:
    """Set <tag attr=value/> in the part's element, adding the element after <mass .../> (or at the end) if absent."""
    m = _search(r"<%s\b[^>]*/>" % tag, body)
    if m:
        return body[:m.start()] + _set_attr(body[m.start():m.end()], attr, value) + body[m.end():]
    anchor = _search(r"<mass\b[^>]*/>", body)
    el = f'<{tag} {attr}="{value}"/>'
    if anchor:
        line = body.rfind("\n", 0, anchor.start())
        ind = body[line + 1:anchor.start()] if line >= 0 else "    "
        return body[:anchor.end()] + f"\n{ind}{el}" + body[anchor.end():]
    close = body.rfind("</")
    if close <= 0:
        raise StonefishFitError(f"part '{part}' is a single empty element: can't add <{tag}>")
    return body[:close] + f"    {el}\n" + body[close:]


def merge_settings(base: Optional[PartSettings], top: Optional[PartSettings]) -> Optional[PartSettings]:
    """`top` applied on top of `base` (fit-stonefish --from-strathe: the fit's changes on top of the exported values):
    drag, added mass and mass/inertia from `top` where it sets them, CG shifts added, buoyant-volume factors multiplied."""
    if base is None or top is None:
        return top if base is None else base
    out = PartSettings(**base.__dict__)
    for k in ("quadratic_drag", "viscous_drag", "added_mass", "added_inertia", "mass", "inertia"):
        if getattr(top, k) is not None:
            setattr(out, k, getattr(top, k))
    if top.added_mass is not None:
        out.anisotropic = top.anisotropic
    if top.cg_shift is not None:
        out.cg_shift = [a + b for a, b in zip(base.cg_shift or [0.0, 0.0, 0.0], top.cg_shift)]
    if top.buoyancy_scale is not None:
        out.buoyancy_scale = (base.buoyancy_scale or 1.0) * top.buoyancy_scale
    return out


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def apply_part_settings(text: str, part: str, s: PartSettings, buoyancy_part: Optional[str] = None) -> str:
    """The scenario text with the tuned part (and the buoyancy part) changed; everything else untouched."""
    span = _part_span(text, part)
    if span:
        a, b = span
        body = text[a:b]
        hydro = {}
        if s.quadratic_drag is not None:
            hydro["quadratic_drag"] = _fmt(s.quadratic_drag)
        if s.viscous_drag is not None:
            hydro["viscous_drag"] = _fmt(s.viscous_drag)
        if s.added_mass is not None:
            hydro["added_mass"] = _fmt(s.added_mass)
            hydro["added_inertia"] = _fmt(s.added_inertia if s.added_inertia is not None else [0, 0, 0])
            hydro["anisotropic_added_mass"] = "true" if s.anisotropic else "false"
        if hydro:
            m = _search(r"<hydrodynamics\b[^>]*/>", body)
            if m:
                el = m.group(0)
                for k, v in hydro.items():
                    el = _set_attr(el, k, v)
                body = body[:m.start()] + el + body[m.end():]
            else:
                el = "<hydrodynamics " + " ".join(f'{k}="{v}"' for k, v in hydro.items()) + "/>"
                close = body.rfind("</")
                indent = re.search(r"\n(\s*)<[^/]", body)
                ind = indent.group(1) if indent else "    "
                body = body[:close] + f"{ind}{el}\n" + body[close:] if close > 0 else body
        if s.mass is not None:
            body = _replace_or_add(body, "mass", "value", f"{float(s.mass):.6g}", part)
        if s.inertia is not None:
            body = _replace_or_add(body, "inertia", "xyz", _fmt(s.inertia), part)
        if s.cg_shift is not None and any(s.cg_shift):
            m = _search(r'<cg\b[^>]*\bxyz="([^"]*)"', body)
            if not m:
                raise StonefishFitError(f"part '{part}' has no <cg xyz=...> to shift; add one (its current centre of "
                                        f"gravity) or leave cg out of --fit")
            xyz = [float(x) + d for x, d in zip(m.group(1).split(), s.cg_shift)]
            body = body[:m.start(1)] + _fmt(xyz) + body[m.end(1):]
        text = text[:a] + body + text[b:]
    if buoyancy_part and s.buoyancy_scale is not None:
        span = _part_span(text, buoyancy_part)
        if span:
            a, b = span
            body = text[a:b]
            m = _search(r"<dimensions\b[^>]*/>", body)
            if not m:
                raise StonefishFitError(f"buoyancy part '{buoyancy_part}' needs <dimensions> (a box, cylinder or "
                                        f"sphere): a mesh part can't be resized without changing its shape")
            el = m.group(0)
            k = s.buoyancy_scale
            if 'height="' in el:  # cylinder: keep the length, scale the radius
                r = float(re.search(r'radius="([^"]*)"', el).group(1))
                el = _set_attr(el, "radius", f"{r * math.sqrt(k):.6g}")
            elif 'radius="' in el:  # sphere
                r = float(re.search(r'radius="([^"]*)"', el).group(1))
                el = _set_attr(el, "radius", f"{r * k ** (1 / 3):.6g}")
            else:  # box
                xyz = [float(x) * k ** (1 / 3) for x in re.search(r'xyz="([^"]*)"', el).group(1).split()]
                el = _set_attr(el, "xyz", _fmt(xyz))
            body = body[:m.start()] + el + body[m.end():]
            text = text[:a] + body + text[b:]
    return text


@dataclass
class Scenario:
    root: str                          # the lab's top-level scenario
    find: dict = field(default_factory=dict)
    part: str = ""
    buoyancy_part: Optional[str] = None
    vehicle_file: Optional[str] = None  # the file that defines the tuned part (set by discover())
    base: Optional[PartSettings] = None  # applied under every change (fit-stonefish --from-strathe: the exported values)

    def files(self) -> dict:
        """{original path: text with $(find) resolved}, for the root and everything it includes."""
        out, todo = {}, [os.path.abspath(self.root)]
        while todo:
            p = todo.pop()
            if p in out:
                continue
            if not os.path.exists(p):
                raise StonefishFitError(f"scenario file not found: {p}")
            text = _resolve_find(open(p, encoding="utf-8").read(), self.find, p)
            out[p] = text
            for inc in re.findall(r'<include\b[^>]*\bfile="([^"]*)"', text):
                q = inc if os.path.isabs(inc) else os.path.join(os.path.dirname(p), inc)
                todo.append(os.path.abspath(q))
        return out

    def discover(self) -> None:
        hits = [p for p, t in self.files().items() if _part_span(t, self.part)]
        if not hits:
            raise StonefishFitError(f"no part named '{self.part}' in {self.root} or the files it includes; give "
                                    f"--part the name of one of the robot's parts (e.g. its hull)")
        self.vehicle_file = hits[0]
        if self.buoyancy_part and not any(_part_span(t, self.buoyancy_part) for t in self.files().values()):
            raise StonefishFitError(f"no part named '{self.buoyancy_part}' (--buoyancy-part) in the scenario")

    def write_runnable(self, out_dir: str, settings: Optional[PartSettings] = None,
                       density: Optional[float] = None) -> str:
        """A copy of the scenario Stonefish can run headless: sensors stripped, $(find) resolved, includes pointing
        at the copies, the tuned part changed. Returns the root's path."""
        os.makedirs(out_dir, exist_ok=True)
        files = self.files()
        names = {p: os.path.join(out_dir, f"{i:02d}_{os.path.basename(p)}") for i, p in enumerate(sorted(files))}
        for p, text in files.items():
            for pat in _STRIP:
                text = pat.sub("", text)

            def inc(m, base=os.path.dirname(p)):
                q = m.group(1) if os.path.isabs(m.group(1)) else os.path.join(base, m.group(1))
                return m.group(0).replace(m.group(1), names[os.path.abspath(q)])
            text = re.sub(r'<include\b[^>]*\bfile="([^"]*)"', inc, text)
            if density is not None:
                text = re.sub(r'(<water\b[^>]*\bdensity=")[^"]*(")', rf"\g<1>{density:g}\g<2>", text)
            merged = merge_settings(self.base, settings)
            if merged is not None:
                text = apply_part_settings(text, self.part, merged, self.buoyancy_part)
            with open(names[p], "w", encoding="utf-8") as f:
                f.write(text)
        return names[os.path.abspath(self.root)]

    def write_calibrated_vehicle(self, path: str, settings: PartSettings) -> None:
        """The lab's own vehicle file (comments, sensors, everything) with only the tuned parts changed."""
        original = open(self.vehicle_file, encoding="utf-8").read()
        with open(path, "w", encoding="utf-8") as f:
            f.write(apply_part_settings(original, self.part, merge_settings(self.base, settings), self.buoyancy_part))


# ---------------------------------------------------------------------------------------------------------------------
# The replay tool
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Replay:
    tool: list                 # argv prefix: [path] or [python, fake.py]
    data_dir: str              # Stonefish's data folder (meshes)
    robot: str
    ref: tuple                 # the logs' reference point in the robot's frame (FRD, m)
    depth: float = 3.0
    hz: float = 200.0
    dt: float = 0.1

    def _argv(self, scenario):
        return self.tool + ["--scenario", scenario, "--data", self.data_dir, "--robot", self.robot,
                            "--ref", *(f"{x:g}" for x in self.ref), "--depth", f"{self.depth:g}", "--hz", f"{self.hz:g}",
                            "--dt", f"{self.dt:g}"]

    def info(self, scenario: str) -> dict:
        r = subprocess.run(self._argv(scenario) + ["--info"], capture_output=True, text=True, timeout=600)
        js = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if r.returncode or not js:
            raise StonefishFitError(f"the Stonefish replay tool couldn't load the scenario: {r.stderr.strip()[-800:]}")
        return json.loads(js[-1])

    def run(self, scenario: str, rows_by_window: list, work_dir: str) -> list:
        """[(V (n,6), Q (n,4))] per window; NaN where Stonefish diverged."""
        os.makedirs(work_dir, exist_ok=True)
        fin, fout = os.path.join(work_dir, "windows.csv"), os.path.join(work_dir, "pred.csv")
        with open(fin, "w") as f:
            f.write("win,k,u,v,w,p,q,r,qx,qy,qz,qw,Fx,Fy,Fz,Tx,Ty,Tz\n")
            for i, rows in enumerate(rows_by_window):
                for k, row in enumerate(rows):
                    f.write(f"{i},{k}," + ",".join(f"{x:.9g}" for x in row) + "\n")
        r = subprocess.run(self._argv(scenario) + ["--windows", fin, "--out", fout], capture_output=True, text=True,
                           timeout=24 * 3600)
        if r.returncode:
            raise StonefishFitError(f"the Stonefish replay failed: {r.stderr.strip()[-800:]}")
        got = {}
        with open(fout) as f:
            for row in csv.DictReader(f):
                got.setdefault(int(row["win"]), []).append([float(row[c]) for c in
                                                             ("u", "v", "w", "p", "q", "r", "qx", "qy", "qz", "qw")])
        out = []
        for i, rows in enumerate(rows_by_window):
            arr = np.full((len(rows), 10), np.nan)
            g = np.array(got.get(i, []), dtype=float).reshape(-1, 10)
            arr[:len(g)] = g
            out.append((arr[:, :6], arr[:, 6:]))
        return out


def find_tool(path: Optional[str]) -> list:
    p = path or os.environ.get("STRATHE_STONEFISH_REPLAY") or DEFAULT_TOOL
    if not (os.path.exists(p) and os.access(p, os.X_OK)):
        raise StonefishFitError(f"the Stonefish replay tool isn't built ({p}). Build it against your Stonefish: "
                                f"calibration/stonefish/README.md, or give its path with --replay-tool")
    return [p]


# ---------------------------------------------------------------------------------------------------------------------
# Windows and scores
# ---------------------------------------------------------------------------------------------------------------------

def windows(run, length_s: float, stride_s: float) -> list:
    n = int(round(length_s / run.dt)) + 1
    return [(a, n) for a in range(0, len(run.real_nu) - n + 1, max(1, int(round(stride_s / run.dt))))]


def window_rows(run, wins) -> list:
    return [np.hstack([run.real_nu[a:a + n], run.orientation_quat[a:a + n], run.forces_torques[a:a + n]])
            for a, n in wins]


def path(V, Q, dt):
    R = Rotation.from_quat(np.nan_to_num(Q) + np.array([0, 0, 0, 1e-12]))
    return np.vstack([np.zeros(3), np.cumsum(R.apply(V[:, :3]) * dt, axis=0)[:-1]])


def score(runs_wins: list, preds: list, horizons=HORIZONS_S) -> dict:
    """Median drift, velocity RMSE per axis and attitude error per horizon, over all windows of all recordings."""
    vel = {h: [] for h in horizons}
    drift = {h: [] for h in horizons}
    att = {h: [] for h in horizons}
    total = diverged = 0
    for (run, wins), pr in zip(runs_wins, preds):
        for (a, n), (V, Q) in zip(wins, pr):
            total += 1
            if np.isnan(V).any():
                diverged += 1
                continue
            rv, rq = run.real_nu[a:a + n], run.orientation_quat[a:a + n]
            Pp, Pr = path(V, Q, run.dt), path(rv, rq, run.dt)
            e = np.degrees(np.abs((Rotation.from_quat(Q).as_euler("xyz") - Rotation.from_quat(rq).as_euler("xyz")
                                   + np.pi) % (2 * np.pi) - np.pi))
            for h in horizons:
                m = int(round(h / run.dt))
                if m >= n:
                    continue
                vel[h].append((V[1:m + 1] - rv[1:m + 1]) ** 2)
                drift[h].append(float(np.linalg.norm(Pp[m] - Pr[m])))
                att[h].append(e[m])
    out = {"windows": total, "diverged": diverged}
    for h in horizons:
        if drift[h]:
            out[f"{h}s"] = {"drift_median_m": round(float(np.median(drift[h])), 3),
                            "drift_p90_m": round(float(np.percentile(drift[h], 90)), 3),
                            "velocity_rmse": np.sqrt(np.concatenate(vel[h]).mean(axis=0)).round(4).tolist(),
                            "attitude_error_median_deg": np.median(np.array(att[h]), axis=0).round(2).tolist()}
    return out


def strathe_free_run(params, vehicle, run, a: int, n: int):
    """Strathe's own model over the same window: its physics step, its own attitude (as validation.free_run)."""
    from dynamics_model import SimulationConfig
    from fast_dynamics import FastDynamics
    acc = FastDynamics(params, vehicle).acceleration_trig
    cfg = SimulationConfig(dt=run.dt)
    sub, h = cfg.n_substeps, cfg.sub_dt
    qx, qy, qz, qw = (float(x) for x in run.orientation_quat[a])
    v = [float(x) for x in run.real_nu[a]]
    V, Q = np.full((n, 6), np.nan), np.full((n, 4), np.nan)
    for k in range(n):
        V[k], Q[k] = v, (qx, qy, qz, qw)
        if k + 1 == n:
            break
        tau = run.forces_torques[a + k].tolist()
        for _ in range(sub):
            r20, r21, r22 = 2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)
            vd = acc(v, tau, r21, r22, -r20, 1.0)
            v = [v[j] + vd[j] * h for j in range(6)]
            if not all(math.isfinite(x) and abs(x) < 1e3 for x in v):
                return V, Q
            wx, wy, wz = v[3] * h, v[4] * h, v[5] * h
            th = math.sqrt(wx * wx + wy * wy + wz * wz)
            if th > 1e-12:
                c, s_ = math.cos(th / 2), math.sin(th / 2) / th
                dx, dy, dz = wx * s_, wy * s_, wz * s_
                qw, qx, qy, qz = (qw * c - qx * dx - qy * dy - qz * dz, qw * dx + qx * c + qy * dz - qz * dy,
                                  qw * dy - qx * dz + qy * c + qz * dx, qw * dz + qx * dy - qy * dx + qz * c)
        nq = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        qx, qy, qz, qw = qx / nq, qy / nq, qz / nq, qw / nq
    return V, Q


def baselines(runs_wins: list) -> dict:
    zero, keep = [], []
    for run, wins in runs_wins:
        zero.append([(np.zeros((n, 6)), np.repeat(run.orientation_quat[a:a + 1], n, axis=0)) for a, n in wins])
        keep.append([(np.repeat(run.real_nu[a:a + 1], n, axis=0), np.repeat(run.orientation_quat[a:a + 1], n, axis=0))
                     for a, n in wins])
    return {"no motion": score(runs_wins, zero), "keeps its starting velocity": score(runs_wins, keep)}


# ---------------------------------------------------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Knobs:
    """The tuned settings, by name, and the parameter vector <-> PartSettings mapping. Each setting is fitted (in the
    vector: log scale for drag and added mass, a log factor for the buoyant volume, metres for the CG shift) or held
    at a value the lab measured (`--hold added_mass_x=10.6`)."""
    fit: tuple
    start: dict             # the tuned part's settings from the replay tool's --info
    axisymmetric: bool = False
    held: dict = field(default_factory=dict)   # {setting name: physical value}

    def _ax(self):
        return ["x", "yz"] if self.axisymmetric else ["x", "y", "z"]

    def all_names(self) -> list:
        n = []
        if "drag" in self.fit:
            n += [f"quadratic_drag_{a}" for a in self._ax()] + [f"viscous_drag_{a}" for a in self._ax()]
        if "added_mass" in self.fit:
            n += [f"added_mass_{a}" for a in self._ax()]
        if "buoyancy" in self.fit:
            n += ["buoyancy_scale"]
        if "cg" in self.fit:
            n += ["cg_shift_x", "cg_shift_z"]
        return n

    def __post_init__(self):
        unknown = set(self.held) - set(self.all_names())
        if unknown:
            raise StonefishFitError(f"can't hold {', '.join(sorted(unknown))}: the settings being fitted are "
                                    f"{', '.join(self.all_names()) or 'none'} (add the group to --fit to hold one of it)")

    def names(self) -> list:
        """The fitted settings (the parameter vector's entries)."""
        return [n for n in self.all_names() if n not in self.held]

    def _start_value(self, name: str) -> float:
        kind, _, axis = name.rpartition("_")
        i = {"x": 0, "y": 1, "yz": 1, "z": 2}[axis]
        floor = {"quadratic_drag": 1e-4, "viscous_drag": 1e-5, "added_mass": 1e-2}[kind]
        if axis == "yz":  # one value across the hull: start between the two (an exported vehicle can differ, #74)
            return math.sqrt(max(float(self.start[kind][1]), floor) * max(float(self.start[kind][2]), floor))
        return max(float(self.start[kind][i]), floor)

    def _param(self, name: str, value: float) -> float:
        """A setting's physical value -> its parameter."""
        if name == "buoyancy_scale":
            return math.log(value)
        if name.startswith("cg_shift"):
            return value
        if value <= 0:
            raise StonefishFitError(f"{name} must be positive, got {value}")
        return math.log(value / self._start_value(name))

    def _value(self, name: str, p: float) -> float:
        if name == "buoyancy_scale":
            return math.exp(p)
        if name.startswith("cg_shift"):
            return p
        return self._start_value(name) * math.exp(p)

    def x0(self) -> np.ndarray:
        return np.zeros(len(self.names()))

    def bounds(self):
        lo, hi = [], []
        for n in self.names():
            if n == "buoyancy_scale":
                lo.append(math.log(0.5)); hi.append(math.log(2.0))
            elif n.startswith("cg_shift"):
                lo.append(-0.15); hi.append(0.15)
            else:
                lo.append(-math.log(30.0)); hi.append(math.log(30.0))
        return np.array(lo), np.array(hi)

    def values(self, x) -> dict:
        """{setting: physical value} for a parameter vector, held settings included."""
        out = dict(zip(self.names(), (self._value(n, float(p)) for n, p in zip(self.names(), x))))
        out.update({n: float(v) for n, v in self.held.items()})
        return {n: out[n] for n in self.all_names()}

    def _axes3(self, v: dict, kind: str) -> list:
        if self.axisymmetric:
            return [v[f"{kind}_x"], v[f"{kind}_yz"], v[f"{kind}_yz"]]
        return [v[f"{kind}_{a}"] for a in "xyz"]

    def settings(self, x) -> PartSettings:
        v = self.values(x)
        s = PartSettings()
        if "drag" in self.fit:
            s.quadratic_drag = self._axes3(v, "quadratic_drag")
            s.viscous_drag = self._axes3(v, "viscous_drag")
        if "added_mass" in self.fit:
            s.added_mass = self._axes3(v, "added_mass")
            s.added_inertia = list(self.start["added_inertia"])
            s.anisotropic = True
        if "buoyancy" in self.fit:
            s.buoyancy_scale = v["buoyancy_scale"]
        if "cg" in self.fit:
            s.cg_shift = [v["cg_shift_x"], 0.0, v["cg_shift_z"]]
        return s


LINEAR_FLOOR, ANGULAR_FLOOR = 0.05, 0.02  # m/s, rad/s: an axis that barely varies isn't weighted without limit


def axis_scales(velocities: np.ndarray, weighting: str = "variation") -> np.ndarray:
    """What each axis's velocity error is divided by in the fit. 'variation' (default): the axis's standard deviation
    over the training recordings, floored, so a vehicle cruising steadily at 0.7 m/s weighs surge errors by how much
    surge actually changes. 'magnitude': its RMS (Strathe's own objective's scale; #72's first runs), which makes
    surge errors cheap during a cruise and let the fit trade surge for pitch and yaw (#73)."""
    if weighting == "magnitude":
        sc = np.sqrt(np.mean(velocities ** 2, axis=0))
    elif weighting == "variation":
        sc = np.std(velocities, axis=0)
    else:
        raise StonefishFitError(f"unknown weighting {weighting!r}: use variation or magnitude")
    return np.maximum(sc, np.array([LINEAR_FLOOR] * 3 + [ANGULAR_FLOOR] * 3))


def fit(scenario: Scenario, replay: Replay, train: list, knobs: Knobs, work: str, densities: dict,
        workers: int = 4, max_iterations: int = 15, drift_weight: float = 1.0, log=print,
        weighting: str = "variation") -> dict:
    """Least squares over Stonefish's own settings on the training recordings (30 s free-running windows every 30 s;
    velocity error per axis scaled by axis_scales, plus end-of-window drift scaled by the no-motion drift). The
    Jacobian's Stonefish runs go in parallel."""
    from scipy.optimize import least_squares
    runs_wins = [(run, windows(run, FIT_WINDOW_S, FIT_STRIDE_S)) for run in train]
    if not any(w for _, w in runs_wins):
        raise StonefishFitError(f"the training recordings are shorter than one {FIT_WINDOW_S:g} s window")
    rows = [window_rows(run, w) for run, w in runs_wins]
    scale = axis_scales(np.vstack([run.real_nu for run in train]), weighting)
    real_paths = [[path(run.real_nu[a:a + n], run.orientation_quat[a:a + n], run.dt) for a, n in w] for run, w in runs_wins]
    d0 = max(float(np.median([np.linalg.norm(P[-1]) for ps in real_paths for P in ps])), 0.1)
    counter = {"n": 0}

    def residuals_at(x, tag):
        d = os.path.join(work, f"eval_{tag}")
        out = []
        for i, ((run, wins), rw, rp) in enumerate(zip(runs_wins, rows, real_paths)):
            scn = scenario.write_runnable(os.path.join(d, f"scn{i}"), knobs.settings(x), densities.get(run.date))
            preds = replay.run(scn, rw, os.path.join(d, f"run{i}"))
            for (a, n), (V, Q), Pr in zip(wins, preds, rp):
                if np.isnan(V).any():
                    out.append(np.full(6 * (n - 1) + 1, 10.0))
                    continue
                e = ((V[1:] - run.real_nu[a + 1:a + n]) / scale).ravel() / math.sqrt(n - 1)
                dr = np.linalg.norm(path(V, Q, run.dt)[-1] - Pr[-1]) / d0
                out.append(np.append(e, math.sqrt(drift_weight) * dr))
        shutil.rmtree(d, ignore_errors=True)
        return np.concatenate(out)

    cache = {}

    def fun(x):
        key = tuple(np.round(x, 12))
        if key in cache:
            return cache[key]
        counter["n"] += 1
        r = residuals_at(x, f"f{counter['n']}")
        log(f"  evaluation {counter['n']}: cost {0.5 * float(r @ r):.4f}")
        cache.clear()
        cache[key] = r
        return r

    lo, hi = knobs.bounds()
    h = 0.05

    def jac(x):
        f0 = fun(x)
        cols = [None] * len(x)

        def col(j):
            xj = x.copy()
            step = h if x[j] + h <= hi[j] else -h
            xj[j] += step
            return j, (residuals_at(xj, f"j{counter['n']}_{j}") - f0) / step
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            for j, c in ex.map(col, range(len(x))):
                cols[j] = c
        return np.column_stack(cols)

    x0 = knobs.x0()
    r0 = fun(x0)  # cached: least_squares' first evaluation reuses it
    res = least_squares(fun, x0, jac=jac, bounds=(lo, hi), method="trf", max_nfev=max_iterations, x_scale=1.0)
    best = res.x
    return {"x": best.tolist(), "names": knobs.names(), "values": knobs.values(best), "axis_scales": scale.tolist(),
            "start_cost": 0.5 * float(r0 @ r0),
            "cost": float(res.cost), "evaluations": counter["n"], "status": res.message,
            "settings": knobs.settings(best)}


# ---------------------------------------------------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------------------------------------------------

def _load_runs(data_dir: str, heldout_dates: Optional[list]):
    from data_split import discover_csv_files, split_by_date
    from objective import DatasetConfig, load_run
    files = discover_csv_files(data_dir)[0]
    split = split_by_date(files, heldout_dates=heldout_dates)
    rate = None
    rp = os.path.join(data_dir, "ingest_report.json")
    if os.path.exists(rp):
        rate = json.load(open(rp, encoding="utf-8")).get("sampling_rate_hz")
    cfg = DatasetConfig(float(rate or 10.0), (0.0, 0.0, 0.0), None, None)  # torques as logged, about the reference
    return [load_run(f, cfg) for f in split.train_files], [load_run(f, cfg) for f in split.heldout_files], split


def _table(results: dict, horizons=HORIZONS_S) -> list:
    L = ["| Model | " + " | ".join(f"drift {h} s" for h in horizons) + " | velocity error over 10 s, u v w (m/s) "
         "| roll, pitch, yaw after 10 s | diverged |", "|---|" + "---|" * (len(horizons) + 3)]
    for name, r in results.items():
        cells = [f"{r[f'{h}s']['drift_median_m']:.2f} m" if f"{h}s" in r else "n/a" for h in horizons]
        v = r.get("10s", {}).get("velocity_rmse")
        a = r.get("10s", {}).get("attitude_error_median_deg")
        L.append(f"| {name} | " + " | ".join(cells) + f" | {', '.join(f'{x:.3f}' for x in v[:3]) if v else 'n/a'} "
                 f"| {', '.join(f'{x:.1f}°' for x in a) if a else 'n/a'} | {r['diverged']}/{r['windows']} |")
    return L


def _plot(results: dict, path_png: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, r in results.items():
        hs = [h for h in HORIZONS_S if f"{h}s" in r]
        ax.plot(hs, [r[f"{h}s"]["drift_median_m"] for h in hs], marker="o", label=name)
    ax.set_xlabel("prediction horizon [s]"); ax.set_ylabel("median position error [m]")
    ax.set_title("Unseen recordings: original vs calibrated Stonefish model"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path_png, dpi=120); plt.close(fig)


def run_fit_stonefish(a) -> int:
    scenario = Scenario(a.scenario, dict(kv.split("=", 1) for kv in (a.find or [])), a.part, a.buoyancy_part)
    scenario.discover()
    replay = Replay(a.replay_tool_argv, a.stonefish_data or os.path.dirname(os.path.abspath(a.scenario)), a.robot,
                    tuple(a.reference_point), a.start_depth, a.physics_hz)
    out = a.out
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="strathe_stonefish_")
    try:
        train, held, split = _load_runs(a.data, a.heldout_dates)
        if not held:
            raise StonefishFitError("no held-out recordings: give --heldout-dates, or logs from at least two sessions")
        replay.dt = train[0].dt
        densities = dict((d, float(r)) for d, r in (kv.split("=", 1) for kv in (a.density_by_date or [])))
        from_strathe = None
        if a.from_strathe:  # start from the Strathe calibration's values, converted (export-stonefish)
            from stonefish_export import axis_speeds, export_settings, load_calibration, training_density
            sp, smass, pfile = load_calibration(a.from_strathe, a.compare_mass)
            counter = {"n": 0}

            def measure(settings):
                counter["n"] += 1
                return replay.info(scenario.write_runnable(os.path.join(work, f"export{counter['n']}"), settings))
            rho = training_density(split.train_groups, densities, measure(None).get("water_density") or 1025.0)
            from_strathe = export_settings(sp, smass, measure, a.part, a.buoyancy_part, rho,
                                           [axis_speeds(train, i) for i in range(3)])
            scenario.base = from_strathe.settings
            a.compare_params, a.compare_mass = a.compare_params or pfile, a.compare_mass or smass
            print(f"Starting from {pfile}'s values, converted into Stonefish's settings (mass {smass:g} kg).", flush=True)
        info = replay.info(scenario.write_runnable(os.path.join(work, "info")))
        parts = {p["name"].split("/")[-1]: p for p in info["parts"]}
        if a.part not in parts:
            raise StonefishFitError(f"Stonefish doesn't report a part '{a.part}' on robot '{a.robot}' (it has: "
                                    f"{', '.join(parts) or 'none listed; is the replay tool built against the patched '
                                    'Stonefish?'})")
        fit_knobs = tuple(k for k in (a.fit or KNOBS))
        notes = []
        if "added_mass" in fit_knobs and not info.get("patched"):
            fit_knobs = tuple(k for k in fit_knobs if k != "added_mass")
            notes.append("Added mass not fitted: the replay tool is built against Stonefish without "
                         "stonefish_added_mass.patch (calibration/stonefish/README.md).")
        if "buoyancy" in fit_knobs and not a.buoyancy_part:
            fit_knobs = tuple(k for k in fit_knobs if k != "buoyancy")
            notes.append("Buoyancy not fitted: give --buoyancy-part (a buoyant box, cylinder or sphere part).")
        holds = {}
        for kv in (a.hold or []):
            name, eq, value = kv.partition("=")
            try:
                holds[name.strip()] = float(value)
            except ValueError:
                raise StonefishFitError(f"--hold takes NAME=VALUE (e.g. added_mass_x=10.6), got {kv!r}")
        knobs = Knobs(fit_knobs, parts[a.part], a.axisymmetric, holds)
        if holds:
            print("  held: " + ", ".join(f"{n} = {v:g}" for n, v in holds.items()), flush=True)
        print(f"Fitting {', '.join(fit_knobs)} on part '{a.part}' ({len(knobs.names())} values) from "
              f"{len(train)} training recording(s); {len(held)} held out ({', '.join(split.heldout_groups)}).", flush=True)
        for n in notes:
            print("  note: " + n, flush=True)
        f = fit(scenario, replay, train, knobs, work, densities, a.workers, a.max_iterations,
                log=lambda s: print(s, flush=True), weighting=a.weighting)
        s = f["settings"]
        vehicle_out = os.path.join(out, os.path.splitext(os.path.basename(scenario.vehicle_file))[0] + "_calibrated.scn")
        scenario.write_calibrated_vehicle(vehicle_out, s)

        print("Scoring the original and calibrated models on the held-out recordings...", flush=True)
        runs_wins = [(run, windows(run, EVAL_WINDOW_S, EVAL_STRIDE_S)) for run in held]
        results = {}
        base = scenario.base
        models = [("Stonefish, original", None, None)]
        if base is not None:
            models.append(("Stonefish with Strathe's values", base, None))
        models.append(("Stonefish, calibrated by Strathe", base, s))
        for m, (label, under, settings) in enumerate(models):
            scenario.base = under
            preds = []
            for i, (run, wins) in enumerate(runs_wins):
                scn = scenario.write_runnable(os.path.join(work, f"eval{m}_scn{i}"), settings, densities.get(run.date))
                preds.append(replay.run(scn, window_rows(run, wins), os.path.join(work, f"eval{m}_run{i}")))
            results[label] = score(runs_wins, preds)
        scenario.base = base
        if a.compare_params:
            from dynamics_model import CalibrationParameters, VehicleConstants
            doc = yaml.safe_load(open(a.compare_params, encoding="utf-8"))
            p = CalibrationParameters(**doc["parameters"])
            mass = a.compare_mass or doc.get("vehicle_mass_kg") or info["base"]["mass"]
            results["Strathe's own model"] = score(runs_wins, [[strathe_free_run(p, VehicleConstants(mass), run, a_, n)
                                                                for a_, n in wins] for run, wins in runs_wins])
        results.update(baselines(runs_wins))

        summary = {"scenario": os.path.abspath(a.scenario), "robot": a.robot, "part": a.part,
                   "buoyancy_part": a.buoyancy_part, "reference_point_m": list(a.reference_point),
                   "fitted": list(fit_knobs), "axisymmetric": bool(a.axisymmetric), "notes": notes,
                   "held": holds, "weighting": a.weighting,
                   "from_strathe": {"calibration": a.compare_params, "exported": {k: v for k, v in
                                    from_strathe.settings.__dict__.items() if v is not None},
                                    "notes": from_strathe.notes} if from_strathe else None, "axis_scales": f["axis_scales"], "values": f["values"],
                   "train_dates": split.train_groups, "heldout_dates": split.heldout_groups,
                   "fit": {k: f[k] for k in ("start_cost", "cost", "evaluations", "status")},
                   "before": {k: parts[a.part][k] for k in ("quadratic_drag", "viscous_drag", "added_mass", "added_inertia")},
                   "after": {k: v for k, v in s.__dict__.items() if v is not None},
                   "stonefish_patched": bool(info.get("patched")),
                   "calibrated_vehicle_file": vehicle_out, "heldout": results}
        yaml.safe_dump(json.loads(json.dumps(summary)), open(os.path.join(out, "stonefish_fit.yaml"), "w"), sort_keys=False)
        _plot(results, os.path.join(out, "drift.png"))
        orig = results["Stonefish, original"].get("60s", results["Stonefish, original"].get("30s", {}))
        cal = results["Stonefish, calibrated by Strathe"].get("60s", results["Stonefish, calibrated by Strathe"].get("30s", {}))
        L = [f"# Stonefish model calibrated from your logs: {os.path.basename(scenario.vehicle_file)}\n",
             "## Bottom line\n"]
        if orig and cal:
            change = 100 * (cal["drift_median_m"] - orig["drift_median_m"]) / max(orig["drift_median_m"], 1e-9)
            L.append(f"On recordings it never saw ({', '.join(split.heldout_groups)}), the calibrated model's position "
                     f"error after the longest horizon is **{cal['drift_median_m']:.2f} m against "
                     f"{orig['drift_median_m']:.2f} m** for the original ({change:+.0f}%). Check every horizon below: "
                     f"a calibration can help over a minute and cost over seconds.\n")
        L += ["## How the models compare on unseen recordings\n",
              f"Windows of {EVAL_WINDOW_S:g} s every {EVAL_STRIDE_S:g} s; each starts from the logged state and is "
              f"driven by the logged forces, the model turning and tilting on its own, in still water. Medians.\n"]
        L += _table(results)
        L += ["", "![Drift](drift.png)", "",
              "\"Keeps its starting velocity\" is a hard baseline over a closed-loop mission: the logged forces react to "
              "the real vehicle, so any open-loop model's error compounds. These numbers rank models; they don't "
              "promise minute-ahead prediction.\n",
              "## What changed in your vehicle file\n",
              f"`{os.path.basename(vehicle_out)}` is `{os.path.basename(scenario.vehicle_file)}` with only part "
              f"`{a.part}`" + (f" and the buoyancy part `{a.buoyancy_part}`" if a.buoyancy_part else "") + " changed:\n"]
        for k in ("quadratic_drag", "viscous_drag", "added_mass"):
            if getattr(s, k) is not None:
                L.append(f"- {k.replace('_', ' ')}: {_fmt(parts[a.part][k])} -> {_fmt(getattr(s, k))}")
        if s.buoyancy_scale is not None:
            L.append(f"- buoyant volume of `{a.buoyancy_part}`: x{s.buoyancy_scale:.3f}")
        if s.cg_shift is not None:
            L.append(f"- centre of gravity moved by {s.cg_shift[0]:+.3f} m (x), {s.cg_shift[2]:+.3f} m (z)")
        if s.added_mass is not None:
            L.append("- `anisotropic_added_mass=\"true\"`: needs Stonefish with "
                     "`calibration/stonefish/stonefish_added_mass.patch` (unpatched Stonefish ignores added mass "
                     "settings and averages it over the three axes)")
        if holds:
            L.append("- held at your values: " + ", ".join(f"{n} = {v:g}" for n, v in holds.items()))
        if from_strathe:
            fs = from_strathe.settings
            L.append(f"- started from your Strathe calibration (`{os.path.basename(os.path.dirname(a.compare_params))}`), "
                     f"converted: mass {fs.mass:.4g} kg, rigid inertia {_fmt(fs.inertia)} kg·m², added inertia "
                     f"{_fmt(fs.added_inertia or [])} kg·m², CG moved by {fs.cg_shift[0]:+.3f} m (x), {fs.cg_shift[2]:+.3f} m "
                     f"(z); the fit then tuned the rest from there (export-stonefish's report lists what can't be carried)")
        L += ["", "## How the fit went\n",
              f"Fitted: {', '.join(fit_knobs)} ({len(knobs.names())} values) on {', '.join(split.train_groups)}. Cost "
              f"{f['start_cost']:.3f} -> {f['cost']:.3f} after {f['evaluations']} evaluations ({f['status']})."]
        L += [f"- {n}" for n in notes]
        L += ["", "Fitting can trade one axis for another, or tailor values to the training sessions: odd values "
              "(drag far from the shape's, added mass at a limit) mean the logs don't pin them down. More varied "
              "manoeuvres at several speeds help."]
        open(os.path.join(out, "stonefish_fit_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
        print(f"Wrote {vehicle_out}, stonefish_fit.yaml and stonefish_fit_report.md in {out}")
        for name, r in results.items():
            hs = [f"{r[f'{h}s']['drift_median_m']:.2f}" for h in HORIZONS_S if f"{h}s" in r]
            print(f"  {name:36s} drift {' / '.join(hs)} m")
        return 0
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_arguments(ap)
    return ap


def add_arguments(ap) -> None:
    ap.add_argument("--scenario", required=True, help="your Stonefish scenario (the world file with the robot in it)")
    ap.add_argument("--robot", required=True, help="the robot's name in the scenario (after $(arg ...) substitution)")
    ap.add_argument("--part", required=True, help="the robot part to tune, e.g. its hull (an external_part name)")
    ap.add_argument("--data", required=True, help="your logs in Strathe's CSV layout (`strathe ingest`, or "
                                                  "`strathe fit-stonefish --ingest-spec`)")
    ap.add_argument("--reference-point", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                    help="where the logs' velocity and forces are measured, in the robot's frame (forward-right-down, "
                         "m from its origin), e.g. the base_link of your URDF")
    ap.add_argument("--heldout-dates", nargs="+", help="recording dates to hold out (default: the latest 30%%)")
    ap.add_argument("--buoyancy-part", help="a buoyant box/cylinder/sphere part resized to fit the net buoyancy")
    ap.add_argument("--fit", nargs="+", choices=KNOBS, help=f"what to fit (default: all of {', '.join(KNOBS)})")
    ap.add_argument("--hold", nargs="+", metavar="NAME=VALUE",
                    help="hold settings at values you measured instead of fitting them, e.g. added_mass_x=10.6 "
                         "(names: quadratic_drag_x/y/z, viscous_drag_x/y/z, added_mass_x/y/z -- _yz with "
                         "--axisymmetric --, buoyancy_scale, cg_shift_x, cg_shift_z)")
    ap.add_argument("--weighting", choices=["variation", "magnitude"], default="variation",
                    help="each axis's error is divided by its velocity's variation (default) or its magnitude (RMS)")
    ap.add_argument("--axisymmetric", action="store_true",
                    help="one drag and added mass across the hull (y = z), for a torpedo-shaped vehicle")
    ap.add_argument("--find", nargs="+", metavar="PKG=PATH", help="folders for $(find PKG) in ROS-style scenarios")
    ap.add_argument("--stonefish-data", help="Stonefish's data folder for meshes (default: the scenario's folder)")
    ap.add_argument("--density-by-date", nargs="+", metavar="DATE=RHO",
                    help="water density per recording date (e.g. 27-03-2026=1000 for fresh water)")
    ap.add_argument("--from-strathe", metavar="CALIBRATION",
                    help="start from your Strathe calibration's values converted into Stonefish's (as export-stonefish: "
                         "mass, inertia, CG, buoyancy, drag, added mass), then fit; its output folder or "
                         "calibrated_parameters.yaml")
    ap.add_argument("--compare-params", help="also score your Strathe calibration (calibrated_parameters.yaml)")
    ap.add_argument("--compare-mass", type=float, help="its vehicle mass, kg (default: the calibration's, else "
                                                       "Stonefish's)")
    ap.add_argument("--start-depth", type=float, default=3.0, help="depth the windows are replayed at, m")
    ap.add_argument("--physics-hz", type=float, default=200.0, help="Stonefish's physics rate (default 200 Hz)")
    ap.add_argument("--max-iterations", type=int, default=15, help="least-squares iterations (each: one Jacobian)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="Stonefish runs at once")
    ap.add_argument("--replay-tool", help="the built strathe_stonefish_replay (default: calibration/stonefish/build/)")
    ap.add_argument("--out", default="stonefish_fit", help="output folder")


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        a.replay_tool_argv = find_tool(a.replay_tool)
        return run_fit_stonefish(a)
    except StonefishFitError as e:
        print(f"❌ {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
