"""
vehicle_describe.py -- write a vehicle description (vehicle_spec.yaml) from basic measurements, or from your vehicle's
model in any simulator (`strathe describe-vehicle`, PROJECT_STATUS.md #95; calibration/DESCRIBE_VEHICLE.md).

    python strathe.py describe-vehicle --mass 11.5 --length 0.457 --width 0.575 --height 0.254 --out my_spec.yaml
    python strathe.py describe-vehicle --mass 33 --length 1.5 --diameter 0.25 --out my_auv_spec.yaml
    python strathe.py describe-vehicle --from-model my_robot.urdf --sim urdf --out my_spec.yaml

Every value calibration fits gets a starting value and a range sized from the hull: its mass, its size, its shape and
the water. The ranges are what matter: the logs decide the values, and a range sized from the hull keeps the fit
from wandering to values that fit the training logs and fail on unseen ones (#77). The rules:

  box frame (an ROV: L x W x H)     faces W*H (surge), L*H (sway), L*W (heave); drag coefficient 0.1-3 on each;
                                    added mass 0 to 4 x mass; inertia m(W^2+H^2)/12 ... plus as much again of water
  torpedo (L x D, L/D > 3)          frontal pi D^2/4, side L*D; drag coefficient 0.05-8 nose-on, 0.03-2.5 broadside;
                                    added mass: surge up to the displaced mass, sway/heave up to 2x strip theory
                                    rho*pi*r^2*L; pitch/yaw inertia plus slender-body rho*pi*r^2*L^3/12
  both                              net buoyancy +-12% of the weight around your trim; CG within half the height
                                    (or radius) of the CB

describe-stonefish (stonefish_describe.py) uses the same torpedo rules with its model's values as starting points.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "templates", "vehicle_spec_template.yaml")
G = 9.81
SLENDER = 3.0                 # length / diameter above which a hull is torpedo-shaped
CD_SURGE = (0.05, 8.0)        # torpedo, nose-on: a faired nose up to thrusters and sensors
CD_BROADSIDE = (0.03, 2.5)    # torpedo, side-on (sway, heave, and strip-theory pitch and yaw)
CD_BOX = (0.1, 3.0)           # an open frame, any face: porous to a solid plate with things hanging off it
LINEAR_SPEED = 1.0            # m/s: linear drag limit = this x the quadratic limit
STOP_TIME = 0.1               # s: an open frame's turning damping may stop its largest inertia this fast
MIN_LENGTH, MIN_MASS = 0.05, 0.01   # m, kg: the smallest hull the limits are checked for (stress/run_95.py FZ rows)
WATER = {"fresh": 1000.0, "sea": 1025.0}


class DescribeError(ValueError):
    pass


def _inside(v: float, lo: float, hi: float) -> float:
    """A starting value kept off its limits: one outside them, or within 2% of the range from either end, starts 2%
    of the range inside (a fit started on a limit can't move off it in that direction's first step)."""
    margin = 0.02 * (hi - lo)
    return min(max(v, lo + margin), hi - margin)


def _round(v: float) -> float:
    """4 decimals, or 3 significant figures below 0.001 (a small hull's inertia would round to 0)."""
    return round(v, 4) if abs(v) >= 1e-3 or abs(v) < 1e-9 else float(f"{v:.3g}")


def _small_hull(b: dict) -> dict:
    """A tiny hull's damping limits: the fixed upper limits (-0.05, -0.2, -0.5) can reach past the size-scaled lower
    ones. A range out of order, or narrower than 2:1, gets an upper limit a tenth of its lower one; others are
    unchanged."""
    for k, (lo, hi, why) in b.items():
        if lo < 0 and hi < 0.5 * lo:
            b[k] = (lo, 0.1 * lo, f"{why}; upper limit a tenth of the lower for a small hull")
    return b


def torpedo_bounds(rho: float, L: float, D: float) -> dict:
    """Limits sized from a slender hull (#77): {name: (lo, hi, why)} for added mass, drag and rotational damping.
    describe-stonefish and describe-vehicle both use these."""
    frontal, side = math.pi * D ** 2 / 4, L * D
    strip = rho * frontal * L
    hull = f"a {L:.2f} m x {D:.2f} m hull"
    b = {"Yv_dot": (0.0, 2 * strip, f"0 to 2x strip theory rho*pi*r^2*L = {strip:.3g} kg for {hull}"),
         "Zw_dot": (0.0, 2 * strip, f"0 to 2x strip theory rho*pi*r^2*L = {strip:.3g} kg for {hull}")}
    for i, (lk, qk) in enumerate((("Xu", "Xuu"), ("Yv", "Yvv"), ("Zw", "Zww"))):
        area, (cmin, cmax) = (frontal, CD_SURGE) if i == 0 else (side, CD_BROADSIDE)
        qlo, qhi = -0.5 * rho * cmax * area, -0.5 * rho * cmin * area
        b[qk] = (qlo, qhi, f"drag coefficient {cmin:g} to {cmax:g} on the {'frontal' if i == 0 else 'side'} area of "
                           f"{hull}")
        b[lk] = (LINEAR_SPEED * qlo, -0.5, f"up to {LINEAR_SPEED:g} m/s x the quadratic limit")
    rot = 0.5 * rho * D * L ** 4 / 32
    for k in ("Mqq", "Nrr"):
        b[k] = (-CD_BROADSIDE[1] * 3 * rot, -max(0.005 * rot, 0.05), "drag coefficient up to 3x sway's, for fins")
    for k, qk in (("Mq", "Mqq"), ("Nr", "Nrr")):
        b[k] = (b[qk][0], -0.2, "up to 1 rad/s x the quadratic limit")
    return _small_hull(b)


def box_bounds(rho: float, L: float, W: float, H: float, m: float, inertia_hi=None) -> dict:
    """Limits sized from an open box frame: {name: (lo, hi, why)}. inertia_hi: the largest roll, pitch and yaw
    inertia allowed (kg m^2), which widens the turning damping limits (#95)."""
    faces = {"x": W * H, "y": L * H, "z": L * W}
    frame = f"a {L:.2f} x {W:.2f} x {H:.2f} m frame"
    b = {}
    for i, (lk, qk, ax) in enumerate((("Xu", "Xuu", "x"), ("Yv", "Yvv", "y"), ("Zw", "Zww", "z"))):
        qlo, qhi = -0.5 * rho * CD_BOX[1] * faces[ax], -0.5 * rho * CD_BOX[0] * faces[ax]
        b[qk] = (qlo, qhi, f"drag coefficient {CD_BOX[0]:g} to {CD_BOX[1]:g} on the {faces[ax]:.3g} m^2 face of {frame}")
        b[lk] = (LINEAR_SPEED * qlo, -0.5, f"up to {LINEAR_SPEED:g} m/s x the quadratic limit")
    # rotation: each face as plates turning about the axis, 0.5*rho*Cd*b*a^4/32 summed over the two faces it sweeps
    rot = {"Kpp": 0.5 * rho * L * (W ** 4 + H ** 4) / 32, "Mqq": 0.5 * rho * W * (L ** 4 + H ** 4) / 32,
           "Nrr": 0.5 * rho * H * (L ** 4 + W ** 4) / 32}
    for k, r in rot.items():
        b[k] = (-CD_BOX[1] * 3 * r, -max(0.005 * r, 0.05), f"plates turning about the axis, drag coefficient up to "
                                                            f"{3 * CD_BOX[1]:g} ({r:.3g} at 1)")
    # Thrusters (their thrust changes with the water flowing through them as the frame turns), a tether and the
    # payload damp turning far more than the frame's outline: DFKI's BlueROV2 needs 10, the plates allow 1.1. So the
    # limit also reaches the damping that stops the largest inertia allowed within STOP_TIME, about one log sample.
    for k, I in zip(("Kpp", "Mqq", "Nrr"), inertia_hi if inertia_hi is not None else ()):
        if I / STOP_TIME > -b[k][0]:
            b[k] = (-I / STOP_TIME, b[k][1], f"{b[k][2]}, or enough to stop the largest inertia allowed ({I:.3g}) "
                                             f"within {STOP_TIME:g} s: thrusters and a tether damp turning too")
    for k, qk in (("Kp", "Kpp"), ("Mq", "Mqq"), ("Nr", "Nrr")):
        b[k] = (b[qk][0], -0.05, "up to 1 rad/s x the quadratic limit")
    for k in ("Xu_dot", "Yv_dot", "Zw_dot"):
        b[k] = (0.0, 4 * m, f"0 to 4x the mass ({4 * m:.3g} kg): an open frame carries 0.5-1.5x its mass of water")
    return _small_hull(b)


def describe_hull(mass: float, length: float, *, width: Optional[float] = None, height: Optional[float] = None,
                  diameter: Optional[float] = None, shape: Optional[str] = None, rho: float = 1025.0,
                  inertia=None, cg_below_cb: Optional[float] = None, net_buoyancy: Optional[float] = None,
                  volume: Optional[float] = None, reference_point=(0.0, 0.0, 0.0),
                  torque_reference_offset=(0.0, 0.0, 0.0), max_force=None, hull_lift: bool = False,
                  name: str = "my_vehicle", source: str = "your measurements") -> tuple:
    """(vehicle description dict, notes). max_force: the thrusters' largest force per axis (N), for the notes."""
    m, L = float(mass), float(length)
    if not (m > 0 and L > 0):
        raise DescribeError("give the vehicle's mass (kg) and length (m), both above 0")
    for k, v in (("width", width), ("height", height), ("diameter", diameter)):
        if v is not None and not v > 0:
            raise DescribeError(f"{k} must be above 0 (m), got {v}")
    if diameter is None and (width is None or height is None):
        raise DescribeError("give the vehicle's size: --width and --height (a frame), or --diameter (a round hull)")
    if m < MIN_MASS or L < MIN_LENGTH:
        raise DescribeError(f"a vehicle under {MIN_LENGTH * 100:g} cm or {MIN_MASS * 1000:g} g is smaller than Strathe's "
                            f"limits are worked out for: check the units (kg and m), got {m:g} kg and {L:g} m")
    if diameter is not None:   # slender and round enough -> torpedo
        across, round_ = diameter, True
    else:
        across, round_ = max(width, height), abs(width - height) / max(width, height) < 0.3
    sized = "torpedo" if round_ and L / across > SLENDER else "box"
    shape = shape or sized
    if shape not in ("box", "torpedo"):
        raise DescribeError(f"--shape is box or torpedo, got {shape!r}")
    D = diameter if diameter is not None else (width + height) / 2
    W = width if width is not None else D
    H = height if height is not None else D
    notes = []
    if shape != sized:
        # The wrong shape's rules size the limits for another hull (#95): a BlueROV2 described as a torpedo lost 0.077
        # held-out skill against the box rules, with its pitch inertia and yaw damping cut off.
        notes.append(f"--shape {shape} goes against the size: {L:.3g} m long and {across:.3g} m across "
                     f"(length {L / across:.2g}x the {'diameter' if diameter is not None else 'width or height'}"
                     f"{'' if round_ else ', not round'}) reads as a {sized}. The {shape} rules size the limits for "
                     f"another hull: a BlueROV2 described as a torpedo lost 0.077 held-out skill and had its pitch "
                     f"inertia and yaw damping cut off. Leave --shape out unless the hull really is {shape}-shaped")
    spec = yaml.safe_load(open(TEMPLATE, encoding="utf-8"))
    T = spec["calibration_targets"]
    src = f"{source}"

    def put(group, key, guess, bounds, why):
        lo, hi = float(bounds[0]), float(bounds[1])
        T[group][key].update({"initial_guess": _round(_inside(float(guess), lo, hi)),
                              "bounds": [_round(lo), _round(hi)], "source": src, "bounds_rationale": why,
                              "confidence": "estimated"})
        T[group][key].pop("reference", None)

    # displaced volume: given, else from the trim, else neutrally buoyant
    if volume is None:
        volume = (m * G + (net_buoyancy or 0.0)) / (rho * G)
    displaced = rho * volume
    trim = rho * G * volume - m * G if net_buoyancy is None else float(net_buoyancy)

    # inertia: rigid (given, else a solid box or cylinder of the mass), moved to the reference point, plus added
    if shape == "torpedo":
        r = D / 2
        rigid = np.array([m * r ** 2 / 2, m * (3 * r ** 2 + L ** 2) / 12, m * (3 * r ** 2 + L ** 2) / 12])
        added = np.array([0.0, 1.0, 1.0]) * rho * math.pi * r ** 2 * L ** 3 / 12
    else:
        rigid = np.array([m * (W ** 2 + H ** 2) / 12, m * (L ** 2 + H ** 2) / 12, m * (L ** 2 + W ** 2) / 12])
        added = 0.75 * rigid   # an open frame carries water as it turns: 0.5-1x the rigid inertia
    if inertia is not None:
        rigid = np.asarray(inertia, dtype=float)
        if rigid.shape != (3,) or not np.all(rigid > 0):
            raise DescribeError("--inertia takes three values above 0: Ixx Iyy Izz (kg m^2, about the centre)")
    d = np.asarray(reference_point, dtype=float)
    shift = m * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    nominal = rigid + shift + added
    inertia_hi = []
    for i, (k, ax) in enumerate((("Ieff_roll", "roll"), ("Ieff_pitch", "pitch"), ("Ieff_yaw", "yaw"))):
        hi = (10 if (shape == "torpedo" and i == 0) else 5) * nominal[i]
        inertia_hi.append(hi)
        put("effective_inertia", k, nominal[i], [nominal[i] / 5, hi],
            f"{'your CAD' if inertia is not None else 'a solid ' + shape} {rigid[i] + shift[i]:.3g} + water "
            f"{added[i]:.3g}, about the reference point; 1/5 to {hi / nominal[i]:g}x")

    # drag and added mass
    if shape == "torpedo":
        b = torpedo_bounds(rho, L, D)
        strip = rho * math.pi * D ** 2 / 4 * L
        b["Xu_dot"] = (0.0, displaced, f"0 to the displaced mass {displaced:.3g} kg")
        guesses = {"Xu_dot": 0.1 * m, "Yv_dot": strip, "Zw_dot": strip,
                   "Xuu": -0.5 * rho * 0.3 * math.pi * D ** 2 / 4, "Yvv": -0.5 * rho * 1.0 * L * D,
                   "Zww": -0.5 * rho * 1.0 * L * D, "Mqq": -0.5 * rho * D * L ** 4 / 32,
                   "Nrr": -0.5 * rho * D * L ** 4 / 32}
    else:
        b = box_bounds(rho, L, W, H, m, inertia_hi)
        faces = np.array([W * H, L * H, L * W])
        ratio = np.clip(faces / faces.mean(), 0.5, 1.5)
        guesses = {"Xu_dot": ratio[0] * m, "Yv_dot": ratio[1] * m, "Zw_dot": ratio[2] * m,
                   **{q: -0.5 * rho * 1.0 * a for q, a in zip(("Xuu", "Yvv", "Zww"), faces)},
                   "Kpp": -0.5 * rho * L * (W ** 4 + H ** 4) / 32, "Mqq": -0.5 * rho * W * (L ** 4 + H ** 4) / 32,
                   "Nrr": -0.5 * rho * H * (L ** 4 + W ** 4) / 32}
    for q, l in (("Xuu", "Xu"), ("Yvv", "Yv"), ("Zww", "Zw"), ("Kpp", "Kp"), ("Mqq", "Mq"), ("Nrr", "Nr")):
        if q in guesses:
            guesses[l] = 0.3 * guesses[q]   # the quadratic value at 0.3 m/s (or rad/s)
    group = {"Xu_dot": "added_mass_diag", "Yv_dot": "added_mass_diag", "Zw_dot": "added_mass_diag",
             **{k: "quadratic_damping" for k in ("Xuu", "Yvv", "Zww", "Kpp", "Mqq", "Nrr")},
             **{k: "linear_damping" for k in ("Xu", "Yv", "Zw", "Kp", "Mq", "Nr")}}
    for k, (lo, hi, why) in b.items():
        put(group[k], k, guesses.get(k, (lo + hi) / 2), [lo, hi], why)

    # trim and the centres
    weight = m * G
    put("buoyancy", "net_buoyancy_z", trim, [trim - 0.12 * weight, trim + 0.12 * weight],
        f"+-12% of the {weight:.3g} N weight around {'your trim' if net_buoyancy is not None else 'neutral'}")
    half = (D if shape == "torpedo" else H) / 2
    put("buoyancy", "cg_cb_offset_z", cg_below_cb if cg_below_cb is not None else 0.1 * 2 * half, [-half, half],
        f"within half the {'diameter' if shape == 'torpedo' else 'height'} ({half:.3g} m) either way")

    if shape == "torpedo" and hull_lift:
        zuw = -0.5 * rho * D ** 2 * 1.2
        T["hull_lift"]["Zuw"].update({"initial_guess": round(zuw, 3), "bounds": [round(8 * zuw, 2), 0.0],
                                      "source": f"slender-body lift for a {D:.2f} m diameter", "confidence": "estimated"})
        T["hull_lift"]["Muw"].update({"bounds": [round(-abs(zuw) * L, 2), round(abs(zuw) * L, 2)]})
        notes.append(f"Slender hull ({L:.2f} m x {D:.2f} m): the hull-lift terms are included. Calibrate with and "
                     f"without them and keep them only if held-out skill improves (LIMITS.md).")
    else:
        T.pop("hull_lift", None)
        if shape != "torpedo":
            T["buoyancy"].pop("cg_cb_offset_x", None)
        elif not hull_lift:
            T["buoyancy"].pop("cg_cb_offset_x", None)
            notes.append(f"Torpedo-shaped ({L:.2f} m x {D:.2f} m): the hull-lift terms are left out. Add --hull-lift "
                         f"to include them; keep them only if held-out skill improves (LIMITS.md).")

    if max_force is not None:   # can the drag limits balance the thrusters at a sensible speed?
        for i, (q, ax) in enumerate((("Xuu", "surge"), ("Yvv", "sway"), ("Zww", "heave"))):
            lo, hi = b[q][0], b[q][1]
            if max_force[i] > 0:
                fastest = math.sqrt(max_force[i] / abs(hi))
                slowest = math.sqrt(max_force[i] / abs(lo))
                if slowest > 3.0:
                    notes.append(f"{ax}: your thrusters ({max_force[i]:.0f} N) would push it faster than "
                                 f"{slowest:.1f} m/s even at the highest drag allowed: check the size and the "
                                 f"thrusters block")
                elif fastest < 0.05:
                    notes.append(f"{ax}: even the lowest drag allowed would hold it under {fastest:.2f} m/s with "
                                 f"{max_force[i]:.0f} N: check the size and the thrusters block")

    spec["fixed_parameters"]["mass_kg"].update({"value": round(m, 4), "source": src, "confidence": "measured"})
    spec["vehicle"]["name"] = name
    spec["data_conventions"]["torque_reference_offset_m"] = [float(x) for x in torque_reference_offset]
    spec["data_conventions"]["torque_reference_rationale"] = (
        "torques taken about this point (describe-vehicle --torque-reference-offset); [0, 0, 0] when your thruster "
        "positions are measured from the point your drag acts through")
    size = f"{L:.3g} x {D:.3g} m" if shape == "torpedo" else f"{L:.3g} x {W:.3g} x {H:.3g} m"
    spec["metadata"]["description"] = (f"{name}: written by `strathe describe-vehicle` from {source} ({m:g} kg, "
                                       f"{shape}, {size}, water {rho:g} kg/m^3). Starting values and limits only: "
                                       f"the logs decide the calibrated values.")
    spec["thruster_model"]["note"] = "Layout and thrust curve are in the ingestion spec's thrusters block."
    return _plain(spec), notes


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# --- reading a model ---------------------------------------------------------------------------------------------------
def _xml(path):
    import xml.etree.ElementTree as ET
    try:
        return ET.parse(path).getroot()
    except Exception as e:
        raise DescribeError(f"{path}: can't read as XML ({type(e).__name__}: {e})")


def _extents(shapes) -> Optional[tuple]:
    """(L, W, H) spanned by [(kind, size, xyz, rpy)] collision shapes: a box's size, a cylinder's radius and length
    (along its own z, as SDF, URDF and MuJoCo have it), a sphere's radius; each turned by its roll, pitch, yaw."""
    from scipy.spatial.transform import Rotation
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for kind, size, xyz, rpy in shapes:
        half = (np.asarray(size, float) / 2 if kind == "box" else
                np.array([size[0], size[0], size[1] / 2]) if kind == "cylinder" else np.full(3, float(size[0])))
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
        pts = Rotation.from_euler("xyz", rpy).apply(corners) + np.asarray(xyz, float)
        lo, hi = np.minimum(lo, pts.min(0)), np.maximum(hi, pts.max(0))
    return None if not np.all(np.isfinite(lo)) else tuple((hi - lo).round(6).tolist())


def from_sdf_or_urdf(path: str, sim: str, model_name: Optional[str] = None) -> dict:
    """Mass, inertia and collision size of the vehicle's main link (the heaviest; Gazebo: the Hydrodynamics plugin's
    link when there is one)."""
    root = _xml(path)
    links = root.findall(".//link")
    if model_name:
        m = next((mm for mm in root.iter("model") if mm.get("name") == model_name), None)
        if m is None:
            raise DescribeError(f"{path}: no model named {model_name!r}")
        links = m.findall(".//link")
    hydro = root.find(".//plugin[@name='gz::sim::systems::Hydrodynamics']/link_name")
    if hydro is not None:
        links = [lk for lk in links if lk.get("name") == hydro.text.strip()] or links

    def mass_of(lk):
        mm = lk.find("inertial/mass")
        if mm is None:
            return 0.0
        return float(mm.get("value") if mm.get("value") is not None else mm.text)
    if not links or max(mass_of(lk) for lk in links) <= 0:
        raise DescribeError(f"{path}: no link with an <inertial><mass>: give --mass and the size instead")
    link = max(links, key=mass_of)
    total = sum(mass_of(lk) for lk in links)
    inert = link.find("inertial/inertia")
    I = None
    if inert is not None:
        get = (lambda k: float(inert.get(k))) if inert.get("ixx") is not None else \
            (lambda k: float(inert.find(k).text))
        I = [get("ixx"), get("iyy"), get("izz")]
    shapes = []
    for col in link.findall("collision"):
        pose = col.find("pose")
        org = col.find("origin")
        if pose is not None and pose.text:
            vals = [float(v) for v in pose.text.split()] + [0.0] * 6
            xyz, rpy = vals[:3], vals[3:6]
        elif org is not None:
            xyz = [float(v) for v in org.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in org.get("rpy", "0 0 0").split()]
        else:
            xyz, rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        g = col.find("geometry")
        if g is None:
            continue
        box, cyl, sph = g.find("box"), g.find("cylinder"), g.find("sphere")
        if box is not None:
            sz = box.get("size") or box.findtext("size")
            shapes.append(("box", [float(v) for v in sz.split()], xyz, rpy))
        elif cyl is not None:
            r = float(cyl.get("radius") or cyl.findtext("radius"))
            ln = float(cyl.get("length") or cyl.findtext("length"))
            shapes.append(("cylinder", [r, ln], xyz, rpy))
        elif sph is not None:
            shapes.append(("sphere", [float(sph.get("radius") or sph.findtext("radius"))], xyz, rpy))
    ext = _extents(shapes)
    return {"mass": total, "inertia": I, "extents": ext, "link": link.get("name"),
            "n_shapes": len(shapes), "frame": "flu"}


def from_mjcf(path: str, body: Optional[str] = None) -> dict:
    try:
        import mujoco
    except ImportError as e:
        raise DescribeError("reading a MuJoCo model needs mujoco (pip install -r requirements.txt)") from e
    try:
        mm = mujoco.MjModel.from_xml_path(path)
    except Exception as e:
        raise DescribeError(f"{path}: MuJoCo can't load it ({e})")
    ids = [i for i in range(1, mm.nbody) if mm.body_parentid[i] == 0]
    if body:
        ids = [mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_BODY, body)]
        if ids[0] < 0:
            raise DescribeError(f"{path}: no body named {body!r}")
    if not ids:
        raise DescribeError(f"{path}: no body under the world")
    b = max(ids, key=lambda i: mm.body_subtreemass[i])
    from scipy.spatial.transform import Rotation
    def under(i):   # the vehicle's body or one of its children
        while i > 0:
            if i == b:
                return True
            i = mm.body_parentid[i]
        return False
    G_ = {k: int(getattr(mujoco.mjtGeom, f"mjGEOM_{k}")) for k in ("BOX", "ELLIPSOID", "CYLINDER", "CAPSULE", "SPHERE")}
    shapes = []
    for g in range(mm.ngeom):
        if not under(int(mm.geom_bodyid[g])):
            continue
        s, xyz, t = mm.geom_size[g], mm.geom_pos[g].tolist(), int(mm.geom_type[g])
        w, x, y, z = mm.geom_quat[g]
        rpy = Rotation.from_quat([x, y, z, w]).as_euler("xyz").tolist()
        if t in (G_["BOX"], G_["ELLIPSOID"]):
            shapes.append(("box", (2 * s).tolist(), xyz, rpy))
        elif t in (G_["CYLINDER"], G_["CAPSULE"]):
            shapes.append(("cylinder", [s[0], 2 * s[1] + (2 * s[0] if t == G_["CAPSULE"] else 0)], xyz, rpy))
        elif t == G_["SPHERE"]:
            shapes.append(("sphere", [s[0]], xyz, rpy))
    return {"mass": float(mm.body_subtreemass[b]), "inertia": mm.body_inertia[b].tolist(), "extents": _extents(shapes),
            "link": mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_BODY, b), "n_shapes": len(shapes), "frame": "flu"}


def from_holoocean(path: str, agent: Optional[str] = None) -> dict:
    import json
    try:
        sc = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        raise DescribeError(f"{path}: can't read as a HoloOcean scenario JSON ({e})")
    agents = sc.get("agents") or []
    a = next((x for x in agents if x.get("agent_name") == (agent or sc.get("main_agent"))), agents[0] if agents else None)
    if a is None:
        raise DescribeError(f"{path}: no agents")
    dyn = a.get("dynamics") or {}
    out = {"mass": dyn.get("mass"), "inertia": None, "extents": None, "link": a.get("agent_name"), "frame": "frd",
           "rho": dyn.get("rho")}
    if dyn.get("length") and (dyn.get("diam") or dyn.get("diameter")):
        d = float(dyn.get("diam") or dyn.get("diameter"))
        out["extents"] = (float(dyn["length"]), d, d)
        out["diameter"] = d
    if dyn.get("r_bg"):
        out["cg_below_cb"] = float(dyn["r_bg"][2]) - float((dyn.get("r_bb") or [0, 0, 0])[2])
    return out


def from_model(path: str, sim: str, name: Optional[str] = None) -> dict:
    if not os.path.exists(path):
        raise DescribeError(f"model not found: {path}")
    if sim in ("gazebo", "urdf"):
        return from_sdf_or_urdf(path, sim, name)
    if sim == "mujoco":
        return from_mjcf(path, name)
    if sim == "holoocean":
        return from_holoocean(path, name)
    raise DescribeError(f"--sim is gazebo, urdf, mujoco or holoocean (Stonefish: python strathe.py "
                        f"describe-stonefish), got {sim!r}")


# --- the command -------------------------------------------------------------------------------------------------------
def max_force_of(thrusters: Optional[str]) -> Optional[list]:
    """The thrusters' largest force per axis (N): a preset name or an ingest spec with a thrusters block."""
    if not thrusters:
        return None
    import thruster_model as tm
    if thrusters == "bluerov2_heavy":
        cfg = tm.bluerov2_heavy()
    else:
        if not os.path.exists(thrusters):
            raise DescribeError(f"--thrusters: {thrusters} isn't a file or a preset (bluerov2_heavy)")
        spec = yaml.safe_load(open(thrusters, encoding="utf-8"))
        spec = spec if isinstance(spec, dict) else {}
        block = (spec.get("wrench") or {}).get("thrusters") or spec.get("thrusters") or spec
        block = {k: v for k, v in block.items() if k in tm.THRUSTER_BLOCK_KEYS}
        try:
            cfg = tm.thruster_block(block, os.path.dirname(os.path.abspath(thrusters)))
        except (ValueError, OSError, KeyError, TypeError) as e:
            raise DescribeError(f"--thrusters: {thrusters} has no usable thrusters block ({e}). Give an ingest spec "
                                f"with one (templates/ingest_csv_pwm_custom_thrusters.yaml) or bluerov2_heavy") from e
    return [float(x) for x in np.abs(cfg.max_wrench()[:3])]


def run(a) -> int:
    notes_pre = []
    model = None
    if a.from_model:
        model = from_model(os.path.expanduser(a.from_model), a.sim, a.model_name)
        ext = model.get("extents")
        gave_size = bool(a.length and (a.diameter or (a.width and a.height)))
        if a.mass is None and model.get("mass"):
            a.mass = float(model["mass"])
        if a.inertia is None and model.get("inertia"):
            a.inertia = model["inertia"]
        if a.cg_below_cb is None and model.get("cg_below_cb") is not None:
            a.cg_below_cb = model["cg_below_cb"]
        if ext:
            L, W, H = ext
            if a.diameter is None and model.get("diameter"):
                a.diameter = model["diameter"]
            a.length = a.length or L
            if a.diameter is None:
                a.width, a.height = a.width or W, a.height or H
            dims = sorted((L, W, H))
            thin = dims[0] < 0.25 * dims[1] and not model.get("diameter")   # a plate, not a hull
            if thin and not gave_size:
                notes_pre.append(f"The model's collision shapes span {L:.3g} x {W:.3g} x {H:.3g} m. One side is much "
                                 f"thinner than the others: if that's a buoyancy block or a plate rather than the hull, "
                                 f"give the hull's real size (--length, --width, --height)")
        elif not gave_size:
            raise DescribeError(
                f"{a.from_model}: the scenario's dynamics has no length and diam: give --length and --diameter"
                if a.sim == "holoocean" else
                f"{a.from_model}: the model has no collision boxes, cylinders or spheres to size the hull from "
                f"(meshes aren't measured): give --length and --width/--height or --diameter")
        if model.get("rho") and a.water is None:
            a.water = str(model["rho"])
    if a.mass is None or a.length is None:
        raise DescribeError("give --mass and --length (and --width/--height or --diameter), or --from-model")
    try:
        rho = WATER.get(a.water or "sea") or float(a.water)
    except ValueError:
        raise DescribeError(f"--water is fresh, sea or a density in kg/m^3; could not read {a.water!r}")
    if not 900 <= rho <= 1100:
        raise DescribeError(f"--water {rho:g} kg/m^3 isn't water (fresh 1000, sea about 1025)")
    ext = model and model.get("extents")
    if ext and not gave_size and model.get("n_shapes") == 1 and abs(ext[1] - ext[2]) < 0.05 * max(ext[1:]):
        # A converter sizes a stand-in cylinder to float the vehicle; a hull with flooded sections is larger outside.
        # URI's ALPHA RISE in Gazebo: 0.17 m read, 0.254 m real; its added-mass limits came out 57% too small.
        shown = rho * math.pi * ext[1] * ext[2] / 4 * ext[0]
        if abs(shown - a.mass) < 0.1 * a.mass:
            notes_pre.append(f"The model's hull ({ext[0]:.3g} m x {ext[1]:.3g} m) displaces "
                             f"{shown:.3g} kg, the vehicle's own mass: it may be sized to float the vehicle rather "
                             f"than to its outer hull. Added mass and drag limits scale with the outer size: if the "
                             f"hull is larger (flooded sections), give --length and --diameter")
    source =(f"{os.path.basename(a.from_model)} ({a.sim}; link {model.get('link')})" if model else
              "your measurements")
    spec, notes = describe_hull(a.mass, a.length, width=a.width, height=a.height, diameter=a.diameter, shape=a.shape,
                                rho=rho, inertia=a.inertia, cg_below_cb=a.cg_below_cb, net_buoyancy=a.net_buoyancy,
                                volume=a.volume, reference_point=tuple(a.reference_point),
                                torque_reference_offset=tuple(a.torque_reference_offset),
                                max_force=max_force_of(a.thrusters), hull_lift=a.hull_lift, name=a.name, source=source)
    out = os.path.abspath(os.path.expanduser(a.out))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    header = ("# Written by `strathe describe-vehicle` (calibration/DESCRIBE_VEHICLE.md). Starting values and limits\n"
              "# sized from the hull; calibration decides the values. Give it to every command with --spec.\n")
    with open(out, "w", encoding="utf-8") as f:
        f.write(header + yaml.safe_dump(spec, sort_keys=False, width=110))
    from optimizer import load_vehicle_spec
    load_vehicle_spec(out)   # the file loads as every command will load it
    print(f"Wrote {out}")
    print(f"  {spec['metadata']['description']}")
    for n in notes_pre + notes:
        print("  note: " + n)
    print(f"Next: python strathe.py calibrate --ingest-spec <your logs' spec> --spec {out} --name {a.name}")
    return 0


def build_parser(add_help: bool = True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 add_help=add_help)
    ap.add_argument("--mass", type=float, help="what the vehicle weighs, as logged (kg)")
    ap.add_argument("--length", type=float, help="overall length, nose to tail (m)")
    ap.add_argument("--width", type=float, help="overall width (m), for a frame")
    ap.add_argument("--height", type=float, help="overall height (m), for a frame")
    ap.add_argument("--diameter", type=float, help="hull diameter (m), for a round hull")
    ap.add_argument("--shape", choices=("box", "torpedo"), help="default: torpedo when length / diameter > 3")
    ap.add_argument("--inertia", type=float, nargs=3, metavar=("IXX", "IYY", "IZZ"),
                    help="rigid inertia about the centre from CAD (kg m^2): the starting value, with water added")
    ap.add_argument("--cg-below-cb", type=float, help="how far the centre of gravity sits below the centre of "
                                                      "buoyancy (m), if you know it")
    ap.add_argument("--net-buoyancy", type=float, help="buoyancy minus weight (N), + floats; default neutral")
    ap.add_argument("--volume", type=float, help="displaced volume (m^3); default from the mass and trim")
    ap.add_argument("--water", help="fresh, sea (default), or a density in kg/m^3")
    ap.add_argument("--reference-point", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                    help="where your logs' velocity is measured, from the vehicle's centre (m, forward-right-down)")
    ap.add_argument("--torque-reference-offset", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                    help="the point your logs' torques are taken about, if not the thruster positions' origin "
                         "(vehicle_spec.yaml data_conventions)")
    ap.add_argument("--thrusters", help="your ingest spec (its thrusters block) or a preset (bluerov2_heavy): "
                                        "checks the drag limits against what they can push")
    ap.add_argument("--hull-lift", action="store_true", help="torpedo: include the hull-lift terms")
    ap.add_argument("--from-model", help="your vehicle's model: Gazebo .sdf/.world, URDF, MuJoCo .xml, HoloOcean "
                                         "scenario .json. Mass, inertia and size come from it; any size you give wins")
    ap.add_argument("--sim", choices=("gazebo", "urdf", "mujoco", "holoocean"), help="what --from-model is")
    ap.add_argument("--model-name", help="the vehicle's model / body / agent in the file (default: the heaviest)")
    ap.add_argument("--name", default="my_vehicle")
    ap.add_argument("--out", default="my_vehicle_spec.yaml")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.from_model and not a.sim:
        ext = os.path.splitext(a.from_model)[1].lower()
        a.sim = {".urdf": "urdf", ".sdf": "gazebo", ".world": "gazebo", ".json": "holoocean"}.get(ext)
        if a.sim is None:
            print("❌ give --sim gazebo | urdf | mujoco | holoocean for that file")
            return 2
    try:
        return run(a)
    except DescribeError as e:
        print(f"❌ {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
