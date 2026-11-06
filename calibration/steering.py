"""
steering.py -- how a vehicle steers, said once: the words labs see, the actuators a description defines, and every refusal
(PROJECT_STATUS.md #97, calibration/FINS.md).

    hover     thrusters steer it; what's logged as force is everything that drives it (the default)
    fin       fins and a propeller (REMUS, Iver, LAUV, Hugin, Autosub, Tethys)
    vectored  a gimballed tail or thruster (Bluefin, Dorado)
    hybrid    fins and steering thrusters
    glider    refused: `strathe fin-check` reads its rudder, calibration needs a propeller's thrust

Shape is a separate thing: ALPHA RISE is a torpedo, and hover.

A description's fins are a preset or a list; logs carry rudder, elevator and aileron, or one channel per fin. Prepared
recordings keep the logged channels as columns (`fin_<channel>` measured, `fincmd_<channel>` commanded, in radians, signed
so positive turns to starboard, pitches the nose up and rolls to starboard), and a run becomes per-fin angles when it loads.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from fin_force import Actuators, Fin, MovingMass, Propeller

STEERING = ("hover", "fin", "vectored", "hybrid", "glider")
LAYOUTS = ("cross", "x", "y3", "rudder_only", "custom")
GANGED_CHANNELS = ("rudder", "elevator", "aileron")
EXTRA_COLUMNS = ("rpm", "gimbal_y", "gimbal_z", "moving_mass", "buoyancy_change")
WORDS = {"hover": "HOVER (thrusters steer it)", "fin": "FIN-STEERED (fins and a propeller)",
         "vectored": "VECTORED THRUST (a gimballed tail)", "hybrid": "HYBRID (fins and steering thrusters)",
         "glider": "GLIDER (rudder check only)"}


class SteeringError(ValueError):
    """A mismatch between the vehicle and what's asked of it. Its message says what to do."""


def steering_of(raw: dict) -> str:
    s = (raw.get("vehicle") or {}).get("steering", "hover")
    if s not in STEERING:
        raise SteeringError(f"vehicle.steering: {s!r} isn't one of {', '.join(STEERING)}. hover: thrusters steer it; fin: fins "
                            f"and a propeller; vectored: a gimballed tail; hybrid: both")
    return s


def banner(raw: dict) -> str:
    """The line every command and report opens with."""
    said = "steering" in (raw.get("vehicle") or {})
    s = steering_of(raw)
    return f"Vehicle type: {WORDS[s]}" + ("" if said else ". Not stated in the vehicle description, so taken as hover")


def _unit(v) -> tuple:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        raise SteeringError(f"fins: span_axis {list(v)} has no length")
    return tuple(float(x) for x in a / n)


def _preset(block: dict) -> list:
    """A layout's fins, around the hull at x_m. angles: where each fin stands, degrees around the hull from starboard, toward
    the bottom (90: down)."""
    layout = block["layout"]
    x, r = float(block["x_m"]), float(block["radius_m"])
    sets = {"cross": (("starboard", 0), ("bottom", 90), ("port", 180), ("top", 270)),
            "x": (("lower_starboard", 45), ("lower_port", 135), ("upper_port", 225), ("upper_starboard", 315)),
            "y3": (("top", 270), ("port", 150), ("starboard", 30)),
            "rudder_only": (("bottom", 90), ("top", 270))}[layout]
    out = []
    for name, a in sets:
        c, s = math.cos(math.radians(a)), math.sin(math.radians(a))
        out.append({"name": name, "position_m": [x, r * c, r * s], "span_axis": [0.0, c, s]})
    return out


def _mixing(fins: list) -> list:
    """How much of each ganged channel each fin takes: its moment about each axis for a positive angle, each axis scaled so the
    strongest fin takes 1. Positive rudder then turns to starboard, positive elevator lifts the nose, positive aileron rolls
    to starboard (each fin helping as much as its place allows)."""
    m = []
    for f in fins:
        s, c = np.array(f["span_axis"]), np.array(f.get("chord_axis", (1.0, 0.0, 0.0)))
        m.append(np.cross(np.array(f["position_m"], dtype=float), np.cross(s, c)))
    m = np.array(m)
    big = np.abs(m).max(axis=0)
    big[big < 1e-9] = 1.0
    return [tuple(float(v) for v in row) for row in m / big]


def actuators_from(raw: dict) -> Optional[Actuators]:
    """A vehicle description's actuators, or None for a vehicle with none. Refuses what can't be right."""
    s = steering_of(raw)
    blocks = {k: raw.get(k) for k in ("fins", "propeller", "moving_mass", "buoyancy_engine") if raw.get(k) is not None}
    if s == "hover":
        if blocks:
            raise SteeringError(f"this description has {', '.join(blocks)} but vehicle.steering is hover. Say "
                                f"`steering: fin` (or vectored, hybrid) under vehicle:, or remove them")
        return None
    if s == "glider":
        return None
    if s in ("fin", "hybrid") and "fins" not in blocks:
        raise SteeringError(f"vehicle.steering is {s} but there's no fins: section. `strathe describe-vehicle --steering fin` "
                            f"writes one; templates/vehicle_spec_fin_template.yaml shows it")
    if s == "vectored" and "propeller" not in blocks:
        raise SteeringError("vehicle.steering is vectored but there's no propeller: section with gimbal: true")
    fins = ()
    fb = blocks.get("fins")
    if fb:
        layout = fb.get("layout", "custom")
        if layout not in LAYOUTS:
            raise SteeringError(f"fins.layout: {layout!r} isn't one of {', '.join(LAYOUTS)}")
        raw_fins = [dict(f) for f in fb["fins"]] if layout == "custom" else _preset(fb)
        if not raw_fins:
            raise SteeringError("fins: a custom layout needs a fins: list")
        names = [f["name"] for f in raw_fins]
        if len(set(names)) != len(names):
            raise SteeringError(f"fins: two fins share a name in {names}")
        mix = _mixing(raw_fins)
        built = []
        for f, m in zip(raw_fins, mix):
            span = _unit(f["span_axis"])
            area = float(f.get("area_m2", fb.get("area_m2", 0.0)))
            length = float(f.get("span_m", fb.get("span_m", 0.0)))
            if area <= 0 or length <= 0:
                raise SteeringError(f"fin {f['name']}: needs area_m2 and span_m, both above 0: its limits and drag come from them")
            group = int(f.get("group", 1 if abs(span[1]) > abs(span[2]) + 1e-9 else 0))  # 0 steers, 1 pitches
            built.append(Fin(f["name"], tuple(float(x) for x in f["position_m"]), span, area, length, group=group, mix=m,
                             stall_rad=math.radians(float(f.get("stall_deg", fb.get("stall_deg", 20.0)))),
                             oswald=float(f.get("oswald", fb.get("oswald", .7))),
                             limit_rad=math.radians(float(f.get("limit_deg", fb.get("limit_deg", 30.0)))),
                             sign=float(f.get("sign", 1.0))))
        fins = tuple(built)
    prop = None
    pb = blocks.get("propeller")
    if pb:
        prop = Propeller(float(pb["diameter_m"]), float(pb["kt0"]), float(pb.get("kt1", 0.0)), float(pb.get("kq0", 0.0)),
                         float(pb.get("kq1", 0.0)), float(pb.get("wake", 1.0)), float(pb.get("thrust_deduction", 0.0)),
                         tuple(float(x) for x in pb.get("position_m", (0.0, 0.0, 0.0))), float(pb.get("spin", 1.0)),
                         bool(pb.get("gimbal", False)))
    mm = blocks.get("moving_mass")
    servo = (fb or {}).get("servo") or {}
    rho = float((raw.get("environment") or {}).get("water_density_kg_m3", 1025.0))
    return Actuators(fins=fins, propeller=prop, moving_mass=MovingMass(float(mm["mass_kg"]), tuple(mm.get("axis", (1.0, 0.0, 0.0))))
                     if mm else None, buoyancy_engine=bool(blocks.get("buoyancy_engine")), rho=rho,
                     servo_rate_rad_s=math.radians(float(servo.get("rate_limit_deg_s", 0.0))))


def logged_channels(columns: Sequence[str]) -> dict:
    """{channel: column} of a prepared recording's fin channels, and whether they're commanded."""
    measured = {c[4:]: c for c in columns if c.startswith("fin_")}
    commanded = {c[7:]: c for c in columns if c.startswith("fincmd_")}
    return {"measured": measured, "commanded": commanded}


def has_actuator_columns(columns: Sequence[str]) -> bool:
    return any(c.startswith(("fin_", "fincmd_")) or c in EXTRA_COLUMNS for c in columns)


def controls_from_frame(df, act: Actuators, where: str = "this recording") -> tuple:
    """(N, Actuators.width) controls from a prepared recording, and whether the fin angles are commanded. Ganged channels are
    mixed to each fin; a recording may instead carry a channel per fin."""
    seen = logged_channels(df.columns)
    if seen["measured"] and seen["commanded"]:
        raise SteeringError(f"{where} has both measured and commanded fin angles: prepare it with one (fins.source)")
    commanded = bool(seen["commanded"])
    chans = seen["commanded"] or seen["measured"]
    n = len(df)
    cols = []
    if act.fins:
        if not chans:
            raise SteeringError(f"{where} has no fin angles, but the vehicle description has fins. Add a fins: block to the "
                                f"ingest spec (FINS.md)")
        names = {f.name for f in act.fins}
        per_fin = set(chans) & names
        if per_fin:
            missing = names - set(chans)
            if missing:
                raise SteeringError(f"{where} has angles for {sorted(per_fin)} but not {sorted(missing)}: log every fin, or log "
                                    f"rudder, elevator and aileron")
            cols += [f.sign * df[chans[f.name]].to_numpy(dtype=float) for f in act.fins]
        else:
            odd = set(chans) - set(GANGED_CHANNELS)
            if odd:
                raise SteeringError(f"{where}: fin channels {sorted(odd)} aren't rudder, elevator, aileron or a fin's name "
                                    f"({sorted(names)})")
            get = {k: df[chans[k]].to_numpy(dtype=float) if k in chans else np.zeros(n) for k in GANGED_CHANNELS}
            cols += [f.mix[0] * get["aileron"] + f.mix[1] * get["elevator"] + f.mix[2] * get["rudder"] for f in act.fins]
    for name, need in (("rpm", act.propeller is not None), ("gimbal_y", bool(act.propeller and act.propeller.gimbal)),
                       ("gimbal_z", bool(act.propeller and act.propeller.gimbal)), ("moving_mass", act.moving_mass is not None),
                       ("buoyancy_change", act.buoyancy_engine)):
        if need:
            if name not in df.columns:
                raise SteeringError(f"{where} has no {name} column, but the vehicle description needs it. Add it to the ingest spec")
            cols.append(df[name].to_numpy(dtype=float))
    return np.stack(cols, 1) if cols else np.zeros((n, 0)), commanded


def check_data(raw: dict, columns: Sequence[str]) -> None:
    """The vehicle description against a prepared recording's columns, before any time is spent."""
    s = steering_of(raw)
    has = has_actuator_columns(columns)
    if s == "glider":
        raise SteeringError("calibration needs thrust a glider doesn't have: it moves by changing its buoyancy and shifting its "
                            "battery. `strathe fin-check` checks its rudder against its logged turning")
    if s == "hover" and has:
        raise SteeringError("these logs have fin or propeller channels but the vehicle description is hover. Say `steering: fin` "
                            "under vehicle: and describe the fins (`strathe describe-vehicle --steering fin`), or leave those "
                            "channels out of the ingest spec")
    if s != "hover" and not has:
        raise SteeringError(f"the vehicle description is {s} but these logs have no fin or propeller channels. Add a fins: block "
                            f"to the ingest spec (FINS.md)")


def check_speed_source(raw: dict, sensors_seen: dict) -> None:
    """A fin's force goes as speed squared, so a fit without a speed sensor is circular."""
    if steering_of(raw) in ("fin", "vectored", "hybrid") and not sensors_seen.get("u"):
        raise SteeringError("a fin-steered vehicle needs a speed source: a fin's force goes as speed squared, so speed can't be "
                            "inferred from the turning it causes. Log a DVL or position fixes")


def refuse_experiment(raw: dict, command: str) -> None:
    if steering_of(raw) == "glider":
        raise SteeringError(f"{command} can't run a glider: there's no thrust model for one")
