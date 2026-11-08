"""
stonefish_describe.py -- write a vehicle description (vehicle_spec.yaml) from a lab's own Stonefish model
(`strathe describe-stonefish`, PROJECT_STATUS.md #77).

A lab with a Stonefish model of its vehicle has already described it: mass, mass distribution, hull geometry, drag
coefficients and where the centres of gravity and buoyancy sit. This reads them through the Stonefish replay tool
(the same one `fit-stonefish` uses) and turns them into Strathe's starting values and limits, so the lab doesn't have
to work them out by hand (LAB_GUIDE.md step 4) or send CAD:
  - mass: the vehicle's, or `--mass` (what it weighs; the model's can differ);
  - effective inertia (rigid + added) about the logs' reference point, the rigid part scaled to `--mass`;
  - added mass per axis (Stonefish's geometric estimate for each external part);
  - linear and quadratic drag per axis from the parts' drag coefficients and mesh areas (Stonefish's quadratic drag
    grows with speed cubed, so it's matched at `--cruise-speed`);
  - pitch and yaw damping from the hull's length and diameter;
  - net buoyancy (the model's trim), and the centre of gravity relative to the centre of buoyancy;
  - for a slender hull, the hull-lift terms (#71).
The model's values are the starting points. The limits come from the hull's size (its length and diameter from the mesh,
its displaced volume, its mass distribution), not from the model's values: Stonefish's geometric estimates can be
several times the physical ones (ALPHA RISE's sway added mass is 3x strip theory), and limits sized from them let the
fit wander far enough to fit the training logs and fail on unseen ones. The logs decide the calibrated values.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile

import numpy as np
import yaml

from stonefish_fit import Replay, Scenario, StonefishFitError, find_tool

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "templates", "vehicle_spec_template.yaml")
# the hull-size rules are shared with describe-vehicle (vehicle_describe.py, #95)
from vehicle_describe import G, SLENDER, _inside, torpedo_bounds  # noqa: E402
# Damping limits are generous on purpose: in an open-loop replay damping also absorbs what the model leaves out, and
# ALPHA RISE's calibration wants more of it than the hull alone gives (#77: 0.3 m/s here cost 0.6 of mean skill).
# Inertia and added mass keep limits from the hull's size: those are where an off model value did the damage.


def describe(info: dict, *, mass: float | None = None, water_density: float | None = None, cruise_speed: float = 0.5,
             reference_point=(0.0, 0.0, 0.0), name: str = "my_vehicle", model_name: str = "your Stonefish model",
             hull_lift: bool = False) -> tuple:
    """(vehicle description dict, notes) from the replay tool's --info."""
    base = info["base"]
    parts = info["parts"]
    if not parts:
        raise StonefishFitError("Stonefish reports no parts for this robot: rebuild the replay tool (calibration/"
                                "stonefish/README.md) and check --robot")
    for k in ("cg", "cb", "inertia_body_axes"):
        if k not in base:
            raise StonefishFitError("the replay tool is older than this command: rebuild it "
                                    "(cmake --build calibration/stonefish/build)")
    rho = float(water_density or info.get("water_density") or 1025.0)
    model_mass = float(base["mass"])
    m = float(mass or model_mass)
    ext = [p for p in parts if p.get("external")]
    notes = []
    src = lambda what: f"{model_name}: {what}"  # noqa: E731
    spec = yaml.safe_load(open(TEMPLATE, encoding="utf-8"))
    T = spec["calibration_targets"]

    def put(group, key, guess, bounds, source, rationale):
        """The model's value as the starting point, inside limits sized from the hull (not from the model's value: a
        model value that's off, e.g. a geometric added-mass estimate 3x strip theory, mustn't widen them)."""
        lo, hi = float(bounds[0]), float(bounds[1])
        start = _inside(float(guess), lo, hi)
        if not lo <= float(guess) <= hi:   # (one just inside is moved off the limit too, but isn't outside)
            outside.append(f"{key} {float(guess):.3g}")
        T[group][key].update({"initial_guess": round(start, 4), "bounds": [round(lo, 4), round(hi, 4)],
                              "source": source, "bounds_rationale": rationale, "confidence": "estimated"})
        T[group][key].pop("reference", None)

    outside = []
    # Geometry from the parts' projected areas: frontal area -> equivalent diameter, side area -> length.
    S = np.sum([p["projected_area"] for p in ext], axis=0)
    D = math.sqrt(4 * max(S[0], 1e-6) / math.pi)
    L = max(S[1], S[2]) / max(D, 1e-6)
    slender = L / max(D, 1e-6) > SLENDER
    hull = f"a {L:.2f} m x {D:.2f} m hull"
    frontal = math.pi * D ** 2 / 4
    displaced = rho * float(base["volume"])

    # Mass and its distribution.
    spec["vehicle"]["name"] = name
    spec["fixed_parameters"]["mass_kg"].update({"value": round(m, 3), "source": "weighed (--mass)" if mass else
                                                src("total mass"), "confidence": "measured" if mass else "estimated"})
    if mass and abs(mass - model_mass) > 0.05 * model_mass:
        notes.append(f"The Stonefish model weighs {model_mass:.1f} kg against the {mass:g} kg you gave: its rigid "
                     f"inertia is scaled to {mass:g} kg, and its trim (net buoyancy) kept as the model has it.")
    added_I = np.sum([p["added_inertia"] for p in parts], axis=0)
    rigid = np.maximum(np.array(base["inertia_body_axes"]) - added_I, 0.0) * (m / model_mass)
    d = np.array(reference_point, dtype=float) - np.array(base["cg"], dtype=float)
    shift = m * np.array([d[1] ** 2 + d[2] ** 2, d[0] ** 2 + d[2] ** 2, d[0] ** 2 + d[1] ** 2])
    Ieff = rigid + added_I + shift
    # Limits: the model's rigid inertia (its mass distribution) plus slender-body added inertia rho*pi*r^2*L^3/12.
    phys_added = rho * frontal * L ** 3 / 12
    for i, (k, ax) in enumerate((("Ieff_roll", "roll"), ("Ieff_pitch", "pitch"), ("Ieff_yaw", "yaw"))):
        nominal = rigid[i] + shift[i] + (0.0 if i == 0 else phys_added)
        nominal = max(nominal, 0.05)
        put("effective_inertia", k, Ieff[i], [nominal / 5, (10 if i == 0 else 3) * nominal],
            src(f"{ax} inertia, rigid + added, about the logs' reference point"),
            f"rigid {rigid[i] + shift[i]:.3g}" + ("" if i == 0 else f" + slender-body added {phys_added:.3g}")
            + f" for {hull}; {'1/5 to 10x' if i == 0 else '1/5 to 3x'}")
    A = np.sum([p["added_mass"] for p in ext], axis=0)
    hb = torpedo_bounds(rho, L, D)   # limits from the hull's size, shared with describe-vehicle (#95)
    put("added_mass_diag", "Xu_dot", A[0], [0.0, displaced], src("surge added mass (geometric estimate)"),
        f"0 to the displaced mass {displaced:.3g} kg for {hull}")
    for k, ax, v in (("Yv_dot", "sway", A[1]), ("Zw_dot", "heave", A[2])):
        put("added_mass_diag", k, v, hb[k][:2], src(f"{ax} added mass (geometric estimate)"), hb[k][2])

    # Drag. Stonefish: quadratic 0.5 rho Cd S |v|^2 |v| (cubic, matched at the cruise speed), viscous rho Cf T v.
    for i, (lk, qk, ax) in enumerate((("Xu", "Xuu", "surge"), ("Yv", "Yvv", "sway"), ("Zw", "Zww", "heave"))):
        quad = -sum(0.5 * rho * p["quadratic_drag"][i] * p["projected_area"][i] for p in ext) * cruise_speed
        lin = -sum(rho * p["viscous_drag"][i] * p["tangential_area"][i] for p in ext)
        put("quadratic_damping", qk, quad, hb[qk][:2],
            src(f"{ax} drag from its coefficients and mesh area, matched at {cruise_speed:g} m/s"), hb[qk][2])
        put("linear_damping", lk, lin, hb[lk][:2], src(f"{ax} skin friction from its coefficients and mesh area"),
            hb[lk][2])
    rot = 0.5 * rho * D * L ** 4 / 32  # strip theory, drag coefficient 1
    for k in ("Mqq", "Nrr"):
        put("quadratic_damping", k, -rot, hb[k][:2], src(f"strip theory 0.5*rho*D*L^4/32 for {hull}"), hb[k][2])
    for k, qk in (("Mq", "Mqq"), ("Nr", "Nrr")):
        qb = T["quadratic_damping"][qk]
        put("linear_damping", k, 0.3 * qb["initial_guess"], [qb["bounds"][0], -0.2],
            "0.3 rad/s x the quadratic value", "up to 1 rad/s x the quadratic limit")

    # Trim and the centres.
    net = rho * G * float(base["volume"]) - model_mass * G
    weight = m * G
    put("buoyancy", "net_buoyancy_z", net, [min(-0.12 * weight, net - 0.12 * weight), max(0.12 * weight, net + 0.12 * weight)],
        src(f"its trim (buoyant volume against its mass, water {rho:g})"), "+-12% of the weight around the model's trim")
    cgcb = np.array(base["cg"]) - np.array(base["cb"])  # body frame forward-right-down: z > 0 = CG below CB
    put("buoyancy", "cg_cb_offset_z", cgcb[2], [-D / 2, D / 2], src("centre of gravity below the centre of buoyancy"),
        f"within the hull's radius ({D / 2:.3g} m) either way")
    if outside:
        notes.append("Outside the limits for the hull's size, so the fit starts just inside them: "
                     + ", ".join(outside) + ". Stonefish's geometric estimates can be several times the physical ones.")
    if slender and hull_lift:
        put("buoyancy", "cg_cb_offset_x", cgcb[0], [cgcb[0] - 0.05, cgcb[0] + 0.05],
            src("centre of gravity ahead of the centre of buoyancy"), "the model's offset +-5 cm")
        zuw = -0.5 * rho * D ** 2 * 1.2
        T["hull_lift"]["Zuw"].update({"initial_guess": round(zuw, 3), "bounds": [round(8 * zuw, 2), 0.0],
                                      "source": src(f"slender-body lift for a {D:.2f} m diameter"), "confidence": "estimated"})
        # the lift's moment: Zuw anywhere in its limits, acting anywhere along the hull (not just its guess: that held the OOI REMUS 600s'
        # pitch about 15% short of what their logs need, #97)
        T["hull_lift"]["Muw"].update({"bounds": [round(-8 * abs(zuw) * L, 2), round(8 * abs(zuw) * L, 2)],
                                      "bounds_rationale": "+-|Zuw| x length, over Zuw's limits"})
        notes.append(f"Slender hull ({L:.2f} m x {D:.2f} m): the hull-lift terms are included. Calibrate with and "
                     f"without them and keep them only if held-out skill improves (LIMITS.md).")
    else:
        T["buoyancy"].pop("cg_cb_offset_x", None)
        T.pop("hull_lift", None)
        if slender:
            notes.append(f"Slender hull ({L:.2f} m x {D:.2f} m): the hull-lift terms are left out. Add --hull-lift to "
                         f"include them; on ordinary missions they often fit the training logs better and unseen ones "
                         f"worse, so keep them only if held-out skill improves (LIMITS.md).")
    spec["metadata"]["description"] = (f"{name}: written by `strathe describe-stonefish` from {model_name} "
                                       f"({m:g} kg, hull about {L:.2f} m x {D:.2f} m). Starting values and limits only: "
                                       f"the logs decide the calibrated values.")
    return _plain(spec), notes


def _plain(x):
    """numpy numbers -> plain floats, so the description writes as ordinary YAML."""
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, np.generic):
        return x.item()
    return x


HEADER = """# Vehicle description written by `strathe describe-stonefish` from {model}.
# Pass it to every command: python strathe.py calibrate --ingest-spec my_ingest.yaml --spec {out} --name my_vehicle \
#     --restarts 4
# Calibrate it from 4 starting points (--restarts 4): its limits are wide, and one start can land badly (on URI's
# ALPHA RISE, one start blew up on the unseen mission; four matched the hand-written description, PROJECT_STATUS #79).
# Also calibrate `strathe describe-vehicle`'s description from the hull's size, and keep the better (#95).
# Each value's `source` says where it came from. They are starting values, with limits sized from the hull; the logs
# decide the calibrated values. Check the mass (weigh the vehicle) and the thruster count; edit anything you know
# better (calibration/templates/vehicle_spec_template.yaml explains every field).
"""


def run(a) -> int:
    scenario = Scenario(a.scenario, dict(kv.split("=", 1) for kv in (a.find or [])), a.part or "")
    replay = Replay(a.replay_tool_argv, a.stonefish_data or os.path.dirname(os.path.abspath(a.scenario)), a.robot,
                    tuple(a.reference_point))
    work = tempfile.mkdtemp(prefix="strathe_describe_")
    info = replay.info(scenario.write_runnable(work))
    model = f"the Stonefish model of '{a.robot}' in {os.path.basename(a.scenario)}"
    spec, notes = describe(info, mass=a.mass, water_density=a.water_density, cruise_speed=a.cruise_speed,
                           reference_point=a.reference_point, name=a.name, model_name=model, hull_lift=a.hull_lift)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(HEADER.format(model=model, out=os.path.basename(a.out)))
        yaml.safe_dump(spec, f, sort_keys=False, width=110)
    from optimizer import initial_parameters, load_vehicle_spec
    initial_parameters(load_vehicle_spec(a.out))  # the description must load as Strathe reads it
    T = spec["calibration_targets"]
    print(f"Wrote {a.out} from {model}.")
    print(f"  mass {spec['fixed_parameters']['mass_kg']['value']:g} kg; inertia roll/pitch/yaw "
          f"{T['effective_inertia']['Ieff_roll']['initial_guess']:.3g} / {T['effective_inertia']['Ieff_pitch']['initial_guess']:.3g} / "
          f"{T['effective_inertia']['Ieff_yaw']['initial_guess']:.3g} kg*m^2; added mass "
          f"{T['added_mass_diag']['Xu_dot']['initial_guess']:.3g} / {T['added_mass_diag']['Yv_dot']['initial_guess']:.3g} / "
          f"{T['added_mass_diag']['Zw_dot']['initial_guess']:.3g} kg; net buoyancy "
          f"{T['buoyancy']['net_buoyancy_z']['initial_guess']:+.1f} N; CG {100 * T['buoyancy']['cg_cb_offset_z']['initial_guess']:.1f} cm "
          f"below CB")
    for n in notes:
        print("  note: " + n)
    print(f"Next: python strathe.py calibrate --spec {a.out} --restarts 4 ... (4 starting points: with limits this "
          f"wide, one start isn't reliable)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, help="your Stonefish scenario (the world file with the robot in it)")
    ap.add_argument("--robot", required=True, help="the robot's name in the scenario")
    ap.add_argument("--reference-point", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X", "Y", "Z"),
                    help="where your logs' velocity and forces are measured, in the robot's frame (forward-right-down, "
                         "m from its origin), e.g. your URDF's base_link (default: the robot's origin)")
    ap.add_argument("--mass", type=float, help="the vehicle's weighed mass, kg (default: the Stonefish model's)")
    ap.add_argument("--water-density", type=float, help="kg/m^3 (default: the scenario's)")
    ap.add_argument("--cruise-speed", type=float, default=0.5, help="typical forward speed, m/s (default 0.5)")
    ap.add_argument("--name", default="my_vehicle", help="the vehicle's name in the description")
    ap.add_argument("--hull-lift", action="store_true", help="slender hulls: include the hull-lift terms and the "
                                                             "fore-aft CG offset (#71); off by default")
    ap.add_argument("--part", help=argparse.SUPPRESS)
    ap.add_argument("--find", nargs="+", metavar="PKG=PATH", help="folders for $(find PKG) in ROS-style scenarios")
    ap.add_argument("--stonefish-data", help="Stonefish's data folder for meshes (default: the scenario's folder)")
    ap.add_argument("--replay-tool", help="the built strathe_stonefish_replay; or docker: Strathe's Stonefish image (setup-sim "
                    "stonefish --docker, once); or docker:IMAGE (default: calibration/stonefish/build/, else the "
                    "Docker image if it's built)")
    ap.add_argument("--out", default="my_vehicle_spec.yaml", help="the vehicle description to write")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        a.replay_tool_argv = find_tool(a.replay_tool)
        return run(a)
    except StonefishFitError as e:
        print(f"❌ {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
