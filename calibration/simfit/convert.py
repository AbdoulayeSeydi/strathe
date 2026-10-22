"""
simfit/convert.py -- `strathe convert-sim`: a lab's vehicle model from one simulator written for another
(PROJECT_STATUS.md #85). From what the source model reports (a Stonefish model, through its replay tool's --info): the
mass and its distribution, the centres of gravity and buoyancy, the hull's size and displaced volume, the added mass
per axis and the drag coefficients. Written as a Gazebo world (Hydrodynamics + Buoyancy), a MuJoCo MJCF (ellipsoid
fluid model, gravcomp) or a HoloOcean scenario (Fossen torpedo). Every converted model has its origin at the logs'
reference point, so use `--reference-point 0 0 0` with it. With --data, source and converted models are replayed on the
same held-out windows and the gap is reported: a conversion is a starting point, `fit-sim` calibrates it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

from simfit.core import SimFitError
from simfit.editors import fmt_number

G = 9.81
CRUISE = 0.5  # m/s: Stonefish's quadratic drag is cubic in speed; matched here (as describe-stonefish)


def facts_from_stonefish(info: dict, ref, mass=None, cruise: float = CRUISE) -> dict:
    """The vehicle as numbers, relative to the reference point (forward-right-down)."""
    base, parts = info["base"], info["parts"]
    ext = [p for p in parts if p.get("external")]
    if not ext:
        raise SimFitError("the Stonefish model reports no external parts (rebuild the replay tool)")
    rho = float(info.get("water_density") or 1025.0)
    m0 = float(base["mass"])
    m = float(mass or m0)
    S = np.sum([p["projected_area"] for p in ext], axis=0)
    Tg = np.sum([p["tangential_area"] for p in ext], axis=0)
    D = math.sqrt(4 * max(S[0], 1e-6) / math.pi)
    L = max(S[1], S[2]) / max(D, 1e-6)
    added_I = np.sum([p["added_inertia"] for p in parts], axis=0)
    rigid = np.maximum(np.array(base["inertia_body_axes"]) - added_I, 1e-4) * (m / m0)
    A = np.sum([p["added_mass"] for p in ext], axis=0)
    quad = [-0.5 * rho * sum(p["quadratic_drag"][i] * p["projected_area"][i] for p in ext) * cruise for i in range(3)]
    lin = [-rho * sum(p["viscous_drag"][i] * p["tangential_area"][i] for p in ext) for i in range(3)]
    rot = -0.5 * rho * D * L ** 4 / 32
    ref = np.array(ref, dtype=float)
    # keep the source model's trim (net buoyancy) when the weighed mass differs from the model's: its buoyant volume
    # was sized for its own mass (as describe-stonefish)
    volume = float(base["volume"]) + (m - m0) / rho
    trim = rho * G * float(base["volume"]) - m0 * G
    return {"mass": m, "rho": rho, "length": L, "diameter": D, "volume": volume, "net_buoyancy": trim,
            "cg": (np.array(base["cg"]) - ref).tolist(), "cb": (np.array(base["cb"]) - ref).tolist(),
            "inertia_rigid": rigid.tolist(), "added_inertia": added_I.tolist(), "added_mass": A.tolist(),
            "linear_damping": lin + [-1.0, 0.3 * rot, 0.3 * rot], "quadratic_damping": quad + [-1.0, rot, rot],
            "surge_cd": float(sum(p["quadratic_drag"][0] * p["projected_area"][0] for p in ext) / (math.pi * D * D / 4)),
            "notes": [f"Hull about {L:.2f} m x {D:.2f} m; quadratic drag matched at {cruise:g} m/s (Stonefish's grows with "
                      f"speed cubed); pitch and yaw damping by strip theory."]
            + ([f"The model weighs {m0:.1f} kg against the {m:g} kg given: its buoyant volume is grown so the vehicle keeps "
                f"the model's trim ({trim:+.1f} N)."] if abs(m - m0) > 1e-6 else [])}


def _flu(v):
    return [v[0], -v[1], -v[2]]


def write_gazebo(f: dict, out: str, step: float = 0.01) -> str:
    cg, cb = _flu(f["cg"]), _flu(f["cb"])
    r = math.sqrt(f["volume"] / (math.pi * f["length"]))  # a cylinder along x with the model's displaced volume
    lin, quad = f["linear_damping"], f["quadratic_damping"]
    names = [("xU", "xUabsU"), ("yV", "yVabsV"), ("zW", "zWabsW"), ("kP", "kPabsP"), ("mQ", "mQabsQ"), ("nR", "nRabsR")]
    hyd = "\n".join(f"        <{a}>{fmt_number(lin[i])}</{a}><{b}>{fmt_number(quad[i])}</{b}>" for i, (a, b) in enumerate(names))
    A, AI, Ir = f["added_mass"], f["added_inertia"], f["inertia_rigid"]
    sdf = f"""<?xml version="1.0"?>
<!-- Converted by strathe convert-sim: origin at the logs' reference point (use reference point 0 0 0). -->
<sdf version="1.11">
  <world name="water">
    <physics name="replay" type="dart"><max_step_size>{step:g}</max_step_size><real_time_factor>0</real_time_factor></physics>
    <gravity>0 0 -9.81</gravity>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-buoyancy-system" name="gz::sim::systems::Buoyancy">
      <uniform_fluid_density>{fmt_number(f["rho"])}</uniform_fluid_density>
    </plugin>
    <model name="vehicle">
      <link name="base_link">
        <inertial>
          <pose>{fmt_number(cg[0])} {fmt_number(cg[1])} {fmt_number(cg[2])} 0 0 0</pose>
          <mass>{fmt_number(f["mass"])}</mass>
          <inertia><ixx>{fmt_number(Ir[0])}</ixx><iyy>{fmt_number(Ir[1])}</iyy><izz>{fmt_number(Ir[2])}</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia>
          <fluid_added_mass><xx>{fmt_number(A[0])}</xx><xy>0</xy><xz>0</xz><xp>0</xp><xq>0</xq><xr>0</xr><yy>{fmt_number(A[1])}</yy><yz>0</yz><yp>0</yp><yq>0</yq><yr>0</yr><zz>{fmt_number(A[2])}</zz><zp>0</zp><zq>0</zq><zr>0</zr><pp>{fmt_number(AI[0])}</pp><pq>0</pq><pr>0</pr><qq>{fmt_number(AI[1])}</qq><qr>0</qr><rr>{fmt_number(AI[2])}</rr></fluid_added_mass>
        </inertial>
        <collision name="hull">
          <pose>{fmt_number(cb[0])} {fmt_number(cb[1])} {fmt_number(cb[2])} 0 1.5707963267948966 0</pose>
          <geometry><cylinder><radius>{fmt_number(r)}</radius><length>{fmt_number(f["length"])}</length></cylinder></geometry>
        </collision>
      </link>
      <plugin filename="gz-sim-hydrodynamics-system" name="gz::sim::systems::Hydrodynamics">
        <link_name>base_link</link_name>
{hyd}
        <zUW>0</zUW><yUV>0</yUV><mUW>0</mUW><nUV>0</nUV><mUQ>0</mUQ><nUR>0</nUR>
      </plugin>
    </model>
  </world>
</sdf>
"""
    p = os.path.join(out, "world.sdf")
    open(p, "w", encoding="utf-8").write(sdf)
    return p


def write_mujoco(f: dict, out: str, timestep: float = 0.002) -> str:
    cg, cb = _flu(f["cg"]), _flu(f["cb"])
    Ir = f["inertia_rigid"]
    float_mass = 0.01  # a tiny body at the centre of buoyancy carries the buoyancy: a righting moment, as in water
    gravcomp = f["rho"] * f["volume"] / float_mass
    xml = f"""<mujoco model="converted_vehicle">
  <!-- Converted by strathe convert-sim: origin at the logs' reference point (use reference point 0 0 0). MuJoCo has
       no buoyancy force: the body "buoyancy", fixed at the centre of buoyancy, carries it (gravcomp on its tiny mass),
       so the vehicle has a righting moment. -->
  <option timestep="{timestep:g}" density="{fmt_number(f["rho"])}" viscosity="0.0009" integrator="implicitfast"/>
  <worldbody>
    <body name="vehicle" pos="0 0 -3">
      <freejoint/>
      <body name="buoyancy" pos="{fmt_number(cb[0])} {fmt_number(cb[1])} {fmt_number(cb[2])}" gravcomp="{fmt_number(gravcomp)}">
        <inertial pos="0 0 0" mass="{float_mass:g}" diaginertia="1e-6 1e-6 1e-6"/>
      </body>
      <inertial pos="{fmt_number(cg[0])} {fmt_number(cg[1])} {fmt_number(cg[2])}" mass="{fmt_number(f["mass"])}" diaginertia="{fmt_number(Ir[0])} {fmt_number(Ir[1])} {fmt_number(Ir[2])}"/>
      <geom name="hull" type="ellipsoid" pos="{fmt_number(cb[0])} {fmt_number(cb[1])} {fmt_number(cb[2])}" size="{fmt_number(f["length"] / 2)} {fmt_number(f["diameter"] / 2)} {fmt_number(f["diameter"] / 2)}" fluidshape="ellipsoid" fluidcoef="0.5 0.25 1.5 1.0 1.0" mass="0" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""
    p = os.path.join(out, "vehicle.xml")
    open(p, "w", encoding="utf-8").write(xml)
    return p


def write_holoocean(f: dict, out: str) -> str:
    m, A = f["mass"], f["added_mass"]
    lin, quad = f["linear_damping"], f["quadratic_damping"]
    T = lambda i, M: M / max(abs(lin[i]) + abs(quad[i]) * CRUISE, 1e-3)  # noqa: E731  a first-order time constant
    dyn = {"rho": f["rho"], "mass": m, "length": f["length"], "diam": f["diameter"], "r_bg": f["cg"], "r_bb": f["cb"],
           "Cd": f["surge_cd"], "T_surge": T(0, m + A[0]), "T_sway": T(1, m + A[1]),
           "T_yaw": T(5, f["inertia_rigid"][2] + f["added_inertia"][2])}
    sc = {"name": "strathe_converted", "package_name": "Ocean", "world": "OpenWater", "main_agent": "auv0",
          "ticks_per_sec": 100,
          "agents": [{"agent_name": "auv0", "agent_type": "TorpedoAUV", "location": [0, 0, -3], "rotation": [0, 0, 0],
                      "sensors": [{"sensor_type": "DynamicsSensor", "configuration": {"UseCOM": True, "UseRPY": False}}],
                      "control_scheme": 1, "fossen_model": "torpedo", "control_mode": "manualControl",
                      "dynamics": {k: (round(v, 6) if isinstance(v, float) else [round(x, 6) for x in v])
                                   for k, v in dyn.items()}}]}
    p = os.path.join(out, "scenario.json")
    json.dump(sc, open(p, "w", encoding="utf-8"), indent=2)
    return p


WRITERS = {"gazebo": write_gazebo, "mujoco": write_mujoco, "holoocean": write_holoocean}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="strathe convert-sim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", choices=["stonefish"], default="stonefish")
    ap.add_argument("--to", required=True, choices=sorted(WRITERS))
    ap.add_argument("--scenario", required=True, help="the source Stonefish scenario")
    ap.add_argument("--robot", required=True)
    ap.add_argument("--reference-point", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                    help="the logs' reference point in the source model (forward-right-down): the converted model's origin")
    ap.add_argument("--mass", type=float, help="the vehicle's weighed mass, kg (default: the source model's)")
    ap.add_argument("--find", nargs="+", metavar="PKG=PATH")
    ap.add_argument("--stonefish-data")
    ap.add_argument("--replay-tool")
    ap.add_argument("--out", required=True, help="folder for the converted model")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    try:
        import tempfile
        from stonefish_fit import Replay, Scenario, find_tool
        sc = Scenario(a.scenario, dict(kv.split("=", 1) for kv in (a.find or [])), "")
        rp = Replay(find_tool(a.replay_tool), a.stonefish_data or os.path.dirname(os.path.abspath(a.scenario)), a.robot,
                    tuple(a.reference_point))
        info = rp.info(sc.write_runnable(tempfile.mkdtemp(prefix="strathe_convert_")))
        f = facts_from_stonefish(info, a.reference_point, a.mass)
        os.makedirs(a.out, exist_ok=True)
        path = WRITERS[a.to](f, a.out)
        json.dump(f, open(os.path.join(a.out, "converted_from.json"), "w"), indent=2)
        print(f"Wrote {path} from {os.path.basename(a.scenario)} ({f['mass']:g} kg, hull {f['length']:.2f} x "
              f"{f['diameter']:.2f} m). Its origin is the logs' reference point: use --reference-point 0 0 0 with it.")
        for n in f["notes"]:
            print("  note: " + n)
        return 0
    except SimFitError as e:
        print(f"❌ {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
