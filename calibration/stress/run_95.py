"""
stress/run_95.py -- stress test of `strathe describe-vehicle` (PROJECT_STATUS.md #95) on the real datasets.

    python stress/run_95.py fast        # every input and model source: specs written and checked (minutes)
    python stress/run_95.py mutants     # each deliberate fault must be caught
    python stress/run_95.py e2e [ROW]   # calibrations: each generated spec against the hand-written one (hours)
    python stress/run_95.py list

The proof is calibration on real logs: each vehicle is calibrated with a spec written by describe-vehicle (from its
published size, or from its model in a simulator) and with the spec a person wrote by hand for it, on the same logs,
held-out days and seed. Pass, per vehicle: held-out mean skill no worse than the hand spec's by more than 0.05; every
value the hand-spec calibration found inside the generated limits; no more values pinned at a limit.
"""
from __future__ import annotations

import json
import os
import shutil
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, CAL)
sys.path.insert(0, HERE)
from common import record, strathe  # noqa: E402

import vehicle_describe as VD  # noqa: E402

F = "95"
DATA = os.path.expanduser("~/strathe_describe_data")
BLUEROV = os.path.expanduser("~/marine_sim_mvp/data/raw/Bluerov2_trajectory_dataset/dataset")
DFKI = os.path.expanduser("~/strathe_session_data/dfki_dated")
URI_LOGS = os.path.expanduser("~/strathe_uri_lab_v2/runs/ingest_I0")
URI_HAND = os.path.expanduser("~/strathe_uri_lab_v2/data/alpha_rise_spec.yaml")
URI_DESCRIBED = os.path.expanduser("~/strathe_uri_lab_v2/data/alpha_rise_spec_from_stonefish_final.yaml")
HAND_BLUEROV = os.path.join(CAL, "vehicle_spec.yaml")
GZ_BLUEROV = os.path.expanduser("~/strathe_gz_public/bluerov2_gz/models/bluerov2_heavy/model.sdf")
GZ_ALPHA = os.path.expanduser("~/strathe_uri_lab_v3/models/alpha_gazebo/world.sdf")
MJ_ALPHA = os.path.expanduser("~/strathe_uri_lab_v3/models/alpha_mujoco/vehicle.xml")
# published sizes: BlueROV2 Heavy (Blue Robotics: 457 x 575 x 254 mm, 11.5 kg); ALPHA RISE (URI: 1.5 m, 10-inch
# hull, 33 kg); DFKI's 6-thruster BlueROV2 (457 x 338 x 254 mm; with its gripper and ballast, 11.5 kg as weighed in
# the dataset's paper is not given, so the standard R4's 11.5 kg)
BLUE = {"mass": 11.5, "length": 0.457, "width": 0.575, "height": 0.254}
ALPHA = {"mass": 33.0, "length": 1.5, "diameter": 0.254}
DFKI_DIMS = {"mass": 11.5, "length": 0.457, "width": 0.338, "height": 0.254}
BLUE_CONV = ["--torque-reference-offset", "0", "0", "-0.085", "--water", "fresh"]   # the dataset's convention, a tank


def describe(out: str, args: list) -> tuple:
    """Run describe-vehicle through strathe; (exit code, spec dict or None, printed text)."""
    log = out + ".log"
    rc, _ = strathe(["describe-vehicle", *args, "--out", out], log)
    text = open(log, encoding="utf-8").read()
    spec = yaml.safe_load(open(out)) if rc == 0 and os.path.exists(out) else None
    return rc, spec, text


def targets(spec: dict) -> dict:
    return {k: v for g in spec["calibration_targets"].values() for k, v in g.items()
            if isinstance(v, dict) and "bounds" in v}


def sane(spec: dict) -> list:
    """Problems with a written spec: limits the wrong way round, a start outside them, damping not negative."""
    bad = []
    damping = {k for g in ("linear_damping", "quadratic_damping") for k in spec["calibration_targets"].get(g, {})}
    for k, v in targets(spec).items():
        lo, hi = v["bounds"]
        if not lo < hi:
            bad.append(f"{k}: limits {lo}..{hi}")
        if not lo <= v["initial_guess"] <= hi:
            bad.append(f"{k}: start {v['initial_guess']} outside {lo}..{hi}")
        if k in damping and hi >= 0:   # damping only (hull lift Muw takes either sign)
            bad.append(f"{k}: damping limit not below 0")
    return bad


# --- fast: the specs ----------------------------------------------------------------------------------------------
def _args(d: dict) -> list:
    out = []
    for k, v in d.items():
        out += [f"--{k.replace('_', '-')}", *(str(x) for x in (v if isinstance(v, (list, tuple)) else [v]))]
    return out


def g_inputs() -> None:
    """Every input, one at a time from each vehicle's published size: the spec loads, its limits are sane, and they
    move the way the physics says."""
    os.makedirs(DATA, exist_ok=True)
    base = {"blue": (BLUE, BLUE_CONV), "alpha": (ALPHA, [])}
    for veh, (dims, extra) in base.items():
        rc, ref, text = describe(os.path.join(DATA, f"G-{veh}-base.yaml"), _args(dims) + extra)
        record(F, f"G-{veh}-base", "inputs", "published size", f"{veh}: from its published size", veh,
               {"exit": rc, "problems": sane(ref) if ref else "no spec"}, "loads; sane limits",
               rc == 0 and ref is not None and not sane(ref))
        t0 = targets(ref)
        for key in dims:
            for f in (0.7, 0.9, 1.1, 1.3):
                d = {**dims, key: round(dims[key] * f, 6)}
                rc, s, _ = describe(os.path.join(DATA, f"G-{veh}-{key}-{f}.yaml"), _args(d) + extra)
                t = targets(s) if s else {}
                # what must move: drag limits with area, inertia with mass and size, added mass limits with size/mass
                moved = {k: t[k]["bounds"][0] / t0[k]["bounds"][0] for k in ("Xuu", "Ieff_pitch") if t and
                         t0[k]["bounds"][0]}
                expect = {"length": ("Ieff_pitch", 1.0), "mass": ("Ieff_pitch", 1.0),
                          "width": ("Xuu", 1.0), "height": ("Xuu", 1.0), "diameter": ("Xuu", 1.0)}[key]
                k, _ = expect
                direction = (moved.get(k, 1.0) - 1.0) * (f - 1.0)
                ok = rc == 0 and s is not None and not sane(s) and direction > 0
                record(F, f"G-{veh}-{key}-{f}", key, f"{f:g}x", f"{veh}: {key} x {f:g}", veh,
                       {"exit": rc, "ratio_" + k: round(moved.get(k, float('nan')), 3), "problems": sane(s) if s else None},
                       f"loads; {k}'s limit moves with {key}", ok)
        for name, more in (("shape-box", ["--shape", "box"]), ("shape-torpedo", ["--shape", "torpedo"]),
                           ("water-fresh", ["--water", "fresh"]), ("water-sea", ["--water", "sea"]),
                           ("water-1010", ["--water", "1010"]), ("inertia", ["--inertia", "0.2", "0.3", "0.4"]),
                           ("cg", ["--cg-below-cb", "0.03"]), ("net-buoyancy", ["--net-buoyancy", "5"]),
                           ("volume", ["--volume", str(round(dims["mass"] / 1000, 5))]),
                           ("reference-point", ["--reference-point", "0.1", "0", "0.05"]),
                           ("thrusters", ["--thrusters", "bluerov2_heavy"]), ("hull-lift", ["--hull-lift"]),
                           ("name", ["--name", f"{veh}_stress"])):
            if veh == "blue" and name in ("water-fresh",):
                continue
            rc, s, text = describe(os.path.join(DATA, f"G-{veh}-{name}.yaml"), _args(dims) + extra + more)
            info = {"exit": rc, "problems": sane(s) if s else "no spec", "notes": [ln.strip() for ln in text.splitlines()
                                                                                 if "note:" in ln][:2]}
            ok = rc == 0 and s is not None and not sane(s)
            if ok and name == "inertia":
                ok = abs(targets(s)["Ieff_roll"]["initial_guess"] - targets(ref)["Ieff_roll"]["initial_guess"]) > 1e-6
            if ok and name == "hull-lift":
                ok = ("hull_lift" in s["calibration_targets"]) == (veh == "alpha")
            if ok and name == "shape-box" and veh == "alpha":
                ok = "box" in s["metadata"]["description"]
            if ok and name.startswith("shape-"):   # the shape the size doesn't suggest is warned about, and only that one
                ok = ("goes against the size" in text) == (name == {"blue": "shape-torpedo", "alpha": "shape-box"}[veh])
            if ok and name.startswith("water"):
                ok = targets(s)["Xuu"]["bounds"] != t0["Xuu"]["bounds"] or name == "water-sea" and veh == "alpha"
            record(F, f"G-{veh}-{name}", "inputs", name, f"{veh}: {name}", veh, info, "loads; the input shows", ok)


def m_models() -> None:
    """Every model source: Gazebo SDF (bluerov2_gz; ALPHA RISE converted), URDF, MuJoCo, HoloOcean."""
    urdf = _urdf_of(GZ_BLUEROV)
    holo = _holoocean_alpha()
    cases = [("gazebo-bluerov2_gz", GZ_BLUEROV, "gazebo", [], "thinner"),
             ("gazebo-bluerov2_gz-sized", GZ_BLUEROV, "gazebo", ["--height", "0.254"], None),
             ("urdf-bluerov2", urdf, "urdf", [], "thinner"),
             ("gazebo-model-name", GZ_BLUEROV, "gazebo", ["--model-name", "bluerov2_heavy", "--height", "0.254"], None),
             ("mujoco-model-name", MJ_ALPHA, "mujoco", ["--model-name", "vehicle"], None),
             ("gazebo-alpha", GZ_ALPHA, "gazebo", [], "sized to float the vehicle"),
             ("gazebo-alpha-sized", GZ_ALPHA, "gazebo", ["--length", "1.5", "--diameter", "0.254"], None),
             ("mujoco-alpha", MJ_ALPHA, "mujoco", [], None),
             ("mujoco-example", os.path.join(CAL, "simulators", "mujoco", "example", "auv.xml"), "mujoco", [], None),
             ("holoocean-alpha", holo, "holoocean", [], None),
             ("holoocean-example-sized", os.path.join(CAL, "simulators", "holoocean", "example", "scenario.json"),
              "holoocean", ["--length", "1.6", "--diameter", "0.19"], None)]
    for name, path, sim, more, note in cases:
        if not path or not os.path.exists(path):
            record(F, f"MS-{name}", "model", sim, f"{name}: model file missing", "-", {"path": path}, "present", False)
            continue
        rc, s, text = describe(os.path.join(DATA, f"MS-{name}.yaml"), ["--from-model", path, "--sim", sim, *more])
        ok = rc == 0 and s is not None and not sane(s) and (note is None or note in text)
        record(F, f"MS-{name}", "model", sim, f"describe-vehicle --from-model ({name})", os.path.basename(path),
               {"exit": rc, "description": (s or {}).get("metadata", {}).get("description", "")[:160],
                "note_expected": note, "problems": sane(s) if s else text[-200:]}, "loads; sane; the expected note", ok)


def _urdf_of(sdf: str) -> str:
    """A URDF of bluerov2_gz's base link (its mass, inertia and collision box, as a URDF has them)."""
    import xml.etree.ElementTree as ET
    root = ET.parse(sdf).getroot()
    link = next(lk for lk in root.iter("link") if lk.get("name") == "base_link")
    m = float(link.findtext("inertial/mass"))
    I = {k: link.findtext(f"inertial/inertia/{k}") for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")}
    size = link.find("collision").findtext("geometry/box/size")
    pose = (link.find("collision").findtext("pose") or "0 0 0 0 0 0").split()
    p = os.path.join(DATA, "bluerov2.urdf")
    open(p, "w").write(f"""<?xml version="1.0"?>
<robot name="bluerov2">
  <link name="base_link">
    <inertial><mass value="{m}"/><inertia ixx="{I['ixx']}" iyy="{I['iyy']}" izz="{I['izz']}" ixy="{I['ixy'] or 0}" ixz="{I['ixz'] or 0}" iyz="{I['iyz'] or 0}"/></inertial>
    <collision><origin xyz="{' '.join(pose[:3])}" rpy="{' '.join(pose[3:6])}"/><geometry><box size="{size}"/></geometry></collision>
  </link>
</robot>
""")
    return p


def _holoocean_alpha() -> str:
    """ALPHA RISE as a HoloOcean scenario, written by convert-sim from URI's Stonefish model (Docker)."""
    out = os.path.join(DATA, "alpha_holoocean")
    p = os.path.join(out, "scenario.json")
    if os.path.exists(p):
        return p
    rc, _ = strathe(["convert-sim", "--from", "stonefish", "--to", "holoocean",
                     "--scenario", os.path.expanduser("~/strathe_uri_lab_v3/stonefish_v2/lab/alpha_world.scn"),
                     "--robot", "alpha_rise", "--find",
                     "world_of_stonefish=" + os.path.expanduser("~/strathe_uri_lab_v2/stonefish/world"),
                     "--stonefish-data", os.path.expanduser("~/strathe_uri_lab_v3/stonefish_v2/world/data"),
                     "--reference-point", "-0.76", "0", "0", "--replay-tool", "docker", "--out", out],
                    os.path.join(DATA, "convert_holoocean.log"))
    return p if rc == 0 and os.path.exists(p) else None   # (converted_from.json next to it is the conversion's record)


def x_refusals() -> None:
    cases = [("no size", ["--mass", "10", "--length", "1"], "give the vehicle's size"),
             ("zero length", ["--mass", "10", "--length", "0", "--diameter", "0.2"], "above 0"),
             ("negative width", ["--mass", "10", "--length", "1", "--width", "-1", "--height", "0.2"], "above 0"),
             ("no mass", ["--length", "1", "--diameter", "0.2"], "give --mass"),
             ("missing model", ["--from-model", "/nope/robot.urdf", "--sim", "urdf"], "not found"),
             ("model without mass", ["--from-model", _nomass(), "--sim", "urdf"], "no link with an <inertial><mass>"),
             ("holoocean without size", ["--from-model", os.path.join(CAL, "simulators", "holoocean", "example",
                                                                       "scenario.json"), "--sim", "holoocean"],
              "no length and diam"),
             ("bad water", ["--mass", "10", "--length", "1", "--diameter", "0.2", "--water", "salty"], "could not"),
             ("bad inertia", ["--mass", "10", "--length", "1", "--diameter", "0.2", "--inertia", "1", "0", "1"],
              "three values above 0"),
             ("unknown thrusters", ["--mass", "10", "--length", "1", "--diameter", "0.2", "--thrusters", "nope"],
              "isn't a file or a preset")]
    for label, args, want in cases:
        out = os.path.join(DATA, f"X-{label.replace(' ', '_')}.yaml")
        rc, s, text = describe(out, args)
        ok = rc != 0 and want.lower() in text.lower() and "Traceback" not in text
        record(F, f"X-{label.replace(' ', '_')}", "refusal", label, f"refused: {label}", "-",
               {"exit": rc, "said": text.strip().splitlines()[-1][:200] if text.strip() else ""},
               f"a plain refusal mentioning '{want}'", ok)


def _nomass() -> str:
    p = os.path.join(DATA, "nomass.urdf")
    open(p, "w").write('<robot name="x"><link name="a"><collision><geometry><box size="1 1 1"/></geometry>'
                       '</collision></link></robot>')
    return p


def w_web() -> None:
    import strathe_web as web
    m = web.JobManager(os.path.join(DATA, "web"))
    h = web.StratheHandler.__new__(web.StratheHandler)
    h.manager = m
    for label, form in (("measurements", {"mass": "11.5", "length": "0.457", "width": "0.575", "height": "0.254",
                                          "water": "fresh", "name": "web_blue"}),
                        ("model", {"from_model": GZ_BLUEROV, "sim": "gazebo", "height": "0.254", "name": "web_gz"}),
                        ("torpedo", {"mass": "33", "length": "1.5", "diameter": "0.254", "shape": "torpedo",
                                     "cg_below_cb": "0.02", "net_buoyancy": "3", "hull_lift": "on", "water": "1023",
                                     "name": "web_alpha"})):
        job = h._start("/jobs/describe_vehicle", form)
        job.proc.wait()
        out = job.argv[job.argv.index("--out") + 1]
        ok = job.proc.returncode == 0 and os.path.exists(out) and not sane(yaml.safe_load(open(out)))
        record(F, f"W-{label}", "web", label, f"the web card: from {label}", "-",
               {"exit": job.proc.returncode, "out": out}, "the job writes a sane spec", ok)
    for label, form, want in (("nothing", {"name": "x"}, "mass and length"),
                              ("model without kind", {"from_model": GZ_BLUEROV, "name": "x"}, "Gazebo, URDF")):
        try:
            h._start("/jobs/describe_vehicle", form)
            ok, msg = False, "accepted"
        except ValueError as e:
            ok, msg = want in str(e), str(e)
        record(F, f"W-refuse-{label.replace(' ', '_')}", "web", label, f"the web card refuses: {label}", "-",
               {"message": msg}, "a plain refusal", ok)


def fast() -> int:
    os.makedirs(DATA, exist_ok=True)
    for step in (g_inputs, m_models, x_refusals, w_web):
        try:
            step()
        except Exception as e:
            import traceback
            traceback.print_exc()
            record(F, f"crash-{step.__name__}", "harness", step.__name__, "crashed", "-", {"error": str(e)[:300]},
                   "runs", False)
    return 0


def mutants() -> int:
    """Bounds that don't scale with size: the input rows must fail. Run in-process on describe_hull."""
    real = VD.box_bounds, VD.torpedo_bounds
    fails = []
    VD.box_bounds = lambda rho, L, W, H, m, *a: real[0](rho, 1.0, 1.0, 1.0, m)
    VD.torpedo_bounds = lambda rho, L, D: real[1](rho, 1.0, 0.2)
    try:
        for veh, dims in (("blue", BLUE), ("alpha", ALPHA)):
            s0, _ = VD.describe_hull(**{k: v for k, v in dims.items()})
            for key in ("length",) + (("width",) if "width" in dims else ("diameter",)):
                s1, _ = VD.describe_hull(**{**dims, key: dims[key] * 1.3})
                a, b = targets(s0)["Xuu"]["bounds"][0], targets(s1)["Xuu"]["bounds"][0]
                if not (b / a > 1.0 if key != "length" or veh == "blue" else True) or abs(b - a) < 1e-12:
                    fails.append(f"{veh} {key}: Xuu limit didn't move")
    finally:
        VD.box_bounds, VD.torpedo_bounds = real
    record(F, "M-bounds_not_scaled", "mutant", "bounds not scaled with size", "deliberate fault: bounds fixed to a "
           "1 m hull", "-", {"caught": fails}, "the size rows fail", bool(fails))
    return 0


# --- end to end ---------------------------------------------------------------------------------------------------
def calibrate(row: str, data: str, spec: str, extra: list) -> dict:
    out = os.path.join(DATA, "cal", row)
    if not os.path.exists(os.path.join(out, "metrics.json")):
        shutil.rmtree(out, ignore_errors=True)
        rc, secs = strathe(["calibrate", "--data", data, "--spec", spec, "--name", row, "--out", out, "--seed", "95",
                            *extra], out + ".log")
        if rc:
            return {"error": f"calibrate exit {rc}", "seconds": secs}
    from calibration_report import pinned_parameters
    from dynamics_model import CalibrationParameters
    from optimizer import load_vehicle_spec
    m = json.load(open(os.path.join(out, "metrics.json")))
    params = CalibrationParameters(**yaml.safe_load(open(os.path.join(out, "calibrated_parameters.yaml")))["parameters"])
    skill = m["skill_vs_zero"]["calibrated_heldout_full_replay"]
    pinned = pinned_parameters(params, load_vehicle_spec(spec))
    return {"out": out, "mean_skill": round(float(np.mean(skill)), 4), "skill": [round(x, 3) for x in skill],
            "pinned": [p["name"] for p in pinned], "pinned_at": {p["name"]: p["bound"] for p in pinned},
            "params": dict(zip(params.param_names, (float(x) for x in params.to_vector())))}


def compare(row: str, hand: dict, gen: dict, gen_spec: str, what: str, data: str) -> None:
    if "error" in hand or "error" in gen:
        record(F, row, "calibration", what, f"{what}: calibration failed", data, {"hand": hand.get("error"),
                                                                                 "generated": gen.get("error")},
               "both calibrate", False)
        return
    t = targets(yaml.safe_load(open(gen_spec)))
    outside = [k for k, v in hand["params"].items() if k in t and isinstance(v, (int, float)) and
               not (t[k]["bounds"][0] - 1e-9 <= v <= t[k]["bounds"][1] + 1e-9)]
    # A value the hand fit left at one of its own limits isn't one it found: the data wanted more than that range
    # allows. It only has to be pinned at the same end in the generated fit; a value it found must be inside.
    pinned_both = [k for k in outside if hand["pinned_at"].get(k) and hand["pinned_at"][k] == gen["pinned_at"].get(k)]
    cut_off = [k for k in outside if k not in pinned_both]
    nums = {"hand_mean_skill": hand["mean_skill"], "generated_mean_skill": gen["mean_skill"],
            "difference": round(gen["mean_skill"] - hand["mean_skill"], 4), "hand_values_outside_generated_limits":
            outside, "of_which_pinned_at_the_same_end_in_both": pinned_both, "found_values_cut_off": cut_off,
            "pinned_hand": len(hand["pinned"]), "pinned_generated": len(gen["pinned"]),
            "skill_hand": hand["skill"], "skill_generated": gen["skill"]}
    ok = gen["mean_skill"] >= hand["mean_skill"] - 0.05 and not cut_off and len(gen["pinned"]) <= len(hand["pinned"])
    record(F, row, "calibration", what, f"{what}: generated spec against the hand-written one", data, nums,
           "skill within 0.05 of the hand spec's or better; every value the hand fit found inside the generated "
           "limits (one pinned at a hand limit: pinned at the same end in both); no more pinned", ok)


def e2e(only=None) -> int:
    os.makedirs(os.path.join(DATA, "cal"), exist_ok=True)
    blue_held = ["--heldout-dates", "16-02-2024", "19-02-2024", "--skip-identifiability"]
    rows = {}

    def spec(name, args):
        p = os.path.join(DATA, "specs", f"{name}.yaml")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p):
            rc, s, text = describe(p, args)
            if rc:
                raise RuntimeError(text[-300:])
        return p

    def blue(name, args, what):
        def run():
            hand = calibrate("blue-hand", BLUEROV, HAND_BLUEROV, blue_held)
            gs = spec(name, args)
            compare(f"E-{name}", hand, calibrate(name, BLUEROV, gs, blue_held), gs, what, "bluerov2 (16, 19 Feb held out)")
        rows[f"E-{name}"] = run

    blue("blue-dims", _args(BLUE) + BLUE_CONV, "BlueROV2 from its published size")
    blue("blue-gz-sized", ["--from-model", GZ_BLUEROV, "--sim", "gazebo", "--height", "0.254"] + BLUE_CONV,
         "BlueROV2 from bluerov2_gz (Gazebo), its height given")
    blue("blue-gz-raw", ["--from-model", GZ_BLUEROV, "--sim", "gazebo"] + BLUE_CONV,
         "BlueROV2 from bluerov2_gz as it is (a 5 cm buoyancy block for the hull)")
    blue("blue-urdf", ["--from-model", os.path.join(DATA, "bluerov2.urdf"), "--sim", "urdf", "--height", "0.254"] +
         BLUE_CONV, "BlueROV2 from a URDF")
    for key, f in (("length", 0.7), ("length", 1.3), ("mass", 0.7), ("mass", 1.3), ("width", 0.7), ("height", 1.3)):
        blue(f"blue-{key}-{f}", _args({**BLUE, key: round(BLUE[key] * f, 6)}) + BLUE_CONV,
             f"BlueROV2, {key} off by {round((f - 1) * 100):+d}%")
    blue("blue-as-torpedo", _args(BLUE) + BLUE_CONV + ["--shape", "torpedo"], "BlueROV2 described as a torpedo")
    blue("blue-sea", _args(BLUE) + ["--torque-reference-offset", "0", "0", "-0.085", "--water", "sea"],
         "BlueROV2, sea water assumed (it was a tank)")

    def blue_restarts():
        hand = calibrate("blue-hand-r4", BLUEROV, HAND_BLUEROV, blue_held + ["--restarts", "4"])
        gs = spec("blue-dims", _args(BLUE) + BLUE_CONV)
        compare("E-blue-restarts4", hand, calibrate("blue-dims-r4", BLUEROV, gs, blue_held + ["--restarts", "4"]), gs,
                "BlueROV2 from its size, 4 starting points", "bluerov2")
    rows["E-blue-restarts4"] = blue_restarts

    def dfki():
        hand = calibrate("dfki-hand", DFKI, HAND_BLUEROV, ["--skip-identifiability"])
        gs = spec("dfki-dims", _args(DFKI_DIMS) + ["--water", "sea"])
        compare("E-dfki-dims", hand, calibrate("dfki-dims", DFKI, gs, ["--skip-identifiability"]), gs,
                "DFKI's 6-thruster BlueROV2 from its size", "dfki")
    rows["E-dfki-dims"] = dfki

    for held, tag in (("27-03-2026", "pondout"), ("05-12-2025", "bayout")):
        extra = ["--heldout-dates", held, "--estimate-current", "--workers", "2", "--skip-identifiability"]

        def alpha(name, args, what, held=held, tag=tag, extra=extra):
            def run():
                hand = calibrate(f"alpha-hand-{tag}", URI_LOGS, URI_HAND, extra)
                gs = spec(name, args)
                compare(f"E-{name}-{tag}", hand, calibrate(f"{name}-{tag}", URI_LOGS, gs, extra), gs, what,
                        f"ALPHA RISE ({held} held out)")
            rows[f"E-{name}-{tag}"] = run
        alpha("alpha-dims", _args(ALPHA), "ALPHA RISE from its published size")
        alpha("alpha-gazebo", ["--from-model", GZ_ALPHA, "--sim", "gazebo"], "ALPHA RISE from its Gazebo model")
        alpha("alpha-gazebo-sized", ["--from-model", GZ_ALPHA, "--sim", "gazebo", "--length", "1.5", "--diameter", "0.254"],
              "ALPHA RISE from its Gazebo model, its real size given")
        alpha("alpha-mujoco", ["--from-model", MJ_ALPHA, "--sim", "mujoco"], "ALPHA RISE from its MuJoCo model")
        alpha("alpha-holoocean", ["--from-model", os.path.join(DATA, "alpha_holoocean", "scenario.json"), "--sim",
                                  "holoocean", "--length", "1.5", "--diameter", "0.254"], "ALPHA RISE from HoloOcean")
        alpha("alpha-as-box", _args(ALPHA) + ["--shape", "box"], "ALPHA RISE described as a box frame")
        alpha("alpha-hull-lift", _args(ALPHA) + ["--hull-lift"], "ALPHA RISE with the hull-lift terms")

        def described(tag=tag, extra=extra, held=held):
            hand = calibrate(f"alpha-hand-{tag}", URI_LOGS, URI_HAND, extra)
            compare(f"E-alpha-describe-stonefish-{tag}", hand,
                    calibrate(f"alpha-described-{tag}", URI_LOGS, URI_DESCRIBED, extra), URI_DESCRIBED,
                    "ALPHA RISE: describe-stonefish's spec (for reference)", f"ALPHA RISE ({held} held out)")
        rows[f"E-alpha-describe-stonefish-{tag}"] = described
    if only == ["__rows__"]:
        return rows
    for rid, fn in rows.items():
        if only and rid not in only and not any(rid.startswith(o) for o in only):
            continue
        print(f"=== {rid}", flush=True)
        try:
            fn()
        except Exception as e:
            record(F, rid, "e2e", rid, "crashed", "-", {"error": f"{type(e).__name__}: {e}"[:300]}, "runs", False)
    return 0


def _e2e_rows() -> dict:
    return e2e(["__rows__"])


def all_rows() -> set:
    """Every row id this harness writes (the coverage test checks the matrix against it)."""
    rows = set()
    for veh, dims in (("blue", BLUE), ("alpha", ALPHA)):
        rows.add(f"G-{veh}-base")
        rows |= {f"G-{veh}-{k}-{f}" for k in dims for f in (0.7, 0.9, 1.1, 1.3)}
        rows |= {f"G-{veh}-{n}" for n in ("shape-box", "shape-torpedo", "water-fresh", "water-sea", "water-1010",
                                          "inertia", "cg", "net-buoyancy", "volume", "reference-point", "thrusters",
                                          "hull-lift", "name") if not (veh == "blue" and n == "water-fresh")}
    rows |= {f"MS-{n}" for n in ("gazebo-bluerov2_gz", "gazebo-bluerov2_gz-sized", "urdf-bluerov2", "gazebo-model-name",
                                 "mujoco-model-name", "gazebo-alpha", "gazebo-alpha-sized", "mujoco-alpha", "mujoco-example", "holoocean-alpha",
                                 "holoocean-example-sized")}
    rows |= {f"X-{n}" for n in ("no_size", "zero_length", "negative_width", "no_mass", "missing_model",
                                "model_without_mass", "holoocean_without_size", "bad_water", "bad_inertia",
                                "unknown_thrusters")}
    rows |= {"W-measurements", "W-model", "W-torpedo", "W-refuse-nothing", "W-refuse-model_without_kind",
             "M-bounds_not_scaled"}
    return rows | set(_e2e_rows())


ROWS_FAST = ("G", "MS", "X", "W", "M")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if cmd == "list":
        print("fast:", ROWS_FAST, "\ne2e:", " ".join(_e2e_rows()))
        sys.exit(0)
    sys.exit({"fast": fast, "mutants": mutants}.get(cmd, lambda: e2e(sys.argv[2:]))())
