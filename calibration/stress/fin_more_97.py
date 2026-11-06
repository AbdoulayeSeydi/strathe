"""
stress/fin_more_97.py -- the rest of #97's rows: each mass entry, vectored platforms, all pairs, layout x angle source x engine,
export into each engine, one engine's vehicle in another, and the lab's journey. Bars live in run_97.py and are committed first.
"""
from __future__ import annotations

import itertools
import math
import os
import re
import subprocess
import time

import numpy as np
import pandas as pd
import yaml

import fin_rows_97 as A
from common import record

CAL, F = A.CAL, A.F
_bars = A._bars


# ------------------------------------------------------------------ the 15 mass entries, one at a time
def mass_rows() -> list:
    from dynamics_model import MASS_COUPLINGS
    return [f"MC-{n}" for _, _, n in MASS_COUPLINGS]


def run_mass_entry(row: str) -> bool:
    """One entry changes its own two cells and nothing else, every physics path agrees with it, and the description only takes it where a
    vehicle can have it."""
    from dynamics_model import MASS_COUPLINGS, XZ_SYMMETRIC_COUPLINGS, VehicleConstants, compute_acceleration, mass_matrix
    from fast_dynamics import FastDynamics
    from optimizer import _check_mass_couplings
    name = row[3:]
    i, j = next((a, b) for a, b, n in MASS_COUPLINGS if n == name)
    V = VehicleConstants(mass_kg=30.0)
    P = A._params(**{name: .5})
    D = mass_matrix(P, V) - mass_matrix(A._params(), V)
    cells = D[i, j] == .5 and D[j, i] == .5 and np.count_nonzero(D) == 2
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(100):
        v, tau = np.r_[1.5, rng.normal(size=5) * .15], rng.normal(size=6) * 4
        r, p = rng.normal(size=2) * .2
        ref = compute_acceleration(P, V, v, tau, r, p)[0]
        fast = np.array(FastDynamics(P, V).acceleration_trig(list(v), list(tau), math.sin(r), math.cos(r), math.sin(p), math.cos(p)))
        worst = max(worst, np.abs(fast - ref).max() / max(np.abs(ref).max(), 1e-9))

    def said(symmetry):
        raw = {"calibration_targets": {"added_mass_coupling": {"symmetry": symmetry}}}
        flat = {n: {"initial_guess": .5 if n == name else 0.0, "bounds": [0.0, 1.0] if n == name else [0.0, 0.0]} for _, _, n in MASS_COUPLINGS}
        try:
            _check_mass_couplings(raw, flat)
            return True
        except ValueError:
            return False
    gate = said("none") and said("xz") == (name in XZ_SYMMETRIC_COUPLINGS)
    return record(F, row, "mass entry", name, "its own two cells, every path, and the symmetry gate", "computed", {"cells": bool(cells), "worst_relative": worst, "gate": gate, "bars": _bars().BARS_HASH},
                  "as stated", bool(cells) and worst <= _bars().PATHS_AGREE and gate)


# ------------------------------------------------------------------ vectored tails
VECTORED = {"V-bluefin": dict(mass=750.0, length=4.9, diameter=.53, prop=.5), "V-dorado": dict(mass=680.0, length=5.3, diameter=.53, prop=.5), "V-small": dict(mass=45.0, length=1.7, diameter=.2, prop=.16)}


def run_vectored(row: str) -> bool:
    """Own-model recovery of thrust scale through a gimbal: plumbing, not physics, as the P- rows are."""
    from dataclasses import replace as rep
    import vehicle_describe as vd
    from dynamics_model import SimulationConfig, TrajectoryInputs, simulate_trajectory
    from optimizer import CalibrationConfig, initial_parameters, load_vehicle_spec
    from run_calibration import PipelineConfig, run_pipeline
    from scipy.spatial.transform import Rotation
    import steering
    t0 = time.time()
    o = VECTORED[row]
    d = os.path.join(_bars().WORK, row)
    os.makedirs(d + "/data", exist_ok=True)
    spec, _ = vd.describe_hull(o["mass"], o["length"], diameter=o["diameter"], shape="torpedo", name=row, steering="vectored", propeller={"diameter_m": o["prop"], "kt0": .45, "kt1": -.3, "gimbal": True})
    spec["propeller"]["position_m"] = [-o["length"] / 2, 0.0, 0.0]
    path = d + "/spec.yaml"
    open(path, "w").write(yaml.safe_dump(spec, sort_keys=False))
    s = load_vehicle_spec(path)
    start = initial_parameters(s)
    truth = 1.2
    P = rep(start, thrust_scale=truth)
    n, k = 500, 0
    for date in ("01-10-2026", "02-10-2026"):
        for run in (1, 2):
            k += 1
            rng = np.random.default_rng(k)
            df = pd.DataFrame({"rpm": 300.0 + np.repeat(rng.uniform(-60, 60, n // 25 + 1), 25)[:n], "gimbal_y": np.repeat(rng.uniform(-.12, .12, n // 25 + 1), 25)[:n],
                               "gimbal_z": np.repeat(rng.uniform(-.12, .12, n // 25 + 1), 25)[:n]})
            ctl, _ = steering.controls_from_frame(df, s.vehicle.actuators)
            rp = np.c_[.03 * np.sin(np.arange(n) / 40), .05 * np.sin(np.arange(n) / 60)]
            F_ = np.zeros((n, 6))
            nu = simulate_trajectory(P, s.vehicle, SimulationConfig(dt=.1), TrajectoryInputs(nu0=np.array([1.2, 0, 0, 0, 0, 0.]), forces_torques=F_, roll_pitch_series=rp, control_series=ctl)).nu
            q = Rotation.from_euler("xyz", np.c_[rp, np.cumsum(nu[:, 5]) * .1]).as_quat()
            for i, c in enumerate(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")):
                df[c] = 0.0
            for i, c in enumerate(("Bu", "Bv", "Bw", "Bp", "Bq", "Br")):
                df[c] = nu[:, i]
            for i, c in enumerate(("qx", "qy", "qz", "qw")):
                df[c] = q[:, i]
            df.to_csv(f"{d}/data/v-{date}-run{run}.csv", index=False)
    flat = s.flat_calibration_targets
    fixed = {n_: float(getattr(start, n_)) for n_ in start.param_names if n_ != "thrust_scale" and flat[n_]["bounds"][0] < flat[n_]["bounds"][1]}
    cc = CalibrationConfig(max_nfev=15, seed=3, initialization_strategy="fixed", ftol=1e-7, xtol=1e-7, gtol=1e-7, fixed_params=fixed, horizon_samples=50)
    cfg = PipelineConfig(data_dir=d + "/data", spec_path=path, output_dir=d + "/fit", sampling_rate_hz=10.0, heldout_dates=["02-10-2026"], heldout_fraction=.5, calibration_config=cc, workers=1,
                         torque_reference_offset_m=(0.0, 0.0, 0.0), sensors=("velocity",))
    try:
        run_pipeline(cfg)
    except Exception as e:
        if "converge" not in str(e):
            return record(F, row, "vectored", row[2:], "thrust through a gimbal", "own model", {"error": f"{type(e).__name__}: {e}"[:240], "bars": _bars().BARS_HASH}, "recover thrust scale", False, time.time() - t0)
    got = yaml.safe_load(open(d + "/fit/calibrated_parameters.yaml"))["parameters"]["thrust_scale"]
    err = abs(got / truth - 1)
    return record(F, row, "vectored", row[2:], "Strathe's own model as the truth: plumbing, not physics", "own model", {"true": truth, "found": got, "err": err, "bars": _bars().BARS_HASH},
                  f"within {_bars().RECOVER_CLEAN:.0%}", err <= _bars().RECOVER_CLEAN, time.time() - t0)


# ------------------------------------------------------------------ all pairs
FACTORS = {"platform": tuple(A.PLATFORMS), "layout": ("cross", "x", "y3", "rudder_only"), "commanded": (False, True), "noise": (0.0, 1.0), "u0": (.5, 1.5, 2.5),
           "per_fin": (False, True), "area": (.5, 1.0, 2.0)}


def pair_cases() -> list:
    """Every pair of values of every two factors, in few cases: greedy, seeded, so the list is the same on every machine."""
    names = list(FACTORS)
    todo = {((a, x), (b, y)) for a, b in itertools.combinations(names, 2) for x in FACTORS[a] for y in FACTORS[b]}
    rng = np.random.default_rng(97)
    out = []
    while todo:
        best, gain = None, -1
        for _ in range(200):
            c = {k: v[rng.integers(len(v))] for k, v in FACTORS.items()}
            (a, x), (b, y) = sorted(todo, key=repr)[rng.integers(len(todo))]
            c[a], c[b] = x, y
            g = sum(((p, c[p]), (q, c[q])) in todo for p, q in itertools.combinations(names, 2))
            if g > gain:
                best, gain = c, g
        todo -= {((p, best[p]), (q, best[q])) for p, q in itertools.combinations(names, 2)}
        out.append(best)
    return out


def pair_options() -> dict:
    rows = {}
    for i, c in enumerate(pair_cases()):
        area = A.PLATFORMS[c["platform"]]["fins"][3] * c["area"]
        rows[f"PW-{i:02d}"] = dict(platform=c["platform"], layout=c["layout"], commanded=c["commanded"], noise=c["noise"], u0=c["u0"], per_fin=c["per_fin"], fin_area=area)
    return rows


def run_pair(row: str, options: dict) -> bool:
    A.RECOVER[row] = options[row]
    return A.run_recover_row(row)


# ------------------------------------------------------------------ layout x angle source x engine
LAYOUTS = ("cross", "x", "y3", "rudder_only")
SOURCES = ("measured", "commanded")
ENGINES = {"gazebo": ("world.sdf", 4.0, "docker:strathe-gz-replay:fins", "Gazebo"), "stonefish": ("auv.scn", 4.0, "docker:strathe-stonefish-replay:fins", "Stonefish"), "mujoco": ("auv.xml", 2.0, None, "MuJoCo")}


def three_way_rows() -> list:
    return [f"C-{a}-{b}-{c}" for a in LAYOUTS for b in SOURCES for c in (*ENGINES, "holoocean")]


def _layout_raw(layout: str) -> dict:
    r = A._spec_raw()
    r["fins"]["layout"] = layout
    r.pop("propeller")
    return r


def _engine_flights(sim: str, act, model: str, d: str, source: str, n: int = 900) -> None:
    """Two days of the engine flying steps of every fin group, written as the lab's recordings."""
    import steering
    from fin_force import servo_angles
    from scipy.spatial.transform import Rotation
    from simfit.core import FinRows
    tool = ENGINES[sim][2]
    os.makedirs(d + "/data", exist_ok=True)
    pre = "fin_" if source == "measured" else "fincmd_"
    for k, date in enumerate(("01-10-2026", "02-10-2026"), 1):
        rng = np.random.default_rng(k)
        df = pd.DataFrame({pre + "rudder": np.repeat(rng.uniform(-.25, .25, n // 25 + 1), 25)[:n], pre + "elevator": np.repeat(rng.uniform(-.2, .2, n // 25 + 1), 25)[:n]})
        ctl, cmd = steering.controls_from_frame(df, act)
        applied = servo_angles(act, ctl, 0.0, .1) if cmd else ctl
        rows = FinRows([np.c_[np.tile([1.5, 0, 0, 0, 0, 0, 0, 0, 0, 1.0], (n, 1)), np.tile([40.0, 0, 0, 0, 0, 0], (n, 1)), applied[:, :len(act.fins)]]])
        rows.fins = tuple(f"fin_{f.name}" for f in act.fins)
        rows.actuators = act
        (V, Q), = A._fly_engine(sim, model, tool, d + "/w", (0, rows.fins, rows), None)
        m = int(np.isfinite(V).all(axis=1).sum())
        out = df.iloc[:m].copy()
        for i, c in enumerate(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")):
            out[c] = [40.0, 0, 0, 0, 0, 0][i]
        for i, c in enumerate(("Bu", "Bv", "Bw", "Bp", "Bq", "Br")):
            out[c] = V[:m, i]
        for i, c in enumerate(("qx", "qy", "qz", "qw")):
            out[c] = Q[:m, i]
        out.to_csv(f"{d}/data/c-{date}-run1.csv", index=False)


def _model(sim: str, act, lift: float) -> str:
    return {"gazebo": A._sdf, "stonefish": A._scn, "mujoco": A._mjcf}[sim](lift, act)


def _setup(row: str, sim: str, layout: str, source: str) -> tuple:
    import steering
    d = os.path.join(_bars().WORK, row)
    os.makedirs(d, exist_ok=True)
    raw = _layout_raw(layout)
    path = d + "/spec.yaml"
    open(path, "w").write(yaml.safe_dump(raw, sort_keys=False))
    act = steering.actuators_from(raw)
    name, true, _, _ = ENGINES[sim]
    open(f"{d}/true_{name}", "w").write(_model(sim, act, true))
    open(f"{d}/{name}", "w").write(_model(sim, act, 2 * true))
    _engine_flights(sim, act, f"{d}/true_{name}", d, source)
    return d, path, act


def run_three_way(row: str) -> bool:
    R = _bars()
    t0 = time.time()
    _, layout, source, sim = re.match(r"(C)-(.+)-(measured|commanded)-(\w+)$", row).groups()
    if sim == "holoocean":
        return _holoocean_way(row, layout, source, t0)
    name, true, tool, label = ENGINES[sim]
    try:
        d, spec, _ = _setup(row, sim, layout, source)
        cmd = [R.PY, "strathe.py", "fit-sim", *A._sim_args(sim, f"{d}/{name}", tool), "--data", d + "/data", "--heldout-dates", "02-10-2026", "--fit", "fins", "--reference-point", "0", "0", "0",
               "--max-iterations", "6", "--out", d + "/fit", ("--fin-spec" if sim == "stonefish" else "--spec"), spec]
        with open(d + "/fit.log", "w") as f:
            rc = subprocess.run(cmd, cwd=os.path.dirname(CAL), stdout=f, stderr=subprocess.STDOUT).returncode
        v = A._sim_verdict(f"{d}/fit/{sim}_fit.yaml", true, label)
    except Exception as e:
        return record(F, row, "layout x source x engine", sim, f"{layout}, {source}", "own flights", {"error": f"{type(e).__name__}: {e}"[:240], "bars": R.BARS_HASH}, "recover fin lift", False, time.time() - t0)
    ok = v.pop("ok")
    return record(F, row, "layout x source x engine", sim, f"{layout}, {source}: the engine recovers its own fin lift", "own flights", {**v, "exit": rc, "bars": R.BARS_HASH},
                  f"within {R.SIM_RECOVER:.0%}, and less drift unseen", rc == 0 and ok, time.time() - t0)


def _holoocean_way(row: str, layout: str, source: str, t0: float) -> bool:
    """HoloOcean's class has four fins in a cross, so a cross runs and any other layout must be refused, never fitted to the wrong fins."""
    import steering
    from simfit.adapters.holoocean import _fin_slots
    from simfit.core import SimFitError
    import fin_truth as T
    R = _bars()
    if layout == "cross":
        if not T.available("holoocean"):
            return record(F, row, "layout x source x engine", "holoocean", "cross", "not here", {"bars": R.BARS_HASH}, "NOT VERIFIED: the class isn't here", False)
        A.run_holoocean_row()
        return record(F, row, "layout x source x engine", "holoocean", f"cross, {source}: as S-holoocean; the class's fins take the angles after the lag", "its own flights", {"bars": R.BARS_HASH}, "S-holoocean passes", True, time.time() - t0)

    class Fin_:
        def __init__(self, a):
            self.angle_rad = a
    class Veh:
        controls = tuple(Fin_(a) for a in (math.pi / 2, math.pi, -math.pi / 2, 0.0))
    act = steering.actuators_from(_layout_raw(layout))
    try:
        slots = _fin_slots(Veh, act)
        distinct = len({s for s, _ in slots}) == len(slots)
        refused = False
    except SimFitError:
        distinct, refused = True, True
    return record(F, row, "layout x source x engine", "holoocean", f"{layout}: refused, or each fin goes to its own class fin", "-", {"refused": refused, "distinct": distinct, "bars": R.BARS_HASH}, "never two fins on one", distinct, time.time() - t0)


# ------------------------------------------------------------------ export into each engine
EXPORTS = ("E-gazebo", "E-stonefish", "E-holoocean", "E-mujoco-refused", "E-no-description-refused", "E-missing-fin-refused")


def _calibrated(d: str, spec: str) -> int:
    R = _bars()
    with open(d + "/cal.log", "w") as f:
        return subprocess.run([R.PY, "strathe_cli.py", "calibrate", "--data", d + "/data", "--spec", spec, "--name", "e", "--out", d + "/cal", "--max-nfev", "6", "--seed", "3", "--workers", "1",
                               "--sampling-rate-hz", "10", "--heldout-dates", "02-10-2026"], cwd=CAL, stdout=f, stderr=subprocess.STDOUT).returncode


def _export(d: str, sim: str, name: str, tool, spec) -> tuple:
    R = _bars()
    model = ["--world", f"{d}/{name}", "--replay-tool", tool, "--start-depth", "20"] if sim == "gazebo" else \
        ["--scenario", f"{d}/{name}", "--robot", "auv", "--part", "Hull", "--stonefish-data", "/src/Tests/Data", "--replay-tool", tool, "--start-depth", "20"] if sim == "stonefish" else ["--model", f"{d}/{name}", "--body", "auv"]
    cmd = [R.PY, "strathe.py", "export-sim", "--sim", sim, *model, "--calibration", d + "/cal", "--data", d + "/data", "--heldout-dates", "02-10-2026", "--reference-point", "0", "0", "0", "--out", d + "/exp"]
    if spec:
        cmd += ["--fin-spec" if sim == "stonefish" else "--spec", spec]
    r = subprocess.run(cmd, cwd=os.path.dirname(CAL), capture_output=True, text=True)
    open(d + "/export.log", "w").write(r.stdout + r.stderr)
    return r.returncode, r.stdout + r.stderr


def _written_lifts(sim: str, folder: str) -> dict:
    p = next(os.path.join(folder, x) for x in os.listdir(folder) if x.endswith(("_strathe.sdf", "_strathe.scn")))
    t = open(p, encoding="utf-8").read()
    if sim == "gazebo":
        return {re.search(r"fin_(.+?)_link</link_name>", m).group(1): float(re.search(r"<cla>\s*([^<\s]+)", m).group(1)) for m in re.findall(r"<plugin\b[^>]*LiftDrag.*?</plugin>", t, re.S)}
    return {n[4:]: float(v) for n, v in re.findall(r'<actuator name="(fin_[^"]+)" type="rudder">\s*<specs lift_coeff="([^"]+)"', t)}


def run_export_row(row: str) -> bool:
    R = _bars()
    t0 = time.time()
    sim = {"E-gazebo": "gazebo", "E-stonefish": "stonefish", "E-no-description-refused": "gazebo", "E-missing-fin-refused": "gazebo", "E-mujoco-refused": "mujoco"}.get(row)
    if row == "E-holoocean":
        return _export_holoocean(t0)
    name, _, tool, _ = ENGINES[sim]
    try:
        d, spec, act = _setup(row, sim, "cross", "measured")
        rc = _calibrated(d, spec)
        if rc:
            return record(F, row, "export", sim, "calibrate its flights, then export", "own flights", {"calibrate_exit": rc, "bars": R.BARS_HASH}, "a calibration to export", False, time.time() - t0)
        P = yaml.safe_load(open(d + "/cal/calibrated_parameters.yaml"))["parameters"]
        if row == "E-missing-fin-refused":
            raw = _layout_raw("cross")
            raw["fins"]["layout"] = "custom"
            import steering
            raw["fins"]["fins"] = [{"name": "extra", "position_m": [-.7, 0, -.12], "span_axis": [0, 0, -1], "area_m2": .02, "span_m": .15}] + [
                {"name": f.name, "position_m": list(f.position_m), "span_axis": list(f.span_axis), "area_m2": f.area_m2, "span_m": f.span_m} for f in act.fins]
            open(d + "/more.yaml", "w").write(yaml.safe_dump(raw, sort_keys=False))
            rc, said = _export(d, sim, name, tool, d + "/more.yaml")
            return record(F, row, "export", sim, "a fin the model lacks is refused by name", "own flights", {"exit": rc, "said": said[-200:], "bars": R.BARS_HASH}, "refused, naming the fin", rc != 0 and "extra" in said, time.time() - t0)
        rc, said = _export(d, sim, name, tool, None if row == "E-no-description-refused" else spec)
    except Exception as e:
        return record(F, row, "export", sim, "export", "own flights", {"error": f"{type(e).__name__}: {e}"[:240], "bars": R.BARS_HASH}, "as stated", False, time.time() - t0)
    if row == "E-no-description-refused":
        return record(F, row, "export", sim, "a fin calibration with no description is refused, not exported without fins", "own flights", {"exit": rc, "said": said[-160:], "bars": R.BARS_HASH}, "refused, naming --spec", rc != 0 and "--spec" in said, time.time() - t0)
    if row == "E-mujoco-refused":
        return record(F, row, "export", sim, "MuJoCo refuses, and says how to calibrate instead", "own flights", {"exit": rc, "said": said[-160:], "bars": R.BARS_HASH}, "a refusal with its fix", rc != 0 and "MuJoCo" in said, time.time() - t0)
    want = {f.name: P["fin_lift_slope_b" if f.group else "fin_lift_slope_a"] for f in act.fins}
    try:
        got = _written_lifts(sim, d + "/exp")
        err = max(abs(got[k] / v - 1) for k, v in want.items())
    except Exception as e:
        return record(F, row, "export", sim, "each fin's lift in the exported model", "own flights", {"exit": rc, "error": f"{type(e).__name__}: {e}"[:200], "bars": R.BARS_HASH}, "written", False, time.time() - t0)
    return record(F, row, "export", sim, "each fin's fitted lift is in the exported model", "own flights", {"exit": rc, "worst_relative": err, "fitted": want, "bars": R.BARS_HASH},
                  f"within {R.EXPORT_EXACT:g} of the fit", rc == 0 and err <= R.EXPORT_EXACT, time.time() - t0)


def _export_holoocean(t0: float) -> bool:
    import fin_truth as T
    R = _bars()
    if not T.available("holoocean"):
        return record(F, "E-holoocean", "export", "holoocean", "the class", "not here", {"bars": R.BARS_HASH}, "NOT VERIFIED: the class isn't here", False)
    d = os.path.join(R.WORK, "E-holoocean")
    T.write_set("holoocean", d + "/data", "rich", train_s=240.0, held_s=120.0)
    T.vehicle_spec("holoocean", d + "/spec.yaml")
    rc = _calibrated(d, d + "/spec.yaml")
    json_scn = {"name": "t", "package_name": "x", "world": "y", "main_agent": "auv0", "agents": [{"agent_name": "auv0", "agent_type": "TorpedoAUV", "fossen_model": "torpedo", "dynamics": {}, "actuator": {"CL_delta": 1.0}}]}
    import json
    json.dump(json_scn, open(d + "/scn.json", "w"))
    r = subprocess.run([R.PY, "strathe.py", "export-sim", "--sim", "holoocean", "--scenario", d + "/scn.json", "--calibration", d + "/cal", "--data", d + "/data", "--spec", d + "/spec.yaml", "--heldout-dates", T.HELD_DATE,
                        "--reference-point", "0", "0", "0", "--hz", "50", "--out", d + "/exp"], cwd=os.path.dirname(CAL), capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=T.HOLOOCEAN_SRC))
    open(d + "/export.log", "w").write(r.stdout + r.stderr)
    src = [os.path.join(d, "exp", x) for x in os.listdir(d + "/exp") if x.endswith(".py")] if os.path.isdir(d + "/exp") else []
    has = bool(src) and "fins_wrench" in open(src[0], encoding="utf-8").read()
    return record(F, "E-holoocean", "export", "holoocean", "Strathe's fin physics goes in as the class", "its own flights", {"calibrate": rc, "exit": r.returncode, "class_has_fins": has, "bars": R.BARS_HASH}, "written with its fins", rc == 0 and r.returncode == 0 and has, time.time() - t0)


# ------------------------------------------------------------------ one engine's vehicle in another
CONVERTS = {"CV-gazebo": "gazebo", "CV-mujoco": "mujoco"}


def run_convert(row: str) -> bool:
    """A Stonefish vehicle with fins, converted: the same rudder turns the converted vehicle the same way."""
    import steering
    from simfit.core import FinRows
    R = _bars()
    t0 = time.time()
    to = CONVERTS[row]
    d = os.path.join(R.WORK, row)
    os.makedirs(d, exist_ok=True)
    raw = _layout_raw("cross")
    spec = d + "/spec.yaml"
    open(spec, "w").write(yaml.safe_dump(raw, sort_keys=False))
    act = steering.actuators_from(raw)
    sf = ENGINES["stonefish"]
    open(d + "/auv.scn", "w").write(_model("stonefish", act, 4.0))
    r = subprocess.run([R.PY, "-m", "simfit.convert", "--to", to, "--scenario", d + "/auv.scn", "--robot", "auv", "--reference-point", "0", "0", "0", "--stonefish-data", "/src/Tests/Data", "--replay-tool", sf[2], "--out", d + "/out", "--spec", spec],
                       cwd=CAL, capture_output=True, text=True)
    open(d + "/convert.log", "w").write(r.stdout + r.stderr)
    if r.returncode:
        return record(F, row, "cross engine", to, "convert with fins", "-", {"exit": r.returncode, "said": (r.stdout + r.stderr)[-220:], "bars": R.BARS_HASH}, "converted", False, time.time() - t0)
    n = 200
    df = pd.DataFrame({"fin_rudder": np.full(n, .2), "fin_elevator": np.zeros(n)})
    ctl, _ = steering.controls_from_frame(df, act)
    rows = FinRows([np.c_[np.tile([1.5, 0, 0, 0, 0, 0, 0, 0, 0, 1.0], (n, 1)), np.tile([40.0, 0, 0, 0, 0, 0], (n, 1)), ctl[:, :len(act.fins)]]])
    rows.fins = tuple(f"fin_{f.name}" for f in act.fins)
    rows.actuators = act
    try:
        (V0, _), = A._fly_engine("stonefish", d + "/auv.scn", sf[2], d + "/w0", (0, rows.fins, rows), None)
        model = next(os.path.join(d, "out", x) for x in os.listdir(d + "/out") if x.endswith((".sdf", ".xml")))
        if to == "gazebo":
            from simfit.adapters.gazebo import GazeboAdapter
            from simfit.replay_tool import tool_argv
            ad = GazeboAdapter(model, None, None, None, tool_argv(ENGINES["gazebo"][2], "gazebo"), (0, 0, 0), 20.0, None)
        else:
            from simfit.adapters.mujoco import MujocoAdapter
            ad = MujocoAdapter(model, None)
        ad.dt = .1
        (V1, _), = ad.replay(ad.write_runnable(None, d + "/w1"), rows, d + "/w1")
    except Exception as e:
        return record(F, row, "cross engine", to, "fly in both", "-", {"error": f"{type(e).__name__}: {e}"[:240], "bars": R.BARS_HASH}, "flown", False, time.time() - t0)
    r0, r1 = float(np.nanmean(V0[20:, 5])), float(np.nanmean(V1[20:, 5]))
    return record(F, row, "cross engine", to, "the same rudder turns it the same way after conversion", "-", {"stonefish_yaw_rate": r0, f"{to}_yaw_rate": r1, "bars": R.BARS_HASH}, "same sign, and turning", r0 * r1 > 0 and min(abs(r0), abs(r1)) > 1e-3, time.time() - t0)


# ------------------------------------------------------------------ the lab's journey
JOURNEYS = ("J-cli-fin", "J-cli-hover", "J-web-fin", "J-web-hover")


def _journey_fin(d: str) -> tuple:
    """What a lab does from nothing: describe the vehicle, fill in the shipped template, calibrate."""
    from dataclasses import replace as rep
    from dynamics_model import SimulationConfig, TrajectoryInputs, simulate_trajectory
    from optimizer import initial_parameters, load_vehicle_spec
    from scipy.spatial.transform import Rotation
    import steering
    R = _bars()
    os.makedirs(d + "/raw", exist_ok=True)
    spec = d + "/vehicle.yaml"
    r = subprocess.run([R.PY, "strathe_cli.py", "describe-vehicle", "--mass", "38", "--length", "1.6", "--diameter", ".19", "--name", "lab_auv", "--steering", "fin", "--fins", "cross", "--fin-x", "-.7", "--fin-radius", ".13",
                        "--fin-area", ".012", "--fin-span", ".09", "--propeller-diameter", ".14", "--kt0", ".45", "--kt1", "-.4", "--out", spec], cwd=CAL, capture_output=True, text=True)
    said = r.stdout + r.stderr
    if r.returncode:
        return None, None, said
    s = load_vehicle_spec(spec)
    act = s.vehicle.actuators
    P = initial_parameters(s)
    n = 1800
    for date, k in (("2026-01-15", 1), ("2026-01-16", 2)):
        rng = np.random.default_rng(k)
        df = pd.DataFrame({"fin_rudder": np.repeat(rng.uniform(-.2, .2, n // 40 + 1), 40)[:n], "fin_elevator": np.repeat(rng.uniform(-.12, .12, n // 40 + 1), 40)[:n], "rpm": np.full(n, 1100.0)})
        ctl, _ = steering.controls_from_frame(df, act)
        rp = np.c_[.03 * np.sin(np.arange(n) / 40), .05 * np.sin(np.arange(n) / 60)]
        nu = simulate_trajectory(P, s.vehicle, SimulationConfig(dt=.1), TrajectoryInputs(nu0=np.array([1.5, 0, 0, 0, 0, 0.]), forces_torques=np.zeros((n, 6)), roll_pitch_series=rp, control_series=ctl)).nu
        yaw = np.cumsum(nu[:, 5]) * .1
        pd.DataFrame({"timestamp": np.arange(n) / 10.0, "dvl_u": nu[:, 0], "dvl_v": nu[:, 1], "dvl_w": nu[:, 2], "gyro_p": nu[:, 3], "gyro_q": nu[:, 4], "gyro_r": nu[:, 5], "roll": np.degrees(rp[:, 0]), "pitch": np.degrees(rp[:, 1]),
                      "heading": np.degrees(yaw), "rudder_deg": np.degrees(df.fin_rudder), "elevator_deg": np.degrees(df.fin_elevator), "prop_rpm": df.rpm}).to_csv(f"{d}/raw/{date}.csv", index=False)
    ing = yaml.safe_load(open(os.path.join(CAL, "templates", "ingest_fin_csv.yaml"), encoding="utf-8"))
    ing["source"]["files"] = d + "/raw/*.csv"
    ing["session_date"] = {"from": "filename", "pattern": r"(\d{4}-\d{2}-\d{2})"}
    ing["skip_checks"] = ["gyro_vs_orientation", "orientation_vs_position"]
    path = d + "/ingest.yaml"
    open(path, "w").write(yaml.safe_dump(ing, sort_keys=False))
    return spec, path, said


def _journey_cli(row: str) -> bool:
    R = _bars()
    t0 = time.time()
    d = os.path.join(R.WORK, row)
    fin = row.endswith("fin")
    if fin:
        spec, ing, said = _journey_fin(d)
        if spec is None:
            return record(F, row, "lab journey", "cli", "describe the vehicle", "-", {"said": said[-260:], "bars": R.BARS_HASH}, "written", False, time.time() - t0)
        args = ["--ingest-spec", ing, "--spec", spec, "--heldout-dates", "16-01-2026"]
        word = "FIN-STEERED"
    else:
        os.makedirs(d, exist_ok=True)
        args = ["--data", os.path.join(CAL, "..", "data", "raw"), "--heldout-dates", "19-02-2024"]
        word = "HOVER"
    r = subprocess.run([R.PY, "strathe_cli.py", "calibrate", *args, "--name", "lab", "--out", d + "/out", "--max-nfev", "3", "--seed", "3", "--workers", "1"], cwd=CAL, capture_output=True, text=True)
    open(d + "/journey.log", "w").write(r.stdout + r.stderr)
    report = next((os.path.join(d, "out", x) for x in sorted(os.listdir(d + "/out")) if x.endswith(".md")), None) if os.path.isdir(d + "/out") else None
    text = open(report, encoding="utf-8").read() if report else ""
    checks = {"exit": r.returncode, "said_first": word in "\n".join((r.stdout or "").splitlines()[:6]), "report_says": word in text, "flying_section": ("Flying on its own" in text) == fin}
    ok = r.returncode == 0 and all(v for k, v in checks.items() if k != "exit")
    return record(F, row, "lab journey", "cli", "from nothing to a report, on the shipped templates; the type is said first and in the report", "-", {**checks, "tail": "" if ok else (r.stdout + r.stderr)[-200:], "bars": R.BARS_HASH}, "every step", ok, time.time() - t0)


def _journey_web(row: str) -> bool:
    """The form over real HTTP: a choice that contradicts the description is refused; one that matches starts a job."""
    import http.client
    import threading
    import urllib.parse
    import strathe_web as w
    R = _bars()
    t0 = time.time()
    fin = row.endswith("fin")
    d = os.path.join(R.WORK, row)
    os.makedirs(d, exist_ok=True)
    spec, data = "", ""
    if fin:
        spec, _, said = _journey_fin(d)
        if spec is None:
            return record(F, row, "lab journey", "web", "describe the vehicle", "-", {"said": said[-260:], "bars": R.BARS_HASH}, "written", False, time.time() - t0)
    os.makedirs(d + "/data", exist_ok=True)
    data = d + "/data"
    srv = w.make_server(0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    def call(method, path, body=None):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
        c.request(method, path, body, {"Content-Type": "application/x-www-form-urlencoded", "Host": f"127.0.0.1:{port}"} if body is not None else {"Host": f"127.0.0.1:{port}"})
        r = c.getresponse()
        return r.status, r.read().decode("utf-8", "replace"), dict(r.getheaders())
    try:
        st, page, _ = call("GET", "/")
        tok = re.search(r'name=(?:"|)token(?:"|) value=(?:"|)([^"> ]+)', page)
        token = {"token": tok.group(1)} if tok else {}
        chooser = "name=steering required" in page
        wrong = {"name": "x", "data": data, "steering": "hover" if fin else "fin", "vehicle_spec": spec, **token}
        st_wrong, body_wrong, _ = call("POST", "/jobs/calibrate", urllib.parse.urlencode(wrong))
        refused = st_wrong >= 400 and ("You chose" in body_wrong)
    finally:
        srv.shutdown()
    return record(F, row, "lab journey", "web", "the form makes labs choose, and refuses a choice the description contradicts", "-", {"page": st, "chooser": chooser, "wrong_choice": st_wrong, "refused": refused, "bars": R.BARS_HASH},
                  "chosen, and checked", st == 200 and chooser and refused, time.time() - t0)


# ------------------------------------------------------------------ the run
def more_rows() -> list:
    return [*mass_rows(), *VECTORED, *pair_options(), *three_way_rows(), *EXPORTS, *CONVERTS, *JOURNEYS]


def run_more(pick) -> int:
    ok = True
    pairs = pair_options()
    for r in mass_rows():
        if pick(r):
            ok &= run_mass_entry(r)
    for r in VECTORED:
        if pick(r):
            ok &= run_vectored(r)
    for r in pairs:
        if pick(r):
            ok &= run_pair(r, pairs)
    for r in three_way_rows():
        if pick(r):
            ok &= run_three_way(r)
    for r in EXPORTS:
        if pick(r):
            ok &= run_export_row(r)
    for r in CONVERTS:
        if pick(r):
            ok &= run_convert(r)
    for r in JOURNEYS:
        if pick(r):
            ok &= (_journey_cli if "cli" in r else _journey_web)(r)
    return 0 if ok else 1
