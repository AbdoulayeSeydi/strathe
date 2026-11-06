"""stress/fin_rows_97.py -- run_97's rows that need no independent truth: identities, refusals, faults, fuzz, mutants, every layout, and real logs."""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import time
from dataclasses import replace

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, CAL)
sys.path.insert(0, os.path.join(CAL, "tests"))
sys.path.insert(0, HERE)
from common import record  # noqa: E402

F = "97"


def _bars():
    import run_97
    return run_97


# ------------------------------------------------------------------ identities
def _spec_raw():
    return yaml.safe_load(open(os.path.join(CAL, "templates", "vehicle_spec_fin_template.yaml"), encoding="utf-8"))


def _params(**kw):
    from dynamics_model import CalibrationParameters
    base = dict(Ieff_roll=2, Ieff_pitch=9, Ieff_yaw=9, Xu=-5, Yv=-6, Zw=-7, Kp=-1, Mq=-3, Nr=-3, Xuu=-5, Yvv=-6, Zww=-7, Kpp=-.5, Mqq=-.5, Nrr=-.5,
                Xu_dot=3, Yv_dot=30, Zw_dot=30, net_buoyancy_z=.2, cg_cb_offset_z=.02, fin_lift_slope_a=3.0, fin_lift_slope_b=2.5, fin_inflow_a=.8,
                fin_inflow_b=.6, fin_drag0=.02)
    base.update(kw)
    return CalibrationParameters(**base)


def _vehicle(layout="cross"):
    import steering
    from dynamics_model import VehicleConstants
    r = _spec_raw()
    r["fins"]["layout"] = layout
    r.pop("propeller")
    return VehicleConstants(mass_kg=30.0, actuators=steering.actuators_from(r))


def f_paths_agree() -> tuple:
    from dynamics_model import compute_acceleration, compute_acceleration_from_rotation_matrix
    from fast_dynamics import FastDynamics
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(0)
    worst = 0.0
    for layout in ("cross", "x", "y3", "rudder_only"):
        V = _vehicle(layout)
        n = len(V.actuators.fins)
        P = _params(Yr_dot=-.1, Zq_dot=.1)
        for _ in range(200):
            v, tau, d, cb = np.r_[1.5, rng.normal(size=5) * .15], rng.normal(size=6) * 4, list(rng.uniform(-.2, .2, n)), rng.normal(size=3) * .2
            roll, pitch = rng.normal(size=2) * .2
            ref = compute_acceleration(P, V, v, tau, roll, pitch, cb, d)[0]
            fast = np.array(FastDynamics(P, V).acceleration_trig(list(v), list(tau), math.sin(roll), math.cos(roll), math.sin(pitch), math.cos(pitch), list(cb), d))
            R = Rotation.from_euler("xyz", [roll, pitch, rng.normal()]).as_matrix()
            rot = compute_acceleration_from_rotation_matrix(P, V, v, tau, R, R @ cb, d)[0]
            s = max(np.abs(ref).max(), 1e-9)
            worst = max(worst, np.abs(fast - ref).max() / s, np.abs(rot - ref).max() / s)
    return worst <= _bars().PATHS_AGREE, {"worst_relative": worst}


def f_oracle() -> tuple:
    import fin_oracle as o
    from fin_force import Actuators, actuator_wrench, prepare
    V = _vehicle("cross")
    rng = np.random.default_rng(1)
    act = Actuators(fins=tuple(replace(f, stall_rad=10.0) for f in V.actuators.fins))
    P = _params()
    worst = 0.0
    for _ in range(500):
        v, d = np.r_[rng.uniform(.5, 2.5), rng.normal(size=2) * .15, rng.normal(size=3) * .2], rng.uniform(-.2, .2, 4)
        got = np.array(actuator_wrench(act, prepare(act, P), P, list(v), list(d), 0, 1, 0, 1))
        want = sum(o.fin_wrench(f.position_m, f.span_axis, f.area_m2, f.span_m, P.fin_lift_slope_b if f.group else P.fin_lift_slope_a,
                                P.fin_inflow_b if f.group else P.fin_inflow_a, P.fin_drag0, f.oswald, x, v) for f, x in zip(act.fins, d))
        worst = max(worst, np.abs(got - want).max() / max(np.abs(want).max(), 1e-9))
    return worst <= _bars().PATHS_AGREE, {"worst_relative": worst}


def f_speed() -> tuple:
    from dynamics_model import VehicleConstants
    from fast_dynamics import FastDynamics
    P = _params()
    hover, fins = FastDynamics(P, VehicleConstants(mass_kg=30.0)).acceleration_trig, FastDynamics(P, _vehicle("cross")).acceleration_trig
    v, tau, c = [1.5, .02, .01, .01, .02, .03], [10, 0, 0, 0, 0, 0], [.1, -.1, .05, -.05]
    best = [9.0, 9.0]
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(100000):
            hover(v, tau, .01, 1, .05, .999)
        a = time.perf_counter() - t
        t = time.perf_counter()
        for _ in range(100000):
            fins(v, tau, .01, 1, .05, .999, None, c)
        b = time.perf_counter() - t
        best = [min(best[0], a), min(best[1], b)]
    return best[1] / best[0] <= _bars().SPEED_FOUR_FINS, {"four_fins_over_none": best[1] / best[0]}


def f_mass_fuzz() -> tuple:
    from dynamics_model import MASS_COUPLINGS, NonPhysicalMass, mass_matrix, VehicleConstants
    rng = np.random.default_rng(2)
    bad = good = 0
    for _ in range(3000):
        p = _params(**{n: float(rng.normal() * 10 ** rng.uniform(-2, 2)) for _, _, n in MASS_COUPLINGS})
        try:
            M = mass_matrix(p, VehicleConstants(30.0))
            assert np.linalg.eigvalsh(M)[0] > 0
            good += 1
        except NonPhysicalMass:
            bad += 1
    return bad + good == 3000 and bad > 0 and good > 0, {"physical": good, "refused": bad}


def f_properties() -> tuple:
    from fin_force import actuator_wrench, prepare
    V = _vehicle("cross")
    act = V.actuators
    P = _params()
    w = lambda v, c: np.array(actuator_wrench(act, prepare(act, P), P, list(v), list(c), 0, 1, 0, 1))
    d = [.1, .1, .05, .05]
    a, b = w([1, 0, 0, 0, 0, 0], d), w([2, 0, 0, 0, 0, 0], d)
    checks = {"speed_squared": bool(np.allclose(b, 4 * a, rtol=1e-12)), "still_is_zero": not w(np.zeros(6), d).any(),
              "finite_from_behind": bool(np.isfinite(w([-1.5, .3, .2, 0, 0, 0], d)).all()), "finite_at_large_angle": bool(np.isfinite(w([.2, 2, 2, 1, 1, 1], [.5] * 4)).all())}
    return all(checks.values()), checks


IDENTITIES = {"F-paths": (f_paths_agree, "fast, reference, rotation physics agree, every layout"), "F-oracle": (f_oracle, "against the independent vector fin"),
              "F-speed": (f_speed, "four fins' cost per step"), "F-mass-fuzz": (f_mass_fuzz, "random mass entries: physical or refused by name"),
              "F-properties": (f_properties, "speed squared, zero at rest, finite in any flow")}


# ------------------------------------------------------------------ faults: each must be refused, or warned about
def _ingest_dir(tmp, n=600, fins=None, flip=1.0, stuck=False, nan=False, swap=False, deg_as_rad=False, speed=1.5, top=None):
    t = np.arange(n) / 10
    rud = 10 * np.sin(t / 3)
    ele = 4 * np.sin(t / 2 + 1)
    if stuck:
        rud = np.full(n, 3.0)
    r = np.radians(rud) * .5 * flip
    q = np.radians(ele) * .4
    df = pd.DataFrame({"t": t, "u": speed, "v": 0., "w": 0., "p": 0., "q": np.r_[0, q[:-1]], "r": np.r_[0, r[:-1]], "roll": 0., "pitch": 0.,
                       "yaw": np.degrees(np.cumsum(r) * .1), "rud": ele if swap else rud, "elev": rud if swap else ele, "rpm": 900.})
    if nan:
        df.loc[100:200, "rud"] = np.nan
    d = os.path.join(tmp, "raw")
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, "r.csv"), index=False)
    s = {"vehicle": "x", "source": {"files": os.path.join(d, "*.csv")}, "time": {"column": "t", "unit": "s"},
         "session_date": {"from": "constant", "value": "2026-01-15"},
         "body_velocity": {"columns": dict(u="u", v="v", w="w", p="p", q="q", r="r"), "linear_unit": "m/s", "angular_unit": "rad/s"},
         "orientation": {"euler": {"roll": "roll", "pitch": "pitch", "yaw": "yaw", "unit": "deg"}},
         "fins": fins or {"source": "measured", "channels": {"rudder": "rud", "elevator": "elev"}, "unit": "rad" if deg_as_rad else "deg"},
         "propeller": {"column": "rpm"}}
    s.update(top or {})
    p = os.path.join(tmp, "s.yaml")
    open(p, "w").write(yaml.safe_dump(s))
    return p


def _expect(word: str, **kw) -> tuple:
    import data_ingestion as di
    with tempfile.TemporaryDirectory() as tmp:
        try:
            s = di.ingest(di.load_ingest_spec(_ingest_dir(tmp, **kw)), os.path.join(tmp, "o"))
        except di.IngestError as e:
            return word in str(e), {"refused": str(e)[:140]}
        text = json.dumps(s)
        return word in text, {"not_refused": True, "found": word in text}


FAULTS = {"X-wrong-sign": (dict(flip=-1.0), "looks reversed", "a reversed rudder"),
          "X-degrees-as-radians": (dict(deg_as_rad=True), "check its unit", "degrees read as radians"),
          "X-stuck-fin": (dict(stuck=True), "barely moved", "a stuck rudder can't be sign-checked, and says so"),
          "X-too-slow": (dict(speed=.05), "barely moved", "too slow for the fins to count"),
          "X-nan-dropout": (dict(nan=True), "NaN", "a dropout in the fin channel"),
          "X-swapped-channels": (dict(swap=True, flip=1.0), "correlation", "rudder and elevator swapped: reported, weakly"),
          "X-percent-no-scale": (dict(fins={"source": "measured", "channels": {"rudder": "rud"}, "unit": "percent"}), "full_scale_deg", "percent with no scale"),
          "X-pwm-no-scale": (dict(fins={"source": "measured", "channels": {"rudder": "rud"}, "unit": "pwm_us"}), "pwm_us_per_deg", "servo PWM with no scale"),
          "X-unknown-key": (dict(top={"fin_block": {}}), "fin_block", "a misspelled key"),
          "X-servo-as-thruster": (dict(top={"wrench": {"thrusters": {"pwm_columns": ["rudder_servo"], "preset": "bluerov2_heavy"}}}), "looks like a fin", "a fin in the thruster columns")}


def refusals() -> dict:
    import steering

    def check(fn, word):
        try:
            fn()
        except steering.SteeringError as e:
            return word in str(e), {"refused": str(e)[:140]}
        return False, {"not_refused": True}
    h, fv = {"vehicle": {}}, {"vehicle": {"steering": "fin"}}
    return {"X-hover-with-fin-logs": (lambda: check(lambda: steering.check_data(h, ["fin_rudder"]), "vehicle description is hover"), "fin logs, hover description"),
            "X-fin-with-hover-logs": (lambda: check(lambda: steering.check_data(fv, ["Fx"]), "no fin or propeller channels"), "hover logs, fin description"),
            "X-glider-calibration": (lambda: check(lambda: steering.check_data({"vehicle": {"steering": "glider"}}, ["fin_rudder"]), "fin-check"), "a glider"),
            "X-no-speed-source": (lambda: check(lambda: steering.check_speed_source(fv, {"u": [], "w": ["depth.d"]}), "speed source"), "fins with depth only")}


# ------------------------------------------------------------------ fuzz
def z_specs(seeds=400) -> tuple:
    """Random fin descriptions and logs: a result or a named refusal, never a traceback."""
    import steering
    bad, ok, refused = [], 0, 0
    rng = np.random.default_rng(7)
    for s in range(seeds):
        r = _spec_raw()
        r["fins"] = {"layout": str(rng.choice(["cross", "x", "y3", "rudder_only", "custom", "star"])), "x_m": float(rng.normal()), "radius_m": float(abs(rng.normal())),
                     "area_m2": float(rng.choice([0, .01, -1, 1])), "span_m": float(rng.choice([0, .1, 2])), "fins": [] if rng.random() < .5 else [
                         {"name": "a", "position_m": [0, 0, 0], "span_axis": [0, 0, float(rng.choice([0, 1]))], "area_m2": .01, "span_m": .1}]}
        r["vehicle"]["steering"] = str(rng.choice(["hover", "fin", "vectored", "hybrid", "glider", "boat"]))
        if rng.random() < .3:
            r.pop("propeller")
        try:
            a = steering.actuators_from(r)
            steering.banner(r)
            if a is not None and a.fins:
                n = 20
                df = pd.DataFrame({"fin_rudder": rng.normal(size=n), "rpm": rng.normal(size=n)})
                steering.controls_from_frame(df, a)
            ok += 1
        except steering.SteeringError:
            refused += 1
        except Exception as e:   # a traceback is a failure
            bad.append(f"{s}: {type(e).__name__}: {e}"[:120])
    return not bad, {"ok": ok, "refused": refused, "tracebacks": bad[:5]}


def z_physics(n=3000) -> tuple:
    from fin_force import actuator_wrench, prepare
    rng = np.random.default_rng(8)
    V = _vehicle("y3")
    act = V.actuators
    P = _params()
    fins = prepare(act, P)
    bad = 0
    for _ in range(n):
        v = list(rng.normal(size=6) * 10 ** rng.uniform(-6, 2))
        c = list(rng.normal(size=3) * 10 ** rng.uniform(-3, 1))
        bad += not np.all(np.isfinite(actuator_wrench(act, fins, P, v, c, 0, 1, 0, 1)))
    return bad == 0, {"cases": n, "not_finite": bad}


# ------------------------------------------------------------------ mutants: each fault must be caught by a named check
MUTANTS = {"M-flip-moment": ("fin_force.py", "N += x * fy - y * fx", "N -= x * fy - y * fx", "tests/test_fin_force.py"),
           "M-drop-rate": ("fin_force.py", "lu = u + b * z - r * y", "lu = u", "tests/test_fin_force.py"),
           "M-swap-lift-drag": ("fin_force.py", "fn = q * U * (CL * vc - CD * vn)", "fn = q * U * (CD * vc - CL * vn)", "tests/test_fin_force.py"),
           "M-drop-inflow": ("fin_force.py", "alpha = c[i] + bias - inflow * phi", "alpha = c[i] + bias", "tests/test_fin_force.py"),
           "M-no-current-forward": ("fast_dynamics.py", "tau, sr, cr, sp, cp, None, ctl)", "tau, sr, cr, sp, cp, None, None)", "tests/test_fin_force.py"),
           "M-no-window-slice": ("objective.py", "controls=None if run.controls is None else run.controls[a:b])\n            parts.append", "controls=run.controls)\n            parts.append", "tests/test_fin_pipeline.py"),
           "M-reverse-mixing": ("steering.py", "m / big", "-m / big", "tests/test_steering.py"),
           "M-ignore-servo": ("objective.py", "if getattr(run, \"servo\", None) is not None:", "if False:", "tests/test_fin_pipeline.py"),
           "M-mass-transpose": ("dynamics_model.py", "M[i, j] = M[j, i] = getattr(params, name)", "M[i, j] = getattr(params, name)", "tests/test_mass_couplings.py"),
           "M-gazebo-sign": ("simfit/adapters/gazebo.py", "Ma[i, j] = Ma[j, i] = flip[i] * flip[j] * getattr(params, name)", "Ma[i, j] = Ma[j, i] = getattr(params, name)", "tests/test_mass_couplings.py"),
           "M-no-fin-check": ("data_ingestion.py", "if best < FIN_REVERSED_CORRELATION:", "if False:", "tests/test_steering.py"),
           "M-no-guard": ("fin_fit.py", 'if g is not None and g["swing_deg"] < MIN_FIN_SWING_DEG:', "if False:", "tests/test_fin_calibration.py")}


def run_mutant(row: str) -> bool:
    """Break the code in a scratch copy of the calibration folder; the named tests must fail."""
    import shutil
    import subprocess
    t0 = time.time()
    f, old, new, tests = MUTANTS[row]
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "calibration")
        shutil.copytree(CAL, dst, ignore=shutil.ignore_patterns("results", "mlruns", "*.zip", "__pycache__", "calibration_output*", "*_policies", "stress"), symlinks=True)
        p = os.path.join(dst, f)
        t = open(p, encoding="utf-8").read()
        applied = old in t
        open(p, "w", encoding="utf-8").write(t.replace(old, new, 1))
        r = subprocess.run([sys.executable, "-m", "pytest", tests, "-q", "-x", "-p", "no:cacheprovider"], cwd=dst, capture_output=True, text=True)
    caught = r.returncode != 0
    return record(F, row, "mutant", f, f"{old[:50]} broken", "tests", {"applied": applied, "caught": caught, "bars": _bars().BARS_HASH}, f"{tests} must fail", applied and caught, time.time() - t0)


def fast_rows() -> list:
    return [*IDENTITIES, *FAULTS, "X-hover-with-fin-logs", "X-fin-with-hover-logs", "X-glider-calibration", "X-no-speed-source", "Z-specs", "Z-physics", *MUTANTS,
            *fast_rows_more()]


def run_fast(pick) -> int:
    h = _bars().BARS_HASH
    ok = True
    for row, (fn, what) in IDENTITIES.items():
        if pick(row):
            t0 = time.time()
            p, n = fn()
            ok &= record(F, row, "identity", "-", what, "computed", {**n, "bars": h}, "the bar in run_97.py", p, time.time() - t0)
    for row, (kw, word, what) in FAULTS.items():
        if pick(row):
            p, n = _expect(word, **kw)
            ok &= record(F, row, "fault", "-", what, "synthetic log", {**n, "bars": h}, f"says: {word}", p)
    for row, (fn, what) in refusals().items():
        if pick(row):
            p, n = fn()
            ok &= record(F, row, "refusal", "-", what, "-", {**n, "bars": h}, "a refusal with its fix", p)
    for row, fn, what in (("Z-specs", z_specs, "random descriptions"), ("Z-physics", z_physics, "random states and angles")):
        if pick(row):
            p, n = fn()
            ok &= record(F, row, "fuzz", "-", what, "random", {**n, "bars": h}, "no traceback, nothing non-finite", p)
    for row in MUTANTS:
        if pick(row):
            ok &= run_mutant(row)
    for f in FORMATS:
        if pick(f"LF-{f}"):
            ok &= run_format(f)
    for c in HOVER_BASE:
        if pick(f"H-{c}"):
            ok &= run_hover(c)
    return 0 if ok else 1


# ------------------------------------------------------------------ every layout and platform, through the whole product
# Strathe's own model is the truth here, so these prove the plumbing, not the physics, and say so. Sizes are published or typical.
PLATFORMS = {
    "remus100": dict(mass=38.0, length=1.6, diameter=.19, fins=("cross", -.7, .13, .012, .09)),
    "remus600": dict(mass=240.0, length=3.25, diameter=.324, fins=("y3", -1.4, .25, .05, .2)),
    "iver": dict(mass=39.0, length=1.8, diameter=.147, fins=("cross", -.8, .12, .01, .08)),
    "lauv": dict(mass=18.0, length=1.1, diameter=.15, fins=("cross", -.5, .11, .008, .07)),
    "hugin": dict(mass=650.0, length=5.0, diameter=.75, fins=("cross", -2.3, .5, .15, .35)),
    "autosub": dict(mass=2800.0, length=7.0, diameter=.9, fins=("cross", -3.2, .6, .3, .5)),
    "tethys": dict(mass=110.0, length=2.3, diameter=.31, fins=("rudder_only", -1.0, .2, .03, .15), moving_mass=8.0, buoyancy=True),
    "xtail": dict(mass=60.0, length=2.0, diameter=.2, fins=("x", -.9, .15, .02, .1)),
}
RECOVER = {f"P-platform-{k}": dict(platform=k) for k in PLATFORMS}
RECOVER.update({f"P-layout-{k}": dict(platform="remus100", layout=k) for k in ("cross", "x", "y3", "rudder_only")})
RECOVER.update({"P-source-commanded": dict(platform="remus100", commanded=True), "P-noise-grade": dict(platform="remus100", noise=1.0),
                "P-per-fin-channels": dict(platform="remus100", per_fin=True), "P-hybrid": dict(platform="remus100", steering="hybrid"),
                "P-fin-far-forward": dict(platform="remus100", fin_x=-.3), "P-fin-small": dict(platform="remus100", fin_area=.004),
                "P-fin-large": dict(platform="remus100", fin_area=.03), "P-speed-low": dict(platform="remus100", u0=.5),
                "P-speed-high": dict(platform="remus100", u0=2.5)})


def _platform_spec(path: str, o: dict) -> dict:
    import vehicle_describe as vd
    s = PLATFORMS[o["platform"]]
    layout, x, r, area, span = s["fins"]
    layout = o.get("layout", layout)
    spec, _ = vd.describe_hull(s["mass"], s["length"], diameter=s["diameter"], shape="torpedo", name=o["platform"], steering=o.get("steering", "fin"),
                               fins={"layout": layout, "x_m": o.get("fin_x", x), "radius_m": r, "area_m2": o.get("fin_area", area), "span_m": span,
                                     "limit_deg": 40.0, "stall_deg": 30.0, "servo": {"rate_limit_deg_s": 0}})
    if s.get("moving_mass"):
        spec["moving_mass"] = {"mass_kg": s["moving_mass"]}
    if s.get("buoyancy"):
        spec["buoyancy_engine"] = True
    open(path, "w").write(yaml.safe_dump(spec, sort_keys=False))
    return spec


def _own_truth(spec_path: str, out: str, o: dict, truth_a: float, truth_b: float) -> None:
    from dataclasses import replace as rep
    import steering
    from dynamics_model import SimulationConfig, TrajectoryInputs, simulate_trajectory
    from fin_force import servo_angles
    from optimizer import initial_parameters, load_vehicle_spec
    from scipy.spatial.transform import Rotation
    s = load_vehicle_spec(spec_path)
    act = s.vehicle.actuators
    P = rep(initial_parameters(s), fin_lift_slope_a=truth_a, fin_lift_slope_b=truth_b)
    n, os_ = 500, os.path
    k = 0
    for date in ("01-10-2026", "02-10-2026"):
        for run in (1, 2):
            k += 1
            rng = np.random.default_rng(k)
            rud = np.repeat(rng.uniform(-.2, .2, n // 15 + 1), 15)[:n]
            ele = np.repeat(rng.uniform(-.12, .12, n // 15 + 1), 15)[:n]
            cols = {"fincmd_rudder" if o.get("commanded") else "fin_rudder": rud, "fincmd_elevator" if o.get("commanded") else "fin_elevator": ele}
            if act.moving_mass:
                cols["moving_mass"] = .02 * np.sin(np.arange(n) / 30)
            if act.buoyancy_engine:
                cols["buoyancy_change"] = np.sin(np.arange(n) / 40)
            df = pd.DataFrame(cols)
            ctl, cmd = steering.controls_from_frame(df, act)
            if o.get("per_fin"):
                for i, f in enumerate(act.fins):
                    df[f"fin_{f.name}"] = ctl[:, i]
                df = df.drop(columns=["fin_rudder", "fin_elevator"])
            applied = servo_angles(act, ctl, 0.0, .1) if cmd else ctl
            rp = np.c_[.03 * np.sin(np.arange(n) / 40), .05 * np.sin(np.arange(n) / 60)]
            u0 = o.get("u0", 1.5)
            F = np.zeros((n, 6))
            F[:, 0] = abs(P.Xuu) * u0 * u0 + abs(P.Xu) * u0
            nu = simulate_trajectory(P, s.vehicle, SimulationConfig(dt=.1), TrajectoryInputs(nu0=np.array([u0, 0, 0, 0, 0, 0.]), forces_torques=F, roll_pitch_series=rp, control_series=applied)).nu
            q = Rotation.from_euler("xyz", np.c_[rp, np.cumsum(nu[:, 5]) * .1]).as_quat()
            for i, c in enumerate(("Fx", "Fy", "Fz", "Tx", "Ty", "Tz")):
                df[c] = F[:, i]
            for i, c in enumerate(("Bu", "Bv", "Bw", "Bp", "Bq", "Br")):
                df[c] = nu[:, i] + (rng.normal(0, _bars().GRADE[c] * o.get("noise", 0), n) if o.get("noise") else 0)
            for i, c in enumerate(("qx", "qy", "qz", "qw")):
                df[c] = q[:, i]
            df.to_csv(f"{out}/p-{date}-run{run}.csv", index=False)


def run_recover_row(row: str) -> bool:
    from dataclasses import fields
    from optimizer import CalibrationConfig, initial_parameters, load_vehicle_spec
    from run_calibration import PipelineConfig, run_pipeline
    o = RECOVER[row]
    t0 = time.time()
    W = _bars().WORK
    d = os.path.join(W, row)
    os.makedirs(d + "/data", exist_ok=True)
    spec = d + "/spec.yaml"
    _platform_spec(spec, o)
    s = load_vehicle_spec(spec)
    start = initial_parameters(s)
    ta, tb = start.fin_lift_slope_a * 1.25, (start.fin_lift_slope_b or 3.0) * .8
    _own_truth(spec, d + "/data", o, ta, tb)
    free = {"fin_lift_slope_a", "fin_lift_slope_b"} & {n for n in start.param_names if s.flat_calibration_targets[n]["bounds"][0] < s.flat_calibration_targets[n]["bounds"][1]}
    fixed = {n: float(getattr(start, n)) for n in start.param_names if n not in free}
    fixed = {k: v for k, v in fixed.items() if s.flat_calibration_targets[k]["bounds"][0] < s.flat_calibration_targets[k]["bounds"][1]}
    cc = CalibrationConfig(max_nfev=15, seed=3, initialization_strategy="fixed", ftol=1e-7, xtol=1e-7, gtol=1e-7, fixed_params=fixed, horizon_samples=50)
    cfg = PipelineConfig(data_dir=d + "/data", spec_path=spec, output_dir=d + "/fit", sampling_rate_hz=10.0, heldout_dates=["02-10-2026"], heldout_fraction=.5,
                         calibration_config=cc, workers=1, torque_reference_offset_m=(0.0, 0.0, 0.0), sensors=("velocity",))
    try:
        run_pipeline(cfg)
    except Exception as e:
        if "converge" not in str(e):
            return record(F, row, "recover", o["platform"], "own model", str(o), {"error": str(e)[:200], "bars": _bars().BARS_HASH}, "recover fin lift", False, time.time() - t0)
    p = yaml.safe_load(open(d + "/fit/calibrated_parameters.yaml"))["parameters"]
    errs = {}
    if "fin_lift_slope_a" in free:
        errs["a"] = abs(p["fin_lift_slope_a"] / ta - 1)
    if "fin_lift_slope_b" in free:
        errs["b"] = abs(p["fin_lift_slope_b"] / tb - 1)
    bar = _bars().RECOVER_NOISY if o.get("noise") else _bars().RECOVER_CLEAN
    return record(F, row, "recover", o["platform"], "Strathe's own model as the truth: plumbing, not physics", str(o), {**errs, "bars": _bars().BARS_HASH},
                  f"fin lift within {bar:.0%}", bool(errs) and max(errs.values()) <= bar, time.time() - t0)


def recover_rows() -> list:
    return list(RECOVER)


def run_recover(pick) -> int:
    return 0 if all([run_recover_row(r) for r in RECOVER if pick(r)]) else 1


# ------------------------------------------------------------------ OOI's REMUS: real vehicles at sea
# Bars fixed before any fit ran on these logs. The logs are 1 Hz, the fins' full scale and the vehicle's sizes are estimates, and the
# propeller's scale is fitted, so these are looser than the flight bars, and flying must beat assuming it holds its course.
REAL_FLY = {"heading_deg": 10.0, "pitch_deg": 3.0, "depth_m": 2.0}
REAL = {"O-turns-same-vehicle": (("CP05MOAS-A6264_R00008", 4), ("CP05MOAS-A6264_R00006", 6), "fly", "train on the turning mission, fly another mission of the same vehicle"),
        "O-turns-other-vehicle": (("CP05MOAS-A6264_R00008", 4), ("CP05MOAS-A6263_R00009", 6), "fly", "train on one REMUS, fly the other"),
        "O-turns-years-later": (("CP05MOAS-A6264_R00008", 4), ("CP16MOAS-A6263_R00001", 6), "fly", "train in 2019, fly the other REMUS in 2024"),
        "O-survey-only": (("CP05MOAS-A6263_R00008", 4), ("CP05MOAS-A6263_R00009", 6), "warned", "survey lines only: must warn")}
PREPARED = os.path.expanduser("~/strathe_fin_data/ooi_prepared")


def remus600_spec(path: str) -> str:
    import vehicle_describe as vd
    spec, _ = vd.describe_hull(240.0, 3.25, diameter=.324, shape="torpedo", hull_lift=True, name="ooi_remus600", steering="fin",
                               propeller={"diameter_m": .3, "kt0": .02, "position_m": [-1.6, 0, 0]},
                               fins={"layout": "y3", "x_m": -1.4, "radius_m": .25, "area_m2": .05, "span_m": .2, "limit_deg": 20.0, "stall_deg": 20.0})
    spec["calibration_targets"]["fins"]["thrust_scale"] = {"unit": "-", "initial_guess": 1.0, "bounds": [.2, 5.0], "bounds_rationale": "its propeller curve isn't published",
                                                            "source": "estimated", "confidence": "assumed"}
    open(path, "w").write(yaml.safe_dump(spec, sort_keys=False))
    return path


def _longest(folder: str, n: int) -> list:
    """The recordings with the most turning, of a mission."""
    import glob
    fs = glob.glob(os.path.join(PREPARED, folder, "*.csv"))
    return sorted(fs, key=lambda f: -float(pd.read_csv(f)["Br"].abs().sum()))[:n]


def run_real_row(row: str) -> bool:
    import re
    import shutil
    import subprocess
    (train, nt), (held, nh), bar, what = REAL[row]
    R = _bars()
    t0 = time.time()
    if not os.path.isdir(os.path.join(PREPARED, train)):
        return record(F, row, "real", train, what, "not here", {"bars": R.BARS_HASH}, "the prepared OOI logs (stress/prepare_ooi_97.py)", False)
    d = os.path.join(R.WORK, row)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d + "/data")
    for f in _longest(train, nt) + _longest(held, nh):
        shutil.copy(f, d + "/data")
    dates = {re.search(r"(\d\d-\d\d-\d{4})", f).group(1) for f in _longest(held, nh)}
    spec = remus600_spec(d + "/spec.yaml")
    with open(d + "/fit.log", "w") as f:
        rc = subprocess.run([R.PY, "strathe_cli.py", "calibrate", "--data", d + "/data", "--spec", spec, "--name", row, "--out", d + "/fit", "--sampling-rate-hz", "1",
                             "--max-nfev", str(R.NFEV), "--seed", "7", "--workers", "2", "--heldout-dates", *sorted(dates), "--sensors", "velocity"],
                            cwd=CAL, stdout=f, stderr=subprocess.STDOUT).returncode
    secs = time.time() - t0
    vj = d + "/fit/validation/validation.json"
    if not os.path.exists(vj):
        return record(F, row, "real", train, what, held, {"exit": rc, "bars": R.BARS_HASH}, "a calibration and its validation", False, secs)
    v = json.load(open(vj))
    fr = v["fin_free_run"]
    c, h = fr["calibrated"], fr["straight_on"]
    n = {k: c[k]["median"] for k in REAL_FLY}
    n.update({f"{k}_hold": h[k]["median"] for k in REAL_FLY}, windows=fr["windows"], diverged=fr["diverged"]["calibrated"], exit=rc, bars=R.BARS_HASH,
             few=json.load(open(d + "/fit/metrics.json"))["fins"]["few_manoeuvres"])
    if bar == "warned":
        return record(F, row, "real", train, what, held, n, "too few manoeuvres flagged", bool(n["few"]), secs)
    ok = all(n[k] is not None and n[k] <= REAL_FLY[k] for k in REAL_FLY) and not n["diverged"]
    ok = ok and n["heading_deg"] <= n["heading_deg_hold"]
    return record(F, row, "real", train, what, held, n, f"30 s median within {REAL_FLY}, and heading better than holding course", ok, secs)


def run_fin_check_row() -> bool:
    import glob
    import fin_check
    R = _bars()
    out, ok = {}, True
    for d in sorted(glob.glob(os.path.join(PREPARED, "*"))):
        r = fin_check.check_folder(d, 1.0)["channels"]
        out[os.path.basename(d)] = {k: v.get("correlation") for k, v in r.items()}
        ok &= not any("problem" in v for v in r.values())
    return record(F, "O-fin-check", "real", "all missions", "fin-check reads each mission's fin signs as right", "OOI REMUS", {**out, "bars": R.BARS_HASH},
                  "no reversed fin on any mission", ok and bool(out))


def real_rows() -> list:
    return [*REAL, "O-fin-check"]


def run_real(pick) -> int:
    ok = all([run_real_row(r) for r in REAL if pick(r)])
    return 0 if (run_fin_check_row() if pick("O-fin-check") else True) and ok else 1


# ------------------------------------------------------------------ every log format: fin channels read as they were written
FORMATS = ("csv", "px4", "moos", "lcm", "ardupilot", "rosbag")


def _channels(n=400):
    t = np.arange(n) / 10.0
    return t, 10 * np.sin(t / 3), 4 * np.sin(t / 2 + 1)


def _lf_spec(fmt, files, top):
    s = {"vehicle": "x", "source": {"format": fmt, "files": files}, "resample_hz": 10.0, "session_date": {"from": "constant", "value": "2026-01-15"},
         "skip_checks": ["fin_direction", "gyro_vs_orientation", "orientation_vs_position"]}
    s.update(top)
    return s


def run_format(fmt: str) -> bool:
    import data_ingestion as di
    from format_writers import Topic, write_alog, write_lcm, write_ulog
    t0 = time.time()
    t, rud, ele = _channels()
    h = _bars().BARS_HASH
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "logs")
        os.makedirs(d)
        att = {"qx": 0 * t, "qy": 0 * t, "qz": 0 * t, "qw": 0 * t + 1}
        rates = {"p": 0 * t, "q": 0 * t, "r": .01 * np.sin(t / 3)}
        vel = {"u": 0 * t + 1.5, "v": 0 * t, "w": 0 * t}
        fins = {"rud": rud, "ele": ele, "rpm": 0 * t + 900}
        bv = {"linear_unit": "m/s", "angular_unit": "rad/s"}
        if fmt == "csv":
            df = pd.DataFrame({"t": t, **att, **rates, **vel, **fins})
            df.to_csv(os.path.join(d, "a.csv"), index=False)
            s = _lf_spec("csv", os.path.join(d, "*.csv"), {"time": {"column": "t", "unit": "s"}, "body_velocity": {"columns": dict(u="u", v="v", w="w", p="p", q="q", r="r"), **bv},
                         "orientation": {"quaternion": dict(x="qx", y="qy", z="qz", w="qw")}, "fins": {"source": "commanded", "channels": {"rudder": "rud", "elevator": "ele"}},
                         "propeller": {"column": "rpm"}})
            s.pop("resample_hz")
        elif fmt == "px4":
            p = os.path.join(d, "a.ulg")
            write_ulog(p, {"att": Topic(t, att), "gyr": Topic(t, rates), "vel": Topic(t, vel), "fin": Topic(t, fins)}, None)
            s = _lf_spec("px4", p, {"body_velocity": {"columns": dict(u="vel.u", v="vel.v", w="vel.w", p="gyr.p", q="gyr.q", r="gyr.r"), **bv},
                         "orientation": {"quaternion": dict(x="att.qx", y="att.qy", z="att.qz", w="att.qw")},
                         "fins": {"source": "commanded", "channels": {"rudder": "fin.rud", "elevator": "fin.ele"}}, "propeller": {"column": "fin.rpm"}})
        elif fmt == "moos":
            p = os.path.join(d, "a.alog")
            tm = t + 1.7e9
            write_alog(p, {"a": Topic(tm, {"QX": att["qx"], "QY": att["qy"], "QZ": att["qz"], "QW": att["qw"]}), "g": Topic(tm, {"GP": rates["p"], "GQ": rates["q"], "GR": rates["r"]}),
                           "v": Topic(tm, {"VU": vel["u"], "VV": vel["v"], "VW": vel["w"]}), "f": Topic(tm, {"DESIRED_RUDDER": rud, "DESIRED_ELEVATOR": ele, "DESIRED_THRUST": fins["rpm"]})}, logstart=tm[0] - 2.0, sources={})
            s = _lf_spec("moos", p, {"body_velocity": {"columns": dict(u="VU", v="VV", w="VW", p="GP", q="GQ", r="GR"), **bv}, "orientation": {"quaternion": dict(x="QX", y="QY", z="QZ", w="QW")},
                         "fins": {"source": "commanded", "channels": {"rudder": "DESIRED_RUDDER", "elevator": "DESIRED_ELEVATOR"}}, "propeller": {"column": "DESIRED_THRUST"}})
        elif fmt == "lcm":
            p = os.path.join(d, "lcmlog-a.00")
            write_lcm(p, os.path.join(d, "types"), {"ATT": Topic(t, att), "GYR": Topic(t, {"rp": rates["p"], "rq": rates["q"], "rr": rates["r"]}), "VEL": Topic(t, vel), "FIN": Topic(t, fins)},
                      with_utime=True, receive_delay_s=0.0)
            s = _lf_spec("lcm", p, {"body_velocity": {"columns": dict(u="VEL.u", v="VEL.v", w="VEL.w", p="GYR.rp", q="GYR.rq", r="GYR.rr"), **bv},
                         "orientation": {"quaternion": dict(x="ATT.qx", y="ATT.qy", z="ATT.qz", w="ATT.qw")},
                         "fins": {"source": "commanded", "channels": {"rudder": "FIN.rud", "elevator": "FIN.ele"}}, "propeller": {"column": "FIN.rpm"}})
            s["source"]["lcm_types"] = [os.path.join(d, "types", "*.lcm")]
        else:   # ArduPilot and ROS log servo outputs: the same fin channels as servo PWM
            n = len(t)
            d_ = pd.DataFrame({"Bu": vel["u"], "Bv": vel["v"], "Bw": vel["w"], "Bp": rates["p"], "Bq": rates["q"], "Br": rates["r"], **att,
                               **{f"pwm{i}": np.full(n, 1500.0) for i in range(1, 9)}})
            d_["pwm1"], d_["pwm2"] = np.round(1500 + 10 * rud), np.round(1500 + 10 * ele)
            name = "ingest_ardupilot_bin.yaml" if fmt == "ardupilot" else "ingest_rosbag_mavros.yaml"
            s = yaml.safe_load(open(os.path.join(CAL, "templates", name), encoding="utf-8"))
            cols = s["wrench"]["thrusters"]["pwm_columns"]
            s.pop("wrench")
            s["fins"] = {"source": "commanded", "unit": "pwm_us", "pwm_us_per_deg": 10.0, "channels": {"rudder": cols[0], "elevator": cols[1]}}
            s["propeller"] = {"column": cols[2]}
            s["skip_checks"] = ["fin_direction"] + [c for c in s.get("skip_checks", [])]
            if fmt == "ardupilot":
                from ardupilot_log_writer import write_bin
                open(os.path.join(d, "a.bin"), "wb").write(write_bin(os.path.join(d, "a.bin"), d_) or b"") if False else write_bin(os.path.join(d, "a.bin"), d_)
                s["source"]["files"] = os.path.join(d, "*.bin")
            else:
                from test_ingest_bags_and_thrusters import write_bag
                write_bag(os.path.join(d, "a.bag"), d_)
                s["source"]["files"] = os.path.join(d, "*.bag")
                tw = "/odom.twist.twist."
                s["body_velocity"] = {"columns": {"u": tw + "linear.x", "v": tw + "linear.y", "w": tw + "linear.z", "p": tw + "angular.x", "q": tw + "angular.y",
                                                  "r": tw + "angular.z"}, "linear_unit": "m/s", "angular_unit": "rad/s"}
                s["orientation"] = {"quaternion": {k: f"/odom.pose.pose.orientation.{k}" for k in "xyzw"}}
            s["session_date"] = {"from": "constant", "value": "2026-01-15"}
        path = os.path.join(tmp, "s.yaml")
        open(path, "w").write(yaml.safe_dump(s))
        try:
            di.ingest(di.load_ingest_spec(path), os.path.join(tmp, "o"))
            out = pd.read_csv(sorted(__import__("glob").glob(os.path.join(tmp, "o", "**", "*.csv"), recursive=True))[0])
        except Exception as e:
            return record(F, f"LF-{fmt}", "log format", fmt, "fin channels", "synthetic", {"error": f"{type(e).__name__}: {e}"[:220], "bars": h}, "fin channels read", False, time.time() - t0)
        col = "fincmd_rudder"
        err = float(np.abs(np.degrees(out[col].to_numpy()) - np.interp(np.arange(len(out)) / 10.0 + 0, t, rud)).max()) if col in out else None
        ok = err is not None and "rpm" in out and float(np.percentile(np.abs(np.degrees(out[col].to_numpy())), 99)) > 8
    return record(F, f"LF-{fmt}", "log format", fmt, "commanded fin angles and rpm through this format", "synthetic", {"max_rudder_deg_err": err, "bars": h}, "fin channels read", ok, time.time() - t0)


# ------------------------------------------------------------------ hover: nothing changed
HOVER_BASE = {"dfki_dvl": "83ebd2ff1ab04831", "dfki_depth_hold": "5b9df3cc316c1384", "kth_dvl": "1119c53e07d3346e", "uri_dvl": "ff01459e3f2ee140", "tank_dvl": "e5f025deab482b15"}
HOVER_NAMES = ("Ieff_roll Ieff_pitch Ieff_yaw Xu Yv Zw Kp Mq Nr Xuu Yvv Zww Kpp Mqq Nrr Xu_dot Yv_dot Zw_dot net_buoyancy_z cg_cb_offset_z "
               "Xq_dot Zq_dot thruster_time_constant_s cg_cb_offset_x Zuw Muw Muq").split()


def run_hover(case: str) -> bool:
    import hashlib
    import subprocess
    W = os.path.expanduser("~/strathe_sensor_data")
    uri = os.path.expanduser("~/strathe_uri_lab_v2/data/alpha_rise_spec.yaml")
    data, spec, held, flags = {"dfki_dvl": (f"{W}/dfki/v", "vehicle_spec.yaml", "02-01-2025", ["--sensors", "velocity"]),
                               "dfki_depth_hold": (f"{W}/dfki/d3", "vehicle_spec.yaml", "02-01-2025", ["--unseen", "hold"]),
                               "kth_dvl": (f"{W}/marinarium/k96/v", "vehicle_spec.yaml", "30-10-2025", ["--sensors", "velocity"]),
                               "uri_dvl": (f"{W}/real/uri_v", uri, "27-03-2026", ["--sensors", "velocity"]),
                               "tank_dvl": (f"{W}/real/rV", "vehicle_spec.yaml", "14-02-2024", ["--sensors", "velocity"])}[case]
    t0 = time.time()
    o = os.path.join(_bars().WORK, f"H-{case}")
    os.makedirs(os.path.dirname(o), exist_ok=True)
    with open(o + ".log", "w") as f:
        rc = subprocess.run([_bars().PY, "strathe_cli.py", "calibrate", "--data", data, "--spec", spec, "--name", case, "--out", o, "--max-nfev", "4", "--seed", "7", "--workers", "1",
                             "--heldout-dates", held, *flags], cwd=CAL, stdout=f, stderr=subprocess.STDOUT).returncode
    p = yaml.safe_load(open(o + "/calibrated_parameters.yaml"))["parameters"]
    z = {k: p[k] for k in HOVER_NAMES if p.get(k)}
    got = hashlib.sha256(json.dumps(z, sort_keys=True).encode()).hexdigest()[:16]
    return record(F, f"H-{case}", "hover", case, "a seeded real calibration, against the commit before fins", "real logs", {"hash": got, "base": HOVER_BASE[case], "exit": rc, "bars": _bars().BARS_HASH},
                  "identical parameters", got == HOVER_BASE[case], time.time() - t0)


# ------------------------------------------------------------------ every simulator: its own physics recovers its own fin lift
SIM_RECOVER = 0.15   # fit-sim from a fin lift twice the truth lands within this of it, and drifts less on unseen recordings


SIM_MODEL = {"mass": 30.0, "x": -.7, "area": .02, "lift": 3.0}


def _fin_set():
    from fin_force import Actuators, Fin
    x = SIM_MODEL["x"]
    return Actuators(fins=(Fin("top", (x, 0, -.12), (0, 0, -1), SIM_MODEL["area"], .15, group=0, mix=(0, 0, 1), stall_rad=.5, limit_rad=.5),
                           Fin("bottom", (x, 0, .12), (0, 0, 1), SIM_MODEL["area"], .15, group=0, mix=(0, 0, 1), stall_rad=.5, limit_rad=.5)), rho=1000.0)


def _sim_spec(path: str) -> str:
    r = _spec_raw()
    r["fins"] = {"layout": "custom", "fins": [{"name": f.name, "position_m": list(f.position_m), "span_axis": list(f.span_axis), "area_m2": f.area_m2, "span_m": f.span_m,
                                               "stall_deg": 28.0, "limit_deg": 28.0} for f in _fin_set().fins]}
    r.pop("propeller")
    open(path, "w").write(yaml.safe_dump(r, sort_keys=False))
    return path


def _sdf(lift: float) -> str:
    from simfit import fin_models as fm
    links, plugins = fm.sdf_fins(_fin_set(), "base_link", 1000.0)
    plugins = plugins.replace("<cla>3</cla>", f"<cla>{lift:g}</cla>")
    return f"""<?xml version="1.0"?>
<sdf version="1.11"><world name="water">
  <physics name="s" type="dart"><max_step_size>0.01</max_step_size><real_time_factor>0</real_time_factor></physics>
  <gravity>0 0 0</gravity>
  <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>
    <model name="vehicle">
      <link name="base_link"><inertial><pose>0 0 0 0 0 0</pose><mass>30</mass><inertia><ixx>1</ixx><iyy>10</iyy><izz>10</izz><ixy>0</ixy><ixz>0</ixz><iyz>0</iyz></inertia></inertial>
        <collision name="hull"><geometry><box><size>0.5 0.3 0.3</size></box></geometry></collision></link>
{links}
      <plugin filename="gz-sim-hydrodynamics-system" name="gz::sim::systems::Hydrodynamics"><link_name>base_link</link_name>
        <xU>-5</xU><yV>-20</yV><zW>-20</zW><kP>-1</kP><mQ>-8</mQ><nR>-8</nR><xUabsU>-10</xUabsU><yVabsV>-60</yVabsV><zWabsW>-60</zWabsW><kPabsP>-1</kPabsP><mQabsQ>-4</mQabsQ><nRabsR>-4</nRabsR>
      </plugin>
{plugins}
    </model>
</world></sdf>
"""


def _scn(lift: float) -> str:
    from simfit import fin_models as fm
    return f"""<?xml version="1.0"?>
<scenario>
 <environment><ned latitude="40.0" longitude="3.0"/><ocean><water density="1000.0" jerlov="0.25"/><current type="uniform"><velocity xyz="0.0 0.0 0.0"/></current></ocean>
  <atmosphere><sun azimuth="120.0" elevation="50.0"/></atmosphere></environment>
 <materials><material name="Neutral" density="1000.0" restitution="0.5"/><friction_table><friction material1="Neutral" material2="Neutral" static="0.5" dynamic="0.2"/></friction_table></materials>
 <looks><look name="gray" gray="0.3" roughness="0.4"/></looks>
 <robot name="auv" fixed="false" self_collisions="false">
  <base_link name="Hull" type="sphere" physics="submerged" buoyant="true"><dimensions radius="0.3"/><origin rpy="0.0 0.0 0.0" xyz="0.0 0.0 0.0"/><material name="Neutral"/><look name="gray"/></base_link>
{fm.stonefish_fins(_fin_set(), "Hull").replace('lift_coeff="3"', f'lift_coeff="{lift:g}"')}
  <world_transform rpy="0.0 0.0 0.0" xyz="0.0 0.0 20.0"/>
 </robot>
</scenario>
"""


def _mjcf(lift: float) -> str:
    from simfit import fin_models as fm
    body = fm.mjcf_fins(_fin_set()).replace('fluidcoef="0.5 0.25 1.5 2.0 0"', f'fluidcoef="0.5 0.25 1.5 {lift:g} 0"')
    return f"""<mujoco model="fins"><option timestep="0.002" density="1000" viscosity="0.0009" integrator="implicitfast" gravity="0 0 0"/>
  <worldbody><body name="auv" pos="0 0 -3"><freejoint/><inertial pos="0 0 0" mass="30" diaginertia="1 10 10"/>
      <geom name="hull" type="ellipsoid" size="0.6 0.15 0.15" fluidshape="ellipsoid" fluidcoef="0.5 0.25 1.5 0 0" mass="0" contype="0" conaffinity="0"/>
{body}
    </body></worldbody></mujoco>
"""


SIMS = {"S-gazebo": ("gazebo", "world.sdf", _sdf, 4.0, "docker:strathe-gz-replay:fins", "fin_cla"),
        "S-stonefish": ("stonefish", "auv.scn", _scn, 4.0, "docker:strathe-stonefish-replay:fins", "fin_lift"),
        "S-mujoco": ("mujoco", "auv.xml", _mjcf, 2.0, None, "fin_top_geom_lift")}


def _sim_args(sim: str, model: str, tool) -> list:
    if sim == "gazebo":
        return ["--sim", "gazebo", "--world", model, "--replay-tool", tool, "--start-depth", "20"]
    if sim == "stonefish":
        return ["--sim", "stonefish", "--scenario", model, "--robot", "auv", "--part", "Hull", "--stonefish-data", "/src/Tests/Data", "--replay-tool", tool, "--start-depth", "20"]
    return ["--sim", "mujoco", "--model", model, "--body", "auv"]


def _fly_engine(sim: str, model: str, tool, out_dir: str, ref_rows, spec_path: str):
    """The engine flying the fin angles with its true lift: recordings it made itself, in Strathe's layout."""
    from simfit.replay_tool import ReplayTool, tool_argv, write_windows
    t0, names, rows = ref_rows
    if sim == "mujoco":
        from simfit.adapters.mujoco import MujocoAdapter
        ad = MujocoAdapter(model, "auv")
        ad.dt = .1
        return ad.replay(ad.write_runnable(None, out_dir), rows, out_dir)
    extra = (["--model-name", "vehicle", "--depth", "20"] if sim == "gazebo" else [])
    if sim == "gazebo":
        rt = ReplayTool(tool_argv(tool, "gazebo"), (0, 0, 0), .1, extra=extra, name="Gazebo")
        return rt.run(model, rows, out_dir)
    from stonefish_fit import Replay
    rp = Replay(tool_argv(tool, "stonefish"), "/src/Tests/Data", "auv", (0, 0, 0))
    return rp.run(model, rows, out_dir)


def run_sim(row: str) -> bool:
    """Fly fin steps in the engine with its true lift, then fit-sim from twice that lift, on those recordings."""
    import subprocess
    from scipy.spatial.transform import Rotation
    from simfit.core import FinRows
    R = _bars()
    sim, name, write, true, tool, knob = SIMS[row]
    t0 = time.time()
    d = os.path.join(R.WORK, row)
    os.makedirs(d + "/data", exist_ok=True)
    spec = _sim_spec(d + "/spec.yaml")
    open(f"{d}/true_{name}", "w").write(write(true))
    open(f"{d}/{name}", "w").write(write(2 * true))
    fins = ("fin_top", "fin_bottom")
    k = 0
    try:
        for date in ("01-10-2026", "02-10-2026"):
            k += 1
            rng = np.random.default_rng(k)
            n = 600
            ang = np.repeat(rng.uniform(-.25, .25, n // 25 + 1), 25)[:n]
            rows = FinRows([np.c_[np.tile([1.5, 0, 0, 0, 0, 0, 0, 0, 0, 1.0], (n, 1)), np.tile([40.0, 0, 0, 0, 0, 0], (n, 1)), ang, -ang]])
            rows.fins = fins
            (V, Q), = _fly_engine(sim, f"{d}/true_{name}", tool, d + "/w", (0, fins, rows), spec)
            ok = np.isfinite(V).all(axis=1)
            m = int(ok.sum())
            q = Q[:m]
            rp = Rotation.from_quat(q).as_euler("xyz")
            df = pd.DataFrame({"Fx": 40.0, "Fy": 0.0, "Fz": 0.0, "Tx": 0.0, "Ty": 0.0, "Tz": 0.0, **{c: V[:m, i] for i, c in enumerate(("Bu", "Bv", "Bw", "Bp", "Bq", "Br"))},
                               **{c: q[:, i] for i, c in enumerate(("qx", "qy", "qz", "qw"))}, "fin_top": ang[:m], "fin_bottom": -ang[:m]})
            df.to_csv(f"{d}/data/s-{date}-run1.csv", index=False)
        args = _sim_args(sim, f"{d}/{name}", tool)
        cmd = [R.PY, "strathe.py", "fit-sim", *args, "--data", d + "/data", "--heldout-dates", "02-10-2026", "--fit", "fins", "--reference-point", "0", "0", "0",
               "--max-iterations", "6", "--out", d + "/fit", ("--fin-spec" if sim == "stonefish" else "--spec"), spec]
        with open(d + "/fit.log", "w") as f:
            rc = subprocess.run(cmd, cwd=os.path.dirname(CAL), stdout=f, stderr=subprocess.STDOUT).returncode
        text = open(d + "/fit.log").read()
        y = yaml.safe_load(open(next(os.path.join(d, "fit", x) for x in os.listdir(d + "/fit") if x.endswith("_fit.yaml"))))
        got = next((v for k_, v in (y.get("values") or y.get("fitted") or y).items() if k_ == knob), None)
    except Exception as e:
        return record(F, row, "simulator", sim, "the engine recovers its own fin lift", "own flights", {"error": f"{type(e).__name__}: {e}"[:260], "bars": R.BARS_HASH}, "recover fin lift", False, time.time() - t0)
    err = None if got is None else abs(float(got) / true - 1)
    return record(F, row, "simulator", sim, "from twice the lift, on recordings the engine flew itself", "own flights", {"true": true, "found": got, "err": err, "exit": rc, "bars": R.BARS_HASH},
                  f"within {SIM_RECOVER:.0%}", rc == 0 and err is not None and err <= SIM_RECOVER, time.time() - t0)


def run_holoocean_row() -> bool:
    import subprocess
    import fin_truth as T
    R = _bars()
    t0 = time.time()
    if not T.available("holoocean"):
        return record(F, "S-holoocean", "simulator", "holoocean", "the real Torpedo class", "not here", {"bars": R.BARS_HASH}, "NOT VERIFIED: the class isn't here", False)
    d = os.path.join(R.WORK, "S-holoocean")
    T.write_set("holoocean", d + "/data", "rich", train_s=240.0, held_s=120.0)
    T.vehicle_spec("holoocean", d + "/spec.yaml")
    json.dump({"name": "t", "package_name": "x", "world": "y", "main_agent": "auv0", "agents": [{"agent_name": "auv0", "agent_type": "TorpedoAUV", "fossen_model": "torpedo", "dynamics": {},
                                                                                                 "actuator": {"CL_delta": 1.0}}]}, open(d + "/scn.json", "w"))
    env = dict(os.environ, PYTHONPATH=T.HOLOOCEAN_SRC)
    with open(d + "/fit.log", "w") as f:
        rc = subprocess.run([R.PY, "strathe.py", "fit-sim", "--sim", "holoocean", "--scenario", d + "/scn.json", "--data", d + "/data", "--spec", d + "/spec.yaml", "--heldout-dates", T.HELD_DATE,
                             "--fit", "fins", "--reference-point", "0", "0", "0", "--hz", "50", "--max-iterations", "6", "--out", d + "/out"], cwd=os.path.dirname(CAL), stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    got = json.load(open(d + "/out/scn_calibrated.json"))["agents"][0]["actuator"]["CL_delta"] if os.path.exists(d + "/out/scn_calibrated.json") else None
    err = None if got is None else abs(got / .5 - 1)
    return record(F, "S-holoocean", "simulator", "holoocean", "the real Torpedo class, from twice the fin lift", "its own flights", {"true": .5, "found": got, "err": err, "exit": rc, "bars": R.BARS_HASH},
                  f"within {SIM_RECOVER:.0%}", rc == 0 and err is not None and err <= SIM_RECOVER, time.time() - t0)


def run_external_row() -> bool:
    import subprocess
    R = _bars()
    t0 = time.time()
    tool = os.path.join(CAL, "simulators", "example", "example_replay.py")
    r = subprocess.run([R.PY, tool, "--features"], capture_output=True, text=True)
    return record(F, "S-external", "simulator", "external", "the example tool says it takes fins (tests/test_simfit_fins.py runs it)", "-", {"features": r.stdout.strip()[:80], "bars": R.BARS_HASH}, '"fins": true', '"fins": true' in r.stdout, time.time() - t0)


def run_refused_row() -> bool:
    from simfit.core import SimFitError, FinRows
    from simfit.replay_tool import ReplayTool
    t = ReplayTool(["unused"], (0, 0, 0))
    t._fins = False
    rows = FinRows([np.zeros((2, 17))])
    rows.fins = ("fin_a",)
    try:
        t.run("m", rows, tempfile.mkdtemp())
        ok = False
    except SimFitError as e:
        ok = "doesn't take fin angles" in str(e)
    return record(F, "S-refused", "simulator", "old tool", "a tool that doesn't take fins is refused", "-", {"bars": _bars().BARS_HASH}, "a refusal with its fix", ok)


def run_sims(pick) -> int:
    ok = True
    for r in SIMS:
        if pick(r):
            ok &= run_sim(r)
    for r, fn in (("S-holoocean", run_holoocean_row), ("S-external", run_external_row), ("S-refused", run_refused_row)):
        if pick(r):
            ok &= fn()
    return 0 if ok else 1


def fast_rows_more() -> list:
    return [f"LF-{f}" for f in FORMATS] + [f"H-{c}" for c in HOVER_BASE]


def sim_rows() -> list:
    return ["S-holoocean", "S-gazebo", "S-stonefish", "S-mujoco", "S-external", "S-refused"]
