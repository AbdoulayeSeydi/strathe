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
    return [*IDENTITIES, *FAULTS, "X-hover-with-fin-logs", "X-fin-with-hover-logs", "X-glider-calibration", "X-no-speed-source", "Z-specs", "Z-physics", *MUTANTS]


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
