"""
fin_fit.py -- what a calibration knows about a fin-steered vehicle before and after the fit (PROJECT_STATUS.md #97, FINS.md).

From geometry: where a fin's lift slope should be, and the off-diagonal mass its fins and centre of gravity make. Those mass entries
are worked out and held: logs rarely separate them, and fitting all of them would look thorough and be false.

From the logs: the range of speed, fin angle and turning the training recordings cover, and how many real manoeuvres they hold. A fit
is only known to hold inside that range. Trained on straight survey lines, a fit can predict more survey lines well while its
rudder value is far off (a REMUS test: 39%; still 27% with the inflow held at its true value, and 127% held at a wrong guess, so
it isn't held). The report says what the logs covered, and a group of fins that barely moved keeps its geometry value.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from dynamics_model import MASS_COUPLINGS
from fin_force import Actuators, Fin

COUPLING_NAMES = {(i, j): n for i, j, n in MASS_COUPLINGS}
MIN_FIN_SWING_DEG = 2.0        # a group of fins moved less than this (standard deviation, at speed) tells nothing about its lift
MIN_SPEED_MPS = 0.3            # under this the fins have no authority: those samples say nothing about them
MIN_FIN_MANOEUVRES = 20        # fewer swings of a group of fins from one side to the other, and its value isn't pinned down


def lift_slope(aspect_ratio: float) -> float:
    """Lift per radian of a low-aspect-ratio fin (Helmbold's form, as Fossen's)."""
    return math.pi * aspect_ratio / (1 + math.sqrt(1 + (aspect_ratio / 2) ** 2))


def _target(unit: str, guess: float, lo: float, hi: float, why: str) -> dict:
    return {"unit": unit, "initial_guess": round(guess, 4), "bounds": [round(lo, 4), round(hi, 4)], "bounds_rationale": why,
            "source": "fin geometry", "confidence": "estimated"}


def fin_targets(act: Actuators) -> dict:
    """calibration_targets.fins: each group's lift slope around its geometry value, and what's held."""
    out = {}
    for suffix, group in (("a", 0), ("b", 1)):
        fins = [f for f in act.fins if f.group == group]
        if not fins:
            continue
        s = float(np.mean([lift_slope(f.span_m ** 2 / f.area_m2) for f in fins]))
        out[f"fin_lift_slope_{suffix}"] = _target("per rad", s, .5 * s, 1.5 * s, "lift slope from aspect ratio, x0.5 to x1.5")
        out[f"fin_inflow_{suffix}"] = _target("-", INFLOW_GUESS, 0.0, 1.2, "the hull straightens the flow reaching the fins")
    return out


INFLOW_GUESS = 0.7


def held_couplings(fins: tuple, rho: float, mass_kg: float, cg_below_cb_m: float = 0.0) -> dict:
    """{entry: value}: the off-diagonal mass from each fin, as a flat plate moving through its normal (strip theory), and the centre
    of gravity's rigid-body terms. Entries are the matrix's own."""
    M = np.zeros((6, 6))
    for f in fins:
        chord = f.area_m2 / f.span_m
        mf = rho * math.pi * (chord / 2) ** 2 * f.span_m
        s, c = np.array(f.span_axis), np.array(f.chord_axis)
        n = np.cross(s, c)
        g = np.r_[n, np.cross(np.array(f.position_m), n)]
        M += mf * np.outer(g, g)
    M[0, 4] += mass_kg * cg_below_cb_m
    M[1, 3] -= mass_kg * cg_below_cb_m
    return {name: round(float(M[i, j]), 6) for (i, j), name in COUPLING_NAMES.items() if abs(M[i, j]) > 1e-6}


def coupling_targets(values: dict) -> dict:
    return {n: {"unit": "kg or kg*m", "initial_guess": v, "bounds": [v, v], "bounds_rationale": "held: worked out from the fins and "
                "the centre of gravity, because logs rarely separate these", "source": "strip theory", "confidence": "estimated"}
            for n, v in values.items()}


def _moving(run, act: Actuators) -> np.ndarray:
    """Samples at speed: from the DVL, or the propeller's rpm with no DVL."""
    u = np.asarray(run.real_nu[:, 0], dtype=float)
    if np.isfinite(u).mean() > .5:
        return np.nan_to_num(u) > MIN_SPEED_MPS
    if act.propeller is not None:
        return run.controls[:, len(act.fins)] > 100.0
    return np.ones(len(u), dtype=bool)


def manoeuvres(a: np.ndarray) -> int:
    """How many times an angle swings from clearly one side of its mean to clearly the other."""
    if len(a) < 3 or float(np.std(a)) == 0:
        return 0
    x = (a - np.mean(a)) / np.std(a)
    side = np.where(x > .5, 1, np.where(x < -.5, -1, 0))
    side = side[side != 0]
    return int(np.sum(side[1:] != side[:-1]))


def group_angle(run, idx: list) -> np.ndarray:
    """A group of fins' common angle: each fin signed by the first's, averaged."""
    c = run.controls[:, idx]
    sign = np.sign(np.sum(c * c[:, :1], axis=0))
    sign[sign == 0] = 1
    return np.mean(c * sign, axis=1)


def envelope(runs: list, act: Actuators) -> dict:
    """What the training recordings cover, at speed: speed, each group's fin angle and manoeuvres, and the turn and pitch rates."""
    if not act.fins:
        return {}
    masks = [_moving(r, act) for r in runs]
    u = np.concatenate([np.nan_to_num(r.real_nu[m, 0]) for r, m in zip(runs, masks)])
    u = u[u > 0]
    out = {"speed_mps": [round(float(x), 3) for x in np.percentile(u, [5, 95])] if len(u) else None,
           "seconds_at_speed": round(float(sum(m.sum() * r.dt for r, m in zip(runs, masks))), 1), "groups": {}}
    for group, word in ((0, "steering"), (1, "pitching")):
        idx = [i for i, f in enumerate(act.fins) if f.group == group]
        if not idx:
            continue
        parts = [group_angle(r, idx)[m] for r, m in zip(runs, masks)]
        a = np.concatenate(parts) if parts else np.zeros(0)
        out["groups"][word] = {"swing_deg": round(float(np.degrees(np.std(a))), 2) if len(a) else 0.0,
                               "angle_deg": [round(float(np.degrees(x)), 2) for x in np.percentile(a, [1, 99])] if len(a) else None,
                               "manoeuvres": int(sum(manoeuvres(p) for p in parts))}
    r = np.concatenate([x.real_nu[m, 5] for x, m in zip(runs, masks)])
    q = np.concatenate([x.real_nu[m, 4] for x, m in zip(runs, masks)])
    out["turn_rate_rad_s"] = round(float(np.percentile(np.abs(r), 99)), 4) if len(r) else 0.0
    out["pitch_rate_rad_s"] = round(float(np.percentile(np.abs(q), 99)), 4) if len(q) else 0.0
    return out


def held_groups(env: dict) -> dict:
    """{parameter word: why} for groups of fins that barely moved: they keep their geometry values."""
    out = {}
    for word, suffix in (("steering", "a"), ("pitching", "b")):
        g = (env.get("groups") or {}).get(word)
        if g is not None and g["swing_deg"] < MIN_FIN_SWING_DEG:
            out[suffix] = f"the {word} fins moved only {g['swing_deg']:.1f}° at speed (under {MIN_FIN_SWING_DEG:g}°)"
    return out


def few_manoeuvres(env: dict) -> list:
    return [w for w, g in (env.get("groups") or {}).items() if g["manoeuvres"] < MIN_FIN_MANOEUVRES
            and g["swing_deg"] >= MIN_FIN_SWING_DEG]


# A fin vehicle's fit replays short windows, not whole recordings (#97). Many fin vehicles are unstable with their fins held, and a real one is flown by its
# autopilot: replaying a 900 s recording on logged fins sends the true values of Prestero's REMUS far from the log (cost 5.9 against 0.6 for a wrong, nearly stable
# vehicle), so a whole-recording fit is rewarded for inventing stability. At 2 s the truth wins (0.16 against 0.36), and the fit found his growth rate, 0.76 s doubling
# against 0.69. A stable vehicle lost nothing: Fossen's REMUS flew 30 s to 1.9 degrees, from 2.7. --prediction-horizon-s overrides it.
FIN_WINDOW_S = 2.0
FIN_MIN_WINDOW_SAMPLES = 5   # a slow log's window still needs a few samples to say anything


# Windows alone aren't enough. Every window feeds the logged attitude back in, so no fit sees what its own pitch does: Prestero's 2 s fit beat his true values tenfold on
# its training windows and still missed pitch by 4.2 degrees after 5 s of flying itself. So a second stage goes on from the first with flights in the objective: from
# real states the model flies itself, on its own attitude, and its roll, pitch and heading are scored against the log. On his unseen recordings that took 5 s pitch to
# 1.05 degrees, under the published values' 1.74, and held 30 s under his autopilot to 1.7 degrees, 1.5 degrees and 0.24 m at each flight's worst moment.
FLIGHT_S = 10.0              # 5 s reached 1.87 degrees, 10 s 1.05: longer is what pins the unstable mode
FLIGHT_SAMPLE_S = 0.5
FLIGHT_MIN_SAMPLES = 5
FLIGHT_MAX_NFEV = 25         # it starts from a fit, so it has refining to do, not searching
FLIGHT_LOST = 10.0           # a flight that diverged, in the residual's scaled units


@dataclass(frozen=True)
class Flights:
    """One recording's flights: where each starts, how long, every how many samples it's scored, and the logged attitude."""
    starts: tuple
    samples: int
    every: int
    logged: np.ndarray       # (N, 3) roll, pitch, yaw
    scale: np.ndarray        # (3,) what assuming the attitude holds gets wrong over a flight, rad


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def plan_flights(train_runs: list) -> int:
    """Put a Flights on each training recording long enough to have one; objective.residuals_for_run then scores them. Returns how many flights."""
    rate = train_runs[0].sample_rate_hz
    n = max(FLIGHT_MIN_SAMPLES, int(round(FLIGHT_S * rate))) + 1
    every = max(1, int(round(FLIGHT_SAMPLE_S * rate)))
    plans, hold = [], []
    for run in train_runs:
        if run.orientation_quat is None or run.n_samples < n:
            plans.append(None)
            continue
        logged = Rotation.from_quat(run.orientation_quat).as_euler("xyz")
        starts = tuple(range(0, run.n_samples - n + 1, n - 1))
        hold += [_wrap(logged[a + n - 1] - logged[a]) for a in starts]
        plans.append((starts, logged))
    if not hold:
        return 0
    scale = np.maximum(np.sqrt(np.mean(np.square(hold), axis=0)), 1e-3)
    count = 0
    for run, p in zip(train_runs, plans):
        if p is not None:
            run.__dict__["_flights"] = Flights(p[0], n, every, p[1], scale)
            count += len(p[0])
    return count


def clear_flights(train_runs: list) -> None:
    for run in train_runs:
        run.__dict__.pop("_flights", None)


def flights_flown(params, run, vehicle, plan: Flights, starts) -> np.ndarray:
    """These flights' attitude errors through their length, in units of plan.scale."""
    from validation import free_run
    n, k = plan.samples, plan.every
    width = len(range(k, n, k)) * 3
    out = []
    for a in starts:
        got = free_run(params, vehicle, run, a, n)
        if got is None:
            out.append(np.full(width, FLIGHT_LOST))
        else:
            out.append(np.nan_to_num(_wrap(got[k::k] - plan.logged[a + k:a + n:k]) / plan.scale).ravel())
    return np.concatenate(out) if out else np.zeros(0)


def flight_weight(plan: Flights) -> float:
    """So a recording's flights count as one unit per axis, however many there are."""
    return math.sqrt(len(plan.starts) * len(range(plan.every, plan.samples, plan.every)))


def flight_residuals(params, run, vehicle, plan: Flights) -> np.ndarray:
    return flights_flown(params, run, vehicle, plan, plan.starts) / flight_weight(plan)


def window_samples(train_runs: list) -> int:
    """A fin vehicle's default prediction window, in samples."""
    return max(FIN_MIN_WINDOW_SAMPLES, int(round(FIN_WINDOW_S * train_runs[0].sample_rate_hz)))


def setup(train_runs: list, act: Actuators, default_params, already_fixed: dict) -> dict:
    """What the fit holds and what the report says, before the fit."""
    env = envelope(train_runs, act)
    held = {}
    why = held_groups(env)
    for suffix in why:
        for name in (f"fin_lift_slope_{suffix}", f"fin_inflow_{suffix}"):
            if name not in (already_fixed or {}):
                held[name] = float(getattr(default_params, name))
    return {"envelope": env, "held": held, "held_why": why, "few_manoeuvres": few_manoeuvres(env),
            "plain": envelope_words(env)}


def envelope_words(env: dict) -> list:
    if not env or not env.get("speed_mps"):
        return ["The training recordings have no samples at speed: nothing here says how the fins behave."]
    w = [f"Trained at {env['speed_mps'][0]:.2f} to {env['speed_mps'][1]:.2f} m/s, turning up to {env['turn_rate_rad_s']:.3f} rad/s."]
    for word, g in env.get("groups", {}).items():
        if g.get("angle_deg"):
            w.append(f"The {word} fins moved between {g['angle_deg'][0]:.1f}° and {g['angle_deg'][1]:.1f}° in {g['manoeuvres']} "
                     f"manoeuvres.")
    w.append("Predictions outside this range aren't supported by these logs.")
    return w
