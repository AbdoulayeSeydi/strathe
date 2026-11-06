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

import numpy as np

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
