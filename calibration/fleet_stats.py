"""
fleet_stats.py -- statistics for fleet v2 experiments.

Episodes are matched across belief conditions by (policy, comm, backend,
episode_key) -- same initial state, same packet-drop pattern -- so
comparisons between conditions are PAIRED:

  paired_contrast()   difference in success (and collision, belief error)
                      between two conditions over matched episodes, with a
                      bootstrap 95% CI and McNemar's exact test on the
                      discordant pairs.
  dose_response()     success rate vs. how far the belief model's
                      parameters are from the truth (alpha), per comm
                      condition: least-squares slope with a paired
                      bootstrap CI (episodes resampled as whole matched sets).
  belief_success_link()  across conditions, does lower belief error go with
                      higher success? Spearman rank correlation.
  paired_by()         the same paired difference between any two values of
                      another field: clean vs noisy simulation (field
                      "mode"), or one group of policies vs another (field
                      "policy_group", e.g. SAC vs PPO; several training seeds
                      in a group are averaged per episode first).

Pure numpy/scipy; no pandas dependency so the report runs anywhere the
simulator does.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy import stats


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def _spearman(x, y) -> tuple:
    """Spearman rho and p, or (nan, nan) when either side is constant
    (correlation undefined -- e.g. every condition had the same success rate)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan"), float("nan")
    rho, p = stats.spearmanr(x, y)
    return float(rho), float(p)


def summarize(rows: Sequence[dict], by: Sequence[str]) -> list:
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in by)].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        n = len(rs)
        s = sum(r["success"] for r in rs)
        c = sum(r["collision"] for r in rs)
        be = np.array([r["mean_belief_error_m"] for r in rs], dtype=float)
        be = be[np.isfinite(be)]
        out.append({**dict(zip(by, key)), "n": n,
                    "success_rate": s / n, "success_ci95": wilson_ci(s, n),
                    "collision_rate": c / n, "collision_ci95": wilson_ci(c, n),
                    "mean_belief_error_m": float(be.mean()) if be.size else float("nan"),
                    "mean_steps": float(np.mean([r["steps"] for r in rs])),
                    "delivery_rate": float(np.mean([r["delivery_rate"] for r in rs]))})
    return out


def _match(rows, belief_a, belief_b, strata=("policy", "comm", "backend")):
    """{stratum: [(row_a, row_b), ...]} for episodes present in both conditions."""
    index = {}
    for r in rows:
        index[(tuple(r[k] for k in strata), r["episode_key"], r["belief"])] = r
    pairs = defaultdict(list)
    for (stratum, ep, belief), ra in index.items():
        if belief != belief_a:
            continue
        rb = index.get((stratum, ep, belief_b))
        if rb is not None:
            pairs[stratum].append((ra, rb))
    return pairs


def paired_contrast(rows, belief_a: str, belief_b: str, group_by: Sequence[str] = ("comm",),
                    n_boot: int = 2000, seed: int = 0) -> list:
    """For each group (default: per comm condition, pooling policies and
    backends), A minus B over matched episodes."""
    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for stratum, pairs in _match(rows, belief_a, belief_b).items():
        strat = dict(zip(("policy", "comm", "backend"), stratum))
        grouped[tuple(strat[g] for g in group_by)] += pairs
    out = []
    for key, pairs in sorted(grouped.items(), key=lambda kv: [str(x) for x in kv[0]]):
        sa = np.array([a["success"] for a, _ in pairs], dtype=float)
        sb = np.array([b["success"] for _, b in pairs], dtype=float)
        ca = np.array([a["collision"] for a, _ in pairs], dtype=float)
        cb = np.array([b["collision"] for _, b in pairs], dtype=float)
        ea = np.array([a["mean_belief_error_m"] for a, _ in pairs], dtype=float)
        eb = np.array([b["mean_belief_error_m"] for _, b in pairs], dtype=float)
        n = len(pairs)
        idx = rng.integers(0, n, size=(n_boot, n))
        d_s = (sa - sb)[idx].mean(axis=1)
        d_c = (ca - cb)[idx].mean(axis=1)
        ok = np.isfinite(ea) & np.isfinite(eb)
        d_e = (ea - eb)
        d_e_boot = np.array([np.mean(d_e[i][ok[i]]) if ok[i].any() else np.nan for i in idx])
        e_ci = (tuple(np.nanpercentile(d_e_boot, [2.5, 97.5]).tolist()) if np.isfinite(d_e_boot).any()
                else (float("nan"), float("nan")))
        b_count = int(np.sum((sa == 1) & (sb == 0)))
        c_count = int(np.sum((sa == 0) & (sb == 1)))
        out.append({**dict(zip(group_by, key)), "a": belief_a, "b": belief_b, "n_pairs": n,
                    "success_a": float(sa.mean()), "success_b": float(sb.mean()),
                    "success_diff": float((sa - sb).mean()),
                    "success_diff_ci95": tuple(np.percentile(d_s, [2.5, 97.5]).tolist()),
                    "a_only_successes": b_count, "b_only_successes": c_count,
                    "mcnemar_p": mcnemar_exact_p(b_count, c_count),
                    "collision_diff": float((ca - cb).mean()),
                    "collision_diff_ci95": tuple(np.percentile(d_c, [2.5, 97.5]).tolist()),
                    "belief_error_diff_m": float(np.mean(d_e[ok])) if ok.any() else float("nan"),
                    "belief_error_diff_ci95": e_ci})
    return out


def dose_response(rows, alpha_by_belief: dict, group_by: Sequence[str] = ("comm",),
                  n_boot: int = 2000, seed: int = 0) -> list:
    """Slope of success rate per unit alpha (alpha=0 true model, 1 generic),
    with a bootstrap CI that resamples matched episode sets."""
    rng = np.random.default_rng(seed)
    labels = list(alpha_by_belief)
    alphas = np.array([alpha_by_belief[l] for l in labels])
    table = defaultdict(dict)  # group -> {(policy, backend, episode_key): {label: success}}
    for r in rows:
        if r["belief"] not in alpha_by_belief:
            continue
        g = tuple(r[k] for k in group_by)
        table[g].setdefault((r["policy"], r["backend"], r["comm"], r["episode_key"]), {})[r["belief"]] = r["success"]
    out = []
    for g, eps in sorted(table.items(), key=lambda kv: [str(x) for x in kv[0]]):
        full = [e for e in eps.values() if len(e) == len(labels)]
        if len(full) < 2:
            continue
        mat = np.array([[e[l] for l in labels] for e in full], dtype=float)  # episodes x conditions
        rates = mat.mean(axis=0)
        slope = float(np.polyfit(alphas, rates, 1)[0])
        idx = rng.integers(0, len(full), size=(n_boot, len(full)))
        boot = np.array([np.polyfit(alphas, mat[i].mean(axis=0), 1)[0] for i in idx])
        rho, rho_p = _spearman(alphas, rates)
        out.append({**dict(zip(group_by, g)), "n_matched_episodes": len(full),
                    "alphas": alphas.tolist(), "success_rates": rates.tolist(),
                    "slope_per_unit_alpha": slope,
                    "slope_ci95": tuple(np.percentile(boot, [2.5, 97.5]).tolist()),
                    "spearman_rho": float(rho), "spearman_p": float(rho_p)})
    return out


def belief_success_link(summary_rows: Sequence[dict], group_by: Sequence[str] = ("comm",),
                        exclude_beliefs: Iterable[str] = ()) -> list:
    """Across belief conditions within each group: Spearman correlation of
    mean belief error vs success rate (negative = better belief, more success)."""
    exclude = set(exclude_beliefs)
    groups = defaultdict(list)
    for s in summary_rows:
        if s["belief"] in exclude or not np.isfinite(s["mean_belief_error_m"]):
            continue
        groups[tuple(s[k] for k in group_by)].append(s)
    out = []
    for g, ss in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
        if len(ss) < 3:
            continue
        rho, p = _spearman([s["mean_belief_error_m"] for s in ss], [s["success_rate"] for s in ss])
        out.append({**dict(zip(group_by, g)), "n_conditions": len(ss),
                    "spearman_rho": float(rho), "spearman_p": float(p)})
    return out


def paired_by(rows, field: str, a, b, match: Sequence[str], group_by: Sequence[str] = ("comm",),
              n_boot: int = 2000, seed: int = 0) -> list:
    """Success (and collisions) with rows[field] == a minus with rows[field] == b, over episodes matched on
    `match` (plus episode_key). Several rows on one side of a matched episode (e.g. several training seeds of a
    policy group) are averaged first, so each matched episode counts once. Bootstrap 95% CI over matched
    episodes; McNemar's test when both sides are single episodes (0/1)."""
    rng = np.random.default_rng(seed)
    cells = defaultdict(lambda: defaultdict(lambda: {a: [], b: []}))
    for r in rows:
        v = r.get(field)
        if v != a and v != b:
            continue
        g = tuple(r[k] for k in group_by)
        key = tuple(r[k] for k in match) + (r["episode_key"],)
        cells[g][key][v].append(r)
    out = []
    for g, eps in sorted(cells.items(), key=lambda kv: [str(x) for x in kv[0]]):
        both = [(sides[a], sides[b]) for sides in eps.values() if sides[a] and sides[b]]
        if not both:
            continue
        sa = np.array([np.mean([r["success"] for r in ra]) for ra, _ in both], dtype=float)
        sb = np.array([np.mean([r["success"] for r in rb]) for _, rb in both], dtype=float)
        ca = np.array([np.mean([r["collision"] for r in ra]) for ra, _ in both], dtype=float)
        cb = np.array([np.mean([r["collision"] for r in rb]) for _, rb in both], dtype=float)
        n = len(both)
        idx = rng.integers(0, n, size=(n_boot, n))
        d_s = (sa - sb)[idx].mean(axis=1)
        binary = all(len(ra) == 1 and len(rb) == 1 for ra, rb in both)
        b_count = int(np.sum((sa == 1) & (sb == 0))) if binary else None
        c_count = int(np.sum((sa == 0) & (sb == 1))) if binary else None
        out.append({**dict(zip(group_by, g)), "field": field, "a": a, "b": b, "n_pairs": n,
                    "success_a": float(sa.mean()), "success_b": float(sb.mean()),
                    "success_diff": float((sa - sb).mean()),
                    "success_diff_ci95": tuple(np.percentile(d_s, [2.5, 97.5]).tolist()),
                    "mcnemar_p": mcnemar_exact_p(b_count, c_count) if binary else None,
                    "collision_a": float(ca.mean()), "collision_b": float(cb.mean())})
    return out
