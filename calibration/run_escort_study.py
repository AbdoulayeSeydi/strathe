#!/usr/bin/env python3
"""
run_escort_study.py -- the heterogeneous escort study: which REMUS model a BlueROV2 needs to escort it.

    python run_escort_study.py --policies escort_policy_seed1 escort_policy_seed2 escort_policy_seed3 scripted
    python run_escort_study.py --policies scripted --per-mission 1 --out results/escort_smoke    # quick check

Every cell (policy x teammate model x comm condition) runs the SAME episodes: the held-out missions
(escort_remus.EVAL_SEEDS, never trained on), the same windows, ROV starts and packet-loss draws. Only the teammate
model differs between beliefs, so differences are paired.

Writes to --out: episodes.csv (one row per episode), summary.csv, contrasts.csv (paired differences with 95%
bootstrap CIs), report.md, metadata.json. A policy is a path to an SB3 .zip (algorithm read from the file) or
`scripted` (escort_env.EscortPursuit, no learning).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from collections import defaultdict

import numpy as np

import escort_env as EE
import escort_remus as ER

BELIEFS = ("oracle", "calibrated", "physics_alpha0.25", "physics_alpha0.5", "physics_alpha0.75", "generic",
           "constant_velocity", "last_beacon")
COMMS = ("L0@1", "L2@3", "L3a@1", "L3a@3")
CONTRASTS = (("calibrated", "generic"), ("calibrated", "constant_velocity"), ("calibrated", "last_beacon"),
             ("constant_velocity", "last_beacon"), ("generic", "last_beacon"), ("oracle", "calibrated"))
ALPHA = {"calibrated": 0.0, "physics_alpha0.25": 0.25, "physics_alpha0.5": 0.5, "physics_alpha0.75": 0.75, "generic": 1.0}
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "escort", "missions_cache")

_W = {}


def parse_comm(s: str) -> EE.CommCondition:
    sev, _, b = s.partition("@")
    return EE.CommCondition(sev, float(b.rstrip("s") or 1.0))


def _init_worker(lib_path: str):
    try:
        import torch
        torch.set_num_threads(1)   # one core per worker
    except ImportError:            # scripted-only runs don't need torch
        pass
    _W["lib"] = ER.MissionLibrary.load(lib_path)
    _W["rov"], _W["remus"], _W["cfg"] = EE.load_rov(), EE.load_remus_models(), EE.EscortConfig()
    _W["policies"] = {}


def _policy(name: str):
    if name not in _W["policies"]:
        if name == "scripted":
            _W["policies"][name] = EE.EscortPursuit(_W["rov"], _W["cfg"].target_range_m)
        else:
            from policy_io import load_policy
            model, _ = load_policy(name)
            if model.observation_space.shape != (EE.OBS_DIM,) or model.action_space.shape != (EE.ACT_DIM,):
                raise ValueError(f"{name} isn't an escort policy (observation {model.observation_space.shape}, "
                                 f"action {model.action_space.shape})")
            _W["policies"][name] = model
    return _W["policies"][name]


def run_cell(task) -> list:
    policy_name, belief, comm_s, episodes = task
    sim = EE.EscortSim(_W["rov"], _W["remus"], _W["lib"], _W["cfg"])
    sim.set_comm(parse_comm(comm_s))
    sim.set_belief(_W["remus"].belief(belief))
    pol = _policy(policy_name)
    rows = []
    label = os.path.basename(policy_name)
    for ep in episodes:
        r = EE.run_episode(sim, pol, ep)
        rows.append({"policy": label, "belief": belief, "comm": sim.comm.label, "backend": "stand_in",
                     "episode_key": ep.key, "mission_seed": sim.library[ep.mission].seed, **r})
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def analyse(rows: list, out: str) -> dict:
    """Summaries, paired contrasts, the dose-response and the straight/turn split; writes the CSVs and report.md."""
    from fleet_stats import paired_contrast, summarize
    summary = summarize(rows, ("policy", "comm", "belief"))
    pooled = summarize(rows, ("comm", "belief"))
    extra = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k in ("in_band_frac", "in_band_frac_straight", "in_band_frac_turn", "belief_error_straight_m",
                  "belief_error_turn_m", "lost", "mean_range_error_m"):
            extra[(r["comm"], r["belief"])][k].append(r[k])
    for s in pooled:
        e = extra[(s["comm"], s["belief"])]
        for k, v in e.items():
            s[f"mean_{k}" if not k.startswith("mean") else k] = float(np.nanmean(np.array(v, dtype=float)))
    contrasts = []
    for a, b in CONTRASTS:
        for group in (("comm",), ("policy", "comm")):
            for c in paired_contrast(rows, a, b, group_by=group):
                c.setdefault("policy", "pooled")
                contrasts.append(c)
    flat = lambda d: {k: (f"[{v[0]:+.3f}, {v[1]:+.3f}]" if isinstance(v, tuple) else v) for k, v in d.items()}
    _write_csv(os.path.join(out, "summary.csv"), [flat(s) for s in summary])
    _write_csv(os.path.join(out, "summary_pooled.csv"), [flat(s) for s in pooled])
    _write_csv(os.path.join(out, "contrasts.csv"), [flat(c) for c in contrasts])
    write_report(rows, pooled, contrasts, out)
    return {"summary": summary, "pooled": pooled, "contrasts": contrasts}


def _pct(x):
    return f"{100 * x:.0f}%"


def write_report(rows, pooled, contrasts, out):
    comms = sorted({r["comm"] for r in rows}, key=lambda c: (c.split("@")[0], float(c.split("@")[1][:-1])))
    beliefs = [b for b in BELIEFS if any(r["belief"] == b for r in rows)]
    pols = sorted({r["policy"] for r in rows})
    P = {(s["comm"], s["belief"]): s for s in pooled}
    L = [f"# Escort study: a BlueROV2 escorting a REMUS 100", "",
         f"{len(rows)} episodes; policies {', '.join(pols)}; pooled over policies unless a table says otherwise. "
         f"Paired: every cell runs the same held-out episodes. CIs are 95% bootstrap over matched episodes; `*` = the CI "
         f"excludes zero.", "", "## Success rate", "", "| Comm | " + " | ".join(f"`{b}`" for b in beliefs) + " |",
         "|---|" + "---|" * len(beliefs)]
    for c in comms:
        L.append(f"| {c} | " + " | ".join(_pct(P[(c, b)]["success_rate"]) if (c, b) in P else "" for b in beliefs) + " |")
    L += ["", "## Mean belief error (m): straight legs / turns", "", "| Comm | " + " | ".join(f"`{b}`" for b in beliefs) + " |",
          "|---|" + "---|" * len(beliefs)]
    for c in comms:
        L.append(f"| {c} | " + " | ".join(
            f"{P[(c, b)]['mean_belief_error_straight_m']:.2f} / {P[(c, b)]['mean_belief_error_turn_m']:.2f}" if (c, b) in P else ""
            for b in beliefs) + " |")
    L += ["", "## Time in the band: straight legs / turns", "", "| Comm | " + " | ".join(f"`{b}`" for b in beliefs) + " |",
          "|---|" + "---|" * len(beliefs)]
    for c in comms:
        L.append(f"| {c} | " + " | ".join(
            f"{_pct(P[(c, b)]['mean_in_band_frac_straight'])} / {_pct(P[(c, b)]['mean_in_band_frac_turn'])}" if (c, b) in P else ""
            for b in beliefs) + " |")
    L += ["", "## Paired differences in success (points), pooled over policies", "",
          "| Comm | " + " | ".join(f"{a} - {b}" for a, b in CONTRASTS) + " |", "|---|" + "---|" * len(CONTRASTS)]
    C = {(c["comm"], c["a"], c["b"]): c for c in contrasts if c["policy"] == "pooled"}
    for cm in comms:
        cells = []
        for a, b in CONTRASTS:
            c = C.get((cm, a, b))
            if c is None:
                cells.append("")
                continue
            lo, hi = c["success_diff_ci95"]
            star = " *" if lo > 0 or hi < 0 else ""
            cells.append(f"{100 * c['success_diff']:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}]{star}")
        L.append(f"| {cm} | " + " | ".join(cells) + " |")
    alphas = [b for b in ALPHA if b in beliefs]
    if len(alphas) >= 3:
        L += ["", "## Dose-response: model parameters moved from calibrated (0) to generic (1)", "",
              "| Comm | " + " | ".join(f"{ALPHA[b]:g}" for b in alphas) + " |", "|---|" + "---|" * len(alphas)]
        for c in comms:
            L.append(f"| {c} | " + " | ".join(
                f"{_pct(P[(c, b)]['success_rate'])} ({P[(c, b)]['mean_belief_error_m']:.2f} m)" if (c, b) in P else ""
                for b in alphas) + " |")
    if len(pols) > 1:
        L += ["", "## Success by policy", ""]
        by = defaultdict(list)
        for r in rows:
            by[(r["policy"], r["comm"], r["belief"])].append(r["success"])
        L += ["| Policy | Comm | " + " | ".join(f"`{b}`" for b in beliefs) + " |", "|---|---|" + "---|" * len(beliefs)]
        for p in pols:
            for c in comms:
                L.append(f"| {p} | {c} | " + " | ".join(
                    _pct(np.mean(by[(p, c, b)])) if by.get((p, c, b)) else "" for b in beliefs) + " |")
    with open(os.path.join(out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policies", nargs="+", required=True, help="SB3 .zip paths (without .zip is fine) and/or 'scripted'")
    ap.add_argument("--beliefs", nargs="+", default=list(BELIEFS))
    ap.add_argument("--comms", nargs="+", default=list(COMMS), help="SEVERITY@INTERVAL_S, e.g. L3a@3")
    ap.add_argument("--per-mission", type=int, default=3, help="episodes per held-out mission (100 missions)")
    ap.add_argument("--missions", type=int, default=len(ER.EVAL_SEEDS))
    ap.add_argument("--episode-seed", type=int, default=2027)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--out", default="results/escort_study")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = list(ER.EVAL_SEEDS)[:args.missions]
    lib = ER.MissionLibrary.build(seeds, cache_dir=args.cache, workers=args.workers)
    lib_path = ER.MissionLibrary.cache_path(args.cache, lib.cfg, seeds)
    cfg = EE.EscortConfig()
    episodes = EE.evaluation_episodes(lib, cfg, args.per_mission, seed=args.episode_seed)
    for c in args.comms:
        parse_comm(c)
    tasks = [(p, b, c, episodes) for p in args.policies for c in args.comms for b in args.beliefs]
    print(f"{len(tasks)} cells x {len(episodes)} episodes = {len(tasks) * len(episodes)} episodes on {args.workers} workers")
    t0 = time.time()
    rows = []
    if args.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(args.workers, initializer=_init_worker, initargs=(lib_path,)) as pool:
            for i, rs in enumerate(pool.imap_unordered(run_cell, tasks)):
                rows += rs
                print(f"  {i + 1}/{len(tasks)} cells ({(time.time() - t0) / 60:.1f} min)", flush=True)
    else:
        _init_worker(lib_path)
        for i, t in enumerate(tasks):
            rows += run_cell(t)
            print(f"  {i + 1}/{len(tasks)} cells ({(time.time() - t0) / 60:.1f} min)", flush=True)
    rows.sort(key=lambda r: (r["policy"], r["comm"], r["belief"], r["episode_key"]))
    _write_csv(os.path.join(args.out, "episodes.csv"), rows)
    analyse(rows, args.out)
    from mlflow_run_utils import get_git_commit, get_git_dirty
    meta = {"policies": args.policies, "beliefs": args.beliefs, "comms": args.comms, "episodes_per_cell": len(episodes),
            "eval_mission_seeds": [seeds[0], seeds[-1]], "mission_config": lib.cfg.__dict__, "task": cfg.__dict__,
            "rov": {"params": EE.ROV_PARAMS, "thrusters": "bluerov2_heavy"},
            "remus": {"spec": EE.REMUS_SPEC, "calibrated": EE.REMUS_CALIBRATED, "buoyancy_change_n": ER.BUOYANCY_CHANGE_N},
            "git_commit": get_git_commit(), "git_uncommitted_changes": get_git_dirty(), "python": platform.python_version(),
            "elapsed_min": round((time.time() - t0) / 60, 1)}
    with open(os.path.join(args.out, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"wrote {args.out}/report.md ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
