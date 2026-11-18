#!/usr/bin/env python3
"""
run_escort_study.py -- the heterogeneous escort study: which REMUS model a BlueROV2 needs to escort it.

    python run_escort_study.py --policies escort_policy_seed1 escort_policy_seed2 escort_policy_seed3 scripted
    python run_escort_study.py --policies scripted --per-mission 1 --out results/escort_smoke    # quick check
    python run_escort_study.py --policies ... --beliefs calibrated_nofins calibrated_intent generic_intent \
        --merge-with results/escort_study/episodes.csv --out results/escort_beacon     # what the beacon carries
    python run_escort_study.py --policies ... --comms L3a@3 --current-speed 0.2 --noise open_water \
        --beliefs oracle calibrated calibrated_aware calibrated_intent calibrated_intent_aware generic constant_velocity \
        --out results/escort_open_water                                                 # robustness slice

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
# What the beacon carries (escort_beliefs.py), run with --beliefs ... --merge-with <the main study's episodes.csv>
BEACON_BELIEFS = ("calibrated_nofins", "calibrated_intent", "generic_intent")
CONTRASTS = (("calibrated", "generic"), ("calibrated", "constant_velocity"), ("calibrated", "last_beacon"),
             ("constant_velocity", "last_beacon"), ("generic", "last_beacon"), ("oracle", "calibrated"))
BEACON_CONTRASTS = (("calibrated", "calibrated_nofins"), ("calibrated_intent", "calibrated"),
                    ("generic_intent", "generic"), ("calibrated_intent", "generic_intent"),
                    ("oracle", "calibrated_intent"))
SCRIPTED_PI_KI = 4.0
_TEXT_FIELDS = {"policy", "belief", "comm", "backend", "episode_key", "noise"}
ALPHA = {"calibrated": 0.0, "physics_alpha0.25": 0.25, "physics_alpha0.5": 0.5, "physics_alpha0.75": 0.75, "generic": 1.0}
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "escort", "missions_cache")

_W = {}


def parse_comm(s: str) -> EE.CommCondition:
    sev, _, b = s.partition("@")
    return EE.CommCondition(sev, float(b.rstrip("s") or 1.0))


def _escort_config(noise, noise_scale):
    from sensor_noise import resolve_sensor_noise
    return EE.EscortConfig(sensor_noise=resolve_sensor_noise(noise, noise_scale))


def _init_worker(lib_path: str, noise=None, noise_scale=1.0):
    try:
        import torch
        torch.set_num_threads(1)   # one core per worker
    except ImportError:            # scripted-only runs don't need torch
        pass
    _W["lib"] = ER.MissionLibrary.load(lib_path)
    _W["rov"], _W["remus"], _W["cfg"] = EE.load_rov(), EE.load_remus_models(), _escort_config(noise, noise_scale)
    _W["noise"] = noise or "none"
    _W["policies"] = {}


def _policy(name: str):
    if name not in _W["policies"]:
        if name == "scripted":
            _W["policies"][name] = EE.EscortPursuit(_W["rov"], _W["cfg"].target_range_m)
        elif name == "scripted_pi":     # with integral action, for runs in a current
            _W["policies"][name] = EE.EscortPursuit(_W["rov"], _W["cfg"].target_range_m, ki=SCRIPTED_PI_KI)
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
                     "episode_key": ep.key, "mission_seed": sim.library[ep.mission].seed,
                     "current_mps": float(np.linalg.norm(sim.library[ep.mission].current)),
                     "noise": _W.get("noise", "none"), **r})
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))   # every row's columns: merged older runs lack newer ones
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


INTERACTIONS = (("calibration x intent", {"calibrated_intent": 1, "generic_intent": -1, "calibrated": -1, "generic": 1}),)


def mission_clustered(rows: list, weights: dict, group_by=("comm",), n_boot: int = 4000, seed: int = 0) -> list:
    """A paired linear combination of success rates (e.g. {a: 1, b: -1} for a - b) with a 95% CI that resamples whole
    missions. Every policy runs the same episodes and each mission gives several, so episodes aren't independent;
    resampling missions keeps every correlated row together. Matched on (policy, comm, episode); a match needs every
    belief in `weights`."""
    by = defaultdict(dict)
    for r in rows:
        if r["belief"] in weights:
            by[(r["policy"], r["comm"], r["episode_key"])][r["belief"]] = (r["success"], r["mission_seed"])
    groups = defaultdict(lambda: defaultdict(list))
    for (pol, comm, _), d in by.items():
        if len(d) != len(weights):
            continue
        val = sum(w * d[b][0] for b, w in weights.items())
        mission = next(iter(d.values()))[1]
        g = tuple({"policy": pol, "comm": comm}[k] for k in group_by)
        groups[g][mission].append(val)
    rng = np.random.default_rng(seed)
    out = []
    for g, per in sorted(groups.items()):
        sums = np.array([sum(v) for v in per.values()], dtype=float)
        ns = np.array([len(v) for v in per.values()], dtype=float)
        idx = rng.integers(0, len(sums), size=(n_boot, len(sums)))
        boot = sums[idx].sum(axis=1) / ns[idx].sum(axis=1)
        out.append({**dict(zip(group_by, g)), "estimate": float(sums.sum() / ns.sum()), "n_pairs": int(ns.sum()),
                    "n_missions": len(sums), "ci95": tuple(np.percentile(boot, [2.5, 97.5]).tolist())})
    return out


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
    for a, b in CONTRASTS + BEACON_CONTRASTS:
        for group in (("comm",), ("policy", "comm")):
            clustered = {tuple(m[k] for k in group): m for m in mission_clustered(rows, {a: 1, b: -1}, group)}
            for c in paired_contrast(rows, a, b, group_by=group):
                m = clustered.get(tuple(c[k] for k in group))
                c["success_diff_ci95_episodes"] = c["success_diff_ci95"]
                if m is not None:      # the report's CI: whole missions resampled
                    c["success_diff_ci95"] = m["ci95"]
                    c["n_missions"] = m["n_missions"]
                c.setdefault("policy", "pooled")
                contrasts.append(c)
    interactions = []
    for name, w in INTERACTIONS:
        for m in mission_clustered(rows, w, ("comm",)):
            interactions.append({"interaction": name, **m})
    _write_csv(os.path.join(out, "interactions.csv"), [
        {k: (f"[{v[0]:+.3f}, {v[1]:+.3f}]" if isinstance(v, tuple) else v) for k, v in d.items()} for d in interactions])
    flat = lambda d: {k: (f"[{v[0]:+.3f}, {v[1]:+.3f}]" if isinstance(v, tuple) else v) for k, v in d.items()}
    _write_csv(os.path.join(out, "summary.csv"), [flat(s) for s in summary])
    _write_csv(os.path.join(out, "summary_pooled.csv"), [flat(s) for s in pooled])
    _write_csv(os.path.join(out, "contrasts.csv"), [flat(c) for c in contrasts])
    write_report(rows, pooled, contrasts, out, interactions)
    return {"summary": summary, "pooled": pooled, "contrasts": contrasts, "interactions": interactions}


def read_episodes(path: str) -> list:
    """An episodes.csv back as rows with numbers where the study wrote numbers."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row = {}
            for k, v in r.items():
                if k in _TEXT_FIELDS:
                    row[k] = v
                else:
                    x = float(v) if v not in ("", "nan") else float("nan")
                    row[k] = int(x) if k in ("success", "collision", "lost", "steps", "beacons_sent",
                                             "beacons_delivered", "mission_seed") and np.isfinite(x) else x
            out.append(row)
    return out


def _pct(x):
    return f"{100 * x:.0f}%"


def write_report(rows, pooled, contrasts, out, interactions=()):
    comms = sorted({r["comm"] for r in rows}, key=lambda c: (c.split("@")[0], float(c.split("@")[1][:-1])))
    present = {r["belief"] for r in rows}
    order = list(BELIEFS + BEACON_BELIEFS)
    order += [f"{b}_aware" for b in order]
    beliefs = [b for b in order if b in present] + sorted(present - set(order))
    pols = sorted({r["policy"] for r in rows})
    P = {(s["comm"], s["belief"]): s for s in pooled}
    L = [f"# Escort study: a BlueROV2 escorting a REMUS 100", "",
         f"{len(rows)} episodes; policies {', '.join(pols)}; pooled over policies unless a table says otherwise. "
         f"Paired: every cell runs the same held-out episodes. CIs are 95% bootstrap resampling whole missions (every "
         f"policy runs the same episodes and each mission gives several, so they aren't independent); `*` = the CI "
         f"excludes zero. contrasts.csv also keeps the per-episode CIs (`success_diff_ci95_episodes`).", "", "## Success rate", "", "| Comm | " + " | ".join(f"`{b}`" for b in beliefs) + " |",
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
    C = {(c["comm"], c["a"], c["b"]): c for c in contrasts if c["policy"] == "pooled"}
    for title, pairs in (("Paired differences in success (points), pooled over policies", CONTRASTS),
                         ("What the beacon carries: paired differences in success (points)", BEACON_CONTRASTS)):
        pairs = [(a, b) for a, b in pairs if any((cm, a, b) in C for cm in comms)]
        if not pairs:
            continue
        L += ["", f"## {title}", "", "| Comm | " + " | ".join(f"{a} - {b}" for a, b in pairs) + " |",
              "|---|" + "---|" * len(pairs)]
        for cm in comms:
            cells = []
            for a, b in pairs:
                c = C.get((cm, a, b))
                if c is None:
                    cells.append("")
                    continue
                lo, hi = c["success_diff_ci95"]
                star = " *" if lo > 0 or hi < 0 else ""
                cells.append(f"{100 * c['success_diff']:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}]{star}")
            L.append(f"| {cm} | " + " | ".join(cells) + " |")
    if interactions:
        L += ["", "## Interactions (points, mission-resampled 95% CI)", "",
              "`calibration x intent` = (calibrated_intent - generic_intent) - (calibrated - generic): how much more "
              "calibration matters when the beacon carries the orders.", "", "| Comm | Interaction | Estimate | CI | Pairs |",
              "|---|---|---|---|---|"]
        for d in interactions:
            lo, hi = d["ci95"]
            star = " *" if lo > 0 or hi < 0 else ""
            L.append(f"| {d['comm']} | {d['interaction']} | {100 * d['estimate']:+.0f}{star} | "
                     f"[{100 * lo:+.0f}, {100 * hi:+.0f}] | {d['n_pairs']} |")
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
    ap.add_argument("--current-speed", type=float, default=0.0, metavar="MPS",
                    help="a steady current of this speed, toward a direction drawn per mission (default: still water). "
                         "Models see it only as <model>_aware, e.g. calibrated_aware")
    ap.add_argument("--noise", default=None, help="the ROV's own sensor noise: none (default), tank, typical, open_water "
                                                  "(SOLAQUA, measured; biases assumed) or a sensor-noise YAML")
    ap.add_argument("--noise-scale", type=float, default=1.0)
    ap.add_argument("--merge-with", default=None, metavar="EPISODES_CSV",
                    help="an earlier run's episodes.csv on the same episodes: its rows are analysed with this run's "
                         "(e.g. new beacon contents against the main study's models). Refused if the episodes differ.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    seeds = list(ER.EVAL_SEEDS)[:args.missions]
    lib = ER.MissionLibrary.build(seeds, ER.MissionConfig(current_speed_mps=args.current_speed), cache_dir=args.cache,
                                  workers=args.workers)
    lib_path = ER.MissionLibrary.cache_path(args.cache, lib.cfg, seeds)
    cfg = _escort_config(args.noise, args.noise_scale)
    episodes = EE.evaluation_episodes(lib, cfg, args.per_mission, seed=args.episode_seed)
    for c in args.comms:
        parse_comm(c)
    tasks = [(p, b, c, episodes) for p in args.policies for c in args.comms for b in args.beliefs]
    print(f"{len(tasks)} cells x {len(episodes)} episodes = {len(tasks) * len(episodes)} episodes on {args.workers} workers")
    t0 = time.time()
    rows = []
    if args.workers > 1:
        from multiprocessing import get_context
        with get_context("spawn").Pool(args.workers, initializer=_init_worker,
                                             initargs=(lib_path, args.noise, args.noise_scale)) as pool:
            for i, rs in enumerate(pool.imap_unordered(run_cell, tasks)):
                rows += rs
                print(f"  {i + 1}/{len(tasks)} cells ({(time.time() - t0) / 60:.1f} min)", flush=True)
    else:
        _init_worker(lib_path, args.noise, args.noise_scale)
        for i, t in enumerate(tasks):
            rows += run_cell(t)
            print(f"  {i + 1}/{len(tasks)} cells ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if args.merge_with:
        old_rows = read_episodes(args.merge_with)
        new_keys = {r["episode_key"] for r in rows}
        if new_keys - {r["episode_key"] for r in old_rows}:
            raise SystemExit(f"{args.merge_with} wasn't run on these episodes (different --missions, --per-mission "
                             "or --episode-seed): the rows can't be paired")
        clash = {(r["policy"], r["comm"], r["belief"]) for r in rows} & {(r["policy"], r["comm"], r["belief"]) for r in old_rows}
        old_rows = [r for r in old_rows if (r["policy"], r["comm"], r["belief"]) not in clash]
        print(f"merged {len(old_rows)} rows from {args.merge_with}" + (f" (re-run cells replaced: {len(clash)})" if clash else ""))
        rows = old_rows + rows
    rows.sort(key=lambda r: (r["policy"], r["comm"], r["belief"], r["episode_key"]))
    _write_csv(os.path.join(args.out, "episodes.csv"), rows)
    analyse(rows, args.out)
    from mlflow_run_utils import get_git_commit, get_git_dirty
    meta = {"policies": args.policies, "beliefs": args.beliefs, "comms": args.comms, "episodes_per_cell": len(episodes),
            "current_speed_mps": args.current_speed, "rov_sensor_noise": args.noise or "none",
            "noise_scale": args.noise_scale,
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
