#!/usr/bin/env python3
"""
run_algorithm_study.py -- are SAC and TD3 fleet policies usable, and how do
they compare with PPO? PROJECT_STATUS.md #58.

    python run_algorithm_study.py [--workers 6]

Judged against criteria written before any SAC/TD3 fleet policy had been
evaluated: results/fleet_v2_algorithms/criteria.md. In short:
  Gate 1 (integration), every trained file: passes validate_fleet_policy.py
    with the algorithm read from the file; saved and reloaded, identical
    actions on 1,000 recorded observations; no NaN/inf action.
  Gate 2 (learning), pooled over each algorithm's three seeds, on L2 and L3a,
    1 s and 3 s beacons, beliefs calibrated and last_beacon, the standard
    8 x 5 x 5 episode grid (clean, stand-in):
      learned  = success >= floor + 0.5 x (PPO - floor), floor = random commands
      parity   = paired (algorithm - PPO) 95% CI includes zero or lies above it
      a seed below floor + 0.25 x (PPO - floor) is a failed seed

Writes results/fleet_v2_algorithms/: episodes.csv, metrics.json, report.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time

import numpy as np

from build_two_agent_cores import load_calibrated_parameters
from evaluate_comm_degradation_curve import COMM_CONFIGS
from fleet_belief import BeliefSpec
from fleet_env import FleetConfig, FleetSim, N_AGENTS
from fleet_eval import RANDOM_POLICY, ZERO_POLICY, CellSpec, CommSpec, run_cells
from fleet_stats import paired_by, wilson_ci
from mlflow_run_utils import get_git_commit, get_git_dirty
from optimizer import load_vehicle_spec

TRUTH = "calibrated_parameters_fleet_truth.yaml"
POLICIES = {"ppo": [f"fleet_policy_seed{k}" for k in (1, 2, 3)],
            "sac": [f"algorithm_policies/fleet_sac_seed{k}" for k in (1, 2, 3)],
            "td3": [f"algorithm_policies/fleet_td3_seed{k}" for k in (1, 2, 3)]}
BELIEFS = ("calibrated", "last_beacon")
COMMS = (("L2", 1.0), ("L2", 3.0), ("L3a", 1.0), ("L3a", 3.0))
SEPARATIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
SPEEDS = [-0.7, -0.35, 0.0, 0.35, 0.7]
COMM_SEEDS = [1, 2, 3, 4, 5]
OUT = "results/fleet_v2_algorithms"


def gate1(path: str) -> dict:
    """Integration checks for one policy file."""
    from policy_io import load_policy
    from validate_fleet_policy import load_fleet_policy
    res = {"policy": path}
    try:
        model = load_fleet_policy(path)
        res["contract"] = True
    except Exception as e:  # noqa: BLE001 - reported, not raised
        return {**res, "contract": False, "error": f"{type(e).__name__}: {e}"}
    # 1,000 real observations, recorded by driving the fleet with the policy itself
    vehicle = load_vehicle_spec("vehicle_spec.yaml").vehicle
    truth = load_calibrated_parameters(TRUTH, quiet=True)
    sim = FleetSim(truth, vehicle, FleetConfig(), COMM_CONFIGS["L2"]["packet_loss_config"],
                   COMM_CONFIGS["L2"]["bandwidth_config"], belief=BeliefSpec("last_beacon", "last_beacon"))
    obs_log, finite = [], True
    k = 0
    while len(obs_log) < 1000:
        obs = sim.reset(comm_seed=k, initial_separation=2.0 + (k % 8) * 0.5, closing_speed=-0.7 + (k % 5) * 0.35)
        k += 1
        while len(obs_log) < 1000:
            act, _ = model.predict(np.stack(obs), deterministic=True)
            finite &= bool(np.all(np.isfinite(act)))
            obs_log += list(obs)
            obs, _, term, trunc, _ = sim.step([act[i] for i in range(N_AGENTS)])
            if term or trunc:
                break
    X = np.stack(obs_log[:1000])
    a1 = model.predict(X, deterministic=True)[0]
    with tempfile.TemporaryDirectory() as d:
        model.save(os.path.join(d, "again"))
        again, alg = load_policy(os.path.join(d, "again"))
        a2 = again.predict(X, deterministic=True)[0]
    res.update(algorithm=alg, reload_identical=bool(np.array_equal(a1, a2)), finite=finite and bool(np.all(np.isfinite(a1))),
               max_abs_action=float(np.abs(a1).max()))
    res["passed"] = bool(res["contract"] and res["reload_identical"] and res["finite"])
    return res


def build_cells():
    vehicle = load_vehicle_spec("vehicle_spec.yaml").vehicle
    truth = load_calibrated_parameters(TRUTH, quiet=True)
    beliefs = {"calibrated": BeliefSpec("physics", "calibrated", truth),
               "last_beacon": BeliefSpec("last_beacon", "last_beacon")}
    fc = FleetConfig()
    cells = []
    entries = ([(alg, p) for alg, ps in POLICIES.items() for p in ps]
               + [("random", RANDOM_POLICY), ("zero", ZERO_POLICY)])
    for group, path in entries:
        for sev, interval in COMMS:
            cs = CommSpec(f"{sev}@{interval:g}s", COMM_CONFIGS[sev]["packet_loss_config"],
                          COMM_CONFIGS[sev]["bandwidth_config"], 1500.0, interval)
            for b in BELIEFS:
                cells.append(CellSpec(path, None if group in ("random", "zero") else group, cs, beliefs[b], truth, vehicle, fc,
                                      SEPARATIONS, SPEEDS, COMM_SEEDS, tags={"policy_group": group}))
    return cells


def rate(rows):
    n = len(rows)
    s = sum(bool(r["success"]) for r in rows)
    return (s / n if n else float("nan")), wilson_ci(s, n), n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    missing = [p for ps in POLICIES.values() for p in ps if not os.path.exists(p + ".zip")]
    if missing:
        print(f"ERROR: policies not trained yet: {missing}", file=sys.stderr)
        return 2
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    print("Gate 1: integration checks...", flush=True)
    g1 = [gate1(p) for alg in ("sac", "td3") for p in POLICIES[alg]]
    for r in g1:
        print(f"  {r['policy']}: {'PASS' if r.get('passed') else 'FAIL'} {r}", flush=True)
    cells = build_cells()
    print(f"Gate 2: {len(cells)} cells x {len(SEPARATIONS) * len(SPEEDS) * len(COMM_SEEDS)} episodes...", flush=True)
    rows = run_cells(cells, workers=args.workers,
                     progress=lambda d, n: print(f"  {d}/{n} cells ({time.time() - t0:.0f}s)", flush=True)
                     if d % 10 == 0 or d == n else None)
    for r in rows:
        r["success"] = int(bool(r["success"]))
        r["collision"] = int(bool(r["collision"]))
    with open(os.path.join(OUT, "episodes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    by = lambda g: [r for r in rows if r["policy_group"] == g]
    floor = rate(by("random"))[0]
    ppo = rate(by("ppo"))[0]
    gap = ppo - floor
    summary, verdicts = {}, {}
    for g in ("ppo", "sac", "td3", "random", "zero"):
        s, ci, n = rate(by(g))
        per_seed = {os.path.basename(p): rate([r for r in by(g) if r["policy"] == os.path.basename(p)])[0]
                    for p in POLICIES.get(g, [])}
        per_comm = {f"{c}|{b}": rate([r for r in by(g) if r["comm"] == c and r["belief"] == b])[0]
                    for c in dict.fromkeys(x["comm"] for x in rows) for b in BELIEFS}
        summary[g] = {"success": s, "ci95": ci, "episodes": n, "per_seed": per_seed, "per_condition": per_comm,
                      "collision": float(np.mean([r["collision"] for r in by(g)]))}
    contrasts = {}
    for g in ("sac", "td3"):
        (pooled,) = paired_by(rows, "policy_group", g, "ppo", match=("belief", "comm", "backend"), group_by=())
        contrasts[g] = {"pooled": pooled,
                        "per_comm": paired_by(rows, "policy_group", g, "ppo", match=("belief", "comm", "backend"),
                                              group_by=("comm",))}
        s = summary[g]
        g1_ok = all(r.get("passed") for r in g1 if f"_{g}_" in r["policy"])
        learned = s["success"] >= floor + 0.5 * gap
        parity = pooled["success_diff_ci95"][1] >= 0
        failed_seeds = [k for k, v in s["per_seed"].items() if v < floor + 0.25 * gap]
        verdicts[g] = {"gate1": g1_ok, "learned": learned, "parity": parity, "failed_seeds": failed_seeds,
                       "validated": bool(g1_ok and learned),
                       "threshold_learned": floor + 0.5 * gap, "threshold_failed_seed": floor + 0.25 * gap}
    wall = {}
    for g, ps in POLICIES.items():
        ts = []
        for p in ps:
            try:
                with open(p + ".json", encoding="utf-8") as f:
                    meta = json.load(f)
                ts.append((meta.get("timesteps"), meta.get("elapsed_s")))
            except OSError:
                pass
        wall[g] = ts
    metrics = {"criteria": "results/fleet_v2_algorithms/criteria.md", "gate1": g1, "summary": summary,
               "floor": floor, "ppo": ppo, "contrasts_vs_ppo": contrasts, "verdicts": verdicts, "training": wall,
               "episodes": len(rows), "git_commit": get_git_commit(), "git_uncommitted_changes": get_git_dirty(),
               "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))

    pct = lambda x: f"{100 * x:.0f}%"
    pp = lambda x: f"{100 * x:+.0f}"
    L = ["# SAC and TD3 for the fleet, against PPO (PROJECT_STATUS.md #58)\n",
         f"{len(rows):,} episodes, clean simulation, stand-in physics. Criteria fixed before any result: "
         "[criteria.md](criteria.md).\n", "**Scope:** the reference BlueROV2, the formation-hold task, clean simulation, SB3-standard settings. The ranking may not carry over to another vehicle or task (under noise, TD3 falls from best to worst: results/fleet_v2_noise). For your vehicle, train two or three seeds of each and compare them in one experiment (`strathe fleet --policies ppo=... sac=...`).\n", "## Verdict\n",
         "| Algorithm | Gate 1 (integration) | Learned (≥ floor + half the gap to PPO) | Parity with PPO | Failed seeds | Validated for the fleet |",
         "|---|---|---|---|---|---|"]
    for g in ("sac", "td3"):
        v = verdicts[g]
        L.append(f"| {g.upper()} | {'pass' if v['gate1'] else 'FAIL'} | {'yes' if v['learned'] else 'no'} "
                 f"({pct(summary[g]['success'])} vs threshold {pct(v['threshold_learned'])}) | "
                 f"{'yes' if v['parity'] else 'no'} | {', '.join(v['failed_seeds']) or 'none'} | "
                 f"**{'yes' if v['validated'] else 'no'}** |")
    L += ["", "## Success, pooled over the grid\n", "| | Success (95% CI) | Collisions | Per seed |", "|---|---|---|---|"]
    names = {"random": "random commands (the floor)", "zero": "no commands (reference, not in the criteria)"}
    for g in ("ppo", "sac", "td3", "random", "zero"):
        s = summary[g]
        seeds = ", ".join(pct(v) for v in s["per_seed"].values()) or "n/a"
        L.append(f"| {names.get(g, g.upper())} | {pct(s['success'])} "
                 f"({pct(s['ci95'][0])}–{pct(s['ci95'][1])}) | {pct(s['collision'])} | {seeds} |")
    L += ["", "## Paired against PPO (same episodes; seeds averaged per episode)\n",
          "| Condition | SAC − PPO (95% CI) | TD3 − PPO (95% CI) |", "|---|---|---|"]
    comms = [c["comm"] for c in contrasts["sac"]["per_comm"]]
    for cm in comms:
        cells = []
        for g in ("sac", "td3"):
            c = next(x for x in contrasts[g]["per_comm"] if x["comm"] == cm)
            cells.append(f"{pp(c['success_diff'])} pts ({pp(c['success_diff_ci95'][0])} to "
                         f"{pp(c['success_diff_ci95'][1])})")
        L.append(f"| {cm} | " + " | ".join(cells) + " |")
    pooled = [contrasts[g]["pooled"] for g in ("sac", "td3")]
    L.append("| **pooled** | " + " | ".join(f"**{pp(c['success_diff'])} pts** ({pp(c['success_diff_ci95'][0])} to "
                                           f"{pp(c['success_diff_ci95'][1])})" for c in pooled) + " |")
    L += ["", "## Training cost\n", "| | Steps per seed | Wall-clock per seed |", "|---|---|---|"]
    for g in ("ppo", "sac", "td3"):
        ts = [t for t in wall[g] if t[0]]
        if ts:
            L.append(f"| {g.upper()} | {ts[0][0]:,} | " + ", ".join(f"{(e or 0) / 60:.0f} min" for _, e in ts) + " |")
    L += ["", "Gate 1 details are in `metrics.json` (`gate1`). Wall-clock times depend on how many runs shared the "
          "machine; PPO's shipped policies were trained with fewer runs in parallel.\n"]
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    from keep_awake import keep_awake
    with keep_awake("algorithm study"):
        sys.exit(main())
