#!/usr/bin/env python3
"""
run_disturbance_study.py -- do realistic roll/pitch disturbances matter for
fleet formation-holding, and can policies learn to cope with them?

    python run_disturbance_study.py [--workers 6]

PROJECT_STATUS.md #52. Paired design (same start states, same packet drops,
same disturbance sequences wherever the disturbance level matches):
  - policies trained in the clean simulation (fleet_policy_seed1-3) and, if
    present, policies trained WITH the fitted disturbance
    (disturbance_policies/fleet_policy_dist_seed1-2);
  - belief models calibrated and last_beacon;
  - comm L2 with 1 s beacons and L3a with 3 s beacons;
  - disturbance off, as fitted to the real logs (x1), and twice as rough (x2);
  - the stand-in engine for everything, MuJoCo for x0 / x1 with seed 1 as the
    cross-engine check.

Writes results/fleet_v2_disturbance/: episodes.csv, summary.csv, metrics.json,
summary_report.md.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

from build_two_agent_cores import load_calibrated_parameters
from disturbance import DEFAULT_FIT_PATH, load_disturbance
from evaluate_comm_degradation_curve import COMM_CONFIGS
from fleet_belief import BeliefSpec
from fleet_env import FleetConfig
from fleet_eval import CellSpec, CommSpec, run_cells
from fleet_stats import paired_contrast, wilson_ci
from mlflow_run_utils import get_git_commit, get_git_dirty
from optimizer import load_vehicle_spec

TRUTH = "calibrated_parameters_fleet_truth.yaml"
CLEAN_POLICIES = {f"clean_seed{k}": f"fleet_policy_seed{k}" for k in (1, 2, 3)}
DIST_POLICIES = {f"dist_seed{k}": f"disturbance_policies/fleet_policy_dist_seed{k}" for k in (1, 2)}
BELIEFS = ("calibrated", "last_beacon")
COMMS = (("L2", 1.0), ("L3a", 3.0))
SCALES = (0.0, 1.0, 2.0)
SEPARATIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
SPEEDS = [-0.7, -0.35, 0.0, 0.35, 0.7]
COMM_SEEDS = [1, 2, 3, 4, 5]
OUT = "results/fleet_v2_disturbance"


def build():
    vehicle = load_vehicle_spec("vehicle_spec.yaml").vehicle
    truth = load_calibrated_parameters(TRUTH)
    specs = {"calibrated": BeliefSpec("physics", "calibrated", truth), "last_beacon": BeliefSpec("last_beacon",
                                                                                                  "last_beacon")}
    policies = dict(CLEAN_POLICIES)
    policies.update({k: v for k, v in DIST_POLICIES.items() if os.path.exists(v + ".zip")})
    cells = []
    for backend in ("stand_in", "mujoco"):
        for scale in SCALES:
            if backend == "mujoco" and scale == 2.0:
                continue
            dist = load_disturbance(DEFAULT_FIT_PATH, scale, truth_params_path=TRUTH) if scale > 0 else None
            fc = FleetConfig(dynamics_backend=backend, disturbance=dist)
            for pname, ppath in policies.items():
                if backend == "mujoco" and pname not in ("clean_seed1", "dist_seed1"):
                    continue
                for sev, iv in COMMS:
                    cs = CommSpec(f"{sev}@{iv:g}s", COMM_CONFIGS[sev]["packet_loss_config"],
                                  COMM_CONFIGS[sev]["bandwidth_config"], 1500.0, iv)
                    for b in BELIEFS:
                        cells.append(CellSpec(ppath, "ppo", cs, specs[b], truth, vehicle, copy.deepcopy(fc),
                                              SEPARATIONS, SPEEDS, COMM_SEEDS,
                                              tags={"policy_set": pname.rsplit("_", 1)[0], "policy_name": pname,
                                                    "seed_index": pname.rsplit("seed", 1)[1],
                                                    "disturbance_scale": scale}))
    return cells


def _contrast(rows, key_a, key_b, relabel, group_by=("comm",)):
    """Paired contrast between two row subsets, via fleet_stats.paired_contrast:
    rows are relabelled so 'belief' holds the side (a/b) and 'policy' holds
    everything else that must match."""
    rel = []
    for r in rows:
        side = relabel(r)
        if side in (key_a, key_b):
            rel.append({**r, "belief": side})
    return paired_contrast(rel, key_a, key_b, group_by=group_by)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    cells = build()
    print(f"{len(cells)} cells x {len(SEPARATIONS) * len(SPEEDS) * len(COMM_SEEDS)} episodes", flush=True)
    rows = run_cells(cells, workers=args.workers,
                     progress=lambda d, n: print(f"  {d}/{n} cells ({time.time() - t0:.0f}s)", flush=True))
    for r in rows:
        r["real_belief"] = r["belief"]
    os.makedirs(OUT, exist_ok=True)

    # success by (backend, policy set, comm, belief, scale)
    groups = defaultdict(list)
    for r in rows:
        groups[(r["backend"], r["policy_set"], r["comm"], r["real_belief"], r["disturbance_scale"])].append(r)
    summary = []
    for (be, ps, cm, b, sc), rs in sorted(groups.items()):
        k = sum(r["success"] for r in rs)
        lo, hi = wilson_ci(k, len(rs))
        summary.append({"backend": be, "policy_set": ps, "comm": cm, "belief": b, "disturbance_scale": sc,
                        "n": len(rs), "success_rate": k / len(rs), "success_ci95": [lo, hi],
                        "collision_rate": float(np.mean([r["collision"] for r in rs])),
                        "mean_abs_separation_error_m": float(np.mean([r["mean_abs_separation_error_m"] for r in rs]))})

    contrasts = {}
    # 1. effect of the disturbance on each policy set (same policy, same episode, scale x vs 0)
    for ps in sorted({r["policy_set"] for r in rows}):
        for sc in (1.0, 2.0):
            sub = [dict(r, policy=f"{r['policy_name']}|{r['real_belief']}") for r in rows
                   if r["backend"] == "stand_in" and r["policy_set"] == ps]
            contrasts[f"{ps}: x{sc:g} - off"] = _contrast(
                sub, f"x{sc:g}", "off", lambda r: "off" if r["disturbance_scale"] == 0 else f"x{r['disturbance_scale']:g}")
    # 2. disturbance-trained vs clean-trained under the disturbance, paired by seed index and episode
    if any(r["policy_set"] == "dist" for r in rows):
        for sc in SCALES:
            sub = [dict(r, policy=f"seed{r['seed_index']}|{r['real_belief']}") for r in rows
                   if r["backend"] == "stand_in" and r["disturbance_scale"] == sc and r["seed_index"] in ("1", "2")]
            contrasts[f"trained with disturbance - clean-trained, at x{sc:g}"] = _contrast(
                sub, "dist", "clean", lambda r: r["policy_set"])
    # 3. cross-engine: same policy, same episode, same disturbance, MuJoCo vs stand-in
    sub = [dict(r, belief=r["backend"], backend="any",
                policy=f"{r['policy_name']}|{r['real_belief']}|x{r['disturbance_scale']:g}")
           for r in rows if r["policy_name"] in ("clean_seed1", "dist_seed1") and r["disturbance_scale"] in (0.0, 1.0)]
    contrasts["MuJoCo - stand-in (seed 1, off and x1)"] = paired_contrast(sub, "mujoco", "stand_in")

    with open(os.path.join(OUT, "episodes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "contrasts": contrasts, "git_commit": get_git_commit(),
                   "git_uncommitted_changes": get_git_dirty(), "disturbance_file": DEFAULT_FIT_PATH,
                   "elapsed_s": round(time.time() - t0, 1)}, f, indent=2, default=float)

    def fmt(c):
        lo, hi = c["success_diff_ci95"]
        return (f"{100 * c['success_diff']:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}]" + ("" if lo <= 0 <= hi else " *")
                + f" (n={c['n_pairs']})")
    L = ["# Fleet v2: do roll/pitch disturbances matter?\n",
         f"{len(rows)} episodes. Disturbance fitted to the real BlueROV2 logs (`{os.path.basename(DEFAULT_FIT_PATH)}`); "
         "x1 = as fitted, x2 = twice as rough. Differences are success-rate points with 95% bootstrap CIs over "
         "matched episodes; `*` = CI excludes zero.\n", "## Success rate (stand-in engine)\n",
         "| Policies | Comm | Belief | off | x1 | x2 |", "|---|---|---|---|---|---|"]
    by = {(s["backend"], s["policy_set"], s["comm"], s["belief"], s["disturbance_scale"]): s for s in summary}
    for ps in sorted({s["policy_set"] for s in summary}):
        for cm in sorted({s["comm"] for s in summary}):
            for b in BELIEFS:
                cellv = [by.get(("stand_in", ps, cm, b, sc)) for sc in SCALES]
                L.append(f"| {ps} | {cm} | `{b}` | " + " | ".join(
                    "n/a" if c is None else f"{100 * c['success_rate']:.0f}%" for c in cellv) + " |")
    L += ["", "## Paired differences\n"]
    for name, cs in contrasts.items():
        L.append(f"**{name}**: " + "; ".join(f"{c['comm']}: {fmt(c)}" for c in cs))
        L.append("")
    with open(os.path.join(OUT, "summary_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    from keep_awake import keep_awake
    with keep_awake("fleet disturbance study"):
        rc = main()
    sys.exit(rc)
