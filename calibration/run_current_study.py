#!/usr/bin/env python3
"""
run_current_study.py -- what does a water current do to fleet formation-holding,
does it matter whether the teammate model knows the current, and can
policies learn to cope?

    python run_current_study.py [--workers 6]

PROJECT_STATUS.md #55. Paired design: every condition sees the same start
states, packet drops and (per speed level) the same current draws. Varies:
  - current speed 0, 0.1, 0.2, 0.3 m/s (direction drawn per episode, any
    heading), plus 0.2 m/s with slow fluctuation (0.05 m/s, 15 deg, 30 s);
  - teammate model: `calibrated` that knows the episode's mean current,
    `calibrated` that doesn't, and `last_beacon` (no prediction);
  - policies trained in still water (fleet_policy_seed1-3) and, if present,
    policies trained in random currents (current_policies/fleet_policy_current_seed1-2);
  - comm L2 with 1 s beacons and L3a with 3 s beacons;
  - the stand-in engine, with MuJoCo at 0.2 m/s for seed 1 as a cross-check.

Writes results/fleet_v2_current/: episodes.csv, summary.csv, metrics.json,
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
from current import CurrentConfig
from evaluate_comm_degradation_curve import COMM_CONFIGS
from fleet_belief import BeliefSpec
from fleet_env import FleetConfig
from fleet_eval import CellSpec, CommSpec, run_cells
from fleet_stats import paired_contrast, wilson_ci
from mlflow_run_utils import get_git_commit, get_git_dirty
from optimizer import load_vehicle_spec

TRUTH = "calibrated_parameters_fleet_truth.yaml"
STILL_POLICIES = {f"still_seed{k}": f"fleet_policy_seed{k}" for k in (1, 2, 3)}
CURRENT_POLICIES = {f"current_seed{k}": f"current_policies/fleet_policy_current_seed{k}" for k in (1, 2)}
COMMS = (("L2", 1.0), ("L3a", 3.0))
CURRENTS = {"0": None, "0.1": dict(speed_mps=0.1), "0.2": dict(speed_mps=0.2), "0.3": dict(speed_mps=0.3),
            "0.2 fluctuating": dict(speed_mps=0.2, speed_std_mps=0.05, direction_std_deg=15, time_scale_s=30)}
MODELS = ("calibrated, knows current", "calibrated, unaware", "last_beacon")
SEPARATIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
SPEEDS = [-0.7, -0.35, 0.0, 0.35, 0.7]
COMM_SEEDS = [1, 2, 3, 4, 5]
OUT = "results/fleet_v2_current"


def build():
    vehicle = load_vehicle_spec("vehicle_spec.yaml").vehicle
    truth = load_calibrated_parameters(TRUTH)
    policies = dict(STILL_POLICIES)
    policies.update({k: v for k, v in CURRENT_POLICIES.items() if os.path.exists(v + ".zip")})
    cells = []
    for backend in ("stand_in", "mujoco"):
        for cur_label, cur_kw in CURRENTS.items():
            if backend == "mujoco" and cur_label not in ("0", "0.2"):
                continue
            for model in MODELS:
                known = model != "calibrated, unaware"
                current = None if cur_kw is None else CurrentConfig(direction_deg=(0.0, 360.0),
                                                                     known_to_beliefs=known, **cur_kw)
                belief = (BeliefSpec("last_beacon", "last_beacon") if model == "last_beacon"
                          else BeliefSpec("physics", model, truth))
                fc = FleetConfig(dynamics_backend=backend, current=current)
                for pname, ppath in policies.items():
                    if backend == "mujoco" and pname not in ("still_seed1", "current_seed1"):
                        continue
                    for sev, iv in COMMS:
                        cs = CommSpec(f"{sev}@{iv:g}s", COMM_CONFIGS[sev]["packet_loss_config"],
                                      COMM_CONFIGS[sev]["bandwidth_config"], 1500.0, iv)
                        cells.append(CellSpec(ppath, "ppo", cs, belief, truth, vehicle, copy.deepcopy(fc),
                                              SEPARATIONS, SPEEDS, COMM_SEEDS,
                                              tags={"policy_set": pname.rsplit("_", 1)[0], "policy_name": pname,
                                                    "seed_index": pname.rsplit("seed", 1)[1],
                                                    "current": cur_label}))
    return cells


def _pair(rows, side, a, b, match):
    """Paired contrast of side-label a vs b, matching on `match` (+ episode)."""
    rel = [dict(r, belief=side(r), policy="|".join(str(r[k]) for k in match), backend="any") for r in rows]
    return paired_contrast([r for r in rel if r["belief"] in (a, b)], a, b)


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
        r["model"] = r["belief"]
    os.makedirs(OUT, exist_ok=True)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["backend"], r["policy_set"], r["comm"], r["model"], r["current"])].append(r)
    summary = []
    for (be, ps, cm, mo, cu), rs in sorted(groups.items()):
        k = sum(r["success"] for r in rs)
        lo, hi = wilson_ci(k, len(rs))
        summary.append({"backend": be, "policy_set": ps, "comm": cm, "model": mo, "current": cu, "n": len(rs),
                        "success_rate": k / len(rs), "success_ci95": [lo, hi],
                        "collision_rate": float(np.mean([r["collision"] for r in rs])),
                        "mean_belief_error_m": float(np.nanmean([r["mean_belief_error_m"] for r in rs])),
                        "mean_abs_separation_error_m": float(np.mean([r["mean_abs_separation_error_m"] for r in rs]))})

    si = [r for r in rows if r["backend"] == "stand_in"]
    contrasts = {}
    for ps in sorted({r["policy_set"] for r in si}):
        sub = [r for r in si if r["policy_set"] == ps]
        for cu in [c for c in CURRENTS if c != "0"]:
            contrasts[f"{ps} policies: current {cu} m/s vs still water"] = _pair(
                [r for r in sub if r["current"] in ("0", cu)], lambda r: r["current"], cu, "0",
                ("policy_name", "model"))
        for cu in [c for c in CURRENTS if c != "0"]:
            contrasts[f"{ps} policies, current {cu} m/s: model knows vs unaware of current"] = _pair(
                [r for r in sub if r["current"] == cu], lambda r: r["model"], "calibrated, knows current",
                "calibrated, unaware", ("policy_name",))
    if any(r["policy_set"] == "current" for r in si):
        for cu in CURRENTS:
            contrasts[f"current-trained vs still-trained policies at {cu} m/s"] = _pair(
                [r for r in si if r["current"] == cu and r["seed_index"] in ("1", "2")], lambda r: r["policy_set"],
                "current", "still", ("seed_index", "model"))
    mj = [r for r in rows if r["policy_name"] in ("still_seed1", "current_seed1") and r["current"] in ("0", "0.2")]
    contrasts["MuJoCo vs stand-in (seed 1, still water and 0.2 m/s)"] = _pair(
        mj, lambda r: r["backend"], "mujoco", "stand_in", ("policy_name", "model", "current"))

    for name, data in (("episodes.csv", rows), ("summary.csv", summary)):
        with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "contrasts": contrasts, "git_commit": get_git_commit(),
                   "git_uncommitted_changes": get_git_dirty(), "elapsed_s": round(time.time() - t0, 1)},
                  f, indent=2, default=float)

    def fmt(c):
        lo, hi = c["success_diff_ci95"]
        return (f"{100 * c['success_diff']:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}]" + ("" if lo <= 0 <= hi else " *")
                + f" (n={c['n_pairs']})")
    by = {(s["backend"], s["policy_set"], s["comm"], s["model"], s["current"]): s for s in summary}
    L = ["# Fleet v2: water current\n",
         f"{len(rows)} episodes. Current direction drawn per episode (any heading); both vehicles feel the same "
         "water. Differences are success-rate points with 95% bootstrap CIs over matched episodes; `*` = CI "
         "excludes zero.\n"]
    for ps in sorted({s["policy_set"] for s in summary}):
        L += [f"## Success rate, {ps}-water-trained policies (stand-in engine)\n",
              "| Comm | Teammate model | " + " | ".join(f"{c} m/s" for c in CURRENTS) + " |",
              "|---|---|" + "---|" * len(CURRENTS)]
        for cm in sorted({s["comm"] for s in summary}):
            for mo in MODELS:
                vals = [by.get(("stand_in", ps, cm, mo, cu)) for cu in CURRENTS]
                L.append(f"| {cm} | {mo} | " + " | ".join("n/a" if v is None else f"{100 * v['success_rate']:.0f}%"
                                                          for v in vals) + " |")
        L.append("")
    L.append("## Paired differences\n")
    for name, cs in contrasts.items():
        L.append(f"**{name}**: " + "; ".join(f"{c['comm']}: {fmt(c)}" for c in cs) + "\n")
    with open(os.path.join(OUT, "summary_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    from keep_awake import keep_awake
    with keep_awake("fleet current study"):
        rc = main()
    sys.exit(rc)
