#!/usr/bin/env python3
"""
run_specialized_study.py -- fleet v2 follow-up: each belief model paired with
a policy TRAINED FOR IT.

    python run_specialized_study.py [--seeds 1 2] [--workers 7]

Why: the main study (fleet_study.yaml) used policies trained on a MIXTURE of
belief models whose observation doesn't say which one is active, so a policy
applies one average staleness correction to all of them. That favours the
last-beacon belief (whose "age" really does mean "this is stale") and even
made perfect information (oracle) score worse than a physics model
(PROJECT_STATUS.md #47). A real fleet is trained with its own onboard
predictor. Here, policy specialized_policies/fleet_policy_<belief>_seed<k>
(train_fleet_policy.py --belief <belief>) is evaluated only with <belief>.

Pairing: every condition runs the same episode grid (start states x comm
seeds -> identical packet-drop patterns), and rows are matched across belief
models by training seed index, so contrasts stay paired. They now compare
(belief model + its own policy) systems, which is the deployment question.

Writes results/fleet_v2_specialized/: episodes.csv, summary.csv,
metrics.json, summary_report.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

from build_two_agent_cores import load_calibrated_parameters
from evaluate_comm_degradation_curve import COMM_CONFIGS
from fleet_belief import BeliefSpec
from fleet_env import FleetConfig
from fleet_eval import CellSpec, CommSpec, run_cells
from fleet_stats import paired_contrast, summarize
from mlflow_run_utils import get_git_commit, get_git_dirty
from optimizer import initial_parameters, load_vehicle_spec
from keep_awake import keep_awake

BELIEFS = ("calibrated", "generic", "last_beacon", "oracle")
SEVERITIES = ("L0", "L1", "L2", "L3a")
INTERVALS = (1.0, 3.0)
SEPARATIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
SPEEDS = [-0.7, -0.35, 0.0, 0.35, 0.7]
COMM_SEEDS = [1, 2, 3, 4, 5]
OUT = "results/fleet_v2_specialized"


def build(seeds):
    spec = load_vehicle_spec("vehicle_spec.yaml")
    truth = load_calibrated_parameters("calibrated_parameters_fleet_truth.yaml")
    generic = initial_parameters(spec)
    specs = {"calibrated": BeliefSpec("physics", "calibrated", truth),
             "generic": BeliefSpec("physics", "generic", generic),
             "last_beacon": BeliefSpec("last_beacon", "last_beacon"),
             "oracle": BeliefSpec("oracle", "oracle")}
    cells = []
    for seed in seeds:
        for b in BELIEFS:
            path = f"specialized_policies/fleet_policy_{b}_seed{seed}"
            if not os.path.exists(path + ".zip"):
                raise FileNotFoundError(f"{path}.zip missing -- train it with: python train_fleet_policy.py "
                                        f"--belief {b} --seed {seed} --out {path}")
            for sev in SEVERITIES:
                for iv in INTERVALS:
                    cs = CommSpec(f"{sev}@{iv:g}s", COMM_CONFIGS[sev]["packet_loss_config"],
                                  COMM_CONFIGS[sev]["bandwidth_config"], 1500.0, iv)
                    cells.append(CellSpec(path, "ppo", cs, specs[b], truth, spec.vehicle, FleetConfig(),
                                          SEPARATIONS, SPEEDS, COMM_SEEDS, tags={"training_seed": f"seed{seed}"}))
    return cells


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    cells = build(args.seeds)
    print(f"{len(cells)} cells x {len(SEPARATIONS) * len(SPEEDS) * len(COMM_SEEDS)} episodes", flush=True)
    rows = run_cells(cells, workers=args.workers,
                     progress=lambda d, n: print(f"  {d}/{n} cells ({time.time() - t0:.0f}s)", flush=True))
    for r in rows:
        r["policy_file"] = r["policy"]
        r["policy"] = r["training_seed"]  # pair across belief models by training seed
    os.makedirs(OUT, exist_ok=True)
    summary = summarize(rows, by=("comm", "belief"))
    contrasts = {f"{a}_vs_{b}": paired_contrast(rows, a, b) for a, b in
                 (("calibrated", "generic"), ("calibrated", "last_beacon"), ("generic", "last_beacon"),
                  ("oracle", "calibrated"), ("oracle", "last_beacon"))}
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
                   "git_uncommitted_changes": get_git_dirty(), "seeds": args.seeds,
                   "elapsed_s": round(time.time() - t0, 1)}, f, indent=2, default=float)

    def fmt(c):
        lo, hi = c["success_diff_ci95"]
        return f"{100 * c['success_diff']:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}]" + ("" if lo <= 0 <= hi else " *")
    comms = sorted({r["comm"] for r in rows})
    L = ["# Fleet v2: belief models with policies trained for them\n",
         f"{len(rows)} episodes; training seeds {args.seeds}; paired by seed, start state and packet losses. "
         "Differences are success-rate points with 95% bootstrap CIs; `*` = CI excludes zero.\n",
         "## Success rate\n", "| Comm | " + " | ".join(f"`{b}`" for b in BELIEFS) + " |", "|---|" + "---|" * len(BELIEFS)]
    by = {(s["comm"], s["belief"]): s for s in summary}
    for cm in comms:
        L.append(f"| {cm} | " + " | ".join(f"{100 * by[(cm, b)]['success_rate']:.0f}%" for b in BELIEFS) + " |")
    L += ["", "## Paired differences\n", "| Comm | " + " | ".join(contrasts) + " |", "|---|" + "---|" * len(contrasts)]
    tabs = {k: {c["comm"]: c for c in v} for k, v in contrasts.items()}
    for cm in comms:
        L.append(f"| {cm} | " + " | ".join(fmt(tabs[k][cm]) for k in contrasts) + " |")
    with open(os.path.join(OUT, "summary_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    with keep_awake("fleet specialized study"):  # PROJECT_STATUS.md #49
        rc = main()
    sys.exit(rc)
