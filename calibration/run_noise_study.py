#!/usr/bin/env python3
"""
run_noise_study.py -- does sensor noise matter for fleet formation-holding,
does training with it help, and how robust are the fleet policies to a
buoyancy change? PROJECT_STATUS.md #58.

    python run_noise_study.py [--workers 6]

Paired design: every condition runs the same episodes (start states, packet
drops, and -- where the noise level matches -- the same sensor errors and
disturbances), so differences are episode by episode.

1. Noise dose-response. Policies trained clean (fleet_policy_seed1-3) and
   with noise (noisy_policies/fleet_policy_noisy_seed1-3, trained with
   `--mode noisy`), under: clean; sensor noise `tank` (the reference logs'
   own noise); `typical` at x0.5, x1 and x2; and the named noisy mode
   (typical x1 + the fitted roll/pitch disturbance).
2. SAC and TD3 (algorithm_policies/, trained clean) under clean and noisy.
3. Buoyancy from -10 N to +10 N, clean, with the physics teammate model
   unaware of the change and aware of it.
Beliefs calibrated and last_beacon; comm L2 @ 1 s and L3a @ 3 s; the standard
8 x 5 grid of start states with 5 packet-loss seeds (3 for the buoyancy sweep).

Writes results/fleet_v2_noise/: episodes.csv, metrics.json, report.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

from build_two_agent_cores import load_calibrated_parameters
from buoyancy import BuoyancyConfig
from disturbance import DEFAULT_FIT_PATH, load_disturbance
from evaluate_comm_degradation_curve import COMM_CONFIGS
from fleet_belief import BeliefSpec
from fleet_env import FleetConfig
from fleet_eval import CellSpec, CommSpec, run_cells
from fleet_stats import paired_by, wilson_ci
from mlflow_run_utils import get_git_commit, get_git_dirty
from optimizer import load_vehicle_spec
from sensor_noise import load_sensor_noise

TRUTH = "calibrated_parameters_fleet_truth.yaml"
GROUPS = {"trained_clean": [f"fleet_policy_seed{k}" for k in (1, 2, 3)],
          "trained_noisy": [f"noisy_policies/fleet_policy_noisy_seed{k}" for k in (1, 2, 3)],
          "sac": [f"algorithm_policies/fleet_sac_seed{k}" for k in (1, 2, 3)],
          "td3": [f"algorithm_policies/fleet_td3_seed{k}" for k in (1, 2, 3)]}
BELIEFS = ("calibrated", "last_beacon")
COMMS = (("L2", 1.0), ("L3a", 3.0))
SEPARATIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
SPEEDS = [-0.7, -0.35, 0.0, 0.35, 0.7]
COMM_SEEDS = [1, 2, 3, 4, 5]
LEVELS = (("clean", None, 0.0, False), ("tank", "tank", 1.0, False), ("typical x0.5", "typical", 0.5, False),
          ("typical x1", "typical", 1.0, False), ("typical x2", "typical", 2.0, False),
          ("noisy mode", "typical", 1.0, True))
BUOYANCY = (-10.0, -5.0, -2.0, 0.0, 2.0, 5.0, 10.0)
OUT = "results/fleet_v2_noise"


def build_cells():
    vehicle = load_vehicle_spec("vehicle_spec.yaml").vehicle
    truth = load_calibrated_parameters(TRUTH, quiet=True)
    beliefs = {"calibrated": BeliefSpec("physics", "calibrated", truth),
               "last_beacon": BeliefSpec("last_beacon", "last_beacon")}
    dist = load_disturbance(DEFAULT_FIT_PATH, 1.0)
    comms = [CommSpec(f"{s}@{i:g}s", COMM_CONFIGS[s]["packet_loss_config"], COMM_CONFIGS[s]["bandwidth_config"],
                      1500.0, i) for s, i in COMMS]
    cells = []

    def add(part, group, path, fc, tags, seeds=COMM_SEEDS):
        for cs in comms:
            for b in BELIEFS:
                cells.append(CellSpec(path, None, cs, beliefs[b], truth, vehicle, fc, SEPARATIONS, SPEEDS, seeds,
                                      tags={"part": part, "policy_group": group, **tags}))

    for label, preset, scale, with_dist in LEVELS:
        fc = FleetConfig(sensor_noise=load_sensor_noise(preset, scale) if preset else None,
                         disturbance=dist if with_dist else None)
        for group in ("trained_clean", "trained_noisy"):
            for p in GROUPS[group]:
                add("noise", group, p, fc, {"level": label})
        if label in ("clean", "noisy mode"):
            for group in ("sac", "td3"):
                for p in GROUPS[group]:
                    add("noise", group, p, fc, {"level": label})
    for off in BUOYANCY:
        for known in (False, True):
            if off == 0.0 and known:
                continue
            fc = FleetConfig(buoyancy=BuoyancyConfig(off, known_to_beliefs=known) if off else None)
            for p in GROUPS["trained_clean"]:
                add("buoyancy", "trained_clean", p, fc, {"buoyancy_n": off, "known": known}, seeds=[1, 2, 3])
    return cells


def rate(rows):
    n = len(rows)
    s = sum(r["success"] for r in rows)
    return (s / n if n else float("nan")), wilson_ci(s, n), n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    missing = [p for ps in GROUPS.values() for p in ps if not os.path.exists(p + ".zip")]
    if missing:
        print(f"ERROR: policies not trained yet: {missing}", file=sys.stderr)
        return 2
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    cells = build_cells()
    print(f"{len(cells)} cells...", flush=True)
    rows = run_cells(cells, workers=args.workers,
                     progress=lambda d, n: print(f"  {d}/{n} cells ({time.time() - t0:.0f}s)", flush=True)
                     if d % 25 == 0 or d == n else None)
    for r in rows:
        r["success"], r["collision"] = int(bool(r["success"])), int(bool(r["collision"]))
        r.setdefault("level", "")
        r.setdefault("buoyancy_n", 0.0)
        r.setdefault("known", False)
    with open(os.path.join(OUT, "episodes.csv"), "w", newline="", encoding="utf-8") as f:
        fields = list(dict.fromkeys(k for r in rows for k in r))
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    noise = [r for r in rows if r["part"] == "noise"]
    buoy = [r for r in rows if r["part"] == "buoyancy"]
    pct = lambda x: f"{100 * x:.0f}%"
    pp = lambda c: (f"{100 * c['success_diff']:+.0f} ({100 * c['success_diff_ci95'][0]:+.0f} to "
                    f"{100 * c['success_diff_ci95'][1]:+.0f})")
    levels = [l for l, *_ in LEVELS]
    m = {"episodes": len(rows), "git_commit": get_git_commit(), "git_uncommitted_changes": get_git_dirty()}

    # 1. dose-response
    dose = {}
    for g in ("trained_clean", "trained_noisy"):
        for lv in levels:
            rs = [r for r in noise if r["policy_group"] == g and r["level"] == lv]
            s, ci, n = rate(rs)
            nav = float(np.mean([r["mean_nav_error_m"] for r in rs]))
            be = np.array([r["mean_belief_error_m"] for r in rs], dtype=float)
            dose[f"{g}|{lv}"] = {"success": s, "ci95": ci, "n": n, "nav_error_m": nav,
                                 "belief_error_m": float(np.nanmean(be)),
                                 "collision": float(np.mean([r["collision"] for r in rs]))}
    vs_clean = {}
    for g in ("trained_clean", "trained_noisy"):
        rs = [r for r in noise if r["policy_group"] == g]
        for lv in levels[1:]:
            (c,) = paired_by(rs, "level", lv, "clean", match=("policy", "belief", "comm", "backend"), group_by=())
            vs_clean[f"{g}|{lv}"] = c
    per_comm = {}
    for lv in levels[1:]:
        rs = [r for r in noise if r["policy_group"] == "trained_clean"]
        per_comm[lv] = paired_by(rs, "level", lv, "clean", match=("policy", "belief", "comm", "backend"),
                                 group_by=("comm", "belief"))
    training = {}
    for lv in levels:
        rs = [r for r in noise if r["level"] == lv]
        (c,) = paired_by(rs, "policy_group", "trained_noisy", "trained_clean", match=("belief", "comm", "backend"),
                         group_by=())
        training[lv] = c
    algos = {}
    for g in ("trained_clean", "sac", "td3"):
        rs = [r for r in noise if r["policy_group"] == g and r["level"] in ("clean", "noisy mode")]
        (c,) = paired_by(rs, "level", "noisy mode", "clean", match=("policy", "belief", "comm", "backend"),
                         group_by=())
        algos[g] = {"clean": rate([r for r in rs if r["level"] == "clean"])[0],
                    "noisy": rate([r for r in rs if r["level"] == "noisy mode"])[0], "noisy_minus_clean": c}
    # 3. buoyancy
    bres = {}
    for off in BUOYANCY:
        for known in ((False, True) if off else (False,)):
            for b in BELIEFS:
                rs = [r for r in buoy if r["buoyancy_n"] == off and r["known"] == known and r["belief"] == b]
                s, ci, n = rate(rs)
                bres[f"{off:+g}|{known}|{b}"] = {"success": s, "ci95": ci, "n": n}
    zero = [r for r in buoy if r["buoyancy_n"] == 0.0]
    bvs = {}
    for off in BUOYANCY:
        if not off:
            continue
        for known in (False, True):
            rs = zero + [r for r in buoy if r["buoyancy_n"] == off and r["known"] == known]
            for r in rs:
                r["_b"] = "zero" if r["buoyancy_n"] == 0.0 else "off"
            for c in paired_by(rs, "_b", "off", "zero", match=("policy", "comm", "backend", "belief"),
                               group_by=("belief",)):
                bvs[f"{off:+g}|{known}|{c['belief']}"] = c
    m.update(dose=dose, noisy_minus_clean=vs_clean, noisy_minus_clean_per_condition=per_comm,
             noise_trained_minus_clean_trained=training, algorithms=algos, buoyancy=bres,
             buoyancy_minus_zero=bvs, elapsed_s=round(time.time() - t0, 1))
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))

    L = ["# Clean vs noisy fleet simulation, and buoyancy (PROJECT_STATUS.md #58)\n",
         f"{len(rows):,} paired episodes, stand-in physics. Comm L2 @ 1 s and L3a @ 3 s, beliefs calibrated and "
         "last_beacon, pooled unless stated.\n",
         "**Read this first:** `tank` is measured from the reference tank logs; `typical` uses assumed sizes for a DVL + AHRS vehicle, not measured on any open-water vehicle. The costs below are for those assumed sizes (x0.5 and x2 bracket them); for your own fleet, fit your sensors with `strathe fit-noise` and rerun.\n",
         "## 1. How much does sensor noise cost?\n",
         "Success for policies trained clean and policies trained with `--mode noisy`, at each noise level. "
         "`noisy mode` = typical sensors + the fitted roll/pitch disturbance.\n",
         "| Level | Trained clean | Noisy − clean level (95% CI) | Trained noisy | Noisy − clean level | "
         "Navigation error | Belief error |", "|---|---|---|---|---|---|---|"]
    for lv in levels:
        a, b = dose[f"trained_clean|{lv}"], dose[f"trained_noisy|{lv}"]
        da = "—" if lv == "clean" else pp(vs_clean[f"trained_clean|{lv}"])
        db = "—" if lv == "clean" else pp(vs_clean[f"trained_noisy|{lv}"])
        L.append(f"| {lv} | {pct(a['success'])} | {da} | {pct(b['success'])} | {db} | "
                 f"{100 * a['nav_error_m']:.0f} cm | {100 * a['belief_error_m']:.0f} cm |")
    L += ["", "Per condition, clean-trained policies (noise level − clean, points):\n",
          "| Level | " + " | ".join(f"{c['comm']} `{c['belief']}`" for c in per_comm[levels[1]]) + " |",
          "|---|" + "---|" * len(per_comm[levels[1]])]
    for lv in levels[1:]:
        L.append(f"| {lv} | " + " | ".join(pp(c) for c in per_comm[lv]) + " |")
    L += ["", "## 2. Does training with the noise help?\n",
          "Trained noisy − trained clean, the same episodes, seeds averaged per episode:\n",
          "| Level | Difference (95% CI) |", "|---|---|"]
    for lv in levels:
        L.append(f"| {lv} | {pp(training[lv])} |")
    L += ["", "## 3. SAC and TD3 under noise\n", "| Policies (trained clean) | Clean | Noisy mode | Noisy − clean (95% CI) |",
          "|---|---|---|---|"]
    for g in ("trained_clean", "sac", "td3"):
        a = algos[g]
        L.append(f"| {'PPO' if g == 'trained_clean' else g.upper()} | {pct(a['clean'])} | {pct(a['noisy'])} | "
                 f"{pp(a['noisy_minus_clean'])} |")
    L += ["", "## 4. Buoyancy\n",
          "Clean-trained PPO, clean sensors. Offset = net buoyancy change from the calibration (− = heavier). "
          "`unaware` / `aware`: whether the calibrated teammate model knows the change.\n",
          "| Offset | calibrated, unaware | vs 0 N | calibrated, aware | vs 0 N | last_beacon | vs 0 N |",
          "|---|---|---|---|---|---|---|"]
    for off in BUOYANCY:
        if off == 0.0:
            L.append(f"| 0 N | {pct(bres['+0|False|calibrated']['success'])} | — | — | — | "
                     f"{pct(bres['+0|False|last_beacon']['success'])} | — |")
            continue
        k = f"{off:+g}"
        L.append(f"| {k} N | {pct(bres[f'{k}|False|calibrated']['success'])} | {pp(bvs[f'{k}|False|calibrated'])} | "
                 f"{pct(bres[f'{k}|True|calibrated']['success'])} | {pp(bvs[f'{k}|True|calibrated'])} | "
                 f"{pct(bres[f'{k}|False|last_beacon']['success'])} | {pp(bvs[f'{k}|False|last_beacon'])} |")
    L.append("")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    from keep_awake import keep_awake
    with keep_awake("noise study"):
        sys.exit(main())
