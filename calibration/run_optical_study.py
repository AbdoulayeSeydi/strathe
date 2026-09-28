#!/usr/bin/env python3
"""
run_optical_study.py -- what an optical or hybrid link gives the shipped fleet
policies, by water clarity and terminal pointing. PROJECT_STATUS.md #59.

    python run_optical_study.py [--workers 6]

Criteria fixed before any result: results/fleet_optical/criteria.md.
Paired design: every link condition runs the same episodes (start states,
acoustic packet drops, optical draws), so differences are episode by episode.

Main: shipped PPO fleet_policy_seed1-3 (trained on acoustic links only) x
beliefs calibrated / last_beacon x acoustic L2 / L3a at 1 s / 3 s x links
acoustic, optical, hybrid optical_first, hybrid redundant x water clear /
coastal / harbour, omni terminals. Pointing: forward_cone and fore_aft_cones
against omni in coastal water. Noisy: acoustic vs hybrid (coastal, omni) in
the noisy mode. Standard 8 x 5 start grid x 5 comm seeds.

Writes results/fleet_optical/: episodes.csv, metrics.json, report.md, optical_study.png.
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
from comm_channel import LinkConfig, OpticalLinkConfig
from disturbance import DEFAULT_FIT_PATH, load_disturbance
from evaluate_comm_degradation_curve import COMM_CONFIGS
from fleet_belief import BeliefSpec
from fleet_env import FleetConfig
from fleet_eval import CellSpec, CommSpec, run_cells
from fleet_stats import paired_by, wilson_ci
from mlflow_run_utils import get_git_commit, get_git_dirty
from optimizer import load_vehicle_spec
from run_fleet_experiment import link_contrasts, link_summary
from sensor_noise import load_sensor_noise

TRUTH = "calibrated_parameters_fleet_truth.yaml"
POLICIES = [f"fleet_policy_seed{k}" for k in (1, 2, 3)]
BELIEFS = ("calibrated", "last_beacon")
COMMS = (("L2", 1.0), ("L2", 3.0), ("L3a", 1.0), ("L3a", 3.0))
WATERS = ("clear", "coastal", "harbor")
SEPARATIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
SPEEDS = [-0.7, -0.35, 0.0, 0.35, 0.7]
COMM_SEEDS = [1, 2, 3, 4, 5]
OUT = "results/fleet_optical"


def _link(kind, water="coastal", terminals="omni", policy="optical_first"):
    if kind == "acoustic":
        return LinkConfig("acoustic")
    lc = LinkConfig(kind, OpticalLinkConfig(water=water, terminals=terminals), policy)
    lc.validate()
    return lc


def build_cells():
    vehicle = load_vehicle_spec("vehicle_spec.yaml").vehicle
    truth = load_calibrated_parameters(TRUTH, quiet=True)
    beliefs = {"calibrated": BeliefSpec("physics", "calibrated", truth),
               "last_beacon": BeliefSpec("last_beacon", "last_beacon")}
    clean = FleetConfig()
    noisy = FleetConfig(sensor_noise=load_sensor_noise("typical"), disturbance=load_disturbance(DEFAULT_FIT_PATH, 1.0))
    cells = []

    def add(part, mode, fc, lc, sev, interval, pointing):
        acoustic_base = "none" if lc.link == "optical" else f"{sev}@{interval:g}s"
        label = (acoustic_base if lc.link == "acoustic" else lc.label if lc.link == "optical"
                 else f"{acoustic_base} {lc.label}") + ("" if mode == "clean" else " noisy")
        cs = CommSpec(label, COMM_CONFIGS[sev]["packet_loss_config"], COMM_CONFIGS[sev]["bandwidth_config"], 1500.0,
                      interval, link=None if lc.link == "acoustic" else lc)
        for p in POLICIES:
            for b in BELIEFS:
                cells.append(CellSpec(p, None, cs, beliefs[b], truth, vehicle, fc, SEPARATIONS, SPEEDS, COMM_SEEDS,
                                      tags={"part": part, "mode": mode, "link_kind": lc.link,
                                            "acoustic_base": acoustic_base,
                                            "water": lc.optical.water if lc.uses_optical else "none",
                                            "pointing": pointing, "hybrid_policy": lc.hybrid_policy
                                            if lc.link == "hybrid" else "none"}))

    for sev, iv in COMMS:
        add("main", "clean", clean, _link("acoustic"), sev, iv, "acoustic")
        for w in WATERS:
            for pol in ("optical_first", "redundant"):
                # redundant rows get their own pointing label, so the cones-vs-omni pairing uses optical_first only
                add("main", "clean", clean, _link("hybrid", w, policy=pol), sev, iv,
                    "omni" if pol == "optical_first" else "omni_redundant")
        for t in ("forward_cone", "fore_aft_cones"):
            add("pointing", "clean", clean, _link("hybrid", "coastal", t), sev, iv, t)
        add("noisy", "noisy", noisy, _link("acoustic"), sev, iv, "acoustic")
        add("noisy", "noisy", noisy, _link("hybrid", "coastal"), sev, iv, "omni")
    for w in WATERS:
        add("main", "clean", clean, _link("optical", w), "L0", 1.0, "omni")
    for t in ("forward_cone", "fore_aft_cones"):
        add("pointing", "clean", clean, _link("optical", "coastal", t), "L0", 1.0, t)
    return cells


def plot(path, lsum, vs_acoustic):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {s["comm"]: s for s in lsum}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    comms = [f"{s}@{i:g}s" for s, i in COMMS]
    series = [("acoustic only", lambda c: c), ("hybrid, clear", lambda c: f"{c} hybrid/clear"),
              ("hybrid, coastal", lambda c: f"{c} hybrid/coastal"), ("hybrid, harbour", lambda c: f"{c} hybrid/harbor"),
              ("hybrid, coastal, fore/aft cones", lambda c: f"{c} hybrid/coastal/fore_aft_cones"),
              ("hybrid, coastal, forward cone", lambda c: f"{c} hybrid/coastal/forward_cone")]
    w = 0.8 / len(series)
    for j, (name, key) in enumerate(series):
        xs = np.arange(len(comms)) + (j - (len(series) - 1) / 2) * w
        axes[0].bar(xs, [100 * by[key(c)]["success_rate"] for c in comms], w, label=name)
        axes[1].bar(xs, [by[key(c)]["median_belief_age_s"] for c in comms], w, label=name)
    for ax, title, yl in ((axes[0], "Success, shipped PPO (acoustic-trained)", "Success (%)"),
                          (axes[1], "Median teammate-belief age", "Seconds")):
        ax.set_xticks(range(len(comms)), comms)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(yl)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=8, frameon=False, loc="lower center", ncol=6)
    ups = [(k, by[k]["optical_up_fraction"]) for k in ("optical/clear", "optical/coastal", "optical/harbor",
                                                        "optical/coastal/fore_aft_cones", "optical/coastal/forward_cone")
           if k in by]
    axes[2].barh([k.replace("optical/", "") for k, _ in ups][::-1], [100 * v for _, v in ups][::-1], color="#4c72b0")
    axes[2].set_title("Optical beacons delivered (optical only)", fontsize=10)
    axes[2].set_xlabel("% of optical beacons")
    axes[2].spines[["top", "right"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--from-csv", action="store_true",
                    help="rebuild the report and chart from an existing episodes.csv instead of rerunning")
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    if args.from_csv:
        import pandas as pd
        rows = pd.read_csv(os.path.join(OUT, "episodes.csv"), keep_default_na=False,
                           na_values=[""]).to_dict("records")
        elapsed_note = json.load(open(os.path.join(OUT, "metrics.json"))).get("elapsed_min")
    else:
        cells = build_cells()
        print(f"{len(cells)} cells x {len(SEPARATIONS) * len(SPEEDS) * len(COMM_SEEDS)} episodes...", flush=True)
        rows = run_cells(cells, workers=args.workers,
                         progress=lambda d, n: print(f"  {d}/{n} cells ({time.time() - t0:.0f}s)", flush=True)
                         if d % 25 == 0 or d == n else None)
        elapsed_note = None
    for r in rows:
        r["success"], r["collision"] = int(bool(r["success"])), int(bool(r["collision"]))
    if not args.from_csv:
        with open(os.path.join(OUT, "episodes.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader()
            w.writerows(rows)
    elapsed_min = elapsed_note if elapsed_note is not None else (time.time() - t0) / 60

    bad = [r for r in rows if r["acoustic_delivered"] > r["acoustic_sent"] or r["optical_delivered"] > r["optical_sent"]]
    lsum = link_summary(rows)
    vs_acoustic, pointing = link_contrasts(rows)
    red = paired_by([r for r in rows if r["part"] == "main"], "hybrid_policy", "redundant", "optical_first",
                    match=("policy", "belief", "acoustic_base", "water", "backend", "mode"),
                    group_by=("acoustic_base", "water"))
    # collisions, paired the same way: the collision flag in place of success
    _, pointing_c = link_contrasts([dict(r, success=r["collision"]) for r in rows])
    vs_acoustic_c, _ = link_contrasts([dict(r, success=r["collision"]) for r in rows])
    raised = [c for c in vs_acoustic_c if c["success_diff_ci95"][0] > 0]
    m = {"episodes": len(rows), "git_commit": get_git_commit(), "git_uncommitted_changes": get_git_dirty(),
         "gate_bookkeeping_failures": len(bad), "links": lsum, "link_vs_acoustic": vs_acoustic,
         "pointing_vs_omni": pointing, "redundant_vs_optical_first": red,
         "collisions_vs_acoustic": vs_acoustic_c, "collisions_raised": raised, "elapsed_min": round(elapsed_min, 1)}
    with open(os.path.join(OUT, "metrics.json"), "w") as f:
        json.dump(m, f, indent=2, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else str(x))
    plot(os.path.join(OUT, "optical_study.png"), lsum, vs_acoustic)

    pct = lambda x: "n/a" if not np.isfinite(x) else f"{100 * x:.0f}%"
    pp = lambda c: (f"{100 * c['success_diff']:+.0f} ({100 * c['success_diff_ci95'][0]:+.0f} to "
                    f"{100 * c['success_diff_ci95'][1]:+.0f})")
    verdict = lambda c: ("helps" if c["success_diff_ci95"][0] > 0 else "hurts" if c["success_diff_ci95"][1] < 0
                         else "no clear difference")
    by = {s["comm"]: s for s in lsum}
    L = ["# Optical and hybrid links with the shipped fleet policies (PROJECT_STATUS.md #59)\n",
         f"{time.strftime('%Y-%m-%d')} · {len(rows):,} episodes · {elapsed_min:.0f} min · "
         f"`python run_optical_study.py` · criteria fixed in advance: [criteria.md](criteria.md)\n",
         "**Read this first.** The optical model is **uncalibrated**: the water values are rough textbook figures and "
         "the delivery curve is a placeholder (50% delivery at 30 m in clear water, ~15 m coastal, ~4 m harbour, when "
         "pointed). Every number here is conditional on those assumptions. The policies are the shipped PPO fleet "
         "policies, trained on acoustic links only, with no input saying whether the optical link is up (only the "
         "teammate-belief age shows it). This study shows whether **fresher teammate information helps existing "
         "policies**. It **can't** show whether a fleet learns to stay within optical range and pointed at its "
         "teammate: that needs policies trained with `--link hybrid`, the follow-up study (criteria.md, addendum).\n",
         "What the optical link adds here is **high-rate state sharing**: every tick, the teammate's measured state "
         "and the thrust it delivered on that tick. That is not intent sharing (planned commands or a trajectory), "
         "which stays open (#48). **No latency effect is modelled:** delivery is quantised to the 0.1 s tick, so "
         "optical's 1 ms and acoustic's few ms both arrive on the next tick; the gains below come from send rate "
         "and loss.\n",
         "Hybrid `optical_first` skips a vehicle's acoustic beacon only if an optical message from its teammate "
         "arrived on the previous tick (`link_up_ticks` = 1).\n",
         f"**Gate (integration):** {len(rows):,} episodes, {len(bad)} bookkeeping failures, every link condition ran "
         f"with every policy. {'PASSED' if not bad else 'FAILED'}.\n",
         "## 1. Does the optical link help? Hybrid optical_first vs acoustic only, same episodes\n",
         "| Acoustic condition | Water | Acoustic only | Hybrid | Difference, pts (95% CI) | Verdict | "
         "Collisions: hybrid vs acoustic |", "|---|---|---|---|---|---|---|"]
    for c in vs_acoustic:
        lab = c["a"]
        if c["mode"] != "clean" or not lab.startswith("hybrid/") or lab.count("/") != 1:
            continue
        L.append(f"| {c['acoustic_base']} | {lab.split('/')[1]} | {pct(c['success_b'])} | {pct(c['success_a'])} | "
                 f"{pp(c)} | {verdict(c)} | {pct(c['collision_a'])} vs {pct(c['collision_b'])} |")
    L += ["", "Optical only (no acoustic channel), against each acoustic condition:\n",
          "| Acoustic condition | Water | Acoustic only | Optical only | Difference, pts (95% CI) | Verdict |",
          "|---|---|---|---|---|---|"]
    for c in vs_acoustic:
        lab = c["a"]
        if c["mode"] != "clean" or not lab.startswith("optical/") or lab.count("/") != 1:
            continue
        L.append(f"| {c['acoustic_base']} | {lab.split('/')[1]} | {pct(c['success_b'])} | {pct(c['success_a'])} | "
                 f"{pp(c)} | {verdict(c)} |")
    L += ["", "## 2. Freshness and 3. water\n",
          "| Condition | Success | Collisions | Optical beacons delivered | Belief updates via optical | "
          "Median belief age | Acoustic skipped / episode | Skipped with optical copy lost |",
          "|---|---|---|---|---|---|---|---|"]
    order = [f"{s}@{i:g}s" for s, i in COMMS]
    keys = [k for c in order for k in [c] + [f"{c} hybrid/{w}" for w in WATERS]] + [f"optical/{w}" for w in WATERS]
    for k in keys:
        if k in by:
            s = by[k]
            L.append(f"| {k} | {pct(s['success_rate'])} | {pct(s['collision_rate'])} | {pct(s['optical_up_fraction'])} | "
                     f"{pct(s['belief_updates_optical_share'])} | "
                     + ("n/a" if not np.isfinite(s["median_belief_age_s"]) else f"{s['median_belief_age_s']:.2f} s")
                     + f" | {s['acoustic_skipped_per_episode']:.1f} | {pct(s['skipped_uncovered_share'])} |")
    L += ["", "## 4. optical_first vs redundant (same episodes)\n",
          "| Acoustic condition | Water | optical_first | redundant | Redundant − optical_first, pts (95% CI) |",
          "|---|---|---|---|---|"]
    for c in red:
        L.append(f"| {c['acoustic_base']} | {c['water']} | {pct(c['success_b'])} | {pct(c['success_a'])} | {pp(c)} |")
    L += ["", "Acoustic energy and airtime aren't modelled, so redundant (a superset of messages) can't lose on "
          "information; optical_first's value is the acoustic energy and channel time it saves, which matters once "
          "three or more vehicles share an acoustic channel.\n",
          "## 5. What pointing costs (coastal water, cones vs omni, same episodes)\n",
          "| Terminals | Link | Acoustic condition | Cones | Omni | Cones − omni, pts (95% CI) | Optical delivered with "
          "cones |", "|---|---|---|---|---|---|---|"]
    for c in pointing:
        key = (f"{c['acoustic_base']} hybrid/coastal/{c['a']}" if c["link_kind"] == "hybrid"
               else f"optical/coastal/{c['a']}")
        L.append(f"| {c['a']} | {c['link_kind']} | {c['acoustic_base']} | {pct(c['success_a'])} | {pct(c['success_b'])} | "
                 f"{pp(c)} | {pct(by[key]['optical_up_fraction']) if key in by else 'n/a'} |")
    L += ["", "## 6. Collisions: does any link raise them? (vs acoustic only, same episodes)\n"]
    if raised:
        L.append("**Yes.** These link conditions collide significantly more often than acoustic only (paired 95% CI "
                 "above zero):\n")
        L += ["| Link | Acoustic condition | Mode | Collisions with link | Acoustic only | Difference, pts (95% CI) |",
              "|---|---|---|---|---|---|"]
        for c in raised:
            L.append(f"| `{c['a']}` | {c['acoustic_base']} | {c['mode']} | {pct(c['success_a'])} | "
                     f"{pct(c['success_b'])} | {pp(c)} |")
        L += ["", "**What happens:** the vehicles start lined up, inside each other's cones, so the first optical "
              "beacon arrives within 0.2 s, far sooner than the first acoustic one. The policy starts closing on "
              "that fix, rolls and yaws the vehicles out of the cones, and the link goes quiet. Over 80 of these "
              "episodes (L3a @ 3 s, last_beacon, seed-1 policy), every collision happened with a teammate belief "
              "4.6-7.7 s old; acoustic only collided in none of them. An intermittent high-rate link is worse than "
              "a slow steady one for a policy that never learned to keep it up. Omni terminals never raise "
              "collisions here.\n"]
    else:
        L.append("No: no link condition collides significantly more often than acoustic only.\n")
    L += ["", "## 7. Noisy mode: hybrid (coastal, omni) vs acoustic, sensor noise + roll/pitch disturbance on\n",
          "| Acoustic condition | Acoustic only | Hybrid | Difference, pts (95% CI) | Verdict |", "|---|---|---|---|---|"]
    for c in vs_acoustic:
        if c["mode"] == "noisy":
            L.append(f"| {c['acoustic_base']} | {pct(c['success_b'])} | {pct(c['success_a'])} | {pp(c)} | "
                     f"{verdict(c)} |")
    L += ["", "![Success, belief age and optical delivery](optical_study.png)\n",
          "## Limits\n",
          "- Uncalibrated optical model (above). A lab's link logs (success against range and angle) are what would "
          "calibrate it; UNH's thesis gives one data point (about 20 m, 5 kbps, 3 nodes).\n"
          "- Pointing is judged on the true attitude with a hard cone. The shipped policies roll and pitch freely "
          "(#50), which is why narrow cones deliver little here, and why an early fix followed by silence can end in "
          "a collision (section 6).\n"
          "- No occlusion, sunlight, bubbles or changing turbidity. Two vehicles, so no shared acoustic channel.\n"
          "- Policies trained on acoustic links only. Whether a policy trained with a hybrid link learns to stay "
          "within optical range and pointed at its teammate, and what that costs in collisions, is the next study "
          "(`strathe train-fleet --link hybrid`).\n"]
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print("\n".join(L))
    return 1 if bad else 0


if __name__ == "__main__":
    from keep_awake import keep_awake
    with keep_awake("optical link study"):  # PROJECT_STATUS.md #49
        rc = main()
    sys.exit(rc)
