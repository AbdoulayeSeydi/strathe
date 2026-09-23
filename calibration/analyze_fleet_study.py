#!/usr/bin/env python3
"""
analyze_fleet_study.py -- the checks a skeptical reader should run on a fleet
v2 result, beyond the headline in summary_report.md.

    python analyze_fleet_study.py results/fleet_v2_belief_to_success

Writes <dir>/extended_analysis.md with, for every comm condition:
  1. the headline contrast (default calibrated vs generic) PER TRAINING SEED --
     a pooled effect carried by one policy isn't a property of the method;
  2. every belief model against `last_beacon` (no prediction) -- is a
     predictive model worth having at all?
  3. `oracle` against `calibrated` -- do the policies turn better information
     into better coordination? If perfect information does worse, belief
     accuracy is not what's limiting them.
All contrasts are paired over matched episodes with bootstrap 95% CIs.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

from fleet_stats import paired_contrast


def load_rows(results_dir: str) -> list:
    with open(os.path.join(results_dir, "episodes.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["success"] = r["success"] == "True"
        r["collision"] = r["collision"] == "True"
        r["mean_belief_error_m"] = float(r["mean_belief_error_m"])
    return rows


def _fmt(c) -> str:
    lo, hi = c["success_diff_ci95"]
    star = "" if lo <= 0 <= hi else " *"
    return f"{100 * c['success_diff']:+.0f} [{100 * lo:+.0f}, {100 * hi:+.0f}]{star}"


def analyze(results_dir: str, a: str = "calibrated", b: str = "generic") -> str:
    rows = load_rows(results_dir)
    beliefs = sorted({r["belief"] for r in rows})
    policies = sorted({r["policy"] for r in rows})
    comms = sorted({r["comm"] for r in rows})
    L = [f"# Extended analysis: {os.path.basename(os.path.abspath(results_dir))}\n",
         "Success-rate differences in percentage points, paired over matched episodes, with 95% bootstrap "
         "CIs. `*` = the CI excludes zero.\n"]
    if a in beliefs and b in beliefs:
        L.append(f"## 1. `{a}` minus `{b}`, per training seed\n")
        L.append("| Comm | " + " | ".join(policies) + " |")
        L.append("|---|" + "---|" * len(policies))
        per = {p: {c["comm"]: c for c in paired_contrast([r for r in rows if r["policy"] == p], a, b)}
               for p in policies}
        for cm in comms:
            L.append(f"| {cm} | " + " | ".join(_fmt(per[p][cm]) if cm in per[p] else "n/a" for p in policies) + " |")
        L.append("")
    if "last_beacon" in beliefs:
        others = [x for x in beliefs if x != "last_beacon"]
        L.append("## 2. Each belief model minus `last_beacon` (no prediction)\n")
        L.append("| Comm | " + " | ".join(f"`{x}`" for x in others) + " |")
        L.append("|---|" + "---|" * len(others))
        table = {x: {c["comm"]: c for c in paired_contrast(rows, x, "last_beacon")} for x in others}
        for cm in comms:
            L.append(f"| {cm} | " + " | ".join(_fmt(table[x][cm]) if cm in table[x] else "n/a" for x in others) + " |")
        L.append("")
    if "oracle" in beliefs and a in beliefs:
        L.append(f"## 3. `oracle` (perfect information) minus `{a}`\n")
        L.append("If perfect information does WORSE, the policies are not limited by belief accuracy: they "
                 "are tuned to the error patterns they were trained on.\n")
        for c in paired_contrast(rows, "oracle", a):
            L.append(f"- {c['comm']}: {_fmt(c)}")
        L.append("")
    text = "\n".join(L) + "\n"
    with open(os.path.join(results_dir, "extended_analysis.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dir")
    ap.add_argument("--a", default="calibrated")
    ap.add_argument("--b", default="generic")
    args = ap.parse_args(argv)
    print(analyze(args.results_dir, args.a, args.b))
    return 0


if __name__ == "__main__":
    sys.exit(main())
