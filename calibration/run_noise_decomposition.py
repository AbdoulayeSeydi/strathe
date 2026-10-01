#!/usr/bin/env python3
"""
run_noise_decomposition.py -- what part of sensor noise costs formation-holding?
PROJECT_STATUS.md #60.

    python run_noise_decomposition.py [--workers 6]

An exploratory follow-up to the noise study (#58), NOT a study with criteria
fixed in advance: it was run after a quick check showed every mix of the
`typical` sensors costing the same 10 points. Each variant switches on one
part of the sensor model, on the same paired episodes:
  - the starting position fix error alone (perfect sensors), at 0.05-0.3 m
  - the typical sensors with a perfect starting fix: all of them, white noise only, biases only
  - typical as shipped (sensors + 0.2 m fix), as in #58
  - open_water (SOLAQUA: measured white noise and fix error, assumed biases), with and without its fix
Shipped PPO fleet_policy_seed1-3; beliefs calibrated and last_beacon; comm L2 @ 1 s and L3a @ 3 s; the
standard 8 x 5 start grid with 5 packet-loss seeds. Writes results/fleet_v2_noise/decomposition.md,
decomposition_metrics.json and decomposition.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from build_two_agent_cores import load_calibrated_parameters
from evaluate_comm_degradation_curve import COMM_CONFIGS
from fleet_belief import BeliefSpec
from fleet_env import FleetConfig
from fleet_eval import CellSpec, CommSpec, run_cells
from fleet_stats import paired_by, wilson_ci
from mlflow_run_utils import get_git_commit, get_git_dirty
from optimizer import load_vehicle_spec
from sensor_noise import SensorNoiseConfig, load_sensor_noise

TRUTH = "calibrated_parameters_fleet_truth.yaml"
POLICIES = [f"fleet_policy_seed{k}" for k in (1, 2, 3)]
BELIEFS = ("calibrated", "last_beacon")
COMMS = (("L2", 1.0), ("L3a", 3.0))
SEPARATIONS = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
SPEEDS = [-0.7, -0.35, 0.0, 0.35, 0.7]
COMM_SEEDS = [1, 2, 3, 4, 5]
OUT = "results/fleet_v2_noise"
WHITE = ("velocity_std_mps", "velocity_dropout_prob", "gyro_std_rps", "attitude_std_rad", "heading_std_rad",
         "depth_std_m")
BIAS = ("velocity_bias_mps", "gyro_bias_rps", "attitude_bias_rad", "heading_bias_rad")


def part(src: SensorNoiseConfig, keep, fix: float, name: str) -> SensorNoiseConfig:
    """A dead-reckoning sensor model with only `keep` taken from `src` and a starting-fix error of `fix` m."""
    d = {k: (getattr(src, k) if k in keep else 0.0) for k in WHITE + BIAS}
    c = SensorNoiseConfig(position="dead_reckoning", position_std_m=fix, source=name, **d)
    c.validate()
    return c


def variants() -> list:
    """[(label, group, SensorNoiseConfig or None)] in report order."""
    typ, ow = load_sensor_noise("typical"), load_sensor_noise("open_water")
    out = [("clean", "reference", None),
           ("typical (as #58)", "as shipped", typ),
           ("open_water (SOLAQUA)", "as shipped", ow)]
    for fix in (0.05, 0.1, 0.2, 0.3):
        out.append((f"starting fix {fix:g} m only", "starting fix", part(typ, (), fix, f"fix_{fix:g}")))
    out += [("typical sensors, perfect fix", "sensors", part(typ, WHITE + BIAS, 0.0, "typical_nofix")),
            ("typical white noise only, perfect fix", "sensors", part(typ, WHITE, 0.0, "typical_white")),
            ("typical biases only, perfect fix", "sensors", part(typ, BIAS, 0.0, "typical_bias")),
            ("open_water sensors, perfect fix", "sensors", part(ow, WHITE + BIAS, 0.0, "open_water_nofix"))]
    return out


def build_cells():
    vehicle = load_vehicle_spec("vehicle_spec.yaml").vehicle
    truth = load_calibrated_parameters(TRUTH, quiet=True)
    beliefs = {"calibrated": BeliefSpec("physics", "calibrated", truth),
               "last_beacon": BeliefSpec("last_beacon", "last_beacon")}
    comms = [CommSpec(f"{s}@{i:g}s", COMM_CONFIGS[s]["packet_loss_config"], COMM_CONFIGS[s]["bandwidth_config"],
                      1500.0, i) for s, i in COMMS]
    cells = []
    for label, group, noise in variants():
        fc = FleetConfig(sensor_noise=noise)
        for cs in comms:
            for p in POLICIES:
                for b in BELIEFS:
                    cells.append(CellSpec(p, None, cs, beliefs[b], truth, vehicle, fc, SEPARATIONS, SPEEDS,
                                          COMM_SEEDS, tags={"variant": label, "group": group}))
    return cells


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    cells = build_cells()
    print(f"{len(cells)} cells x {len(SEPARATIONS) * len(SPEEDS) * len(COMM_SEEDS)} episodes...", flush=True)
    rows = run_cells(cells, workers=args.workers,
                     progress=lambda d, n: print(f"  {d}/{n} cells ({time.time() - t0:.0f}s)", flush=True)
                     if d % 20 == 0 or d == n else None)
    for r in rows:
        r["success"], r["collision"] = int(bool(r["success"])), int(bool(r["collision"]))
    labels = [v[0] for v in variants()]
    res = []
    for lab in labels:
        rs = [r for r in rows if r["variant"] == lab]
        s = sum(r["success"] for r in rs)
        entry = {"variant": lab, "n": len(rs), "success": s / len(rs), "success_ci95": wilson_ci(s, len(rs)),
                 "mean_nav_error_m": float(np.mean([r["mean_nav_error_m"] for r in rs])),
                 "collision": float(np.mean([r["collision"] for r in rs]))}
        if lab != "clean":
            c = paired_by(rows, "variant", lab, "clean", match=("policy", "belief", "comm", "backend"), group_by=())
            entry["diff_vs_clean"] = c[0]["success_diff"]
            entry["diff_ci95"] = c[0]["success_diff_ci95"]
        res.append(entry)
    elapsed = (time.time() - t0) / 60
    m = {"episodes": len(rows), "elapsed_min": round(elapsed, 1), "git_commit": get_git_commit(),
         "git_uncommitted_changes": get_git_dirty(), "variants": res}
    with open(os.path.join(OUT, "decomposition_metrics.json"), "w") as f:
        json.dump(m, f, indent=2)

    pp = lambda e: (f"{100 * e['diff_vs_clean']:+.0f} ({100 * e['diff_ci95'][0]:+.0f} to "
                    f"{100 * e['diff_ci95'][1]:+.0f})") if "diff_vs_clean" in e else "—"
    L = ["# What part of sensor noise costs formation-holding? (PROJECT_STATUS.md #60)\n",
         f"{time.strftime('%Y-%m-%d')} · {len(rows):,} paired episodes · {elapsed:.0f} min · "
         "`python run_noise_decomposition.py`\n",
         "**Exploratory follow-up to the noise study (#58), not a study with criteria fixed in advance.** It was "
         "run after a quick check showed every mix of the `typical` sensors costing the same 10 points. Each row "
         "switches on one part of the sensor model on the same episodes: shipped PPO seeds 1-3, beliefs "
         "calibrated and last_beacon, L2 @ 1 s and L3a @ 3 s, the standard start grid with 5 packet-loss seeds.\n",
         "| Variant | Success | Noise − clean, pts (95% CI) | Mean navigation error |", "|---|---|---|---|"]
    for e in res:
        L.append(f"| {e['variant']} | {100 * e['success']:.0f}% | {pp(e)} | {100 * e['mean_nav_error_m']:.0f} cm |")
    by = {e["variant"]: e for e in res}
    L += ["",
          "## What it shows\n",
          f"- **The starting position fix is the cost.** A 0.2 m starting-fix error with perfect sensors costs "
          f"{pp(by['starting fix 0.2 m only'])} points; the `typical` sensors with a perfect fix cost "
          f"{pp(by['typical sensors, perfect fix'])}. Each vehicle's fix error is independent, so the relative "
          "position error is about 1.4 times larger, against a ±0.2 m formation tolerance; over a 20 s episode, "
          "dead-reckoning drift barely builds up.",
          f"- **The noise study's (#58) explanation was wrong.** It said realistic sensors cost 10 points through "
          "dead-reckoning drift. The 10 points are real, but they come from the 0.2 m starting fix inside the "
          "`typical` preset, not from the sensors or their drift.",
          f"- **Measured open-water sensors (SOLAQUA) cost nothing on their own** "
          f"({pp(by['open_water sensors, perfect fix'])}); with the measured USBL starting fix (0.17 m) they cost "
          f"{pp(by['open_water (SOLAQUA)'])}, close to `typical` because the assumed 0.2 m fix was about right.",
          "- **What to do with it:** report the starting-fix error with any noisy result, set it to your "
          "positioning's accuracy (`--start-fix-error`), and treat relative positioning between the vehicles "
          "(acoustic ranging, a shared USBL) as the lever.\n",
          "![Cost by part of the sensor model](decomposition.png)\n"]
    with open(os.path.join(OUT, "decomposition.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2))
    rows_ = [e for e in res if "diff_vs_clean" in e][::-1]
    ys = np.arange(len(rows_))
    vals = [100 * e["diff_vs_clean"] for e in rows_]
    lo = [100 * (e["diff_vs_clean"] - e["diff_ci95"][0]) for e in rows_]
    hi = [100 * (e["diff_ci95"][1] - e["diff_vs_clean"]) for e in rows_]
    colors = ["#c44e52" if "fix" in e["variant"] and "perfect" not in e["variant"] or "as #58" in e["variant"]
              or "SOLAQUA" in e["variant"] else "#4c72b0" for e in rows_]
    ax.barh(ys, vals, xerr=[lo, hi], color=colors, capsize=2)
    ax.set_yticks(ys, [e["variant"] for e in rows_], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Success vs clean simulation, points (95% CI)")
    ax.set_title("Red: a starting-fix error is on. Blue: perfect starting fix.", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "decomposition.png"), dpi=130)
    plt.close(fig)
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    from keep_awake import keep_awake
    with keep_awake("noise decomposition"):  # PROJECT_STATUS.md #49
        rc = main()
    sys.exit(rc)
