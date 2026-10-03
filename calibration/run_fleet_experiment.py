#!/usr/bin/env python3
"""
run_fleet_experiment.py -- run a Strathe fleet v2 experiment from one YAML file.

    python run_fleet_experiment.py                          # default_experiment.yaml
    python run_fleet_experiment.py --config my_exp.yaml

Two simulated BlueROV2s hold formation while exchanging state over a lossy
acoustic link. Between beacons each vehicle predicts where its teammate is
with a belief model; the experiment compares belief models (for example a
calibrated physics model against generic default physics) episode by
episode, under identical initial conditions and identical packet losses.

Outputs in output_dir:
    summary_report.md   plain-language results, confidence intervals, caveats
    summary.csv         one row per (comm condition, belief) -- pooled over policies
    episodes.csv        every episode
    metrics.json        everything above, machine-readable, plus the statistics
    metadata.json       git commit (+ whether the tree had uncommitted changes),
                        package versions, resolved config
    experiment.yaml     the resolved config -- pass it back with --config to rerun
    plots/              success rate and belief error by condition
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime
from typing import Optional

import numpy as np
import yaml

from build_two_agent_cores import load_calibrated_parameters
from fleet_belief import BeliefSpec, interpolate_params
from disturbance import load_disturbance
from fleet_env import FleetConfig
from fleet_eval import CellSpec, CommSpec, run_cells
from fleet_experiment_schema import (ConfigError, PhysicsAlphaBelief, PhysicsParamsBelief,
                                     belief_label, load_fleet_experiment)
from fleet_stats import belief_success_link, dose_response, paired_by, paired_contrast, summarize
from formation_hold_task import FormationHoldConfig
from mlflow_run_utils import get_git_commit, get_git_dirty
from optimizer import initial_parameters, load_vehicle_spec
from validate_fleet_policy import FleetPolicyError, load_fleet_policy
from keep_awake import keep_awake

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG_PATH = os.path.join(HERE, "default_experiment.yaml")
VEHICLE_SPEC_PATH = os.path.join(HERE, "vehicle_spec.yaml")


# ----------------------------------------------------------------- building cells

def resolve_beliefs(cfg, truth, generic) -> list:
    """[(BeliefSpec, alpha or None)]; alpha places physics beliefs on the
    truth(0) -> generic(1) axis so a dose-response can be fitted."""
    out = []
    for b in cfg.beliefs:
        label = belief_label(b)
        if b == "calibrated":
            out.append((BeliefSpec("physics", label, truth), 0.0))
        elif b == "generic":
            out.append((BeliefSpec("physics", label, generic), 1.0))
        elif isinstance(b, str):
            out.append((BeliefSpec(b, label), None))
        elif isinstance(b, PhysicsAlphaBelief):
            out.append((BeliefSpec("physics", label, interpolate_params(truth, generic, b.physics_alpha)),
                        b.physics_alpha))
        elif isinstance(b, PhysicsParamsBelief):
            out.append((BeliefSpec("physics", label, load_calibrated_parameters(b.physics_params)), None))
    return out


def fleet_config_for(cfg, backend: str, mode: str = None) -> FleetConfig:
    """mode: 'clean' or 'noisy' (one half of mode: both); default = the experiment's own mode."""
    t = cfg.task
    task = FormationHoldConfig(target_separation=t.target_separation_m, tolerance=t.tolerance_m,
                               hold_steps=t.hold_steps, safety_radius=t.collision_radius_m,
                               influence_radius=max(1.5, t.collision_radius_m + 1.0))
    seps = cfg.initial_conditions.separations_m
    speeds = cfg.initial_conditions.closing_speeds_mps
    noise, dist_path, dist_scale = cfg.variant(mode or cfg.modes()[-1])
    disturbance = (load_disturbance(dist_path, dist_scale, truth_params_path=cfg.truth_params)
                   if dist_path else None)
    return FleetConfig(max_episode_steps=t.max_episode_steps, task=task, dynamics_backend=backend,
                       initial_separation_range=(min(seps), max(seps)),
                       initial_closing_speed_range=(min(speeds), max(speeds)), disturbance=disturbance,
                       current=cfg.current.to_config() if cfg.current is not None else None,
                       thrusters=cfg.thruster_config(), sensor_noise=noise, buoyancy=cfg.buoyancy_config())


def build_cells(cfg, truth, generic, vehicle) -> tuple:
    beliefs = resolve_beliefs(cfg, truth, generic)
    backends = ["stand_in", "mujoco"] if cfg.dynamics_backend == "both" else [cfg.dynamics_backend]
    comm = cfg.communication
    cells = []
    modes = cfg.modes()
    link_configs = comm.link_configs()
    for backend in backends:
        for mode in modes:
            fc = fleet_config_for(cfg, backend, mode)
            for pol in cfg.policies:
                optical_done = set()
                for level, plc, bwc in comm.levels():
                    for interval in comm.beacon_interval_s:
                        acoustic_base = f"{level}@{interval:g}s"
                        for lc, pointing in link_configs:
                            if lc.link == "acoustic":
                                base = acoustic_base          # unchanged labels for acoustic-only runs
                            elif lc.link == "hybrid":
                                base = f"{acoustic_base} {lc.label}"
                            else:
                                # optical only: the acoustic severity and interval don't apply, so it runs once
                                base = lc.label
                                if base in optical_done:
                                    continue
                                optical_done.add(base)
                            # with mode: both, clean and noisy are separate conditions in every table
                            cs = CommSpec(base + (f" {mode}" if len(modes) > 1 else ""), plc, bwc,
                                          comm.speed_of_sound_mps, interval,
                                          link=None if lc.link == "acoustic" else lc)
                            tags = {"mode": mode, "comm_base": base, "policy_group": pol.group,
                                    "link_kind": lc.link, "acoustic_base": "none" if lc.link == "optical"
                                    else acoustic_base, "water": lc.optical.water if lc.uses_optical else "none",
                                    "pointing": pointing}
                            for spec, _alpha in beliefs:
                                cells.append(CellSpec(pol.model_path, pol.algorithm, cs, spec, truth, vehicle, fc,
                                                      cfg.initial_conditions.separations_m,
                                                      cfg.initial_conditions.closing_speeds_mps,
                                                      cfg.evaluation.comm_seeds, tags=dict(tags)))
    alphas = {spec.label: a for spec, a in beliefs if a is not None}
    return cells, alphas


# ----------------------------------------------------------------- outputs

def _pct(x):
    return f"{100 * x:.0f}%"


def _pp(x):
    return f"{100 * x:+.0f} pts"


def headline_sentences(contrasts, a, b) -> list:
    out = []
    for c in contrasts:
        lo, hi = c["success_diff_ci95"]
        if lo > 0:
            verdict = f"**{a} did better**"
        elif hi < 0:
            verdict = f"**{b} did better**"
        else:
            verdict = "**no clear difference**"
        err = (f"belief error {c['belief_error_diff_m'] * 100:+.1f} cm" if np.isfinite(c["belief_error_diff_m"])
               else "no beacons arrived, so no belief error to compare")
        out.append(f"- **{c['comm']}**: {verdict}. Success {_pct(c['success_a'])} with `{a}` vs "
                   f"{_pct(c['success_b'])} with `{b}` ({_pp(c['success_diff'])}, 95% CI "
                   f"{_pp(lo)} to {_pp(hi)}; McNemar p = {c['mcnemar_p']:.2g}; {c['n_pairs']} matched episodes). "
                   f"Difference in {err}; collisions {_pp(c['collision_diff'])}.")
    return out


def policies_trained_elsewhere(cfg) -> list:
    """[(policy name, the parameters file it was trained on)] for policies whose training vehicle (from the
    metadata JSON train_fleet_policy.py writes next to each policy) isn't this experiment's true vehicle.
    Compared by parameter values, not file names. Policies without metadata are left out (unknown)."""
    try:
        with open(cfg.truth_params, encoding="utf-8") as f:
            truth = (yaml.safe_load(f) or {}).get("parameters") or {}
    except OSError:
        return []
    out = []
    for p in cfg.policies:
        meta_path = os.path.splitext(p.model_path)[0] + ".json"
        try:
            with open(meta_path, encoding="utf-8") as f:
                trained_on = json.load(f).get("truth_params")
        except (OSError, ValueError):
            continue
        if not trained_on:
            continue
        cands = [trained_on] if os.path.isabs(trained_on) else \
            [os.path.join(os.path.dirname(os.path.abspath(meta_path)), trained_on), os.path.join(HERE, trained_on)]
        path = next((c for c in cands if os.path.exists(c)), None)
        if path is None:
            continue
        with open(path, encoding="utf-8") as f:
            theirs = (yaml.safe_load(f) or {}).get("parameters") or {}
        same = set(theirs) == set(truth) and all(np.isclose(float(theirs[k]), float(truth[k]), rtol=1e-9, atol=1e-12)
                                                  for k in truth)
        if not same:
            out.append((os.path.basename(p.model_path), os.path.basename(path)))
    return out


def trained_elsewhere_note(cfg) -> Optional[str]:
    other = policies_trained_elsewhere(cfg)
    if not other:
        return None
    names = ", ".join(n for n, _ in other)
    sources = ", ".join(sorted({f"`{t}`" for _, t in other}))
    return (f"**Caution: the policies were trained on a different vehicle.** {names} learned to hold formation on "
            f"{sources}, not on this experiment's vehicle (`{os.path.basename(cfg.truth_params)}`). Policies tuned "
            "to another vehicle hold formation less often, and that can blur or reverse the belief-model "
            "comparison below. For conclusions about your vehicle, train policies on it first: `python strathe.py "
            f"train-fleet --params {cfg.truth_params} --out my_policy` (about 40 min; add the "
            "same `--thrusters` as here), then run this experiment with `--policies my_policy`.\n")


def simulation_lines(cfg) -> list:
    """How clean or noisy the simulation was, in plain words."""
    L = []
    modes = cfg.modes()
    if len(modes) > 1:
        L.append("**Clean and noisy, side by side (`mode: both`).** Every condition was run twice on the same "
                 "episodes: *clean* (exact sensor readings, no roll/pitch disturbance) and *noisy* (below); the two "
                 "are labelled in every table and compared episode by episode further down.\n")
    noise, dist_path, dist_scale = cfg.variant(modes[-1])
    lead = "In the noisy runs: " if len(modes) > 1 else ""
    if noise is not None:
        L.append(f"{lead}**Sensor noise on** ({noise.describe()}). Each vehicle's policy sees what its own sensors "
                 "report, and its teammate relative to where it believes it is; beacons carry what the sender "
                 "measured. Success, collisions and belief errors are still judged on the true positions. "
                 f"Most of what sensor noise costs formation-holding is the starting position fix "
                 f"(±{noise.start_fix:.2g} m per axis here; `--start-fix-error` sets it on its own): with a perfect "
                 "fix, even the assumed `typical` sensors cost about 2 points (PROJECT_STATUS.md #60).\n")
    else:
        L.append("Exact sensor readings: no sensor noise (`mode: noisy`, or a `sensor_noise:` block, adds "
                 "realistic DVL, gyro, attitude, compass and depth noise).\n")
    if dist_path:
        L.append(f"{lead}**Roll/pitch disturbance on** (`{os.path.basename(dist_path)}`, scale "
                 f"{dist_scale:g}): each vehicle is pushed by random roll and pitch torques fitted to the "
                 "motion real BlueROV2 logs show but the thrusters don't explain. Every condition sees the same "
                 "disturbances.\n")
    else:
        L.append("No roll/pitch disturbance (set `mode: noisy` or `disturbance:` in the YAML to add one).\n")
    return L


def uses_optical(cfg) -> bool:
    return any(l != "acoustic" for l in cfg.communication.links())


def link_summary(rows) -> list:
    """Per comm condition, pooled over policies and beliefs: how each link did."""
    groups = {}
    for r in rows:
        groups.setdefault(r["comm"], []).append(r)

    def nanmean(xs):
        xs = np.array([x for x in xs if x is not None], dtype=float)
        xs = xs[np.isfinite(xs)]
        return float(xs.mean()) if xs.size else float("nan")

    def nanmedian(xs):
        xs = np.array([x for x in xs if x is not None], dtype=float)
        xs = xs[np.isfinite(xs)]
        return float(np.median(xs)) if xs.size else float("nan")
    out = []
    for comm, rs in groups.items():
        out.append({"comm": comm, "link": rs[0].get("link", "acoustic"), "link_kind": rs[0].get("link_kind", "acoustic"),
                    "n": len(rs), "success_rate": float(np.mean([r["success"] for r in rs])),
                    "collision_rate": float(np.mean([r["collision"] for r in rs])),
                    "acoustic_delivery_share": nanmean(r["acoustic_delivery_share"] for r in rs),
                    "acoustic_skipped_per_episode": float(np.mean([r["acoustic_skipped"] for r in rs])),
                    # of the skipped acoustic beacons, the share whose optical copy was lost too
                    "skipped_uncovered_share": (sum(r.get("acoustic_skipped_uncovered", 0) for r in rs)
                                                / sum(r["acoustic_skipped"] for r in rs)
                                                if sum(r["acoustic_skipped"] for r in rs) else float("nan")),
                    "optical_up_fraction": nanmean(r["optical_up_fraction"] for r in rs),
                    "belief_updates_optical_share": nanmean(r["belief_updates_optical_share"] for r in rs),
                    "median_belief_age_s": nanmedian(r["median_belief_age_s"] for r in rs)})
    return out


def link_contrasts(rows) -> tuple:
    """(each optical/hybrid link minus acoustic, pointing cones minus omni), paired episode by episode.
    Optical-only runs have no acoustic severity, so they are paired against every acoustic one."""
    kinds = {r.get("link_kind", "acoustic") for r in rows}
    if kinds == {"acoustic"}:
        return [], []
    acoustic_bases = sorted({r["acoustic_base"] for r in rows if r.get("link_kind") == "acoustic"})
    aug = []
    for r in rows:
        if r.get("link_kind") == "optical":
            aug += [dict(r, acoustic_base=b) for b in acoustic_bases]
        else:
            aug.append(r)
    vs_acoustic = []
    if "acoustic" in kinds:
        for lab in dict.fromkeys(r["link"] for r in rows if r.get("link_kind") != "acoustic"):
            vs_acoustic += paired_by(aug, "link", lab, "acoustic", match=("policy", "belief", "acoustic_base",
                                                                         "backend", "mode"),
                                     group_by=("acoustic_base", "mode"))
    pointing = []
    for p in dict.fromkeys(r["pointing"] for r in rows if r.get("pointing") not in ("acoustic", "omni", None)
                           and not str(r.get("pointing")).startswith("omni")):
        pointing += paired_by(rows, "pointing", p, "omni", match=("policy", "belief", "acoustic_base", "link_kind",
                                                                  "water", "backend", "mode"),
                              group_by=("link_kind", "water", "acoustic_base", "mode"))
    return vs_acoustic, pointing


def _ci(c) -> str:
    lo, hi = c["success_diff_ci95"]
    return f"{_pp(c['success_diff'])} ({_pp(lo)} to {_pp(hi)})"


def link_lines(cfg, lsum, vs_acoustic, pointing) -> list:
    """The report's optical-link section."""
    comm = cfg.communication
    L = ["## Optical link\n"]
    seps = cfg.initial_conditions.separations_m
    seen = set()
    for lc, _p in comm.link_configs():
        if lc.link == "acoustic" or lc.label in seen:
            continue
        seen.add(lc.label)
        L.append(f"- `{lc.label}`: {lc.describe()}.")
        note = lc.start_geometry_note(seps)
        if note:
            L.append(f"  - **Start geometry: {note}**")
    L.append("")
    L.append("Optical delivery is judged on the true positions and attitudes (the light follows the real vehicle); "
             "the beacon carries what the sender measured. A belief keeps whichever beacon was sent most recently, "
             "whichever link it came on, so a late acoustic copy never rolls it back.\n")
    L.append("**No latency advantage is modelled.** Messages are delivered on the 0.1 s tick: anything sent on one "
             "tick with any latency above zero arrives on the next. Optical's 1 ms and acoustic's few milliseconds at "
             "these ranges (under 150 m) are therefore identical here. Optical's advantage in this simulation is its "
             "send rate (every tick) and its loss, not its latency.\n")
    if "hybrid" in comm.links():
        pol, why = next(lc for lc, _ in comm.link_configs() if lc.link == "hybrid").effective_hybrid_policy()
        if why:
            L.append(f"**Hybrid policy changed:** {why}.\n")
        L.append("*Why `optical_first` exists:* acoustic energy and airtime aren't modelled here, so sending every "
                 "beacon on both links (`redundant`) is never worse in this simulation. `optical_first` saves acoustic "
                 "energy and channel time, which starts to matter once three or more vehicles share one acoustic "
                 f"channel. Here a vehicle skips its acoustic beacon only if an optical message from its teammate "
                 f"arrived in the last {next(lc for lc, _ in comm.link_configs() if lc.link == 'hybrid').link_up_ticks} "
                 f"tick(s); the table counts skipped beacons whose optical copy was lost too.\n")
    L.append("| Comm | Success | Collision | Acoustic delivered | Acoustic beacons skipped / episode | "
             "Skipped with optical copy lost | Optical up | Belief updates via optical | Median belief age |")
    L.append("|---|---|---|---|---|---|---|---|---|")

    def f(x, pct=True):
        return "n/a" if not np.isfinite(x) else (_pct(x) if pct else f"{x:.2f} s")
    for s in sorted(lsum, key=lambda s: s["comm"]):
        L.append(f"| {s['comm']} | {_pct(s['success_rate'])} | {_pct(s['collision_rate'])} | "
                 f"{f(s['acoustic_delivery_share'])} | {s['acoustic_skipped_per_episode']:.1f} | "
                 f"{f(s['skipped_uncovered_share'])} | "
                 f"{f(s['optical_up_fraction'])} | {f(s['belief_updates_optical_share'])} | "
                 f"{f(s['median_belief_age_s'], pct=False)} |")
    L.append("\n*Optical up* = share of optical beacons that got through; *belief updates via optical* = share of "
             "the beacons that actually refreshed a teammate belief that came by light. Pooled over policies and "
             "belief models.\n")
    if vs_acoustic:
        L.append("### Does the optical link help? (vs acoustic only, same episodes)\n")
        L.append("| Link | Acoustic condition | Mode | With link | Acoustic only | Difference (95% CI) | "
                 "Collisions with link vs acoustic | Episodes |")
        L.append("|---|---|---|---|---|---|---|---|")
        for c in vs_acoustic:
            L.append(f"| `{c['a']}` | {c['acoustic_base']} | {c['mode']} | {_pct(c['success_a'])} | "
                     f"{_pct(c['success_b'])} | {_ci(c)} | {_pct(c['collision_a'])} vs {_pct(c['collision_b'])} | "
                     f"{c['n_pairs']} |")
        L.append("\nOptical-only runs have no acoustic channel; they are paired with each acoustic condition on the "
                 "same episodes.\n")
    if pointing:
        L.append("### What pointing costs (cones vs omnidirectional terminals, same episodes)\n")
        L.append("| Terminals | Link | Water | Acoustic condition | Mode | Cones | Omni | Cones − omni (95% CI) | "
                 "Episodes |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for c in pointing:
            L.append(f"| `{c['a']}` | {c['link_kind']} | {c['water']} | {c['acoustic_base']} | {c['mode']} | "
                     f"{_pct(c['success_a'])} | {_pct(c['success_b'])} | {_ci(c)} | {c['n_pairs']} |")
        L.append("")
    L.append("**Limits of the optical model:** uncalibrated (rough textbook water values, a placeholder delivery "
             "curve: fit it to your own link logs of success against range and angle before quoting numbers); a hard "
             "cone (inside = full power, outside = none); no occlusion, sunlight, bubbles or turbidity changes; "
             "pointing uses the true attitude, so results are more sensitive to roll and pitch than acoustic ones "
             "(see LIMITS.md).\n")
    return L


def write_report(path, cfg, summary, contrasts, doses, links, elapsed, n_eps, dirty, mode_contrasts=None,
                 group_contrasts=None, group_summary=None, link_section=None):
    a, b = (cfg.compare.a, cfg.compare.b) if cfg.compare else (None, None)
    L = []
    L.append(f"# Fleet experiment: {cfg.name or 'unnamed'}\n")
    L.append(f"{datetime.now():%Y-%m-%d %H:%M} · {n_eps} episodes · {elapsed / 60:.1f} min"
             + (" · **ran with uncommitted code changes**" if dirty else "") + "\n")
    L.append("## What was tested\n")
    L.append("Two simulated vehicles must hold "
             f"{cfg.task.target_separation_m} m ± {cfg.task.tolerance_m} m apart for {cfg.task.hold_steps} "
             f"consecutive ticks (0.1 s each) within {cfg.task.max_episode_steps} ticks, without coming closer than "
             f"{cfg.task.collision_radius_m} m. They share their state over "
             + ("an acoustic link that loses packets. " if not uses_optical(cfg) else
                f"a link that loses packets ({', '.join(cfg.communication.links())}; see *Optical link* below). ")
             + "Between beacons, each vehicle predicts where its teammate is with a **belief model**; that model is the "
             "only thing that changes between the conditions below. Both vehicles' true physics is "
             f"`{os.path.basename(cfg.truth_params)}` in every condition, and every condition sees the same starting "
             "positions and the same lost packets.\n")
    L += simulation_lines(cfg)
    if uses_optical(cfg):
        for lc, _p in cfg.communication.link_configs():
            note = lc.start_geometry_note(cfg.initial_conditions.separations_m) if lc.link != "acoustic" else None
            if note:
                L.append(f"**Warning (`{lc.label}`):** {note}\n")
    cur = cfg.current.to_config() if cfg.current is not None else None
    if cur is not None and cur.active:
        L.append(f"**Water current on:** {cur.describe()}. Both vehicles feel the same current; the policy sees only "
                 "its own velocity over ground. Every condition sees the same currents.\n")
    else:
        L.append("Still water: no current (set `current:` in the YAML for open-water conditions).\n")
    th = cfg.thruster_config()
    if th is not None:
        import thruster_model as tm
        L.append(f"**Thruster limits on:** {tm.describe(th)}. Every command is limited to what these thrusters can "
                 "deliver; the beacons carry the delivered force.\n")
    else:
        L.append(f"No thruster limits: any command up to ±{FleetConfig().action_bound:g} N / N·m per axis is "
                 "delivered (set `thrusters:` in the YAML to use your vehicle's real thrusters).\n")
    buoy = cfg.buoyancy_config()
    if buoy is not None and buoy.active:
        L.append(f"**Buoyancy changed:** {buoy.describe()}. Every condition sees the same buoyancy.\n")
    if os.path.basename(cfg.vehicle_spec) != "vehicle_spec.yaml" or os.path.dirname(os.path.abspath(cfg.vehicle_spec)) != HERE:
        L.append(f"Vehicle description: `{cfg.vehicle_spec}`.\n")
    note = trained_elsewhere_note(cfg)
    if note:
        L.append(note)
    if contrasts:
        L.append(f"## Headline: `{a}` vs `{b}`\n")
        L += headline_sentences(contrasts, a, b)
        L.append("")
    if mode_contrasts:
        L.append("## Does noise matter? Clean vs noisy\n")
        L.append("The same episodes (start state, packet losses) run clean and noisy. A negative difference means the "
                 "noise costs success.\n")
        L.append("| Comm | Belief | Clean | Noisy | Noisy − clean (95% CI) | Episodes |")
        L.append("|---|---|---|---|---|---|")
        for c in mode_contrasts:
            lo, hi = c["success_diff_ci95"]
            L.append(f"| {c['comm_base']} | `{c['belief']}` | {_pct(c['success_b'])} | {_pct(c['success_a'])} | "
                     f"{_pp(c['success_diff'])} ({_pp(lo)} to {_pp(hi)}) | {c['n_pairs']} |")
        L.append("")
    if link_section:
        L += link_section
    if group_summary:
        groups = list(dict.fromkeys(p.group for p in cfg.policies))
        L.append("## Policy groups compared\n")
        L.append(f"Policies grouped by their label (or training algorithm): {', '.join(f'`{g}`' for g in groups)}. "
                 "Every group runs the same episodes; a group's training seeds are averaged per episode before "
                 f"groups are compared. Differences are against `{groups[0]}`.\n")
        seeds = {g: sum(1 for p in cfg.policies if p.group == g) for g in groups}
        L.append("These differences hold for this vehicle, this task and these training seeds. "
                 + (f"Groups with a single policy ({', '.join(f'`{g}`' for g, n in seeds.items() if n < 2)}) can't "
                    "separate an algorithm from the luck of one training seed; train two or three seeds each. "
                    if any(n < 2 for n in seeds.values()) else "")
                 + "SAC and TD3 were validated against PPO on the reference BlueROV2 only (PROJECT_STATUS.md #58).\n")
        by = {(g["policy_group"], g["comm"]): g for g in group_summary}
        comms = list(dict.fromkeys(g["comm"] for g in group_summary))
        L.append("| Comm | " + " | ".join(f"`{g}`" for g in groups) + " | "
                 + " | ".join(f"`{g}` − `{groups[0]}` (95% CI)" for g in groups[1:]) + " |")
        L.append("|---|" + "---|" * (2 * len(groups) - 1))
        gc = {(c["comm"], c["a"]): c for c in (group_contrasts or [])}
        for cm in comms:
            cells = [(f"{_pct(by[(g, cm)]['success_rate'])}" if (g, cm) in by else "n/a") for g in groups]
            diffs = []
            for g in groups[1:]:
                c = gc.get((cm, g))
                diffs.append("n/a" if c is None else f"{_pp(c['success_diff'])} ({_pp(c['success_diff_ci95'][0])} "
                                                     f"to {_pp(c['success_diff_ci95'][1])})")
            L.append(f"| {cm} | " + " | ".join(cells) + " | " + " | ".join(diffs) + " |")
        L.append("\nPooled over every belief model in the run.\n")
    L.append("## All conditions\n")
    L.append("| Comm | Belief | Success (95% CI) | Collision | Mean belief error | Episodes |")
    L.append("|---|---|---|---|---|---|")
    for s in summary:
        lo, hi = s["success_ci95"]
        L.append(f"| {s['comm']} | `{s['belief']}` | {_pct(s['success_rate'])} ({_pct(lo)}–{_pct(hi)}) | "
                 f"{_pct(s['collision_rate'])} | "
                 + (f"{s['mean_belief_error_m'] * 100:.1f} cm" if np.isfinite(s['mean_belief_error_m']) else "n/a")
                 + f" | {s['n']} |")
    L.append("")
    if doses:
        L.append("## Does belief-model error drive success? (dose-response)\n")
        L.append("Belief models placed on a line from the true parameters (alpha = 0) to generic defaults "
                 "(alpha = 1). A negative slope means success falls as the model gets less accurate.\n")
        L.append("| Comm | Success at alpha = " + ", ".join(f"{x:g}" for x in doses[0]["alphas"])
                 + " | Slope per unit alpha (95% CI) |")
        L.append("|---|---|---|")
        for d in doses:
            lo, hi = d["slope_ci95"]
            L.append(f"| {d['comm']} | " + ", ".join(_pct(r) for r in d["success_rates"])
                     + f" | {_pp(d['slope_per_unit_alpha'])} ({_pp(lo)} to {_pp(hi)}) |")
        L.append("")
    if links:
        L.append("## Belief error vs success across all belief models\n")
        L.append("Spearman correlation between a condition's mean belief error and its success rate "
                 "(negative = lower error goes with more success; the oracle is excluded because its error is "
                 "zero by construction).\n")
        for k in links:
            if np.isfinite(k["spearman_rho"]):
                L.append(f"- {k['comm']}: rho = {k['spearman_rho']:+.2f} over {k['n_conditions']} belief models "
                         f"(p = {k['spearman_p']:.2g})")
            else:
                L.append(f"- {k['comm']}: not defined (every belief model had the same success rate or error)")
        L.append("")
    L.append("## How to read this, and its limits\n")
    L.append("- Differences are **paired**: the same episode (start state, packet losses) is run under each belief "
             "model, so the comparison isn't blurred by luck of the draw. CIs are bootstrap intervals over episodes; "
             "McNemar's test counts episodes that one model won and the other lost.\n"
             "- `calibrated` means a model whose parameters equal the simulated vehicle's true parameters. It shows "
             "the value of an accurate model; how close a real calibration gets to a real vehicle is measured "
             "separately, on held-out real logs (see README, calibration results).\n"
             "- Simulation only. No hardware has been in the loop.\n"
             f"- Policies: {', '.join(os.path.basename(p.model_path) for p in cfg.policies)}. Results pool them; "
             "`episodes.csv` has the per-policy breakdown.\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def write_plots(plots_dir, summary, doses):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(plots_dir, exist_ok=True)
    comms = sorted({s["comm"] for s in summary})
    beliefs = list(dict.fromkeys(s["belief"] for s in summary))
    by = {(s["comm"], s["belief"]): s for s in summary}
    for metric, fname, ylabel in (("success_rate", "success_by_belief.png", "Success rate"),
                                  ("mean_belief_error_m", "belief_error_by_belief.png", "Mean belief error (m)")):
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(comms) * len(beliefs) / 3), 4))
        w = 0.8 / len(beliefs)
        for j, bel in enumerate(beliefs):
            xs = np.arange(len(comms)) + (j - (len(beliefs) - 1) / 2) * w
            ys = [by[(c, bel)][metric] if (c, bel) in by else np.nan for c in comms]
            if metric == "success_rate":
                err = np.array([[ys[i] - by[(c, bel)]["success_ci95"][0], by[(c, bel)]["success_ci95"][1] - ys[i]]
                                if (c, bel) in by else [0, 0] for i, c in enumerate(comms)]).T
                # a 100% (or 0%) cell's Wilson bound sits a rounding error past the value; matplotlib refuses
                # negative error bars and the plots were skipped
                ax.bar(xs, ys, w, yerr=np.clip(err, 0.0, None), capsize=2, label=bel)
            else:
                ax.bar(xs, ys, w, label=bel)
        ax.set_xticks(range(len(comms)), comms)
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, fname), dpi=130)
        plt.close(fig)
    if doses:
        fig, ax = plt.subplots(figsize=(6, 4))
        for d in doses:
            ax.plot(d["alphas"], d["success_rates"], marker="o", label=d["comm"])
        ax.set_xlabel("Belief-model parameters: 0 = true vehicle, 1 = generic defaults")
        ax.set_ylabel("Success rate")
        ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, "dose_response.png"), dpi=130)
        plt.close(fig)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def _log_mlflow(cfg, out_dir, summary, contrasts):
    """Observational only: never lets a logging problem affect the run."""
    try:
        import mlflow
        mlflow.set_tracking_uri(f"sqlite:///{os.path.join(HERE, 'mlflow.db')}")
        mlflow.set_experiment("strathe_fleet")
        with mlflow.start_run(run_name=cfg.name or "fleet_experiment"):
            mlflow.set_tags({"script": "run_fleet_experiment.py", "schema_version": "2",
                             "git_commit": get_git_commit(), "dynamics_backend": cfg.dynamics_backend,
                             "disturbance": (f"{os.path.basename(cfg.disturbance.path())}x{cfg.disturbance.scale:g}"
                                             if cfg.disturbance.path() else "none"),
                             "current": (cfg.current.to_config().describe() if cfg.current is not None
                                         and cfg.current.to_config().active else "none"),
                             "thrusters": str(cfg.thrusters) if cfg.thrusters is not None else "none",
                             "vehicle_spec": cfg.vehicle_spec, "mode": cfg.mode,
                             "sensor_noise": (f"{cfg.sensor_noise.preset}x{cfg.sensor_noise.scale:g}"
                                              if cfg.sensor_noise is not None else "none"),
                             "buoyancy": (cfg.buoyancy_config().describe() if cfg.buoyancy is not None
                                          and cfg.buoyancy_config().active else "none"),
                             "algorithms": ",".join(sorted({p.algorithm for p in cfg.policies})),
                             "link": ",".join(cfg.communication.links()),
                             "optical_water": (",".join(cfg.communication.optical.waters())
                                               if cfg.communication.optical is not None else "none"),
                             "optical_terminals": (str(cfg.communication.optical.terminals)
                                                   if cfg.communication.optical is not None else "none"),
                             "hybrid_policy": (cfg.communication.hybrid_policy
                                               if "hybrid" in cfg.communication.links() else "none")})
            for s in summary:
                key = f"{s['comm']}.{s['belief']}".replace("@", "_at_")
                mlflow.log_metric(f"success.{key}", s["success_rate"])
                mlflow.log_metric(f"belief_error_m.{key}", s["mean_belief_error_m"])
            for c in contrasts:
                mlflow.log_metric(f"success_diff.{c['comm'].replace('@', '_at_')}", c["success_diff"])
            mlflow.log_artifacts(out_dir)
    except Exception as e:  # pragma: no cover - depends on local mlflow state
        print(f"  (MLflow logging skipped: {type(e).__name__}: {e})")


# ----------------------------------------------------------------- main

def run_experiment(config_path: str) -> dict:
    t0 = time.monotonic()
    print("=" * 70 + "\nSTRATHE FLEET EXPERIMENT (v2)\n" + "=" * 70)
    print(f"Config: {config_path}\n\n[1/4] Validating config...")
    cfg = load_fleet_experiment(config_path)
    spec = load_vehicle_spec(cfg.vehicle_spec)
    truth = load_calibrated_parameters(cfg.truth_params)
    generic = initial_parameters(spec)
    print(f"  OK: {len(cfg.policies)} policy(ies), beliefs {[belief_label(b) for b in cfg.beliefs]}, "
          f"comm {[l for l, _, _ in cfg.communication.levels()]} x beacon {cfg.communication.beacon_interval_s} s, "
          f"backend {cfg.dynamics_backend}, disturbance "
          + (f"{os.path.basename(cfg.disturbance.path())} x{cfg.disturbance.scale:g}" if cfg.disturbance.path()
             else "off")
          + ", current " + (cfg.current.to_config().describe() if cfg.current is not None
                            and cfg.current.to_config().active else "none")
          + ", thrusters " + ("limited" if cfg.thrusters is not None else "unlimited")
          + f", mode {cfg.mode}"
          + (f" (sensor noise {cfg.sensor_noise.preset} x{cfg.sensor_noise.scale:g})"
             if cfg.sensor_noise is not None and cfg.sensor_noise.config() is not None else "")
          + (", buoyancy " + cfg.buoyancy_config().describe()
             if cfg.buoyancy is not None and cfg.buoyancy_config().active else "")
          + f", link {'/'.join(cfg.communication.links())}")
    if uses_optical(cfg):
        seen = set()
        for lc, _p in cfg.communication.link_configs():
            if lc.link == "acoustic" or lc.label in seen:
                continue
            seen.add(lc.label)
            print(f"  {lc.label}: {lc.describe()}")
            note = lc.start_geometry_note(cfg.initial_conditions.separations_m)
            if note:
                print(f"  WARNING: {note}")

    print("\n[2/4] Checking policies against the per-vehicle contract...")
    for p in cfg.policies:
        load_fleet_policy(p.model_path, p.algorithm)
        print(f"  PASS {p.model_path} ({p.algorithm.upper()}" + (f", group {p.label}" if p.label else "") + ")")
    for name, trained_on in policies_trained_elsewhere(cfg):
        print(f"  NOTE {name} was trained on {trained_on}, not on this experiment's vehicle; the report explains "
              f"what that means (train-fleet trains policies for your vehicle)")

    cells, alphas = build_cells(cfg, truth, generic, spec.vehicle)
    n_per_cell = (len(cfg.initial_conditions.separations_m) * len(cfg.initial_conditions.closing_speeds_mps)
                  * len(cfg.evaluation.comm_seeds))
    print(f"\n[3/4] Running {len(cells)} cells x {n_per_cell} episodes = {len(cells) * n_per_cell} episodes...")
    last = [0.0]

    def progress(done, total):
        if time.monotonic() - last[0] > 10 or done == total:
            last[0] = time.monotonic()
            print(f"  {done}/{total} cells ({time.monotonic() - t0:.0f}s)", flush=True)
    rows = run_cells(cells, workers=cfg.evaluation.workers, progress=progress)

    print("\n[4/4] Statistics and report...")
    out = cfg.output_dir
    os.makedirs(out, exist_ok=True)
    summary = summarize(rows, by=("comm", "belief"))
    contrasts = paired_contrast(rows, cfg.compare.a, cfg.compare.b) if cfg.compare else []
    doses = dose_response(rows, alphas) if len(alphas) >= 3 else []
    links = belief_success_link(summary, exclude_beliefs=[l for l in {r["belief"] for r in rows}
                                                          if l == "oracle"])
    mode_contrasts = (paired_by(rows, "mode", "noisy", "clean", match=("policy", "belief", "comm_base", "backend"),
                                group_by=("comm_base", "belief")) if len(cfg.modes()) > 1 else [])
    groups = list(dict.fromkeys(p.group for p in cfg.policies))
    group_summary = summarize(rows, by=("policy_group", "comm")) if len(groups) > 1 else []
    group_contrasts = [c for g in groups[1:] for c in
                       paired_by(rows, "policy_group", g, groups[0], match=("belief", "comm", "backend"),
                                 group_by=("comm",))] if len(groups) > 1 else []
    lsum = link_summary(rows) if uses_optical(cfg) else []
    vs_acoustic, pointing = link_contrasts(rows) if uses_optical(cfg) else ([], [])
    elapsed = time.monotonic() - t0
    dirty = get_git_dirty()

    with open(os.path.join(out, "episodes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    metrics = {"summary": summary, "paired_contrast": contrasts, "dose_response": doses, "belief_success_link": links,
               "clean_vs_noisy": mode_contrasts, "policy_groups": group_summary,
               "policy_group_contrasts": group_contrasts, "links": lsum, "link_vs_acoustic": vs_acoustic,
               "pointing_vs_omni": pointing}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(_jsonable(metrics), f, indent=2)
    from importlib.metadata import version

    def v(pkg):
        try:
            return version(pkg)
        except Exception:
            return "unknown"
    meta = {"git_commit": get_git_commit(), "git_uncommitted_changes": dirty, "python": platform.python_version(),
            "platform": platform.platform(), "packages": {p: v(p) for p in
                                                          ("numpy", "scipy", "stable-baselines3", "torch", "mujoco")},
            "timestamp": datetime.now().isoformat(), "elapsed_s": round(elapsed, 1), "n_episodes": len(rows),
            "resolved_config": cfg.model_dump(mode="json")}
    with open(os.path.join(out, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out, "experiment.yaml"), "w") as f:
        yaml.safe_dump(cfg.model_dump(mode="json", exclude_none=True), f, sort_keys=False)
    write_report(os.path.join(out, "summary_report.md"), cfg, summary, contrasts, doses, links, elapsed, len(rows), dirty,
                 mode_contrasts=mode_contrasts, group_contrasts=group_contrasts, group_summary=group_summary,
                 link_section=link_lines(cfg, lsum, vs_acoustic, pointing) if uses_optical(cfg) else None)
    try:
        write_plots(os.path.join(out, "plots"), summary, doses)
    except Exception as e:  # plots are a convenience; the numbers are already written
        print(f"  (plots skipped: {type(e).__name__}: {e})")
    _log_mlflow(cfg, out, summary, contrasts)

    print("\n" + "=" * 70)
    if contrasts:
        for line in headline_sentences(contrasts, cfg.compare.a, cfg.compare.b):
            print(line.replace("**", "").replace("`", ""))
    for c in mode_contrasts:
        print(f"- noisy - clean, {c['comm_base']}, {c['belief']}: {100 * c['success_diff']:+.0f} pts "
              f"({100 * c['success_diff_ci95'][0]:+.0f} to {100 * c['success_diff_ci95'][1]:+.0f})")
    for c in group_contrasts:
        print(f"- {c['a']} - {c['b']}, {c['comm']}: {100 * c['success_diff']:+.0f} pts "
              f"({100 * c['success_diff_ci95'][0]:+.0f} to {100 * c['success_diff_ci95'][1]:+.0f})")
    for c in vs_acoustic:
        print(f"- {c['a']} - acoustic, {c['acoustic_base']} {c['mode']}: {100 * c['success_diff']:+.0f} pts "
              f"({100 * c['success_diff_ci95'][0]:+.0f} to {100 * c['success_diff_ci95'][1]:+.0f})")
    for c in pointing:
        print(f"- {c['a']} - omni, {c['link_kind']} {c['water']} {c['acoustic_base']} {c['mode']}: "
              f"{100 * c['success_diff']:+.0f} pts "
              f"({100 * c['success_diff_ci95'][0]:+.0f} to {100 * c['success_diff_ci95'][1]:+.0f})")
    print(f"\nDone in {elapsed / 60:.1f} min. Report: {os.path.join(out, 'summary_report.md')}")
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    args = ap.parse_args(argv)
    try:
        run_experiment(args.config)
    except (ConfigError, FleetPolicyError) as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    with keep_awake("fleet experiment"):  # PROJECT_STATUS.md #49
        rc = main()
    sys.exit(rc)
