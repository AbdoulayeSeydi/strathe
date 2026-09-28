#!/usr/bin/env python3
"""
stress_test_optical.py -- put the optical and hybrid links through every
combination of settings, and check what must hold exactly. PROJECT_STATUS.md #59.

    python stress_test_optical.py [--workers 6]

A. Grid: every link (acoustic, optical, hybrid optical_first, hybrid
   redundant) x water (clear, coastal, harbour, custom c = 1/m) x terminals
   (omni, forward_cone, fore_aft_cones, down_cone, and an asymmetric cone
   pair) x comm L0 / L2 / L3a / L3b x beacons every 1 s and 3 s x clean /
   noisy x buoyancy (none, -5, +5, -5..+5 N) x policy (PPO, SAC, TD3,
   random, zero). One episode each on the stand-in physics, plus every link
   configuration on MuJoCo. Every episode must run to the end with finite
   observations and consistent message bookkeeping.
B. Exact checks with the zero policy (no commands, so the vehicles move the
   same way whatever the link, and differences are the link's alone):
   - optical delivery is nested by water: clear >= coastal >= custom >= harbour
   - forward cones deliver nothing from the line start (clean)
   - hybrid redundant: acoustic deliveries identical to acoustic only, and the
     teammate belief never older
   - acoustic drops paired: a hybrid run uses exactly the acoustic draws an
     acoustic-only run uses (sent + skipped = sent)
   - asymmetric terminals run optical_first as redundant (nothing skipped)
C. Determinism: a sample of combinations rerun gives identical results.
D. Acoustic unchanged: 96 acoustic cells (stand-in and MuJoCo, clean and
   noisy with buoyancy, L0-L3b, beacons 0.5-3 s, two belief models) hashed
   observation by observation against the reference taken on main before
   this change (results/stress_optical/acoustic_reference_8588809.json).

Writes results/stress_optical/report.md and results.json. Exits 1 if any
check fails.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

OUT = "results/stress_optical"
REFERENCE = os.path.join(OUT, "acoustic_reference_8588809.json")
TRUTH = "calibrated_parameters_fleet_truth.yaml"
POLICIES = {"ppo": "fleet_policy_seed1", "sac": "algorithm_policies/fleet_sac_seed1",
            "td3": "algorithm_policies/fleet_td3_seed1", "random": "random", "zero": "zero"}
SEVERITIES = ("L0", "L2", "L3a", "L3b")
INTERVALS = (1.0, 3.0)
WATERS = ("clear", "coastal", "custom", "harbor")          # in order of murkiness (custom c = 1.0/m)
TERMINALS = ("omni", "forward_cone", "fore_aft_cones", "down_cone", "asymmetric")
LINKS = ("acoustic", "optical", "hybrid", "hybrid_redundant")
BUOYANCY = (None, -5.0, 5.0, (-5.0, 5.0))

_cache = {}


def _setup():
    if "truth" not in _cache:
        from build_two_agent_cores import load_calibrated_parameters
        from optimizer import load_vehicle_spec
        _cache["truth"] = load_calibrated_parameters(TRUTH, quiet=True)
        _cache["vehicle"] = load_vehicle_spec("vehicle_spec.yaml").vehicle
    return _cache["truth"], _cache["vehicle"]


def _policy(name, seed=0):
    if name == "random":
        # a fresh stream per episode: a cached one would depend on what this worker ran before
        from fleet_eval import RandomPolicy
        return RandomPolicy(seed)
    if name not in _cache:
        from fleet_eval import _load_policy
        _cache[name] = _load_policy(POLICIES[name], None)
    return _cache[name]


def link_config(link, water, terminals):
    from comm_channel import LinkConfig, OpticalLinkConfig
    if link == "acoustic":
        return LinkConfig("acoustic")
    kw = {"water": water, "attenuation_per_m": 1.0 if water == "custom" else None}
    if terminals == "asymmetric":
        kw.update(terminals="fore_aft_cones", beam_half_angle_deg=30.0, receiver_half_fov_deg=60.0)
    else:
        kw["terminals"] = terminals
    oc = OpticalLinkConfig(**kw)
    lc = LinkConfig("hybrid" if link.startswith("hybrid") else link, oc,
                    "redundant" if link == "hybrid_redundant" else "optical_first")
    lc.validate()
    return lc


def run_combo(c: dict) -> dict:
    """One episode; returns totals, a hash of everything observed, and any failure."""
    import torch
    torch.set_num_threads(1)
    from buoyancy import BuoyancyConfig
    from evaluate_comm_degradation_curve import COMM_CONFIGS
    from fleet_belief import BeliefSpec
    from fleet_env import FleetConfig, FleetSim, N_AGENTS
    from sensor_noise import load_sensor_noise
    from disturbance import DEFAULT_FIT_PATH, load_disturbance
    truth, vehicle = _setup()
    out = dict(c)
    try:
        kw = {"dynamics_backend": c["backend"]}
        if c["mode"] == "noisy":
            kw["sensor_noise"] = load_sensor_noise("typical")
            kw["disturbance"] = load_disturbance(DEFAULT_FIT_PATH, 1.0, truth_params_path=TRUTH)
        if c["buoyancy"] is not None:
            b = c["buoyancy"]
            kw["buoyancy"] = BuoyancyConfig(offset_n=tuple(b) if isinstance(b, (list, tuple)) else b)
        cfg = FleetConfig(**kw)
        lc = link_config(c["link"], c["water"], c["terminals"])
        belief = (BeliefSpec("physics", "calibrated", truth) if c["belief"] == "calibrated"
                  else BeliefSpec(c["belief"], c["belief"]))
        sim = FleetSim(truth, vehicle, cfg, COMM_CONFIGS[c["sev"]]["packet_loss_config"],
                       COMM_CONFIGS[c["sev"]]["bandwidth_config"], belief=belief,
                       beacon_interval_s=c["interval"], link=lc)
        pol = _policy(c["policy"], c["seed"])
        obs = sim.reset(comm_seed=c["seed"], initial_separation=c["d0"], closing_speed=c["v0"])
        h = hashlib.sha256()
        tot = {"acoustic_sent": 0, "optical_sent": 0, "acoustic_skipped": 0, "acoustic_delivered": 0,
               "optical_delivered": 0, "updates_optical": 0, "updates_acoustic": 0}
        ages, trace, steps, reason = [], [], 0, None
        while True:
            a, _ = pol.predict(np.stack(obs), deterministic=True)
            obs, r, te, tr, info = sim.step([a[i] for i in range(N_AGENTS)])
            steps += 1
            o = np.stack(obs)
            if not (np.all(np.isfinite(o)) and np.isfinite(r)):
                raise FloatingPointError(f"non-finite observation or reward at step {steps}")
            h.update(o.tobytes())
            h.update(np.float64(r).tobytes())
            tot["acoustic_sent"] += info["sent"]["acoustic"]
            tot["optical_sent"] += info["sent"]["optical"]
            tot["acoustic_skipped"] += info["acoustic_skipped"]
            tot["updates_optical"] += info["belief_updates"]["optical"]
            tot["updates_acoustic"] += info["belief_updates"]["acoustic"]
            for links in info["delivered_by"]:
                tot["acoustic_delivered"] += links.count("acoustic")
                tot["optical_delivered"] += links.count("optical")
            ages += [x for x in info["belief_ages"] if np.isfinite(x)]
            trace += [float(x) if np.isfinite(x) else 1e9 for x in info["belief_ages"]]
            # bookkeeping: an optical-only link sends nothing acoustic, an acoustic link nothing optical
            if lc.link == "optical" and (info["sent"]["acoustic"] or info["acoustic_skipped"]):
                raise AssertionError("optical-only link sent an acoustic beacon")
            if lc.link == "acoustic" and info["sent"]["optical"]:
                raise AssertionError("acoustic link sent an optical beacon")
            if te or tr:
                reason = info["terminal_reason"] or "timeout"
                break
        if tot["acoustic_delivered"] > tot["acoustic_sent"] or tot["optical_delivered"] > tot["optical_sent"]:
            raise AssertionError(f"more delivered than sent: {tot}")
        if tot["updates_optical"] > tot["optical_delivered"] or tot["updates_acoustic"] > tot["acoustic_delivered"]:
            raise AssertionError(f"more belief updates than deliveries: {tot}")
        out.update(tot, steps=steps, reason=reason, hash=h.hexdigest(),
                   median_age=float(np.median(ages)) if ages else float("nan"),
                   max_age=float(np.max(ages)) if ages else float("nan"),
                   draws_left=float(sim.channel._rng.random()), ok=True,
                   age_trace=trace if c["block"] == "exact" else None)
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-800:])
    return out


def build_grid() -> list:
    rng = np.random.default_rng(59)
    combos = []
    k = 0
    starts = [(2.5, 0.35), (4.5, -0.35), (5.5, 0.7), (3.0, 0.0)]
    link_cfgs = [("acoustic", "none", "none")] + [(l, w, t) for l in LINKS[1:] for w in WATERS for t in TERMINALS]
    for (link, water, terms), mode, policy in itertools.product(link_cfgs, ("clean", "noisy"), POLICIES):
        comms = [("L0", 1.0)] if link == "optical" else [(s, i) for s in SEVERITIES for i in INTERVALS]
        for sev, interval in comms:
            d0, v0 = starts[k % len(starts)]
            combos.append({"block": "grid", "link": link, "water": water, "terminals": terms, "sev": sev,
                           "interval": interval, "mode": mode, "policy": policy,
                           "buoyancy": BUOYANCY[k % len(BUOYANCY)], "belief": ("calibrated", "last_beacon")[k % 2],
                           "backend": "stand_in", "seed": int(rng.integers(1, 10_000)), "d0": d0, "v0": v0})
            k += 1
    for link, water, terms in link_cfgs:      # every link configuration on MuJoCo too
        combos.append({"block": "mujoco", "link": link, "water": water, "terminals": terms, "sev": "L2",
                       "interval": 1.0, "mode": "clean", "policy": "ppo", "buoyancy": None, "belief": "calibrated",
                       "backend": "mujoco", "seed": 5, "d0": 4.5, "v0": 0.35})
    return combos


def exact_grid() -> list:
    """Zero-policy episodes for the exact checks (block B)."""
    combos = []
    for mode, (sev, interval), (d0, v0), seed in itertools.product(
            ("clean", "noisy"), (("L2", 1.0), ("L3a", 3.0)), ((2.5, 0.35), (4.5, 0.2), (5.5, 0.7)), (1, 2)):
        base = {"block": "exact", "sev": sev, "interval": interval, "mode": mode, "policy": "zero",
                "buoyancy": None, "belief": "last_beacon", "backend": "stand_in", "seed": seed, "d0": d0, "v0": v0}
        combos.append(dict(base, link="acoustic", water="none", terminals="none"))
        for link, water, terms in itertools.product(LINKS[1:], WATERS, TERMINALS):
            combos.append(dict(base, link=link, water=water, terminals=terms))
    return combos


def acoustic_hashes() -> dict:
    """Block D: the same 96 cells the reference was taken on (see the module docstring)."""
    import yaml
    from buoyancy import BuoyancyConfig
    from dynamics_model import CalibrationParameters
    from evaluate_comm_degradation_curve import COMM_CONFIGS
    from fleet_belief import BeliefSpec
    from fleet_env import FleetConfig, FleetSim
    from optimizer import load_vehicle_spec
    from sensor_noise import load_sensor_noise
    from validate_fleet_policy import load_fleet_policy
    spec = load_vehicle_spec("vehicle_spec.yaml")
    with open("calibrated_parameters_a1.yaml") as f:
        truth = CalibrationParameters(**yaml.safe_load(f)["parameters"])
    pol = load_fleet_policy("fleet_policy_seed1", None)
    out = {}
    cfgs = {"clean": FleetConfig(), "noisy": FleetConfig(sensor_noise=load_sensor_noise("typical"),
                                                         buoyancy=BuoyancyConfig(offset_n=(-5.0, 5.0)))}
    for backend in ("stand_in", "mujoco"):
        for mode, fc in cfgs.items():
            fc = FleetConfig(**{**fc.__dict__, "dynamics_backend": backend})
            for sev in ("L0", "L2", "L3a", "L3b"):
                for interval in (0.5, 1.0, 3.0):
                    for belief in ("calibrated", "last_beacon"):
                        bs = (BeliefSpec("physics", "calibrated", truth) if belief == "calibrated"
                              else BeliefSpec(belief, belief))
                        sim = FleetSim(truth, spec.vehicle, fc, COMM_CONFIGS[sev]["packet_loss_config"],
                                       COMM_CONFIGS[sev]["bandwidth_config"], belief=bs, beacon_interval_s=interval)
                        h = hashlib.sha256()
                        for seed, d0, v0 in ((1, 2.5, 0.3), (7, 5.0, -0.5)):
                            if backend == "mujoco" and seed == 7:
                                continue
                            obs = sim.reset(comm_seed=seed, initial_separation=d0, closing_speed=v0)
                            for _ in range(200):
                                a, _ = pol.predict(np.stack(obs), deterministic=True)
                                obs, r, term, trunc, info = sim.step(list(a))
                                h.update(np.stack(obs).tobytes())
                                h.update(np.float64(r).tobytes())
                                h.update(repr((info["delivered"], info["belief_errors"], info["true_separation"],
                                               term, trunc)).encode())
                                if term or trunc:
                                    break
                        out[f"{backend}|{mode}|{sev}|{interval}|{belief}"] = h.hexdigest()
    return out


def exact_checks(rows) -> list:
    """Block B. Returns [(name, passed, detail)]."""
    checks = []
    key = lambda r: (r["mode"], r["sev"], r["interval"], r["d0"], r["v0"], r["seed"])
    by = {}
    for r in rows:
        by[(key(r), r["link"], r["water"], r["terminals"])] = r
    keys = sorted({key(r) for r in rows}, key=str)

    bad = [(k, l, t) for k in keys for l in LINKS[1:] for t in TERMINALS
           if not all(by[(k, l, WATERS[i], t)]["optical_delivered"] >= by[(k, l, WATERS[i + 1], t)]["optical_delivered"]
                      for i in range(len(WATERS) - 1))]
    n = len(keys) * (len(LINKS) - 1) * len(TERMINALS)
    checks.append(("Optical deliveries nested by water: clear >= coastal >= custom (1/m) >= harbour, episode by "
                   "episode", not bad, f"{n - len(bad)}/{n} episode sets" + (f"; first failure {bad[0]}" if bad else "")))

    fwd = [by[(k, l, w, "forward_cone")] for k in keys if k[0] == "clean" for l in LINKS[1:] for w in WATERS]
    leaked = [r for r in fwd if r["optical_delivered"]]
    checks.append(("Forward cones deliver nothing from the line start (clean, zero policy)", not leaked,
                   f"{len(fwd) - len(leaked)}/{len(fwd)} episodes with no optical delivery"))

    red_bad, age_bad, n_red = [], [], 0
    for k in keys:
        a = by[(k, "acoustic", "none", "none")]
        for w, t in itertools.product(WATERS, TERMINALS):
            r = by[(k, "hybrid_redundant", w, t)]
            n_red += 1
            if (r["acoustic_sent"], r["acoustic_delivered"], r["draws_left"]) != (a["acoustic_sent"],
                                                                                a["acoustic_delivered"], a["draws_left"]):
                red_bad.append((k, w, t))
            if len(r["age_trace"]) != len(a["age_trace"]) or any(
                    x > y + 1e-9 for x, y in zip(r["age_trace"], a["age_trace"])):
                age_bad.append((k, w, t))
    checks.append(("Hybrid redundant: acoustic sends, deliveries and draws identical to acoustic only", not red_bad,
                   f"{n_red - len(red_bad)}/{n_red}" + (f"; first failure {red_bad[0]}" if red_bad else "")))
    checks.append(("Hybrid redundant: the teammate belief is never older than with acoustic only, tick by tick",
                   not age_bad, f"{n_red - len(age_bad)}/{n_red}" + (f"; first failure {age_bad[0]}" if age_bad else "")))

    pair_bad, n_pair = [], 0
    for k in keys:
        a = by[(k, "acoustic", "none", "none")]
        for w, t in itertools.product(WATERS, TERMINALS):
            r = by[(k, "hybrid", w, t)]
            n_pair += 1
            if r["acoustic_sent"] + r["acoustic_skipped"] != a["acoustic_sent"] or r["draws_left"] != a["draws_left"]:
                pair_bad.append((k, w, t))
    checks.append(("Hybrid optical_first: acoustic draws paired with acoustic only (sent + skipped = sent; same "
                   "next draw)", not pair_bad, f"{n_pair - len(pair_bad)}/{n_pair}"))

    asym = [by[(k, "hybrid", w, "asymmetric")] for k in keys for w in WATERS]
    skipped = [r for r in asym if r["acoustic_skipped"]]
    checks.append(("Asymmetric terminals: optical_first runs as redundant (no acoustic beacon skipped)", not skipped,
                   f"{len(asym) - len(skipped)}/{len(asym)}"))
    omni_skip = [by[(k, "hybrid", w, "omni")] for k in keys for w in ("clear", "coastal")]
    checks.append(("Omni terminals in clear/coastal water: optical_first skips acoustic beacons (the link is up)",
                   all(r["acoustic_skipped"] > 0 for r in omni_skip),
                   f"{sum(r['acoustic_skipped'] > 0 for r in omni_skip)}/{len(omni_skip)}"))
    return checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--skip-acoustic-reference", action="store_true", help="skip block D (about 1 minute)")
    args = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    grid, exact = build_grid(), exact_grid()
    rng = np.random.default_rng(0)
    rerun_idx = sorted(rng.choice(len(grid), size=min(120, len(grid)), replace=False).tolist())
    todo = grid + exact + [dict(grid[i], block="rerun", rerun_of=i) for i in rerun_idx]
    print(f"{len(grid)} grid + {len(exact)} exact + {len(rerun_idx)} reruns = {len(todo)} episodes, "
          f"{args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(run_combo, todo, chunksize=8))
    g = [r for r in results if r["block"] in ("grid", "mujoco")]
    ex = [r for r in results if r["block"] == "exact"]
    re = [r for r in results if r["block"] == "rerun"]
    failed = [r for r in results if not r["ok"]]
    checks = [("Every episode runs to the end with finite observations and consistent bookkeeping", not failed,
               f"{len(results) - len(failed)}/{len(results)} episodes"
               + (f"; first failure: {failed[0]['error']}" if failed else ""))]
    if not [r for r in ex if not r["ok"]]:
        checks += exact_checks(ex)
    mism = [i for i, r in zip(rerun_idx, re) if not r["ok"] or r["hash"] != results[i]["hash"]]
    checks.append(("Determinism: reruns give identical observations and rewards", not mism,
                   f"{len(re) - len(mism)}/{len(re)} reruns identical"))
    if not args.skip_acoustic_reference:
        if os.path.exists(REFERENCE):
            ref = json.load(open(REFERENCE))
            now = acoustic_hashes()
            diff = [k for k in ref if ref[k] != now.get(k)]
            checks.append(("Acoustic runs unchanged: every observation, reward and delivery identical to main "
                           "before this change", not diff, f"{len(ref) - len(diff)}/{len(ref)} cells identical"))
        else:
            checks.append(("Acoustic runs unchanged", False, f"reference {REFERENCE} missing"))
    elapsed = time.time() - t0

    # a few descriptive numbers from the grid (not pass/fail)
    ok = [r for r in g if r["ok"] and r["block"] == "grid"]

    def up(rs):
        s = sum(r["optical_sent"] for r in rs)
        return sum(r["optical_delivered"] for r in rs) / s if s else float("nan")
    desc = {t: up([r for r in ok if r["terminals"] == t and r["link"] != "acoustic" and r["policy"] != "zero"])
            for t in TERMINALS}
    desc_water = {w: up([r for r in ok if r["water"] == w and r["terminals"] == "omni"]) for w in WATERS}

    lines = [f"# Stress test: optical and hybrid links (PROJECT_STATUS.md #59)\n",
             f"{time.strftime('%Y-%m-%d')} · {len(results)} episodes · {elapsed / 60:.1f} min · "
             f"`python stress_test_optical.py`\n",
             "## What was run\n",
             f"- **Grid ({sum(r['block'] == 'grid' for r in g)} episodes, stand-in physics):** every link (acoustic, "
             "optical, hybrid optical_first, hybrid redundant) x water (clear, coastal, harbour, custom c = 1/m) x "
             "terminals (omni, forward_cone, fore_aft_cones, down_cone, and an asymmetric fore/aft pair with a 30 deg "
             "beam and 60 deg receiver half-angle) x comm L0/L2/L3a/L3b x beacons every 1 s and 3 s x clean/noisy x "
             "policy (PPO, SAC, TD3, random, zero), with buoyancy cycling through none, -5, +5 and -5..+5 N and the "
             "belief between calibrated and last_beacon. Optical-only runs have no acoustic channel, so they run once "
             "per water, terminals, mode and policy.",
             f"- **MuJoCo ({sum(r['block'] == 'mujoco' for r in g)} episodes):** every link configuration once.",
             f"- **Exact checks ({len(ex)} episodes):** the zero policy (no commands, so the vehicles move the same way "
             "whatever the link) on L2@1s and L3a@3s, three starts, two seeds, clean and noisy.",
             f"- **Determinism:** {len(re)} grid episodes rerun.",
             "- **Acoustic unchanged:** 96 acoustic cells hashed observation by observation against the reference "
             "taken on main (8588809) before this change.\n",
             "## Checks\n", "| Check | Result | Detail |", "|---|---|---|"]
    for name, passed, detail in checks:
        lines.append(f"| {name} | {'PASS' if passed else '**FAIL**'} | {detail} |")
    lines += ["", "## What the shipped policies do to pointing (descriptive)\n",
              "Share of optical beacons delivered across the grid (PPO, SAC, TD3 and random policies; every water):\n",
              "| Terminals | Optical delivered |", "|---|---|"]
    lines += [f"| {t} | {100 * v:.0f}% |" for t, v in desc.items()]
    lines += ["", "With omnidirectional terminals, by water (every policy):\n", "| Water | Optical delivered |",
              "|---|---|"]
    lines += [f"| {w}{' (c = 1/m)' if w == 'custom' else ''} | {100 * v:.0f}% |" for w, v in desc_water.items()]
    lines += ["", "Cone terminals deliver little because the shipped policies were never rewarded for holding "
              "attitude: they roll and pitch freely while holding formation (#50), so a narrow cone rarely stays "
              "on the teammate. Omni terminals are the fair baseline for these policies.\n"]
    n_fail = sum(not p for _, p, _ in checks)
    lines.append(f"**{len(checks) - n_fail}/{len(checks)} checks passed.**\n")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump({"checks": [{"name": n, "passed": bool(p), "detail": d} for n, p, d in checks],
                   "episodes": len(results), "elapsed_s": round(elapsed, 1),
                   "failures": [{k: v for k, v in r.items() if k not in ("hash", "age_trace")} for r in failed[:20]],
                   "optical_delivered_by_terminals": desc, "optical_delivered_by_water_omni": desc_water},
                  f, indent=2, default=str)
    print("\n".join(lines))
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
