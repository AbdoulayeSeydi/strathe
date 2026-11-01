#!/usr/bin/env python3
"""
comms_study.py -- `strathe comms-study`: how far a fleet's coordination survives acoustic latency, bandwidth limits and
packet loss (PROJECT_STATUS.md #89, calibration/COMMS_STUDY.md).

It runs the N-vehicle formation task (fleet_coordination.py) over the shared acoustic channel (acoustic_network.py)
on Strathe's calibrated vehicle physics, starting from one base setting and changing ONE thing at a time:

    latency          the modem's decoding delay, s (on top of travel time and time on the air)
    bit_rate         the modem's bit rate, bit/s (sets each beacon's time on the air, so the TDMA cycle)
    loss             packet loss (independent)
    burst            the same loss in bursts of this many packets on average
    fleet_size       vehicles sharing the channel
    beacon_interval  seconds between each vehicle's beacons
    spacing          formation spacing, m (range: travel time, guard times)
    nav_drift        navigation drift, % of distance travelled (each vehicle's dead reckoning)
    fix_error        each vehicle's starting position-fix error, m
    mac              tdma (take turns) or aloha (send when due)
    belief           how a vehicle predicts a teammate between beacons
    controller       the coordination strategy (built in, or your own module:Class)

Every value runs the same episodes (same seeds: same leader turns, same starts), so differences come from the
setting, not luck. It writes:
    episodes.csv   one row per episode
    summary.csv    one row per setting: success rate with its 95% interval, formation error, separation, the channel
    report.md      what holds and where it breaks, per setting, with the plots
    plots/*.png    success and formation error against each setting
    study.yaml     everything needed to repeat it

    python comms_study.py --out my_study                       # the default study (~10 min on 8 cores)
    python comms_study.py --modem slow --fleet-size 6 --out s6 # your modem class and fleet
    python comms_study.py --sweep latency loss --values loss=0,0.2,0.4,0.6 --out lat_loss
    python comms_study.py --controller my_ctrl:MyFormation --controller-path ~/my_code --out mine
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from acoustic_network import MODEM_PRESETS, LossModel, ModemConfig, NetworkConfig  # noqa: E402
from fleet_coordination import FORMATIONS, CoordinationConfig, CoordinationSim, FormationSpec  # noqa: E402
from fleet_stats import wilson_ci  # noqa: E402

DEFAULT_PARAMS = os.path.join(HERE, "calibrated_parameters_fleet_truth.yaml")
DEFAULT_SPEC = os.path.join(HERE, "vehicle_spec.yaml")
SUCCESS_LINE = 0.8   # "coordination holds" = at least this success rate

INF = math.inf

# name: (what it is, unit, parse, default values (easy -> hard), harder_when_larger, categorical)
AXES = {
    "latency": ("Modem decoding delay", "s", float, [0, 1, 2, 5, 10, 20], True, False),
    "bit_rate": ("Bit rate", "bit/s", float, [INF, 5000, 1000, 300, 100, 50], False, False),
    "loss": ("Packet loss (independent)", "", float, [0, 0.1, 0.2, 0.3, 0.5, 0.7], True, False),
    "burst": ("Mean loss-burst length", "packets", float, [1, 2, 4, 8, 16], True, False),
    "fleet_size": ("Vehicles sharing the channel", "", int, [2, 3, 4, 6, 8], True, False),
    "beacon_interval": ("Time between beacons", "s", float, [1, 2, 5, 10, 20], True, False),
    "spacing": ("Formation spacing", "m", float, [5, 10, 20, 50, 100], True, False),
    "nav_drift": ("Navigation drift", "% of distance", float, [0, 0.5, 1, 2, 5], True, False),
    "fix_error": ("Starting position-fix error", "m", float, [0, 0.5, 1, 2, 5], True, False),
    "mac": ("Channel sharing", "", str, ["tdma", "aloha"], None, True),
    "belief": ("Teammate prediction between beacons", "", str,
               ["oracle", "physics", "constant_velocity", "last_beacon"], None, True),
    "controller": ("Coordination strategy", "", str, ["leader_follower"], None, True),
}
DEFAULT_SWEEP = ["latency", "bit_rate", "loss", "burst", "fleet_size", "beacon_interval", "mac", "belief", "nav_drift",
                 "fix_error"]
BURST_LOSS_IF_NONE = 0.3


def fmt(axis: str, v) -> str:
    if axis == "bit_rate":
        return "unlimited" if v == INF else f"{v:g}"
    if axis == "loss":
        return f"{v:.0%}"
    if axis == "burst":
        return "independent" if v == 1 else f"{v:g}"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def apply(base: CoordinationConfig, axis: str, v) -> CoordinationConfig:
    c = copy.deepcopy(base)
    net, modem = c.network, c.network.modem
    if axis == "latency":
        modem.processing_latency_s = float(v)
    elif axis == "bit_rate":
        modem.bit_rate_bps = float(v)
    elif axis == "loss":
        net.loss = replace(net.loss, loss=float(v))
    elif axis == "burst":
        loss = net.loss.loss or BURST_LOSS_IF_NONE
        net.loss = replace(net.loss, loss=loss, burst_length=None if v == 1 else float(v))
    elif axis == "fleet_size":
        c.n_vehicles = int(v)
    elif axis == "beacon_interval":
        net.beacon_interval_s = float(v)
    elif axis == "spacing":
        c.formation = replace(c.formation, spacing_m=float(v))
    elif axis == "nav_drift":
        c.nav_drift_pct = float(v)
    elif axis == "fix_error":
        c.start_fix_error_m = float(v)
    elif axis == "mac":
        net.mac = str(v)
    elif axis == "belief":
        c.belief = str(v)
    elif axis == "controller":
        c.controller = str(v)
    else:
        raise ValueError(f"unknown setting {axis!r}")
    return c


# --- workers -----------------------------------------------------------------------------------------------------------
_SIM_CACHE = {}


def _init_worker(paths):
    for p in paths:
        if p not in sys.path:
            sys.path.insert(0, p)


def _run(task):
    key, axis, label, cfg, seed, truth, vehicle, belief_params = task
    sim = _SIM_CACHE.get(key)
    if sim is None:
        sim = _SIM_CACHE[key] = CoordinationSim(truth, vehicle, cfg, belief_params)
    row = sim.run_episode(seed)
    row.update(axis=axis, value=label)
    return row


# --- statistics --------------------------------------------------------------------------------------------------------
def _mean_ci(x):
    x = np.asarray([v for v in x if v == v and v != INF], float)
    if len(x) == 0:
        return math.nan, math.nan, math.nan
    m = float(x.mean())
    h = 1.96 * float(x.std(ddof=1)) / math.sqrt(len(x)) if len(x) > 1 else math.nan
    return m, m - h, m + h


def summarize(rows, axis, values) -> list:
    out = []
    for v in values:
        r = [x for x in rows if x["axis"] == axis and x["value"] == fmt(axis, v)]
        if not r:
            continue
        k = sum(x["success"] for x in r)
        lo, hi = wilson_ci(k, len(r))
        e, elo, ehi = _mean_ci([x["formation_error_rms_m"] for x in r])
        out.append({
            "axis": axis, "value": fmt(axis, v), "episodes": len(r), "success_rate": k / len(r),
            "success_ci_low": lo, "success_ci_high": hi,
            "formation_error_m": e, "formation_error_ci_low": elo, "formation_error_ci_high": ehi,
            "formation_error_median_m": float(np.nanmedian([x["formation_error_rms_m"] for x in r])),
            "min_separation_m": float(min(x["min_separation_m"] for x in r)),
            "collisions": sum(x["collision"] for x in r),
            "belief_error_m": _mean_ci([x["belief_error_m"] for x in r])[0],
            "beacon_age_s": _mean_ci([x["beacon_age_s"] for x in r])[0],
            "delivery_ratio": _mean_ci([x["delivery_ratio"] for x in r])[0],
            "utilisation": _mean_ci([x["utilisation"] for x in r])[0],
            "nav_error_m": _mean_ci([x.get("nav_error_m", 0.0) for x in r])[0],
            "airtime_s": r[0]["airtime_s"], "tdma_cycle_s": r[0]["tdma_cycle_s"],
        })
    return out


def verdict(axis, summ) -> str:
    """One sentence: where coordination holds and where it breaks, for an ordered setting."""
    name, unit = AXES[axis][0], AXES[axis][1]
    u = f" {unit}" if unit and unit not in ("",) else ""
    if not summ:
        return f"**{name}:** no episodes."
    first_bad = next((s for s in summ if s["success_rate"] < SUCCESS_LINE), None)
    e0, e1 = summ[0]["formation_error_m"], summ[-1]["formation_error_m"]
    trend = f"Formation error goes from {e0:.2f} m to {e1:.2f} m across the range."
    if first_bad is None:
        return (f"**{name}:** coordination holds across the whole range tested (to {summ[-1]['value']}{u}): "
                f"success never drops below {SUCCESS_LINE:.0%}. {trend}")
    i = summ.index(first_bad)
    s = first_bad
    ci = f"{s['success_rate']:.0%} (95% interval {s['success_ci_low']:.0%}-{s['success_ci_high']:.0%})"
    if i == 0:
        return f"**{name}:** coordination already fails at {s['value']}{u}: success {ci}. {trend}"
    ok = summ[i - 1]
    return (f"**{name}:** coordination holds up to {ok['value']}{u} ({ok['success_rate']:.0%} success) and breaks at "
            f"{s['value']}{u}: success {ci}. {trend}")


def compare(axis, summ) -> str:
    name = AXES[axis][0]
    parts = [f"{s['value']}: {s['success_rate']:.0%} success ({s['success_ci_low']:.0%}-{s['success_ci_high']:.0%}), "
             f"{s['formation_error_m']:.2f} m" for s in summ]
    best = max(summ, key=lambda s: (s["success_rate"], -s["formation_error_m"])) if summ else None
    return f"**{name}:** " + "; ".join(parts) + (f". Best: {best['value']}." if best else "")


# --- plots -------------------------------------------------------------------------------------------------------------
def plot(axis, summ, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(summ))
    labels = [s["value"] for s in summ]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6))
    sr = np.array([s["success_rate"] for s in summ])
    a.errorbar(x, sr * 100, yerr=[(sr - [s["success_ci_low"] for s in summ]) * 100,
                                  (np.array([s["success_ci_high"] for s in summ]) - sr) * 100], fmt="o-", capsize=4)
    a.axhline(SUCCESS_LINE * 100, ls="--", lw=1, color="grey")
    a.set_ylim(-5, 105)
    a.set_ylabel("success (%)")
    e = np.array([s["formation_error_m"] for s in summ])
    lo = e - np.array([s["formation_error_ci_low"] for s in summ])
    hi = np.array([s["formation_error_ci_high"] for s in summ]) - e
    b.errorbar(x, e, yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], fmt="o-", capsize=4, color="C3")
    b.set_ylabel("formation error, RMS (m)")
    for ax in (a, b):
        ax.set_xticks(x, labels)
        unit = AXES[axis][1]
        ax.set_xlabel(AXES[axis][0] + (f" ({unit})" if unit else ""))
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# --- the study ---------------------------------------------------------------------------------------------------------
def base_config(a) -> CoordinationConfig:
    modem = ModemConfig.preset(a.modem, bit_rate_bps=a.bit_rate, packet_overhead_s=a.packet_overhead,
                               processing_latency_s=a.latency)
    if a.max_range is not None:
        modem.max_range_m = a.max_range
    net = NetworkConfig(modem=modem, loss=LossModel(a.loss, None if a.burst in (None, 1) else a.burst),
                        mac=a.mac, beacon_interval_s=a.beacon_interval, beacon_bytes=a.beacon_bytes)
    offsets = None
    if a.offsets:
        nums = [float(x) for x in a.offsets.replace(",", " ").split()]
        if len(nums) % 2:
            raise ValueError("--offsets takes x y pairs")
        offsets = [nums[i:i + 2] for i in range(0, len(nums), 2)]
    form = FormationSpec(a.formation if not offsets else "custom", a.spacing, offsets, a.formation_orientation)
    cfg = CoordinationConfig(n_vehicles=a.fleet_size, formation=form, duration_s=a.duration, speed_mps=a.speed,
                             turn_deg=a.turn, tolerance_m=a.tolerance, controller=a.controller, belief=a.belief,
                             network=net, nav_drift_pct=a.nav_drift, start_fix_error_m=a.start_fix_error)
    cfg.validate()
    return cfg


def parse_values(specs) -> dict:
    out = {}
    for s in specs or []:
        if "=" not in s:
            raise ValueError(f"--values takes SETTING=v1,v2,...; got {s!r}")
        axis, vals = s.split("=", 1)
        if axis not in AXES:
            raise ValueError(f"unknown setting {axis!r}; choose from {', '.join(AXES)}")
        parse = AXES[axis][2]
        out[axis] = [INF if v.strip().lower() in ("inf", "unlimited") else parse(v) for v in vals.split(",") if v.strip()]
    return out


def write_csv(path, rows, fields=None):
    fields = fields or list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run_study(a) -> int:
    from build_two_agent_cores import load_calibrated_parameters
    from optimizer import load_vehicle_spec
    try:
        base = base_config(a)
        values = parse_values(a.values)
    except ValueError as e:
        print(f"❌ {e}")
        return 2
    sweep = a.sweep or DEFAULT_SWEEP
    for ax in sweep:
        if ax not in AXES:
            print(f"❌ unknown setting {ax!r}; choose from {', '.join(AXES)}")
            return 2
    if a.controller != "leader_follower" and "controller" in sweep and "controller" not in values:
        values["controller"] = ["leader_follower", a.controller]
    truth = load_calibrated_parameters(a.params)
    vehicle = load_vehicle_spec(a.vehicle_spec).vehicle
    belief_params = load_calibrated_parameters(a.belief_params, quiet=True) if a.belief_params else None
    if a.controller_path:
        for p in a.controller_path:
            sys.path.insert(0, os.path.abspath(os.path.expanduser(p)))
    paths = [os.path.abspath(os.path.expanduser(p)) for p in (a.controller_path or [])] + [os.getcwd()]
    _init_worker(paths)
    plan = {}
    for ax in sweep:
        vals = values.get(ax, AXES[ax][3])
        if a.quick and ax not in values:
            vals = vals[::2] if len(vals) > 3 else vals
        cfgs = []
        for v in vals:
            try:
                cfgs.append((v, apply(base, ax, v)))
                cfgs[-1][1].validate()
            except ValueError as e:
                print(f"❌ {ax}={fmt(ax, v)} doesn't make a valid setting: {e}")
                return 2
        plan[ax] = cfgs
    try:  # a controller that can't load fails here, not in every worker
        from fleet_coordination import LeaderPlan, load_controller
        for ax, cfgs in plan.items():
            for _, c in cfgs:
                load_controller(c.controller, c, LeaderPlan(c, np.random.default_rng(0)))
    except ValueError as e:
        print(f"❌ {e}")
        return 2
    seeds = [a.seed + k for k in range(a.episodes)]
    tasks = []
    for ax, cfgs in plan.items():
        for v, c in cfgs:
            key = (ax, fmt(ax, v))
            tasks += [(key, ax, fmt(ax, v), c, s, truth, vehicle, belief_params) for s in seeds]
    os.makedirs(a.out, exist_ok=True)
    os.makedirs(os.path.join(a.out, "plots"), exist_ok=True)
    print(f"Base setting: {base.n_vehicles} vehicles, {base.formation.shape} at {base.formation.spacing_m:g} m, "
          f"{base.belief} beliefs, {base.controller}.", flush=True)
    from acoustic_network import AcousticNetwork
    print("Channel: " + AcousticNetwork(base.n_vehicles, base.network, 0, base.max_distance_m()).describe(), flush=True)
    print(f"Running {len(tasks)} episodes ({len(plan)} settings, {a.episodes} episodes per value) on "
          f"{a.workers} worker(s)...", flush=True)
    rows, t0 = [], time.time()
    if a.workers <= 1:
        for k, t in enumerate(tasks, 1):
            rows.append(_run(t))
            if k % max(1, len(tasks) // 20) == 0:
                print(f"  {k}/{len(tasks)} ({time.time() - t0:.0f} s)", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker, initargs=(paths,)) as ex:
            futs = [ex.submit(_run, t) for t in tasks]
            for k, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if k % max(1, len(tasks) // 20) == 0:
                    el = time.time() - t0
                    print(f"  {k}/{len(tasks)} ({el:.0f} s, about {el / k * (len(tasks) - k):.0f} s left)", flush=True)
    rows.sort(key=lambda r: (list(plan).index(r["axis"]), r["value"], r["seed"]))
    write_csv(os.path.join(a.out, "episodes.csv"), rows)
    summary, sections = [], []
    for ax, cfgs in plan.items():
        summ = summarize(rows, ax, [v for v, _ in cfgs])
        summary += summ
        plot(ax, summ, os.path.join(a.out, "plots", f"{ax}.png"))
        sections.append((ax, summ))
    write_csv(os.path.join(a.out, "summary.csv"), summary)
    with open(os.path.join(a.out, "study.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"command": " ".join(sys.argv), "base": _plain(base.to_dict()), "episodes": a.episodes,
                        "seeds": seeds, "params": os.path.abspath(a.params),
                        "vehicle_spec": os.path.abspath(a.vehicle_spec),
                        "settings": {ax: [fmt(ax, v) for v, _ in cfgs] for ax, cfgs in plan.items()}},
                       f, sort_keys=False)
    write_report(a, base, sections, os.path.join(a.out, "report.md"), time.time() - t0)
    print(f"\n✅ Wrote {os.path.join(a.out, 'report.md')} (and episodes.csv, summary.csv, plots/)")
    for ax, summ in sections:
        print("  " + (compare(ax, summ) if AXES[ax][5] else verdict(ax, summ)).replace("**", ""))
    return 0


def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return x


def write_report(a, base, sections, path, wall_s):
    from acoustic_network import AcousticNetwork
    net = AcousticNetwork(base.n_vehicles, base.network, 0, base.max_distance_m())
    L = ["# Fleet coordination under acoustic limits\n",
         f"{base.n_vehicles} vehicles in a {base.formation.shape} formation ({base.formation.spacing_m:g} m spacing, "
         f"{'compass-fixed' if base.formation.orientation == 'fixed' else 'turning with the leader'}) follow a leader "
         f"whose {base.turn_deg:g}-degree turns they can't predict, at {base.speed_mps:g} m/s, for "
         f"{base.duration_s:g} s. They know where the leader and each other are only through acoustic beacons. "
         f"Vehicle physics: `{os.path.basename(a.params)}`. Controller: `{base.controller}`.\n",
         f"**Base channel:** {net.describe()}. Teammate prediction between beacons: `{base.belief}`.\n",
         f"**Success** = formation error under {base.tolerance_m:g} m RMS (each follower's distance from its true "
         f"slot, after {base.settle_s:g} s) and no two vehicles closer than {base.safety_radius_m:g} m. "
         f"**Coordination holds** = at least {SUCCESS_LINE:.0%} success. Each value ran the same {a.episodes} episodes "
         f"(seeds {a.seed}-{a.seed + a.episodes - 1}: same leader turns, same starts); intervals are 95%.\n",
         "## What holds and what breaks\n"]
    for ax, summ in sections:
        L.append("- " + (compare(ax, summ) if AXES[ax][5] else verdict(ax, summ)))
    L.append("")
    for ax, summ in sections:
        name, unit = AXES[ax][0], AXES[ax][1]
        L.append(f"## {name}" + (f" ({unit})" if unit else "") + "\n")
        L.append(f"![{name}](plots/{ax}.png)\n")
        L.append("| Value | Success | Formation error (m) | Closest approach (m) | Beacon age (s) | Delivered | "
                 "Channel busy | Time on air (s) | TDMA cycle (s) |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for s in summ:
            cyc = "" if s["tdma_cycle_s"] != s["tdma_cycle_s"] else f"{s['tdma_cycle_s']:.1f}"
            L.append(f"| {s['value']} | {s['success_rate']:.0%} ({s['success_ci_low']:.0%}-{s['success_ci_high']:.0%}) "
                     f"| {s['formation_error_m']:.2f} ({s['formation_error_ci_low']:.2f}-{s['formation_error_ci_high']:.2f}) "
                     f"| {s['min_separation_m']:.1f} | {s['beacon_age_s']:.1f} | {s['delivery_ratio']:.0%} | "
                     f"{s['utilisation']:.0%} | {s['airtime_s']:.2f} | {cyc} |")
        L.append("")
        if ax == "burst":
            L.append(f"Bursts are at the base loss ({base.network.loss.loss or BURST_LOSS_IF_NONE:.0%}); "
                     f"'independent' is the same loss as coin flips.\n")
        if ax in ("nav_drift", "fix_error"):
            L.append("Each vehicle's own position error (mean, after the settling time): "
                     + ", ".join(f"{s['value']}: {s['nav_error_m']:.2f} m" for s in summ) + ". A follower's slot is off "
                     "by the difference between its error and the leader's: beacons can't fix that without ranging.\n")
        if ax == "bit_rate" and base.network.mac == "tdma":
            L.append("With TDMA each vehicle waits for its slot, so the cycle (vehicles x (time on air + guard)) is "
                     "the real update interval once it exceeds the beacon interval: that is how bandwidth limits "
                     "coordination.\n")
    L += ["## How to read this\n",
          "- **Beacon age** is how old the newest state each vehicle holds about a teammate is, on average. It is "
          "what latency, bandwidth and loss all turn into.",
          "- **Delivered** counts packets that reached a teammate; collisions, half-duplex losses (a vehicle can't "
          "hear while it transmits), range and fades are in `episodes.csv` (`n_collision`, `n_half_duplex`, "
          "`n_lost`, `n_out_of_range`, `n_no_slot`, `n_superseded`).",
          "- **Channel busy** is the fraction of time someone was transmitting; above 100% means transmissions "
          "overlapped (random access).",
          "- Model and limits: calibration/COMMS_STUDY.md and LIMITS.md. The channel model is generic: measure "
          "your own modem (`strathe fit-link`) for a lab-specific answer.\n",
          "## Repeat it\n", "```", "python strathe.py comms-study " + " ".join(sys.argv[1:]), "```",
          f"\n{len(sections)} settings, {sum(s['episodes'] for _, sm in sections for s in sm)} episodes, "
          f"{wall_s / 60:.1f} min.\n"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="strathe comms-study", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="comms_study", help="output folder (default comms_study)")
    g = ap.add_argument_group("the study")
    g.add_argument("--sweep", nargs="+", metavar="SETTING", help=f"settings to change one at a time (default: "
                   f"{' '.join(DEFAULT_SWEEP)}); any of {', '.join(AXES)}")
    g.add_argument("--values", nargs="+", metavar="SETTING=V1,V2", help="your own values for a setting, e.g. "
                   "latency=0,3,6,12 or bit_rate=inf,2000,500 (otherwise each setting's defaults)")
    g.add_argument("--episodes", type=int, default=30, help="episodes per value (default 30)")
    g.add_argument("--seed", type=int, default=1000, help="first episode seed (default 1000)")
    g.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1), help="CPU cores to use")
    g.add_argument("--quick", action="store_true", help="every other default value, for a first look")
    f = ap.add_argument_group("the fleet")
    f.add_argument("--fleet-size", type=int, default=4, help="vehicles (2-12, default 4)")
    f.add_argument("--formation", choices=[x for x in FORMATIONS if x != "custom"], default="wedge")
    f.add_argument("--offsets", help="your own formation: x y for each follower, m in the leader's frame "
                                     "(x forward, y starboard), e.g. '-10 10 -10 -10 -20 0'")
    f.add_argument("--formation-orientation", choices=["fixed", "leader"], default="fixed",
                   help="fixed: keeps its compass orientation (default); leader: turns with the leader")
    f.add_argument("--spacing", type=float, default=10.0, help="formation spacing, m (default 10)")
    f.add_argument("--speed", type=float, default=0.2, help="leader speed, m/s (default 0.2)")
    f.add_argument("--turn", type=float, default=60.0, help="leader turn size, degrees (default 60)")
    f.add_argument("--duration", type=float, default=300.0, help="episode length, s (default 300)")
    f.add_argument("--tolerance", type=float, default=3.0, help="success: RMS formation error under this, m (default 3)")
    f.add_argument("--controller", default="leader_follower",
                   help="leader_follower (built in) or your own module:Class (COMMS_STUDY.md)")
    f.add_argument("--controller-path", nargs="+", help="folders to import your controller from")
    f.add_argument("--belief", default="constant_velocity",
                   choices=["last_beacon", "constant_velocity", "physics", "oracle"],
                   help="how a vehicle predicts a teammate between beacons (default constant_velocity)")
    f.add_argument("--nav-drift", type=float, default=0.0, metavar="PCT",
                   help="navigation drift: each vehicle's dead reckoning drifts this %% of the distance it travels "
                        "(default 0: exact)")
    f.add_argument("--start-fix-error", type=float, default=0.0, metavar="M",
                   help="each vehicle's starting position fix is off by this much, RMS, m (default 0)")
    f.add_argument("--params", default=DEFAULT_PARAMS, help="the vehicles' calibrated physics (default: the fleet's)")
    f.add_argument("--vehicle-spec", default=DEFAULT_SPEC, help="the vehicle description for --params")
    f.add_argument("--belief-params", help="physics beliefs: the model the vehicles predict teammates with "
                                           "(default: the true one); e.g. an uncalibrated model")
    c = ap.add_argument_group("the channel (base setting)")
    c.add_argument("--modem", choices=list(MODEM_PRESETS), default="medium",
                   help="; ".join(f"{k}: {v['note']}" for k, v in MODEM_PRESETS.items()) + " (default medium)")
    c.add_argument("--bit-rate", type=float, help="bit/s (overrides the preset)")
    c.add_argument("--packet-overhead", type=float, help="preamble on the air per packet, s (overrides the preset)")
    c.add_argument("--latency", type=float, help="modem decoding delay, s (overrides the preset)")
    c.add_argument("--max-range", type=float, help="modem range, m (default 3000)")
    c.add_argument("--loss", type=float, default=0.0, help="packet loss, 0-1 (default 0)")
    c.add_argument("--burst", type=float, help="losses come in bursts of this many packets on average")
    c.add_argument("--mac", choices=["tdma", "aloha"], default="tdma",
                   help="tdma: take turns (default); aloha: send when due, collisions possible")
    c.add_argument("--beacon-interval", type=float, default=2.0, help="s between each vehicle's beacons (default 2)")
    c.add_argument("--beacon-bytes", type=int, default=32, help="beacon size, bytes (default 32)")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.episodes < 1:
        print("❌ --episodes must be at least 1")
        return 2
    for name in ("params", "vehicle_spec", "belief_params"):
        p = getattr(a, name)
        if p and not os.path.exists(p):
            print(f"❌ {name.replace('_', '-')} file not found: {p}")
            return 2
    return run_study(a)


if __name__ == "__main__":
    sys.exit(main())
