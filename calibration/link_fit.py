"""
Calibrate the communication links from a lab's own message logs (PROJECT_STATUS.md #68): `strathe fit-link`.

Input: one or more CSV files, one row per message sent (templates/link_log_template.csv):
    send_time_s       when it was sent (s, any clock shared by both vehicles)
    receive_time_s    when it arrived; empty if it was lost        (or: delivered, 1 / 0)
    range_m           distance between sender and receiver when it was sent (m)
    link              acoustic | optical (optional; default from --link)
    pointed           optical only, optional: 1 if the terminals were facing each other, 0 if not (0 rows are
                      left out: the fit is for a pointed link)

Acoustic fit (comm_channel.CommChannel):
- delivery: the drop probability's distance ramp (PacketLossConfig: near/far distance, near/far drop probability),
  by maximum likelihood; kept only if it beats one constant drop probability by AIC, else a constant is used;
- latency: fixed delay + range / speed of sound, by robust least squares. The fixed delay is the modem's own
  processing and packet airtime (often seconds for acoustic modems); with too narrow a range span, or an implausible
  speed, the speed is held at 1500 m/s and only the delay is fitted.
Optical fit (comm_channel.OpticalLinkConfig): the delivery S-curve on received light power exp(-c d)/d^2. Its rated
range and curve width by maximum likelihood, with the water's attenuation c held (--optical-water,
--optical-attenuation, or coastal): logged messages can't measure c (fit_optical_delivery says why), and it only
shapes delivery beyond the logged ranges. Latency: the median delay.

Output (in --out): link_calibration.yaml, which `strathe fleet` / `train-fleet` (--link-calibration) and the
experiment YAML (communication.link_calibration) use in place of the assumed link values; link_calibration_report.md;
and plots of delivery and latency against range.

    python link_fit.py --log modem_log.csv --out link_cal
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

from comm_channel import (OpticalLinkConfig, PacketLossConfig, WATER_ATTENUATION_PER_M, _REFERENCE_WATER,
                          _MIN_RANGE_M, log_received_power)

FORMAT_VERSION = 1
LINKS = ("acoustic", "optical")
N_BINS = 8
EPS = 1e-6


# ------------------------------------------------------------------ loading

def load_logs(paths: list, default_link: str = "acoustic") -> pd.DataFrame:
    frames = []
    for p in paths:
        d = pd.read_csv(p, comment="#")
        d.columns = [c.strip() for c in d.columns]
        missing = {"send_time_s", "range_m"} - set(d.columns)
        if missing:
            raise ValueError(f"{p}: missing column(s) {sorted(missing)} (see templates/link_log_template.csv)")
        if "receive_time_s" not in d.columns and "delivered" not in d.columns:
            raise ValueError(f"{p}: needs receive_time_s (empty when lost) or delivered (1/0)")
        if "receive_time_s" in d.columns:
            d["receive_time_s"] = pd.to_numeric(d["receive_time_s"], errors="coerce")
            d["delivered"] = d["receive_time_s"].notna().astype(int) if "delivered" not in d.columns else d["delivered"]
        else:
            d["receive_time_s"] = np.nan
        d["delivered"] = pd.to_numeric(d["delivered"], errors="coerce")
        if d["delivered"].isna().any() or not set(d["delivered"].unique()) <= {0, 1}:
            raise ValueError(f"{p}: delivered must be 1 or 0 on every row")
        if "link" not in d.columns:
            d["link"] = default_link
        d["link"] = d["link"].astype(str).str.strip().str.lower()
        bad = set(d["link"]) - set(LINKS)
        if bad:
            raise ValueError(f"{p}: unknown link value(s) {sorted(bad)}; use acoustic or optical")
        d["range_m"] = pd.to_numeric(d["range_m"], errors="coerce")
        if d["range_m"].isna().any() or (d["range_m"] < 0).any():
            raise ValueError(f"{p}: range_m must be a distance in metres on every row")
        d["file"] = os.path.basename(p)
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    if "pointed" in out.columns:
        out["pointed"] = pd.to_numeric(out["pointed"], errors="coerce").fillna(1).astype(int)
    else:
        out["pointed"] = 1
    return out


# ------------------------------------------------------------------ fits

def _bernoulli_nll(p_deliver, y):
    p = np.clip(p_deliver, EPS, 1 - EPS)
    return float(-np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ramp_drop(d, near_d, far_d, near_p, far_p):
    """PacketLossConfig.drop_probability, vectorised (the fit evaluates it many times)."""
    d = np.asarray(d, dtype=float)
    if far_d <= near_d:
        return np.where(d <= near_d, near_p, far_p)
    return np.interp(d, [near_d, far_d], [near_p, far_p])


def bins_table(d, y, predict, n_bins=N_BINS) -> list:
    """Observed and fitted delivery in range bins of about equal message counts."""
    edges = np.unique(np.quantile(d, np.linspace(0, 1, n_bins + 1)))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d >= lo) & ((d < hi) | (hi == edges[-1]) & (d <= hi))
        if not m.any():
            continue
        n, k = int(m.sum()), int(y[m].sum())
        z = 1.96  # Wilson interval
        phat = k / n
        den = 1 + z * z / n
        c = (phat + z * z / (2 * n)) / den
        h = z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
        out.append({"range_m": [round(float(lo), 2), round(float(hi), 2)], "messages": n,
                    "delivered": round(phat, 3), "ci95": [round(max(0, c - h), 3), round(min(1, c + h), 3)],
                    "fitted": round(float(np.mean(predict(d[m]))), 3)})
    return out


def fit_acoustic_delivery(d: np.ndarray, y: np.ndarray) -> dict:
    """PacketLossConfig ramp by maximum likelihood over knot pairs at range quantiles; constant if not better."""
    q = np.unique(np.quantile(d, np.linspace(0, 1, 21)))
    best = None
    for i, dn in enumerate(q):
        for df in q[i + 1:]:
            def nll(x, dn=dn, df=df):
                return _bernoulli_nll(1 - ramp_drop(d, dn, df, x[0], x[1]), y)
            r = minimize(nll, x0=[0.2, 0.5], bounds=[(0, 1), (0, 1)], method="L-BFGS-B")
            if best is None or r.fun < best[0]:
                best = (float(r.fun), float(dn), float(df), float(r.x[0]), float(r.x[1]))
    p_const = 1 - float(y.mean())
    nll_const = _bernoulli_nll(np.full(len(y), 1 - p_const), y)
    aic_const, aic_ramp = 2 * 1 + 2 * nll_const, (2 * 4 + 2 * best[0]) if best else math.inf
    if best is not None and aic_ramp < aic_const - 2:
        _, dn, df, pn, pf = best
        model = "ramp"
    else:
        dn, df, pn, pf = 0.0, float(d.max()) if len(d) else 1000.0, p_const, p_const
        model = "constant"
    plc = PacketLossConfig(round(dn, 2), round(max(df, dn), 2), round(pn, 4), round(pf, 4))
    plc.validate()
    return {"model": model, "packet_loss": {"near_distance_m": plc.near_distance_m, "far_distance_m": plc.far_distance_m,
                                            "near_field_prob": plc.near_field_prob, "far_field_prob": plc.far_field_prob},
            "aic": {"ramp": round(aic_ramp, 1) if best else None, "constant": round(aic_const, 1)},
            "bins": bins_table(d, y, lambda x: 1 - ramp_drop(x, plc.near_distance_m, plc.far_distance_m,
                                                             plc.near_field_prob, plc.far_field_prob))}


def fit_latency(d: np.ndarray, delay: np.ndarray, warnings: list, plausible=(1300.0, 1700.0)) -> dict:
    """delay = fixed + d / c, robustly (two passes, dropping residuals beyond 3 MADs). c held at 1500 m/s when the
    range span is too narrow or the fitted speed isn't plausible for sound in water."""
    if (delay < 0).any():
        warnings.append(f"{int((delay < 0).sum())} message(s) arrive before they were sent: the two vehicles' clocks "
                        f"aren't synchronized; latency below is unreliable")
    keep = np.ones(len(d), bool)
    span = float(d.max() - d.min()) if len(d) else 0.0
    c, a = 1500.0, float(np.median(delay - d / 1500.0)) if len(d) else 0.0
    fitted_speed = False
    if len(d) >= 10 and span >= 100.0:
        for _ in range(2):
            b, a_ = np.polyfit(d[keep], delay[keep], 1)
            res = delay - (a_ + b * d)
            mad = np.median(np.abs(res[keep] - np.median(res[keep]))) or 1e-9
            keep = np.abs(res) <= 3 * 1.4826 * mad
        if b > 0 and plausible[0] <= 1 / b <= plausible[1]:
            c, a, fitted_speed = float(1 / b), float(a_), True
        else:
            warnings.append(f"the latency's slope gives a speed of sound of {('%.0f' % (1 / b)) if b > 0 else 'n/a'} "
                            f"m/s, outside {plausible[0]:.0f}-{plausible[1]:.0f}: held at 1500 m/s, only the fixed "
                            f"delay fitted")
            a = float(np.median(delay - d / 1500.0))
    elif len(d):
        warnings.append(f"ranges span only {span:.0f} m (under 100 m): the speed of sound is held at 1500 m/s and "
                        f"only the fixed delay is fitted")
    res = delay - (a + d / c)
    return {"speed_of_sound_mps": round(c, 1), "fixed_latency_s": round(max(a, 0.0), 4),
            "speed_fitted": fitted_speed, "residual_p90_s": round(float(np.percentile(np.abs(res), 90)), 4)
            if len(res) else None, "messages_used": int(keep.sum())}


def optical_probability(d, c, rated_range_m, curve_width):
    """OpticalLinkConfig.success_probability for a pointed link, vectorised."""
    ref = log_received_power(rated_range_m, WATER_ATTENUATION_PER_M[_REFERENCE_WATER])
    dd = np.maximum(np.atleast_1d(np.asarray(d, dtype=float)), _MIN_RANGE_M)
    logp = -c * dd - 2.0 * np.log(dd)
    return 1.0 / (1.0 + np.exp(-np.clip((logp - ref) / curve_width, -500, 500)))


DEFAULT_OPTICAL_WATER = "coastal"


def fit_optical_delivery(d: np.ndarray, y: np.ndarray, warnings: list, water: str = None,
                         attenuation_per_m: float = None) -> dict:
    """The rated range and curve width by maximum likelihood, with the water's attenuation c HELD: at a measured
    value (attenuation_per_m, e.g. from a transmissometer), a textbook water type, or coastal by default.

    Why c isn't fitted: delivery against range is one S-curve, pinned by its midpoint and steepness; c, the rated
    range and the width trade off along it, so logged messages can't separate the water's clarity from the modem's
    sensitivity (checked on synthetic logs over 0.2-80 m in clear, coastal and harbour water: every water fits
    within one log-likelihood unit). Inside the logged ranges the fitted curve is the same whatever c is held at;
    c only shapes how delivery falls off beyond them."""
    if attenuation_per_m is not None:
        c, held = float(attenuation_per_m), f"measured ({attenuation_per_m:g}/m)"
    else:
        name = water or DEFAULT_OPTICAL_WATER
        c, held = WATER_ATTENUATION_PER_M[name], f"{name} water ({WATER_ATTENUATION_PER_M[name]:g}/m)"
        if water is None:
            warnings.append(f"optical: the water's clarity is held at {held}: delivery against range can't measure "
                            f"it. Inside the logged ranges ({d.min():.0f}-{d.max():.0f} m) that doesn't matter; "
                            f"beyond them it sets how fast delivery falls off. Give --optical-water, or "
                            f"--optical-attenuation from a transmissometer, if you know it")
    best = None
    for r0 in (5.0, 15.0, 30.0, 80.0, 200.0):
        for w0 in (0.3, 1.0, 3.0):
            res = minimize(lambda x: _bernoulli_nll(optical_probability(d, c, math.exp(x[0]), math.exp(x[1])), y),
                           [math.log(r0), math.log(w0)], bounds=[(math.log(0.1), math.log(500)),
                                                                 (math.log(0.05), math.log(10))], method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
    R, w = math.exp(best.x[0]), math.exp(best.x[1])
    out = {"attenuation_per_m": round(c, 4), "attenuation_held": held, "rated_range_m": round(R, 3),
           "curve_width": round(w, 4)}
    cfg = OpticalLinkConfig(water="custom", attenuation_per_m=min(max(c, 1e-3), 20.0), rated_range_m=R,
                            curve_width=w)
    out["range_at_50pct_m"] = round(cfg.range_at(0.5), 2)
    out["bins"] = bins_table(d, y, lambda x: optical_probability(x, c, R, w))
    return out


def fit_logs(logs: pd.DataFrame, optical_water: str = None, optical_attenuation: float = None) -> dict:
    res = {"strathe_link_calibration": FORMAT_VERSION, "source": {"files": sorted(logs["file"].unique().tolist())},
           "warnings": []}
    for link in LINKS:
        g = logs[logs["link"] == link]
        if link == "optical":
            skipped = int((g["pointed"] == 0).sum())
            g = g[g["pointed"] == 1]
            if skipped:
                res["warnings"].append(f"optical: {skipped} message(s) sent with the terminals not facing each other "
                                       f"left out (the fit is for a pointed link)")
        if len(g) == 0:
            continue
        if len(g) < 30:
            res["warnings"].append(f"{link}: only {len(g)} messages; the fit is rough (a few hundred is better)")
        d, y = g["range_m"].to_numpy(float), g["delivered"].to_numpy(int)
        entry = {"messages": int(len(g)), "delivered_fraction": round(float(y.mean()), 4),
                 "range_m": [round(float(d.min()), 2), round(float(d.max()), 2)]}
        delivered = g[g["delivered"] == 1]
        delays = (delivered["receive_time_s"] - delivered["send_time_s"]).to_numpy(float)
        has_delay = np.isfinite(delays).any()
        if link == "acoustic":
            entry.update(fit_acoustic_delivery(d, y))
            if has_delay:
                ok = np.isfinite(delays)
                entry.update(fit_latency(delivered["range_m"].to_numpy(float)[ok], delays[ok], res["warnings"]))
            else:
                entry.update({"speed_of_sound_mps": 1500.0, "fixed_latency_s": 0.0, "speed_fitted": False})
                res["warnings"].append("acoustic: no receive times (delivered only), so latency isn't calibrated: "
                                       "range / 1500 m/s, no fixed delay")
        else:
            entry.update(fit_optical_delivery(d, y, res["warnings"], optical_water, optical_attenuation))
            ok = np.isfinite(delays)
            entry["latency_s"] = round(float(np.clip(np.median(delays[ok]), 0.0, 1.0)), 4) if ok.any() else 0.001
            if not ok.any():
                res["warnings"].append("optical: no receive times, latency left at the default 1 ms")
        if float(d.max()) > 0:
            entry["note"] = (f"fitted on messages sent at {entry['range_m'][0]:g}-{entry['range_m'][1]:g} m; "
                             f"beyond that range the model extrapolates")
        res[link] = entry
    if not any(l in res for l in LINKS):
        raise ValueError("no messages to fit")
    return res


# ------------------------------------------------------------------ using the result

def load_link_calibration(path: str) -> dict:
    """A link_calibration.yaml, checked: {'acoustic': {...}?, 'optical': {...}?, 'file': path}."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if raw.get("strathe_link_calibration") != FORMAT_VERSION:
        raise ValueError(f"{path} isn't a link calibration from `strathe fit-link` (format {FORMAT_VERSION})")
    out = {"file": os.path.abspath(path)}
    if raw.get("acoustic"):
        a = raw["acoustic"]
        plc = PacketLossConfig(**a["packet_loss"])
        plc.validate()
        out["acoustic"] = {"packet_loss_config": plc, "speed_of_sound_mps": float(a["speed_of_sound_mps"]),
                           "fixed_latency_s": float(a.get("fixed_latency_s", 0.0)), "range_m": a.get("range_m")}
    if raw.get("optical"):
        o = raw["optical"]
        out["optical"] = {"attenuation_per_m": float(o["attenuation_per_m"]), "rated_range_m": float(o["rated_range_m"]),
                          "curve_width": float(o["curve_width"]), "latency_s": float(o.get("latency_s", 0.001)),
                          "range_m": o.get("range_m")}
    if "acoustic" not in out and "optical" not in out:
        raise ValueError(f"{path} has neither an acoustic nor an optical fit")
    return out


def with_optical_calibration(link_config, cal: dict):
    """A comm_channel.LinkConfig with the calibration's optical fit (water, curve, latency) in place of the assumed
    values; unchanged if it has no optical part or the link isn't optical."""
    from dataclasses import replace
    if "optical" not in cal or not link_config.uses_optical:
        return link_config
    o = cal["optical"]
    opt = replace(link_config.optical, water="custom", attenuation_per_m=o["attenuation_per_m"],
                  rated_range_m=o["rated_range_m"], curve_width=o["curve_width"], latency_s=o["latency_s"],
                  calibrated_from=cal["file"])
    out = replace(link_config, optical=opt)
    out.validate()
    return out


# ------------------------------------------------------------------ report and plots

def write_report(res: dict, path: str, plots: list) -> None:
    L = ["# Link calibration\n", f"From {', '.join(res['source']['files'])}.\n"]
    if "acoustic" in res:
        a = res["acoustic"]
        pl = a["packet_loss"]
        L.append("## Acoustic\n")
        L.append(f"- **Messages:** {a['messages']}, {100 * a['delivered_fraction']:.0f}% delivered, sent at "
                 f"{a['range_m'][0]:g}-{a['range_m'][1]:g} m.")
        if a["model"] == "ramp":
            L.append(f"- **Delivery:** {100 * (1 - pl['near_field_prob']):.0f}% up to {pl['near_distance_m']:g} m, "
                     f"falling to {100 * (1 - pl['far_field_prob']):.0f}% at {pl['far_distance_m']:g} m and beyond.")
        else:
            L.append(f"- **Delivery:** {100 * (1 - pl['near_field_prob']):.0f}% at every range (a distance ramp didn't "
                     f"fit these messages better).")
        L.append(f"- **Latency:** {a['fixed_latency_s']:g} s fixed (the modem's processing and airtime) + range / "
                 f"{a['speed_of_sound_mps']:g} m/s" + ("" if a.get("speed_fitted") else " (speed held, not fitted)")
                 + (f"; 90% of messages within {a['residual_p90_s']:g} s of that." if a.get("residual_p90_s") is not None
                    else "."))
        L += _bins_md(a["bins"])
    if "optical" in res:
        o = res["optical"]
        L.append("## Optical\n")
        L.append(f"- **Messages:** {o['messages']}, {100 * o['delivered_fraction']:.0f}% delivered, sent at "
                 f"{o['range_m'][0]:g}-{o['range_m'][1]:g} m.")
        L.append(f"- **Delivery:** 50% at {o['range_at_50pct_m']:g} m when pointed (rated range {o['rated_range_m']:g} m, "
                 f"curve width {o['curve_width']:g}). The water's attenuation is held at {o['attenuation_held']}: "
                 f"messages can't measure it, and it only matters beyond the logged ranges.")
        L.append(f"- **Latency:** {1000 * o['latency_s']:g} ms.")
        L += _bins_md(o["bins"])
    if res["warnings"]:
        L += ["## Warnings\n"] + [f"- {w}" for w in res["warnings"]] + [""]
    L.append("## Using it\n")
    L.append("`strathe fleet --link-calibration link_calibration.yaml` (and `train-fleet`, the experiment YAML's "
             "`communication.link_calibration`, or the web page) uses these values instead of the assumed ones.\n")
    for p in plots:
        L.append(f"![{p}]({p})\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def _bins_md(bins) -> list:
    L = ["", "| Range | Messages | Delivered (95% interval) | Fitted |", "|---|---|---|---|"]
    for b in bins:
        L.append(f"| {b['range_m'][0]:g}-{b['range_m'][1]:g} m | {b['messages']} | {100 * b['delivered']:.0f}% "
                 f"({100 * b['ci95'][0]:.0f}-{100 * b['ci95'][1]:.0f}%) | {100 * b['fitted']:.0f}% |")
    return L + [""]


def plot(res: dict, logs: pd.DataFrame, out_dir: str) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    made = []
    for link in LINKS:
        if link not in res:
            continue
        e = res[link]
        g = logs[(logs["link"] == link) & (logs["pointed"] == 1)]
        d = g["range_m"].to_numpy(float)
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
        mids = [0.5 * (b["range_m"][0] + b["range_m"][1]) for b in e["bins"]]
        axs[0].errorbar(mids, [b["delivered"] for b in e["bins"]],
                        yerr=[[b["delivered"] - b["ci95"][0] for b in e["bins"]],
                              [b["ci95"][1] - b["delivered"] for b in e["bins"]]],
                        fmt="o", color="#222222", label="logged (95% interval)")
        xs = np.linspace(0, max(d.max() * 1.1, 1e-3), 200)
        if link == "acoustic":
            pl = e["packet_loss"]
            ys = 1 - ramp_drop(xs, pl["near_distance_m"], pl["far_distance_m"], pl["near_field_prob"],
                               pl["far_field_prob"])
        else:
            ys = optical_probability(xs, e["attenuation_per_m"], e["rated_range_m"], e["curve_width"])
        axs[0].plot(xs, ys, color="#1f77b4", label="fitted")
        axs[0].set_ylim(-0.02, 1.02)
        axs[0].set_xlabel("range when sent [m]")
        axs[0].set_ylabel("messages delivered")
        axs[0].set_title(f"{link.capitalize()}: delivery against range", fontsize=10)
        axs[0].grid(alpha=0.3)
        axs[0].legend(fontsize=8)
        dl = g[g["delivered"] == 1]
        delay = (dl["receive_time_s"] - dl["send_time_s"]).to_numpy(float)
        ok = np.isfinite(delay)
        axs[1].plot(dl["range_m"].to_numpy(float)[ok], delay[ok], ".", color="#222222", alpha=0.4, ms=3,
                    label="logged")
        if link == "acoustic":
            axs[1].plot(xs, e["fixed_latency_s"] + xs / e["speed_of_sound_mps"], color="#1f77b4", label="fitted")
        else:
            axs[1].axhline(e["latency_s"], color="#1f77b4", label="fitted (median)")
        axs[1].set_xlabel("range when sent [m]")
        axs[1].set_ylabel("latency [s]")
        axs[1].set_title(f"{link.capitalize()}: latency against range", fontsize=10)
        axs[1].grid(alpha=0.3)
        axs[1].legend(fontsize=8)
        fig.tight_layout()
        name = f"{link}_link.png"
        fig.savefig(os.path.join(out_dir, name), dpi=110)
        plt.close(fig)
        made.append(name)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", nargs="+", required=True, help="message log CSV(s) (templates/link_log_template.csv)")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--link", choices=LINKS, default="acoustic",
                    help="the link of rows with no link column (default acoustic)")
    ap.add_argument("--optical-water", choices=sorted(WATER_ATTENUATION_PER_M),
                    help="the water's clarity, held (messages can't measure it; default coastal)")
    ap.add_argument("--optical-attenuation", type=float, default=None, metavar="C",
                    help="or the water's measured attenuation coefficient, per metre (e.g. a transmissometer)")
    args = ap.parse_args(argv)
    try:
        logs = load_logs(args.log, args.link)
        if args.optical_water and args.optical_attenuation is not None:
            raise ValueError("give --optical-water or --optical-attenuation, not both")
        res = fit_logs(logs, args.optical_water, args.optical_attenuation)
    except (ValueError, OSError) as e:
        print(f"❌ {e}")
        return 2
    os.makedirs(args.out, exist_ok=True)
    plots = plot(res, logs, args.out)
    with open(os.path.join(args.out, "link_calibration.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(res, f, sort_keys=False)
    write_report(res, os.path.join(args.out, "link_calibration_report.md"), plots)
    for link in LINKS:
        if link not in res:
            continue
        e = res[link]
        if link == "acoustic":
            pl = e["packet_loss"]
            print(f"acoustic: {e['messages']} messages, {100 * e['delivered_fraction']:.0f}% delivered; delivery "
                  f"{100 * (1 - pl['near_field_prob']):.0f}% -> {100 * (1 - pl['far_field_prob']):.0f}% "
                  f"({e['model']}); latency {e['fixed_latency_s']:g} s + range / {e['speed_of_sound_mps']:g} m/s",
                  flush=True)
        else:
            print(f"optical: {e['messages']} messages, {100 * e['delivered_fraction']:.0f}% delivered; 50% at "
                  f"{e['range_at_50pct_m']:g} m, attenuation {e['attenuation_per_m']:g}/m, latency "
                  f"{1000 * e['latency_s']:g} ms", flush=True)
    for w in res["warnings"]:
        print(f"⚠️  {w}", flush=True)
    print(f"Wrote {os.path.join(args.out, 'link_calibration.yaml')}. Next: strathe fleet --link-calibration "
          f"{os.path.join(args.out, 'link_calibration.yaml')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
