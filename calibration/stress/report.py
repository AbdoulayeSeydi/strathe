"""
stress/report.py -- write STRESS_TEST_93_95.md (#93-#96) from stress/results_<feature>.csv: every row, grouped, with pass/fail
and the numbers, plus each feature's proof tables.

    python stress/report.py
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import REPO, load_rows  # noqa: E402

import inventory  # noqa: E402

TITLES = {"93": "One field day: hold out recordings, or a block of time",
          "94": "PX4, MOOS-IvP and LCM logs, read directly",
          "95": "Vehicle spec from basic measurements, or from any simulator model",
          "96": "Calibrate from what the vehicle measures: no DVL needed"}
# Each feature's row groups, by the row id's prefix (the part before the first "-"; #93's are numbered: F1, E8b).
GROUPS = {"93": {"F": "Fast checks on the real recordings", "M": "Deliberate faults (each must be caught)",
                 "E": "End to end on real data"},
          "94": {"R": "Every real recording through each format, against the CSV route",
                 "V": "One variable at a time", "P": "All pairs of variables", "K": "Field kinds",
                 "D": "Damaged logs", "DF": "Randomly damaged logs", "S": "Size: a one-hour log", "X": "Refusals",
                 "I": "Independent decoders (pyulog, plain text, lcm's own)",
                 "EP": "Entry points: ingest, calibrate --ingest-spec, the web card", "C": "strathe check",
                 "M": "Deliberate faults (each must be caught)", "E": "End to end on real data"},
          "95": {"G": "Inputs, each varied", "MS": "Model sources", "X": "Refusals", "W": "The web card",
                 "M": "Deliberate faults (each must be caught)", "FZ": "Random inputs (fuzz)", "E": "End to end on real data"},
          "96": {"R": "Real: the reference BlueROV2 tank, motion-capture velocity hidden",
                 "D": "Real: DFKI's BlueROV2, its own Bar30, IMU and attitude, motion-capture velocity hidden",
                 "U": "Real: URI's ALPHA RISE at sea, the raw DVL hidden",
                 "B": "Known truth, one variable at a time", "M": "Every sensor combination (what is calibrated, what is held)",
                 "S": "Real: SOLAQUA's USBL against its DVL (the smoother alone)",
                 "E": "End to end through strathe, no DVL", "W": "The web form", "FS": "fit-sim in each simulator"}}


def _group(feature: str, row: str) -> str:
    key = row.split("-")[0]
    return key[0] if feature == "93" else key


def _latest(rows: list) -> list:
    """The last result of each row id (a row re-run after a fix replaces the earlier one)."""
    out = {}
    for r in rows:
        out[r["row"]] = r
    return list(out.values())


def _short(numbers: str, n: int = 160) -> str:
    s = numbers.replace("|", "/")
    return s if len(s) <= n else s[:n] + "..."


def optimism_93(rows: list) -> list:
    """Within-session skill against the same calibration on sessions it never saw."""
    L = ["### How good a one-session check is (measured)\n",
         "Each calibration trained on one session, scored on its own held-out part and on sessions it never saw "
         "(mean skill over the 6 axes against predicting no motion; position drift after 30 s, median).\n",
         "| Calibrated on | Held out | Within the session: mean skill, drift 30 s | Other sessions: mean skill, drift 30 s "
         "| Within minus other (skill) |", "|---|---|---|---|---|"]
    for r in rows:
        try:
            n = json.loads(r["numbers"])
        except ValueError:
            continue
        if "other_sessions" not in n:
            continue
        w, o = n["within"], n["other_sessions"]
        L.append(f"| {r['value']} | {n.get('mode')} | {w['mean_skill']:+.3f}, {w['drift_m'].get('30s')} m | "
                 f"{o['mean_skill']:+.3f}, {o['drift_m'].get('30s')} m | {n['optimism_mean_skill']:+.3f} |")
    return L if len(L) > 4 else []


AXES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
DATASETS_96 = (("R", "The reference BlueROV2 tank (held-out days, against the hidden motion-capture velocity)"),
               ("D", "DFKI's BlueROV2 basin (held-out recordings, against the hidden motion-capture velocity)"),
               ("U", "URI's ALPHA RISE at sea (the held-out day, against the hidden raw DVL)"))


def real_96(rows: list) -> list:
    """Each real dataset: its velocity fit (V), each no-velocity configuration and default physics, per axis."""
    by = {r["row"]: r for r in rows}
    L = []
    for key, title in DATASETS_96:
        grp = [r for r in rows if r["row"].startswith(key + "-")]
        if not grp:
            continue
        L += [f"### {title}\n", "Held-out skill per axis (1 is perfect, 0 is no better than predicting no motion); "
              "60 s drift is against what the fit had (the position fixes, or depth), median.\n",
              "| Row | Sensors | " + " | ".join(AXES) + " | Mean | Drift 60 s (default) | Not calibrated | Bar |",
              "|---" * 12 + "|"]
        first = None
        for r in sorted(grp, key=lambda r: (r["row"][-1] != "V", r["row"])):
            n = json.loads(r["numbers"])
            first = first or n
            dr = (f"{n['drift_60s_m']:.2f} m ({n['drift_60s_default_m']:.2f})" if n.get("drift_60s_m") is not None
                  else "")
            blind = ", ".join(a for a, by_ in (n.get("axes_seen_by") or {}).items() if not by_) or "none"
            L.append(f"| {r['row']} | {r['what']} | " + " | ".join(f"{x:+.2f}" for x in n["skill"]) +
                     f" | **{n['mean_skill']:+.3f}** | {dr} | {blind} | {'✅' if r['passed'] == 'pass' else '❌'} |")
        d = first["default_skill"]
        L += [f"| default | the description's physics, nothing fitted | " + " | ".join(f"{x:+.2f}" for x in d) +
              f" | {sum(d) / 6:+.3f} | | | |", ""]
    return L


def synthetic_96(rows: list) -> list:
    """Known truth: each sensor set against the true parameters, scored on held-out data."""
    grp = [r for r in rows if r["row"].startswith(("B-set-", "B-ref-"))]
    if not grp:
        return []
    L = ["### Known truth: what each sensor set recovers\n",
         "A vehicle with known parameters replays the real logged thrust; its sensors are simulated. Held-out skill "
         "against its true velocity, and the median parameter error as a share of each parameter's range.\n",
         "| Row | Sensors | " + " | ".join(AXES) + " | Mean | Parameter error | Held | Bar |", "|---" * 12 + "|"]
    for r in grp:
        n = json.loads(r["numbers"])
        if "skill" not in n:
            continue
        held = len(n.get("held_unobserved") or [])
        L.append(f"| {r['row']} | {r['value']} | " + " | ".join(f"{x:+.2f}" for x in n["skill"]) +
                 f" | **{n['mean_skill']:+.3f}** | {n.get('median_param_error_of_range', float('nan')):.3f} | "
                 f"{held} | {'✅' if r['passed'] == 'pass' else '❌'} |")
    return L + [""]


PROOF = {"93": optimism_93, "96": lambda rows: real_96(rows) + synthetic_96(rows)}


def render() -> str:
    L = ["# Stress test: #93-#96\n", f"Written by `calibration/stress/report.py` on {time.strftime('%Y-%m-%d %H:%M')}. "
         "Every variable each feature has is listed by `stress/inventory.py` from the code, and "
         "`tests/test_stress_coverage.py` fails if one has no row here.\n"]
    for f in sorted(TITLES):
        rows = _latest(load_rows(f))
        if not rows:
            continue
        passed = sum(r["passed"] == "pass" for r in rows)
        inv = inventory.inventory([f]) if f in inventory.FEATURES else []
        L += [f"## #{f}. {TITLES[f]}\n",
              f"**{passed} of {len(rows)} rows pass** ({len(inv)} variables found in the code, each with a row).\n"]
        L += PROOF.get(f, lambda r: [])(rows) + [""]
        groups = dict(GROUPS.get(f, {}))
        groups.update({k: f"Other rows ({k})" for k in sorted({_group(f, r["row"]) for r in rows}) if k not in groups})
        for g, title in groups.items():   # every row is listed: unknown prefixes get their own group
            grp = [r for r in rows if _group(f, r["row"]) == g]
            if not grp:
                continue
            L += [f"### {title}\n", "| Row | Varied | What | Data | Result | Numbers | Time |",
                  "|---|---|---|---|---|---|---|"]
            for r in grp:
                L.append(f"| {r['row']} | {r['variable']} = {r['value']} | {r['what']} | {os.path.basename(r['data'])} "
                         f"| {'✅' if r['passed'] == 'pass' else '❌'} | {_short(r['numbers'])} | "
                         f"{float(r['seconds']) / 60:.0f} min |")
            L.append("")
    return "\n".join(L) + "\n"


def main() -> int:
    out = os.path.join(REPO, "STRESS_TEST_93_95.md")
    open(out, "w", encoding="utf-8").write(render())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
