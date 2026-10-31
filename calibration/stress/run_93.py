"""
stress/run_93.py -- stress test of one-session calibration (PROJECT_STATUS.md #93) on the real datasets.

    python stress/run_93.py fast                 # every split check, on the real recordings (minutes)
    python stress/run_93.py mutants              # each deliberate fault must be caught by the fast checks
    python stress/run_93.py e2e [ROW ...]        # the end-to-end runs (calibrations: hours); default all
    python stress/run_93.py list                 # the rows

Results go to stress/results_93.csv; stress/report.py writes them up. Data (made by stress/data_93.sh):
    ~/strathe_session_data/bluerov_<DD-MM-YYYY>   one day of oceansystemslab's BlueROV2 Heavy (17/9/8/6 recordings)
    ~/strathe_session_data/dfki_one              DFKI's BlueROV2, its 22 recordings as the one session they were
    ~/strathe_session_data/uri_bay, uri_pond     URI's ALPHA RISE, one recording each (time-block mode)
Independent checks: the fast rows check the split's promises from the files themselves (row ranges re-read with
pandas, file sets compared), not with data_split's own assert_disjoint, so a fault in data_split shows up.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import yaml

HERE =os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, CAL)
sys.path.insert(0, HERE)
from common import all_pairs, record, strathe  # noqa: E402

import data_split as ds  # noqa: E402

F = "93"
DATA = os.path.expanduser("~/strathe_session_data")
OUT = os.path.join(DATA, "e2e")
BLUEROV_ALL = os.path.join(os.path.dirname(os.path.dirname(CAL)), "marine_sim_mvp", "data", "raw",
                           "Bluerov2_trajectory_dataset")
if not os.path.isdir(BLUEROV_ALL):
    BLUEROV_ALL = os.path.expanduser("~/marine_sim_mvp/data/raw/Bluerov2_trajectory_dataset")
DAYS = ("12-01-2024", "14-02-2024", "16-02-2024", "19-02-2024")
URI_SPEC = os.path.expanduser("~/strathe_uri_lab_v2/data/alpha_rise_spec.yaml")
DFKI_DATED = os.path.join(DATA, "dfki_dated")   # DFKI with the worked example's two placeholder dates


def files(folder: str) -> list:
    return ds.discover_csv_files(folder)[0]


def day(d: str) -> str:
    return os.path.join(DATA, f"bluerov_{d}")


# --- independent checks ---------------------------------------------------------------------------------------------
def _rows_of(ref: str) -> tuple:
    """(recording, set of row indices) a reference covers, re-read from the file."""
    p, a, b = ds.parse_ref(ref)
    n = len(pd.read_csv(p, usecols=[0]))
    return os.path.abspath(p), set(range(n) if a is None else range(a, b))


def independent(split) -> dict:
    """Overlap between training and held-out rows, from the files themselves."""
    train, held = {}, {}
    for r in split.train_files:
        p, s = _rows_of(r)
        train.setdefault(p, set()).update(s)
    for r in split.heldout_files:
        p, s = _rows_of(r)
        held.setdefault(p, set()).update(s)
    overlap = sum(len(train.get(p, set()) & s) for p, s in held.items())
    gaps = {}
    for p in set(train) & set(held):
        gaps[os.path.basename(p)] = (min(held[p]) - max(train[p]) - 1) if train[p] and held[p] else None
    return {"overlap_rows": overlap, "gap_rows": gaps, "train": len(split.train_files), "held": len(split.heldout_files)}


def _old_split():
    """data_split.py as it was before #93 (feat/docker), for the regression rows."""
    src = subprocess.run(["git", "-C", CAL, "show", "feat/docker:calibration/data_split.py"], capture_output=True,
                         text=True, check=True).stdout
    path = os.path.join(tempfile.mkdtemp(), "data_split_before93.py")
    open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("data_split_before93", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# --- fast rows ------------------------------------------------------------------------------------------------------
def f1_auto(sets: dict) -> None:
    """Auto split on every real set: mode, count, latest by run number, no overlap, warning only within a session."""
    for name, fs in sets.items():
        t0 = time.time()
        sp = ds.split_by_date(fs)
        ind = independent(sp)
        dates = {ds.extract_date(f) for f in fs}
        want_mode = "dates" if len(dates) > 1 else ("recordings" if len(fs) > 1 else "time block")
        ok = sp.mode == want_mode and ind["overlap_rows"] == 0 and (sp.warning is not None) == (len(dates) == 1)
        if want_mode == "recordings":
            order = sorted((ds.recording_name(f) for f in fs), key=ds._natural)
            n = max(1, min(len(order) - 1, round(len(order) * 0.3)))
            ok = ok and sp.heldout_groups == order[-n:]
        again = ds.split_by_date(fs)
        ok = ok and again.train_files == sp.train_files and again.heldout_files == sp.heldout_files
        record(F, f"F1-{name}", "mode", want_mode, f"auto split of {name} ({len(fs)} recordings, {len(dates)} date(s))",
               name, {**ind, "mode": sp.mode, "held": sp.heldout_groups[:8]},
               f"mode {want_mode}, latest 30% held out, no overlap, deterministic, warning iff one session", ok,
               time.time() - t0)


def f2_named() -> None:
    fs = files(day("12-01-2024"))
    names = sorted((ds.recording_name(f) for f in fs), key=ds._natural)
    for label, held in (("one", names[2:3]), ("two", [names[2], names[8]]), ("all-but-one", names[1:])):
        sp = ds.split_by_date(fs, heldout_dates=held)
        ind = independent(sp)
        ok = sp.mode == "recordings" and sorted(sp.heldout_groups) == sorted(held) and ind["overlap_rows"] == 0 \
            and len(sp.train_files) == len(fs) - len(held)
        record(F, f"F2-{label}", "kind", "recording", f"--heldout-recordings, {label} named", "bluerov 12-01",
               ind, "exactly the named recordings held out, the rest trained, no overlap", ok)
    try:
        ds.split_by_date(fs, heldout_dates=names)
        ok, msg = False, "no error"
    except ValueError as e:
        ok, msg = "at least one" in str(e), str(e)
    record(F, "F2-all", "kind", "recording", "every recording named: refused", "bluerov 12-01", {"message": msg},
           "a plain refusal", ok)


def f3_time_block() -> None:
    """The last part of one recording, for each fraction: row counts, the gap, and the rows load_run reads."""
    from objective import DatasetConfig, load_run
    cases = [("uri_bay", files(os.path.join(DATA, "uri_bay"))[0]), ("uri_pond", files(os.path.join(DATA, "uri_pond"))[0]),
             ("bluerov 19-02 run1", files(day("19-02-2024"))[0])]
    for name, path in cases:
        n, rate = ds._rows(path), ds._rate_hz(path)
        raw = pd.read_csv(path)
        for frac in (0.2, 0.3, 0.5):
            t0 = time.time()
            sp = ds.split_by_date([path], heldout_dates=[f"{ds.recording_name(path)}::last{round(frac * 100)}%"])
            ind = independent(sp)
            held_n = int(np.floor(n * frac))
            gap = int(round(ds.TIME_BLOCK_GAP_S * rate))
            _, a, b = ds.parse_ref(sp.heldout_files[0])
            _, ta, tb = ds.parse_ref(sp.train_files[0])
            run_h = load_run(sp.heldout_files[0], DatasetConfig(rate))
            same = np.allclose(run_h.forces_torques, raw[["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]].to_numpy()[a:b])
            ok = (b - a == held_n and tb - ta == n - held_n - gap and a - tb == gap and ind["overlap_rows"] == 0
                  and same and sp.mode == "time block" and sp.warning)
            record(F, f"F3-{name}-{frac}", "kind", "last part", f"last {frac:.0%} of one recording",
                   name, {"rows": n, "held_rows": b - a, "train_rows": tb - ta, "gap_rows": a - tb, **ind,
                          "rows_read_match": bool(same)},
                   f"held {held_n} rows, gap {gap}, train the rest, load_run reads exactly those rows", ok,
                   time.time() - t0)


def f4_gap() -> None:
    path = files(os.path.join(DATA, "uri_pond"))[0]
    raw = pd.read_csv(path)
    for gap_s in (0.0, 10.0, 20.0):
        sp = ds._split_time_block(path, 0.3, gap_s)
        _, a, b = ds.parse_ref(sp.heldout_files[0])
        _, ta, tb = ds.parse_ref(sp.train_files[0])
        jump = float(np.abs(raw[["Bu", "Bv", "Bw"]].to_numpy()[a] - raw[["Bu", "Bv", "Bw"]].to_numpy()[tb - 1]).max())
        ok = a - tb == int(round(gap_s * ds._rate_hz(path))) and independent(sp)["overlap_rows"] == 0
        record(F, f"F4-gap{gap_s:g}", "const", "TIME_BLOCK_GAP_S", f"gap of {gap_s:g} s between the parts", "uri_pond",
               {"gap_rows": a - tb, "velocity_jump_across_gap_mps": round(jump, 4)},
               "the gap in rows is gap x rate; at 0 s the parts touch (the leak a gap removes)", ok)


def f5_refusals(tmp: str) -> None:
    short = os.path.join(tmp, "my_rov-05-01-2030-run1.csv")
    pd.read_csv(files(day("19-02-2024"))[0]).iloc[:1000].to_csv(short, index=False)   # 100 s at 10 Hz
    fs = files(day("12-01-2024"))
    multi = files(os.path.join(BLUEROV_ALL, "dataset"))
    cases = [
        ("too short", lambda: ds.split_by_date([short]), "too short"),
        ("unknown recording", lambda: ds.split_by_date(fs, heldout_dates=["odom-12-01-2024-run99"]), "Unknown"),
        ("date and recording", lambda: ds.split_by_date(multi, heldout_dates=["16-02-2024", "odom-12-01-2024-run1"]),
         "not both"),
        ("the only date", lambda: ds.split_by_date(fs, heldout_dates=["12-01-2024"]), "no other day"),
        ("last 0%", lambda: ds.split_by_date(fs[:1], heldout_dates=[f"{ds.recording_name(fs[0])}::last0%"]),
         "between 0 and 1"),
        ("last 100%", lambda: ds.split_by_date(fs[:1], heldout_dates=[f"{ds.recording_name(fs[0])}::last100%"]),
         "between 0 and 1"),
        ("last part of two", lambda: ds.split_by_date(fs, heldout_dates=[f"{ds.recording_name(f)}::last30%"
                                                                          for f in fs[:2]]), "one recording only"),
        ("whole and last part", lambda: ds.split_by_date(fs, heldout_dates=[ds.recording_name(fs[1]),
                                                                             f"{ds.recording_name(fs[0])}::last30%"]),
         "the last part of one, not both"),
        ("percent not a number", lambda: ds.split_by_date(fs[:1], heldout_dates=[f"{ds.recording_name(fs[0])}::lastx%"]),
         "percentage"),
        ("unknown date", lambda: ds.split_by_date(multi, heldout_dates=["01-01-1999"]), "Unknown heldout"),
        ("fraction outside (0, 1)", lambda: ds.split_by_date(fs, heldout_fraction=1.0), "strictly between"),
    ]
    for label, fn, want in cases:
        try:
            fn()
            ok, msg = False, "no error"
        except ValueError as e:
            ok, msg = want in str(e), str(e)
        record(F, f"F5-{label.replace(' ', '_')}", "refusal", label, f"refused: {label}", "bluerov", {"message": msg[:300]},
               f"a ValueError mentioning '{want}'", ok)


PAIRS_93 = {"dataset": ["6 recordings", "2 recordings", "3 recordings", "5 recordings", "1 recording", "2 days"],
            "held_out": ["automatic", "one name", "two names", "last part", "a date"],
            "fraction": [0.2, 0.3, 0.5]}


def _pair_files(dataset: str) -> list:
    d19 = files(day("19-02-2024"))
    return {"6 recordings": d19, "2 recordings": files(day("12-01-2024") + "_2recs"),
            "3 recordings": files(day("12-01-2024") + "_3recs"), "5 recordings": files(day("12-01-2024") + "_5recs"),
            "1 recording": d19[:1], "2 days": sorted(d19 + files(day("16-02-2024")))}[dataset]


def p_pairwise() -> None:
    """Every pair of dataset, held-out kind and fraction: the split each gives is worked out here independently
    (row counts read with pandas at 10 Hz) and compared; refusals only where the rules say so."""
    for i, c in enumerate(all_pairs(PAIRS_93, 93)):
        fs = _pair_files(c["dataset"])
        order = sorted((ds.recording_name(p) for p in fs), key=ds._natural)
        dates = sorted({ds.extract_date(p) for p in fs})
        f = c["fraction"]
        held, expect = None, None   # expect: ("refused", text) or (mode, held-out names)
        if c["held_out"] == "automatic":
            if len(dates) > 1:
                expect = ("dates", [max(dates, key=lambda d: ds._parse_date(d))])
            elif len(fs) == 1:
                held, expect = None, ("time block", None)
            else:
                n = max(1, min(len(order) - 1, round(len(order) * f)))
                expect = ("recordings", order[-n:])
        elif c["held_out"] in ("one name", "two names"):
            k = 1 if c["held_out"] == "one name" else 2
            held = order[:k]
            expect = ("refused", "Every recording is held out") if k >= len(order) else ("recordings", held)
        elif c["held_out"] == "last part":
            held = [f"{order[0]}::last{round(f * 100)}%"]
            rows = len(pd.read_csv([p for p in fs if ds.recording_name(p) == order[0]][0]))
            short = int(rows * f) < 600 or rows - int(rows * f) - 100 < 600   # 60 s each side, a 10 s gap, at 10 Hz
            expect = ("refused", "too short") if short else ("time block", None)
        else:
            held = [dates[0]]
            expect = ("refused", "no other day") if len(dates) == 1 else ("dates", [dates[0]])
        problems = []
        try:
            sp = ds.split_by_date(fs, held, f)
            ds.assert_disjoint(sp)
            if expect[0] == "refused":
                problems.append(f"split ({sp.mode}) where a refusal was expected: {expect[1]}")
            elif sp.mode != expect[0]:
                problems.append(f"mode {sp.mode}, expected {expect[0]}")
            elif expect[1] is not None and sorted(sp.heldout_groups) != sorted(expect[1]):
                problems.append(f"held out {sp.heldout_groups}, expected {expect[1]}")
            if sp.mode == "time block":   # the part's rows: last floor(n*f), a 10 s gap, the rest trains
                path, a, b = ds.parse_ref(sp.heldout_files[0])
                n = len(pd.read_csv(path))
                tr = [ds.parse_ref(r) for r in sp.train_files if ds.parse_ref(r)[0] == path][0]
                if (a, b) != (n - int(n * f), n) or tr[1:] != (0, n - int(n * f) - 100):
                    problems.append(f"rows held {a}-{b}, trained {tr[1]}-{tr[2]} of {n}")
                if len(sp.train_files) != len(fs):
                    problems.append("the other recordings don't train")
            if (sp.warning is not None) != (len(dates) == 1):
                problems.append("the one-session warning is missing, or shown for two days")
        except ValueError as e:
            if expect[0] != "refused" or expect[1] not in str(e):
                problems.append(f"refused: {str(e)[:160]}")
        record(F, f"P-{i}", "pairwise", json.dumps(c), f"all-pairs case {i + 1}: {c['dataset']}, {c['held_out']}, "
               f"{f:g}", c["dataset"], {"held": held, "expected": expect, "problems": problems},
               "the split worked out independently, or the refusal the rules give", not problems)


def f6_regression() -> None:
    """Two or more dates: identical to data_split before #93."""
    old = _old_split()
    multi = files(os.path.join(BLUEROV_ALL, "dataset"))
    cases = [("bluerov default", multi, None), ("bluerov 16-02 19-02", multi, ["16-02-2024", "19-02-2024"])]
    cases += [(f"bluerov {d}", multi, [d]) for d in DAYS]
    if os.path.isdir(DFKI_DATED):
        cases.append(("dfki placeholder dates", files(DFKI_DATED), None))
    for label, fs, held in cases:
        a, b = old.split_by_date(fs, heldout_dates=held), ds.split_by_date(fs, heldout_dates=held)
        same = all(getattr(a, k) == getattr(b, k) for k in ("train_files", "heldout_files", "train_groups",
                                                             "heldout_groups", "n_train", "n_heldout"))
        record(F, f"F6-{label.replace(' ', '_')}", "mode", "dates", f"regression: {label}", label,
               {"held": b.heldout_groups}, "identical to before #93", same and b.mode == "dates" and b.warning is None)


def f7_reproduce(sets: dict) -> None:
    """What a calibration stores in metrics.json splits the same way again (report, validate, --finish)."""
    for name, fs in sets.items():
        sp = ds.split_by_date(fs)
        again = ds.split_by_date(fs, heldout_dates=sp.heldout_groups)
        ok = again.train_files == sp.train_files and again.heldout_files == sp.heldout_files and again.mode == sp.mode
        record(F, f"F7-{name}", "mode", sp.mode, "re-split from the stored held-out groups", name,
               {"stored": sp.heldout_groups[:6]}, "the same files and rows", ok)


def f8_web() -> None:
    import strathe_web as web
    good = ["16-02-2024", "odom-12-01-2024-run7", "alpha_rise-05-12-2025-run1::last30%"]
    bad = ["yesterday", "16/02/2024", "run7", "x::last30%", "odom-12-01-2024-run1::last300%"]
    ok_good = all(web._heldout_args(g) == ["--heldout-dates", g] for g in good)
    errs = []
    for b in bad:
        try:
            web._heldout_args(b)
            errs.append(f"{b}: accepted")
        except ValueError:
            pass
    record(F, "F8-format", "web", "all", "web held-out field: valid entries pass, malformed ones are refused", "-",
           {"problems": errs}, "dates, recordings and last parts pass; the rest refused", ok_good and not errs)
    try:
        web._heldout_args("odom-01-01-1999-run1", day("12-01-2024"))
        ok, msg = False, "accepted"
    except ValueError as e:
        ok, msg = "isn't a date or recording" in str(e) and "odom-12-01-2024-run1" in str(e), str(e)
    record(F, "F8-folder", "web", "all", "a name not in the logs folder is refused, listing the real ones",
           "bluerov 12-01", {"message": msg[:200]}, "refused, with the folder's dates and recordings", ok)
    m = web.JobManager(tempfile.mkdtemp(), launcher=lambda cmd, log: None)
    h = web.StratheHandler.__new__(web.StratheHandler)
    h.manager = m
    for field, path, form in (
            ("cal-heldout", "/jobs/calibrate", {"data": day("12-01-2024"), "name": "x"}),
            ("sf-heldout", "/jobs/fit_stonefish", None), ("simf-heldout", "/jobs/fit_sim", None)):
        if form is None:   # the Stonefish and simulator cards: their argv builders
            fn = web.stonefish_cli_args if field == "sf-heldout" else (lambda f: web.sim_cli_args(f, "fit"))
            scn = os.path.join(tempfile.mkdtemp(), "w.scn")
            open(scn, "w").write("<scenario/>")
            base = ({"scenario": scn, "robot": "r", "part": "Hull", "reference_point": "0 0 0"}
                    if field == "sf-heldout" else {"sim": "mujoco", "model": _mjcf(), "reference_point": "0 0 0"})
            argv = fn({**base, "data": day("12-01-2024"), "heldout": "odom-12-01-2024-run3"})
        else:
            job = h._start(path, {**form, "heldout": "odom-12-01-2024-run3"})
            argv = job.argv
        ok = "odom-12-01-2024-run3" in argv[argv.index("--heldout-dates") + 1:]
        record(F, f"F8-{field}", "web", field, f"web field {field} passes a recording name to the command", "-",
               {"argv_tail": argv[-6:]}, "--heldout-dates odom-12-01-2024-run3 in the command", ok)


def _mjcf() -> str:
    p = glob.glob(os.path.join(CAL, "simulators", "mujoco", "example", "*.xml"))
    return p[0]


def f9_flags() -> None:
    """Every parser with --heldout-recordings stores it where --heldout-dates goes (the inventory's flag rows)."""
    import ast
    import inventory
    for rel, var in inventory.heldout_parsers(inventory.reachable()):
        t = ast.parse(open(os.path.join(CAL, rel), encoding="utf-8").read())
        calls = [n for n in ast.walk(t) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_argument"
                 and getattr(n.func.value, "id", None) == var
                 and any(isinstance(a, ast.Constant) and a.value == "--heldout-recordings" for a in n.args)]
        ok = all(any(isinstance(a, ast.Constant) and a.value == "--heldout-dates" for a in c.args)
                 and any(k.arg == "dest" and getattr(k.value, "value", None) == "heldout_dates" for k in c.keywords)
                 for c in calls)
        record(F, f"F9-{rel}:{var}", "flag", f"{rel}:{var}", "--heldout-recordings is --heldout-dates' alias", rel,
               {"calls": len(calls)}, "same option, dest heldout_dates", ok and bool(calls))
    import strathe_cli as cli
    p = cli.build_parser()
    for sub, extra in (("calibrate", ["--data", "x"]), ("report", ["--result", "x"]),
                       ("fit-stonefish", ["--scenario", "s", "--robot", "r", "--part", "p", "--data", "d",
                                          "--reference-point", "0", "0", "0"]),
                       ("export-stonefish", ["--calibration", "c", "--scenario", "s", "--robot", "r", "--part", "p",
                                             "--reference-point", "0", "0", "0"])):
        a = p.parse_args([sub, *extra, "--heldout-recordings", "odom-12-01-2024-run3", "x::last20%"])
        record(F, f"F9-parse-{sub}", "flag", sub, f"strathe {sub} --heldout-recordings parses", "-",
               {"heldout_dates": a.heldout_dates}, "both names in heldout_dates",
               a.heldout_dates == ["odom-12-01-2024-run3", "x::last20%"])


def f10_density() -> None:
    from stonefish_export import training_density
    cases = [(["27-03-2026"], 1000), (["alpha_rise-27-03-2026-run1::first part"], 1000),
             (["alpha_rise-27-03-2026-run1", "alpha_rise-05-12-2025-run1"], 1025)]
    res = [training_density(g, {"27-03-2026": 1000, "05-12-2025": 1023}, 1025) for g, _ in cases]
    record(F, "F10-density", "consumer", "stonefish_export.py", "--density-by-date with recording groups", "-",
           {"got": res}, "1000, 1000, the model's when mixed", res == [w for _, w in cases])


def fast() -> int:
    tmp = tempfile.mkdtemp(prefix="stress93_")
    multi = files(os.path.join(BLUEROV_ALL, "dataset"))
    d12 = files(day("12-01-2024"))
    sets = {f"bluerov_{d}": files(day(d)) for d in DAYS}
    sets.update({"dfki_one": files(os.path.join(DATA, "dfki_one")), "uri_bay": files(os.path.join(DATA, "uri_bay")),
                 "uri_pond": files(os.path.join(DATA, "uri_pond")), "bluerov_all_days": multi,
                 "bluerov_12-01_2recs": d12[:2], "bluerov_12-01_3recs": d12[:3], "bluerov_12-01_5recs": d12[:5]})
    f1_auto(sets)
    f2_named()
    f3_time_block()
    f4_gap()
    f5_refusals(tmp)
    f6_regression()
    f7_reproduce(sets)
    f8_web()
    f9_flags()
    f10_density()
    p_pairwise()
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


# --- mutants --------------------------------------------------------------------------------------------------------
def mutants() -> int:
    """Each fault, applied in-process, must make at least one fast row fail. The rows are written to a scratch file,
    not the results."""
    import common
    real_record, real = common.record, {k: getattr(ds, k) for k in ("_split_time_block", "_split_recordings",
                                                                     "assert_disjoint")}
    import objective
    real_read = objective.read_run_csv

    def off_by_one(path, fraction, gap_s=ds.TIME_BLOCK_GAP_S):   # held-out part starts one row inside training
        sp = real["_split_time_block"](path, fraction, gap_s)
        p, a, b = ds.parse_ref(sp.heldout_files[0])
        _, ta, tb = ds.parse_ref(sp.train_files[0])
        sp.heldout_files = [ds.part_ref(p, tb - 1, b)]
        return sp

    def leak(filepaths, heldout, frac):   # one held-out recording also trained on
        sp = real["_split_recordings"](filepaths, heldout, frac)
        sp.train_files = sorted(sp.train_files + sp.heldout_files[:1])
        return sp

    def no_slice(filepath):   # rows ignored: the whole recording read for a part
        return pd.read_csv(ds.parse_ref(filepath)[0])

    def wrong_count(filepaths, heldout, frac):   # one recording too many held out
        if heldout is None:
            order = sorted((ds.recording_name(p) for p in filepaths), key=ds._natural)
            n = max(1, min(len(order) - 1, round(len(order) * frac)))
            heldout = order[-min(len(order) - 1, n + 1):]
        return real["_split_recordings"](filepaths, heldout, frac)

    faults = {"time block off by one": ("_split_time_block", off_by_one), "held-out recording leaks into training":
              ("_split_recordings", leak), "rows of a part ignored": ("read", no_slice),
              "held-out count wrong": ("_split_recordings", wrong_count)}
    for label, (target, fn) in faults.items():
        fails = []
        common.record = lambda *a, **k: fails.append(a[1]) if not a[8] else None  # noqa: E731
        globals()["record"] = common.record
        ds.assert_disjoint = lambda split: None   # the mutant gets past data_split's own guard
        if target == "read":
            objective.read_run_csv = fn
        else:
            setattr(ds, target, fn)
        try:
            fast()
        except Exception as e:   # a crash also counts as caught
            fails.append(f"crash: {type(e).__name__}: {e}")
        finally:
            for k, v in real.items():
                setattr(ds, k, v)
            objective.read_run_csv = real_read
            common.record = real_record
            globals()["record"] = real_record
        record(F, f"M-{label.replace(' ', '_')}", "mutant", label, f"deliberate fault: {label}", "fast rows",
               {"rows_failed": fails[:8], "n_failed": len(fails)}, "at least one fast row fails", bool(fails))
    return 0


# --- end to end -----------------------------------------------------------------------------------------------------
def score(result: str, data: str, heldout, spec: str, whole: bool = False) -> dict:
    """Per-axis skill against predicting no motion, and position drift, of a calibration on any recordings.
    whole=True: every recording in `data`, whole, as a calibration scores a session it held out; when the calibration
    estimated currents, theirs are estimated with it too (strathe estimate-current), as calibrate does for held-out
    recordings."""
    import yaml
    from dataclasses import replace
    from dynamics_model import CalibrationParameters
    from objective import load_run, read_currents_file, stable_rmse_by_dof, zero_baseline_rmse
    from optimizer import load_vehicle_spec
    from validation import dataset_config_from_metrics, trajectory_drift
    metrics = json.load(open(os.path.join(result, "metrics.json"), encoding="utf-8"))
    params = CalibrationParameters(**yaml.safe_load(open(os.path.join(result, "calibrated_parameters.yaml")))["parameters"])
    vehicle = load_vehicle_spec(spec).vehicle
    dc = dataset_config_from_metrics(metrics)
    if whole:
        targets = files(data)
        current = None
        if (metrics.get("objective") or {}).get("current_by_file"):   # the calibration estimated currents
            est = os.path.join(result, f"other_session_currents_{os.path.basename(os.path.normpath(data))}.csv")
            if not os.path.exists(est):
                strathe(["estimate-current", "--data", data, "--params", os.path.join(result, "calibrated_parameters.yaml"),
                         "--spec", spec, "--out", est], est + ".log")
            current = read_currents_file(est)
        dc = replace(dc, current_by_file=current, current_world_mps=None)
        held = [ds.recording_name(f) for f in targets]
    else:
        sp = ds.split_by_date(files(data), heldout_dates=heldout)
        targets, held = sp.heldout_files, sp.heldout_groups
    runs = [load_run(f, dc) for f in targets]
    c, z = stable_rmse_by_dof(params, runs, vehicle)[0], zero_baseline_rmse(runs)
    skill = (1 - np.asarray(c) / np.asarray(z)).round(3).tolist()
    drift = trajectory_drift({"calibrated": params}, runs, vehicle)
    return {"skill": skill, "mean_skill": round(float(np.mean(skill)), 3), "heldout": held[:6],
            "drift_m": {f"{h}s": round(float(drift["calibrated"][i]["median"]), 2) if drift["calibrated"][i]
                        else None for i, h in enumerate(drift["horizons_s"])}}


def _e2e_calibrate(row: str, variable: str, value, what: str, data: str, args: list, spec: str = None,
                   cross=None, expect_mode=None) -> None:
    out = os.path.join(OUT, row)
    shutil.rmtree(out, ignore_errors=True)
    argv = ["calibrate", "--data", data, "--name", row, "--out", out, *(["--spec", spec] if spec else []), *args]
    rc, secs = strathe(argv, os.path.join(OUT, f"{row}.log"))
    nums = {"exit": rc}
    ok = rc == 0 and os.path.exists(os.path.join(out, "calibration_report.md"))
    if ok:
        m = json.load(open(os.path.join(out, "metrics.json"), encoding="utf-8"))
        rep = open(os.path.join(out, "calibration_report.md"), encoding="utf-8").read()
        mode = m["dataset"].get("split_mode", "dates")
        nums.update({"mode": mode, "held": m["dataset"]["heldout_dates"][:6],
                     "converged": m.get("optimizer", {}).get("converged"),
                     "within": score(out, data, m["dataset"]["heldout_dates"], spec or _default_spec())})
        ok = ok and (expect_mode is None or mode == expect_mode)
        r = m.get("rmse") or {}
        nums["train_rmse_start_vs_fit"] = [r.get("default_train"), r.get("calibrated_train")]
        # the fit has to have worked: better than the starting description on its own training recordings (#93)
        ok = ok and r.get("calibrated_train") is not None and r["calibrated_train"] < 0.9 * r["default_train"]
        if mode != "dates":
            ok = ok and "within one session" in rep and os.path.exists(os.path.join(out, "validation", "validation.json"))
        if cross:   # the same calibration on sessions it never saw: the optimism of a one-session check, measured
            nums["other_sessions"] = (score(out, cross[0], None, spec or _default_spec(), whole=True) if cross[1] is None
                                      else score(out, cross[0], cross[1], spec or _default_spec()))
            nums["optimism_mean_skill"] = round(nums["within"]["mean_skill"] - nums["other_sessions"]["mean_skill"], 3)
    record(F, row, variable, value, what, data, nums,
           f"completes; split by {expect_mode or 'as given'}; the fit beats the starting description by >10% on its "
           f"training recordings; report states the within-session warning", ok, secs)


def _default_spec() -> str:
    return os.path.join(CAL, "vehicle_spec.yaml")


def _e2e_rows() -> dict:
    """row id -> callable. Each is one end-to-end run on real data."""
    multi = os.path.join(BLUEROV_ALL, "dataset")
    rows = {}
    for i, d in enumerate(DAYS, 1):
        others = [x for x in DAYS if x != d]
        rows[f"E{i}-bluerov_{d}"] = (lambda d=d, i=i, others=others: _e2e_calibrate(
            f"E{i}-bluerov_{d}", "dataset", f"bluerov {d}", f"calibrate on one day ({d}), recordings held out",
            day(d), [], cross=(multi, others), expect_mode="recordings"))
    rows["E5-dfki_one"] = lambda: _e2e_calibrate("E5-dfki_one", "dataset", "dfki one session",
                                                 "calibrate DFKI's single session", os.path.join(DATA, "dfki_one"), [],
                                                 expect_mode="recordings")
    for i, (m, other) in enumerate((("bay", "pond"), ("pond", "bay")), 6):
        rows[f"E{i}-uri_{m}"] = (lambda m=m, other=other, i=i: _e2e_calibrate(
            f"E{i}-uri_{m}", "dataset", f"uri {m}", f"calibrate URI's {m} mission alone (time block)",
            os.path.join(DATA, f"uri_{m}"), ["--estimate-current", "--workers", "2"], URI_SPEC,
            cross=(os.path.join(DATA, f"uri_{other}"), None), expect_mode="time block"))
    rows["E8-named"] = lambda: _e2e_calibrate("E8-named", "kind", "recording", "--heldout-recordings, two named",
                                              day("12-01-2024"), ["--heldout-recordings", "odom-12-01-2024-run3",
                                                                  "odom-12-01-2024-run9"], expect_mode="recordings")
    rows["E8b-retry"] = lambda: _e2e_retry()
    for i, pct in ((9, 20), (10, 50)):
        rows[f"E{i}-last{pct}"] = (lambda pct=pct, i=i: _e2e_calibrate(
            f"E{i}-last{pct}", "kind", "last part", f"--heldout-recordings ...::last{pct}%",
            os.path.join(DATA, "uri_pond"), ["--estimate-current", "--workers", "2", "--heldout-recordings",
                                              f"alpha_rise-27-03-2026-run1::last{pct}%"], URI_SPEC,
            expect_mode="time block"))
    for i, n in ((11, 2), (12, 3), (13, 5)):
        rows[f"E{i}-{n}recs"] = (lambda n=n, i=i: _subset_and_calibrate(f"E{i}-{n}recs", n))
    rows["E14-restarts4"] = lambda: _e2e_calibrate("E14-restarts4", "fit", "--restarts 4", "one day, 4 starting points",
                                                   day("19-02-2024"), ["--restarts", "4"], expect_mode="recordings")
    rows["E15-no_current"] = lambda: _e2e_calibrate("E15-no_current", "fit", "no --estimate-current",
                                                    "URI bay alone, current not estimated", os.path.join(DATA, "uri_bay"),
                                                    ["--workers", "2"], URI_SPEC, expect_mode="time block")
    rows["E16-finish"] = e16_finish
    rows["E17-report_validate"] = e17_report_validate
    rows["E18-fit_sim_mujoco"] = e18_fit_sim_mujoco
    rows["E19-fit_sim_gazebo"] = e19_fit_sim_gazebo
    rows["E20-fit_stonefish"] = e20_fit_stonefish
    rows["E21-export_stonefish"] = e21_export_stonefish
    rows["E25-export_sim_mujoco"] = e25_export_sim_mujoco
    rows["E25b-export_sim_gazebo"] = e25b_export_sim_gazebo
    rows["E26-fit_noise"] = e26_fit_noise
    rows["E22-web"] = e22_web
    rows["E23-sound_untouched"] = e23_sound_untouched
    rows["E24-wandering"] = e24_wandering
    return rows


def _e2e_retry() -> None:
    """E8's split with the seed whose single start failed (training cost 1599 against E1's 20): the fit must notice,
    try more starting points, and end with a working calibration (#93)."""
    row = "E8b-retry"
    _e2e_calibrate(row, "fit", "failed start retried", "E8 again, same seed: a failed single start is retried",
                   day("12-01-2024"), ["--heldout-recordings", "odom-12-01-2024-run3", "odom-12-01-2024-run9",
                                       "--seed", "70209514"], expect_mode="recordings")
    m = json.load(open(os.path.join(OUT, row, "metrics.json")))
    auto = (m.get("optimizer") or {}).get("auto_restart")
    r = m["rmse"]
    e8 = json.load(open(os.path.join(OUT, "E8-named", "metrics.json")))["optimizer"]["cost"]
    same_start = bool(auto) and auto["first_cost"] == e8   # --seed repeats the failed fit bit for bit
    ok = same_start and r["calibrated_train"] < 0.9 * r["default_train"]
    record(F, f"{row}-check", "fit", "auto restart", "the failed start was noticed and retried", day("12-01-2024"),
           {"auto_restart": auto, "first_cost_is_E8s": same_start, "E8_cost": e8,
            "train_rmse": [r["default_train"], r["calibrated_train"]],
            "heldout_skill": m["skill_vs_zero"]["calibrated_heldout_full_replay"]},
           "the first fit repeats E8's exactly; retried, and the kept fit beats the starting description by >10%", ok)


def e23_sound_untouched() -> None:
    """E4 again (19 Feb alone), same seed: a sound fit is probed but not retried, and its result is unchanged to the
    last bit (the probe only evaluates after the fit)."""
    row = "E23-sound_untouched"
    seed = yaml.safe_load(open(os.path.join(OUT, "E4-bluerov_19-02-2024", "calibrated_parameters.yaml")))["seed"]
    _e2e_calibrate(row, "fit", "sound fit untouched", "E4 again, same seed: probed, not retried", day("19-02-2024"),
                   ["--seed", str(seed), "--skip-identifiability"], expect_mode="recordings")
    m = json.load(open(os.path.join(OUT, row, "metrics.json")))["optimizer"]
    e4 = json.load(open(os.path.join(OUT, "E4-bluerov_19-02-2024", "metrics.json")))["optimizer"]["cost"]
    ok = "auto_restart" not in m and m["cost"] == e4 and m["n_restarts"] == 1
    record(F, f"{row}-check", "fit", "sound fit untouched", "no retry, same cost to the last bit", day("19-02-2024"),
           {"cost": m["cost"], "E4_cost": e4, "retried": "auto_restart" in m}, "not retried; cost identical to E4's", ok)


WANDERING = os.path.expanduser("~/strathe_formats_data/E-bluerov-px4/csv/ingested")   # #94's CSV route


def e24_wandering() -> None:
    """#94's CSV route, seed 94: the first fit used its whole budget on a rough cost (training cost 103, held-out skill
    -0.02; the PX4 route on the same recordings reached 39 and +0.38). It must be caught as rough and retried."""
    row = "E24-wandering"
    _e2e_calibrate(row, "fit", "wandering fit retried", "#94's CSV route, seed 94: a fit that wandered out of budget",
                   WANDERING, ["--seed", "94", "--skip-identifiability", "--heldout-dates", "16-02-2024", "19-02-2024"])
    m = json.load(open(os.path.join(OUT, row, "metrics.json")))
    auto = (m.get("optimizer") or {}).get("auto_restart") or {}
    r = m["rmse"]
    skill = m["skill_vs_zero"]["calibrated_heldout_full_replay"]
    ok = auto.get("why") == "rough" and auto["best_cost"] < auto["first_cost"] and \
        r["calibrated_train"] < 0.9 * r["default_train"]
    record(F, f"{row}-check", "fit", "wandering fit retried", "caught as rough, retried, a better fit kept", WANDERING,
           {"auto_restart": auto, "train_rmse": [r["default_train"], r["calibrated_train"]], "heldout_skill": skill,
            "heldout_mean_skill": round(sum(skill) / len(skill), 3), "before_fix": {"cost": 102.597, "mean_skill": -0.024}},
           "retried because rough; the kept fit has a lower training cost and beats the starting description", ok)


def _subset_and_calibrate(row: str, n: int) -> None:
    folder = os.path.join(DATA, f"bluerov_12-01-2024_{n}recs")
    os.makedirs(folder, exist_ok=True)
    for f in files(day("12-01-2024"))[:n]:
        dst = os.path.join(folder, os.path.basename(f))
        if not os.path.exists(dst):
            os.symlink(os.path.realpath(f), dst)
    _e2e_calibrate(row, "recordings available", n, f"one day with {n} recordings", folder, [], expect_mode="recordings")


def e16_finish() -> None:
    """An interrupted one-session fit (budget 3), finished from its checkpoint: the split and warning survive."""
    row, out = "E16-finish", os.path.join(OUT, "E16-finish")
    shutil.rmtree(out, ignore_errors=True)
    base = ["calibrate", "--data", day("19-02-2024"), "--name", row, "--out", out, "--skip-identifiability"]
    rc1, s1 = strathe(base + ["--max-nfev", "3"], os.path.join(OUT, f"{row}_a.log"))
    rc2, s2 = strathe(base + ["--finish"], os.path.join(OUT, f"{row}_b.log"))
    m = json.load(open(os.path.join(out, "metrics.json"), encoding="utf-8")) if os.path.exists(
        os.path.join(out, "metrics.json")) else {}
    rep = open(os.path.join(out, "calibration_report.md")).read() if os.path.exists(
        os.path.join(out, "calibration_report.md")) else ""
    ok = rc1 == 0 and rc2 == 0 and m.get("dataset", {}).get("split_mode") == "recordings" and "within one session" in rep
    record(F, row, "consumer", "finalize_checkpoint.py", "--finish on a one-session fit", day("19-02-2024"),
           {"exit": [rc1, rc2], "dataset": m.get("dataset")}, "finished; split and warning kept", ok, s1 + s2)


def e17_report_validate() -> None:
    row, res = "E17-report_validate", os.path.join(OUT, "E4-bluerov_19-02-2024")
    if not os.path.exists(os.path.join(res, "metrics.json")):
        return record(F, row, "consumer", "calibration_report.py validation.py", "needs E4 first", "-", {}, "E4 done",
                      False)
    before = json.load(open(os.path.join(res, "validation", "validation.json")))
    rc1, s1 = strathe(["report", "--result", res, "--data", day("19-02-2024")], os.path.join(OUT, f"{row}_r.log"))
    rc2, s2 = strathe(["validate", "--result", res, "--data", day("19-02-2024")], os.path.join(OUT, f"{row}_v.log"))
    after = json.load(open(os.path.join(res, "validation", "validation.json")))
    rep = open(os.path.join(res, "calibration_report.md")).read()
    same = before["heldout_dates"] == after["heldout_dates"] and before["trajectory"] == after["trajectory"]
    record(F, row, "consumer", "calibration_report.py validation.py", "strathe report and validate, standalone, on a "
           "one-session calibration", day("19-02-2024"), {"exit": [rc1, rc2], "held": after["heldout_dates"]},
           "same held-out recordings and numbers as calibrate wrote; warning in the report",
           rc1 == 0 and rc2 == 0 and same and "within one session" in rep, s1 + s2)


def _sim_fit(row: str, value: str, argv: list, data: str, report_name: str) -> None:
    out = os.path.join(OUT, row)
    shutil.rmtree(out, ignore_errors=True)
    rc, secs = strathe([*argv, "--data", data, "--out", out], os.path.join(OUT, f"{row}.log"))
    rep = os.path.join(out, report_name)
    text = open(rep, encoding="utf-8").read() if os.path.exists(rep) else ""
    log = open(os.path.join(OUT, f"{row}.log"), encoding="utf-8").read()
    record(F, row, "consumer", value, f"{argv[0]} on one session", data,
           {"exit": rc, "report": os.path.exists(rep), "tail": log.strip().splitlines()[-3:]},
           "completes; the within-session warning printed and in the report",
           rc == 0 and "within one session" in log and "within one session" in text, secs)


def e18_fit_sim_mujoco() -> None:
    _sim_fit("E18-fit_sim_mujoco", "simfit/runner.py simfit/core.py",
             ["fit-sim", "--sim", "mujoco", "--model", _mjcf(), "--reference-point", "0", "0", "0", "--fit", "drag",
              "--max-iterations", "5", "--workers", "4"], day("19-02-2024"), "mujoco_fit_report.md")


def e19_fit_sim_gazebo() -> None:
    H = os.path.expanduser("~/strathe_gz_public")
    _sim_fit("E19-fit_sim_gazebo", "simfit (gazebo, Docker)",
             ["fit-sim", "--sim", "gazebo", "--world",
              os.path.join(CAL, "simulators", "gazebo", "example_bluerov2_gz", "world.sdf"),
              "--model-file", f"{H}/bluerov2_gz/models/bluerov2_heavy/model.sdf", "--resource-path",
              f"{H}/bluerov2_gz/models", "--reference-point", "0", "0", "0", "--fit", "buoyancy",
              "--max-iterations", "3", "--workers", "3"], day("19-02-2024"), "gazebo_fit_report.md")


URI_SF = ["--scenario", os.path.expanduser("~/strathe_uri_lab_v3/stonefish_v2/lab/alpha_world.scn"), "--robot",
          "alpha_rise", "--part", "Hull", "--reference-point", "-0.76", "0", "0", "--find",
          "world_of_stonefish=" + os.path.expanduser("~/strathe_uri_lab_v2/stonefish/world"), "--stonefish-data",
          os.path.expanduser("~/strathe_uri_lab_v3/stonefish_v2/world/data"), "--buoyancy-part", "Foam"]


def e20_fit_stonefish() -> None:
    _sim_fit("E20-fit_stonefish", "stonefish_fit.py",
             ["fit-stonefish", *URI_SF, "--fit", "drag", "--axisymmetric", "--max-iterations", "3", "--workers", "3",
              "--replay-tool", "docker"], os.path.join(DATA, "uri_bay"), "stonefish_fit_report.md")


def e21_export_stonefish() -> None:
    _sim_fit("E21-export_stonefish", "stonefish_export.py",
             ["export-stonefish", "--calibration", os.path.expanduser("~/strathe_uri_lab_v2/runs/cal_B0r1_pondout"),
              "--mass", "33", *URI_SF, "--heldout-recordings", "alpha_rise-27-03-2026-run1::last30%",
              "--replay-tool", "docker"], os.path.join(DATA, "uri_pond"), "stonefish_export_report.md")


def e25_export_sim_mujoco() -> None:
    """MuJoCo can't hold Strathe's model (no per-axis linear damping, no buoyancy moment): export-sim says so plainly."""
    row, res = "E25-export_sim_mujoco", os.path.join(OUT, "E4-bluerov_19-02-2024")
    rc, secs = strathe(["export-sim", "--sim", "mujoco", "--model", _mjcf(), "--reference-point", "0", "0", "0",
                        "--calibration", res, "--data", day("19-02-2024"), "--out", os.path.join(OUT, row)],
                       os.path.join(OUT, f"{row}.log"))
    log = open(os.path.join(OUT, f"{row}.log"), encoding="utf-8").read()
    record(F, row, "refusal", "export-sim --sim mujoco", "export-sim to MuJoCo on one session: a plain refusal",
           day("19-02-2024"), {"exit": rc}, "exit 2, naming what MuJoCo lacks and fit-sim instead",
           rc == 2 and "fit-sim --sim mujoco" in log and "Traceback" not in log, secs)


def e25b_export_sim_gazebo() -> None:
    """export-sim scores a calibration in Gazebo on the held-out part: the one-session split must reach it."""
    H = os.path.expanduser("~/strathe_gz_public")
    _sim_fit("E25b-export_sim_gazebo", "simfit/runner.py (export)",
             ["export-sim", "--sim", "gazebo", "--world",
              os.path.join(CAL, "simulators", "gazebo", "example_bluerov2_gz", "world.sdf"),
              "--model-file", f"{H}/bluerov2_gz/models/bluerov2_heavy/model.sdf", "--resource-path",
              f"{H}/bluerov2_gz/models", "--reference-point", "0", "0", "0",
              "--calibration", os.path.join(OUT, "E4-bluerov_19-02-2024"), "--workers", "3"], day("19-02-2024"),
             "gazebo_export_report.md")


def e26_fit_noise() -> None:
    """fit-noise reads every recording (it holds nothing out): one field day must work as before."""
    row, out = "E26-fit_noise", os.path.join(OUT, "E26-fit_noise.yaml")
    rc, secs = strathe(["fit-noise", "--data", day("19-02-2024"), "--out", out], os.path.join(OUT, f"{row}.log"))
    ok = rc == 0 and os.path.exists(out) and bool(yaml.safe_load(open(out)))
    record(F, row, "consumer", "fit-noise", "fit-noise on one session", day("19-02-2024"), {"exit": rc,
           "written": os.path.exists(out)}, "completes and writes the noise file", ok, secs)


def e22_web() -> None:
    """The web page's calibrate card with a recording name, as a lab would submit it (short budget)."""
    import strathe_web as web
    row = "E22-web"
    m = web.JobManager(os.path.join(OUT, "web_jobs"))
    h = web.StratheHandler.__new__(web.StratheHandler)
    h.manager = m
    t0 = time.time()
    job = h._start("/jobs/calibrate", {"data": day("16-02-2024"), "name": "stress93web", "heldout":
                                       "odom-16-02-2024-run2", "max_nfev": "5"})
    job.proc.wait()
    log = open(job.log_path, encoding="utf-8").read()
    ok = job.proc.returncode == 0 and "within one session" in log and "odom-16-02-2024-run2" in log
    record(F, row, "web", "cal-heldout", "web calibrate card, one session, a named recording", day("16-02-2024"),
           {"exit": job.proc.returncode, "tail": log.strip().splitlines()[-2:]}, "the job completes with the split",
           ok, time.time() - t0)


def rescore(rows) -> int:
    """Score finished rows again (e.g. after a fix to the scoring), keeping their calibrations."""
    import csv
    last = {r["row"]: r for r in csv.DictReader(open(os.path.join(HERE, "results_93.csv"), encoding="utf-8"))}
    for rid in rows:
        r = last[rid]
        n = json.loads(r["numbers"])
        out = os.path.join(OUT, rid)
        spec = URI_SPEC if "uri" in rid else _default_spec()
        other = os.path.join(DATA, "uri_pond" if rid.endswith("bay") else "uri_bay")
        n["within"] = score(out, r["data"], n["held"], spec)
        n["other_sessions"] = score(out, other, None, spec, whole=True)
        n["optimism_mean_skill"] = round(n["within"]["mean_skill"] - n["other_sessions"]["mean_skill"], 3)
        n["rescored"] = "other session scored whole, with its current estimated (as calibrate scores held-out sessions)"
        record(F, rid, r["variable"], r["value"], r["what"], r["data"], n, r["expected"], r["passed"] == "pass",
               float(r["seconds"]))
    return 0


def e2e(only=None) -> int:
    rows = _e2e_rows()
    for rid, fn in rows.items():
        if only and rid not in only and not any(rid.startswith(o) for o in only):
            continue
        print(f"=== {rid} ({time.strftime('%H:%M')})", flush=True)
        try:
            fn()
        except Exception as e:
            record(F, rid, "e2e", rid, "crashed", "-", {"error": f"{type(e).__name__}: {e}"}, "completes", False)
    return 0


ROWS_FAST = ("F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "P")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fast"
    if cmd == "list":
        print("fast:", " ".join(ROWS_FAST))
        print("e2e:", " ".join(_e2e_rows()))
        sys.exit(0)
    if cmd == "rescore":
        sys.exit(rescore(sys.argv[2:]))
    sys.exit({"fast": fast, "mutants": mutants}.get(cmd, lambda: e2e(sys.argv[2:]))())
