"""
stress/inventory.py -- every variable a feature has, found in the code, not listed by hand (PROJECT_STATUS.md #93-#95).

    python stress/inventory.py            # writes stress/variables.csv

tests/test_stress_coverage.py checks that each variable here has a row in that feature's stress matrix
(stress/matrix_<feature>.csv), so an option added later without a stress test fails CI.

What counts as a variable, per feature:
  93 (one session): each lab-facing module that splits the logs (reachable from `strathe` / the web page: found by
     following imports and launched scripts), each parser with --heldout-recordings, each split mode and held-out
     kind, each web held-out field, and the split's constants.
"""
from __future__ import annotations

import ast
import csv
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, CAL)
ENTRY = ("strathe_cli.py", "strathe_web.py", "simfit/cli.py")
SPLITTERS = ("split_by_date", "load_runs", "_load_runs")


def _module_file(name: str):
    """A local module name -> its file under calibration/, or None."""
    for p in (name.replace(".", "/") + ".py", name.replace(".", "/") + "/__init__.py"):
        if os.path.exists(os.path.join(CAL, p)):
            return p
    return None


def _tree(rel: str):
    try:
        return ast.parse(open(os.path.join(CAL, rel), encoding="utf-8").read())
    except (OSError, SyntaxError):
        return None


def reachable() -> set:
    """Files `strathe` and the web page can reach: imports (also inside functions) and scripts they launch."""
    seen, todo = set(), list(ENTRY)
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        t = _tree(rel)
        if t is None:
            continue
        for n in ast.walk(t):
            names = []
            if isinstance(n, ast.Import):
                names = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                names = [n.module] + [f"{n.module}.{a.name}" for a in n.names]
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                if re.fullmatch(r"[a-z_0-9]+\.py", n.value) and os.path.exists(os.path.join(CAL, n.value)):
                    todo.append(n.value)
                elif re.fullmatch(r"simfit(\.[a-z_]+)+", n.value):   # `python -m simfit.cli`
                    names = [n.value]
            for name in names:
                f = _module_file(name)
                if f:
                    todo.append(f)
    return seen


def _calls(rel: str, names) -> bool:
    t = _tree(rel)
    return t is not None and any(isinstance(n, ast.Call) and
                                 (getattr(n.func, "attr", None) or getattr(n.func, "id", None)) in names
                                 for n in ast.walk(t))


def heldout_parsers(files) -> list:
    """(file, parser variable) for every add_argument(... '--heldout-recordings' ...)."""
    out = []
    for rel in sorted(files):
        t = _tree(rel)
        for n in ast.walk(t) if t else []:
            if (isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "add_argument"
                    and any(isinstance(a, ast.Constant) and a.value == "--heldout-recordings" for a in n.args)):
                out.append((rel, getattr(n.func.value, "id", "?")))
    return out


def variables_93() -> list:
    import data_split as ds
    files = reachable()
    rows = [("93", f"consumer:{rel}", "module that splits the logs", "dates / recordings / time block")
            for rel in sorted(files) if rel != "data_split.py" and _calls(rel, SPLITTERS)]
    rows += [("93", f"flag:{rel}:{var}", "--heldout-recordings on a parser", "names, last part, refusals")
             for rel, var in heldout_parsers(files)]
    rows += [("93", f"mode:{m}", "split mode", m) for m in ds.SPLIT_MODES]
    rows += [("93", f"kind:{k}", "held-out entry kind", k) for k in ds.HELDOUT_KINDS]
    web = open(os.path.join(CAL, "strathe_web.py"), encoding="utf-8").read()
    rows += [("93", f"web:{i}", "web held-out field", "dates, recordings, last part, refusals")
             for i in sorted(set(re.findall(r"id=([\w-]+) name=heldout", web)))]
    rows += [("93", f"const:{c}", "split constant", getattr(ds, c))
             for c in ("DEFAULT_HELDOUT_FRACTION", "TIME_BLOCK_GAP_S", "MIN_PART_S")]
    import run_calibration as rc
    rows += [("93", f"const:{c}", "failed-fit retry", getattr(rc, c))   # found by #93's and #94's stress tests
             for c in ("AUTO_RESTARTS", "AUTO_RESTART_GAIN", "AUTO_RESTART_ROUGHNESS")]
    return rows


FEATURES = {"93": variables_93}


def inventory(features=None) -> list:
    out = []
    for f in features or FEATURES:
        out += FEATURES[f]()
    return out


def matrix(feature: str) -> dict:
    """stress/matrix_<feature>.csv: variable -> (row ids, note)."""
    path = os.path.join(HERE, f"matrix_{feature}.csv")
    rows = csv.DictReader(open(path, encoding="utf-8")) if os.path.exists(path) else []
    return {r["variable"]: (r["rows"].split(), r.get("note", "")) for r in rows}


if __name__ == "__main__":
    out = os.path.join(HERE, "variables.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(("feature", "variable", "kind", "values"))
        w.writerows(inventory())
    print(f"{len(inventory())} variables -> {out}")
