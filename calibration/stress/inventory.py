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
    return rows


LOG_READERS = {"px4": "ulog_reader", "moos": "moos_reader", "lcm": "lcm_reader"}


def _schema(model, prefix="") -> list:
    """(path, values) for every field of a pydantic model, recursively: a Literal's values (also inside Optional,
    list or a dict's keys), else 'given'."""
    import typing
    from pydantic import BaseModel

    def literal_or_model(t):
        o = typing.get_origin(t)
        if o is typing.Literal:
            return t
        if o is typing.Union:
            for a in typing.get_args(t):
                if a is not type(None):
                    r = literal_or_model(a)
                    if r is not None:
                        return r
            return None
        if o in (list, dict):
            for a in typing.get_args(t):
                r = literal_or_model(a)
                if r is not None:
                    return r
            return None
        return t if isinstance(t, type) and issubclass(t, BaseModel) else None

    out = []
    for name, f in model.model_fields.items():
        path = f"{prefix}{f.alias or name}"
        inner = literal_or_model(f.annotation)
        if inner is not None and typing.get_origin(inner) is typing.Literal:
            out.append((path, tuple(typing.get_args(inner))))
        elif inner is not None:
            out.append((path, ("given",)))
            out += _schema(inner, path + ".")
        else:
            out.append((path, ("given",)))
    return out


def variables_94() -> list:
    """Per log format: every ingest-spec field and option value, the thruster block's keys and command units, the
    reader's column kinds, and the entry points."""
    import importlib
    import data_ingestion as di
    import thruster_model as tm
    rows = []
    for fmt, mod in LOG_READERS.items():
        for path, vals in _schema(di.IngestSpec):
            for v in vals:
                if path == "source.format" and v != fmt:   # each format's own value
                    continue
                rows.append(("94", f"{fmt}|spec:{path}={v}", "ingest spec field / value", v))
        rows += [("94", f"{fmt}|thrusters:{k}", "thruster block key", k) for k in sorted(tm.THRUSTER_BLOCK_KEYS)]
        rows += [("94", f"{fmt}|command:{k}", "thruster command units", k) for k in tm.COMMAND_UNITS]
        rows += [("94", f"{fmt}|column:{k}", "column reference kind", k)
                 for k in importlib.import_module(mod).COLUMN_KINDS]
        rows += [("94", f"{fmt}|entry:{k}", "entry point", k) for k in ("strathe ingest", "calibrate --ingest-spec",
                                                                         "web Convert logs", "strathe check")]
    return rows


def variables_95() -> list:
    """Every describe-vehicle option and choice, the water kinds, and the web card's fields."""
    import vehicle_describe as VD
    rows = []
    for act in VD.build_parser(add_help=False)._actions:
        if not act.option_strings:
            continue
        rows.append(("95", f"option:{act.option_strings[0]}", "describe-vehicle option", "given / left out"))
        for c in act.choices or ():
            rows.append(("95", f"option:{act.option_strings[0]}={c}", "describe-vehicle choice", c))
    rows += [("95", f"water:{w}", "water", w) for w in (*VD.WATER, "a density")]
    rows += [("95", f"shape-rules:{k}", "bound rules", k) for k in ("box_bounds", "torpedo_bounds")]
    web = open(os.path.join(CAL, "strathe_web.py"), encoding="utf-8").read()
    i = web.index("/jobs/describe_vehicle>")
    card = web[i:web.index("</form>", i)]
    rows += [("95", f"web:{n}", "web card field", "dates") for n in sorted(set(re.findall(r" name=(\w+)", card)))]
    return rows


FEATURES = {"93": variables_93, "94": variables_94, "95": variables_95}


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
