"""
stress/common.py -- shared pieces of the stress tests for #93-#95 (STRESS_TEST_93_95.md).

Every check appends one row to stress/results_<feature>.csv: what was varied, the command or call, the data, the
numbers, what was expected and whether it passed. stress/report.py turns the rows into the write-up.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
REPO = os.path.dirname(CAL)
PY = sys.executable
FIELDS = ("feature", "row", "variable", "value", "what", "data", "numbers", "expected", "passed", "seconds", "when")


def results_path(feature: str) -> str:
    return os.path.join(HERE, f"results_{feature}.csv")


def record(feature: str, row: str, variable: str, value, what: str, data: str, numbers: dict, expected: str,
           passed: bool, seconds: float = 0.0) -> bool:
    """Append one result (and print it). Returns `passed`."""
    path = results_path(feature)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow({"feature": feature, "row": row, "variable": variable, "value": str(value), "what": what,
                    "data": data, "numbers": json.dumps(numbers, default=str), "expected": expected,
                    "passed": "pass" if passed else "FAIL", "seconds": f"{seconds:.1f}",
                    "when": time.strftime("%Y-%m-%d %H:%M")})
    print(f"[{'pass' if passed else 'FAIL'}] {row} {variable}={value}: {what}" + ("" if passed else f" ({numbers})"),
          flush=True)
    return passed


def strathe(args: list, log: str, timeout: Optional[float] = None) -> tuple:
    """Run `python strathe.py ARGS` from the repository; (exit code, seconds). Output goes to `log`."""
    t0 = time.time()
    os.makedirs(os.path.dirname(log), exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run([PY, os.path.join(REPO, "strathe.py"), *args], cwd=REPO, stdout=f,
                            stderr=subprocess.STDOUT, timeout=timeout).returncode
    return rc, time.time() - t0


def load_rows(feature: str) -> list:
    path = results_path(feature)
    return list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else []


def all_pairs(options: dict, seed: int) -> list:
    """All-pairs combinations of options {variable: [values]}: a greedy cover in which every pair of values of any two
    variables appears in at least one case."""
    import itertools
    import random
    names = list(options)
    need = {(a, x, b, y) for a, b in itertools.combinations(names, 2) for x in options[a] for y in options[b]}
    rnd, cases = random.Random(seed), []
    while need:
        best, gain = None, -1
        for _ in range(80):
            c = {n: rnd.choice(options[n]) for n in names}
            g = sum((a, c[a], b, c[b]) in need for a, b in itertools.combinations(names, 2))
            if g > gain:
                best, gain = c, g
        cases.append(best)
        need -= {(a, best[a], b, best[b]) for a, b in itertools.combinations(names, 2)}
    return cases
