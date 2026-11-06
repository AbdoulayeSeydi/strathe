"""Every variable a feature has (found in the code by stress/inventory.py) has a row in its stress matrix, and every
row the matrix names exists in the feature's harness, so an option added without a stress test fails here
(PROJECT_STATUS.md #93-#95; STRESS_TEST_93_95.md)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
STRESS = os.path.join(os.path.dirname(HERE), "stress")
sys.path.insert(0, STRESS)

import inventory  # noqa: E402


@pytest.mark.parametrize("feature", sorted(inventory.FEATURES))
def test_every_variable_has_a_stress_row(feature):
    m = inventory.matrix(feature)
    missing = [v[1] for v in inventory.inventory([feature]) if v[1] not in m]
    assert not missing, f"no stress row in stress/matrix_{feature}.csv for: {missing}"
    for var, (rows, note) in m.items():
        assert rows, f"{var}: no rows"
        if rows == ["exempt"]:
            assert len(note) > 20, f"{var}: an exemption needs its reason"


@pytest.mark.parametrize("feature", sorted(inventory.FEATURES))
def test_every_matrix_row_exists_in_the_harness(feature):
    import importlib
    run = importlib.import_module(f"run_{feature}")
    if hasattr(run, "all_rows"):   # the harness lists its row ids: the matrix must name real ones
        known = run.all_rows()
    else:                          # #93's rows are named by group (F1, E1, ...)
        known = set(run.ROWS_FAST) | set(run._e2e_rows())
        known |= {k.split("-")[0] for k in known}
    for var, (rows, _) in inventory.matrix(feature).items():
        for r in rows:
            assert r == "exempt" or r in known, f"{var}: row {r} isn't in stress/run_{feature}.py"


def test_the_report_lists_every_row():
    """STRESS_TEST_93_95.md once counted all 349 #94 rows but listed 15: rows outside the F/M/E groups were dropped."""
    import report
    from common import load_rows
    text = report.render()
    for feature in report.TITLES:
        missing = [r["row"] for r in load_rows(feature) if f"| {r['row']} |" not in text]
        assert not missing, f"#{feature} rows missing from the report: {missing[:5]}"


def test_the_results_page_lists_every_row():
    """The private results page (stress/page.py) is built from the same rows as the report, and drops none."""
    import page
    from common import load_rows
    text = page.render()
    for feature in ("93", "94", "95", "96", "97"):
        missing = [r["row"] for r in load_rows(feature) if f"<code>{page.E(r['row'])}</code>" not in text]
        assert not missing, f"#{feature} rows missing from the page: {missing[:5]}"


def test_what_other_matrices_owe_97_names_rows_that_exist():
    """An exemption saying "proved in #97" is only true if the row does: #94 once said it of 111 options before any row did."""
    import re
    import run_97
    known = set(run_97.all_rows())
    for feature in ("94", "96"):
        for var, (rows, note) in inventory.matrix(feature).items():
            if rows == ["exempt"] and "#97" in note:
                named = re.findall(r"\b(?:LF|X|XS|P|S|E|J)-[\w.-]*\w", note)
                assert named, f"{var}: says #97 proves it without naming the row"
                assert set(named) <= known, f"{var}: names rows #97 doesn't have: {set(named) - known}"


def _results_97():
    from common import load_rows
    return load_rows("97")


def test_97s_results_were_judged_by_the_bars_in_force():
    """Results keep the hash of the bars they met: moving a bar after seeing a result can't go unnoticed."""
    import json
    import run_97
    old = [r["row"] for r in _results_97() if json.loads(r["numbers"].replace("'", '"') if r["numbers"].startswith("{'") else r["numbers"] or "{}").get("bars") not in (None, run_97.BARS_HASH)]
    assert not old, f"judged by other bars: {old[:5]}"


def test_97s_failing_rows_are_written_down():
    """A failure is a finding: every failing row is in LIMITS, with what it means for a lab."""
    limits = open(os.path.join(os.path.dirname(HERE), "..", "LIMITS.md"), encoding="utf-8").read()
    missing = [r["row"] for r in _results_97() if r["passed"] != "pass" and r["row"] not in limits]
    assert not missing, f"failing rows with no LIMITS.md entry: {missing}"
