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
