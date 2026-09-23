"""fleet_stats.py tests on synthetic episodes with known ground truth."""
import numpy as np
import pytest

from fleet_stats import (belief_success_link, dose_response, mcnemar_exact_p, paired_contrast,
                         summarize, wilson_ci)


def _row(belief, ep, success, collision=False, err=0.1, policy="p1", comm="L2", backend="stand_in"):
    return {"belief": belief, "episode_key": str(ep), "success": bool(success), "collision": bool(collision),
            "mean_belief_error_m": err, "policy": policy, "comm": comm, "backend": backend,
            "steps": 100, "delivery_rate": 0.7}


def test_wilson_ci_known_values():
    lo, hi = wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-3) and hi == pytest.approx(0.7634, abs=1e-3)


def test_mcnemar_exact():
    assert mcnemar_exact_p(0, 10) == pytest.approx(2 * 0.5 ** 10)
    assert mcnemar_exact_p(5, 5) == 1.0
    assert mcnemar_exact_p(0, 0) == 1.0


def test_paired_contrast_detects_a_real_difference():
    rng = np.random.default_rng(0)
    rows = []
    for ep in range(200):
        rows.append(_row("calibrated", ep, True, err=0.05))
        rows.append(_row("generic", ep, rng.random() < 0.5, err=0.20))
    (res,) = paired_contrast(rows, "calibrated", "generic")
    assert res["n_pairs"] == 200
    assert res["success_diff"] == pytest.approx(1 - np.mean([r["success"] for r in rows if r["belief"] == "generic"]))
    assert res["success_diff_ci95"][0] > 0.35
    assert res["mcnemar_p"] < 1e-10 and res["b_only_successes"] == 0
    assert res["belief_error_diff_m"] == pytest.approx(-0.15)


def test_paired_contrast_null_effect_ci_covers_zero():
    rng = np.random.default_rng(1)
    rows = []
    for ep in range(300):
        s = rng.random() < 0.4
        rows += [_row("a", ep, s), _row("b", ep, s)]
    (res,) = paired_contrast(rows, "a", "b")
    assert res["success_diff"] == 0 and res["mcnemar_p"] == 1.0
    assert res["success_diff_ci95"][0] <= 0 <= res["success_diff_ci95"][1]


def test_paired_contrast_only_uses_matched_episodes_and_strata():
    rows = [_row("a", 1, True), _row("b", 1, False),
            _row("a", 2, True),                          # no partner -> excluded
            _row("b", 3, True, policy="p2")]              # different policy -> excluded
    (res,) = paired_contrast(rows, "a", "b")
    assert res["n_pairs"] == 1


def test_paired_contrast_groups_by_comm():
    rows = []
    for ep in range(50):
        rows += [_row("a", ep, True, comm="L0"), _row("b", ep, True, comm="L0")]
        rows += [_row("a", ep, True, comm="L3a"), _row("b", ep, False, comm="L3a")]
    res = {r["comm"]: r for r in paired_contrast(rows, "a", "b")}
    assert res["L0"]["success_diff"] == 0 and res["L3a"]["success_diff"] == 1


def test_dose_response_recovers_a_negative_slope():
    rng = np.random.default_rng(2)
    alphas = {"a0": 0.0, "a25": 0.25, "a5": 0.5, "a75": 0.75, "a1": 1.0}
    rows = []
    for ep in range(400):
        u = rng.random()  # shared latent difficulty -> realistic within-episode correlation
        for label, a in alphas.items():
            rows.append(_row(label, ep, u < 0.8 - 0.6 * a))
    (res,) = dose_response(rows, alphas)
    assert res["slope_per_unit_alpha"] == pytest.approx(-0.6, abs=0.05)
    assert res["slope_ci95"][1] < 0
    assert res["spearman_rho"] == pytest.approx(-1.0)


def test_dose_response_flat_ci_covers_zero():
    rng = np.random.default_rng(3)
    alphas = {"a0": 0.0, "a5": 0.5, "a1": 1.0}
    rows = [_row(l, ep, rng.random() < 0.5) for ep in range(300) for l in alphas]
    (res,) = dose_response(rows, alphas)
    assert res["slope_ci95"][0] < 0 < res["slope_ci95"][1]


def test_summarize_and_belief_success_link():
    rows = []
    for label, err, p in (("x", 0.02, 0.9), ("y", 0.05, 0.7), ("z", 0.10, 0.5), ("w", 0.20, 0.3)):
        rows += [_row(label, ep, ep < p * 100, err=err) for ep in range(100)]
    summ = summarize(rows, by=("comm", "belief"))
    assert {s["belief"]: s["success_rate"] for s in summ} == {"w": 0.3, "x": 0.9, "y": 0.7, "z": 0.5}
    (link,) = belief_success_link(summ)
    assert link["spearman_rho"] == pytest.approx(-1.0) and link["n_conditions"] == 4
    (link2,) = belief_success_link(summ, exclude_beliefs=["x"])
    assert link2["n_conditions"] == 3
