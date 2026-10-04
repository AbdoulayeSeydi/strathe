"""drag_reference.py: published BlueROV2 drag values and the opt-in --drag modes (PROJECT_STATUS.md #63)."""
import numpy as np
import pytest

import drag_reference as dr
import strathe_cli as cli
from optimizer import initial_parameters, load_vehicle_spec, parameter_bounds

SPEC = load_vehicle_spec("vehicle_spec.yaml")
NAMES = list(initial_parameters(SPEC).param_names)
LB, UB = parameter_bounds(SPEC)


def test_published_values_are_wu_2018_table_5_3():
    v = dr.reference_values("bluerov2_wu2018")
    assert (v["Xu"], v["Xuu"], v["Yv"], v["Yvv"], v["Zw"], v["Zww"]) == (-4.03, -18.18, -6.22, -21.66, -5.18, -36.99)
    assert all(v[k] == -0.07 for k in ("Kp", "Mq", "Nr")) and all(v[k] == -1.55 for k in ("Kpp", "Mqq", "Nrr"))
    with pytest.raises(ValueError, match="unknown drag reference"):
        dr.reference_values("bluerov9")


def test_published_values_are_clamped_into_the_spec_bounds_and_say_so():
    clamped, notes = dr.within_bounds(dr.reference_values("bluerov2_wu2018"), NAMES, LB, UB)
    assert clamped["Xu"] == -4.03 and clamped["Kp"] == -0.1 and clamped["Nr"] == -0.2  # -0.07 is outside
    assert len(notes) == 3 and all("outside" in n for n in notes)
    assert dr.DRAG_MODES == ("free", "prior")  # linear-fixed made DFKI's replay unstable (#63) and was removed


def test_prior_adds_one_log_distance_residual_per_term():
    targets, _ = dr.within_bounds(dr.reference_values("bluerov2_wu2018"), NAMES, LB, UB)
    base = lambda x, *a: np.array([1.0, 2.0])
    f = dr.prior_residual_wrapper(base, NAMES, targets, weight=0.25)
    x = initial_parameters(SPEC).to_vector().copy()
    for k, v in targets.items():
        x[NAMES.index(k)] = v
    r = f(x)
    assert r.shape == (2 + 12,) and np.allclose(r[2:], 0)  # at the published values: no pull
    x[NAMES.index("Xu")] = 2 * targets["Xu"]
    assert f(x)[2 + list(targets).index("Xu")] == pytest.approx(0.5 * np.log(2))  # sqrt(0.25) * ln 2


def test_calibrate_passes_drag_to_the_fit_only(tmp_path):
    calls = []
    run = lambda cmd: calls.append(cmd) or 0
    assert cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "o"), "--drag", "prior"],
                    runner=run) == 0
    fit = next(c for c in calls if c[1] == "run_calibration.py")
    assert fit[fit.index("--drag") + 1] == "prior"
    assert all("--drag" not in c for c in calls if c[1] != "run_calibration.py")
