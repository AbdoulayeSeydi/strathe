"""Objective additions for PROJECT_STATUS.md #50: prediction windows, skill vs
"predict zero", torque reference shift, parallel evaluation, excluded axes."""
import json
import os
from dataclasses import replace

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from data_split import discover_csv_files
from dynamics_model import SimulationConfig, TrajectoryInputs, simulate_trajectory
from objective import (DatasetConfig, ParallelJointResiduals, RunData, _raw_residuals, compute_zero_baseline_scales,
                       horizon_windows, joint_residuals, load_run, rmse_by_dof, shift_torque_reference, skill_vs_zero,
                       zero_baseline_rmse)
from optimizer import CalibrationConfig, initial_parameters, load_vehicle_spec
from thruster_model import BLUEROV2_HEAVY_ALLOCATION

SPEC = load_vehicle_spec("vehicle_spec.yaml")
VEHICLE = SPEC.vehicle
GENERIC = initial_parameters(SPEC)
SYNTH = sorted(discover_csv_files("synth_test_data")[0])


def _runs(offset=(0.0, 0.0, 0.0)):
    return [load_run(f, DatasetConfig(10.0, offset)) for f in SYNTH]


# ----------------------------------------------------------------- windows

def test_horizon_windows_cover_the_run():
    assert horizon_windows(100, None) == [(0, 100)] and horizon_windows(100, 500) == [(0, 100)]
    w = horizon_windows(95, 10)
    assert w[0] == (0, 10) and w[-1] == (90, 95) and all(b - a >= 2 for a, b in w)
    assert horizon_windows(91, 10)[-1] == (80, 90)          # a 1-sample tail predicts nothing: dropped
    with pytest.raises(ValueError):
        horizon_windows(100, 1)


def test_windowed_residuals_restart_from_the_real_state_each_window():
    run = _runs()[0]
    H = 30
    res = _raw_residuals(GENERIC, run, VEHICLE, horizon_samples=H)
    wins = horizon_windows(run.n_samples, H)
    assert res.shape == (sum(b - a - 1 for a, b in wins), 6)
    a, b = wins[1]
    sim = simulate_trajectory(GENERIC, VEHICLE, SimulationConfig(dt=run.dt),
                              TrajectoryInputs(run.real_nu[a], run.forces_torques[a:b], run.roll_pitch[a:b])).nu
    first = wins[0][1] - wins[0][0] - 1
    assert np.allclose(res[first:first + (b - a - 1)], sim[1:] - run.real_nu[a + 1:b])


def test_no_horizon_is_the_original_objective():
    run = _runs()[0]
    sim = simulate_trajectory(GENERIC, VEHICLE, SimulationConfig(dt=run.dt),
                              TrajectoryInputs(run.nu0, run.forces_torques, run.roll_pitch)).nu
    assert np.array_equal(_raw_residuals(GENERIC, run, VEHICLE), sim - run.real_nu)


def test_whole_run_replay_biases_damping_up_under_an_unlogged_disturbance():
    """The finding behind #50, on synthetic ground truth: a roll torque the
    model doesn't see makes the whole-run objective prefer more roll damping
    than the truth (1.6-1.9x over three 300 s realizations in the #50
    investigation; this is the first), because extra damping is its only way
    to limit the drift. How large the bias is depends on the realization."""
    rng = np.random.default_rng(0)

    def ar1(n, scale, a):
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = a * x[i - 1] + rng.normal() * scale
        return x
    true = replace(GENERIC, Kp=-0.8, Kpp=-1.0)
    n = 3000
    F = np.zeros((n, 6))
    F[:, 0] = ar1(n, 3, 0.95)
    F[:, 5] = ar1(n, 0.5, 0.95)
    F[:, 3] = ar1(n, 0.3, 0.95)
    hidden = np.zeros((n, 6))
    hidden[:, 3] = ar1(n, 0.4, 0.97)
    rp = np.zeros((n, 2))
    real = simulate_trajectory(true, VEHICLE, SimulationConfig(dt=0.1), TrajectoryInputs(np.zeros(6), F + hidden, rp)).nu
    run = RunData("synthetic", F, rp, real, real[0].copy(), 0.1, 10.0)
    fit = minimize_scalar(lambda kp: rmse_by_dof(replace(true, Kp=kp), run, VEHICLE)[3], bounds=(-10, -0.1),
                          method="bounded", options={"xatol": 1e-2}).x
    assert fit < 1.5 * true.Kp       # at least 1.5x the true damping (both negative)


# ----------------------------------------------------------------- skill

def test_zero_baseline_and_skill():
    runs = _runs()
    z = zero_baseline_rmse(runs)
    assert np.allclose(z, np.mean([np.sqrt(np.mean(r.real_nu ** 2, axis=0)) for r in runs], axis=0))
    pooled = compute_zero_baseline_scales(runs)
    assert np.allclose(pooled, np.maximum(np.sqrt(np.mean(np.concatenate([r.real_nu for r in runs]) ** 2, axis=0)), 1e-3))
    assert np.allclose(skill_vs_zero([0.0, 1.0, 2.0], [1.0, 1.0, 1.0]), [1.0, 0.0, -1.0])


# ----------------------------------------------------------------- torque reference

def test_shift_torque_reference():
    W = np.random.default_rng(1).normal(size=(50, 6))
    assert shift_torque_reference(W, (0, 0, 0)) is W
    r = np.array([0.01, -0.02, -0.085])
    out = shift_torque_reference(W, r)
    assert np.allclose(out[:, :3], W[:, :3]) and np.allclose(out[:, 3:], W[:, 3:] - np.cross(r, W[:, :3]))
    # the BlueROV2 Heavy's horizontal thrusters sit 8.5 cm above the dataset's
    # torque reference: moving the reference up removes exactly their roll/pitch torque
    horiz = shift_torque_reference(BLUEROV2_HEAVY_ALLOCATION[:, :4].T, (0, 0, -0.085))
    assert np.allclose(horiz[:, 3:5], 0, atol=1e-4)
    assert np.allclose(horiz[:, 5], BLUEROV2_HEAVY_ALLOCATION[5, :4])


def test_dataset_config_applies_and_validates_the_offset():
    plain, shifted = _runs()[0], _runs((0, 0, -0.085))[0]
    assert np.allclose(shifted.forces_torques, shift_torque_reference(plain.forces_torques, (0, 0, -0.085)))
    with pytest.raises(ValueError):
        DatasetConfig(10.0, (0.0, float("nan"), 0.0)).validate()


# ----------------------------------------------------------------- parallel evaluation

def test_parallel_evaluation_is_identical_to_serial():
    runs = _runs()
    x = GENERIC.to_vector()
    scales = compute_zero_baseline_scales(runs)
    with ParallelJointResiduals(runs, VEHICLE, 2) as pj:
        for args in ((None, True, None, None), (scales, True, None, 20), (scales, False, None, None)):
            assert np.array_equal(pj(x, runs, VEHICLE, *args), joint_residuals(x, runs, VEHICLE, *args))
        with pytest.raises(ValueError, match="different set of runs"):
            pj(x, runs[:-1], VEHICLE)
        with pytest.raises(ValueError, match="stand-in"):
            pj(x, runs, VEHICLE, None, True, lambda *a: None)
    with pytest.raises(ValueError):
        ParallelJointResiduals(runs, VEHICLE, 1)


# ----------------------------------------------------------------- run_calibration wiring

def test_excluded_axes_zero_their_weight_and_hold_their_parameters():
    from run_calibration import excluded_axis_setup
    scales, held = excluded_axis_setup(("p",), None, GENERIC)
    assert np.isinf(scales[3]) and np.all(scales[[0, 1, 2, 4, 5]] == 1)
    assert held == {"Kp": GENERIC.Kp, "Kpp": GENERIC.Kpp, "Ieff_roll": GENERIC.Ieff_roll}
    scales, held = excluded_axis_setup(("p", "q"), np.full(6, 2.0), GENERIC)
    assert np.isinf(scales[3:5]).all() and scales[0] == 2.0 and held["cg_cb_offset_z"] == GENERIC.cg_cb_offset_z
    with pytest.raises(ValueError):
        excluded_axis_setup(("r",), None, GENERIC)


def test_pipeline_records_skill_and_objective_settings(tmp_path):
    """End to end on the synthetic logs, with every new option on. Most
    parameters are held so the fit converges in seconds."""
    from run_calibration import PipelineConfig, run_pipeline
    names = GENERIC.param_names
    free = {"Xu", "Nr"}
    cc = CalibrationConfig(max_nfev=60, eval_workers=2, horizon_samples=20,
                           fixed_params={n: float(getattr(GENERIC, n)) for n in names if n not in free})
    report = run_pipeline(PipelineConfig(
        data_dir="synth_test_data", spec_path="vehicle_spec.yaml", output_dir=str(tmp_path), sampling_rate_hz=10.0,
        heldout_dates=["03-03-2024"], heldout_fraction=0.2, calibration_config=cc, workers=1,
        axis_scaling="zero_rms", exclude_axes=("p", "q"), torque_reference_offset_m=(0.0, 0.0, -0.085)))
    m = json.load(open(os.path.join(tmp_path, "metrics.json"), encoding="utf-8"))
    assert m["objective"] == {"axis_scaling": "zero_rms", "prediction_horizon_samples": 20,
                              "excluded_axes": ["p", "q"], "torque_reference_offset_m": [0.0, 0.0, -0.085],
                              "held_parameters": m["objective"]["held_parameters"]}
    assert {"Kp", "Kpp", "Mq", "Mqq", "cg_cb_offset_z"} <= set(m["objective"]["held_parameters"])
    sk = m["skill_vs_zero"]
    assert len(sk["zero_heldout"]) == 6 and sk["short_window_s"] == 3.0
    assert np.allclose(sk["calibrated_heldout_full_replay"],
                       1 - np.array(m["per_dof"]["calibrated_heldout"]) / np.array(sk["zero_heldout"]))
    assert report.calibrated_parameters.Kp == GENERIC.Kp


def test_torque_reference_default_comes_from_the_vehicle_spec():
    from types import SimpleNamespace
    from run_calibration import spec_torque_reference_offset
    assert spec_torque_reference_offset(SPEC) == (0.0, 0.0, -0.085)
    assert spec_torque_reference_offset(SimpleNamespace(raw={})) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        spec_torque_reference_offset(SimpleNamespace(raw={"data_conventions": {"torque_reference_offset_m": [1, 2]}}))


def test_finalize_checkpoint_writes_a_usable_not_converged_result(tmp_path):
    import yaml as _yaml
    from calibration_report import build_report
    from finalize_checkpoint import finalize
    ck = tmp_path / "checkpoint.yaml"
    ck.write_text(_yaml.safe_dump({"call_count": 5, "best_cost_so_far": 1.5, "best_parameters_so_far":
                                   {n: float(getattr(GENERIC, n)) for n in GENERIC.param_names}}))
    (tmp_path / "eval_timing.csv").write_text("eval\n1\n2\n3\n")
    m = finalize(str(ck), "synth_test_data", "vehicle_spec.yaml", ["03-03-2024"], str(tmp_path), 10.0,
                 axis_scaling="zero_rms", exclude_axes=("p", "q"), reason="test stop")
    assert m["optimizer"]["converged"] is False and m["optimizer"]["nfev"] == 3
    assert "test stop" in m["optimizer"]["message"]
    assert m["objective"]["torque_reference_offset_m"] == [0.0, 0.0, -0.085]    # from vehicle_spec.yaml
    assert "Kp" in m["objective"]["held_parameters"]
    assert "not formally converged" in (tmp_path / "calibrated_parameters.yaml").read_text()
    r = build_report(str(tmp_path))
    v = {x["axis"]: x for x in r["verdicts"]}
    assert v["p"]["excluded"] and "not calibrated" in v["p"]["why"]
    assert "stopped before fully converging" in open(r["report"], encoding="utf-8").read()
