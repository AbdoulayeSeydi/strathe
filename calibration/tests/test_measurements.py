"""The sensor suite's measurement model and state smoother (PROJECT_STATUS.md #96): at the true parameters, with the
true starting velocities, every sensor's residual is zero; unknown window starts are removed exactly; the smoother
recovers the true velocity from position fixes or depth; which axes the sensors see, and what's held."""
import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

import measurements as ms
import state_smoother as sm
from dynamics_model import SimulationConfig, TrajectoryInputs, simulate_trajectory
from objective import DatasetConfig, convert_run_dataframe, residuals_for_run, rmse_by_dof, joint_residuals
from optimizer import initial_parameters, load_vehicle_spec

SPEC = load_vehicle_spec("vehicle_spec.yaml")
TRUTH = initial_parameters(SPEC)
RATE = 10.0


def truth_frame(n=900, seed=0, dvl=True, pos_every=1, depth_every=1, pos_noise=0.0, acc=False, offset=(5.0, -3.0, 2.0)):
    """A recording made by the truth vehicle itself: smooth thrust, a slowly turning orientation, every sensor read
    from the simulated motion (positions integrated with the logged orientation)."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / RATE

    def smooth(scale, k):
        x = np.convolve(rng.normal(0, 1, n + k), np.ones(k) / k, mode="valid")[:n]
        return scale * x / (np.std(x) + 1e-9)
    F = np.stack([smooth(20, 40), smooth(10, 40), smooth(12, 40), smooth(1, 40), smooth(1, 40), smooth(2, 40)], 1)
    yaw = np.cumsum(smooth(0.2, 60)) / RATE
    R_ = Rotation.from_euler("ZYX", np.stack([yaw, smooth(0.04, 30), smooth(0.04, 30)], 1))
    q = R_.as_quat()
    from dynamics_model import quaternion_to_roll_pitch
    roll, pitch = quaternion_to_roll_pitch(q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    nu = simulate_trajectory(TRUTH, SPEC.vehicle, SimulationConfig(dt=1 / RATE),
                             TrajectoryInputs(nu0=np.zeros(6), forces_torques=F,
                                              roll_pitch_series=np.stack([roll, pitch], 1))).nu
    R = R_.as_matrix()
    path = np.vstack([np.zeros(3), np.cumsum(np.einsum("nij,nj->ni", R, nu[:, :3])[:-1] / RATE, 0)]) + np.array(offset)
    df = pd.DataFrame(F, columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
    df[["Bp", "Bq", "Br"]] = nu[:, 3:6]
    df[["qx", "qy", "qz", "qw"]] = q
    if dvl:
        df[["Bu", "Bv", "Bw"]] = nu[:, :3]
    pos = path + pos_noise * rng.normal(size=path.shape)
    keep = np.zeros(n, bool)
    keep[::pos_every] = True
    pos[~keep] = np.nan
    df["pos_usbl_n"], df["pos_usbl_e"], df["pos_usbl_d"] = pos[:, 0], pos[:, 1], pos[:, 2]
    dep = path[:, 2].copy()
    dk = np.zeros(n, bool)
    dk[::depth_every] = True
    dep[~dk] = np.nan
    df["depth_main"] = dep
    if acc:
        vdot = np.gradient(nu[:, :3], 1 / RATE, axis=0)
        f = vdot + np.cross(nu[:, 3:6], nu[:, :3]) + np.einsum("nji,j->ni", R, np.array([0, 0, -ms.G]))
        df[["acc_x", "acc_y", "acc_z"]] = f
    return df, nu, path


def as_run(df, sensors=("all",), accel=False, anchors=None):
    run = convert_run_dataframe(df, "synthetic-01-11-2026-run1.csv",
                                DatasetConfig(sampling_rate_hz=RATE, sensors=sensors, use_accelerometer=accel))
    if anchors is not None:
        run.anchor_lin = anchors
    return run


# ------------------------------------------------------------------ the measurement model

@pytest.mark.parametrize("H", [None, 50, 100])
def test_every_sensor_reads_exactly_what_the_truth_did(H):
    df, nu, _ = truth_frame(acc=True)
    run = as_run(df, accel=True, anchors=nu[:, :3])
    parts = ms.raw_channel_residuals(TRUTH, run, SPEC.vehicle, H)
    labels = ms.channel_labels(run.channels)
    assert labels == ["gyro.p", "gyro.q", "gyro.r", "dvl.u", "dvl.v", "dvl.w", "main.d", "usbl.n", "usbl.e",
                      "usbl.d", "acc.ax", "acc.ay", "acc.az"]
    for lab, r in zip(labels, parts):
        assert len(r) > 0
        tol = 1e-3 if lab.startswith("acc") else 1e-9   # the accelerometer differentiates numerically
        assert np.max(np.abs(r)) < tol, (lab, np.max(np.abs(r)))


def test_an_unknown_start_position_is_removed_exactly():
    df, nu, _ = truth_frame()
    df2 = df.copy()
    df2[["pos_usbl_n", "pos_usbl_e", "pos_usbl_d"]] += [1234.5, -987.0, 42.0]   # a different origin
    df2["depth_main"] -= 0.73                                                   # a depth sensor's own zero
    a = ms.raw_channel_residuals(TRUTH, as_run(df, anchors=nu[:, :3]), SPEC.vehicle, 100)
    b = ms.raw_channel_residuals(TRUTH, as_run(df2, anchors=nu[:, :3]), SPEC.vehicle, 100)
    for x, y in zip(a, b):
        assert np.allclose(x, y, atol=1e-6)


def test_wrong_parameters_show_in_every_sensor():
    df, nu, _ = truth_frame()
    run = as_run(df, sensors=("usbl", "main"), anchors=nu[:, :3])
    wrong = TRUTH.__class__(**{**TRUTH.__dict__, "Xu": TRUTH.Xu * 3, "Zw": TRUTH.Zw * 3})
    rm_t = ms.channel_rmse(TRUTH, run, SPEC.vehicle, 100)
    rm_w = ms.channel_rmse(wrong, run, SPEC.vehicle, 100)
    assert all(v < 1e-9 for v in rm_t.values())
    assert rm_w["usbl.n"] > 1e-3 and rm_w["main.d"] > 1e-3


def test_hidden_velocity_never_reaches_the_run():
    """The firewall: with the DVL not among the sensors used, the run's u, v, w are NaN, whatever the file held."""
    df, _, _ = truth_frame()
    run = as_run(df, sensors=("usbl", "main"))
    assert np.isnan(run.real_nu[:, :3]).all()
    assert "dvl" not in [c.kind for c in run.channels]
    assert np.isfinite(run.real_nu[:, 3:]).all()


def test_the_residual_vector_keeps_its_length_and_survives_a_blow_up():
    df, nu, _ = truth_frame(pos_every=10, depth_every=5)
    run = as_run(df, sensors=("usbl", "main"), anchors=nu[:, :3])
    run.sensor_scales = ms.still_scales([run], 100)
    v1 = residuals_for_run(TRUTH, run, SPEC.vehicle, horizon_samples=100)
    unstable = TRUTH.__class__(**{**TRUTH.__dict__, "Xu": 500.0, "Xuu": 500.0})   # damping that pumps energy in
    v2 = residuals_for_run(unstable, run, SPEC.vehicle, horizon_samples=100)
    assert v1.shape == v2.shape and np.all(np.isfinite(v2)) and np.abs(v2).max() > np.abs(v1).max()
    assert np.abs(v1).max() < 1e-9


def test_each_sensor_axis_weighs_the_same_whatever_its_rate():
    """A 1 Hz fix and a 10 Hz depth, both no better than standing still: each contributes about one unit."""
    df, nu, _ = truth_frame(pos_every=10, depth_every=1)
    run = as_run(df, sensors=("usbl", "main"), anchors=nu[:, :3])
    run.sensor_scales = ms.still_scales([run], 100)
    plan = ms.plan_for(run, 100)
    labels = ms.channel_labels(run.channels)
    for lab, s in zip(labels, plan.still):
        if lab.startswith(("usbl", "main")):
            share = float(np.sum((s / run.sensor_scales[lab] / np.sqrt(len(s))) ** 2))
            assert share == pytest.approx(1.0, rel=1e-9)


def test_a_window_needs_two_fixes():
    df, nu, _ = truth_frame(pos_every=150)
    run = as_run(df, sensors=("usbl",), anchors=nu[:, :3])
    plan = ms.plan_for(run, 100)   # 10 s windows, a fix every 15 s: never two in one window
    ci = [c.name for c in run.channels].index("usbl")
    assert all(ci not in pw for pw in plan.picks)


def test_lever_arm_of_a_position_sensor():
    df, nu, path = truth_frame()
    arm = np.array([0.4, -0.1, 0.3])
    R = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].values).as_matrix()
    moved = path + np.einsum("nij,j->ni", R, arm)
    df2 = df.copy()
    df2[["pos_usbl_n", "pos_usbl_e", "pos_usbl_d"]] = moved
    run = convert_run_dataframe(df2, "s-01-11-2026-run1.csv", DatasetConfig(
        sampling_rate_hz=RATE, sensors=("usbl",), lever_arms={"usbl": arm}))
    run.anchor_lin = nu[:, :3]
    r = ms.raw_channel_residuals(TRUTH, run, SPEC.vehicle, 100)
    assert max(np.abs(x).max() for x in r) < 1e-9
    run0 = as_run(df2, sensors=("usbl",), anchors=nu[:, :3])   # the same readings, the arm ignored
    assert max(np.abs(x).max() for x in ms.raw_channel_residuals(TRUTH, run0, SPEC.vehicle, 100)[3:]) > 1e-3


def test_velocity_view_reports_nan_without_a_dvl():
    df, nu, _ = truth_frame(dvl=False)
    run = as_run(df, anchors=nu[:, :3])
    run._horizon = 100
    r = rmse_by_dof(TRUTH, run, SPEC.vehicle, horizon_samples=100)
    assert np.isnan(r[:3]).all() and np.all(r[3:] < 1e-9)


def test_runs_with_a_dvl_at_every_sample_keep_the_original_objective():
    df, nu, _ = truth_frame()
    run = as_run(df, sensors=("auto",))
    assert run.channels is None
    v = joint_residuals(TRUTH.to_vector(), [run], SPEC.vehicle, horizon_samples=100)
    assert len(v) == 6 * (900 - 9)   # (N - windows) x 6: the original re-anchored velocity residuals


# ------------------------------------------------------------------ what the sensors see

def test_axes_seen_and_the_values_held():
    df, nu, _ = truth_frame(dvl=False)
    full = as_run(df, anchors=nu[:, :3])
    assert ms.axes_seen([full], 100) == {"u": ["position"], "v": ["position"], "w": ["depth", "position"]}
    depth_only = as_run(df, sensors=("main",), anchors=nu[:, :3])
    seen = ms.axes_seen([depth_only], 100)
    assert seen == {"u": [], "v": [], "w": ["depth"]}
    held = ms.unobservable_holds(seen, TRUTH)
    assert {"Xu", "Xuu", "Xu_dot", "Yv", "Yvv", "Yv_dot", "Xq_dot"} <= set(held)
    assert not {"Zw", "Zww", "Zw_dot", "Nr", "Ieff_yaw"} & set(held)
    gyro_only = as_run(df, sensors=("none",), anchors=nu[:, :3])
    assert [c.kind for c in gyro_only.channels] == ["gyro"]
    assert ms.axes_seen([gyro_only], 100) == {"u": [], "v": [], "w": []}
    words = ms.seen_words(seen)
    assert words[0].startswith("Surge: not calibrated") and words[2] == "Heave: calibrated from depth."


# ------------------------------------------------------------------ the smoother

def test_smoother_recovers_the_velocity_from_position_fixes():
    df, nu, _ = truth_frame(n=1500, dvl=False, pos_every=10, pos_noise=0.05)
    run = as_run(df, sensors=("usbl", "main"))
    res = sm.smooth(run, TRUTH, SPEC.vehicle, accel_sigma=0.03, noise={"usbl": 0.05, "main": 0.002})
    err = np.abs(res.velocity[50:-50] - nu[50:-50, :3])
    assert res.fallback is None
    assert float(np.sqrt(np.mean(err ** 2))) < 0.02     # 2 cm/s from 1 Hz fixes with 5 cm noise
    # the filter's uncertainty isn't overconfident (NIS well above 1); below 1 here, since the physics is exact and
    # any process noise is more than it needs
    assert 0.1 < res.nis_mean < 3.0


def test_smoother_with_the_true_physics_and_no_horizontal_fix_still_tracks():
    """Depth only: heave from the depth sensor; surge and sway from the physics alone, which is exact here."""
    df, nu, _ = truth_frame(n=900, dvl=False)
    run = as_run(df, sensors=("main",))
    res = sm.smooth(run, TRUTH, SPEC.vehicle, accel_sigma=0.01, noise={"main": 0.002})
    assert float(np.sqrt(np.mean((res.velocity[20:-20, 2] - nu[20:-20, 2]) ** 2))) < 0.01
    assert float(np.sqrt(np.mean((res.velocity[20:-20, :2] - nu[20:-20, :2]) ** 2))) < 0.02


def test_smoother_picks_a_process_noise_and_falls_back_safely():
    df, nu, _ = truth_frame(n=600, dvl=False, pos_every=5, pos_noise=0.02)
    run = as_run(df, sensors=("usbl",))
    sigma, scores = sm.choose_accel_sigma([run], TRUTH, SPEC.vehicle, {"usbl": 0.02})
    assert sigma in sm.ACCEL_SIGMA_GRID and all(np.isfinite(v) for v in scores.values())
    bad = TRUTH.__class__(**{**TRUTH.__dict__, "Xu": 5000.0, "Xuu": 5000.0})
    res = sm.smooth(run, bad, SPEC.vehicle, 0.1)
    assert res.fallback is not None and np.all(res.velocity == 0)


def test_anchors_use_the_dvl_where_it_read():
    df, nu, _ = truth_frame(n=600)
    lost = (np.arange(600) // 60) % 2 == 1
    df.loc[lost, ["Bu", "Bv", "Bw"]] = np.nan
    run = as_run(df)
    sm.set_anchors([run], TRUTH, SPEC.vehicle, 0.03, {"usbl": 0.01})
    assert np.allclose(run.anchor_lin[~lost], nu[~lost, :3])
    assert float(np.sqrt(np.mean((run.anchor_lin[lost] - nu[lost, :3]) ** 2))) < 0.01


# ------------------------------------------------------------------ the sensors' mounts and a DVL's track

def test_a_water_tracking_dvl_in_a_current():
    """A DVL locked on the water reads the velocity through the water: over ground minus the current. At the truth, in
    a current, its residuals are zero only when calibration knows it tracks the water."""
    from objective import run_with_current
    df, nu, _ = truth_frame(n=500)
    c_world = np.array([0.15, -0.05, 0.0])
    R = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].values).as_matrix()
    c_body = np.einsum("nji,j->ni", R, c_world)
    # the truth itself, re-simulated in the current (its velocity over ground changes)
    from dynamics_model import quaternion_to_roll_pitch
    q = df[["qx", "qy", "qz", "qw"]].values
    roll, pitch = quaternion_to_roll_pitch(q[:, 0], q[:, 1], q[:, 2], q[:, 3])
    nu_c = simulate_trajectory(TRUTH, SPEC.vehicle, SimulationConfig(dt=1 / RATE),
                               TrajectoryInputs(nu0=np.zeros(6), forces_torques=df[["Fx", "Fy", "Fz", "Tx", "Ty",
                                                                                     "Tz"]].values,
                                                roll_pitch_series=np.stack([roll, pitch], 1),
                                                current_body_series=c_body)).nu
    df[["Bp", "Bq", "Br"]] = nu_c[:, 3:6]
    df[["Bu", "Bv", "Bw"]] = nu_c[:, :3] - c_body          # what a water-tracking DVL reads
    mounts = ms.sensor_mounts({"sensors": {"dvl": {"track": "water"}}})
    run = convert_run_dataframe(df, "w-01-11-2026-run1.csv", DatasetConfig(
        sampling_rate_hz=RATE, sensors=("dvl",), lever_arms=mounts))
    run = run_with_current(run, c_world)
    sm.set_anchors([run], TRUTH, SPEC.vehicle, 0.01, {})
    assert np.allclose(run.anchor_lin, nu_c[:, :3])         # its readings turned back into over-ground starts
    r = ms.raw_channel_residuals(TRUTH, run, SPEC.vehicle, 100)
    assert max(np.abs(x).max() for x in r) < 1e-9
    run_b = convert_run_dataframe(df, "w-01-11-2026-run1.csv", DatasetConfig(sampling_rate_hz=RATE, sensors=("dvl",)))
    run_b = run_with_current(run_b, c_world)
    sm.set_anchors([run_b], TRUTH, SPEC.vehicle, 0.01, {})
    assert max(np.abs(x).max() for x in ms.raw_channel_residuals(TRUTH, run_b, SPEC.vehicle, 100)) > 0.05


def test_a_dvl_mounted_away_from_the_reference_point():
    df, nu, _ = truth_frame(n=500)
    arm = np.array([-0.3, 0.1, 0.2])
    df[["Bu", "Bv", "Bw"]] = nu[:, :3] + np.cross(nu[:, 3:6], arm)
    mounts = ms.sensor_mounts({"sensors": {"dvl": {"position_m": arm.tolist()}}})
    run = convert_run_dataframe(df, "a-01-11-2026-run1.csv", DatasetConfig(sampling_rate_hz=RATE, sensors=("dvl",),
                                                                          lever_arms=mounts))
    sm.set_anchors([run], TRUTH, SPEC.vehicle, 0.01, {})
    assert np.allclose(run.anchor_lin, nu[:, :3])
    assert max(np.abs(x).max() for x in ms.raw_channel_residuals(TRUTH, run, SPEC.vehicle, 100)) < 1e-9


@pytest.mark.parametrize("bad, message", [
    ({"usbl": {"position_m": [1, 2]}}, "three numbers"),
    ({"dvl": {"track": "seabed"}}, "bottom or water"),
    ({"usbl": {"track": "water"}}, "only a DVL"),
    ({"main": {"grade": "bar99"}}, "isn't a sensor grade. Known: bar02, bar30"),
    ({"usbl": {"grade": "ugps_g2"}}, "give range_m"),
    ({"main": {"noise": -1}}, "a positive number"),
    ({"main": {"noise": "low"}}, "a positive number"),
    ({"usbl": {"range_m": 0}}, "range_m"),
    ({"usbl": {"delay_s": -0.5}}, "a number of seconds \\(0 or more\\)"),
    ({"usbl": {"delay_s": "late"}}, "sensors.usbl.delay_s: how long after"),
])
def test_bad_sensor_mounts_are_refused(bad, message):
    with pytest.raises(ValueError, match=message):
        ms.sensor_mounts({"sensors": bad})


def test_every_sensor_grade_is_a_noise_with_its_source():
    """sensor_grades.yaml: each preset is a sensor whose noise the smoother uses, one reading's noise (or a share of
    the range) that is plausible for its kind, and the datasheet it came from."""
    plausible = {"dvl": (1e-4, 0.05), "depth": (1e-5, 0.1), "position": (1e-3, 10.0)}
    for name, g in ms.sensor_grades().items():
        assert g["kind"] in ms.NOISE_KINDS, name
        assert ("noise" in g) != ("noise_fraction_of_range" in g), name
        if "noise" in g:
            lo, hi = plausible[g["kind"]]
            assert lo <= g["noise"] <= hi, name
        assert len(g["source"]) > 20, name


def test_the_smoother_noise_comes_from_your_number_then_the_logs_then_the_grade(tmp_path):
    import json
    df, _, _ = truth_frame(n=200, dvl=True, pos_every=10)
    run = as_run(df)
    mounts = ms.sensor_mounts({"sensors": {"main": {"grade": "bar30"}, "usbl": {"grade": "ugps_g2", "range_m": 40},
                                           "dvl": {"grade": "nucleus1000", "noise": 0.02}}})
    noise, src = ms.smoother_noise(None, mounts, [run])
    assert noise == pytest.approx({"main": 0.002, "usbl": 0.08, "dvl": 0.02})
    assert src == {"main": "grade bar30", "usbl": "grade ugps_g2", "dvl": "your description"}
    # what ingestion measured on unsmoothed readings wins over a grade; a smoothed sensor's measurement doesn't count
    (tmp_path / "ingest_report.json").write_text(json.dumps({"sensors": {"sensors": {
        "main": {"noise_std": [0.004]}, "usbl": {"noise_std": [0.3], "filtered": True}}}}), encoding="utf-8")
    noise, src = ms.smoother_noise(str(tmp_path), mounts, [run])
    assert noise["main"] == 0.004 and src["main"] == "measured from your logs"
    assert noise["usbl"] == pytest.approx(0.08) and src["usbl"] == "grade ugps_g2"
    # nothing given: the smoother's default for the kind, and said so
    noise, src = ms.smoother_noise(None, {}, [run])
    assert noise == {} and src == {"dvl": "default", "usbl": "default", "main": "default"}


def test_a_grade_of_the_wrong_kind_is_refused():
    df, _, _ = truth_frame(n=200, dvl=False, pos_every=10)
    with pytest.raises(ValueError, match="bar30 is a depth sensor, but usbl in your logs is a position sensor"):
        ms.smoother_noise(None, ms.sensor_mounts({"sensors": {"usbl": {"grade": "bar30"}}}), [as_run(df)])


def test_the_noise_the_smoother_assumes_matters_both_ways():
    """Fixes with 5 cm noise. Without physics the fixes are all there is, so discounting them (a 1.27 m GNSS grade)
    costs most; with Strathe's physics, trusting them too much (an 8.5 mm RTK datasheet) costs most. Why measured
    noise comes before a datasheet grade (SENSORS.md, "Noise")."""
    df, nu, _ = truth_frame(n=1500, dvl=False, pos_every=10, pos_noise=0.05)

    def err(physics, fix_noise):
        run = as_run(df, sensors=("usbl", "main"))
        noise = {"usbl": fix_noise, "main": 0.002}
        veh = SPEC.vehicle if physics else None
        sigma, _ = sm.choose_accel_sigma([run], TRUTH if physics else None, veh, noise)
        sm.set_anchors([run], TRUTH if physics else None, veh, sigma, noise)
        return np.sqrt(np.mean((run.anchor_lin[50:-50] - nu[50:-50, :3]) ** 2))
    assert err(False, 1.27) > 2 * err(False, 0.05)     # measured: 0.21 against 0.080 m/s
    assert err(True, 0.0085) > 3 * err(True, 0.05)     # measured: 0.052 against 0.0084 m/s


def test_the_smoother_is_the_exact_least_squares_answer_on_a_linear_system():
    """With no physics and a fixed orientation, the model is linear and Gaussian: the forward filter and the
    Rauch-Tung-Striebel pass must give exactly the batch least-squares answer over every state at once (prior, each
    step's process noise and each reading, each weighted by its own variance)."""
    n, dt, every, sig_v, noise = 120, 1.0 / RATE, 5, 0.05, 0.03
    rng = np.random.default_rng(7)
    v = np.cumsum(rng.normal(0, sig_v * np.sqrt(dt), (n, 3)), 0)
    p = np.vstack([np.zeros(3), np.cumsum(v[:-1] * dt, 0)]) + np.array([1.0, 2.0, 3.0])
    fixes = np.full((n, 3), np.nan)
    fixes[::every] = p[::every] + noise * rng.normal(size=(len(p[::every]), 3))
    df = pd.DataFrame(np.zeros((n, 6)), columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
    df[["Bp", "Bq", "Br"]] = 0.0
    df[["qx", "qy", "qz", "qw"]] = [0.0, 0.0, 0.0, 1.0]
    df["pos_usbl_n"], df["pos_usbl_e"], df["pos_usbl_d"] = fixes[:, 0], fixes[:, 1], fixes[:, 2]
    res = sm.smooth(as_run(df, sensors=("usbl",)), None, None, accel_sigma=sig_v, noise={"usbl": noise})
    assert res.fallback is None

    # the batch answer: unknowns x_k = [v_k, p_k] for every k, each equation divided by its standard deviation
    d = 6
    F = np.eye(d)
    F[3:, :3] = dt * np.eye(3)
    q = np.r_[[sig_v ** 2 * dt] * 3, [sm.POSITION_PROCESS_STD ** 2] * 3]
    p0 = np.r_[[sm.INITIAL_VELOCITY_STD ** 2] * 3, [sm.UNKNOWN_POSITION_VAR] * 3]
    s_fix = max(noise, sm.NOISE_FLOOR["position"])
    rows, rhs = [], []

    def eq(coefs, value, std):
        r = np.zeros(n * d)
        for k, c in coefs:
            r[k * d:(k + 1) * d] += c
        rows.append(r / std)
        rhs.append(value / std)
    for j in range(d):                                   # the prior on the first state (zero, as the filter starts)
        e = np.zeros(d)
        e[j] = 1.0
        eq([(0, e)], 0.0, np.sqrt(p0[j]))
    for k in range(n - 1):                               # x_{k+1} - F x_k = process noise
        for j in range(d):
            e = np.zeros(d)
            e[j] = 1.0
            eq([(k + 1, e), (k, -F[j])], 0.0, np.sqrt(q[j]))
    for k in np.flatnonzero(np.isfinite(fixes[:, 0])):   # each fix reads the position
        for j in range(3):
            e = np.zeros(d)
            e[3 + j] = 1.0
            eq([(k, e)], fixes[k, j], s_fix)
    x = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0].reshape(n, d)
    assert np.max(np.abs(res.velocity - x[:, :3])) < 1e-6


def test_late_fixes_are_found_and_on_time_ones_are_not():
    """state_smoother.fix_delays (#96): fixes reported 1 s after the vehicle was there are found late, and said;
    the same fixes on time aren't."""
    df, _, _ = truth_frame(n=900, dvl=False, pos_every=10, pos_noise=0.05)
    late = df.copy()
    cols = ["pos_usbl_n", "pos_usbl_e", "pos_usbl_d"]
    late[cols] = df[cols].shift(10)      # each fix logged 10 samples (1 s) after the vehicle was there
    found = {}
    for tag, frame in (("on time", df), ("late", late)):
        run = as_run(frame, sensors=("usbl", "main"))
        found[tag] = sm.fix_delays([run], TRUTH, SPEC.vehicle, 0.03, {"usbl": 0.05, "main": 0.002})["usbl"]
    assert not found["on time"]["late"] and found["on time"]["delay_s"] <= 0.2
    assert found["late"]["late"] and 0.8 <= found["late"]["delay_s"] <= 1.2


def test_a_delay_taken_off_once_stays_taken_off():
    """state_smoother.shift_fixes (#96): the fixes move earlier by the delay, the windows are rebuilt from them, and
    shifting the same runs to the same delay again (a calibration reloaded) changes nothing."""
    df, _, _ = truth_frame(n=300, dvl=False, pos_every=10, pos_noise=0.0)
    run = as_run(df, sensors=("usbl", "main"))
    before = next(c for c in run.channels if c.name == "usbl").values.copy()
    ms.plan_for(run, 50)
    sm.shift_fixes([run], "usbl", 1.0)
    after = next(c for c in run.channels if c.name == "usbl").values
    assert "_plans" not in run.__dict__
    np.testing.assert_array_equal(after[:-10], before[10:])
    assert np.isnan(after[-10:]).all()
    sm.shift_fixes([run], "usbl", 1.0)
    np.testing.assert_array_equal(next(c for c in run.channels if c.name == "usbl").values, after)
    sm.shift_fixes([run], "main", 1.0)   # a depth sensor isn't shifted: only position fixes are scanned
    np.testing.assert_array_equal(next(c for c in run.channels if c.name == "main").values[:, 0],
                                  df["depth_main"].to_numpy())
    # and a calibration's delay, carried by the dataset config, gives the same fixes from the file
    again = convert_run_dataframe(df, "synthetic-01-11-2026-run1.csv",
                                  DatasetConfig(sampling_rate_hz=RATE, sensors=("usbl", "main"),
                                                fix_delays=(("usbl", 1.0),)))
    np.testing.assert_array_equal(next(c for c in again.channels if c.name == "usbl").values, after)


def test_a_delay_given_in_both_places_is_refused(tmp_path):
    import json
    (tmp_path / "ingest_report.json").write_text(json.dumps(
        {"sensors": {"sensors": {"usbl": {"kind": "position", "delay_s": 0.3}, "main": {"kind": "depth"}}}}))
    assert ms.given_delays(str(tmp_path), {}) == {"usbl": (0.0, "the ingest spec's delay_s (0.3 s, already taken off)")}
    assert ms.given_delays(None, {"usbl": {"delay_s": 0.0}})["usbl"][0] == 0.0
    with pytest.raises(ValueError, match="given in both the ingest spec and the vehicle description: give it once"):
        ms.given_delays(str(tmp_path), {"usbl": {"delay_s": 0.3}})
