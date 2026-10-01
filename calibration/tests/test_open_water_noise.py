"""
The open_water sensor-noise preset (measured from SINTEF's SOLAQUA data) and
the starting-fix error as its own setting: the preset and its provenance, the
fit's estimators, the starting fix in the simulator, and its wiring through
the experiment YAML, the command line and the web form. PROJECT_STATUS.md #60.
"""
import glob
import os

import numpy as np
import pytest
import yaml

import strathe_cli as cli
from sensor_noise import PRESETS, SensorNoiseConfig, SensorSuite, load_sensor_noise, resolve_sensor_noise

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
SOLAQUA = os.path.join(os.path.dirname(CAL), "data", "raw", "solaqua")
BIASES = ("velocity_bias_mps", "gyro_bias_rps", "attitude_bias_rad", "heading_bias_rad")


# ----------------------------------------------------------------- the preset

def test_open_water_preset_is_measured_with_its_assumed_biases_named():
    ow = load_sensor_noise("open_water")
    typ = load_sensor_noise("typical")
    assert "open_water" in PRESETS and ow.provenance == "measured"
    assert ow.assumed_fields == list(BIASES)
    for b in BIASES:                                  # the biases this data can't measure are typical's
        assert getattr(ow, b) == getattr(typ, b)
    d = ow.describe()
    assert "measured from logs" in d and "ASSUMED" in d and "SOLAQUA" in d
    assert ow.position == "dead_reckoning" and 0.05 < ow.position_std_m < 0.5
    # measured white noise is below the assumed sizes (PROJECT_STATUS.md #60)
    for k in ("velocity_std_mps", "gyro_std_rps", "attitude_std_rad", "heading_std_rad", "depth_std_m"):
        assert 0 < getattr(ow, k) < getattr(typ, k), k
    raw = yaml.safe_load(open(PRESETS["open_water"], encoding="utf-8"))
    assert raw["source_data"]["license"] == "CC BY-SA 4.0" and "SINTEF" in raw["source_data"]["attribution"]
    assert raw["fit"]["recordings"] >= 50
    header = open(PRESETS["open_water"], encoding="utf-8").read(800)
    assert "CC BY-SA 4.0" in header and "ASSUMED" in header


def test_typical_and_tank_descriptions_unchanged_in_meaning():
    assert "ASSUMED" in load_sensor_noise("typical").describe()
    assert load_sensor_noise("tank").assumed_fields == []


# ----------------------------------------------------------------- the starting fix

def test_start_fix_overrides_only_the_starting_fix_and_keeps_the_random_stream():
    typ = load_sensor_noise("typical")
    assert typ.start_fix == pytest.approx(typ.position_std_m)
    assert load_sensor_noise("typical", 2.0).start_fix == pytest.approx(2 * typ.position_std_m)
    zero = typ.with_start_fix(0.0)
    assert zero.start_fix == 0.0 and zero.position_std_m == typ.position_std_m and "set on its own" in zero.describe()
    assert typ.with_start_fix(None) is typ
    pos, R, nu = np.array([1.0, 2.0, 3.0]), np.eye(3), np.zeros(6)
    a, b = SensorSuite(typ, 0.1, 7), SensorSuite(zero, 0.1, 7)
    ma, mb = a.start(pos, R, nu), b.start(pos, R, nu)
    np.testing.assert_allclose(mb.position[:2], pos[:2])                 # perfect fix: starts exactly right
    assert np.linalg.norm(ma.position[:2] - pos[:2]) > 0
    np.testing.assert_allclose(a.bias_v, b.bias_v)                        # the same draws, in the same order
    assert a._rng.random() == b._rng.random()
    # the error is the configured size: per-axis std over many vehicles
    errs = np.array([SensorSuite(typ.with_start_fix(0.3), 0.1, s).start(pos, R, nu).position[:2] - pos[:2]
                     for s in range(4000)])
    assert errs.std(axis=0) == pytest.approx([0.3, 0.3], rel=0.05)


def test_start_fix_validation():
    with pytest.raises(ValueError, match="needs sensor noise on"):
        resolve_sensor_noise(None, start_fix=0.1)
    with pytest.raises(ValueError, match="between 0 and 20"):
        load_sensor_noise("typical").with_start_fix(25.0)
    with pytest.raises(ValueError, match="only applies to dead reckoning"):
        load_sensor_noise("tank").with_start_fix(0.1)                     # tank positioning is external
    only_fix = SensorNoiseConfig(position="dead_reckoning", start_fix_m=0.2)
    assert only_fix.active                                                # a fix error alone is still noise


# ----------------------------------------------------------------- the fit's estimators

def test_usbl_error_estimator_recovers_a_known_error():
    from fit_solaqua_noise import usbl_fix_error
    rng = np.random.default_rng(1)
    for sigma in (0.05, 0.2):
        ests = []
        for _ in range(20):
            t = np.arange(0, 90, 0.11)
            x = np.cumsum(0.2 + 0.05 * np.sin(t / 7)) * 0.11
            y = np.cumsum(0.05 * np.cos(t / 5)) * 0.11
            th, drift = rng.uniform(0, 2 * np.pi), rng.normal(0, 0.1, 2)
            tu = np.arange(0.5, 89, 2.0)
            xu, yu = np.interp(tu, t, x), np.interp(tu, t, y)
            E = np.cos(th) * xu - np.sin(th) * yu + drift[0] * tu + rng.normal(0, sigma, len(tu))
            N = np.sin(th) * xu + np.cos(th) * yu + drift[1] * tu + rng.normal(0, sigma, len(tu))
            ests.append(usbl_fix_error(np.column_stack([tu, E, N, np.full_like(tu, 0.1)]),
                                       np.column_stack([t, x, y]))[0])
        assert np.median(ests) == pytest.approx(sigma, rel=0.1)


def test_dvl_dropout_counts_invalid_readings_and_long_gaps():
    from fit_solaqua_noise import dvl_dropout
    t = np.arange(0, 100, 0.1)
    valid = np.ones_like(t)
    valid[::20] = 0                                                       # 5% invalid
    keep = ~((t > 50) & (t < 55))                                         # a 5 s gap = 5% of the time
    dv = np.column_stack([t, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t), valid, np.ones_like(t)])[keep]
    frac, rate = dvl_dropout(dv)
    assert rate == pytest.approx(10, rel=0.01) and frac == pytest.approx(0.10, abs=0.01)


@pytest.mark.skipif(not glob.glob(os.path.join(SOLAQUA, "*_data.bag")),
                    reason="SOLAQUA data bags not downloaded (see fit_solaqua_noise.py)")
def test_fit_on_real_solaqua_bags_matches_the_shipped_preset():
    from fit_solaqua_noise import fit_run, summarise
    bags = sorted(glob.glob(os.path.join(SOLAQUA, "2024-08-20_16-*_data.bag")))[:4]
    s = summarise([fit_run(b) for b in bags])["sensors"]
    ow = load_sensor_noise("open_water")
    for k in ("velocity_std_mps", "gyro_std_rps", "attitude_std_rad", "depth_std_m"):
        assert s[k] == pytest.approx(getattr(ow, k), rel=0.6), k            # a 4-run subset, same order of size


# ----------------------------------------------------------------- wiring

def test_cli_open_water_and_start_fix(tmp_path):
    dest = tmp_path / "x.yaml"
    assert cli.main(["fleet", "--noise", "open_water", "--start-fix-error", "0.05", "--save-config", str(dest)]) == 0
    assert yaml.safe_load(dest.read_text())["sensor_noise"] == {"preset": "open_water", "scale": 1.0,
                                                                "start_fix_error_m": 0.05}
    assert cli.main(["fleet", "--mode", "noisy", "--start-fix-error", "0", "--save-config", str(dest)]) == 0
    from fleet_experiment_schema import load_fleet_experiment
    cwd = os.getcwd()
    try:
        os.chdir(CAL)
        exp = load_fleet_experiment(str(dest))
    finally:
        os.chdir(cwd)
    assert exp.sensor_noise.preset == "typical" and exp.sensor_noise.config().start_fix == 0.0
    assert cli.main(["fleet", "--start-fix-error", "0.1", "--save-config", str(tmp_path / "y.yaml")]) == 2  # clean
    assert cli.main(["fleet", "--noise", "typical", "--start-fix-error", "30",
                     "--save-config", str(tmp_path / "z.yaml")]) == 2
    calls = []
    run = lambda argv: calls.append(argv) or 0
    assert cli.main(["arena", "--mode", "noisy", "--noise", "open_water", "--start-fix-error", "0.1",
                     "--out", str(tmp_path / "a")], runner=run) == 0
    a = calls[-1]
    assert a[a.index("--noise") + 1] == "open_water" and a[a.index("--start-fix-error") + 1] == "0.1"
    import run_arena_experiment as rae
    rae.build_parser().parse_args(a[2:])
    params = os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml")
    assert cli.main(["train-fleet", "--params", params, "--mode", "noisy", "--start-fix-error", "0.05",
                     "--out", str(tmp_path / "p")], runner=run) == 0
    assert "--start-fix-error" in calls[-1]


def test_web_form_offers_open_water_and_the_start_fix():
    import strathe_web as w
    html = w.noise_fields("x")
    assert "value=open_water" in html and "name=start_fix_error" in html and "SOLAQUA" in html
    args = w.noise_cli_args({"mode": "noisy", "noise": "open_water", "noise_scale": "1", "start_fix_error": "0.05"})
    assert args[args.index("--noise") + 1] == "open_water" and args[args.index("--start-fix-error") + 1] == "0.05"
    with pytest.raises(ValueError, match="needs sensor noise on"):
        w.noise_cli_args({"start_fix_error": "0.1"})


def test_fleet_report_names_the_starting_fix(tmp_path):
    import run_fleet_experiment as rfe
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump({
        "schema_version": 2, "name": "ow", "policies": [{"model_path": os.path.join(CAL, "fleet_policy_seed1")}],
        "truth_params": os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml"),
        "vehicle_spec": os.path.join(CAL, "vehicle_spec.yaml"), "beliefs": ["calibrated", "last_beacon"],
        "compare": {"a": "calibrated", "b": "last_beacon"}, "communication": {"severity": "L2"},
        "mode": "noisy", "sensor_noise": {"preset": "open_water", "start_fix_error_m": 0.0},
        "initial_conditions": {"separations_m": [2.5], "closing_speeds_mps": [0.35]},
        "evaluation": {"comm_seeds": [1], "workers": 1}, "output_dir": str(tmp_path / "out")}))
    rfe.run_experiment(str(p))
    rep = (tmp_path / "out" / "summary_report.md").read_text(encoding="utf-8")
    assert "starting position fix" in rep and "±0 m per axis" in rep and "ASSUMED" in rep
