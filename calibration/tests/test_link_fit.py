"""
Link calibration from message logs (link_fit.py, `strathe fit-link`, PROJECT_STATUS.md #68). The logs are drawn from
Strathe's own channel models with known settings (link_log_synth.py): the fit must recover them, say when it can't,
and its result must reach the fleet simulation, training, the CLI and the web page.
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import link_fit as F
from comm_channel import BandwidthConfig, CommChannel, PacketLossConfig
from link_log_synth import acoustic_log, optical_log

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
ROOT = os.path.dirname(CAL)


def _fit(tmp_path, *frames, **kw):
    p = tmp_path / "log.csv"
    pd.concat(frames).to_csv(p, index=False)
    return F.fit_logs(F.load_logs([str(p)]), **kw)


def test_acoustic_loss_ramp_and_latency_are_recovered(tmp_path):
    r = _fit(tmp_path, acoustic_log())["acoustic"]
    pl = r["packet_loss"]
    assert r["model"] == "ramp"
    assert pl["near_field_prob"] == pytest.approx(0.05, abs=0.03) and pl["far_field_prob"] == pytest.approx(0.6, abs=0.06)
    assert pl["near_distance_m"] == pytest.approx(50, abs=30) and pl["far_distance_m"] == pytest.approx(400, abs=40)
    assert r["speed_of_sound_mps"] == pytest.approx(1480, abs=25) and r["speed_fitted"]
    assert r["fixed_latency_s"] == pytest.approx(2.5, abs=0.05)


def test_a_loss_that_doesnt_depend_on_range_is_fitted_as_a_constant(tmp_path):
    r = _fit(tmp_path, acoustic_log(plc=PacketLossConfig(0, 1000, 0.3, 0.3)))["acoustic"]
    assert r["model"] == "constant"
    assert r["packet_loss"]["near_field_prob"] == pytest.approx(0.3, abs=0.04)
    assert r["packet_loss"]["near_field_prob"] == r["packet_loss"]["far_field_prob"]


def test_too_narrow_ranges_hold_the_speed_of_sound(tmp_path):
    res = _fit(tmp_path, acoustic_log(ranges=(40.0, 90.0)))
    r = res["acoustic"]
    assert r["speed_of_sound_mps"] == 1500.0 and not r["speed_fitted"]
    assert r["fixed_latency_s"] == pytest.approx(2.5, abs=0.05)
    assert any("held at 1500 m/s" in w for w in res["warnings"])


def test_unsynchronized_clocks_are_flagged(tmp_path):
    a = acoustic_log(n=200)
    a["receive_time_s"] = a["receive_time_s"] - 5.0  # receiver's clock 5 s behind
    res = _fit(tmp_path, a)
    assert any("clocks aren't synchronized" in w for w in res["warnings"])


def test_optical_delivery_is_recovered_where_messages_were_logged(tmp_path):
    o, cfg = optical_log()
    res = _fit(tmp_path, o)
    r = res["optical"]
    assert r["range_at_50pct_m"] == pytest.approx(cfg.range_at(0.5), rel=0.05)
    assert r["latency_s"] == pytest.approx(0.002, abs=1e-6)
    # the water is held (coastal by default, the truth here), so the modem's settings come back too
    assert r["attenuation_per_m"] == 0.4 and r["rated_range_m"] == pytest.approx(30.0, rel=0.1)
    assert r["curve_width"] == pytest.approx(1.0, rel=0.2)
    assert any("clarity is held at coastal" in w for w in res["warnings"])


def test_the_water_is_held_not_fitted_and_doesnt_change_the_logged_range(tmp_path):
    """Logged messages can't measure the water's clarity: held at the wrong water, the fit still reproduces delivery
    where messages were logged; only the extrapolation differs."""
    o, cfg = optical_log(water="harbor", rated=60.0, ranges=(0.5, 12.0), n=3000)
    right = _fit(tmp_path, o, optical_water="harbor")["optical"]
    wrong = _fit(tmp_path, o, optical_water="clear")["optical"]
    measured = _fit(tmp_path, o, optical_attenuation=2.2)["optical"]
    assert measured["attenuation_held"] == "measured (2.2/m)" and measured["rated_range_m"] == right["rated_range_m"]
    for r in (right, wrong):
        assert r["range_at_50pct_m"] == pytest.approx(cfg.range_at(0.5), rel=0.05)
        for b in r["bins"]:
            assert b["fitted"] == pytest.approx(b["delivered"], abs=0.08)


def test_messages_sent_unpointed_are_left_out(tmp_path):
    o, _ = optical_log(n=300)
    o["pointed"] = 1
    o.loc[o.index[:50], "pointed"] = 0
    res = _fit(tmp_path, o)
    assert res["optical"]["messages"] == 250 and any("50 message(s)" in w for w in res["warnings"])


@pytest.mark.parametrize("csv,match", [
    ("send_time_s,receive_time_s\n0,1\n", "missing column"),
    ("send_time_s,range_m\n0,10\n", "receive_time_s .* or delivered"),
    ("send_time_s,range_m,delivered\n0,10,2\n", "delivered must be 1 or 0"),
    ("send_time_s,range_m,delivered,link\n0,10,1,radio\n", "unknown link"),
])
def test_log_errors_say_what_to_fix(tmp_path, csv, match):
    (tmp_path / "bad.csv").write_text(csv)
    with pytest.raises(ValueError, match=match):
        F.load_logs([str(tmp_path / "bad.csv")])


def test_the_template_loads():
    d = F.load_logs([os.path.join(CAL, "templates", "link_log_template.csv")])
    assert len(d) == 5 and d["delivered"].tolist() == [1, 0, 1, 1, 0]
    assert ((d["receive_time_s"] - d["send_time_s"]).dropna() >= 0).all()


def test_the_acoustic_channel_adds_the_modem_delay():
    ch = CommChannel(PacketLossConfig(0, 1000, 0, 0), BandwidthConfig(1e9), speed_of_sound_mps=1500,
                     fixed_latency_s=2.0, seed=1)
    m = ch.send("a", "b", 10.0, (0, 0, 0), (300, 0, 0), "position_beacon")
    assert m.arrival_time == pytest.approx(10.0 + 2.0 + 0.2)
    assert CommChannel(PacketLossConfig(0, 1000, 0, 0), BandwidthConfig(1e9), seed=1).send(
        "a", "b", 10.0, (0, 0, 0), (300, 0, 0), "position_beacon").arrival_time == pytest.approx(10.2)


@pytest.fixture
def cal_file(tmp_path):
    o, _ = optical_log(n=800)
    res = _fit(tmp_path, acoustic_log(n=800), o, optical_water="coastal")
    p = tmp_path / "link_calibration.yaml"
    p.write_text(yaml.safe_dump(res, sort_keys=False))
    return str(p), res


def test_an_experiment_uses_the_measured_link(cal_file):
    from fleet_experiment_schema import CommunicationConfig
    path, res = cal_file
    c = CommunicationConfig(link=["acoustic", "hybrid"], link_calibration=path, beacon_interval_s=1.0)
    (label, plc, _bw), = c.levels()
    assert label == "measured" and c.acoustic_calibrated()
    assert plc.far_field_prob == res["acoustic"]["packet_loss"]["far_field_prob"]
    assert c.speed_of_sound_mps == res["acoustic"]["speed_of_sound_mps"]
    assert c.acoustic_fixed_latency_s == res["acoustic"]["fixed_latency_s"]
    hybrid = [lc for lc, _ in c.link_configs() if lc.link == "hybrid"][0]
    assert hybrid.optical.water == "custom" and hybrid.optical.attenuation_per_m == 0.4
    assert "Calibrated from your link logs (link_calibration.yaml)" in hybrid.optical.describe()
    # reloading the saved, resolved config gives the same thing
    again = CommunicationConfig(**c.model_dump())
    assert again.levels()[0][0] == "measured" and again.acoustic_fixed_latency_s == c.acoustic_fixed_latency_s
    with pytest.raises(ValueError, match="at most one severity"):
        CommunicationConfig(severities=["L1", "L2"], link_calibration=path)


def test_fleet_sim_and_training_get_the_modem_delay(cal_file):
    import train_fleet_policy as T
    from link_fit import load_link_calibration
    path, res = cal_file
    conds = T.comm_conditions(intervals=[1.0, 3.0], acoustic=load_link_calibration(path)["acoustic"])
    assert [c.label for c in conds] == ["measured@1s", "measured@3s"]
    assert all(c.acoustic_fixed_latency_s == res["acoustic"]["fixed_latency_s"] for c in conds)
    from fleet_env import FleetSim
    from optimizer import initial_parameters, load_vehicle_spec
    spec = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml"))
    sim = FleetSim(initial_parameters(spec), spec.vehicle, acoustic_fixed_latency_s=2.0)
    sim.reset(comm_seed=1, initial_separation=3.0, closing_speed=0.0)
    assert sim.channel.fixed_latency_s == 2.0


def test_strathe_fit_link_then_fleet_and_train_fleet_flags(tmp_path):
    import strathe_cli as cli
    o, _ = optical_log(n=600)
    pd.concat([acoustic_log(n=600), o]).to_csv(tmp_path / "msgs.csv", index=False)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "strathe.py"), "fit-link", "--log", str(tmp_path / "msgs.csv"),
                        "--out", str(tmp_path / "lc")], capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env={**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "acoustic: 600 messages" in r.stdout and "optical: 600 messages" in r.stdout
    for f in ("link_calibration.yaml", "link_calibration_report.md"):
        assert (tmp_path / "lc" / f).exists()
    report = (tmp_path / "lc" / "link_calibration_report.md").read_text(encoding="utf-8")
    assert "## Acoustic" in report and "## Optical" in report
    lc = str(tmp_path / "lc" / "link_calibration.yaml")
    args = cli.build_parser().parse_args(["fleet", "--link", "hybrid", "--link-calibration", lc])
    assert cli.fleet_setting_overrides(args)["communication"]["link_calibration"] == lc
    calls = []
    cli.main(["train-fleet", "--params", os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml"), "--out",
              str(tmp_path / "p"), "--link-calibration", lc], runner=lambda c: calls.append(c) or 0)
    t = calls[0]
    assert t[t.index("--link-calibration") + 1] == lc


def test_web_link_card_and_field(tmp_path):
    import strathe_web as web
    form = {"link": "hybrid", "link_calibration": str(tmp_path / "nope.yaml")}
    with pytest.raises(ValueError, match="Link calibration not found"):
        web.link_cli_args(form)
    (tmp_path / "lc.yaml").write_text("x: 1\n")
    assert "--link-calibration" in web.link_cli_args({"link": "acoustic", "link_calibration": str(tmp_path / "lc.yaml")})
