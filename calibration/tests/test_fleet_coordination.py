"""N-vehicle coordination over the shared acoustic channel, and `strathe comms-study` (PROJECT_STATUS.md #89).
Episodes are kept short (60-120 s) so this stays cheap in CI."""
import os
import sys

import numpy as np
import pytest
import yaml

import comms_study as CS
import strathe_cli as cli
from acoustic_network import LossModel, ModemConfig, NetworkConfig
from build_two_agent_cores import load_calibrated_parameters
from fleet_belief import Beacon, BeliefSpec
from fleet_coordination import CoordinationConfig, CoordinationSim, FormationSpec, LeaderPlan, load_controller
from optimizer import load_vehicle_spec

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
TRUTH = load_calibrated_parameters(os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml"), quiet=True)
VEHICLE = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml")).vehicle


def _cfg(**kw):
    net = kw.pop("network", NetworkConfig(modem=ModemConfig.preset("medium")))
    return CoordinationConfig(duration_s=kw.pop("duration_s", 120.0), settle_s=20.0, leg_s=40.0, network=net, **kw)


def test_formation_slots():
    assert FormationSpec("line", 10).slots(4)[1:, :2].tolist() == [[0, 10], [0, -10], [0, 20]]
    assert FormationSpec("column", 5).slots(3)[1:, :2].tolist() == [[-5, 0], [-10, 0]]
    assert FormationSpec("wedge", 10).slots(3)[1:, :2].tolist() == [[-10, 10], [-10, -10]]
    assert FormationSpec("custom", 10, [[-3, 4]]).slots(2)[1, :2].tolist() == [-3, 4]
    with pytest.raises(ValueError, match="needs 2 offsets"):
        FormationSpec("custom", 10, [[-3, 4]]).slots(3)


def test_settings_that_cant_fly_are_refused_with_the_reason():
    with pytest.raises(ValueError, match="2 to 12 vehicles"):
        _cfg(n_vehicles=13).validate()
    with pytest.raises(ValueError, match="slow the turn"):   # a wedge turning with the leader at 3 deg/s
        _cfg(formation=FormationSpec("wedge", 10, orientation="leader")).validate()
    _cfg(formation=FormationSpec("wedge", 10, orientation="leader"), turn_rate_dps=0.1).validate()


def test_a_good_channel_holds_the_formation_and_a_saturated_one_doesnt():
    good = CoordinationSim(TRUTH, VEHICLE, _cfg()).run_episode(1)
    assert good["success"] and good["formation_error_rms_m"] < 1.5 and good["delivery_ratio"] == 1.0
    jammed = _cfg(network=NetworkConfig(modem=ModemConfig.preset("slow"), mac="aloha", beacon_interval_s=2.0))
    bad = CoordinationSim(TRUTH, VEHICLE, jammed).run_episode(1)
    assert not bad["success"] and bad["n_collision"] > 0 and bad["formation_error_rms_m"] > 3 * good[
        "formation_error_rms_m"]


def test_episodes_repeat_exactly_and_the_leader_is_the_same_across_settings():
    a = CoordinationSim(TRUTH, VEHICLE, _cfg(duration_s=60.0)).run_episode(7)
    b = CoordinationSim(TRUTH, VEHICLE, _cfg(duration_s=60.0)).run_episode(7)
    assert a == b
    p1 = LeaderPlan(_cfg(), np.random.default_rng([7, 0xF1EE7]))
    p2 = LeaderPlan(_cfg(network=NetworkConfig(loss=LossModel(0.5))), np.random.default_rng([7, 0xF1EE7]))
    assert p1.turns == p2.turns


def test_more_vehicles_on_one_channel():
    r = CoordinationSim(TRUTH, VEHICLE, _cfg(n_vehicles=6, duration_s=60.0)).run_episode(2)
    assert r["n_sent"] > 0 and r["tdma_cycle_s"] == pytest.approx(6 * (r["airtime_s"] + 0.05 +
                                                                        CoordinationConfig(n_vehicles=6).max_distance_m() / 1500), rel=1e-6)


def test_every_belief_gives_position_velocity_and_heading():
    b = Beacon(send_tick=1, position=np.array([1.0, 2, 3]), orientation=np.eye(3), nu=np.array([0.5, 0, 0, 0, 0, 0]),
               tau=np.zeros(6))
    for kind in ("last_beacon", "constant_velocity", "physics", "oracle"):
        bel = BeliefSpec(kind, kind, TRUTH if kind == "physics" else None).build(0.1, VEHICLE)
        assert bel.estimate_state((np.zeros(3), np.zeros(3), np.eye(3))) is None
        bel.on_tick(1, [b])
        bel.on_tick(2, [])
        p, v, R = bel.estimate_state((np.zeros(3), np.zeros(3), np.eye(3)))
        assert p.shape == v.shape == (3,) and R.shape == (3, 3)


def test_a_teams_own_controller(tmp_path, monkeypatch):
    (tmp_path / "my_ctrl.py").write_text("import numpy as np\nclass Idle:\n    def act(self, i, own, teammates, t):\n"
                                         "        return np.array([0, 0 if i == 0 else 20.0, 0, 0, 0, 0])\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    r = CoordinationSim(TRUTH, VEHICLE, _cfg(controller="my_ctrl:Idle", duration_s=60.0)).run_episode(1)
    assert not r["success"]
    with pytest.raises(ValueError, match="can't load controller"):
        load_controller("my_ctrl:Nope", _cfg(), None)
    with pytest.raises(ValueError, match="module:Class"):
        load_controller("my_ctrl", _cfg(), None)


# --- the study ---------------------------------------------------------------------------------------------------------
def test_each_setting_changes_only_itself():
    base = CS.base_config(CS.build_parser().parse_args([]))
    assert CS.apply(base, "latency", 7).network.modem.processing_latency_s == 7
    assert CS.apply(base, "bit_rate", 80).network.modem.bit_rate_bps == 80
    assert CS.apply(base, "loss", 0.4).network.loss.loss == 0.4
    burst = CS.apply(base, "burst", 8).network.loss
    assert burst.loss == CS.BURST_LOSS_IF_NONE and burst.burst_length == 8   # bursts need a loss: 30% if none
    assert CS.apply(base, "burst", 1).network.loss.burst_length is None
    assert CS.apply(base, "fleet_size", 6).n_vehicles == 6
    assert CS.apply(base, "mac", "aloha").network.mac == "aloha"
    assert base.network.modem.processing_latency_s == 0.3 and base.n_vehicles == 4   # the base is untouched
    assert CS.parse_values(["bit_rate=inf,500", "fleet_size=2,3"]) == {"bit_rate": [float("inf"), 500.0],
                                                                       "fleet_size": [2, 3]}
    with pytest.raises(ValueError, match="unknown setting"):
        CS.parse_values(["warp=1"])


def test_the_verdicts():
    def s(v, rate, e=1.0):
        return {"value": v, "success_rate": rate, "success_ci_low": rate - 0.1, "success_ci_high": min(1, rate + 0.1),
                "formation_error_m": e}
    assert "holds up to 5 s" in CS.verdict("latency", [s("0", 1), s("5", 0.9), s("10", 0.3, 4)])
    assert "breaks at 10 s" in CS.verdict("latency", [s("0", 1), s("5", 0.9), s("10", 0.3, 4)])
    assert "across the whole range" in CS.verdict("loss", [s("0%", 1), s("70%", 0.85)])
    assert "already fails" in CS.verdict("loss", [s("0%", 0.5)])


def test_a_small_study_end_to_end(tmp_path, capsys):
    out = tmp_path / "s"
    rc = CS.main(["--sweep", "mac", "loss", "--values", "loss=0,0.6", "--episodes", "2", "--duration", "60",
                  "--workers", "1", "--fleet-size", "3", "--out", str(out)])
    assert rc == 0
    for f in ("episodes.csv", "summary.csv", "report.md", "study.yaml", "plots/mac.png", "plots/loss.png"):
        assert (out / f).exists(), f
    report = (out / "report.md").read_text()
    assert "Channel sharing" in report and "Packet loss" in report and "Repeat it" in report
    assert yaml.safe_load((out / "study.yaml").read_text())["settings"]["loss"] == ["0%", "60%"]
    assert CS.main(["--sweep", "warp", "--out", str(tmp_path / "x")]) == 2
    assert CS.main(["--controller", "nowhere:Thing", "--sweep", "mac", "--out", str(tmp_path / "y")]) == 2
    assert "can't load controller" in capsys.readouterr().out


def test_strathe_comms_study_passes_everything_through(tmp_path, monkeypatch):
    calls = []
    monkeypatch.chdir(tmp_path)
    assert cli.main(["comms-study", "--quick", "--out", "mine", "--controller", "my_ctrl:C"],
                    runner=lambda c: calls.append(c) or 0) == 0
    argv = calls[0]
    assert os.path.basename(argv[1]) == "comms_study.py" and "--quick" in argv
    assert argv[argv.index("--out") + 1] == str(tmp_path / "mine")
    assert argv[argv.index("--controller-path") + 1] == str(tmp_path)   # a controller next to where you run
    calls.clear()
    cli.main(["comms-study", "--help"], runner=lambda c: calls.append(c) or 0)
    assert "--help" in calls[0] and "--out" not in calls[0]
    assert "comms-study" in cli.build_parser().format_help()
