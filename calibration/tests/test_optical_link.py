"""
Optical and hybrid links in the fleet comm model (comm_channel.OpticalChannel,
LinkConfig): the physics (range, water, pointing), pairing and determinism,
the hybrid skip rule, stale copies, bit-identical acoustic runs, and the
wiring into the experiment runner, training, the command line and the web form.
"""
import ast
import os
import sys

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

import strathe_cli as cli
from build_two_agent_cores import load_calibrated_parameters
from comm_channel import (BandwidthConfig, CommChannel, LinkConfig, OpticalChannel, OpticalLinkConfig,
                          OpticalTerminal, PacketLossConfig, optical_terminals, pointing_gain, received_power)
from fleet_belief import Beacon, BeliefSpec, LastBeaconBelief, PhysicsBelief
from fleet_env import FleetConfig, FleetSim
from optimizer import load_vehicle_spec
from test_strathe_cli import server  # noqa: F401  (pytest fixture)

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
TRUTH = load_calibrated_parameters(os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml"), quiet=True)
VEHICLE = load_vehicle_spec(os.path.join(CAL, "vehicle_spec.yaml")).vehicle
L2 = (PacketLossConfig(0.0, 1000.0, 0.3, 0.3), BandwidthConfig(1e9))
L3A = (PacketLossConfig(0.0, 1000.0, 0.6, 0.6), BandwidthConfig(1e9))
YAW180 = Rotation.from_euler("z", 180, degrees=True).as_matrix()


def _optical(**kw):
    c = OpticalLinkConfig(**kw)
    c.validate()
    return c


# ----------------------------------------------------------------- pointing

def test_cone_edges_in_both_directions():
    t = [OpticalTerminal((1, 0, 0), 30.0, 20.0)]
    def at(deg):  # receiver at `deg` off the sender's +x axis, facing back at the sender
        a = np.radians(deg)
        return (np.cos(a), np.sin(a), 0.0), Rotation.from_euler("z", 180 + deg, degrees=True).as_matrix()
    pos, R = at(29.9)
    assert pointing_gain((0, 0, 0), np.eye(3), pos, R, t) == 1.0          # inside the 30 deg beam
    pos, R = at(30.1)
    assert pointing_gain((0, 0, 0), np.eye(3), pos, R, t) == 0.0          # just outside the beam
    # the receiver's own field of view (20 deg half-angle) is checked too
    pos, _ = at(0.0)
    assert pointing_gain((0, 0, 0), np.eye(3), pos, Rotation.from_euler("z", 180 + 19.9, degrees=True).as_matrix(), t) == 1.0
    assert pointing_gain((0, 0, 0), np.eye(3), pos, Rotation.from_euler("z", 180 + 20.1, degrees=True).as_matrix(), t) == 0.0


def test_180_is_every_direction():
    omni = optical_terminals("omni")
    rng = np.random.default_rng(0)
    for _ in range(50):
        Rs, Rr = Rotation.random(2, random_state=1).as_matrix()
        assert pointing_gain(rng.normal(size=3), Rs, rng.normal(size=3), Rr, omni) == 1.0


def test_presets_at_the_line_start():
    # vehicle 0 at the origin, vehicle 1 ahead on +x, both facing +x (the task's start layout)
    fwd, fa, down = (optical_terminals(p) for p in ("forward_cone", "fore_aft_cones", "down_cone"))
    assert pointing_gain((0, 0, 0), None, (3, 0, 0), None, fwd) == 0.0     # the front vehicle faces away
    assert pointing_gain((3, 0, 0), None, (0, 0, 0), None, fwd) == 0.0
    assert pointing_gain((0, 0, 0), None, (3, 0, 0), None, fa) == 1.0
    assert pointing_gain((3, 0, 0), None, (0, 0, 0), None, fa) == 1.0
    assert pointing_gain((0, 0, 0), None, (3, 0, 0), None, down) == 0.0
    assert pointing_gain((0, 0, 0), None, (0, 0, 3), None, down) == 0.0     # below, but its receiver faces down too
    # face to face, the forward cone connects both ways
    assert pointing_gain((0, 0, 0), np.eye(3), (3, 0, 0), YAW180, fwd) == 1.0
    assert pointing_gain((3, 0, 0), YAW180, (0, 0, 0), np.eye(3), fwd) == 1.0


def test_several_terminals_any_pair_counts():
    terms = [OpticalTerminal((0, 1, 0), 10, 10), OpticalTerminal((1, 0, 0), 10, 10)]
    assert pointing_gain((0, 0, 0), np.eye(3), (5, 0, 0), YAW180, terms) == 1.0   # +x terminals see each other
    # side by side: the sender's +y terminal faces the receiver; the receiver (yawed 180) faces back with its +y one
    assert pointing_gain((0, 0, 0), np.eye(3), (0, 5, 0), YAW180, terms) == 1.0
    assert pointing_gain((0, 0, 0), np.eye(3), (0, 5, 0), np.eye(3), terms) == 0.0


@pytest.mark.parametrize("kw,msg", [
    ({"terminals": "forward_cone", "beam_half_angle_deg": 360}, "HALF-angle"),
    ({"terminals": "forward_cone", "receiver_half_fov_deg": 0}, r"half-angle in \(0, 180\]"),
    ({"terminals": "forward_cone", "beam_half_angle_deg": -5}, r"half-angle in \(0, 180\]"),
    ({"water": "custom"}, "needs attenuation_per_m"),
    ({"water": "clear", "attenuation_per_m": 0.3}, "only used with water: custom"),
    ({"water": "murky"}, "must be one of"),
    ({"data_rate_bps": 0}, "data_rate_bps must be positive"),
    ({"terminals": "sideways"}, "must be one of"),
    ({"terminals": "omni", "beam_half_angle_deg": 30}, "omni terminals see every direction"),
    ({"rated_range_m": 0}, "rated_range_m"),
])
def test_bad_settings_are_rejected_with_a_plain_message(kw, msg):
    with pytest.raises(ValueError, match=msg):
        OpticalLinkConfig(**kw).validate()


# ----------------------------------------------------------------- range and water

def test_success_falls_with_distance_and_murkiness():
    clear, coastal, harbor = (_optical(water=w) for w in ("clear", "coastal", "harbor"))
    ds = np.linspace(0.5, 60, 200)
    for c in (clear, coastal, harbor):
        p = [c.success_probability(d) for d in ds]
        assert all(a >= b for a, b in zip(p, p[1:]))
    for d in ds:
        assert clear.success_probability(d) >= coastal.success_probability(d) >= harbor.success_probability(d)
    assert clear.range_at(0.5) == pytest.approx(30.0, abs=0.1)     # rated range is quoted in clear water
    assert coastal.range_at(0.5) == pytest.approx(14.8, abs=0.3)
    assert harbor.range_at(0.5) == pytest.approx(3.9, abs=0.3)
    assert harbor.success_probability(2.0) > 0.99 and harbor.success_probability(8.0) < 0.01
    assert _optical(water="custom", attenuation_per_m=0.4).range_at(0.5) == pytest.approx(coastal.range_at(0.5))
    assert received_power(1000.0, 2.2) == 0.0 and np.isfinite(harbor.success_probability(1000.0))   # no log(0)
    assert harbor.success_probability(1.0, gain=0.0) == 0.0


def test_description_says_uncalibrated_and_half_angles():
    d = _optical(terminals="fore_aft_cones").describe()
    assert "Uncalibrated" in d and "half-angles" in d and "assumed" in d


# ----------------------------------------------------------------- the channel

def _send_many(ch, n=400, dist=3.0, R_r=None):
    got = []
    for k in range(n):
        m = ch.send("0", "1", 0.1 * k, (0, 0, 0), (dist, 0, 0), "position_beacon", payload=k, payload_size_bytes=24,
                    sender_R=np.eye(3), receiver_R=YAW180 if R_r is None else R_r)
        got.append(m is not None)
    return got


def test_deterministic_and_one_draw_per_send_even_when_blocked():
    cfg = _optical(water="harbor", terminals="forward_cone")
    a = _send_many(OpticalChannel(cfg, 0.1, seed=7), dist=4.0)
    b = _send_many(OpticalChannel(cfg, 0.1, seed=7), dist=4.0)
    assert a == b and 0 < sum(a) < len(a)
    blocked = OpticalChannel(cfg, 0.1, seed=7)
    assert not any(_send_many(blocked, n=10, R_r=np.eye(3)))           # receiver faces away: nothing gets through
    fresh = OpticalChannel(cfg, 0.1, seed=7)
    for _ in range(10):
        fresh._rng.random()
    assert blocked._rng.random() == fresh._rng.random()               # ...but each send still used one draw


def test_optical_latency_bandwidth_and_link_tag():
    ch = OpticalChannel(_optical(latency_s=0.001, data_rate_bps=8 * 240 / 0.1), 0.1, seed=0)   # 240 bytes per tick
    for k in range(20):
        ch.send("0", "1", 0.0, (0, 0, 0), (1, 0, 0), "position_beacon", payload=k, payload_size_bytes=24)
    ch.step(0.0)
    assert ch.receive("1") == []                                       # 1 ms later, not this tick
    ch.step(0.1)
    msgs = ch.receive("1")
    assert len(msgs) == 10 and all(m.link == "optical" for m in msgs)  # the rest were over the budget
    ac = CommChannel(*L2, seed=0)
    ac.send("0", "1", 0.0, (0, 0, 0), (1, 0, 0), "position_beacon", payload=1, payload_size_bytes=24,
            sender_R=np.eye(3), receiver_R=np.eye(3))
    ac.step(1.0)
    assert all(m.link == "acoustic" for m in ac.receive("1"))


def test_cones_without_orientation_are_an_error():
    ch = OpticalChannel(_optical(terminals="forward_cone"), 0.1, seed=0)
    with pytest.raises(ValueError, match="needs sender_R and receiver_R"):
        ch.send("0", "1", 0.0, (0, 0, 0), (1, 0, 0), "position_beacon")
    OpticalChannel(_optical(), 0.1, seed=0).send("0", "1", 0.0, (0, 0, 0), (1, 0, 0), "position_beacon")


def test_comm_channel_still_imports_no_physics():
    tree = ast.parse(open(os.path.join(CAL, "comm_channel.py"), encoding="utf-8").read())
    mods = {n.module if isinstance(n, ast.ImportFrom) else a.name
            for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert mods <= {"__future__", "heapq", "dataclasses", "typing", "numpy"}, mods


# ----------------------------------------------------------------- hybrid policy and stale copies

def test_hybrid_policy_falls_back_to_redundant_for_asymmetric_terminals():
    sym = LinkConfig("hybrid", _optical(terminals="fore_aft_cones"))
    assert sym.effective_hybrid_policy() == ("optical_first", None)
    asym = LinkConfig("hybrid", _optical(terminals="forward_cone", beam_half_angle_deg=30, receiver_half_fov_deg=60))
    pol, why = asym.effective_hybrid_policy()
    assert pol == "redundant" and "symmetric" in why
    assert "redundant" in asym.describe()
    assert LinkConfig("hybrid", _optical(), "redundant").effective_hybrid_policy() == ("redundant", None)
    assert LinkConfig("acoustic").effective_hybrid_policy() == (None, None)


def test_stale_acoustic_copy_never_rolls_the_belief_back():
    """An optical beacon sent at t = 5.0 s, then an acoustic one sent at 4.9 s arriving later: keep 5.0."""
    new = Beacon(send_tick=50, position=np.array([5.0, 0, 0]), orientation=np.eye(3), nu=np.zeros(6), tau=np.zeros(6))
    old = Beacon(send_tick=49, position=np.array([4.9, 0, 0]), orientation=np.eye(3), nu=np.zeros(6), tau=np.zeros(6))
    for b in (LastBeaconBelief(0.1), PhysicsBelief(0.1, TRUTH, VEHICLE)):
        b.on_tick(51, [new])
        est = b.estimate().copy()
        b.on_tick(52, [old])                 # the acoustic copy, sent earlier, arrives later
        assert b._beacon.send_tick == 50 and b.age_s == pytest.approx(0.2)
        if isinstance(b, LastBeaconBelief):
            np.testing.assert_allclose(b.estimate(), [5.0, 0, 0])
        b2 = type(b)(0.1) if isinstance(b, LastBeaconBelief) else PhysicsBelief(0.1, TRUTH, VEHICLE)
        b2.on_tick(51, [new])
        b2.on_tick(52, [])
        np.testing.assert_allclose(b.estimate(), b2.estimate())   # as if the stale copy never came
        b3 = type(b)(0.1) if isinstance(b, LastBeaconBelief) else PhysicsBelief(0.1, TRUTH, VEHICLE)
        b3.on_tick(51, [new])
        b3.on_tick(52, [new])                # a same-tick duplicate (redundant) doesn't reset dead reckoning
        np.testing.assert_allclose(b3.estimate(), b2.estimate())
        assert est is not None


# ----------------------------------------------------------------- in the fleet simulation

def _run(link, sev=L3A, interval=3.0, seed=3, d0=4.5, v0=0.35, steps=120, cfg=None, policy=None):
    sim = FleetSim(TRUTH, VEHICLE, cfg or FleetConfig(), *sev, belief=BeliefSpec("last_beacon", "last_beacon"),
                   beacon_interval_s=interval, link=link)
    obs = sim.reset(comm_seed=seed, initial_separation=d0, closing_speed=v0)
    rng = np.random.default_rng(0)
    out = []
    for _ in range(steps):
        acts = policy(obs) if policy else [rng.uniform(-0.3, 0.3, 6) for _ in range(2)]
        obs, r, te, tr, info = sim.step(acts)
        out.append((np.stack(obs).copy(), r, info))
        if te or tr:
            break
    return sim, out


def test_acoustic_link_is_bit_identical_to_no_link():
    for sev in (L2, L3A):
        _, a = _run(None, sev)
        _, b = _run(LinkConfig("acoustic"), sev)
        assert len(a) == len(b)
        for (oa, ra, ia), (ob, rb, ib) in zip(a, b):
            assert np.array_equal(oa, ob) and ra == rb and ia["delivered"] == ib["delivered"]
            assert ia["sent"]["optical"] == 0 and ia["belief_updates"]["optical"] == 0


def test_acoustic_drops_are_paired_between_acoustic_and_hybrid():
    """The hybrid's acoustic beacons (sent or skipped) use the same draws as an acoustic-only run."""
    def acoustic_draws(link):
        sim, out = _run(link, L2, interval=1.0, steps=100)
        return sim.channel._rng.random()      # the next draw: equal only if the same number were used
    ref = acoustic_draws(None)
    assert acoustic_draws(LinkConfig("hybrid", _optical())) == ref                     # optical_first: skips
    assert acoustic_draws(LinkConfig("hybrid", _optical(), "redundant")) == ref
    assert acoustic_draws(LinkConfig("hybrid", _optical(terminals="forward_cone"))) == ref


def test_optical_first_skips_acoustic_while_the_link_is_up_and_resumes_when_it_drops():
    _, out = _run(LinkConfig("hybrid", _optical()), L2, interval=1.0, steps=100)
    assert sum(i["acoustic_skipped"] for _, _, i in out) > 0
    assert sum(i["sent"]["acoustic"] for _, _, i in out) == 0         # omni, 4.5 m, coastal: always up
    # cones the random commands soon point away from: acoustic beacons must resume (no starvation)
    _, out = _run(LinkConfig("hybrid", _optical(terminals="fore_aft_cones", beam_half_angle_deg=10,
                                                receiver_half_fov_deg=10)), L2, interval=1.0, steps=150,
                  policy=lambda obs: [np.array([0, 0, 0, 1.0, 0.5, 0.8])] * 2)
    assert sum(i["sent"]["acoustic"] for _, _, i in out) > 0
    _, out = _run(LinkConfig("hybrid", _optical(), "redundant"), L2, interval=1.0, steps=60)
    assert sum(i["acoustic_skipped"] for _, _, i in out) == 0 and sum(i["sent"]["acoustic"] for _, _, i in out) > 0


def test_optical_link_keeps_beliefs_fresh_and_harbor_water_fails_at_range():
    _, out = _run(LinkConfig("optical", _optical(water="coastal")), steps=60)
    ages = [a for _, _, i in out for a in i["belief_ages"] if np.isfinite(a)]
    assert np.median(ages) == pytest.approx(0.1)
    assert all(i["sent"]["acoustic"] == 0 for _, _, i in out)
    still = lambda obs: [np.zeros(6)] * 2
    _, far = _run(LinkConfig("optical", _optical(water="harbor")), d0=5.4, v0=0.6, steps=40, policy=still)
    near = _run(LinkConfig("optical", _optical(water="harbor")), d0=2.5, v0=0.0, steps=40, policy=still)[1]
    # the last tick's beacons are still in flight when the episode ends
    rate = lambda o: (sum(len(x) for _, _, i in o for x in i["delivered_by"])
                      / (sum(i["sent"]["optical"] for _, _, i in o) - 2))
    assert rate(near) > 0.9 and rate(far) < 0.2


def test_forward_cones_carry_nothing_at_the_line_start():
    _, out = _run(LinkConfig("optical", _optical(terminals="forward_cone")), steps=30,
                  policy=lambda obs: [np.zeros(6)] * 2)
    assert sum(i["sent"]["optical"] for _, _, i in out) == 60
    assert not any(any(x) for _, _, i in out for x in i["delivered_by"])


def test_optical_link_is_deterministic_and_the_start_geometry_note():
    link = LinkConfig("hybrid", _optical(water="harbor", terminals="fore_aft_cones"))
    a, b = _run(link, steps=80)[1], _run(link, steps=80)[1]
    assert all(np.array_equal(x[0], y[0]) and x[1] == y[1] for x, y in zip(a, b))
    note = LinkConfig("optical", _optical(terminals="forward_cone")).start_geometry_note([2.0, 5.5])
    assert "0% of vehicle pairs are inside each other's cones" in note
    assert LinkConfig("optical", _optical()).start_geometry_note([2.0, 5.5]) is None
    assert "only" in LinkConfig("optical", _optical(water="custom", attenuation_per_m=5.0)).start_geometry_note([4, 5])


def test_noisy_mode_and_mujoco_with_hybrid():
    from sensor_noise import load_sensor_noise
    link = LinkConfig("hybrid", _optical(terminals="fore_aft_cones"))
    for cfg in (FleetConfig(sensor_noise=load_sensor_noise("typical")), FleetConfig(dynamics_backend="mujoco")):
        _, out = _run(link, cfg=cfg, steps=40)
        assert all(np.all(np.isfinite(o)) for o, _, _ in out)


# ----------------------------------------------------------------- experiment runner, training, CLI, web

def _experiment(tmp_path, comm):
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.safe_dump({
        "schema_version": 2, "name": "optical_test", "policies": [{"model_path": os.path.join(CAL, "fleet_policy_seed1")}],
        "truth_params": os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml"),
        "vehicle_spec": os.path.join(CAL, "vehicle_spec.yaml"),
        "beliefs": ["calibrated", "last_beacon"], "compare": {"a": "calibrated", "b": "last_beacon"},
        "communication": comm, "initial_conditions": {"separations_m": [2.5, 4.5], "closing_speeds_mps": [0.35]},
        "evaluation": {"comm_seeds": [1], "workers": 1}, "output_dir": str(tmp_path / "out")}))
    return p


def test_schema_links_waters_and_defaults():
    from fleet_experiment_schema import CommunicationConfig as C
    c = C(severity="L2", link=["acoustic", "hybrid", "optical"],
          optical={"water": ["clear", "harbor"], "terminals": "forward_cone", "compare_omni": True})
    labels = [lc.label for lc, _ in c.link_configs()]
    assert labels[0] == "acoustic" and "hybrid/harbor/forward_cone" in labels and "optical/clear" in labels
    assert len(labels) == 1 + 2 * 2 * 2
    assert C(severity="L2", link="hybrid").optical.water == "coastal"       # default written in
    assert C(link="optical").severities == ["L0"]                          # no acoustic severity needed
    for bad, msg in ((dict(severity="L2", optical={}), "link is acoustic"),
                     (dict(severity="L2", link="hybrid", optical={"compare_omni": True}), "already omnidirectional"),
                     (dict(severity="L2", link="hybrid", optical={"beacon_interval_s": 0.15}), "multiple of the 0.1"),
                     (dict(link="hybrid"), "at least one severity")):
        with pytest.raises(ValueError, match=msg):
            C(**bad)


def test_experiment_end_to_end_with_links_and_pointing(tmp_path):
    import run_fleet_experiment as rfe
    p = _experiment(tmp_path, {"severity": "L3a", "beacon_interval_s": 3.0, "link": ["acoustic", "optical", "hybrid"],
                               "optical": {"water": "harbor", "terminals": "forward_cone", "compare_omni": True}})
    m = rfe.run_experiment(str(p))
    rep = (tmp_path / "out" / "summary_report.md").read_text(encoding="utf-8")
    assert "## Optical link" in rep and "Uncalibrated" in rep and "0% of vehicle pairs" in rep
    assert "Does the optical link help?" in rep and "What pointing costs" in rep and "Why `optical_first` exists" in rep
    assert "Median belief age" in rep and "Optical up" in rep
    comms = {s["comm"] for s in m["links"]}
    assert comms == {"L3a@3s", "optical/harbor", "optical/harbor/forward_cone", "L3a@3s hybrid/harbor",
                     "L3a@3s hybrid/harbor/forward_cone"}
    assert m["link_vs_acoustic"] and m["pointing_vs_omni"]
    by = {s["comm"]: s for s in m["links"]}
    assert by["optical/harbor"]["median_belief_age_s"] == pytest.approx(0.1)
    assert by["L3a@3s"]["median_belief_age_s"] > 1.0
    saved = yaml.safe_load((tmp_path / "out" / "experiment.yaml").read_text())
    assert saved["communication"]["optical"]["terminals"] == "forward_cone"
    import csv
    rows = list(csv.DictReader(open(tmp_path / "out" / "episodes.csv")))
    assert {"link", "optical_up_fraction", "belief_updates_optical_share", "median_belief_age_s"} <= set(rows[0])


def test_asymmetric_hybrid_is_reported_as_redundant(tmp_path):
    import run_fleet_experiment as rfe
    p = _experiment(tmp_path, {"severity": "L2", "link": "hybrid",
                               "optical": {"terminals": "fore_aft_cones", "beam_half_angle_deg": 30,
                                           "receiver_half_fov_deg": 60}})
    rfe.run_experiment(str(p))
    rep = (tmp_path / "out" / "summary_report.md").read_text(encoding="utf-8")
    assert "Hybrid policy changed" in rep and "uses redundant" in rep


def test_acoustic_only_report_has_no_optical_section(tmp_path):
    import run_fleet_experiment as rfe
    rfe.run_experiment(str(_experiment(tmp_path, {"severity": "L2"})))
    rep = (tmp_path / "out" / "summary_report.md").read_text(encoding="utf-8")
    assert "Optical link" not in rep and "over an acoustic link that loses packets" in rep


def test_training_conditions_carry_the_link():
    import train_fleet_policy as tfp
    hyb = LinkConfig("hybrid", _optical(water="harbor"))
    conds = tfp.comm_conditions(link=hyb)
    assert len(conds) == len(tfp.comm_conditions()) and all(c.link is hyb for c in conds)
    assert conds[0].label.endswith("hybrid/harbor")
    opt = tfp.comm_conditions(link=LinkConfig("optical", _optical()))
    assert len(opt) == 1 and opt[0].label == "optical/coastal"
    env = tfp.build_training_env(1, 0, link=hyb)
    env.reset()
    for _ in range(12):
        env.step(np.zeros((2, 6), dtype=np.float32))
    assert env.sims[0].optical_channel is not None


def test_cli_fleet_and_train_fleet_link_flags(tmp_path):
    dest = tmp_path / "x.yaml"
    assert cli.main(["fleet", "--link", "acoustic", "hybrid", "--optical-water", "clear", "harbor",
                     "--optical-terminals", "fore_aft_cones", "--optical-beam", "20", "--optical-fov", "20",
                     "--optical-compare-omni", "--hybrid-policy", "redundant", "--save-config", str(dest)]) == 0
    comm = yaml.safe_load(dest.read_text())["communication"]
    assert comm["link"] == ["acoustic", "hybrid"] and comm["hybrid_policy"] == "redundant"
    assert comm["optical"] == {"water": ["clear", "harbor"], "terminals": "fore_aft_cones", "beam_half_angle_deg": 20.0,
                               "receiver_half_fov_deg": 20.0, "compare_omni": True}
    for bad in (["--link", "hybrid", "--optical-terminals", "forward_cone", "--optical-beam", "360"],
                ["--optical-water", "harbor"],                                    # optical settings, acoustic link
                ["--link", "hybrid", "--optical-water", "custom"]):
        assert cli.main(["fleet", *bad, "--save-config", str(tmp_path / "y.yaml")]) == 2, bad
    calls = []
    params = os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml")
    assert cli.main(["train-fleet", "--params", params, "--link", "hybrid", "--optical-water", "harbor",
                     "--optical-terminals", "fore_aft_cones", "--out", str(tmp_path / "p")],
                    runner=lambda argv: calls.append(argv) or 0) == 0
    t = calls[-1]
    assert t[t.index("--link") + 1] == "hybrid" and t[t.index("--optical-water") + 1] == "harbor"
    assert cli.main(["train-fleet", "--params", params, "--optical-water", "harbor", "--out", str(tmp_path / "p")],
                    runner=lambda argv: calls.append(argv) or 0) == 2
    import argparse
    import sim_mode
    ap = argparse.ArgumentParser()
    sim_mode.add_link_args(ap)
    ap.add_argument("--out")
    assert sim_mode.link_from_args(ap.parse_args(t[t.index("--link"):])).label == "hybrid/harbor/fore_aft_cones"


def test_web_fleet_and_train_fleet_forms_start_the_cli_with_link_flags(server):
    request, token, started, manager, tmp_path = server
    body = [("token", token), ("shipped", "default_experiment.yaml"), ("link", "acoustic"), ("link", "hybrid"),
            ("optical_water", "coastal"), ("optical_water", "harbor"), ("optical_terminals", "forward_cone"),
            ("optical_compare_omni", "1"), ("backend", "stand_in")]
    status, text, _ = request("POST", "/jobs/fleet", body)
    assert status == 303, text
    cmd = started[-1]
    assert cmd[cmd.index("--link") + 1:cmd.index("--link") + 3] == ["acoustic", "hybrid"]
    assert "--optical-compare-omni" in cmd and cmd[cmd.index("--optical-terminals") + 1] == "forward_cone"
    parsed = cli.build_parser().parse_args(cmd[3:])            # the CLI accepts exactly what the form sends
    assert parsed.optical_water == ["coastal", "harbor"]
    status, text, _ = request("POST", "/jobs/fleet", (
        [("token", token), ("shipped", "default_experiment.yaml"), ("link", "hybrid"),
         ("optical_terminals", "forward_cone"), ("optical_beam", "360"), ("backend", "stand_in")]))
    assert status == 400 and "HALF-angle" in text
    status, text, _ = request("POST", "/jobs/train_fleet", (
        [("token", token), ("params", os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml")),
         ("out", str(tmp_path / "pol")), ("link", "optical"), ("optical_water", "harbor"), ("seed", "1")]))
    assert status == 303, text
    cmd = started[-1]
    assert cmd[cmd.index("--link") + 1] == "optical" and cmd[cmd.index("--optical-water") + 1] == "harbor"
    cli.build_parser().parse_args(cmd[3:])


def test_help_and_web_labels_say_uncalibrated_and_half_angle():
    import strathe_web as w
    html = w.link_fields("x", sweep=True)
    assert "uncalibrated" in html and "half-angle" in html and "180 = every direction" in html
    fleet = cli.build_parser()._subparsers._group_actions[0].choices["fleet"].format_help()
    assert "UNCALIBRATED" in fleet and "HALF-angle" in fleet and "--optical-compare-omni" in fleet
    tf = cli.build_parser()._subparsers._group_actions[0].choices["train-fleet"].format_help()
    assert "--link" in tf and "--optical-terminals" in tf


# ----------------------------------------------------------------- review follow-ups

def test_skip_only_when_heard_on_the_previous_tick_and_uncovered_skips_are_counted():
    """A vehicle skips its acoustic beacon only if an optical message arrived within link_up_ticks; skipped
    beacons whose optical copy was lost too are counted."""
    still_apart = lambda obs: [np.zeros(6)] * 2
    link = lambda k: LinkConfig("hybrid", _optical(water="harbor"), link_up_ticks=k)
    # ~3.7 m apart in harbour water: the optical link is intermittent (about 60% delivery)
    for k in (1, 30):
        _, out = _run(link(k), L2, interval=1.0, d0=3.5, v0=0.7, steps=200, policy=still_apart)
        prev_optical = [0, 0]
        for _, _, i in out:
            if k == 1:
                assert i["acoustic_skipped"] <= sum(prev_optical)      # only right after an optical receipt
            assert i["acoustic_skipped_uncovered"] <= i["acoustic_skipped"]
            prev_optical = [int("optical" in x) for x in i["delivered_by"]]
        if k == 1:
            one = (sum(i["acoustic_skipped"] for _, _, i in out), sum(i["acoustic_skipped_uncovered"] for _, _, i in out))
    wide = (sum(i["acoustic_skipped"] for _, _, i in out), sum(i["acoustic_skipped_uncovered"] for _, _, i in out))
    assert wide[0] > one[0] and wide[1] > one[1]      # a wide window skips more, and more of them uncovered
    with pytest.raises(ValueError, match="whole number of ticks"):
        LinkConfig("hybrid", _optical(), link_up_ticks=0).validate()


def test_down_cone_points_at_a_vehicle_below():
    # body z is down (Fossen) and so is the simulators' world z: a buoyant vehicle rises to negative z
    down = optical_terminals("down_cone")
    from comm_channel import beam_covers
    assert beam_covers((0, 0, 0), np.eye(3), (0, 0, 5), down)          # directly below
    assert not beam_covers((0, 0, 0), np.eye(3), (0, 0, -5), down)     # directly above
    assert not beam_covers((0, 0, 0), np.eye(3), (5, 0, 0), down)      # level, ahead
    from buoyancy import with_buoyancy_offset
    from dynamics_model import SimulationConfig
    from fast_dynamics import FastCore
    core = FastCore(with_buoyancy_offset(TRUTH, 20.0), VEHICLE, SimulationConfig(dt=0.1))
    core.reset(position=np.zeros(3), nu=np.zeros(6))
    for _ in range(30):
        core.step(np.zeros(6))
    assert core.position_array()[2] < 0                                # buoyant rises: world z is down


def test_optical_interval_must_be_whole_ticks(tmp_path):
    lc = LinkConfig("optical", _optical(beacon_interval_s=0.15))
    with pytest.raises(ValueError, match="multiple of dt"):
        FleetSim(TRUTH, VEHICLE, FleetConfig(), *L2, link=lc)
    params = os.path.join(CAL, "calibrated_parameters_fleet_truth.yaml")
    assert cli.main(["train-fleet", "--params", params, "--link", "optical", "--optical-interval", "0.15",
                     "--out", str(tmp_path / "p")], runner=lambda argv: 0) == 2
    assert cli.main(["fleet", "--link", "optical", "--optical-interval", "0.15",
                     "--save-config", str(tmp_path / "x.yaml")]) == 2


def test_no_latency_advantage_both_links_arrive_on_the_next_tick():
    no_loss = (PacketLossConfig(0.0, 1000.0, 0.0, 0.0), BandwidthConfig(1e9))
    for link in (None, LinkConfig("optical", _optical(water="clear"))):
        _, out = _run(link, no_loss, interval=1.0, steps=12, d0=3.0, v0=0.0,
                      policy=lambda obs: [np.zeros(6)] * 2)
        first = next(k for k, (_, _, i) in enumerate(out, start=1) if any(i["delivered"]))
        assert first == (11 if link is None else 2)       # sent on tick 10 (acoustic) or 1 (optical): +1 tick
