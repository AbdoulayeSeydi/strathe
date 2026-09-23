"""train_fleet_policy.py: the training distribution is what its docstring says."""
import json
import sys
from collections import Counter

import numpy as np
import pytest

import train_fleet_policy as tfp
from build_two_agent_cores import load_calibrated_parameters
from optimizer import initial_parameters, load_vehicle_spec


def test_comm_conditions_cover_every_severity_and_beacon_interval():
    conds = tfp.comm_conditions()
    assert len(conds) == len(tfp.TRAIN_SEVERITIES) * len(tfp.TRAIN_BEACON_INTERVALS_S)
    assert {c.label for c in conds} >= {"L0@0.5s", "L3a@3s"}
    assert all(c.beacon_interval_s in tfp.TRAIN_BEACON_INTERVALS_S for c in conds)


def test_belief_sampler_matches_the_documented_mix():
    truth = load_calibrated_parameters(tfp.TRUTH_PARAMS_PATH)
    generic = initial_parameters(load_vehicle_spec(tfp.VEHICLE_SPEC_PATH))
    sample = tfp.make_belief_sampler(truth, generic)
    rng = np.random.default_rng(0)
    specs = [sample(rng) for _ in range(4000)]
    counts = Counter("physics_interpolated" if s.kind == "physics" else s.kind for s in specs)
    for kind, p in tfp.BELIEF_MIX.items():
        assert counts[kind] / 4000 == pytest.approx(p, abs=0.03), kind
    # interpolated models really span truth -> generic
    physics = [s for s in specs if s.kind == "physics"]
    xu = np.array([s.params.Xu for s in physics])
    lo, hi = sorted([truth.Xu, generic.Xu])
    assert xu.min() >= lo - 1e-9 and xu.max() <= hi + 1e-9 and np.ptp(xu) > 0.8 * (hi - lo)


def test_training_runs_end_to_end_and_writes_provenance(tmp_path, monkeypatch):
    out = tmp_path / "pol"
    monkeypatch.setattr(sys, "argv", ["train_fleet_policy.py", "--timesteps", "2048", "--seed", "0",
                                      "--n-sims", "2", "--out", str(out)])
    tfp.main()
    meta = json.loads((tmp_path / "pol.json").read_text(encoding="utf-8"))
    assert (tmp_path / "pol.zip").exists()
    assert meta["contract"]["per_vehicle"] and meta["timesteps"] == 2048
    assert meta["training_distribution"]["beacon_intervals_s"] == list(tfp.TRAIN_BEACON_INTERVALS_S)
    from validate_fleet_policy import load_fleet_policy
    load_fleet_policy(str(out))  # the trained file meets the lab-facing contract


@pytest.mark.parametrize("kind,expect_kind", [("calibrated", "physics"), ("generic", "physics"),
                                               ("last_beacon", "last_beacon"), ("oracle", "oracle")])
def test_fixed_belief_sampler(kind, expect_kind):
    truth = load_calibrated_parameters(tfp.TRUTH_PARAMS_PATH)
    generic = initial_parameters(load_vehicle_spec(tfp.VEHICLE_SPEC_PATH))
    sample = tfp.make_fixed_belief_sampler(kind, truth, generic)
    rng = np.random.default_rng(0)
    specs = {sample(rng).label for _ in range(50)}
    assert specs == {kind} and sample(rng).kind == expect_kind
    if kind == "calibrated":
        np.testing.assert_array_equal(sample(rng).params.to_vector(), truth.to_vector())


def test_unknown_fixed_belief_is_rejected():
    with pytest.raises(ValueError, match="--belief"):
        tfp.make_fixed_belief_sampler("telepathy", None, None)
