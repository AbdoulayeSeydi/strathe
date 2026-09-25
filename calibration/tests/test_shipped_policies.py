"""Every policy shipped in the repo loads and meets the fleet v2 per-vehicle contract."""
import glob
import os

import pytest

from validate_fleet_policy import load_fleet_policy

HERE = os.path.join(os.path.dirname(__file__), "..")
SHIPPED = sorted(glob.glob(os.path.join(HERE, "fleet_policy_seed*.zip"))
                 + glob.glob(os.path.join(HERE, "specialized_policies", "*.zip"))
                 + glob.glob(os.path.join(HERE, "disturbance_policies", "*.zip"))
                 + glob.glob(os.path.join(HERE, "current_policies", "*.zip")))


def test_policies_are_shipped():
    assert len(SHIPPED) >= 15  # 3 mixed + 8 specialized + 2 disturbance-trained + 2 current-trained


@pytest.mark.parametrize("path", SHIPPED, ids=[os.path.basename(p) for p in SHIPPED])
def test_shipped_policy_meets_contract(path):
    load_fleet_policy(path)
    assert os.path.exists(path[:-4] + ".json"), "every policy ships with its training provenance"


def test_disturbance_policies_record_their_training_disturbance():
    import json
    for k in (1, 2):
        meta = json.load(open(os.path.join(HERE, "disturbance_policies", f"fleet_policy_dist_seed{k}.json")))
        assert meta["training_distribution"]["disturbance"] == {"file": "bluerov2_rollpitch.yaml", "scale": 1.0}


def test_current_policies_record_their_training_current():
    import json
    for k in (1, 2):
        meta = json.load(open(os.path.join(HERE, "current_policies", f"fleet_policy_current_seed{k}.json")))
        assert meta["truth_params"] == "calibrated_parameters_fleet_truth.yaml"
        assert "0-0.3 m/s" in meta["training_distribution"]["current"]
