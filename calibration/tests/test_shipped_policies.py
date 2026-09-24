"""Every policy shipped in the repo loads and meets the fleet v2 per-vehicle contract."""
import glob
import os

import pytest

from validate_fleet_policy import load_fleet_policy

HERE = os.path.join(os.path.dirname(__file__), "..")
SHIPPED = sorted(glob.glob(os.path.join(HERE, "fleet_policy_seed*.zip"))
                 + glob.glob(os.path.join(HERE, "specialized_policies", "*.zip"))
                 + glob.glob(os.path.join(HERE, "disturbance_policies", "*.zip")))


def test_policies_are_shipped():
    assert len(SHIPPED) >= 13  # 3 mixed-belief + 8 specialized + 2 trained with the roll/pitch disturbance


@pytest.mark.parametrize("path", SHIPPED, ids=[os.path.basename(p) for p in SHIPPED])
def test_shipped_policy_meets_contract(path):
    load_fleet_policy(path)
    assert os.path.exists(path[:-4] + ".json"), "every policy ships with its training provenance"


def test_disturbance_policies_record_their_training_disturbance():
    import json
    for k in (1, 2):
        meta = json.load(open(os.path.join(HERE, "disturbance_policies", f"fleet_policy_dist_seed{k}.json")))
        assert meta["training_distribution"]["disturbance"] == {"file": "bluerov2_rollpitch.yaml", "scale": 1.0}
