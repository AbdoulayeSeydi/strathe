"""Every policy shipped in the repo loads and meets the fleet v2 per-vehicle contract."""
import glob
import os

import pytest

from validate_fleet_policy import load_fleet_policy

HERE = os.path.join(os.path.dirname(__file__), "..")
SHIPPED = sorted(glob.glob(os.path.join(HERE, "fleet_policy_seed*.zip"))
                 + glob.glob(os.path.join(HERE, "specialized_policies", "*.zip")))


def test_policies_are_shipped():
    assert len(SHIPPED) >= 11  # 3 mixed-belief + 8 specialized


@pytest.mark.parametrize("path", SHIPPED, ids=[os.path.basename(p) for p in SHIPPED])
def test_shipped_policy_meets_contract(path):
    load_fleet_policy(path)
    assert os.path.exists(path[:-4] + ".json"), "every policy ships with its training provenance"
