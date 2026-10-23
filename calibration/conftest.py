# Standard pytest mechanism: pytest automatically adds the directory
# containing conftest.py to sys.path, so tests/ can import dynamics_model
# and other calibration/ modules without needing PYTHONPATH set manually.
import tempfile

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "uses_docker: may use the Docker images built on this machine")


@pytest.fixture(autouse=True)
def _docker_isolated(request, monkeypatch, tmp_path_factory):
    """Tests don't see the Docker images a developer has built (they'd change what `strathe check` and the replay-tool
    lookups say), and Docker's shared work folder is a temporary one (PROJECT_STATUS.md #92)."""
    import sim_docker
    monkeypatch.setattr(tempfile, "tempdir", tempfile.tempdir)   # restored after each test
    monkeypatch.setattr(sim_docker, "SHARED_TMP", str(tmp_path_factory.mktemp("strathe_tmp")))
    monkeypatch.setattr(sim_docker, "_seen", set())
    if not request.node.get_closest_marker("uses_docker"):
        monkeypatch.setattr(sim_docker, "auto_image", lambda sim: None)
