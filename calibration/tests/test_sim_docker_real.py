"""Strathe's real Docker images (PROJECT_STATUS.md #92): run when Docker is running and `python strathe.py setup-sim
gazebo --docker` / `... stonefish --docker` has built them; skipped otherwise (CI)."""
import os
import shutil
import tempfile

import pytest

import sim_docker as D
from simfit import core
from simfit.adapters.gazebo import GazeboAdapter
from simfit.replay_tool import tool_argv
from test_simfit_gazebo import WORLD, _window

pytestmark = pytest.mark.uses_docker


def _built(sim: str) -> bool:
    return bool(shutil.which("docker")) and D.state()[0] == "running" and D.image_present(D.IMAGES[sim])


needs_gazebo_image = pytest.mark.skipif(not _built("gazebo"), reason="Strathe's Gazebo Docker image not built")
needs_stonefish_image = pytest.mark.skipif(not _built("stonefish"), reason="Strathe's Stonefish Docker image not built")


@pytest.fixture
def shared(monkeypatch):
    """Work folders under the home folder, which every Docker set-up shares (pytest's own temp folder may not be)."""
    root = os.path.join(os.path.expanduser("~"), ".strathe", "tmp")
    os.makedirs(root, exist_ok=True)
    d = tempfile.mkdtemp(prefix="pytest_", dir=root)
    monkeypatch.setattr(D, "SHARED_TMP", d)
    D.use_shared_temp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@needs_gazebo_image
def test_the_gazebo_image_runs_and_sees_the_shared_folder(shared):
    assert D.features(D.IMAGES["gazebo"])["gz_sim"].startswith("8.")
    D.check_shared(D.IMAGES["gazebo"], [shared])


@needs_gazebo_image
def test_gazebo_in_docker_gives_the_local_tools_steady_states(shared):
    """The same frames and numbers as test_simfit_gazebo's real-engine test."""
    ad = GazeboAdapter(WORLD, tool=tool_argv("docker", "gazebo"))
    d = tempfile.mkdtemp()
    model = ad.write_runnable(None, d)
    for F, axis, want in (((20, 0, 0, 0, 0, 0), 0, 0.737), ((0, 20, 0, 0, 0, 0), 1, 0.434),
                          ((0, 0, 0, 0, 0, 0.5), 5, 0.2247)):
        (V, _), = ad.replay(model, core.window_rows(_window(F), [(0, 101)]), d)
        assert V[-1, axis] == pytest.approx(want, abs=2e-3), (F, V[-1])
    assert ad.info()["mass"] == pytest.approx(10.0)


@needs_stonefish_image
def test_the_stonefish_image_is_patched():
    f = D.features(D.IMAGES["stonefish"])
    assert f["patched"] and f["axis_damping"]
