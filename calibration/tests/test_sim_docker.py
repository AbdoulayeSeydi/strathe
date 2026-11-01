"""Stonefish and Gazebo in Docker, with nothing to build (sim_docker.py; PROJECT_STATUS.md #92). Docker itself is faked
here; test_sim_docker_real.py runs the real images when they're built on the machine."""
import os
import re
import subprocess
import tempfile

import pytest

import sim_docker as D
import stonefish_fit as SF
import strathe_cli as cli
from simfit import replay_tool as RT
from simfit.core import SimFitError


def _docker(monkeypatch, st="running", images=("strathe-gz-replay", "strathe-stonefish-replay")):
    monkeypatch.setattr(D, "state", lambda: (st, "29.5.2" if st == "running" else ""))
    monkeypatch.setattr(D, "image_present", lambda image: image in images)


def test_docker_means_strathes_image_for_that_simulator():
    assert D.image_for("docker", "gazebo") == "strathe-gz-replay"
    assert D.image_for("docker", "stonefish") == "strathe-stonefish-replay"
    assert D.image_for("docker:lab/sim:2", None) == "lab/sim:2"
    with pytest.raises(D.DockerError, match="docker:IMAGE"):
        D.image_for("docker", "external")
    assert D.is_docker("docker") and D.is_docker("docker:x") and not D.is_docker("dockerfile") and not D.is_docker(None)


def test_each_missing_piece_says_what_to_do(monkeypatch):
    _docker(monkeypatch, "missing")
    with pytest.raises(D.DockerError, match="Docker Desktop.*Colima.*Docker Engine"):
        D.require("strathe-gz-replay", "gazebo")
    _docker(monkeypatch, "stopped")
    with pytest.raises(D.DockerError, match="colima start"):
        D.require("strathe-gz-replay", "gazebo")
    _docker(monkeypatch, images=())
    with pytest.raises(D.DockerError, match="setup-sim gazebo --docker"):
        D.require("strathe-gz-replay", "gazebo")
    with pytest.raises(D.DockerError, match="docker pull lab/sim"):
        D.require("lab/sim", None)
    _docker(monkeypatch)
    D.require("strathe-gz-replay", "gazebo")


def _host(mount):
    """The host side of a -v HOST:CONTAINER mount (a Windows host path has its own colon)."""
    return mount.rsplit(":", 1)[0]


def test_the_container_runs_as_you_with_the_folders_at_the_same_paths(tmp_path):
    argv = D.run_prefix("img", [str(tmp_path / "b"), str(tmp_path / "a"), str(tmp_path / "a")], {"GZ_X": "/m"})
    assert argv[:3] == ["docker", "run", "--rm"] and argv[-1] == "img"
    if hasattr(os, "getuid"):
        assert argv[argv.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"   # files it writes are yours
    else:
        assert "--user" not in argv   # Windows: Docker Desktop maps file ownership itself
    assert "HOME=/tmp" in argv and "GZ_X=/m" in argv
    mounts = [argv[i + 1] for i, x in enumerate(argv) if x == "-v"]
    assert mounts == [f"{d}:{D.inside(d)}" for d in (str(tmp_path / "a"), str(tmp_path / "b"))]


def test_windows_paths_as_the_container_sees_them(monkeypatch):
    monkeypatch.setattr(D.os, "name", "nt")
    assert D.inside("C:\\Users\\lab\\w") == "/c/Users/lab/w"
    assert D.translate(["--model", "D:\\m\\world.sdf", "--dt", "0.1"]) == ["--model", "/d/m/world.sdf", "--dt", "0.1"]
    assert D.env_value("C:\\a;D:\\b") == "/c/a:/d/b"   # a Windows path list becomes a Linux one
    monkeypatch.setattr(D.os, "name", "posix")
    assert D.translate(["/x/y"]) == ["/x/y"] and D.inside("/x") == "/x"


def test_work_folders_go_where_docker_can_see_them(tmp_path, monkeypatch):
    shared = tmp_path / "shared"
    monkeypatch.setattr(D, "SHARED_TMP", str(shared))
    shared.mkdir()
    (shared / "old").mkdir()
    os.utime(shared / "old", (0, 0))
    (shared / "new").mkdir()
    D.use_shared_temp()
    assert tempfile.mkdtemp().startswith(str(shared))
    assert not (shared / "old").exists() and (shared / "new").exists()   # older than two days: removed


def test_folders_the_model_points_at_are_mounted_too(tmp_path):
    meshes = tmp_path / "meshes"
    meshes.mkdir()
    (meshes / "hull.dae").write_text("")
    (tmp_path / "w").mkdir()
    (tmp_path / "w" / "world.sdf").write_text(f'<uri>file://{meshes}/hull.dae</uri><mesh filename="/no/such.obj"/>'
                                              f'<include file="{meshes}"/>')
    (tmp_path / "w" / "windows.csv").write_text(f'"{tmp_path}"')   # not a model file: not read
    assert D.referenced_dirs(str(tmp_path / "w")) == {str(meshes)}


def test_a_folder_docker_cant_see_is_named_with_the_fix(tmp_path, monkeypatch):
    """Docker mounts an empty folder when it can't share one (Colima and the system temp folder): caught up front."""
    seen, calls = tmp_path / "seen", []
    hidden = tmp_path / "hidden"
    seen.mkdir()
    hidden.mkdir()

    def fake_run(argv, timeout=60):
        calls.append(argv)
        mounted = [_host(argv[i + 1]) for i, x in enumerate(argv) if x == "-v"]
        assert all(os.path.exists(os.path.join(d, f".strathe_docker_check_{os.getpid()}")) for d in mounted)
        return subprocess.CompletedProcess(argv, 0, f"{hidden}\n" if str(hidden) in mounted else "", "")
    monkeypatch.setattr(D, "_run", fake_run)
    h = re.escape(str(hidden))
    with pytest.raises(D.DockerError, match=rf"can't see {h}.*colima start --mount {h}:w.*File sharing"):
        D.check_shared("img", [str(seen), str(hidden)])
    assert not list(seen.iterdir()) and not list(hidden.iterdir())   # the marker files are removed
    D.check_shared("img", [str(seen)])
    D.check_shared("img", [str(seen / "sub"), str(seen)])   # checked once: no new container
    assert len(calls) == 2 and calls[0][:5] == ["docker", "run", "--rm", "--entrypoint", "sh"]


def test_gazebo_uses_the_image_when_the_tool_isnt_built(tmp_path, monkeypatch, capsys):
    _docker(monkeypatch)
    monkeypatch.setattr(D, "auto_image", lambda sim: D.IMAGES.get(sim))
    monkeypatch.delenv("STRATHE_GZ_REPLAY", raising=False)
    argv = RT.find(None, "STRATHE_GZ_REPLAY", str(tmp_path / "not_built"), "Gazebo", "Build it", sim="gazebo")
    assert argv == ["docker:strathe-gz-replay"] and "Using Strathe's Docker image" in capsys.readouterr().out
    assert tempfile.gettempdir() == D.SHARED_TMP
    assert RT.find("docker", "STRATHE_GZ_REPLAY", None, "Gazebo", "", sim="gazebo") == ["docker:strathe-gz-replay"]
    monkeypatch.setattr(D, "auto_image", lambda sim: None)
    with pytest.raises(SimFitError, match="setup-sim gazebo --docker"):
        RT.find(None, "STRATHE_GZ_REPLAY", str(tmp_path / "not_built"), "Gazebo", "Build it", sim="gazebo")
    _docker(monkeypatch, images=())
    with pytest.raises(SimFitError, match="isn't built yet.*setup-sim gazebo --docker"):
        RT.tool_argv("docker", "gazebo")


def test_stonefish_uses_the_image_when_the_tool_isnt_built(tmp_path, monkeypatch, capsys):
    _docker(monkeypatch)
    monkeypatch.setattr(SF, "DEFAULT_TOOL", str(tmp_path / "not_built"))
    monkeypatch.delenv("STRATHE_STONEFISH_REPLAY", raising=False)
    monkeypatch.setattr(D, "auto_image", lambda sim: D.IMAGES.get(sim))
    assert SF.find_tool(None) == ["docker:strathe-stonefish-replay"]
    assert "Stonefish replay tool isn't built here" in capsys.readouterr().out
    assert SF.find_tool("docker") == ["docker:strathe-stonefish-replay"]
    monkeypatch.setattr(D, "auto_image", lambda sim: None)
    with pytest.raises(SF.StonefishFitError, match="setup-sim stonefish --docker"):
        SF.find_tool(None)


def test_the_replay_runs_in_the_container_with_its_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "check_shared", lambda image, dirs: None)
    t = RT.ReplayTool(["docker:img"], (0.0, 0.0, 0.0), extra=["--model-dir", str(tmp_path / "models")],
                      env={"GZ_SIM_RESOURCE_PATH": str(tmp_path)})
    argv = t._argv(str(tmp_path / "w" / "world.sdf"), (str(tmp_path / "work"),))
    assert argv[argv.index("img") + 1:argv.index("img") + 3] == ["--model", D.inside(str(tmp_path / "w" / "world.sdf"))]
    mounts = {_host(argv[i + 1]) for i, x in enumerate(argv) if x == "-v"}
    assert mounts == {str(tmp_path / "w"), str(tmp_path / "work"), str(tmp_path / "models"), str(tmp_path)}
    assert f"GZ_SIM_RESOURCE_PATH={D.env_value(str(tmp_path))}" in argv


def test_strathe_passes_docker_through_untouched(tmp_path):
    """It used to make every --replay-tool a path: docker:IMAGE became /cwd/docker:IMAGE (#92)."""
    calls = []
    (tmp_path / "s.scn").write_text("<scenario/>")
    cli.main(["fit-stonefish", "--scenario", str(tmp_path / "s.scn"), "--robot", "r", "--part", "Hull", "--data",
              str(tmp_path), "--reference-point", "0", "0", "0", "--replay-tool", "docker"],
             runner=lambda c: calls.append(c) or 0)
    assert calls[0][calls[0].index("--replay-tool") + 1] == "docker"
    assert cli._tool("docker:lab/img") == "docker:lab/img" and os.path.isabs(cli._tool("build/tool"))


def test_setup_sim_docker_builds_the_image_and_checks_it(monkeypatch, capsys):
    _docker(monkeypatch)
    built = []
    monkeypatch.setattr(D, "build", lambda sim: built.append(sim) or 0)
    monkeypatch.setattr(D, "auto_image", lambda sim: D.IMAGES[sim])
    monkeypatch.setattr(D, "features", lambda image: {"simulator": "gazebo", "gz_sim": "8.15.0"})
    assert cli.main(["setup-sim", "gazebo", "--docker"]) == 0
    out = capsys.readouterr().out
    assert built == ["gazebo"] and "✅ Gazebo replay tool in Docker (strathe-gz-replay) (gz-sim 8.15.0)" in out
    assert "--replay-tool docker" in out
    monkeypatch.setattr(D, "features", lambda image: {"patched": True, "axis_damping": True})
    assert cli.main(["setup-sim", "stonefish", "--docker"]) == 0
    assert "Stonefish replay tool in Docker (strathe-stonefish-replay), with the patch" in capsys.readouterr().out
    monkeypatch.setattr(D, "build", lambda sim: 1)
    assert cli.main(["setup-sim", "gazebo", "--docker"]) == 2
    assert "docker system df" in capsys.readouterr().out
    _docker(monkeypatch, "stopped")
    assert cli.main(["setup-sim", "gazebo", "--docker"]) == 2 and "isn't running" in capsys.readouterr().out
    assert cli.main(["setup-sim", "mujoco", "--docker"]) == 0 and "no Docker needed" in capsys.readouterr().out


def test_strathe_check_reports_docker_and_the_images(monkeypatch, tmp_path):
    _docker(monkeypatch, "missing")
    assert "Docker not installed (optional" in D.check_line()
    _docker(monkeypatch, "stopped")
    assert "not running" in D.check_line()
    _docker(monkeypatch, images=("strathe-gz-replay",))
    assert D.check_line().startswith("✅ Docker 29.5.2, running; Strathe's images built: gazebo (python strathe.py "
                                     "setup-sim stonefish --docker)")
    assert "not built" in cli.gazebo_status(str(tmp_path / "none"))   # auto_image is off in tests
    monkeypatch.setattr(D, "auto_image", lambda sim: D.IMAGES[sim])
    monkeypatch.setattr(D, "features", lambda image: {"gz_sim": "8.15.0"})
    assert cli.gazebo_status(str(tmp_path / "none")) == "✅ Gazebo replay tool in Docker (strathe-gz-replay) (gz-sim 8.15.0)"


def test_the_web_page_builds_an_image_and_takes_docker_as_the_tool(tmp_path, monkeypatch):
    import strathe_web as web
    _docker(monkeypatch)
    started = []
    m = web.JobManager(str(tmp_path), launcher=lambda cmd, log: started.append(cmd))
    page = web.home_page(m, "t").decode()
    assert "nothing to build (Docker)" in page and "action=/jobs/setup_docker" in page
    h = web.StratheHandler.__new__(web.StratheHandler)
    h.manager = m
    job = h._start("/jobs/setup_docker", {"sim": "gazebo"})
    assert started[0][-3:] == ["setup-sim", "gazebo", "--docker"] and job.kind == "setup_docker"
    with pytest.raises(ValueError, match="Stonefish or Gazebo"):
        h._start("/jobs/setup_docker", {"sim": "mujoco"})
    assert web._replay_tool_args({"replay_tool": "docker"}) == ["--replay-tool", "docker"]
    with pytest.raises(ValueError, match="docker for Strathe's Docker image"):
        web._replay_tool_args({"replay_tool": str(tmp_path / "none")})


def test_run_me_says_gazebo_must_be_installed_to_look_at_a_docker_fit():
    from simfit.adapters.gazebo import GazeboAdapter
    from test_simfit_gazebo import WORLD
    assert "ran Gazebo in Docker" in GazeboAdapter(WORLD, tool=["docker:strathe-gz-replay"]).run_instructions("w.sdf")
    assert "Docker" not in GazeboAdapter(WORLD, tool=["/opt/strathe_gz_replay"]).run_instructions("w.sdf")
