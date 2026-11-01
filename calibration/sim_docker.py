"""
sim_docker.py -- run a simulator's replay tool in Docker, so a lab has nothing to build (PROJECT_STATUS.md #92;
calibration/simulators/DOCKER.md).

Stonefish and Gazebo each have an image Strathe builds once (`strathe setup-sim stonefish --docker`, `... gazebo
--docker`). Then `--replay-tool docker` uses it, and so does every Stonefish and Gazebo command when the replay tool
isn't built on this machine. `--replay-tool docker:IMAGE` runs any image that speaks the replay contract
(calibration/simulators/SIMULATOR_PROTOCOL.md).

How a replay runs: `docker run --rm` with the model's folder, the simulator's data folders and the work folder mounted
at the same paths (on Windows, C:\\x is /c/x inside), as the user running Strathe (so the files it writes are theirs),
with HOME=/tmp. Work folders go under ~/.strathe/tmp: every Docker set-up shares the home folder, and Colima doesn't
share the system temp folder. The first replay checks Docker can see each folder, and says how to share one it can't.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Optional

CAL_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES = {"stonefish": "strathe-stonefish-replay", "gazebo": "strathe-gz-replay"}
_BUILD = {"stonefish": "stonefish", "gazebo": os.path.join("simulators", "gazebo")}   # build context, under CAL_DIR
BUILD_MINUTES = {"stonefish": "5-20", "gazebo": "3-10"}   # 8 and 4 on an M1 MacBook Air (#92)
SHARED_TMP = os.path.join(os.path.expanduser("~"), ".strathe", "tmp")
GUIDE = "calibration/simulators/DOCKER.md"

NOT_INSTALLED = (f"Docker isn't installed. Install one of: Docker Desktop (docker.com/products/docker-desktop; Mac, "
                 f"Windows, Linux), Colima on a Mac without admin rights (brew install colima docker && colima start), "
                 f"or Docker Engine on Linux (docs.docker.com/engine/install). See {GUIDE}")
NOT_RUNNING = ("Docker is installed but its engine isn't running: open Docker Desktop, or run `colima start` if you use "
               "Colima, or `sudo systemctl start docker` on Linux")


class DockerError(RuntimeError):
    pass


def _run(argv: list, timeout: float = 60) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return subprocess.CompletedProcess(argv, 1, "", str(e))


def state() -> tuple:
    """("missing" | "stopped" | "running", server version or "")."""
    if not shutil.which("docker"):
        return "missing", ""
    r = _run(["docker", "version", "--format", "{{.Server.Version}}"], 30)
    return ("running", r.stdout.strip()) if r.returncode == 0 and r.stdout.strip() else ("stopped", "")


def image_present(image: str) -> bool:
    return _run(["docker", "image", "inspect", image], 30).returncode == 0


def image_for(spec: str, sim: Optional[str]) -> str:
    """`docker` -> Strathe's image for that simulator; `docker:IMAGE` -> IMAGE."""
    if spec == "docker":
        if sim not in IMAGES:
            raise DockerError("--replay-tool docker uses Strathe's image, and there's one for Stonefish and Gazebo; for "
                              "your own simulator give your image: --replay-tool docker:IMAGE")
        return IMAGES[sim]
    return spec[len("docker:"):]


def is_docker(spec: Optional[str]) -> bool:
    return bool(spec) and (spec == "docker" or spec.startswith("docker:"))


def require(image: str, sim: Optional[str] = None) -> None:
    """Docker is running and the image is there, or a DockerError that says what to do."""
    st, _ = state()
    if st != "running":
        raise DockerError(NOT_INSTALLED if st == "missing" else NOT_RUNNING)
    if not image_present(image):
        if sim in IMAGES and image == IMAGES[sim]:
            raise DockerError(f"Strathe's {sim} image ({image}) isn't built yet. Build it once (about "
                              f"{BUILD_MINUTES[sim]} minutes): python strathe.py setup-sim {sim} --docker")
        raise DockerError(f"the Docker image {image} isn't on this machine: build it (docker build -t {image} ...) "
                          f"or pull it (docker pull {image})")


def auto_image(sim: str) -> Optional[str]:
    """Strathe's image for this simulator, if Docker is running and it's built; else None (quietly)."""
    if sim not in IMAGES or not shutil.which("docker"):
        return None
    return IMAGES[sim] if image_present(IMAGES[sim]) else None


def use_shared_temp() -> str:
    """Work folders under ~/.strathe/tmp from now on (folders older than two days there are removed)."""
    os.makedirs(SHARED_TMP, exist_ok=True)
    old = time.time() - 2 * 86400
    for name in os.listdir(SHARED_TMP):
        p = os.path.join(SHARED_TMP, name)
        try:
            if os.path.getmtime(p) < old:
                shutil.rmtree(p, ignore_errors=True) if os.path.isdir(p) else os.remove(p)
        except OSError:
            pass
    tempfile.tempdir = SHARED_TMP
    return SHARED_TMP


# --- paths inside the container ----------------------------------------------------------------------------------------
_WIN_ABS = re.compile(r"^([A-Za-z]):[\\/]")


def inside(path: str) -> str:
    """A host path as the container sees it: the same path, except C:\\x\\y -> /c/x/y on Windows."""
    m = _WIN_ABS.match(path)
    if os.name == "nt" and m:
        return "/" + m.group(1).lower() + "/" + path[3:].replace("\\", "/")
    return path


def translate(args: list) -> list:
    """Arguments with Windows paths (also inside A;B lists) as the container sees them; unchanged elsewhere."""
    if os.name != "nt":
        return list(args)
    return [";".join(inside(p) for p in str(a).split(";")) if _WIN_ABS.match(str(a)) else a for a in args]


def env_value(v: str) -> str:
    """An environment value for the (Linux) container: on Windows, A;B path lists become /a:/b."""
    return ":".join(inside(p) for p in v.split(";")) if os.name == "nt" and _WIN_ABS.match(v) else v


def run_prefix(image: str, dirs=(), env: Optional[dict] = None) -> list:
    """`docker run` up to the image name: the folders mounted, the user's own id, the environment."""
    user = ["--user", f"{os.getuid()}:{os.getgid()}"] if hasattr(os, "getuid") else []
    mounts = [x for d in sorted({os.path.abspath(d) for d in dirs}) for x in ("-v", f"{d}:{inside(d)}")]
    envs = [x for k, v in (env or {}).items() for x in ("-e", f"{k}={env_value(str(v))}")]
    return ["docker", "run", "--rm", *user, "-e", "HOME=/tmp", *mounts, *envs, image]


_ABS_IN_TEXT = re.compile(r"""(?:["'>]|file://)((?:/|[A-Za-z]:[\\/])[^"'<>\s]*)""")


def referenced_dirs(folder: str) -> set:
    """Folders that the model files Strathe wrote in `folder` point at by absolute path (meshes, included models), so
    they're mounted too."""
    found = set()
    for name in os.listdir(folder) if os.path.isdir(folder) else []:
        if not name.lower().endswith((".scn", ".sdf", ".world", ".xml", ".urdf", ".config")):
            continue
        try:
            text = open(os.path.join(folder, name), encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in _ABS_IN_TEXT.finditer(text):
            p = m.group(1)
            if len(p) > 1 and os.path.exists(p):
                d = p if os.path.isdir(p) else os.path.dirname(p)
                if d not in ("/", "") and os.path.dirname(d) != d:
                    found.add(os.path.abspath(d))
    return found


# --- can Docker see the folders? ---------------------------------------------------------------------------------------
_seen: set = set()


def check_shared(image: str, dirs) -> None:
    """Once per folder: a marker file written here must be visible in the container. Docker otherwise mounts an empty
    folder silently, and the replay's results would vanish."""
    todo = [os.path.abspath(d) for d in dirs
            if not any(os.path.abspath(d) == s or os.path.abspath(d).startswith(s + os.sep) for s in _seen)]
    todo = sorted({d for d in todo if os.path.isdir(d)})
    if not todo:
        return
    mark = f".strathe_docker_check_{os.getpid()}"
    made = []
    for d in todo:
        try:
            open(os.path.join(d, mark), "w").close()
            made.append(d)
        except OSError:   # a read-only folder: can't check it here
            pass
    try:
        if not made:
            return
        script = "; ".join(f"[ -f '{inside(d)}/{mark}' ] || echo '{d}'" for d in made)
        r = _run(["docker", "run", "--rm", "--entrypoint", "sh",
                  *[x for d in made for x in ("-v", f"{d}:{inside(d)}")], image, "-c", script], 120)
        if r.returncode:
            raise DockerError(f"Docker couldn't start {image}: {(r.stderr or r.stdout).strip()[-400:]}")
        missing = [ln.strip() for ln in r.stdout.splitlines() if ln.strip()]
        if missing:
            raise DockerError(
                f"Docker can't see {missing[0]}: it mounts an empty folder instead, so the replay would lose its "
                f"files. Colima shares only your home folder by default: move the folder under {os.path.expanduser('~')}, "
                f"or restart Colima sharing it (colima stop && colima start --mount {missing[0]}:w). Docker Desktop: "
                f"Settings > Resources > File sharing. ({GUIDE})")
        _seen.update(made)
    finally:
        for d in made:
            try:
                os.remove(os.path.join(d, mark))
            except OSError:
                pass


def features(image: str) -> dict:
    """The tool's --features line, run in the image."""
    r = _run(run_prefix(image) + ["--features"], 180)
    js = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
    if r.returncode or not js:
        raise DockerError(f"the replay tool in {image} didn't report its features: "
                          f"{(r.stderr or r.stdout).strip()[-400:]}")
    return json.loads(js[-1])


def build(sim: str) -> int:
    """`docker build` Strathe's image for this simulator, its output on the terminal. Returns docker's exit code."""
    ctx = os.path.join(CAL_DIR, _BUILD[sim])
    return subprocess.run(["docker", "build", "-t", IMAGES[sim], "-f", os.path.join(ctx, "Dockerfile"), ctx]).returncode


def check_line() -> str:
    """One line for `strathe check`."""
    st, ver = state()
    if st == "missing":
        return f"➖ Docker not installed (optional: runs Stonefish and Gazebo with nothing to build; {GUIDE})"
    if st == "stopped":
        return f"⚠️  Docker is installed but not running (optional; {NOT_RUNNING})"
    built = [s for s, img in IMAGES.items() if image_present(img)]
    return (f"✅ Docker {ver}, running; Strathe's images built: {', '.join(built) or 'none yet'}"
            + ("" if len(built) == len(IMAGES) else
               f" (python strathe.py setup-sim {' / '.join(s for s in IMAGES if s not in built)} --docker)"))
