"""
simfit/replay_tool.py -- run a simulator's replay command (PROJECT_STATUS.md #81). The contract is written up for labs
in calibration/simulators/SIMULATOR_PROTOCOL.md; Stonefish's tool (calibration/stonefish/) was the first.

    TOOL --model MODEL --ref X Y Z --dt DT [simulator's own options] --features
    TOOL --model MODEL --ref X Y Z --dt DT [...] --info
    TOOL --model MODEL --ref X Y Z --dt DT [...] --windows IN.csv --out OUT.csv

  IN.csv   win,k,u,v,w,p,q,r,qx,qy,qz,qw,Fx,Fy,Fz,Tx,Ty,Tz   (one row per logged sample of each window)
  OUT.csv  win,k,u,v,w,p,q,r,qx,qy,qz,qw                     (k = 0 is the logged start; stop a window if it diverges)
  Velocity: body axes forward-right-down at the reference point (m/s, rad/s); quaternion: body -> world north-east-down
  (x, y, z, w); wrench: body axes, about the reference point (N, N m), held over each sample. --info prints one JSON
  line (whatever the simulator knows: mass, version, ...); --features one JSON line without a model.

A tool is a path, a command (`python my_tool.py`), `docker:IMAGE`, or `docker` (Strathe's image for that simulator): the
tool then runs in that image (`docker run`), with the model's and the work folders mounted at the same paths, so nothing
has to be built (sim_docker.py; calibration/simulators/DOCKER.md).
"""
from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

import sim_docker
from simfit.core import SimFitError

IN_COLUMNS = "win,k,u,v,w,p,q,r,qx,qy,qz,qw,Fx,Fy,Fz,Tx,Ty,Tz"
OUT_COLUMNS = ("u", "v", "w", "p", "q", "r", "qx", "qy", "qz", "qw")
INFO_TIMEOUT_S, RUN_TIMEOUT_S = 600, 24 * 3600


def _windows(windows: Optional[bool]) -> bool:
    return os.name == "nt" if windows is None else windows


def split_command(spec: str, windows: Optional[bool] = None) -> list:
    """A command line -> argv. On Windows a backslash is a path separator, not an escape, and quotes group as in cmd."""
    if not _windows(windows):
        return shlex.split(spec)
    return [p[1:-1] if len(p) > 1 and p[0] == p[-1] and p[0] in "\"'" else p for p in shlex.split(spec, posix=False)]


def join_command(parts: list, windows: Optional[bool] = None) -> str:
    """argv -> a command line that split_command reads back."""
    return subprocess.list2cmdline(parts) if _windows(windows) else " ".join(shlex.quote(p) for p in parts)


def write_windows(path: str, rows_by_window: list) -> None:
    """A fin vehicle's windows carry each fin's angle after the wrench, named fin_<name> (radians)."""
    extra = getattr(rows_by_window, "fins", ())
    with open(path, "w") as f:
        f.write(",".join((IN_COLUMNS, *extra)) + "\n")
        for i, rows in enumerate(rows_by_window):
            for k, row in enumerate(rows):
                f.write(f"{i},{k}," + ",".join(f"{x:.9g}" for x in row) + "\n")


def read_predictions(path: str, rows_by_window: list) -> list:
    """[(V (n,6), Q (n,4))] per window; NaN rows past where the simulator stopped (diverged)."""
    got = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            got.setdefault(int(row["win"]), []).append([float(row[c]) for c in OUT_COLUMNS])
    out = []
    for i, rows in enumerate(rows_by_window):
        arr = np.full((len(rows), 10), np.nan)
        g = np.array(got.get(i, []), dtype=float).reshape(-1, 10)[:len(rows)]
        arr[:len(g)] = g
        out.append((arr[:, :6], arr[:, 6:]))
    return out


def why_failed(r) -> str:
    """The end of what a replay said. Stonefish reports on stdout, so an empty stderr doesn't mean it said nothing."""
    said = (r.stderr or "").strip() or (r.stdout or "").strip()
    return said[-800:] or f"it stopped with exit code {r.returncode} and said nothing"


def tool_argv(spec: str, sim: Optional[str] = None) -> list:
    """A --replay-tool value -> argv prefix. `docker` / `docker:IMAGE` -> ["docker:IMAGE"] (ReplayTool mounts the
    folders), once Docker is running and the image is there; work folders then go where Docker can see them."""
    if sim_docker.is_docker(spec):
        try:
            image = sim_docker.image_for(spec, sim)
            sim_docker.require(image, sim)
        except sim_docker.DockerError as e:
            raise SimFitError(str(e))
        sim_docker.use_shared_temp()
        return ["docker:" + image]
    if os.path.exists(spec):
        if not os.access(spec, os.X_OK):
            raise SimFitError(f"the replay tool {spec} isn't executable (chmod +x, or give it as 'python {spec}')")
        return [spec]
    argv = split_command(spec)
    if not argv or not (shutil.which(argv[0]) or os.path.exists(argv[0])):
        raise SimFitError(f"the replay tool {spec!r} wasn't found: give its path, a command, or docker:IMAGE")
    return argv


@dataclass
class ReplayTool:
    tool: list                       # argv prefix, or ["docker:IMAGE"]
    ref: tuple                       # the logs' reference point in the model's frame (FRD, m)
    dt: float = 0.1
    extra: list = field(default_factory=list)   # the simulator's own options
    name: str = "simulator"          # for messages
    env: dict = field(default_factory=dict)     # extra environment for the command
    _fins: Optional[bool] = field(default=None, repr=False)

    def _argv(self, model: str, folders=()) -> list:
        base = ["--model", model, "--ref", *(f"{x:g}" for x in self.ref), "--dt", f"{self.dt:g}", *self.extra]
        if self.tool and self.tool[0].startswith("docker:"):
            # the model's and work folders, and any folder the environment points at (model paths), at the same paths
            dirs = {os.path.dirname(os.path.abspath(model)), *folders}
            for v in self.env.values():
                dirs |= {d for d in str(v).split(os.pathsep) if os.path.isabs(d) and os.path.isdir(d)}
            if "--model-dir" in self.extra:
                dirs.add(self.extra[self.extra.index("--model-dir") + 1])
            dirs |= sim_docker.referenced_dirs(os.path.dirname(os.path.abspath(model)))
            image = self.tool[0][len("docker:"):]
            try:
                sim_docker.check_shared(image, dirs)
            except sim_docker.DockerError as e:
                raise SimFitError(str(e))
            return sim_docker.run_prefix(image, dirs, self.env) + sim_docker.translate(base)
        return self.tool + base

    def _call(self, argv: list, timeout: int, what: str):
        try:
            env = {**os.environ, **self.env} if self.env else None
            return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            raise SimFitError(f"the {self.name} replay tool didn't {what} within {timeout // 60} minutes; is the machine "
                              f"asleep or overloaded? Try fewer --workers")
        except OSError as e:
            raise SimFitError(f"the {self.name} replay tool couldn't start: {e}")

    def features(self) -> dict:
        if self.tool[0].startswith("docker:"):
            argv = sim_docker.run_prefix(self.tool[0][len("docker:"):]) + ["--features"]
        else:
            argv = [*self.tool, "--features"]
        r = self._call(argv, 120, "answer")
        js = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if r.returncode or not js:
            raise SimFitError(f"the {self.name} replay tool didn't report its features: {r.stderr.strip()[-400:]}")
        return json.loads(js[-1])

    def takes_fins(self) -> bool:
        if self._fins is None:
            self._fins = bool(self.features().get("fins"))
        return self._fins

    def info(self, model: str) -> dict:
        r = self._call(self._argv(model) + ["--info"], INFO_TIMEOUT_S, "load the model")
        js = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if r.returncode or not js:
            raise SimFitError(f"the {self.name} replay tool couldn't load the model: {r.stderr.strip()[-800:]}")
        return json.loads(js[-1])

    def run(self, model: str, rows_by_window: list, work_dir: str) -> list:
        os.makedirs(work_dir, exist_ok=True)
        if getattr(rows_by_window, "fins", ()) and not self.takes_fins():
            raise SimFitError(f"the {self.name} replay tool doesn't take fin angles: its --features must say \"fins\": true and it "
                              f"must apply fin_<name> columns (calibration/simulators/SIMULATOR_PROTOCOL.md)")
        fin, fout = os.path.join(work_dir, "windows.csv"), os.path.join(work_dir, "pred.csv")
        write_windows(fin, rows_by_window)
        io = ["--windows", fin, "--out", fout]
        r = self._call(self._argv(model, (os.path.abspath(work_dir),))
                       + (sim_docker.translate(io) if self.tool[0].startswith("docker:") else io),
                       RUN_TIMEOUT_S, "finish the replay")
        if r.returncode or not os.path.exists(fout):
            raise SimFitError(f"the {self.name} replay failed: {why_failed(r)}")
        return read_predictions(fout, rows_by_window)


def find(spec: Optional[str], env: str, default: Optional[str], what: str, guide: str, sim: Optional[str] = None) -> list:
    """The replay tool: --replay-tool, else the environment variable, else the default build location, else Strathe's
    Docker image for this simulator when it's built (the tool then needn't be built here)."""
    s = spec or os.environ.get(env) or default
    if not spec and not os.environ.get(env) and s and not os.path.exists(s) and sim:
        image = sim_docker.auto_image(sim)
        if image:
            print(f"Using Strathe's Docker image {image}: the {what} replay tool isn't built here "
                  f"({sim_docker.GUIDE})", flush=True)
            s = "docker"
    if not s or (not sim_docker.is_docker(s) and " " not in s and not os.path.exists(s)):
        raise SimFitError(f"the {what} replay tool isn't built{f' ({s})' if s else ''}. {guide}; or, with nothing "
                          f"to build, use Docker: python strathe.py setup-sim {sim or 'SIMULATOR'} --docker. Or give "
                          f"--replay-tool PATH (or docker:IMAGE)")
    return tool_argv(s, sim)
