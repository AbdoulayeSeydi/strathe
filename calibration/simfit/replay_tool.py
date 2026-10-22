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

A tool is a path, a command (`python my_tool.py`), or `docker:IMAGE`: the tool then runs in that image (`docker run`),
with the model's and the work folders mounted at the same paths, so nothing has to be built.
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

from simfit.core import SimFitError

IN_COLUMNS = "win,k,u,v,w,p,q,r,qx,qy,qz,qw,Fx,Fy,Fz,Tx,Ty,Tz"
OUT_COLUMNS = ("u", "v", "w", "p", "q", "r", "qx", "qy", "qz", "qw")
INFO_TIMEOUT_S, RUN_TIMEOUT_S = 600, 24 * 3600


def write_windows(path: str, rows_by_window: list) -> None:
    with open(path, "w") as f:
        f.write(IN_COLUMNS + "\n")
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


def tool_argv(spec: str) -> list:
    """A --replay-tool value -> argv prefix. `docker:IMAGE` is kept as is (ReplayTool mounts the folders)."""
    if spec.startswith("docker:"):
        if not shutil.which("docker"):
            raise SimFitError("--replay-tool docker:... needs Docker (docker.com); or build the tool (the simulator's "
                              "guide in calibration/simulators/)")
        return [spec]
    if os.path.exists(spec):
        if not os.access(spec, os.X_OK):
            raise SimFitError(f"the replay tool {spec} isn't executable (chmod +x, or give it as 'python {spec}')")
        return [spec]
    argv = shlex.split(spec)
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

    def _argv(self, model: str, folders=()) -> list:
        base = ["--model", model, "--ref", *(f"{x:g}" for x in self.ref), "--dt", f"{self.dt:g}", *self.extra]
        if self.tool and self.tool[0].startswith("docker:"):
            # the model's and work folders, and any folder the environment points at (model paths), at the same paths
            dirs = {os.path.dirname(os.path.abspath(model)), *folders}
            for v in self.env.values():
                dirs |= {d for d in str(v).split(os.pathsep) if os.path.isabs(d) and os.path.isdir(d)}
            if "--model-dir" in self.extra:
                dirs.add(self.extra[self.extra.index("--model-dir") + 1])
            mounts = [x for d in sorted(dirs) for x in ("-v", f"{d}:{d}")]
            envs = [x for k, v in self.env.items() for x in ("-e", f"{k}={v}")]
            return ["docker", "run", "--rm", *mounts, *envs, self.tool[0][len("docker:"):], *base]
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
            argv = ["docker", "run", "--rm", self.tool[0][len("docker:"):], "--features"]
        else:
            argv = [*self.tool, "--features"]
        r = self._call(argv, 120, "answer")
        js = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if r.returncode or not js:
            raise SimFitError(f"the {self.name} replay tool didn't report its features: {r.stderr.strip()[-400:]}")
        return json.loads(js[-1])

    def info(self, model: str) -> dict:
        r = self._call(self._argv(model) + ["--info"], INFO_TIMEOUT_S, "load the model")
        js = [ln for ln in r.stdout.splitlines() if ln.startswith("{")]
        if r.returncode or not js:
            raise SimFitError(f"the {self.name} replay tool couldn't load the model: {r.stderr.strip()[-800:]}")
        return json.loads(js[-1])

    def run(self, model: str, rows_by_window: list, work_dir: str) -> list:
        os.makedirs(work_dir, exist_ok=True)
        fin, fout = os.path.join(work_dir, "windows.csv"), os.path.join(work_dir, "pred.csv")
        write_windows(fin, rows_by_window)
        r = self._call(self._argv(model, (os.path.abspath(work_dir),)) + ["--windows", fin, "--out", fout],
                       RUN_TIMEOUT_S, "finish the replay")
        if r.returncode or not os.path.exists(fout):
            raise SimFitError(f"the {self.name} replay failed: {r.stderr.strip()[-800:]}")
        return read_predictions(fout, rows_by_window)


def find(spec: Optional[str], env: str, default: Optional[str], what: str, guide: str) -> list:
    """The replay tool: --replay-tool, else the environment variable, else the default build location."""
    s = spec or os.environ.get(env) or default
    if not s or (not s.startswith("docker:") and " " not in s and not os.path.exists(s)):
        raise SimFitError(f"the {what} replay tool isn't built{f' ({s})' if s else ''}. {guide}, or give "
                          f"--replay-tool PATH (or docker:IMAGE)")
    return tool_argv(s)
