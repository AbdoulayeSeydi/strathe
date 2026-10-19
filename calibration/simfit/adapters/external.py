"""
simfit/adapters/external.py -- any simulator, described by the lab (PROJECT_STATUS.md #81). The lab writes a small
YAML (calibration/simulators/SIMULATOR_PROTOCOL.md):

    simulator: my_sim              # a name for the reports
    replay: python my_replay.py    # a command speaking the replay contract (a path, a command, or docker:IMAGE)
    model: vehicle.sdf             # the vehicle's model file (relative to this YAML)
    format: xml                    # xml | json | yaml | regex (default: from the extension)
    options: [--depth, "3"]        # extra options for the replay command (optional)
    density: "//fluid/density"    # where the water density is (optional: --density-by-date writes it)
    knobs:                         # what Strathe may tune, where it is, and its limits
      - {name: drag_x, group: drag, at: "//xUabsU", sign: -1}                       # x1/30 to x30 of the magnitude
      - {name: mass, group: mass, at: "//inertial/mass", transform: log_factor, lo: 1.2, hi: 1.2}
      - {name: cg_x, group: cg, at: "//inertial/pose#0", transform: linear, lo: -0.1, hi: 0.1}
    export:                        # optional: where Strathe's values go (export-sim, --from-strathe)
      - {param: Xuu, at: "//xUabsU"}
      - {param: Xu_dot, at: "//xDotU", scale: -1}
"""
from __future__ import annotations

import os

import yaml

from simfit.adapters import ExportDef, FileAdapter, KnobDef
from simfit.core import SimFitError
from simfit.replay_tool import ReplayTool, tool_argv

KNOB_KEYS = {"name", "group", "at", "transform", "sign", "lo", "hi", "axisym", "note"}
EXPORT_KEYS = {"param", "at", "scale", "offset"}


def load_description(path: str) -> dict:
    if not os.path.exists(path):
        raise SimFitError(f"the simulator description {path} doesn't exist")
    try:
        d = yaml.safe_load(open(path, encoding="utf-8")) or {}
    except yaml.YAMLError as e:
        raise SimFitError(f"{path} isn't valid YAML: {e}")
    for key in ("replay", "model", "knobs"):
        if key not in d:
            raise SimFitError(f"{path}: needs `{key}:` (calibration/simulators/SIMULATOR_PROTOCOL.md)")
    for i, k in enumerate(d["knobs"]):
        bad = set(k) - KNOB_KEYS
        if bad or not {"name", "group", "at"} <= set(k):
            raise SimFitError(f"{path}: knob {i + 1} needs name, group and at" + (f"; unknown keys {sorted(bad)}" if bad else ""))
    for i, e in enumerate(d.get("export") or []):
        if set(e) - EXPORT_KEYS or not {"param", "at"} <= set(e):
            raise SimFitError(f"{path}: export entry {i + 1} needs param and at (and optionally scale, offset)")
    return d


def resolve_command(replay, here: str) -> str:
    """A replay command from a description: parts that are files next to the description get its folder."""
    import shlex
    if isinstance(replay, list):
        parts = [str(p) for p in replay]
    elif str(replay).startswith("docker:"):
        return str(replay)
    else:
        parts = shlex.split(str(replay))
    parts = [os.path.join(here, p) if (not os.path.isabs(p) and os.path.exists(os.path.join(here, p))) else p
             for p in parts]
    return " ".join(shlex.quote(p) for p in parts)


def adapter_from_args(a) -> FileAdapter:
    d = load_description(a.simulator)
    here = os.path.dirname(os.path.abspath(a.simulator))
    model = a.model or os.path.join(here, d["model"])
    replay = a.replay_tool or d["replay"]
    if not a.replay_tool:  # relative paths in the description's command are relative to the description
        replay = resolve_command(replay, here)
    name = str(d.get("simulator", "external")).strip() or "external"
    sim = "".join(c if c.isalnum() else "_" for c in name.lower())
    tool = ReplayTool(tool_argv(replay), tuple(a.reference_point), extra=[str(x) for x in d.get("options") or []]
                      + ["--model-dir", os.path.dirname(os.path.abspath(model))], name=name)
    knobs = [KnobDef(**{k: v for k, v in kd.items()}) for kd in d["knobs"]]
    exports = [ExportDef(**e) for e in d.get("export") or []]
    return FileAdapter(sim, name, model, tool, knobs, exports, d.get("format"), d.get("density"))


def add_arguments(ap) -> None:
    ap.add_argument("--simulator", required=True, metavar="DESCRIPTION.yaml",
                    help="your simulator's description: the replay command, the model file, the knobs "
                         "(calibration/simulators/SIMULATOR_PROTOCOL.md)")
    ap.add_argument("--model", help="the model file (default: the description's)")
    ap.add_argument("--replay-tool", help="the replay command (default: the description's)")
