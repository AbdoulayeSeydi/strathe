"""
simfit/adapters -- one per simulator (PROJECT_STATUS.md #81). `strathe fit-sim --sim NAME` / `export-sim` pick one.

Most simulators keep a vehicle in a model file and can replay logged windows through a command (the replay contract,
simfit/replay_tool.py): FileAdapter does everything for those from a list of knobs (where each tunable number is in the
file) and, for export, where each of Strathe's values goes. A simulator adapter is then mostly that list.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from simfit.core import SimFitError
from simfit.editors import fmt_number, guess_format, open_model
from simfit.knobs import KnobSpec


@dataclass
class KnobDef:
    """Where one tunable number is in the model file, and how it's fitted."""
    name: str
    group: str
    at: str                         # editor path (simfit/editors.py)
    transform: str = "log_factor"
    sign: float = 1.0               # knob value = sign x file value (fit a negative damping by its magnitude: -1)
    lo: Optional[float] = None
    hi: Optional[float] = None
    axisym: Optional[str] = None    # with --axisymmetric, knobs sharing this key take one value
    note: str = ""
    if_zero: Optional[float] = None  # where to start a log_factor knob that is 0 in the model


@dataclass
class ExportDef:
    """Where one of Strathe's calibrated values goes in the model file: file value = scale x param + offset."""
    param: str                      # a CalibrationParameters field (Xu, Xuu, Yv_dot, ...) or "mass"
    at: str
    scale: float = 1.0
    offset: float = 0.0


@dataclass
class FileAdapter:
    sim: str
    label: str
    model_path: str
    tool: object                    # a ReplayTool, or anything with info(model), run(model, rows, folder)
    knob_defs: list
    export_defs: list = field(default_factory=list)
    fmt: Optional[str] = None
    density_at: Optional[str] = None   # where the water density is in the file (--density-by-date writes it)
    notes: list = field(default_factory=list)
    base: Optional[dict] = None
    dt: float = 0.1

    def __post_init__(self):
        if not os.path.exists(self.model_path):
            raise SimFitError(f"the model file {self.model_path} doesn't exist")
        self.model_path = os.path.abspath(self.model_path)
        self.fmt = self.fmt or guess_format(self.model_path)
        self.text = open(self.model_path, encoding="utf-8").read()
        open_model(self.text, self.fmt)  # fails early with a plain message if the file can't be read
        names = [k.name for k in self.knob_defs]
        if len(set(names)) != len(names):
            raise SimFitError("two knobs share a name")

    @property
    def groups(self) -> tuple:
        out = []
        for k in self.knob_defs:
            if k.group not in out:
                out.append(k.group)
        return tuple(out)

    # values: {knob name: knob value} and/or {path: file value} ("@" + path, from export)
    def _apply(self, values: Optional[dict], rho: Optional[float] = None) -> str:
        ed = open_model(self.text, self.fmt)
        by_name = {k.name: k for k in self.knob_defs}
        for layer in (self.base, values):
            for key, v in (layer or {}).items():
                if key in by_name:
                    ed.set(by_name[key].at, float(v) * by_name[key].sign)
                elif key.startswith("@"):
                    ed.set(key[1:], float(v))
                else:
                    raise SimFitError(f"unknown setting {key!r}")
        if rho is not None and self.density_at:
            ed.set(self.density_at, rho)
        return ed.text

    def set_base(self, values: Optional[dict]) -> None:
        self.base = dict(values) if values else None

    def _file_value(self, k: KnobDef) -> float:
        return open_model(self._apply(None), self.fmt).get(k.at) * k.sign

    def knob_specs(self, info: dict, groups: tuple, axisymmetric: bool) -> list:
        specs = []
        for k in self.knob_defs:
            if k.group not in groups:
                continue
            v = self._file_value(k)
            if k.transform == "log_factor" and abs(v) < 1e-12 and not k.if_zero:
                raise SimFitError(f"knob {k.name} is 0 in your model, and a factor of 0 stays 0: give it a starting value "
                                  f"(`if_zero: VALUE` in its knob line), or use `transform: log` with lo/hi limits")
            specs.append(KnobSpec(k.name, k.group, v, k.transform, k.lo, k.hi,
                                  tie=k.axisym if (axisymmetric and k.axisym) else None, note=k.note, if_zero=k.if_zero))
        return specs

    def write_runnable(self, values: Optional[dict], folder: str, rho: Optional[float] = None) -> str:
        os.makedirs(folder, exist_ok=True)
        p = os.path.join(folder, os.path.basename(self.model_path))
        open(p, "w", encoding="utf-8").write(self._apply(values, rho))
        return p

    def replay(self, model: str, rows_by_window: list, folder: str) -> list:
        self.tool.dt = self.dt
        return self.tool.run(model, rows_by_window, folder)

    def info(self) -> dict:
        import tempfile
        d = tempfile.mkdtemp(prefix=f"strathe_{self.sim}_info_")
        self.tool.dt = self.dt
        return self.tool.info(self.write_runnable(None, d))

    def write_calibrated(self, values: dict, out_dir: str, suffix: str) -> str:
        stem, ext = os.path.splitext(os.path.basename(self.model_path))
        p = os.path.join(out_dir, f"{stem}_{suffix}{ext}")
        open(p, "w", encoding="utf-8").write(self._apply(values))
        return p

    def changes(self, values: dict) -> list:
        by_name = {k.name: k for k in self.knob_defs}
        orig = open_model(self.text, self.fmt)
        L = []
        for key, v in values.items():
            k = by_name.get(key)
            at = k.at if k else key[1:]
            new = float(v) * (k.sign if k else 1.0)
            try:
                old = orig.get(at)
            except SimFitError:
                old = None
            if old is None or abs(old - new) > 1e-12 * max(1.0, abs(old)):
                L.append(f"- {key if k else at}: {fmt_number(old) if old is not None else '?'} -> {fmt_number(new)}"
                         + (f" (`{at}`)" if k else ""))
        return L or ["- nothing (the fit kept every value)"]

    def export_values(self, params, mass: float, info: dict, train: list, rho: float):
        if not self.export_defs:
            raise SimFitError(f"the {self.label} model has no mapping for Strathe's values (an `export:` list in the "
                              f"simulator description, calibration/simulators/SIMULATOR_PROTOCOL.md)")
        values, used = {}, set()
        for e in self.export_defs:
            v = float(mass) if e.param == "mass" else getattr(params, e.param, None)
            if v is None:
                raise SimFitError(f"export: {e.param!r} isn't one of Strathe's calibrated values")
            values["@" + e.at] = e.scale * float(v) + e.offset
            used.add(e.param)
        from dynamics_model import CalibrationParameters
        missing = [n for n in CalibrationParameters.__dataclass_fields__ if n not in used and getattr(params, n)]
        notes = [f"Not carried (no place for it in the model's export list): {', '.join(missing)}."] if missing else []
        return values, notes


def adapter_names() -> list:
    return ["stonefish", "gazebo", "mujoco", "holoocean", "external"]
