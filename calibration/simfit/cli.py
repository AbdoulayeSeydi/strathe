"""
simfit/cli.py -- `python -m simfit.cli fit|export --sim NAME ...`, what `strathe fit-sim` / `strathe export-sim` run
(PROJECT_STATUS.md #81). --sim stonefish goes to fit-stonefish / export-stonefish, unchanged.
"""
from __future__ import annotations

import argparse
import importlib
import sys

from simfit import runner
from simfit.adapters import adapter_names
from simfit.core import SimFitError

ADAPTER_MODULES = {"external": "simfit.adapters.external", "gazebo": "simfit.adapters.gazebo",
                   "mujoco": "simfit.adapters.mujoco", "holoocean": "simfit.adapters.holoocean"}


def _module(sim: str):
    try:
        return importlib.import_module(ADAPTER_MODULES[sim])
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("simfit.adapters"):
            raise SimFitError(f"--sim {sim} isn't available in this version yet")
        raise


def build_parser(command: str, sim: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"strathe {command}-sim --sim {sim}",
                                 description=f"{'Calibrate your' if command == 'fit' else 'Put a Strathe calibration into your'} "
                                             f"{sim} model (calibration/simulators/README.md)")
    _module(sim).add_arguments(ap)
    if command == "fit":
        runner.add_shared_arguments(ap, f"{sim}_fit")
        runner.add_fit_arguments(ap)
    else:
        ap.add_argument("--calibration", help="your Strathe calibration's output folder")
        ap.add_argument("--params", help="or: a calibrated_parameters.yaml (then give --data)")
        ap.add_argument("--mass", type=float, help="the vehicle's mass, kg (default: the calibration's)")
        runner.add_shared_arguments(ap, f"{sim}_export")
        for act in ap._actions:  # export reads the logs from the calibration by default
            if act.dest == "data":
                act.required = False
    return ap


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 1 or argv[0] not in ("fit", "export"):
        print("usage: python -m simfit.cli fit|export --sim NAME ...")
        return 2
    command, rest = argv[0], argv[1:]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--sim", choices=adapter_names(), required=True)
    known, rest2 = pre.parse_known_args(rest)
    sim = known.sim
    if sim == "stonefish":
        import stonefish_export
        import stonefish_fit
        return (stonefish_fit if command == "fit" else stonefish_export).main(rest2)
    try:
        a = build_parser(command, sim).parse_args(rest2)
        adapter = _module(sim).adapter_from_args(a)
        return runner.run_fit(adapter, a) if command == "fit" else runner.run_export(adapter, a)
    except SimFitError as e:
        print(f"❌ {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
