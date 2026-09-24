"""
strathe_cli.py -- one command for everything a lab does with Strathe.

    python strathe.py check                              is my machine set up?
    python strathe.py calibrate --data <logs folder>     calibrate a vehicle, end to end
    python strathe.py report --result <output folder>    plain-language report for a result
    python strathe.py fleet [--config my_experiment.yaml]  run a fleet experiment
    python strathe.py ingest --spec my_vehicle.yaml --out <folder>   convert your own logs
    python strathe.py web                                the same, in a browser (local only)

(strathe.py at the repository root is a thin launcher for this module; it can
be run from any folder.)

`calibrate` replaces the old sequence of separate terminal commands:
  1. pre-flight data checks            (preflight_check.py)
  2. calibration with the recommended settings (run_calibration.py)
  3. the plain-language report         (calibration_report.py)
and stops with a clear message at the first step that fails. Every step is
still a normal script that specialists can run on its own.
"""
from __future__ import annotations

import argparse
import importlib
import os
import re
import subprocess
import sys
from typing import Callable, Optional

CAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(CAL_DIR)
DEFAULT_SPEC = "vehicle_spec.yaml"
REFERENCE_DATA = os.path.join(REPO_DIR, "data", "raw", "Bluerov2_trajectory_dataset")

# Recommended calibration settings (PROJECT_STATUS.md #50): score every axis
# against its "predict zero" error, and evaluate runs on all but one core.
RECOMMENDED_CALIBRATION_FLAGS = ["--axis-scaling", "zero_rms"]

REQUIRED_PACKAGES = [("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"), ("yaml", "pyyaml"),
                     ("pydantic", "pydantic"), ("mujoco", "mujoco"), ("stable_baselines3", "stable-baselines3"),
                     ("torch", "torch")]
OPTIONAL_PACKAGES = [("mlflow", "mlflow", "experiment tracking"),
                     ("holoocean", "holoocean", "HoloOcean engine (not needed for calibration or the fleet kit)")]


def _abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def safe_name(name: str) -> str:
    """A folder-name-safe version of a user-supplied run name."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    if not cleaned:
        raise ValueError(f"run name {name!r} has no usable characters (use letters, digits, _ or -)")
    return cleaned[:60]


# ----------------------------------------------------------------- commands as argv lists

def preflight_argv(data: str, spec: str, heldout_dates: Optional[list]) -> list:
    argv = ["preflight_check.py", "--data-dir", _abs(data), "--spec", spec]
    return argv + (["--heldout-dates", *heldout_dates] if heldout_dates else [])


def calibrate_argv(data: str, out: str, spec: str, heldout_dates: Optional[list], workers: int,
                   max_nfev: int, extra: Optional[list] = None) -> list:
    argv = ["run_calibration.py", "--data-dir", _abs(data), "--spec", spec, "--output-dir", _abs(out),
            "--max-nfev", str(max_nfev), "--eval-workers", str(workers), *RECOMMENDED_CALIBRATION_FLAGS]
    if heldout_dates:
        argv += ["--heldout-dates", *heldout_dates]
    return argv + list(extra or [])


def report_argv(result: str, data: Optional[str], spec: str, heldout_dates: Optional[list]) -> list:
    argv = ["calibration_report.py", "--result", _abs(result), "--spec", spec]
    if data:
        argv += ["--data-dir", _abs(data)]
    if heldout_dates:
        argv += ["--heldout-dates", *heldout_dates]
    return argv


def fleet_argv(config: Optional[str]) -> list:
    return ["run_fleet_experiment.py"] + (["--config", _abs(config)] if config else [])


def ingest_argv(spec: str, out: str) -> list:
    return ["data_ingestion.py", "--spec", _abs(spec), "--out", _abs(out)]


# ----------------------------------------------------------------- running steps

def run_step(argv: list, runner: Optional[Callable] = None) -> int:
    """Run one Strathe script with this Python, from the calibration folder,
    streaming its output. Returns the exit code."""
    cmd = [sys.executable, *argv]
    if runner is not None:
        return runner(cmd)
    return subprocess.call(cmd, cwd=CAL_DIR)


def _banner(i, n, text):
    print(f"\n{'=' * 70}\n[{i}/{n}] {text}\n{'=' * 70}", flush=True)


def cmd_check(args, runner=None) -> int:
    ok = True
    print(f"Python {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info < (3, 10):
        ok = False
        print("❌ Python 3.10 or newer is required")
    for mod, pip_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(mod)
            print(f"✅ {pip_name}")
        except Exception as e:
            ok = False
            print(f"❌ {pip_name} is missing or broken ({type(e).__name__}). Fix: pip install -r requirements.txt")
    for mod, pip_name, what in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(mod)
            print(f"✅ {pip_name} (optional: {what})")
        except Exception:
            print(f"➖ {pip_name} not installed (optional: {what})")
    if os.path.isdir(REFERENCE_DATA):
        print(f"✅ reference BlueROV2 dataset found ({REFERENCE_DATA})")
    else:
        print("➖ reference BlueROV2 dataset not downloaded (only needed to reproduce the published results; "
              "see README 'Setup')")
    policies = [p for p in ("fleet_policy_seed1.zip", "fleet_policy_seed2.zip", "fleet_policy_seed3.zip")
                if os.path.exists(os.path.join(CAL_DIR, p))]
    print(("✅" if len(policies) == 3 else "❌") + f" shipped fleet policies ({len(policies)}/3)")
    ok = ok and len(policies) == 3
    print("\nReady." if ok else "\nNot ready: fix the ❌ items above.")
    return 0 if ok else 1


def cmd_calibrate(args, runner=None) -> int:
    data = _abs(args.data)
    if not os.path.isdir(data):
        print(f"❌ data folder not found: {data}")
        return 2
    out = _abs(args.out) if args.out else os.path.join(REPO_DIR, f"calibration_output_{safe_name(args.name)}")
    extra = []
    if args.torque_reference_offset_m:
        extra += ["--torque-reference-offset-m", *map(str, args.torque_reference_offset_m)]
    if args.exclude_axes:
        extra += ["--exclude-axes", *args.exclude_axes]
    steps = [
        ("Checking your data before spending compute on it", preflight_argv(data, args.spec, args.heldout_dates)),
        (f"Calibrating (this takes a while: minutes on a small log set, about an hour on the reference dataset; "
         f"the machine is kept awake)", calibrate_argv(data, out, args.spec, args.heldout_dates, args.workers,
                                                        args.max_nfev, extra)),
        ("Writing the plain-language report", report_argv(out, data, args.spec, args.heldout_dates)),
    ]
    for i, (title, argv) in enumerate(steps, 1):
        _banner(i, len(steps), title)
        rc = run_step(argv, runner)
        if rc != 0:
            hint = {1: "The data check found problems; its messages above say what to fix.",
                    2: "Calibration didn't finish. If it ran out of evaluations, re-run with a larger --max-nfev, "
                       "or turn the best result so far (checkpoint.yaml in the output folder) into a usable, "
                       "clearly-marked result with calibration/finalize_checkpoint.py.",
                    3: "The report couldn't be written; the calibration itself is saved in the output folder."}[i]
            print(f"\n❌ Step {i} failed (exit code {rc}). {hint}")
            return rc
    report = os.path.join(out, "calibration_report.md")
    print(f"\n✅ Done.\n   Report:     {report}\n   Parameters: {os.path.join(out, 'calibrated_parameters.yaml')}")
    if runner is None and os.path.exists(report):
        _print_bottom_line(report)
    return 0


def _print_bottom_line(report_path: str) -> None:
    with open(report_path, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"## Bottom line\n\n(.+?)\n", text)
    if m:
        print("\n" + m.group(1).replace("**", ""))


def cmd_report(args, runner=None) -> int:
    return run_step(report_argv(args.result, args.data, args.spec, args.heldout_dates), runner)


def cmd_fleet(args, runner=None) -> int:
    return run_step(fleet_argv(args.config), runner)


def cmd_ingest(args, runner=None) -> int:
    return run_step(ingest_argv(args.spec, args.out), runner)


def cmd_web(args, runner=None) -> int:
    from strathe_web import serve
    return serve(port=args.port, open_browser=not args.no_browser)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="strathe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="check that this machine can run Strathe").set_defaults(fn=cmd_check)

    c = sub.add_parser("calibrate", help="calibrate a vehicle from its logs, end to end")
    c.add_argument("--data", required=True, help="folder of logs in Strathe's CSV layout (see `ingest`)")
    c.add_argument("--name", default="my_vehicle", help="short name for this calibration (names the output folder)")
    c.add_argument("--out", help="output folder (default: calibration_output_<name> in the repository)")
    c.add_argument("--spec", default=DEFAULT_SPEC, help="vehicle description (default: the BlueROV2 Heavy)")
    c.add_argument("--heldout-dates", nargs="+",
                   help="recording dates to hold out for validation, e.g. 16-02-2024 (default: the last 20%%)")
    c.add_argument("--workers", type=int, default=default_workers(), help="CPU cores to use")
    c.add_argument("--max-nfev", type=int, default=100, help="optimizer budget (default 100)")
    c.add_argument("--torque-reference-offset-m", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="advanced: re-express logged torques about another point (see run_calibration.py --help)")
    c.add_argument("--exclude-axes", nargs="+", choices=["p", "q"],
                   help="advanced: leave roll (p) / pitch (q) out of the fit (see run_calibration.py --help)")
    c.set_defaults(fn=cmd_calibrate)

    r = sub.add_parser("report", help="plain-language report for a calibration result")
    r.add_argument("--result", required=True)
    r.add_argument("--data", help="the logs it was calibrated on (enables the per-axis skill check)")
    r.add_argument("--spec", default=DEFAULT_SPEC)
    r.add_argument("--heldout-dates", nargs="+")
    r.set_defaults(fn=cmd_report)

    f = sub.add_parser("fleet", help="run a fleet coordination experiment")
    f.add_argument("--config", help="experiment YAML (default: the shipped default_experiment.yaml)")
    f.set_defaults(fn=cmd_fleet)

    i = sub.add_parser("ingest", help="convert your own logs into Strathe's CSV layout")
    i.add_argument("--spec", required=True, help="ingestion spec YAML (see data_ingestion.py)")
    i.add_argument("--out", required=True)
    i.set_defaults(fn=cmd_ingest)

    w = sub.add_parser("web", help="open the local web interface")
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--no-browser", action="store_true")
    w.set_defaults(fn=cmd_web)
    return ap


def main(argv=None, runner=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args, runner)


if __name__ == "__main__":
    sys.exit(main())
