"""
strathe_cli.py -- one command for everything a lab does with Strathe.

    python strathe.py check                              is my machine set up?
    python strathe.py calibrate --data <logs folder>     calibrate a vehicle, end to end
    python strathe.py calibrate --ingest-spec my.yaml    ... straight from your own logs (CSV or ROS bags)
    python strathe.py report --result <output folder>    plain-language report for a result
    python strathe.py estimate-current --data <logs>     the water current in each open-water recording
    python strathe.py arena --params <calibration>       one vehicle: train and test a controller on a task
    python strathe.py fleet [--config my_experiment.yaml]  two vehicles: run a fleet experiment
    python strathe.py train-fleet --params <calibration> train a fleet policy (PPO, SAC or TD3) for your vehicle
    python strathe.py fit-noise --data <logs> --out n.yaml  how noisy your vehicle's sensors are, for --noise
    python strathe.py ingest --spec my_vehicle.yaml --out <folder>   convert your own logs
    python strathe.py web                                the same, in a browser (local only)

(strathe.py at the repository root is a thin launcher for this module; it can
be run from any folder.)

`calibrate` replaces the old sequence of separate terminal commands:
  0. (with --ingest-spec) convert your logs  (data_ingestion.py)
  1. pre-flight data checks            (preflight_check.py)
  2. calibration with the recommended settings (run_calibration.py)
  3. which parameters the data can pin down (automated_identifiability_report.py)
  4. the plain-language report         (calibration_report.py)
and stops with a clear message at the first step that fails (step 3 only
warns). The logs' sampling rate and torque convention come from the
ingestion report when there is one. Every step is still a normal script that
specialists can run on its own.
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
OPTIONAL_PACKAGES = [("rosbags", "rosbags", "reading ROS 1/ROS 2 bags directly"),
                     ("pymavlink", "pymavlink", "reading ArduPilot .bin / MAVLink .tlog logs directly"),
                     ("mlflow", "mlflow", "experiment tracking"),
                     ("holoocean", "holoocean", "HoloOcean engine (not needed for calibration or the fleet kit)")]


def _abs(path: str) -> str:
    return os.path.abspath(os.path.expanduser(path))


def spec_file(path: str) -> str:
    """Absolute path of a vehicle description given on the command line (see spec_path); must exist."""
    p = spec_path(path)
    p = p if os.path.isabs(p) else os.path.join(CAL_DIR, p)
    if not os.path.exists(p):
        raise ValueError(f"vehicle description not found: {p}")
    return p


def spec_path(path: str) -> str:
    """A vehicle description given on the command line: relative to where the user is, else one of the
    files shipped in calibration/ (the scripts run from there)."""
    p = os.path.expanduser(path)
    if os.path.isabs(p):
        return p
    if os.path.exists(p):
        return os.path.abspath(p)
    return p


def ingest_summary(data: str) -> dict:
    """The ingest_report.json data_ingestion.py wrote into a logs folder, or {}."""
    import json
    path = os.path.join(data, "ingest_report.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def safe_name(name: str) -> str:
    """A folder-name-safe version of a user-supplied run name."""
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")
    if not cleaned:
        raise ValueError(f"run name {name!r} has no usable characters (use letters, digits, _ or -)")
    return cleaned[:60]


# ----------------------------------------------------------------- commands as argv lists

def preflight_argv(data: str, spec: str, heldout_dates: Optional[list], rate: Optional[float] = None) -> list:
    argv = ["preflight_check.py", "--data-dir", _abs(data), "--spec", spec]
    if rate is not None:
        argv += ["--sampling-rate-hz", f"{rate:g}"]
    return argv + (["--heldout-dates", *heldout_dates] if heldout_dates else [])


def calibrate_argv(data: str, out: str, spec: str, heldout_dates: Optional[list], workers: int,
                   max_nfev: int, extra: Optional[list] = None, rate: Optional[float] = None) -> list:
    argv = ["run_calibration.py", "--data-dir", _abs(data), "--spec", spec, "--output-dir", _abs(out),
            "--max-nfev", str(max_nfev), "--eval-workers", str(workers), *RECOMMENDED_CALIBRATION_FLAGS]
    if rate is not None:
        argv += ["--sampling-rate-hz", f"{rate:g}"]
    if heldout_dates:
        argv += ["--heldout-dates", *heldout_dates]
    return argv + list(extra or [])


def finalize_argv(out: str, data: str, spec: str, heldout_dates: Optional[list], rate: Optional[float],
                  extra: list, reason: str) -> list:
    """finalize_checkpoint.py with the same objective settings the calibration step was given."""
    argv = ["finalize_checkpoint.py", "--checkpoint", os.path.join(_abs(out), "checkpoint.yaml"),
            "--data-dir", _abs(data), "--spec", spec, *RECOMMENDED_CALIBRATION_FLAGS, *extra, "--reason", reason]
    if rate is not None:
        argv += ["--sampling-rate-hz", f"{rate:g}"]
    return argv + (["--heldout-dates", *heldout_dates] if heldout_dates else [])


def estimate_current_argv(data: str, params: Optional[str], spec: str, workers: int, rate: Optional[float],
                          out_csv: Optional[str] = None) -> list:
    argv = ["estimate_current.py", "--data-dir", _abs(data), "--spec", spec, "--workers", str(workers)]
    if params:
        argv += ["--params", _abs(params)]
    if rate is not None:
        argv += ["--sampling-rate-hz", f"{rate:g}"]
    return argv + (["--out", _abs(out_csv)] if out_csv else [])


def identifiability_argv(out: str, data: str, spec: str, workers: int) -> list:
    return ["automated_identifiability_report.py", "--result", _abs(out), "--data-dir", _abs(data), "--spec", spec,
            "--workers", str(workers)]


def report_argv(result: str, data: Optional[str], spec: str, heldout_dates: Optional[list]) -> list:
    argv = ["calibration_report.py", "--result", _abs(result), "--spec", spec]
    if data:
        argv += ["--data-dir", _abs(data)]
    if heldout_dates:
        argv += ["--heldout-dates", *heldout_dates]
    return argv


def fleet_argv(config: Optional[str]) -> list:
    return ["run_fleet_experiment.py"] + (["--config", _abs(config)] if config else [])


DEFAULT_FLEET_CONFIG = os.path.join(CAL_DIR, "default_experiment.yaml")
JOBS_DIR = os.path.join(REPO_DIR, "strathe_jobs")


def current_block(speed, direction, speed_std: float = 0.0, direction_std: float = 0.0,
                  time_scale: float = 30.0, known: bool = True) -> dict:
    """A validated fleet-YAML `current:` block. speed/direction: one number or
    [min, max] (drawn per episode)."""
    if not time_scale >= 1.0:
        raise ValueError(f"current time scale must be at least 1 s (the physics assumes a slowly varying "
                         f"current), got {time_scale}")
    block = {"speed_mps": speed, "direction_deg": direction, "known_to_beliefs": bool(known)}
    if speed_std or direction_std:
        block["variability"] = {"speed_std_mps": speed_std, "direction_std_deg": direction_std,
                                "time_scale_s": time_scale}
    from current import CurrentConfig
    CurrentConfig.from_dict(block)          # raises ValueError with a specific message
    return block


def experiment_with_current(config: Optional[str], block: dict, jobs_dir: Optional[str] = None) -> str:
    """Copy an experiment YAML with a `current:` block added (see experiment_with_overrides)."""
    return experiment_with_overrides(config, {"current": block}, jobs_dir)


def experiment_with_overrides(config: Optional[str], overrides: dict, jobs_dir: Optional[str] = None) -> str:
    """Copy an experiment YAML with some top-level fields replaced (current, thrusters,
    vehicle_spec, truth_params) and its output redirected into jobs_dir (so shipped
    results are never overwritten). Returns the new YAML's path."""
    import time
    import yaml
    jobs_dir = jobs_dir or JOBS_DIR
    src = _abs(config) if config else DEFAULT_FLEET_CONFIG
    with open(src, encoding="utf-8") as f:
        exp = yaml.safe_load(f)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    suffix = "_current" if set(overrides) == {"current"} else "_custom"
    name = safe_name(f"{exp.get('name') or os.path.splitext(os.path.basename(src))[0]}{suffix}")
    os.makedirs(jobs_dir, exist_ok=True)
    exp["name"] = overrides.pop("name", None) or name
    for k, v in overrides.items():  # nested blocks (communication, task, ...) are merged, not replaced
        if isinstance(v, dict) and isinstance(exp.get(k), dict) and k not in ("current", "thrusters"):
            before = exp[k]
            exp[k] = {**exp[k], **v}
            if k == "communication" and "severities" in v:
                exp[k].pop("severity", None)
            if k == "communication" and isinstance(v.get("optical"), dict) and isinstance(before.get("optical"), dict):
                exp[k]["optical"] = {**before["optical"], **v["optical"]}   # --optical-* flags change one field each
        else:
            exp[k] = v
    exp["output_dir"] = os.path.join(jobs_dir, f"{safe_name(exp['name'])}_{stamp}")
    out = os.path.join(jobs_dir, f"{name}_{stamp}.yaml")
    what = ", ".join({"current": "a water current", "thrusters": "thruster limits", "vehicle_spec": "your vehicle "
                      "description", "truth_params": "your calibration"}.get(k, k) for k in overrides) or "no changes"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"# Generated by `strathe fleet` from {src}, with {what}.\n")
        yaml.safe_dump(exp, f, sort_keys=False)
    return out


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
    if bool(args.data) == bool(args.ingest_spec):
        print("❌ give either --data <folder of Strathe CSVs> or --ingest-spec <your ingestion YAML>")
        return 2
    out = _abs(args.out) if args.out else os.path.join(REPO_DIR, f"calibration_output_{safe_name(args.name)}")
    spec = spec_path(args.spec)
    if args.ingest_spec:
        if not os.path.exists(_abs(args.ingest_spec)):
            print(f"❌ ingestion spec not found: {_abs(args.ingest_spec)}")
            return 2
        data = os.path.join(out, "ingested_logs")
    else:
        data = _abs(args.data)
        if not os.path.isdir(data):
            print(f"❌ data folder not found: {data}")
            return 2

    memo = {}

    def settings():
        """Sampling rate and torque convention: flags, else the ingestion report, else the defaults.
        Read once, after any ingestion step has run."""
        if "v" in memo:
            return memo["v"]
        info = ingest_summary(data)
        rate = args.sampling_rate_hz or info.get("sampling_rate_hz")
        extra = []
        offset = args.torque_reference_offset_m or info.get("torque_reference_offset_m")
        if offset is not None:
            extra += ["--torque-reference-offset-m", *map(str, offset)]
        if args.exclude_axes:
            extra += ["--exclude-axes", *args.exclude_axes]
        if args.current_world_mps:
            extra += ["--current-world-mps", *map(str, args.current_world_mps)]
        if rate is None:
            print("ℹ️  No sampling rate given and no ingest_report.json in the data folder: assuming 10 Hz "
                  "(pass --sampling-rate-hz if your logs are at another rate).")
        elif not args.sampling_rate_hz:
            print(f"ℹ️  Logs are at {rate:g} Hz (from ingest_report.json)."
                  + (f" Torques re-expressed about {offset} m, as the ingestion recorded." if offset else ""))
        memo["v"] = (None if rate is None else float(rate)), extra
        return memo["v"]

    steps = []
    if args.ingest_spec:
        steps.append(("Converting your logs into Strathe's layout",
                      lambda: ingest_argv(args.ingest_spec, data) + ["--no-next-step"],
                      "Your logs couldn't be converted; the message above says what's wrong with them or with "
                      "the ingestion spec (templates are in calibration/templates/).", True))
    steps.append(("Checking your data before spending compute on it",
                  lambda: preflight_argv(data, spec, args.heldout_dates, settings()[0]),
                  "The data check found problems; its messages above say what to fix.", True))
    fit_hint = ("Calibration didn't finish. If it ran out of evaluations, re-run with a larger --max-nfev. If it was "
                "interrupted (the machine slept, or the run was stopped), run the same command again with --finish: "
                "it turns the best result so far (checkpoint.yaml in the output folder) into a usable, "
                "clearly-marked result and writes the report.")
    fit_title = ("Calibrating (this takes a while: minutes on a small log set, about an hour on the reference "
                 "dataset; the machine is kept awake)")
    if args.estimate_current and args.current_world_mps:
        print("❌ give either --current-world-mps (a known current) or --estimate-current, not both")
        return 2
    currents_file = None
    if not args.estimate_current and not args.current_world_mps and os.path.exists(os.path.join(data, "currents.csv")):
        currents_file = os.path.join(data, "currents.csv")
        print(f"ℹ️  Using each recording's own water current from {currents_file}.")

    def fit_argv(dest, currents=None, warm=None):
        rate, extra = settings()
        argv = calibrate_argv(data, dest, spec, args.heldout_dates, args.workers, args.max_nfev, extra, rate)
        if currents:
            argv += ["--currents-file", currents]
        if warm and os.path.exists(warm):
            argv += ["--resume-from-checkpoint", warm]
        return argv

    if args.estimate_current:
        rounds = max(1, args.current_rounds)
        prev = None
        for k in range(1, rounds + 1):
            dest = os.path.join(out, f"current_round{k}")
            cur_in = None if k == 1 else os.path.join(out, f"current_round{k - 1}", "currents.csv")
            steps.append((f"Current round {k} of {rounds}: calibrating " + ("in still water" if k == 1 else
                                                                            "with the estimated currents"),
                          (lambda d=dest, c=cur_in, w=prev: fit_argv(d, c, w)), fit_hint, True))
            steps.append((f"Current round {k} of {rounds}: estimating each recording's current",
                          (lambda d=dest: estimate_current_argv(data, os.path.join(d, "calibrated_parameters.yaml"),
                                                                spec, args.workers, settings()[0],
                                                                os.path.join(d, "currents.csv"))),
                          "The current estimate failed; the message above says why.", True))
            prev = os.path.join(dest, "checkpoint.yaml")
        final_currents = os.path.join(out, f"current_round{rounds}", "currents.csv")
        steps.append((fit_title + " -- final fit, with the estimated currents",
                      lambda: fit_argv(out, final_currents, prev), fit_hint, True))
    else:
        steps.append((fit_title, lambda: fit_argv(out, currents_file), fit_hint, True))
    if args.finish:
        # an interrupted fit: keep its best parameters, with the settings this command gives the fit
        if args.estimate_current:
            print("❌ --finish doesn't support --estimate-current runs; re-run them, or use "
                  "calibration/finalize_checkpoint.py on the last round's checkpoint")
            return 2
        if not os.path.exists(os.path.join(out, "checkpoint.yaml")):
            print(f"❌ --finish: no checkpoint.yaml in {out}; nothing to finish (did the fit start?)")
            return 2
        if args.ingest_spec and os.path.isdir(data):
            steps = [st for st in steps if not st[0].startswith("Converting")]  # already converted
        steps = [st for st in steps if not st[0].startswith(("Checking your data", "Calibrating"))]

        def finish_argv():
            rate, extra = settings()
            cur = ["--currents-file", currents_file] if currents_file else []
            return finalize_argv(out, data, spec, args.heldout_dates, rate, extra + cur,
                                 "interrupted; finished from its checkpoint with strathe calibrate --finish")
        steps.append(("Turning the interrupted fit's best result into a calibration (marked not converged)",
                      finish_argv, "The checkpoint couldn't be turned into a result; the message above says why.",
                      True))
    if not args.skip_identifiability:
        steps.append(("Checking which values your data can pin down (a few minutes)",
                      lambda: identifiability_argv(out, data, spec, args.workers),
                      "The check of which values the data can pin down failed; the report is written without "
                      "that section.", False))
    steps.append(("Writing the plain-language report", lambda: report_argv(out, data, spec, args.heldout_dates),
                  "The report couldn't be written; the calibration itself is saved in the output folder.", True))
    for i, (title, make_argv, hint, fatal) in enumerate(steps, 1):
        _banner(i, len(steps), title)
        argv = make_argv()
        rc = run_step(argv, runner)
        dest = argv[argv.index("--output-dir") + 1] if argv[0] == "run_calibration.py" else out
        if rc != 0 and argv[0] == "run_calibration.py" and os.path.exists(os.path.join(dest, "checkpoint.yaml")):
            # the fit stopped early (usually: the --max-nfev budget ran out). Keep the best result so far as a
            # clearly-marked, not-converged result and carry on, rather than leaving the lab with nothing.
            print(f"\n⚠️  The fit stopped before converging (exit code {rc}; usually the evaluation budget, "
                  f"--max-nfev {args.max_nfev}, ran out). Saving the best result so far, marked NOT converged; "
                  f"the report will say so. Re-run with a larger --max-nfev for a converged fit.", flush=True)
            rate, extra = settings()
            cur = [argv[argv.index("--currents-file")], argv[argv.index("--currents-file") + 1]] \
                if "--currents-file" in argv else []
            rc = run_step(finalize_argv(dest, data, spec, args.heldout_dates, rate, extra + cur,
                                        f"the optimizer stopped before converging (budget --max-nfev "
                                        f"{args.max_nfev})"), runner)
        if rc != 0 and fatal:
            print(f"\n❌ Step {i} failed (exit code {rc}). {hint}")
            return rc
        if rc != 0:
            print(f"\n⚠️  Step {i} failed (exit code {rc}). {hint}")
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


def cmd_estimate_current(args, runner=None) -> int:
    data = _abs(args.data)
    if not os.path.isdir(data):
        print(f"❌ data folder not found: {data}")
        return 2
    try:
        params = _file_arg(args.params, "calibration file") if args.params else None
    except ValueError as e:
        print(f"❌ {e}")
        return 2
    return run_step(estimate_current_argv(data, params, spec_path(args.spec), args.workers, args.sampling_rate_hz,
                                          args.out), runner)


def cmd_report(args, runner=None) -> int:
    return run_step(report_argv(args.result, args.data, spec_path(args.spec), args.heldout_dates), runner)


def _range_arg(xs, name):
    if xs is None:
        return None
    if len(xs) == 1:
        return xs[0]
    if len(xs) == 2:
        return [xs[0], xs[1]]
    raise ValueError(f"--{name} takes one value or MIN MAX")


def _file_arg(path: Optional[str], what: str) -> Optional[str]:
    if path is None:
        return None
    p = _abs(path)
    if not os.path.exists(p):
        raise ValueError(f"{what} not found: {p}")
    return p


def thrusters_arg(value: Optional[str]) -> Optional[str]:
    """--thrusters: the preset name, or a YAML file with a thrusters block (made absolute). Validated."""
    if not value:
        return None
    v = value if not value.endswith((".yaml", ".yml")) else _file_arg(value, "thrusters file")
    from fleet_experiment_schema import resolve_thrusters
    resolve_thrusters(v)  # raises ValueError with a specific message
    return v


def noise_arg(value: Optional[str]) -> Optional[str]:
    """--noise: none, a preset (tank, typical, open_water), or a sensor-noise YAML (made absolute). Validated."""
    if value in (None, ""):
        return None
    from sensor_noise import PRESETS
    v = value if value == "none" or value in PRESETS else _file_arg(value, "sensor-noise file")
    if v != "none":
        from sensor_noise import load_sensor_noise
        load_sensor_noise(v)  # raises ValueError with a specific message
    return v


def start_fix_arg(value: Optional[float]) -> Optional[float]:
    """--start-fix-error: metres per axis, 0 to 20. Validated."""
    if value is None:
        return None
    if not (0.0 <= float(value) <= 20.0):
        raise ValueError(f"--start-fix-error must be between 0 and 20 m, got {value}")
    return float(value)


def buoyancy_block(values, known: bool = False) -> Optional[dict]:
    """--buoyancy N [N] -> a validated fleet-YAML `buoyancy:` block, or None."""
    if not values:
        return None
    if len(values) not in (1, 2):
        raise ValueError("--buoyancy takes one value (N) or MIN MAX")
    from buoyancy import BuoyancyConfig
    block = {"offset_n": float(values[0]) if len(values) == 1 else [float(values[0]), float(values[1])],
             "known_to_beliefs": bool(known)}
    BuoyancyConfig.from_dict(block)  # raises ValueError with a specific message
    return block


def simulation_argv(args, known_flag: bool = False) -> list:
    """--mode/--noise/--noise-scale/--disturbance/--disturbance-scale/--buoyancy for the scripts, validated."""
    argv = []
    if getattr(args, "mode", None):
        argv += ["--mode", args.mode]
    if getattr(args, "noise", None) is not None:
        if args.noise_scale < 0:
            raise ValueError("--noise-scale must be 0 or more")
        argv += ["--noise", noise_arg(args.noise), "--noise-scale", str(args.noise_scale)]
    if getattr(args, "start_fix_error", None) is not None:
        start_fix_arg(args.start_fix_error)
        argv += ["--start-fix-error", str(args.start_fix_error)]
    if getattr(args, "disturbance", None) is not None:
        if args.disturbance_scale < 0:
            raise ValueError("--disturbance-scale must be 0 or more")
        argv += ["--disturbance", args.disturbance if args.disturbance in ("none", "fitted")
                 else _file_arg(args.disturbance, "disturbance file"),
                 "--disturbance-scale", str(args.disturbance_scale)]
    if getattr(args, "buoyancy", None):
        buoyancy_block(args.buoyancy)
        argv += ["--buoyancy", *map(str, args.buoyancy)]
        if known_flag and getattr(args, "buoyancy_known_to_beliefs", False):
            argv.append("--buoyancy-known-to-beliefs")
    return argv


def policy_entries(values) -> list:
    """--policies [LABEL=]PATH ... -> experiment `policies:` entries; the algorithm is read from each file."""
    out = []
    for v in values:
        label, path = (v.split("=", 1) if "=" in v and not any(c in v.split("=", 1)[0] for c in "/\\")
                       else (None, v))
        entry = {"model_path": _file_arg(path if path.endswith(".zip") else path + ".zip", "policy")[:-4]}
        if label:
            entry["label"] = label
        out.append(entry)
    return out


SEVERITIES = ("L0", "L1", "L2", "L3a", "L3b")
BELIEF_NAMES = ("calibrated", "generic", "last_beacon", "constant_velocity", "oracle")


def link_block(args) -> dict:
    """`communication:` fields set by --link / --optical-* / --hybrid-policy (optical settings are checked when
    the experiment YAML is validated)."""
    out = {}
    if args.link:
        out["link"] = list(dict.fromkeys(args.link)) if len(args.link) > 1 else args.link[0]
    optical = {k: v for k, v in (("water", (list(args.optical_water) if args.optical_water and len(args.optical_water) > 1
                                            else (args.optical_water[0] if args.optical_water else None))),
                                 ("attenuation_per_m", args.optical_attenuation),
                                 ("terminals", args.optical_terminals),
                                 ("beam_half_angle_deg", args.optical_beam),
                                 ("receiver_half_fov_deg", args.optical_fov),
                                 ("rated_range_m", args.optical_rated_range),
                                 ("data_rate_bps", args.optical_data_rate),
                                 ("beacon_interval_s", args.optical_interval)) if v is not None}
    if args.optical_compare_omni:
        optical["compare_omni"] = True
    if optical:
        out["optical"] = optical
    if args.hybrid_policy:
        out["hybrid_policy"] = args.hybrid_policy
    return out


def fleet_setting_overrides(args) -> dict:
    """The experiment-YAML fields set by `strathe fleet` flags (comm, task, sample size, engine, ...)."""
    o = {}
    comm = {}
    if args.comm:
        comm["severities"] = list(args.comm)
    if args.beacon_interval:
        comm["beacon_interval_s"] = [float(x) for x in args.beacon_interval]
    comm.update(link_block(args))
    if comm:
        o["communication"] = comm
    task = {k: v for k, v in (("target_separation_m", args.target_separation), ("tolerance_m", args.tolerance),
                              ("hold_steps", args.hold_steps), ("max_episode_steps", args.episode_steps),
                              ("collision_radius_m", args.collision_radius)) if v is not None}
    if task:
        o["task"] = task
    ic = {k: [float(x) for x in v] for k, v in (("separations_m", args.separations),
                                                ("closing_speeds_mps", args.closing_speeds)) if v}
    if ic:
        o["initial_conditions"] = ic
    if args.seeds is not None:
        if args.seeds < 1:
            raise ValueError("--seeds must be at least 1")
        o["evaluation"] = {"comm_seeds": list(range(1, args.seeds + 1))}
    if args.beliefs:
        o["beliefs"] = list(dict.fromkeys(args.beliefs))
        if args.compare is None:
            a, b = ("calibrated", "generic")
            o["compare"] = {"a": a, "b": b} if {a, b} <= set(o["beliefs"]) else None
    if args.compare:
        o["compare"] = {"a": args.compare[0], "b": args.compare[1]}
    if args.backend:
        o["dynamics_backend"] = args.backend
    if args.disturbance:
        o["disturbance"] = {"roll_pitch": args.disturbance if args.disturbance in ("none", "fitted")
                            else _file_arg(args.disturbance, "disturbance file"),
                            "scale": args.disturbance_scale}
    if args.policies:
        o["policies"] = policy_entries(args.policies)
    if args.mode:
        o["mode"] = args.mode
    if args.noise is not None:
        if args.noise_scale < 0:
            raise ValueError("--noise-scale must be 0 or more")
        o["sensor_noise"] = {"preset": noise_arg(args.noise), "scale": args.noise_scale}
    if getattr(args, "start_fix_error", None) is not None:
        o.setdefault("sensor_noise", {})["start_fix_error_m"] = start_fix_arg(args.start_fix_error)
    if args.buoyancy:
        o["buoyancy"] = buoyancy_block(args.buoyancy, args.buoyancy_known_to_beliefs)
    if args.name:
        o["name"] = safe_name(args.name)
    return o


def validate_experiment(path: str) -> None:
    """Load a (generated) experiment YAML the way run_fleet_experiment.py will; raise ValueError if invalid."""
    from fleet_experiment_schema import ConfigError, load_fleet_experiment
    cwd = os.getcwd()
    try:
        os.chdir(CAL_DIR)  # relative paths in experiment YAMLs resolve from calibration/, as when it runs
        load_fleet_experiment(path)
    except ConfigError as e:
        raise ValueError(str(e)) from e
    finally:
        os.chdir(cwd)


def cmd_fleet(args, runner=None) -> int:
    config = args.config
    overrides = {}
    try:
        overrides.update(fleet_setting_overrides(args))
        if args.current_speed is not None:
            overrides["current"] = current_block(_range_arg(args.current_speed, "current-speed"),
                                                 _range_arg(args.current_direction, "current-direction"),
                                                 args.current_speed_std, args.current_direction_std,
                                                 args.current_time_scale, not args.current_unknown_to_beliefs)
        if args.thrusters:
            overrides["thrusters"] = thrusters_arg(args.thrusters)
        if args.vehicle_spec:
            overrides["vehicle_spec"] = _file_arg(args.vehicle_spec, "vehicle description")
        if args.params:
            overrides["truth_params"] = _file_arg(args.params, "calibration file")
    except ValueError as e:
        print(f"❌ {e}")
        return 2
    if overrides or args.save_config:
        config = experiment_with_overrides(config, overrides)
        try:
            validate_experiment(config)
        except ValueError as e:
            print(f"❌ These settings don't make a valid experiment:\n{e}")
            os.remove(config)
            return 2
        if args.save_config:
            import shutil
            dest = _abs(args.save_config)
            shutil.move(config, dest)
            print(f"✅ Wrote {dest} (not run). Run it with: python strathe.py fleet --config {dest}")
            return 0
        print(f"Running {config}\n(your experiment with "
              + (", ".join(sorted(overrides)).replace("truth_params", "calibration") or "no changes")
              + " set; the file is kept, so you can rerun or edit it)", flush=True)
    return run_step(fleet_argv(config), runner)


def cmd_arena(args, runner=None) -> int:
    try:
        argv = ["run_arena_experiment.py", "--params", _file_arg(args.params, "calibration file"),
                "--spec", spec_file(args.spec),
                "--task", args.task, "--seeds", str(args.seeds), "--timesteps", str(args.timesteps),
                "--episodes", str(args.episodes), "--workers", str(args.workers)]
        if args.compare_generic:
            argv.append("--compare-generic")
        if args.policy:
            argv += ["--policy", _file_arg(args.policy, "policy")]
        if args.current_speed is not None:
            current_block(_range_arg(args.current_speed, "current-speed"),
                          _range_arg(args.current_direction, "current-direction"),
                          args.current_speed_std, args.current_direction_std, args.current_time_scale)
            argv += ["--current-speed", *map(str, args.current_speed),
                     "--current-direction", *map(str, args.current_direction),
                     "--current-speed-std", str(args.current_speed_std),
                     "--current-direction-std", str(args.current_direction_std),
                     "--current-time-scale", str(args.current_time_scale)]
        if args.thrusters:
            argv += ["--thrusters", thrusters_arg(args.thrusters)]
        argv += ["--backend", args.backend, "--algorithm", args.algorithm]
        argv += simulation_argv(args)
        if args.compare_clean_training:
            argv.append("--compare-clean-training")
    except ValueError as e:
        print(f"❌ {e}")
        return 2
    import time
    out = _abs(args.out) if args.out else os.path.join(JOBS_DIR, f"arena_{safe_name(args.task)}_"
                                                                  f"{time.strftime('%Y%m%d-%H%M%S')}")
    argv += ["--out", out]
    print(f"Report: {os.path.join(out, 'report.md')}", flush=True)
    return run_step(argv, runner)


def link_argv(args) -> list:
    """--link / --optical-* flags for train_fleet_policy.py, checked here first (ValueError)."""
    import sim_mode
    sim_mode.link_from_args(args)
    argv = []
    for flag, v in (("--link", args.link), ("--optical-water", args.optical_water),
                    ("--optical-attenuation", args.optical_attenuation), ("--optical-terminals", args.optical_terminals),
                    ("--optical-beam", args.optical_beam), ("--optical-fov", args.optical_fov),
                    ("--optical-rated-range", args.optical_rated_range), ("--optical-data-rate", args.optical_data_rate),
                    ("--optical-interval", args.optical_interval), ("--hybrid-policy", args.hybrid_policy)):
        if v is not None:
            argv += [flag, str(v)]
    return argv


def cmd_train_fleet(args, runner=None) -> int:
    try:
        argv = ["train_fleet_policy.py", "--truth-params", _file_arg(args.params, "calibration file"),
                "--spec", spec_file(args.spec), "--algorithm", args.algorithm,
                "--seed", str(args.seed), "--belief", args.belief, "--out", _abs(args.out)]
        if args.timesteps is not None:
            argv += ["--timesteps", str(args.timesteps)]
        argv += simulation_argv(args, known_flag=True)
        if args.current_speed is not None:
            current_block(_range_arg(args.current_speed, "current-speed"),
                          _range_arg(args.current_direction, "current-direction"))
            argv += ["--current-speed", *map(str, args.current_speed),
                     "--current-direction", *map(str, args.current_direction)]
        if args.thrusters:
            argv += ["--thrusters", thrusters_arg(args.thrusters)]
        argv += link_argv(args)
    except ValueError as e:
        print(f"❌ {e}")
        return 2
    print(f"Policy: {_abs(args.out)}.zip — use it in a fleet experiment YAML under policies: "
          f"(see CUSTOM_EXPERIMENT.md section 5)", flush=True)
    return run_step(argv, runner)


def cmd_fit_noise(args, runner=None) -> int:
    argv = ["fit_sensor_noise.py", "--data-dir", _abs(args.data), "--out", _abs(args.out)]
    if args.position:
        argv += ["--position", args.position]
    if args.sampling_rate_hz:
        argv += ["--sampling-rate-hz", str(args.sampling_rate_hz)]
    return run_step(argv, runner)


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
    c.add_argument("--data", help="folder of logs in Strathe's CSV layout (see `ingest`)")
    c.add_argument("--ingest-spec", help="or: your ingestion spec (CSV or ROS bag logs, forces or thruster PWM); "
                                         "the logs are converted first. Templates: calibration/templates/")
    c.add_argument("--sampling-rate-hz", type=float,
                   help="the logs' sampling rate (default: from ingest_report.json, else 10 Hz)")
    c.add_argument("--skip-identifiability", action="store_true",
                   help="skip the check of which values the data can pin down (saves a few minutes)")
    c.add_argument("--name", default="my_vehicle", help="short name for this calibration (names the output folder)")
    c.add_argument("--out", help="output folder (default: calibration_output_<name> in the repository)")
    c.add_argument("--spec", default=DEFAULT_SPEC, help="vehicle description (default: the BlueROV2 Heavy)")
    c.add_argument("--heldout-dates", nargs="+",
                   help="recording dates to hold out for validation, e.g. 16-02-2024 (default: the last 20%%)")
    c.add_argument("--workers", type=int, default=default_workers(), help="CPU cores to use")
    c.add_argument("--max-nfev", type=int, default=100, help="optimizer budget (default 100)")
    c.add_argument("--torque-reference-offset-m", type=float, nargs=3, metavar=("X", "Y", "Z"),
                   help="advanced: re-express logged torques about another point (default: what the ingestion "
                        "recorded, else vehicle_spec.yaml; see run_calibration.py --help)")
    c.add_argument("--current-world-mps", type=float, nargs=3, metavar=("VX", "VY", "VZ"),
                   help="open-water logs: the known, roughly constant water current during the recordings, in "
                        "the logs' world frame (m/s). Default: vehicle_spec.yaml (still water)")
    c.add_argument("--estimate-current", action="store_true",
                   help="open water, current unknown: estimate each recording's (constant) current from the logs, "
                        "alternating with the fit (calibrate, estimate, recalibrate). About 2-3x the time")
    c.add_argument("--current-rounds", type=int, default=2, help="with --estimate-current: rounds (default 2)")
    c.add_argument("--exclude-axes", nargs="+", choices=["p", "q"],
                   help="advanced: leave roll (p) / pitch (q) out of the fit (see run_calibration.py --help)")
    c.add_argument("--finish", action="store_true",
                   help="the fit was interrupted (the machine slept, the run was stopped): turn its best result so "
                        "far into a calibration, marked not converged, and write the report. Same command and "
                        "flags as the interrupted run, plus --finish")
    c.set_defaults(fn=cmd_calibrate)

    e = sub.add_parser("estimate-current", help="estimate the water current in each open-water recording")
    e.add_argument("--data", required=True, help="logs in Strathe's CSV layout")
    e.add_argument("--params", help="a calibration to use as the vehicle model (default: the description's defaults)")
    e.add_argument("--spec", default=DEFAULT_SPEC)
    e.add_argument("--sampling-rate-hz", type=float)
    e.add_argument("--workers", type=int, default=default_workers())
    e.add_argument("--out", help="default: <data>/currents.csv, which calibrate then uses automatically")
    e.set_defaults(fn=cmd_estimate_current)

    r = sub.add_parser("report", help="plain-language report for a calibration result")
    r.add_argument("--result", required=True)
    r.add_argument("--data", help="the logs it was calibrated on (enables the per-axis skill check)")
    r.add_argument("--spec", default=DEFAULT_SPEC)
    r.add_argument("--heldout-dates", nargs="+")
    r.set_defaults(fn=cmd_report)

    f = sub.add_parser("fleet", help="run a fleet coordination experiment")
    f.add_argument("--config", help="experiment YAML (default: the shipped default_experiment.yaml)")
    f.add_argument("--current-speed", type=float, nargs="+", metavar="MPS",
                   help="add a water current: speed in m/s, or MIN MAX to draw one per episode")
    f.add_argument("--current-direction", type=float, nargs="+", default=[0.0, 360.0], metavar="DEG",
                   help="direction the water flows toward, or MIN MAX (default: any direction)")
    f.add_argument("--current-speed-std", type=float, default=0.0, help="slow speed fluctuation (m/s)")
    f.add_argument("--current-direction-std", type=float, default=0.0, help="slow direction fluctuation (deg)")
    f.add_argument("--current-time-scale", type=float, default=30.0, help="fluctuation time scale (s)")
    f.add_argument("--current-unknown-to-beliefs", action="store_true",
                   help="teammate models don't know the current (default: they know its mean)")
    f.add_argument("--thrusters", help="limit commands to what the thrusters can deliver: bluerov2_heavy, or a YAML "
                                       "file with a thrusters block (your ingest spec works)")
    f.add_argument("--params", help="your calibrated_parameters.yaml as the vehicles' true physics")
    f.add_argument("--vehicle-spec", help="your vehicle description (vehicle_spec.yaml format)")
    g = f.add_argument_group("experiment settings (each overrides the YAML; no editing needed)")
    g.add_argument("--comm", nargs="+", choices=SEVERITIES,
                   help="packet-loss levels, one result cell each: L0 0%%, L1 10%%, L2 30%%, L3a 60%%, "
                        "L3b 60%% + bandwidth one byte short of a beacon")
    g.add_argument("--beacon-interval", nargs="+", type=float, metavar="S", help="seconds between beacons (list = sweep)")
    g.add_argument("--beliefs", nargs="+", choices=BELIEF_NAMES, help="teammate models to compare")
    g.add_argument("--compare", nargs=2, metavar=("A", "B"), help="the headline pair (default calibrated generic)")
    g.add_argument("--target-separation", type=float, metavar="M")
    g.add_argument("--tolerance", type=float, metavar="M")
    g.add_argument("--hold-steps", type=int, help="ticks (0.1 s) the formation must be held")
    g.add_argument("--episode-steps", type=int, help="episode length in ticks (0.1 s)")
    g.add_argument("--collision-radius", type=float, metavar="M")
    g.add_argument("--separations", nargs="+", type=float, metavar="M", help="starting separations")
    g.add_argument("--closing-speeds", nargs="+", type=float, metavar="MPS", help="starting closing speeds")
    g.add_argument("--seeds", type=int, help="packet-loss seeds per starting condition (sample size)")
    g.add_argument("--backend", choices=["stand_in", "mujoco", "both"], help="physics engine")
    g.add_argument("--mode", choices=["clean", "noisy", "both"],
                   help="clean (exact sensors, no disturbance), noisy (sensor noise + roll/pitch disturbance), or "
                        "both: every condition run clean and noisy, compared episode by episode")
    g.add_argument("--noise", help="sensor noise: none, tank (measured from the reference tank logs), typical (ASSUMED sizes for a DVL + AHRS vehicle, not measured), open_water (MEASURED on a BlueROV2 at a fish farm, SOLAQUA; biases assumed), or your file from `strathe fit-noise`")
    g.add_argument("--noise-scale", type=float, default=1.0, help="0 = off, 2 = twice as noisy")
    g.add_argument("--start-fix-error", type=float, metavar="M", help="error of each vehicle's starting position fix, m per axis, for dead reckoning (default: the noise preset's; open_water 0.17, typical 0.2). In the fleet it is most of what sensor noise costs (PROJECT_STATUS.md #60)")
    g.add_argument("--buoyancy", type=float, nargs="+", metavar="N",
                   help="net buoyancy change from the calibration in N (- = heavier), or MIN MAX drawn per episode")
    g.add_argument("--buoyancy-known-to-beliefs", action="store_true",
                   help="physics teammate models know the buoyancy change (default: they don't)")
    g.add_argument("--disturbance", help="roll/pitch disturbance: none, fitted, or a fit file")
    g.add_argument("--disturbance-scale", type=float, default=1.0)
    import sim_mode
    sim_mode.add_link_args(g, sweep=True)
    g.add_argument("--optical-compare-omni", action="store_true",
                   help="also run every optical/hybrid condition with omnidirectional terminals on the same "
                        "episodes, to show what pointing costs")
    g.add_argument("--policies", nargs="+", metavar="[LABEL=]ZIP",
                   help="policy files to use (PPO, SAC or TD3; read from each file). Policies with different "
                        "labels, or different algorithms, are compared in the report, e.g. "
                        "ppo=fleet_policy_seed1 sac=algorithm_policies/fleet_sac_seed1")
    g.add_argument("--name", help="experiment name (names the output folder)")
    f.add_argument("--save-config", metavar="YAML",
                   help="write the resulting experiment YAML here instead of running it")
    f.set_defaults(fn=cmd_fleet)

    a = sub.add_parser("arena", help="one vehicle: train and test a controller on a task with your calibration")
    a.add_argument("--params", default=os.path.join(CAL_DIR, "calibrated_parameters_fleet_truth.yaml"),
                   help="your calibrated_parameters.yaml (default: the shipped BlueROV2 calibration)")
    a.add_argument("--spec", default=DEFAULT_SPEC, help="vehicle description (default: the BlueROV2 Heavy)")
    a.add_argument("--task", default="waypoint", choices=["waypoint", "station_keeping", "docking",
                                                          "obstacle_avoidance"])
    a.add_argument("--seeds", type=int, default=2, help="training seeds per condition")
    a.add_argument("--timesteps", type=int, default=150_000, help="training steps per policy (~5 min each)")
    a.add_argument("--episodes", type=int, default=100, help="evaluation episodes per controller")
    a.add_argument("--compare-generic", action="store_true",
                   help="also train on the default (uncalibrated) physics, to see whether calibration matters")
    a.add_argument("--policy", help="evaluate your own trained policy (.zip: PPO, SAC or TD3) instead of training")
    a.add_argument("--algorithm", choices=["ppo", "sac", "td3"], default="ppo",
                   help="training algorithm (default ppo; sac and td3 with the settings validated on the reference BlueROV2 in PROJECT_STATUS.md #37; on your vehicle, compare seeds)")
    a.add_argument("--mode", choices=["clean", "noisy"],
                   help="clean (default) or noisy: realistic sensor noise + the fitted roll/pitch disturbance")
    a.add_argument("--noise", help="sensor noise: none, tank (measured from the reference tank logs), typical (ASSUMED sizes for a DVL + AHRS vehicle, not measured), open_water (MEASURED on a BlueROV2 at a fish farm, SOLAQUA; biases assumed), or your file from `strathe fit-noise`")
    a.add_argument("--noise-scale", type=float, default=1.0, help="0 = off, 2 = twice as noisy")
    a.add_argument("--start-fix-error", type=float, metavar="M", help="error of each vehicle's starting position fix, m per axis, for dead reckoning (default: the noise preset's; open_water 0.17, typical 0.2). In the fleet it is most of what sensor noise costs (PROJECT_STATUS.md #60)")
    a.add_argument("--buoyancy", type=float, nargs="+", metavar="N",
                   help="net buoyancy change from the calibration in N (- = heavier), or MIN MAX drawn per episode")
    a.add_argument("--compare-clean-training", action="store_true",
                   help="with noise or disturbance on: also train on a clean simulation and compare on your test")
    a.add_argument("--current-speed", type=float, nargs="+", metavar="MPS")
    a.add_argument("--current-direction", type=float, nargs="+", default=[0.0, 360.0], metavar="DEG")
    a.add_argument("--current-speed-std", type=float, default=0.0)
    a.add_argument("--current-direction-std", type=float, default=0.0)
    a.add_argument("--current-time-scale", type=float, default=30.0)
    a.add_argument("--thrusters", help="bluerov2_heavy, or a YAML file with a thrusters block")
    a.add_argument("--backend", choices=["stand_in", "mujoco", "both"], default="stand_in",
                   help="physics engine for the tests: stand_in (fast, default), mujoco, or both side by side")
    a.add_argument("--disturbance", default=None,
                   help="random roll/pitch torques: none, fitted, or your own fit file (default: from --mode)")
    a.add_argument("--disturbance-scale", type=float, default=1.0, help="0 = off, 1 = as fitted, 2 = twice as rough")
    a.add_argument("--workers", type=int, default=default_workers())
    a.add_argument("--out", help="output folder (default: strathe_jobs/arena_<task>_<time>)")
    a.set_defaults(fn=cmd_arena)

    t = sub.add_parser("train-fleet", help="train a fleet (two-vehicle) policy for your vehicle")
    t.add_argument("--params", required=True, help="your calibrated_parameters.yaml")
    t.add_argument("--spec", default=DEFAULT_SPEC, help="vehicle description")
    t.add_argument("--algorithm", choices=["ppo", "sac", "td3"], default="ppo",
                   help="ppo (default), sac or td3. SAC/TD3 were validated on the reference BlueROV2 and the formation task (PROJECT_STATUS.md #58); for your vehicle, train a few seeds of each and compare them with `strathe fleet --policies ppo=... sac=...`")
    t.add_argument("--timesteps", type=int, default=None,
                   help="default: 3M for ppo, 1M for sac/td3 (~40 min each on an M1)")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--belief", default="mix",
                   help="teammate model(s) to train with: mix (recommended) or calibrated/generic/last_beacon/"
                        "constant_velocity/oracle")
    t.add_argument("--current-speed", type=float, nargs="+", metavar="MPS")
    t.add_argument("--current-direction", type=float, nargs="+", default=[0.0, 360.0], metavar="DEG")
    t.add_argument("--thrusters", help="bluerov2_heavy, or a YAML file with a thrusters block")
    t.add_argument("--mode", choices=["clean", "noisy"],
                   help="train clean (default) or noisy: sensor noise + roll/pitch disturbance")
    t.add_argument("--noise", help="sensor noise: none, tank (measured from the reference tank logs), typical (ASSUMED sizes for a DVL + AHRS vehicle, not measured), open_water (MEASURED on a BlueROV2 at a fish farm, SOLAQUA; biases assumed), or your file from `strathe fit-noise`")
    t.add_argument("--noise-scale", type=float, default=1.0)
    t.add_argument("--start-fix-error", type=float, metavar="M", help="error of each vehicle's starting position fix, m per axis, for dead reckoning (default: the noise preset's; open_water 0.17, typical 0.2). In the fleet it is most of what sensor noise costs (PROJECT_STATUS.md #60)")
    t.add_argument("--disturbance", help="roll/pitch disturbance: none, fitted, or a fit file (default: from --mode)")
    t.add_argument("--disturbance-scale", type=float, default=1.0)
    t.add_argument("--buoyancy", type=float, nargs="+", metavar="N",
                   help="net buoyancy change in N, or MIN MAX drawn per episode (train across trims)")
    t.add_argument("--buoyancy-known-to-beliefs", action="store_true")
    sim_mode.add_link_args(t)
    t.add_argument("--out", required=True, help="policy path without .zip")
    t.set_defaults(fn=cmd_train_fleet)

    n = sub.add_parser("fit-noise", help="measure how noisy your vehicle's sensors are, from its logs")
    n.add_argument("--data", required=True, help="logs in Strathe's CSV layout (see `ingest`)")
    n.add_argument("--out", required=True, help="sensor-noise YAML to write (use it with --noise)")
    n.add_argument("--position", choices=["dead_reckoning", "external"],
                   help="how the vehicle knows its position (default: external if the logs have x/y)")
    n.add_argument("--sampling-rate-hz", type=float)
    n.set_defaults(fn=cmd_fit_noise)

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
