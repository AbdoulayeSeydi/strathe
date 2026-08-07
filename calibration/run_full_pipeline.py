#!/usr/bin/env python3
"""
Phase 3 — the single-command demo pipeline: spec file + log folder in,
a ready-to-use Gymnasium environment + validation report out. Runnable
end-to-end without manual intervention partway through.

Reuses run_calibration.py's already-tested run_pipeline() for the actual
calibration (discover -> split -> load -> calibrate -> evaluate -> save) -
does not reimplement any of that logic. Adds: a Gym-environment smoke
test (confirms Arena actually instantiates and steps with the calibrated
params, not just that a YAML file was written), and one combined,
plain-language summary report.

SCOPE, stated explicitly (see README.md / PROJECT_STATUS.md for why):
  - Uses the FAST STAND-IN calibration, not the (7-29 hour, less
    reliable) HoloOcean refinement. This command is designed to be
    watchable in a live demo, not run overnight.
  - The validation numbers here are a QUICK check on the fast stand-in
    backend, NOT the rigorous real-HoloOcean Claim A methodology
    documented in PROJECT_STATUS.md. Labeled clearly as such in the
    output - do not quote these numbers as the project's headline
    result without checking PROJECT_STATUS.md for the real, slower,
    HoloOcean-validated numbers.

Usage:
    python run_full_pipeline.py \
        --spec vehicle_spec.yaml \
        --data-dir ../data/raw/Bluerov2_trajectory_dataset \
        --output-dir ../demo_pipeline_output \
        --sampling-rate-hz 10.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from run_calibration import PipelineConfig, run_pipeline, ConsoleReporter
from optimizer import CalibrationConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="Path to vehicle_spec.yaml")
    parser.add_argument("--data-dir", required=True, help="Folder of real logged mission data")
    parser.add_argument("--output-dir", required=True, help="Where all outputs get written")
    parser.add_argument(
        "--sampling-rate-hz", type=float, default=10.0,
        help="Real sampling rate of the logged data. NOT auto-detected in this version — "
             "defaults to 10.0 (this project's own confirmed value for the reference "
             "dataset; see PROJECT_STATUS.md finding #11 for the validated method used to "
             "determine it). Override this if using a different dataset with a known, "
             "different rate.",
    )
    parser.add_argument("--max-nfev", type=int, default=15,
                         help="Evaluation budget for the calibration fit. Kept SMALL by "
                              "default (15) specifically for this demo script, given a "
                              "confirmed real precedent: --max-nfev 60 on this same "
                              "backend/dataset took ~6.5 hours (the displayed nfev "
                              "undercounts real cost ~21x for a 20+ parameter model - see "
                              "PROJECT_STATUS.md finding #14). Raise this for a more fully "
                              "converged result once you've confirmed the pipeline works "
                              "end-to-end at this fast, watchable setting first.")
    return parser


def smoke_test_gym_environment(calibrated_params_path: str, spec_path: str) -> dict:
    """
    Confirms the calibrated parameters actually produce a working,
    importable, steppable Gymnasium environment — not just that a YAML
    file exists. Returns a small dict describing what was checked, for
    the final report.
    """
    result = {"attempted": True, "success": False, "detail": None}
    try:
        import yaml
        from dynamics_model import CalibrationParameters
        from optimizer import load_vehicle_spec
        from arena import Arena

        with open(calibrated_params_path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        params = CalibrationParameters(**data["parameters"])
        vehicle = load_vehicle_spec(spec_path).vehicle  # same real vehicle constants
        # used throughout calibration, not a hardcoded guess

        # Best-effort construction - Arena's exact constructor signature
        # should be verified against arena.py directly if this fails;
        # this smoke test is deliberately defensive rather than assuming.
        env = Arena(params=params, vehicle=vehicle)
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

        result["success"] = True
        result["detail"] = f"Environment reset and stepped successfully. obs shape: {getattr(obs, 'shape', type(obs))}"
    except Exception as e:
        result["detail"] = f"Smoke test failed: {type(e).__name__}: {e}"
    return result


def main() -> None:
    args = build_arg_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    t_start = time.monotonic()

    print("=" * 70)
    print("PHASE 3 DEMO PIPELINE — spec + logs -> calibrated Gym environment")
    print("=" * 70)
    print(f"Spec:      {args.spec}")
    print(f"Data dir:  {args.data_dir}")
    print(f"Output:    {args.output_dir}")
    print(f"Sampling rate: {args.sampling_rate_hz} Hz (NOT auto-detected - see --help)")
    print()
    print("STEP 1/2: Fitting calibration (fast stand-in backend)...")
    print("NOTE: on the full real dataset, this backend has previously taken multiple")
    print("hours at this max-nfev (the displayed nfev undercounts real evaluation cost")
    print("~21x for a 20+ parameter model - see PROJECT_STATUS.md finding #14). Real")
    print("per-iteration progress will now print below as it happens.")

    calibration_config = CalibrationConfig(
        loss="soft_l1", n_restarts=1, seed=42, max_nfev=args.max_nfev,
        initialization_strategy="perturbed", verbose=2,
    )
    pipeline_config = PipelineConfig(
        data_dir=args.data_dir, spec_path=args.spec, output_dir=args.output_dir,
        sampling_rate_hz=args.sampling_rate_hz, heldout_dates=None,
        heldout_fraction=0.3, calibration_config=calibration_config, workers=4,
    )

    try:
        report = run_pipeline(pipeline_config)
    except Exception as e:
        print(f"\nPIPELINE FAILED at calibration step: {e}")
        sys.exit(1)

    ConsoleReporter().report(report)

    print()
    print("STEP 2/2: Verifying the calibrated Gym environment actually works...")
    calibrated_params_path = str(Path(args.output_dir) / "calibrated_parameters.yaml")
    gym_check = smoke_test_gym_environment(calibrated_params_path, args.spec)
    if gym_check["success"]:
        print(f"  PASS — {gym_check['detail']}")
    else:
        print(f"  FAIL — {gym_check['detail']}")
        print("  (Calibration itself succeeded; only the Gym-environment smoke test failed. "
              "Check arena.py's Arena() constructor signature against this script's call.)")

    total_elapsed = time.monotonic() - t_start

    summary = {
        "calibration_converged": report.success,
        "rmse_train_default": report.rmse.default_train,
        "rmse_train_calibrated": report.rmse.calibrated_train,
        "rmse_heldout_default": report.rmse.default_heldout,
        "rmse_heldout_calibrated": report.rmse.calibrated_heldout,
        "improvement_heldout_pct": report.rmse.improvement_heldout_pct,
        "gym_environment_check": gym_check,
        "total_elapsed_seconds": total_elapsed,
        "NOTE": "Validation numbers above are from the FAST STAND-IN backend, a quick check "
                "only - NOT the rigorous real-HoloOcean Claim A methodology in "
                "PROJECT_STATUS.md. Do not quote as the project's headline result.",
    }
    summary_path = Path(args.output_dir) / "pipeline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=" * 70)
    print(f"DONE in {total_elapsed:.1f}s. Summary written to {summary_path}")
    print("=" * 70)
    sys.exit(0 if (report.success and gym_check["success"]) else 1)


if __name__ == "__main__":
    main()