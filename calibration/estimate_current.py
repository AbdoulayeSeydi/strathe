#!/usr/bin/env python3
"""
estimate_current.py -- estimate the water current each open-water recording
was made in, from the logs themselves.

    python estimate_current.py --data-dir <logs> [--params calibrated_parameters.yaml] \
        [--spec vehicle_spec.yaml] [--sampling-rate-hz 10] [--out <logs>/currents.csv]
    (or: python strathe.py estimate-current --data <logs>; or let
     `strathe calibrate --estimate-current` do it as part of calibrating)

Why: a calibration assumes it knows the water the vehicle moved through. A
current it doesn't know about shows up as unexplained error and skews every
drag value (PROJECT_STATUS.md #55). Most labs don't have an ADCP reading for
each dive; this recovers the current from how the vehicle actually moved
under its logged thrust.

How: for each recording, the constant current (north, east, down; m/s, in
the logs' world frame) that makes the vehicle model's short-term (3 s)
predictions best match the logged velocities. The model is your calibration
(--params) or, without one, the vehicle description's defaults; an
uncalibrated model gives rougher currents, which is why
`strathe calibrate --estimate-current` alternates: calibrate, estimate the
currents with that calibration, calibrate again with them.

Checks it reports for each recording, in plain words:
  - whether the vehicle model was a calibration or the rough defaults (a
    drag error along the vehicle's path looks just like a current)
  - how much of the model's error the current explains (information)
  - whether the current looks constant: it is also estimated separately on
    the first and second half of the recording; if the two disagree by more
    than max(5 cm/s, 30% of the speed), the current changed during the
    recording (or the model can't separate it from drag), and the estimate
    should be treated with care
  - how much the vehicle turned: a vehicle that holds one heading can't tell
    a current from a drag error along that heading

Writes currents.csv (file, vx_mps, vy_mps, vz_mps, ...; calibration reads it
automatically when it sits in the logs folder) and current_estimates.md.
Limit: one constant current per recording. A current that changes within a
recording, or varies from place to place, isn't modelled.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd

CHANGE_ABS_MPS = 0.05
CHANGE_REL = 0.30
MIN_TURN_DEG = 90.0
WINDOW_S = 3.0


def _fit(run, params, vehicle, horizon, scales, c0=None):
    from scipy.optimize import least_squares
    from objective import residuals_for_run, run_with_current

    def f(c):
        return residuals_for_run(params, run_with_current(run, c), vehicle, scales, True, None, horizon)
    r = least_squares(f, np.zeros(3) if c0 is None else c0, bounds=(-3.0, 3.0), method="trf", x_scale=0.1,
                      diff_step=1e-3, max_nfev=60)
    return r.x, float(np.sqrt(np.mean(r.fun ** 2)))


def _half(run, a, b):
    from dataclasses import replace
    if run.channels is not None:   # the sensor suite (#96): its channels and window starts go with it
        from measurements import slice_run
        return replace(slice_run(run, a, b), current_body=None)
    return replace(run, forces_torques=run.forces_torques[a:b], roll_pitch=run.roll_pitch[a:b],
                   real_nu=run.real_nu[a:b], nu0=run.real_nu[a].copy(), current_body=None,
                   orientation_quat=run.orientation_quat[a:b],
                   controls=None if run.controls is None else run.controls[a:b])


def estimate_run(job) -> dict:
    """job: (filepath, params_vector, vehicle, dataset_config). One recording's current and its checks. A recording on
    which these parameters make the simulation blow up gets no current (still water, flagged `unstable`) instead of
    stopping every other recording's estimate and the calibration around it (PROJECT_STATUS.md #76)."""
    try:
        with np.errstate(over="ignore", invalid="ignore"):  # an unstable recording is reported below, not as warnings
            return _estimate_run(job)
    except FloatingPointError as e:
        return {"file": os.path.basename(job[0]), "vx_mps": 0.0, "vy_mps": 0.0, "vz_mps": 0.0, "speed_mps": 0.0,
                "direction_deg": 0.0, "first_half": None, "second_half": None, "half_difference_mps": None,
                "looks_constant": None, "heading_range_deg": 0.0, "error_explained_pct": 0.0,
                "error_still_water": None, "error_with_current": None, "samples": None, "unstable": str(e)[:200]}


def _estimate_run(job) -> dict:
    import os as _os
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        _os.environ[var] = "1"
    from scipy.spatial.transform import Rotation
    from dynamics_model import CalibrationParameters
    from objective import compute_zero_baseline_scales, load_run, residuals_for_run
    path, pvec, vehicle, dc = job
    params = CalibrationParameters.from_vector(pvec)
    run = load_run(path, dc)
    horizon = max(2, int(round(WINDOW_S * run.sample_rate_hz)))
    if run.channels is not None:
        # the sensor suite (#96): windows start from the smoother's velocity with these parameters, each sensor scaled
        # by its own still RMS. A position fix or a bottom-lock DVL measures motion over ground: the current shows
        import measurements as ms
        import state_smoother as sm
        sigma, _ = sm.choose_accel_sigma([run], params, vehicle, dc.sensor_noise)
        sm.set_anchors([run], params, vehicle, sigma, dc.sensor_noise)
        run.sensor_scales = ms.still_scales([run], max(horizon, int(round(10.0 * run.sample_rate_hz))))
        horizon = max(horizon, int(round(10.0 * run.sample_rate_hz)))   # position needs time to move
    scales = compute_zero_baseline_scales([run])
    still = float(np.sqrt(np.mean(residuals_for_run(params, run, vehicle, scales, True, None, horizon) ** 2)))
    c, err = _fit(run, params, vehicle, horizon, scales)
    n = run.n_samples
    halves = []
    if n >= 4 * horizon:
        for a, b in ((0, n // 2), (n // 2, n)):
            halves.append(_fit(_half(run, a, b), params, vehicle, horizon, scales, c0=c)[0])
    yaw = np.unwrap(Rotation.from_quat(run.orientation_quat).as_euler("xyz")[:, 2])
    turn = float(np.degrees(yaw.max() - yaw.min()))
    speed = float(np.linalg.norm(c[:2]))
    change = float(np.linalg.norm(halves[0] - halves[1])) if halves else None
    steady = None if change is None else change <= max(CHANGE_ABS_MPS, CHANGE_REL * float(np.linalg.norm(c)))
    return {"file": os.path.basename(path), "vx_mps": float(c[0]), "vy_mps": float(c[1]), "vz_mps": float(c[2]),
            "speed_mps": speed, "direction_deg": float(np.degrees(np.arctan2(c[1], c[0])) % 360.0),
            "first_half": None if not halves else [float(x) for x in halves[0]],
            "second_half": None if not halves else [float(x) for x in halves[1]],
            "half_difference_mps": change, "looks_constant": steady, "heading_range_deg": turn,
            "error_explained_pct": 100.0 * (1 - err / still) if still > 0 else 0.0,
            "error_still_water": still, "error_with_current": err, "samples": n}


def estimate_currents(files: list, params, vehicle, dataset_config, workers: int = 1) -> list:
    jobs = [(f, params.to_vector(), vehicle, dataset_config) for f in files]
    if workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            return list(ex.map(estimate_run, jobs))
    return [estimate_run(j) for j in jobs]


def verdict(r: dict) -> str:
    if r.get("unstable"):
        return ("no current estimated: with these parameters the simulation blew up on this recording, so it's held "
                "at still water. The parameters are unstable for its motion; check the calibration's limits")
    notes = []
    if r.get("model") == "default":
        notes.append("estimated with the uncalibrated default model, so rough: a drag error along the vehicle's "
                     "path looks just like a current (in tests, 20-60 cm/s off); estimate with a calibration "
                     "(--params), or use `strathe calibrate --estimate-current`, which alternates the two")
    if r["looks_constant"] is False:
        notes.append(f"the first and second halves disagree by {100 * r['half_difference_mps']:.0f} cm/s: the "
                     f"current changed during this recording, or the model can't separate it from drag; treat "
                     f"with care")
    elif r["looks_constant"] is None:
        notes.append("too short to check whether the current stayed constant")
    if r["heading_range_deg"] < MIN_TURN_DEG:
        notes.append(f"the vehicle only turned {r['heading_range_deg']:.0f} deg; a current along one heading is "
                     f"hard to tell from a drag error along it")
    return "; ".join(notes) if notes else "steady, and the vehicle turned enough to pin it down"


def write_outputs(rows: list, out_csv: str, model_label: str) -> str:
    df = pd.DataFrame([{k: r[k] for k in ("file", "vx_mps", "vy_mps", "vz_mps", "speed_mps", "direction_deg",
                                          "half_difference_mps", "looks_constant", "heading_range_deg",
                                          "error_explained_pct")} for r in rows])
    df.to_csv(out_csv, index=False, float_format="%.4f")
    md = os.path.join(os.path.dirname(os.path.abspath(out_csv)), "current_estimates.md")
    L = ["# Estimated water currents\n",
         f"One constant current per recording, estimated from how the vehicle moved under its logged thrust, "
         f"with {model_label}. Directions are where the water flows toward, degrees clockwise from north (the "
         f"logs' world x axis). Calibration reads `{os.path.basename(out_csv)}` automatically when it's in the logs "
         f"folder.\n", "| Recording | Current | Toward | Error explained | Check |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['file']} | {100 * r['speed_mps']:.0f} cm/s (vertical {100 * r['vz_mps']:+.0f}) | "
                 f"{r['direction_deg']:.0f}° | {r['error_explained_pct']:.0f}% | {verdict(r)} |")
    flagged = sum(1 for r in rows if r["looks_constant"] is False)
    unstable = [r["file"] for r in rows if r.get("unstable")]
    if rows and rows[0].get("model") == "default":
        L.append("\n**Rough estimates:** made with the uncalibrated default model. Use them only as a starting "
                 "point; `strathe calibrate --estimate-current` refines them with a calibration.")
    L.append("")
    L.append(f"**{flagged} of {len(rows)} recording(s)** look like the current changed during them."
             if flagged else "Every recording's current looks constant over the recording.")
    if unstable:
        L.append(f"\n**No current for {len(unstable)} recording(s)** ({', '.join(unstable)}): the simulation blew up "
                 f"on them with these parameters, so they're held at still water.")
    L.append("\nLimit: this estimates one constant current per recording. A current that changes within a "
             "recording, or varies from place to place, isn't modelled (see LIMITS.md).")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(os.path.dirname(os.path.abspath(out_csv)), "current_estimates.json"), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return md


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="logs in Strathe's CSV layout")
    ap.add_argument("--params", help="calibrated_parameters.yaml to use as the vehicle model "
                                     "(default: the vehicle description's defaults, rougher)")
    ap.add_argument("--spec", default="vehicle_spec.yaml")
    ap.add_argument("--sampling-rate-hz", type=float, help="default: ingest_report.json, else 10")
    ap.add_argument("--torque-reference-offset-m", type=float, nargs=3,
                    help="default: ingest_report.json, else vehicle_spec.yaml")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", help="default: <data-dir>/currents.csv")
    ap.add_argument("--sensors", nargs="+", default=["auto"], help="as run_calibration.py --sensors (#96)")
    args = ap.parse_args(argv)
    from build_two_agent_cores import load_calibrated_parameters
    from data_split import discover_csv_files
    from objective import DatasetConfig
    from optimizer import initial_parameters, load_vehicle_spec
    from run_calibration import spec_torque_reference_offset
    info = {}
    ip = os.path.join(args.data_dir, "ingest_report.json")
    if os.path.exists(ip):
        with open(ip, encoding="utf-8") as f:
            info = json.load(f)
    spec = load_vehicle_spec(args.spec)
    rate = args.sampling_rate_hz or info.get("sampling_rate_hz") or 10.0
    offset = args.torque_reference_offset_m or info.get("torque_reference_offset_m") or \
        spec_torque_reference_offset(spec)
    files, _ = discover_csv_files(args.data_dir)
    if not files:
        print(f"ERROR: no logs in {args.data_dir}", file=sys.stderr)
        return 2
    params = load_calibrated_parameters(args.params) if args.params else initial_parameters(spec)
    label = (f"the calibration `{os.path.basename(os.path.dirname(os.path.abspath(args.params)))}`" if args.params
             else "the vehicle description's default (uncalibrated) values")
    import measurements as ms
    from run_calibration import _sensor_mounts
    noise, _ = ms.smoother_noise(args.data_dir, ms.sensor_mounts(spec.raw), [])   # as calibrate's smoother has it
    late = {k: v[0] for k, v in ms.given_delays(args.data_dir, ms.sensor_mounts(spec.raw)).items()}
    mp = os.path.join(os.path.dirname(os.path.abspath(args.params)), "metrics.json") if args.params else ""
    if os.path.exists(mp):   # the delay the calibration took off a late sensor's fixes
        with open(mp, encoding="utf-8") as f:
            late = {k: d["applied_s"] for k, d in ((json.load(f).get("sensor_suite") or {}).get("fix_delays")
                                                    or {}).items() if "applied_s" in d}
    dc = DatasetConfig(float(rate), tuple(offset), None, sensors=tuple(args.sensors), lever_arms=_sensor_mounts(spec),
                       sensor_noise=noise, fix_delays=tuple((k, v) for k, v in sorted(late.items()) if v) or None,
                       actuators=spec.vehicle.actuators)
    print(f"Estimating the current in {len(files)} recording(s) at {rate:g} Hz with {label}...", flush=True)
    rows = estimate_currents(files, params, spec.vehicle, dc, args.workers)
    for r in rows:
        r["model"] = "calibration" if args.params else "default"
    out = args.out or os.path.join(args.data_dir, "currents.csv")
    md = write_outputs(rows, out, label)
    for r in rows:
        if r.get("unstable"):
            print(f"  {r['file']:<40} {verdict(r)}")
            continue
        print(f"  {r['file']:<40} {100 * r['speed_mps']:5.1f} cm/s toward {r['direction_deg']:5.0f} deg  "
              f"({r['error_explained_pct']:.0f}% of the error explained) - {verdict(r)}")
    print(f"Wrote {out} and {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
