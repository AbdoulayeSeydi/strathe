"""
Tests whether MuJoCo's residual gradient (as trf's own finite-difference
Jacobian would see it) is STABLE across step sizes, or swings wildly --
the specific, cheaper hypothesis proposed after the multi-hour
calibration experiment showed MuJoCo's optimization progress collapsing
~40x in rate partway through, without reaching anywhere near the
stand-in's converged cost.

DELIBERATELY SMALL, for speed (see conversation -- a full 22-parameter
Jacobian at 3 step sizes would take over an hour at MuJoCo's measured
~8.6s/call): tests only 5 parameters, chosen to span two contrasting
groups --
  - "rotational/restoring" (cg_cb_offset_z, Mq, Ieff_pitch): directly
    tied to the near-gimbal-lock, near-inversion dynamics already
    confirmed central to this whole investigation. If the gradient
    problem is specifically THERE, these should show it.
  - "translational" (Xu, Xu_dot): unrelated to rotational dynamics, as
    a contrast group. If these are ALSO unstable, the problem is more
    general than the rotational-dynamics hypothesis suggests.

Tests at TWO points: x0 (where MuJoCo's fit was still making real
progress) and the point where MuJoCo's fit actually stalled (taken
directly from the most recent calibration experiment's own printed
MuJoCo-fit values) -- comparing gradient stability at these two points
could directly explain "good early progress, then stall" if instability
is worse at the second point.

Uses forward differences (1 extra eval per param per step size, not
central difference's 2) to keep this fast, at some cost in estimate
quality -- appropriate for a quick stability check, not a precise
gradient value.

*** UNVERIFIED against real data in this sandbox -- no real CSV files,
no real calibrated_parameters_a1.yaml, in this sandbox. This has NOT
been run against real data before being handed off.
"""
import argparse
import time

import numpy as np
import mujoco

import objective
from objective import DatasetConfig, joint_residuals
from optimizer import load_vehicle_spec, initial_parameters
from dynamics_model import CalibrationParameters
from mujoco_backend import make_mujoco_simulate_fn, simulate_trajectory_mujoco

MJCF_PATH = "b0_single_body_placeholder.xml"
VEHICLE_SPEC_PATH = "vehicle_spec.yaml"

# Directly from the most recent calibration experiment's own printed
# MuJoCo-fit column (see conversation) -- the point where its
# optimization progress collapsed.
MUJOCO_STALL_POINT_KWARGS = dict(
    Ieff_roll=0.2642, Ieff_pitch=0.2301, Ieff_yaw=0.4898,
    Xu=-9.2372, Yv=-15.6986, Zw=-42.9097,
    Kp=-0.2537, Mq=-0.5961, Nr=-1.1518,
    Xuu=-65.2373, Yvv=-99.8429, Zww=-102.8253,
    Kpp=-0.5924, Mqq=-0.6322, Nrr=-2.0404,
    Xu_dot=4.9776, Yv_dot=15.3327, Zw_dot=19.6128,
    net_buoyancy_z=0.4274, cg_cb_offset_z=0.0214,
    Xq_dot=0.1778, Zq_dot=-0.1337,
)

TESTED_PARAMS = ["cg_cb_offset_z", "Mq", "Ieff_pitch", "Xu", "Xu_dot"]
RELATIVE_STEP_SIZES = [1e-6, 1e-4, 1e-2]


def _load_runs(data_dir, dates, sampling_rate_hz, limit=None):
    from data_split import discover_csv_files, extract_date
    all_files, _ = discover_csv_files(data_dir)
    selected = sorted(f for f in all_files if extract_date(f) in dates)
    if limit is not None:
        selected = selected[:limit]
    dataset_config = DatasetConfig(sampling_rate_hz=sampling_rate_hz)
    runs = [objective.load_run(f, dataset_config) for f in selected]
    return runs


def _cost(param_vec, runs, vehicle, simulate_fn):
    residuals = joint_residuals(param_vec, runs, vehicle, simulate_fn=simulate_fn)
    return 0.5 * float(np.sum(residuals ** 2))


def _test_point(label, x_vec, param_names, runs, vehicle, simulate_fn, simulate_fn_label):
    print(f"\n  --- {label}, evaluated with {simulate_fn_label} ---")
    t0 = time.monotonic()
    base_cost = _cost(x_vec, runs, vehicle, simulate_fn)
    print(f"    base cost = {base_cost:.6f} ({time.monotonic() - t0:.1f}s)")

    results = {}
    for pname in TESTED_PARAMS:
        idx = param_names.index(pname)
        x_val = x_vec[idx]
        derivs = []
        for rel_step in RELATIVE_STEP_SIZES:
            h = rel_step * max(1.0, abs(x_val))
            x_perturbed = x_vec.copy()
            x_perturbed[idx] = x_val + h
            t0 = time.monotonic()
            perturbed_cost = _cost(x_perturbed, runs, vehicle, simulate_fn)
            dt = time.monotonic() - t0
            d_cost = (perturbed_cost - base_cost) / h
            derivs.append(d_cost)
            print(f"    {pname:<16} rel_step={rel_step:<8.0e} h={h:<12.6g} "
                  f"d(cost)/d({pname})={d_cost:>14.4f}  ({dt:.1f}s)")
        derivs = np.array(derivs)
        nonzero = derivs[np.abs(derivs) > 1e-12]
        if len(nonzero) >= 2:
            spread_ratio = np.max(np.abs(nonzero)) / np.min(np.abs(nonzero))
        else:
            spread_ratio = float("nan")
        flag = "  <-- UNSTABLE (>10x spread across step sizes)" if spread_ratio > 10 else ""
        print(f"    {pname:<16} spread ratio (max/min |deriv|) across step sizes: "
              f"{spread_ratio:.2f}x{flag}")
        results[pname] = {"derivs": derivs, "spread_ratio": spread_ratio}
    return results


def _make_teacher_forced_simulate_fn(model, data):
    # make_mujoco_simulate_fn never exposes teacher_forced (always uses
    # the forward-sim default) -- this wrapper follows the exact same
    # pattern, but passes teacher_forced=True explicitly, to test whether
    # the gradient instability found at the stall point is specific to
    # self-integrated orientation (which teacher-forced mode never does).
    def simulate_fn(params, vehicle, sim_config, trajectory):
        return simulate_trajectory_mujoco(params, vehicle, sim_config, trajectory, model, data,
                                            teacher_forced=True)
    return simulate_fn


def run_diagnostic(data_dir: str, sampling_rate_hz: float,
                    vehicle_spec_path: str = VEHICLE_SPEC_PATH):
    from phase2_config import PHASE1_TRAIN_DATES

    loaded = load_vehicle_spec(vehicle_spec_path)
    vehicle = loaded.vehicle
    param_names = initial_parameters(loaded).param_names
    x0 = initial_parameters(loaded).to_vector()
    stall_point = CalibrationParameters(**MUJOCO_STALL_POINT_KWARGS).to_vector()

    train_runs = _load_runs(data_dir, PHASE1_TRAIN_DATES, sampling_rate_hz, limit=2)
    print(f"Testing gradient stability on {len(train_runs)} run(s), "
          f"parameters: {TESTED_PARAMS}, step sizes: {RELATIVE_STEP_SIZES}")

    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    mujoco_simulate_fn = make_mujoco_simulate_fn(model, data)

    all_results = {}
    print("\n" + "=" * 70)
    print("TEST POINT 1: x0 (initial guess -- where MuJoCo was still making progress)")
    print("=" * 70)
    all_results[("x0", "stand-in")] = _test_point(
        "x0", x0, param_names, train_runs, vehicle, None, "stand-in")
    all_results[("x0", "mujoco")] = _test_point(
        "x0", x0, param_names, train_runs, vehicle, mujoco_simulate_fn, "mujoco")

    print("\n" + "=" * 70)
    print("TEST POINT 2: MuJoCo's stalled fit point (where progress collapsed)")
    print("=" * 70)
    all_results[("stall", "stand-in")] = _test_point(
        "stall point", stall_point, param_names, train_runs, vehicle, None, "stand-in")
    all_results[("stall", "mujoco")] = _test_point(
        "stall point", stall_point, param_names, train_runs, vehicle, mujoco_simulate_fn, "mujoco")

    # NEW: teacher-forced, at the stall point specifically -- where the
    # instability was actually worst. Fresh model/data, since the earlier
    # mujoco_simulate_fn calls above have left this model/data in some
    # arbitrary internal state from their own last substep.
    model2 = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data2 = mujoco.MjData(model2)
    teacher_forced_simulate_fn = _make_teacher_forced_simulate_fn(model2, data2)
    all_results[("stall", "teacher_forced")] = _test_point(
        "stall point", stall_point, param_names, train_runs, vehicle,
        teacher_forced_simulate_fn, "mujoco (teacher_forced=True)")

    print("\n" + "=" * 70)
    print("SUMMARY: spread ratio (max/min |d(cost)/d(param)| across step sizes)")
    print("=" * 70)
    print(f"{'param':<16} {'x0, stand-in':>14} {'x0, mujoco':>14} "
          f"{'stall, stand-in':>16} {'stall, mujoco':>16} {'stall, teacher_forced':>22}")
    for pname in TESTED_PARAMS:
        row = [all_results[(pt, fn)][pname]["spread_ratio"]
               for pt, fn in [("x0", "stand-in"), ("x0", "mujoco"),
                              ("stall", "stand-in"), ("stall", "mujoco"),
                              ("stall", "teacher_forced")]]
        print(f"{pname:<16} {row[0]:>14.2f} {row[1]:>14.2f} {row[2]:>16.2f} "
              f"{row[3]:>16.2f} {row[4]:>22.2f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("If stand-in spread ratios stay low (near 1x) everywhere while MuJoCo's are")
    print("consistently high (>10x), that's concrete evidence the gradient ITSELF is")
    print("unreliable for MuJoCo -- not just that the landscape is harder. If this is")
    print("WORSE at the stall point than at x0, that would directly explain the observed")
    print("pattern (real early progress, then stalling) -- the optimizer would be running")
    print("on decreasingly trustworthy gradient information as it approaches that region.")
    print("If MuJoCo's rotational-group parameters (cg_cb_offset_z, Mq, Ieff_pitch) show")
    print("much worse spread than the translational ones (Xu, Xu_dot), that specifically")
    print("implicates the near-gimbal-lock/restoring-force dynamics, not a generic MuJoCo")
    print("noise problem across all parameters equally.")
    print()
    print("The 'stall, teacher_forced' column is the direct, practical test: if it stays")
    print("low like the stand-in's (rather than exploding like forward-sim MuJoCo's), that")
    print("confirms teacher_forced=True sidesteps this instability entirely, since it never")
    print("self-integrates orientation into the region where this occurs -- supporting the")
    print("existing recommendation to keep it as the practical calibration default,")
    print("independent of whatever the underlying MuJoCo integration mechanism turns out")
    print("to be.")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    args = parser.parse_args()
    run_diagnostic(args.data_dir, args.sampling_rate_hz)