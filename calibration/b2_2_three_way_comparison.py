"""
B2.2 -- three-way held-out RMSE comparison: stand-in, HoloOcean
(already-recorded), MuJoCo (this track's own new number). Explicitly
labeled as NOT directly comparable (teacher-forced-vs-forward-simulation
caveat, plus HoloOcean and MuJoCo are different backends entirely) --
useful only as a directional check: is MuJoCo's error the same rough
order of magnitude as the other two, or is something clearly broken.

Reuses B2.1's own stand-in/MuJoCo computation directly (b2_1_heldout_rmse
.run_b2_1) rather than recomputing it separately -- avoids having two
places that could silently drift apart on what "MuJoCo's held-out RMSE"
means.

HoloOcean's number is NOT computed here and is NOT hardcoded -- it must
be supplied explicitly, either as a direct --holoocean-heldout-rmse value
or via --holoocean-metrics-json pointing at the real, already-saved
metrics file, with an explicit --holoocean-metrics-key naming which field
in that file to use (schemas differ across HoloOcean calibration runs in
this project's own history -- e.g. "fitted_rmse_holoocean" vs
"rmse_heldout_calibrated" -- so this is not guessed).

*** UNVERIFIED against real data in this sandbox -- no real CSV files,
no real calibrated_parameters_a1.yaml, no real HoloOcean metrics file,
in this sandbox. This has NOT been run against real data before being
handed off.
"""
import argparse
import json

from b2_1_heldout_rmse import run_b2_1


def _load_holoocean_rmse(direct_value, metrics_json_path, metrics_key):
    if direct_value is not None:
        return direct_value, "--holoocean-heldout-rmse (direct value)"
    if metrics_json_path is not None:
        if metrics_key is None:
            raise ValueError("--holoocean-metrics-json requires --holoocean-metrics-key "
                              "(the field name in that file to read) -- schemas differ "
                              "across this project's own HoloOcean calibration runs, so "
                              "this is not guessed.")
        with open(metrics_json_path, encoding="utf-8") as f:
            metrics = json.load(f)
        if metrics_key not in metrics:
            raise KeyError(f"{metrics_key!r} not found in {metrics_json_path}. "
                            f"Available keys: {sorted(metrics.keys())}")
        return metrics[metrics_key], f"{metrics_json_path} [{metrics_key}]"
    raise ValueError("Must supply either --holoocean-heldout-rmse or "
                      "--holoocean-metrics-json + --holoocean-metrics-key.")


def run_b2_2(data_dir: str, sampling_rate_hz: float,
             holoocean_rmse_direct: float = None,
             holoocean_metrics_json: str = None, holoocean_metrics_key: str = None,
             calibrated_params_path: str = "calibrated_parameters_a1.yaml"):
    holoocean_rmse, holoocean_source = _load_holoocean_rmse(
        holoocean_rmse_direct, holoocean_metrics_json, holoocean_metrics_key)

    print("Running B2.1's own stand-in/MuJoCo held-out RMSE computation...")
    _, standin_rmse, mujoco_rmse = run_b2_1(data_dir, sampling_rate_hz, calibrated_params_path)

    print("\n" + "=" * 70)
    print("B2.2 THREE-WAY HELD-OUT RMSE COMPARISON")
    print("=" * 70)
    print(f"{'Backend':<20} {'Held-out RMSE':>15}   Source")
    print(f"{'Stand-in':<20} {standin_rmse:>15.6f}   this run, calibrated_parameters_a1.yaml, "
          f"simulate_fn=None")
    print(f"{'HoloOcean':<20} {holoocean_rmse:>15.6f}   {holoocean_source}")
    print(f"{'MuJoCo':<20} {mujoco_rmse:>15.6f}   this run, calibrated_parameters_a1.yaml, "
          f"make_mujoco_simulate_fn (forward-sim)")

    print()
    print("NOT DIRECTLY COMPARABLE, by design:")
    print("  - Stand-in and MuJoCo here are both teacher-forced-vs-forward-sim as already")
    print("    established (MuJoCo uses forward-sim per this task's own literal wording);")
    print("    HoloOcean's own number was produced by an entirely separate backend, on its")
    print("    own calibration run, not necessarily the identical held-out runs or the")
    print("    identical parameter set as calibrated_parameters_a1.yaml used here.")
    print("  - This is a DIRECTIONAL check only: is MuJoCo's error the same rough order of")
    print("    magnitude as the other two, or is something clearly broken (e.g. orders of")
    print("    magnitude larger, non-finite, or wildly inconsistent).")

    ratio_to_standin = mujoco_rmse / standin_rmse if standin_rmse > 0 else float("inf")
    ratio_to_holoocean = mujoco_rmse / holoocean_rmse if holoocean_rmse > 0 else float("inf")
    print(f"\nMuJoCo / stand-in ratio:   {ratio_to_standin:.2f}x")
    print(f"MuJoCo / HoloOcean ratio:  {ratio_to_holoocean:.2f}x")

    return {"standin_rmse": standin_rmse, "holoocean_rmse": holoocean_rmse, "mujoco_rmse": mujoco_rmse}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sampling-rate-hz", type=float, required=True)
    parser.add_argument("--holoocean-heldout-rmse", type=float, default=None,
                         help="Direct value, if you already know the number.")
    parser.add_argument("--holoocean-metrics-json", default=None,
                         help="Path to HoloOcean's already-saved metrics.json, as an "
                              "alternative to --holoocean-heldout-rmse.")
    parser.add_argument("--holoocean-metrics-key", default=None,
                         help="Field name to read from --holoocean-metrics-json "
                              "(required if that flag is used).")
    parser.add_argument("--calibrated-params", default="calibrated_parameters_a1.yaml")
    args = parser.parse_args()
    run_b2_2(args.data_dir, args.sampling_rate_hz,
              args.holoocean_heldout_rmse, args.holoocean_metrics_json, args.holoocean_metrics_key,
              args.calibrated_params)