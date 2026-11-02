"""
stress/solaqua_96.py -- the state smoother on a real USBL (#96): SOLAQUA's BlueROV2 at a fish farm, its velocity
estimated from the USBL's 0.5 Hz fixes and depth alone (no physics, no DVL), against its Water Linked A50 DVL.

    python stress/solaqua_96.py [bags folder]      (default: data/raw/solaqua; records row S-solaqua)

SOLAQUA has no thruster forces, so it can't calibrate anything; it is the only public log we have with a real USBL,
so it checks the part of the sensor suite that turns fixes into the velocity each window starts from. Two things
stand between the two sensors (PROJECT_STATUS.md #60), and are fitted, as #60 did, before comparing:
  - the DVL locks onto the fish-farm net, which moves with the current: a constant drift per 20 s window;
  - ArduSub's heading and the USBL's east/north disagree by a rotation (#60 found -1.48 rad): one per recording,
    with the handedness either way.
The comparison is in the world frame, horizontal only: the smoother's velocity over ground (its body velocity turned
by the logged attitude) against the DVL's body velocity turned the same way, then rotated and drift-corrected.
Skill per recording: 1 - RMS(difference) / RMS(DVL velocity after the drift is removed): 1 is perfect, 0 is no
better than saying the vehicle didn't move.

Bar (written before the first run): median skill over the in-water recordings >= 0.3, and above 0 in most of them.
"""
from __future__ import annotations

import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.dirname(HERE)
sys.path.insert(0, CAL)
sys.path.insert(0, HERE)
BAGS = os.path.join(os.path.dirname(CAL), "data", "raw", "solaqua")
RATE = 10.0
WINDOW_S = 20.0
BAR_MEDIAN = 0.3
MAX_SPEED_MPS = 5.0     # PositionSpec's default
USBL_NOISE_M = 0.11     # #60's measured USBL fix error


def run_frame(d: dict):
    """A canonical frame (no forces: nothing is simulated) on a 10 Hz grid over the attitude stream, with the USBL's
    fixes and the depth placed where they read; and the DVL's valid readings on the same grid."""
    import sensor_ingest as si
    from scipy.spatial.transform import Rotation
    at, us, dp, dv = (d[k] for k in ("/sensor/attitude", "/sensor/usbl", "/sensor/depth_temperature",
                                       "/sensor/dvl_velocity"))
    t = np.arange(at[0, 0], at[-1, 0], 1.0 / RATE)
    unwrap = np.c_[at[:, 1], at[:, 2], np.unwrap(at[:, 3])]
    rpy = np.column_stack([np.interp(t, at[:, 0], unwrap[:, i]) for i in range(3)])
    rates = np.column_stack([np.interp(t, at[:, 0], at[:, i]) for i in (4, 5, 6)])
    q = Rotation.from_euler("ZYX", rpy[:, ::-1]).as_quat()     # body FRD -> world NED
    df = pd.DataFrame(np.zeros((len(t), 6)), columns=["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"])
    df[["Bp", "Bq", "Br"]] = rates
    df[["qx", "qy", "qz", "qw"]] = q
    # as ingestion does with a position: block's defaults: fixes the vehicle couldn't reach at 5 m/s are dropped
    tu, fixes, _ = si.drop_unreachable_fixes(us[:, 0], np.c_[us[:, 2], us[:, 1]], MAX_SPEED_MPS,
                                             3.0 * np.sqrt(2.0) * USBL_NOISE_M)
    ne = si.place_on_grid(tu, fixes, t)                                     # north, east
    df["pos_usbl_n"], df["pos_usbl_e"] = ne[:, 0], ne[:, 1]
    df["depth_main"] = si.place_on_grid(dp[:, 0], dp[:, 1:2], t)[:, 0]
    ok = dv[:, 4] > 0.5
    dvl = si.place_on_grid(dv[ok, 0], dv[ok, 1:4], t)
    return df, dvl, t


def compare(world_s: np.ndarray, world_d: np.ndarray, t: np.ndarray) -> dict:
    """Fit one rotation (and handedness) per recording and a drift per window, then score."""
    good = np.isfinite(world_d).all(1) & np.isfinite(world_s).all(1)
    win = np.floor((t - t[0]) / WINDOW_S).astype(int)
    best = None
    for flip in (1.0, -1.0):
        d = world_d.copy()
        d[:, 1] *= flip
        for ang in np.radians(np.arange(-180, 180, 1.0)):
            c, s = np.cos(ang), np.sin(ang)
            r = d @ np.array([[c, s], [-s, c]])
            res, sig = [], []
            for w in np.unique(win[good]):
                m = good & (win == w)
                if m.sum() < 50:
                    continue
                drift = np.mean(r[m] - world_s[m], 0)           # the net's motion over the window
                res.append(world_s[m] - (r[m] - drift))
                sig.append(r[m] - np.mean(r[m], 0))
            if not res:
                return {}
            res, sig = np.vstack(res), np.vstack(sig)
            rms = float(np.sqrt(np.mean(res ** 2)))
            if best is None or rms < best["rms_mps"]:
                best = {"rms_mps": rms, "still_rms_mps": float(np.sqrt(np.mean(sig ** 2))), "angle_deg": float(
                    np.degrees(ang)), "flip": flip, "samples": int(len(res))}
    best["skill"] = 1.0 - best["rms_mps"] / best["still_rms_mps"] if best["still_rms_mps"] > 0 else float("nan")
    return best


def one(path: str) -> dict:
    import state_smoother as sm
    from fit_solaqua_noise import read_bag
    from objective import DatasetConfig, convert_run_dataframe
    d = read_bag(path)
    if min(len(d[k]) for k in ("/sensor/attitude", "/sensor/usbl", "/sensor/dvl_velocity")) < 20:
        return {}
    df, dvl, t = run_frame(d)
    run = convert_run_dataframe(df, f"solaqua-20-08-2024-{os.path.basename(path)[11:19]}.csv",
                                DatasetConfig(sampling_rate_hz=RATE, sensors=("all",)))
    noise = {"usbl": USBL_NOISE_M, "main": 0.0024}             # #60's measured USBL fix error and depth noise
    sigma, _ = sm.choose_accel_sigma([run], None, None, noise)
    res = sm.smooth(run, None, None, sigma, noise)
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(df[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    world_s = np.einsum("nij,nj->ni", R, res.velocity)[:, :2]
    world_d = np.einsum("nij,nj->ni", R, dvl)[:, :2]
    out = compare(world_s, world_d, t)
    if out:
        out.update({"recording": os.path.basename(path)[:19], "accel_sigma": sigma,
                    "fixes": int(np.isfinite(df["pos_usbl_n"]).sum()), "fallback": res.fallback})
        out["control_skill"] = control(df, world_d, t, noise, int(path.encode().__len__()))
    return out


def control(df, world_d, t, noise, seed) -> float:
    """What these fixes could show at best: the same smoother on fixes made from the DVL's own track (its velocity,
    turned to the world and integrated), at the real fixes' times, with the USBL's measured noise. If this is near or
    below 0 too, the limit is what 0.5 Hz fixes at this noise can resolve of this motion, not the smoother."""
    import state_smoother as sm
    from objective import DatasetConfig, convert_run_dataframe
    good = np.isfinite(world_d).all(1)
    if good.sum() < 50:
        return float("nan")
    v = np.column_stack([np.interp(t, t[good], world_d[good, i]) for i in range(2)])
    path = np.vstack([np.zeros(2), np.cumsum(v[:-1] / RATE, 0)])
    rng = np.random.default_rng(seed)
    c = df.copy()
    has = np.isfinite(df["pos_usbl_n"]).to_numpy()
    fake = path + USBL_NOISE_M * rng.normal(size=path.shape)
    c["pos_usbl_n"], c["pos_usbl_e"] = np.where(has, fake[:, 0], np.nan), np.where(has, fake[:, 1], np.nan)
    run = convert_run_dataframe(c, "solaqua-20-08-2024-control.csv", DatasetConfig(sampling_rate_hz=RATE,
                                                                                    sensors=("all",)))
    sigma, _ = sm.choose_accel_sigma([run], None, None, noise)
    res = sm.smooth(run, None, None, sigma, noise)
    from scipy.spatial.transform import Rotation
    R = Rotation.from_quat(c[["qx", "qy", "qz", "qw"]].to_numpy()).as_matrix()
    return compare(np.einsum("nij,nj->ni", R, res.velocity)[:, :2], world_d, t).get("skill", float("nan"))


def main(bags: str = BAGS) -> int:
    from fit_solaqua_noise import FIRST_IN_WATER
    from common import record
    files = [f for f in sorted(glob.glob(os.path.join(bags, "*_data.bag"))) if os.path.basename(f) >= FIRST_IN_WATER]
    if not files:
        print(f"no SOLAQUA data bags in {bags} (see fit_solaqua_noise.py)")
        return 1
    t0 = time.time()
    rows = []
    for f in files:
        try:
            r = one(f)
        except Exception as e:   # a damaged recording: listed, not hidden
            r = {"recording": os.path.basename(f)[:19], "error": f"{type(e).__name__}: {e}"[:200]}
        if r:
            rows.append(r)
            print(r.get("recording"), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                                       if k != "recording"}, flush=True)
    sk = np.array([r["skill"] for r in rows if np.isfinite(r.get("skill", np.nan))])
    nums = {"recordings": len(rows), "scored": int(len(sk)), "median_skill": float(np.median(sk)) if len(sk) else None,
            "share_above_zero": float(np.mean(sk > 0)) if len(sk) else None,
            "quartiles": [float(x) for x in np.percentile(sk, [25, 75])] if len(sk) else None,
            "median_rms_mps": float(np.median([r["rms_mps"] for r in rows if "rms_mps" in r])) if len(sk) else None,
            "median_still_rms_mps": float(np.median([r["still_rms_mps"] for r in rows if "still_rms_mps" in r]))
            if len(sk) else None,
            "angle_deg_median": float(np.median([r["angle_deg"] for r in rows if "angle_deg" in r])) if len(sk) else None,
            "control_median_skill": float(np.nanmedian([r.get("control_skill", np.nan) for r in rows])),
            "control_share_above_zero": float(np.nanmean([r.get("control_skill", np.nan) > 0 for r in rows
                                                          if np.isfinite(r.get("control_skill", np.nan))])),
            "errors": [r["recording"] for r in rows if "error" in r]}
    ok = bool(len(sk)) and nums["median_skill"] >= BAR_MEDIAN and nums["share_above_zero"] > 0.5
    record("96", "S-solaqua", "real USBL", "SOLAQUA", "the smoother's velocity from a real USBL (0.5 Hz) and depth, "
           "no physics, against the A50 DVL (net drift and heading rotation fitted)", "SINTEF SOLAQUA BlueROV2",
           nums, f"median skill >= {BAR_MEDIAN} and above 0 in most recordings", ok, time.time() - t0)
    print(("[pass] " if ok else "[FAIL] ") + str(nums))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
