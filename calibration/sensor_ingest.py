"""
sensor_ingest.py -- what a vehicle's sensors physically report, turned into canonical sensor channels
(PROJECT_STATUS.md #96). Used by data_ingestion.py.

The canonical run CSV always has the thruster wrench, the gyro rates and the orientation. Body velocity (a DVL, or
motion capture) is one sensor among several, not a requirement. Each other sensor adds its own columns:

    Bu Bv Bw                     DVL / body velocity, m/s, body FRD. NaN where the DVL had no reading
                                 (with linear_gaps: keep), or absent when the vehicle has none
    pos_<name>_n _e _d           a position fix (USBL, GPS, LBL, motion capture), m, world NED. _d is absent
                                 for a fix that has no vertical (GPS)
    depth_<name>                 depth, m, positive down (a depth sensor, or pressure converted)
    acc_x acc_y acc_z            accelerometer specific force, m/s^2, body FRD (includes gravity: about
                                 (0, 0, -9.81) at rest, level)

A sensor's readings are placed on the recording's time grid where they happened: each grid sample holds the mean of
the readings within half a sample of it, and NaN when there were none. They are never interpolated. A 1 Hz USBL fix
interpolated onto a 10 Hz grid would invent nine smooth readings out of every ten, and hide the noise the calibration
has to weigh (the SOLAQUA lesson, PROJECT_STATUS.md #60). Noise is measured on each sensor's own readings, at its own
rate, before they are placed.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

G = 9.80665
LENGTH_UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001}
PRESSURE_UNITS = {"Pa": 1.0, "hPa": 100.0, "kPa": 1000.0, "mbar": 100.0, "bar": 1e5, "psi": 6894.757}
WATER_DENSITY = {"fresh": 997.0, "salt": 1025.0}
ACCEL_UNITS = {"m/s^2": 1.0, "g": G}

# WGS84
_A = 6378137.0
_F = 1.0 / 298.257223563
_E2 = _F * (2.0 - _F)

# Plausibility limits: beyond these a declared unit or frame is almost certainly wrong
MAX_PLAUSIBLE_DEPTH_M = 11000.0
MAX_PLAUSIBLE_DEPTH_RATE_MPS = 5.0
OUTLIER_SHARE_REFUSED = 0.2       # more than this share of position fixes impossible to reach: units or frame
# A generous per-axis noise for each kind of fix (m), for the outlier check's tolerance only: motion capture ~mm,
# USBL/LBL 0.1-1% of range (0.17 m measured on SOLAQUA, #60), standalone GPS 2-5 m (RTK: cm)
FIX_NOISE_M = {"mocap": 0.02, "usbl": 1.0, "lbl": 1.0, "gps": 5.0, "odometry": 1.0, "other": 2.0}
GYRO_SLOPE_TOLERANCE = 3.0        # logged gyro against the orientation's own rates: a factor of 57 is deg vs rad
GYRO_CLEAR_R2 = 0.5               # below this the two don't follow each other well enough to judge unit or sign
GYRO_LAG_WARN_S = 0.25            # a larger offset between gyro and orientation is said
MIN_ROTATION_RADPS = 0.05         # rates below this don't say much about scale
ACCEL_G_TOLERANCE = 0.25          # quasi-static |f| within 25% of 9.81
STATIC_GYRO_RADPS = 0.05
STATIC_ACCEL_SPREAD = 0.5


class SensorError(ValueError):
    """A sensor's data would silently corrupt a calibration (an IngestError is raised from it)."""


# ------------------------------------------------------------------ conversions

def latlon_to_ned(lat_deg: np.ndarray, lon_deg: np.ndarray, lat0_deg: float, lon0_deg: float) -> np.ndarray:
    """Geodetic latitude/longitude (WGS84, on the ellipsoid) -> local north/east (m) on the tangent plane at
    (lat0, lon0). Exact through ECEF, so it holds over any distance a vehicle covers in one recording."""
    lat, lon = np.radians(np.asarray(lat_deg, float)), np.radians(np.asarray(lon_deg, float))
    lat0, lon0 = np.radians(lat0_deg), np.radians(lon0_deg)

    def ecef(la, lo):
        n = _A / np.sqrt(1.0 - _E2 * np.sin(la) ** 2)
        return np.stack([n * np.cos(la) * np.cos(lo), n * np.cos(la) * np.sin(lo), n * (1 - _E2) * np.sin(la)], -1)
    d = ecef(lat, lon) - ecef(np.array(lat0), np.array(lon0))
    sl, cl, so, co = np.sin(lat0), np.cos(lat0), np.sin(lon0), np.cos(lon0)
    north = -sl * co * d[..., 0] - sl * so * d[..., 1] + cl * d[..., 2]
    east = -so * d[..., 0] + co * d[..., 1]
    return np.stack([north, east], -1)


def pressure_to_depth(p: np.ndarray, unit: str, water, surface) -> tuple:
    """Absolute or gauge pressure -> depth (m, positive down), and what surface pressure was used. surface: a value
    in the same unit, or "auto": the lowest 1% of the readings (the vehicle at or near the surface; a vehicle that
    never surfaced gets a constant offset, which calibration doesn't see: it compares depth changes)."""
    pa = np.asarray(p, float) * PRESSURE_UNITS[unit]
    rho = WATER_DENSITY[water] if isinstance(water, str) else float(water)
    if surface == "auto":
        p0 = float(np.percentile(pa, 1)) if len(pa) else 0.0
        how = "auto: the lowest 1% of readings"
    else:
        p0, how = float(surface) * PRESSURE_UNITS[unit], "given"
    return (pa - p0) / (rho * G), {"surface_pa": round(p0, 1), "surface": how, "water_density_kg_m3": rho}


def collapse_repeats(t: np.ndarray, v: np.ndarray) -> tuple:
    """A held reading (the same fix logged again until a new one arrives) isn't a new measurement: keep the first of
    each run of identical readings. Returns (t, v, n_dropped)."""
    if len(t) < 2:
        return t, v, 0
    same = np.all(v[1:] == v[:-1], axis=1)
    keep = np.r_[True, ~same]
    return t[keep], v[keep], int((~keep).sum())


def drop_unreachable_fixes(t: np.ndarray, v: np.ndarray, max_speed_mps: float, allowance_m: float = 0.0) -> tuple:
    """Position fixes the vehicle couldn't have reached from the last good one at max_speed_mps (USBL multipath, a
    GPS jump): dropped. allowance_m is what the fixes' own noise can add (3 x sqrt(2) x their noise), so a noisy but
    honest fix isn't taken for a jump. Walks out from the fix nearest the median position, so one bad first fix
    doesn't condemn the rest. Returns (t, v, n_dropped)."""
    n = len(t)
    if n < 3:
        return t, v, 0
    # start from the fix closest to the median position (robust against a bad first fix)
    med = np.median(v, axis=0)
    start = int(np.argmin(np.linalg.norm(v - med, axis=1)))
    keep = np.zeros(n, dtype=bool)
    keep[start] = True
    for direction in (1, -1):
        last = start
        i = start + direction
        while 0 <= i < n:
            dt = abs(t[i] - t[last])
            if np.linalg.norm(v[i] - v[last]) <= max_speed_mps * dt + allowance_m + 1e-9:
                keep[i] = True
                last = i
            i += direction
    return t[keep], v[keep], int((~keep).sum())


def place_on_grid(t_native: np.ndarray, v_native: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Readings at their own times -> the grid: each grid sample is the mean of the readings within half a sample
    of it, NaN where there were none. Never interpolated (see module docstring). Works for a non-uniform grid."""
    v_native = np.asarray(v_native, float).reshape(len(t_native), -1)
    out = np.full((len(grid), v_native.shape[1]), np.nan)
    if not len(t_native) or not len(grid):
        return out
    if len(grid) == 1:
        half = np.array([np.inf])
    else:
        sp = np.diff(grid)
        half = 0.5 * np.r_[sp[0], np.minimum(sp[:-1], sp[1:]), sp[-1]]
    j = np.clip(np.searchsorted(grid, t_native), 1, len(grid) - 1) if len(grid) > 1 else np.zeros(len(t_native), int)
    if len(grid) > 1:
        j = np.where(np.abs(t_native - grid[j - 1]) <= np.abs(t_native - grid[j]), j - 1, j)
    ok = np.abs(t_native - grid[j]) <= half[j] + 1e-9
    j, vals = j[ok], v_native[ok]
    sums = np.zeros_like(out)
    counts = np.zeros(len(grid))
    np.add.at(sums, j, vals)
    np.add.at(counts, j, 1.0)
    has = counts > 0
    out[has] = sums[has] / counts[has, None]
    return out


def native_stats(t: np.ndarray, v: np.ndarray, unit_scale: float = 1.0) -> dict:
    """A sensor's own rate and white noise, from its own readings (fit_sensor_noise's robust second difference)."""
    from fit_sensor_noise import white_noise_std, noise_correlation, FILTERED_ABOVE
    v = np.asarray(v, float).reshape(len(t), -1)
    out = {"readings": int(len(t))}
    if len(t) < 2:
        return out
    dts = np.diff(t)
    med = float(np.median(dts))
    out["rate_hz"] = round(1.0 / med, 3) if med > 0 else None
    out["gaps_over_3_periods"] = int(np.sum(dts > 3 * med)) if med > 0 else 0
    if len(t) >= 20:
        noise = [white_noise_std(v[:, k]) * unit_scale for k in range(v.shape[1])]
        corr = [noise_correlation(v[:, k]) for k in range(v.shape[1])]
        out["noise_std"] = [round(float(x), 6) for x in noise]
        # smoothed readings (a filtered nav solution, an AHRS): the second difference is motion, not noise
        out["filtered"] = bool(np.nanmax(corr) > FILTERED_ABOVE) if np.all(np.isfinite(corr)) else None
    return out


# ------------------------------------------------------------------ derived rates and checks

def rates_from_orientation(t: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Body angular rates (rad/s) from consecutive orientations: the rotation between samples k-1 and k+1 in the body
    frame, over their time apart (a central difference; the ends use one-sided steps)."""
    n = len(t)
    if n < 2:
        return np.zeros((n, 3))
    rot = Rotation.from_matrix(R)
    out = np.zeros((n, 3))
    lo = np.r_[0, np.arange(n - 2), n - 2]
    hi = np.r_[1, np.arange(2, n), n - 1]
    rel = (rot[lo].inv() * rot[hi]).as_rotvec()
    dt = (t[hi] - t[lo])
    out[:] = rel / np.where(dt > 0, dt, np.nan)[:, None]
    return np.nan_to_num(out)


def gyro_vs_orientation(t: np.ndarray, R: np.ndarray, gyro: np.ndarray) -> dict:
    """Do the logged rates turn the vehicle the way its orientation turns? Per axis: the lag (s) at which the gyro best
    matches the orientation's own rates (by cross-correlation within +-2 s), and at that lag the least-squares slope
    and its R^2. A slope near 57 or 1/57 is deg/s against rad/s; near -1 is a sign or frame error; a lag is two
    clocks (or a slow attitude filter). Judged only where the two clearly follow each other (R^2 >= 0.5)."""
    ref = rates_from_orientation(t, R)
    out = {"slope": {}, "lag_s": {}, "r2": {}}
    problems, unclear, late = [], [], []
    rate = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    m = int(min(2.0 * rate, len(t) // 4))
    for k, ax in enumerate("pqr"):
        x, y = ref[:, k], gyro[:, k]
        if (np.abs(x) > MIN_ROTATION_RADPS).sum() < 20:
            out["slope"][ax] = None
            continue
        L = 0
        if m >= 1:
            xc, yc = x - x.mean(), y - y.mean()
            lags = np.arange(-m, m + 1)
            c = [abs(np.dot(xc[max(0, -j):len(xc) - max(0, j)], yc[max(0, j):len(yc) - max(0, -j)])) for j in lags]
            L = int(lags[int(np.argmax(c))])
        xs, ys = (x[:len(x) - L], y[L:]) if L >= 0 else (x[-L:], y[:len(y) + L])
        moving = np.abs(xs) > MIN_ROTATION_RADPS
        slope = float(np.dot(xs[moving], ys[moving]) / np.dot(xs[moving], xs[moving]))
        r2 = 1.0 - float(np.sum((ys[moving] - slope * xs[moving]) ** 2) / max(np.sum(ys[moving] ** 2), 1e-12))
        out["slope"][ax], out["r2"][ax], out["lag_s"][ax] = round(slope, 3), round(r2, 3), round(L / rate, 3)
        if r2 < GYRO_CLEAR_R2:
            # they don't move together well enough to judge scale or sign (a heavily filtered attitude, or test
            # data): said, not refused
            unclear.append(f"{ax} (R^2 {r2:.2f})")
            continue
        if abs(L / rate) > GYRO_LAG_WARN_S:
            late.append(f"{ax} {L / rate:+.2f} s")
        if slope < -0.5:
            problems.append(f"{ax}: the gyro turns the opposite way to the orientation (slope {slope:+.2f}): a sign or "
                            f"frame error (check frame: and the gyro's axes)")
        elif slope > GYRO_SLOPE_TOLERANCE or (0 < slope < 1.0 / GYRO_SLOPE_TOLERANCE):
            hint = (" (57: logged in deg/s but declared rad/s)" if 20 < slope < 120 else
                    " (1/57: logged in rad/s but declared deg/s)" if 1 / 120 < slope < 1 / 20 else "")
            problems.append(f"{ax}: the gyro reads {slope:.3g} times the orientation's own rate of turn{hint}; check "
                            f"angular_unit")
    if problems:
        out["problem"] = "; ".join(problems)
    warnings = []
    if unclear:
        warnings.append(f"the logged gyro and the orientation's own rate of turn don't follow each other closely on "
                        f"{', '.join(unclear)}, so the gyro's unit and sign weren't checked there: check they are one "
                        f"sensor's, on one clock")
    if late:
        warnings.append(f"the gyro and the orientation match best shifted in time ({', '.join(late)}; positive: the "
                        f"gyro is late): different clocks, or a slow attitude filter. Calibration lines them up as "
                        f"logged")
    if warnings:
        out["warning"] = "; ".join(warnings)
    return out


def accelerometer_gravity(R: np.ndarray, acc: np.ndarray, gyro: Optional[np.ndarray]) -> dict:
    """At rest an accelerometer reads gravity's reaction: (0, 0, -9.81) rotated into the body (FRD), whatever the
    vehicle's attitude. On quasi-static samples (little rotation, readings close to their neighbours), checks the
    magnitude (a unit error: g declared as m/s^2 reads 1) and the direction (a frame or mount error)."""
    ok = np.all(np.isfinite(acc), axis=1)
    if gyro is not None:
        ok &= np.all(np.abs(gyro) < STATIC_GYRO_RADPS, axis=1)
    if ok.sum() < 20:
        ok = np.all(np.isfinite(acc), axis=1)
    if ok.sum() < 20:
        return {"note": "too few accelerometer readings to check"}
    f = acc[ok]
    expect = np.einsum("nji,j->ni", R[ok], np.array([0.0, 0.0, -G]))
    mag = float(np.median(np.linalg.norm(f, axis=1)))
    cos = np.einsum("ni,ni->n", f, expect) / (np.linalg.norm(f, axis=1) * G + 1e-12)
    out = {"median_magnitude_mps2": round(mag, 3), "median_cos_to_gravity": round(float(np.median(cos)), 3)}
    if abs(mag - G) > ACCEL_G_TOLERANCE * G:
        hint = " (about 1: logged in g, declare unit: g)" if 0.5 < mag < 1.5 else \
               " (about 96: logged in m/s^2, declared g)" if 50 < mag < 150 else ""
        out["problem"] = (f"the accelerometer reads {mag:.2f} m/s^2 at rest, not 9.81{hint}; check its unit")
    elif float(np.median(cos)) < 0.8:
        out["problem"] = (f"the accelerometer's gravity points {np.degrees(np.arccos(np.clip(np.median(cos), -1, 1))):.0f}"
                          f" degrees away from where the orientation says down is: a frame or mount error (check "
                          f"frame: and mount_rpy_deg; a ROS IMU is FLU and reads +9.81 on z at rest)")
    return out


def depth_vs_position(depth: np.ndarray, pz: np.ndarray) -> dict:
    """Two vertical sensors should agree up to an offset (a depth sensor's zero against a positioning system's)."""
    ok = np.isfinite(depth) & np.isfinite(pz)
    if ok.sum() < 20 or np.std(pz[ok]) < 0.05:
        return {"note": "too little vertical motion seen by both to compare"}
    a, b = pz[ok] - pz[ok].mean(), depth[ok] - depth[ok].mean()
    slope = float(np.dot(a, b) / np.dot(a, a))
    out = {"slope": round(slope, 3), "rms_difference_m": round(float(np.sqrt(np.mean((b - a) ** 2))), 4)}
    if slope < -0.5:
        out["problem"] = ("depth moves opposite to the position fix's down axis (slope %+.2f): one of them has its sign "
                          "flipped (depth positive: down|up, or the position frame)" % slope)
    elif not (0.5 < slope < 2.0):
        out["problem"] = f"depth changes {slope:.2f} times as much as the position fix's down axis: check the units"
    return out


def fix_velocities(t: np.ndarray, pos: np.ndarray, max_gap_s: float = 3.0, min_dt_s: float = 0.5) -> tuple:
    """World velocity between position fixes (grid arrays, NaN = no fix), from each fix to the next one at least
    min_dt_s later and at most max_gap_s. Returns (grid indices at the midpoints, velocities)."""
    idx = np.flatnonzero(np.all(np.isfinite(pos), axis=1))
    mids, vels = [], []
    j = 0
    for a in range(len(idx)):
        ia = idx[a]
        while j < len(idx) and t[idx[j]] - t[ia] < min_dt_s:
            j += 1
        if j >= len(idx):
            break
        ib = idx[j]
        dt = t[ib] - t[ia]
        if dt > max_gap_s:
            continue
        mids.append(int(np.argmin(np.abs(t - 0.5 * (t[ia] + t[ib])))))
        vels.append((pos[ib] - pos[ia]) / dt)
    return np.array(mids, dtype=int), np.array(vels).reshape(-1, pos.shape[1])


def heading_vs_fixes(t: np.ndarray, pos: np.ndarray, R: np.ndarray, body_vec: np.ndarray, what: str) -> dict:
    """The heading check without a dense position track: the horizontal velocity between fixes against `body_vec`
    (the body velocity, or with no DVL the thrust, which points where the vehicle goes in steady motion) turned into
    the world by the orientation. Same test as data_ingestion.orientation_vs_position: a complex gain whose angle is
    the heading offset; a mirrored heading matches the conjugate instead."""
    mids, vw = fix_velocities(t, pos[:, :2])
    if len(mids) < 20:
        return {"note": "too few position fixes in motion to check the heading"}
    b3 = np.einsum("nij,nj->ni", R[mids], body_vec[mids])
    a, b = vw[:, 0] + 1j * vw[:, 1], b3[:, 0] + 1j * b3[:, 1]
    moving = (np.abs(a) > 0.05) & (np.abs(b) > 1e-6)
    if moving.sum() < 20:
        return {"note": "too little horizontal motion between fixes to check the heading"}
    a, b = a[moving], b[moving]

    def fit(x):
        c = np.vdot(x, a) / np.vdot(x, x)
        return float(np.angle(c, deg=True)), float(1 - np.linalg.norm(a - c * x) ** 2 / np.linalg.norm(a) ** 2)
    off, r2 = fit(b)
    moff, mr2 = fit(np.conj(b))
    out = {"compared_with": what, "heading_offset_deg": round(off, 1), "match_r2": round(r2, 3),
           "mirrored_match_r2": round(mr2, 3)}
    if mr2 > 0.3 and mr2 > r2 + 0.2:
        out["problem"] = (f"the orientation's heading is mirrored relative to the position fixes (the {what} matches "
                          f"the fixes' motion only with the heading's sign flipped, R^2 {mr2:.2f} against {r2:.2f}). "
                          f"Check the heading source's frame or frame:")
    elif r2 >= 0.3 and abs(off) > 30:
        out["problem"] = (f"the orientation's heading is about {off:.0f} degrees off the direction the position fixes "
                          f"show the vehicle moving ({what}, R^2 {r2:.2f}). Check the heading source (a compass offset "
                          f"or declination) or the positions' world frame")
    return out
