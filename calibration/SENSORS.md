# Your vehicle's sensors

Strathe calibrates from what your vehicle actually measures. A DVL is one sensor among several, not a
requirement (PROJECT_STATUS.md #96). This page covers which sensors Strathe reads, which motion each one
lets it calibrate, and how to log them.

`strathe calibrate` uses whatever your logs have:
- **A DVL reading at every sample:** the original velocity fit, unchanged.
- **Anything else:** the sensor suite. No DVL, a DVL with gaps, depth, position fixes, GPS.

`--sensors` chooses explicitly.

## What each sensor tells calibration

| Sensor | What it measures | Axes it lets Strathe calibrate |
|---|---|---|
| Gyro (any IMU) | body rates p, q, r | roll, pitch, yaw. Always present: with no gyro logged, the rates are derived from the orientation |
| Orientation (AHRS, IMU filter, compass) | roll, pitch, heading | required: it turns body motion into the world |
| DVL, or motion-capture velocity | body velocity u, v, w | surge, sway, heave |
| Depth sensor (pressure) | depth | heave |
| USBL, LBL, motion capture | world position, with a vertical | surge, sway, heave |
| GPS (at the surface) | world position, horizontal only | surge, sway (while surfaced) |
| Accelerometer | specific force (acceleration plus gravity's reaction) | read and checked; used only with `--use-accelerometer` (experimental: #96 measures what it adds) |

No software can calibrate an axis that no sensor sees. With only a depth sensor and an IMU, Strathe has
nothing that shows how fast the vehicle moves forward or sideways. Ingestion says so ("no sensor sees surge,
sway motion"), and calibration keeps your vehicle description's values for those axes rather than inventing
them.

## How calibration uses them

Each recording is replayed in short windows: 10 s by default, `--prediction-horizon-s` to change.

**Where each window starts:**
- the gyro's rates;
- a body velocity: the DVL's reading where it has one. Otherwise it comes from a *state smoother*: a Kalman filter
  and smoother running Strathe's own physics with every sensor's readings, before and after that moment.

**What's compared:** each sensor's actual readings with what it would have read on the simulated motion:
- **Gyro:** the simulated body rates.
- **DVL:** the simulated body velocity, where it read.
- **Position fix:** the simulated velocity turned by the logged orientation and summed into a path. Where the
  window started isn't known to the fix's accuracy, so that start is removed exactly: each window's residuals are
  taken about their own mean.
- **Depth:** the same as a position fix's vertical.
- **Accelerometer (opt-in):** the simulated acceleration plus gravity's reaction, its own bias removed per window.

**Scaling.** Each sensor axis is scaled by what predicting no motion gets wrong on it, and by its number of
readings. So a 1 Hz GPS at the surface counts as much as a 50 Hz gyro, and neither swamps the other.

**Not fitted against itself.** The smoother only chooses where windows start. The fit is always scored against the
real readings, never against the smoother's estimate.

**Refitting with better starts.** After the fit, the smoother runs again with the fitted values and the fit is
repeated (`--anchor-rounds`, 2 by default). Better physics gives better starting velocities.

**Where each sensor sits.** The vehicle description's optional `sensors:` section
(`templates/vehicle_spec_template.yaml`) gives each sensor's mount, by its ingest name, in body FRD metres from the
torques' reference point.
- A DVL there reads v + ω×r.
- A USBL transponder's or GPS antenna's fix is of its own point.
- A pressure port reads its own depth.
- `track: water` marks a DVL that measures velocity through the water rather than over the seabed: calibration adds
  the water current back.

Calibration models all of these. Leaving a mount out means the sensor is at the reference point.

**Axes no sensor sees.** An axis with no sensor (surge and sway with no DVL and no horizontal fixes) has its
parameters held at your vehicle description's values. Calibration says so in the log, `metrics.json`,
`calibrated_parameters.yaml` and the report ("Surge: not calibrated. Your logs have no DVL or horizontal position
fixes for it").

**The report.** It gains a section, "What your sensors could calibrate". It gives each sensor's own held-out skill
(1 = perfect, 0 = no better than assuming the vehicle didn't move), and judges each motion by the sensors that
measured it.

**Validation.** Position drift is measured against your vehicle's own position fixes (or depth): an independent
measurement, not its velocity.

**Everything else works the same with sensors:**
- `--restarts`, the automatic retry of a failed fit, `--finish`;
- `--estimate-current`, `--drag prior`, `--free-decay`, `--exclude-axes`;
- the one-session split and the web page.

## Logging them: the ingest spec

Start from a template:
- `templates/ingest_ardusub_no_dvl.yaml`: an ArduSub vehicle with no DVL (`.bin` logs);
- `templates/ingest_rosbag_sensors.yaml`: a ROS vehicle's IMU, pressure, GPS, USBL and DVL.

```yaml
body_velocity:            # leave out u, v, w with no DVL; leave out p, q, r if no gyro is logged
  columns: {p: ..., q: ..., r: ...}
  # with a DVL: u, v, w too, and linear_gaps: keep so its dropouts stay dropouts
depth:                    # a depth reading, or a pressure (Pa, hPa, kPa, mbar, bar, psi)
  pressure: {column: ..., unit: mbar, water: salt, surface: auto}
position:                 # one fix, or a list of them
  - {use: calibrate, source: usbl, x: ..., y: ..., z: ..., delay_s: 0.3}
  - {use: calibrate, source: gps, latlon: {lat: ..., lon: ...},
     drop_when: [{column: <fix status>, below: 0}]}
accelerometer: {x: ..., y: ..., z: ..., unit: m/s^2}
```

- **Placed where they happened, never interpolated.** A 1 Hz USBL fix on a 10 Hz recording fills one sample
  in ten; the other nine are empty. Interpolating would invent smooth readings and hide the noise
  calibration has to weigh. A sensor faster than the recording is averaged into each sample.
- **Doesn't shorten the recording.** GPS that only has a fix at the surface leaves the underwater part
  empty for GPS, but the recording keeps its full length.
- **`drop_when:`** drops readings by another column, e.g. GPS whose fix status is below 0, or below 0.5 m of
  depth.
- **`delay_s:`** is a sensor's latency (a USBL fix arrives after the vehicle was there). It's taken off the
  readings' times.
- **Canonical columns** in the ingested CSVs:
  - `pos_<name>_n`, `_e`, `_d`: world NED, m;
  - `depth_<name>`: m, positive down;
  - `acc_x`, `_y`, `_z`: body FRD, m/s²;
  - `Bu`, `Bv`, `Bw`: NaN where the DVL had no reading.
- **What not to use as a sensor:** a filtered navigation solution (robot_localization's `/odometry/filtered`,
  ArduPilot's EKF velocity, MOOS `NAV_*`). It usually fuses the DVL and a model, so it isn't an independent
  measurement. `source: odometry` with `use: calibrate` is refused.

## What ingestion checks

Every check prints its numbers into `ingest_report.json`. A check that finds a problem stops ingestion with
what to fix; `skip_checks:` turns it into a warning for logs you know are right.

| Check | Catches |
|---|---|
| `gyro_vs_orientation` | deg/s declared as rad/s (slope 57), a flipped axis (slope −1), the gyro and the orientation on different clocks (their lag is named). Only judged where the two clearly follow each other |
| `accelerometer` | g declared as m/s² (reads 1 at rest), a frame or mount error (gravity not along the orientation's down; a ROS IMU reads +9.81 on z at rest, which is right) |
| `depth_vs_position` | a depth sign flipped against a position fix's vertical, or the wrong unit |
| depth rate | depth changing faster than 5 m/s: a wrong unit, or not a depth |
| position outliers | fixes the vehicle couldn't reach at `max_speed_mps` from the last good one are dropped. Over 20% dropped is refused (two systems mixed, or the wrong unit). The tolerance comes from the kind of fix (`noise_m:` to set it) |
| `heading_vs_position` | with no DVL: a mirrored or offset heading, from the fixes against the thrust |
| force direction | with no DVL: thrust pushing against the motion between fixes and depth readings |

Real data so far:
- **DFKI's and URI's gyros:** match their orientation with slope 1.00–1.04 and R² ≥ 0.99.
- **URI's raw IMU:** reads 9.80–9.82 m/s² along gravity at rest.
- **URI's RTK GPS track:** matches the thrust direction within 13–22°.
- **DFKI's heading:** mirrored on purpose in a test, it is now refused by the gyro check. Before #96 it was
  only warned about.

## Noise

`ingest_report.json` gives each sensor's own rate and its white noise, measured on its own readings with
`fit-noise`'s robust second difference. A sensor smoothed by the logging system (a filtered depth, an RTK
solution) says so, since its noise can't be measured from the log. `strathe fit-noise` measures the position
and depth channels too. For a sensor the vehicle doesn't have (no DVL), it writes the `typical` preset's value
and says it wasn't measured: the simulators model a DVL and a depth sensor on every vehicle.
