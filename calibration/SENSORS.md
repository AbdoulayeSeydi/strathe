# Your vehicle's sensors

Strathe calibrates from what your vehicle actually measures. A DVL is one sensor among several, not a
requirement (PROJECT_STATUS.md #96). This page covers which sensors Strathe reads, which motion each one
lets it calibrate, and how to log them.

> **Status:** ingestion reads every sensor below, checks it, and records which axes your sensors see.
> Calibrating *from* depth, position fixes and the accelerometer, instead of body velocity, is the next step
> of #96. Until it lands, `strathe calibrate` on logs with no DVL stops and says so.

## What each sensor tells calibration

| Sensor | What it measures | Axes it lets Strathe calibrate |
|---|---|---|
| Gyro (any IMU) | body rates p, q, r | roll, pitch, yaw. Always present: with no gyro logged, the rates are derived from the orientation |
| Orientation (AHRS, IMU filter, compass) | roll, pitch, heading | required: it turns body motion into the world |
| DVL, or motion-capture velocity | body velocity u, v, w | surge, sway, heave |
| Depth sensor (pressure) | depth | heave |
| USBL, LBL, motion capture | world position, with a vertical | surge, sway, heave |
| GPS (at the surface) | world position, horizontal only | surge, sway (while surfaced) |
| Accelerometer | specific force (acceleration plus gravity's reaction) | read and checked; how much it adds is being measured (#96) |

No software can calibrate an axis that no sensor sees. With only a depth sensor and an IMU, Strathe has
nothing that shows how fast the vehicle moves forward or sideways. Ingestion says so ("no sensor sees surge,
sway motion"), and calibration keeps your vehicle description's values for those axes rather than inventing
them.

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
