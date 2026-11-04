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

**Your own simulator.** `fit-sim` and `export-sim` work from the sensors too: Stonefish, Gazebo, MuJoCo,
HoloOcean and external simulators.
- Your simulator runs free over each window, as with a DVL.
- It starts from the estimated velocity.
- It's scored against what each sensor read: its path, under its own attitude, against the fixes and depth; its rates
  against the gyro; and its velocity against the DVL where the DVL read.
- Drift is measured against the position fixes.

Starting velocities come from a physics-free smoother, since your simulator may model any vehicle. Give a Strathe
calibration and they come from its physics instead, which was about 4x closer to the truth in our tests (2 against
8 cm/s from 1 Hz fixes):
- `--from-strathe CALIBRATION` also puts the calibration's values into your model, and `export-sim` always has one.
  A simulator that can't take Strathe's model (MuJoCo) refuses it.
- `--starts-from CALIBRATION` gives only the starts, for any simulator.

With either, or with `export-sim`, a delay the calibration took off a late position sensor's fixes is taken off
here too, so the simulator is fitted to the same fixes. fit-sim doesn't look for a delay itself: without a
calibration, give it as `delay_s:` in the ingest spec.

On real data, with the hidden truth (#96, FS-real-*), held-out mean skill over 10 s free runs:

| | fit-sim | `--starts-from` | Strathe's calibration, same sensors | the model before fitting |
|---|---|---|---|---|
| DFKI BlueROV2: depth, gyro, attitude | 0.067 | 0.061 | −0.107 | −1.53 |
| URI ALPHA RISE: GPS at the surface, depth, IMU | 0.033 | 0.032 | 0.101 | −0.36 |
| URI ALPHA RISE with its DVL (a control) | 0.123 | | 0.173 | −0.36 |

The physics-aided starts made no measurable difference on either. On URI with no DVL, fit-sim fitted surge to 0.15,
where the model already had 0.82. With the DVL it kept 0.81. The cause isn't found (LIMITS.md).

**Everything else works the same with sensors:**
- `--restarts`, the automatic retry of a failed fit, `--finish`;
- `--estimate-current`, `--drag prior`, `--free-decay`, `--exclude-axes`;
- the one-session split and the web page.

## Logging them: the ingest spec

Start from a template:
- `templates/ingest_ardusub_no_dvl.yaml`: an ArduSub vehicle with no DVL (`.bin` logs);
- `templates/ingest_rosbag_sensors.yaml`: a ROS vehicle's IMU, pressure, GPS, USBL and DVL;
- `templates/ingest_csv_usbl.yaml`: CSV logs with a USBL, LBL or motion-capture fix and a depth sensor.

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
- **`delay_s:`** is a sensor's latency (a USBL fix arrives after the vehicle was there; depth read through a slow
  link or filter). Given in the ingest spec's `position:` or `depth:` block, it's taken off the readings' times at
  ingest. Given as `delay_s:` on the sensor in the
  vehicle description's `sensors:`, calibration takes it off. Either way it's taken as said, 0 included (0: the
  fixes are on time). Give it in one place: both is refused.
- **Left out, calibration looks for a delay** in each position and depth sensor's readings. It shifts them
  earlier, 0 to 2 s, over the minute with the most readings, and asks the state smoother (the description's physics
  driven by the logged thrust) which shift explains them best. A shift of 0.5 s or more that explains the readings
  clearly better than no shift is **taken off before the fit**, and the report warns that it was: for a position
  sensor, at least 0.5 better in log-likelihood per fix; for depth, logged at 1 to 10 Hz, at least 0.3 per second
  of readings. A weaker shift of 0.5 s or more is only said. Reloading the calibration (validation,
  identifiability, `estimate-current`, fit-sim and export-sim given it) takes off the same delay. A gyro that lags
  the orientation is said at ingest, not shifted, since either one could be the late one. Measured through
  calibration (PROJECT_STATUS.md #96), position fixes:
  - **Fixes with no delay** (the tank's and DFKI's, made from motion capture, and DFKI's motion capture itself):
    0.0 s, no gain, nothing taken off.
  - **URI's real GPS:** at least 2 s, but 0.125 per fix over 26 fixes: said, not taken off. URI's DVL couldn't
    confirm a delay either way: a vehicle at the surface barely accelerates.
  - **Fixes late, on a known truth** (B-applied-*): 1 s late was found at 0.8 s and taken off, and held-out skill
    went from 0.876 to 0.982 (0.993 with the fixes on time). On other recordings, 1 s late: 0.915 to 0.975; 2 s
    late, found "at least 2.0 s": 0.875 to 0.995. Fixes on time: nothing taken off, skill unchanged. 0.3 s late is
    too little to tell (0.2 s, 0.035 per fix): nothing taken off, and it still costs 0.035 of skill.
  - **A delay given** (B-given-*): `delay_s: 1` on the same 1 s late fixes reached 0.9925. `delay_s: 0` was taken
    as said (0.876), and the report says the fixes still look 0.8 s late.
  - **The thresholds' history:** the first, 2 per fix, missed the 1 s case (1.21 per fix; B-latency-1 failed).
    0.5 was then chosen from these numbers and tested on recordings and noise it wasn't chosen on
    (B-latency-fresh-*): 0 s quiet (0.0 per fix), 1 s warned (0.8 s, 2.39 per fix), 2 s warned ("at least
    2.0 s", 7.23 per fix). Those rows ran when a delay was only said.
  Depth (B-depthlag-*; the bar, 0.3 per second, was chosen from the scan alone on the reference recordings and on
  real depth, then tested on fresh recordings):
  - **Real depth on time:** the tank's and DFKI's Bar30 0.0 s; URI's 1 Hz depth 0.5 s, but at 0.003 per second:
    nothing taken off.
  - **1 s late, nothing given:** found 1.0 s (2.26 per second) and taken off: held-out skill 0.9925, the same as
    on time (0.9926). Given `delay_s: 0`, it's taken as said: 0.948, and the report says the readings look late.
    Given `delay_s: 1`: 0.9925.
  - **Fresh recordings, depth the only vertical sensor:** 0, 0.5, 1 and 2 s late found at 0.0, 0.5, 1.0 and 2.0 s
    (0.53 to 0.90 per second when late) and taken off; on time, nothing taken off.
  - **Depth and the gyro only:** 1 s late found at 1.0 s and taken off; on time, nothing.
  - **1 Hz depth** (as URI logs it): 1 s late found at 0.8 s (0.67 per second) and taken off; on time, nothing.
  - **0.3 s late** read 0.2 to 0.3 s at 0.30 to 0.43 per second, under the 0.5 s it takes: not taken off.
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
| `depth_vs_thrust` | with depth as the only vertical sensor: a flipped sign, from depth falling when the thrusters push down (correlation −0.4 or lower refuses, −0.2 to −0.4 warns). It needs 60 s and 0.5 m of depth change to judge |
| depth rate | depth changing faster than 5 m/s: a wrong unit, or not a depth |
| position outliers | fixes the vehicle couldn't reach at `max_speed_mps` from the last good one are dropped. Over 20% dropped is refused (two systems mixed, or the wrong unit). The tolerance comes from the kind of fix (`noise_m:` to set it) |
| `heading_vs_position` | with no DVL: a mirrored or offset heading, from the fixes against the thrust |
| force direction | with no DVL: thrust pushing against the motion between fixes and depth readings |

Real data so far:
- **DFKI's and URI's gyros:** match their orientation with slope 1.00–1.04 and R² ≥ 0.99.
- **Depth against the thrust:** on the tank's 9 recordings, URI's 2 and DFKI's 5 long enough to judge, none was
  refused. With the sign flipped, all 9 tank recordings and all 5 DFKI ones were refused; on URI one was refused, and
  the other (correlation +0.26 as logged) got a warning.
- **URI's raw IMU:** reads 9.80–9.82 m/s² along gravity at rest.
- **URI's RTK GPS track:** matches the thrust direction within 13–22°.
- **DFKI's heading:** mirrored on purpose in a test, it is now refused by the gyro check. Before #96 it was
  only warned about.
- **KTH's Marinarium BlueROV2 (PX4):** its motion capture is in PX4's frames, gravity matching its accelerometer to
  0.6 m/s². Its velocity matches its positions to 1 cm/s by the messages' own stamps, not by the bag's receive times.
  Its gyro was too quiet on roll and pitch on one day to check against the attitude, and the report says so
  (`examples/marinarium/`).

## Noise

`ingest_report.json` gives each sensor's own rate and its white noise, measured on its own readings with
`fit-noise`'s robust second difference. A sensor smoothed by the logging system (a filtered depth, an RTK
solution) says so, since its noise can't be measured from the log. `strathe fit-noise` measures the position
and depth channels too. For a sensor the vehicle doesn't have (no DVL), it writes the `typical` preset's value
and says it wasn't measured: the simulators model a DVL and a depth sensor on every vehicle.

### The noise the calibration assumes

Where a window has no DVL reading, its starting velocity comes from a state smoother that weighs each sensor by
its reading noise. Strathe takes that noise, in order, from:

1. `noise:` in your vehicle description's `sensors:` section: one reading's standard deviation (m, or m/s for a
   DVL);
2. what ingestion measured on your logs, when the readings weren't smoothed;
3. `grade:`, a datasheet preset from `calibration/sensor_grades.yaml`: `nucleus1000` (DVL, 0.5 cm/s), `bar30`
   (2 mm), `bar02` (0.16 mm), `rtk_f9p` (RTK GNSS, 0.01 m CEP), `gnss_f9p` (standalone, 1.5 m CEP), `ugps_g2`
   (Water Linked Underwater GPS G2, 0.2% of range: give `range_m:` too);
4. otherwise, a default for the sensor's kind.

```yaml
sensors:
  bar30: {grade: bar30, position_m: [-0.2, 0.0, 0.1]}
  usbl:  {grade: ugps_g2, range_m: 40}
  dvl:   {noise: 0.008}              # your own number wins
```

The report lists the noise used for each sensor and where it came from. A datasheet number is a best case,
which is why your logs' own measurement comes first. On a test vehicle with 5 cm fixes
(`tests/test_measurements.py`), assuming 8.5 mm made the physics-aided starts 6× worse (5.2 against 0.84 cm/s),
and assuming 1.27 m made the physics-free ones 2.6× worse (21 against 8 cm/s). Grades exist only for sensors whose
datasheet states a per-reading noise. The Water Linked A50 publishes only its long-term accuracy (±1.01%), which
is a scale error, not noise, so give its `noise:` yourself.

**Two sensors of the same kind, of different quality** (a USBL and a GPS, two pressure sensors) are each their own
channel, weighed by their own noise, so say each one's. With a 0.17 m USBL and a 1.27 m GPS both left at the
0.2 m default, the GPS was trusted 6× too much, and adding it lowered held-out skill from 0.993 to 0.960
(B-redundant-position, known truth). Give the GPS `grade: gnss_f9p` or its `noise:`.
