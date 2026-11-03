# Templates: getting your own vehicle's logs into Strathe

Copy the template closest to your logs next to them, change the right-hand
sides (your column or topic names, units, frame), describe your vehicle (below),
and run one command:

```bash
python strathe.py calibrate --ingest-spec my_vehicle_ingest.yaml --spec my_vehicle_spec.yaml --name my_vehicle
```

That converts the logs, checks them, calibrates, checks which values the data
can pin down, and writes a plain-language report. To only convert:
`python strathe.py ingest --spec my_vehicle_ingest.yaml --out <folder>`.

| Your logs have | Start from |
|---|---|
| CSV with body-frame forces and torques | `ingest_csv_forces.yaml` |
| CSV with thruster PWM, BlueROV2 Heavy | `ingest_csv_pwm_bluerov2_heavy.yaml` |
| CSV with thruster PWM, any other vehicle | `ingest_csv_pwm_custom_thrusters.yaml` |
| ROS 1 `.bag` or ROS 2 bag folders (`.mcap`, `.db3`), e.g. ArduSub + MAVROS | `ingest_rosbag_mavros.yaml` |
| ArduSub / ArduPilot DataFlash logs from the SD card (`.bin`, `.log`) | `ingest_ardupilot_bin.yaml` |
| MAVLink telemetry logs from a ground station (`.tlog`, e.g. QGroundControl) | `ingest_mavlink_tlog.yaml` |
| PX4 logs (`.ulg`) | `ingest_px4_ulog.yaml` (`../LOG_FORMATS.md`) |
| MOOS-IvP logs (`.alog`, written by pLogger) | `ingest_moos_alog.yaml` (`../LOG_FORMATS.md`) |
| LCM event logs (lcm-logger), with your `.lcm` message definitions | `ingest_lcm.yaml` (`../LOG_FORMATS.md`; `pip install lcm`) |
| Free-decay tank tests on an ArduSub vehicle, for roll and pitch (`strathe free-decay`; `../FREE_DECAY_PROTOCOL.md`) | `ingest_free_decay_ardupilot.yaml` |
| CSV logs (any logger: DUNE, vendor software, MATLAB) with **normalized** thruster commands (e.g. −1..1) plus your command-to-thrust table | `ingest_csv_normalized_thrusters.yaml` with `thrust_curve_normalized_template.csv` |
| ROS 1 / ROS 2 bags with an EKF (`nav_msgs/Odometry`) or DVL and **normalized** thruster commands (array topic, one topic per thruster, or a custom message) | `ingest_rosbag_normalized_thrusters.yaml` with `thrust_curve_normalized_template.csv` |
| **No DVL**: an ArduSub vehicle's IMU, depth sensor and (if it has one) a USBL/UGPS fix, from `.bin` logs | `ingest_ardusub_no_dvl.yaml` (`../SENSORS.md`) |
| A ROS vehicle's own sensors: IMU, pressure, GPS at the surface, a USBL, a DVL with its gaps kept | `ingest_rosbag_sensors.yaml` (`../SENSORS.md`) |
| **No DVL**, CSV logs: thrust, gyro, attitude, depth and a USBL / LBL / motion-capture fix in a local frame, empty cells between fixes | `ingest_csv_usbl.yaml` (`../SENSORS.md`) |
| Your modems' message logs, to calibrate the acoustic/optical link (`strathe fit-link`) | `link_log_template.csv` |

Any other format (DUNE/IMC, vendor binaries): export to CSV and start from a CSV template. See `LIMITS.md`.

CSV ingestion reads **one table per recording**. If your logs are one CSV per sensor at different
rates (a common export), merge them onto one time base first; `../examples/dfki_tether/` does exactly
that for a public dataset (motion capture + ArduSub telemetry) and is a worked example to copy.

## Your vehicle's description

`vehicle_spec_template.yaml` describes the vehicle itself: its mass, and a first
guess and an allowed range for its inertia, drag, added mass and buoyancy, with
the formulas to work them out from its mass and size. **Easier:** `python strathe.py describe-vehicle --mass ...
--length ... --width ... --height ...` writes it for you (`../DESCRIBE_VEHICLE.md`). Or copy the template, fill it
in, and pass it with `--spec`. Without it Strathe uses a BlueROV2 Heavy (11.2 kg), which
is right only for that vehicle (`../../LAB_GUIDE.md` step 4).

## What every spec needs

- **Velocity** (surge, sway, heave, roll/pitch/yaw rates): from a DVL or
  odometry. Body frame by default; `linear_frame: world` takes world-frame
  (NED/ENU) velocity, as ArduPilot logs it, and rotates it into the body frame.
  A sensor mounted rotated in the vehicle (a DVL fitted upside down, a gyro turned 90°): give its roll, pitch and
  yaw in degrees as `linear_mount_rpy_deg` (DVL) or `angular_mount_rpy_deg` (gyro), in the spec's frame, and the
  readings are turned into the vehicle's axes. If ingestion says a force pushes against the motion on one axis, a
  rotated mount is a common cause (URI's DVL is rolled 180°: heave −0.48 without it, +0.48 with it).
- **Orientation**: a quaternion or roll/pitch/yaw.
- **What pushed the vehicle**: logged forces and torques, or the command sent to each
  thruster (ESC PWM in µs, or normalized) plus a description of the thrusters.
- **Time**: a timestamp column (any unit), or the sampling rate. Bags carry their
  own. The rate is recorded, and calibration uses it automatically.
- **A date per recording**: calibration holds out whole days to check itself on
  sessions it never saw (with one day, recordings within it: `../ONE_SESSION.md`). Dates come from the filename,
  a constant, or the recording's own clock (`from: recording`).
- **Frame**: `frd` (forward-right-down, NED world; the physics model's) or
  `flu` (ROS: forward-left-up, ENU world). Everything is converted to FRD/NED.

## Thrusters

- `preset: bluerov2_heavy`: the validated BlueROV2 Heavy, in ArduSub's motor
  numbering and FRD. It reproduces the reference dataset's forces from PWM on all
  183,555 rows (that dataset is in ROS FLU, so they match after flipping y and z;
  PROJECT_STATUS.md #62).
- `layout:`: any vehicle. Give one entry per thruster, in PWM-column order:
  - `position_m`: x, y from the vehicle's centre; z from the plane of the
    horizontal thrusters.
  - `direction`: the way it pushes the vehicle for positive thrust.
  - `reversed`: ArduSub `MOT_n_DIRECTION = -1`.
  - **When thrusters differ** (optional, per entry):
    - `curve_csv`: that thruster's own command-to-thrust table;
    - `thrust_scale`: a multiplier on the shared curve, e.g. 0.5 for a smaller thruster or a 50% software limit;
    - `max_forward_n` / `max_reverse_n`: a cap in newtons.

    Thrust, the force limits and what the simulators can deliver then use each thruster's own. A thruster with its
    own `curve_csv` can't be rescaled by battery voltage (`voltage_column`); Strathe says so.

  The custom-thrusters template is a BlueROV2 Heavy written this way, as an
  example. A test checks it matches the preset exactly. For ArduSub, "positive
  thrust" is a positive motor output, in the direction ArduSub's own motor factors
  give that motor (forward; lateral = right; throttle -1 = pushes down).
  `../examples/dfki_tether/ingest_dfki_tether.yaml` is a standard 6-thruster
  BlueROV2 (ArduSub's vectored frame), checked against motion capture.
- `allocation_csv:` + `directions:`: if you already have the 6 x n matrix
  (`allocation_template.csv`).
- `curve: t200` (default), optionally with `voltage:` (12–20 V). Or give your own
  `curve_csv:` (`thrust_curve_template.csv`: PWM in µs → thrust in N).
- **Normalized commands** (logged as e.g. −1..1 or 0..1 rather than µs): add
  `command: normalized` and `command_range: [low, high]`; stopped is the middle of
  the range. Give your command → thrust table in the same units as `curve_csv:`
  (`thrust_curve_normalized_template.csv`, header `command,thrust_n`); without it the
  T200 curve is used with the range mapped onto 1100–1900 µs. A reversed thruster's
  command is mirrored about stopped. Works with every source format (CSV, ROS bags);
  ArduPilot logs are already in µs.

Vehicles whose thrusters can't push every axis are fine; ingestion names the
axes. For example, a 6-thruster BlueROV2 has no pitch control. The same
`thrusters:` block (or the whole ingest spec file) can limit the simulated
vehicle's commands in `strathe arena`, `strathe fleet` and `strathe train-fleet`
(`--thrusters my_vehicle_ingest.yaml`).

## Optional extras

- `voltage_column:` (under `wrench: thrusters:`): the battery voltage logged with each sample. Each sample's
  thrust is then scaled to its own voltage instead of one `voltage:` for the whole spec. A battery that sags
  from 15.3 to 13.1 V loses 17% of its maximum thrust (Blue Robotics' published T200 data), so this matters
  for long sessions. T200 thrusters: the built-in curve, or your own `curve_csv` table with `curve_voltage:`
  (the battery voltage it was measured at): it keeps its shape and is rescaled from that voltage by the
  published T200 ratio, forward and reverse separately (#69).
- `position: {x: ..., y: ..., z: ..., unit: m}`: world positions (motion capture, USBL, odometry), in the
  spec's world frame. By default (`use: check`) only to check the orientation (below).
- **Your vehicle's own sensors (#96, `../SENSORS.md`):** `depth:` (a depth reading, or a pressure in Pa, hPa,
  mbar, bar or psi), `position:` with `use: calibrate` (a USBL, LBL or motion-capture fix in metres, or GPS
  `latlon:`; several as a list), `accelerometer:`, and a DVL whose dropouts are kept
  (`body_velocity.linear_gaps: keep`). `body_velocity` can leave out u, v, w (no DVL), or p, q, r (no gyro
  logged: the rates come from the orientation, with a warning). Each sensor's readings are placed at the
  samples where they happened and never interpolated, its noise is measured at its own rate, and
  `drop_when:` drops readings by another column (GPS has no fix underwater), `delay_s:` takes a sensor's
  latency off its times (a position or depth sensor's; left out, calibration takes off a delay the readings
  clearly show; SENSORS.md).

## What it refuses, and what it warns about

It refuses data that would silently corrupt a calibration:
- deg/s declared as rad/s
- non-unit quaternions (usually wrong columns)
- non-monotonic or jittery time (unless you set `resample_hz`)
- files at different rates
- PWM that isn't in ESC microseconds
- topics or fields missing from a bag

It also checks the data makes physical sense, and refuses it when it doesn't:
- **Force direction:** on surge, sway, heave and yaw, force and velocity should move together (thrust
  mostly balances drag). A clearly negative correlation (below -0.2) on any of them means the forces are
  reversed: a mirrored thruster layout, wrong `reversed:` flags or the wrong `frame:`, and is refused. One
  near zero (-0.2 to +0.2) is warned about: with world-frame velocity, usually a wrong heading. Every axis's
  correlation is printed. Roll and pitch are printed, not judged: restoring moments dominate them.
- **Orientation against position** (with `position:` and body-frame velocity): the body velocity, turned by
  the orientation, must follow the position track. A mirrored heading or a steady compass offset is named,
  with its size. With world-frame velocity this can't test the heading (the velocity was turned with that same
  heading). There, a wrong heading shows as force and velocity barely moving together on surge or sway, which
  ingestion warns about (it doesn't refuse: a mirrored heading took DFKI's surge to about 0, not below -0.2).
- **The sensors' own checks (#96):** the gyro must turn the way the orientation turns (deg/s against rad/s,
  a flipped axis, and a clock offset between them are named); an accelerometer must read 9.81 m/s^2 along
  the orientation's down at rest (g against m/s^2, a frame or mount error); a depth sensor and a position
  fix's vertical must move together (a flipped sign, the wrong unit); depth can't change faster than
  5 m/s; position fixes the vehicle couldn't have reached are dropped as outliers (refused if over 20%);
  with no DVL, the heading is checked against the fixes with the thrust, and force direction with the
  velocity between fixes and depth readings.
- `skip_checks: [force_direction]`, `[orientation_vs_position]`, `[gyro_vs_orientation]`, `[accelerometer]`,
  `[depth_vs_position]` or `[heading_vs_position]` turns a refusal into a warning, for logs you know are
  right.

It warns about gaps in time (with their total length and share of the recording) and orientation jumps
over 45° between samples (sensor glitches). Everything is recorded in `ingest_report.json`. It only sees
the timeline you give it: if you merge per-sensor files yourself, check each stream's gaps when you merge.

## ArduPilot / MAVLink notes

- Columns are `<MESSAGE>.<field>`, e.g. `RCOU.C1`, `ATT.Roll`, `SERVO_OUTPUT_RAW.servo1_raw`.
- `[i]` picks one instance of a message logged for several sensors: `IMU[0].GyrX`, `XKF1[0].VN`.
  If you leave it out when there are several instances, ingestion says which ones exist.
- List a log's messages with `mavlogdump.py --types my.bin` (installed with pymavlink), or in Mission Planner.
- **Velocity needs a DVL or positioning system.** The EKF's velocity without one is dead reckoning.
- **Dates:** `.bin` logs have a real date only with GPS; underwater they usually have time since boot, so date recordings from the filename or a constant. `.tlog` files carry the ground station's clock.
- **Telemetry rates:** raise them to at least 10 Hz before recording `.tlog` files.
- **Damage:** corrupt or cut-off bytes (a log cut at power-down) are skipped and counted. The rest of the log is used.

## ROS bag notes

- Columns are `<topic>.<field path>`, with `[i]` for array elements, e.g.
  `/mavros/rc/out.channels[0]`.
- Topics at different rates are put on one grid at `resample_hz`:
  - velocities: linear interpolation
  - orientation: spherical interpolation
  - PWM: held at the last command
- Timestamps come from each message's `header.stamp` (`stamp: receive` uses the
  time the bag recorded it instead).
- ROS 1 bags and ROS 2 `.mcap` bags include their message definitions. ROS 2
  `.db3` bags from before Iron don't: list the `.msg` files of any non-standard
  message (e.g. `mavros_msgs/msg/RCOut.msg`) under `source.message_types`.
