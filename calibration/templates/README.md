# Templates: getting your own vehicle's logs into Strathe

Copy the template closest to your logs next to them, change the right-hand
sides (your column or topic names, units, frame), and run one command:

```bash
python strathe.py calibrate --ingest-spec my_vehicle_ingest.yaml --name my_vehicle
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

## What every spec needs

- **Body-frame velocity** (surge, sway, heave, roll/pitch/yaw rates): from a DVL
  or odometry, in the vehicle's frame, not the world frame.
- **Orientation**: a quaternion or roll/pitch/yaw.
- **What pushed the vehicle**: logged forces and torques, or the PWM sent to each
  thruster's ESC plus a description of the thrusters.
- **Time**: a timestamp column (any unit), or the sampling rate. Bags carry their
  own. The rate is recorded, and calibration uses it automatically.
- **A date per recording**: calibration holds out whole days to check itself on
  sessions it never saw. Dates come from the filename, a constant, or the
  recording's own clock (`from: recording`).
- **Frame**: `frd` (forward-right-down, NED world; the physics model's) or
  `flu` (ROS: forward-left-up, ENU world). Everything is converted to FRD/NED.

## Thrusters

- `preset: bluerov2_heavy`: the validated BlueROV2 Heavy. It reproduces the
  reference dataset's forces from PWM on all 183,555 rows.
- `layout:`: any vehicle. Give one entry per thruster, in PWM-column order:
  - `position_m`: x, y from the vehicle's centre; z from the plane of the
    horizontal thrusters.
  - `direction`: the way it pushes the vehicle for positive thrust.
  - `reversed`: ArduSub `MOT_n_DIRECTION = -1`.

  The custom-thrusters template is a BlueROV2 Heavy written this way, as an
  example. A test checks it matches the preset exactly.
- `allocation_csv:` + `directions:`: if you already have the 6 x n matrix
  (`allocation_template.csv`).
- `curve: t200` (default), optionally with `voltage:` (12–20 V). Or give your own
  `curve_csv:` (`thrust_curve_template.csv`: PWM in µs → thrust in N).

Vehicles whose thrusters can't push every axis are fine; ingestion names the
axes. For example, a 6-thruster BlueROV2 has no pitch control. The same
`thrusters:` block (or the whole ingest spec file) can limit the simulated
vehicle's commands in `strathe arena`, `strathe fleet` and `strathe train-fleet`
(`--thrusters my_vehicle_ingest.yaml`).

## What it refuses, and what it warns about

It refuses data that would silently corrupt a calibration:
- deg/s declared as rad/s
- non-unit quaternions (usually wrong columns)
- non-monotonic or jittery time (unless you set `resample_hz`)
- files at different rates
- PWM that isn't in ESC microseconds
- topics or fields missing from a bag

It warns about gaps in time and orientation jumps over 45° between samples
(sensor glitches). Everything is recorded in `ingest_report.json`.

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
