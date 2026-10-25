# Your logs, as they are: PX4, MOOS-IvP and LCM

Strathe reads these straight from the files your vehicle writes, with nothing to convert first:
- CSV;
- ROS 1 and ROS 2 bags;
- ArduPilot `.bin` and MAVLink `.tlog`;
- **PX4 `.ulg`, MOOS-IvP `.alog` and LCM event logs.**

Copy the template for your format, write your topic or variable names on the right-hand side, and run:

```bash
python strathe.py calibrate --ingest-spec my_vehicle.yaml --name my_auv     # converts, checks, calibrates, reports
python strathe.py ingest --spec my_vehicle.yaml --out converted/            # or only convert
```

| You log with | Template | Needs |
|---|---|---|
| PX4 (`.ulg`) | `templates/ingest_px4_ulog.yaml` | `pyulog`: in `requirements.txt` |
| MOOS-IvP (`.alog`, pLogger) | `templates/ingest_moos_alog.yaml` | nothing: it's text |
| LCM (lcm-logger's logs) | `templates/ingest_lcm.yaml` | `pip install lcm` (it ships lcm-gen). It has no wheel for Intel Macs: there, export to CSV or use another machine |

`python strathe.py check` says whether `pyulog` and `lcm` are installed. On the web page (`python strathe.py web`),
"Convert logs" takes your spec, and "Calibrate a vehicle" takes it as the ingestion spec.

## Naming what to read

Each format has its own way of naming a column. Every other part of the spec is the same as for CSV and bags:
- units;
- `frame: frd | flu`;
- world or body velocity;
- a rotated sensor mount;
- forces, or thruster PWM or normalized commands with your thruster layout.

| | Write a column as | Examples |
|---|---|---|
| **PX4** | `topic.field`; `topic[i].field` for one instance of a topic logged for several devices; arrays as `field[k]` | `vehicle_attitude.q[0]`, `vehicle_local_position.vx`, `actuator_outputs[0].output[2]` |
| **MOOS-IvP** | `VARIABLE`; `VARIABLE.key` for one key of a `key=value,...` posting; `VARIABLE@app` for one app's postings | `NAV_HEADING`, `NODE_REPORT_LOCAL.SPD`, `DVL_U@pDVL` |
| **LCM** | `CHANNEL.field`, `[k]` for array elements, dots into nested types | `STATE.vel_body[0]`, `NAV.pose.orientation[2]` |

**Format by format:**
- **PX4** is already forward-right-down / north-east-down (`frame: frd`).
  - Its quaternion is w, x, y, z: map `w: vehicle_attitude.q[0]`.
  - Its velocity is in the world frame: `linear_frame: world`.
  - Find names with `ulog_info -v my.ulg` (comes with pyulog) or PX4 Flight Review.
- **MOOS-IvP:**
  - `NAV_HEADING` is a compass heading in degrees, so NED yaw.
  - Calibration needs body velocity (your DVL driver's posting) and what each thruster was told (your thruster
    driver's posting), not `DESIRED_SPEED` or `DESIRED_THRUST`.
  - Find names with `alogscan my.alog`.
- **LCM:**
  - Give your message definitions as `lcm_types: [lcmtypes/*.lcm]`. Strathe turns them into Python with lcm-gen once
    (cached in `~/.strathe/lcm_types`) and decodes each message by its type's fingerprint.
  - Point `files:` at the logs (`lcmlog-*`), not the `.lcm` type files.
  - A channel name with dots works too.

## Time and the session date

| | Clock | `session_date: {from: recording}` |
|---|---|---|
| PX4 | `timestamp`: microseconds since the autopilot booted; UTC when the log has GPS time | Works with GPS time. Underwater logs usually have none: use `{from: filename}` or `{from: constant}` |
| MOOS-IvP | `%% LOGSTART` (Unix time) plus the log's seconds | Works |
| LCM | The message's own `utime` field when it has one (`stamp: header`, the default), else lcm-logger's receive time (`stamp: receive`) | Works |

**Every topic, variable or channel is put on one grid at `resample_hz`,** over the time they all cover:
- continuous values are interpolated;
- angles are unwrapped first;
- quaternions are interpolated on the sphere;
- thruster commands keep their last value until the next one.

The same holds for ROS bags and ArduPilot logs.

## When a log is damaged
| What | What happens |
|---|---|
| Cut off at power-down, or damaged in the middle | Read up to the damage. The ingest report says so, and the recording is shorter |
| Gaps (a sensor dropped out) | Interpolated across, and counted in the ingest report |
| NaN, repeated or out-of-order messages | Sorted, dropped and counted |
| A string where a number should be (MOOS: `DVL_U = no_lock`) | Skipped and counted |
| A name that isn't in the log | Refused, listing the names the log has |
| A topic logged for several instances, with no `[i]` | Refused, listing the instances |
| LCM: no type file matches a channel's messages | Refused, naming the channel |
| A key that doesn't apply (`time:`, or `stamp:` on PX4/MOOS, `lcm_types:` elsewhere) | Refused, saying why |
| An empty file | Refused |

## How we know it works
PROJECT_STATUS.md #94 and `STRESS_TEST_93_95.md` have every check.
- **Few public logs to test against:** we found no public real-vehicle MOOS-IvP or LCM logs with thruster commands,
  and PX4's public archive is mostly aircraft.
- **So each reader is proven on real data written into its format.** All of the BlueROV2 dataset's recordings (its
  logged forces and its real ESC PWM) and URI's ALPHA RISE missions were written as a vehicle logs them:
  - each topic at its own rate (attitude 50 Hz, velocity 10 Hz, commands 20 Hz);
  - commands held between updates.

  They were also written with the same content as a CSV.
- **Ingesting both gives the same recordings,** to float rounding:
  - for every option of the spec (each unit, frame, orientation and thruster setup), one at a time and in all
    pairs;
  - for every way of naming a column;
  - and calibrating both gives the same model.
- **Damaged logs:** every kind in the table above was made deliberately and gives the result it shows.
- **Independent decoders:** values match pyulog's own `ulog2csv`, a plain-text read of the `.alog`, and lcm's own
  decoder.
- **Not yet run on a lab's own MOOS-IvP or LCM log.** If yours misbehaves, send it (or its first minute): it's the
  check we're missing.
