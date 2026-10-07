# Strathe for labs: from your vehicle's logs to results

This page is the whole path for a lab: what to record, which file to start from, which commands to run, and what
comes back. Every command also has a form on the local web page (`python strathe.py web`), except where noted.
Known limits, with what to do about each, are in [LIMITS.md](LIMITS.md).

## 1. Install and check

```bash
pip install -r requirements.txt
python strathe.py check
```

`check` says what's missing, if anything. Python 3.10 or newer; macOS, Linux or Windows.

## 2. What to record

**The minimum: calibration from ordinary runs.**
- **Logs from two or more days.** Strathe checks itself on whole days it never saw during calibration.
- **Velocity:** from a DVL, an EKF that fuses one, or a positioning system.
- **Orientation**, and the **body rates** from the IMU.
- **What each thruster was told:** ESC PWM in µs, normalized commands (with your command-to-thrust table), or logged forces.
- **Varied driving:** manoeuvres on every axis you care about, at several speeds.
- **The tether slack.** Taut stretches get learned as drag. If you log tension, Strathe can drop them.

**Better:**
- **Battery voltage:** thrust drops as the battery sags.
- **An independent position track** (USBL, motion capture), to check position accuracy.
- **A note per log:** conditions, current, tether, what was run.

**Optional extras, each with its own step:**

| You record | It calibrates | Protocol / format |
|---|---|---|
| Tilt-and-release tests in a tank, about 30–60 min | roll and pitch (free decay) | [FREE_DECAY_PROTOCOL.md](calibration/FREE_DECAY_PROTOCOL.md) |
| Your modems' message logs: send time, receive time or lost, range | the acoustic and/or optical link | [link_log_template.csv](calibration/templates/link_log_template.csv) |
| Logs in a steady current | calibration in that current | `calibrate --current-world-mps`, or `--estimate-current` if unknown |
| Nothing extra: the same logs | your sensors' white noise and DVL dropouts, for realistic simulation (biases need an independent reference) | `fit-noise` (command line only) |

## 3. Pick a template

Copy the closest one from [calibration/templates/](calibration/templates/README.md) next to your logs and change the
column or topic names, units and frame. Each template's comments say what every line means.

| Your logs | Template |
|---|---|
| CSV with forces and torques | `ingest_csv_forces.yaml` |
| CSV with ESC PWM, BlueROV2 Heavy | `ingest_csv_pwm_bluerov2_heavy.yaml` |
| CSV with ESC PWM, any vehicle | `ingest_csv_pwm_custom_thrusters.yaml` |
| CSV with normalized commands (e.g. −1..1), any vehicle | `ingest_csv_normalized_thrusters.yaml` |
| ROS 1 / ROS 2 bags, ArduSub + MAVROS | `ingest_rosbag_mavros.yaml` |
| ROS 1 / ROS 2 bags, EKF or DVL + normalized commands | `ingest_rosbag_normalized_thrusters.yaml` |
| ArduSub DataFlash `.bin` / `.log` | `ingest_ardupilot_bin.yaml` |
| MAVLink `.tlog` from a ground station | `ingest_mavlink_tlog.yaml` |
| Free-decay tank tests on ArduSub | `ingest_free_decay_ardupilot.yaml` |

**Other formats:**
- **PX4 `.ulg`, MOOS, LCM, vendor binaries:** export to CSV, one table per recording.
- **One CSV per sensor:** merge the files onto one time base first ([examples/dfki_tether/](calibration/examples/dfki_tether/README.md) does exactly that, step by step).

**Thrusters:**
- **BlueROV2 Heavy:** the preset.
- **Any other vehicle:** a `layout:` with each thruster's position, the direction it pushes, and whether it's reversed, always in forward-right-down axes.
- **Thrust curve:** give your own command-to-thrust table, or the T200 curve is used.

## 4. Run it

```bash
# 1. calibrate: convert, check, fit, identifiability, compare with the real vehicle, report
python strathe.py calibrate --ingest-spec my_rov.yaml --name my_rov

# 2. (optional) roll and pitch from tank tests, then calibrate holding them
python strathe.py free-decay --ingest-spec my_tank_tests.yaml --out my_tank
python strathe.py calibrate --ingest-spec my_rov.yaml --name my_rov_fd --free-decay my_tank/free_decay.yaml

# 3. (optional) your communication link from message logs
python strathe.py fit-link --log messages.csv --out my_link

# 4. use the calibration
python strathe.py arena --params calibration_output_my_rov/calibrated_parameters.yaml --task station_keeping --compare-generic
python strathe.py fleet --params calibration_output_my_rov/calibrated_parameters.yaml --link-calibration my_link/link_calibration.yaml
python strathe.py train-fleet --params calibration_output_my_rov/calibrated_parameters.yaml --out my_policy
```

**Timing:**
- **Calibration:** a few minutes on a small log set, about an hour on large ones.
- **Keep it awake:** the machine is kept awake, but closing a laptop lid still sleeps it, so keep it on mains power.
- **If it was interrupted:** run the same command again with `--finish` to keep the best result so far.

## 5. What you get back

**`calibration_output_<name>/`**
- **`calibration_report.md`:**
  - a bottom line;
  - a verdict per motion axis (judged against "predict no motion");
  - values stuck at limits, and values the data can't separate;
  - **how the simulation compares with the real vehicle** on unseen recordings: position drift, orientation with the model turning on its own, the 1 s response, command-step timing, and plots;
  - next steps.
- **`calibrated_parameters.yaml`:** your vehicle's model. Every simulation command takes it with `--params`.
- **`validation/`:** the comparison's numbers (`validation.json`) and plots.

**From the optional steps:**
- **`free-decay`:** `free_decay.yaml` and a short report: the roll/pitch swing period, damping and righting.
- **`fit-link`:** `link_calibration.yaml`, a report and plots: delivery against range and latency for each link.
- **`arena` / `fleet`:** a report each, with confidence intervals, comparing your calibrated model against generic physics.

## 6. When something isn't right

- **Every step stops with a plain message saying what to fix**, for example:
  - a column that isn't there;
  - forces that push against the motion (a reversed layout, a mirrored heading);
  - commands outside their range;
  - only one recording day.
- **Read the report's "Be careful with" and next steps.** It says which axes not to trust and what data would fix them.
- **"Unstable" at the top of a report** means the calibrated simulation blew up on some unseen recordings. Don't simulate with it.
- **What Strathe can't do yet:** [LIMITS.md](LIMITS.md) lists every known limit with what to do about it. Roll and pitch prediction and the thruster lag are the main open ones.

## 7. What to send us, if you'd like help

- **The report folder** (`calibration_output_<name>/`): no raw logs needed.
- **Your ingest spec and `ingest_report.json`,** for conversion problems.
