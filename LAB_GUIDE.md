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
- **Battery voltage:** thrust drops as the battery sags (17% from 15.3 to 13.1 V on a T200). With your own command-to-thrust table, say the battery voltage it was measured at (`curve_voltage:`).
- **An independent position track** (USBL, motion capture), to check position accuracy.
- **A note per log:** conditions, current, tether, what was run.

**Optional extras, each with its own step:**

| You record | It calibrates | Protocol / format |
|---|---|---|
| Tilt-and-release tests in a tank, about 30–60 min | roll and pitch (free decay) | [FREE_DECAY_PROTOCOL.md](calibration/FREE_DECAY_PROTOCOL.md) |
| Your modems' message logs: send time, receive time or lost, range | the acoustic and/or optical link | [link_log_template.csv](calibration/templates/link_log_template.csv) |
| Logs in a steady current | calibration in that current | `calibrate --current-world-mps`, or `--estimate-current` if unknown |
| Nothing extra: the same logs | your sensors' white noise and DVL dropouts, for realistic simulation (biases need an independent reference) | `fit-noise` (command line only) |
| Nothing extra: the same logs, plus your **Stonefish** scenario | your own Stonefish vehicle model: its drag, added mass, buoyancy and CG, scored against the original on unseen recordings | `fit-stonefish` ([calibration/stonefish/README.md](calibration/stonefish/README.md)) |
| Nothing extra: your calibration, plus your **Stonefish** scenario | your calibration's values put into your Stonefish vehicle (minutes), scored the same way | `export-stonefish` ([calibration/stonefish/README.md](calibration/stonefish/README.md)) |

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

## 4. Describe your vehicle

The calibration fits your logs starting from a description of the vehicle: its mass, and a first guess and an
allowed range for its inertia, drag, added mass and buoyancy. **Without `--spec`, Strathe uses a BlueROV2 Heavy
(11.2 kg).** That's right only for a BlueROV2 Heavy. Any other vehicle needs its own description, or the fit is poor
and the simulation can blow up (a 33 kg AUV calibrated as a BlueROV2: no axis trustworthy, PROJECT_STATUS.md #70).

Copy [calibration/templates/vehicle_spec_template.yaml](calibration/templates/vehicle_spec_template.yaml) next to your
logs and fill it in. You need the vehicle's **mass** and its **size**; the template's comments give the formulas for
the rest:

| Value | First guess | Allowed range |
|---|---|---|
| mass | weigh it, in the configuration you logged | fixed |
| inertia (roll, pitch, yaw) | box: m(W²+H²)/12 etc.; cylinder: m·r²/2 (roll), m(3r²+L²)/12 (pitch, yaw); plus the water carried along | about 5× either way |
| added mass | box frame: 0.5–1.5 × mass; cylinder: 0.1 × mass in surge, ρπr²L sideways and vertically | 0 to about 4× |
| drag at speed | ½ρ·Cd·area facing the motion (Cd ≈ 1 broadside, ≈ 0.2 streamlined), or steady thrust ÷ speed² from your logs | about 5× either way |
| net buoyancy | 0 (positive floats) | about ±10% of the weight |
| hull lift, its moment, speed-dependent damping, fore-aft CG offset | **torpedo-shaped hulls only:** keep the template's `hull_lift:` section and `cg_cb_offset_x`; delete them for a box frame | as in the template |

- **From CAD or a pendulum test?** Use the value as the guess, and add it as `reference:`; the report then flags a
  fitted value far from it.
- **The report says a value is stuck at a limit?** Widen that limit once if the physics allows. If it hits the new
  limit too, the model is missing something for your vehicle ([LIMITS.md](LIMITS.md)); don't keep widening.
- **The same file goes to every command:** `calibrate --spec`, `arena --spec`, `train-fleet --spec`,
  `fleet --vehicle-spec`. On the web page, it's the "Vehicle description" field of each card.

`calibrate` warns before it starts when your logs' thrusters aren't a BlueROV2 Heavy's and no `--spec` was given.

## 5. Run it

```bash
# 1. calibrate: convert, check, fit, identifiability, compare with the real vehicle, report
python strathe.py calibrate --ingest-spec my_rov.yaml --spec my_vehicle_spec.yaml --name my_rov

# 2. (optional) roll and pitch from tank tests, then calibrate holding them
python strathe.py free-decay --ingest-spec my_tank_tests.yaml --out my_tank
python strathe.py calibrate --ingest-spec my_rov.yaml --spec my_vehicle_spec.yaml --name my_rov_fd --free-decay my_tank/free_decay.yaml

# 3. (optional) your communication link from message logs
python strathe.py fit-link --log messages.csv --out my_link

# 4. (optional) your own Stonefish model, calibrated from the same logs (calibration/stonefish/README.md);
#    export-stonefish converts your calibration's values in minutes and shows how far your model is from them
python strathe.py fit-stonefish --scenario my_world.scn --robot my_rov --part Hull --ingest-spec my_rov.yaml --reference-point 0 0 0
python strathe.py export-stonefish --calibration calibration_output_my_rov --scenario my_world.scn --robot my_rov --part Hull --reference-point 0 0 0

# 5. use the calibration
python strathe.py arena --params calibration_output_my_rov/calibrated_parameters.yaml --spec my_vehicle_spec.yaml --thrusters my_rov.yaml --task station_keeping --compare-generic
python strathe.py train-fleet --params calibration_output_my_rov/calibrated_parameters.yaml --spec my_vehicle_spec.yaml --thrusters my_rov.yaml --out my_policy
python strathe.py fleet --params calibration_output_my_rov/calibrated_parameters.yaml --vehicle-spec my_vehicle_spec.yaml --thrusters my_rov.yaml --policies my_policy.zip --link-calibration my_link/link_calibration.yaml
```

The BlueROV2 Heavy needs no `--spec` (it's the default) and can use `--thrusters bluerov2_heavy`. Train your own
fleet policy before running `fleet` on another vehicle: the shipped policies were trained on the BlueROV2.

**Timing:**
- **Calibration:** a few minutes on a small log set, about an hour on large ones.
- **Keep it awake:** the machine is kept awake, but closing a laptop lid still sleeps it, so keep it on mains power.
- **If it was interrupted:** run the same command again with `--finish` to keep the best result so far.
- **To repeat a calibration exactly:** add `--seed N` with the seed it printed (also in its report).
- **A hard calibration:** add `--restarts 3` (or 4). The fit starts from that many points and keeps the one that fits
  your training recordings best; the first fit takes that many times as long. The report lists each start's result,
  so you can see whether the starting point mattered.
  - **With `--estimate-current`,** only the first round restarts, and the start with the lowest round-1 cost isn't
    always the one that ends best. Add `--restart-whole-fit` to run the whole fit, every round, from each starting
    point and keep the lowest *final* training cost. It takes that many times as long; each start's result is kept
    in its `startN` folder, and the report lists their costs round by round.

## 6. What you get back

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
- **`fit-stonefish`:** your Stonefish vehicle file with the tuned part changed, and a report comparing it with the original on unseen recordings.
- **`export-stonefish`:** your Stonefish vehicle file with your calibration's values in it, the same comparison, and what Stonefish can't carry.
- **`arena` / `fleet`:** a report each, with confidence intervals, comparing your calibrated model against generic physics.

## 7. When something isn't right

- **Every step stops with a plain message saying what to fix**, for example:
  - a column that isn't there;
  - forces that push against the motion (a reversed layout, a mirrored heading);
  - commands outside their range;
  - only one recording day.
- **Read the report's "Be careful with" and next steps.** It says which axes not to trust and what data would fix them.
- **Many values stuck at their limits, or no axis trustworthy?** Check the vehicle description first (step 4): the
  report says so at the top of "Be careful with" when a BlueROV2's description was used for other thrusters.
- **"Unstable" at the top of a report** means the calibrated simulation blew up on some unseen recordings. Don't simulate with it.
- **What Strathe can't do yet:** [LIMITS.md](LIMITS.md) lists every known limit with what to do about it. Roll and pitch prediction and the thruster lag are the main open ones.

## 8. What to send us, if you'd like help

- **The report folder** (`calibration_output_<name>/`): no raw logs needed.
- **Your ingest spec and `ingest_report.json`,** for conversion problems.
- **Your vehicle description** (`my_vehicle_spec.yaml`).
