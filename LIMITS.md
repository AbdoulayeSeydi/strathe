# Known limits

Every known limit of Strathe in one place, from a sweep of the code, its docs
and the research log. Each entry says what the limit means for a lab and what
to do about it. When a limit is lifted, it moves to the "Lifted" list at the
end, with the finding that lifted it. `PROJECT_STATUS.md #N` points to the
evidence.

Quick answers:
- **Can I use my own vehicle?** Yes: calibrate it from its logs, then use it in
  one- and two-vehicle experiments ([Your vehicle](#your-vehicle-and-its-thrusters)).
- **Are my log formats supported?** CSV, ROS 1/2 bags, ArduPilot `.bin` and
  MAVLink `.tlog` are read directly ([Logs](#logs-and-data-formats)).
- **Is it validated on hardware?** Not yet. Everything is simulation
  checked against held-out real logs ([Validation](#validation)).

---

## Validation

| Limit | What it means | What to do |
|---|---|---|
| **No hardware-in-the-loop validation yet** | Results are simulation. Calibration is checked on recording days it never saw, not on a live vehicle. | Run the same commands on the vehicle and in the calibrated simulator and compare. That's the partnership we're looking for. |
| **"calibrated" in fleet results means "a model equal to the simulated vehicle"** | The fleet studies show the value of an accurate teammate model. How close a real calibration gets to a real vehicle is measured separately, per axis, in the calibration report. | Read both reports together. |
| **HoloOcean results are provisional** | A frame fix and a tick-rate guard are unconfirmed in HoloOcean itself; it doesn't run on macOS (#40). MuJoCo and the stand-in agree with each other. | Run `verify_holoocean_frame_conventions.py` on a HoloOcean machine before relying on HoloOcean numbers. |

## Logs and data formats

| Limit | What it means | What to do |
|---|---|---|
| **Supported formats: CSV, ROS 1 `.bag`, ROS 2 `.mcap`/`.db3`, ArduPilot DataFlash `.bin`/`.log`, MAVLink `.tlog`** | Other formats aren't read directly: PX4 `.ulg`, LCM, MOOS `.alog`, vendor DVL binaries, spreadsheets. | Export to CSV (one row per sample, any column names) and use `templates/ingest_csv_*.yaml`. |
| **Older ROS 2 `.db3` bags (before Iron) with custom messages** | They don't store message definitions, so custom types like `mavros_msgs/RCOut` can't be decoded on their own. | List the `.msg` files under `source.message_types` (templates README). ROS 1 bags and `.mcap` bags are fine. |
| **Body-frame velocity is required** | Calibration needs velocity (surge/sway/heave) from a DVL, visual odometry or a positioning system. World-frame velocity works too (`linear_frame: world`). | ArduSub's EKF velocity *without* a DVL or positioning system is dead reckoning and won't calibrate anything useful. |
| **ArduPilot `.bin` without GPS has no calendar date** | Underwater logs usually have no GPS, so their clock is time since boot. Calibration splits recordings by date to test on days it never saw. | Use `session_date: {from: filename}` or `{from: constant, value: ...}`. `.tlog` files carry the ground station's clock. |
| **MAVLink telemetry is low-rate by default** | Ground stations stream ATTITUDE/SERVO_OUTPUT_RAW at a few Hz unless told otherwise. | Raise the stream rates to at least 10 Hz before recording (template comments), and set `resample_hz` no higher. |
| **Orientation glitches** | Real IMU logs contain sudden jumps (the reference data has 239 jumps over 20° within 0.1 s). Ingestion slerps across them and warns about jumps over 45°. | Check the warnings in `ingest_report.json`; trim or drop recordings with many. |
| **One vehicle per ingestion spec, one sampling rate per calibration** | All files converted together must end up at one rate. | Set `resample_hz`. Use separate specs for separate vehicles. |
| **Session dates** | At least two recording days are needed, so one can be held out. | Record on more than one day, or give each session a distinct date. |

## Your vehicle and its thrusters

| Limit | What it means | What to do |
|---|---|---|
| **One built-in preset: BlueROV2 Heavy** | Any other vehicle is described once with a `layout:`: position, direction and reversed flag per thruster. There is no one-line preset for, e.g., the standard 6-thruster BlueROV2. | Copy `templates/ingest_csv_pwm_custom_thrusters.yaml` and edit the layout. Vehicles that can't push every axis are supported. |
| **The thruster model isn't calibrated** | The PWM → thrust curve and the layout are taken as given (T200 data, or your curve file). Errors in them become errors in the calibrated drag. | Use a bollard-pull curve at your battery voltage if you have one (`curve_csv`). |
| **No thruster response lag, no battery sag** | PWM maps to thrust instantly, at one voltage per spec. | Set `voltage:` to your typical battery voltage. |
| **The thruster-limit mixer isn't ArduSub's exact mixer** | In the simulators a command is split over the thrusters by least squares and scaled down as a whole if any thruster saturates. That matches ArduSub's direction-preserving behaviour, not its exact code. | Fine for experiments; don't expect saturation to match the vehicle to the newton. |
| **Default action bound is ±30 N / N·m per axis without thruster limits** | Without `thrusters:`, any policy command up to 30 on every axis is delivered, including ~30 N·m of pitch torque a BlueROV2 Heavy can't make (~18 N·m). | Turn on `thrusters:` (`--thrusters`) for realistic authority. |

## Calibration (the physics model and the fit)

| Limit | What it means | What to do |
|---|---|---|
| **Roll and pitch aren't predictive on the reference data** | After the torque-arm fix they are at about zero skill: no better than "no motion" (#50). The dataset's roll/pitch motion is mostly not caused by the thrusters (tether, waves). This doesn't affect formation-holding (#52). | For attitude-sensitive work, record free-decay tests and deliberate roll/pitch moves with the tether slack. `--exclude-axes p q` leaves them out of the fit. |
| **Many parameters end at their allowed limits** | On the reference data, 8–15 of 22 parameters sit at `vehicle_spec.yaml` bounds (#50). The report lists them in plain words. | Review the bounds with someone who knows the vehicle, widen where physical, and refit. |
| **Some parameters can't be told apart** | E.g. linear vs quadratic drag on the same axis. The report's identifiability section names the pairs and groups. | Treat those individual values as uncertain. Manoeuvres at varied speeds while turning separate them. |
| **Model structure** | Diagonal added mass, a centre-of-gravity offset in z only, no drag moments, no payload changes, no off-diagonal added-mass coupling (`vehicle_spec.yaml` assumptions). | These are the model's assumptions; they are listed in the vehicle description. |
| **Teacher-forced orientation during calibration** | The fit replays logged roll/pitch rather than integrating them (`dynamics_model.py`). Its error isn't directly comparable to a free-running simulator's. | The report's per-axis skill is measured the same way for every model, so comparisons within Strathe are fair. |
| **Explicit Euler integration** | The physics uses small sub-steps (0.005 s). An unusual vehicle could need smaller ones. | If a simulation raises `FloatingPointError`, lower `physics_dt` before blaming the parameters. |
| **Budget** | A full fit on a large log set takes about an hour (7 cores). If the evaluation budget runs out, the best result so far is kept and marked "not converged". | Re-run with a larger `--max-nfev` for a converged fit. |
| **Long runs need the machine awake** | Long runs keep the machine awake themselves (#49), but closing a laptop lid still sleeps it. | Leave the lid open, or run on a desktop or server. |

## Water current

| Limit | What it means | What to do |
|---|---|---|
| **One constant current per recording** | Calibration, and current estimation, assume the current is the same throughout each recording. A current that changes within a recording, or differs from place to place, isn't modelled. | Keep recordings short enough (a few minutes) that the current is steady. `estimate-current` checks each recording by estimating its two halves separately, and flags ones where they disagree. |
| **Estimating an unknown current needs a decent vehicle model and some turning** | With uncalibrated default physics the estimates are unreliable, and the tool says so ("explains only N% of the error"). A vehicle that never turns can't separate a current from drag along its heading. | Use `strathe calibrate --estimate-current`, which alternates fitting and estimating. Include turns in open-water recordings. |
| **Simulated currents are uniform in space** | In experiments the current can vary in time (slow fluctuation), and it's the same everywhere. No shear or eddies; both fleet vehicles feel the same water. | Enough for station-keeping and formation studies over tens of metres. |
| **Policies don't observe the current** | They see their own velocity over ground, like a DVL on bottom track. | Pass the mean current to physics teammate models with `known_to_beliefs`. |

## Single-vehicle arena (`strathe arena`)

| Limit | What it means | What to do |
|---|---|---|
| **Training always uses the stand-in engine** | MuJoCo (`--backend mujoco` or `both`) is for testing; training on it would be ~3× slower. The two engines agree closely (#55, and the arena stress test). | Use `--backend both` to check that the results hold on both engines. |
| **Four tasks** | Reach a point, hold station, dock, avoid one obstacle. New tasks need a reward and a config in `arena.py`. | Ask us, or subclass `Task`. |
| **Episodes start level, at the origin, at rest** | Targets are random within ±3 m. | |
| **The fitted disturbance was fitted to the reference BlueROV2** | `--disturbance fitted` reproduces that vehicle's tether/wave roll/pitch wobble. On another vehicle it's a generic "rough water" setting. | Fit your own with `fit_rollpitch_disturbance.py`, then pass the file. |
| **PPO only** | The runner trains PPO. SAC and TD3 exist in `train_multiseed.py` for specialists. | `--policy` evaluates a PPO policy you trained elsewhere. |
| **Training-seed variance** | Reinforcement learning results vary between seeds. A heavily damped vehicle is slow to learn on (10% vs 96% success at 100k vs 300k steps, #56). | Use several seeds and a longer `--timesteps`; read the per-seed numbers. |

## Fleet (two vehicles)

| Limit | What it means | What to do |
|---|---|---|
| **Exactly two vehicles** | N ≠ 2 is rejected. The per-vehicle observation and shared policy are the groundwork for N. | |
| **Both vehicles are the same vehicle** | They share `truth_params`, `vehicle_spec` and `thrusters`. | Two different vehicles in one fleet isn't supported yet. |
| **One task: formation hold** | Hold a separation for a number of ticks without colliding. | Separation, tolerance, hold time, episode length and collision radius are all settable (`strathe fleet --target-separation ...`). |
| **Start state** | Level, facing +x, on a line. Separation and closing speed are chosen per start. | `--separations`, `--closing-speeds`. |
| **Policies act on beacons plus a teammate model; no intent sharing** | Calibrated prediction ties "last beacon" in still water (#54). It pulls ahead in a current it knows about (#55). | Intent-sharing beacons are the next lever (research). |
| **Policies trained for one teammate model vary across seeds** | Policies specialised to the calibrated model ranged 35–98% (#54). The mixed-trained policies are consistent. | Use `--belief mix` (the default) for policies you report. |
| **The shipped policies were trained on the reference BlueROV2** | They run on other vehicles, but results for your vehicle should come from policies trained on it. | `strathe train-fleet --params <your calibration>` (~40 min). |
| **No sensor noise on a vehicle's own state** | Each vehicle knows its own velocity and attitude exactly. Roll/pitch disturbance is modelled; sensor noise isn't. | |

## Communication model

| Limit | What it means | What to do |
|---|---|---|
| **Distance-based packet loss, formula mode only** | No obstruction or multipath, no fitting to measured modem data, no message priorities (`comm_channel.py`). | The named levels L0–L3b span 0–60% loss and a bandwidth-starved case. |
| **Beacon size is nominal (24 bytes)** | It isn't a real modem framing format. | Matters only for the bandwidth-starved level L3b. |

## Engines

| Limit | What it means | What to do |
|---|---|---|
| **Three engines** | Stand-in (fast, used for training), MuJoCo (independent check), HoloOcean (provisional, see Validation). MuJoCo integrates the same calibrated equations. It is a check on integration and frames, not a different hydrodynamic model. | |
| **Parallel restarts don't checkpoint** | `run_calibration.py --parallel` can't write a checkpoint from several processes. | Use `--eval-workers` (the default in `strathe calibrate`), which checkpoints. |

## Interfaces

| Limit | What it means | What to do |
|---|---|---|
| **The web page is local and single-user** | It listens on this computer only and runs Strathe's own commands. It isn't a shared server. | Each lab member runs their own. |
| **Some settings are YAML-only** | Comm overrides (custom loss probabilities, speed of sound), custom teammate-model entries (`physics_params`, `physics_alpha`) and the task's reward weights. | `strathe fleet --save-config my.yaml` writes a full experiment file to edit. Everything else is a flag or a web field. |
| **Python 3.10+; HoloOcean needs a manual install** | `pip install -r requirements.txt` covers everything else, on macOS, Linux and Windows (CI runs on Windows). | |

---

## Lifted

What used to be a limit, and where it was fixed:
- **Logs in other formats (#56, #57):**
  - ROS bags are read directly (#56).
  - ArduPilot `.bin` and MAVLink `.tlog` are read directly (#57).
- **Custom thrusters needed hand-made matrix files (#56):** a `layout:` in the ingestion YAML now does it.
- **Water current:**
  - A known current is supported (#55).
  - An unknown one is estimated per recording (#57).
- **Custom fleet experiments needed YAML editing (#57):** comm levels, beacon interval, teammate models, task, starting conditions, sample size, engine and disturbance are flags and web fields now.
- **Single-vehicle arena on the stand-in engine only, with no roll/pitch disturbance (#57):** `--backend mujoco|both` and `--disturbance fitted` now exist.
- **The one-command calibration skipped the identifiability check (#56).**
- **It assumed 10 Hz (#56).**
- **The data check crashed on converted logs (#56).**
- **The overnight "4-hour evaluation" stall (#49):** the machine was sleeping.
