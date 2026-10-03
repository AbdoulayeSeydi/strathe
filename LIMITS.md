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
- **Can the simulation be realistic rather than perfect?** Yes: `--mode noisy`
  adds sensor noise and roll/pitch disturbance, `--mode both` runs clean and
  noisy side by side, and `--buoyancy` changes the trim
  ([Clean vs noisy](#clean-vs-noisy-simulation-and-buoyancy)).
- **Can the fleet use an optical link?** Yes: `--link optical | hybrid`, with water
  clarity and terminal pointing, but the optical model is uncalibrated
  ([Communication model](#communication-model)).

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
| **One table per recording (CSV)** | CSV ingestion reads one file per recording with every column in it. Logs exported as one CSV per sensor, at different rates, have to be merged first. | Merge them onto one time base; `calibration/examples/dfki_tether/` does it for a public dataset (motion capture + ArduSub telemetry, #62). |
| **Session dates** | At least two recording sessions are needed, so one can be held out. Ingestion warns, and calibration stops with a plain message, when every recording has the same date. | Record on more than one day, or split one day's recordings into sessions you can tell apart (e.g. morning and afternoon) and give each its own date. |

## Your vehicle and its thrusters

| Limit | What it means | What to do |
|---|---|---|
| **One built-in preset: BlueROV2 Heavy** | Any other vehicle is described once with a `layout:`: position, direction and reversed flag per thruster. There is no one-line preset for, e.g., the standard 6-thruster BlueROV2. | Copy `templates/ingest_csv_pwm_custom_thrusters.yaml` and edit the layout; `examples/dfki_tether/ingest_dfki_tether.yaml` is a standard 6-thruster BlueROV2. Vehicles that can't push every axis are supported. |
| **The shipped reference calibration is a mirror image** | The reference dataset is in ROS axes (forward-left-up, z up) and was calibrated as it is. Its model is the same vehicle turned half a turn about x: identical for every experiment, except that its fitted net buoyancy and surge-pitch coupling (`Xq_dot`) have the opposite sign to the real vehicle's, and with `--thrusters bluerov2_heavy` the vertical thrusters' stronger direction (47.9 vs 37.2 N each) is reversed relative to gravity (#62). | Nothing for formation or station-keeping experiments. Don't quote the reference calibration's net buoyancy as the real vehicle's. Your own calibration, from logs converted at ingestion, is in FRD/NED. |
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

## Clean vs noisy simulation, and buoyancy

| Limit | What it means | What to do |
|---|---|---|
| **The sensor model is simple** | White noise, a constant bias per episode, and DVL dropouts, per 0.1 s tick (`sensor_noise.py`). Not modelled: compass errors that depend on heading, gyro bias that wanders during a run, scale-factor errors, sensor latency, DVL altitude limits. | Enough to test whether a result survives realistic sensors. Measure your own sizes with `strathe fit-noise`. |
| **`typical` is assumed; `tank` and `open_water` are measured** | `tank` is the noise in the reference tank logs (small, with external positioning). `open_water` is measured from SINTEF's SOLAQUA data, a BlueROV2 at a fish farm (#60): white noise, DVL dropouts and the USBL starting-fix error. Its biases are `typical`'s assumed values. `typical` (DVL + AHRS + dead reckoning) uses stated sizes of a common magnitude, **not measured on any vehicle**, and is still the default of `--mode noisy`. Every report says which. The measured open-water white noise is 2–35× smaller than `typical`'s; the starting fix (0.17 m measured, 0.2 m assumed) is close. | For your vehicle, measure your own with `strathe fit-noise --data <logs>` and pass the file, adding datasheet biases. `--noise open_water` is the measured open-water reference until then. |
| **Biases can't be measured from logs** | A constant offset per dive is invisible without an independent reference, so `fit-noise` writes biases as 0 and says so. SOLAQUA has a USBL reference, but its DVL tracks the fish-farm net, which moves with the current, so it can't measure them either (#60). | Put your sensors' datasheet biases in the file. A bias needs a seabed-locked DVL logged with an independent position and heading reference. |
| **The starting position fix is most of what sensor noise costs the fleet** | Each vehicle starts dead reckoning from its own fix, with an independent error, so the relative position is off by about 1.4× the fix error against a ±0.2 m tolerance. With perfect sensors, a 0.05 / 0.1 / 0.2 / 0.3 m fix costs 1 / 3 / 9 / 18 points; with a perfect fix, the `typical` or measured open-water sensors cost about 1 (#60). | Set `--start-fix-error` (or `sensor_noise.start_fix_error_m`) to your positioning's accuracy, and report it with any noisy result. Relative positioning between the vehicles (acoustic ranging) is the lever. |
| **`open_water`'s DVL tracked a net at 0.5–2 m** | In SOLAQUA the DVL locks onto the fish-farm net, not the seabed. A DVL at that range is quieter than one bottom-locked at 10–50 m, and gyro, roll/pitch and heading are ArduSub's filtered estimates, not raw IMU samples. | Treat `open_water`'s DVL noise as a lower bound for deep bottom-lock; `--noise-scale 2` brackets it. |
| **`fit-noise` on resampled logs underestimates noise** | `fit-noise` reads one CSV per recording at one rate. Logs converted from a different native rate are interpolated, which smooths white noise. | Convert at the sensors' native rate (`resample_hz`), or measure each stream at its own rate as `fit_solaqua_noise.py` does. |
| **Navigation is raw dead reckoning or external positioning** | No mid-run USBL/LBL fixes and no navigation filter: the policy sees dead-reckoned position that drifts from a starting fix. | Use `position: external` (with its accuracy) for a vehicle with continuous positioning. |
| **Beacons carry each vehicle's believed position; no acoustic ranging** | Real modems can also measure the range between vehicles, which would cancel the two vehicles' independent starting-fix errors, the main cost of sensor noise here (#60). | Research: ranging-aided teammate models. |
| **A buoyancy change is a net force only** | `--buoyancy` changes the net buoyancy of both vehicles by the same amount. The centre of gravity, the righting moment and the mass don't change; nothing changes with depth. | Recalibrate after a large payload change. |
| **Policies can push back at most 30 N vertically without thruster limits** | Beyond that a policy can't hold depth; with `--thrusters` the real limit is lower. Values beyond ±50 N are refused as a probable units mistake. | |

## Single-vehicle arena (`strathe arena`)

| Limit | What it means | What to do |
|---|---|---|
| **Training always uses the stand-in engine** | MuJoCo (`--backend mujoco` or `both`) is for testing; training on it would be ~3× slower. The two engines agree closely (#55, and the arena stress test). | Use `--backend both` to check that the results hold on both engines. |
| **Four tasks** | Reach a point, hold station, dock, avoid one obstacle. New tasks need a reward and a config in `arena.py`. | Ask us, or subclass `Task`. |
| **Episodes start level, at the origin, at rest** | Targets are random within ±3 m. | |
| **The fitted disturbance was fitted to the reference BlueROV2** | `--disturbance fitted` reproduces that vehicle's tether/wave roll/pitch wobble. On another vehicle it's a generic "rough water" setting. | Fit your own with `fit_rollpitch_disturbance.py`, then pass the file. |
| **SAC and TD3 use stable-baselines3's own settings** | `--algorithm sac\|td3` trains with one environment and SB3's defaults, the setup validated on this vehicle (#37). No hyperparameter tuning per vehicle or task. | Compare seeds; PPO remains the default. |
| **Training-seed variance** | Reinforcement learning results vary between seeds. A heavily damped vehicle is slow to learn on (10% vs 96% success at 100k vs 300k steps, #56). | Use several seeds and a longer `--timesteps`; read the per-seed numbers. |

## Fleet (two vehicles)

| Limit | What it means | What to do |
|---|---|---|
| **Exactly two vehicles** | N ≠ 2 is rejected. The per-vehicle observation and shared policy are the groundwork for N. | |
| **Both vehicles are the same vehicle** | They share `truth_params`, `vehicle_spec` and `thrusters`. | Two different vehicles in one fleet isn't supported yet. |
| **One task: formation hold** | Hold a separation for a number of ticks without colliding. | Separation, tolerance, hold time, episode length and collision radius are all settable (`strathe fleet --target-separation ...`). |
| **Start state** | Level, facing +x, on a line. Separation and closing speed are chosen per start. | `--separations`, `--closing-speeds`. |
| **Policies act on beacons plus a teammate model; no intent sharing** | Calibrated prediction ties "last beacon" in still water (#54). It pulls ahead in a current it knows about (#55). A beacon carries the teammate's state and the thrust it delivered on the send tick, never a planned command. An optical link (a beacon every tick) gives high-rate *state* sharing (#59), not intent sharing. | `--link hybrid` for high-rate state sharing. Intent sharing (planned commands or a trajectory in the beacon) is research, the open lever from #48. |
| **Policies trained for one teammate model vary across seeds** | Policies specialised to the calibrated model ranged 35–98% (#54). The mixed-trained policies are consistent. | Use `--belief mix` (the default) for policies you report. |
| **The shipped policies were trained on the reference BlueROV2** | They run on other vehicles, but results for your vehicle should come from policies trained on it. | `strathe train-fleet --params <your calibration>` (~40 min). |
| **SAC and TD3 fleet policies were validated on one task and one vehicle** | Three seeds each, 1M steps, passed criteria fixed in advance: SAC ties PPO (−0 points, 95% CI −1 to +1) and TD3 is 4 points better (+3 to +6) over 17,600 paired episodes (#58). That's formation hold on the reference BlueROV2, clean simulation, SB3-standard settings with no tuning for another vehicle. The ranking doesn't necessarily carry over: TD3 went from best in clean simulation to worst under noise (−22 points). | For your vehicle, train two or three seeds of each (`strathe train-fleet --params <yours> --algorithm sac --seed 1 ...`) and compare them in one experiment (`--policies ppo=a ppo=b sac=c sac=d`), ideally with `--mode both`. The report's "Policy groups compared" table gives the paired difference with a 95% CI. |
| **SAC and TD3 cost more per step** | Each off-policy step includes gradient updates from a replay buffer: about 500-650 steps/s on one M1 core vs about 770 for PPO. With a third of the steps, a 1M-step run takes about 30-35 min alone, against about 65 min for PPO's 3M. Several runs sharing a machine slow each other down (the study's own runs took 2-2.5 h with 9 at once). | Run fewer at a time than cores, as with PPO. |

## Communication model

| Limit | What it means | What to do |
|---|---|---|
| **Distance-based packet loss, formula mode only** | No obstruction or multipath, no fitting to measured modem data, no message priorities (`comm_channel.py`). | The named levels L0–L3b span 0–60% loss and a bandwidth-starved case. |
| **Beacon size is nominal (24 bytes)** | It isn't a real modem framing format. | Matters only for the bandwidth-starved level L3b. |
| **The optical link is uncalibrated** | Delivery follows received light power `exp(-c·d)/d²` through a smooth S-curve whose midpoint is set by `rated_range_m` (50% delivery at 30 m in clear water by default). Both the curve and its width are placeholders, not fitted to any modem. UNH's thesis gives one data point (about 20 m, 5 kbps, 3 nodes). Every report that uses the optical link says it is uncalibrated. | Set `rated_range_m` and `data_rate_bps` from your modem's datasheet. Calibrating the curve needs your link logs: delivery against range and angle. |
| **Water clarity values are textbook figures** | `clear` 0.15, `coastal` 0.40, `harbor` 2.2 per metre: classic measured attenuation coefficients as usually quoted, not your site. Turbidity doesn't change during a run. | `water: custom` with `attenuation_per_m` from a transmissometer at your site; list several waters to bracket it. |
| **Pointing is a hard cone on the true attitude** | Inside a terminal's half-angle the link is at full strength; outside it, nothing. The true roll, pitch and yaw decide it, so optical results are more sensitive to attitude than acoustic ones, and roll/pitch isn't predictive yet (#50). The shipped policies roll and pitch freely while holding formation, so narrow cones rarely stay pointed at the teammate (#59). | `terminals: omni` (the default) is the fair baseline. Use `compare_omni` to measure what pointing costs; train with `--link hybrid` to see whether a policy learns to keep its terminals pointed. |
| **Not modelled for light: occlusion, sunlight, bubbles, multipath** | The other vehicle's hull, the seabed or anything in between never blocks the beam; ambient light near the surface and bubbles don't degrade it. The same holds for acoustic occlusion. | Treat near-surface optical results as optimistic. |
| **Two vehicles, so no shared acoustic channel** | Acoustic energy and airtime cost nothing here, so hybrid `redundant` (every beacon on both links) is never worse than `optical_first`. The value of `optical_first` (saving acoustic energy and channel time) can't show up until three or more vehicles share a channel. | Report `redundant` for what the information is worth; `optical_first` for what a real link manager would do. |
| **`optical_first` decides the link is up from what it has heard** | A vehicle skips its acoustic beacon only if an optical message from its teammate arrived within the last `link_up_ticks` ticks (default 1: on the previous tick). Symmetric terminals make the geometry reciprocal at an instant, but each direction draws its losses independently, so a skipped acoustic beacon's optical copy can still be lost. The report counts those (3.2% of skips across the stress grid). With asymmetric terminals (beam ≠ receiver half-angle) the inference is unsafe, so it runs as `redundant` automatically, and the report says so. No acknowledgements are modelled. | Set `link_up_ticks` (YAML) to match your modem's link-status logic; wider windows skip more and lose more. |
| **No latency advantage for optical** | Messages are delivered on the 0.1 s tick: anything sent with any latency above zero arrives on the next tick. Optical's 1 ms and acoustic's few milliseconds at fleet ranges (under 150 m) are therefore identical in the simulation. Optical's advantage here is its send rate (every tick) and its loss, not its latency. | Don't read a latency effect into optical results. Acoustic latency starts to cost a tick beyond ~150 m. |
| **The shipped fleet policies never saw an optical link** | They were trained with acoustic beacons every 1–3 s. With omni terminals an optical or hybrid link lifts them to 100% in every loss condition. With 30° cones they lose the link as they roll, and fore/aft cones raise collisions: an early fix goes stale while they close (#59). What a policy trained with a link could do isn't measured yet. | `strathe train-fleet --link hybrid` (and the web form) trains with it. |

## Engines

| Limit | What it means | What to do |
|---|---|---|
| **Three engines** | Stand-in (fast, used for training), MuJoCo (independent check), HoloOcean (provisional, see Validation). MuJoCo integrates the same calibrated equations. It is a check on integration and frames, not a different hydrodynamic model. | |
| **Parallel restarts don't checkpoint** | `run_calibration.py --parallel` can't write a checkpoint from several processes. | Use `--eval-workers` (the default in `strathe calibrate`), which checkpoints. |

## Interfaces

| Limit | What it means | What to do |
|---|---|---|
| **`fit-noise`, `report`, `check` and a standalone `estimate-current` have no web form** | They run from the command line only (`python strathe.py fit-noise --data <logs> --out my_noise.yaml`). The web page can *use* a noise file that `fit-noise` wrote ("From my own file" on the one-vehicle, fleet and training forms), and estimating currents during calibration is a checkbox on its form. | Run `fit-noise` once from a terminal, then pick the file on the web page. |
| **The web page is local and single-user** | It listens on this computer only and runs Strathe's own commands. It isn't a shared server. | Each lab member runs their own. |
| **Some settings are YAML-only** | Comm overrides (custom loss probabilities, speed of sound), custom teammate-model entries (`physics_params`, `physics_alpha`), the task's reward weights, and the finer optical settings (a custom terminal layout, the delivery curve's width, the modem latency, `link_up_ticks`). | `strathe fleet --save-config my.yaml` writes a full experiment file to edit. Everything else is a flag or a web field. |
| **Python 3.10+; HoloOcean needs a manual install** | `pip install -r requirements.txt` covers everything else, on macOS, Linux and Windows (CI runs on Windows). | |

---

## Lifted

What used to be a limit, and where it was fixed:
- **Logs in other formats (#56, #57):**
  - ROS bags are read directly (#56).
  - ArduPilot `.bin` and MAVLink `.tlog` are read directly (#57).
- **Custom thrusters needed hand-made matrix files (#56):** a `layout:` in the ingestion YAML now does it.
- **The BlueROV2 Heavy preset was in ROS axes (#62):** it pushed sideways, vertically, in pitch and in yaw the wrong way for ArduSub logs. It is now in ArduSub's motor numbering and FRD, checked against ArduSub's motor table and against a second, independent BlueROV2 with motion capture.
- **`strathe calibrate` only worked on logs with the reference dataset's dates (#62):** the data check, the identifiability step and the report defaulted to its held-out dates; they now use the same split as calibration.
- **Water current:**
  - A known current is supported (#55).
  - An unknown one is estimated per recording (#57).
- **Custom fleet experiments needed YAML editing (#57):** comm levels, beacon interval, teammate models, task, starting conditions, sample size, engine and disturbance are flags and web fields now.
- **Single-vehicle arena on the stand-in engine only, with no roll/pitch disturbance (#57):** `--backend mujoco|both` and `--disturbance fitted` now exist.
- **No sensor noise on a vehicle's own state (#58):** `--mode noisy` / `--noise`, in both simulators, training, the CLI and the web page; `--mode both` compares clean and noisy episode by episode.
- **Training with PPO only (#58):** `--algorithm sac|td3` for one vehicle and for the fleet; policies of any of the three are recognised from the file.
- **A vehicle heavier or lighter than calibrated couldn't be simulated (#58):** `--buoyancy`.
- **No open-water sensor noise was measured (#60):** `--noise open_water`, from SINTEF's SOLAQUA data (white noise, dropouts, USBL fix; biases still assumed).
- **Acoustic links only (#59):** `--link optical | hybrid`, with water clarity and terminal pointing, in the fleet simulation, training, the CLI and the web page.
- **The one-command calibration skipped the identifiability check (#56).**
- **It assumed 10 Hz (#56).**
- **The data check crashed on converted logs (#56).**
- **The overnight "4-hour evaluation" stall (#49):** the machine was sleeping.
