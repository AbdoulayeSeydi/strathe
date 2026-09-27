# Strathe — Communication-Constrained Multi-Agent Simulation for Underwater Autonomy

A simulation and RL-training platform for studying how real-world communication constraints — latency, packet loss, limited bandwidth — affect coordination between multiple underwater vehicles, grounded in physics calibrated against real vehicle mission logs, not an idealized model.

**Current real scope:**
- one-vehicle tasks and two-vehicle formation-hold
- the reference BlueROV2, or a lab's own vehicle calibrated from its logs

The published results are for the reference BlueROV2. Larger fleets are a planned next step.

**Every known limit is listed in one place: [LIMITS.md](LIMITS.md).**

**What this answers:** when two underwater vehicles coordinate over a sparse, lossy acoustic link, each has to predict where its teammate is between messages. Does it matter whether that prediction uses a physics model calibrated against real vehicle logs, or generic textbook physics?

**Answer so far** (simulation only; 38,400 paired episodes in the main study, 3 training seeds; confirmed on MuJoCo and with a second "true vehicle"):
- **A calibrated teammate model beats a generic one by +13 to +25 points** of formation success wherever the task isn't nearly solved. The gap is largest when beacons are sparse or lossy.
- **An uncalibrated predictive model is worse than no prediction at all**, by 12 to 25 points in those conditions.
- **Better teammate information is worth up to +28 points** (perfect information vs none).
- **Calibrated physics prediction now ties simply using the last message** (−1 to +3 points). The remaining error comes from the teammate's changing commands, not its physics, so sharing intent in the beacon is the next lever.
- **Caveat:** policies trained for the calibrated model alone vary a lot between training seeds with the current truth vehicle. The claims above rest on policies trained on a mixture of models, which are consistent (PROJECT_STATUS.md #54).

See [Fleet coordination results](#fleet-coordination-results). An earlier headline metric (fleet v1, "Claim C") was withdrawn after it turned out not to measure calibration at all ([PROJECT_STATUS.md](PROJECT_STATUS.md) #46).

This rests on a system-identification pipeline that fits physics to real BlueROV2 logs, validated on held-out recording sessions (see [Single-Vehicle Calibration Results](#single-vehicle-calibration-results)).

## What a lab can do with it

Every row works from one command (`python strathe.py ...`) and from the local web page (`python strathe.py web`).

**1. Calibrate your own vehicle from its logs**

| You want to... | How |
|---|---|
| Calibrate from raw logs in one step | `calibrate --ingest-spec my_rov.yaml`: converts, checks, fits, finds what the data can't pin down, and writes a plain-language report |
| Use CSV logs with your own column names and units | a CSV template in `calibration/templates/` |
| Use ROS 1 / ROS 2 bags | read directly: `templates/ingest_rosbag_mavros.yaml` |
| Use ArduSub / ArduPilot logs | read directly: `.bin` from the SD card (`templates/ingest_ardupilot_bin.yaml`), `.tlog` from the ground station (`templates/ingest_mavlink_tlog.yaml`) |
| Use logs with only thruster PWM, no forces | BlueROV2 Heavy preset, or any vehicle's thruster `layout:` |
| Calibrate logs recorded in a known current | `calibrate --current-world-mps VX VY VZ` |
| Calibrate logs recorded in an unknown current | `calibrate --estimate-current`: each recording's current is estimated from the logs |

**2. Run experiments with your vehicle**

| You want to... | One vehicle (`arena`) | Two vehicles (`fleet`) |
|---|---|---|
| Use your calibration and vehicle description | `--params`, `--spec` | `--params`, `--vehicle-spec` |
| Limit commands to your real thrusters | `--thrusters my_rov.yaml` | `--thrusters my_rov.yaml` |
| Add a water current | `--current-speed 0 0.3` | `--current-speed 0 0.3` |
| Add roll/pitch disturbance (tether, waves) | `--disturbance fitted --disturbance-scale 2` | `--disturbance fitted --disturbance-scale 2` |
| Test on MuJoCo, or both engines side by side | `--backend mujoco` / `both` | `--backend mujoco` / `both` |
| See whether calibration matters | `--compare-generic` | the default `calibrated` vs `generic` headline |
| Set the task | `--task station_keeping` (or `waypoint`, `docking`, `obstacle_avoidance`) | `--target-separation`, `--tolerance`, `--hold-steps`, `--episode-steps` |
| Set comm and sample size | none (a single vehicle has no link) | `--comm L2 L3a --beacon-interval 1 3 --seeds 5` |
| Train policies for your vehicle | built in | `train-fleet --params ...` |

**3. Know what the limits are:** [LIMITS.md](LIMITS.md) lists every known one, what it means for you, and what to do.

## Recent fixes (September 2026)

What changed, and why the numbers in this README differ from earlier versions. Details are in [PROJECT_STATUS.md](PROJECT_STATUS.md) #39–55.

**Fleet coordination: rebuilt (fleet v2)**
- **Claim C withdrawn (#46).** The v1 metric measured how far the *teammate* had moved since its last message, not anything about calibration. The same "calibrated advantage" appeared with both vehicles running identical physics.
- **What else v1 got wrong, and what v2 does instead (#47):**
  - v1 let one controller see both vehicles' true state. In v2, each vehicle sees only its own state plus its belief about its teammate.
  - v1's reward paid a vehicle to crash early and end a bad episode. In v2, a collision costs the rest of the episode's penalty.
  - v1 sent beacons 10 times a second. In v2, the interval is set by `beacon_interval_s` (1 s and 3 s in the studies).
  - v1 ignored most of its config file. In v2, every YAML setting reaches the simulation, including start states (`initial_conditions`), task and comm settings.
- **The fleet now uses the best converged calibration** as its "true" vehicle, not an unconverged quick fit.
- **Belief models were separated from policy training (#48).** Policies trained on a mixture of belief models made perfect information score *below* a calibrated model. Policies trained for their own belief model fix that.

**Physics and calibration**
- **Rotating-frame term (#39, #40).** Both physics backends were missing the rotating-frame (ω × v) term when converting the body-frame acceleration into the world-frame acceleration the engine integrates.
  - **MuJoCo:** fixed, and confirmed against real MuJoCo.
  - **HoloOcean:** fixed in code, plus a new check that HoloOcean's tick length matches the 0.1 s logs. Neither is confirmed in HoloOcean itself yet (it doesn't run on macOS), so HoloOcean numbers are provisional.
- **Roll and pitch: corrected claim, and a real fix (#50).** An earlier version of this README said calibration "improves every axis, including roll". That compared against default physics, which is very poor on roll and pitch. Measured against simply predicting "no motion", calibrated roll and pitch were *worse* than zero. Yaw is fine (skill +0.60).
  - **Cause:** the dataset takes thruster torques about a point 8.5 cm below the horizontal thrusters. Every forward or sideways thrust command therefore also logs a pitch or roll torque, which the real vehicle doesn't feel.
  - **Fix:** torques are now re-expressed about the thruster plane (`vehicle_spec.yaml` → `data_conventions`). Refitting with that correction lifts held-out roll/pitch skill from −0.49/−0.84 to −0.03/−0.10, with no cost on the other axes.
  - **What's left:** roll and pitch are still not *predictive*. Most of the real roll/pitch motion in this dataset isn't caused by the thrusters (tether, waves or contact), so this data can't calibrate it.
- **Does that matter for the fleet? No (#52).** Formation-holding success is unchanged, over 15,200 paired episodes, when realistic roll/pitch wobble is added (−0 to +2 points, not significant).
- **Roll/pitch disturbance, fitted to the real logs (#51).** Random roll/pitch torques whose size and timing reproduce the real vehicle's wobble on held-out logs, in MuJoCo and the stand-in. Turn it on with `disturbance:` in a fleet YAML; it's the first "noisy simulation" mode.
- **Calibration reports now judge each axis against "predict no motion"**, so they can no longer call an axis trustworthy just because default physics is worse. New calibration options: `--axis-scaling zero_rms`, `--exclude-axes`, `--prediction-horizon-s`, `--eval-workers` (parallel, bit-identical), and `--torque-reference-offset-m` (#50).
- **The "one evaluation takes 4 hours" stall is solved (#42, #49): the Mac was asleep.**
  - The new stall watchdog times every evaluation on three clocks and dumps stack traces when one runs long.
  - It caught 7 stalls in an overnight run. In each, the process computed for ~26 s while up to 67 min passed on the wall clock.
  - Every one lines up to the second with a macOS sleep/wake event.
  - Fix: every long-running script now keeps the machine awake by itself (`keep_awake.py`; see step 2 below).
- **Stand-in calibration is 3× faster** (47.5 s → 16.1 s per full-dataset evaluation). The model is unchanged; the evaluation now uses the equivalence-tested fast integrator.

**Using a lab's own data**: new PWM → thrust model, log ingestion and plain-language reports. See [Bring your own vehicle's logs](#bring-your-own-vehicles-logs).

**Easier open water, more log formats, fewer YAML edits (#57):**
- **The single-vehicle arena can run on MuJoCo too** (`strathe arena --backend mujoco`, or `both` side by side). It can add the roll/pitch disturbance fitted to real logs (`--disturbance fitted [--disturbance-scale 2]`). The same choices are on the web form.
- **ArduPilot logs are read directly:** DataFlash `.bin` from the SD card, and MAVLink `.tlog` from a ground station. There are templates, and world-frame velocity is rotated into the body frame for you.
- **Unknown water currents are estimated from the logs.** `strathe estimate-current` estimates each recording's current. `strathe calibrate --estimate-current` alternates fitting and estimating, and each recording gets its own current.
- **Fleet experiments without editing YAML.** `strathe fleet` flags and web fields cover:
  - comm levels and beacon interval
  - teammate models
  - the task
  - starting conditions and sample size
  - the physics engine and disturbance

  `--save-config` writes the resulting experiment file.
- **[LIMITS.md](LIMITS.md) lists every known limit and what to do about it.**

**A lab's own vehicle, end to end (#56).** A lab can now go from its raw logs to a calibrated simulator, a report, and single-vehicle and fleet experiments with its own vehicle:
- **Raw logs, one command:** `python strathe.py calibrate --ingest-spec my_rov.yaml` converts the logs, checks them, fits, checks which values the data can pin down, and writes the report. There are ready-to-edit templates in `calibration/templates/`.
- **ROS bags read directly:** ROS 1 `.bag`, and ROS 2 `.mcap` or `.db3`, with no ROS install.
- **Any thruster layout:** give each thruster's position, direction and whether it's reversed. Vehicles that can't push every axis work too (a 6-thruster BlueROV2 has no pitch control).
- **Thruster limits in the simulators:** `thrusters:` limits every command to what the vehicle's real thrusters can deliver, in the single-vehicle arena, the fleet (both engines) and training. A BlueROV2 Heavy makes about 18 N·m of pitch torque, not the 30 N·m any policy could command before.
- **One vehicle:** `python strathe.py arena` trains and tests controllers on a task with your calibration, and can compare training on calibrated vs default physics.
- **Two vehicles:** fleet experiments take your calibration and vehicle description, and `python strathe.py train-fleet` trains fleet policies for your vehicle.
- **The report now says which values your data can't pin down, in plain words.** The check uses the same objective the fit used.
- **Bugs the end-to-end stress test found and fixed:**
  - The data check crashed on every converted log, because it needed position columns the conversion doesn't produce.
  - `strathe calibrate` always assumed 10 Hz. On 20 Hz logs that made held-out error 5–11% worse on four axes.
  - Resampling interpolated orientation linearly. The reference data has 239 orientation jumps over 20° within 0.1 s, and linear interpolation produced invalid rotations across them. It now uses spherical interpolation.
  - Resampling time grids built from Unix timestamps drifted by microseconds per sample.

**Water current (#55)**: open-water labs can now add a current, off by default and configurable. You set speed and direction (fixed or drawn per episode), vertical flow, slow fluctuation, and whether teammate models know it. It's available in the fleet YAML, `strathe fleet` flags, the web page, the single-vehicle environment, and calibration from logs recorded in a known current. See [the fleet workflow](#the-fleet-coordination-workflow-fleet-v2) and [Bring your own vehicle's logs](#bring-your-own-vehicles-logs).

**Onboarding (#53)**: one command, `python strathe.py`, replaces the separate terminal steps, and `python strathe.py web` opens a local web page to run and read everything. See [Quick start](#quick-start).

**Tests and CI**
- A fake `mujoco` module in one test file leaked into every later file. Test results depended on file order.
- One MuJoCo test file never actually ran; it passed values between tests in a way pytest doesn't support.
- CI now downloads the real dataset, pinned to a fixed commit, so the real-data tests run on every push.
- CI covers the fleet suite and a smoke run of the lab entry point. It runs on Windows, and several Windows-only failures were fixed (file encoding, permissions).
- 396 tests in `calibration/tests/` plus 142 in the fleet suite, all passing.

## Quick start

After `pip install -r requirements.txt`, everything goes through one command, run from any folder:

```bash
python strathe.py check
```

`check` confirms this machine can run Strathe and says what to fix if not. The other commands:

| Command | What it does |
|---|---|
| `python strathe.py web` | Opens a local web page in your browser. From it you can calibrate, run single-vehicle and fleet experiments, train fleet policies, watch jobs run, and read the reports. |
| `python strathe.py calibrate --ingest-spec my_rov.yaml --name my_rov` | From your raw logs (CSV, ROS bags, ArduPilot `.bin` or MAVLink `.tlog`; forces or thruster PWM): converts, checks, calibrates, checks which values the data can pin down, and writes the plain-language report. Add `--estimate-current` for open-water logs in an unknown current. Templates: `calibration/templates/`. |
| `python strathe.py calibrate --data <logs folder> --name my_rov` | The same, for logs already in Strathe's CSV layout. |
| `python strathe.py arena --params <your calibrated_parameters.yaml> --task station_keeping --compare-generic` | One vehicle: trains controllers on your calibrated vehicle and tests them on a task (reach a point, hold station, dock, avoid an obstacle). Add `--backend mujoco` (or `both`), `--disturbance fitted`, `--current-speed ...` and `--thrusters ...` as needed. |
| `python strathe.py fleet` | Two vehicles: runs the shipped fleet experiment. Add `--config my_experiment.yaml` for your own. `--params`, `--vehicle-spec`, `--thrusters` and `--current-speed` use your vehicle, its thrusters and a current. `--comm`, `--beacon-interval`, `--beliefs`, `--target-separation`, `--seeds`, `--backend`, `--disturbance` (and more; `--help`) change the experiment without editing YAML. |
| `python strathe.py estimate-current --data <logs>` | Open water: estimates the current in each recording. Calibration then uses each recording's own current. |
| `python strathe.py train-fleet --params <your calibration> --out my_policy` | Trains a fleet policy for your vehicle (~40 min). |
| `python strathe.py ingest --spec my_vehicle.yaml --out <folder>` | Only converts your own logs into Strathe's layout. |
| `python strathe.py report --result <output folder> --data <logs>` | Writes the plain-language report for an existing calibration. |

The web page has a form for each of these. It runs on this computer only: it can't be reached from the network, and it only starts Strathe's own commands. Long jobs keep the computer awake by themselves. The individual scripts described below still work for specialists.

## Setup

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. HoloOcean (the underlying simulator)

HoloOcean can't be distributed via `pip`/PyPI directly, due to the Unreal
Engine EULA — it requires a one-time account-linking step first:

1. **Link your GitHub account to an Epic Games account** and accept the
   Unreal Engine EULA — follow
   [Epic's official steps](https://www.unrealengine.com/en-US/ue-on-github).
   **Accept the email invitation from Unreal Engine** — easy to miss, and
   the next step fails silently without it.
2. **Clone and install the HoloOcean client library:**
   ```bash
   git clone git@github.com:byu-holoocean/HoloOcean.git holoocean
   cd holoocean/client
   pip install .
   ```
   (If cloning gives a "Page Not Found" error, step 1's account link hasn't
   gone through yet.)
3. **Install the Ocean world package** (several GB, one-time download):
   ```bash
   python -c "import holoocean; holoocean.install('Ocean')"
   ```

Full official docs:
[HoloOcean Installation Guide](https://byu-holoocean.github.io/holoocean-docs/v2.0.0/usage/installation.html).

MuJoCo (used for the multi-agent fleet work below) needs no manual install step — it's a normal pip package, already pinned in `requirements.txt`.

### 3. Use the calibrated model directly

```bash
python -c "
from calibration.dynamics_model import CalibrationParameters
import yaml

with open('calibration_output_holoocean_coriolis/calibrated_parameters.yaml') as f:
   params = CalibrationParameters(**yaml.safe_load(f)['parameters'])
print(params)
"
```

*(Confirmed working as-written, run from the project root.)*

### 4. Try the fast demo pipeline

A single, watchable command — pre-flight data-quality checks, a fast
calibration fit, and a real Gym-environment smoke test, end to end:

```bash
cd calibration
python run_full_pipeline.py \
   --spec vehicle_spec.yaml \
   --data-dir <path-to-logs> \
   --output-dir <output-dir> \
   --sampling-rate-hz 10.0 \
   --max-train-runs 3 \
   --max-nfev 5
```

Optionally pick a task for the delivered environment with `--task waypoint`
(default), `--task station_keeping`, `--task docking`, or
`--task obstacle_avoidance` (this last one has a larger, 15-dim
observation space rather than 11 - see PROJECT_STATUS.md finding 36).
This is a fast, truncated demo for confirming the mechanism works end
to end — not the source of the reportable numbers below. See the real
workflow below for that.

## The real (non-demo) workflow: single-vehicle calibration

The pipeline above is a fast, truncated demo. The real pipeline, run
against the full dataset and real HoloOcean physics, is a sequence of
separate scripts, run in order:

1. **Pre-flight data-quality check** (always run this first):
   ```bash
   cd calibration
   python preflight_check.py --data-dir <path-to-logs> --spec vehicle_spec.yaml
   ```
   Validates sampling rate, Coriolis-relevant excitation, schema
   (required columns), structural data integrity (empty/malformed files,
   NaN/Inf), and physical/units plausibility (e.g. catches a
   degrees-vs-radians unit mistake directly) on the REAL, full dataset
   before committing real compute time to a calibration that might not
   succeed.

2. **Stand-in calibration.** This is what `strathe.py calibrate` runs. One evaluation of the full dataset takes about 16 s on one M1 core, or about 5 s with `--eval-workers 6`; a fit takes a few hundred evaluations.
   ```bash
   python run_calibration.py --data-dir <path> --spec vehicle_spec.yaml \
      --sampling-rate-hz 10.0 --heldout-dates 16-02-2024 19-02-2024 \
      --output-dir <output-dir> --max-nfev 60 --eval-workers 6 --axis-scaling zero_rms
   ```
   - **Torque convention.** Logged torques are re-expressed about the point set in `vehicle_spec.yaml` → `data_conventions` (the thruster plane for the BlueROV2 dataset, PROJECT_STATUS.md #50). Override it with `--torque-reference-offset-m X Y Z`.
   - **Output.** `metrics.json` records each axis's held-out skill against predicting no motion, and the objective settings used.
   - **If a run stops early** (interrupted, or out of evaluations), `python finalize_checkpoint.py --checkpoint <output-dir>/checkpoint.yaml --data-dir <path> --heldout-dates ...` turns its best parameters into a normal, clearly-marked "not converged" result.
   **The machine is kept awake automatically while the run lasts.** This uses
   `caffeinate` on macOS, `SetThreadExecutionState` on Windows and
   `systemd-inhibit` on Linux. Without it, a Mac sleeps and evaluations that
   take ~16 s stretch to over an hour (PROJECT_STATUS.md #49).
   - The same applies to HoloOcean calibration, the demo pipeline, training
     and fleet studies.
   - On a laptop, keep the lid open, because closing it still sleeps the machine.
   - Set `STRATHE_ALLOW_SLEEP=1` to turn this off.
   - If it can't block sleep, the run prints a warning and carries on.

   Every run writes `eval_timing.csv`, which has each
   evaluation's time on three clocks. If an evaluation runs long, the run also
   writes `stall_log.txt`, which records whether the machine slept, the
   process waited, or it was really computing.

   Add --parallel (optionally --max-workers N) to the stand-in step above to run multiple restarts across real CPU cores instead of sequentially - confirmed real speedup (roughly 4 restarts in the time 1 would normally take) on real, smaller data. Honest caveat: checkpointing is disabled in this mode (a real, deliberate tradeoff - see PROJECT_STATUS.md finding 34), and the combination of --parallel with the FULL, untruncated dataset has not yet been confirmed to finish in a reasonable time - recommend testing on a smaller subset first, or preferring the default (sequential) mode for full-dataset runs until that has been verified.

3. **HoloOcean refinement** (slow, hours — real, confirmed cost ~124s per
   evaluation on this dataset; checkpointed and resumable):
   ```bash
   python run_calibration_holoocean.py --data-dir <path> --spec vehicle_spec.yaml \
      --sampling-rate-hz 10.0 --resume-from-checkpoint <path>/checkpoint.yaml \
      --n-steps-per-run 150 --max-nfev 15 --output-dir <output-dir>
   ```
   If resuming from a checkpoint created before a `vehicle_spec.yaml` bounds
   change, first run `check_and_fix_checkpoint_bounds.py` against it — an
   out-of-date checkpoint can hold values now outside the current bounds,
   which will crash the resume. Add `--no-dof-weighting` to disable the
   custom per-DOF weighting described in the Results section below, for a
   standard, unweighted objective.

4. **Real validation (Claim A)** against real HoloOcean physics:
   ```bash
   python replay_comparison.py --data-dir <path> --spec vehicle_spec.yaml \
      --sampling-rate-hz 10.0 --calibrated-params <output-dir>/calibrated_parameters.yaml \
      --result-type final --output-dir <output-dir>
   ```

5. **Real multi-seed PPO training (Claim B)** — optionally pick a task
   (`waypoint` default, `station_keeping`, `docking`, or
   `obstacle_avoidance` - note: obstacle_avoidance generates its own
   randomized goal each episode and will print a warning that the
   real-data-grounded waypoint below is computed but not used for
   this task):
   ```bash
   python train_multiseed.py --data-dir <path> --spec vehicle_spec.yaml \
      --calibrated-params <output-dir>/calibrated_parameters.yaml \
      --result-type final --n-seeds 4 --task waypoint --algorithm ppo --output-dir <output-dir>
   ```

6. **Optional — deeper diagnostics**, once you have a real calibrated
   result:
   - **Per-DOF real-vs-simulated plots and force-breakdown charts** (fast
   — a few minutes, no Jacobian evaluation):
   ```bash
   python diagnose_calibration.py --data-dir <path> --spec vehicle_spec.yaml \
         --calibrated-params <output-dir>/calibrated_parameters.yaml \
         --sampling-rate-hz 10.0 --heldout-dates 16-02-2024 19-02-2024 \
         --skip-jacobian --plot --plot-output-dir <output-dir>/plots
   ```
   Shows exactly WHERE in a trajectory the fit breaks down, per DOF —
   not just the aggregate RMSE number.
   - **Automated parameter identifiability report** (real cost ~15-17 min
   on the full dataset — condition number, correlated pairs,
   multi-parameter flat directions):
   ```bash
   python automated_identifiability_report.py --data-dir <path> --spec vehicle_spec.yaml \
         --sampling-rate-hz 10.0 --calibrated-params <output-dir>/calibrated_parameters.yaml
   ```

See `PROJECT_STATUS.md` for the full history of real runs, real bugs found
and fixed along the way, and the honest per-DOF results.

## Bring your own vehicle's logs

The calibration pipeline reads one canonical CSV layout. The shortest route
from a lab's own logs is one command:
```bash
cp calibration/templates/ingest_rosbag_mavros.yaml my_logs/my_rov.yaml   # or a CSV template; edit the names
python strathe.py calibrate --ingest-spec my_logs/my_rov.yaml --name my_rov
```
It runs five steps and stops at the first that fails, saying what to fix:
1. converts the logs
2. checks them
3. calibrates
4. checks which values the data can pin down
5. writes the report

If the fit runs out of its evaluation budget (`--max-nfev`) before converging, the best result so far is kept, marked "not converged", and the remaining steps still run. The report says so.

The web page's calibration form does the same. Each piece is below.

**1. Thruster PWM → force** (`calibration/thruster_model.py`). Most logs record
the PWM each ESC was sent, not force. `ThrusterConfig.wrench(pwm)` applies
motor directions, a thrust curve and the thruster allocation matrix.
- **BlueROV2 Heavy preset** (8 × T200): extracted from the public dataset, which
  logs all three stages. It reproduces the dataset's own forces from raw PWM on
  all 183,555 rows. Its thrust curve (+47.9 N / −37.2 N) matches Blue Robotics'
  published T200 figures at ~15 V (a 4S battery), and `voltage:` rescales it to
  12–20 V.
- **Any other vehicle:** give a `layout:`, one entry per thruster with:
  - its position
  - the way it pushes
  - whether it's reversed (ArduSub `MOT_n_DIRECTION`)

  Alternatively, give an allocation-matrix file. Use the T200 curve or your own
  curve file. `templates/ingest_csv_pwm_custom_thrusters.yaml` writes the Heavy
  this way, and a test checks it matches the preset exactly.
- **Vehicles that can't push every axis** work too: ingestion names the axes
  (a 6-thruster BlueROV2 can't pitch on command).

**2. Log ingestion** (`calibration/data_ingestion.py`). A short YAML spec says
where each quantity lives in your logs, in what units and frame:
- velocity, orientation, and force or PWM
- ROS FLU/ENU frames are converted to Fossen FRD/NED

Start from a template in [`calibration/templates/`](calibration/templates/README.md):

| Your logs | Template |
|---|---|
| CSV with forces | `ingest_csv_forces.yaml` |
| CSV with BlueROV2 Heavy PWM | `ingest_csv_pwm_bluerov2_heavy.yaml` |
| CSV with any vehicle's PWM | `ingest_csv_pwm_custom_thrusters.yaml` |
| ROS bags | `ingest_rosbag_mavros.yaml` |
| ArduSub / ArduPilot DataFlash logs (`.bin`) | `ingest_ardupilot_bin.yaml` |
| MAVLink telemetry logs (`.tlog`) | `ingest_mavlink_tlog.yaml` |

**ArduPilot logs are read directly** (`ardupilot_reader.py`, using pymavlink):
- DataFlash `.bin`/`.log` from the autopilot's SD card, and MAVLink `.tlog` from a ground station such as QGroundControl.
- Columns are `<MESSAGE>.<field>`, e.g. `RCOU.C1` or `ATT.Roll`. Add `[i]` for one of several sensors, e.g. `IMU[0].GyrX`.
- ArduPilot logs velocity in the world frame; `linear_frame: world` rotates it into the body frame with the logged attitude.
- Corrupt or cut-off records (common when power is cut) are skipped and counted.
- Tested on all 39 reference runs written as damaged `.bin` and `.tlog` logs: every sample recovered (#57).
- The velocity is only as good as the vehicle's DVL or positioning system. Without one, ArduSub's velocity is dead reckoning. `.bin` logs without GPS have no calendar date, so date them from the filename (templates, LIMITS.md).

**ROS bags are read directly** (`rosbag_reader.py`, no ROS install needed):
- ROS 1 `.bag`, and ROS 2 bag folders with `.mcap` or `.db3`.
- Columns are `<topic>.<field>`, e.g. `/mavros/rc/out.channels[0]`.
- Topics at different rates are put on one time grid:
  - velocities are interpolated
  - orientation uses spherical interpolation
  - PWM holds the last command
- Timestamps come from message headers.

Tested by writing all 39 reference runs as MAVROS-style ROS 1 bags. Converting
them with the shipped template recovers:
- velocities exactly
- forces to 0.001 N
- orientation to 1e-13°
- the calibration's held-out error, unchanged

It refuses data that would silently corrupt a calibration:
- deg/s declared as rad/s
- non-unit quaternions
- non-monotonic or irregular timestamps (unless you ask it to resample)
- files at different rates
- PWM values that aren't ESC microseconds
- missing topics or fields

It warns about time gaps and orientation jumps over 45° between samples. `ingest_report.json` records:
- the measured sampling rate
- how torques should be referenced
- time gaps
- anything dropped

`strathe calibrate` uses the rate and torque convention automatically. Before
#56 it assumed 10 Hz whatever the logs were.
To only convert: `python strathe.py ingest --spec my_rov.yaml --out <folder>`.

**3. A report written for non-specialists** (`calibration/calibration_report.py`). `strathe calibrate` writes it automatically, including the check of which values the data can pin down (`automated_identifiability_report.py --result`, run with the same objective settings as the fit). To write it for an existing result:
```bash
python strathe.py report --result calibration_output_x --data <logs>
# the check of what the data can pin down, for a result made before it was automatic:
cd calibration && python automated_identifiability_report.py --result ../calibration_output_x --data-dir <logs> --spec vehicle_spec.yaml --workers 8
```
It writes `calibration_report.md`, with:
- a bottom line
- a verdict for each motion axis (trust it / use with care / don't rely on it), with the reason
- parameters stuck at their limits
- physically implausible values
- values your data can't tell apart: pairs that trade off, and groups that shift together
- concrete next steps

Example: `calibration_output_10hz_coriolis/calibration_report.md`.

**4. Logs recorded in open water.** The reference data comes from a tank, so still water is the default. If your recordings were made in a known current, give it to the calibration. Otherwise the whole current shows up as unexplained error and skews the drag values. In synthetic tests with a 0.2 m/s current, ignoring it left 0.2 m/s of error and shifted the drag values by 1–5%, depending on the vehicle; supplying it recovered them exactly (PROJECT_STATUS.md #55).
```bash
python strathe.py calibrate --data <logs> --current-world-mps 0.15 -0.05 0   # vx vy vz, in the logs' world frame
```
Or set `environment.current_*` in your `vehicle_spec.yaml`.

**Current unknown? Estimate it** (#57):
```bash
python strathe.py calibrate --data <logs> --estimate-current     # or --ingest-spec ...; also a web checkbox
python strathe.py estimate-current --data <logs> --params <a calibration>   # only the estimate
```
How `--estimate-current` works:
- It calibrates in still water, estimates each recording's constant current with that calibration, then calibrates again with the currents (two rounds by default).
- Each recording gets its own current, written to `currents.csv`.
- A `currents.csv` in the logs folder is used automatically, so you can also fill one in from ADCP readings.
- Each estimate is checked by estimating it separately on the first and second half of the recording. If the halves disagree, the current changed during the recording, and `current_estimates.md` says so in plain words.

Tested on six simulated open-water dives, each in a different current from 7 to 30 cm/s:
- It recovered every current to within about 1 cm/s and 2°.
- Held-out error was 14× lower than a calibration that ignores the currents (0.007 vs 0.097).
- Parameters came within 0.1% of the truth (median).

Estimates made with the uncalibrated defaults are flagged as rough, because a drag error along the vehicle's path looks like a current. That's why `--estimate-current` calibrates first.

**Limit:** one constant current per recording. A current that changes within a recording, or varies from place to place, isn't modelled. Keep open-water recordings to a few minutes, and include turns: a vehicle that holds one heading can't tell a current from a drag error along it. See [LIMITS.md](LIMITS.md).

## One vehicle: the single-vehicle arena

```bash
python strathe.py arena --params calibration_output_my_rov/calibrated_parameters.yaml \
    --task station_keeping --compare-generic --current-speed 0 0.3 --thrusters my_logs/my_rov.yaml
```

**What it does:**
- Trains PPO controllers on your calibrated vehicle for one of four tasks:
  - reach a point
  - reach a point and hold station
  - dock slowly and level
  - reach a point around an obstacle
- Tests them on that vehicle, with two references that need no training: a
  simple hand-written controller and random commands.
- With `--compare-generic`, also trains on the default, uncalibrated physics.
  The report then says whether the calibration mattered for this task, as a
  paired difference with a 95% CI.

**Options and outputs:**
- Every controller sees the same episodes (same targets, currents and disturbances).
- **Physics engine** for the tests: `--backend stand_in` (fast, default), `mujoco`, or `both`. With `both`, the report shows each engine and how closely they agree. Policies always train on the stand-in.
- **Roll/pitch disturbance:** `--disturbance fitted` adds the random roll/pitch torques fitted to the real BlueROV2 logs (tether and waves). `--disturbance-scale 2` makes them twice as rough. Your own fit file also works.
- A water current and thruster limits are optional, as in the fleet.
- `--policy my_policy.zip` evaluates your own policy instead of training.
- `report.md` is plain language. `results.json` and `episodes.csv` hold the
  numbers, and the trained policies are saved.
- There's a form for it on the web page.
- The underlying script is `calibration/run_arena_experiment.py`.

**What it found on a test vehicle** (PROJECT_STATUS.md #56): a simulated 6-thruster vehicle with no pitch authority, heavily damped, in currents up to 0.3 m/s.
- **Reaching a point:** after 300k training steps, policies trained on its calibration succeeded 96% of the time, and policies trained on default physics 100%.
- **After 100k steps:** 10% vs 75%. A heavily damped vehicle is slow to learn on, and more training (`--timesteps`) closes most of the gap.
- **So calibration doesn't always matter.** For a forgiving single-vehicle task it may not, which is what `--compare-generic` is for. The fleet task, where each vehicle predicts its teammate, is where it did matter for the same vehicle (below).

## The fleet coordination workflow (fleet v2)

**The setup, in one paragraph.**
- **Two simulated BlueROV2s must hold 2 m ± 0.2 m apart** for 2 s within a 20 s episode, without coming within 0.5 m.
- **Both run the same "true" physics**: `calibrated_parameters_fleet_truth.yaml`, the torque-arm-corrected calibration (armfix; see Single-Vehicle Calibration Results).
- **Each broadcasts its state** (position, orientation, velocity, last thrust) every `beacon_interval_s` over an acoustic channel that drops packets.
- **Between beacons, each vehicle predicts its teammate with a belief model**, which is the only thing varied between conditions:
  - `calibrated`: physics with the true parameters
  - `generic`: physics with default parameters
  - interpolations between those two
  - `constant_velocity`, `last_beacon` (no prediction), `oracle` (perfect knowledge)
- **One shared policy drives each vehicle** from that vehicle's own state plus its belief (13 numbers in, 6 thrusts out).
- **Comparisons are paired:** every condition sees the same start states and the same lost packets.

```bash
cd calibration
python run_fleet_experiment.py                       # shipped experiment (default_experiment.yaml), a few minutes
cp default_experiment.yaml my_experiment.yaml        # then edit: comm, beacon interval, beliefs, task, start states
python run_fleet_experiment.py --config my_experiment.yaml
```
Or change the experiment with flags, no editing. For example:
```bash
python strathe.py fleet --comm L2 L3a --beacon-interval 1 3 --beliefs calibrated generic last_beacon \
    --target-separation 3 --hold-steps 30 --seeds 3 --backend both --disturbance fitted
python strathe.py fleet ... --save-config my_experiment.yaml     # write the file instead of running it
```
Settings that make an impossible experiment are refused before anything runs, e.g. a hold time longer than the episode, or a headline pair that isn't in the teammate models. The web page's fleet form has the same fields.

| Level | Packet loss | Bandwidth |
|---|---|---|
| L0 | 0% | unlimited |
| L1 | 10% | unlimited |
| L2 | 30% | unlimited |
| L3a | 60% | unlimited |
| L3b | 60% | 1 byte short of a beacon |

What the run produces:
- `summary_report.md`: a plain-language headline with 95% CIs, every condition, and limits
- `episodes.csv`, `metrics.json`
- `metadata.json`: git commit and **whether the code had uncommitted changes**
- `experiment.yaml`: reruns the exact experiment
- plots

Every setting is documented in [`calibration/CUSTOM_EXPERIMENT.md`](calibration/CUSTOM_EXPERIMENT.md), and the acceptance spec for handing it to a lab is in [`calibration/LAB_USER_JOURNEY.md`](calibration/LAB_USER_JOURNEY.md).

Training your own per-vehicle policy (~40 min on an M1 for 3M steps):
```bash
python train_fleet_policy.py --timesteps 3000000 --seed 4 --out my_policy   # --belief <model> to specialize
python validate_fleet_policy.py my_policy.zip
```
Shipped policies:
- `fleet_policy_seed1-3.zip`: trained on a mixture of belief models.
- `specialized_policies/fleet_policy_<belief>_seed1-2.zip`: each trained for one belief model (see results).
- `disturbance_policies/fleet_policy_dist_seed1-2.zip`: trained with the fitted roll/pitch disturbance on.

Add realistic roll/pitch wobble to any experiment with `disturbance: {roll_pitch: fitted}` in its YAML (CUSTOM_EXPERIMENT.md section 8).

**Your own vehicle and its thrusters.** The experiment YAML can set:
- `truth_params:` your calibration
- `vehicle_spec:` your vehicle description
- `thrusters:` its real thrusters (`bluerov2_heavy`, or your ingestion spec file)

The same settings are flags on `strathe fleet` (`--params`, `--vehicle-spec`,
`--thrusters`) and fields on the web page. Train policies for your vehicle with:
```bash
python strathe.py train-fleet --params <your calibration> --thrusters my_rov.yaml --out my_policy
```
Thruster limits change what a command delivers: a command is scaled down until
no thruster exceeds its curve, and axes the thrusters can't push get nothing.
The beacons carry the delivered force. See CUSTOM_EXPERIMENT.md sections 10–11.

Tested on a vehicle the pipeline had never seen (PROJECT_STATUS.md #56):
- The vehicle: a simulated 6-thruster layout with no pitch authority, logged as PWM only.
- Calibration: one `strathe calibrate` command recovered all 22 parameters within 4%.
- A fleet policy trained for it with `strathe train-fleet`.
- A fleet experiment with its thrusters and a current, on both engines:
  - The calibrated teammate model beat the generic one by +1 to +10 points, +10 (+3, +17) at L3a with 1 s beacons.
  - The ordering matches the reference study: oracle > calibrated ≥ last beacon ≥ generic.

**Open water: add a water current** (off by default). In the experiment YAML:
```yaml
current: {speed_mps: [0.0, 0.3], direction_deg: [0, 360]}   # a speed and heading drawn per episode
```
Options:
- speed and direction, fixed or drawn per episode
- vertical flow
- slow fluctuation
- whether teammate models know the current

The same options are available as flags on `python strathe.py fleet --current-speed 0.2 ...`, and as fields on the web page's fleet form. See CUSTOM_EXPERIMENT.md section 9. `current_policies/` holds two policies trained in random currents up to 0.3 m/s. The single-vehicle environment takes the same setting: `Arena(..., current=CurrentConfig(...))`.

What a current does to formation-holding (`run_current_study.py`, 34,800 paired episodes, PROJECT_STATUS.md #55):
- **Little, directly.** With 1 s beacons, nothing changes up to 0.3 m/s. With lossy 3 s beacons, 0.3 m/s costs 5 points.
- **The teammate model needs to know the current.** At 0.2–0.3 m/s, a calibrated model that knows it beats one that doesn't by 6–13 points. Once aware, it beats no prediction by +8 at 0.3 m/s (vs +3 in still water).
- **Training policies in currents didn't help;** the still-water policies already cope.

Reproduce the studies below: `python run_fleet_experiment.py --config fleet_study.yaml` (~30 min), `fleet_study_alt_truth.yaml`, `fleet_study_mujoco.yaml`, then `python analyze_fleet_study.py results/<dir>` for per-seed checks.

The retired fleet v1 setup still runs (`run_fleet_experiment_v1.py`) for reproducing old results only.

## What's in this repo

- **`calibration/`** — the full calibration pipeline: data loading, the
physics model, the optimizer, the HoloOcean integration, the Gym/`Arena`
RL training environment (four tasks: `waypoint`, `station_keeping`,
`docking`, `obstacle_avoidance`), three RL training algorithms
(`--algorithm ppo` (default), `sac`, or `td3` - all real, rigorously
validated against the same locked criteria, see PROJECT_STATUS.md
finding 37), diagnostic and reporting tools, and a real, automated
test suite (`calibration/tests/`, CI-verified on every push).
- **`calibration/comm_channel.py`** — a standalone, vehicle-agnostic underwater acoustic communication model (latency, distance-conditioned packet loss, per-tick bandwidth budget). Zero physics dependency by construction — any vehicle can sit on either end.
- **`calibration/fleet_env.py`, `fleet_belief.py`, `fleet_vec_env.py`** — fleet v2: the two-vehicle simulation with per-vehicle observations, the teammate belief models, and the wrapper that trains one shared per-vehicle policy with stock stable-baselines3.
- **`calibration/run_fleet_experiment.py`, `fleet_experiment_schema.py`, `fleet_eval.py`, `fleet_stats.py`** — the lab entry point: YAML config, paired evaluation across processes, statistics (Wilson CIs, paired bootstrap, McNemar, dose-response), plain-language report.
- **`calibration/train_fleet_policy.py`, `validate_fleet_policy.py`, `analyze_fleet_study.py`, `run_specialized_study.py`** — training, the policy contract check, and the study analyses.
- **`calibration/fast_dynamics.py`** — the same 6-DOF equations as `ArenaCore`, 6.4x faster (equivalence-tested), which makes fleet training practical. Stand-in calibration uses it too, which makes it 3× faster per evaluation.
- **`strathe.py`, `calibration/strathe_cli.py`, `strathe_web.py`** — the one-command entry point and the local web page (see Quick start).
- **`calibration/current.py`, `run_current_study.py`, `current_policies/`** — the configurable water current, the study of its effect on formation-holding, and two policies trained in random currents.
- **`calibration/disturbance.py`, `fit_rollpitch_disturbance.py`, `disturbances/`** — the roll/pitch disturbance model, its fitting script, and the shipped fit for the fleet truth vehicle.
- **`calibration/run_disturbance_study.py`, `disturbance_policies/`** — the study showing roll/pitch doesn't affect formation-holding, and the two policies trained with the disturbance.
- **`calibration/keep_awake.py`** — stops the machine sleeping during long runs (macOS, Windows, Linux).
- **`calibration/stall_watchdog.py`** — times every calibration evaluation on three clocks and dumps stack traces for a stalled one (`eval_timing.csv`, `stall_log.txt`).
- **`calibration/verify_holoocean_frame_conventions.py`, `diagnose_rotational_weakness.py`** — the HoloOcean frame check (needs a HoloOcean machine), and a per-axis comparison of held-out results across calibrations.
- **`calibration/mujoco_backend.py`** — the MuJoCo physics backend (used as an independent "truth" engine for fleet evaluation).
- **`calibration/thruster_model.py`, `data_ingestion.py`, `rosbag_reader.py`, `calibration_report.py`, `templates/`** — PWM → force for any thruster layout (and thruster limits for the simulators), lab log ingestion from CSV or ROS bags, plain-language calibration reports, and ready-to-edit ingestion specs (see "Bring your own vehicle's logs").
- **`calibration/run_arena_experiment.py`** — the single-vehicle experiment on a lab's own calibrated vehicle, on the stand-in or MuJoCo, with optional disturbance, current and thruster limits (`strathe arena`).
- **`calibration/ardupilot_reader.py`** — reads ArduPilot DataFlash `.bin` and MAVLink `.tlog` logs for ingestion.
- **`calibration/estimate_current.py`** — estimates each open-water recording's water current from the logs (`strathe estimate-current`, `calibrate --estimate-current`).
- **[`LIMITS.md`](LIMITS.md)** — every known limit, what it means, and what to do about it.
- Fleet v1 (`two_agent_*.py`, `evaluate_comm_degradation_curve*.py`, `run_fleet_experiment_v1.py`) — retired, kept for reproducing old results.
- **`calibration_output_*/`** — saved, ready-to-use calibrated parameter
sets (`calibrated_parameters.yaml` + `metrics.json` in each).
- **`PROJECT_STATUS.md`** — the full research log: every finding, every bug
found and fixed, every validation check, in chronological order. This is
the detailed record for anyone who wants to verify the methodology or
understand exactly how a specific number was produced. Not required
reading to just use the model.

## Fleet coordination results

**Main study** (`fleet_study.yaml`):
- **Episodes:** 3 training seeds × 8 belief models × 4 loss levels × 2 beacon intervals × 200 paired episodes.
- **Setup:** true physics = the torque-arm-corrected calibration (armfix), stand-in engine.
- **Outputs:** full tables and plots in `calibration/results/fleet_v2_belief_to_success/`, per-seed checks in `extended_analysis.md`.

Formation success rate by teammate-prediction model (policies trained on a mixture of belief models):

| Comm | calibrated | generic | last beacon (no prediction) | oracle (perfect) | calibrated − generic (95% CI) |
|---|---|---|---|---|---|
| L0, beacon 3 s | 97% | 76% | 98% | 100% | **+21 (+17, +25)** |
| L1, beacon 3 s | 94% | 70% | 94% | 100% | **+25 (+20, +29)** |
| L2, beacon 1 s | 100% | 98% | 100% | 100% | +2 (+1, +2) |
| L2, beacon 3 s | 86% | 64% | 84% | 100% | **+22 (+17, +26)** |
| L3a, beacon 1 s | 98% | 84% | 96% | 100% | +13 (+10, +16) |
| L3a, beacon 3 s | 68% | 52% | 65% | 93% | +16 (+12, +21) |

(Every model scores 98–100% at L0 and L1 with 1 s beacons.)

What it shows:
1. **Calibration matters when a vehicle predicts its teammate with physics.** Calibrated beats generic by +13 to +25 points wherever the task isn't nearly solved.
   - Per training seed, the gain is +10 to +38 points in those conditions, significant for every seed except one at L3a with 3 s beacons.
   - Success falls steadily as the model's parameters move from true toward generic: all 8 dose-response slopes are negative with CIs excluding zero.
2. **An uncalibrated model is worse than none.** Generic loses to last-beacon by 12 to 25 points wherever beacons are 3 s apart or packet loss is heavy.
3. **Calibrated physics prediction now ties using the last message** (−1 to +3 points, no significant difference). With the previous truth vehicle it trailed by 2–7 points.
4. **Better information helps.** Perfect information beats calibrated by +3 to +24 points, most where comms are worst.

These results use the torque-arm-corrected truth vehicle (armfix), with all policies retrained on it; PROJECT_STATUS.md #54 compares them with the previous truth.

**Follow-up: each belief model with a policy trained for it** (`run_specialized_study.py`, 2 seeds each, 12,800 paired episodes, `calibration/results/fleet_v2_specialized/`):

| Comm | calibrated | generic | last beacon | oracle |
|---|---|---|---|---|
| L0, beacon 3 s | 40% | 72% | 100% | 100% |
| L1, beacon 3 s | 58% | 73% | 100% | 100% |
| L2, beacon 3 s | 74% | 70% | 95% | 100% |
| L3a, beacon 1 s | 98% | 88% | 99% | 100% |
| L3a, beacon 3 s | 66% | 57% | 81% | 97% |

(Every belief model scores 100% at L0, L1 and L2 with 1 s beacons.)

- **Policies trained for the calibrated model are unreliable with this truth vehicle.** At 3 s beacons, both study seeds often fail to settle (timeouts, not collisions).
  - Two more seeds, trained the same way, reached 98% and 57% at L0 with 3 s beacons (`replication_calibrated_seeds3-4.md`). That's large seed-to-seed variance, not a systematic failure.
  - Policies trained on a mixture of models (the main study above) are consistent across seeds, so the main study is the basis for the calibration claims.
- **Where it can be compared, calibrated still beats generic**: +10 at L3a with 1 s beacons and +9 at L3a with 3 s, both significant.
- **Perfect information beats the last beacon** by +5 (L2, 3 s) and +16 (L3a, 3 s); last-beacon-trained policies are strong in this setup.

**Robustness:**
- **With a second "true vehicle"** (the least bound-limited calibration): calibrated beats generic by +13 to +21 points in 3 of 4 conditions, with no difference in the fourth (L2 with 1 s beacons, −1, not significant).
- **With MuJoCo as the truth engine:** calibrated beats generic by +15 (L3a, 1 s beacons) and +14 (L3a, 3 s), against +13 and +16 with the stand-in.

**Limits:**
- Simulation only, 2 vehicles, formation hold, BlueROV2.
- In simulation `calibrated` equals the simulated vehicle, so this measures the value of model accuracy. How close a real calibration gets to a real vehicle is the single-vehicle result below.
- Success rates range from about 40% to 100%, depending mostly on comm conditions, the belief model, and the policy.

## Single-Vehicle Calibration Results

**Methodology:** real BlueROV2 mission logs (public `oceansystemslab`
dataset), held out by entire collection *date* (not a random split) — so
validation checks generalization to entire unseen sessions, not just
unseen samples. Real logged thruster commands are replayed open-loop through
default and calibrated physics and compared against the real logged
velocities, per axis.

**1. Stand-in and MuJoCo (the engines this product uses), on the 14 held-out
runs.** For each axis, the table shows **skill**: 1 − error / error of simply
predicting "no motion on this axis". 1 is perfect, 0 is no better than
predicting no motion, and negative is worse. Whole runs are replayed with the
real orientation fed in; free-running 3 s windows in MuJoCo give the same
picture (PROJECT_STATUS.md #50).

| Calibration | Surge (u) | Sway (v) | Heave (w) | Roll rate (p) | Pitch rate (q) | Yaw rate (r) |
|---|---|---|---|---|---|---|
| default physics (uncalibrated) | +0.52 | −0.22 | −1.29 | −7.97 | −11.25 | +0.19 |
| expA (fleet truth until 2026-09-24) | +0.71 | +0.51 | +0.58 | −0.49 | −0.84 | +0.60 |
| **armfix (recommended, and the fleet truth vehicle)** | +0.72 | +0.51 | +0.59 | −0.03 | −0.10 | +0.58 |

- **Surge, sway, heave and yaw are predictive**, and calibration clearly helps.
- **Roll and pitch are not predictive yet.** The torque-arm correction took them from worse-than-nothing to about zero. Most of the real roll/pitch motion in this dataset isn't caused by the thrusters, so no fit to this data can predict it.
- That doesn't affect fleet formation-holding (PROJECT_STATUS.md #52). It would matter for attitude-sensitive tasks, and calibrating it needs roll/pitch tests from a lab (free-decay tests, deliberate roll/pitch manoeuvres).
- An earlier version of this table showed "+82–87% improvement" on roll and pitch. That was improvement over default physics, which is very poor on those axes; it didn't mean the model predicts them.
- Caveat: every converged fit has several parameters at their `vehicle_spec.yaml` bounds. For roll/pitch damping that's an artifact of fitting motion the thrusters don't cause (#50); the translational ones need a review by someone who knows the vehicle.

**2. Inside HoloOcean (an independent engine) — provisional.** The earlier
HoloOcean result (~3–8% scale-normalized overall; surge +61–63%, sway
+29–30%, heave +27–35%, pitch +25–35%, roll and yaw *worse* than default)
predates two problems found in the HoloOcean backend (PROJECT_STATUS.md
finding #40): it omitted the rotating-frame term when handing HoloOcean a
world-frame acceleration (the same omission was confirmed and fixed in the
MuJoCo backend), and it never checked that HoloOcean's tick length matches
the 0.1 s logs (HoloOcean defaults to 30 ticks/s). Both are fixed or guarded
in code but not yet confirmed in HoloOcean itself, which can't run on macOS.
Run `calibration/verify_holoocean_frame_conventions.py` on a Windows or Linux
machine with HoloOcean, then re-run Claim A, before quoting HoloOcean numbers.

## Status

- ✅ Real-data calibration pipeline, validated (held-out-by-date, multi-restart stability); stand-in calibration 3× faster
- 🔄 HoloOcean integration: runs, but the rotating-frame fix and tick-rate check are unconfirmed in HoloOcean (see below)
- ✅ Gym-compatible RL environment (`Arena`), four tasks, PPO/SAC/TD3 training confirmed working
- ✅ Automated pre-flight data-quality checks, automated identifiability reporting
- ✅ Lab data tools: PWM → thrust (reproduces all 183,555 dataset rows) for any thruster layout, log ingestion from CSV or ROS 1/ROS 2 bags with silent-error guards, ready-to-edit templates, plain-language calibration reports that include which values the data can't pin down
- ✅ A lab's own vehicle end to end (#56): raw logs → one command → calibration and report → single-vehicle (`strathe arena`) and fleet experiments and fleet training with its calibration, vehicle description, thruster limits and water current; all on the web page too
- ✅ Calibration stall diagnosed (the Mac was asleep) and fixed: long runs keep the machine awake, with a watchdog on every run
- ✅ Automated test suite (396 + 142 fleet tests, incl. real-data validation on the dataset fetched in CI), Windows CI on every push
- ✅ Comm channel module (latency, packet loss, bandwidth), vehicle-agnostic, zero physics dependency
- ✅ Fleet v2: per-vehicle policies, physical teammate belief models, paired belief-to-success study with robustness checks
- ✅ MuJoCo physics backend, independently validated (frame conventions, held-out RMSE, multi-body zero-cross-talk)
- ❌ Fleet v1 "Claim C" withdrawn — its metric didn't measure calibration (PROJECT_STATUS.md #46)
- 🔄 HoloOcean-backed results provisional — backend frame fix and tick-rate guard need confirming on a HoloOcean machine (`verify_holoocean_frame_conventions.py`)
- 🔄 Roll/pitch not predictive yet: torque-arm correction brought them from worse than "no motion" to about zero skill; this dataset can't calibrate them further. Doesn't affect formation-holding (15,200 paired episodes); needs lab roll/pitch tests for attitude-sensitive tasks
- ✅ Surge, sway, heave and yaw predictive on held-out data (skill +0.51 to +0.72)
- ✅ Roll/pitch disturbance fitted to real logs (noisy-simulation mode), MuJoCo and stand-in
- ✅ One-command onboarding (`python strathe.py`) and a local web page (`python strathe.py web`)
- ✅ Water current (open water): configurable and off by default, in both engines, the fleet, the single-vehicle environment, the CLI, the web page and calibration from logs in a known current
- ✅ Unknown water currents estimated per recording (`--estimate-current`), with checks (#57)
- ⏳ A current that changes within a recording, or varies across space
- ✅ Single-vehicle arena on MuJoCo too, with the fitted roll/pitch disturbance (#57)
- ✅ ArduPilot `.bin` / MAVLink `.tlog` logs read directly (#57)
- ✅ Fleet experiments from flags and web fields, no YAML editing (#57)
- ✅ Every known limit listed in [LIMITS.md](LIMITS.md)
- ✅ Single-vehicle env physics step fixed (0.02 → 0.005 s): the old default diverged under random commands (#55)
- ✅ Fleet truth switched to the torque-arm-corrected calibration, all 13 policies retrained and every fleet study rerun: calibrated beats generic by +13 to +25; better information worth up to +28 pts
- 🔄 Policies trained for the calibrated model alone are unreliable across seeds with the new truth (35–98% at L0, 3 s beacons); the mixed-trained policies are consistent
- 🔄 Physics dead reckoning now ties last-beacon (no longer trails); intent-sharing beacons are the next lever
- ⏳ Fleet policies are trained on the stand-in engine; MuJoCo is used as an independent truth engine for evaluation
- ⏳ Real hardware validation — not yet done, for either the single-vehicle or fleet work; looking for lab partners with BlueROV2 access

## Experiment Tracking (MLflow)

Every real calibration run - `run_calibration.py`, `run_calibration_holoocean.py`,
and `run_full_pipeline.py` - automatically logs to a local MLflow experiment
(`strathe_calibration`). This is purely observational: if MLflow is
missing or logging fails for any reason, the calibration itself is
completely unaffected - you'll see a warning printed, never a crash.

Fleet v2 experiments (`run_fleet_experiment.py`) log to a separate `strathe_fleet` MLflow
experiment: success rate and belief error per (comm condition, belief model), the headline
paired differences, and the whole output folder as artifacts, with the same observational
never-crash guarantee. (Retired fleet v1 scripts log under the older fleet schema.)

### Viewing logged runs

```bash
cd calibration
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open the URL it prints (default `http://127.0.0.1:5000`) in a
browser. **Always pass `--backend-store-uri sqlite:///mlflow.db`
explicitly** - a real issue found while building this: `mlflow ui` and
the Python client library can each independently fall back to a
*different* default storage location if neither is told explicitly,
silently showing an empty UI even though real data exists. This project
always logs to `sqlite:///mlflow.db` (a local file, `calibration/mlflow.db`
- not tracked in git, regenerated automatically).

### What's tracked

- **Parameters**: which vehicle spec and dataset were used (including a
  dataset "signature" - file count, total size, modification time - an
  *identifier*, not a cryptographic guarantee of uniqueness), the
  optimizer and its settings, sampling rate, and (where applicable)
  algorithm/seed/task, plus real, live-queried package versions
  (numpy, scipy, stable-baselines3, mlflow itself) for reproducibility.
  For fleet runs: comm-condition metadata (packet-loss/bandwidth config,
  comm seed) as well.
- **Metrics**: RMSE results (aggregate, and per-DOF where the script
  computes them without extra real cost), runtime, and evaluation count.
  For fleet runs: success rate and belief error per condition, and the
  headline paired differences.
- **Status**: every run is tagged with one of four honest outcomes -
  `SUCCESS` (the optimizer genuinely converged), `CHECKPOINT_FALLBACK`
  (didn't converge, but a real, usable checkpoint result was
  intentionally substituted instead), `FAILED` (no usable result at
  all), or `INTERRUPTED` - plus a `status_reason` with the real, specific
  detail (e.g. the actual scipy convergence message).
- **Artifacts**: the real `vehicle_spec.yaml` used, the saved calibrated
  parameters, and the saved metrics/report file.

### What's deliberately NOT tracked

Raw trajectory data is never logged - it's large, already lives in your
own dataset directory, and would bloat the tracking store for no real
benefit. Only the dataset's identity (path, file count, size, mtime) is
recorded, so you can tell *which* dataset a run used without duplicating
the data itself.

## License

Strathe is proprietary software — all rights reserved. See [LICENSE](LICENSE).
Access to this repository does not grant any right to use, copy, modify or
distribute it; labs and other partners use Strathe under a separate written
evaluation, pilot or commercial agreement.

## Contact

Abdoul Seydi — abdoulayeaseydi@gmail.com

Looking for feedback, collaboration, or lab partners with real BlueROV2
(or similar hovering-AUV) access for hardware validation.
