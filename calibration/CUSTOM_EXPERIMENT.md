# Running your own fleet experiment (fleet v2)

Every change below is a YAML edit. Copy the shipped config and edit the copy:

```bash
cd calibration
cp default_experiment.yaml my_experiment.yaml
python run_fleet_experiment.py --config my_experiment.yaml
```

**Or don't edit YAML at all.** Every common setting is a `strathe fleet` flag, and a field on the web page's
fleet form. Flags override the chosen YAML (`--config`, default `default_experiment.yaml`). A copy with your
settings is written to `strathe_jobs/` and run; the shipped files are never changed:

| Setting | Flag | YAML field |
|---|---|---|
| Packet-loss levels | `--comm L0 L2 L3a` | `communication.severities` |
| Seconds between beacons | `--beacon-interval 1 3` | `communication.beacon_interval_s` |
| Link: acoustic, optical or hybrid (section 15) | `--link acoustic hybrid --optical-water coastal harbor` | `communication.link`, `communication.optical.water` |
| Optical terminals (half-angles) | `--optical-terminals fore_aft_cones --optical-beam 20 --optical-fov 20` | `communication.optical.*` |
| What pointing costs | `--optical-compare-omni` | `communication.optical.compare_omni` |
| Hybrid policy | `--hybrid-policy redundant` | `communication.hybrid_policy` |
| Teammate models | `--beliefs calibrated generic last_beacon oracle` | `beliefs` |
| Headline pair | `--compare calibrated generic` | `compare` |
| Target separation / tolerance | `--target-separation 3 --tolerance 0.2` | `task.*` |
| Hold time / episode length (0.1 s ticks) | `--hold-steps 20 --episode-steps 200` | `task.*` |
| Collision radius | `--collision-radius 0.5` | `task.collision_radius_m` |
| Starting separations / closing speeds | `--separations 2.5 3.5 --closing-speeds -0.35 0 0.35` | `initial_conditions.*` |
| Sample size | `--seeds 5` | `evaluation.comm_seeds` |
| Physics engine | `--backend stand_in \| mujoco \| both` | `dynamics_backend` |
| Clean or noisy simulation | `--mode clean \| noisy \| both` | `mode` |
| Sensor noise | `--noise tank \| typical \| my_noise.yaml --noise-scale 2` | `sensor_noise` |
| Buoyancy change | `--buoyancy -3` or `--buoyancy -5 5` (+ `--buoyancy-known-to-beliefs`) | `buoyancy` |
| Roll/pitch disturbance | `--disturbance fitted --disturbance-scale 1.5` | `disturbance` |
| Water current | `--current-speed 0 0.3 --current-direction 0 360 ...` | `current` |
| Your vehicle | `--params ... --vehicle-spec ... --thrusters ...` | `truth_params`, `vehicle_spec`, `thrusters` |
| Policies (PPO, SAC or TD3; compared by label) | `--policies my_policy.zip` or `--policies ppo=fleet_policy_seed1 sac=my_sac` | `policies` |
| Name | `--name my_study` | `name` |

`--save-config my_experiment.yaml` writes the resulting file instead of running it. Settings that can't
make a valid experiment are refused before anything runs: a hold time longer than the episode, a starting
separation inside the collision radius, or a headline pair that isn't among the teammate models. Only
custom loss probabilities, speed of sound, `physics_params`/`physics_alpha` teammate models, reward
weights and the finer optical settings (a custom terminal layout, the delivery curve's width, the modem
latency, `link_up_timeout_s`) need YAML editing (LIMITS.md).

Every field reaches the simulation. The v1 runner validated task settings and
comm overrides and then ignored them; both are now covered by regression tests
in `tests/test_fleet_experiment.py`.

Results go to `output_dir`:

| File | What it is |
|---|---|
| `summary_report.md` | Plain-language results: headline comparison with 95% CIs, clean vs noisy (with `mode: both`), policy groups compared (with several labels or algorithms), every condition, caveats |
| `summary.csv` | One row per (comm condition, belief model), pooled over policies |
| `episodes.csv` | Every episode, with its policy, start state and comm seed |
| `metrics.json` | All of the above plus the statistics, machine-readable |
| `metadata.json` | Git commit, whether the code had uncommitted changes, package versions, resolved config |
| `experiment.yaml` | The resolved config; pass it back with `--config` to rerun exactly |
| `plots/` | Success rate and belief error by condition, dose-response if applicable |

---

## 1. Communication

```yaml
communication:
  severities: [L2, L3a]      # one result cell per level
  beacon_interval_s: 1.0     # or a list to sweep: [0.5, 1.0, 3.0]
```

| Level | Packet loss | Bandwidth |
|---|---|---|
| L0 | 0% | unlimited |
| L1 | 10% | unlimited |
| L2 | 30% | unlimited |
| L3a | 60% | unlimited |
| L3b | 60% | 1 byte short of a beacon (nothing gets through) |

`beacon_interval_s` is how often each vehicle transmits its state. Real
acoustic modems manage roughly one state update every 1–10 s. v1 beaconed
10 times a second, which kept every belief within a few centimetres of the
truth even at 60% loss. It must be a multiple of the 0.1 s tick.

A custom condition, on top of one named level or on its own:

```yaml
communication:
  severity: L2
  overrides: {near_field_prob: 0.45, far_field_prob: 0.45, bytes_per_tick: 1e9}
  speed_of_sound_mps: 1480.0
```

Custom levels are labelled `L2_custom` (or `custom` without a base level).

## 2. Belief models: what's being compared

Between beacons each vehicle predicts where its teammate is. That prediction
model is the experimental variable:

```yaml
beliefs:
  - calibrated                  # physics with the true vehicle's parameters
  - generic                     # physics with vehicle_spec.yaml defaults
  - {physics_alpha: 0.5}        # parameters halfway from true to generic
  - {physics_params: my_fit.yaml, label: my_fit}   # YOUR calibration file
  - constant_velocity           # extrapolate the last beacon at its velocity
  - last_beacon                 # no prediction at all
  - oracle                      # perfect knowledge (upper bound)
compare: {a: calibrated, b: generic}   # the headline paired comparison
```

With three or more of `calibrated`, `physics_alpha` and `generic`, the report
adds a **dose-response**: success rate against how far the model's parameters
are from the truth.

`physics_params` lets a lab test **its own calibration** as a teammate model:
how much coordination does your fit buy you compared with generic defaults?

## 3. The task and episode length

```yaml
task:
  target_separation_m: 2.0
  tolerance_m: 0.2
  hold_steps: 20          # consecutive 0.1 s ticks inside tolerance = success
  max_episode_steps: 200  # 20 s
  collision_radius_m: 0.5
```

The config is rejected if success would be impossible (`hold_steps > max_episode_steps`)
or if holding formation would itself count as a collision.

## 4. Initial conditions

```yaml
initial_conditions:
  separations_m: [2.0, 3.0, 4.0, 5.0]
  closing_speeds_mps: [-0.5, 0.0, 0.5]   # + = the vehicles start moving apart
```

Every combination is run once per comm seed. The vehicles start level, facing
+x, with the closing speed split evenly between them.

## 5. Your own policy

```bash
python train_fleet_policy.py --timesteps 3000000 --seed 4 --out my_policy   # ~40 min on an M1
python validate_fleet_policy.py my_policy.zip
```
For your own vehicle, train on its calibration (section 11):
`python strathe.py train-fleet --params <your calibration> --spec <your vehicle description> [--thrusters ...] --out my_policy`,
or use the web page's "Train a fleet policy for your vehicle" form.

```yaml
policies:
  - {model_path: my_policy}                 # PPO, SAC or TD3: read from the file
  - {model_path: fleet_policy_seed1}        # list several to pool over training seeds
```

The per-vehicle contract: observation `Box(13,)` and action `Box(6,)` in
[−1, 1], one network applied to each vehicle separately. The observation
layout is in `fleet_env.py`. The validator explains the common mismatches,
for example a v1 centralized `Box(24,)` policy or a single-vehicle Arena
policy. PPO, SAC and TD3 policies are all accepted, and which one a file holds
is read from the file (`algorithm:` is optional; a wrong one is refused with the
right name). To train SAC or TD3, and to compare algorithms, see section 14.

## 6. Sample size

```yaml
evaluation:
  comm_seeds: [1, 2, 3, 4, 5]   # each seed = a different packet-loss pattern
  workers: 7                    # parallel processes (default: CPU count - 1)
```

Episodes per cell = separations × closing speeds × comm seeds (8 × 5 × 5 = 200
with the defaults). Comparisons between belief models are paired, so they need
far fewer episodes than two independent samples would. If a CI in the report
straddles zero, add comm seeds before drawing a conclusion.

## 7. Physics engine

```yaml
dynamics_backend: stand_in   # or mujoco, or both
```

`mujoco` simulates the vehicles' true motion with MuJoCo's integrator. The
belief models always use the stand-in equations, as an onboard model would.

## 8. Roll/pitch disturbance (part of the noisy mode, section 12)

```yaml
disturbance:
  roll_pitch: fitted     # none (default) | fitted | path/to/your_fit.yaml
  scale: 1.0             # 0 = off, 1 = as fitted to real logs, 2 = twice as rough
```

A real BlueROV2 in the reference logs rolls and pitches by roughly 10–14°, and
the recorded thruster commands don't explain that motion. It comes from
something unlogged, such as tether pull, waves or contact. With `fitted`, each
vehicle is also pushed by random roll and pitch torques whose size and timing
were fitted to bring the simulated wobble as close to those logs as the truth
vehicle allows (PROJECT_STATUS.md #51, #54). With the current truth, the
disturbance brings roll/pitch rates and their timing closer to the real logs,
but the simulated vehicle tilts further than the real one with or without it.
Roll/pitch can't be calibrated from this dataset (#50), and formation-holding
doesn't depend on them (#52).

- The torques act on the true vehicles only. They never appear in the beacons,
  because a teammate can't know about them.
- Every condition in an experiment sees the same disturbances, so comparisons
  stay paired.
- The shipped fit belongs to the shipped truth vehicle. With your own
  `truth_params`, the run warns you; re-fit with
  `python fit_rollpitch_disturbance.py --params your_truth.yaml`.

To train a policy that has experienced the disturbance, run
`python train_fleet_policy.py --disturbance fitted --out my_policy`.

## 9. Water current (open water)

```yaml
current:                   # leave out for still water (the default: a tank)
  speed_mps: 0.2           # or [0.0, 0.3]: a speed drawn per episode
  direction_deg: 90        # direction the water flows toward, or [0, 360] for any direction
  vertical_mps: 0.0        # + = downward flow
  variability:             # optional slow fluctuation around the episode's mean
    speed_std_mps: 0.05
    direction_std_deg: 10
    time_scale_s: 30       # at least 1 s; fluctuations should be slow
  known_to_beliefs: true   # do physics teammate models know the episode's mean current?
```

How it works:
- The calibrated drag and added-mass forces act on each vehicle's velocity *relative to the water*. Positions move with velocity over ground, so an unpowered vehicle drifts with the current.
- Both vehicles feel the same current.
- Policies don't observe the current directly. They see their own velocity over ground, as with a DVL tracking the bottom.
- Every condition in an experiment sees the same currents, so comparisons stay paired.

Choosing the options:
- **`known_to_beliefs: false`** makes the physics teammate models ignore the current. Use it to measure what estimating the current is worth to your fleet.
- **Ranges (`[min, max]`)** draw a new speed or direction each episode. That's the usual way to test robustness.

To train a policy that has experienced currents:
```bash
python train_fleet_policy.py --current-speed 0 0.3 --current-direction 0 360 --out my_policy
```
Add `--current-speed-std`, `--current-direction-std` and `--current-time-scale` for fluctuation. The single-vehicle environment takes the same settings: `Arena(..., current=CurrentConfig(...))`.

To calibrate from logs recorded in a known current, see the README ("Bring your own vehicle's logs").

## 10. Thruster limits (your vehicle's real thrusters)

```yaml
thrusters: bluerov2_heavy                 # the validated BlueROV2 Heavy (8 x T200)
# or a file with a thrusters: block -- your ingestion spec works as is:
thrusters: ../my_logs/my_vehicle_ingest.yaml
# or inline, for any vehicle (templates/ingest_csv_pwm_custom_thrusters.yaml explains each field):
thrusters:
  layout:
    - {position_m: [0.13, -0.13, 0.0], direction: [-1, -1, 0], reversed: true}
    - ...
  curve: t200
  voltage: 16
```

Without it (the default, and every shipped result), a policy's command of up to
±30 N / N·m on each axis is delivered as is. With it, every command goes through
the vehicle's thrusters:
- It is split over the thrusters.
- If any thruster would exceed its thrust curve, the whole command is scaled down
  so none does. The direction of the command is kept, as ArduSub's mixer does.
- Axes the thrusters can't push get nothing.

Examples: a BlueROV2 Heavy can make about 18 N·m of pitch torque, not 30. A
6-thruster BlueROV2 can't pitch on command at all.

Beacons carry the delivered force, so the teammate models predict with what the
vehicle actually did. The report states the limits. The same option exists for:
- training: `train_fleet_policy.py --thrusters`, `strathe train-fleet --thrusters`
- the single-vehicle arena: `strathe arena --thrusters`
- the command line: `strathe fleet --thrusters`
- the web page's "Your vehicle" fields

## 11. Your own vehicle

```yaml
truth_params: ../calibration_output_my_rov/calibrated_parameters.yaml   # the vehicles' true physics
vehicle_spec: ../my_vehicle_spec.yaml    # mass, parameter bounds and defaults ("generic" uses these)
thrusters: ../my_logs/my_vehicle_ingest.yaml
policies:
  - {model_path: ../my_policies/my_rov_seed1}   # trained for this vehicle (section 5)
```

- `truth_params` is what both simulated vehicles really are.
- `vehicle_spec` gives the mass and the default ("generic") parameters. The
  `generic` teammate model is still "what you'd have without calibrating".
- The shipped policies were trained on the reference BlueROV2. They run on
  another vehicle, but for results you can report, train policies on your own
  calibration (section 5).

From the command line, `strathe fleet --params ... --vehicle-spec ... --thrusters ...`
writes a copy of your experiment with those fields set into `strathe_jobs/` and runs
it. The web page's fleet form does the same. For one vehicle instead of two, see
`strathe arena` in the README.

---

## 12. Clean vs noisy simulation (sensor noise)

```yaml
mode: noisy              # clean (default) | noisy | both
sensor_noise:            # optional; with mode noisy it defaults to typical
  preset: typical        # none | tank | typical | path/to/my_noise.yaml
  scale: 1.0             # 0 = off, 2 = twice as noisy
```

- **clean**: each vehicle knows its own velocity, attitude and position exactly,
  and nothing pushes it but its thrusters (the original simulation).
- **noisy**: realistic sensors plus the fitted roll/pitch disturbance
  (section 8). Unless you say otherwise, `sensor_noise: typical` and
  `disturbance: fitted`, both at scale 1; the saved `experiment.yaml` spells
  them out.
- **both**: every condition runs clean and noisy on the same episodes, and the
  report adds a "Does noise matter? Clean vs noisy" table, episode by episode.

What sensor noise means (`sensor_noise.py`): the policy sees what the vehicle's
own DVL (velocity), gyro (rates), attitude sensor (roll, pitch) and compass
(heading) report, and its teammate relative to where it *believes* it is. That
believed position is dead-reckoned from the DVL and compass, so it drifts.
Beacons carry what the sender believes, so the teammate's prediction starts
from that. Success, collisions and belief errors are still judged on the true
positions, and the report shows how far the navigation drifted
(`mean_nav_error_m` in `episodes.csv`).

| Preset | What it is |
|---|---|
| `tank` | The noise actually in the reference BlueROV2 tank logs, measured with `fit_sensor_noise.py`: about 2 mm/s velocity, 0.4°/s gyro, 0.1° attitude, 0.2° heading, 1 mm depth, and external positioning (3 mm, no drift). Small. |
| `typical` | A DVL + AHRS + depth-sensor vehicle navigating by dead reckoning: 1 cm/s velocity (0.5 cm/s bias, 2% dropouts), 0.6°/s gyro, 0.5° attitude, 2° heading (2° bias), 1 cm depth, 20 cm starting fix. **Assumed, not measured:** sizes of a common magnitude, fixed before any result, not taken from any open-water vehicle (`sensor_noise/typical_dvl_ahrs.yaml` gives each one). It's the default for `mode: noisy`, and reports that use it say the sizes are assumed. |
| your file | `python strathe.py fit-noise --data <your logs> --out my_noise.yaml` measures the white noise, dropouts and positioning from your own logs. Add your sensors' biases from their datasheets: logs can't show a constant offset. `fit-noise` is command line only (no web form yet); the web forms can use the file it writes ("From my own file"). |

**Which preset to use.** `tank` for a vehicle in a tank with external positioning. For open water, your own
fitted file; `typical` is a placeholder until you have one, and `--noise-scale 0.5` / `2` show how sensitive
your result is to it. Results quoted with `typical` (e.g. the 10-point cost in PROJECT_STATUS.md #58) are about
the assumed sizes, not a measured vehicle.

Every condition sees the same sensor errors, so comparisons stay paired.
To train a policy that has experienced the noise:
`python strathe.py train-fleet --params ... --mode noisy --out my_noisy_policy`.

## 13. Buoyancy change (payload, trim)

```yaml
buoyancy:
  offset_n: -3.0           # N added to the calibration's net buoyancy; - = heavier (payload), + = floats up
                           # or [-5, 5]: a value drawn per episode (training across trims)
  known_to_beliefs: false  # do the physics teammate models know about the change?
```

The true vehicles' net buoyancy differs from what they were calibrated with,
as after adding a sensor or retrimming. A payload that weighs 1 kg more than
the water it displaces is about −9.8 N. The vertical restoring force changes in every
engine; the righting moment doesn't. By default the teammate models keep the
calibrated value, since a calibration made before the change can't know it.
`known_to_beliefs: true` gives them the new value. Both vehicles get the same
change. Values beyond ±50 N are refused as a probable units mistake. The
default action bound (30 N) caps how much a policy can push back.

## 14. Training algorithms: PPO, SAC, TD3

```bash
python strathe.py train-fleet --params <calibration> --algorithm sac --seed 1 --out my_sac_seed1   # 1M steps by default
python strathe.py fleet --policies ppo=fleet_policy_seed1 sac=my_sac_seed1 --mode both
```

- `--algorithm ppo` (default, 3M steps), `sac` or `td3` (off-policy; 1M steps
  by default). The settings are in `train_fleet_policy.py` and each policy's
  `.json` records them.
- Policies with different labels (or, without labels, different algorithms)
  are compared in the report's "Policy groups compared" table: the same
  episodes, training seeds averaged per episode, a 95% CI per condition.
- How they compare on this task, judged against criteria written before any
  result (`results/fleet_v2_algorithms/`): over 17,600 paired episodes, SAC
  ties PPO (−0 points, 95% CI −1 to +1) and TD3 is 4 points better (+3 to +6),
  each with a third of PPO's training steps. Trained SAC and TD3 policies ship
  in `algorithm_policies/` (PROJECT_STATUS.md #58).
- **That validation covers the reference BlueROV2 and this formation task only**,
  in clean simulation, with SB3-standard settings not tuned for any other vehicle.
  The ranking doesn't necessarily carry over: TD3 was best clean and worst under
  noise. For your own vehicle, train two or three seeds of each and compare them
  in one run:

  ```bash
  python strathe.py train-fleet --params <yours> --algorithm ppo --seed 1 --out my_ppo_1   # and seed 2
  python strathe.py train-fleet --params <yours> --algorithm sac --seed 1 --out my_sac_1   # and seed 2
  python strathe.py fleet --params <yours> --mode both \
      --policies ppo=my_ppo_1 ppo=my_ppo_2 sac=my_sac_1 sac=my_sac_2
  ```
  Policies sharing a label are pooled as seeds of one group; the "Policy groups
  compared" table gives each group's success and the paired difference with a
  95% CI.

## 15. Optical and hybrid links

```yaml
communication:
  severity: L2                 # the acoustic channel, as before (not needed for link: optical)
  beacon_interval_s: 1.0       # acoustic beacons
  link: hybrid                 # acoustic (default) | optical | hybrid, or a list: [acoustic, hybrid]
  optical:
    water: coastal             # clear | coastal | harbor | custom (then attenuation_per_m), or a list
    terminals: omni            # omni (default) | forward_cone | fore_aft_cones | down_cone
    beam_half_angle_deg: 30    # cone presets only; HALF-angles, 180 = every direction
    receiver_half_fov_deg: 30
    rated_range_m: 30          # 50% delivery when pointed, in clear water
    data_rate_bps: 1.0e6
    beacon_interval_s: 0.1     # optical beacons: every tick
    compare_omni: false        # true: also run with omni terminals, same episodes
  hybrid_policy: optical_first # or redundant
```

```bash
python strathe.py fleet --link acoustic hybrid --optical-water clear coastal harbor
python strathe.py fleet --link hybrid --optical-terminals fore_aft_cones --optical-compare-omni
python strathe.py train-fleet --params <calibration> --link hybrid --optical-water coastal --out my_hybrid
```

**What the optical link is.** A second link beside the acoustic one:
- **Range:** short, set by water clarity.
- **Latency:** 1 ms, the modem's processing delay.
- **Bandwidth:** Mbps, so a beacon can go every tick.

**How delivery works:**
- **Received power.** Light power falls off as `exp(-c·d) / d²`, where `c` is the water's attenuation per metre:
  - clear 0.15
  - coastal 0.40
  - harbour 2.2
  - `custom`: your own value
- **Pointing gain.** The power is multiplied by a pointing gain: 1 when the receiver is inside a sender terminal's beam and the sender is inside a receiver terminal's field of view, 0 otherwise.
- **Delivery probability.** A smooth S-curve turns the result into a delivery probability. At the defaults, a pointed link delivers half its messages at 30 m in clear water, about 15 m in coastal water and about 4 m in harbour water.
- **Judged on the truth.** Delivery uses the true positions and attitudes, because the light follows the real vehicle. The beacon carries what the sender measured.

**Uncalibrated.** The water values are rough textbook figures and the S-curve is a placeholder. Every report says so. A lab's link logs, with success against range and angle, are what would calibrate it; there's no `fit` command for it yet. Check a vendor's specs before quoting a number.

**Terminals:**
- **`omni`** (the default) covers every direction, like LED arrays. The link then depends only on range and water, which is the fair baseline.
- **Cone presets** put 30° cones (half-angles) facing forward, forward and aft, or down. A list of `{axis_body: [x, y, z], beam_half_angle_deg, receiver_half_fov_deg}` gives any layout.
- **Watch the start layout.** The task starts both vehicles in a line facing +x, so `forward_cone` can't connect at the start, and the report says so up front. `fore_aft_cones` connects both ways.
- **`compare_omni: true`** runs every optical and hybrid condition again with omni terminals on the same episodes. The report's "What pointing costs" table is the paired difference.

**Hybrid policies:**
- **`optical_first`:** optical beacons at their own rate. A vehicle skips its acoustic beacon while the optical link is up, meaning it heard its teammate optically within `link_up_timeout_s` (default 3 optical intervals, YAML only).
- **Asymmetric terminals** (beam ≠ receiver half-angle): hearing your teammate doesn't mean it hears you. `optical_first` then runs as `redundant`, and the report says so.
- **`redundant`:** every beacon on both links.
- **Why keep `optical_first`?** Acoustic energy and airtime aren't modelled, so `redundant` is never worse here. `optical_first` becomes the realistic choice once more vehicles share one acoustic channel.

**Stale copies:** a teammate belief keeps the beacon with the newest send time, whichever link it came on, so a late acoustic copy never rolls it back.

**Pairing:**
- The optical link draws from its own random stream, once per optical send.
- A skipped acoustic beacon still uses its draw.
- So acoustic, optical and hybrid runs of an experiment share their start states and acoustic packet losses. The "Does the optical link help?" table compares them episode by episode.

**What the report adds** (section "Optical link"):
- per condition:
  - the share of acoustic and optical beacons delivered (*optical up*)
  - acoustic beacons skipped
  - the share of belief updates that came by light
  - the median age of the teammate belief
- the paired comparisons against acoustic only, and of cones against omni
- a start-geometry warning where the cones can't connect

`episodes.csv` has the same fields per episode.

**The shipped policies were trained on acoustic links only.** What they do with an optical link is in PROJECT_STATUS.md #59 (`results/fleet_optical/`). To see whether a policy learns to stay within optical range and pointed at its teammate, train one with `--link hybrid`.

## Still not configurable

| What | Why / where |
|---|---|
| More than 2 vehicles | `fleet_env.py` is built for 2; the per-vehicle observation and shared policy are the groundwork for N |
| Optical occlusion, sunlight, bubbles, a soft-edged beam | the optical link is range x water x a hard pointing cone (section 15) |
| Tasks other than formation hold | `formation_hold_task.py` |
| Start orientation (always level, facing +x) | `FleetSim.reset` |
| Correlated sensor errors (e.g. a compass error that depends on heading), DVL altitude limits | the sensor model is white noise + a constant bias per episode + dropouts (section 12) |
| A buoyancy change that moves the centre of gravity | the offset changes the net buoyancy only (section 13) |
| A current that varies across space (shear, eddies) | the current is uniform over the fleet; it can vary in time (section 9) |
| Thruster response lag, battery sag within a run | `thruster_model.py` maps PWM to thrust instantly at one voltage (section 10) |
| Two different vehicles in one fleet | both vehicles share `truth_params`, `vehicle_spec` and `thrusters` |

The retired fleet v1 setup (`run_fleet_experiment_v1.py`, `default_experiment_v1.yaml`)
still runs, for reproducing old results only. PROJECT_STATUS.md finding #46
explains why its headline metric was withdrawn.
