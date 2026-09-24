# Running your own fleet experiment (fleet v2)

Every change below is a YAML edit. Copy the shipped config and edit the copy:

```bash
cd calibration
cp default_experiment.yaml my_experiment.yaml
python run_fleet_experiment.py --config my_experiment.yaml
```

Every field reaches the simulation. The v1 runner validated task settings and
comm overrides and then ignored them; both are now covered by regression tests
in `tests/test_fleet_experiment.py`.

Results go to `output_dir`:

| File | What it is |
|---|---|
| `summary_report.md` | Plain-language results: headline comparison with 95% CIs, every condition, caveats |
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

```yaml
policies:
  - {model_path: my_policy, algorithm: ppo}
  - {model_path: fleet_policy_seed1}        # list several to pool over training seeds
```

The per-vehicle contract: observation `Box(13,)` and action `Box(6,)` in
[−1, 1], one network applied to each vehicle separately. The observation
layout is in `fleet_env.py`. The validator explains the common mismatches,
for example a v1 centralized `Box(24,)` policy or a single-vehicle Arena
policy. PPO is validated by this project's training; SAC and TD3 load but
haven't been trained for fleet v2.

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

## 8. Roll/pitch disturbance (clean vs noisy simulation)

```yaml
disturbance:
  roll_pitch: fitted     # none (default) | fitted | path/to/your_fit.yaml
  scale: 1.0             # 0 = off, 1 = as fitted to real logs, 2 = twice as rough
```

A real BlueROV2 in the reference logs rolls and pitches by roughly 10–14°, and
the recorded thruster commands don't explain that motion. It comes from
something unlogged, such as tether pull, waves or contact. With `fitted`, each
vehicle is also pushed by random roll and pitch torques whose size and timing
were fitted so the simulated wobble matches those logs (PROJECT_STATUS.md #51).

- The torques act on the true vehicles only. They never appear in the beacons,
  because a teammate can't know about them.
- Every condition in an experiment sees the same disturbances, so comparisons
  stay paired.
- The shipped fit belongs to the shipped truth vehicle. With your own
  `truth_params`, the run warns you; re-fit with
  `python fit_rollpitch_disturbance.py --params your_truth.yaml`.

To train a policy that has experienced the disturbance, run
`python train_fleet_policy.py --disturbance fitted --out my_policy`.

---

## Still not configurable

| What | Why / where |
|---|---|
| More than 2 vehicles | `fleet_env.py` is built for 2; the per-vehicle observation and shared policy are the groundwork for N |
| Tasks other than formation hold | `formation_hold_task.py` |
| Start orientation (always level, facing +x) | `FleetSim.reset` |
| Sensor noise on the vehicle's own state | not modelled yet (roll/pitch *disturbance* is, see section 8) |
| Water current / translational disturbance | not modelled yet |

The retired fleet v1 setup (`run_fleet_experiment_v1.py`, `default_experiment_v1.yaml`)
still runs, for reproducing old results only. PROJECT_STATUS.md finding #46
explains why its headline metric was withdrawn.
