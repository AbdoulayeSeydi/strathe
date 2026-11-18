# Heterogeneous escort: a BlueROV2 escorting a REMUS 100

A hover vehicle and a fin-steered vehicle on one acoustic link. The REMUS flies its own survey; the BlueROV2 must stay
3 +- 1 m from it, knowing where it is only from its beacons and a model of it between them. Which model the ROV carries
is the experimental variable.

| | BlueROV2 (escort) | REMUS 100 (surveyor) |
|---|---|---|
| Steers with | 8 thrusters (BlueROV2 Heavy allocation, `thruster_model.bluerov2_heavy`) | 4 fins and a propeller |
| Can hover | yes | no (0.5 to 0.7 m/s at 700 rpm) |
| Truth physics | Strathe's BlueROV2 model calibrated from HoloOcean simulator data (`calibrated_parameters_fleet_truth_alt.yaml`, a checkpoint, not converged) | Prestero's 2001 equations (`stress/prestero_remus.py`), not Strathe's model |
| Controlled by | the learned policy (or `EscortPursuit`) | its own autopilot (the #97 Prestero autopilot plus a depth loop) |
| Sends | nothing | a beacon every 0.5 to 3 s: position, attitude, velocity, fin angles, rpm |

## Teammate models (what the ROV believes about the REMUS)

| Name | What it is |
|---|---|
| `oracle` | the true position (upper bound) |
| `calibrated` | Strathe's fin model, parameters calibrated from logs of this REMUS (#97 `T-prestero-slow`), holding the beacon's fins and rpm |
| `physics_alpha0.25/0.5/0.75` | parameters that far from calibrated toward generic (the dose-response) |
| `generic` | the same model with `describe-vehicle`'s tape-measure starting values |
| `constant_velocity` | last position + last velocity x age |
| `last_beacon` | where the last beacon said |

### What the beacon carries (`escort_beliefs.py`)

| Name | Beacon | Prediction between beacons |
|---|---|---|
| `calibrated_nofins` | state only | calibrated model, fins centred, the mission's rpm |
| `calibrated`, `generic` | state + fins + rpm | the model holding the fins and rpm (zero-order hold) |
| `calibrated_intent`, `generic_intent` | state + fins + rpm + heading and depth orders | the model flown under the REMUS's own autopilot (same law, 100 Hz) toward the orders |

The intent models assume the ROV knows the REMUS's autopilot law and gains, and that the orders on the send tick hold
until the next beacon. The pitch integrator isn't sent; it is set so the law gives back the reported elevator angle.

The calibrated model is not the truth: Prestero's equations have lumped lift terms and off-diagonal mass that Strathe's
model was not written from, so its errors are real model error.

## What was changed from the #97 vehicle, and why

- **Re-ballasted 6 N lighter** (from +7 N to +1 N). At the speeds a BlueROV2 can match, Prestero's +7 N beats the fins
  and the vehicle surfaces. It enters as a smaller B in his equations (exact; `test_reballasting_is_a_smaller_B...`).
  The calibrated model is told about the change (`net_buoyancy_z - 6`); the generic one keeps its neutral guess.
- **700 rpm** (~0.7 m/s): the calibration's logs were at 750 to 1150 rpm, so this is slightly below what it saw.
- **A depth loop** turning depth error into the autopilot's pitch order (a survey holds depth).
- **The ROV uses the least bound-limited BlueROV2 calibration** (3 of 22 parameters at bounds). The fleet's default truth
  has surge damping pinned at its bound and tops out near 0.8 m/s on the Heavy's thrusters, too close to the REMUS.

## Files

| File | |
|---|---|
| `escort_remus.py` | the REMUS truth, its missions, and the precomputed mission library |
| `escort_env.py` | the escort simulation, teammate models, task, scripted controller |
| `escort_beliefs.py` | the state-only and intent beacon contents and their models |
| `escort_gym.py` | gymnasium wrapper for SB3 training |
| `train_escort_policy.py` | trains a policy (fleet v2's hyperparameters) |
| `run_escort_study.py` | the paired evaluation and its report |
| `escort/remus_spec.yaml` | `describe-vehicle`'s REMUS description (identical to `fin_truth.vehicle_spec("prestero")`) |
| `escort/remus_calibrated_T-prestero-slow.yaml` | the #97 calibration the `calibrated` model uses |
| `tests/test_escort.py` | |

`fleet_belief.Beacon` gained an optional `controls` field (a fin vehicle's fins and rpm), which `PhysicsBelief` holds
between beacons; hover beacons leave it `None` and behave as before.

## Running it

```bash
cd calibration
python -m pytest tests/test_escort.py                          # ~30 s
python train_escort_policy.py --timesteps 20000 --seed 1 --out /tmp/escort_smoke   # smoke: builds the 400 training missions once
python train_escort_policy.py --seed 1 --out escort_policy_seed1   # 3M PPO steps; run seeds 2 and 3 side by side
python run_escort_study.py --policies escort_policy_seed1 escort_policy_seed2 escort_policy_seed3 scripted \
    --out results/escort_study
python run_escort_study.py --policies escort_policy_seed1 escort_policy_seed2 escort_policy_seed3 scripted \
    --beliefs calibrated_nofins calibrated_intent generic_intent \
    --merge-with results/escort_study/episodes.csv --out results/escort_beacon     # what the beacon carries
```
Robustness slice. The still-water policies can't be used in a current: like the scripted PD escort, they are
memoryless and learned their drag from ground velocity, so a current holds them off the band whatever they know about
the REMUS (the PD escort with perfect knowledge drops to 0%). So the slice uses policies trained with currents and the
ROV's sensor noise drawn per episode, and `scripted_pi`, the PD escort plus integral action:
```bash
python train_escort_policy.py --seed 1 --current-speeds 0 0.1 0.2 0.3 --noise open_water --out escort_robust_seed1
python run_escort_study.py --policies escort_robust_seed1 escort_robust_seed2 escort_robust_seed3 scripted_pi \
    --comms L3a@3 --current-speed 0.2 --noise open_water \
    --beliefs oracle calibrated calibrated_aware calibrated_intent calibrated_intent_aware generic constant_velocity \
    --out results/escort_open_water
```
Training in currents tells a physics model the current half the time (`--aware-prob`).
- `--current-speed`: a steady horizontal current toward a direction drawn per mission. It acts on both vehicles (the
  REMUS through Prestero's own current term, the ROV through its physics). A model named `<model>_aware` is told the
  current; without the suffix it assumes still water. `constant_velocity` sees the drift in the beacon's ground velocity.
- `--noise open_water`: the ROV's own sensors with the noise measured on a BlueROV2 at a fish farm (SOLAQUA; biases
  assumed). The policy sees what they measure, including a dead-reckoned own position; reward, success and the REMUS
  stay true. The REMUS's beacons stay exact (an assumption: no measured REMUS navigation error is used).
- The current changes the missions, so they get their own cache; still-water caches are unchanged.

CIs in the report resample whole missions (`mission_clustered`), since every policy runs the same episodes and each
mission gives several; `contrasts.csv` keeps the per-episode CIs too. `interactions.csv` has
calibration x intent = (calibrated_intent - generic_intent) - (calibrated - generic).

`--merge-with` pairs the new models with the main study's episodes; use the same policy names and the default
`--missions`, `--per-mission` and `--episode-seed`, or it refuses.

Missions are cached under `escort/missions_cache/` (training: seeds 0 to 399; evaluation: 100000 to 100099, never
trained on). The study runs every cell on the same episodes (3 per held-out mission by default), so all differences
are paired; `report.md` gives success, belief error and time in band split into straight legs and turns, the paired
contrasts with bootstrap CIs, and the dose-response.

## Limits

- One-way link: the REMUS doesn't hear or react to the ROV. A leader-follower escort, not mutual coordination.
- Still water and exact sensors by default; the robustness slice adds a current and the ROV's measured sensor noise, never the REMUS's navigation error.
- The ROV's BlueROV2 model was calibrated from HoloOcean simulator data (a non-converged checkpoint), not from real logs. The study holds it fixed; only the REMUS model varies.
- The calibrated REMUS model was fitted at 750 to 1150 rpm and is used at 700 rpm.
