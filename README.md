# Strathe — Communication-Constrained Multi-Agent Simulation for Underwater Autonomy

A simulation and RL-training platform for studying how real-world communication constraints — latency, packet loss, limited bandwidth — affect coordination between multiple underwater vehicles, grounded in physics calibrated against real vehicle mission logs, not an idealized model.

**Current real scope: two-agent formation-hold, BlueROV2 physics only.** Every real result in this repo has been built and validated against a two-vehicle fleet and one vehicle class. Scaling to larger fleets and additional vehicle types is a real, planned next step — see PRODUCT_VISION.md for the honest scope of that gap.

**What this answers:** when two underwater vehicles coordinate over a sparse, lossy acoustic link, each has to predict where its teammate is between messages. Does it matter whether that prediction uses a physics model calibrated against real vehicle logs, or generic textbook physics?

**Answer so far** (simulation only; 38,400 + 12,800 paired episodes; confirmed on MuJoCo and with a second "true vehicle"):
- **A calibrated teammate model beats a generic one by +11 to +26 points** of formation success in every condition not already at 100%, when each vehicle's policy is trained for its own belief model. With policies trained on a mixture of belief models, the gap is +3 to +25. Either way, it grows as communication gets worse.
- **An uncalibrated predictive model is worse than no prediction at all**, by 13 to 31 points.
- **Better teammate information is worth up to +28 points** (perfect information vs none).
- **Physics-based dead reckoning still doesn't beat simply using the last message** (−2 to −7 points). The remaining error comes from the teammate's changing commands, not its physics. Sharing intent in the beacon is the next lever.

See [Fleet coordination results](#fleet-coordination-results). An earlier headline metric (fleet v1, "Claim C") was withdrawn after it turned out not to measure calibration at all ([PROJECT_STATUS.md](PROJECT_STATUS.md) #46).

This rests on a system-identification pipeline that fits physics to real BlueROV2 logs, validated on held-out recording sessions (see [Single-Vehicle Calibration Results](#single-vehicle-calibration-results)).

## Recent fixes (September 2026)

What changed, and why the numbers in this README differ from earlier versions. Details are in [PROJECT_STATUS.md](PROJECT_STATUS.md) #39–49.

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
- **Roll/yaw "weakness" (#40).** It isn't in the calibration. With the physics the fit was made in, calibration improves held-out roll by ~82% and yaw by 23–51%. The weakness only appeared inside HoloOcean, where the rotating-frame term was missing.
- **The "one evaluation takes 4 hours" stall is solved (#42, #49): the Mac was asleep.**
  - The new stall watchdog times every evaluation on three clocks and dumps stack traces when one runs long.
  - It caught 7 stalls in an overnight run. In each, the process computed for ~26 s while up to 67 min passed on the wall clock.
  - Every one lines up to the second with a macOS sleep/wake event.
  - Fix: run long calibrations under `caffeinate -i` (see step 2 below).
- **Stand-in calibration is 3× faster** (47.5 s → 16.1 s per full-dataset evaluation). The model is unchanged; the evaluation now uses the equivalence-tested fast integrator.

**Using a lab's own data**: new PWM → thrust model, log ingestion and plain-language reports. See [Bring your own vehicle's logs](#bring-your-own-vehicles-logs).

**Tests and CI**
- A fake `mujoco` module in one test file leaked into every later file. Test results depended on file order.
- One MuJoCo test file never actually ran; it passed values between tests in a way pytest doesn't support.
- CI now downloads the real dataset, pinned to a fixed commit, so the real-data tests run on every push.
- CI covers the fleet suite and a smoke run of the lab entry point. It runs on Windows, and several Windows-only failures were fixed (file encoding, permissions).
- 197 tests in `calibration/tests/` plus 142 in the fleet suite, all passing.

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

2. **Stand-in calibration** (~16 s per evaluation on the full dataset on an
   M1; a `--max-nfev 60` run is a few hundred evaluations, so plan on hours):
   ```bash
   caffeinate -i python run_calibration.py --data-dir <path> --spec vehicle_spec.yaml \
      --sampling-rate-hz 10.0 --heldout-dates 16-02-2024 19-02-2024 \
      --output-dir <output-dir> --max-nfev 60
   ```
   **On macOS, keep `caffeinate -i` in front of long runs.** Without it the Mac
   sleeps, and evaluations that take ~16 s stretch to over an hour
   (PROJECT_STATUS.md #49). Keep the lid open, since `caffeinate -i` doesn't
   stop lid-close sleep. Every run writes `eval_timing.csv`, which has each
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

The calibration pipeline reads one canonical CSV layout. Three tools get a
lab's own data into it and explain the result.

**1. Thruster PWM → force** (`calibration/thruster_model.py`). Most logs record
the PWM each ESC was sent, not force. `ThrusterConfig.wrench(pwm)` applies
motor directions, a thrust curve and the thruster allocation matrix. The
BlueROV2 Heavy preset (8 × T200) was extracted from the public dataset, which
logs all three stages. It reproduces the dataset's own forces from raw PWM
on all 183,555 rows. Its thrust curve (+47.9 N / −37.2 N) matches Blue
Robotics' published T200 figures at ~15 V (a 4S battery), and
`at_voltage(V)` rescales it to 12–20 V. Other vehicles use
`allocation_from_geometry()` plus their own curve file.

**2. Log ingestion** (`calibration/data_ingestion.py`). A short YAML spec says
which of your columns hold velocity, orientation and force (or PWM), in what
units and frame. ROS FLU/ENU frames are converted to Fossen FRD/NED.
```bash
cd calibration
python data_ingestion.py --spec my_vehicle_ingest.yaml --out ../my_vehicle_logs
python preflight_check.py --data-dir ../my_vehicle_logs --spec vehicle_spec.yaml
```
It refuses data that would silently corrupt a calibration:
- deg/s declared as rad/s
- non-unit quaternions
- non-monotonic or irregular timestamps (unless you ask it to resample)
- files at different rates
- PWM values that aren't ESC microseconds

`ingest_report.json` records the detected rate, time gaps and anything dropped. The
full example spec is in the module docstring. ROS bags: export the topics to
CSV first (for example with `rosbags-convert` or `ros2 bag` plus a small
script); direct bag reading isn't built yet.

**3. A report written for non-specialists** (`calibration/calibration_report.py`).
```bash
python calibration_report.py --result ../calibration_output_x --identifiability identifiability_report.json
```
It writes `calibration_report.md`, with:
- a bottom line
- a verdict for each motion axis (trust it / use with care / don't rely on it), with the reason
- parameters stuck at their limits, physically implausible values, and parameter pairs your data can't tell apart
- concrete next steps

Example: `calibration_output_10hz_coriolis/calibration_report.md`.

## The fleet coordination workflow (fleet v2)

**The setup, in one paragraph.**
- **Two simulated BlueROV2s must hold 2 m ± 0.2 m apart** for 2 s within a 20 s episode, without coming within 0.5 m.
- **Both run the same "true" physics**: `calibrated_parameters_fleet_truth.yaml`, the best held-out calibration.
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
Shipped policies: `fleet_policy_seed1-3.zip` (trained on a mixture of belief models) and `specialized_policies/fleet_policy_<belief>_seed1-2.zip` (each trained for one belief model; see results).

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
- **`calibration/stall_watchdog.py`** — times every calibration evaluation on three clocks and dumps stack traces for a stalled one (`eval_timing.csv`, `stall_log.txt`).
- **`calibration/verify_holoocean_frame_conventions.py`, `diagnose_rotational_weakness.py`** — the HoloOcean frame check (needs a HoloOcean machine), and a per-axis comparison of held-out results across calibrations.
- **`calibration/mujoco_backend.py`** — the MuJoCo physics backend (used as an independent "truth" engine for fleet evaluation).
- **`calibration/thruster_model.py`, `data_ingestion.py`, `calibration_report.py`** — PWM → force, lab log ingestion, plain-language calibration reports (see "Bring your own vehicle's logs").
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
- **Setup:** true physics = the best held-out calibration, stand-in engine.
- **Outputs:** full tables and plots in `calibration/results/fleet_v2_belief_to_success/`, per-seed checks in `extended_analysis.md`.

Formation success rate by teammate-prediction model:

| Comm | calibrated | generic | last beacon (no prediction) | oracle (perfect) | calibrated − generic (95% CI) |
|---|---|---|---|---|---|
| L0, beacon 1 s | 100% | 97% | 100% | 100% | +3 (+2, +5) |
| L1, beacon 3 s | 90% | 78% | 99% | 84% | +12 (+7, +15) |
| L2, beacon 1 s | 100% | 96% | 100% | 93% | +4 (+2, +5) |
| L2, beacon 3 s | 89% | 64% | 90% | 82% | **+25 (+20, +29)** |
| L3a, beacon 1 s | 97% | 86% | 99% | 86% | +11 (+8, +14) |
| L3a, beacon 3 s | 64% | 48% | 67% | 69% | +15 (+10, +20) |

What it shows:
1. **Calibration matters when a vehicle predicts its teammate with physics.** Calibrated beats generic in 7 of 8 conditions, and success falls steadily as the model's parameters move from true toward generic (dose-response, all 8 slopes below zero). Per training seed the gain is robust in the hard conditions and seed-dependent in the easy ones.
2. **An uncalibrated model is worse than none.** Generic loses to last-beacon in every condition, by 3 to 26 points.
3. **These policies were trained on a mixture of belief models**, and their input doesn't say which one is active, so perfect information scored *below* a calibrated model here. Policies trained for their own belief model fix this (next table).

**Follow-up: each belief model with a policy trained for it** (`run_specialized_study.py`, 2 seeds each, 12,800 paired episodes, `calibration/results/fleet_v2_specialized/`):

| Comm | calibrated | generic | last beacon | oracle | calibrated − generic (95% CI) |
|---|---|---|---|---|---|
| L0, beacon 3 s | 98% | 84% | 100% | 100% | +14 (+10, +18) |
| L1, beacon 3 s | 93% | 80% | 100% | 100% | +14 (+9, +18) |
| L2, beacon 3 s | 86% | 62% | 94% | 100% | **+24 (+18, +30)** |
| L3a, beacon 1 s | 97% | 86% | 99% | 100% | +11 (+8, +15) |
| L3a, beacon 3 s | 70% | 44% | 70% | 98% | **+26 (+20, +32)** |

(Every belief model scores 100% at L0, L1 and L2 with 1 s beacons.)

- **Better information now wins:** perfect information beats the last beacon by up to +28 points.
- **Calibrated beats generic by +11 to +26 points** wherever the task isn't already solved, and generic is 13–31 points *worse* than no prediction.
- **Calibrated dead reckoning ties or trails the last beacon (−2 to −7):** it holds the teammate's last thrust for up to 3 s while the teammate's controller keeps changing it. The next lever is sharing intent (planned thrust or trajectory) in the beacon, a natural research question for a lab partner (PROJECT_STATUS.md #48).

**Robustness:**
- **With a second "true vehicle"** (the least bound-limited calibration): calibrated beats generic by +14 to +19 points in 3 of 4 conditions, with no difference in the fourth.
- **With MuJoCo as the truth engine:** results match the stand-in within 1–2 points (L3a with 1 s beacons: 98 / 88 / 99% vs 97 / 86 / 99%).

**Limits:**
- Simulation only, 2 vehicles, formation hold, BlueROV2.
- In simulation `calibrated` equals the simulated vehicle, so this measures the value of model accuracy. How close a real calibration gets to a real vehicle is the single-vehicle result below.
- Success rates range from 44% to 100%, depending mostly on comm conditions and the belief model.

## Single-Vehicle Calibration Results

**Methodology:** real BlueROV2 mission logs (public `oceansystemslab`
dataset), held out by entire collection *date* (not a random split) — so
validation checks generalization to entire unseen sessions, not just
unseen samples. Real logged thruster commands are replayed open-loop through
default and calibrated physics and compared against the real logged
velocities, per axis.

**1. With the physics the calibration was fitted in (stand-in model), on the
14 held-out runs** (`calibration/diagnose_rotational_weakness.py`),
improvement over uncalibrated default physics:

| Calibration | Surge (u) | Sway (v) | Heave (w) | Roll rate (p) | Pitch rate (q) | Yaw rate (r) |
|---|---|---|---|---|---|---|
| expA (best converged fit) | +40% | +59% | +82% | +83% | +85% | +51% |
| 10hz_coriolis (converged) | +46% | +58% | +71% | +82% | +87% | +51% |
| holoocean_coriolis (Claim A) | +40% | +55% | +73% | +82% | +83% | +23% |

Calibration improves every axis, including roll and yaw. Caveat: every
converged fit has several parameters sitting at their `vehicle_spec.yaml`
bounds (PROJECT_STATUS.md, open items), so the bounds need review.

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
- ✅ Lab data tools: PWM → thrust (reproduces all 183,555 dataset rows), log ingestion with silent-error guards, plain-language calibration reports
- ✅ Calibration stall diagnosed (the Mac was asleep), with a watchdog on every run
- ✅ Automated test suite (197 + 142 fleet tests, incl. real-data validation on the dataset fetched in CI), Windows CI on every push
- ✅ Comm channel module (latency, packet loss, bandwidth), vehicle-agnostic, zero physics dependency
- ✅ Fleet v2: per-vehicle policies, physical teammate belief models, paired belief-to-success study with robustness checks
- ✅ MuJoCo physics backend, independently validated (frame conventions, held-out RMSE, multi-body zero-cross-talk)
- ❌ Fleet v1 "Claim C" withdrawn — its metric didn't measure calibration (PROJECT_STATUS.md #46)
- 🔄 HoloOcean-backed results provisional — backend frame fix and tick-rate guard need confirming on a HoloOcean machine (`verify_holoocean_frame_conventions.py`); with the stand-in physics, calibration improves all six axes on held-out data
- ✅ Specialized-policy follow-up: better teammate information → better coordination (+28 pts); calibrated beats generic by +11 to +26
- 🔄 Physics dead reckoning doesn't beat last-beacon yet — intent-sharing beacons are the next lever
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
