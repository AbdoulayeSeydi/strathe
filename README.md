# Strathe — Communication-Constrained Multi-Agent Simulation for Underwater Autonomy

A simulation and RL-training platform for studying how real-world communication constraints — latency, packet loss, limited bandwidth — affect coordination between multiple underwater vehicles, grounded in physics calibrated against real vehicle mission logs, not an idealized model.

**Current real scope: two-agent formation-hold, BlueROV2 physics only.** Every real result in this repo has been built and validated against a two-vehicle fleet and one vehicle class. Scaling to larger fleets and additional vehicle types is a real, planned next step — see PRODUCT_VISION.md for the honest scope of that gap.

**What this answers:** does calibrating each agent's physics against real vehicle data change how a fleet coordinates under realistic communication constraints, compared to using generic, uncalibrated physics — and does that effect survive the move from a fast physics stand-in to a validated, independent physics engine (MuJoCo)?

**Answer so far:** yes, on the metric that matters most. Under moderate-to-severe communication degradation, the calibrated agent maintains a measurably better belief about its teammate's position than the generic agent — an effect that holds up quantitatively, not just directionally, across both physics backends. See [Fleet Coordination Results](#fleet-coordination-results) below for the full, honest breakdown, including where this is still early.

This result rests on a real, separately-validated foundation: a system-identification pipeline that fits physics to real vehicle logs, integrated into [HoloOcean](https://byu-holoocean.github.io/holoocean-docs/v2.0.0/index.html) for high-fidelity single-vehicle validation. That pipeline's own results are real and reported below (see [Single-Vehicle Calibration Results](#single-vehicle-calibration-results)), but calibration accuracy is no longer this project's headline claim — the multi-agent coordination result is.

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

2. **Stand-in calibration** (fast, minutes):
   ```bash
   python run_calibration.py --data-dir <path> --spec vehicle_spec.yaml \
      --sampling-rate-hz 10.0 --heldout-dates 16-02-2024 19-02-2024 \
      --output-dir <output-dir> --max-nfev 60
   ```

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

## The fleet coordination workflow: multi-agent, comm-constrained

**Two agents, two physics models — stated up front before any commands.**
- **Calibrated agent:** uses `calibrated_parameters_a1.yaml`, parameters fitted to real BlueROV2 mission logs by the single-vehicle calibration pipeline above. This is the agent that "knows" the vehicle's real hydrodynamics.
- **Generic agent:** uses uncalibrated default physics — same vehicle model, no data fitting. This is the baseline.
- **Task:** both agents try to maintain a 2m separation (formation hold). Neither agent can see the other directly — they can only communicate position beacons over an acoustic channel.
- **The question:** does having calibrated physics change how well the calibrated agent coordinates under realistic acoustic communication constraints? And does that effect survive a switch from the training simulator (stand-in) to an independently-validated physics engine (MuJoCo)?

**Acoustic comm severity levels — what the named levels actually mean:**

| Level | Packet drop prob | Bandwidth | Physical meaning |
|-------|-----------------|-----------|-----------------|
| L0    | 0%              | Unlimited | Perfect comm — baseline, no degradation |
| L1    | 10%             | Unlimited | Mild: 1 in 10 beacons lost |
| L2    | 30%             | Unlimited | Moderate: nearly 1 in 3 beacons lost |
| L3a   | 60%             | Unlimited | Severe: majority of beacons lost |
| L3b   | 60%             | 23 bytes/tick | Severe packet loss + bandwidth-starved (23 bytes is 1 byte below a full position beacon — every bandwidth-limited message fails) |

Drop probability is distance-independent in the current model. Speed of sound is fixed at 1500 m/s (latency = distance / 1500). Both are hardcoded — see the "Configurable vs. hardcoded" note below if you need to change them.

1. **Gate 1 — fast sanity check** (does PPO even construct against the two-agent env, no crash):
   ```bash
   cd calibration
   python train_formation_baseline.py --mode gate1
   ```

2. **Gate 2 — real training** (stand-in physics backend, per this project's own training/validation split — see `phase2_config.py`'s Decision 2):
   ```bash
   python train_formation_baseline.py --mode gate2 --total-timesteps 20000 --run-name my_run
   ```
   Saves a trained policy (`formation_hold_ppo_my_run.zip`) and logs the run to MLflow (see below).

   **Policy interface contract** — what makes a policy compatible with this environment (relevant whether you use the pre-shipped policies or substitute your own):
   - **Observation space:** `Box(24,)` float32 — `[body_frame_velocity(6), orientation_roll_pitch(2), comm_obs(4)]` per agent × 2 agents = 24 total
   - **Action space:** `Box(12,)` float32, bounded `[-1.0, 1.0]` — 6-dim normalized thruster command per agent × 2 agents = 12 total (scaled internally by `action_bound=30.0 N·m`)
   - **Compatible algorithms:** PPO (proven in this project for fleet training), SAC and TD3 (supported by the interface, not yet validated by any real fleet training run)
   - **Not compatible:** single-agent policies (`Box(11,)` or `Box(22,)` obs — these will fail at load time with a clear shape-mismatch error), recurrent/LSTM policies, custom SB3 subclasses

3. **Comm-degradation sweep** (stand-in backend, across severities `L0`/`L1`/`L2`/`L3a`):

   The four `--model-paths` below (`formation_hold_ppo_curriculum_v5`/`_L1`/`_L2`/`_L3a`) are **already-included, pre-trained policies shipped in this repo** — one per severity — not something step 2 above produces. Step 2's own output (`formation_hold_ppo_my_run.zip`) is a separate, fresh model; it demonstrates that training works, but by itself it's only one model, and this sweep needs one per severity to compare. To run the sweep on your own freshly-trained models instead, repeat step 2 once per severity (`--run-name L0`, `--run-name L1`, etc.) and substitute your own four filenames below.

   ```bash
   python evaluate_comm_degradation_curve.py \
       --severities L0 L1 L2 L3a \
       --model-paths formation_hold_ppo_curriculum_v5 formation_hold_ppo_curriculum_L1 \
                      formation_hold_ppo_curriculum_L2 formation_hold_ppo_curriculum_L3a \
       --csv-prefix comm_curve
   ```

4. **MuJoCo-backend validation** — the same already-trained policies, re-evaluated on real MuJoCo dynamics instead of the stand-in (no retraining; see `phase2_config.py`'s `Claim C`):
   ```bash
   python evaluate_baseline_sanity_mujoco.py \
       --model-paths formation_hold_ppo_curriculum_v5 formation_hold_ppo_curriculum_L1 \
                      formation_hold_ppo_curriculum_L2 formation_hold_ppo_curriculum_L3a \
       --labels L0 L1 L2 L3a

   python evaluate_comm_degradation_curve_mujoco.py \
       --severities L0 L1 L2 L3a \
       --model-paths formation_hold_ppo_curriculum_v5 formation_hold_ppo_curriculum_L1 \
                      formation_hold_ppo_curriculum_L2 formation_hold_ppo_curriculum_L3a \
       --csv-prefix comm_curve_mujoco
   ```
   Expect the same qualitative shape as step 3's stand-in sweep, not identical numbers — the two physics backends are expected to differ somewhat, the same way MuJoCo's single-vehicle held-out RMSE was never expected to exactly match the stand-in's.

**After each run, results are automatically logged to MLflow.** To browse them:
```bash
cd calibration
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Open `http://127.0.0.1:5000` in a browser. Each sweep cell appears as a separate run with its comm severity, backend, success rate, and belief-error delta already tagged and filterable.

**What you can configure today vs. what requires source edits:**

Configurable via CLI flags right now:
- Which severity levels to sweep (`--severities L0 L1 L2 L3a`)
- Which trained policies to use (`--model-paths ...`)
- Number of comm noise seeds (`--comm-seeds`, default: 5)
- Episode seeds per grid point (`--n-seeds-per-grid`, default: 5)
- Output file prefix (`--csv-prefix`)

Currently hardcoded — requires editing source to change:
- Initial agent separation (randomized 2.0–5.5m range per episode)
- Task type (formation-hold only; `target_separation`=2m, `tolerance`=0.2m, `hold_steps`=20 consecutive steps)
- Episode length (200 steps max)
- Speed of sound / acoustic latency (fixed at 1500 m/s)
- Raw packet-loss probabilities (only the named severity levels above are exposed — you cannot set arbitrary drop probabilities via CLI today)

A dedicated experiment configuration file (`experiment.yaml`) is planned as the next engineering priority (`F1`) — this will expose the full parameter surface as a single editable YAML, no source edits needed.

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
- **`calibration/formation_hold_task.py`, `two_agent_orchestrator.py`** — the two-agent formation-hold coordination task and the orchestration loop tying two vehicle cores, the comm channel, and the task together into one Gym environment (`two_agent_formation_env.py`).
- **`calibration/mujoco_backend.py`** — the validated MuJoCo physics backend, used two ways: as a shared pose source for the coordination task, and (`MuJoCoArenaCore`) as each agent's own independent, live dynamics engine for re-evaluating trained policies.
- **`calibration/train_formation_baseline.py`, `evaluate_comm_degradation_curve.py` (+ `_mujoco.py`), `evaluate_baseline_sanity_mujoco.py`** — the fleet training and evaluation scripts described in the workflow above.
- **`calibration_output_*/`** — saved, ready-to-use calibrated parameter
sets (`calibrated_parameters.yaml` + `metrics.json` in each).
- **`PROJECT_STATUS.md`** — the full research log: every finding, every bug
found and fixed, every validation check, in chronological order. This is
the detailed record for anyone who wants to verify the methodology or
understand exactly how a specific number was produced. Not required
reading to just use the model.

## Fleet Coordination Results

**Methodology:** four PPO policies (`L0`–`L3a`), each trained on the stand-in physics backend at a fixed comm-degradation severity, then evaluated two ways: a baseline sanity check (trained policy vs. a random baseline, zero comm degradation) and a full comm-degradation sweep (32 initial-condition grid points × 5 comm seeds × 5 per-grid seeds, at each of `L0`/`L1`/`L2`/`L3a`), each run on both the stand-in backend and, separately, on real MuJoCo dynamics with no retraining.

**Baseline sanity:** all four trained policies clearly beat a random baseline on mean reward and on how long they hold formation, on both backends. `success_rate` alone is noisy at this sample size (`n=20`) — one cell (`L2`) scored *below* random on that single metric while still clearly beating it on the other two, which we read as sampling noise, not a broken transfer, per this project's own established decision rule for a similar situation.

**Comm-degradation sweep:** delivery rate matches almost exactly between backends at every severity (`~99%/89%/69%/40%` for `L0`–`L3a`) — expected, since delivery depends only on the comm channel's own randomness, not the physics backend, and confirms the comm model is genuinely backend-independent. Collision rate shows the same qualitative shape on both backends (a dip at `L1`, then a sharp climb through `L2`/`L3a`) at different absolute numbers.

**The main result:** `Delta_physics(belief_error)` — the calibrated agent's belief-error advantage over the generic agent — is near-zero and noisy at `L0`/`L1` on both backends, but shows a consistent, similarly-sized negative gap (calibrated doing better) at `L2` (stand-in `-0.017` vs. MuJoCo `-0.020`) and `L3a` (`-0.022` vs. `-0.024`). Under real communication degradation, the calibrated agent's advantage holds up quantitatively, not just directionally, across the switch from a fast physics stand-in to an independently-validated physics engine.

This is labeled **Claim C** in `phase2_config.py`, matching the same explicit labeling discipline as Claim A and Claim B below: complementary to both, not a third accuracy proof. It shows that a real physics-calibration difference produces a measurable, cross-backend-consistent coordination effect under communication constraints — evidence supporting the technical differentiation this project's moat claim rests on, not proof of the moat by itself.

**Honest limitations:** `success_rate` and off-axis drift do *not* transfer cleanly between backends — but `L0`–`L3a` are four separately-trained policies, so this reflects each training run's own idiosyncrasies as much as any backend effect, a real confound in the experiment design, not something introduced by the MuJoCo validation itself. No MuJoCo-based *training* has been done — every trained policy above learned on the stand-in only, per this project's training/validation split (`phase2_config.py` Decision 2); this result is about transfer of an already-trained policy, not about what MuJoCo-based training would produce. Absolute performance is proof-of-concept grade, not production-ready — success rates in the `15`–`30%` range. No real hardware validation has been done for the fleet work, same open item as the single-vehicle work below.

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

- ✅ Real-data calibration pipeline, validated (held-out-by-date, multi-restart stability)
- ✅ HoloOcean integration, frame conventions confirmed
- ✅ Gym-compatible RL environment (`Arena`), three tasks, PPO training confirmed working
- ✅ Automated pre-flight data-quality checks, automated identifiability reporting
- ✅ Real, automated regression test suite (26 tests), CI-verified on every push
- ✅ Comm channel module (latency, packet loss, bandwidth), vehicle-agnostic, zero physics dependency
- ✅ Two-agent coordination task and centralized PPO training, stand-in backend
- ✅ MuJoCo physics backend, independently validated (frame conventions, held-out RMSE, multi-body zero-cross-talk)
- ✅ Multi-agent comm-degradation result reproduced on the validated MuJoCo backend (Claim C) — same qualitative shape as the stand-in, and the calibrated-vs-generic belief-error effect holds up quantitatively at moderate-to-severe degradation
- 🔄 HoloOcean-backed results provisional — backend frame fix and tick-rate guard need confirming on a HoloOcean machine (`verify_holoocean_frame_conventions.py`); with the stand-in physics, calibration improves all six axes on held-out data
- 🔄 Fleet coordination policies are proof-of-concept grade (15–30% success rates), not production-ready
- ⏳ No MuJoCo-based training yet — every fleet policy above is stand-in-trained, MuJoCo-validated only
- ⏳ Real hardware validation — not yet done, for either the single-vehicle or fleet work; looking for lab partners with BlueROV2 access

## Experiment Tracking (MLflow)

Every real calibration run - `run_calibration.py`, `run_calibration_holoocean.py`,
and `run_full_pipeline.py` - automatically logs to a local MLflow experiment
(`strathe_calibration`). This is purely observational: if MLflow is
missing or logging fails for any reason, the calibration itself is
completely unaffected - you'll see a warning printed, never a crash.

The fleet coordination scripts (`train_formation_baseline.py`,
`evaluate_comm_degradation_curve.py` and its MuJoCo counterpart,
`evaluate_baseline_sanity_mujoco.py`) log the same way, via a schema
extension (v2) adding fleet-specific fields — comm-condition metadata,
per-agent reward, coordination-success rate — as first-class entries
alongside the original single-vehicle schema, same observational
never-crash guarantee.

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
  For fleet runs: per-agent reward and coordination-success rate.
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
