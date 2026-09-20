# Lab User Journey — Mera Fleet Experiment Platform

**Status:** Written after F0.1 (cold dogfood pass) and F0.2 (gap analysis).
This document is the acceptance spec for F1–F7. Every engineering milestone
must be checked against the criteria here, not just against its own internal
tests.

---

## Goal

> Simulate two BlueROV2s coordinating under degraded acoustic communication,
> compare a calibrated-physics agent against a generic-physics agent, run a
> reproducible multi-seed experiment, and get an interpretable result — without
> asking the developer for help.

**Level 1 (reproduction):** run the developer's experiment.
**Level 2 (configuration):** change the experiment.
**Level 3 (research use):** formulate a new experiment with your own policy.

Current status: Level 1 is fully working. Level 2 is partially working (YAML
config exposes the main knobs; initial conditions and task structure still
require source edits). Level 3 works for policies trained via
`train_formation_baseline.py`; arbitrary external SB3 policies require
`validate_policy_compatibility.py` to confirm compatibility first (F3 work,
now complete).

---

## Starting point

- Git clone of this repo
- Python 3.13, macOS (Apple Silicon confirmed) or compatible Linux
- `pip install -r requirements.txt` — includes `mujoco`, `mlflow`,
  `stable_baselines3`, `gymnasium` (all now pinned in requirements.txt)
- **HoloOcean is NOT required for the fleet workflow.** It's only needed for
  the single-vehicle calibration pipeline.
- No other manual install steps.

---

## Required inputs

**To run the shipped default experiment (zero extra setup):**
- Nothing. `python run_fleet_experiment.py` uses `default_experiment.yaml`
  and the pre-shipped `formation_hold_ppo_curriculum_v5.zip` model.

**To run a custom experiment:**
- A copy of `default_experiment.yaml` edited with your parameters
- Optionally: a trained policy `.zip` (your own, or one of the four
  pre-shipped curriculum models)

**To train your own policy:**
- No additional data required — uses the same shipped `vehicle_spec.yaml`
  and `calibrated_parameters_a1.yaml`

---

## The commands

### Step 0: Confirm setup (once)

```bash
cd calibration
python3 -c "import mujoco, mlflow, stable_baselines3; print('Dependencies OK')"
```

### Step 1: Run the default experiment (Level 1)

```bash
cd calibration
python run_fleet_experiment.py
```

Expected runtime: ~7 minutes (stand_in backend, 5 comm_seeds).

### Step 2: Run a custom experiment (Level 2)

```bash
cp default_experiment.yaml my_experiment.yaml
# Edit my_experiment.yaml: change severity, episodes, model paths, etc.
python run_fleet_experiment.py --config my_experiment.yaml
```

No source edits required for any parameter exposed in the YAML.

### Step 3: Train your own policy (Level 3, optional)

```bash
# Fast sanity check first
python train_formation_baseline.py --mode gate1

# Real training (adjust timesteps as needed)
python train_formation_baseline.py --mode gate2 --total-timesteps 200000 --run-name my_policy

# Validate your policy is compatible before using it in an experiment
python validate_policy_compatibility.py formation_hold_ppo_my_policy.zip

# Use it in an experiment
# Set model_path: formation_hold_ppo_my_policy in my_experiment.yaml
python run_fleet_experiment.py --config my_experiment.yaml
```

---

## Expected output

### Console (run_fleet_experiment.py)

```
======================================================================
MERA FLEET EXPERIMENT
======================================================================
Config: /path/to/default_experiment.yaml

[1/4] Loading and validating experiment config...
  OK: 2 vehicles, severity=L2, backend=stand_in

[2/4] Loading and validating policies (pre-flight shape check)...
  [validate_policy_compatibility] PASS: ...
  OK: all 2 policies passed compatibility check

  WARNING: both vehicles use the same model ('...'). delta_physics_belief_error
  will be near zero by construction -- no real calibrated-vs-generic difference
  exists. ...
  [only appears when both vehicles use identical model_path]

[3/4] Running fleet experiment sweep...
  Written to results/.../L2_stand_in.csv

  ====================================================================
    fleet_baseline_sweep@L2  (averaged across 5 comm_seeds)
  ====================================================================
  Reading guide: negative delta_physics values = calibrated doing better
  Format: mean  std  [min, max] across comm_seeds

  success_rate                     +X.XXXX  ±X.XXXX  [...]
      fraction of episodes where both agents held formation for 20+ steps
  ...
  delta_physics_belief_error       -X.XXXX  ±X.XXXX  [...]
      *** KEY RESULT *** calibrated MINUS generic belief error about teammate
      ...

[4/4] Writing report and metadata...

======================================================================
EXPERIMENT COMPLETE in ~445s (7.4 min)
======================================================================
  Report:   results/fleet_baseline_sweep/summary_report.md
  Metrics:  results/fleet_baseline_sweep/metrics.json
  Metadata: results/fleet_baseline_sweep/metadata.json
```

### Output files

| File | What it is |
|------|-----------|
| `summary_report.md` | Human-readable markdown: setup, results table, caveats |
| `metrics.json` | Machine-readable aggregated stats per cell |
| `metadata.json` | Git commit, package versions, resolved config, timestamp |
| `*.csv` | Per-episode raw data, one file per sweep cell |

---

## Acceptance criteria

These are F0.2's 11 questions restated as pass/fail criteria. Every F1–F7
milestone must not regress any criterion that passes here.

### Q1: How do I simulate two BlueROV2s?

**Status: PASS.**
README fleet section now explains upfront (before any commands) that there are
two agents, one calibrated and one generic, using `calibrated_parameters_a1.yaml`
and default physics respectively. The task and the central question are both
stated before step 1.

**Acceptance criterion:** A stranger reading only the README can state which
agent uses calibrated physics and which doesn't, before running anything.

---

### Q2: Where is comm defined, and what numeric parameters do the severity levels use?

**Status: PASS.**
README now contains a severity table (L0–L3b) with packet drop probability,
bandwidth, and physical meaning for each level. `comm_channel.py` is listed
in "What's in this repo" with a plain-language description.

**Acceptance criterion:** A stranger can find the packet drop probability for
`L2` without opening any source file.

---

### Q3: What can I configure, and what's hardcoded?

**Status: PARTIAL PASS.** What's configurable via the experiment YAML is now
documented in the README and in `default_experiment.yaml` itself. What remains
hardcoded is also explicitly listed.

Remaining hardcoded (source edit required):
- Initial agent separation (randomized 2.0–5.5m range)
- Task type (formation-hold only)
- Episode length (200 steps)

**Acceptance criterion:** A stranger can distinguish "I can change this in
the YAML" from "I need to edit source" without asking.

---

### Q4: What makes a policy compatible, and how do I know if mine is?

**Status: PASS.**
Policy interface contract is documented in the README fleet section (obs
`Box(24,)`, action `Box(12,)`, bounds `[-1,1]`). `validate_policy_compatibility.py`
runs this check pre-flight with specific error messages naming the mismatch.

**Acceptance criterion:** A policy trained on a single-agent Arena environment
fails at load time with a message explaining it's a single-agent policy, not
mid-run with a cryptic shape error.

---

### Q5: How do I change initial conditions, task parameters, seeds, episode count?

**Status: PARTIAL PASS.**
Seeds (`comm_seeds`, `grid_seeds`) and episode count are configurable via YAML.
Task parameters (`target_separation_m`, `tolerance_m`, `hold_steps`) are
configurable via YAML since F1. Initial agent separation and task type remain
hardcoded.

**Acceptance criterion (current):** A researcher can change comm_seeds,
grid_seeds, episode count, and task parameters without source edits.

**Acceptance criterion (complete, pending):** Initial separation and task type
are also configurable via YAML. This is the remaining F1 gap.

---

### Q6: How do I know whether comm actually affected coordination?

**Status: PASS.**
`delta_physics_belief_error` is now labeled `*** KEY RESULT ***` in the
printed output, with a multi-line explanation of what it means, the expected
direction (negative = calibrated doing better), and when to expect signal vs.
noise (L0/L1 vs. L2/L3a).

**Acceptance criterion:** A stranger reading only the console output can
identify the key metric, state the expected sign for a meaningful result,
and understand why L0/L1 results are expected to be near zero.

---

### Q7: How do I run my own model?

**Status: PASS.**
`train_formation_baseline.py --mode gate2 --run-name myname` produces a
compatible `.zip`. README step 3 now documents the pre-shipped curriculum
models explicitly and explains how to substitute your own. `validate_policy_compatibility.py`
confirms compatibility before any compute is spent.

**Acceptance criterion:** A stranger can produce and use their own policy
without guessing the expected filename format or debugging a mid-run shape error.

---

### Q8: How do I run my own experiment?

**Status: PASS.**
`default_experiment.yaml` + `--config my_experiment.yaml` flag exposes the
full documented parameter surface. No source edits required for anything in
the YAML.

**Acceptance criterion:** A researcher can run a new experiment with different
`comm_seeds`, `severity`, `episodes`, and `model_paths` by editing a YAML file
only.

---

### Q9: What does the output mean?

**Status: PASS.**
`print_cell_stats` now explains every metric inline. `summary_report.md`
provides a structured human-readable summary. `metadata.json` records full
reproducibility information.

**Acceptance criterion:** A researcher receiving only the output directory
(no conversation, no README) can state the headline result and its main caveat.
This is the `F5.4` dogfood checkpoint, not yet run with a real second person.

---

### Q10: How do I verify the experiment is reproducible?

**Status: PASS (infrastructure).**
`metadata.json` records git commit, package versions, full resolved config,
and timestamp on every run. MLflow logs the same.

**Acceptance criterion:** Given only `metadata.json` from a previous run, a
researcher can exactly reproduce the environment and rerun the experiment.

---

### Q11: What is the scope of this tool — what can't it do?

**Status: PASS.**
README "What's in this repo" and fleet workflow sections both contain explicit
scope statements. `validate_experiment_config` rejects N≠2 vehicles with a
clear message naming generalization as explicitly out of scope.

**Acceptance criterion:** A stranger trying to use 3 or more vehicles gets an
explicit "out of scope" error, not a silent misfire.

---

## Known gaps (open as of F0.3)

| Gap | Severity | Planned fix |
|-----|----------|-------------|
| Initial agent separation hardcoded | Medium | Expose in YAML via task config (F1 remaining work) |
| Task type hardcoded (formation-hold only) | Medium | Expose via YAML task.type field (F1 remaining work) |
| Episode length hardcoded (200 steps) | Low | Expose in YAML (F1 remaining work) |
| `success_rate` noisy at small n | Research caveat | Documented in output, not a bug to fix |
| No MuJoCo-based training | Research scope | Out of scope for this track |
| No hardware validation | Research scope | Out of scope for this track |

---

## What this does NOT do

Explicitly out of scope for the current MVP. Attempting these will produce clear
errors, not silent misfires:

- **N ≠ 2 vehicles.** Rejected at config validation with an explicit "out of
  scope" message.
- **Non-BlueROV2 vehicle physics.** The calibrated agent uses
  `calibrated_parameters_a1.yaml`, which is fitted to a specific BlueROV2
  dataset. Substituting a different vehicle class requires a separate
  calibration run (the single-vehicle pipeline) and a different parameter file.
- **Real hardware.** No hardware-in-the-loop, no real acoustic modems, no
  live vehicle control. This is a simulation-only platform.
- **Arbitrary RL algorithms.** SAC/TD3 are supported by the interface but
  unvalidated by any real fleet training run. PPO is the only proven algorithm.
- **GUI, web dashboard, or hosted service.** CLI only.
