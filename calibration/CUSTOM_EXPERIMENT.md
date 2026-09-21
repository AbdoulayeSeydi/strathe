# Running Your Own Fleet Experiment

This walkthrough shows four concrete changes you can make without editing
any source code. Each change is a YAML edit only.

Start by copying the shipped default config:

```bash
cd calibration
cp default_experiment.yaml my_experiment.yaml
```

Then make any combination of the changes below, and run:

```bash
python run_fleet_experiment.py --config my_experiment.yaml
```

Results go to whatever `output_dir` you set. The output structure is
identical to the default run: `summary_report.md`, `experiment.yaml`
(the resolved config), `metrics.json`, `metadata.json`, `plots/`.

---

## Change 1: Comm severity

**What:** test a different acoustic degradation level — L0 (perfect),
L1 (mild), L2 (moderate), L3a (severe), or L3b (severe + bandwidth-limited).

**Edit in `my_experiment.yaml`:**

```yaml
# Before
communication:
  severity: L2

# After — test severe degradation
communication:
  severity: L3a
```

**What to expect:** at L3a, `delivery_rate` drops to ~40%, and
`delta_physics_belief_error` typically stays meaningfully negative
(-0.022 on the stand-in backend in the reference run). At L0, delivery
is near-perfect and `delta_physics_belief_error` is near zero — the
calibrated physics advantage only appears when communication is unreliable.

**Or — test a custom condition without a named severity:**

```yaml
communication:
  overrides:
    near_field_prob: 0.45
    far_field_prob: 0.45
    speed_of_sound_mps: 1200.0  # shallower/warmer water
```

The label in the report will be `custom` rather than a named level.
Any named severity can also be combined with overrides:

```yaml
communication:
  severity: L2              # base: 30% drop probability
  overrides:
    speed_of_sound_mps: 1200.0  # override just this one parameter
```

The label will be `L2_custom`.

---

## Change 2: Formation target separation

**What:** change the target distance the two agents must hold.

**Edit in `my_experiment.yaml`:**

```yaml
# Before
task:
  target_separation_m: 2.0
  tolerance_m: 0.2
  hold_steps: 20

# After — tighter formation at 1m, harder hold criterion
task:
  target_separation_m: 1.0
  tolerance_m: 0.1
  hold_steps: 30
```

**What to expect:** tighter formations and longer hold requirements
make success harder — expect lower `success_rate` and higher
`collision_rate`. The `delta_physics_belief_error` signal should be
unaffected (it measures belief accuracy, not task success).

**Note on what's still hardcoded:** the initial conditions for each
episode (starting separation 2.0–5.5m, starting velocity ±0.7 m/s)
are not yet configurable via the YAML. This is a known remaining gap —
see `LAB_USER_JOURNEY.md` Q5. Initial conditions will be YAML-exposed
in a future version.

---

## Change 3: Episode count and seeds

**What:** run a faster/cheaper experiment for exploration, or a more
statistically robust one for publication.

**Edit in `my_experiment.yaml`:**

```yaml
# Fast exploration -- ~90 seconds instead of 8 minutes
evaluation:
  episodes: 32
  comm_seeds: [1]
  grid_seeds: 1

# Robust -- more seeds, tighter confidence intervals
evaluation:
  episodes: 640
  comm_seeds: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  grid_seeds: 5
```

**Tip:** `comm_seeds` drives most of the variance in
`delta_physics_belief_error`. Running with `comm_seeds: [1]` is fine
for a quick sanity check; use 5+ seeds for any result you want to
report. The std across seeds (`±0.0006` in the reference run) is your
confidence measure.

---

## Change 4: Swap in your own trained policy

**What:** use a policy you trained yourself instead of the pre-shipped
curriculum models.

**Step 1 — train a policy** (skip if you already have one):

```bash
python train_formation_baseline.py --mode gate2 \
    --total-timesteps 200000 \
    --run-name my_policy_L2
```

This produces `formation_hold_ppo_my_policy_L2.zip` in `calibration/`.

**Step 2 — validate compatibility** (pre-flight check, ~2 seconds):

```bash
python validate_policy_compatibility.py formation_hold_ppo_my_policy_L2.zip
```

Expected: `PASS: 'formation_hold_ppo_my_policy_L2.zip' -- obs (24,), act (12,), bounds [-1.0, 1.0]`

If validation fails, the error message names exactly what's wrong
(wrong observation shape, wrong action bounds, file not found, etc.)
before any compute is spent.

**Step 3 — edit `my_experiment.yaml`** (this step is required — copying the file is not enough):

Open `my_experiment.yaml` in any text editor and change `vehicles[0].model_path`:


```yaml
# Before
vehicles:
  - model_path: formation_hold_ppo_curriculum_L2
    algorithm: ppo
    role: calibrated
  - model_path: formation_hold_ppo_curriculum_v5
    algorithm: ppo
    role: generic

# After -- your own policy
vehicles:
  - model_path: formation_hold_ppo_my_policy_L2   # your model, no .zip suffix needed
    algorithm: ppo
    role: my_policy
  - model_path: formation_hold_ppo_curriculum_v5  # second entry validated but not used in sweep
    algorithm: ppo
    role: reference
```

Also change `name:` and `output_dir:` so your results don't overwrite the defaults:

```yaml
name: "my_policy_L2_experiment"
output_dir: results/my_policy_L2_experiment
```

**Step 4 — run:**

```bash
python run_fleet_experiment.py --config my_experiment.yaml
```

The pre-flight compatibility check (step 2) runs automatically again
inside `run_fleet_experiment.py` before any episodes execute.

**Compatible policy requirements** (from `POLICY_INTERFACE_CONTRACT` in
`validate_policy_compatibility.py`):
- Observation space: `Box(24,)` float32
- Action space: `Box(12,)` float32, bounded `[-1.0, 1.0]`
- Algorithm: PPO (proven), SAC or TD3 (supported but unvalidated)

Any policy trained via `train_formation_baseline.py` is guaranteed
compatible. External policies trained on a different environment may
not be — run `validate_policy_compatibility.py` first.

---

## Putting it together — a complete custom experiment

Here is a complete `my_experiment.yaml` that changes all four things
at once. Copy this, edit the model path, and run it:

```yaml
name: "severe_degradation_tighter_formation"

vehicles:
  - model_path: formation_hold_ppo_my_policy_L2   # your policy
    algorithm: ppo
    role: my_policy
  - model_path: formation_hold_ppo_curriculum_v5
    algorithm: ppo
    role: reference

communication:
  severity: L3a   # severe degradation

task:
  target_separation_m: 1.5   # tighter than the default 2.0m
  tolerance_m: 0.15
  hold_steps: 20

evaluation:
  episodes: 160
  comm_seeds: [1, 2, 3, 4, 5]
  grid_seeds: 5

dynamics_backend: stand_in
output_dir: results/severe_tight_experiment
```

```bash
python run_fleet_experiment.py --config my_experiment.yaml
```

---

## What is NOT yet configurable via YAML

These require source edits or a future config version:

| Parameter | Current state | Where it lives |
|-----------|--------------|----------------|
| Initial episode separation range | Hardcoded 2.0–5.5m | `TwoAgentFormationEnv.__init__` |
| Initial episode velocity range | Hardcoded ±0.7 m/s | `TwoAgentFormationEnv.__init__` |
| Task type | Formation-hold only | `TwoAgentFormationEnv` |
| Episode max length | Hardcoded 200 steps | `run_fleet_experiment.py` |
| Raw packet-loss probabilities | Only named severities via `severity:` (though `overrides:` exposes the underlying fields) | `experiment_schema.py` `CommOverrides` |

These are documented gaps, not silent limitations. See
`LAB_USER_JOURNEY.md` for the full acceptance criteria list.
