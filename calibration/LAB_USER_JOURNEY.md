# Lab user journey: Strathe fleet experiment platform (fleet v2)

**What this is:** the acceptance spec for handing the fleet kit to a lab.
Each question below is something a new user must be able to answer
without asking us. Any change to the fleet code must keep every PASS.

## Goal

> Simulate two BlueROV2s coordinating over a degraded acoustic link, compare
> how well they coordinate with different teammate-prediction models (for
> example a calibrated physics model against generic default physics), and
> get a reproducible, interpretable, statistically honest answer, without
> asking the developer for help.

- **Level 1 (reproduce):** run the shipped experiment.
- **Level 2 (configure):** change comm, task, starting conditions, belief models and sample size.
- **Level 3 (research):** bring your own policy or your own calibration file.

All three work today, with the scope limits at the end of this document.

## Starting point

- Clone the repo; Python 3.13 on macOS (Apple Silicon) or Linux, or 3.14 on Windows (CI).
- `pip install -r requirements.txt`. HoloOcean is **not** needed for the fleet kit.
- Access to the repo is under a written agreement (see LICENSE).

## The commands

```bash
cd calibration
python run_fleet_experiment.py                               # Level 1: ~5 min on an M1 (3 policies, 16 cells)
cp default_experiment.yaml my_experiment.yaml                # Level 2: edit, then
python run_fleet_experiment.py --config my_experiment.yaml
python train_fleet_policy.py --timesteps 3000000 --out my_policy   # Level 3 (~40 min)
python validate_fleet_policy.py my_policy.zip
```

## Acceptance criteria

**Q1. What exactly is being simulated?** PASS. The README fleet section and
every report's "What was tested" paragraph say it up front. There are two
vehicles with identical true physics, and each predicts its teammate between
beacons with a belief model. The belief model is the only difference between
conditions.

**Q2. What do the comm levels mean numerically?** PASS. There's a table in the
README and in CUSTOM_EXPERIMENT.md, and the resolved values of any run are in
`metadata.json`.

**Q3. What can I configure, and what's hardcoded?** PASS. CUSTOM_EXPERIMENT.md
covers every field, and its closing table lists what still needs source edits:
N > 2 vehicles, other tasks, start orientation, sensor noise.

**Q4. Is my policy compatible?** PASS. `validate_fleet_policy.py` checks the
per-vehicle contract before any compute is spent, and names the common
mismatches: a v1 centralized policy, or a single-vehicle Arena policy.

**Q5. Can I change starting conditions, task, seeds and episode length without editing code?** PASS.
All are in the YAML, and all reach the simulation. The two v1 regressions
(ignored task settings, crashing comm overrides) are covered by tests.

**Q6. How do I know whether the belief model affected coordination?** PASS.
The report opens with the paired headline comparison: success difference,
95% bootstrap CI, McNemar p and the number of matched episodes. It also states
the verdict in words ("calibrated did better" / "no clear difference").

**Q7. How do I run my own model or my own calibration?** PASS. Use
`policies: [{model_path: ...}]` and `beliefs: [{physics_params: my_fit.yaml, label: ...}]`.

**Q8. What does the output mean?** PASS. `summary_report.md` explains the
setup, results, how to read them and the limits, in plain language. The
numbers are in CSV and JSON.

**Q9. Is it reproducible?** PASS. `metadata.json` records the git commit,
**whether the working tree had uncommitted changes**, package versions and the
fully resolved config. `experiment.yaml` reruns the exact experiment. Every
episode is seeded.

**Q10. Is the comparison fair?** PASS, and tested. Conditions share start states
and packet-drop patterns (`test_drop_pattern_is_identical_across_belief_conditions`),
and one policy trained across all conditions is used for all of them. Neither
vehicle can observe its teammate except through the channel
(`test_observation_is_decentralized`).

**Q11. What can't it do?** PASS. N ≠ 2 vehicles is rejected. It's simulation
only, with one vehicle type (BlueROV2) and one task (formation hold). The
"true vehicle" is a calibrated model, so `calibrated` means "a model that
matches the simulated vehicle". How close a real calibration gets to a real
vehicle is measured separately, on held-out logs.

## Scope limits a lab should know before starting

| Limit | Status |
|---|---|
| 2 vehicles only | per-vehicle observations and a shared policy are the groundwork for N |
| Formation hold only | new tasks need a reward, a config and training |
| No sensor noise on a vehicle's own state | not modelled |
| No hardware in the loop | the reason to partner with a lab |
| Policies trained on the stand-in engine only | MuJoCo truth is available for evaluation |
