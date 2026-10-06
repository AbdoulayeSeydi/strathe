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

The simplest route is `python strathe.py web` from the repository root: a local
web page to start experiments, watch them run and read the reports.
`python strathe.py fleet [--config my_experiment.yaml]` does the same from the
terminal. Open-water labs can add a water current from the web form, with
`--current-speed` / `--current-direction`, or with a `current:` block in the YAML
(CUSTOM_EXPERIMENT.md section 9). The underlying scripts:

```bash
cd calibration
python run_fleet_experiment.py                               # Level 1: ~5 min on an M1 (3 policies, 16 cells)
cp default_experiment.yaml my_experiment.yaml                # Level 2: edit, then
python run_fleet_experiment.py --config my_experiment.yaml
python train_fleet_policy.py --timesteps 3000000 --out my_policy   # Level 3 (~40 min)
python train_fleet_policy.py --algorithm sac --out my_sac            # or SAC / TD3 (1M steps)
python validate_fleet_policy.py my_policy.zip
```

Clean or noisy: `python strathe.py fleet --mode both` runs every condition with
exact sensors and with realistic ones (plus the fitted roll/pitch disturbance)
on the same episodes, and reports the difference. `--buoyancy -3` makes the
vehicles 3 N heavier than they were calibrated (CUSTOM_EXPERIMENT.md sections
12-14).

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
N > 2 vehicles, other tasks, start orientation, correlated sensor errors.

**Q4. Is my policy compatible?** PASS. `validate_fleet_policy.py` checks the
per-vehicle contract before any compute is spent, and names the common
mismatches: a v1 centralized policy, or a single-vehicle Arena policy. PPO,
SAC and TD3 policies all work, and which one a file holds is read from the
file.

**Q5. Can I change starting conditions, task, seeds and episode length without editing code?** PASS.
All are in the YAML, and all reach the simulation. They are also `strathe fleet` flags and web fields,
so editing YAML is optional (CUSTOM_EXPERIMENT.md). The two v1 regressions
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

**Q11. What can't it do?** PASS. N ≠ 2 vehicles is rejected, and there is one
task (formation hold). It's simulation only. The "true vehicle" is a calibrated
model, so `calibrated` means "a model that matches the simulated vehicle". How
close a real calibration gets to a real vehicle is measured separately, on
held-out logs.

**Q13. Clean vs noisy: does my result survive realistic sensors?** PASS.
`mode: both` (`--mode both`, or the web form) runs every condition clean and
noisy on the same episodes and adds a paired "Does noise matter?" table. The
noise can be the shipped `tank` preset (measured from the reference tank logs),
the `typical` preset (**assumed** sizes for a DVL + AHRS vehicle), the `open_water`
preset (**measured** on a BlueROV2 at a fish farm, SINTEF's SOLAQUA data; biases
assumed), or your own, measured from your logs with
`strathe fit-noise` (command line only; the web forms can then use the file)
(CUSTOM_EXPERIMENT.md section 12).
Success is always judged on the true positions. Most of what noise costs the fleet
is each vehicle's starting position fix, not the sensors; set it to your
positioning's accuracy with `--start-fix-error` (PROJECT_STATUS.md #60).

**Q14. What if the vehicle is heavier or lighter than when it was calibrated?**
PASS. `buoyancy: {offset_n: -3}` (`--buoyancy -3`), or a range to train across
trims; the teammate models know about it only if you say so (section 13).

**Q15. Can I train with something other than PPO, and compare?** PASS.
`strathe train-fleet --algorithm sac|td3`; list the policies with labels
(`--policies ppo=... sac=...`) and the report compares them episode by episode
(section 14). How SAC and TD3 compare with PPO here: PROJECT_STATUS.md #58.
That validation is on the reference BlueROV2 and the formation task only; for
your vehicle, train two or three seeds of each and compare them in one run
(section 14 shows the commands).

**Q16. Can the vehicles use an optical link, or a hybrid of optical and acoustic?**
PASS. `communication.link: optical | hybrid` (`--link`, or the web form), with
water clarity, terminal layout (omnidirectional by default; forward, fore/aft or
down-facing cones) and a hybrid policy (section 15). The report adds the optical
link's delivery, which link refreshed each belief, the median belief age, and
paired comparisons against acoustic only and of cones against omni. It warns up
front when the cones can't connect at the start layout. The optical model is
**uncalibrated** (textbook water values, a placeholder delivery curve); a lab's link
logs would calibrate it. What it gives the shipped policies: PROJECT_STATUS.md #59.

**Q12. Can I use my own vehicle, not the reference BlueROV2?** PASS. Calibrate
it from its logs (below), then set `truth_params`, `vehicle_spec` and
`thrusters` in the experiment YAML (CUSTOM_EXPERIMENT.md sections 10-11). Train
a policy for it with `strathe train-fleet`. Each is also a flag on
`strathe fleet` and a field on the web page. Tested end to end with a
simulated 6-thruster vehicle (PROJECT_STATUS.md #56).

## The calibration journey: from a lab's raw logs to a report

The goal: a lab with its own vehicle's logs gets a calibrated simulator and an
honest plain-language report, without asking us.

```bash
cp calibration/templates/ingest_rosbag_mavros.yaml my_logs/my_rov.yaml   # or a CSV template; edit names
python strathe.py calibrate --ingest-spec my_logs/my_rov.yaml --name my_rov
```

**C1. My logs are ROS bags, ArduPilot logs, or CSVs with my own column names, units and frame.** PASS.
- ArduSub DataFlash `.bin` and MAVLink `.tlog` are read directly too (templates included). World-frame
  velocity, as ArduPilot logs it, is rotated into the body frame.
- One ingestion spec maps them (`calibration/templates/`, with a README).
- ROS 1 `.bag` and ROS 2 `.mcap`/`.db3` are read directly.
- Units and ROS FLU/ENU frames are converted.
- It refuses the classic silent errors.
- Logs exported as one CSV per sensor need merging onto one time base first: `calibration/examples/dfki_tether/`
  is a worked example on a second public BlueROV2 dataset (motion capture + ArduSub telemetry, #62).
- Ingestion checks the physics, not just the format: forces that push against the motion (a reversed thruster
  layout, a mirrored heading) are refused with the axis named, and a logged battery voltage (`voltage_column:`)
  or bad stretches (`exclude_samples:`, e.g. a taut tether) are handled (#63).

**C2. My logs have thruster PWM, not forces.** PASS. In ArduSub's motor numbering, FRD (#62). Choose one:
- the BlueROV2 Heavy preset
- a per-thruster `layout` for any vehicle, including ones that can't push every axis
- your own allocation matrix and thrust curve

**C3. One command, not six.** PASS. `strathe calibrate --ingest-spec` runs, in order:
1. conversion
2. data checks
3. the fit
4. which values the data can pin down
5. the report

It stops with a plain message at the first failing step. Checked end to end, every step as its own process, on
logs with their own dates (`tests/test_calibrate_end_to_end.py`): until #62 the data check only worked on the
reference dataset's dates. If the fit runs out of its evaluation budget,
the best result so far is kept, marked not converged, and the report is still written.
If it is interrupted (the machine slept, the run was stopped), the same command with `--finish`
does the same from its checkpoint. The web page has the same form.

**C4. My logs aren't at 10 Hz.** PASS. The rate measured from the timestamps is
recorded and used by every step. Before this, `strathe calibrate` silently
assumed 10 Hz (PROJECT_STATUS.md #56).

**C5. Can a non-specialist read the result?** PASS. `calibration_report.md` has:
- a bottom line
- a verdict per motion axis, judged against "predict no motion"
- values stuck at limits
- values the data can't separate, in plain words
- next steps
- how the simulation compares with the real vehicle on unseen recordings (#65): position drift, orientation with the model on its own, the 1 s response, command steps, and plots, shown inline on the web page too

**C6. My logs were recorded in open water.** PASS.
- A known current: `--current-world-mps`, or the web form.
- An unknown one: `--estimate-current` (a web checkbox too) estimates each recording's constant current and
  calibrates with it. It flags recordings where the current changed or the vehicle didn't turn.
- Limit: one constant current per recording (LIMITS.md).

**C7. My report says roll and pitch aren't predicted.** PASS, with a limit.
- `FREE_DECAY_PROTOCOL.md` says how to record free-decay tank tests (30-60 minutes), with an ArduSub template.
- `strathe free-decay` measures the unforced roll and pitch from them; `calibrate --free-decay` holds those values, and the report says they were measured.
- Limit: the thruster-driven roll/pitch isn't fitted yet. On the reference dataset, holding the measured values made whole-mission roll/pitch prediction worse (PROJECT_STATUS.md #64), so compare with a calibration made without them before simulating attitude-sensitive tasks.

## The single-vehicle journey

```bash
python strathe.py arena --params calibration_output_my_rov/calibrated_parameters.yaml \
    --task station_keeping --compare-generic [--current-speed 0 0.3] [--thrusters my_logs/my_rov.yaml]
```

**S1. Does a controller trained in my calibrated simulator do my task?** PASS.
`report.md` gives success rates with 95% CIs, time to succeed, final distance
and effort, for:
- policies trained on your calibration
- a hand-written controller
- random commands

Every controller sees the same episodes.

**S2. Does the calibration matter for this task?** PASS. `--compare-generic`
also trains on default physics. The report gives the paired difference with a
95% CI and says in words whether it is clear of zero.

**S3. Open water and real thruster limits?** PASS: the same current and
`thrusters` options as the fleet.

**S5. MuJoCo, and rough water?** PASS. `--backend mujoco` or `both`, and
`--disturbance fitted [--disturbance-scale 2]`, on the command line and the web form. Policies train on the
stand-in; with `both`, the report shows each engine and how closely they agree.

**S4. My own single-vehicle policy?** PASS. `--policy my_policy.zip` evaluates it
against the same references, whether it is PPO, SAC or TD3. A two-vehicle policy
given here is refused with an explanation.

**S6. Clean vs noisy, buoyancy, and other algorithms?** PASS. `--mode noisy`
(realistic sensors + roll/pitch disturbance, or `--noise` on its own),
`--compare-clean-training` (does training with the noise pay off? paired, with a
95% CI), `--buoyancy -3`, and `--algorithm sac|td3` (the settings validated for
one vehicle in PROJECT_STATUS.md #37). All are on the web form.

## Scope limits a lab should know before starting

The full list, with what to do about each, is in [`LIMITS.md`](../LIMITS.md). The short version:

| Limit | Status |
|---|---|
| 2 vehicles only (or 1 with `strathe arena`) | per-vehicle observations and a shared policy are the groundwork for N |
| Formation hold only | new tasks need a reward, a config and training |
| Sensor noise is white noise + a per-dive bias + DVL dropouts | no heading-dependent compass error or other correlated errors (LIMITS.md) |
| Optical link is uncalibrated | textbook water values, a hard pointing cone, no occlusion or sunlight (LIMITS.md) |
| No hardware in the loop | the reason to partner with a lab |
| Policies trained on the stand-in engine only | MuJoCo truth is available for evaluation |
