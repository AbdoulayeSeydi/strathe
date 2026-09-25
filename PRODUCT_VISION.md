# Product Vision — Strathe

*Companion to PROJECT_STATUS.md (the research log) — this document answers
"what are we actually building, for whom, and what's the gap between what
exists today and the full product." Written at the same level of detail
as the findings in PROJECT_STATUS.md, not an executive summary.*

*Updated after MVP completion — Part 3 substantially expanded with the
full post-MVP gap list. Further updated this session — see the note at
the end of Part 3.*

---

## The one-sentence answer, stated precisely

**We are building the simulation-and-calibration layer that underwater
robotics companies use to develop and validate their autonomy software —
not the autonomy software itself.**

A customer already has, or is building, navigation/docking/inspection/
obstacle-avoidance software for their AUV or ROV. What they don't have is
a simulator that actually behaves like *their specific vehicle* — a
generic simulator gives false confidence, because the sim-to-real gap is
large and vehicle-specific. We sit underneath their autonomy stack:
*"give us your vehicle's real logs, we give you back a simulator that
behaves like your real vehicle, plus the tools to train and validate
autonomy on it."*

We are explicitly **not**: a company that builds navigation/docking/
mission-planning software ourselves, a hardware company, or a company that
operates vehicles. Those are potential customers, not us.

---

## Part 1 — The full product

### 1.1 Target customer

- **AUV/ROV manufacturers** who need a calibrated simulator for every
  vehicle variant they ship, so their own customers (or internal teams)
  can develop autonomy without needing constant real water access
- **Marine robotics research labs** developing novel control/perception
  algorithms who need a trustworthy sim environment, not a generic one
- **Offshore inspection / defense / survey companies** operating fleets of
  vehicles, who need to validate autonomy updates before risking real
  hardware in the field
- **NOT the target customer, at least initially:** hobbyists, one-off
  academic projects without a real vehicle, anyone without real mission
  logs to calibrate against (the whole value proposition depends on real
  data existing)

### 1.2 The full user flow, step by step, in detail

**Step 1 — Vehicle onboarding**

The customer specifies their vehicle class — a structural fork in the
entire pipeline:

- **Hovering-style** (thrusters providing force/torque in multiple
  directions simultaneously — BlueROV2, most observation-class ROVs).
  HoloOcean's `HoveringAUV` "Custom Dynamics" control scheme — the
  integration path this MVP is built on.
- **Torpedo-style** (fin-actuated, primarily forward-moving — REMUS,
  Iver-class, most long-range survey AUVs). HoloOcean's `TorpedoAUV` has a
  genuinely different, native Fossen-dynamics customization interface —
  never touched. Not a variation on what exists; a second, parallel
  integration effort of comparable scope to the entire HoloOcean
  integration already built.
- The system also needs the vehicle's known constants (mass, approximate
  dimensions, thruster/fin count and layout, sensor suite) to generate a
  starting `vehicle_spec.yaml`-equivalent — templated per vehicle class,
  not hand-authored per vehicle the way this MVP's single spec was.

**Step 2 — Data upload**

Customer uploads real mission logs. In production this cannot assume one
fixed CSV schema — needs to:

- Parse multiple real-world formats (ROS bag files are the realistic
  format most customers would actually have)
- Automatically detect the real sampling rate (the validated methodology
  from this project's own sampling-rate investigation — now available as
  a real, standalone pre-flight tool, `preflight_check.py`, but not yet a
  fully automated pipeline stage in a hosted product sense)
- Automatically check maneuver diversity/excitation across every DOF
  before committing compute to a calibration that can't succeed (the same
  pre-flight tool covers this too)
- Flag data-quality problems before the customer waits hours for a result
  that was never going to be trustworthy

**Step 3 — Automated calibration**

Runs the real calibration methodology as a **managed cloud job**, not a
person watching a terminal:

- Real, structured progress a customer can check via a dashboard
- Parallelized multi-restart, cutting wall-clock time roughly N×
- **Must be reliable** — a customer-facing product cannot have the
  still-unexplained performance variance this MVP has (a single
  evaluation sometimes taking 100x longer than normal; real negative
  evidence has narrowed but not resolved the cause — see Part 3, Tier 0)

**Step 4 — Calibration report**

The automated, customer-facing equivalent of this project's own research
log:

- Per-DOF accuracy numbers, held-out-by-date validation results
- Honest confidence flags — e.g. "your data doesn't sufficiently excite
  roll; this axis's calibration should not be trusted yet"
- Multi-parameter identifiability warnings, surfaced automatically (the
  underlying analysis now exists as a real, working tool —
  `automated_identifiability_report.py` — that produces a structured
  report; it is not yet written for a non-specialist reader, see item 23)
- **Written for a non-specialist**, not someone with this project's own
  depth of understanding of Jacobians and SVD near-null-space directions

**Step 5 — Simulator delivery**

- A calibrated parameter file
- A ready-to-use Gym-compatible training environment, pre-loaded with the
  customer's calibrated physics, with a choice of task (`waypoint`,
  `station_keeping`, `docking` all now real, working options — see Part 2)
- **Two distinct delivery modes, not one** (see Part 3.E) — a clean/
  deterministic mode for calibration and validation, and a realistic/
  noisy mode (real sensor noise, real disturbances) for actually training
  and evaluating policies meant to run on real hardware
- Access to run locally, or against hosted HoloOcean compute

**Step 6 — Training & development**

- Customer trains their own autonomy using our environment as the
  trustworthy simulator underneath
- We provide example tasks/baseline policies, but do not build their
  mission-specific autonomy

**Step 7 — Real-world feedback loop**

- Customer deploys on real hardware, ideally feeds real performance data
  back in
- System tracks whether sim-to-real transfer actually holds over time
- **This is the step that proves the entire value proposition, and we do
  not have it yet.** Everything validated so far is against held-out
  *logs*, never a live vehicle.

---

## Part 2 — What the MVP actually is, today, in detail

**Be precise: this works for exactly one vehicle (BlueROV2), on exactly
one dataset, one physics regime (tank/pool, benign conditions), requires a
technically sophisticated user, and has one real, unresolved reliability
issue.**

### 2.1 The real, literal user flow today

A single command (`run_full_pipeline.py`) runs a fast, truncated demo end
to end — automated pre-flight checks, calibration, a real Gym-environment
smoke test — genuinely watchable in minutes, with a choice of training
task. The *real*, reportable-quality pipeline is a documented 6-step
sequence of separate scripts (pre-flight → stand-in fit → HoloOcean
refinement → Claim A validation → Claim B training → optional deeper
diagnostics), still requiring someone comfortable with the command line
and real patience (the HoloOcean refinement step alone can run for many
hours).

### 2.2 What the MVP genuinely, honestly proves

**The methodology works.** Calibration measurably improves how closely a
simulator tracks real vehicle behavior — held-out-by-date validation,
multi-restart stability, real bugs found and fixed with real evidence at
every step (a structural parameter-identifiability bug, a sampling-rate
error, a Forward-Euler numerical instability, a Coriolis-physics gap, a
non-physical inertia/buoyancy drift, several more). This is a real,
defensible research result. It is **not** yet something a robotics
engineer at another company could pick up and run on *their* vehicle
without personal guidance.

**A genuine, honest limitation, more precisely characterized than before:**
two rotational axes (roll rate, yaw rate) do not yet show the same
improvement as the other four. Two specific, well-reasoned hypotheses for
this have since been tested directly and refuted with real evidence — a
custom objective-weighting approach, and a fix for a separately-confirmed
non-physical parameter drift that fed directly into the relevant physics
coupling term. Neither helped. This is a stronger, more specific negative
result than "still investigating" — see Part 3, item 29.

---

## Part 3 — Detailed gap list: MVP → full product

> **Status update, 2026-09-23** (details in PROJECT_STATUS.md #39-#55). Items below are unchanged text; this is what has moved since:
>
> | Item | Now |
> |---|---|
> | 1. Performance mystery | **Solved: the machine was asleep.** The stall watchdog (`stall_watchdog.py`) caught 7 stalls in an overnight run. Each lined up to the second with a macOS sleep/wake event, while the process computed only ~26 s per evaluation (PROJECT_STATUS.md #49). Fixed: long runs now keep the machine awake themselves (`keep_awake.py`: macOS, Windows, Linux). |
> | 3, 12. Thruster allocation, raw-PWM datasets | **Done for T200-based vehicles.** `thruster_model.py` reproduces the dataset's forces from raw PWM on every row; the allocation matrix can be built from geometry for other vehicles. ESC lag and battery sag are not modelled. |
> | 9. Multi-agent | **Fleet v2 (2 vehicles):** per-vehicle policies, physical teammate beliefs, paired belief-to-success study. |
> | 11, 13. Data ingestion, garbage input | **CSV done** (`data_ingestion.py`: columns, units, ROS frames, PWM, timestamps, with guards against silent unit/frame/time errors). ROS bags: export to CSV first. |
> | 19. CI | Windows CI on every push, with the public dataset fetched so real-data tests run. HoloOcean-dependent code still can't run in CI; `verify_holoocean_frame_conventions.py` is the manual check. |
> | 27, 30. Customer confidence reporting | **Done:** `calibration_report.py` writes a plain-language per-axis verdict, limits, confounded parameters and next steps. |
> | 29. Rotational weakness | **Corrected and partly fixed (#50-#52).** Yaw is predictive (skill +0.60 vs predicting no motion). Roll and pitch were *worse* than predicting no motion; the earlier "+82% vs default" hid that. The main cause was a thrust moment arm in the logged torques. Correcting it (`vehicle_spec.yaml` `data_conventions`) and refitting lifts roll/pitch to about zero skill. The rest of this dataset's roll/pitch motion isn't caused by the thrusters, so it can't be calibrated from these logs; that needs lab free-decay tests. Formation-holding is unaffected (15,200 paired episodes with realistic wobble). |
> | Onboarding (six terminal commands) | **Done:** `python strathe.py calibrate --data <logs>` runs check, fit and report; `python strathe.py web` is a local web page for calibration, fleet runs, log conversion and reports (#53). |
> | Disturbances / clean vs noisy modes | **Started:** a roll/pitch disturbance fitted to the real logs is a YAML switch (`disturbance:`) and a training flag (#51). Sensor noise is not modelled yet. |
> | Water current (open water) | **Done, off by default (#55):** speed and direction (fixed or drawn per episode), vertical flow, slow fluctuation, and whether teammate models know it. Works in both engines, the fleet, the single-vehicle env, the CLI, the web page and training. Calibration accepts a known, roughly constant current per recording. Not built: estimating an unknown or changing current from logs, and currents that vary across space. |
> | Unit economics | The stand-in replay is 3x faster (`fast_dynamics.py`, measured A/B); see UNIT_ECONOMICS.md. |


Organized by category, roughly in priority tiers. Each item states not
just what's missing but why it matters.

### Tier 0 — reliability, before anything else scales

1. **The unexplained performance mystery.** A single evaluation has been
   observed taking up to ~4 hours against a confirmed ~100-second
   baseline, for reasons never fully diagnosed. This is not a "someday"
   item — it's a real reliability blocker for any future customer-facing
   product, not just an inconvenience during development. **Real negative
   evidence has since been gathered, narrowing but not resolving the
   cause:** the affected working directory is confirmed on local disk, not
   cloud-synced (ruling out OneDrive/Dropbox-style interference); a direct,
   isolated test of rapid repeated file writes (matching the real
   checkpoint-write pattern exactly, 500 real writes) came back completely
   clean, ruling out simple I/O contention alone as a sufficient
   explanation. The mystery likely requires the *combination* of sustained
   file I/O and heavy concurrent CPU/memory load (i.e., only appears
   inside a real, long-running optimization process) — a live process
   profiler (`py-spy`) attached to an actual stall in progress remains the
   one clear, undone, decisive diagnostic step, not yet completed.

### A. Vehicle & physics generality

2. **Torpedo AUV support** — see Part 1, Step 1. A second, comparably-sized
   integration effort using HoloOcean's native Fossen customization.
   Blocks any customer whose vehicle isn't hovering-style.
3. **Thruster allocation/dynamics** — real PWM-to-thrust mapping (deadband,
   motor response lag, saturation, nonlinear thrust curve). This MVP
   always assumed clean, pre-computed net force/torque already existed in
   the dataset — no real deployment will hand over data like that.
4. **Water current / disturbance modeling** — currently hardcoded to zero;
   `vehicle_spec.yaml` explicitly flags this as a tank/pool-only
   assumption that breaks for real open-water customer data.
5. **Open-water / real-ocean environmental range** — waves, varying
   depth/pressure, real temperature/fluid-density variation (distinct
   from current — even benign open water has physical constants that
   shift with depth/temperature in ways a tank never exercises).
6. **Vehicle-class bound templates** — every bound in this MVP's spec file
   was hand-derived for one vehicle over the course of one long session; a
   real product needs these generated from basic specs, not hand-tuned
   per customer.
7. **Off-diagonal added-mass generalization** — currently BlueROV2-specific
   and shelved (confirmed not load-bearing for this vehicle specifically);
   a different vehicle's real asymmetry would need different coupling
   terms, decided systematically, not re-derived from scratch each time.
8. **Payload modeling** — payload changes mass, buoyancy, CG, and CB
   simultaneously; nothing propagates a payload change through all four
   together yet.
9. **Multi-agent / multi-vehicle scenarios** — no vehicle-to-vehicle
   interaction, formation control, or multi-vehicle docking; single-
   instance only.
10. **Sensor suite generalization** — the pipeline assumes one specific
    measurement format (this dataset's exact velocity ground truth). Real
    vehicles vary widely (DVL or not, different IMU quality) — a
    genuinely different problem than data *format* ingestion; it's about
    what's actually measured at all.

### B. Data ingestion

11. **General data ingestion** (ROS bags, arbitrary CSV schemas) — hardcoded
    to one dataset's exact column format today. The mechanical part (a
    basic reader) is genuinely tractable in 1-2 weeks; the real work is a
    validation/mapping layer robust enough to catch a wrong assumption
    (e.g. an undetected sampling-rate error, the exact class of bug this
    project's own dataset already had) rather than silently miscalibrate
    against unfamiliar customer data — realistically 4-8 weeks for
    something genuinely trustworthy, not 1-2.
12. **Raw-PWM-only datasets** — ties directly to item 3; if a customer's
    logs don't already have pre-computed forces, the pipeline currently
    can't use them at all.
13. **Garbage-input robustness** — pre-flight checks catch *sub-optimal*
    data (bad sampling rate, poor excitation), but nothing has been tested
    against genuinely corrupted, wrong-unit, or wrong-vehicle input.

### C. Infrastructure & scale

14. **Parallelized restarts** — each restart is already independent for the
    stand-in backend; needs real multi-process wiring, genuinely a few
    days of work. For the HoloOcean backend specifically, there's a real,
    unconfirmed open question about whether multiple real Unreal Engine
    instances can even run simultaneously on one machine — if not, this
    item is really the cloud-orchestration item (15) in disguise, not a
    small addition.
15. **Cloud compute orchestration** — the multi-hour, manually-watched
    terminal sessions this whole project has relied on need to become a
    real, managed job queue.
16. **Multi-tenant infrastructure** — zero customer data isolation exists
    today.
17. **Web dashboard** — upload, watch progress, view results, download the
    simulator, no terminal required. The pipeline's clean
    `PipelineConfig`-in/`CalibrationReport`-out architecture was built in
    a way that should make this a wrapping exercise, not a rewrite.
18. **Experiment tracking/versioning** — real auto-backup protection now
    exists on both the stand-in and HoloOcean calibration scripts, closing
    the specific "silently overwritten, unrecoverable result" risk that
    caused a real loss earlier in this project. There is still no formal
    system (MLflow-style) tracking which spec version produced which
    result more broadly.
19. **CI coverage gap for HoloOcean-dependent code.** A real, automated
    test suite now exists (26 tests, covering the physics model, the data
    split logic, and the Gym environment's task logic against real
    instances, not mocks) and runs automatically on every push via GitHub
    Actions — but it can only cover code paths that don't need HoloOcean
    itself (no PyPI distribution, can't install in a standard CI runner).
    A real product still needs a strategy for testing the HoloOcean-
    dependent paths too (e.g. a self-hosted runner).

### D. ML/training capability

20. **Imitation learning from real logs** — reuses existing `Arena`
    infrastructure. A real design subtlety worth flagging: `Arena`'s
    observation space bakes in "position error to a goal waypoint," a
    concept that doesn't naturally exist in open, real logged trajectories
    never collected as goal-directed episodes — resolving this cleanly is
    a genuine design decision, not just wiring, and affects the honest
    time estimate (2-4 weeks if caught early; longer if discovered after
    building around the wrong assumption).
21. **Training algorithms beyond PPO** — SAC, TD3, etc.; currently
    hardcoded to one algorithm.
22. **Richer, more realistic task library.** Two new tasks were added this
    session — `station_keeping` (requires sustained presence near a
    target, not just a single touch) and `docking` (requires position,
    orientation alignment, and low speed simultaneously) — both real,
    tested, and wired into the actual training scripts, not just the
    demo. Both remain simplified proxies, though: no real contact or
    collision-aware docking against an actual structure yet.

### E. The two-mode simulation architecture (currently unaddressed)

23. **Clean-vs-noisy simulation modes.** Real, deliberate effort went into
    zeroing HoloOcean's sensor noise specifically so calibration would be
    reproducible. But a policy trained on a noiseless sim won't handle
    real sensor noise at deployment. The product needs two distinct,
    clearly-separated modes — clean/deterministic for calibration and
    validation, realistic/noisy for actually training and evaluating
    deployable policies. Only the first has really been built.

### F. Safety

24. **Emergency-stop / safe-envelope enforcement** — nothing in the current
    pipeline treats "don't let the vehicle do something dangerous" as a
    first-class concern; no guardrail layer exists.
25. **Collision avoidance as a baseline safety property** — currently only
    exists as something a policy could learn to value via a training
    task, not as an always-active safety layer independent of what the
    policy has learned.

### G. Validation & trust

26. **Real hardware validation** — still the single most important missing
    piece; nothing here has ever touched a live vehicle. Everything
    validated is against held-out logs.
27. **Automated per-customer confidence reporting** —
    `automated_identifiability_report.py` now exists as a real, working,
    structured-output tool (condition number, correlated pairs,
    multi-parameter near-null-space directions), confirmed working
    end-to-end this session, but it is not yet a polished, non-specialist-
    facing customer report — see item 30.
28. **Continuous real-world feedback loop** — deployment performance
    feeding back into recalibration doesn't exist as a concept yet.
29. **The rotational-physics weakness itself — now more precisely
    characterized, still genuinely open.** Real, direct kinematic evidence
    has ruled out a data-coverage explanation (the real dataset excites
    every relevant Coriolis coupling term comparably well). Two further,
    specific, well-reasoned hypotheses have since been tested directly
    against real HoloOcean physics and refuted: a custom per-DOF objective
    weighting (made the regression monotonically worse across three real
    segments), and a fix for a separately-confirmed non-physical inertia/
    buoyancy parameter drift that fed directly into the relevant Coriolis
    coupling term (the coupling term itself changed by close to the
    largest amount that could be tested, and the result was the single
    worst of any test run this session). A real, specific pattern has
    emerged across every test: sway and heave stay roughly stable while
    surge and all three rotational axes regress together, regardless of
    which fix is applied. The leading remaining hypothesis — every
    HoloOcean-backed test so far has trained on the same short, 15-second
    truncated window of real data — has not yet been tested, and doing so
    requires giving up the speed shortcut that made rapid iteration
    possible so far. This would matter directly to any customer whose task
    depends heavily on roll/yaw accuracy.

### H. Legal, IP, and compliance

30. **Export control / dual-use review.** Underwater autonomy plus AI is a
    genuinely ITAR/EAR-adjacent space for a US company, and defense-
    adjacent survey companies are an explicitly named target customer.
    Needs real legal review before, not after, that kind of customer
    signs on.
31. **Dataset licensing** — the calibration methodology has been validated
    against a public research dataset; its license should be confirmed to
    permit this kind of commercial use before it's central to a pitch.
32. **Customer data ownership & retention terms** — a hosted product means
    customers upload proprietary mission logs; needs real, explicit
    answers on ownership, retention, deletion rights, and any GDPR-style
    exposure for EU customers.

### I. Business/economics

33. **Unit economics.** Real, confirmed compute costs now exist (HoloOcean
    calibration running many real hours at real cloud rates) — this is
    answerable today, from data already in hand, not speculative. Worth
    checking whether the business model actually accounts for what this
    genuinely costs to run per customer.

---

## Summary: the honest priority order

**Before anything else scales:** resolve the performance-reliability
mystery (item 1) — a product this unpredictable can't be handed to a
customer regardless of how good the underlying science is. Real, direct
negative evidence has narrowed the search; a live `py-spy` catch of an
actual stall is the clear remaining step.

**Then, roughly in order of "worth it once there's real customer
traction":** general data ingestion, parallelized restarts, imitation
learning, vehicle-class bound templates, and the two-mode simulation
architecture (items 11, 14, 20, 6, 23) — each is well-scoped, moderate
effort, and directly serves a second real customer or a more trustworthy
deployed policy.

**Held back until a specific customer or lab is actually pulling for it:**
torpedo AUV support, cloud orchestration, the web dashboard, multi-
tenancy, real actuator dynamics, and off-diagonal generalization — each a
genuinely large, multi-day-plus effort, and building any of them
speculatively is the highest-risk use of time right now.

**Running in parallel with everything else, not sequenced after it:**
the legal/compliance items (30-32) and real hardware validation (26) —
both have long lead times independent of engineering effort, so starting
them early is worth more than deferring them until "the product is done."

---

*Session update note:* several items originally listed as build-scope gaps
have since been closed for real, not just attempted — automated pre-flight
checks, automated identifiability reporting, real regression-tested Gym
task variants, and output-versioning protection on both calibration
scripts are now genuinely wired into the actual production scripts (not
just standalone tools), each confirmed via live command output. The
rotational-weakness item (29) saw real, decisive negative progress — two
specific hypotheses eliminated with direct evidence — which sharpens
rather than resolves it as the most important open technical question.