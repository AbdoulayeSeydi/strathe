# Product Vision — Calibrated Digital Twins for Marine Autonomy

*Companion to PROJECT_STATUS.md (the research log) — this document answers
"what are we actually building, for whom, and what's the gap between what
exists today and the full product." Written at the same level of detail
as the findings in PROJECT_STATUS.md, not an executive summary.*

*Updated after MVP completion — Part 3 substantially expanded with the
full post-MVP gap list.*

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
  from this session's own sampling-rate investigation, currently a
  standalone script, not an automated pipeline stage)
- Automatically check maneuver diversity/excitation across every DOF
  before committing compute to a calibration that can't succeed
- Flag data-quality problems before the customer waits hours for a result
  that was never going to be trustworthy

**Step 3 — Automated calibration**

Runs the real calibration methodology as a **managed cloud job**, not a
person watching a terminal:

- Real, structured progress a customer can check via a dashboard
- Parallelized multi-restart, cutting wall-clock time roughly N×
- **Must be reliable** — a customer-facing product cannot have the
  still-unexplained performance variance this MVP has (a single
  evaluation sometimes taking 100x longer than normal, for reasons never
  fully diagnosed)

**Step 4 — Calibration report**

The automated, customer-facing equivalent of this project's own research
log:

- Per-DOF accuracy numbers, held-out-by-date validation results
- Honest confidence flags — e.g. "your data doesn't sufficiently excite
  roll; this axis's calibration should not be trusted yet"
- Multi-parameter identifiability warnings, surfaced automatically
- **Written for a non-specialist**, not someone with this project's own
  depth of understanding of Jacobians and SVD near-null-space directions

**Step 5 — Simulator delivery**

- A calibrated parameter file
- A ready-to-use Gym-compatible training environment, pre-loaded with the
  customer's calibrated physics
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
to end — pre-flight checks, calibration, a real Gym-environment smoke
test — genuinely watchable in minutes. The *real*, reportable-quality
pipeline is a documented 6-step sequence of separate scripts (pre-flight →
stand-in fit → HoloOcean refinement → Claim A validation → Claim B
training → optional deeper diagnostics), still requiring someone
comfortable with the command line and real patience (the HoloOcean
refinement step alone can run for many hours).

### 2.2 What the MVP genuinely, honestly proves

**The methodology works.** Calibration measurably improves how closely a
simulator tracks real vehicle behavior — held-out-by-date validation,
multi-restart stability, real bugs found and fixed with real evidence at
every step (a structural parameter-identifiability bug, a sampling-rate
error, a Forward-Euler numerical instability, a Coriolis-physics gap, a
non-physical inertia drift, several more). This is a real, defensible
research result. It is **not** yet something a robotics engineer at
another company could pick up and run on *their* vehicle without personal
guidance.

---

## Part 3 — Detailed gap list: MVP → full product

Organized by category, roughly in priority tiers. Each item states not
just what's missing but why it matters.

### Tier 0 — reliability, before anything else scales

1. **The unexplained performance mystery.** A single evaluation has been
   observed taking up to ~4 hours against a confirmed ~100-second
   baseline, for reasons never fully diagnosed despite extensive
   investigation this session (isolated tests, live process profiling
   attempts, a prepared-but-not-yet-decisive timing log). This is not a
   "someday" item — it's a real reliability blocker for any future
   customer-facing product, not just an inconvenience during development.
   Needs a genuinely different debugging approach (e.g. sustained,
   hours-long live profiling) before broad rollout.

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
    to one dataset's exact column format today.
12. **Raw-PWM-only datasets** — ties directly to item 3; if a customer's
    logs lack pre-computed forces, the pipeline currently can't use them
    at all.
13. **Garbage-input robustness** — pre-flight checks catch *sub-optimal*
    data (bad sampling rate, poor excitation), but nothing has been tested
    against genuinely corrupted, wrong-unit, or wrong-vehicle input.

### C. Infrastructure & scale

14. **Parallelized restarts** — each restart is already independent;
    needs real multi-process/multi-machine wiring, not new algorithmic
    work.
15. **Cloud compute orchestration** — the multi-hour, manually-watched
    terminal sessions this whole project has relied on need to become a
    real, managed job queue.
16. **Multi-tenant infrastructure** — zero customer data isolation exists
    today.
17. **Web dashboard** — upload, watch progress, view results, download the
    simulator, no terminal required. The pipeline's clean
    `PipelineConfig`-in/`CalibrationReport`-out architecture was built in
    a way that should make this a wrapping exercise, not a rewrite.
18. **Experiment tracking/versioning** — a real auto-backup mechanism now
    exists, but there's no formal system (MLflow-style) tracking which
    spec version produced which result.
19. **CI coverage gap for HoloOcean-dependent code** — the real, passing
    test suite built this session can only cover code paths that don't
    need HoloOcean itself (no PyPI distribution, can't install in a
    standard CI runner). A real product needs a strategy for testing the
    HoloOcean-dependent paths too (e.g. a self-hosted runner).

### D. ML/training capability

20. **Imitation learning from real logs** — a known recipe, reuses existing
    Gym-environment infrastructure, not yet built.
21. **Training algorithms beyond PPO** — currently hardcoded to one
    algorithm.
22. **Richer, more realistic task library** — the docking task built this
    session is a simplified proxy (position + orientation + speed
    thresholds), not real contact/collision-aware docking against an
    actual structure.
23. **Non-expert-facing reporting** — the deeper diagnostic tools (Jacobian
    correlation, SVD near-null-space analysis) are built for someone with
    this project's own depth of understanding, not a lab without in-house
    research bandwidth — the explicitly stated target customer.

### E. The two-mode simulation architecture (currently unaddressed)

24. **Clean-vs-noisy simulation modes.** Real, deliberate effort went into
    zeroing HoloOcean's sensor noise specifically so calibration would be
    reproducible. But a policy trained on a noiseless sim won't handle
    real sensor noise at deployment. The product needs two distinct,
    clearly-separated modes — clean/deterministic for calibration and
    validation, realistic/noisy for actually training and evaluating
    deployable policies. Only the first has really been built.

### F. Safety

25. **Emergency-stop / safe-envelope enforcement** — nothing in the current
    pipeline treats "don't let the vehicle do something dangerous" as a
    first-class concern; no guardrail layer exists.
26. **Collision avoidance as a baseline safety property** — currently only
    exists as something a policy could learn to value via a training
    task, not as an always-active safety layer independent of what the
    policy has learned.

### G. Validation & trust

27. **Real hardware validation** — still the single most important missing
    piece; nothing here has ever touched a live vehicle. Everything
    validated is against held-out logs.
28. **Continuous real-world feedback loop** — deployment performance
    feeding back into recalibration doesn't exist as a concept yet.
29. **The rotational-physics weakness itself** — still open; a real
    fidelity gap that would matter directly to any customer whose task
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
customer regardless of how good the underlying science is.

**Then, roughly in order of "worth it once there's real customer
traction":** general data ingestion, parallelized restarts, imitation
learning, vehicle-class bound templates, and the two-mode simulation
architecture (items 11, 14, 20, 6, 24) — each is well-scoped, moderate
effort, and directly serves a second real customer or a more trustworthy
deployed policy.

**Held back until a specific customer or lab is actually pulling for it:**
torpedo AUV support, cloud orchestration, the web dashboard, multi-
tenancy, real actuator dynamics, and off-diagonal generalization — each a
genuinely large, multi-day-plus effort, and building any of them
speculatively is the highest-risk use of time right now.

**Running in parallel with everything else, not sequenced after it:**
the legal/compliance items (30-32) and real hardware validation (27) —
both have long lead times independent of engineering effort, so starting
them early is worth more than deferring them until "the product is done."