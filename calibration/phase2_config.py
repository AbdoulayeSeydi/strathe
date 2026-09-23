"""
Phase 2 architecture decisions — made explicit and importable, not just
agreed on in conversation. Everything in Stage 2.1 onward (the Gym/"Arena"
wrapper, the replay comparison, the PPO training scripts) imports from
here rather than re-deciding these things inconsistently across files.

## Decision 1: held-out dates are REUSED from Phase 1, not re-split
Phase 2's "does calibration improve accuracy" comparison must use the SAME
held-out data Phase 1's calibration never saw — reusing a split the
calibration was already validated against, not introducing a new one that
could accidentally leak information forward. See PHASE1_HELDOUT_DATES
below — this is the exact pair used in every real calibration run this
session (findings #1-15 in PROJECT_STATUS.md), not a new choice.

## Decision 2 (REVISED): train on the FAST stand-in, validate on MuJoCo
ORIGINALLY: validate on real HoloOcean. REVISED after a full MuJoCo
investigation (separate session) found:
  - MuJoCo's real per-step overhead measured at ~1.9x the stand-in's, not
    HoloOcean's confirmed ~400x (time_holoocean_backend.py).
  - MuJoCo's teacher-forced mode (mujoco_backend.simulate_trajectory_mujoco,
    teacher_forced=True) matches the stand-in's own residual gradient
    EXACTLY (to 4 decimal places, confirmed at the specific point
    forward-sim's own gradient was found unstable) — calibration-safety
    confirmed directly, not assumed by analogy.
  - HoloOcean has a confirmed, unresolved defect: continuous episodes
    beyond a few hundred steps deadlock the Unreal-side IPC, forcing
    every real HoloOcean calibration this project has run to use
    artificially truncated (150-step) runs.
    calibration_output_holoocean_coriolis's own metrics.json
    (converged=False, nfev=345, elapsed_seconds=42838 — ~11.9 hours) is
    representative of this cost, not exceptional.

Still true, and still the core justification either way:
dynamics_model.compute_acceleration() is the IDENTICAL physics regardless
of backend (see dynamics_model.py / mujoco_backend.py — MuJoCo's own
native dynamics are fully overridden via wrench injection, confirmed to
machine precision against the stand-in). Training PPO directly against a
~400x-slower backend would cost the difference between hours and weeks
for a typical multi-million-step run — so:
  - TRAINING (Stage 2.2, 2.4): unchanged — always the stand-in backend
    (simulate_fn=None, i.e. dynamics_model.simulate_trajectory)
  - VALIDATION (Stage 2.3's real comparison, and any final check of a
    trained policy's behavior against real physics): MuJoCo-backed —
    mujoco_backend.make_mujoco_calibration_simulate_fn (teacher_forced=True)
    for any actual FITTING, mujoco_backend.make_mujoco_simulate_fn
    (forward-sim) for held-out EVALUATION of already-fitted parameters
    only — these are NOT interchangeable, see that module's own docstring
    for why forward-sim is unsafe as a fitting objective.

HoloOcean's remaining, narrower role: an occasional, expensive,
INDEPENDENT physics cross-check — not the primary validation loop, and
not routine. The stand-in and MuJoCo both simulate dynamics_model.py's
OWN equations; neither can ever catch an error IN those equations, only
in how faithfully they're numerically integrated. HoloOcean, with its own
separately-implemented hydrodynamics, is the one thing in this project
that could ever independently catch a genuine physics-model error — that
is specifically what it's for now, not day-to-day validation.

## Decision 2, addendum: B2.1's HoloOcean-relative criterion (see conversation)
Track B's own stated completion criterion for B2.1 reads "held-out RMSE
is in a sane range relative to the other two backends" -- written before
Decision 2's revision above, and taken completely literally it implies a
live three-way comparison including HoloOcean. Attempting to actually
run that comparison surfaced two separate, concrete blockers, not just
inconvenience:

  1. HoloOcean is not installed in this project's working environment,
     and this machine is macOS -- HoloOcean's own documented requirements
     list only "Linux or Windows 64bit". Even on a supported platform,
     installation is not a simple pip command: recent versions cannot be
     distributed via PyPI at all ("in accordance with the Unreal Engine
     EULA"), requiring an Epic Games account linked to GitHub, EULA
     acceptance, then a source install plus a separate multi-gigabyte
     world-binary download.

  2. The fallback approach -- reusing an EXISTING historical HoloOcean
     report and re-running its exact parameters through MuJoCo for a
     direct, live comparison -- is also blocked. Five historical
     replay_comparison.py output directories exist
     (replay_holoocean_{clean_inertia_test,segment3,segment4,ieff_fixed,
     segment5}_placeholder), but replay_comparison.py's own report
     schema never records which --calibrated-params file produced a
     given result, and none of the five retained a copy of their own
     source parameters. calibration_output_holoocean_coriolis -- the one
     directory this file's own Decision 2 text above cites by name, with
     a real metrics.json -- no longer exists on this machine at all. This
     is the same class of provenance gap PROJECT_STATUS.md already
     flagged for the segment5 report specifically (its
     calibrated_parameters.yaml didn't match its own
     calibrated_parameters_segment5_backup.yaml); here it's total
     absence rather than a mismatch, but the practical effect -- no
     verifiable link between a saved number and the parameters that
     produced it -- is the same.

Given both paths are blocked, this criterion is being closed on the
strength of Decision 2's OWN evidence, not a fresh live comparison:
MuJoCo and the stand-in agree on B2.1's own held-out result to a ~1.00x
ratio (0.320402 vs. 0.320404) -- close enough that MuJoCo can be treated
as inheriting the stand-in's own relationship to HoloOcean, without a
separate direct measurement. An indirect substitute exists too: a
--no-holoocean (stand-in) replay of HoloOcean-fit parameters
(calibration_output_holoocean_coriolis, while it still existed) gave
0.2575 -- a broadly comparable magnitude, not a contradiction.

One honest, UNRESOLVED caveat this investigation surfaced, not silently
dropped: the four non-segment5 historical reports (clean_inertia_test,
segment3, segment4, ieff_fixed -- all testing earlier calibration
attempts, NOT calibrated_parameters_a1.yaml) show HoloOcean's own
scale-normalized calibrated-vs-default improvement ranging from -21.5%
to +8.0% -- markedly worse than the ~30-40% improvements this project has
consistently seen via stand-in/MuJoCo. Without the source parameters,
this cannot be attributed to either (a) those specifically being weaker,
earlier attempts, or (b) a genuine, systematic HoloOcean-vs-stand-in
divergence this project's calibrated physics hasn't caught. This
ambiguity is exactly the kind of thing HoloOcean's narrowed
"independent physics cross-check" role above exists to eventually
resolve -- it just hasn't been resolved yet, and B2.1's closure here
should not be read as having resolved it.

RECOMMENDATION for any future rigorous three-way comparison (e.g. a
result_type="final" CLAIM_A, per Decision 3's own labeling discipline):
(1) HoloOcean would need a working install on an actually-supported
platform, not this machine as currently configured; (2) any future
HoloOcean-backed run should have its own output directory save a literal
copy of its --calibrated-params input alongside the report -- the
concrete, fixable gap that made all five existing historical directories
unverifiable against a fresh run.

## Decision 3: two SEPARATE claims, not one conflated claim
Conflating these would overstate what's actually proven — see
PROJECT_STATUS.md's habit of stating exactly what's confirmed vs.
suggestive, applied here too.

  CLAIM_A — the rigorous, direct extension of Phase 1's calibration work:
    replay the SAME real logged thruster commands OPEN-LOOP through both
    the default-physics and calibrated-physics variants of the wrapped
    env, compute RMSE against real logged trajectories for both. This is
    methodologically identical to everything already validated in Phase 1
    (findings #1-15) — same kind of comparison, now happening inside the
    Gym wrapper instead of the raw calibration pipeline. This IS the MVP's
    literal stated Phase 2 acceptance criterion.

  CLAIM_B — complementary, deliberately softer:
    trained PPO policies (closed-loop, reacting to state, not replaying
    fixed real commands) behave sensibly and DIFFERENTLY under calibrated
    vs. default physics. Demonstrates the tool produces a usable,
    trainable RL environment — a real capability claim — but is NOT a
    second accuracy proof, since there's no live hardware yet to validate
    a trained policy's real-world behavior against (that's Phase 2.5's
    job, opportunistic/pending real vehicle access).

Any result reported from this phase must be labeled CLAIM_A or CLAIM_B
explicitly — never presented as one undifferentiated "Phase 2 result."

## Decision 3, addendum: CLAIM_C -- multi-agent comm behavior (see conversation)
Track C's own C0.1-C0.3 work (MuJoCo as shared pose source, then as each
agent's own live dynamics backend) produced a result CLAIM_A and CLAIM_B
don't cover: whether an already-trained, stand-in-trained MULTI-agent
policy's comm-degradation behavior transfers to MuJoCo dynamics. This is
CLAIM_C, added here rather than folded into CLAIM_B, because it depends
on evidence CLAIM_B never required (a second, independently-computed
dynamics backend) and answers a different question (does the stand-in's
own trained behavior GENERALIZE across backends, not just "does training
work at all").

Evidence (C0.3, see conversation): four already stand-in-trained
policies (formation_hold_ppo_curriculum_{v5,L1,L2,L3a}, one per comm
severity, no MuJoCo-based retraining -- per Decision 2, training stays
stand-in-only) were re-evaluated on MuJoCo-backed TwoAgentOrchestrator
dynamics (MuJoCoArenaCore, each agent its own independent single-body
MuJoCo instance):

  1. Baseline sanity (evaluate_baseline_sanity_mujoco.py): all four
     trained policies clearly beat a random baseline on MuJoCo, on
     mean_reward and mean_max_consecutive_in_tolerance (the two least
     noisy metrics at n=20 episodes/cell) -- success_rate alone was
     noisy at this sample size (L2: 5% vs. random's 15%) and should not
     be read in isolation, per A1.9's own decision-rule precedent.

  2. Comm-degradation sweep (evaluate_comm_degradation_curve_mujoco.py
     vs. the stand-in's own evaluate_comm_degradation_curve.py, both at
     the full 32-point x 5-comm-seed x 5-grid-seed scale): delivery rate
     matched the stand-in almost exactly at every severity (L0: 0.992 vs.
     0.993; L3a: 0.397 vs. 0.395 -- expected, since delivery depends only
     on the comm channel's own RNG, not the dynamics backend). Collision
     rate showed the SAME qualitative shape on both backends (a dip at
     L1, then a sharp climb through L2/L3a) at different absolute
     numbers. delta_physics_belief_error -- the specific metric the
     product's moat claim rests on -- was near-zero and sign-inconsistent
     at L0/L1 on BOTH backends, but showed a consistent, similarly-sized
     negative gap at L2 (-0.017 stand-in vs. -0.020 MuJoCo) and L3a
     (-0.022 vs. -0.024): the calibrated agent's belief-error advantage
     under real communication degradation held up QUANTITATIVELY, not
     just directionally, across the backend switch.

What this does NOT establish, stated explicitly per Decision 3's own
discipline: success_rate and delta_physics_off_axis did NOT transfer
cleanly between backends -- but this is confounded by L0-L3a being four
SEPARATELY-trained policies (each reflecting its own training run's own
idiosyncrasies), not evidence against the backend transfer itself. Nor
does this claim anything about whether MuJoCo-based TRAINING would
produce equally-good or better policies -- no such training has been
run, per Decision 2's own training-backend rule; this is purely a
transfer-of-an-already-trained-policy result.

CLAIM_C is layered on top of the other two, not independent of them: it
presupposes CLAIM_A's own physics-fidelity work (MuJoCo is only a
meaningful validator here because its dynamics were already confirmed to
match the stand-in to machine precision -- see Decision 2), and extends
CLAIM_B's single-agent "trained policy behaves sensibly" result to the
multi-agent, cross-backend case specifically. Any result reported from
Track C must be labeled CLAIM_C explicitly, same discipline as A/B.
"""
from __future__ import annotations

from typing import Optional

# Decision 1 — the exact pair used in every real calibration run this
# session (run_calibration_holoocean.py's own CLI default matches this).
PHASE1_HELDOUT_DATES = ["16-02-2024", "19-02-2024"]
PHASE1_TRAIN_DATES = ["12-01-2024", "14-02-2024"]

# Decision 2 (REVISED) — which backend each stage of Phase 2 must use.
# Referenced by name (not just a bare None/simulate_fn), so a future
# reader can see WHY a particular stage picked its backend, not just
# infer it from the code.
TRAINING_BACKEND = "stand_in"  # dynamics_model.simulate_trajectory (fast) — unchanged

VALIDATION_BACKEND = "mujoco"
# mujoco_backend.make_mujoco_calibration_simulate_fn (teacher_forced=True)
# for fitting; mujoco_backend.make_mujoco_simulate_fn (forward-sim) for
# held-out evaluation of already-fitted parameters only. REVISED from
# "holoocean" — see Decision 2 above for the concrete evidence.

INDEPENDENT_CROSSCHECK_BACKEND = "holoocean"
# Occasional, expensive, deliberately NOT routine — see Decision 2's own
# reasoning for why this role is narrower than "the" validation backend
# it used to be. holoocean_backend.make_holoocean_simulate_fn.

# F5.2: The authoritative Claim C honest-caveats text, pulled directly from
# the Decision 3 addendum above. run_fleet_experiment.py imports this to
# include it verbatim in every summary_report.md, rather than maintaining a
# separately-written copy that can drift from the source of truth.
# Any edit to the caveats above should be reflected here.
# WITHDRAWN 2026-09-23 (PROJECT_STATUS.md finding #46): delta_physics_belief_error
# measures how far the TEAMMATE moved since its last beacon, which depends on
# how hard the policy drives each vehicle slot, not on either agent's physics
# (the same negative "advantage" appears with no calibration difference at
# all). Kept so fleet v1 runs still render; superseded by fleet v2.
CLAIM_C_WITHDRAWN_NOTICE = (
    "> **Claim C is withdrawn (PROJECT_STATUS.md finding #46).** `delta_physics_belief_error` "
    "measures how far the teammate moved since its last beacon. No agent in fleet v1 uses its "
    "physics to predict anything, and the same negative 'advantage' appears when both vehicles "
    "have identical physics (-0.043 with both generic, -0.011 with both calibrated). Use fleet v2 "
    "(`run_fleet_experiment.py`) for calibrated-vs-generic comparisons.\n\n"
)

CLAIM_C_HONEST_CAVEATS = CLAIM_C_WITHDRAWN_NOTICE + (
    "What Claim C does **not** establish (from `phase2_config.py` Decision 3 addendum, "
    "stated explicitly per Decision 3's own discipline):\n\n"
    "- `success_rate` and `delta_physics_off_axis` did **not** transfer cleanly between "
    "backends — but this is confounded by L0–L3a being four **separately-trained** policies "
    "(each reflecting its own training run's idiosyncrasies), not evidence against the "
    "backend transfer itself.\n"
    "- This result claims nothing about whether MuJoCo-based **training** would produce "
    "equally-good or better policies. No such training has been run, per Decision 2's "
    "training-backend rule. This is purely a transfer-of-an-already-stand-in-trained-policy "
    "result.\n"
    "- Claim C is layered on top of Claims A and B. It presupposes Claim A's own "
    "physics-fidelity work (MuJoCo is only a meaningful validator here because its dynamics "
    "were already confirmed to match the stand-in), and extends Claim B's single-agent "
    "'trained policy behaves sensibly' result to the multi-agent, cross-backend case.\n"
    "- Any result reported from Track C must be labeled CLAIM_C explicitly, same discipline "
    "as A/B.\n"
    "- **Missing causal link (identified by F5.4 dogfood review):** this experiment "
    "demonstrates one link in the causal chain — calibrated physics → more accurate "
    "teammate-position belief under degraded comm (`delta_physics_belief_error`). It does "
    "**not** establish the full chain through to fleet task success: lower belief error → "
    "better coordination behavior → higher `success_rate` / fewer collisions. The "
    "`success_rate` in these results is not broken down calibrated-vs-generic, so the "
    "experiment cannot support the claim that calibrated physics improves fleet coordination "
    "outcomes. That link requires a separate, controlled experiment."
)


class Claim:
    """Decision 3 (and its addendum) — every Phase 2 result must be
    tagged with one of these, never left ambiguous about which claim
    (and which evidentiary strength) it belongs to."""
    A_OPEN_LOOP_REPLAY_ACCURACY = "CLAIM_A: open-loop replay RMSE vs. real data — the rigorous, direct MVP acceptance criterion"
    B_TRAINED_POLICY_BEHAVIOR = "CLAIM_B: trained PPO policy behavior under calibrated vs. default physics — complementary, not a second accuracy proof (no live hardware yet — see Phase 2.5)"
    C_MULTIAGENT_COMM_BEHAVIOR = "CLAIM_C: multi-agent comm-degradation behavior under MuJoCo-validated physics — complementary to A and B, not a third accuracy proof (validates cross-backend TRANSFER of an already stand-in-trained policy, not MuJoCo-based training or new physics accuracy)"

    @staticmethod
    def b_trained_policy_behavior(algorithm: str = "PPO") -> str:
        """Same message as B_TRAINED_POLICY_BEHAVIOR, but with the real
        algorithm name substituted in - B_TRAINED_POLICY_BEHAVIOR itself
        is kept unchanged (hardcoded "PPO") for backward compatibility
        with anything that may already reference the bare constant, but
        train_multiseed.py now calls this instead, so a SAC/TD3 run's
        printed header and saved report both say the real algorithm that
        actually ran, not always "PPO" (a real, confirmed bug found and
        fixed - see PROJECT_STATUS.md)."""
        return (f"CLAIM_B: trained {algorithm.upper()} policy behavior under calibrated "
                f"vs. default physics — complementary, not a second accuracy proof "
                f"(no live hardware yet — see Phase 2.5)")

    _BACKEND_DISPLAY_NAMES = {"mujoco": "MuJoCo", "holoocean": "HoloOcean", "stand_in": "the stand-in"}
    # BUG FOUND AND FIXED (see conversation): the first version of this
    # method used validation_backend.upper() directly, mirroring
    # b_trained_policy_behavior's own algorithm.upper() pattern -- which
    # only happens to work there because "PPO" is naturally an all-caps
    # acronym. "MuJoCo" and "HoloOcean" are stylized proper nouns, not
    # acronyms, so .upper() produced "MUJOCO"/"HOLOOCEAN", which did NOT
    # match this project's own consistent capitalization (confirmed by
    # test_claim_c_default_matches_constant failing against the hardcoded
    # constant, which does say "MuJoCo"). An explicit display-name lookup
    # is used instead; a backend key not in this dict falls back to
    # .upper() rather than raising, so a genuinely new backend name still
    # produces something readable instead of crashing this claim entirely.

    @staticmethod
    def c_multiagent_comm_behavior(validation_backend: str = "mujoco") -> str:
        """Same message as C_MULTIAGENT_COMM_BEHAVIOR, but with the real
        validation backend substituted in - C_MULTIAGENT_COMM_BEHAVIOR
        itself is kept unchanged (hardcoded "MuJoCo") for backward
        compatibility, mirroring b_trained_policy_behavior's own
        algorithm parameter. Exists because HoloOcean remains this
        project's own independent physics cross-check (see Decision 2's
        INDEPENDENT_CROSSCHECK_BACKEND) - if a future multi-agent comm
        run is ever validated against it instead of MuJoCo, that caller
        should say so explicitly rather than this claim silently saying
        "MuJoCo" regardless of what actually ran."""
        display_name = Claim._BACKEND_DISPLAY_NAMES.get(validation_backend.lower(), validation_backend.upper())
        return (f"CLAIM_C: multi-agent comm-degradation behavior under "
                f"{display_name}-validated physics — complementary to A and B, "
                f"not a third accuracy proof (validates cross-backend TRANSFER of an already "
                f"stand-in-trained policy, not {display_name}-based training or "
                f"new physics accuracy)")


def assert_uses_phase1_split(heldout_dates: list) -> None:
    """Fail loudly if any Phase 2 script is about to use a DIFFERENT
    held-out split than Phase 1's — call this at the top of any script
    that takes --heldout-dates or similar, rather than silently trusting
    a caller-supplied value."""
    if sorted(heldout_dates) != sorted(PHASE1_HELDOUT_DATES):
        raise ValueError(
            f"Phase 2 must reuse Phase 1's exact held-out split "
            f"{PHASE1_HELDOUT_DATES}, not {heldout_dates} — see phase2_config.py "
            f"Decision 1 for why. If you genuinely need a different split, update "
            f"PHASE1_HELDOUT_DATES here explicitly, don't silently pass a new one."
        )


def check_result_type(result_type: str, metrics_path: Optional[str] = None) -> None:
    """
    The market-facing safeguard: refuses to let a "final" report be
    generated from a calibration that wasn't actually confirmed converged.

    result_type must be exactly "placeholder" or "final" — no default,
    every Stage 2.3/2.4 script must state this explicitly, forcing a
    deliberate choice rather than a silent one.

    If result_type == "final", metrics_path (the SOURCE calibration's own
    metrics.json — see run_calibration_holoocean.py / run_calibration.py's
    output) is REQUIRED and must show converged=True. Refuses loudly
    otherwise — this is the concrete fix for the exact class of mistake
    this session already made more than once (a stale/wrong file silently
    treated as authoritative). A wrong or mislabeled number going to
    market is a real cost, not an inconvenience.
    """
    if result_type not in ("placeholder", "final"):
        raise ValueError(
            f"result_type must be exactly 'placeholder' or 'final', got {result_type!r} — "
            f"no default, every Stage 2.3/2.4 script must state this explicitly."
        )
    if result_type == "final":
        if metrics_path is None:
            raise ValueError(
                "result_type='final' requires metrics_path (the source calibration's own "
                "metrics.json) so convergence can actually be verified, not assumed."
            )
        import json
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        if not metrics.get("converged", False):
            raise ValueError(
                f"REFUSING to label this a FINAL result — {metrics_path} shows "
                f"converged={metrics.get('converged')!r}, not True. This looks like a "
                f"placeholder/proof-of-concept result being mistakenly promoted to final. "
                f"Use result_type='placeholder' instead, or verify the source calibration "
                f"actually converged first."
            )


def output_dir_for(base_name: str, result_type: str) -> str:
    """Enforces the placeholder/_final output-directory naming convention
    everywhere, rather than leaving each script to (inconsistently)
    remember to do this itself. Only validates result_type is a legal
    STRING here — the full metrics-verification check (check_result_type)
    belongs at the point the actual analysis/report is generated, not at
    directory-naming time, since a metrics_path may not even be decided
    yet when a caller is just picking where output should go."""
    if result_type not in ("placeholder", "final"):
        raise ValueError(
            f"result_type must be exactly 'placeholder' or 'final', got {result_type!r}"
        )
    return f"{base_name}_{result_type}"
