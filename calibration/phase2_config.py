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

## Decision 2: train on the FAST stand-in, validate on real HoloOcean
NOT a compromise — dynamics_model.compute_acceleration() is the IDENTICAL
physics either way (see dynamics_model.py / holoocean_backend.py — the
HoloOcean backend calls this same function every tick, it only replaces
the INTEGRATOR, not the physics). Training PPO directly against real
HoloOcean would cost ~400x more wall-clock time per step (confirmed,
time_holoocean_backend.py) — for a typical multi-million-step PPO run,
the difference between "hours" and "weeks". So:
  - TRAINING (Stage 2.2, 2.4): always use the stand-in backend
    (simulate_fn=None, i.e. dynamics_model.simulate_trajectory)
  - VALIDATION (Stage 2.3's real comparison, and any final check of a
    trained policy's behavior against real HoloOcean physics): use the
    HoloOcean-backed simulate_fn (holoocean_backend.make_holoocean_simulate_fn)

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
"""
from __future__ import annotations

from typing import Optional

# Decision 1 — the exact pair used in every real calibration run this
# session (run_calibration_holoocean.py's own CLI default matches this).
PHASE1_HELDOUT_DATES = ["16-02-2024", "19-02-2024"]
PHASE1_TRAIN_DATES = ["12-01-2024", "14-02-2024"]

# Decision 2 — which backend each stage of Phase 2 must use. Referenced by
# name (not just a bare None/simulate_fn), so a future reader can see WHY
# a particular stage picked its backend, not just infer it from the code.
TRAINING_BACKEND = "stand_in"     # dynamics_model.simulate_trajectory (fast)
VALIDATION_BACKEND = "holoocean"  # holoocean_backend.simulate_trajectory_holoocean (slow, real)


class Claim:
    """Decision 3 — every Phase 2 result must be tagged with one of these,
    never left ambiguous about which claim (and which evidentiary
    strength) it belongs to."""
    A_OPEN_LOOP_REPLAY_ACCURACY = "CLAIM_A: open-loop replay RMSE vs. real data — the rigorous, direct MVP acceptance criterion"
    B_TRAINED_POLICY_BEHAVIOR = "CLAIM_B: trained PPO policy behavior under calibrated vs. default physics — complementary, not a second accuracy proof (no live hardware yet — see Phase 2.5)"

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