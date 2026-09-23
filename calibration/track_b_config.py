"""
Track B architecture decisions — made explicit and importable, not just
agreed on in conversation, following the exact discipline established by
phase2_config.py's own Decision 2 (train on the fast stand-in, validate on
real HoloOcean): document the choice, back it with concrete evidence, and
expose it as an importable constant so downstream code (mujoco_backend.py,
B1 onward) references the SAME decision rather than silently re-deciding
or drifting from it.

## Decision 1: full-override, not native-rigid-body, for MuJoCo's dynamics
B0.5 and B0.6 tested BOTH candidate approaches for wiring the stand-in's
already-validated calibrated physics (dynamics_model.compute_acceleration())
into MuJoCo, using four identical (velocity, force) test cases at identity
orientation:

  - B0.5 (full override): cancel MuJoCo's entire bias force (gravity +
    Coriolis) and inject a wrench computed as M_mj @ desired_v_dot +
    qfrc_bias, using MuJoCo's OWN mass matrix read back at runtime. This
    reproduces the stand-in's calibrated acceleration EXACTLY -- confirmed
    to machine precision (1e-16 to 1e-17) across all four test cases (see
    b0_5_spike_full_override.py's actual run output).

  - B0.6 (native rigid-body): inject only the force-like hydrodynamic
    terms (damping, buoyancy) and let MuJoCo's own placeholder mass matrix
    handle gravity and Coriolis natively. Compared directly against B0.5's
    calibrated reference (see b0_6_spike_native_rigid_body.py):

      at rest (mass-matrix structure only -- added mass + cross-coupling,
      no Coriolis involved at all):
        pure surge force:          max abs diff = 0.312  (vs. 0.134 calibrated -- >100% relative error)
        combined force+torque:     max abs diff = 0.166  (vs. 1.366 calibrated)

      nonzero velocity (~0.5 m/s translational, ~0.02-0.1 rad/s rotational
      -- SMALLER than the velocity ranges A1.9's trained PPO policies
      routinely produced, e.g. pitch rates up to +/-1.5 rad/s):
        damping+Coriolis only:     max abs diff = 3.827
        fully general case:        max abs diff = 3.573

NOT a case of simply following the task's stated "recommended default"
without checking -- both approaches were independently spiked and
quantitatively compared. The evidence supports full-override for two
separate, independent reasons:

  1. Added mass and cross-coupling (Xq_dot, Zq_dot) are FIRST-ORDER
     effects for this vehicle, not small corrections -- confirmed by the
     >100% relative error present even at rest, with zero velocity and
     zero Coriolis involved.
  2. The native-rigid-body mismatch grows roughly 10-20x once realistic
     motion is introduced, AT VELOCITIES SMALLER than what this project's
     own trained policies actually produce -- meaning the real,
     product-relevant operating regime is likely to show WORSE divergence
     than even B0.6's own numbers, not better.

What this gives up: MuJoCo's own native rigid-body Coriolis/dynamics
computation goes unused for the vehicle's own motion. What it does NOT
give up: MuJoCo's native COLLISION detection (B3.4) is a separate,
purely geometric computation that runs regardless of how forces are
applied to a body -- choosing full-override for the dynamics does not
forfeit that future capability.
"""
from __future__ import annotations

# Decision 1 -- referenced by name (not a bare boolean/None), so a future
# reader of mujoco_backend.py can see WHY this specific approach was
# chosen, not just infer it from the code. mujoco_backend.py (B1.1) must
# implement exactly this approach, not silently drift to the alternative.
MUJOCO_DYNAMICS_INTEGRATION_APPROACH = "full_override"

# The rejected alternative, kept as a named constant (not just a comment)
# so any future code that wants to reference "the other approach we
# considered" points at the same string, not a re-typed guess.
MUJOCO_DYNAMICS_INTEGRATION_APPROACH_REJECTED = "native_rigid_body"


def assert_uses_full_override(approach: str) -> None:
    """Fail loudly if mujoco_backend.py (or anything built on top of it)
    is about to use a DIFFERENT integration approach than Decision 1 --
    call this at the top of mujoco_backend.py's own wrench-computation
    function, rather than silently trusting the approach was implemented
    correctly."""
    if approach != MUJOCO_DYNAMICS_INTEGRATION_APPROACH:
        raise ValueError(
            f"Track B must use '{MUJOCO_DYNAMICS_INTEGRATION_APPROACH}' for "
            f"MuJoCo dynamics integration, not {approach!r} -- see "
            f"track_b_config.py Decision 1 for why (B0.5/B0.6's quantified "
            f"spike-test comparison). If you genuinely need to revisit this, "
            f"update MUJOCO_DYNAMICS_INTEGRATION_APPROACH here explicitly, "
            f"don't silently pass a different value."
        )