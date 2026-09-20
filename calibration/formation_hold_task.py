"""
FormationHoldTask — A1.3's frozen design, implemented.

## Called directly, not through Arena.step()
Task.step(self, arena, distance, truncated) assumes exactly one Arena/one
ArenaCore. Formation-hold is inherently two-agent (needs both agents'
actions for the asymmetric control term, and ground-truth separation no
single Arena can compute alone) — so this inherits from Task for
interface/pattern consistency (same extra_obs_dim()/extra_obs()
extension point, same terminal-reason vocabulary), but A1.6's
orchestration loop calls FormationHoldTask.step() directly, with an
extended signature, not via Arena.step().

## THE CRITICAL INVARIANT, stated explicitly per design review
All reward and termination calculations use GROUND-TRUTH simulator
state (the true separation between both ArenaCore instances' real
positions). Agent OBSERVATIONS are communication/observation-limited
(delayed, dropped, or bandwidth-throttled per the comm channel) and do
NOT affect environment evaluation. The environment judges the true
outcome; the agent has to learn to act well despite not always knowing
where its partner currently is. Conflating the two would mean a
well-behaved policy could be penalized for a "bad outcome" that was
actually just stale information — the wrong thing to punish.

Concretely: `separation` passed into step() is always ground truth,
supplied by the orchestrator directly from both ArenaCore.state.position
values. step() ALSO accepts `comm_revealed_other_position` as an explicit
parameter, per A1.5's literal requirement — but it is used for nothing
but an audit-trail entry in info; the reward/termination math never reads
it. This is proven BEHAVIORALLY, not just by convention — see
test_step_reward_and_termination_are_invariant_to_comm_position, which
calls step() twice with identical everything except this one parameter
and asserts the reward/terminated/reward_components output is bit-for-bit
identical either way. extra_obs_for_agent() remains the real mechanism
for building each agent's actual comm-limited observation (called once
per agent, since the two directions of comm may reveal different things)
— this parameter's addition to step() doesn't change that division of
responsibility, it only satisfies the literal interface requirement while
keeping the invariant provably intact underneath.

## Reward equation (frozen)
R_i,t = -w_sep * |d_t - d_target|
        - w_ctrl * ||u_i,t||^2
        - w_prox * f(d_t)
        - w_coll * 1[d_t < d_safe]
      + success_bonus (once, on the step formation is first achieved)

where f(d) = max(0, (d_influence - d) / d_influence) when d < d_influence,
else 0 — the same continuous "getting too close" shaping pattern
ObstacleAvoidanceTask already uses for f(d_obstacle), reused rather than
invented.

d_t is symmetric (both agents share it); ||u_i,t||^2 is asymmetric (each
agent's own action). The combined per-step reward the orchestrator sums
into one scalar for the centralized policy is R_a,t + R_b,t.

## Success vs. reward, kept separate per design review
Success is a STATE (consecutive steps within tolerance), tracked
internally exactly like StationKeepingTask's _consecutive_steps_in_radius
— not baked into the reward's conceptual definition. The reward function
only ever adds success_bonus once, on the transition step, and reports it
as a separate, auditable component in info["reward_components"] rather
than folding it silently into one opaque scalar.

## Termination, collision-first (matching ObstacleAvoidanceTask's
established priority order)
    d_t < d_safe                              -> terminated, COLLISION
    else |d_t - d_target| < eps for N steps   -> terminated, FORMATION_ACHIEVED
    else                                       -> not terminated (Arena's
                                                   own truncated/timeout
                                                   handling applies)
A simultaneous collision + formation-achieved on the same step is
COLLISION, never silently overridden by success — same invariant
ObstacleAvoidanceTask already established for its own two termination
conditions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Reused from arena.py's existing terminal-reason vocabulary — NOT a
# separate, parallel constant set (A1.4's explicit requirement).
#
# Made LOUD deliberately: a silent try/except fallback here would make
# "imported from arena.py" and "arena.py's edit was never made" look
# identical from the outside — every test would still pass either way,
# which defeats the point of A1.4 actually requiring the edit. This
# fails LOUDLY with a clear, actionable message instead.
try:
    from arena import TERMINAL_REASON_COLLISION, TERMINAL_REASON_FORMATION_ACHIEVED
    _IMPORTED_FROM_ARENA = True
except ImportError as e:
    raise ImportError(
        "Could not import TERMINAL_REASON_COLLISION / TERMINAL_REASON_FORMATION_ACHIEVED "
        "from arena.py. Either arena.py is missing the one required new line:\n"
        '    TERMINAL_REASON_FORMATION_ACHIEVED = "formation_achieved"\n'
        "(add it next to the existing TERMINAL_REASON_GOAL/TERMINAL_REASON_COLLISION "
        "constants), or arena.py itself failed to import for an unrelated reason "
        "(e.g. gymnasium not installed) — check the original error below.\n"
        f"Original error: {e}"
    ) from e


@dataclass
class FormationHoldConfig:
    """Grouped, structured task configuration — same pattern
    ObstacleAvoidanceConfig already established."""
    target_separation: float = 2.0
    safety_radius: float = 0.5       # d_safe — collision threshold
    influence_radius: float = 1.5    # d_influence — proximity shaping starts here
    tolerance: float = 0.2           # epsilon — "close enough" to target separation
    hold_steps: int = 20             # N consecutive steps within tolerance to succeed

    reward_sep_weight: float = 1.0
    reward_ctrl_weight: float = 0.01
    reward_prox_weight: float = 1.0
    reward_coll_weight: float = 10.0
    success_bonus: float = 10.0

    def validate(self) -> None:
        if self.safety_radius >= self.influence_radius:
            raise ValueError(
                f"safety_radius ({self.safety_radius}) must be < influence_radius "
                f"({self.influence_radius}) — otherwise proximity shaping never "
                f"activates before collision, which defeats its purpose."
            )
        if self.tolerance >= self.target_separation:
            raise ValueError(
                f"tolerance ({self.tolerance}) must be < target_separation "
                f"({self.target_separation}) — otherwise 'success' could include "
                f"separation=0, which should be a collision, not a success."
            )


class FormationHoldTask:
    """See module docstring for the full frozen spec. NOT a drop-in
    Task subclass usable through Arena.step() — see module docstring's
    'Called directly' section for why."""

    def __init__(self, config: Optional[FormationHoldConfig] = None):
        self.config = config or FormationHoldConfig()
        self.config.validate()
        self._consecutive_steps_in_tolerance = 0

    def reset(self) -> None:
        self._consecutive_steps_in_tolerance = 0

    def step(self, separation: float, action_a: np.ndarray, action_b: np.ndarray,
              truncated: bool, comm_revealed_other_position: Optional[np.ndarray] = None) -> tuple:
        """
        separation: GROUND-TRUTH distance between both agents' true
            positions, supplied by the orchestrator — never comm-derived.
        action_a, action_b: each agent's own raw action, for the
            asymmetric control-effort term.
        comm_revealed_other_position: A1.5's literal requirement — accepted
            here as an explicit input to step() itself, per the plan's
            exact wording. NEVER read by the reward/termination math below
            — recorded into info purely for audit/logging (so a later
            analysis of a training run can see what the channel revealed
            alongside what actually happened), and nothing else. See
            test_step_reward_and_termination_are_invariant_to_comm_position
            for the behavioral proof this is genuinely inert, not just a
            convention.

            NOTE, worth having on record: since packet loss/latency can
            differ per direction, agent A's view of B and agent B's view
            of A are not necessarily the same value on a given tick. This
            single joint step() call can only carry one such value for
            audit purposes; the actual PER-AGENT comm-limited observation
            each agent's policy sees is built separately by
            extra_obs_for_agent(), called once per agent with that
            agent's own comm-revealed view — that division of
            responsibility is unchanged by this parameter's addition.
        Returns (reward, terminated, info) — info always carries
        "reward_components" (a dict) and, on termination, "terminal_reason".
        """
        cfg = self.config
        d = float(separation)

        sep_cost = cfg.reward_sep_weight * abs(d - cfg.target_separation)
        ctrl_cost = cfg.reward_ctrl_weight * (
            float(np.sum(np.square(action_a))) + float(np.sum(np.square(action_b)))
        )
        prox_cost = 0.0
        if d < cfg.influence_radius:
            prox_cost = cfg.reward_prox_weight * max(
                0.0, (cfg.influence_radius - d) / cfg.influence_radius
            )
        collision = d < cfg.safety_radius
        coll_cost = cfg.reward_coll_weight if collision else 0.0

        # Success is a STATE, tracked separately from reward — per design
        # review, not baked into the reward's conceptual definition.
        within_tolerance = abs(d - cfg.target_separation) < cfg.tolerance
        if within_tolerance and not collision:
            self._consecutive_steps_in_tolerance += 1
        else:
            self._consecutive_steps_in_tolerance = 0
        formation_achieved = self._consecutive_steps_in_tolerance >= cfg.hold_steps

        success_bonus = cfg.success_bonus if formation_achieved else 0.0

        reward = -sep_cost - ctrl_cost - prox_cost - coll_cost + success_bonus

        info = {
            "reward_components": {
                "separation": -sep_cost,
                "control": -ctrl_cost,
                "proximity": -prox_cost,
                "collision": -coll_cost,
                "success_bonus": success_bonus,
            },
            "consecutive_steps_in_tolerance": self._consecutive_steps_in_tolerance,
            "true_separation": d,
            # Audit-only — see the parameter's own docstring above. Never
            # read back by anything in this method.
            "comm_revealed_other_position_at_step_time": (
                None if comm_revealed_other_position is None
                else np.asarray(comm_revealed_other_position).tolist()
            ),
        }

        # Collision checked first — a simultaneous collision + formation-
        # achieved on the same step is COLLISION, never silently overridden
        # by success. Same invariant ObstacleAvoidanceTask already established.
        if collision:
            terminated = True
            info["terminal_reason"] = TERMINAL_REASON_COLLISION
        elif formation_achieved:
            terminated = True
            info["terminal_reason"] = TERMINAL_REASON_FORMATION_ACHIEVED
        else:
            terminated = False

        return reward, terminated, info

    def extra_obs_dim(self) -> int:
        return 4  # comm-revealed other-agent relative position (3) + a validity flag (1)

    def extra_obs_for_agent(self, own_position: np.ndarray,
                             comm_revealed_other_position: Optional[np.ndarray]) -> np.ndarray:
        """
        THE comm-limited half — deliberately separate from step()'s
        ground-truth reward/termination logic, per the critical invariant
        above. comm_revealed_other_position may be None (dropped, or
        nothing has arrived yet this tick) — the validity flag lets the
        policy distinguish "I know where they are" from "I don't,"
        rather than silently treating a stale zero-vector as real
        information.
        """
        if comm_revealed_other_position is None:
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        rel = np.asarray(comm_revealed_other_position, dtype=float) - np.asarray(own_position, dtype=float)
        return np.concatenate([rel, [1.0]]).astype(np.float32)