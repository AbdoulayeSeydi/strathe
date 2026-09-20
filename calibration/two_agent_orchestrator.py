"""
A1.6 — Orchestration loop. Owns the two ArenaCore instances from A1.2
(generic + calibrated), the comm channel from A0, and one shared
FormationHoldTask from A1.3-A1.5. NOT a gym.Env yet — that's A1.7. This
is the bare, steppable loop everything above wraps, so each piece can be
tested and inspected directly before any Gym wrapping happens.

## Fixed roles: "generic" and "calibrated," not interchangeable slots
This is a heterogeneous two-agent scenario, matching A1.2's own naming.
The generic agent and the calibrated agent experience the SAME trained
policy and the SAME communication conditions every episode — any
difference in their resulting behavior is attributable to the
calibration difference alone, which is the whole point A1.10's
degradation-curve comparison rests on.

## Position beacons, sent automatically every tick
Each agent broadcasts its own real position to the other EVERY tick
(MESSAGE_TYPE_POSITION_BEACON) — the comm channel's own physics
(latency/dropout/bandwidth) determines whether/when it's actually
received, not any explicit "should I send now" decision. Smarter send
policies are a real, separate, later extension — not needed to prove the
comm-degradation thesis.

## A real, physical consequence worth stating explicitly: minimum one-tick lag
Latency = distance / 1500 m/s is NEVER exactly zero for any nonzero
separation, even with packet loss and bandwidth both configured to be
maximally permissive ("zero degradation" in the A1.8 sense). Because a
message's arrival_time is checked against the CURRENT tick's time
immediately after it was just sent at that same tick, a message sent at
tick N always arrives at tick N+1 at the earliest, never tick N itself —
this is correct discrete-time behavior (nothing can be received in the
same instant it's sent), not a bug. See
test_agent_always_lags_by_at_least_one_tick.

## The audit-only comm_revealed_other_position passed to task.step()
FormationHoldTask.step() is a SHARED, joint call — it can only carry ONE
comm_revealed_other_position value for its audit-trail field (A1.5),
even though the two DIRECTIONS of comm (generic's view of calibrated, and
calibrated's view of generic) can genuinely differ. This deliberately
passes the CALIBRATED agent's revealed view of the GENERIC agent for
that single slot — an arbitrary but documented choice, since the field
is audit-only and never affects reward/termination (already proven by
A1.5's own invariance test). Each agent's ACTUAL observation is built
separately via extra_obs_for_agent(), called once per agent with that
agent's own correct, possibly-different comm-revealed view — this
simplification only touches one non-functional log field, nothing
behavioral.

## C0.1 addition (see conversation): MuJoCo as the shared-scene POSE source
Track B's validated two-body MuJoCo scene (b3_two_body_placeholder.xml,
B3.1-B3.3) is now the source of the position value every consumer below
actually reads (channel.send, true_separation, info, observations) —
NOT ArenaCore.state.position directly anymore. This is DELIBERATELY,
STRICTLY a pose-source swap, nothing more:

  - ArenaCore.step() STILL does 100% of the actual velocity/dynamics
    integration, completely unchanged. This file only takes its RESULT
    (position, orientation) and mirrors it into the shared MuJoCo scene.
  - Each step: ArenaCore.step() runs as before -> its resulting position
    AND a full, valid orientation quaternion are written into that
    body's own qpos slice via mujoco_backend.set_body_pose() (which
    propagates via mj_kinematics -- forward kinematics ONLY, no
    collision detection or dynamics -- see that function's own
    docstring) -> the position actually used everywhere below is read
    back via mujoco_backend.get_body_position(), never
    ArenaCore.state.position directly.
  - mujoco_backend owns every MuJoCo-specific representation detail
    (qpos layout, quaternion convention) -- this file only ever calls
    set_body_pose(position, orientation) / get_body_position(), never
    touches qpos or a body_id-to-address mapping directly.
  - Collision detection, dynamics, and termination are explicitly OUT of
    scope for this change -- MuJoCo is not yet authoritative for any of
    those; existing distance-threshold collision logic in
    FormationHoldTask is untouched. That is later, separate scope, once
    the shared scene is actually exploited for physical interaction.
  - The intended result is BEHAVIORALLY identical to the pre-change
    orchestrator, not necessarily bit-for-bit identical -- writing to
    MuJoCo's qpos and reading back through mj_kinematics can introduce
    tiny representation/normalization differences (see
    test_c0_1_pose_source_equivalence.py's own explicit numerical
    tolerance, rather than an exact-equality assumption).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import mujoco

# comm_channel.py deliberately lives at the project root (sibling to
# calibration/), not inside it — see A0's own design note on being
# vehicle-agnostic and reusable outside this specific calibration
# pipeline. This file lives INSIDE calibration/ (matching every other
# script here, and matching where these scripts are actually run from —
# vehicle_spec.yaml/calibrated_parameters_a1.yaml are loaded via relative
# paths that assume that working directory). Resolving the project root
# from THIS FILE'S OWN absolute path — not the current working directory —
# means this works correctly regardless of where it's launched from,
# rather than silently breaking if someone runs it from a different folder.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from build_two_agent_cores import build_two_agent_cores
from comm_channel import CommChannel, PacketLossConfig, BandwidthConfig, MESSAGE_TYPE_POSITION_BEACON
from formation_hold_task import FormationHoldTask, FormationHoldConfig
from arena_core import ArenaCore
from mujoco_backend import set_body_pose, get_body_position, MuJoCoArenaCore

# 3 floats @ 8 bytes + a small header allowance — a reasonable MVP
# placeholder, not a modeled real acoustic-modem framing format (a real,
# separate later refinement, same as the comm module's own scope notes).
POSITION_BEACON_SIZE_BYTES = 24

# C0.1 addition (see conversation): the shared, two-body MuJoCo scene
# this orchestrator mirrors pose into. B3.1's own file, unmodified --
# body 1 = auv_placeholder, body 2 = auv_placeholder_2 (confirmed
# MuJoCo convention: index 0 is always the implicit world body). Fixed,
# explicit role assignment -- matching this file's own "generic"/
# "calibrated" naming discipline (see module docstring) -- not an
# interchangeable/arbitrary mapping.
TWO_BODY_MJCF_PATH = "b3_two_body_placeholder.xml"
_GENERIC_BODY_ID = 1
_CALIBRATED_BODY_ID = 2


@dataclass
class OrchestratorConfig:
    dt: float = 0.1  # matches ArenaCore's/Arena's own default external timestep
    max_episode_steps: int = 200
    # Both ArenaCore.reset() calls default to position=[0,0,0] — spawning
    # both agents there means separation=0.0 BEFORE either agent ever
    # acts, which is already inside any sane safety_radius. Every episode
    # would "collide" on step one regardless of policy — confirmed
    # directly: a random-action Gate 2 baseline run showed collision_rate
    # 1.0, mean_episode_length 1.0, on every single episode, before this
    # fix.
    #
    # 4.0m, NOT 2.0m (FormationHoldConfig's default target_separation) —
    # confirmed via a direct test this matters, not a style preference:
    # starting exactly AT the target let a random policy "succeed" 40% of
    # the time purely by not drifting far from where it already started,
    # which would make any later PPO-vs-random comparison far less
    # informative. Starting meaningfully farther than target forces an
    # actual closing maneuver to succeed, regardless of policy.
    initial_separation: float = 4.0


class TwoAgentOrchestrator:
    """A1.6's bare loop — a plain Python object with an explicit
    step(action_generic, action_calibrated) method."""

    def __init__(self,
                 calibrated_params_path: str = "calibrated_parameters_a1.yaml",
                 packet_loss_config: Optional[PacketLossConfig] = None,
                 bandwidth_config: Optional[BandwidthConfig] = None,
                 task_config: Optional[FormationHoldConfig] = None,
                 orchestrator_config: Optional[OrchestratorConfig] = None,
                 comm_seed: Optional[int] = None,
                 mjcf_path: str = TWO_BODY_MJCF_PATH,
                 dynamics_backend: str = "stand_in",
                 speed_of_sound_mps: float = 1500.0):
        """
        dynamics_backend: C0.3 addition (see conversation). "stand_in"
        (default, EXACT prior behavior for every existing caller) uses
        ArenaCore for both agents -- fast, self-integrating stand-in
        dynamics, with C0.1's shared two-body MuJoCo scene as a separate
        POSE source only (see _sync_pose_to_mujoco). "mujoco" uses
        mujoco_backend.MuJoCoArenaCore for both agents instead -- genuine,
        self-integrating MuJoCo dynamics, each agent its OWN independent
        single-body MuJoCo instance (not the shared two-body scene --
        see MuJoCoArenaCore's own docstring for why two independent
        instances, not an interleaved shared one). In this mode, the
        core's own state.position IS ALREADY genuine MuJoCo-derived
        position -- no separate pose-mirroring scene is created at all.

        This parameter exists specifically for C0.3's sim-to-sim
        validation: the SAME already-trained (on the stand-in, per
        phase2_config.py's own Decision 2) policy evaluated against
        MuJoCo dynamics instead, with the policy, observation, reward,
        action scaling, communication model, and task/termination logic
        all held completely fixed -- dynamics_backend is the ONLY
        intentional change.
        """
        if dynamics_backend not in ("stand_in", "mujoco"):
            raise ValueError(
                f"dynamics_backend must be 'stand_in' or 'mujoco', got {dynamics_backend!r}"
            )
        self._dynamics_backend = dynamics_backend
        self.config = orchestrator_config or OrchestratorConfig()

        core_class = ArenaCore if dynamics_backend == "stand_in" else MuJoCoArenaCore
        self.core_generic, self.core_calibrated = build_two_agent_cores(
            calibrated_params_path, dt=self.config.dt, core_class=core_class,
        )

        # C0.1's shared MuJoCo scene, used ONLY as a separate pose source
        # for the stand_in backend -- NOT created at all for the mujoco
        # backend, where each core's own state.position is already the
        # genuine, correct value (see _sync_pose_to_mujoco below).
        if dynamics_backend == "stand_in":
            self._mujoco_model = mujoco.MjModel.from_xml_path(mjcf_path)
            self._mujoco_data = mujoco.MjData(self._mujoco_model)

        # Maximally permissive defaults (no packet loss, effectively
        # unlimited bandwidth) — A1.8's zero-DEGRADATION check needs this,
        # not zero latency, which is never actually achievable (see module
        # docstring). A1.10 explicitly overrides these later to turn real
        # degradation on.
        self.packet_loss_config = packet_loss_config or PacketLossConfig(
            near_distance_m=0.0, far_distance_m=1000.0,
            near_field_prob=0.0, far_field_prob=0.0,
        )
        self.bandwidth_config = bandwidth_config or BandwidthConfig(bytes_per_tick=1e9)
        # F2.1: speed_of_sound_mps is now a real parameter, threaded through
        # from the experiment schema's comm_overrides. Default 1500.0 m/s
        # (real seawater value) preserves exact prior behavior for every
        # existing caller that doesn't pass this argument.
        self.speed_of_sound_mps = speed_of_sound_mps
        self.channel = CommChannel(
            self.packet_loss_config, self.bandwidth_config,
            speed_of_sound_mps=speed_of_sound_mps, seed=comm_seed,
        )

        self.task = FormationHoldTask(task_config)

        self._t = 0.0
        self._step_count = 0
        self._last_revealed_to_generic: Optional[np.ndarray] = None
        self._last_revealed_to_calibrated: Optional[np.ndarray] = None

    def _sync_pose_to_mujoco(self, body_id: int, core) -> np.ndarray:
        """C0.1 addition, extended for C0.3 (see conversation).

        stand_in backend: mirrors one ArenaCore's current (position,
        orientation) into the shared MuJoCo scene, then returns the
        MuJoCo-derived position that everything downstream actually
        uses -- the exact original C0.1 behavior, unchanged.

        mujoco backend: the core itself (a MuJoCoArenaCore) is already
        genuinely MuJoCo-backed -- its own state.position IS the correct,
        real MuJoCo-derived value already, with no separate shared-scene
        mirroring step needed or possible (no self._mujoco_model/_data
        exist in this mode at all). Returns it directly.

        Either way, this remains the ONE, single place this orchestrator
        ever resolves an agent's position -- reset() and step() both
        call this rather than branching on backend themselves."""
        if self._dynamics_backend == "mujoco":
            return core.state.position.copy()
        set_body_pose(self._mujoco_model, self._mujoco_data, body_id,
                      core.state.position, core.state.orientation)
        return get_body_position(self._mujoco_model, self._mujoco_data, body_id)

    def reset(self, initial_separation: Optional[float] = None,
              initial_calibrated_surge_velocity: Optional[float] = None):
        """
        initial_separation/initial_calibrated_surge_velocity: OPTIONAL
        explicit overrides for the calibrated agent's starting x-position
        and body-frame surge velocity. When None (default), falls back to
        self.config.initial_separation with zero velocity -- the exact
        original fixed-start behavior, unchanged. This orchestrator does
        not sample or randomize these itself (matching its existing
        no-opinion-on-rescaling stance in step()) -- that's a caller
        concern, e.g. TwoAgentFormationEnv sampling a curriculum
        distribution before calling this.
        """
        # Deliberate nonzero starting separation — see OrchestratorConfig's
        # own docstring for why both agents at the origin is a real bug,
        # not a harmless default.
        sep = initial_separation if initial_separation is not None else self.config.initial_separation
        cal_nu = np.zeros(6)
        if initial_calibrated_surge_velocity is not None:
            cal_nu[0] = initial_calibrated_surge_velocity
        self.core_generic.reset(position=np.array([0.0, 0.0, 0.0]))
        self.core_calibrated.reset(position=np.array([sep, 0.0, 0.0]), nu=cal_nu)

        # C0.1 addition (see conversation): sync the freshly-reset initial
        # pose into the shared MuJoCo scene BEFORE the first observation is
        # built, so _build_observations() (called at the end of this
        # method) already reads MuJoCo-derived position, not stale/zero
        # MuJoCo state from a previous episode or the scene's own default
        # construction.
        self._sync_pose_to_mujoco(_GENERIC_BODY_ID, self.core_generic)
        self._sync_pose_to_mujoco(_CALIBRATED_BODY_ID, self.core_calibrated)

        self.channel.reset()
        self.task.reset()
        self._t = 0.0
        self._step_count = 0
        self._last_revealed_to_generic = None
        self._last_revealed_to_calibrated = None
        return self._build_observations()

    def step(self, action_generic: np.ndarray, action_calibrated: np.ndarray,
              action_generic_for_reward: Optional[np.ndarray] = None,
              action_calibrated_for_reward: Optional[np.ndarray] = None):
        """
        action_generic/action_calibrated: raw, UN-normalized 6-dim
        force/torque vectors used for the actual physics — this
        orchestrator does not do the [-1,1]-normalized-action rescaling
        Arena.step() does; that's an A1.7 concern, and TwoAgentFormationEnv
        already rescales before calling here.

        action_*_for_reward: OPTIONAL — if given, THESE are what's passed
        to FormationHoldTask.step()'s control-cost term instead of the raw
        physics actions. This exists to fix a real, confirmed scale bug:
        FormationHoldTask's reward_ctrl_weight=0.01 was calibrated to match
        ObstacleAvoidanceTask's convention, where the control penalty is
        computed on the NORMALIZED [-1,1] action (Arena.step() stores
        _last_action before rescaling). Without this parameter, this
        orchestrator was passing the RESCALED, real-Newton action straight
        into that same 0.01-weighted term — at full 30N single-axis
        thrust, that's a control cost of 9.0 versus a MAXIMUM possible
        separation-cost swing of only 2.0 (initial_separation=4.0 minus
        target_separation=2.0). A 900x scale mismatch, confirmed by direct
        calculation, not assumed — strong enough on its own to make any
        meaningful thrust reward-dominated by its own control penalty,
        regardless of PPO configuration. When these two parameters are
        None (default), falls back to using action_generic/
        action_calibrated directly — preserving exact prior behavior for
        any caller (e.g. A1.6's own existing tests) that doesn't know
        about a normalized/rescaled distinction at all.

        Returns ((obs_generic, obs_calibrated), reward, terminated,
        truncated, info) — reward is the single joint scalar
        FormationHoldTask already computes (per A1.3's frozen design:
        R_a,t + R_b,t combined, not split per agent), matching what A1.7's
        centralized policy will actually need.

        C0.1 (see conversation): position is now MuJoCo-derived (see
        module docstring) — ArenaCore.step() itself, and everything about
        its own velocity/dynamics integration, is completely unchanged.
        """
        self.core_generic.step(np.asarray(action_generic, dtype=float))
        self.core_calibrated.step(np.asarray(action_calibrated, dtype=float))
        self._step_count += 1
        self._t += self.config.dt

        pos_generic = self._sync_pose_to_mujoco(_GENERIC_BODY_ID, self.core_generic)
        pos_calibrated = self._sync_pose_to_mujoco(_CALIBRATED_BODY_ID, self.core_calibrated)

        # Ground-truth separation — used for reward/termination, NEVER for
        # what either agent's observation reveals.
        true_separation = float(np.linalg.norm(pos_generic - pos_calibrated))

        self.channel.send(
            sender_id="generic", receiver_id="calibrated", send_time=self._t,
            sender_pos=pos_generic, receiver_pos=pos_calibrated,
            message_type=MESSAGE_TYPE_POSITION_BEACON,
            payload=pos_generic.tolist(), payload_size_bytes=POSITION_BEACON_SIZE_BYTES,
        )
        self.channel.send(
            sender_id="calibrated", receiver_id="generic", send_time=self._t,
            sender_pos=pos_calibrated, receiver_pos=pos_generic,
            message_type=MESSAGE_TYPE_POSITION_BEACON,
            payload=pos_calibrated.tolist(), payload_size_bytes=POSITION_BEACON_SIZE_BYTES,
        )
        self.channel.step(current_time=self._t)

        delivered_to_generic = self.channel.receive("generic")
        delivered_to_calibrated = self.channel.receive("calibrated")
        if delivered_to_generic:
            self._last_revealed_to_generic = np.array(delivered_to_generic[-1].payload)
        if delivered_to_calibrated:
            self._last_revealed_to_calibrated = np.array(delivered_to_calibrated[-1].payload)

        truncated = self._step_count >= self.config.max_episode_steps

        reward_action_a = (action_generic_for_reward if action_generic_for_reward is not None
                            else action_generic)
        reward_action_b = (action_calibrated_for_reward if action_calibrated_for_reward is not None
                            else action_calibrated)

        reward, terminated, task_info = self.task.step(
            separation=true_separation,
            action_a=np.asarray(reward_action_a, dtype=float),
            action_b=np.asarray(reward_action_b, dtype=float),
            truncated=truncated,
            comm_revealed_other_position=self._last_revealed_to_calibrated,
        )

        obs_generic, obs_calibrated = self._build_observations()

        info = dict(task_info)
        info.update({
            "true_separation": true_separation,
            "position_generic": pos_generic,
            "position_calibrated": pos_calibrated,
            "revealed_to_generic": self._last_revealed_to_generic,
            "revealed_to_calibrated": self._last_revealed_to_calibrated,
            "delivered_to_generic_this_tick": bool(delivered_to_generic),
            "delivered_to_calibrated_this_tick": bool(delivered_to_calibrated),
        })

        return (obs_generic, obs_calibrated), reward, terminated, truncated, info

    def _build_observations(self):
        # BUG FOUND AND FIXED (see conversation): this previously called
        # get_body_position(self._mujoco_model, ...) directly and
        # unconditionally -- correct for the stand_in backend (C0.1,
        # where the scene is already in sync from whichever of
        # reset()/step() called this), but self._mujoco_model/_data are
        # NEVER created at all in "mujoco" mode (see __init__), so this
        # would have raised AttributeError the instant reset() was
        # called with dynamics_backend="mujoco" -- a missed call site
        # from C0.3's own refactor (_sync_pose_to_mujoco was updated
        # correctly; this method was not). Fixed by reusing
        # _sync_pose_to_mujoco itself -- idempotent in stand_in mode
        # (re-syncing the same, unchanged core state and reading it back
        # gives the same result), and exactly what's needed in mujoco
        # mode (returns core.state.position directly) -- rather than
        # duplicating the backend branch in a second place.
        pos_generic = self._sync_pose_to_mujoco(_GENERIC_BODY_ID, self.core_generic)
        pos_calibrated = self._sync_pose_to_mujoco(_CALIBRATED_BODY_ID, self.core_calibrated)
        obs_generic = self.task.extra_obs_for_agent(
            own_position=pos_generic,
            comm_revealed_other_position=self._last_revealed_to_generic,
        )
        obs_calibrated = self.task.extra_obs_for_agent(
            own_position=pos_calibrated,
            comm_revealed_other_position=self._last_revealed_to_calibrated,
        )
        return obs_generic, obs_calibrated
