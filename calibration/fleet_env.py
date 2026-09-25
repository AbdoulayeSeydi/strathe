"""
fleet_env.py -- fleet v2 simulation: two vehicles, decentralized
observations, a physical teammate-belief model.

## What changed from fleet v1 (two_agent_formation_env.py), and why
1. Per-vehicle observations. v1 gave ONE centralized policy both vehicles'
   exact velocity and orientation every tick, so part of the teammate's
   state bypassed the acoustic channel entirely. Here each vehicle's
   policy input is built only from its own sensors plus what it believes
   about its teammate. One policy network is shared by both vehicles
   (parameter sharing), applied to each vehicle's own observation.
2. Identical true physics for both vehicles. v1 put calibrated physics on
   one vehicle and generic physics on the other, which made "which slot
   the policy drives harder" indistinguishable from "which physics is
   better". Both vehicles here run the same truth parameters.
3. A physical belief model (fleet_belief.py). What each vehicle uses to
   predict its teammate between beacons is the experimental variable.
4. Every episode setting is a parameter: initial separation and closing
   speed (ranges for training, exact values for evaluation grids), episode
   length, task tolerances, comm condition.

## Observation (per vehicle, 13 floats)
    [0:6]   own body velocity nu = [u, v, w, p, q, r]
    [6:8]   own roll, pitch (rad)
    [8:11]  believed teammate position relative to self, in OWN body frame (m)
    [11]    belief age: min(age_s, age_cap_s) / age_cap_s (1.0 when no beacon yet)
    [12]    1.0 if any beacon has ever arrived, else 0.0 (then [8:11] are zero)
## Action (per vehicle, 6 floats in [-1, 1])
    body-frame [Fx, Fy, Fz, Tx, Ty, Tz], scaled by action_bound (30 N / N*m)

## Episode layout
Vehicle 0 starts at the origin, vehicle 1 at (d0, 0, 0), both level and
facing +x. closing_speed v0 is split symmetrically: vehicle 0 surges at
-v0/2 and vehicle 1 at +v0/2, so v0 > 0 means they start separating.
Reward and termination are FormationHoldTask's, computed from TRUE
separation; the reward is shared by both vehicles (cooperative task).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from comm_channel import BandwidthConfig, CommChannel, PacketLossConfig, MESSAGE_TYPE_POSITION_BEACON
from current import CurrentConfig, WaterCurrent
from disturbance import DisturbanceConfig, RollPitchDisturbance, episode_seed
from dynamics_model import CalibrationParameters, SimulationConfig, VehicleConstants
from fast_dynamics import FastCore
from fleet_belief import Beacon, BeliefSpec
from formation_hold_task import FormationHoldConfig, FormationHoldTask

OBS_DIM = 13
ACT_DIM = 6
N_AGENTS = 2
BEACON_SIZE_BYTES = 24  # nominal; keeps the named severities' meaning (see fleet_belief.py)


@dataclass
class FleetConfig:
    dt: float = 0.1
    physics_dt: float = 0.005
    max_episode_steps: int = 200
    action_bound: float = 30.0
    age_cap_s: float = 10.0
    initial_separation_range: tuple = (2.0, 5.5)
    initial_closing_speed_range: tuple = (-0.7, 0.7)
    dynamics_backend: str = "stand_in"   # truth physics engine: "stand_in" or "mujoco"
    # How often each vehicle transmits its state. Acoustic modems exchange
    # state every ~1-10 s in practice; fleet v1 beaconed every 0.1 s tick,
    # which kept every belief within a few cm of the truth even at 60%
    # packet loss, so no belief model could ever matter. Must be a whole
    # number of ticks.
    beacon_interval_s: float = 1.0
    # A collision ends the episode. Without this, ending early escapes all
    # remaining per-step separation cost, so an early crash could score
    # better than holding formation imperfectly for the full episode (fleet
    # v1 used that reward; its policies collided in ~20-50% of episodes).
    # When True, a collision also charges the remaining steps at the cost of
    # sitting at zero separation: reward_sep_weight * target_separation each.
    absorbing_collision_penalty: bool = True
    task: FormationHoldConfig = field(default_factory=FormationHoldConfig)
    # Random roll/pitch torques shaped like the real motion the logged
    # thrusters don't explain (disturbance.py, PROJECT_STATUS.md #51). None =
    # clean simulation (the original behaviour). Added to each vehicle's
    # commanded wrench before the physics step; never part of the beacon,
    # since a teammate can't know it.
    disturbance: Optional[DisturbanceConfig] = None
    # Water current (current.py, PROJECT_STATUS.md #55). None = still water
    # (the tank the reference data comes from; the original behaviour). Both
    # vehicles feel the same current; the policy doesn't observe it directly,
    # only through its own velocity over ground.
    current: Optional[CurrentConfig] = None

    @property
    def beacon_every_ticks(self) -> int:
        return int(round(self.beacon_interval_s / self.dt))

    def validate(self) -> None:
        lo, hi = self.initial_separation_range
        if not (0 < lo <= hi):
            raise ValueError(f"initial_separation_range must satisfy 0 < min <= max, got {self.initial_separation_range}")
        if lo <= self.task.safety_radius:
            raise ValueError(
                f"initial separation min ({lo} m) must exceed the collision radius "
                f"({self.task.safety_radius} m), or episodes start already collided")
        vlo, vhi = self.initial_closing_speed_range
        if vlo > vhi:
            raise ValueError(f"initial_closing_speed_range min > max: {self.initial_closing_speed_range}")
        if self.max_episode_steps <= 0:
            raise ValueError(f"max_episode_steps must be positive, got {self.max_episode_steps}")
        k = self.beacon_interval_s / self.dt
        if self.beacon_interval_s <= 0 or abs(k - round(k)) > 1e-9:
            raise ValueError(f"beacon_interval_s must be a positive multiple of dt={self.dt}, got {self.beacon_interval_s}")
        if self.max_episode_steps < self.task.hold_steps:
            raise ValueError(f"max_episode_steps ({self.max_episode_steps}) < hold_steps ({self.task.hold_steps}): "
                             f"success would be impossible")
        if self.dynamics_backend not in ("stand_in", "mujoco"):
            raise ValueError(f"dynamics_backend must be 'stand_in' or 'mujoco', got {self.dynamics_backend!r}")
        if self.disturbance is not None:
            self.disturbance.validate()
        if self.current is not None:
            self.current.validate()
        self.task.validate()


class _MuJoCoTruth:
    """Adapter giving MuJoCoArenaCore the few accessors FleetSim reads."""

    def __init__(self, params, vehicle, sim_config):
        from mujoco_backend import MuJoCoArenaCore
        self._core = MuJoCoArenaCore(params, vehicle, sim_config)

    def reset(self, position=None, nu=None):
        self._core.reset(position=position, nu=nu)

    def step(self, tau):
        self._core.step(tau)

    @property
    def current_world(self):
        return self._core.current_world

    @current_world.setter
    def current_world(self, value):
        self._core.current_world = value

    def position_array(self):
        return self._core.state.position.copy()

    def nu_array(self):
        return self._core.state.nu.copy()

    def rotation_matrix(self):
        return self._core.state.orientation.as_matrix()

    def roll_pitch(self):
        r, p, _ = self._core.state.orientation.as_euler("xyz")
        return r, p


class _FastTruth(FastCore):
    def rotation_matrix(self):
        return np.array(self.R)


class FleetSim:
    """Two-vehicle formation-hold simulation with a comm channel and a
    per-vehicle teammate belief. Not a gym.Env: the training wrapper
    (fleet_vec_env.py) and the evaluator drive it directly."""

    def __init__(self, truth_params: CalibrationParameters, vehicle: VehicleConstants,
                 config: Optional[FleetConfig] = None,
                 packet_loss_config: Optional[PacketLossConfig] = None,
                 bandwidth_config: Optional[BandwidthConfig] = None,
                 speed_of_sound_mps: float = 1500.0,
                 belief: Optional[BeliefSpec] = None,
                 beacon_interval_s: Optional[float] = None):
        self.cfg = copy.deepcopy(config) if config is not None else FleetConfig()
        self.cfg.validate()
        self.vehicle = vehicle
        sim_config = SimulationConfig(dt=self.cfg.dt, physics_dt=self.cfg.physics_dt)
        truth_cls = _FastTruth if self.cfg.dynamics_backend == "stand_in" else _MuJoCoTruth
        self.cores = [truth_cls(truth_params, vehicle, sim_config) for _ in range(N_AGENTS)]
        self.task = FormationHoldTask(self.cfg.task)
        self.set_comm(packet_loss_config, bandwidth_config, speed_of_sound_mps, beacon_interval_s)
        self.set_belief(belief or BeliefSpec("last_beacon", "last_beacon"))
        self._tick = 0
        self._last_tau = [np.zeros(ACT_DIM) for _ in range(N_AGENTS)]
        self._disturbances = None
        self._current = None

    # --- configuration that may change between episodes -------------------
    def set_comm(self, packet_loss_config=None, bandwidth_config=None, speed_of_sound_mps=1500.0,
                 beacon_interval_s: Optional[float] = None):
        self.packet_loss_config = packet_loss_config or PacketLossConfig(0.0, 1000.0, 0.0, 0.0)
        self.bandwidth_config = bandwidth_config or BandwidthConfig(bytes_per_tick=1e9)
        self.speed_of_sound_mps = speed_of_sound_mps
        if beacon_interval_s is not None:
            self.cfg.beacon_interval_s = beacon_interval_s
            self.cfg.validate()

    def set_belief(self, spec: BeliefSpec):
        self.belief_spec = spec
        self.beliefs = [spec.build(self.cfg.dt, self.vehicle) for _ in range(N_AGENTS)]

    # --- episode control -----------------------------------------------------
    def reset(self, comm_seed: int, initial_separation: Optional[float] = None,
              closing_speed: Optional[float] = None, rng: Optional[np.random.Generator] = None) -> list:
        """initial_separation / closing_speed: exact values (evaluation), or
        None to sample from the configured ranges with `rng` (training)."""
        rng = rng or np.random.default_rng(comm_seed)
        d0 = initial_separation if initial_separation is not None else float(rng.uniform(*self.cfg.initial_separation_range))
        v0 = closing_speed if closing_speed is not None else float(rng.uniform(*self.cfg.initial_closing_speed_range))
        if d0 <= self.cfg.task.safety_radius:
            raise ValueError(f"initial separation {d0} m is inside the collision radius")
        self.initial_separation, self.closing_speed = d0, v0
        self.cores[0].reset(position=np.zeros(3), nu=np.array([-v0 / 2, 0, 0, 0, 0, 0]))
        self.cores[1].reset(position=np.array([d0, 0.0, 0.0]), nu=np.array([v0 / 2, 0, 0, 0, 0, 0]))
        self.channel = CommChannel(self.packet_loss_config, self.bandwidth_config,
                                   speed_of_sound_mps=self.speed_of_sound_mps, seed=comm_seed)
        self.task.reset()
        for b in self.beliefs:
            b.reset()
        dist = self.cfg.disturbance
        # seeded from the episode's identity only, so paired conditions (same
        # start state and comm seed, different belief model) share disturbances
        self._disturbances = ([RollPitchDisturbance(dist, self.cfg.dt, episode_seed(comm_seed, i, d0, v0))
                               for i in range(N_AGENTS)] if dist is not None and dist.active else None)
        cur = self.cfg.current
        # one current for both vehicles (the same water), seeded like the disturbances
        self._current = (WaterCurrent(cur, self.cfg.dt, episode_seed(comm_seed, N_AGENTS, d0, v0, salt=(0xC0,)))
                         if cur is not None and cur.active else None)
        known = self._current.mean_world if (self._current is not None and cur.known_to_beliefs) else None
        for b in self.beliefs:
            b.set_water_current(known)
        for core in self.cores:
            core.current_world = None
        self._tick = 0
        self._last_tau = [np.zeros(ACT_DIM) for _ in range(N_AGENTS)]
        return self._observations()

    def step(self, actions: list):
        """actions: one (6,) array in [-1, 1] per vehicle.
        Returns (obs list, shared reward, terminated, truncated, info)."""
        norm = [np.clip(np.asarray(a, dtype=float), -1.0, 1.0) for a in actions]
        taus = [a * self.cfg.action_bound for a in norm]
        current_now = None
        if self._current is not None:
            current_now = self._current.next()
            for core in self.cores:
                core.current_world = current_now
        for i, (core, tau) in enumerate(zip(self.cores, taus)):
            core.step(tau if self._disturbances is None else tau + self._disturbances[i].next())
        self._tick += 1
        self._last_tau = taus
        t = self._tick * self.cfg.dt

        pos = [c.position_array() for c in self.cores]
        true_sep = float(np.linalg.norm(pos[0] - pos[1]))

        # Each vehicle beacons its end-of-tick state every beacon_every_ticks
        # ticks; delivery happens on a later tick (latency > 0). The comm RNG
        # is drawn once per send, and send ticks don't depend on actions, so
        # the drop pattern is the same whatever the vehicles do.
        sends = [] if self._tick % self.cfg.beacon_every_ticks else range(N_AGENTS)
        for i in sends:
            j = 1 - i
            beacon = Beacon(send_tick=self._tick, position=pos[i], orientation=self.cores[i].rotation_matrix(),
                            nu=self.cores[i].nu_array(), tau=taus[i].copy())
            self.channel.send(sender_id=str(i), receiver_id=str(j), send_time=t,
                              sender_pos=pos[i], receiver_pos=pos[j],
                              message_type=MESSAGE_TYPE_POSITION_BEACON, payload=beacon,
                              payload_size_bytes=BEACON_SIZE_BYTES)
        self.channel.step(current_time=t)
        delivered = [[m.payload for m in self.channel.receive(str(i))] for i in range(N_AGENTS)]
        for i in range(N_AGENTS):
            self.beliefs[i].on_tick(self._tick, delivered[i])

        truncated = self._tick >= self.cfg.max_episode_steps
        reward, terminated, task_info = self.task.step(
            separation=true_sep, action_a=norm[0], action_b=norm[1], truncated=truncated)
        collision_tail_penalty = 0.0
        if task_info.get("terminal_reason") == "collision" and self.cfg.absorbing_collision_penalty:
            remaining = self.cfg.max_episode_steps - self._tick
            collision_tail_penalty = remaining * self.cfg.task.reward_sep_weight * self.cfg.task.target_separation
            reward -= collision_tail_penalty

        estimates = [self.beliefs[i].estimate(true_teammate_position=pos[1 - i]) for i in range(N_AGENTS)]
        belief_errors = [None if e is None else float(np.linalg.norm(e - pos[1 - i]))
                         for i, e in enumerate(estimates)]
        info = {
            "true_separation": true_sep,
            "positions": pos,
            "belief_errors": belief_errors,
            "delivered": [bool(d) for d in delivered],
            "terminal_reason": task_info.get("terminal_reason"),
            "current_world": current_now,
            "reward_components": dict(task_info["reward_components"], collision_tail=-collision_tail_penalty),
        }
        return self._observations(estimates), float(reward), bool(terminated), bool(truncated and not terminated), info

    # --- observations --------------------------------------------------------
    def _observations(self, estimates: Optional[list] = None) -> list:
        if estimates is None:
            pos = [c.position_array() for c in self.cores]
            estimates = [self.beliefs[i].estimate(true_teammate_position=pos[1 - i]) for i in range(N_AGENTS)]
        obs = []
        for i, core in enumerate(self.cores):
            roll, pitch = core.roll_pitch()
            belief = self.beliefs[i]
            if estimates[i] is None:
                rel_body, age, valid = np.zeros(3), 1.0, 0.0
            else:
                rel_world = estimates[i] - core.position_array()
                rel_body = core.rotation_matrix().T @ rel_world
                age, valid = min(belief.age_s, self.cfg.age_cap_s) / self.cfg.age_cap_s, 1.0
            obs.append(np.concatenate([core.nu_array(), [roll, pitch], rel_body, [age, valid]]).astype(np.float32))
        return obs
