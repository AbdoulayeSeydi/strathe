"""
escort_env.py -- a heterogeneous pair: a BlueROV2 escorts a fin-steered REMUS 100 over an acoustic link.

## The team
- The REMUS (escort_remus.py) flies its own survey on its own autopilot and never reacts to the ROV. It sends an
  acoustic beacon every beacon_interval_s: position, attitude, body velocity, and its fin angles and propeller rpm.
- The BlueROV2 (Strathe's calibrated BlueROV2 physics, the BlueROV2 Heavy's thrusters) must stay within band_m of
  target_range_m of the REMUS. It never sees the REMUS directly: between beacons it predicts it with a teammate model
  (fleet_belief.py), which is the experimental variable.

The two vehicles share nothing but the water: one hovers on thrusters, the other can't hover and steers with fins,
so the ROV must predict a vehicle whose physics differ from its own.

## Teammate models (what the ROV believes about the REMUS)
    oracle             the true REMUS position (upper bound; not physical)
    calibrated         Strathe's fin-vehicle model with the parameters Strathe calibrated from logs of this REMUS
                       (#97 T-prestero-slow), dead-reckoned from the beacon holding its fins and rpm
    generic            the same model with the starting values `describe-vehicle` writes from a tape measure
    physics_alphaA     A of the way from calibrated to generic, parameter by parameter
    constant_velocity  last beacon position + its velocity x age
    last_beacon        the REMUS assumed to sit where its last beacon said
The calibrated model is not the truth: the truth is Prestero's equations, which Strathe's model was not written from.

## Re-ballasting
escort_remus re-ballasts the REMUS BUOYANCY_CHANGE_N lighter than the vehicle the calibration logs came from. The
ROV's crew knows that change, so the calibrated model gets it (net_buoyancy_z + change). The generic model keeps the
neutral buoyancy `describe-vehicle` assumes.

## Observation (16 floats, the ROV's own frame)
    [0:6]   own body velocity
    [6:8]   own roll, pitch
    [8:11]  believed REMUS position relative to self, own body frame (m)
    [11:14] believed REMUS velocity relative to own, own body frame (m/s). Each model's own velocity: last_beacon
            believes the REMUS is still (0), constant_velocity its beacon's velocity, physics its model's velocity
    [14]    belief age / age_cap_s
    [15]    1.0 once a beacon has arrived (always: the REMUS's launch state is handed over at tick 0)
## Action (4 floats in [-1, 1])
    surge, sway, heave force and yaw moment, scaled by the BlueROV2 Heavy's largest wrench on each axis and limited to
    what its eight thrusters can deliver together (thruster_model.bluerov2_heavy). Roll and pitch are left to its
    righting moment.

## Task
Reward each tick: -w_range |range - target| - w_ctrl |a|^2 - w_prox * max(0, (influence - range) / influence).
Closer than collision_m ends the episode (a collision); farther than lost_m ends it (contact lost). Either charges the
rest of the episode at the worst per-tick cost, so ending early never pays.
Success: no collision, never lost, and within the band on at least success_in_band of the ticks after acquire_s.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from comm_channel import BandwidthConfig, CommChannel, PacketLossConfig, MESSAGE_TYPE_POSITION_BEACON
from dynamics_model import CalibrationParameters, SimulationConfig, VehicleConstants
from escort_remus import Mission, MissionLibrary, controls_basis, controls_for, BUOYANCY_CHANGE_N
from fast_dynamics import FastCore
from fleet_belief import Beacon, BeliefSpec, interpolate_params

OBS_DIM = 16
ACT_DIM = 4
ACTION_AXES = (0, 1, 2, 5)          # surge, sway, heave, yaw in the 6-DOF wrench
BEACON_SIZE_BYTES = 24              # fleet_env's nominal beacon, so the named severities keep their meaning

REMUS_SPEC = "escort/remus_spec.yaml"
REMUS_CALIBRATED = "escort/remus_calibrated_T-prestero-slow.yaml"
# The ROV: Strathe's least bound-limited BlueROV2 calibration (3 of 22 parameters at a bound). The fleet's default truth
# has surge damping pinned at its bounds, which caps it near 0.8 m/s on the Heavy's thrusters: too close to the REMUS's
# 0.7 m/s for an escort. This one reaches ~1 m/s (a real BlueROV2 is rated to 1.5 m/s).
ROV_PARAMS = "calibrated_parameters_fleet_truth_alt.yaml"
ROV_SPEC = "vehicle_spec.yaml"

# Named loss levels: the same values as evaluate_comm_degradation_curve.COMM_CONFIGS (test_escort.py checks it)
SEVERITIES = {
    "L0": (PacketLossConfig(0.0, 1000.0, 0.0, 0.0), BandwidthConfig(bytes_per_tick=1e9)),
    "L1": (PacketLossConfig(0.0, 1000.0, 0.1, 0.1), BandwidthConfig(bytes_per_tick=1e9)),
    "L2": (PacketLossConfig(0.0, 1000.0, 0.3, 0.3), BandwidthConfig(bytes_per_tick=1e9)),
    "L3a": (PacketLossConfig(0.0, 1000.0, 0.6, 0.6), BandwidthConfig(bytes_per_tick=1e9)),
    "L3b": (PacketLossConfig(0.0, 1000.0, 0.6, 0.6), BandwidthConfig(bytes_per_tick=BEACON_SIZE_BYTES - 1)),
}


@dataclass(frozen=True)
class CommCondition:
    severity: str = "L0"
    beacon_interval_s: float = 1.0
    speed_of_sound_mps: float = 1500.0
    fixed_latency_s: float = 0.0

    @property
    def label(self) -> str:
        return f"{self.severity}@{self.beacon_interval_s:g}s"


@dataclass
class EscortConfig:
    dt: float = 0.1
    physics_dt: float = 0.005
    episode_steps: int = 600            # 60 s
    target_range_m: float = 3.0
    band_m: float = 1.0
    collision_m: float = 1.2            # REMUS 1.33 m long, BlueROV2 ~0.5 m: centres closer than this touch
    lost_m: float = 15.0
    acquire_s: float = 10.0
    success_in_band: float = 0.8
    start_range_m: tuple = (2.0, 6.0)
    start_depth_offset_m: tuple = (-0.5, 0.5)
    age_cap_s: float = 10.0
    w_range: float = 1.0
    w_ctrl: float = 0.01
    w_prox: float = 1.0
    influence_m: float = 2.5

    def validate(self) -> None:
        if self.collision_m >= self.target_range_m - self.band_m:
            raise ValueError("the band reaches into the collision radius")
        if self.start_range_m[0] <= self.collision_m:
            raise ValueError("episodes would start collided")
        if self.lost_m <= self.target_range_m + self.band_m:
            raise ValueError("lost_m must be beyond the band")
        if self.acquire_s >= self.episode_steps * self.dt:
            raise ValueError("acquire_s leaves no ticks to judge")
        if not 0.0 < self.success_in_band <= 1.0:
            raise ValueError("success_in_band must be in (0, 1]")

    @property
    def worst_tick_cost(self) -> float:
        return self.w_range * max(self.target_range_m, self.lost_m - self.target_range_m) + self.w_prox + self.w_ctrl * ACT_DIM


@dataclass(frozen=True)
class EscortEpisode:
    """Everything that decides an episode but the ROV's actions and the teammate model: paired conditions share it."""
    mission: int                 # index in the MissionLibrary
    start_tick: int
    offset: tuple                # ROV start position minus REMUS's, world (m)
    comm_seed: int

    @property
    def key(self) -> str:
        return f"m{self.mission}-k{self.start_tick}-o{self.offset[0]:.3f},{self.offset[1]:.3f},{self.offset[2]:.3f}-c{self.comm_seed}"


def sample_episode(rng: np.random.Generator, library: MissionLibrary, cfg: EscortConfig) -> EscortEpisode:
    m = int(rng.integers(len(library)))
    last = len(library[m]) - 1 - cfg.episode_steps
    if last < 0:
        raise ValueError("missions are shorter than an episode")
    k0 = int(rng.integers(0, last + 1))
    rng_m = float(rng.uniform(*cfg.start_range_m))
    bearing = float(rng.uniform(-math.pi, math.pi))
    dz = float(rng.uniform(*cfg.start_depth_offset_m))
    horiz = math.sqrt(max(rng_m ** 2 - dz ** 2, 0.0))
    return EscortEpisode(m, k0, (horiz * math.cos(bearing), horiz * math.sin(bearing), dz), int(rng.integers(2 ** 31)))


def evaluation_episodes(library: MissionLibrary, cfg: EscortConfig, per_mission: int, seed: int = 0) -> list:
    """A fixed, reproducible set: per_mission episodes from every mission in the library."""
    rng = np.random.default_rng(seed)
    out = []
    for m in range(len(library)):
        sub = MissionLibrary([library[m]], library.cfg)
        for _ in range(per_mission):
            e = sample_episode(rng, sub, cfg)
            out.append(dataclasses.replace(e, mission=m))
    return out


# --------------------------------------------------------------------------- models of the REMUS

@dataclass
class RemusModels:
    vehicle: VehicleConstants        # Strathe's description of the REMUS (fins, propeller)
    calibrated: CalibrationParameters
    generic: CalibrationParameters
    basis: tuple                     # Prestero fins -> Strathe controls

    def belief(self, name: str) -> BeliefSpec:
        if name == "calibrated":
            return BeliefSpec("physics", "calibrated", self.calibrated)
        if name == "generic":
            return BeliefSpec("physics", "generic", self.generic)
        if name.startswith("physics_alpha"):
            alpha = float(name[len("physics_alpha"):])
            return BeliefSpec("physics", name, interpolate_params(self.calibrated, self.generic, alpha))
        return BeliefSpec(name, name)


def load_remus_models(spec_path: str = REMUS_SPEC, calibrated_path: str = REMUS_CALIBRATED,
                      buoyancy_change_n: float = BUOYANCY_CHANGE_N) -> RemusModels:
    from build_two_agent_cores import load_calibrated_parameters
    from optimizer import initial_parameters, load_vehicle_spec
    spec = load_vehicle_spec(spec_path)
    cal = load_calibrated_parameters(calibrated_path, quiet=True)
    cal = dataclasses.replace(cal, net_buoyancy_z=cal.net_buoyancy_z + buoyancy_change_n)
    cal.validate()
    return RemusModels(spec.vehicle, cal, initial_parameters(spec), controls_basis(spec.vehicle.actuators))


@dataclass
class RovModel:
    params: CalibrationParameters
    vehicle: VehicleConstants
    thrusters: object                # thruster_model.ThrusterConfig (for_simulation form)
    scale: np.ndarray                # (4,) N, N, N, N*m for action +-1


def load_rov(params_path: str = ROV_PARAMS, spec_path: str = ROV_SPEC) -> RovModel:
    from build_two_agent_cores import load_calibrated_parameters
    from optimizer import load_vehicle_spec
    from thruster_model import bluerov2_heavy
    thr = bluerov2_heavy().for_simulation()
    return RovModel(load_calibrated_parameters(params_path, quiet=True), load_vehicle_spec(spec_path).vehicle, thr,
                    np.asarray(thr.max_wrench(), dtype=float)[list(ACTION_AXES)])


def _yaw(R) -> float:
    return math.atan2(R[1][0], R[0][0])


def _yaw_matrix(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# --------------------------------------------------------------------------- the simulation

class EscortSim:
    def __init__(self, rov: RovModel, remus: RemusModels, library: MissionLibrary, config: Optional[EscortConfig] = None):
        self.cfg = config or EscortConfig()
        self.cfg.validate()
        if abs(library.cfg.dt - self.cfg.dt) > 1e-12:
            raise ValueError("missions and escort tick at different rates")
        self.rov, self.remus, self.library = rov, remus, library
        from scipy.spatial.transform import Rotation
        self._Rotation = Rotation
        self.core = FastCore(rov.params, rov.vehicle, SimulationConfig(dt=self.cfg.dt, physics_dt=self.cfg.physics_dt))
        self.set_comm(CommCondition())
        self.set_belief(BeliefSpec("last_beacon", "last_beacon"))
        self.episode = None

    def set_comm(self, comm: CommCondition) -> None:
        k = comm.beacon_interval_s / self.cfg.dt
        if comm.beacon_interval_s <= 0 or abs(k - round(k)) > 1e-9:
            raise ValueError(f"beacon_interval_s must be a positive multiple of {self.cfg.dt}")
        if comm.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {sorted(SEVERITIES)}")
        self.comm = comm
        self._every = int(round(k))

    def set_belief(self, spec: BeliefSpec) -> None:
        self.belief_spec = spec
        self.belief = spec.build(self.cfg.dt, self.remus.vehicle)

    # ------------------------------------------------------------------ episode
    def reset(self, episode: EscortEpisode) -> np.ndarray:
        self.episode = episode
        self.mission: Mission = self.library[episode.mission]
        if episode.start_tick + self.cfg.episode_steps >= len(self.mission):
            raise ValueError("the episode runs past the end of its mission")
        self._k0 = episode.start_tick
        self.tick = 0
        m, k = self.mission, self._k0
        psi = _yaw(m.R[k])
        Rr = _yaw_matrix(psi)
        nu = np.zeros(6)
        nu[:3] = Rr.T @ m.world_velocity(k)       # the ROV launches moving with the REMUS
        self.core.reset(position=m.pos[k] + np.asarray(episode.offset), nu=nu,
                        orientation=self._Rotation.from_matrix(Rr))
        plc, bwc = SEVERITIES[self.comm.severity]
        self.channel = CommChannel(plc, bwc, speed_of_sound_mps=self.comm.speed_of_sound_mps, seed=episode.comm_seed,
                                   fixed_latency_s=self.comm.fixed_latency_s)
        self.belief.reset()
        self.belief.on_tick(0, [self._beacon(0)])   # the launch state is handed over before the dive
        self._done = False
        self._stats = dict(ticks=0, judged=0, in_band=0, range_err=0.0, belief_err=[], turn=[], band=[], sent=0, delivered=0,
                           age=0.0, collision=False, lost=False, max_rov_speed=0.0)
        return self._observation(self._estimate())

    def _beacon(self, tick: int) -> Beacon:
        m, k = self.mission, self._k0 + tick
        dr, ds, rpm = m.fins[k]
        return Beacon(send_tick=tick, position=m.pos[k].copy(), orientation=m.R[k].copy(), nu=m.nu[k].copy(),
                      tau=np.zeros(6), controls=controls_for(self.remus.basis, dr, ds, rpm))

    def remus_true(self, tick: Optional[int] = None) -> tuple:
        k = self._k0 + (self.tick if tick is None else tick)
        return self.mission.pos[k], self.mission.world_velocity(k), self.mission.R[k]

    def _estimate(self) -> tuple:
        """(believed REMUS position, believed REMUS world velocity)."""
        p_true, v_true, _ = self.remus_true()
        b = self.belief
        pos = b.estimate(true_teammate_position=p_true)
        kind = self.belief_spec.kind
        if kind == "oracle":
            vel = v_true
        elif kind == "last_beacon":
            vel = np.zeros(3)
        elif kind == "constant_velocity":
            bb = b._beacon
            vel = bb.orientation @ bb.nu[:3]
        else:
            vel = b._core.world_velocity()
        return pos, np.asarray(vel, dtype=float)

    def step(self, action) -> tuple:
        if self._done:
            raise RuntimeError("step() after the episode ended: reset() first")
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        tau = np.zeros(6)
        tau[list(ACTION_AXES)] = a * self.rov.scale
        tau = np.asarray(self.rov.thrusters.achievable_wrench(tau), dtype=float)
        self.core.step(tau)
        self.tick += 1
        t = self.tick * self.cfg.dt
        p_rov = self.core.position_array()
        p_rem, v_rem, _ = self.remus_true()

        delivered = []
        if self.tick % self._every == 0:
            self.channel.send(sender_id="remus", receiver_id="rov", send_time=t, sender_pos=p_rem, receiver_pos=p_rov,
                              message_type=MESSAGE_TYPE_POSITION_BEACON, payload=self._beacon(self.tick),
                              payload_size_bytes=BEACON_SIZE_BYTES)
            self._stats["sent"] += 1
        self.channel.step(current_time=t)
        delivered = [msg.payload for msg in self.channel.receive("rov")]
        self._stats["delivered"] += len(delivered)
        self.belief.on_tick(self.tick, delivered)
        est_pos, est_vel = self._estimate()

        cfg = self.cfg
        rng_true = float(np.linalg.norm(p_rem - p_rov))
        reward = (-cfg.w_range * abs(rng_true - cfg.target_range_m) - cfg.w_ctrl * float(a @ a)
                  - cfg.w_prox * max(0.0, (cfg.influence_m - rng_true) / cfg.influence_m))
        collision = rng_true < cfg.collision_m
        lost = rng_true > cfg.lost_m
        terminated = collision or lost
        truncated = (self.tick >= cfg.episode_steps) and not terminated
        if terminated:
            reward -= (cfg.episode_steps - self.tick) * cfg.worst_tick_cost
        s = self._stats
        s["ticks"] = self.tick
        in_band = abs(rng_true - cfg.target_range_m) <= cfg.band_m
        turning = self.mission.turning(self._k0 + self.tick)
        berr = float(np.linalg.norm(est_pos - p_rem))
        s["belief_err"].append(berr)
        s["turn"].append(turning)
        s["range_err"] += abs(rng_true - cfg.target_range_m)
        s["age"] += self.belief.age_s
        s["max_rov_speed"] = max(s["max_rov_speed"], float(np.linalg.norm(self.core.nu_array()[:3])))
        if t > cfg.acquire_s + 1e-9:
            s["judged"] += 1
            s["in_band"] += int(in_band)
            s["band"].append((in_band, turning))
        s["collision"] |= collision
        s["lost"] |= lost
        self._done = terminated or truncated
        info = {"range": rng_true, "belief_error": berr, "in_band": in_band, "turning": turning,
                "delivered": bool(delivered), "belief_age_s": self.belief.age_s,
                "terminal_reason": "collision" if collision else "lost" if lost else ("time" if truncated else None)}
        return self._observation((est_pos, est_vel)), float(reward), bool(terminated), bool(truncated), info

    def _observation(self, est) -> np.ndarray:
        est_pos, est_vel = est
        c = self.core
        R = np.array(c.R)
        roll, pitch = c.roll_pitch()
        rel = R.T @ (est_pos - c.position_array())
        relv = R.T @ (est_vel - c.world_velocity())
        age = min(self.belief.age_s, self.cfg.age_cap_s) / self.cfg.age_cap_s
        return np.concatenate([c.nu_array(), [roll, pitch], rel, relv, [age, 1.0]]).astype(np.float32)

    def summary(self) -> dict:
        """The episode's outcome once it has ended."""
        s, cfg = self._stats, self.cfg
        judged_ticks = int(round((cfg.episode_steps * cfg.dt - cfg.acquire_s) / cfg.dt))
        frac = s["in_band"] / judged_ticks if judged_ticks else float("nan")   # ticks lost to an early end count as out
        be = np.array(s["belief_err"], dtype=float)
        tu = np.array(s["turn"], dtype=bool)
        band = s["band"]
        ib_turn = [b for b, t in band if t]
        ib_str = [b for b, t in band if not t]
        ok = not s["collision"] and not s["lost"] and frac >= cfg.success_in_band
        return {
            "success": int(ok), "collision": int(s["collision"]), "lost": int(s["lost"]), "steps": s["ticks"],
            "in_band_frac": frac,
            "in_band_frac_straight": float(np.mean(ib_str)) if ib_str else float("nan"),
            "in_band_frac_turn": float(np.mean(ib_turn)) if ib_turn else float("nan"),
            "mean_range_error_m": s["range_err"] / max(s["ticks"], 1),
            "mean_belief_error_m": float(be.mean()) if be.size else float("nan"),
            "belief_error_straight_m": float(be[~tu].mean()) if (~tu).any() else float("nan"),
            "belief_error_turn_m": float(be[tu].mean()) if tu.any() else float("nan"),
            "p90_belief_error_m": float(np.percentile(be, 90)) if be.size else float("nan"),
            "turn_frac": float(tu.mean()) if tu.size else float("nan"),
            "beacons_sent": s["sent"], "beacons_delivered": s["delivered"],
            "delivery_rate": s["delivered"] / s["sent"] if s["sent"] else float("nan"),
            "mean_belief_age_s": s["age"] / max(s["ticks"], 1),
            "max_rov_speed_mps": s["max_rov_speed"],
        }


def run_episode(sim: EscortSim, policy, episode: EscortEpisode) -> dict:
    obs = sim.reset(episode)
    done = False
    while not done:
        act, _ = policy.predict(obs[None, :], deterministic=True)
        obs, _, term, trunc, _ = sim.step(np.asarray(act).reshape(-1)[:ACT_DIM])
        done = term or trunc
    return sim.summary()


# --------------------------------------------------------------------------- a scripted escort (no learning)

class EscortPursuit:
    """A hand-written escort controller on the same observation: drag feed-forward for the ROV's own velocity (the ROV
    knows its own physics), plus a PD on the believed range error and relative velocity, and a yaw loop that faces the
    believed REMUS. Shows a result isn't an artefact of one learned policy, and checks the task is feasible."""

    def __init__(self, rov: RovModel, target_range_m: float = 3.0, kp: float = 0.6, kd: float = 1.6,
                 k_yaw: float = 1.5, k_r: float = 2.0, mass_kg: float = 25.0):
        self.rov, self.target = rov, target_range_m
        self.kp, self.kd, self.k_yaw, self.k_r, self.m = kp, kd, k_yaw, k_r, mass_kg
        p = rov.params
        self.lin = np.array([p.Xu, p.Yv, p.Zw])
        self.quad = np.array([p.Xuu, p.Yvv, p.Zww])
        self.n_lin, self.n_quad = p.Nr, p.Nrr

    def predict(self, obs, deterministic: bool = True):
        out = []
        for o in np.atleast_2d(obs):
            v = o[0:3].astype(float)
            r = float(o[5])
            rel, relv = o[8:11].astype(float), o[11:14].astype(float)
            d = float(np.linalg.norm(rel))
            los = rel / d if d > 1e-6 else np.zeros(3)
            ff = -(self.lin + self.quad * np.abs(v)) * v            # cancels the ROV's own drag
            F = ff + self.m * (self.kp * (d - self.target) * los + self.kd * relv)
            yaw_err = math.atan2(rel[1], rel[0]) if d > 1e-6 else 0.0
            N = -(self.n_lin + self.n_quad * abs(r)) * r + self.k_yaw * yaw_err - self.k_r * r
            a = np.array([F[0], F[1], F[2], N]) / self.rov.scale
            out.append(np.clip(a, -1.0, 1.0))
        return np.array(out, dtype=np.float32), None
