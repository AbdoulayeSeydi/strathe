"""
fleet_coordination.py -- N vehicles keeping a formation over a shared acoustic channel (PROJECT_STATUS.md #89).

What a team tests with it: does our coordination still work with THIS modem's latency, bandwidth and losses, with
THIS many vehicles? The vehicles run Strathe's calibrated physics; every teammate is known only through acoustic
beacons (acoustic_network.py) and a belief model between them (fleet_belief.py).

## The task: follow an unpredictable leader
Vehicle 0 (the leader) runs a survey whose turns the others can't know in advance: straight legs at `speed_mps`, with
a turn of +-`turn_deg` at a random time inside each leg window (seeded per episode). Every other vehicle holds its
slot in the formation, defined in the leader's frame (`formation`, `spacing_m`). A follower only knows where the
leader is from the leader's beacons, so the formation holds exactly as well as the channel lets it.

Formations (slot offsets, metres: x forward, y starboard), in one of two frames (`orientation`):
    fixed    the formation keeps its compass orientation (north = the survey's first heading), so each slot moves
             with the leader's velocity. The default: what survey formations usually do, and always flyable.
    leader   the formation turns with the leader. A slot r metres from the leader then moves at
             speed + r x turn rate, which can exceed what the vehicle can do: the config refuses a turn faster than
             `max_slot_speed_mps` allows.

    line     side by side:   (0, +-s), (0, +-2s), ...
    column   one behind another: (-s, 0), (-2s, 0), ...
    wedge    a V behind the leader: (-s, +-s), (-2s, +-2s), ...
    custom   your own offsets (FormationSpec.offsets)

## Controllers
- `leader_follower` (built in): each follower steers to (believed leader position + the slot rotated by the
  believed leader heading), with the believed leader velocity as a feed-forward, plus a repulsion from any teammate
  believed closer than `avoid_radius_m`. The leader tracks its own planned path. A PD controller with saturation at
  the vehicle's thrust limit.
- your own: any class with `act(i, own, teammates, t) -> 6 wrench values` (COMMS_STUDY.md), passed as
  `module:Class`. `own` is the vehicle's own state; `teammates` maps each teammate id to what this vehicle believes
  about it (or None before its first beacon).

## Scored on the truth
- formation error: each follower's distance from where its slot really is (the TRUE leader + the slot rotated by the
  TRUE leader heading), RMS over followers and time, after a settling time;
- minimum separation between any two vehicles, and collisions (closer than `safety_radius_m`);
- belief error (believed vs true teammate position) and the age of the newest beacon each vehicle holds;
- the channel: delivered / collided / lost packets, utilisation, the TDMA cycle.
An episode succeeds when the formation error stays under `tolerance_m` (RMS) and nothing collides.
"""
from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

from acoustic_network import AcousticNetwork, NetworkConfig
from dynamics_model import CalibrationParameters, SimulationConfig, VehicleConstants
from fast_dynamics import FastCore
from fleet_belief import Beacon, BeliefSpec

FORMATIONS = ("line", "column", "wedge", "custom")
MAX_VEHICLES = 12


@dataclass
class FormationSpec:
    shape: str = "wedge"
    spacing_m: float = 10.0
    offsets: Optional[list] = None   # custom: [(x, y), ...] for vehicles 1..N-1, metres in the leader's frame
    orientation: str = "fixed"       # "fixed" (compass) or "leader" (turns with the leader)

    def slots(self, n: int) -> np.ndarray:
        """(n, 3) offsets in the leader's frame; row 0 (the leader) is zero."""
        s = self.spacing_m
        out = [(0.0, 0.0)]
        if self.shape == "custom":
            if not self.offsets or len(self.offsets) < n - 1:
                raise ValueError(f"a custom formation needs {n - 1} offsets (x y for each follower)")
            out += [tuple(map(float, o[:2])) for o in self.offsets[:n - 1]]
        else:
            for k in range(1, n):
                rank, side = (k + 1) // 2, (1 if k % 2 else -1)
                if self.shape == "line":
                    out.append((0.0, side * rank * s))
                elif self.shape == "column":
                    out.append((-k * s, 0.0))
                elif self.shape == "wedge":
                    out.append((-rank * s, side * rank * s))
                else:
                    raise ValueError(f"formation must be one of {FORMATIONS}, got {self.shape!r}")
        return np.array([[x, y, 0.0] for x, y in out])

    def validate(self, n: int) -> None:
        if self.shape not in FORMATIONS:
            raise ValueError(f"formation must be one of {FORMATIONS}, got {self.shape!r}")
        if self.orientation not in ("fixed", "leader"):
            raise ValueError(f"formation orientation must be 'fixed' or 'leader', got {self.orientation!r}")
        if not 1.0 <= self.spacing_m <= 1000.0:
            raise ValueError(f"spacing must be 1 to 1000 m, got {self.spacing_m}")
        sl = self.slots(n)
        d = [np.linalg.norm(sl[i] - sl[j]) for i in range(n) for j in range(i)]
        if d and min(d) < 1.0:
            raise ValueError("two formation slots are less than 1 m apart")


@dataclass
class CoordinationConfig:
    n_vehicles: int = 4
    formation: FormationSpec = field(default_factory=FormationSpec)
    duration_s: float = 300.0
    settle_s: float = 30.0            # formation error is scored after this
    dt: float = 0.1
    physics_dt: float = 0.005
    speed_mps: float = 0.2
    leg_s: float = 60.0               # the leader turns once in each leg window, at a random time
    turn_deg: float = 60.0            # ... by +-this many degrees, over turn_rate_dps
    turn_rate_dps: float = 3.0
    depth_m: float = 10.0
    start_jitter_m: float = 1.0       # followers start this far (random) from their slots
    tolerance_m: float = 3.0          # success: RMS formation error under this, and no collision
    safety_radius_m: float = 1.5
    avoid_radius_m: float = 4.0
    max_force_n: float = 30.0
    max_torque_nm: float = 5.0
    max_slot_speed_mps: float = 0.3   # "leader" formations: the fastest a slot may move in a turn
    controller: str = "leader_follower"   # or "module:Class"
    belief: str = "constant_velocity"
    network: NetworkConfig = field(default_factory=NetworkConfig)

    def validate(self) -> None:
        if not 2 <= self.n_vehicles <= MAX_VEHICLES:
            raise ValueError(f"the fleet must have 2 to {MAX_VEHICLES} vehicles, got {self.n_vehicles}")
        self.formation.validate(self.n_vehicles)
        if not self.duration_s > self.settle_s >= 0:
            raise ValueError("duration must be longer than the settling time")
        k = self.dt / self.physics_dt
        if abs(k - round(k)) > 1e-9:
            raise ValueError("dt must be a whole number of physics steps")
        for name in ("speed_mps", "leg_s", "turn_rate_dps", "tolerance_m", "safety_radius_m", "max_force_n",
                     "max_torque_nm"):
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.turn_deg <= 180:
            raise ValueError("turn must be 0 to 180 degrees")
        if self.belief not in ("last_beacon", "constant_velocity", "physics", "oracle"):
            raise ValueError(f"belief must be last_beacon, constant_velocity, physics or oracle, got {self.belief!r}")
        self.network.validate()
        if self.formation.orientation == "leader":
            r = max(np.linalg.norm(o[:2]) for o in self.formation.slots(self.n_vehicles))
            fastest = self.speed_mps + r * math.radians(self.turn_rate_dps)
            if fastest > self.max_slot_speed_mps:
                raise ValueError(
                    f"with the formation turning with the leader, the outermost slot ({r:.0f} m out) moves at "
                    f"{fastest:.2f} m/s in a {self.turn_rate_dps:g} deg/s turn, faster than {self.max_slot_speed_mps:g} "
                    f"m/s: slow the turn to {math.degrees((self.max_slot_speed_mps - self.speed_mps) / r):.2f} deg/s "
                    f"or less, or keep the formation's orientation fixed")

    def max_distance_m(self) -> float:
        sl = self.formation.slots(self.n_vehicles)
        return max(np.linalg.norm(sl[i] - sl[j]) for i in range(len(sl)) for j in range(i)) * 1.5 + 5.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["network"] = self.network.to_dict()
        return d


def _yaw(R) -> float:
    return math.atan2(R[1][0], R[0][0])


def _rz(psi: float) -> np.ndarray:
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class LeaderPlan:
    """The leader's survey: heading changes by +-turn_deg once per leg window, at a random time. Only the leader
    knows it."""

    def __init__(self, cfg: CoordinationConfig, rng: np.random.Generator):
        self.cfg = cfg
        n_legs = int(math.ceil(cfg.duration_s / cfg.leg_s)) + 1
        self.turns = []   # (start time, delta heading rad)
        for k in range(1, n_legs):
            t = (k - 1) * cfg.leg_s + cfg.leg_s * float(rng.uniform(0.4, 1.0)) + cfg.settle_s
            self.turns.append((t, math.radians(cfg.turn_deg) * (1 if rng.random() < 0.5 else -1)))
        self._rate = math.radians(cfg.turn_rate_dps)

    def heading(self, t: float) -> float:
        psi = 0.0
        for t0, dpsi in self.turns:
            if t <= t0:
                break
            psi += math.copysign(min(abs(dpsi), self._rate * (t - t0)), dpsi)
        return psi

    def state(self, t: float, dt: float) -> tuple:
        """Reference position and velocity at t (integrated heading at speed), cached per tick."""
        return self._pos(t), self.cfg.speed_mps * np.array([math.cos(self.heading(t)), math.sin(self.heading(t)), 0.0])

    def _pos(self, t: float) -> np.ndarray:
        # integrate the heading in 0.5 s steps (exact enough for a reference)
        h = 0.5
        if not hasattr(self, "_cache"):
            self._cache = [np.array([0.0, 0.0, self.cfg.depth_m])]
        while (len(self._cache) - 1) * h < t:
            k = len(self._cache) - 1
            psi = self.heading((k + 0.5) * h)
            self._cache.append(self._cache[-1] + self.cfg.speed_mps * h * np.array([math.cos(psi), math.sin(psi), 0]))
        k = int(t // h)
        frac = (t - k * h) / h
        a = self._cache[k]
        b = self._cache[min(k + 1, len(self._cache) - 1)]
        return a + (b - a) * frac


@dataclass
class VehicleState:
    """What a controller gets about itself."""
    position: np.ndarray      # world (north, east, down), m
    velocity: np.ndarray      # world, m/s
    R: np.ndarray             # body -> world
    nu: np.ndarray            # body velocity [u v w p q r]


@dataclass
class TeammateView:
    """What a controller gets about a teammate: this vehicle's belief, never the truth."""
    position: np.ndarray
    velocity: np.ndarray
    heading: float            # rad
    age_s: float              # how old the newest beacon is


class LeaderFollower:
    """The built-in controller (see the module docstring)."""

    def __init__(self, cfg: CoordinationConfig, plan: LeaderPlan):
        self.cfg, self.plan = cfg, plan
        self.slots = cfg.formation.slots(cfg.n_vehicles)
        self._integral = {}   # per vehicle: integrated position error (holds the force drag needs at cruise)

    def _track(self, own: VehicleState, p_ref, v_ref, psi_ref, extra=None, i: int = 0) -> np.ndarray:
        c = self.cfg
        e = np.asarray(p_ref) - own.position
        I = self._integral.get(i, np.zeros(3)) + e * c.dt
        I = np.clip(I, -c.max_force_n / 0.6, c.max_force_n / 0.6)   # anti-windup: at most the full force
        self._integral[i] = I
        f_world = 6.0 * e + 0.6 * I + 40.0 * (np.asarray(v_ref) - own.velocity)
        if extra is not None:
            f_world = f_world + extra
        f_world[2] = 40.0 * e[2] - 40.0 * own.velocity[2]
        f_body = own.R.T @ f_world
        f_body = np.clip(f_body, -c.max_force_n, c.max_force_n)
        roll = math.atan2(own.R[2][1], own.R[2][2])
        pitch = -math.asin(max(-1.0, min(1.0, own.R[2][0])))
        tq = np.array([-4.0 * roll - 2.0 * own.nu[3], -4.0 * pitch - 2.0 * own.nu[4],
                       3.0 * _wrap(psi_ref - _yaw(own.R)) - 2.0 * own.nu[5]])
        return np.concatenate([f_body, np.clip(tq, -c.max_torque_nm, c.max_torque_nm)])

    def act(self, i: int, own: VehicleState, teammates: dict, t: float) -> np.ndarray:
        c = self.cfg
        if i == 0:
            p, v = self.plan.state(t, c.dt)
            return self._track(own, p, v, self.plan.heading(t), i=0)
        lead = teammates.get(0)
        if lead is None:   # nothing heard from the leader yet: hold position
            return self._track(own, own.position, np.zeros(3), _yaw(own.R), i=i)
        Rl = _rz(lead.heading if c.formation.orientation == "leader" else 0.0)
        target = lead.position + Rl @ self.slots[i]
        target[2] = c.depth_m
        push = np.zeros(3)
        for j, tm in teammates.items():
            if tm is None:
                continue
            d = own.position - tm.position
            d[2] = 0.0
            r = float(np.linalg.norm(d))
            if 1e-6 < r < c.avoid_radius_m:
                push += 30.0 * (c.avoid_radius_m - r) / c.avoid_radius_m * d / r
        return self._track(own, target, lead.velocity, lead.heading, push, i=i)


def load_controller(spec: str, cfg: CoordinationConfig, plan: LeaderPlan):
    if spec == "leader_follower":
        return LeaderFollower(cfg, plan)
    if ":" not in spec:
        raise ValueError(f"controller must be 'leader_follower' or 'module:Class', got {spec!r}")
    mod, cls = spec.split(":", 1)
    try:
        C = getattr(importlib.import_module(mod), cls)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"can't load controller {spec!r}: {e}") from e
    try:
        return C(cfg, plan)
    except TypeError:
        return C()


class CoordinationSim:
    """One fleet, many episodes: `run_episode(seed)` -> a dict of metrics."""

    def __init__(self, truth: CalibrationParameters, vehicle: VehicleConstants, cfg: CoordinationConfig,
                 belief_params: Optional[CalibrationParameters] = None):
        cfg.validate()
        self.truth, self.vehicle, self.cfg = truth, vehicle, cfg
        self.belief_spec = BeliefSpec(cfg.belief, cfg.belief,
                                      (belief_params or truth) if cfg.belief == "physics" else None)

    def run_episode(self, seed: int, record: bool = False) -> dict:
        cfg = self.cfg
        n = cfg.n_vehicles
        rng = np.random.default_rng([seed, 0xF1EE7])
        plan = LeaderPlan(cfg, rng)
        ctrl = load_controller(cfg.controller, cfg, plan)
        sim = SimulationConfig(dt=cfg.dt, physics_dt=cfg.physics_dt)
        cores = [FastCore(self.truth, self.vehicle, sim) for _ in range(n)]
        slots = cfg.formation.slots(n)
        p0, v0 = plan.state(0.0, cfg.dt)
        for i, core in enumerate(cores):
            jitter = np.zeros(3) if i == 0 else np.append(rng.normal(0, cfg.start_jitter_m, 2), 0.0)
            core.reset(position=p0 + slots[i] + jitter, nu=np.array([cfg.speed_mps, 0, 0, 0, 0, 0]))
        net = AcousticNetwork(n, cfg.network, seed=int(rng.integers(2**31)), max_distance_m=cfg.max_distance_m())
        beliefs = [{j: self.belief_spec.build(cfg.dt, self.vehicle) for j in range(n) if j != i} for i in range(n)]
        taus = [np.zeros(6) for _ in range(n)]
        steps = int(round(cfg.duration_s / cfg.dt))
        settle = int(round(cfg.settle_s / cfg.dt))
        err2, n_err, worst_err = 0.0, 0, 0.0
        min_sep, collided = math.inf, False
        bel_err, ages, n_bel = 0.0, 0.0, 0
        per_follower = np.zeros(n)
        trace = [] if record else None
        diverged = False
        for k in range(1, steps + 1):
            t = k * cfg.dt
            try:
                for core, tau in zip(cores, taus):
                    core.step(tau)
            except FloatingPointError:
                diverged = True
                break
            pos = [c.position_array() for c in cores]
            vel = [c.world_velocity() for c in cores]
            Rs = [np.array(c.R) for c in cores]
            nus = [c.nu_array() for c in cores]

            def beacon(i, _k=k):
                return Beacon(send_tick=_k, position=pos[i].copy(), orientation=Rs[i].copy(), nu=nus[i].copy(),
                              tau=taus[i].copy())
            delivered = net.advance(t, pos, lambda i: (i, beacon(i)))
            for i in range(n):
                by_sender = {}
                for sender, b in delivered[i]:
                    by_sender.setdefault(sender, []).append(b)
                for j, bel in beliefs[i].items():
                    bel.on_tick(k, by_sender.get(j, []))
            # scoring on the truth
            Rl = _rz(_yaw(Rs[0]) if cfg.formation.orientation == "leader" else 0.0)
            for i in range(n):
                for j in range(i):
                    d = float(np.linalg.norm(pos[i] - pos[j]))
                    min_sep = min(min_sep, d)
                    collided |= d < cfg.safety_radius_m
            if k > settle:
                for i in range(1, n):
                    e = float(np.linalg.norm((pos[i] - (pos[0] + Rl @ slots[i]))[:2]))
                    err2 += e * e
                    per_follower[i] += e * e
                    worst_err = max(worst_err, e)
                    n_err += 1
            views = []
            for i in range(n):
                view = {}
                for j, bel in beliefs[i].items():
                    st = bel.estimate_state((pos[j], vel[j], Rs[j]))
                    if st is None:
                        view[j] = None
                        continue
                    view[j] = TeammateView(st[0], st[1], _yaw(st[2]), bel.age_s)
                    if k > settle:
                        bel_err += float(np.linalg.norm(st[0] - pos[j]))
                        ages += min(bel.age_s, cfg.duration_s)
                        n_bel += 1
                views.append(view)
            for i in range(n):
                own = VehicleState(pos[i], vel[i], Rs[i], nus[i])
                taus[i] = np.asarray(ctrl.act(i, own, views[i], t), dtype=float)
            if record:
                trace.append({"t": t, "positions": [p.copy() for p in pos],
                              "slots": [pos[0] + Rl @ slots[i] for i in range(n)]})
        rms = math.sqrt(err2 / n_err) if n_err else math.nan
        out = {
            "seed": seed, "formation_error_rms_m": rms, "formation_error_max_m": worst_err,
            "worst_follower_rms_m": float(np.sqrt(per_follower[1:].max() / max(1, n_err // max(1, n - 1)))),
            "min_separation_m": min_sep, "collision": bool(collided), "diverged": diverged,
            "belief_error_m": bel_err / n_bel if n_bel else math.nan,
            "beacon_age_s": ages / n_bel if n_bel else math.nan,
            "success": bool((not diverged) and (not collided) and rms <= cfg.tolerance_m),
            "delivery_ratio": net.delivery_ratio(), "utilisation": net.utilisation(),
            "tdma_cycle_s": net.cycle if cfg.network.mac == "tdma" else math.nan,
            "airtime_s": net.airtime, **{f"n_{k_}": v for k_, v in net.counts.items()},
        }
        if record:
            out["trace"] = trace
        return out
