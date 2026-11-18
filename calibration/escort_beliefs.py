"""
escort_beliefs.py -- what the REMUS's beacon carries, and how the ROV's model of it uses that.

Three beacon contents, all predicted with Strathe's fin model of the REMUS (calibrated or generic parameters):

    state only           position, attitude, velocity. The ROV knows the mission's propeller rpm (set before the
                         dive) but not the fins, so its model holds them centred.         -> `calibrated_nofins`
    state + actuators    the above plus the fin angles and rpm on the send tick, held       -> `calibrated`, `generic`
                         until the next beacon (zero-order hold). The main study's models.
    state + intent       the above plus the autopilot's heading and depth orders. Between    -> `calibrated_intent`,
                         beacons the ROV flies its model under the REMUS's own autopilot law      `generic_intent`
                         (escort_remus.autopilot and its depth loop, at the same 100 Hz), so the predicted vehicle
                         is steered as the real one is, instead of coasting on frozen fins.

Assumptions of the intent model, stated in the paper: the ROV knows the REMUS's autopilot law and gains (a team
operates its own vehicles), and the orders on the send tick stay in force until the next beacon (a turn ordered
between beacons is missed until one arrives). The autopilot's pitch integrator isn't sent: it is set on each beacon
so the law reproduces the elevator angle the beacon reports.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from dynamics_model import CalibrationParameters, SimulationConfig, VehicleConstants
from escort_remus import AUTOPILOT_H, DEPTH_GAIN, PITCH_ORDER_LIMIT_RAD, autopilot, controls_for
from fast_dynamics import FastCore
from fleet_belief import Beacon, PhysicsBelief

INTENT_BYTES = 8     # two float32 orders


@dataclass
class EscortBeacon(Beacon):
    """A REMUS beacon. fins: Prestero's (delta_r, delta_s, rpm) on the send tick (the same information as `controls`,
    in his convention); intent: the autopilot's (heading order rad, depth order m) on the send tick."""
    fins: Optional[tuple] = None
    intent: Optional[tuple] = None


class NoFinsBelief(PhysicsBelief):
    """State-only beacons: the fins are unknown and held centred; the rpm is the mission's."""
    kind = "physics_nofins"

    def __init__(self, dt, params, vehicle, basis, physics_dt=0.005):
        super().__init__(dt, params, vehicle, physics_dt)
        self._basis = basis

    def _on_new_beacon(self, beacon, now_tick):
        rpm = beacon.fins[2]
        super()._on_new_beacon(dataclasses.replace(beacon, controls=controls_for(self._basis, 0.0, 0.0, rpm)), now_tick)


class IntentBelief(PhysicsBelief):
    """State + intent beacons: the model is flown under the REMUS autopilot toward the beaconed orders."""
    kind = "physics_intent"

    def __init__(self, dt, params, vehicle, basis, physics_dt=0.005, autopilot_h=AUTOPILOT_H):
        super().__init__(dt, params, vehicle, physics_dt)
        n = dt / autopilot_h
        if abs(n - round(n)) > 1e-9:
            raise ValueError("the tick must be a whole number of autopilot steps")
        self._n, self._h = int(round(n)), autopilot_h
        self._basis = basis
        # one autopilot step per core step, as on the vehicle
        self._core = FastCore(params, vehicle, SimulationConfig(dt=autopilot_h, physics_dt=min(physics_dt, autopilot_h)))
        self._ii = 0.0

    def set_buoyancy_offset(self, offset_n):
        raise NotImplementedError("not used by the escort")

    def _state(self):
        c = self._core
        R = np.array(c.R)
        roll, pitch = c.roll_pitch()
        nu = c.nu_array()
        return pitch, float(nu[4]), math.atan2(R[1, 0], R[0, 0]), float(nu[5]), float(c.position_array()[2])

    def _on_new_beacon(self, beacon, now_tick):
        if beacon.intent is None or beacon.fins is None:
            raise ValueError("an intent model needs beacons with fins and orders")
        self._core.reset(position=beacon.position, nu=beacon.nu, orientation=Rotation.from_matrix(beacon.orientation))
        self._psi_d, self._depth_d = float(beacon.intent[0]), float(beacon.intent[1])
        self._rpm = float(beacon.fins[2])
        # set the pitch integrator so the law gives the elevator angle the beacon reports (if it isn't saturated)
        pitch, q, _, _, z = self._state()
        th_d = float(np.clip(DEPTH_GAIN * (z - self._depth_d), -PITCH_ORDER_LIMIT_RAD, PITCH_ORDER_LIMIT_RAD))
        ds = float(beacon.fins[1])
        self._ii = float(np.clip((ds - 1.5 * (pitch - th_d) - 1.2 * q) / 0.4, -.5, .5))
        for _ in range(max(0, now_tick - beacon.send_tick)):
            self._step()

    def _step(self):
        try:
            for _ in range(self._n):
                pitch, q, yaw, r, z = self._state()
                th_d = float(np.clip(DEPTH_GAIN * (z - self._depth_d), -PITCH_ORDER_LIMIT_RAD, PITCH_ORDER_LIMIT_RAD))
                ds, dr, self._ii = autopilot(pitch, q, yaw, r, self._psi_d, th_d, self._ii, self._h)
                self._core.controls = [float(c) for c in controls_for(self._basis, dr, ds, self._rpm)]
                self._core.step([0.0] * 6)
        except FloatingPointError:
            b = self._beacon
            self._core.reset(position=b.position, nu=np.zeros(6), orientation=Rotation.from_matrix(b.orientation))


@dataclass
class EscortBeliefSpec:
    """Like fleet_belief.BeliefSpec, for the escort-only beacon contents."""
    kind: str                       # "physics_nofins" | "physics_intent"
    label: str
    params: CalibrationParameters
    basis: tuple

    def build(self, dt: float, vehicle: VehicleConstants):
        cls = {"physics_nofins": NoFinsBelief, "physics_intent": IntentBelief}[self.kind]
        return cls(dt, self.params, vehicle, self.basis)
