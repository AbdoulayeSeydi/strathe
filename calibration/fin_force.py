"""
fin_force.py -- what a fin-steered vehicle's actuators do to it (PROJECT_STATUS.md #97).

A thruster's force can be logged. A fin's can't: it depends on how fast the water passes the fin and at what angle, so it's
worked out inside the physics at every step, from the state and the logged fin angle. This is the one place that's done:
dynamics_model.py and fast_dynamics.py both call actuator_wrench.

Each sample's controls are one row of numbers (Actuators.width of them), in this order: every fin's angle (rad), the
propeller's rpm, the two gimbal angles of a vectored thruster (rad), the moving mass's travel (m), the buoyancy engine's change
(N). A vehicle only has the columns its actuators use.

Per fin, with the water's velocity past the fin's centre of pressure resolved on its chord (v_c) and normal (v_n):
    alpha = angle + bias - inflow * atan2(v_n, v_c)      inflow: how much of the flow the fin sees past the hull
    U^2   = v_c^2 + v_n^2                                 spanwise flow makes no lift
    F     = 0.5 rho S U [ CL (-v_n c + v_c n) - CD (v_c c + v_n n) ]    lift across the flow, drag along it
    CL    = slope * alpha, up to stall, then blending to Cn sin(alpha) cos(alpha)
    CD    = cd0 + CL^2 / (pi e AR)
There's no division by speed, so it's safe at rest. A centred fin still pushes back when the flow meets it at an angle, which is
why fins add damping that grows with speed. `slope` is the effective fin-and-hull product: logs can't separate the two.
Positive angle lifts toward +n (n = span x chord). Attitude isn't used here on purpose: a free run passes pseudo-angles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

STALL_BAND_RAD = 0.1   # alpha over which lift goes from the linear line to the stalled curve
STALLED_CN = 2.0       # a flat plate's normal force coefficient past stall


@dataclass(frozen=True)
class Fin:
    name: str
    position_m: tuple           # centre of pressure, from the torque reference point, forward-right-down
    span_axis: tuple            # unit vector along the fin's hinge, forward-right-down
    area_m2: float
    span_m: float
    group: int = 0              # 0: lift slope and inflow _a; 1: _b
    mix: tuple = (0.0, 0.0, 0.0)  # how much of the roll, pitch and yaw biases this fin takes
    stall_rad: float = 0.35
    oswald: float = 0.7
    limit_rad: float = 0.6      # the servo's end stop
    chord_axis: tuple = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class Propeller:
    diameter_m: float
    kt0: float                  # thrust coefficient at no advance
    kt1: float = 0.0            # its slope with advance ratio
    kq0: float = 0.0            # torque coefficient (reaction torque in roll)
    kq1: float = 0.0
    wake: float = 1.0           # advance speed = wake * forward speed through the water
    thrust_deduction: float = 0.0
    position_m: tuple = (0.0, 0.0, 0.0)
    spin: float = 1.0           # +1: reaction torque rolls to starboard for forward thrust; -1 the other way
    gimbal: bool = False        # a vectored thruster: two angle columns follow rpm


@dataclass(frozen=True)
class MovingMass:
    mass_kg: float              # what moves
    axis: tuple = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class Actuators:
    fins: tuple = ()
    propeller: Optional[Propeller] = None
    moving_mass: Optional[MovingMass] = None
    buoyancy_engine: bool = False
    rho: float = 1025.0
    servo_rate_rad_s: float = 0.0   # the fins' top speed; 0: none

    @property
    def width(self) -> int:
        p = self.propeller
        return (len(self.fins) + (1 if p else 0) + (2 if p and p.gimbal else 0) + (1 if self.moving_mass else 0)
                + (1 if self.buoyancy_engine else 0))

    def columns(self) -> list:
        """Names of a sample's control columns, in order."""
        c = [f"fin_{f.name}" for f in self.fins]
        if self.propeller:
            c.append("rpm")
            if self.propeller.gimbal:
                c += ["gimbal_y", "gimbal_z"]
        if self.moving_mass:
            c.append("moving_mass")
        if self.buoyancy_engine:
            c.append("buoyancy_change")
        return c


def _lift(alpha: float, slope: float, stall: float) -> float:
    a = abs(alpha)
    if a <= stall:
        return slope * alpha
    after = STALLED_CN * math.sin(alpha) * math.cos(alpha)
    if a >= stall + STALL_BAND_RAD:
        return after
    t = (a - stall) / STALL_BAND_RAD
    t = t * t * (3 - 2 * t)
    return (1 - t) * slope * alpha + t * after


def prepare(act: Actuators, params) -> tuple:
    """Everything per fin that doesn't change during a run, as plain numbers for the hot loop."""
    rows = []
    for i, f in enumerate(act.fins):
        cx, cy, cz = f.chord_axis
        sx, sy, sz = f.span_axis
        nx, ny, nz = sy * cz - sz * cy, sz * cx - sx * cz, sx * cy - sy * cx
        x, y, z = f.position_m
        slope = params.fin_lift_slope_b if f.group else params.fin_lift_slope_a
        inflow = params.fin_inflow_b if f.group else params.fin_inflow_a
        bias = f.mix[0] * params.fin_bias_roll + f.mix[1] * params.fin_bias_pitch + f.mix[2] * params.fin_bias_yaw
        ar = f.span_m * f.span_m / f.area_m2
        rows.append((i, x, y, z, cx, cy, cz, nx, ny, nz, .5 * act.rho * f.area_m2, slope, inflow, bias,
                     params.fin_drag0, 1 / (math.pi * f.oswald * ar), f.stall_rad))
    return tuple(rows)


def actuator_wrench(act: Actuators, fins: tuple, params, v, c, sr: float, cr: float, sp: float, cp: float) -> list:
    """The 6-DOF force and moment of every actuator at this instant. v: velocity through the water (u, v, w, p, q, r); c: this
    sample's controls; sr..cp as dynamics' restoring terms use them (only the products they use)."""
    u, vv, w, a, b, r = v
    X = Y = Z = K = M = N = 0.0
    for i, x, y, z, cx, cy, cz, nx, ny, nz, q, slope, inflow, bias, cd0, kind, stall in fins:
        lu = u + b * z - r * y          # velocity of the point: v + omega x r
        lv = vv + r * x - a * z
        lw = w + a * y - b * x
        vc = lu * cx + lv * cy + lw * cz
        vn = lu * nx + lv * ny + lw * nz
        U = math.sqrt(vc * vc + vn * vn)
        if U < 1e-12:
            continue
        phi = math.atan2(vn, vc)
        if phi > math.pi / 2:           # flow from behind: the same fin, seen from the other end
            phi -= math.pi
        elif phi < -math.pi / 2:
            phi += math.pi
        alpha = c[i] + bias - inflow * phi
        CL = _lift(alpha, slope, stall)
        CD = cd0 + CL * CL * kind
        fc = q * U * (-CL * vn - CD * vc)   # along the chord
        fn = q * U * (CL * vc - CD * vn)    # along the normal
        fx, fy, fz = fc * cx + fn * nx, fc * cy + fn * ny, fc * cz + fn * nz
        X += fx
        Y += fy
        Z += fz
        K += y * fz - z * fy
        M += z * fx - x * fz
        N += x * fy - y * fx
    k = len(act.fins)
    p = act.propeller
    if p is not None:
        n = c[k] / 60.0
        k += 1
        s = params.thrust_scale
        ja = p.wake * u / p.diameter_m
        if n > 0:
            T = act.rho * p.diameter_m ** 4 * (p.kt0 * n * n + p.kt1 * ja * n)
            Q = act.rho * p.diameter_m ** 5 * (p.kq0 * n * n + p.kq1 * ja * n)
        else:
            T = -act.rho * p.diameter_m ** 4 * p.kt0 * n * n
            Q = -act.rho * p.diameter_m ** 5 * p.kq0 * n * n
        T *= s * (1 - p.thrust_deduction)
        fx, fy, fz = T, 0.0, 0.0
        if p.gimbal:
            gy, gz = c[k], c[k + 1]
            k += 2
            fx, fy, fz = T * math.cos(gy) * math.cos(gz), T * math.sin(gz) * math.cos(gy), -T * math.sin(gy)
        x, y, z = p.position_m
        X += fx
        Y += fy
        Z += fz
        K += y * fz - z * fy + p.spin * Q
        M += z * fx - x * fz
        N += x * fy - y * fx
    if act.moving_mass is not None:
        d = c[k] * act.moving_mass.mass_kg * 9.81     # the shifted weight's moment arm: travel x its own weight
        k += 1
        ax, ay, az = act.moving_mass.axis
        dx, dz = ax * d, az * d
        K -= dz * cp * sr
        M -= dx * cp * cr + dz * sp
        N += dx * cp * sr
    if act.buoyancy_engine:
        dB = c[k]
        X += dB * sp
        Y -= dB * cp * sr
        Z -= dB * cp * cr
    return [X, Y, Z, K, M, N]


def servo_angles(act: Actuators, controls, time_constant_s: float, dt: float):
    """Commanded fin angles as the fins reach them: held to their end stops, lagged, then limited in speed. Applied to a
    whole recording, as the thruster lag is (objective.with_thruster_lag). Other columns aren't touched."""
    import numpy as np
    from dynamics_model import apply_thruster_lag
    out = np.array(controls, dtype=float)
    n = len(act.fins)
    if not n:
        return out
    out[:, :n] = np.clip(out[:, :n], [-f.limit_rad for f in act.fins], [f.limit_rad for f in act.fins])
    if time_constant_s > 0:
        out[:, :n] = apply_thruster_lag(out[:, :n], time_constant_s, dt)
    if act.servo_rate_rad_s > 0:
        step = act.servo_rate_rad_s * dt
        for k in range(1, len(out)):
            out[k, :n] = out[k - 1, :n] + np.clip(out[k, :n] - out[k - 1, :n], -step, step)
    return out
