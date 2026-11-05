"""An independent fin: written from the physics with vectors and a rotation, not from fin_force.py. A fin is a flat foil that
turns about its span. Water passes it; lift is across the flow, drag along it, both in the plane normal to the span."""
import numpy as np


def rot(axis, angle):
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def fin_wrench(pos, span, area, span_len, slope, inflow, cd0, oswald, delta, v, rho=1025.0, stall=10.0):
    """Below stall. Returns the 6-vector about the origin. delta > 0 lifts toward span x chord."""
    pos, span = np.asarray(pos, float), np.asarray(span, float)
    chord = np.array([1.0, 0, 0])
    V = np.asarray(v[:3], float) + np.cross(v[3:], pos)       # the point's velocity through the water
    Vp = V - (V @ span) * span                                # spanwise flow doesn't count
    U = np.linalg.norm(Vp)
    if U == 0:
        return np.zeros(6)
    normal = np.cross(span, chord)
    flow = np.arctan2(Vp @ normal, Vp @ chord)
    flow = (flow + np.pi / 2) % np.pi - np.pi / 2
    alpha = delta - inflow * flow
    CL = slope * alpha
    AR = span_len ** 2 / area
    CD = cd0 + CL ** 2 / (np.pi * oswald * AR)
    drag_dir = -Vp / U                                        # the water pushes the fin back along its motion
    lift_dir = rot(span, np.pi / 2) @ (Vp / U)                # across the motion: for forward motion, along +normal
    if (lift_dir @ normal) * (Vp @ chord) < 0:
        lift_dir = -lift_dir
    F = .5 * rho * area * U ** 2 * (CL * lift_dir + CD * drag_dir)
    return np.r_[F, np.cross(pos, F)]
