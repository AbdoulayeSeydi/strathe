"""
stress/prestero_remus.py -- REMUS as Prestero (2001, MIT) modelled it: an independent truth for the fin proof (#97).

Written from the thesis's published coefficients, not from Strathe's model: lift and damping are his lumped derivatives
(Yuv, Yur, Muw, Muq: "body and fin lift"), his fins are Yuudr/Nuudr/Zuuds/Muuds, and his mass matrix has the off-diagonal terms
Yr_dot, Nv_dot, Zq_dot, Mw_dot and m*zg. A fit that recovers this vehicle has met flow-driven fin force and off-diagonal mass
it wasn't written from. Cross-checked against the vertical-plane values in Chuhran's 2003 NPS thesis at 1.5 m/s.

Yvv: -131, as Zww on an axisymmetric hull (a copy of the code in circulation has -1310, a slip).
Signs here are his: positive delta_r gives positive Y and negative N; positive delta_s gives negative Z and M.
"""
from __future__ import annotations

import math

import numpy as np

W, B, G = 299.0, 306.0, 9.8
M = W / G
ZG = 1.96e-2
XB = 0.0                     # the origin is at the centre of buoyancy
IXX, IYY, IZZ = 1.77e-1, 3.45, 3.45
XUU = -1.62
XWQ, XQQ, XVR, XRR = -35.5, -1.93, 35.5, -1.93
YVV, YRR, YUV, YWP, YUR, YPQ = -131.0, .632, -28.6, 35.5, 5.22, 1.93
ZWW, ZQQ, ZUW, ZUQ, ZVP, ZRP = -131.0, -.632, -28.6, -5.22, -35.5, 1.93
KPP, KPROP, KPDOT = -.13, -.543, -7.04e-2
MWW, MQQ, MRP, MUQ, MUW, MVP = 3.18, -188.0, 4.86, -2.0, 24.0, -1.93
NVV, NRR, NUV, NPQ, NWP, NUR = -3.18, -94.0, -24.0, -4.86, -1.93, -2.0
YUUDR, NUUDR, ZUUDS, MUUDS = 9.64, -6.15, -9.64, -6.15
XUDOT, YVDOT, ZWDOT, MQDOT, NRDOT = -.93, -35.5, -35.5, -4.88, -4.88
YRDOT, NVDOT, ZQDOT, MWDOT = 1.93, 1.93, -1.93, -1.93
THRUST_AT_TRIM_N, TRIM_RPM = 3.86, 1500.0       # his propeller term holds 1.5 m/s

MASS = np.array([[M - XUDOT, 0, 0, 0, M * ZG, 0],
                 [0, M - YVDOT, 0, -M * ZG, 0, -YRDOT],
                 [0, 0, M - ZWDOT, 0, -ZQDOT, 0],
                 [0, -M * ZG, 0, IXX - KPDOT, 0, 0],
                 [M * ZG, 0, -MWDOT, 0, IYY - MQDOT, 0],
                 [0, -NVDOT, 0, 0, 0, IZZ - NRDOT]])
MINV = np.linalg.inv(MASS)


def forces(nu, phi, theta, delta_s, delta_r, rpm) -> np.ndarray:
    """His equations of motion's right-hand side, in the water (a current is taken off nu by the caller)."""
    u, v, w, p, q, r = nu
    s1, c1, s2, c2 = math.sin(phi), math.cos(phi), math.sin(theta), math.cos(theta)
    xprop = THRUST_AT_TRIM_N * (rpm / TRIM_RPM) * abs(rpm / TRIM_RPM)
    kprop = KPROP * (rpm / TRIM_RPM) * abs(rpm / TRIM_RPM)
    X = -(W - B) * s2 + XUU * u * abs(u) + (XWQ - M) * w * q + XQQ * q * q + (XVR + M) * v * r + XRR * r * r - M * ZG * p * r + xprop
    Y = ((W - B) * c2 * s1 + YVV * v * abs(v) + YRR * r * abs(r) + YUV * u * v + (YWP + M) * w * p + (YUR - M) * u * r
         - M * ZG * q * r + YPQ * p * q + YUUDR * u * u * delta_r)
    Z = ((W - B) * c2 * c1 + ZWW * w * abs(w) + ZQQ * q * abs(q) + ZUW * u * w + (ZUQ + M) * u * q + (ZVP - M) * v * p
         + M * ZG * p * p + M * ZG * q * q + ZRP * r * p + ZUUDS * u * u * delta_s)
    K = -ZG * W * c2 * s1 + KPP * p * abs(p) - (IZZ - IYY) * q * r - M * ZG * w * p + M * ZG * u * r + kprop
    Mm = (-ZG * W * s2 + MWW * w * abs(w) + MQQ * q * abs(q) + (MRP - (IXX - IZZ)) * r * p + M * ZG * v * r - M * ZG * w * q
          + MUQ * u * q + MUW * u * w + MVP * v * p + MUUDS * u * u * delta_s)
    N = NVV * v * abs(v) + NRR * r * abs(r) + NUV * u * v + (NPQ - (IYY - IXX)) * p * q + NWP * w * p + NUR * u * r + NUUDR * u * u * delta_r
    return np.array([X, Y, Z, K, Mm, N])


def derivative(x, delta_s, delta_r, rpm, current_ned=(0.0, 0.0, 0.0)) -> np.ndarray:
    """x = u v w p q r | north east down | roll pitch yaw."""
    nu = x[:6]
    phi, th, ps = x[9:12]
    c1, s1, c2, s2, c3, s3 = math.cos(phi), math.sin(phi), math.cos(th), math.sin(th), math.cos(ps), math.sin(ps)
    R = np.array([[c3 * c2, c3 * s2 * s1 - s3 * c1, s3 * s1 + c3 * c1 * s2],
                  [s3 * c2, c1 * c3 + s1 * s2 * s3, c1 * s2 * s3 - c3 * s1],
                  [-s2, c2 * s1, c1 * c2]])
    cb = R.T @ np.asarray(current_ned, dtype=float)
    rel = np.r_[nu[:3] - cb, nu[3:]]
    acc = MINV @ forces(rel, phi, th, delta_s, delta_r, rpm)
    p, q, r = nu[3:]
    pos = R @ nu[:3]
    t2 = s2 / c2
    ang = [p + s1 * t2 * q + c1 * t2 * r, c1 * q - s1 * r, s1 / c2 * q + c1 / c2 * r]
    return np.r_[acc, pos, ang]


def step(x, delta_s, delta_r, rpm, h, current_ned=(0.0, 0.0, 0.0)) -> np.ndarray:
    """One fourth-order Runge-Kutta step: the truth shouldn't carry the fit's own Euler error."""
    f = lambda y: derivative(y, delta_s, delta_r, rpm, current_ned)
    k1 = f(x)
    k2 = f(x + .5 * h * k1)
    k3 = f(x + .5 * h * k2)
    k4 = f(x + h * k3)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
