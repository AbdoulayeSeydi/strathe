"""A stand-in for holoocean.fossen_dynamics.torpedo with the same interface (Torpedo.Minv, Torpedo.dynamics(eta, nu,
u_actual, u_control, sampleTime) -> (nu_dot, u_actual), configure_torpedo_from_scenario) and the documented dynamics
keys, for the tests of `strathe fit-sim --sim holoocean` on machines without HoloOcean. Its physics is a simple
time-constant model, not HoloOcean's."""
import math

import numpy as np


class Torpedo:
    dimU = 3

    def __init__(self, controlSystem="stepInput", *args):
        self.controlSystem = controlSystem
        self.rho, self.m, self.length, self.diam = 1026.0, 30.0, 1.6, 0.19
        self.T_surge, self.T_sway, self.T_yaw, self.K_nomoto = 20.0, 20.0, 1.0, 0.25
        self.zeta_roll, self.zeta_pitch, self.r44, self.Cd, self.e = 0.3, 0.8, 0.3, 0.42, 0.7
        self.r_bg, self.r_bb = [0.0, 0.0, 0.02], [0.0, 0.0, 0.0]

    def _M(self):
        m = self.m
        Ix = 0.5 * m * (self.diam / 2) ** 2
        Iy = Iz = m * (3 * (self.diam / 2) ** 2 + self.length ** 2) / 12
        return np.diag([m * 1.1, m * 1.9, m * 1.9, Ix * (1 + self.r44), Iy * 1.5, Iz * 1.5])

    @property
    def Minv(self):
        return np.linalg.inv(self._M())

    def dynamics(self, eta, nu, u_actual, u_control, sampleTime):
        u, v, w, p, q, r = nu
        phi, theta = eta[3], eta[4]
        M = self._M()
        W = self.m * 9.81
        zg = self.r_bg[2] - self.r_bb[2]
        A = math.pi * (self.diam / 2) ** 2
        tau = np.array([
            -M[0, 0] / self.T_surge * u - 0.5 * self.rho * self.Cd * A * abs(u) * u,
            -M[1, 1] / self.T_sway * v,
            -M[2, 2] / self.T_sway * w,
            -2 * self.zeta_roll * math.sqrt(W * max(zg, 1e-3) * M[3, 3]) * p - W * zg * math.cos(theta) * math.sin(phi),
            -2 * self.zeta_pitch * math.sqrt(W * max(zg, 1e-3) * M[4, 4]) * q - W * zg * math.sin(theta),
            -M[5, 5] / self.T_yaw * r])
        return np.linalg.solve(M, tau), u_actual


def configure_torpedo_from_scenario(scenario, vehicle_name, vehicle):
    agent = next(a for a in scenario["agents"] if a["agent_name"] == vehicle_name)
    for k, v in agent.get("dynamics", {}).items():
        setattr(vehicle, {"mass": "m"}.get(k, k), v)
