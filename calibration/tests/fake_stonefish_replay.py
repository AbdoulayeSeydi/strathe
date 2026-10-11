"""A stand-in for calibration/stonefish/strathe_stonefish_replay, for tests (no Stonefish needed): the same command
line and files, and a simple vehicle whose motion follows the scenario's settings on the tuned part ('Hull'), so a fit
can recover them. Per axis (body frame, decoupled): (m + A) dv/dt = F - k_q*Cd*|v|v - k_f*Cf*v (+ net buoyancy on
heave, from the buoyancy part's size); rotation: I dw/dt = T - c*w; the attitude follows w."""
import json
import math
import re
import sys

import numpy as np
from scipy.spatial.transform import Rotation

K_Q, K_F, RHO, G = 20.0, 200.0, 1000.0, 9.81
DEFAULTS = {"quadratic_drag": [0.2, 1.0, 1.0], "viscous_drag": [0.02, 0.1, 0.1], "added_mass": [5.0, 40.0, 40.0],
            "added_inertia": [0.1, 2.0, 2.0]}


def arg(name, n=1):
    i = sys.argv.index(name)
    return sys.argv[i + 1:i + 1 + n] if n > 1 else sys.argv[i + 1]


def scenario_text(path):
    text = open(path).read()
    for inc in re.findall(r'<include\b[^>]*\bfile="([^"]*)"', text):
        text += scenario_text(inc)
    return text


def part(text, name):
    m = re.search(r'<(external_part|internal_part)\b[^>]*name="%s"[^>]*>(.*?)</\1>' % name, text, re.S)
    return m.group(2) if m else ""


def vec(body, attr, default):
    m = re.search(r'%s="([^"]*)"' % attr, body)
    return [float(x) for x in m.group(1).split()] if m else list(default)


def model(text):
    hull = part(text, "Hull")
    h = re.search(r"<hydrodynamics\b[^>]*/>", hull)
    hy = h.group(0) if h else ""
    mass = float(re.search(r'<mass value="([^"]*)"', hull).group(1))
    foam = part(text, "Foam")
    r = float(re.search(r'radius="([^"]*)"', foam).group(1)) if foam else 0.0
    hgt = float(re.search(r'height="([^"]*)"', foam).group(1)) if foam else 0.0
    return {"mass": mass, "Cd": vec(hy, "quadratic_drag", DEFAULTS["quadratic_drag"]),
            "Cf": vec(hy, "viscous_drag", DEFAULTS["viscous_drag"]), "A": vec(hy, "added_mass", DEFAULTS["added_mass"]),
            "AI": vec(hy, "added_inertia", DEFAULTS["added_inertia"]),
            "buoyancy": RHO * G * math.pi * r * r * hgt - mass * G}


def main():
    text = scenario_text(arg("--scenario"))
    m = model(text)
    if "--info" in sys.argv:
        print("[INFO] fake Stonefish")
        print(json.dumps({"patched": True, "base": {"name": f"{arg('--robot')}/Vehicle", "mass": m["mass"]},
                          "parts": [{"name": f"{arg('--robot')}/Vehicle/Hull", "mass": m["mass"],
                                     "quadratic_drag": m["Cd"], "viscous_drag": m["Cf"], "added_mass": m["A"],
                                     "added_inertia": m["AI"]}]}))
        return 0
    dt, hz = float(arg("--dt")), float(arg("--hz"))
    sub = max(1, int(round(dt * hz)))
    h = dt / sub
    rows = {}
    for line in open(arg("--windows")).read().splitlines()[1:]:
        v = [float(x) for x in line.split(",")]
        rows.setdefault(int(v[0]), []).append(v[2:])
    M = m["mass"] + np.array(m["A"])
    I = np.array([1.0, 3.0, 3.0]) + np.array(m["AI"])
    with open(arg("--out"), "w") as f:
        f.write("win,k,u,v,w,p,q,r,qx,qy,qz,qw\n")
        for win, rr in rows.items():
            v = np.array(rr[0][:6], dtype=float)
            q = Rotation.from_quat(rr[0][6:10])
            for k, row in enumerate(rr):
                f.write(f"{win},{k}," + ",".join(f"{x:.9g}" for x in list(v) + list(q.as_quat())) + "\n")
                if k + 1 == len(rr):
                    break
                F, T = np.array(row[10:13]), np.array(row[13:16])
                for _ in range(sub):
                    lin = v[:3]
                    acc = (F - K_Q * np.array(m["Cd"]) * np.abs(lin) * lin - K_F * np.array(m["Cf"]) * lin
                           + np.array([0.0, 0.0, -m["buoyancy"]])) / M
                    rot = (T - 2.0 * v[3:]) / I
                    v = v + h * np.concatenate([acc, rot])
                    q = q * Rotation.from_rotvec(v[3:] * h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
