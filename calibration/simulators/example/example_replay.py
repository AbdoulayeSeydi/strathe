"""An example simulator speaking the replay contract (calibration/simulators/SIMULATOR_PROTOCOL.md): start from it to
connect your own simulator to `strathe fit-sim --sim external` (PROJECT_STATUS.md #81). The tests run it too. Its model
is a JSON file (vehicle.json next to this file):
  {"mass": 30, "added_mass": [x, y, z], "inertia": [roll, pitch, yaw],
   "linear_damping": [Xu, Yv, Zw, Kp, Mq, Nr], "quadratic_damping": [Xuu, ...], "water": {"density": 1000}}
and each axis moves on its own: (m + A) dv/dt = tau + (linear + quadratic |v|) v (coefficients negative), which is
Strathe's damping law, so an exact export reproduces Strathe's own model on decoupled motion."""
import json
import math
import sys

import numpy as np


def arg(name, n=1, default=None):
    if name not in sys.argv:
        return default
    i = sys.argv.index(name)
    return sys.argv[i + 1:i + 1 + n] if n > 1 else sys.argv[i + 1]


def main():
    if "--features" in sys.argv:
        print(json.dumps({"simulator": "fake", "version": "1.0"}))
        return 0
    model = json.load(open(arg("--model")))
    m = float(model["mass"])
    M = np.array([m + a for a in model["added_mass"]] + list(model["inertia"]), dtype=float)
    lin = np.array(model["linear_damping"], dtype=float)
    quad = np.array(model["quadratic_damping"], dtype=float)
    if "--info" in sys.argv:
        print("[fake] loaded")
        print(json.dumps({"mass": m, "water_density": model.get("water", {}).get("density", 1000.0),
                          "model_dir": arg("--model-dir")}))
        return 0
    dt = float(arg("--dt"))
    sub = 10
    h = dt / sub
    rows = {}
    with open(arg("--windows")) as f:
        f.readline()
        for line in f:
            x = [float(v) for v in line.split(",")]
            rows.setdefault(int(x[0]), []).append(x[2:])
    with open(arg("--out"), "w") as f:
        f.write("win,k,u,v,w,p,q,r,qx,qy,qz,qw\n")
        for w, rs in rows.items():
            v = np.array(rs[0][:6])
            q = np.array(rs[0][6:10])
            for k, row in enumerate(rs):
                f.write(f"{w},{k}," + ",".join(f"{x:.9g}" for x in [*v, *q]) + "\n")
                if k + 1 == len(rs):
                    break
                tau = np.array(row[10:16])
                for _ in range(sub):
                    v = v + h * (tau + (lin + quad * np.abs(v)) * v) / M
                    wx, wy, wz = v[3:] * h
                    th = math.sqrt(wx * wx + wy * wy + wz * wz)
                    if th > 1e-12:
                        c, s = math.cos(th / 2), math.sin(th / 2) / th
                        dx, dy, dz = wx * s, wy * s, wz * s
                        x_, y_, z_, w_ = q
                        q = np.array([w_ * dx + x_ * c + y_ * dz - z_ * dy, w_ * dy - x_ * dz + y_ * c + z_ * dx,
                                      w_ * dz + x_ * dy - y_ * dx + z_ * c, w_ * c - x_ * dx - y_ * dy - z_ * dz])
                        q /= np.linalg.norm(q)
                if not np.all(np.isfinite(v)) or np.abs(v).max() > 1e3:
                    break
    return 0


if __name__ == "__main__":
    sys.exit(main())
