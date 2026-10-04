"""
drag_reference.py -- published drag values, and two opt-in ways to use them in a calibration
(PROJECT_STATUS.md #63).

The problem it addresses: logged motion can rarely separate linear from quadratic drag (on DFKI's
BlueROV2, Xu/Xuu correlate at -0.99, Kp/Kpp at -0.94), so the fit slides along that trade-off until one
term hits its bound, and tether pull or other unmodelled forces end up in whichever term is free.

    --drag free     (default) every drag term is fitted, as before
    --drag prior    all twelve are fitted, with a weak penalty on their log-distance from the published values,
                    so a term the data can't pin settles nearer the published value instead of at a bound

Tested on DFKI (#63): the prior gave a little better surge and moved two quadratic terms off their bounds.
Holding the linear terms at the published values instead (all six, or only Xu/Yv/Zw) made the open-loop
replay unstable (sway skill -2.42 and -1.20, roll -11.3 and -4.8): the published linear drag is too small
for real logged motion. That option was removed.

The published values: Wu (2018), "6-DoF Modelling and Control of a Remotely Operated Vehicle",
Flinders University, Table 5.3, a standard 6-thruster BlueROV2. Wu derived them from published
literature (Sandoy 2016), not from his own measurements on the vehicle, and took roll and pitch equal to
yaw. Treat them as a starting point, not as the vehicle's measured drag.
"""
from __future__ import annotations

import numpy as np

REFERENCES = {
    "bluerov2_wu2018": {
        "source": "Wu (2018), Flinders University thesis, Table 5.3 (standard BlueROV2; from Sandoy 2016)",
        "linear": {"Xu": -4.03, "Yv": -6.22, "Zw": -5.18, "Kp": -0.07, "Mq": -0.07, "Nr": -0.07},
        "quadratic": {"Xuu": -18.18, "Yvv": -21.66, "Zww": -36.99, "Kpp": -1.55, "Mqq": -1.55, "Nrr": -1.55},
    },
}
DRAG_MODES = ("free", "prior")
DEFAULT_PRIOR_WEIGHT = 0.1


def reference_values(name: str) -> dict:
    if name not in REFERENCES:
        raise ValueError(f"unknown drag reference {name!r}; available: {sorted(REFERENCES)}")
    r = REFERENCES[name]
    return {**r["linear"], **r["quadratic"]}


def within_bounds(values: dict, names: list, lb, ub) -> tuple:
    """(values clamped to the spec's bounds, [notes for the ones that moved])."""
    out, notes = {}, []
    for k, v in values.items():
        i = names.index(k)
        c = float(np.clip(v, lb[i], ub[i]))
        if c != v:
            notes.append(f"{k}: published {v:g} is outside vehicle_spec.yaml's bounds [{lb[i]:g}, {ub[i]:g}]; "
                         f"using {c:g}")
        out[k] = c
    return out, notes


def prior_residual_wrapper(base_fn, names: list, targets: dict, weight: float):
    """Append sqrt(weight) * ln(theta / target) for each targeted parameter to the residual vector.
    Log-distance: twice and half the published value cost the same. Drag terms keep their sign within
    their bounds, so the ratio is positive."""
    idx = np.array([names.index(k) for k in targets])
    tgt = np.array([targets[k] for k in targets], dtype=float)
    w = float(np.sqrt(weight))

    def wrapped(x, *args, **kw):
        r = base_fn(x, *args, **kw)
        return np.concatenate([np.asarray(r, dtype=float).ravel(), w * np.log(np.asarray(x)[idx] / tgt)])
    return wrapped
