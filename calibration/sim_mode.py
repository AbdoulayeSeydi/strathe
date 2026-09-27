"""
sim_mode.py -- clean vs noisy simulation as one switch, shared by every tool.

    clean   exact sensor readings, no roll/pitch disturbance (the default)
    noisy   sensor noise `typical` (sensor_noise.py) + roll/pitch disturbance
            `fitted` (disturbance.py), each at scale 1

Explicit settings win: `--mode noisy --noise tank` is noisy with the tank's
sensor noise; `--mode clean --noise typical` is clean except for the sensors.
Water current, thruster limits and buoyancy are properties of the site and the
vehicle, not of the mode: they are set on their own in either mode.
"""
from __future__ import annotations

from typing import Optional

MODES = ("clean", "noisy")
NOISY_DEFAULTS = {"noise": "typical", "disturbance": "fitted"}


def resolve(mode: str = "clean", noise: Optional[str] = None, disturbance: Optional[str] = None) -> tuple:
    """(noise preset or 'none', disturbance or 'none') for a mode, with explicit values taking precedence."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    noisy = mode == "noisy"
    return (noise if noise not in (None, "") else (NOISY_DEFAULTS["noise"] if noisy else "none"),
            disturbance if disturbance not in (None, "") else (NOISY_DEFAULTS["disturbance"] if noisy else "none"))


def buoyancy_from_flags(values, known: bool = False):
    """--buoyancy N or --buoyancy MIN MAX -> BuoyancyConfig (validated), or None."""
    if not values:
        return None
    from buoyancy import BuoyancyConfig
    if len(values) not in (1, 2):
        raise ValueError("--buoyancy takes one value (N) or MIN MAX")
    cfg = BuoyancyConfig(offset_n=float(values[0]) if len(values) == 1 else (float(values[0]), float(values[1])),
                         known_to_beliefs=known)
    cfg.validate()
    return cfg if cfg.active else None
