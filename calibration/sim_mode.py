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


OPTICAL_FLAG_FIELDS = {"water": "--optical-water", "attenuation_per_m": "--optical-attenuation",
                       "terminals": "--optical-terminals", "beam_half_angle_deg": "--optical-beam",
                       "receiver_half_fov_deg": "--optical-fov", "rated_range_m": "--optical-rated-range",
                       "data_rate_bps": "--optical-data-rate", "beacon_interval_s": "--optical-interval"}


def link_from_flags(link: Optional[str] = None, hybrid_policy: Optional[str] = None, **optical):
    """--link and --optical-* flags -> comm_channel.LinkConfig (validated), or None for acoustic.
    optical: keyword arguments named as OpticalLinkConfig fields; None = default."""
    from comm_channel import LinkConfig, OpticalLinkConfig
    given = {k: v for k, v in optical.items() if v is not None}
    if link in (None, "", "acoustic"):
        if given or hybrid_policy:
            names = ", ".join(OPTICAL_FLAG_FIELDS.get(k, k) for k in given) or "--hybrid-policy"
            raise ValueError(f"{names} set but the link is acoustic; add --link optical or --link hybrid")
        return None
    if hybrid_policy and link != "hybrid":
        raise ValueError("--hybrid-policy only applies to --link hybrid")
    lc = LinkConfig(link, OpticalLinkConfig(**given), hybrid_policy or "optical_first")
    lc.validate()
    from fleet_env import FleetConfig
    dt = FleetConfig().dt
    k = lc.optical.beacon_interval_s / dt
    if abs(k - round(k)) > 1e-9 or round(k) < 1:
        raise ValueError(f"--optical-interval must be a positive multiple of the {dt:g} s tick, "
                         f"got {lc.optical.beacon_interval_s}")
    return lc


def add_link_args(ap, sweep: bool = False) -> None:
    """--link and --optical-* flags (shared with `strathe fleet` / `train-fleet`). sweep: lists allowed."""
    many = "+" if sweep else None
    ap.add_argument("--link", choices=["acoustic", "optical", "hybrid"], nargs=many, default=None,
                    help="which link carries the beacons: acoustic (default, today's model), optical, or hybrid "
                         "(both)" + ("; a list compares them on the same episodes" if sweep else "")
                         + ". The optical model is UNCALIBRATED (see LIMITS.md)")
    ap.add_argument("--optical-water", choices=["clear", "coastal", "harbor", "custom"], nargs=many, default=None,
                    help="water clarity for the optical link (default coastal; custom needs --optical-attenuation)")
    ap.add_argument("--optical-attenuation", type=float, default=None, metavar="C",
                    help="water attenuation coefficient per metre, for --optical-water custom (~0.15 clear ocean, "
                         "~0.4 coastal, 2+ harbour)")
    ap.add_argument("--optical-terminals", choices=["omni", "forward_cone", "fore_aft_cones", "down_cone"],
                    default=None, help="optical terminals on each vehicle (default omni: every direction)")
    ap.add_argument("--optical-beam", type=float, default=None, metavar="DEG",
                    help="transmit beam HALF-angle in degrees for cone terminals (default 30; 180 = every direction)")
    ap.add_argument("--optical-fov", type=float, default=None, metavar="DEG",
                    help="receiver field-of-view HALF-angle in degrees for cone terminals (default 30)")
    ap.add_argument("--optical-rated-range", type=float, default=None, metavar="M",
                    help="range at which a pointed link delivers half its messages in clear water (default 30 m)")
    ap.add_argument("--optical-data-rate", type=float, default=None, metavar="BPS", help="default 1e6 bit/s")
    ap.add_argument("--optical-interval", type=float, default=None, metavar="S",
                    help="seconds between optical beacons (default 0.1: every tick)")
    ap.add_argument("--link-calibration", default=None, metavar="YAML",
                    help="link_calibration.yaml from `strathe fit-link`: the acoustic losses and latency, and the "
                         "optical water, delivery curve and latency, measured from your message logs")
    ap.add_argument("--hybrid-policy", choices=["optical_first", "redundant"], default=None,
                    help="optical_first (default): skip the acoustic beacon while the optical link is up; "
                         "redundant: send every beacon on both links")


def optical_kwargs(args) -> dict:
    return {"attenuation_per_m": args.optical_attenuation, "terminals": args.optical_terminals,
            "beam_half_angle_deg": args.optical_beam, "receiver_half_fov_deg": args.optical_fov,
            "rated_range_m": args.optical_rated_range, "data_rate_bps": args.optical_data_rate,
            "beacon_interval_s": args.optical_interval}


def link_from_args(args):
    """The link from single-valued --link/--optical-* flags (None = acoustic), with --link-calibration's optical
    fit applied. Raises ValueError."""
    lc = link_from_flags(args.link, args.hybrid_policy, water=args.optical_water, **optical_kwargs(args))
    cal = getattr(args, "link_calibration", None)
    if lc is not None and cal:
        from link_fit import load_link_calibration, with_optical_calibration
        lc = with_optical_calibration(lc, load_link_calibration(cal))
    return lc
