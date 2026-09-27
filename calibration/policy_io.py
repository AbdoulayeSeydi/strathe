"""
policy_io.py -- load a trained stable-baselines3 policy without having to say
which algorithm trained it.

A policy .zip records its own policy class; PPO, SAC and TD3 are told apart
from that, so `--policy my.zip` and `policies: [{model_path: my}]` work for
any of the three. Saying the algorithm explicitly still works, and a
mismatch is reported in plain words instead of a deserialisation error.
"""
from __future__ import annotations

import json
import os
import zipfile
from typing import Optional

ALGORITHMS = ("ppo", "sac", "td3")


class PolicyLoadError(ValueError):
    pass


def zip_path(path: str) -> str:
    return path if path.endswith(".zip") else f"{path}.zip"


def detect_algorithm(path: str) -> Optional[str]:
    """'ppo', 'sac' or 'td3' from the file's own metadata; None if it isn't one of those (or unreadable)."""
    try:
        with zipfile.ZipFile(zip_path(path)) as z:
            data = json.loads(z.read("data"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None
    module = str((data.get("policy_class") or {}).get("__module__", ""))
    if module.startswith("stable_baselines3.sac."):
        return "sac"
    if module.startswith("stable_baselines3.td3."):
        return "td3"
    if module.startswith("stable_baselines3.common.") and "clip_range" in data:
        return "ppo"
    return None


def describe_file(path: str) -> str:
    try:
        with zipfile.ZipFile(zip_path(path)) as z:
            data = json.loads(z.read("data"))
        return str((data.get("policy_class") or {}).get("__module__", "unknown policy class"))
    except Exception:
        return "not a stable-baselines3 policy file"


def resolve_algorithm(path: str, algorithm: Optional[str] = None) -> str:
    """The algorithm to load `path` with: the given one (checked against the file), or the detected one."""
    alg = None if algorithm in (None, "", "auto") else str(algorithm).lower()
    if alg is not None and alg not in ALGORITHMS:
        raise PolicyLoadError(f"algorithm must be one of {ALGORITHMS} (or omitted to detect it), got {algorithm!r}")
    found = detect_algorithm(path)
    if alg is None:
        if found is None:
            raise PolicyLoadError(
                f"{zip_path(path)!r} isn't a PPO, SAC or TD3 policy saved by stable-baselines3 "
                f"(it holds: {describe_file(path)}). Recurrent (sb3-contrib) policies and other frameworks "
                f"aren't supported; see LIMITS.md")
        return found
    if found is not None and found != alg:
        raise PolicyLoadError(f"{zip_path(path)!r} was trained with {found.upper()}, not {alg.upper()}; "
                              f"use algorithm: {found} (or leave it out and it is detected)")
    return alg


def load_policy(path: str, algorithm: Optional[str] = None):
    """(model, algorithm). Raises PolicyLoadError with a plain-language message."""
    zp = zip_path(path)
    if not os.path.exists(zp):
        raise PolicyLoadError(f"policy file not found: {zp!r}")
    alg = resolve_algorithm(zp, algorithm)
    from stable_baselines3 import PPO, SAC, TD3
    cls = {"ppo": PPO, "sac": SAC, "td3": TD3}[alg]
    try:
        return cls.load(zp, device="cpu"), alg
    except Exception as e:  # SB3 raises a variety of errors for wrong-class or corrupt files
        raise PolicyLoadError(f"could not load {zp!r} as a {alg.upper()} policy: {type(e).__name__}: {e}") from e
