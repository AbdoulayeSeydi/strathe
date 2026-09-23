"""
C0.5 addition (see conversation) -- small, shared utilities for wiring
log_fleet_run() into real scripts. Kept separate from mlflow_logging.py
itself (which stays a pure MLflow adapter, no git/subprocess concerns)
and separate from evaluate_comm_degradation_curve.py (a per-episode
reward decomposition is a real computation, not MLflow plumbing).

*** get_git_commit() is a standalone, best-effort implementation --
this session never saw wire_mlflow_run_calibration.py, so if that file
already has an equivalent helper with different behavior (e.g. a short
hash vs. full hash), swap this for that one; nothing else here depends
on which implementation is used, only that git_commit() returns a
string.
"""
from __future__ import annotations

import subprocess
from typing import Optional

import numpy as np


def get_git_commit() -> str:
    """Best-effort full git commit hash for provenance. Never raises --
    returns 'unknown' if git isn't available or this isn't a git repo at
    all, matching mlflow_logging.py's own _safe_pkg_version never-raise
    discipline (a missing commit hash should never block a real run)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def compute_per_agent_rewards(combined_reward: float, action_generic: np.ndarray,
                                action_calibrated: np.ndarray, reward_ctrl_weight: float) -> tuple:
    """Derives a genuine, principled per-agent reward split from the
    FormationHoldTask's own single combined reward (R_a,t + R_b,t) --
    see conversation for why this is necessary (no per-agent reward
    exists anywhere in the current pipeline) and why this specific split
    is defensible rather than arbitrary: every reward_components term
    EXCEPT control cost is shared/symmetric (separation, proximity,
    collision, success_bonus all depend on d_t, not per-agent data) --
    control cost is the one term the reward equation itself defines
    per-agent (||u_i,t||^2 each), just summed into one combined value
    before being returned. This does NOT change training or the actual
    reward FormationHoldTask computes anywhere -- purely a derived
    metric for logging/observability, computed after the fact from
    already-public quantities (the combined reward, each agent's own
    action, and reward_ctrl_weight read LIVE from the real task config
    in use, not a hardcoded guess that could silently go stale if a
    caller ever customizes FormationHoldConfig).

    combined_reward = shared_components - ctrl_generic - ctrl_calibrated
    => shared_components = combined_reward + ctrl_generic + ctrl_calibrated
    => reward_generic     = shared_components - ctrl_generic
       reward_calibrated  = shared_components - ctrl_calibrated
    """
    ctrl_generic = reward_ctrl_weight * float(np.sum(np.square(action_generic)))
    ctrl_calibrated = reward_ctrl_weight * float(np.sum(np.square(action_calibrated)))
    shared_components = combined_reward + ctrl_generic + ctrl_calibrated
    reward_generic = shared_components - ctrl_generic
    reward_calibrated = shared_components - ctrl_calibrated
    return reward_generic, reward_calibrated

def get_git_dirty() -> Optional[bool]:
    """True if the working tree has uncommitted changes to TRACKED files
    (so git_commit alone does not identify the code that ran), False if
    clean, None if git is unavailable. Never raises."""
    try:
        result = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=5, check=True)
        return bool(result.stdout.strip())
    except Exception:
        return None
