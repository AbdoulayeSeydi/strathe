"""
fleet_eval.py -- run fleet v2 policies over a grid of episodes.

One "cell" = (policy, comm condition, belief condition, truth backend).
Every cell runs the SAME episode grid -- initial separations x closing
speeds x comm seeds -- and each episode's comm seed is derived only from
(comm seed, grid point), never from the belief condition or policy. With a
named severity (equal near/far drop probability) that makes the packet-drop
pattern identical across belief conditions, so conditions are compared
episode-by-episode (tested in tests/test_fleet_v2.py).

Cells are independent, so run_cells() spreads them across processes.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from comm_channel import BandwidthConfig, LinkConfig, PacketLossConfig
from dynamics_model import CalibrationParameters, VehicleConstants
from fleet_belief import BeliefSpec
from fleet_env import FleetConfig, FleetSim, N_AGENTS


@dataclass(frozen=True)
class CommSpec:
    label: str
    packet_loss_config: PacketLossConfig
    bandwidth_config: BandwidthConfig
    speed_of_sound_mps: float = 1500.0
    beacon_interval_s: float = 1.0
    link: Optional[LinkConfig] = None   # None = acoustic only (the original behaviour)
    acoustic_fixed_latency_s: float = 0.0   # the acoustic modem's own delay (a link calibration, #68)


@dataclass
class CellSpec:
    policy_path: str
    algorithm: str
    comm: CommSpec
    belief: BeliefSpec
    truth_params: CalibrationParameters
    vehicle: VehicleConstants
    fleet_config: FleetConfig
    separations: Sequence[float]
    closing_speeds: Sequence[float]
    comm_seeds: Sequence[int]
    tags: dict = field(default_factory=dict)


def episode_comm_seed(comm_seed: int, i_sep: int, i_speed: int) -> int:
    return int(np.random.SeedSequence([comm_seed, i_sep, i_speed]).generate_state(1)[0])


def run_episode(sim: FleetSim, policy, comm_seed: int, d0: float, v0: float) -> dict:
    obs = sim.reset(comm_seed=comm_seed, initial_separation=d0, closing_speed=v0)
    belief_errs, delivered, sep_err, nav_err, ages = [], [], [], [], []
    sent = {"acoustic": 0, "optical": 0}
    got = {"acoustic": 0, "optical": 0}
    updates = {"acoustic": 0, "optical": 0}
    skipped = uncovered = 0
    target = sim.cfg.task.target_separation
    info = {"terminal_reason": None}
    steps = 0
    while True:
        actions, _ = policy.predict(np.stack(obs), deterministic=True)
        obs, _, term, trunc, info = sim.step([actions[i] for i in range(N_AGENTS)])
        steps += 1
        belief_errs += [e for e in info["belief_errors"] if e is not None]
        delivered += info["delivered"]
        sep_err.append(abs(info["true_separation"] - target))
        nav_err += info.get("nav_errors") or []
        for k in sent:
            sent[k] += info["sent"][k]
            updates[k] += info["belief_updates"][k]
        for links in info["delivered_by"]:
            for k in links:
                got[k] += 1
        skipped += info["acoustic_skipped"]
        uncovered += info["acoustic_skipped_uncovered"]
        ages += [a for a in info["belief_ages"] if np.isfinite(a)]
        if term or trunc:
            break
    reason = info["terminal_reason"]
    return {
        "steps": steps,
        "terminal_reason": reason or "timeout",
        "success": reason == "formation_achieved",
        "collision": reason == "collision",
        "mean_belief_error_m": float(np.mean(belief_errs)) if belief_errs else float("nan"),
        "p90_belief_error_m": float(np.percentile(belief_errs, 90)) if belief_errs else float("nan"),
        "delivery_rate": float(np.mean(delivered)),
        "mean_abs_separation_error_m": float(np.mean(sep_err)),
        "mean_nav_error_m": float(np.mean(nav_err)) if nav_err else 0.0,
        "buoyancy_offset_n": float(info.get("buoyancy_offset_n", 0.0)),
        # per link: beacons sent / delivered (a delivered copy may still be older than what the belief holds)
        "acoustic_sent": sent["acoustic"], "acoustic_delivered": got["acoustic"], "acoustic_skipped": skipped,
        "acoustic_skipped_uncovered": uncovered,
        "optical_sent": sent["optical"], "optical_delivered": got["optical"],
        "acoustic_delivery_share": got["acoustic"] / sent["acoustic"] if sent["acoustic"] else float("nan"),
        # share of optical sends that got through: the fraction of time the optical link was up
        "optical_up_fraction": got["optical"] / sent["optical"] if sent["optical"] else float("nan"),
        # which link carried the beacons that actually refreshed a belief
        "belief_updates_optical_share": (updates["optical"] / (updates["optical"] + updates["acoustic"])
                                         if updates["optical"] + updates["acoustic"] else float("nan")),
        "median_belief_age_s": float(np.median(ages)) if ages else float("nan"),
    }


RANDOM_POLICY = "random"


class RandomPolicy:
    """Uniformly random commands in [-1, 1]: the floor any trained fleet policy should beat. Seeded, so a run
    is reproducible; use policy_path RANDOM_POLICY in a CellSpec."""

    def __init__(self, seed: int = 0):
        self._rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic: bool = True):
        return self._rng.uniform(-1.0, 1.0, (len(obs), 6)).astype(np.float32), None


ZERO_POLICY = "zero"


class ZeroPolicy:
    """No commands at all: the vehicles only drift. A second reference for how hard a grid is."""

    def predict(self, obs, deterministic: bool = True):
        return np.zeros((len(obs), 6), dtype=np.float32), None


def _load_policy(path: str, algorithm: str):
    if path == RANDOM_POLICY:
        return RandomPolicy(0)
    if path == ZERO_POLICY:
        return ZeroPolicy()
    from validate_fleet_policy import load_fleet_policy
    return load_fleet_policy(path, algorithm)


def run_cell(cell: CellSpec) -> list:
    import torch
    torch.set_num_threads(1)
    policy = _load_policy(cell.policy_path, cell.algorithm)
    sim = FleetSim(cell.truth_params, cell.vehicle, cell.fleet_config,
                   cell.comm.packet_loss_config, cell.comm.bandwidth_config,
                   cell.comm.speed_of_sound_mps, belief=cell.belief,
                   beacon_interval_s=cell.comm.beacon_interval_s, link=cell.comm.link,
                   acoustic_fixed_latency_s=cell.comm.acoustic_fixed_latency_s)
    rows = []
    for comm_seed in cell.comm_seeds:
        for i_d, d0 in enumerate(cell.separations):
            for i_v, v0 in enumerate(cell.closing_speeds):
                r = run_episode(sim, policy, episode_comm_seed(comm_seed, i_d, i_v), d0, v0)
                r.update({"policy": os.path.basename(cell.policy_path), "comm": cell.comm.label,
                          "belief": cell.belief.label, "belief_kind": cell.belief.kind,
                          "beacon_interval_s": cell.comm.beacon_interval_s,
                          "link": sim.link.label,
                          "backend": cell.fleet_config.dynamics_backend, "comm_seed": comm_seed,
                          "initial_separation_m": d0, "closing_speed_mps": v0,
                          "episode_key": f"{comm_seed}|{i_d}|{i_v}"})
                r.update(cell.tags)
                rows.append(r)
    return rows


def run_cells(cells: Sequence[CellSpec], workers: Optional[int] = None, progress=None) -> list:
    """Runs every cell; returns all episode rows. workers=1 runs in-process."""
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    rows = []
    if workers == 1:
        for k, cell in enumerate(cells):
            rows += run_cell(cell)
            if progress:
                progress(k + 1, len(cells))
        return rows
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for k, cell_rows in enumerate(pool.map(run_cell, cells)):
            rows += cell_rows
            if progress:
                progress(k + 1, len(cells))
    return rows
