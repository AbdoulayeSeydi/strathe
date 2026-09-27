#!/usr/bin/env python3
"""
stress_test_noise_algorithms.py -- put clean/noisy simulation, buoyancy
changes and SAC/TD3 policies through difficult conditions, for one vehicle
and for two. PROJECT_STATUS.md #58.

    python stress_test_noise_algorithms.py [--combos 400] [--workers 6]

A. Two vehicles, randomized combinations of every setting: comm L0-L3b,
   beacons 0.5-3 s, buoyancy -25..+25 N (known or not), sensor noise off /
   tank / typical at x1-x10, roll/pitch disturbance up to x3, water current
   up to 0.5 m/s with fluctuation, thruster limits, stand-in or MuJoCo, PPO /
   SAC / TD3 / random policies, every belief model. Plus the worst corners.
   Every episode must run to the end with finite observations, finite errors
   and consistent bookkeeping.
B. One vehicle, the same idea over all four tasks and the three algorithms.
C. Measured against what was configured: the sensor errors' size, bias,
   dropout rate and drift; the buoyancy terminal velocity against the
   calibrated damping; clean runs unchanged against the shipped fleet
   results; determinism; pairing (noise never changes start states,
   targets, packet losses or disturbances).

Writes results/stress_noise_algorithms/report.md and results.json. Exits 1
if any check fails.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

import numpy as np

OUT = "results/stress_noise_algorithms"
TRUTH = "calibrated_parameters_fleet_truth.yaml"
FLEET_POLICIES = {"ppo": "fleet_policy_seed1", "ppo_noisy": "noisy_policies/fleet_policy_noisy_seed1",
                  "sac": "algorithm_policies/fleet_sac_seed1", "td3": "algorithm_policies/fleet_td3_seed1",
                  "random": "random"}
SEVERITIES = ("L0", "L1", "L2", "L3a", "L3b")
BELIEFS = ("calibrated", "generic", "last_beacon", "constant_velocity", "oracle")
NOISE = (None, ("tank", 1.0), ("typical", 1.0), ("typical", 3.0), ("typical", 10.0))
TASKS = ("waypoint", "station_keeping", "docking", "obstacle_avoidance")


# ----------------------------------------------------------------- A. two vehicles

def sample_fleet_combo(rng: np.random.Generator, k: int) -> dict:
    noise = NOISE[rng.integers(len(NOISE))]
    buoy = float(rng.choice([-25, -10, -3, 0, 3, 10, 25]))
    return {"id": k, "severity": str(rng.choice(SEVERITIES)), "beacon_s": float(rng.choice([0.5, 1.0, 2.0, 3.0])),
            "buoyancy": ([buoy - 5, buoy + 5] if rng.random() < 0.2 else buoy), "known": bool(rng.random() < 0.5),
            "noise": noise, "disturbance": float(rng.choice([0.0, 1.0, 3.0])),
            "current": float(rng.choice([0.0, 0.2, 0.5])), "thrusters": bool(rng.random() < 0.4),
            "backend": "mujoco" if rng.random() < 0.2 else "stand_in",
            "policy": str(rng.choice(list(FLEET_POLICIES))), "belief": str(rng.choice(BELIEFS)),
            "d0": float(rng.uniform(1.0, 8.0)), "v0": float(rng.uniform(-1.0, 1.0)), "seed": int(rng.integers(1e6))}


def fleet_corners() -> list:
    base = dict(beacon_s=3.0, known=False, current=0.5, thrusters=True, d0=1.0, v0=-1.0, seed=7)
    out = []
    for i, (pol, backend) in enumerate([(p, b) for p in FLEET_POLICIES for b in ("stand_in", "mujoco")]):
        out.append({**base, "id": 10000 + i, "severity": "L3b", "buoyancy": 25.0 if i % 2 else -25.0,
                    "noise": ("typical", 10.0), "disturbance": 3.0, "backend": backend, "policy": pol,
                    "belief": BELIEFS[i % len(BELIEFS)]})
    return out


def run_fleet_combo(c: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    from build_two_agent_cores import load_calibrated_parameters
    from buoyancy import BuoyancyConfig
    from current import CurrentConfig
    from disturbance import DEFAULT_FIT_PATH, load_disturbance
    from evaluate_comm_degradation_curve import COMM_CONFIGS
    from fleet_belief import BeliefSpec
    from fleet_env import FleetConfig, FleetSim
    from fleet_eval import _load_policy
    from fleet_experiment_schema import resolve_thrusters
    from formation_hold_task import FormationHoldConfig
    from optimizer import initial_parameters, load_vehicle_spec
    from sensor_noise import load_sensor_noise
    t0 = time.time()
    try:
        spec = load_vehicle_spec("vehicle_spec.yaml")
        truth = load_calibrated_parameters(TRUTH, quiet=True)
        if c["belief"] == "calibrated":
            belief = BeliefSpec("physics", "calibrated", truth)
        elif c["belief"] == "generic":
            belief = BeliefSpec("physics", "generic", initial_parameters(spec))
        else:
            belief = BeliefSpec(c["belief"], c["belief"])
        b = c["buoyancy"]
        cfg = FleetConfig(dynamics_backend=c["backend"], beacon_interval_s=c["beacon_s"],
                          initial_separation_range=(0.6, 8.0), task=FormationHoldConfig(),
                          sensor_noise=load_sensor_noise(*c["noise"]) if c["noise"] else None,
                          buoyancy=(BuoyancyConfig(tuple(b) if isinstance(b, list) else b, c["known"])
                                    if b else None),
                          disturbance=load_disturbance(DEFAULT_FIT_PATH, c["disturbance"]) if c["disturbance"] else None,
                          current=(CurrentConfig(speed_mps=c["current"], direction_deg=(0.0, 360.0), speed_std_mps=0.05,
                                                 direction_std_deg=15.0) if c["current"] else None),
                          thrusters=resolve_thrusters("bluerov2_heavy") if c["thrusters"] else None)
        sev = COMM_CONFIGS[c["severity"]]
        sim = FleetSim(truth, spec.vehicle, cfg, sev["packet_loss_config"], sev["bandwidth_config"], belief=belief)
        policy = _load_policy(FLEET_POLICIES[c["policy"]], None)
        obs = sim.reset(comm_seed=c["seed"], initial_separation=max(0.6, c["d0"]), closing_speed=c["v0"])
        problems, steps, info = [], 0, {}
        nav, berr = [], []
        while True:
            X = np.stack(obs)
            if not np.all(np.isfinite(X)):
                problems.append(f"non-finite observation at step {steps}")
                break
            act, _ = policy.predict(X, deterministic=True)
            obs, r, term, trunc, info = sim.step([act[0], act[1]])
            steps += 1
            if not math.isfinite(r):
                problems.append(f"non-finite reward at step {steps}")
            nav += info["nav_errors"]
            berr += [e for e in info["belief_errors"] if e is not None]
            if term or trunc:
                break
        if not all(math.isfinite(x) and x >= 0 for x in nav + berr):
            problems.append("non-finite or negative navigation/belief error")
        if (c["noise"] is None) != all(x == 0 for x in nav):
            problems.append("navigation error present without noise, or absent with it")
        if info.get("terminal_reason") not in (None, "formation_achieved", "collision"):
            problems.append(f"unexpected terminal reason {info.get('terminal_reason')}")
        if not (term or trunc) or steps > cfg.max_episode_steps:
            problems.append("episode did not end properly")
        off = info.get("buoyancy_offset_n", 0.0)
        lo, hi = (b if isinstance(b, list) else (b, b)) if b else (0.0, 0.0)
        if not (lo - 1e-9 <= off <= hi + 1e-9):
            problems.append(f"buoyancy offset {off} outside {lo}..{hi}")
        return {**c, "ok": not problems, "problems": problems, "steps": steps,
                "outcome": info.get("terminal_reason") or "timeout", "max_nav_m": max(nav) if nav else 0.0,
                "mean_belief_m": float(np.mean(berr)) if berr else None, "s": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001 - every failure is a result
        return {**c, "ok": False, "problems": [f"{type(e).__name__}: {e}", traceback.format_exc(limit=3)],
                "s": round(time.time() - t0, 2)}


# ----------------------------------------------------------------- B. one vehicle

def train_tiny_arena_policies(out_dir: str) -> dict:
    """One short-trained policy per algorithm for 11-dim tasks and for obstacle avoidance (15-dim)."""
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO, SAC, TD3
    from arena import Arena
    from build_two_agent_cores import load_calibrated_parameters
    from optimizer import load_vehicle_spec
    truth = load_calibrated_parameters(TRUTH, quiet=True)
    vehicle = load_vehicle_spec("vehicle_spec.yaml").vehicle
    out = {}
    for task in ("waypoint", "obstacle_avoidance"):
        for name, cls in (("ppo", PPO), ("sac", SAC), ("td3", TD3)):
            env = Arena(truth, vehicle, task=task, max_episode_steps=100)
            kw = {"n_steps": 256} if name == "ppo" else {"learning_starts": 200}
            m = cls("MlpPolicy", env, seed=0, verbose=0, device="cpu", **kw)
            m.learn(1500)
            p = os.path.join(out_dir, f"arena_{name}_{task}")
            m.save(p)
            out[(name, "obstacle" if task == "obstacle_avoidance" else "base")] = p
    return out


def sample_arena_combo(rng, k, policies) -> dict:
    task = str(rng.choice(TASKS))
    alg = str(rng.choice(["ppo", "sac", "td3", "hand", "random"]))
    kind = "obstacle" if task == "obstacle_avoidance" else "base"
    return {"id": k, "task": task, "controller": alg,
            "policy": policies.get((alg, kind)) if alg in ("ppo", "sac", "td3") else None,
            "noise": NOISE[rng.integers(len(NOISE))], "buoyancy": float(rng.choice([-25, -10, -3, 0, 3, 10, 25])),
            "disturbance": float(rng.choice([0.0, 1.0, 3.0])), "current": float(rng.choice([0.0, 0.2, 0.5])),
            "thrusters": bool(rng.random() < 0.4), "backend": "mujoco" if rng.random() < 0.25 else "stand_in",
            "seed": int(rng.integers(1e6))}


def run_arena_combo(c: dict) -> dict:
    import torch
    torch.set_num_threads(1)
    from arena import Arena, TERMINAL_REASON_GOAL
    from build_two_agent_cores import load_calibrated_parameters
    from buoyancy import BuoyancyConfig
    from current import CurrentConfig
    from disturbance import DEFAULT_FIT_PATH, load_disturbance
    from fleet_experiment_schema import resolve_thrusters
    from optimizer import load_vehicle_spec
    from sensor_noise import load_sensor_noise
    t0 = time.time()
    try:
        truth = load_calibrated_parameters(TRUTH, quiet=True)
        env = Arena(truth, load_vehicle_spec("vehicle_spec.yaml").vehicle, task=c["task"], backend=c["backend"],
                    sensor_noise=load_sensor_noise(*c["noise"]) if c["noise"] else None,
                    buoyancy=BuoyancyConfig(c["buoyancy"]) if c["buoyancy"] else None,
                    disturbance=load_disturbance(DEFAULT_FIT_PATH, c["disturbance"]) if c["disturbance"] else None,
                    current=CurrentConfig(speed_mps=c["current"], direction_deg=(0.0, 360.0)) if c["current"] else None,
                    thrusters=resolve_thrusters("bluerov2_heavy") if c["thrusters"] else None)
        if c["controller"] in ("ppo", "sac", "td3"):
            from policy_io import load_policy
            model, alg = load_policy(c["policy"])
            assert alg == c["controller"], (alg, c["controller"])
            act = lambda o, rng: model.predict(o, deterministic=True)[0]
        elif c["controller"] == "hand":
            from evaluate_policy_arena import proportional_controller
            act = lambda o, rng: proportional_controller(o)
        else:
            act = lambda o, rng: rng.uniform(-1, 1, 6)
        rng = np.random.default_rng(c["seed"])
        obs, _ = env.reset(seed=c["seed"])
        problems, steps, info, done = [], 0, {}, False
        while not done:
            if not np.all(np.isfinite(obs)):
                problems.append(f"non-finite observation at step {steps}")
                break
            obs, r, term, trunc, info = env.step(act(obs, rng))
            steps += 1
            done = term or trunc
            if not math.isfinite(r):
                problems.append("non-finite reward")
            if not (math.isfinite(info["nav_error_m"]) and info["nav_error_m"] >= 0):
                problems.append("bad navigation error")
            true_d = float(np.linalg.norm(env._waypoint - info["position"]))
            if abs(true_d - info["distance"]) > 1e-9:
                problems.append("distance isn't the true distance")
        if (c["noise"] is None) != (info.get("nav_error_m", 0.0) == 0.0):
            problems.append("navigation error present without noise, or absent with it")
        if info.get("buoyancy_offset_n") != c["buoyancy"]:
            problems.append("buoyancy offset not applied")
        if info.get("terminal_reason") == TERMINAL_REASON_GOAL and info["distance"] >= env.waypoint_radius:
            problems.append("success declared from the believed position, not the true one")
        return {**c, "ok": not problems, "problems": problems, "steps": steps,
                "outcome": info.get("terminal_reason") or "timeout", "nav_m": info.get("nav_error_m"),
                "s": round(time.time() - t0, 2)}
    except Exception as e:  # noqa: BLE001
        return {**c, "ok": False, "problems": [f"{type(e).__name__}: {e}", traceback.format_exc(limit=3)],
                "s": round(time.time() - t0, 2)}


# ----------------------------------------------------------------- C. measured against what was configured

def check_sensor_statistics() -> list:
    from sensor_noise import SensorSuite, load_sensor_noise, matrix_from_euler
    cfg = load_sensor_noise("typical", 1.0)
    out = []
    v_err, g_err, h_err, bias_v, drops = [], [], [], [], []
    for seed in range(300):
        s = SensorSuite(cfg, 0.1, seed)
        R = matrix_from_euler(0.1, -0.05, 0.7)
        nu = np.array([0.3, -0.1, 0.05, 0.02, -0.01, 0.1])
        s.start(np.zeros(3), R, nu)
        prev = None
        for k in range(100):
            m = s.update(np.zeros(3), R, nu)
            e = m.nu - nu
            if prev is not None:
                drops.append(bool(np.all(m.nu[:3] == prev)))
                if not drops[-1]:
                    v_err.append(e[:3] - s.bias_v)
            prev = m.nu[:3].copy()
            g_err.append(e[3:] - s.bias_g)
            from sensor_noise import euler_from_matrix
            h_err.append(((euler_from_matrix(m.R)[2] - 0.7 - s.bias_h + math.pi) % (2 * math.pi)) - math.pi)
        bias_v.append(s.bias_v)

    def rel(measured, want):
        return abs(measured / want - 1)
    checks = [("DVL white noise std", float(np.std(np.array(v_err))), cfg.velocity_std_mps),
              ("gyro white noise std", float(np.std(np.array(g_err))), cfg.gyro_std_rps),
              ("heading white noise std", float(np.std(h_err)), cfg.heading_std_rad),
              ("DVL bias std across dives", float(np.std(np.array(bias_v))), cfg.velocity_bias_mps),
              ("DVL dropout rate", float(np.mean(drops)), cfg.velocity_dropout_prob)]
    for name, got, want in checks:
        tol = 0.25 if "dropout" in name else 0.1
        out.append({"check": name, "configured": want, "measured": got, "ok": rel(got, want) < tol,
                    "tolerance": f"±{100 * tol:.0f}%"})
    return out


def check_buoyancy_terminal_velocity() -> list:
    """With no thrust and level attitude, the vertical speed settles where the calibrated damping balances the
    net buoyancy: (lin + quad |w|) w = net_buoyancy (fast_dynamics sign convention)."""
    from arena import Arena
    from build_two_agent_cores import load_calibrated_parameters
    from buoyancy import BuoyancyConfig
    from fast_dynamics import FastDynamics
    from optimizer import load_vehicle_spec
    truth = load_calibrated_parameters(TRUTH, quiet=True)
    vehicle = load_vehicle_spec("vehicle_spec.yaml").vehicle
    out = []
    for offset in (-20.0, -5.0, -1.0, 1.0, 5.0, 20.0):
        from buoyancy import with_buoyancy_offset
        p = with_buoyancy_offset(truth, offset)
        dyn = FastDynamics(p, vehicle)
        lin, quad, nb = dyn.lin[2], dyn.quad[2], p.net_buoyancy_z
        # solve (lin + quad |w|) w = nb for w (damping terms are negative, so w has the opposite sign of nb)
        from scipy.optimize import brentq
        w_pred = brentq(lambda w: (lin + quad * abs(w)) * w - nb, -10, 10)
        for backend in ("stand_in", "mujoco"):
            env = Arena(truth, vehicle, backend=backend, buoyancy=BuoyancyConfig(offset), max_episode_steps=400)
            env.reset(seed=0)
            # hold roll/pitch level with no thrust: the vertical restoring force is along body z
            for _ in range(300):
                _, _, _, _, info = env.step(np.zeros(6))
            w = float(env._core.state.nu[2])
            out.append({"check": f"terminal vertical speed, {offset:+g} N, {backend}", "predicted_mps": w_pred,
                        "simulated_mps": w, "ok": abs(w - w_pred) < max(0.02 * abs(w_pred), 2e-3)})
    return out


def check_clean_unchanged() -> list:
    """Clean episodes, rerun with this code, against the same episodes run on the code before this change
    (clean_regression_check.py): every field of 6,616 fleet episodes, and every observation of 16 arena runs."""
    import clean_regression_check as crc
    with open(crc.REFERENCE) as f:
        ref = json.load(f)
    c = crc.compare(ref, crc.run())
    return [{"check": "clean simulation unchanged: identical episodes to the code before this change", **c}]


def check_pairing_and_determinism() -> list:
    from build_two_agent_cores import load_calibrated_parameters
    from evaluate_comm_degradation_curve import COMM_CONFIGS
    from fleet_belief import BeliefSpec
    from fleet_env import FleetConfig, FleetSim
    from optimizer import load_vehicle_spec
    from sensor_noise import load_sensor_noise
    from disturbance import DEFAULT_FIT_PATH, load_disturbance
    truth = load_calibrated_parameters(TRUTH, quiet=True)
    vehicle = load_vehicle_spec("vehicle_spec.yaml").vehicle
    sev = COMM_CONFIGS["L3a"]
    acts = np.random.default_rng(3).uniform(-0.4, 0.4, (200, 2, 6))

    def roll(noise, dist, backend="stand_in"):
        sim = FleetSim(truth, vehicle, FleetConfig(sensor_noise=noise, disturbance=dist, dynamics_backend=backend),
                       sev["packet_loss_config"], sev["bandwidth_config"],
                       belief=BeliefSpec("last_beacon", "last_beacon"))
        sim.reset(comm_seed=42, initial_separation=3.3, closing_speed=0.2)
        delivered, pos, obs = [], [], []
        for a in acts:
            o, _, term, trunc, info = sim.step([a[0], a[1]])
            delivered.append(tuple(info["delivered"]))
            pos.append(np.concatenate(info["positions"]))
            obs.append(np.concatenate(o))
            if term or trunc:
                break
        return delivered, np.array(pos), np.array(obs)
    dist = load_disturbance(DEFAULT_FIT_PATH)
    d_clean, p_clean, o_clean = roll(None, dist)
    d_noisy, p_noisy, o_noisy = roll(load_sensor_noise("typical", 3.0), dist)
    n = min(len(p_clean), len(p_noisy))
    out = [{"check": "packet losses identical with noise off and on (same actions)",
            "ok": d_clean[:n] == d_noisy[:n]},
           {"check": "true motion identical with noise off and on (same actions, same disturbances)",
            "ok": bool(np.allclose(p_clean[:n], p_noisy[:n], atol=1e-12))},
           {"check": "observations differ with noise on", "ok": bool(np.abs(o_clean[:n] - o_noisy[:n]).max() > 1e-3)}]
    for backend in ("stand_in", "mujoco"):
        a = roll(load_sensor_noise("typical"), dist, backend)
        b = roll(load_sensor_noise("typical"), dist, backend)
        out.append({"check": f"noisy rollout deterministic ({backend})",
                    "ok": a[0] == b[0] and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])})
    return out


# ----------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--combos", type=int, default=400, help="random combinations per part (A and B)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args(argv)
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    for name, p in FLEET_POLICIES.items():
        if p != "random" and not os.path.exists(p + ".zip"):
            print(f"note: {p}.zip missing; '{name}' combinations are skipped")
    have = {k for k, p in FLEET_POLICIES.items() if p == "random" or os.path.exists(p + ".zip")}
    rng = np.random.default_rng(58)
    fleet = [c for c in (sample_fleet_combo(rng, k) for k in range(args.combos)) if c["policy"] in have]
    fleet += [c for c in fleet_corners() if c["policy"] in have]
    print(f"A. {len(fleet)} two-vehicle combinations...", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        a_res = list(ex.map(run_fleet_combo, fleet, chunksize=4))
    tiny_dir = os.path.join(OUT, "tiny_policies")
    os.makedirs(tiny_dir, exist_ok=True)
    print("B. training tiny single-vehicle policies (PPO, SAC, TD3)...", flush=True)
    pols = train_tiny_arena_policies(tiny_dir)
    arena = [sample_arena_combo(rng, k, pols) for k in range(args.combos)]
    print(f"B. {len(arena)} one-vehicle combinations...", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        b_res = list(ex.map(run_arena_combo, arena, chunksize=4))
    print("C. measured against what was configured...", flush=True)
    c_res = (check_sensor_statistics() + check_buoyancy_terminal_velocity() + check_pairing_and_determinism()
             + check_clean_unchanged())

    fails_a = [r for r in a_res if not r["ok"]]
    fails_b = [r for r in b_res if not r["ok"]]
    fails_c = [r for r in c_res if r["ok"] is False]
    res = {"A_two_vehicles": a_res, "B_one_vehicle": b_res, "C_checks": c_res,
           "elapsed_s": round(time.time() - t0, 1)}
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)

    def tally(rows, key):
        out = {}
        for r in rows:
            k = str(r[key])
            ok, n = out.get(k, (0, 0))
            out[k] = (ok + int(r["ok"]), n + 1)
        return out
    L = ["# Stress test: clean/noisy simulation, buoyancy, SAC/TD3 (PROJECT_STATUS.md #58)\n",
         f"{time.strftime('%Y-%m-%d %H:%M')} · {res['elapsed_s'] / 60:.0f} min\n",
         f"## A. Two vehicles: {len(a_res) - len(fails_a)}/{len(a_res)} combinations ran cleanly\n",
         "Randomized over comm level, beacon interval, buoyancy (−25..+25 N, fixed or a range, known or not), sensor "
         "noise (off, tank, typical ×1/×3/×10), roll/pitch disturbance (×0/1/3), current (0-0.5 m/s, fluctuating), "
         "thruster limits, engine, policy (PPO, noise-trained PPO, SAC, TD3, random), belief model and start state "
         "(0.6-8 m apart, ±1 m/s), plus the worst corner of all of them on every policy and engine.\n"]
    for key in ("policy", "backend", "severity", "noise", "buoyancy"):
        L.append(f"- by {key}: " + ", ".join(f"{k} {ok}/{n}" for k, (ok, n) in sorted(tally(a_res, key).items())))
    outcomes = {}
    for r in a_res:
        outcomes[r.get("outcome", "error")] = outcomes.get(r.get("outcome", "error"), 0) + 1
    L.append(f"- outcomes: {outcomes}; largest navigation error {max(r.get('max_nav_m', 0) for r in a_res):.1f} m")
    L += ["", f"## B. One vehicle: {len(b_res) - len(fails_b)}/{len(b_res)} combinations ran cleanly\n",
          "Randomized over the four tasks, controller (PPO, SAC, TD3 policies, hand-written, random), sensor noise, "
          "buoyancy, disturbance, current, thruster limits and engine. Every step checks that the reported distance "
          "is the true one and that success is never declared from the believed position.\n"]
    for key in ("task", "controller", "backend", "noise", "buoyancy"):
        L.append(f"- by {key}: " + ", ".join(f"{k} {ok}/{n}" for k, (ok, n) in sorted(tally(b_res, key).items())))
    L += ["", f"## C. Measured against what was configured: {sum(1 for r in c_res if r['ok'])}/{len(c_res)} passed\n",
          "| Check | Result | Details |", "|---|---|---|"]
    for r in c_res:
        det = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("check", "ok")}
        L.append(f"| {r['check']} | {'pass' if r['ok'] else ('n/a' if r['ok'] is None else '**FAIL**')} | {det} |")
    if fails_a or fails_b:
        L += ["", "## Failures\n"]
        for r in (fails_a + fails_b)[:30]:
            L.append(f"- {r['id']}: {r['problems'][0]}")
    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    return 1 if (fails_a or fails_b or fails_c) else 0


if __name__ == "__main__":
    from keep_awake import keep_awake
    with keep_awake("stress test"):
        sys.exit(main())
