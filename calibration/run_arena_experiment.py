#!/usr/bin/env python3
"""
run_arena_experiment.py -- the single-vehicle experiment for a lab's own vehicle.

    python run_arena_experiment.py --params ../calibration_output_my_rov/calibrated_parameters.yaml \
        --spec vehicle_spec.yaml --task station_keeping --seeds 2 --timesteps 450000 \
        [--current-speed 0 0.3] [--thrusters bluerov2_heavy] [--compare-generic] \
        [--algorithm sac] [--mode noisy [--compare-clean-training]] [--buoyancy -3] --out ../arena_my_rov
    (or: python strathe.py arena ...)

The question it answers: how well does a controller trained in the simulator
do a task on YOUR vehicle, and (with --compare-generic) how much does the
calibration matter for that?

- "Your vehicle" is the calibrated physics (--params, --spec), plus the water
  current and thruster limits you choose. Every policy is evaluated on it.
- Policies are trained in the same simulator (--algorithm: PPO by default,
  or SAC or TD3 with the settings validated in PROJECT_STATUS.md #37): on the
  calibrated physics, and with --compare-generic also on the default,
  uncalibrated physics from the vehicle description (the "no calibration"
  alternative).
- Every policy sees the same evaluation episodes (same targets, same
  currents), so the calibrated-vs-default difference is paired.
- Two references that need no training: random commands, and a simple
  hand-written controller that pushes toward the target.
- Options (all off by default): a water current (--current-speed ...), thruster
  limits (--thrusters), random roll/pitch torques (--disturbance fitted
  [--disturbance-scale 2]), and the physics engine for the tests (--backend
  mujoco, or both). Policies train on the fast stand-in with the same current,
  thrusters and disturbance; the test engine is your choice.
- Clean or noisy (--mode noisy = realistic sensor noise + the fitted roll/pitch
  disturbance; --noise and --disturbance set each part), and a buoyancy change
  from the calibration (--buoyancy N, e.g. -3 for an added payload). With
  --compare-clean-training, policies are also trained on a clean simulation
  and compared on your noisy test: does training with the noise pay off?

Writes report.md (plain language), results.json, episodes.csv and the
trained policies to --out. This is simulation only: the "vehicle" is your
calibrated model, not the real one; hardware trials are the real test.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = ("waypoint", "station_keeping", "docking", "obstacle_avoidance")
TASK_PLAIN = {
    "waypoint": "reach a target point (within 0.5 m) that is up to 3 m away in each direction",
    "station_keeping": "reach a target point and stay within 0.5 m of it for 20 consecutive steps (2 s)",
    "docking": "reach a target point slowly (under 0.3 m/s) and level (within 0.2 rad)",
    "obstacle_avoidance": "reach a target point while routing around an obstacle",
}


# ----------------------------------------------------------------- environment

ENGINE_PLAIN = {"stand_in": "stand-in", "mujoco": "MuJoCo"}


def make_env(params, ek: dict, backend: str = "stand_in"):
    """ek: everything needed to rebuild the environment in a worker process (picklable)."""
    from arena import Arena
    from current import CurrentConfig
    from optimizer import load_vehicle_spec
    vehicle = load_vehicle_spec(ek["spec"]).vehicle
    current = CurrentConfig.from_dict(ek["current"]) if ek.get("current") else None
    thrusters = None
    if ek.get("thrusters"):
        from fleet_experiment_schema import resolve_thrusters
        thrusters = resolve_thrusters(ek["thrusters"])
    disturbance = None
    if ek.get("disturbance"):
        from disturbance import load_disturbance
        disturbance = load_disturbance(ek["disturbance"], ek.get("disturbance_scale", 1.0))
    noise = None
    if ek.get("noise"):
        from sensor_noise import resolve_sensor_noise
        noise = resolve_sensor_noise(ek["noise"], ek.get("noise_scale", 1.0), ek.get("start_fix"))
    buoyancy = None
    if ek.get("buoyancy"):
        from buoyancy import BuoyancyConfig
        buoyancy = BuoyancyConfig.from_dict(ek["buoyancy"])
    return Arena(params, vehicle, task=ek["task"], max_episode_steps=ek["max_episode_steps"], current=current,
                 thrusters=thrusters, backend=backend, disturbance=disturbance, sensor_noise=noise, buoyancy=buoyancy)


def clean_env_kwargs(ek: dict) -> dict:
    """The same environment without sensor noise or disturbance (current, thrusters and buoyancy kept)."""
    return dict(ek, noise=None, disturbance=None)


def physics_for(label: str, params_path: str, spec: str, quiet: bool = True):
    """label: a training condition; every one but 'generic' trains on (and every test runs on) the calibration."""
    from build_two_agent_cores import load_calibrated_parameters
    from optimizer import initial_parameters, load_vehicle_spec
    if label == "generic":
        return initial_parameters(load_vehicle_spec(spec))
    return load_calibrated_parameters(params_path, quiet=quiet)


# ----------------------------------------------------------------- training / evaluation (worker processes)

ALGORITHMS = ("ppo", "sac", "td3")


def train_job(job: dict) -> dict:
    """PPO: 4 environments, n_steps 512 (the single-vehicle default since #56). SAC/TD3: one environment and
    stable-baselines3's own settings, exactly as validated on this vehicle in PROJECT_STATUS.md #37."""
    import torch
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1)
    params = physics_for(job["physics"], job["params"], job["env"]["spec"])
    env_kwargs = clean_env_kwargs(job["env"]) if job.get("train_clean") else job["env"]
    alg = job.get("algorithm", "ppo")
    if alg == "ppo":
        env = DummyVecEnv([lambda: make_env(params, env_kwargs) for _ in range(4)])
        model = PPO("MlpPolicy", env, seed=job["seed"], verbose=0, device="cpu", n_steps=512)
    else:
        env = DummyVecEnv([lambda: make_env(params, env_kwargs)])
        model = {"sac": SAC, "td3": TD3}[alg]("MlpPolicy", env, seed=job["seed"], verbose=0, device="cpu")
    t0 = time.time()
    model.learn(total_timesteps=job["timesteps"])
    model.save(job["out"])
    return {"policy": job["out"] + ".zip", "physics": job["physics"], "seed": job["seed"], "algorithm": alg,
            "train_s": round(time.time() - t0, 1)}


def evaluate_job(job: dict) -> list:
    """Run one controller on the truth environment for every episode seed. Returns episode rows."""
    import torch
    torch.set_num_threads(1)
    truth = physics_for("calibrated", job["params"], job["env"]["spec"])
    env = make_env(truth, job["env"], job.get("engine", "stand_in"))
    kind = job["controller"]
    if kind == "policy":
        from policy_io import load_policy
        model, _alg = load_policy(job["policy"], job.get("algorithm"))
        act = lambda obs, rng: model.predict(obs, deterministic=True)[0]
    elif kind == "hand_written":
        from evaluate_policy_arena import proportional_controller
        act = lambda obs, rng: proportional_controller(obs)
    else:
        act = lambda obs, rng: rng.uniform(-1.0, 1.0, 6)
    rows = []
    from arena import TERMINAL_REASON_COLLISION, TERMINAL_REASON_GOAL
    for ep_seed in job["episode_seeds"]:
        rng = np.random.default_rng(ep_seed + 7)
        obs, info = env.reset(seed=int(ep_seed))
        done = False
        steps, effort, info, nav = 0, 0.0, {}, []
        while not done:
            obs, _, term, trunc, info = env.step(act(obs, rng))
            steps += 1
            effort += float(np.sum(np.abs(info["tau_applied"]))) * 0.1
            nav.append(info.get("nav_error_m", 0.0))
            done = term or trunc
        reason = info.get("terminal_reason")
        cur = info.get("current_world")
        rows.append({"controller": job["label"], "engine": job.get("engine", "stand_in"),
                     "physics": job.get("physics", ""), "seed": job.get("seed", ""),
                     "episode_seed": int(ep_seed), "success": int(reason == TERMINAL_REASON_GOAL),
                     "collision": int(reason == TERMINAL_REASON_COLLISION), "steps": steps,
                     "final_distance_m": round(float(info["distance"]), 4), "effort_Ns": round(effort, 2),
                     "current_speed_mps": 0.0 if cur is None else round(float(np.linalg.norm(cur)), 4),
                     "mean_nav_error_m": round(float(np.mean(nav)), 4) if nav else 0.0,
                     "buoyancy_offset_n": round(float(info.get("buoyancy_offset_n", 0.0)), 4)})
    return rows


# ----------------------------------------------------------------- statistics and report

def summarize(rows: list, label: str, engine: str = "stand_in") -> dict:
    from fleet_stats import wilson_ci
    r = [x for x in rows if x["controller"] == label and x.get("engine", "stand_in") == engine]
    n, s = len(r), sum(x["success"] for x in r)
    lo, hi = wilson_ci(s, n)
    ok = [x["steps"] for x in r if x["success"]]
    return {"controller": label, "engine": engine, "episodes": n, "success_rate": s / n if n else float("nan"), "ci": [lo, hi],
            "collision_rate": sum(x["collision"] for x in r) / n if n else float("nan"),
            "mean_time_to_success_s": round(0.1 * float(np.mean(ok)), 2) if ok else None,
            "mean_final_distance_m": round(float(np.mean([x["final_distance_m"] for x in r])), 3) if r else None,
            "mean_effort_Ns": round(float(np.mean([x["effort_Ns"] for x in r])), 1) if r else None,
            "mean_nav_error_m": (round(float(np.mean([x.get("mean_nav_error_m", 0.0) for x in r])), 3)
                                 if r else None)}


def paired_difference(rows: list, a: str, b: str, n_boot: int = 2000, seed: int = 0,
                      engine: str = "stand_in") -> Optional[dict]:
    """Success(a) - success(b), paired by (training seed, episode seed); 95% bootstrap CI over episodes."""
    rows = [x for x in rows if x.get("engine", "stand_in") == engine]
    ra = {(x["seed"], x["episode_seed"]): x["success"] for x in rows if x["physics"] == a}
    rb = {(x["seed"], x["episode_seed"]): x["success"] for x in rows if x["physics"] == b}
    keys = sorted(set(ra) & set(rb))
    if not keys:
        return None
    d = np.array([ra[k] - rb[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return {"diff": float(d.mean()), "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "n_pairs": len(keys)}


def write_report(path: str, meta: dict, summaries: list, contrasts: dict) -> None:
    pct = lambda x: f"{100 * x:.0f}%"
    L = [f"# Single-vehicle experiment: {meta['name']}\n",
         f"{datetime.now():%Y-%m-%d %H:%M} · {meta['n_episodes']} evaluation episodes per controller · "
         f"{meta['elapsed_s'] / 60:.1f} min\n", "## What was tested\n",
         f"One simulated vehicle must {TASK_PLAIN[meta['task']]}, within {meta['max_episode_steps']} steps "
         f"(0.1 s each). **The vehicle** is your calibrated physics (`{os.path.basename(meta['params'])}`, "
         f"`{os.path.basename(meta['spec'])}`)"
         + (f", in a water current: {meta['current_described']}" if meta.get("current_described") else ", in still water")
         + (f", with thruster limits: {meta['thrusters_described']}" if meta.get("thrusters_described")
            else ", with no thruster limits (any command up to ±30 N / N·m per axis)")
         + (f", pushed by random roll/pitch torques ({meta['disturbance_described']})"
            if meta.get("disturbance_described") else ", with no roll/pitch disturbance")
         + (f", with sensor noise ({meta['noise_described']}): the controllers see what the vehicle's own sensors "
            "report and where it believes it is, and success is judged on the true position"
            if meta.get("noise_described") else ", with exact sensor readings")
         + (f", with a {meta['buoyancy_described']}" if meta.get("buoyancy_described") else "")
         + f". Physics engine for the tests: {' and '.join(ENGINE_PLAIN[e] for e in meta['engines'])}"
         + (" (policies are trained on the stand-in, then tested on each)" if meta["engines"] != ["stand_in"] else "")
         + ". Every controller below drives that same vehicle through the same episodes (same targets"
         + (", same currents" if meta.get("current_described") else "")
         + (", same disturbances" if meta.get("disturbance_described") else "")
         + (", same sensor noise" if meta.get("noise_described") else "") + ").\n",
         "Controllers:\n",
         f"- **trained on your calibration**: {meta.get('algorithm', 'ppo').upper()} trained in the simulator on your "
         f"calibrated physics, with the same conditions as the test "
         f"({meta['seeds']} training seed(s), {meta['timesteps']:,} steps each)"]
    if meta["compare_generic"]:
        L.append("- **trained on default physics**: the same training, but on the uncalibrated default values from the "
                 "vehicle description, i.e. what you would have without calibrating")
    if meta.get("compare_clean_training"):
        L.append("- **trained clean**: the same training on your calibration, but with exact sensors and no roll/pitch "
                 "disturbance, then tested on the same noisy vehicle")
    L += ["- **hand-written controller**: pushes toward the target in proportion to the distance; no training",
          "- **random commands**: a floor every controller should beat\n", "## Results\n",
          "| Controller | Engine | Success (95% CI) | Mean time to succeed | Mean final distance | Mean effort* |"
          + (" Mean navigation error† |" if meta.get("noise_described") else ""),
          "|---|---|---|---|---|---|" + ("---|" if meta.get("noise_described") else "")]
    for s in summaries:
        t = "n/a" if s["mean_time_to_success_s"] is None else f"{s['mean_time_to_success_s']:.1f} s"
        L.append(f"| {s['controller']} | {ENGINE_PLAIN[s['engine']]} | {pct(s['success_rate'])} "
                 f"({pct(s['ci'][0])}–{pct(s['ci'][1])}) | {t} | "
                 f"{s['mean_final_distance_m']:.2f} m | {s['mean_effort_Ns']:.0f} |"
                 + (f" {100 * (s['mean_nav_error_m'] or 0):.0f} cm |" if meta.get("noise_described") else ""))
    if len(meta["engines"]) == 2:
        by = {(s["controller"], s["engine"]): s["success_rate"] for s in summaries}
        gaps = [abs(by[(c, "stand_in")] - by[(c, "mujoco")]) for c in {s["controller"] for s in summaries}
                if (c, "stand_in") in by and (c, "mujoco") in by]
        L.append(f"\nThe two engines agree to within {100 * max(gaps):.0f} points of success on every controller."
                 if gaps else "")
    L.append("\n*Effort: the delivered forces (N) and torques (N·m), in absolute value, summed over the episode and "
             "multiplied by the 0.1 s step. It's a relative measure of how hard the thrusters work (lower = gentler), "
             "not a physical energy.\n"
             + ("\n†Navigation error: how far the vehicle's believed position was from its true position, averaged "
                "over the episode. Final distance and success use the true position.\n"
                if meta.get("noise_described") else ""))
    for engine, nc in (meta.get("noise_training_contrasts") or {}).items():
        if nc is None:
            continue
        sig = nc["ci"][0] > 0 or nc["ci"][1] < 0
        L.append("## Does training with the noise pay off?"
                 + (f" ({ENGINE_PLAIN[engine]})" if len(meta["engines"]) > 1 else "") + "\n")
        L.append(f"On your noisy test, policies trained with the noise succeed {100 * nc['diff']:+.0f} points "
                 f"({100 * nc['ci'][0]:+.0f} to {100 * nc['ci'][1]:+.0f}, 95% CI; {nc['n_pairs']} paired episodes) "
                 "compared with the same training on a clean simulation. "
                 + ("That difference is clear of zero." if sig else
                    "That interval includes zero: here, training clean did about as well.") + "\n")
    for engine, contrast in (contrasts or {}).items():
        if contrast is None:
            continue
        sig = contrast["ci"][0] > 0 or contrast["ci"][1] < 0
        L.append("## Does the calibration matter for this task?"
                 + (f" ({ENGINE_PLAIN[engine]})" if len(contrasts) > 1 else "") + "\n")
        L.append(f"Policies trained on your calibration succeed {100 * contrast['diff']:+.0f} points "
                 f"({100 * contrast['ci'][0]:+.0f} to {100 * contrast['ci'][1]:+.0f}, 95% CI; "
                 f"{contrast['n_pairs']} paired episodes) compared with policies trained on default physics. "
                 + ("That difference is clear of zero." if sig else
                    "That interval includes zero: on this task, this experiment can't tell the two apart.")
                 + " Training-seed variation is real in reinforcement learning; per-seed numbers are in "
                   "`results.json`, and more seeds (`--seeds`) narrow the interval.\n")
        if contrast["ci"][1] < 0:
            L.append("Training on your calibration did *worse* here. That usually means the training budget was too "
                     "short for your vehicle, not that the calibration is wrong. Every policy is tested on your "
                     "calibrated vehicle, and a heavily damped vehicle is slower to learn on than the lighter default "
                     "physics. In our tests, tripling `--timesteps` took calibration-trained policies from 10% to 96% "
                     "on the reference vehicle (PROJECT_STATUS.md #56) and from 78% to 100% on a second one (#62); "
                     "the default is now that tripled budget (450k). Re-run with a larger `--timesteps` before "
                     "drawing conclusions, and don't read this as calibration hurting.\n")
    L += ["## What this does and doesn't show\n",
          "- It is simulation only: the \"vehicle\" is your calibrated model. How well that model matches the real "
          "vehicle is what your calibration report measures; a hardware trial is the final test.",
          "- Success rates are for these episodes and this training budget; a longer `--timesteps` usually helps "
          "harder tasks (station keeping, docking).",
          f"- Reproduce: `{meta['command']}`\n"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


# ----------------------------------------------------------------- main

def run(args) -> dict:
    t0 = time.time()
    import sim_mode
    args.noise, args.disturbance = sim_mode.resolve(getattr(args, "mode", "clean"), getattr(args, "noise", None),
                                                    args.disturbance)
    from current import CurrentConfig
    from optimizer import load_vehicle_spec
    load_vehicle_spec(args.spec)  # fail early on a bad description
    physics_for("calibrated", args.params, args.spec, quiet=False)  # fail early; say once if not converged
    current = None
    if args.current_speed:
        pick = lambda xs: xs[0] if len(xs) == 1 else list(xs[:2])
        current = {"speed_mps": pick(args.current_speed), "direction_deg": pick(args.current_direction),
                   "variability": {"speed_std_mps": args.current_speed_std,
                                   "direction_std_deg": args.current_direction_std,
                                   "time_scale_s": args.current_time_scale}}
        CurrentConfig.from_dict(current)  # validates
    dist_path = None
    dist_desc = None
    if args.disturbance != "none" and args.disturbance_scale > 0:
        from disturbance import DEFAULT_FIT_PATH, load_disturbance
        dist_path = DEFAULT_FIT_PATH if args.disturbance == "fitted" else os.path.abspath(args.disturbance)
        d = load_disturbance(dist_path, args.disturbance_scale, truth_params_path=args.params)  # validates; warns once
        dist_desc = (f"`{os.path.basename(dist_path)}` x{args.disturbance_scale:g}: roll {d.roll.sigma_nm * d.scale:.2f} "
                     f"N·m, pitch {d.pitch.sigma_nm * d.scale:.2f} N·m typical size")
    engines = ["stand_in", "mujoco"] if args.backend == "both" else [args.backend]
    noise_desc = None
    if args.noise not in (None, "none") and args.noise_scale > 0:
        from sensor_noise import resolve_sensor_noise
        from sensor_noise import PRESETS
        nc = resolve_sensor_noise(args.noise if args.noise in PRESETS else os.path.abspath(args.noise),
                                  args.noise_scale, args.start_fix_error)
        noise_desc = nc.describe() if nc is not None else None
    elif args.start_fix_error is not None:
        raise ValueError("--start-fix-error needs sensor noise on (--mode noisy, or --noise)")
    from sensor_noise import PRESETS as _PRESETS
    noise_ref = (None if noise_desc is None else
                 (args.noise if args.noise in _PRESETS else os.path.abspath(args.noise)))
    from sim_mode import buoyancy_from_flags
    buoy = buoyancy_from_flags(args.buoyancy)
    if args.compare_clean_training and noise_ref is None and dist_path is None:
        raise ValueError("--compare-clean-training compares training with and without sensor noise and roll/pitch "
                         "disturbance, but both are off; add --mode noisy (or --noise / --disturbance)")
    ek = {"spec": os.path.abspath(args.spec), "task": args.task, "max_episode_steps": args.max_episode_steps,
          "current": current, "thrusters": args.thrusters, "disturbance": dist_path,
          "disturbance_scale": args.disturbance_scale, "noise": noise_ref, "noise_scale": args.noise_scale,
          "start_fix": args.start_fix_error if noise_ref is not None else None,
          "buoyancy": None if buoy is None else buoy.to_dict()}
    thr_desc = None
    if args.thrusters:
        import thruster_model as tm
        from fleet_experiment_schema import resolve_thrusters
        thr_desc = tm.describe(resolve_thrusters(args.thrusters))
    os.makedirs(os.path.join(args.out, "policies"), exist_ok=True)
    params = os.path.abspath(args.params)
    workers = max(1, args.workers)

    policies = []
    if args.policy:
        from policy_io import PolicyLoadError, load_policy
        from build_two_agent_cores import load_calibrated_parameters
        try:
            model, alg = load_policy(os.path.abspath(args.policy))
        except PolicyLoadError as e:
            raise ValueError(str(e)) from e
        want = make_env(load_calibrated_parameters(args.params, quiet=True), ek).observation_space.shape
        have = getattr(model.observation_space, "shape", None)
        if have != want or getattr(model.action_space, "shape", None) != (6,):
            hint = (" It looks like a two-vehicle (fleet) policy; use it with `strathe fleet --policies`."
                    if have == (13,) else "")
            raise ValueError(f"{args.policy} expects observations of shape {have}, but the {args.task} task gives "
                             f"{want} (one vehicle: velocity, target in body axes, roll, pitch"
                             + (", obstacle" if args.task == "obstacle_avoidance" else "") + f").{hint}")
        policies = [{"policy": os.path.abspath(args.policy), "physics": "your policy", "seed": 0, "train_s": 0,
                     "algorithm": alg}]
    else:
        conditions = (["calibrated"] + (["generic"] if args.compare_generic else [])
                      + (["calibrated_clean"] if args.compare_clean_training else []))
        jobs = [{"physics": c, "params": params, "env": ek, "seed": s, "timesteps": args.timesteps,
                 "algorithm": args.algorithm, "train_clean": c == "calibrated_clean",
                 "out": os.path.join(os.path.abspath(args.out), "policies", f"{args.algorithm}_{c}_seed{s}")}
                for c in conditions for s in range(1, args.seeds + 1)]
        print(f"Training {len(jobs)} policies ({args.timesteps:,} steps each, {min(workers, len(jobs))} at a time)...",
              flush=True)
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            for r in ex.map(train_job, jobs):
                print(f"  trained {os.path.basename(r['policy'])} in {r['train_s'] / 60:.1f} min", flush=True)
                policies.append(r)

    episode_seeds = list(range(1000, 1000 + args.episodes))
    label = {"calibrated": "trained on your calibration", "generic": "trained on default physics",
             "calibrated_clean": "trained clean", "your policy": "your policy"}
    eval_jobs = [{"controller": "policy", "policy": p["policy"], "physics": p["physics"], "seed": p["seed"],
                  "algorithm": p.get("algorithm"),
                  "label": label[p["physics"]], "params": params, "env": ek, "episode_seeds": episode_seeds,
                  "engine": e} for e in engines for p in policies]
    eval_jobs += [{"controller": c, "label": lab, "params": params, "env": ek, "episode_seeds": episode_seeds,
                   "engine": e}
                  for e in engines for c, lab in (("hand_written", "hand-written controller"),
                                                  ("random", "random commands"))]
    print(f"Evaluating {len(eval_jobs)} controllers on {len(episode_seeds)} episodes each...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=min(workers, len(eval_jobs))) as ex:
        for r in ex.map(evaluate_job, eval_jobs):
            rows += r
    order = list(dict.fromkeys(j["label"] for j in eval_jobs))
    summaries = [summarize(rows, lab, e) for e in engines for lab in order]
    contrasts = ({e: paired_difference(rows, "calibrated", "generic", engine=e) for e in engines}
                 if args.compare_generic and not args.policy else {})
    noise_contrasts = ({e: paired_difference(rows, "calibrated", "calibrated_clean", engine=e) for e in engines}
                       if args.compare_clean_training and not args.policy else {})
    per_seed = [{"physics": p["physics"], "seed": p["seed"], "engine": e,
                 "success_rate": float(np.mean([x["success"] for x in rows if x["physics"] == p["physics"]
                                                and x["seed"] == p["seed"] and x["engine"] == e]))}
                for e in engines for p in policies]

    with open(os.path.join(args.out, "episodes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {"name": args.name or os.path.basename(os.path.abspath(args.out)), "task": args.task,
            "params": params, "spec": os.path.abspath(args.spec), "current": current,
            "current_described": CurrentConfig.from_dict(current).describe(teammates=False) if current else None,
            "thrusters": args.thrusters, "thrusters_described": thr_desc, "seeds": args.seeds,
            "engines": engines, "disturbance": dist_path, "disturbance_scale": args.disturbance_scale,
            "disturbance_described": dist_desc, "algorithm": args.algorithm,
            "mode": args.mode, "noise": noise_ref, "noise_scale": args.noise_scale, "noise_described": noise_desc,
            "start_fix_error_m": args.start_fix_error,
            "buoyancy": None if buoy is None else buoy.to_dict(),
            "buoyancy_described": None if buoy is None else buoy.describe(teammates=False),
            "compare_clean_training": bool(args.compare_clean_training and not args.policy),
            "noise_training_contrasts": noise_contrasts,
            "timesteps": args.timesteps, "n_episodes": args.episodes, "max_episode_steps": args.max_episode_steps,
            "compare_generic": bool(args.compare_generic and not args.policy), "policy": args.policy,
            "command": "python " + " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
            "python": platform.python_version(), "elapsed_s": round(time.time() - t0, 1)}
    contrast = contrasts.get(engines[0]) if contrasts else None
    results = {"meta": meta, "summary": summaries, "calibrated_vs_default": contrast,
               "noise_trained_vs_clean_trained_by_engine": noise_contrasts,
               "calibrated_vs_default_by_engine": contrasts, "per_seed": per_seed,
               "policies": policies}
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_report(os.path.join(args.out, "report.md"), meta, summaries, contrasts)
    print(f"\nReport: {os.path.join(args.out, 'report.md')}")
    for s in summaries:
        print(f"  {s['controller']:<28} {ENGINE_PLAIN[s['engine']]:<9} success {100 * s['success_rate']:5.1f}% "
              f"({100 * s['ci'][0]:.0f}-{100 * s['ci'][1]:.0f}%)")
    for e, c in contrasts.items():
        if c:
            print(f"  calibrated - default training ({ENGINE_PLAIN[e]}): {100 * c['diff']:+.1f} points "
                  f"({100 * c['ci'][0]:+.1f} to {100 * c['ci'][1]:+.1f})")
    for e, c in noise_contrasts.items():
        if c:
            print(f"  noise-trained - clean-trained ({ENGINE_PLAIN[e]}): {100 * c['diff']:+.1f} points "
                  f"({100 * c['ci'][0]:+.1f} to {100 * c['ci'][1]:+.1f})")
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=os.path.join(HERE, "calibrated_parameters_fleet_truth.yaml"),
                    help="your calibrated_parameters.yaml (default: the shipped BlueROV2 calibration)")
    ap.add_argument("--spec", default=os.path.join(HERE, "vehicle_spec.yaml"), help="vehicle description")
    ap.add_argument("--task", choices=TASKS, default="waypoint")
    ap.add_argument("--seeds", type=int, default=2, help="training seeds per condition")
    ap.add_argument("--timesteps", type=int, default=450_000,
                    help="training steps per policy (450k: 150k was too short for heavily damped vehicles, #56, #63)")
    ap.add_argument("--episodes", type=int, default=100, help="evaluation episodes per controller")
    ap.add_argument("--max-episode-steps", type=int, default=200)
    ap.add_argument("--compare-generic", action="store_true",
                    help="also train on the default (uncalibrated) physics and compare")
    ap.add_argument("--policy", help="evaluate this trained policy (.zip; PPO, SAC or TD3, read from the file) "
                                     "instead of training new ones")
    ap.add_argument("--algorithm", choices=ALGORITHMS, default="ppo",
                    help="training algorithm: ppo (default), sac or td3 (as validated in PROJECT_STATUS.md #37)")
    ap.add_argument("--mode", choices=["clean", "noisy"], default="clean",
                    help="clean (exact sensors, no disturbance; default) or noisy (sensor noise 'typical' + roll/pitch "
                         "disturbance 'fitted'); --noise and --disturbance override either part")
    ap.add_argument("--noise", default=None,
                    help="sensor noise: none, tank (measured), typical (assumed sizes, not measured), open_water "
                         "(measured on a BlueROV2 at a fish farm; biases assumed), or a file from fit_sensor_noise.py "
                         "(default: from --mode)")
    ap.add_argument("--noise-scale", type=float, default=1.0, help="0 = off, 2 = twice as noisy")
    ap.add_argument("--start-fix-error", type=float, default=None, metavar="M",
                    help="error of each vehicle's starting position fix, m per axis, for dead reckoning "
                         "(default: the noise preset's; open_water 0.17, typical 0.2). Most of what sensor "
                         "noise costs in the fleet (PROJECT_STATUS.md #60)")
    ap.add_argument("--buoyancy", type=float, nargs="+", metavar="N",
                    help="net buoyancy change from the calibration in N (- = heavier, e.g. a payload), or MIN MAX "
                         "drawn per episode")
    ap.add_argument("--compare-clean-training", action="store_true",
                    help="also train on a clean simulation and compare on your noisy test")
    ap.add_argument("--current-speed", type=float, nargs="+", metavar="MPS",
                    help="water current speed in m/s, or MIN MAX drawn per episode")
    ap.add_argument("--current-direction", type=float, nargs="+", default=[0.0, 360.0], metavar="DEG")
    ap.add_argument("--current-speed-std", type=float, default=0.0)
    ap.add_argument("--current-direction-std", type=float, default=0.0)
    ap.add_argument("--current-time-scale", type=float, default=30.0)
    ap.add_argument("--thrusters", help="limit commands to what the thrusters can deliver: bluerov2_heavy, or a YAML "
                                        "file with a thrusters block (an ingest spec works)")
    ap.add_argument("--backend", choices=["stand_in", "mujoco", "both"], default="stand_in",
                    help="physics engine for the tests: the fast stand-in (default), MuJoCo, or both side by side. "
                         "Policies always train on the stand-in")
    ap.add_argument("--disturbance", default=None,
                    help="random roll/pitch torques: none, fitted (the fit to the reference BlueROV2 logs), "
                         "or a path to your own fit (fit_rollpitch_disturbance.py); default: from --mode. Used in "
                         "training and testing")
    ap.add_argument("--disturbance-scale", type=float, default=1.0,
                    help="0 = off, 1 = as fitted, 2 = twice as rough")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--name", default="")
    ap.add_argument("--out", required=True)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.seeds < 1 or args.episodes < 1 or args.timesteps < 1000:
        print("ERROR: --seeds and --episodes must be >= 1 and --timesteps >= 1000", file=sys.stderr)
        return 2
    try:
        run(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    from keep_awake import keep_awake
    with keep_awake("single-vehicle experiment"):  # PROJECT_STATUS.md #49
        sys.exit(main())
