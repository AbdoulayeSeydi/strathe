#!/usr/bin/env python3
"""
run_arena_experiment.py -- the single-vehicle experiment for a lab's own vehicle.

    python run_arena_experiment.py --params ../calibration_output_my_rov/calibrated_parameters.yaml \
        --spec vehicle_spec.yaml --task station_keeping --seeds 2 --timesteps 150000 \
        [--current-speed 0 0.3] [--thrusters bluerov2_heavy] [--compare-generic] --out ../arena_my_rov
    (or: python strathe.py arena ...)

The question it answers: how well does a controller trained in the simulator
do a task on YOUR vehicle, and (with --compare-generic) how much does the
calibration matter for that?

- "Your vehicle" is the calibrated physics (--params, --spec), plus the water
  current and thruster limits you choose. Every policy is evaluated on it.
- Policies are trained with PPO in the same simulator: on the calibrated
  physics, and with --compare-generic also on the default, uncalibrated
  physics from the vehicle description (the "no calibration" alternative).
- Every policy sees the same evaluation episodes (same targets, same
  currents), so the calibrated-vs-default difference is paired.
- Two references that need no training: random commands, and a simple
  hand-written controller that pushes toward the target.

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

def env_kwargs(args_like: dict) -> dict:
    """Everything needed to rebuild the environment in a worker process (picklable)."""
    return {k: args_like[k] for k in ("spec", "task", "max_episode_steps", "current", "thrusters")}


def make_env(params, ek: dict):
    from arena import Arena
    from current import CurrentConfig
    from optimizer import load_vehicle_spec
    vehicle = load_vehicle_spec(ek["spec"]).vehicle
    current = CurrentConfig.from_dict(ek["current"]) if ek.get("current") else None
    thrusters = None
    if ek.get("thrusters"):
        from fleet_experiment_schema import resolve_thrusters
        thrusters = resolve_thrusters(ek["thrusters"])
    return Arena(params, vehicle, task=ek["task"], max_episode_steps=ek["max_episode_steps"], current=current,
                 thrusters=thrusters)


def physics_for(label: str, params_path: str, spec: str, quiet: bool = True):
    from build_two_agent_cores import load_calibrated_parameters
    from optimizer import initial_parameters, load_vehicle_spec
    if label == "generic":
        return initial_parameters(load_vehicle_spec(spec))
    return load_calibrated_parameters(params_path, quiet=quiet)


# ----------------------------------------------------------------- training / evaluation (worker processes)

def train_job(job: dict) -> dict:
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    torch.set_num_threads(1)
    params = physics_for(job["physics"], job["params"], job["env"]["spec"])
    env = DummyVecEnv([lambda: make_env(params, job["env"]) for _ in range(4)])
    model = PPO("MlpPolicy", env, seed=job["seed"], verbose=0, device="cpu", n_steps=512)
    t0 = time.time()
    model.learn(total_timesteps=job["timesteps"])
    model.save(job["out"])
    return {"policy": job["out"] + ".zip", "physics": job["physics"], "seed": job["seed"],
            "train_s": round(time.time() - t0, 1)}


def evaluate_job(job: dict) -> list:
    """Run one controller on the truth environment for every episode seed. Returns episode rows."""
    import torch
    torch.set_num_threads(1)
    truth = physics_for("calibrated", job["params"], job["env"]["spec"])
    env = make_env(truth, job["env"])
    kind = job["controller"]
    if kind == "policy":
        from stable_baselines3 import PPO
        model = PPO.load(job["policy"], device="cpu")
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
        steps, effort, info = 0, 0.0, {}
        while not done:
            obs, _, term, trunc, info = env.step(act(obs, rng))
            steps += 1
            effort += float(np.sum(np.abs(info["tau_applied"]))) * 0.1
            done = term or trunc
        reason = info.get("terminal_reason")
        cur = info.get("current_world")
        rows.append({"controller": job["label"], "physics": job.get("physics", ""), "seed": job.get("seed", ""),
                     "episode_seed": int(ep_seed), "success": int(reason == TERMINAL_REASON_GOAL),
                     "collision": int(reason == TERMINAL_REASON_COLLISION), "steps": steps,
                     "final_distance_m": round(float(info["distance"]), 4), "effort_Ns": round(effort, 2),
                     "current_speed_mps": 0.0 if cur is None else round(float(np.linalg.norm(cur)), 4)})
    return rows


# ----------------------------------------------------------------- statistics and report

def summarize(rows: list, label: str) -> dict:
    from fleet_stats import wilson_ci
    r = [x for x in rows if x["controller"] == label]
    n, s = len(r), sum(x["success"] for x in r)
    lo, hi = wilson_ci(s, n)
    ok = [x["steps"] for x in r if x["success"]]
    return {"controller": label, "episodes": n, "success_rate": s / n if n else float("nan"), "ci": [lo, hi],
            "collision_rate": sum(x["collision"] for x in r) / n if n else float("nan"),
            "mean_time_to_success_s": round(0.1 * float(np.mean(ok)), 2) if ok else None,
            "mean_final_distance_m": round(float(np.mean([x["final_distance_m"] for x in r])), 3) if r else None,
            "mean_effort_Ns": round(float(np.mean([x["effort_Ns"] for x in r])), 1) if r else None}


def paired_difference(rows: list, a: str, b: str, n_boot: int = 2000, seed: int = 0) -> Optional[dict]:
    """Success(a) - success(b), paired by (training seed, episode seed); 95% bootstrap CI over episodes."""
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


def write_report(path: str, meta: dict, summaries: list, contrast: Optional[dict]) -> None:
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
         + ". Every controller below drives that same vehicle through the same episodes (same targets"
         + (", same currents" if meta.get("current_described") else "") + ").\n",
         "Controllers:\n",
         f"- **trained on your calibration**: PPO trained in the simulator on your calibrated physics "
         f"({meta['seeds']} training seed(s), {meta['timesteps']:,} steps each)"]
    if meta["compare_generic"]:
        L.append("- **trained on default physics**: the same training, but on the uncalibrated default values from the "
                 "vehicle description, i.e. what you would have without calibrating")
    L += ["- **hand-written controller**: pushes toward the target in proportion to the distance; no training",
          "- **random commands**: a floor every controller should beat\n", "## Results\n",
          "| Controller | Success (95% CI) | Mean time to succeed | Mean final distance | Mean effort* |",
          "|---|---|---|---|---|"]
    for s in summaries:
        t = "n/a" if s["mean_time_to_success_s"] is None else f"{s['mean_time_to_success_s']:.1f} s"
        L.append(f"| {s['controller']} | {pct(s['success_rate'])} ({pct(s['ci'][0])}–{pct(s['ci'][1])}) | {t} | "
                 f"{s['mean_final_distance_m']:.2f} m | {s['mean_effort_Ns']:.0f} |")
    L.append("\n*Effort: the delivered forces (N) and torques (N·m), in absolute value, summed over the episode and "
             "multiplied by the 0.1 s step. It's a relative measure of how hard the thrusters work (lower = gentler), "
             "not a physical energy.\n")
    if contrast is not None:
        sig = contrast["ci"][0] > 0 or contrast["ci"][1] < 0
        L.append("## Does the calibration matter for this task?\n")
        L.append(f"Policies trained on your calibration succeed {100 * contrast['diff']:+.0f} points "
                 f"({100 * contrast['ci'][0]:+.0f} to {100 * contrast['ci'][1]:+.0f}, 95% CI; "
                 f"{contrast['n_pairs']} paired episodes) compared with policies trained on default physics. "
                 + ("That difference is clear of zero." if sig else
                    "That interval includes zero: on this task, this experiment can't tell the two apart.")
                 + " Training-seed variation is real in reinforcement learning; per-seed numbers are in "
                   "`results.json`, and more seeds (`--seeds`) narrow the interval.\n")
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
    ek = {"spec": os.path.abspath(args.spec), "task": args.task, "max_episode_steps": args.max_episode_steps,
          "current": current, "thrusters": args.thrusters}
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
        policies = [{"policy": os.path.abspath(args.policy), "physics": "your policy", "seed": 0, "train_s": 0}]
    else:
        conditions = ["calibrated"] + (["generic"] if args.compare_generic else [])
        jobs = [{"physics": c, "params": params, "env": ek, "seed": s, "timesteps": args.timesteps,
                 "out": os.path.join(os.path.abspath(args.out), "policies", f"{c}_seed{s}")}
                for c in conditions for s in range(1, args.seeds + 1)]
        print(f"Training {len(jobs)} policies ({args.timesteps:,} steps each, {min(workers, len(jobs))} at a time)...",
              flush=True)
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            for r in ex.map(train_job, jobs):
                print(f"  trained {os.path.basename(r['policy'])} in {r['train_s'] / 60:.1f} min", flush=True)
                policies.append(r)

    episode_seeds = list(range(1000, 1000 + args.episodes))
    label = {"calibrated": "trained on your calibration", "generic": "trained on default physics",
             "your policy": "your policy"}
    eval_jobs = [{"controller": "policy", "policy": p["policy"], "physics": p["physics"], "seed": p["seed"],
                  "label": label[p["physics"]], "params": params, "env": ek, "episode_seeds": episode_seeds}
                 for p in policies]
    eval_jobs += [{"controller": c, "label": lab, "params": params, "env": ek, "episode_seeds": episode_seeds}
                  for c, lab in (("hand_written", "hand-written controller"), ("random", "random commands"))]
    print(f"Evaluating {len(eval_jobs)} controllers on {len(episode_seeds)} episodes each...", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=min(workers, len(eval_jobs))) as ex:
        for r in ex.map(evaluate_job, eval_jobs):
            rows += r
    order = list(dict.fromkeys(j["label"] for j in eval_jobs))
    summaries = [summarize(rows, lab) for lab in order]
    contrast = paired_difference(rows, "calibrated", "generic") if args.compare_generic and not args.policy else None
    per_seed = [{"physics": p["physics"], "seed": p["seed"],
                 "success_rate": float(np.mean([x["success"] for x in rows
                                                if x["physics"] == p["physics"] and x["seed"] == p["seed"]]))}
                for p in policies]

    with open(os.path.join(args.out, "episodes.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    meta = {"name": args.name or os.path.basename(os.path.abspath(args.out)), "task": args.task,
            "params": params, "spec": os.path.abspath(args.spec), "current": current,
            "current_described": CurrentConfig.from_dict(current).describe(teammates=False) if current else None,
            "thrusters": args.thrusters, "thrusters_described": thr_desc, "seeds": args.seeds,
            "timesteps": args.timesteps, "n_episodes": args.episodes, "max_episode_steps": args.max_episode_steps,
            "compare_generic": bool(args.compare_generic and not args.policy), "policy": args.policy,
            "command": "python " + " ".join([os.path.basename(sys.argv[0])] + sys.argv[1:]),
            "python": platform.python_version(), "elapsed_s": round(time.time() - t0, 1)}
    results = {"meta": meta, "summary": summaries, "calibrated_vs_default": contrast, "per_seed": per_seed,
               "policies": policies}
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_report(os.path.join(args.out, "report.md"), meta, summaries, contrast)
    print(f"\nReport: {os.path.join(args.out, 'report.md')}")
    for s in summaries:
        print(f"  {s['controller']:<28} success {100 * s['success_rate']:5.1f}% "
              f"({100 * s['ci'][0]:.0f}-{100 * s['ci'][1]:.0f}%)")
    if contrast:
        print(f"  calibrated - default training: {100 * contrast['diff']:+.1f} points "
              f"({100 * contrast['ci'][0]:+.1f} to {100 * contrast['ci'][1]:+.1f})")
    return results


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--params", default=os.path.join(HERE, "calibrated_parameters_fleet_truth.yaml"),
                    help="your calibrated_parameters.yaml (default: the shipped BlueROV2 calibration)")
    ap.add_argument("--spec", default=os.path.join(HERE, "vehicle_spec.yaml"), help="vehicle description")
    ap.add_argument("--task", choices=TASKS, default="waypoint")
    ap.add_argument("--seeds", type=int, default=2, help="training seeds per condition")
    ap.add_argument("--timesteps", type=int, default=150_000, help="training steps per policy")
    ap.add_argument("--episodes", type=int, default=100, help="evaluation episodes per controller")
    ap.add_argument("--max-episode-steps", type=int, default=200)
    ap.add_argument("--compare-generic", action="store_true",
                    help="also train on the default (uncalibrated) physics and compare")
    ap.add_argument("--policy", help="evaluate this trained PPO policy (.zip) instead of training new ones")
    ap.add_argument("--current-speed", type=float, nargs="+", metavar="MPS",
                    help="water current speed in m/s, or MIN MAX drawn per episode")
    ap.add_argument("--current-direction", type=float, nargs="+", default=[0.0, 360.0], metavar="DEG")
    ap.add_argument("--current-speed-std", type=float, default=0.0)
    ap.add_argument("--current-direction-std", type=float, default=0.0)
    ap.add_argument("--current-time-scale", type=float, default=30.0)
    ap.add_argument("--thrusters", help="limit commands to what the thrusters can deliver: bluerov2_heavy, or a YAML "
                                        "file with a thrusters block (an ingest spec works)")
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
