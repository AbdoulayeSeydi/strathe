#!/usr/bin/env python3
"""
clean_regression_check.py -- prove that clean simulation is unchanged: rerun a
fixed set of clean episodes and compare them, field by field, with the same
episodes run on the code before sensor noise, buoyancy and SAC/TD3 fleet
training were added (commit 6c8a2ef, PROJECT_STATUS.md #58).

    python clean_regression_check.py            # compare (about 10 min on 6 cores)
    python clean_regression_check.py --write X  # write a new reference to X

The episodes: the shipped default experiment's cells for two policy seeds
(6,400 episodes, every belief and comm condition), the same cells with the
roll/pitch disturbance, a fluctuating current and thruster limits on, a few on
MuJoCo, and single-vehicle runs of every task on both engines with and without
those options (every observation and reward hashed).
"""
import hashlib, json, os, sys
import numpy as np
REFERENCE = "results/stress_noise_algorithms/clean_reference_6c8a2ef.json"
from build_two_agent_cores import load_calibrated_parameters
from optimizer import initial_parameters, load_vehicle_spec
from fleet_experiment_schema import load_fleet_experiment
from run_fleet_experiment import build_cells
from fleet_eval import run_cells
from disturbance import DEFAULT_FIT_PATH, load_disturbance
from current import CurrentConfig
from fleet_experiment_schema import resolve_thrusters
from arena import Arena
import dataclasses


def run() -> dict:
    res = {}
    cfg = load_fleet_experiment("default_experiment.yaml")
    spec = load_vehicle_spec(cfg.vehicle_spec)
    truth = load_calibrated_parameters(cfg.truth_params, quiet=True)
    cells, _ = build_cells(cfg, truth, initial_parameters(spec), spec.vehicle)
    cells = [c for c in cells if c.policy_path.endswith(("seed1", "seed2"))]
    # the same cells again with everything that existed before on: disturbance, current, thrusters, MuJoCo
    extra = []
    for c in cells[:8]:
        fc = dataclasses.replace(c.fleet_config, disturbance=load_disturbance(DEFAULT_FIT_PATH, 1.5),
                                 current=CurrentConfig(speed_mps=(0.0, 0.3), direction_deg=(0.0, 360.0), speed_std_mps=0.03),
                                 thrusters=resolve_thrusters("bluerov2_heavy"))
        extra.append(dataclasses.replace(c, fleet_config=fc, separations=c.separations[:4], closing_speeds=c.closing_speeds[:3],
                                         comm_seeds=[1, 2]))
    for c in cells[:4]:
        fc = dataclasses.replace(c.fleet_config, dynamics_backend="mujoco")
        extra.append(dataclasses.replace(c, fleet_config=fc, separations=c.separations[:3], closing_speeds=c.closing_speeds[:2],
                                         comm_seeds=[1]))
    rows = run_cells(cells + extra, workers=6)
    keep = ("policy", "comm", "belief", "backend", "episode_key", "steps", "terminal_reason", "success", "collision",
            "mean_belief_error_m", "p90_belief_error_m", "delivery_rate", "mean_abs_separation_error_m")
    res["fleet"] = [{k: (None if isinstance(r[k], float) and np.isnan(r[k]) else r[k]) for k in keep}
                    | {"variant": i >= sum(len(c.separations) * len(c.closing_speeds) * len(c.comm_seeds) for c in cells)}
                    for i, r in enumerate(rows)]
    # single vehicle: every task, both engines, with and without the older options; hash every observation
    from evaluate_policy_arena import proportional_controller
    arena = []
    for task in ("waypoint", "station_keeping", "docking", "obstacle_avoidance"):
        for backend in ("stand_in", "mujoco"):
            for opts in (0, 1):
                kw = {} if not opts else dict(disturbance=load_disturbance(DEFAULT_FIT_PATH, 2.0),
                                              current=CurrentConfig(speed_mps=0.2, direction_deg=(0.0, 360.0)),
                                              thrusters=resolve_thrusters("bluerov2_heavy"))
                env = Arena(truth, spec.vehicle, task=task, backend=backend, **kw)
                h = hashlib.sha256()
                for ep in range(3):
                    o, _ = env.reset(seed=100 + ep)
                    h.update(o.tobytes())
                    done = False
                    while not done:
                        o, r, t, tr, info = env.step(proportional_controller(o) if task != "obstacle_avoidance"
                                                     else np.clip(o[6:9] / 3, -1, 1).tolist() + [0, 0, 0])
                        h.update(o.tobytes()); h.update(np.float64(r).tobytes())
                        done = t or tr
                arena.append({"task": task, "backend": backend, "opts": opts, "hash": h.hexdigest()})
    res["arena"] = arena
    return res


def compare(ref: dict, new: dict) -> dict:
    fleet_diff = sum(1 for a, b in zip(ref["fleet"], new["fleet"]) if a != b) + abs(len(ref["fleet"]) - len(new["fleet"]))
    arena_diff = sum(1 for a, b in zip(ref["arena"], new["arena"]) if a != b) + abs(len(ref["arena"]) - len(new["arena"]))
    return {"fleet_episodes": len(ref["fleet"]), "fleet_differ": fleet_diff,
            "with_older_options_on": sum(r["variant"] for r in ref["fleet"]),
            "arena_runs": len(ref["arena"]), "arena_differ": arena_diff, "ok": fleet_diff == 0 and arena_diff == 0}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--write", metavar="JSON", help="write a new reference instead of comparing")
    args = ap.parse_args(argv)
    res = run()
    if args.write:
        with open(args.write, "w") as f:
            json.dump(res, f, separators=(",", ":"))
        print(f"wrote {args.write}")
        return 0
    with open(REFERENCE) as f:
        ref = json.load(f)
    c = compare(ref, res)
    print(json.dumps(c))
    return 0 if c["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
