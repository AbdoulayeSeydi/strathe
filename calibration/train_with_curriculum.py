"""
train_with_curriculum.py — trains PPO with a staged curriculum: the
environment's randomized-start difficulty ceiling (curriculum_max_difficulty,
see two_agent_formation_env.py) is raised in stages over the course of
training, using CurriculumCallback below.

Stage thresholds are DATA-DRIVEN, not arbitrary: they come directly from
classify_state_difficulty.py's heuristic-measured difficulty table (the
same heuristic controller confirmed 32/32 on the diagnostic grid).

*** UNVERIFIED — this sandbox has no network access and could not
install stable_baselines3, so NONE of the SB3-specific code below
(BaseCallback, self.training_env, self.num_timesteps, .envs vs
.set_attr) has been run against a real model, unlike every
non-RL script in this investigation. The API usage is believed correct
for SB3 2.9.0 (your confirmed installed version) based on that
version's documented callback interface, but treat this with the same
caution as check_value_calibration.py: run it and report the exact
traceback if anything fails, rather than assuming it's correct. ***

This is a NEW, self-contained script rather than a modification of your
existing train_formation_baseline.py, since I don't have that file's
source -- only its CLI and output format, inferred from your pasted
runs. After this trains and saves a model, use your EXISTING, already-
tested evaluate_recovery_grid.py to assess it -- I'm not attempting to
replicate train_formation_baseline.py's Gate2-style reporting, since
guessing at its internals risks a subtly wrong recreation.
"""
import argparse

import numpy as np

from two_agent_formation_env import TwoAgentFormationEnv
from comm_channel import PacketLossConfig, BandwidthConfig
from difficulty_lookup import DifficultyLookup


def build_curriculum_env(difficulty_csv_path: str, calibrated_params_path: str = "calibrated_parameters_a1.yaml",
                          packet_loss_config: PacketLossConfig = None, bandwidth_config: BandwidthConfig = None):
    """
    packet_loss_config/bandwidth_config: OPTIONAL overrides for A1.10's
    comm-severity sweep. When both are None (default), falls back to the
    exact zero-loss/unlimited-bandwidth config every A1.9 run used --
    preserving v5's exact training behavior for any caller that doesn't
    ask for degraded comm.
    """
    lookup = DifficultyLookup(difficulty_csv_path)
    env = TwoAgentFormationEnv(
        calibrated_params_path=calibrated_params_path,
        packet_loss_config=packet_loss_config or PacketLossConfig(0.0, 1000.0, 0.0, 0.0),
        bandwidth_config=bandwidth_config or BandwidthConfig(bytes_per_tick=1e9),
        randomize_initial_state=True,
        curriculum_lookup=lookup,
        anchor_probability=0.0,  # Stage 1 starts at exactly 0% -- the callback ramps this per stage
    )
    return env, lookup


def make_curriculum_callback(lookup: DifficultyLookup, total_timesteps: int,
                              stage_fractions=(0.25, 0.5, 0.75, 1.0),
                              anchor_stage_probabilities=(0.0, 0.0, 0.05, 0.10), verbose: int = 1):
    """
    stage_fractions default reverted to the ORIGINAL even 25%-per-stage
    split -- the rebalanced (0.15,0.35,0.6,1.0) timing tried once (v3) was
    strictly worse on every metric (9.4% success vs 31.25%, reintroduced a
    collision), so this reverts that change; anchor_stage_probabilities is
    the actual v5 experiment.

    anchor_stage_probabilities: per-stage anchor_probability, ramped rather
    than flat. v4's flat 15%-from-stage-1 fixed the original (d0=4.0,
    v0=0.0) scenario (F -> R, 17/20 hold steps) but caused a severe
    regression -- 40.6% collision rate (vs 0% for v2), concentrated at
    d0=2.0-3.0 -- because injecting a mid-difficulty fixed point into
    Stage 1 diluted exactly the low-risk, safety-establishing exposure
    that stage was supposed to provide. This ramp keeps Stage 1 (and 2)
    at EXACTLY 0% anchor -- no exceptions -- so v2's safety behavior can
    be established first, only introducing the anchor once the curriculum
    threshold has already earned that difficulty neighborhood (stage 3),
    and capping at 10% (not 15%) rather than assuming more exposure is
    strictly better.
    """
    from stable_baselines3.common.callbacks import BaseCallback

    class CurriculumCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=verbose)
            min_d, max_d = lookup.min_difficulty, lookup.max_difficulty
            self.stage_thresholds = [min_d + (max_d - min_d) * f for f in stage_fractions]
            self._current_stage = -1

        def _on_step(self) -> bool:
            progress = self.num_timesteps / total_timesteps
            stage = len(stage_fractions) - 1
            for i, frac in enumerate(stage_fractions):
                if progress <= frac:
                    stage = i
                    break
            if stage != self._current_stage:
                self._current_stage = stage
                new_threshold = self.stage_thresholds[stage]
                new_anchor_prob = anchor_stage_probabilities[stage]
                try:
                    # DummyVecEnv exposes the underlying envs directly.
                    for e in self.training_env.envs:
                        e.curriculum_max_difficulty = new_threshold
                        e.anchor_probability = new_anchor_prob
                except AttributeError:
                    # Other VecEnv implementations (e.g. SubprocVecEnv) --
                    # set_attr is the documented cross-implementation path.
                    self.training_env.set_attr("curriculum_max_difficulty", new_threshold)
                    self.training_env.set_attr("anchor_probability", new_anchor_prob)
                if self.verbose:
                    print(f"[curriculum] step {self.num_timesteps}/{total_timesteps} "
                          f"({progress*100:.0f}%): stage {stage+1}/{len(stage_fractions)}, "
                          f"max_difficulty={new_threshold:.3f} "
                          f"(full range is [{lookup.min_difficulty:.1f}, {lookup.max_difficulty:.1f}]), "
                          f"anchor_probability={new_anchor_prob:.2f}")
            return True

    return CurriculumCallback()


def main(difficulty_csv_path: str, total_timesteps: int, run_name: str,
         calibrated_params_path: str = "calibrated_parameters_a1.yaml",
         stage_fractions=(0.25, 0.5, 0.75, 1.0),
         anchor_stage_probabilities=(0.0, 0.0, 0.05, 0.10),
         packet_loss_prob: float = 0.0, bandwidth_bytes_per_tick: float = 1e9):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import DummyVecEnv
    from stable_baselines3.common.monitor import Monitor

    packet_loss_config = PacketLossConfig(0.0, 1000.0, packet_loss_prob, packet_loss_prob)
    bandwidth_config = BandwidthConfig(bytes_per_tick=bandwidth_bytes_per_tick)
    env, lookup = build_curriculum_env(difficulty_csv_path, calibrated_params_path,
                                        packet_loss_config, bandwidth_config)
    env = Monitor(env, filename=f"monitor_{run_name}.csv")
    vec_env = DummyVecEnv([lambda: env])

    print(f"Comm severity: packet_loss_prob={packet_loss_prob} (constant, both directions), "
          f"bandwidth={bandwidth_bytes_per_tick:.0f} bytes/tick")
    print(f"Difficulty range from {difficulty_csv_path}: "
          f"[{lookup.min_difficulty:.1f}, {lookup.max_difficulty:.1f}]")
    print(f"Stage boundaries (fraction of training): {stage_fractions}")
    print(f"Anchor probability ramp per stage: {anchor_stage_probabilities} "
          f"(0% during easy stages, only introduced once the curriculum threshold "
          f"reaches the anchor's own neighborhood)")
    print(f"Stage 1 starts at max_difficulty={lookup.min_difficulty:.1f} "
          f"(easiest states only) and widens at each boundary.\n")

    model = PPO("MlpPolicy", vec_env, verbose=1)
    callback = make_curriculum_callback(lookup, total_timesteps, stage_fractions=stage_fractions,
                                         anchor_stage_probabilities=anchor_stage_probabilities)
    model.learn(total_timesteps=total_timesteps, callback=callback)

    model.save(f"formation_hold_ppo_{run_name}")
    print(f"\nSaved trained model to formation_hold_ppo_{run_name}.zip")
    print(f"Saved per-episode training log to monitor_{run_name}.csv")
    print(f"\nNext steps -- evaluate with your existing tooling:")
    print(f"  python3 evaluate_recovery_grid.py --model-path formation_hold_ppo_{run_name} --n-seeds 5")
    print(f"  python3 check_original_scenario.py --model-path formation_hold_ppo_{run_name}")
    print(f"  python3 evaluate_recovery_grid.py --model-path formation_hold_ppo_{run_name} "
          f"--separations 3.5 3.75 4.0 4.25 4.5 --velocities -0.15 0 0.15 --n-seeds 5 "
          f"--csv-path neighborhood_{run_name}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--difficulty-csv", required=True,
                         help="Output of classify_state_difficulty.py")
    parser.add_argument("--total-timesteps", type=int, default=400000)
    parser.add_argument("--run-name", default="curriculum")
    parser.add_argument("--calibrated-params", default="calibrated_parameters_a1.yaml")
    parser.add_argument("--stage-fractions", default="0.25,0.5,0.75,1.0",
                         help="Comma-separated cumulative fractions of total_timesteps "
                              "at which each stage begins.")
    parser.add_argument("--anchor-stage-probabilities", default="0.0,0.0,0.05,0.10",
                         help="Comma-separated anchor_probability for each stage, ramped "
                              "rather than flat. v4's flat 15%% from stage 1 fixed the "
                              "original (d0=4.0,v0=0.0) scenario (F->R, 17/20 hold steps) "
                              "but caused 40.6%% collisions (vs 0%% for v2), concentrated "
                              "at d0=2.0-3.0 -- diluting exactly the safety-establishing "
                              "exposure Stage 1 needs. Default keeps stages 1-2 at exactly "
                              "0%%, only introduces the anchor at stage 3, caps at 10%%.")
    parser.add_argument("--packet-loss-prob", type=float, default=0.0,
                         help="A1.10: constant packet-loss probability, both directions "
                              "(e.g. 0.3 for L2's ~30%% severity level). Default 0.0 "
                              "preserves A1.9's exact zero-degradation training.")
    parser.add_argument("--bandwidth-bytes-per-tick", type=float, default=1e9,
                         help="A1.10: per-agent, per-tick byte budget. A single position "
                              "beacon is 24 bytes; set below that to make bandwidth itself "
                              "binding (as in the proposed L3 level), otherwise it's "
                              "effectively redundant with packet-loss-prob in this "
                              "two-agent, one-message-type scenario. Default 1e9 (effectively "
                              "unlimited) preserves A1.9's exact training.")
    args = parser.parse_args()
    stage_fractions = tuple(float(x) for x in args.stage_fractions.split(","))
    anchor_stage_probabilities = tuple(float(x) for x in args.anchor_stage_probabilities.split(","))
    main(args.difficulty_csv, args.total_timesteps, args.run_name,
         args.calibrated_params, stage_fractions, anchor_stage_probabilities,
         args.packet_loss_prob, args.bandwidth_bytes_per_tick)