"""
trace_episode_detailed.py — full state instrumentation for one deterministic
episode. Captures per-agent force/torque, position, orientation, body- and
world-frame velocity, the separation VECTOR (not just its magnitude) and
its derivative, written to CSV for full offline analysis, plus a condensed
console view focused on pitch and pitch-torque specifically.

Captures the RESCALED, real-Newton tau values by intercepting
TwoAgentOrchestrator.step() directly — the exact force/torque
ArenaCore.step() actually receives, not a recomputation from the raw
[-1,1] action that could silently drift from what the physics saw.

Built specifically to test: does the calibrated agent's known surge-pitch
mass coupling (Xq_dot in the CURRENT calibrated_parameters_a1.yaml —
this is about what actually drove THIS trace's observed behavior, not a
re-litigation of whether that specific fitted value is stable across
calibration seeds, which is a separate, already-settled question) show
up connecting surge thrust to pitch growth to divergence, the exact
causal chain hypothesized after trace_episode.py's aggregate result.
"""
import argparse
import csv

import numpy as np

from two_agent_formation_env import TwoAgentFormationEnv
from comm_channel import PacketLossConfig, BandwidthConfig
from two_agent_orchestrator import OrchestratorConfig

FIELDNAMES = [
    "step", "separation", "dsep_dt", "consec", "reward", "cum_reward", "terminal_reason",
    "gen_Fx", "gen_Fy", "gen_Fz", "gen_Tx", "gen_Ty", "gen_Tz",
    "cal_Fx", "cal_Fy", "cal_Fz", "cal_Tx", "cal_Ty", "cal_Tz",
    "gen_pos_x", "gen_pos_y", "gen_pos_z",
    "cal_pos_x", "cal_pos_y", "cal_pos_z",
    "rel_pos_x", "rel_pos_y", "rel_pos_z",
    "gen_roll", "gen_pitch", "gen_yaw",
    "cal_roll", "cal_pitch", "cal_yaw",
    "gen_u", "gen_v", "gen_w", "gen_p", "gen_q", "gen_r",
    "cal_u", "cal_v", "cal_w", "cal_p", "cal_q", "cal_r",
    "gen_world_vx", "gen_world_vy", "gen_world_vz",
    "cal_world_vx", "cal_world_vy", "cal_world_vz",
    "rel_world_vx", "rel_world_vy", "rel_world_vz",
]


def build_env(max_episode_steps=200):
    return TwoAgentFormationEnv(
        calibrated_params_path="calibrated_parameters_a1.yaml",
        packet_loss_config=PacketLossConfig(0.0, 1000.0, 0.0, 0.0),
        bandwidth_config=BandwidthConfig(bytes_per_tick=1e9),
        orchestrator_config=OrchestratorConfig(max_episode_steps=max_episode_steps),
    )


def run_episode(policy_fn, seed: int, print_every: int, csv_path: str, n_steps: int = 200):
    """policy_fn(obs) -> action. Split out from main() so it can be tested
    with a random-action policy_fn (no SB3/model file needed) before
    trusting it with a real trained model."""
    env = build_env(max_episode_steps=n_steps)

    captured_tau = {}
    original_step = env._orchestrator.step

    def spy_step(action_generic, action_calibrated, **kwargs):
        captured_tau["generic"] = np.array(action_generic, dtype=float)
        captured_tau["calibrated"] = np.array(action_calibrated, dtype=float)
        return original_step(action_generic, action_calibrated, **kwargs)

    env._orchestrator.step = spy_step

    obs, _ = env.reset(seed=seed)

    rows = []
    cum_reward = 0.0
    prev_sep = None
    dt = env._orchestrator.config.dt

    print(f"{'step':>5} {'sep':>8} {'dsep/dt':>9} {'consec':>7} "
          f"{'cal_pitch':>10} {'cal_Ty(pitch trq)':>18} {'cal_Fx':>8} {'reward':>9}")

    last_step = 0
    for step in range(1, n_steps + 1):
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward
        last_step = step

        core_g = env._orchestrator.core_generic
        core_c = env._orchestrator.core_calibrated

        sep = info["true_separation"]
        dsep_dt = (sep - prev_sep) / dt if prev_sep is not None else 0.0
        prev_sep = sep
        consec = info["consecutive_steps_in_tolerance"]
        reason = info.get("terminal_reason", "")

        tau_g = captured_tau["generic"]
        tau_c = captured_tau["calibrated"]

        g_roll, g_pitch, g_yaw = core_g.state.orientation.as_euler("xyz")
        c_roll, c_pitch, c_yaw = core_c.state.orientation.as_euler("xyz")
        g_world_v = core_g.state.orientation.apply(core_g.state.nu[:3])
        c_world_v = core_c.state.orientation.apply(core_c.state.nu[:3])
        rel_pos = core_c.state.position - core_g.state.position
        rel_world_v = c_world_v - g_world_v

        row = {
            "step": step, "separation": sep, "dsep_dt": dsep_dt, "consec": consec,
            "reward": reward, "cum_reward": cum_reward, "terminal_reason": reason,
            "gen_Fx": tau_g[0], "gen_Fy": tau_g[1], "gen_Fz": tau_g[2],
            "gen_Tx": tau_g[3], "gen_Ty": tau_g[4], "gen_Tz": tau_g[5],
            "cal_Fx": tau_c[0], "cal_Fy": tau_c[1], "cal_Fz": tau_c[2],
            "cal_Tx": tau_c[3], "cal_Ty": tau_c[4], "cal_Tz": tau_c[5],
            "gen_pos_x": core_g.state.position[0], "gen_pos_y": core_g.state.position[1],
            "gen_pos_z": core_g.state.position[2],
            "cal_pos_x": core_c.state.position[0], "cal_pos_y": core_c.state.position[1],
            "cal_pos_z": core_c.state.position[2],
            "rel_pos_x": rel_pos[0], "rel_pos_y": rel_pos[1], "rel_pos_z": rel_pos[2],
            "gen_roll": g_roll, "gen_pitch": g_pitch, "gen_yaw": g_yaw,
            "cal_roll": c_roll, "cal_pitch": c_pitch, "cal_yaw": c_yaw,
            "gen_u": core_g.state.nu[0], "gen_v": core_g.state.nu[1], "gen_w": core_g.state.nu[2],
            "gen_p": core_g.state.nu[3], "gen_q": core_g.state.nu[4], "gen_r": core_g.state.nu[5],
            "cal_u": core_c.state.nu[0], "cal_v": core_c.state.nu[1], "cal_w": core_c.state.nu[2],
            "cal_p": core_c.state.nu[3], "cal_q": core_c.state.nu[4], "cal_r": core_c.state.nu[5],
            "gen_world_vx": g_world_v[0], "gen_world_vy": g_world_v[1], "gen_world_vz": g_world_v[2],
            "cal_world_vx": c_world_v[0], "cal_world_vy": c_world_v[1], "cal_world_vz": c_world_v[2],
            "rel_world_vx": rel_world_v[0], "rel_world_vy": rel_world_v[1], "rel_world_vz": rel_world_v[2],
        }
        rows.append(row)

        if step % print_every == 0 or terminated or truncated or step <= 5:
            print(f"{step:>5} {sep:>8.3f} {dsep_dt:>9.3f} {consec:>7} "
                  f"{c_pitch:>10.4f} {tau_c[4]:>18.4f} {tau_c[0]:>8.3f} {reward:>9.3f}")

        if terminated or truncated:
            break

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nFull per-step state ({last_step} steps, {len(FIELDNAMES)} columns) written to {csv_path}")
    return rows


def main(model_path: str, seed: int, print_every: int, csv_path: str):
    from stable_baselines3 import PPO
    model = PPO.load(model_path)

    def policy_fn(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    run_episode(policy_fn, seed, print_every, csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=5)
    parser.add_argument("--csv-path", default="trace_detailed.csv")
    args = parser.parse_args()
    main(args.model_path, args.seed, args.print_every, args.csv_path)