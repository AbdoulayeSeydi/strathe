# Stress test: clean/noisy simulation, buoyancy, SAC/TD3 (PROJECT_STATUS.md #58)

2026-09-27 14:48 · 14 min

## A. Two vehicles: 410/410 combinations ran cleanly

Randomized over comm level, beacon interval, buoyancy (−25..+25 N, fixed or a range, known or not), sensor noise (off, tank, typical ×1/×3/×10), roll/pitch disturbance (×0/1/3), current (0-0.5 m/s, fluctuating), thruster limits, engine, policy (PPO, noise-trained PPO, SAC, TD3, random), belief model and start state (0.6-8 m apart, ±1 m/s), plus the worst corner of all of them on every policy and engine.

- by policy: ppo 79/79, ppo_noisy 79/79, random 77/77, sac 84/84, td3 91/91
- by backend: mujoco 92/92, stand_in 318/318
- by severity: L0 73/73, L1 83/83, L2 88/88, L3a 70/70, L3b 96/96
- by noise: ('tank', 1.0) 69/69, ('typical', 1.0) 90/90, ('typical', 10.0) 82/82, ('typical', 3.0) 81/81, None 88/88
- by buoyancy: -10.0 39/39, -25.0 45/45, -3.0 52/52, 0.0 53/53, 10.0 47/47, 25.0 39/39, 3.0 47/47, [-15.0, -5.0] 8/8, [-2.0, 8.0] 15/15, [-30.0, -20.0] 10/10, [-5.0, 5.0] 14/14, [-8.0, 2.0] 10/10, [20.0, 30.0] 11/11, [5.0, 15.0] 20/20
- outcomes: {'formation_achieved': 193, 'timeout': 205, 'collision': 12}; largest navigation error 10.3 m

## B. One vehicle: 400/400 combinations ran cleanly

Randomized over the four tasks, controller (PPO, SAC, TD3 policies, hand-written, random), sensor noise, buoyancy, disturbance, current, thruster limits and engine. Every step checks that the reported distance is the true one and that success is never declared from the believed position.

- by task: docking 89/89, obstacle_avoidance 107/107, station_keeping 107/107, waypoint 97/97
- by controller: hand 81/81, ppo 78/78, random 79/79, sac 87/87, td3 75/75
- by backend: mujoco 105/105, stand_in 295/295
- by noise: ('tank', 1.0) 76/76, ('typical', 1.0) 101/101, ('typical', 10.0) 75/75, ('typical', 3.0) 75/75, None 73/73
- by buoyancy: -10.0 50/50, -25.0 62/62, -3.0 50/50, 0.0 75/75, 10.0 63/63, 25.0 45/45, 3.0 55/55

## C. Measured against what was configured: 23/23 passed

| Check | Result | Details |
|---|---|---|
| DVL white noise std | pass | {'configured': 0.01, 'measured': 0.01, 'tolerance': '±10%'} |
| gyro white noise std | pass | {'configured': 0.01, 'measured': 0.01, 'tolerance': '±10%'} |
| heading white noise std | pass | {'configured': 0.035, 'measured': 0.0349, 'tolerance': '±10%'} |
| DVL bias std across dives | pass | {'configured': 0.005, 'measured': 0.0051, 'tolerance': '±10%'} |
| DVL dropout rate | pass | {'configured': 0.02, 'measured': 0.0204, 'tolerance': '±25%'} |
| terminal vertical speed, -20 N, stand_in | pass | {'predicted_mps': 0.1543, 'simulated_mps': 0.1543} |
| terminal vertical speed, -20 N, mujoco | pass | {'predicted_mps': 0.1543, 'simulated_mps': 0.1543} |
| terminal vertical speed, -5 N, stand_in | pass | {'predicted_mps': 0.0379, 'simulated_mps': 0.0379} |
| terminal vertical speed, -5 N, mujoco | pass | {'predicted_mps': 0.0379, 'simulated_mps': 0.0379} |
| terminal vertical speed, -1 N, stand_in | pass | {'predicted_mps': 0.0, 'simulated_mps': 0.0} |
| terminal vertical speed, -1 N, mujoco | pass | {'predicted_mps': 0.0, 'simulated_mps': 0.0} |
| terminal vertical speed, +1 N, stand_in | pass | {'predicted_mps': -0.0194, 'simulated_mps': -0.0194} |
| terminal vertical speed, +1 N, mujoco | pass | {'predicted_mps': -0.0194, 'simulated_mps': -0.0194} |
| terminal vertical speed, +5 N, stand_in | pass | {'predicted_mps': -0.0554, 'simulated_mps': -0.0554} |
| terminal vertical speed, +5 N, mujoco | pass | {'predicted_mps': -0.0554, 'simulated_mps': -0.0554} |
| terminal vertical speed, +20 N, stand_in | pass | {'predicted_mps': -0.1678, 'simulated_mps': -0.1678} |
| terminal vertical speed, +20 N, mujoco | pass | {'predicted_mps': -0.1678, 'simulated_mps': -0.1678} |
| packet losses identical with noise off and on (same actions) | pass | {} |
| true motion identical with noise off and on (same actions, same disturbances) | pass | {} |
| observations differ with noise on | pass | {} |
| noisy rollout deterministic (stand_in) | pass | {} |
| noisy rollout deterministic (mujoco) | pass | {} |
| clean simulation unchanged: identical episodes to the code before this change | pass | {'fleet_episodes': 6616, 'fleet_differ': 0, 'with_older_options_on': 216, 'arena_runs': 16, 'arena_differ': 0} |
