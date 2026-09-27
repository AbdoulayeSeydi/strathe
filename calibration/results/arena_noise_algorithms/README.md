# Single-vehicle runs for PROJECT_STATUS.md #58

Reach a point (`waypoint`) on the reference BlueROV2 (`calibrated_parameters_fleet_truth.yaml`), 100 test
episodes per controller, the same episode seeds in every run. Commands, from `calibration/`:

| Folder | Command |
|---|---|
| `clean_waypoint_ppo`, `_sac`, `_td3` | `python run_arena_experiment.py --task waypoint --algorithm <ppo\|sac\|td3> --seeds 2 --timesteps 150000 --episodes 100 --out ...` |
| `noisy_waypoint` | `python run_arena_experiment.py --task waypoint --mode noisy --compare-clean-training --seeds 2 --timesteps 300000 --episodes 100 --backend both --out ...` |
| `algorithms_under_noise/<alg>_seed<k>` | `python run_arena_experiment.py --task waypoint --policy clean_waypoint_<alg>/policies/<alg>_calibrated_seed<k>.zip --mode noisy --episodes 100 --out ...` |
| `buoyancy_sweep/seed<k>_b<N>` | `python run_arena_experiment.py --task waypoint --policy clean_waypoint_ppo/policies/ppo_calibrated_seed<k>.zip --buoyancy <N> --episodes 100 --out ...` |

The SAC and TD3 policy files (about 3-6 MB each) aren't kept; the first command trains equivalent ones
(seeded, though not guaranteed bit-identical across machines).
