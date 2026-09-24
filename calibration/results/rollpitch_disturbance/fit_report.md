# Roll/pitch disturbance fit

Parameters: `calibrated_parameters_fleet_truth.yaml`. Fitted on 60 training windows (60 s), checked on 130 held-out windows. Fitted: roll sigma 0.920 N*m, tau 0.42 s; pitch sigma 1.080 N*m, tau 0.30 s.

| Held-out statistic | Real | No disturbance | Fitted disturbance | Fitted, MuJoCo (real, same windows) |
|---|---|---|---|---|
| roll rate std (rad/s) | 0.144 | 0.068 | 0.102 | 0.105 (0.148) |
| roll angle std (deg) | 11.3 | 21.4 | 23.8 | 20.9 (11.3) |
| roll rate autocorrelation at 0.2 s | +0.79 | +0.62 | +0.56 | +0.57 (+0.80) |
| roll rate autocorrelation at 0.5 s | +0.37 | +0.49 | +0.31 | +0.31 (+0.40) |
| roll rate autocorrelation at 1.0 s | +0.09 | +0.34 | +0.12 | +0.12 (+0.09) |
| pitch rate std (rad/s) | 0.202 | 0.114 | 0.177 | 0.176 (0.199) |
| pitch angle std (deg) | 12.3 | 17.3 | 19.3 | 18.7 (12.3) |
| pitch rate autocorrelation at 0.2 s | +0.80 | +0.84 | +0.62 | +0.61 (+0.80) |
| pitch rate autocorrelation at 0.5 s | +0.27 | +0.60 | +0.31 | +0.29 (+0.25) |
| pitch rate autocorrelation at 1.0 s | -0.10 | +0.34 | +0.11 | +0.07 (-0.11) |

Diverged windows: held-out, no disturbance: 0, held-out, fitted disturbance: 0, held-out, fitted disturbance, MuJoCo: 0

Elapsed 1342 s.
