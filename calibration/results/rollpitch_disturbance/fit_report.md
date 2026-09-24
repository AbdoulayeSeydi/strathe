# Roll/pitch disturbance fit

Parameters: `calibrated_parameters_fleet_truth.yaml`. Fitted on 60 training windows (60 s), checked on 130 held-out windows. Fitted: roll sigma 1.600 N*m, tau 0.70 s; pitch sigma 2.160 N*m, tau 1.00 s.

| Held-out statistic | Real | No disturbance | Fitted disturbance | Fitted, MuJoCo (real, same windows) |
|---|---|---|---|---|
| roll rate std (rad/s) | 0.144 | 0.066 | 0.135 | 0.136 (0.148) |
| roll angle std (deg) | 11.3 | 4.1 | 9.5 | 9.3 (11.3) |
| roll rate autocorrelation at 0.2 s | +0.79 | +0.59 | +0.64 | +0.65 (+0.80) |
| roll rate autocorrelation at 0.5 s | +0.37 | +0.46 | +0.37 | +0.37 (+0.40) |
| roll rate autocorrelation at 1.0 s | +0.09 | +0.29 | +0.11 | +0.12 (+0.09) |
| pitch rate std (rad/s) | 0.202 | 0.084 | 0.214 | 0.214 (0.199) |
| pitch angle std (deg) | 12.3 | 5.9 | 13.4 | 13.6 (12.3) |
| pitch rate autocorrelation at 0.2 s | +0.80 | +0.72 | +0.69 | +0.69 (+0.80) |
| pitch rate autocorrelation at 0.5 s | +0.27 | +0.35 | +0.35 | +0.34 (+0.25) |
| pitch rate autocorrelation at 1.0 s | -0.10 | +0.02 | +0.04 | +0.02 (-0.11) |

Diverged windows: held-out, no disturbance: 0, held-out, fitted disturbance: 0, held-out, fitted disturbance, MuJoCo: 0

Elapsed 701 s.
