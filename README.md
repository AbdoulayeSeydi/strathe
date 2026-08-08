# Calibrated Digital Twin for BlueROV2 (Marine Autonomy)

A system-identification pipeline that fits a physics model to real BlueROV2
mission logs, integrated into [HoloOcean](https://byu-holoocean.github.io/holoocean-docs/v2.0.0/index.html)
for high-fidelity underwater simulation — plus a Gym-compatible RL training
environment built on top of the calibrated physics.

**What this answers:** does calibrating a simulator against real logged
vehicle data measurably improve how closely it tracks real behavior,
compared to an uncalibrated default model?

**Answer so far:** yes, on most axes. Real, held-out-by-date validation
(never-before-seen collection sessions, not just held-out samples) shows a
genuine accuracy improvement — see [Results](#results) below for the honest,
full breakdown, including where it's still a work in progress.

## Setup

### 1. Python dependencies

```bash
pip install -r requirements.txt
```

### 2. HoloOcean (the underlying simulator)

HoloOcean can't be distributed via `pip`/PyPI directly, due to the Unreal
Engine EULA — it requires a one-time account-linking step first:

1. **Link your GitHub account to an Epic Games account** and accept the
   Unreal Engine EULA — follow
   [Epic's official steps](https://www.unrealengine.com/en-US/ue-on-github).
   **Accept the email invitation from Unreal Engine** — easy to miss, and
   the next step fails silently without it.
2. **Clone and install the HoloOcean client library:**
   ```bash
   git clone git@github.com:byu-holoocean/HoloOcean.git holoocean
   cd holoocean/client
   pip install .
   ```
   (If cloning gives a "Page Not Found" error, step 1's account link hasn't
   gone through yet.)
3. **Install the Ocean world package** (several GB, one-time download):
   ```bash
   python -c "import holoocean; holoocean.install('Ocean')"
   ```

Full official docs:
[HoloOcean Installation Guide](https://byu-holoocean.github.io/holoocean-docs/v2.0.0/usage/installation.html).

### 3. Use the calibrated model directly

```bash
python -c "
from calibration.dynamics_model import CalibrationParameters
import yaml

with open('calibration_output_holoocean_coriolis/calibrated_parameters.yaml') as f:
    params = CalibrationParameters(**yaml.safe_load(f)['parameters'])
print(params)
"
```

*(Confirmed working as-written, run from the project root.)*

## The real (non-demo) workflow

`run_full_pipeline.py` (above) is a fast, truncated DEMO - good for
showing the mechanism works end-to-end, not for producing the final,
reportable numbers. The real pipeline, run against the full dataset and
real HoloOcean physics, is a sequence of separate scripts, run in order:

1. **Pre-flight data-quality check** (always run this first):
```bash
   cd calibration
   python preflight_check.py --data-dir <path-to-logs> --spec vehicle_spec.yaml
```
   Validates sampling rate and Coriolis-relevant excitation on the REAL,
   full dataset before committing real compute time to a calibration that
   might not succeed.

2. **Stand-in calibration** (fast, minutes):
```bash
   python run_calibration.py --data-dir <path> --spec vehicle_spec.yaml \
       --sampling-rate-hz 10.0 --heldout-dates 16-02-2024 19-02-2024 \
       --output-dir <output-dir> --max-nfev 60
```

3. **HoloOcean refinement** (slow, hours - real, confirmed cost ~124s per
   evaluation on this dataset; checkpointed and resumable):
```bash
   python run_calibration_holoocean.py --data-dir <path> --spec vehicle_spec.yaml \
       --sampling-rate-hz 10.0 --resume-from-checkpoint <path>/checkpoint.yaml \
       --n-steps-per-run 150 --max-nfev 15 --output-dir <output-dir>
```

4. **Real validation (Claim A)** against real HoloOcean physics:
```bash
   python replay_comparison.py --data-dir <path> --spec vehicle_spec.yaml \
       --sampling-rate-hz 10.0 --calibrated-params <output-dir>/calibrated_parameters.yaml \
       --result-type final --output-dir <output-dir>
```

5. **Real multi-seed PPO training (Claim B)** - optionally pick a task
   (`waypoint` default, `station_keeping`, or `docking`):
```bash
   python train_multiseed.py --data-dir <path> --spec vehicle_spec.yaml \
       --calibrated-params <output-dir>/calibrated_parameters.yaml \
       --result-type final --n-seeds 4 --task waypoint --output-dir <output-dir>
```

6. **Optional - deeper diagnostics**, once you have a real calibrated
   result:
   - **Per-DOF real-vs-simulated plots and force-breakdown charts** (fast
     - a few minutes, no Jacobian evaluation):
```bash
     python diagnose_calibration.py --data-dir <path> --spec vehicle_spec.yaml \
         --calibrated-params <output-dir>/calibrated_parameters.yaml \
         --sampling-rate-hz 10.0 --heldout-dates 16-02-2024 19-02-2024 \
         --skip-jacobian --plot --plot-output-dir <output-dir>/plots
```
     Shows exactly WHERE in a trajectory the fit breaks down, per DOF -
     not just the aggregate RMSE number.
   - **Automated parameter identifiability report** (real cost ~15-17 min
     on the full dataset - condition number, correlated pairs,
     multi-parameter flat directions):
```bash
     python automated_identifiability_report.py --data-dir <path> --spec vehicle_spec.yaml \
         --sampling-rate-hz 10.0 --calibrated-params <output-dir>/calibrated_parameters.yaml
```

See `PROJECT_STATUS.md` for the full history of real runs, real bugs
found and fixed along the way, and the honest per-DOF results.

## What's in this repo

- **`calibration/`** — the full calibration pipeline: data loading, the
  physics model, the optimizer, the HoloOcean integration, and the
  Gym/`Arena` RL training environment.
- **`calibration_output_*/`** — saved, ready-to-use calibrated parameter
  sets (`calibrated_parameters.yaml` + `metrics.json` in each).
- **`PROJECT_STATUS.md`** — the full research log: every finding, every bug
  found and fixed, every validation check, in chronological order. This is
  the detailed record for anyone who wants to verify the methodology or
  understand exactly how a specific number was produced. Not required
  reading to just use the model.

## Results

**Methodology:** real BlueROV2 mission logs (public `oceansystemslab`
dataset), held out by entire collection *date* (not a random split) —
so validation checks generalization to entire unseen sessions, not just
unseen samples. Real logged thruster commands replayed open-loop through
both default and calibrated physics, evaluated inside real HoloOcean
simulation, compared against real logged trajectories.

**Headline: ~3–8% scale-normalized RMSE improvement** over uncalibrated
default physics, on held-out real data (varies by calibration segment —
see `PROJECT_STATUS.md` for the full history).

Per-axis breakdown (RMSE improvement over default, most recent result):

| Axis | Improvement |
|---|---|
| Surge (u) | +61–63% |
| Sway (v) | +29–30% |
| Heave (w) | +27–35% |
| Roll rate (p) | **known limitation — currently under active investigation** |
| Pitch rate (q) | +25–35% |
| Yaw rate (r) | **known limitation — currently under active investigation** |

We're being upfront about roll and yaw rate — they're real, honest results,
not polished ones. Multiple fixes have been tried (bound adjustments,
custom objective weighting); the leading remaining hypothesis is that the
truncated training-run length used for speed during refinement doesn't
capture their true dynamics timescale. See `PROJECT_STATUS.md` (findings
#20–27) for the full technical investigation.

## Status

- ✅ Real-data calibration pipeline, validated (held-out-by-date, multi-restart stability)
- ✅ HoloOcean integration, frame conventions confirmed
- ✅ Gym-compatible RL environment (`Arena`), PPO training confirmed working
- 🔄 Rotational-axis accuracy (roll, yaw rate) — actively being improved
- ⏳ Real hardware validation — not yet done; looking for lab partners with BlueROV2 access

## Contact

Abdoul Seydi — abdoulayeaseydi@gmail.com

Looking for feedback, collaboration, or lab partners with real BlueROV2
(or similar hovering-AUV) access for hardware validation.