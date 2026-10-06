# Pilot rehearsal: a second public BlueROV2 dataset through the lab flow

PROJECT_STATUS.md #62 (first run) and #63 (rerun). Data: DFKI's 6-thruster BlueROV2 with motion capture
(Khanmeh et al., Scientific Data 2025, Zenodo 10.5281/zenodo.17360027, CC BY 4.0). How to reproduce:
`calibration/examples/dfki_tether/README.md`. Local paths are written as `<calibration output>`, `<pilot runs>` and
`<repo>`.

## Current: the rerun (#63)

What changed from the first run:
- **Spec:** `ingest_dfki_tether_slack.yaml`, which drops the stretches where the tether was pulled taut (surface tension above 5 N).
- **Thrust:** each sample's thrust scaled to the logged battery voltage.
- **Same split:** tethered sessions train, the free sessions are held out.

| Folder | What it is |
|---|---|
| `calibration/` | `strathe calibrate`. It converged (4.5 min). Held-out skill: surge +0.80, sway +0.61, yaw +0.56, heave +0.45, pitch +0.01, roll −0.25. The report calls 0 of 6 axes trustworthy, because each better-predicted axis now relies on a drag value at its limit. |
| `arena/` | `strathe arena --task station_keeping --compare-generic` at the new 450k default: calibration-trained and default-trained policies both 100%, hand-written controller 85%. |
| `fleet_shipped_policies/` | `strathe fleet --params` with the shipped fleet policies (trained on the reference Heavy): calibrated beats generic at 3 of 4 comm levels (+8, +7, +6 points), with the training-vehicle caution at the top. |
| `fleet_own_policy/` | One policy trained on this vehicle (27 min): calibrated beats generic by 12, 10 and 22 points at L2@3s, L3a@1s and L3a@3s (p ≤ 0.002), and ties 100%/100% at L2@1s. One policy seed, simulation only. |

## `first_run_62/`: the first run (#62)

That run used the tethered data as it was and one nominal voltage.
- **Calibration:** held-out surge +0.60, worse than the default model.
- **Arena:** 78% at the default budget then (150k), 100% at 450k.
- **Fleet, shipped policies:** a muddled comparison.
- **Fleet, own policy:** calibrated +12 points at 3 of 4 comm levels.
