# Pilot rehearsal: a second public BlueROV2 dataset through the lab flow

PROJECT_STATUS.md #62. Data: DFKI's 6-thruster BlueROV2 with motion capture (Khanmeh et al., Scientific Data 2025,
Zenodo 10.5281/zenodo.17360027, CC BY 4.0). How to reproduce: `calibration/examples/dfki_tether/README.md`.
Local paths are written as `<calibration output>`, `<pilot runs>` and `<repo>`.

| Folder | What it is |
|---|---|
| `calibration/` | `strathe calibrate` on the prepared logs: the report a lab reads, the parameters, metrics, identifiability, and the ingestion report. Finished with `--finish` after the laptop slept with the fit already converged. |
| `arena_150k/` | `strathe arena --task station_keeping --compare-generic` at the default budget: calibration-trained policies 78%, default-trained 100%; the report says to train longer. |
| `arena_450k/` | The same with `--timesteps 450000`, following that advice: both 100%. |
| `fleet_shipped_policies/` | `strathe fleet --params` with the shipped fleet policies (trained on the reference Heavy): 46–63% success and a muddled belief comparison, with the new caution at the top of the report. |
| `fleet_own_policy/` | The same with one policy trained on this vehicle (`strathe train-fleet`, 20.6 min): calibrated beats generic by 12 points in three of four conditions. |
