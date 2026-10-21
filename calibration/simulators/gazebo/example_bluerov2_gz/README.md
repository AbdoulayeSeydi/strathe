# Worked example: a public Gazebo BlueROV2 Heavy, calibrated on public logs

Everything here is public, so a lab can repeat it before using its own vehicle:
- **The model:** [bluerov2_gz](https://github.com/clydemcqueen/bluerov2_gz) (MIT licence), a BlueROV2 Heavy for Gazebo
  Harmonic. It has quadratic drag only, no added mass, and buoyancy from one collision box. That's typical of a
  model written by hand, and leaves calibration a lot to fix.
- **The logs:** oceansystemslab's [BlueROV2 trajectory dataset](https://github.com/oceansystemslab/Bluerov2_trajectory_dataset):
  a BlueROV2 Heavy in a tank, 39 recordings on 4 days. Strathe's own reference calibration uses it.

Strathe's code only reads the model's files. Nothing from the model's repository is run.

## 1. Get the model and the logs

```bash
git clone --depth 1 https://github.com/clydemcqueen/bluerov2_gz.git ~/bluerov2_gz   # about 80 MB
bash data/download_dataset.sh                                                     # the logs, once
python calibration/simulators/gazebo/example_bluerov2_gz/prepare_logs.py ~/bluerov2_logs
```

`prepare_logs.py` puts the logs in Strathe's layout:
- **Axes:** the dataset is in ROS axes (forward-left-up, z up), converted to forward-right-down as `frame: flu`
  ingestion does.
- **Torques:** they get the 8.5 cm shift Strathe's calibration uses for this dataset, so they're about the
  horizontal-thruster plane. That's bluerov2_gz's `base_link` origin, so the reference point is `0 0 0`, the case
  where the Gazebo export is exact.

## 2. Calibrate the model

```bash
python strathe.py fit-sim --sim gazebo \
    --world calibration/simulators/gazebo/example_bluerov2_gz/world.sdf \
    --model-file ~/bluerov2_gz/models/bluerov2_heavy/model.sdf --resource-path ~/bluerov2_gz/models \
    --data ~/bluerov2_logs --reference-point 0 0 0 --heldout-dates 16-02-2024 19-02-2024 --out bluerov2_gz_fit
```

- **`world.sdf`** has only what the replay needs: physics at a 0.01 s step (10 Hz logs), the Buoyancy system and the
  vehicle. bluerov2_gz's own worlds also pull a sun and a seabed from Gazebo Fuel over the network.
- **What it tunes:** the Hydrodynamics plugin's drag, added mass (the plugin's `xDotU`...), the buoyancy box and the
  centre of mass: 18 values.
- **Held out:** the 16 and 19 February recordings, the days Strathe's reference calibration holds out.
- **Time:** several hours on 2 cores, because there are 25 training recordings. `--workers` uses more cores.

## 3. Results

Drift on the two held-out days (16 and 19 February, 712 windows of 60 s), medians:

| Model | 10 s | 30 s | 60 s | Velocity error over 10 s, u v w (m/s) | Roll, pitch, yaw after 10 s |
|---|---|---|---|---|---|
| bluerov2_gz as published | 3.51 m | 8.11 m | 11.99 m | 0.33, 0.30, 0.20 | 3.9°, 7.8°, 41° |
| Calibrated by Strathe, all 18 values (15 evaluations) | 1.87 m | 4.17 m | 6.02 m | 0.22, 0.10, 0.08 | 7.3°, 17.3°, 50° |
| **Calibrated, centre of gravity held** (`--fit drag added_mass buoyancy`, converged after 27), at a 0.005 s step | **1.90 m** | **3.73 m** | **5.46 m** | **0.23, 0.11, 0.08** | **3.5°, 6.0°, 42°** |
| No motion | 3.06 m | 3.58 m | 3.29 m | | |
| Keeps its starting velocity | 4.64 m | 14.72 m | 29.53 m | | |

- **What it changed:**
  - linear drag on every axis, from 0 (surge 29.7, sway 19.0, heave 55.2 N·s/m);
  - quadratic drag about 5× the published values on the translational axes;
  - added mass, from 0, to 10.1 / 8.8 / 3.6 kg;
  - the centre of gravity, raised from 1.1 to 2.7 cm;
  - the buoyant volume, ×0.963.
- **Velocity and position** improve a lot: drift halves at every horizon.
- **Attitude gets worse.** Pitch error more than doubles. Raising the centre of gravity shortens the righting arm
  (the buoyancy box is at 6 cm), so the vehicle tilts more. The report now says so at the top ("Be careful: ...
  turns or tilts worse").
- **In a tank, standing still beats both models after 30 s,** because the vehicle stays within a few metres. The report
  says this too. Trust the 10 s horizon and the velocity errors.
- **Holding the centre of gravity** keeps the gains and fixes the attitude: drift falls 55% at 60 s, and roll and pitch
  are slightly better than the published model's. Its training cost is also lower (874 against 996), because the first
  run hadn't finished at the 15-evaluation default. Run it so:

  ```bash
  python strathe.py fit-sim --sim gazebo ... --fit drag added_mass buoyancy --max-iterations 30
  ```
- **The world's step has to be shorter than the default 0.01 s for that model.** Scored at 0.01 s, it blew up in 35 of
  712 unseen windows. At 0.005 s it blew up in none, with the same medians. Its fitted roll and pitch damping (17 and 14
  N·m·s on 0.26 and 0.23 kg·m²) has an 11 ms time constant, and Gazebo applies damping step by step. The report now
  says so after the fit and names the step: `<max_step_size>0.005</max_step_size>`.
- **Time:** the first run took about 9.5 h on 2 cores, while another calibration used the other 6; the second about
  7.5 h on 7 cores, partly on battery.
