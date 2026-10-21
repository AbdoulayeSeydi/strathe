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

To be added from the run in PROJECT_STATUS.md.
