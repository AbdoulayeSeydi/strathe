# Worked example: a second public BlueROV2 dataset, from raw files to a calibration

This is what a new lab does with its own logs, done on public data (PROJECT_STATUS.md #62).

**The data.** Khanmeh, Wehbe, Simetti and Indiveri, *Tracking data of a Remotely Operated Vehicle and its tether using
a motion capture system and a tension sensor*, Scientific Data (2025). Zenodo
[10.5281/zenodo.17360027](https://doi.org/10.5281/zenodo.17360027), CC BY 4.0.
- **Vehicle:** a standard 6-thruster BlueROV2 (ArduSub), with a gripper and ballast.
- **Where:** DFKI's 23 × 19 × 8 m saltwater test basin.
- **Tracking:** 12 Qualisys motion-capture cameras.
- **Tether:** its pull is measured at the surface.
- **Runs:** 14, about 63 minutes in all: 11 with a limited tether length (5–15 m) and 3 piloted freely (circle, serpentine, random).

**Why it's a good test.** It is unlike the reference dataset in every way that matters to a lab:
- **Files:** one CSV per sensor, at different rates, instead of one table per run.
- **Vehicle:** a 6-thruster vehicle, not the Heavy.
- **Velocity:** world-frame motion capture, not body-frame.
- **Dates:** none.

## Steps

```bash
# 1. download and unzip (223 MB zip, 865 MB unzipped), into data/raw/ (ignored by git)
mkdir -p data/raw/dfki_rov_tether && cd data/raw/dfki_rov_tether
curl -L -o rov_tether_dataset_v3.zip "https://zenodo.org/api/records/17360027/files/rov_tether_dataset_v3.zip/content"
unzip -q rov_tether_dataset_v3.zip && cd ../../..

# 2. one CSV per recording (this folder's script; see its docstring for each choice)
python calibration/examples/dfki_tether/prepare_dfki_tether.py \
    data/raw/dfki_rov_tether/rov_tether_dataset_v1.2 data/raw/dfki_rov_tether/strathe_csv

# 3. calibrate: convert, check, fit, identifiability, report
python strathe.py calibrate --ingest-spec calibration/examples/dfki_tether/ingest_dfki_tether.yaml --name dfki_bluerov2
# or, better, without the stretches where the tether was pulled taut (#63)
python strathe.py calibrate --ingest-spec calibration/examples/dfki_tether/ingest_dfki_tether_slack.yaml --name dfki_bluerov2_slack
```

**The tether matters** (PROJECT_STATUS.md #63).
- **The bursts:** in the limited-tether runs the cable goes taut in bursts of tens of newtons for 20–40% of the time.
- **What the fit does with them:** trained on those runs, the calibration learns the pull as the vehicle's own drag, pushing surge drag to its limits.
- **On held-out free runs, the surge skill by training set:**

| Training set | Surge skill |
|---|---|
| tethered runs | +0.59 |
| tethered runs without the taut stretches, plus free runs (`ingest_dfki_tether_slack.yaml`) | +0.75 |
| free runs only | +0.79 |

- **Thrust and battery:** each sample's thrust is scaled to its logged battery voltage (`voltage_column: battery_v`; 13.1–15.3 V here).

## What the preparation had to work out

- **Velocity:** `velocity.csv` is the motion-capture velocity streamed live, in the basin's frame. It misses up to a third of each run. Instead, the script takes velocity from the origin marker's track in `qtm_rov.csv`, which is seen about 100% of the time. Where both exist they agree to about 2 cm/s, at zero lag.
- **Heading:**
  - **The markers:** they give the body x axis (back-right → front-right), but drop out often.
  - **ArduSub's yaw:** mirrored relative to the basin's frame, offset by about 175°, and drifting by up to 15° near the basin's steel.
  - **What the script uses:** ArduSub's yaw, corrected by its offset from the markers, tracked wherever the markers are seen.
- **Thrusters:**
  - **Frame:** ArduSub's "vectored" frame. Motors 1–4 are as on the Heavy, and motors 5 and 6 are vertical, on the sides.
  - **Which motors are reversed (`reversed`):** recovered by regressing each motor's output on the pilot's sticks (R² 0.94): motors 1, 2 and 5.
  - **Assumed:** where the vertical thrusters sit. It only affects the roll torque.
- **Sessions:** there are no dates. The limited-tether runs and the free runs are labelled as two sessions, so calibration holds out the free runs.

## What it found in Strathe (all fixed in #62)

- **The BlueROV2 Heavy preset was in ROS axes.** The reference dataset turned out to be forward-left-up / z-up, and the preset had been recovered from it. On this ArduSub vehicle, its yaw torque ran against the vehicle's own gyro in 14 of 14 runs. The preset is now FRD in ArduSub's numbering. With the right layout, every force here moves with its velocity: surge 0.80, sway 0.57, heave 0.74, yaw 0.65.
- **`strathe calibrate` failed for any lab.** Its data check, identifiability step and report kept the reference dataset's held-out dates as their defaults.
- **Problems the review round fixed (#63):**
  - **Battery sag:** ignored until `voltage_column:`.
  - **Checks:** reversed forces or a mirrored heading went unnoticed until ingestion checked them.
  - **Tether stretches:** there was no way to drop them (`exclude_samples:`).
- **Smaller problems:**
  - **Out-of-order output:** the data check's output arrived out of order in log files.
  - **A stray "Next:" hint:** printed in the middle of `calibrate`.
  - **One recording session:** a lab with only one got a message in jargon.

The calibration result is in `calibration/results/pilot_dfki_tether/`.
