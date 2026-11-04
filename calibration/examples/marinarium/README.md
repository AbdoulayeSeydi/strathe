# KTH's Marinarium BlueROV2 (PX4), from its ROS 2 bags

A fourth real vehicle for the sensor suite (PROJECT_STATUS.md #96): a BlueROV2 Heavy run by PX4 in KTH's Marinarium
tank, with a motion-capture system. Strathe reads its ROS 2 bags directly with `ingest_marinarium.yaml`.

## Where the data comes from

[github.com/ViktorNfa/bluerov2_dynamics](https://github.com/ViktorNfa/bluerov2_dynamics), MIT licence, under
`rosbags/`:

| Bag | Day | Length | Size |
|---|---|---|---|
| `rosbag2_2025_10_30-16_31_20` | 30 Oct 2025 | 20 min | 233 MB |
| `rosbag2_2025_11_06-manual` | 6 Nov 2025 | 15 min | 175 MB |
| `rosbag2_2025_11_06-stabilized` | 6 Nov 2025 | 15 min | 176 MB |

- **The bags are Git LFS files.** Fetch them with `git lfs pull`, or from
  `media.githubusercontent.com/media/ViktorNfa/bluerov2_dynamics/main/rosbags/...`. Check each one against the
  SHA-256 in its LFS pointer.
- **The PX4 message definitions are needed too.** They're in the repository's `rosbags/types/px4_msgs/msg/`.
  Copy them to `px4_msgs/msg/` next to this spec, and put the bag folders under `bags/`.
- **Don't use the repository's CSVs** (`koopman_dataset_50Hz*.csv`). Their body rates don't follow their own
  attitude. Their `*_sp` columns are normalized wrench setpoints, not newtons.

## What's in the bags, and what was checked

- **Thrust:** PX4's normalized motor commands, `actuator_motors.control[0..7]`, from −1 to 1.
  - PX4 sends **NaN for a stopped motor**. It's taken as stopped (LOG_FORMATS.md); 22–35% of the motor messages
    have one.
  - The thrusters are PX4's BlueROV2 Heavy geometry, as the repository's own allocation has it. A test checks the
    spec against it (`tests/test_ingest_bags_and_thrusters.py`).
  - The battery voltage is logged, but its message definition isn't in the repository. So thrust uses Strathe's
    T200 curve at about 15 V.
- **Motion capture:** `/mocap/itrl_rov_1/odom`, about 100 Hz, in PX4's frames (world NED, body FRD).
  - That frame was checked against PX4's accelerometer: gravity matches to 0.6 m/s², against 2.8–11 m/s² for the
    other frames.
  - With each message's own stamp, its body velocity matches its positions' derivative to 1 cm/s (correlation
    0.997–0.999). By the bag's receive times it didn't, so `stamp: header`.
- **IMU:** PX4's `sensor_combined`, about 100 Hz, FRD. Its gyro follows the motion capture's rate of turn.
- **No pressure sensor or DVL** was logged. Depth for the no-velocity configurations is the motion capture's z, as
  for the reference tank.

## What ingest says

Ingest passes every check: 3 recordings over 2 days at 10 Hz. Force against velocity is clearly positive on
surge, sway, heave, pitch and yaw. It warns about:

- motion-capture gaps (interpolated across);
- the NaN commands;
- the motion capture's orientation jumping by more than 45° a few times (22, 2 and 5 times per recording);
- roll and pitch rates too small, on 6 Nov, to check the gyro against the attitude.

## In the stress test

`stress/prepare_marinarium_96.py` makes the K-* rows of #96 (STRESS_TEST_93_95.md). It ingests the bags with this
spec, then cuts the recordings down to each sensor set, with the motion capture's velocity hidden:

- motion-capture positions and depth;
- USBL-like fixes and depth;
- depth, gyro and attitude only;
- the same with PX4's accelerometer.
