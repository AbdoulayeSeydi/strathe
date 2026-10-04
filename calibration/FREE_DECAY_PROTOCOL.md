# Free-decay tank tests: calibrating roll and pitch

**Why.** Mission logs calibrate surge, sway, heave and yaw, but not roll and pitch (LIMITS.md).
- **Stability:** a BlueROV2 rights itself, because buoyancy sits above weight.
- **Commands:** pilots rarely command roll or pitch, and a 6-thruster BlueROV2 can't pitch on command.
- **The result:** most of the roll and pitch in a mission log comes from the tether and the water, and logged thrust can't explain it.

A free-decay test is the standard way to measure roll and pitch: tilt the vehicle, let go with the thrusters stopped, and record how it swings back.
- **The period** gives how strongly it rights itself.
- **How fast the swing dies out** gives the damping.

It takes 30 to 60 minutes in a pool or tank.

## What you need

- **Water:** at least 2 m deep and 2 m from any wall. The vehicle must stay fully submerged, at least 0.5 m below the surface, because buoyancy changes at the surface.
- **The vehicle as it dives:** the same ballast, floats and payload as on missions. Any change moves the centre of gravity, which is what this test measures.
- **A slack tether:** someone pays it out so it never pulls. A tug during a swing spoils that release; Strathe detects and rejects tugged swings, but you lose the data.
- **Attitude logging:** ArduSub's default 10 Hz is enough. With LOG_BITMASK's ATTITUDE_FAST bit, log faster.
- **Thrusters stopped at release:**
  - **ArduSub:** stay armed in **MANUAL** mode with the sticks centred, so every motor output reads 1500.
  - **Never STABILIZE or DEPTH HOLD:** the autopilot would fight the tilt, and the test would measure the autopilot instead of the vehicle.

## The test, for each axis (roll, then pitch)

1. **Settle:** let the vehicle settle level, submerged, away from walls. Record 10 s of it at rest.
2. **Tilt:** by hand (a diver, or a pole from the side), about **10°**.
   - **On a BlueROV2 Heavy:** you can instead command a short roll or pitch and then centre the sticks. The thrusters must be fully stopped from that moment on.
3. **Release:** hold it still for a second, then **let go cleanly**: no push, hands away.
4. **Wait:** until it is level and still again, about 10 s. Don't touch it or the tether.
5. **Repeat:** 4 releases at about 10° and 4 at about 25°, half each way (nose up and nose down; left and right). Two different angles are what separate linear from quadratic damping.

That makes about 16 releases per axis, 5–10 minutes each. One log file or several, both are fine.

**Optional, for later validation (BlueROV2 Heavy):** with the tether slack, command roll and pitch steps and slow back-and-forth sweeps at two or three sizes. Strathe doesn't fit these yet. They are what would check the measured values against thruster-driven motion.

## In Strathe

```bash
# 1. convert the tank-test logs (templates/ingest_free_decay_ardupilot.yaml for ArduSub .bin; any other ingest
#    template works too) and fit roll and pitch
python strathe.py free-decay --ingest-spec my_tank_tests.yaml --out free_decay_out
#    or, if they are already in Strathe's CSV layout:
python strathe.py free-decay --data <folder> --out free_decay_out

# 2. calibrate from your mission logs as usual, holding what the tank tests measured
python strathe.py calibrate --ingest-spec my_missions.yaml --name my_rov --free-decay free_decay_out/free_decay.yaml
```

**What step 1 does:**
- **Finds the releases:** stretches with the thrusters stopped that start 8–35° from rest and swing back.
- **Fits each axis on its own:**
  - the righting arm (centre of gravity below buoyancy, `cg_cb_offset_z`)
  - linear and quadratic damping (`Kp`, `Kpp` for roll; `Mq`, `Mqq` for pitch)
  - the angle the vehicle rests at
- **Writes:** `free_decay.yaml` and a short `free_decay_report.md`.

**What step 2 does:** the calibration holds those values fixed instead of fitting them from the mission logs. Its report has a "Roll and pitch from tank tests" section.

## What good looks like

- **Releases:** at least 3 per axis, ideally 8 or more, from two different angles.
- **Fit:** it explains 80% or more of each axis's swing.
- **The swing:** a BlueROV2 typically swings with a period of 2–4 s and dies out in a few swings (damping ratio about 0.1–0.5).
- **Roll and pitch agree on the righting arm:** a vehicle has one centre of gravity, so roll and pitch should give the same righting arm. If they don't, the roll/pitch inertia in `vehicle_spec.yaml` is probably off, and the file says by how much.

**An axis is not held,** and the file says why, when it has:
- **too few releases**;
- **a poor fit;**
- **a value at the edge of what the fit allows:** that happens when something else keeps the swing going, such as a tether or a wave. Such a fit can still look good, so it is checked separately.

## What the values mean

**What a free decay measures:** ratios to the vehicle's inertia, not the inertia itself.
- **Inertia:** the values are given the roll and pitch inertia in `vehicle_spec.yaml`, which is the inertia the calibration uses too, so holding them reproduces the measured swing exactly.
- **Period and damping ratio:** in the report, and they don't depend on the inertia.
- **Quoting the damping:** to quote the damping as a vehicle property, measure the inertia separately (CAD, or a bifilar pendulum).
