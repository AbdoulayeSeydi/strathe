# Calibrate your Stonefish vehicle model from your logs

`strathe fit-stonefish` makes the simulator you already use match your real vehicle.
- **What it fits:** your own Stonefish model's settings on one part of the vehicle (usually the hull):
  - quadratic and viscous drag per axis;
  - added mass per axis;
  - the net buoyancy (by resizing a buoyant box, cylinder or sphere part);
  - the centre of gravity's position.
- **What it fits to:** your logs, the same logs `strathe calibrate` uses.
- **How it checks itself:** it scores the original and calibrated models on recordings it never saw.
- **What you get back:** your vehicle file, with only those settings changed.

On URI's ALPHA RISE, Stonefish calibrated this way had 27–35% less position error after 60 s on unseen missions than
URI's original model (PROJECT_STATUS.md #71, #72).

## 1. Build Stonefish with the added-mass patch (recommended)
Stonefish gives a body one added mass: the average of its three axes. A slender vehicle carries far less water moving
forward than sideways (ALPHA RISE: about 11 kg forward, about 120 kg sideways), so the average makes its surge
response wrong. `stonefish_added_mass.patch` (about 100 lines) adds two things:
- **Per-part settings in the scenario:** `<hydrodynamics added_mass="x y z" added_inertia="x y z"
  anisotropic_added_mass="true"/>`.
- **Per-axis added mass, applied each step:** the difference from the average, and the Munk moment, as forces. It's
  off unless asked for, so other scenarios behave exactly as before.

```bash
cd stonefish                                  # your Stonefish checkout (e.g. github.com/patrykcieslak/stonefish)
git apply /path/to/strathe/calibration/stonefish/stonefish_added_mass.patch
cmake -S . -B build && cmake --build build -j && sudo cmake --install build
```

Without the patch everything still works, except added mass: `fit-stonefish` then fits drag, buoyancy and the CG only,
and says so.

## 2. Build the replay tool
```bash
cmake -S calibration/stonefish -B calibration/stonefish/build -DCMAKE_PREFIX_PATH=<your Stonefish install prefix>
cmake --build calibration/stonefish/build
```
This builds `calibration/stonefish/build/strathe_stonefish_replay`, which `fit-stonefish` finds on its own
(or give `--replay-tool`). Stonefish is Linux-first.
- **macOS:** build Stonefish and this tool with OpenMP from Homebrew `libomp`, passing `-DOpenMP_CXX_FLAGS="-Xpreprocessor
  -fopenmp -I$(brew --prefix libomp)/include" -DOpenMP_CXX_LIB_NAMES=omp
  -DOpenMP_omp_LIBRARY=$(brew --prefix libomp)/lib/libomp.dylib`.
- **Windows:** use WSL.

## 3. Run it
```bash
python strathe.py fit-stonefish --scenario my_world.scn --robot my_auv --part Hull \
    --ingest-spec my_auv_ingest.yaml --reference-point -0.76 0 0 --heldout-dates 27-03-2026 \
    --buoyancy-part Foam --axisymmetric --find my_stonefish_pkg=/path/to/my_stonefish_pkg
```
- `--scenario`: the world you normally run, with the robot in it. ROS-style `$(find PKG)` paths need `--find PKG=path`;
  `<include>` with `<arg>` works as in Stonefish. Sensors are removed for the replay (they don't move the vehicle and
  need a GPU); your files are never changed.
- `--reference-point`: where your logs' velocity and forces are measured, in the robot's Stonefish frame (x forward,
  y right, z down, metres from the robot's origin). For a URDF `base_link` 0.76 m behind a Stonefish origin at the
  nose: `-0.76 0 0`.
- `--axisymmetric`: a torpedo-shaped hull; one value across the hull (sideways = vertical).
- `--density-by-date 27-03-2026=1000`: fresh-water recordings in a sea-water world.
- `--compare-params calibrated_parameters.yaml`: also scores your Strathe calibration, for reference.
- It takes an hour or more: every step replays your recordings in Stonefish, several at once (`--workers`).

## What you get (in `--out`, default `stonefish_fit_<name>/`)
- `<your vehicle file>_calibrated.scn`: your vehicle file with only the tuned part (and buoyancy part) changed. Swap it
  in for the original.
- `stonefish_fit_report.md`: the held-out comparison (drift after 10, 30 and 60 s, velocity and attitude error), with
  `drift.png`.
- `stonefish_fit.yaml`: the settings before and after, the fit's cost and the held-out numbers.

## Limits
- **One part is tuned,** with diagonal (per-axis) drag and added mass. Stonefish computes the rotational drag from the
  same coefficients over the part's mesh; it can't be set on its own.
- **No hull lift.** Stonefish has no lift term for a hull; the fit may raise sideways drag to stand in for it.
- **The replay is open loop:** logged forces, still water, the vehicle at `--start-depth` (default 3 m), so the
  surface isn't modelled. Over a closed-loop mission, "keep the starting velocity" is a hard baseline; the report
  shows it.
- **Odd fitted values** (drag far from the shape's, values at their limits) mean the logs don't pin them down.
  Manoeuvres at several speeds help.
