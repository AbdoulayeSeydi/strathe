# Calibrate your Gazebo model from your logs

For Gazebo Harmonic (gz-sim 8) vehicles that use Gazebo's **Hydrodynamics** and **Buoyancy** systems.

```bash
python strathe.py fit-sim    --sim gazebo --world my_world.sdf --ingest-spec my_rov.yaml --reference-point 0 0 0
python strathe.py export-sim --sim gazebo --world my_world.sdf --calibration calibration_output_my_rov --reference-point 0 0 0
```

- **`fit-sim`** tunes your model's own Hydrodynamics coefficients, added mass, buoyancy and centre of mass. It does
  this by replaying your logs in Gazebo, then scores the result against your original model on recordings it never saw.
- **`export-sim`** puts the values `strathe calibrate` found into your model. Gazebo's Hydrodynamics is the same damping
  law as Strathe's (Fossen), so the export is exact.
  - **Checked on Gazebo 8.10:** with the logs' reference point at the link's origin, the exported model and Strathe's
    own model differ by at most 0.006 m/s and rad/s over 30 s. That held with a moving, tilted start, forces and
    torques, added mass and a buoyant vehicle.

## 1. Set it up (choose one)

**A. Build the replay tool on your machine** (macOS, Linux, Windows via conda-forge):
```bash
# Gazebo Harmonic, if you don't have it. conda-forge works anywhere without admin rights (about 3 GB):
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)   # micromamba, once
micromamba create -p ~/strathe_gz_env/env -c conda-forge gz-sim8 cmake ninja cxx-compiler pkg-config
# then build the tool and check it:
python strathe.py setup-sim gazebo --prefix ~/strathe_gz_env/env
```
- **Already have Gazebo Harmonic?** apt on Ubuntu (`gz-harmonic libgz-sim8-dev`) or Homebrew on macOS
  (`brew tap osrf/simulation && brew install gz-harmonic`). Then `python strathe.py setup-sim gazebo` (add `--prefix`
  if it's in a conda env).
- **Check it:** `python strathe.py check` shows the tool and its Gazebo version.

**B. Nothing to build, with Docker:**
```bash
docker build -t strathe-gz-replay calibration/simulators/gazebo   # once
python strathe.py fit-sim --sim gazebo --replay-tool docker:strathe-gz-replay ...
```
Your world's folder is mounted at the same path inside the container. **Not yet tested:** the development machine had
no Docker, so route A is the one that's been run.

## 2. What your world needs
- **The vehicle:** a model with the Hydrodynamics system on its link (`<link_name>`). It can be in the world file, or in
  a model folder the world `<include>`s; then give `--model-file path/to/model.sdf`.
- **Buoyancy:** the Buoyancy system in the world (`<uniform_fluid_density>` or `<graded_buoyancy>`). It uses the link's
  collision shapes. To tune or export buoyancy, use boxes, cylinders or spheres; a mesh's volume isn't known.
- **The physics step must divide the logs' sample time:** `<max_step_size>0.01</max_step_size>` for 10 Hz logs. The
  tool refuses other steps with a plain message.
- **It must also be short enough for the damping.** Gazebo applies the Hydrodynamics forces step by step, so a step
  longer than about half an axis's damping time constant (inertia ÷ damping) can blow up. A light vehicle with stiff
  fitted roll or pitch damping can need 0.005 s. Strathe checks this after the fit. When your step is too long, it
  scores the models at a safe step and names the axes and the step in the report. Then `calibrated_world/` (below)
  uses that step.
- **A worked example:** [`example/world.sdf`](example/world.sdf), a 10 kg neutral vehicle. The tests run on it.

## 3. Run it
```bash
python strathe.py fit-sim --sim gazebo --world my_world.sdf --model-file models/my_rov/model.sdf \
    --ingest-spec my_rov.yaml --reference-point 0 0 0 --heldout-dates 27-03-2026 --axisymmetric
```

| Option | Meaning |
|---|---|
| `--reference-point X Y Z` | Where your logs' velocity and thrust are measured, from the link's origin, in the link's axes as forward-right-down. **Use the link origin (`base_link`, `0 0 0`) when you can.** Gazebo applies damping there, and then the export is exact. Elsewhere, linear damping and added mass are moved exactly, but quadratic damping can't be (Gazebo has no term for it), and the report says so. |
| `--fit drag added_mass buoyancy cg mass` | What to tune. Default: everything but `mass` (weigh your vehicle). |
| `--hold NAME=VALUE` | Keep a value you measured. The names are the plugin's (`xDotU`, `xUabsU`, ...), `added_mass_x/y/z` for `<fluid_added_mass>`, `buoyancy_scale`, `cg_x`, `cg_z`, `mass`. |
| `--axisymmetric` | A torpedo hull: sway = heave, pitch = yaw. |
| `--model-name`, `--link` | When the world has several vehicles. |
| `--start-depth` | Depth of the replay (default 3 m), away from the surface. |
| `--density-by-date DATE=RHO` | Fresh or salt water per recording. It's written into the Buoyancy system. |
| `--from-strathe CALIBRATION` | Start from your Strathe calibration, put in as `export-sim` does. |
| `--compare-params`, `--workers`, `--max-iterations` | As for every simulator ([../README.md](../README.md)). |

**What it tunes, as found in your files:**

| Group | What changes in your files |
|---|---|
| `drag` | the Hydrodynamics plugin's `xU…nR` and `xUabsU…nRabsR` that are present (negative; their magnitudes are tuned) |
| `added_mass` | the link's `<fluid_added_mass>` `xx`, `yy`, `zz`, or the plugin's `xDotU`, `yDotV`, `zDotW` |
| `buoyancy` | the size of the link's box/cylinder/sphere collisions |
| `cg` | the link's `<inertial><pose>` x and z |
| `mass` | the link's `<inertial><mass>` |

**What export-sim writes:**
- **Damping and hull lift:** Strathe's damping, and hull lift if calibrated (`zUW`, `yUV`, `mUW`, `nUV`, `mUQ`, `nUR`),
  into the plugin.
- **Added mass:** into `<fluid_added_mass>`, which Gazebo's physics engine handles with its own Coriolis and Munk terms.
  The plugin's older `xDotU…` terms are set to 0: they estimate accelerations explicitly, and on a test vehicle they
  turned it up to 0.65 rad/s away from Strathe's model.
- **Mass, inertia and the centre of mass:** the mass; Strathe's effective inertia (rigid plus added) as the link's
  inertia; the centre of mass at the reference point.
- **Buoyancy:** the collisions are resized and moved, so the net buoyancy and the centre of buoyancy match the
  calibration.

**What you get, in `--out`:**
- `<your model file>_calibrated.sdf` (or `_strathe.sdf` from `export-sim`), with only those numbers changed;
- `calibrated_world/` (`fit-sim`): a copy of your world with the calibrated model in it, ready to run with
  the command in `calibrated_world/RUN_ME.md` (for an included model:
  `GZ_SIM_RESOURCE_PATH=calibrated_world:$GZ_SIM_RESOURCE_PATH gz sim calibrated_world/<your world>.sdf`). An included
  model folder is copied in, and the world refers to it as `model://`, so the folder still runs when moved or shared. When your world's physics step is
  too long for the fitted damping, this copy uses a safe one. Your own files aren't changed;
- when the first fit made the attitude worse and Strathe fitted again with the centre of gravity held, the other model
  as `<your model file>_calibrated_cg_tuned.sdf` or `_cg_held.sdf` (`--no-cg-rerun` skips the second fit);
- `gazebo_fit_report.md`, `gazebo_fit.yaml` and `drift.png`;
- `gazebo_fit_checkpoint.json`: the fit's best point. If the fit is interrupted, run the same command with `--finish`.

## Frames
Strathe's logs are north-east-down / forward-right-down; Gazebo is east-north-up / forward-left-up. The replay tool
converts, so nothing changes on your side. Checked against analytic steady states:
- 20 N forward gives 0.737 m/s;
- 20 N to starboard gives +0.434 m/s;
- 0.5 N·m of yaw gives +0.225 rad/s, turning clockwise seen from above.
