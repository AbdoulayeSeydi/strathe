# Calibrate your Stonefish vehicle model from your logs

Two commands put Strathe's calibration into the simulator you already use:
- **`strathe fit-stonefish`** fits your Stonefish model's own settings by replaying your logs in Stonefish (an hour or
  more). This is the one that improves Stonefish (53–66% less error after a minute on ALPHA RISE, #73).
- **`strathe export-stonefish`** puts your Strathe calibration into your Stonefish model (minutes), checks each value,
  and scores the result. With Stonefish built with the patch (section 1), Strathe's damping goes in as it is and the
  model behaves like Strathe's (#80). Without it, the drag has to be converted, which hurts over seconds (see
  [Why converting isn't enough](#why-converting-isnt-enough)).

`fit-stonefish` makes the simulator you already use match your real vehicle.
- **What it fits:** your own Stonefish model's settings on one part of the vehicle (usually the hull):
  - quadratic and viscous drag per axis;
  - added mass per axis;
  - the net buoyancy (by resizing a buoyant box, cylinder or sphere part);
  - the centre of gravity's position.
- **What it fits to:** your logs, the same logs `strathe calibrate` uses.
- **How it checks itself:** it scores the original and calibrated models on recordings it never saw.
- **What you get back:** your vehicle file, with only those settings changed.

On URI's ALPHA RISE, Stonefish calibrated this way had 53–66% less position error after 60 s on unseen missions than
URI's original model, holding the surge added mass at a slender-body estimate (PROJECT_STATUS.md #71–#73).

## 0. The quick way
- **One command:** `python strathe.py setup-sim stonefish` clones the Stonefish fork the patch is made for
  (GSO-soslab/stonefish @ 7d52673), applies the patch, builds and installs Stonefish into `~/.strathe/stonefish` and
  builds the replay tool. It finishes with the `strathe check` line. Your own checkout: `--source PATH`. Elsewhere:
  `--prefix PATH`.
  - **Dependencies:** git, cmake, a C++ compiler, GLM, SDL2, Freetype and OpenGL. On macOS also Homebrew's `libomp`,
    whose flags are added for you.
- **Nothing to build, with Docker:** `python strathe.py setup-sim stonefish --docker` builds an image with the patched
  Stonefish and the replay tool, once (about 8 minutes). Every Stonefish command then uses it when the tool isn't built
  here, or when you give `--replay-tool docker`. Install Docker and the details: [../simulators/DOCKER.md](../simulators/DOCKER.md).

Sections 1 and 2 below do the same steps by hand.

## 1. Build Stonefish with the added-mass patch (recommended)
Stonefish gives a body one added mass: the average of its three axes. A slender vehicle carries far less water moving
forward than sideways (ALPHA RISE: about 11 kg forward, about 120 kg sideways), so the average makes its surge
response wrong. Stonefish's drag also can't take a calibration's values (below: it grows with speed cubed and couples
the axes). `stonefish_added_mass.patch` (about 200 lines) adds:
- **Per-axis added mass:** `<hydrodynamics added_mass="x y z" added_inertia="x y z" anisotropic_added_mass="true"/>`
  on a part. The difference from the average, and the Munk moment, are applied as forces each step.
- **Per-axis damping, the same law Strathe calibrates** (#80): `<hydrodynamics damping_linear="Xu Yv Zw"
  damping_linear_rot="Kp Mq Nr" damping_quadratic="Xuu Yvv Zww" damping_quadratic_rot="Kpp Mqq Nrr"
  damping_point="x y z"/>`. Force per axis = (linear + quadratic·|v|)·v, at `damping_point` (in the robot's frame), in
  place of the whole vehicle's mesh drag. `export-stonefish` writes it.

Both are off unless a vehicle file asks for them, so other scenarios behave exactly as before. Stonefish without the
patch ignores the extra settings.

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
**Check what your build supports:** `python strathe.py check` reports the replay tool and whether its Stonefish has the
patch's added mass and damping. Updated the patch? Re-apply it to a clean Stonefish checkout, rebuild and reinstall
Stonefish, then rebuild this tool.
(Built it before #76? Rebuild it: `export-stonefish` needs the mesh areas, inertia and centres it reports (#74), and until
#76 a tool built against unpatched Stonefish listed no parts, so neither command could run without the patch.)
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
- `--from-strathe calibration_output_my_auv`: start from your Strathe calibration's values, converted as
  `export-stonefish` does, then fit. The report compares three Stonefish models: the original, the converted values
  and the fit. **Not recommended yet:** on ALPHA RISE it ended worse than fitting from URI's own model (60 s drift
  29–33 m against 13–22 m, #74), stuck near the converted values.
- `--hold NAME=VALUE`: keep a setting at a value you measured instead of fitting it. **Recommended for surge added
  mass**, which ordinary missions can't pin down. On ALPHA RISE, holding it at a slender-body estimate
  (`--hold added_mass_x=3.3`, about 0.1 × mass) halved the 60 s position error again (#73). For example,
  `--hold added_mass_x=10.6`
  (from a tank test, CAD or your Strathe calibration's `Xu_dot`). The names are:
  - `quadratic_drag_x/y/z`, `viscous_drag_x/y/z` and `added_mass_x/y/z` (`_yz` with `--axisymmetric`);
  - `buoyancy_scale`;
  - `cg_shift_x`, `cg_shift_z`.
- How each axis's error is weighted: by how much that axis's velocity varies in your logs (the default). A vehicle
  cruising steadily at 0.7 m/s would otherwise make surge errors cheap, and the fit would trade surge for pitch and
  yaw. `--weighting magnitude` reproduces the first version.
- It takes an hour or more: every step replays your recordings in Stonefish, several at once (`--workers`).

## What you get (in `--out`, default `stonefish_fit_<name>/`)
- `<your vehicle file>_calibrated.scn`: your vehicle file with only the tuned part (and buoyancy part) changed. Swap it
  in for the original.
- `stonefish_fit_report.md`: the held-out comparison (drift after 10, 30 and 60 s, velocity and attitude error), with
  `drift.png`.
- `stonefish_fit.yaml`: the settings before and after, the fit's cost and the held-out numbers.
- **Read the report's bottom line first.** It says when the calibrated model turns or tilts worse than your original,
  blows up in some unseen windows, or loses to standing still.
- **When the fit made the attitude worse by moving the centre of gravity,** Strathe fits again with it held, scores
  both and keeps the better (`calibration/simulators/README.md`, "What the fit does for you"). The other is next to it
  as `<your vehicle file>_calibrated_cg_tuned.scn` or `_cg_held.scn`. The second fit starts from your model as it is
  and takes about as long as the first. `--no-cg-rerun` (or the web card's checkbox) skips it.
- **Interrupted?** The best point so far is in `stonefish_fit_checkpoint.json`. Run the same command again with
  `--finish` to score it and write the vehicle file and report, without fitting again.

## Describe your vehicle for Strathe from your Stonefish model: `describe-stonefish`

```bash
python strathe.py describe-stonefish --scenario my_world.scn --robot my_auv --reference-point -0.76 0 0 \
    --mass 33 --find my_stonefish_pkg=/path/to/my_stonefish_pkg --out my_vehicle_spec.yaml
```

It writes the vehicle description `strathe calibrate --spec` needs, from what your model already says, so you don't
fill in `templates/vehicle_spec_template.yaml` by hand. It takes:
- **mass:** the model's, or `--mass` (weigh it: the model's can differ, and its rigid inertia is scaled to it);
- **inertia:** roll, pitch and yaw, rigid plus added, about the logs' reference point;
- **added mass** per axis: Stonefish's geometric estimate for each external part;
- **drag** per axis: from the parts' coefficients and mesh areas, Stonefish's cubic law matched at `--cruise-speed`;
- **pitch and yaw damping:** from the hull's length and diameter;
- **net buoyancy:** the model's trim;
- **the centre of gravity's offset** below the centre of buoyancy;
- **the hull-lift terms:** with `--hull-lift`, for a slender hull.

**Limits** come from the hull's size, not the model's values:
- **inertia:** the model's rigid inertia plus slender-body added inertia, 1/5 to 3× (roll 1/5 to 10×);
- **added mass:** surge up to the displaced mass, sway and heave up to 2× strip theory (ρπr²L);
- **drag:** a drag coefficient of 0.05–8 on the frontal area (surge) and 0.03–2.5 on the side area (sway, heave,
  and pitch and yaw by strip theory, up to 3× for fins); linear drag up to 1 m/s × the quadratic limit. Damping is
  kept generous on purpose: in an open-loop replay it also absorbs what the model leaves out;
- **net buoyancy:** ±12% of the weight around the model's trim; **CG below CB:** within the hull's radius.

The model's values can be far off either way. ALPHA RISE's sway and heave added mass are 3× strip theory, and its nose
drag is 20× smaller than the fitted one. With limits sized around those values (5× either way), the fit landed
somewhere that matched the training mission and drifted 64 m on the unseen one. A model value outside the limits
starts just inside them, and the command lists it.

**Calibrate it with `--restarts 4`.** Limits sized from the hull are wide, so where the fit ends up depends on where it
starts. On URI's ALPHA RISE (PROJECT_STATUS.md #77, #79), mean skill on the unseen mission:

| Mission held out | One start | 4 starts | Hand-written description |
|---|---|---|---|
| Bay | −3.18 (blew up: 66 m drift at 60 s) | +0.191, 4.0 m at 60 s | +0.183, 4.0 m |
| Pond | +0.159 | +0.205, 5.3 m at 60 s | +0.227, 3.5 m |

Four starts take about four times as long. Strathe keeps the start with the lowest training cost, so the choice never
uses the unseen mission, and the report lists each start's cost.

## Put your Strathe calibration into Stonefish: `export-stonefish`
```bash
python strathe.py export-stonefish --calibration calibration_output_my_auv --scenario my_world.scn --robot my_auv \
    --part Hull --buoyancy-part Foam --reference-point -0.76 0 0 --find my_stonefish_pkg=/path/to/my_stonefish_pkg
```
It reads the calibration's parameters, its logs (`ingested_logs/`), its held-out dates (`metrics.json`) and the vehicle
mass (calibrations from before #74 need `--mass`). It then sets, on the one part:
- **mass:** the part takes the difference, so the vehicle weighs the calibration's mass;
- **net buoyancy:** resizes `--buoyancy-part`, in the water the calibration was trained in (`--density-by-date`);
- **centre of gravity:** moves the part's `<cg>` so CG − CB matches (fore-aft only when the calibration fitted it);
- **inertia:** roll, pitch and yaw, rigid plus added. It uses added inertia first, then the part's rigid `<inertia>`
  where that isn't enough, never below 5% of Stonefish's own;
- **added mass per axis** (needs the patch);
- **drag:** with the patch, Strathe's per-axis damping as it is, all six axes, at `--reference-point`, replacing the
  vehicle's mesh drag. Without it, Cd and Cf fitted to Strathe's drag over the speeds your logs cover, after what the
  other parts already give (an approximation: see below).

Each value is checked by reloading your vehicle in Stonefish (the report's "before and after" table). You get
`<your vehicle file>_strathe.scn`, `stonefish_export_report.md` (with what can't be carried, and why) and
`stonefish_export.yaml`.

## Why converting isn't enough
Measured on Stonefish's code and in the replay (PROJECT_STATUS.md #74):
- **Stonefish's "quadratic" drag is cubic in speed.** Each mesh face facing the flow adds `|v|² · v_n · area`, so a
  pure translation gives 0.5·ρ·Cd·S·|v|³; the viscous term is linear, ρ·Cf·T·v.
- **It couples the axes.** The force is along the whole flow, so once the hull pitches a few degrees, its large side
  faces add surge drag. Converted per axis, ALPHA RISE lost forward speed (0.61 → 0.46 m/s in a second) under thrust
  that Strathe's model, and the real vehicle, held.
- **Rotational drag** comes from the same coefficients over the mesh, so Strathe's rotational damping can't be set.
- **Strathe's values balance against Strathe's own drag and righting, not Stonefish's.** Transplanted onto URI's ALPHA
  RISE, after 10 s:
  - **better:** heave (velocity error 0.24 → 0.07–0.09 m/s) and heading (20–33° → 14.5°);
  - **worse:** surge (0.15 → 0.42 m/s), roll (2.4 → 5.1°, 4.6 → 10.6°) and pitch (8.9 → 11.3°, 10.9 → 24.2°);
  - **mixed:** sway (better in the pond, about the same in the bay).

  Added mass and buoyancy aren't the cause: they're transplanted exactly. **Surge** is the drag coupling above.
  **Roll** is the transplanted CG − CB (Strathe's 3–5 mm against Stonefish's 97 mm): keep Stonefish's CG and roll
  returns to the original's. **Pitch** is partly the CG (pond: 24° → 16° when it's kept) and partly the rotational
  drag. The export report lists, per axis, what got better and worse.

So with Stonefish's own drag, values can't be copied in; they have to be fitted through it. `fit-stonefish` does that.
Starting it from the converted values (`--from-strathe`) didn't help on ALPHA RISE (#74); start from your own model.

**With the patch's damping (#80), they can.** Stonefish then applies Strathe's damping law itself. Strathe's model is
written about the logs' reference point, so the export also puts the vehicle's centre of gravity there and moves the
buoyancy part to put the centre of buoyancy at the calibrated offset from it (`--buoyancy-part` needed). Otherwise the
logged thrust, acting about 10 cm from Stonefish's own CG, pitches the vehicle. On URI's ALPHA RISE (the baseline
calibration, position error after 10 / 30 / 60 s on the unseen mission):

| Model | Pond held out | Bay held out |
|---|---|---|
| URI's Stonefish model | 2.40 / 15.9 / 47.6 m | 1.92 / 11.4 / 37.4 m |
| Strathe's values, drag converted | 4.16 / 14.8 / 34.3 m | 4.48 / 13.9 / 30.9 m |
| **Strathe's values, with the patch's damping** | **1.85 / 10.1 / 36.0 m** | **1.44 / 5.6 / 17.6 m** |
| Strathe's own model | 1.74 / 10.3 / 26.4 m | 1.47 / 9.1 / 30.6 m |
| `fit-stonefish`, surge added mass held | 2.89 / 9.7 / 21.6 m | 2.48 / 7.2 / 13.8 m |

- **Speeds match Strathe's model** (forward speed error after 10 s: 0.44 → 0.09 m/s and 0.42 → 0.13 m/s; Strathe's
  0.09 and 0.12), and the error after 10 s falls 56–68% against the conversion.
- **Pitch:** 4° after 10 s on the bay, better than Strathe's own model (10°). On the pond, 19° against Strathe's 11°.
  That was the water density: these runs used `--density-by-date`. Scored without it, the pond export reproduces
  Strathe's own model (pitch 11.2° against 10.9°, identical velocity errors, #88).
- **`--density-by-date`:** give it only when the vehicle kept the same ballast in both waters. URI's ALPHA RISE scored
  better without it, so it was probably re-trimmed for the pond, as vehicles usually are for fresh water.
- **The steps that got here:** carrying the damping alone left pitch at 36° / 23°; carrying Strathe's fore-aft offset
  27° / 14°; the CG at the reference point 19° / 4°.
- **So:** export with the patch for a model that behaves like Strathe's; `fit-stonefish` when the best over a minute
  matters (better on the pond).

## Limits
- **One part is tuned,** with diagonal (per-axis) drag and added mass. Without the patch's damping, Stonefish computes
  the rotational drag from the same coefficients over the part's mesh, so it can't be set on its own. (`fit-stonefish`
  fits the mesh coefficients; it doesn't use the patch's damping yet.)
- **No hull lift.** Stonefish has no lift term for a hull; the fit may raise sideways drag to stand in for it.
- **The replay is open loop:** logged forces, still water, the vehicle at `--start-depth` (default 3 m), so the
  surface isn't modelled. Over a closed-loop mission, "keep the starting velocity" is a hard baseline; the report
  shows it.
- **Odd fitted values** (drag far from the shape's, values at their limits) mean the logs don't pin them down.
  Manoeuvres at several speeds help.
