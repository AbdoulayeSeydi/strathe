# Your vehicle's description, from a tape measure and a scale

Every Strathe command takes a vehicle description (`--spec`). It holds a starting value and a range for each value
calibration fits: inertia, drag, added mass, buoyancy and the centre of gravity. **`strathe describe-vehicle` writes
it for you**:
- from the vehicle's mass and size;
- or from its model in Gazebo, URDF, MuJoCo or HoloOcean.

There's no template to fill in by hand. (For a Stonefish model, `describe-stonefish` does the same: see
`stonefish/README.md`.)

```bash
# a frame (an ROV): mass, length, width, height
python strathe.py describe-vehicle --mass 11.5 --length 0.457 --width 0.575 --height 0.254 --out my_rov_spec.yaml
# a torpedo-shaped AUV: mass, length, diameter
python strathe.py describe-vehicle --mass 33 --length 1.5 --diameter 0.254 --out my_auv_spec.yaml
# from your simulator model (any size you give wins over the model's)
python strathe.py describe-vehicle --from-model my_robot.urdf --sim urdf --out my_spec.yaml
# then
python strathe.py calibrate --ingest-spec my_logs.yaml --spec my_rov_spec.yaml --name my_rov
```

On the web page (`python strathe.py web`), use the card "Describe your vehicle (no simulator needed)".

## What to measure

| Measure | How | Needed |
|---|---|---|
| Mass | Weigh it as you logged it: batteries, payload, ballast | yes |
| Length | Nose to tail, overall (from 5 cm; a vehicle under 5 cm or 10 g is refused, as a likely unit mistake) | yes |
| Width and height (a frame) | Overall, including thrusters | yes, or the diameter |
| Diameter (a round hull) | The hull, not the fins | yes, or width and height |
| Water | `--water fresh` (a tank, a lake), `sea` (the default), or a density | when not the sea |
| Inertia from CAD | `--inertia IXX IYY IZZ` (kg·m², about the centre) | optional |
| Centre of gravity below the centre of buoyancy | `--cg-below-cb 0.02` (m) | optional |
| Trim | `--net-buoyancy 3` (N; + floats) | optional, default neutral |
| Thrusters | `--thrusters my_logs.yaml` (your ingest spec, when it logs thruster commands: a preset, a layout or an allocation matrix) or `bluerov2_heavy`: checks the drag ranges against what they can push | optional |
| Torque convention | `--torque-reference-offset X Y Z`, when your logs' torques are taken about another point | rarely |

**The shape comes from the size:**
- **A torpedo** is longer than 3 diameters and round in section (width and height within 30%).
- **Everything else is a frame.**
- Override with `--shape box | torpedo` only if the hull really is that shape. Strathe warns ("goes against the
  size") when the shape you give isn't the one the size reads as: a BlueROV2 described as a torpedo lost 0.077
  held-out skill, with its pitch inertia and yaw damping cut off.

## How the ranges are set

The logs decide the values. The ranges matter more than the starting points: a range sized from the hull keeps the
fit from wandering to values that fit the training logs and fail on unseen ones (PROJECT_STATUS.md #77).

| Value | Frame (L × W × H) | Torpedo (L × D) |
|---|---|---|
| Quadratic drag | 0.5·ρ·Cd·face area; Cd 0.1–3 on each face (W×H surge, L×H sway, L×W heave) | Cd 0.05–8 nose-on (π D²/4), 0.03–2.5 broadside (L×D) |
| Linear drag | up to 1 m/s × the quadratic range | the same |
| Rotational drag | each face as plates turning about the axis (Cd up to 9), or enough to stop the largest inertia allowed within 0.1 s, whichever is larger: thrusters and a tether resist turning far more than the frame's outline (DFKI's BlueROV2 needs 10 N·m·s; the plates alone allow 1.1) | strip theory 0.5·ρ·D·L⁴/32, up to 7.5× |
| Added mass | 0 to 4 × the mass | surge up to the displaced mass; sway and heave up to 2× strip theory ρπr²L |
| Inertia | a solid box of the mass (or your CAD), plus 0.75× that of water; 1/5 to 5× | a solid cylinder (or your CAD), plus slender-body ρπr²L³/12 for pitch and yaw; 1/5 to 3–10× |
| Net buoyancy | ±12% of the weight around your trim | the same |
| CG below CB | within half the height | within the radius |

The file records each range's reason (`bounds_rationale`), so you can see where every number came from. The
torpedo rules are the ones `describe-stonefish` uses.

## From a simulator model

`--from-model FILE --sim gazebo | urdf | mujoco | holoocean` (the kind is guessed from `.sdf`, `.world`, `.urdf`,
`.json`):

| | Mass, inertia | Size |
|---|---|---|
| Gazebo (SDF), URDF | the vehicle's link: the Hydrodynamics plugin's, else the heaviest | its collision boxes, cylinders and spheres, turned by their poses |
| MuJoCo | the vehicle's body and everything attached to it | its geoms (boxes, ellipsoids, cylinders, capsules, spheres) |
| HoloOcean | the scenario's `dynamics: mass` | `length` and `diam` in `dynamics`, else give `--length --diameter` |

**Check the size it reports.** Collision shapes are often not the hull:
- **bluerov2_gz's only collision shape is a 5 cm thick buoyancy block.** Strathe says so ("one side is much thinner
  than the others") when a side is under a quarter of the next.
- **A converted model's hull may be sized to float the vehicle, not to its outside.** URI's ALPHA RISE in Gazebo is a
  0.17 m cylinder that displaces its 33 kg; the real hull, with flooded sections, is 0.254 m. Its added-mass limits came
  out 57% too small and the fit pinned them. Strathe says so ("it may be sized to float the vehicle") when a single
  round hull displaces the vehicle's own mass to within 10%.
- **A close size still moves the limits a little.** URI's ALPHA RISE in MuJoCo reads 1.45 × 0.28 m (the real hull
  1.5 × 0.254 m); its pitch-damping limit, which grows with the length to the fourth power, came out 0.5% short of
  where the hand-written fit stopped.
- **Give the real size** (`--height 0.254`; `--length 1.5 --diameter 0.254`), which wins over the model's; mass and
  inertia still come from the model.
- **Meshes aren't measured.** A model with only meshes needs the size given.

## How we know it works

Each generated description was calibrated on its vehicle's real logs, against the description a person wrote for
it, with the same logs, held-out days and seed (PROJECT_STATUS.md #95, `STRESS_TEST_93_95.md`):

| Vehicle | Described from | Held out | Hand-written | Generated | Pinned (hand / gen.) | Passes |
|---|---|---|---|---|---|---|
| BlueROV2 Heavy | its size and mass | 16, 19 Feb | 0.379 | **0.435** | 14 / 5 | yes |
| BlueROV2 Heavy | bluerov2_gz (Gazebo), height given | 16, 19 Feb | 0.379 | **0.434** | 14 / 5 | yes |
| BlueROV2 Heavy | a URDF of it, height given | 16, 19 Feb | 0.379 | **0.434** | 14 / 5 | yes |
| DFKI BlueROV2, 6 thrusters | its size and mass | its held-out runs | 0.297 | **0.303** | 11 / 8 | yes |
| ALPHA RISE | its size and mass | pond day | 0.227 | **0.221** | 12 / 11 | yes |
| ALPHA RISE | its size and mass | bay day | 0.183 | **0.197** | 11 / 7 | no: roll inertia and heave drag: not held by the pond day's logs |
| ALPHA RISE | its Gazebo model, real size given | pond day | 0.227 | **0.221** | 12 / 7 | yes |
| ALPHA RISE | its Gazebo model, real size given | bay day | 0.183 | **0.197** | 11 / 8 | no: heave drag: not held by the pond day's logs |
| ALPHA RISE | its MuJoCo model | pond day | 0.227 | **0.216** | 12 / 8 | no: pitch damping: the model is 1.45 m long, not 1.5 |
| ALPHA RISE | its MuJoCo model | bay day | 0.183 | **0.196** | 11 / 8 | yes |
| ALPHA RISE | HoloOcean, size given | pond day | 0.227 | **0.222** | 12 / 12 | yes |
| ALPHA RISE | HoloOcean, size given | bay day | 0.183 | **0.192** | 11 / 9 | no: roll inertia and heave drag: not held by the pond day's logs |
| ALPHA RISE | its Stonefish model (`describe-stonefish`) | pond day | 0.227 | **0.216** | 12 / 10 | yes |
| ALPHA RISE | its Stonefish model, 1 start | bay day | 0.183 | **0.123** | 11 / 9 | no: yaw damping stuck; the fit ran out of evaluations |
| ALPHA RISE | its Stonefish model, 4 starts as it advises | bay day | 0.183 | **0.123** | 11 / 9 | no: the same fit: the first start was the best |

Skill is the mean over the six axes on the held-out recordings (1 is perfect, 0 is no better than predicting no motion);
"pinned" counts values that ended stuck at a limit. Each "no" but the last two scores within 0.05 of the hand-written
description, or better, and misses on a value the training day's logs don't decide (both fits leave it stuck at a limit,
in different places; LIMITS.md) or on a model a little off the real size. `describe-stonefish`'s description scores 0.06
below on the bay day, with one start or four (seed 95): calibrate this command's description too
(`stonefish/README.md`).

- **Measurements off by up to 30%** (mass, length, width or height, one at a time, on the BlueROV2): 0.428 to 0.439,
  against 0.435 with the right values. A tape measure and a bathroom scale are enough.
- **Four starting points** (`--restarts 4`) on the BlueROV2 found the same fit as one, for both descriptions, at 3 to
  5.5× the time.
- **In a simulator:** both BlueROV2 calibrations, put into bluerov2_gz with `export-sim gazebo` and scored there on
  the held-out days: neither diverged, and the one from the size drifted 5.41 m in 30 s, the hand-written one 4.98 m.
- **Any size, any mix of inputs:** every pair of values of eleven inputs (26 descriptions), and 600 random ones from
  5 cm to 10 m, each with any optional inputs: every one loads as `calibrate` loads it.
- **What a wrong input costs:** a model's thin buoyancy block read as the hull, 0.312; the BlueROV2 described as a
  torpedo, 0.358; ALPHA RISE as a box, −0.097 on one day (+0.224 on the other). Strathe's notes say so in each case.

**Pass, per vehicle:**
- held-out skill within 0.05 of the hand-written description's, or better;
- every value the hand-written calibration found inside the generated ranges. A value both calibrations left stuck at
  the same end of their ranges is excused: the logs didn't hold it down, so neither found it;
- no more values stuck at a range's end.
