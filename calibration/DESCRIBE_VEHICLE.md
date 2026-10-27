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
| Length | Nose to tail, overall | yes |
| Width and height (a frame) | Overall, including thrusters | yes, or the diameter |
| Diameter (a round hull) | The hull, not the fins | yes, or width and height |
| Water | `--water fresh` (a tank, a lake), `sea` (the default), or a density | when not the sea |
| Inertia from CAD | `--inertia IXX IYY IZZ` (kg·m², about the centre) | optional |
| Centre of gravity below the centre of buoyancy | `--cg-below-cb 0.02` (m) | optional |
| Trim | `--net-buoyancy 3` (N; + floats) | optional, default neutral |
| Thrusters | `--thrusters my_logs.yaml` (your ingest spec) or `bluerov2_heavy`: checks the drag ranges against what they can push | optional |
| Torque convention | `--torque-reference-offset X Y Z`, when your logs' torques are taken about another point | rarely |

**The shape comes from the size:**
- **A torpedo** is longer than 3 diameters and round in section (width and height within 30%).
- **Everything else is a frame.**
- Override with `--shape box | torpedo`.

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
- **Give the real size** (`--height 0.254`), which wins over the model's.
- **Meshes aren't measured.** A model with only meshes needs the size given.

## How we know it works

Each generated description was calibrated on its vehicle's real logs, against the description a person wrote for
it, with the same logs, held-out days and seed (PROJECT_STATUS.md #95, `STRESS_TEST_93_95.md`):

MEASURED_95_TABLE

**Pass, per vehicle:**
- held-out skill within 0.05 of the hand-written description's, or better;
- every value the hand-written calibration found inside the generated ranges;
- no more values stuck at a range's end.
