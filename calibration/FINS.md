# Fin-steered vehicles

Strathe calibrates two kinds of vehicle, and every command and report says which one it's on.

| | Hover | Fin-steered |
|---|---|---|
| What steers it | Thrusters | Fins and a propeller (or a gimballed tail) |
| Examples | BlueROV2, ALPHA RISE | REMUS, Iver, LAUV, Hugin, Autosub, Tethys, Bluefin, Dorado |
| What the logs must have | Thruster commands or forces | Fin angles and propeller rpm |
| Vehicle description | `vehicle_spec_template.yaml` | `vehicle_spec_fin_template.yaml` |

A hull's shape doesn't decide this: ALPHA RISE is torpedo-shaped, and its thrusters steer it, so it's hover.

## Say what kind of vehicle it is

In the vehicle description:

```yaml
vehicle:
  name: my_auv
  steering: fin     # hover | fin | vectored | hybrid
```

- `hover`: thrusters steer it. This is the default.
- `fin`: fins and a propeller.
- `vectored`: a gimballed tail or thruster. Give `propeller:` with `gimbal: true`.
- `hybrid`: fins and steering thrusters.
- `glider`: refused. Calibration needs a propeller's thrust, and a glider moves by changing its buoyancy.

## Describe the fins

`describe-vehicle` writes it from a tape measure:

```bash
python strathe.py describe-vehicle --mass 30.5 --length 1.6 --diameter 0.19 --steering fin --fins cross --fin-x -0.7 --fin-radius 0.12 --fin-area 0.0133 --fin-span 0.086 --propeller-diameter 0.14 --kt0 0.45 --kt1 -0.4 --out my_auv.yaml
```

Fin lift's limits come from the fins' shape. The off-diagonal mass the fins and centre of gravity make is worked out and held,
because logs rarely separate it. By hand, the sections look like this:

```yaml
fins:
  layout: cross       # cross | x | y3 | rudder_only | custom
  x_m: -0.7           # from the torque reference point, forward positive
  radius_m: 0.15      # centre of pressure from the hull's axis
  area_m2: 0.012      # one fin
  span_m: 0.1         # one fin, root to tip
  limit_deg: 20       # the servos' end stops
  stall_deg: 20
  servo: {rate_limit_deg_s: 60}

propeller:
  diameter_m: 0.14
  kt0: 0.45           # thrust coefficient with no forward speed
  kt1: -0.4           # its change with advance ratio
  position_m: [-0.75, 0, 0]
```

For an unusual layout, use `layout: custom` and list each fin with `name`, `position_m` and `span_axis`.

Moving mass and a buoyancy engine (Tethys-class vehicles):

```yaml
moving_mass: {mass_kg: 6.0, axis: [1, 0, 0]}
buoyancy_engine: true
```

## Log the fin angles

In the ingest spec, in place of `wrench:`:

```yaml
fins:
  source: measured              # measured: what the fins did. commanded: what they were told
  unit: deg                     # deg | rad | percent (with full_scale_deg) | pwm_us (with pwm_us_per_deg)
  channels: {rudder: rudder_deg, elevator: elevator_deg}   # or one per fin, by its name in the description
  positive_turns: starboard     # which way a positive rudder turns the vehicle
  positive_pitches: nose_up     # which way a positive elevator pitches it
propeller: {column: prop_rpm, unit: rpm}
```

Other blocks: `gimbal: {y: ..., z: ..., unit: deg}`, `moving_mass: {column: ..., unit: mm}`, and
`buoyancy_engine: {column: ..., unit: litre}`.

Prefer measured angles if you have them. Commanded ones go through the servo's end stops, lag and top speed.

Templates: `ingest_fin_csv.yaml`, `ingest_fin_moos.yaml`, `ingest_fin_rosbag.yaml`.

## What ingestion checks

| Check | What it catches |
|---|---|
| `fin_direction` | A rudder or elevator whose sign is reversed: positive angles go with the opposite motion. It says which key to flip |
| A fin channel past 90° | The wrong `unit` |
| A thruster column named like a fin | A servo that would be read as thrust |

`skip_checks: [fin_direction]` skips the sign check, and the report still shows the numbers.

## What's refused, and what to do

| Message says | Do this |
|---|---|
| Logs have fin channels, description is hover | Say `steering: fin` and describe the fins, or leave the channels out of the ingest spec |
| Description is fin, logs have no fin channels | Add a `fins:` block to the ingest spec |
| Needs a speed source | Log a DVL or position fixes. A fin's force goes as speed squared, so speed can't be inferred from the turning it causes |
| A glider | `strathe fin-check` reads a glider's rudder; calibration doesn't apply |

## Check a rudder without calibrating

```bash
python strathe.py fin-check --data my_prepared_logs
```

It reads recordings from `strathe ingest` and says whether each fin's sign is right, how quickly the vehicle responds and how much
it turns per unit of fin angle. It's the only fin command for a glider, and a quick first step for any vehicle.

## How far to trust a fit

The report has a Fins section. It says what the logs covered: the speeds, fin angles and turning. **Predictions outside that range
aren't supported.**

- **Log some real manoeuvres.** Straight survey lines don't pin down a fin's lift. On a simulated REMUS, a fit trained only on survey
  lines predicted more survey lines well, but its rudder lift was 39% off and its predictions of hard manoeuvres were poor. Under 20
  manoeuvres per group of fins, the report warns. Deliberate turns and dives fix this.
- **A group of fins that barely moved keeps its geometry value**, and the report says so.
- **Fin lift and inertia are found together.** If the description's turning inertia is 20% off, so is the lift, while predictions
  barely change. Treat the prediction as the result, not the fin value alone.
- **Read "Flying on its own" first.** The fit's score feeds the logged pitch back in, which hides what the elevator does to depth.
  That section has the model fly itself for 30 s on the logged fin angles, on recordings it never saw.

## In your simulator

`fit-sim` and `export-sim` take `--spec` with the vehicle description (`--fin-spec` for `fit-stonefish`). The fins go through the
simulator's own physics: Strathe sends each fin's logged angle, and the propeller, gimbal, moving mass and buoyancy engine are already
in the logged force.

| Simulator | What a fin is there | What `fit-sim` tunes |
|---|---|---|
| Gazebo | A link on a revolute joint `fin_<name>`, with a `LiftDrag` plugin | `fin_cla`, the lift slope |
| Stonefish | A `rudder` actuator `fin_<name>` | `fin_lift`, every rudder's `lift_coeff` |
| HoloOcean | The Fossen class's own fins, matched by where they stand | `CL_delta` |
| MuJoCo | A child body on a hinge `fin_<name>` with an ellipsoid geom | Each fin's Kutta lift |
| Your own | `fin_<name>` columns in the replay contract (`simulators/SIMULATOR_PROTOCOL.md`) | Your knobs |

`convert-sim --spec` writes the fins into a converted model. Strathe's Docker images need rebuilding to take fin angles:
`strathe setup-sim gazebo --docker`, or `stonefish`.

**MuJoCo isn't a foil.** Its fluid model gives almost no lift below about 3° and isn't linear above that. Stonefish's `Rudder` drops
lift to zero at stall and never uses `drag_coeff`.

## How fins are modelled

A thruster's force can be logged. A fin's can't: it depends on how fast the water passes the fin and at what angle. So it's
worked out inside the physics at every step, from the state and the logged angle. A centred fin still pushes back when the
flow meets it at an angle, which is why fins add damping that grows with speed.
