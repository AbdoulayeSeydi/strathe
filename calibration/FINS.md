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

If the autopilot also rolls the vehicle with its fins, log the `aileron` channel too, or each fin's own angle. Without it the roll
command is taken as zero, and calibration warns: the fit then can't see what holds the roll, and makes up for it elsewhere. OOI's
REMUS 600s log one, and their first real fits were run without it (#97).

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
| A fin calibration, with nothing to say what its fins are | Give `--spec` to `export-sim`: otherwise the fins would be left out of the model |
| The model has no fin for one you described | Add it to the model, or let `convert-sim --spec` write them |
| A glider | `strathe fin-check` reads a glider's rudder; calibration doesn't apply |

## Check a rudder without calibrating

```bash
python strathe.py fin-check --data my_prepared_logs
```

It reads recordings from `strathe ingest` and says whether each fin's sign is right, how quickly the vehicle responds and how much
it turns per unit of fin angle. It's the only fin command for a glider, and a quick first step for any vehicle.

A Slocum glider's logs need no ingest: point `--data` at a folder of `dbd2asc` text files (`.asc`). It reads `m_fin`, `c_fin`,
`m_heading` and `m_depth`, works out the turn rate from the heading, and leaves out time at the surface.

## How far to trust a fit

The report has a Fins section. It says what the logs covered: the speeds, fin angles and turning. **Predictions outside that range
aren't supported.**

- **Log some real manoeuvres.** Straight survey lines don't pin down a fin's lift. On a simulated REMUS, a fit trained only on survey
  lines predicted more survey lines well, but its rudder lift was 39% off and its predictions of hard manoeuvres were poor. Under 20
  manoeuvres per group of fins, the report warns. Deliberate turns and dives fix this.
- **A group of fins that barely moved keeps its geometry value**, and the report says so.
- **No DVL? Log longer.** Position fixes can stand in for a DVL, but speed is then estimated. On a simulated REMUS with clean fixes at
  1 Hz, 100 s of training left the pitching fins' lift 20% off; 10 minutes brought it to 2.4%.
- **Fin lift and inertia are found together.** If the description's turning inertia is 20% off, so is the lift, while predictions
  barely change. Treat the prediction as the result, not the fin value alone.
- **Read "Flying on its own" first.** The fit's windows feed the logged pitch back in, which hides what the elevator does to depth.
  That section has the model fly itself for 30 s on the logged fin angles, on recordings it never saw.

### How a fin fit works

A fin fit runs in two stages, and takes longer than a hover fit because of it.

1. **Short windows.** Each recording is replayed in 2 s windows, each starting from the real state. A hover fit replays whole recordings,
   but that's wrong for many fin vehicles: see below. `--prediction-horizon-s` changes the length.
2. **Flights.** The fit goes on from there with the model flying itself for 10 s at a time, on its own attitude, and scores its roll, pitch
   and heading against the log. Windows alone never show the fit what its own pitch does. Recordings shorter than 10 s can't do this,
   and the report says so.
   This stage stops when it has converged, as the first does, or after `--max-nfev` steps (at most 100). A stable vehicle stops early; an
   unstable one uses most of them. Its time grows with the length of the logs: about 30 evaluations a step, each flying every 10 s of the
   recordings. On a 1.6 h real log that is about 25 minutes a step, so up to about 40 hours; a smaller `--max-nfev` trades accuracy for time.

### A vehicle that's unstable on its own

Many torpedo AUVs are unstable with their fins held still, and fly straight because an autopilot is always correcting them. Published
values for a REMUS double any small error every 0.7 s. The report warns when a fit finds this, and gives a 5 s flight next to the 30 s one.

**On logged fins, nothing can fly such a vehicle far.** Not even its true equations: fed the same 10 Hz fin log, they end up 68 and 146
degrees from where it really went after 30 s. So a large 30 s number on such a vehicle doesn't mean the fit is bad. Use the 5 s number, and
to see it fly for longer, put your autopilot on it in your simulator, as you would on the real vehicle.

## In your simulator

`fit-sim`, `export-sim`, `fit-stonefish`, `export-stonefish` and `convert-sim` all take `--spec` with the vehicle description. The fins go
through the simulator's own physics: Strathe sends each fin's logged angle, and the propeller, gimbal, moving mass and buoyancy engine are already
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

**Exporting a calibration.** `export-sim` writes each fin's own lift, so rudders and elevators can differ, and scales it if the model's fin
area differs from the description's. Each fin needs its own place in the model, named as above. What can't be carried is in the report:
the engine uses the full flow angle rather than the fitted inflow, so a centred fin damps more there, and fin zero offsets stay on your
joints or commands. A `fit-sim` from those values tunes one lift for all the fins. MuJoCo can't take an export; calibrate your MuJoCo model
with `fit-sim`.

**HoloOcean's fins are a cross.** Its class has four fins. Another layout is refused or goes fin by fin to the nearest, never two on one.

**In Docker,** `--stonefish-data /src/Tests/Data` is the image's own folder of meshes, which generated fins use.

A fit with fin lift in it first tries seven lifts from 0.3 to 2.4 times the model's, and starts from the best: a vehicle that turns hard goes
round in circles, which gives a local search a rough cost past the true lift.

**MuJoCo isn't a foil.** On a fin in steady flow, its fluid model's lift jumps by about 0.17 in lift coefficient at the smallest angle, then rises nearly linearly at about 2.6 per radian, so a centred fin can chatter. Stonefish's `Rudder` drops
lift to zero at stall and never uses `drag_coeff`.

## In experiments

`arena`, `fleet` and `train-fleet` run a calibrated fin vehicle on the stand-in physics. An action's six numbers ask for the same things:
surge is the propeller, and roll, pitch and yaw go to the fins through the same mixing the logs use. Sway and heave ask nothing of it: a
fin vehicle can't push sideways, so a policy has to turn. It also can't steer below the speed its fins need. The MuJoCo backend takes a
wrench, so it refuses a fin vehicle.

## How fins are modelled

A thruster's force can be logged. A fin's can't: it depends on how fast the water passes the fin and at what angle. So it's
worked out inside the physics at every step, from the state and the logged angle. A centred fin still pushes back when the
flow meets it at an angle, which is why fins add damping that grows with speed.
