# Connect any simulator to Strathe

Strathe can calibrate a vehicle model in any simulator, including an in-house one, Isaac Sim, or a simulator Strathe
doesn't build in. Your simulator needs two things:
1. **A replay command** that replays logged windows from a CSV and writes back what the vehicle did. This page specifies it.
2. **A short YAML** that says where each tunable number sits in your model file, with its limits.

```bash
python strathe.py fit-sim --sim external --simulator my_sim.yaml --ingest-spec my_logs.yaml --reference-point 0 0 0
python strathe.py export-sim --sim external --simulator my_sim.yaml --calibration calibration_output_my_rov --reference-point 0 0 0
```

A complete working example is in [`example/`](example/): `example_replay.py` (about 80 lines), `vehicle.json` and
`my_sim.yaml`. The test suite runs it end to end. Start from it.

## 1. The replay command
Strathe calls your command in three ways:

```text
TOOL --model MODEL --ref X Y Z --dt DT --model-dir DIR [your options] --features
TOOL --model MODEL --ref X Y Z --dt DT --model-dir DIR [your options] --info
TOOL --model MODEL --ref X Y Z --dt DT --model-dir DIR [your options] --windows IN.csv --out OUT.csv
```

| Argument | Meaning |
|---|---|
| `--model` | A copy of your model file with Strathe's values written in. It's in a temporary folder. |
| `--model-dir` | The folder your original model file is in. Resolve relative paths in the model (meshes, includes) against it. |
| `--ref X Y Z` | Where the logs' velocity and forces are measured, in the vehicle's frame: metres from the model's origin, x forward, y right, z down. |
| `--dt` | The logs' sample time, in seconds. Hold each logged force for `dt`, stepping your physics as finely as it needs inside. |
| `--features` | Print one JSON line about the simulator, e.g. its version (and `"fins": true` if it applies fin angles), without loading a model. `strathe check` uses it. |
| `--info` | Load the model and print one JSON line with what you know. Lines that don't start with `{` are ignored. Useful keys: `mass` (kg) and `water_density` (kg/m³). |
| `--windows IN.csv --out OUT.csv` | Replay every window in `IN.csv` and write `OUT.csv`. |

**`IN.csv`** has one row per logged sample of each window:

```text
win,k,u,v,w,p,q,r,qx,qy,qz,qw,Fx,Fy,Fz,Tx,Ty,Tz
```

- `win` is the window number and `k` the sample within it, from 0.
- `u v w p q r` is the vehicle's velocity at the reference point, in body axes forward-right-down (m/s, rad/s).
  For each window, only row `k = 0` is used, as the start state.
- `qx qy qz qw` is the orientation: the quaternion from body to world, with the world north-east-down. Only row 0 is used.
- `Fx Fy Fz Tx Ty Tz` is the logged force and torque from the thrusters, in body axes and about the reference point
  (N, N·m). Apply row `k`'s wrench from sample `k` to `k + 1`.

**A fin-steered vehicle** (`FINS.md`) adds one column per fin after `Tz`, named `fin_<name>` from the vehicle description, in radians.
A positive angle lifts toward span × chord. Fin force depends on your simulator's own flow, so apply these with your fin model. The
propeller, gimbal, moving mass and buoyancy engine are already in the wrench. Say `"fins": true` in `--features`, or fit-sim refuses
fin logs with your tool. Hover vehicles' windows don't change.

**`OUT.csv`** has the same windows, with what your vehicle did:

```text
win,k,u,v,w,p,q,r,qx,qy,qz,qw
```

- Use the same axes and frames as the input.
- Row `k = 0` is the start state as given.
- The vehicle moves on its own: don't reset its velocity or attitude to the log during a window.
- If the simulation diverges, stop that window. Strathe treats the missing rows as diverged.

**Rules:**
- **Deterministic:** the same input gives the same output, every time. The fit compares nearby settings, and noise
  would swamp the differences. Turn off sensor noise, random currents and anything else random.
- **No controller:** apply the logged wrench and any logged fin angles as they are, with your vehicle's thrusters and controllers out
  of the loop.
- **Free water:** a fixed depth, away from the surface and the seabed, in still water (Strathe models the current from
  the logs separately).
- **Fast is better:** every step of the fit replays all training windows several times. Strathe runs `--workers`
  copies of your command at once.

**Frames:** if your simulator is east-north-up / forward-left-up (Gazebo, ROS), convert at the edge of your command.
Body velocity and wrench: y → −y, z → −z. Orientation: R_NED = T · R_ENU · T_b, where T swaps x and y and negates z,
and T_b = diag(1, −1, −1).

**The command can be:**
- a path;
- a command line (`python my_replay.py`);
- `docker:IMAGE`: Strathe then runs it with `docker run`, mounting your model's folder, the folders your model refers
  to by full path and the work folders at the same paths. It runs as your user id with `HOME=/tmp`, so your command
  mustn't need root ([DOCKER.md](DOCKER.md) section 6).

On Windows, write paths with backslashes as usual; put a path with spaces in double quotes.

## 2. The simulator description (`my_sim.yaml`)

```yaml
simulator: My sim                 # the name used in the reports
replay: python my_replay.py       # paths are relative to this file
model: vehicle.sdf                # your vehicle's model file
format: xml                       # xml | json | yaml | regex (default: from the file extension)
options: [--depth, "3"]           # extra options for your command (optional)
density: //fluid/density          # where the water density is in the model (optional; --density-by-date writes it)
knobs:                            # what Strathe may tune
  - {name: drag_x, group: drag, at: "//xUabsU", sign: -1}
  - {name: drag_y, group: drag, at: "//yVabsV", sign: -1, axisym: drag_yz}
  - {name: drag_z, group: drag, at: "//zWabsW", sign: -1, axisym: drag_yz}
  - {name: mass, group: mass, at: "//link[@name='base_link']/inertial/mass", lo: 1.2, hi: 1.2}
  - {name: cg_x, group: cg, at: "//link[@name='base_link']/inertial/pose#0", transform: linear, lo: -0.1, hi: 0.1}
export:                           # optional: where Strathe's own calibrated values go (export-sim)
  - {param: Xuu, at: "//xUabsU"}
  - {param: Xu_dot, at: "//xDotU", scale: -1}
  - {param: mass, at: "//link[@name='base_link']/inertial/mass"}
```

**Where a number is (`at`):**

| `format` | `at` | Examples |
|---|---|---|
| `xml` (SDF, URDF, MJCF, any XML) | A small XPath. Steps are `/` (child) or `//` (anywhere below). A step is a tag or `*`, with `[@attr='value']` and/or `[n]` (from 1). A final `/@attr` picks an attribute, and `#k` picks the k-th of several space-separated numbers (from 0). | `//plugin[@name='gz::sim::systems::Hydrodynamics']/xUabsU`, `//geom[@name='hull']/@fluidcoef#2` |
| `json`, `yaml` | A JSON pointer. | `/agents/0/dynamics/mass` |
| `regex` | A regular expression with one `( )` around the number. It must match once. | `mass\s*=\s*([-\d.eE]+)` |

Strathe changes only those numbers.
- **XML:** every other byte stays as it was (comments, formatting, order).
- **JSON:** it's rewritten with the same keys, in the same order.
- **YAML:** it's rewritten, so its comments are lost. Keep a copy, or use `regex`.

**Each knob:**

| Key | Meaning | Default |
|---|---|---|
| `name` | What `--hold NAME=VALUE` and the report use. | required |
| `group` | `--fit drag added_mass ...` picks groups. Any names you like. | required |
| `at` | Where the number is. | required |
| `transform` | `log_factor`: tuned as a factor of its starting value, with limits a factor either way (`lo`, `hi`, default 30). `log`: tuned on a log scale, with limits in its own units. `linear`: tuned by adding to it, with limits as offsets. | `log_factor` |
| `sign` | `-1` when the file holds a negative number. Strathe then tunes its magnitude, e.g. damping. | `1` |
| `lo`, `hi` | The limits (see `transform`). | 30× either way; ±(\|start\| + 1) for `linear` |
| `axisym` | With `--axisymmetric`, knobs sharing this key take one value (a torpedo hull: sway = heave). | none |
| `note` | A line for the report. | none |
| `if_zero` | Where to start a `log_factor` knob that is 0 in your model. A factor of 0 stays 0, so Strathe refuses a zero without it. Size it from your vehicle, e.g. 0.3 × the mass for added mass. | none |

Choose what to tune by which physics in your simulator actually moves the vehicle: drag, added mass, buoyancy and the
centre of gravity first. Keep the list short. Each knob adds a replay of all training windows to every step of the fit.

**`export`:** each entry puts one of Strathe's calibrated values into the file: `file value = scale × value + offset`.
- **Which values:** `param` is a field of `calibrated_parameters.yaml` (`Xu`, `Yv`, … `Nr`, `Xuu` … `Nrr`, `Xu_dot`,
  `Yv_dot`, `Zw_dot`, `Ieff_roll` …, `net_buoyancy_z`, `cg_cb_offset_z`, …), or `mass`.
- **When it's exact:** if your simulator uses the same Fossen law (damping (linear + quadratic·|v|)·v per axis, diagonal
  added mass), the exported model behaves like Strathe's own.
- **What it can't place:** what your simulator has no setting for is listed in the report as not carried.

## 3. Run it

```bash
python strathe.py fit-sim --sim external --simulator my_sim.yaml --data my_logs/ --reference-point 0 0 0 \
    --heldout-dates 27-03-2026 --axisymmetric --hold added_mass_x=3.3
```

| Option | Meaning |
|---|---|
| `--data` / `--ingest-spec` | Your logs in Strathe's layout, or raw logs converted first (LAB_GUIDE.md step 3). |
| `--heldout-dates` | Recordings kept back for scoring. Default: the latest 30%. |
| `--fit GROUP ...` | Tune only these groups. |
| `--hold NAME=VALUE` | Keep a knob at a value you measured. |
| `--axisymmetric` | Use the `axisym` pairs. |
| `--weighting variation\|magnitude` | How each axis's error counts (`variation`: by how much it varies in your logs). |
| `--from-strathe CALIBRATION` | Start from your Strathe calibration's values (needs `export:`). |
| `--density-by-date DATE=RHO` | Fresh or salt water per recording (needs `density:`). |
| `--compare-params calibrated_parameters.yaml` | Also score Strathe's own model. |
| `--max-iterations`, `--workers` | The fit's budget, and how many copies of your command run at once. |

**What you get, in `--out`:**
- `vehicle_calibrated.<ext>`: your model with only the knobs changed.
- `<sim>_fit_report.md`: the held-out comparison (drift after 10/30/60 s, velocity and attitude error) against your
  original model and the baselines, what changed, and how the fit went.
- `<sim>_fit.yaml`: the same numbers.
- `drift.png`: a plot of the comparison.

`export-sim` writes `vehicle_strathe.<ext>`, `<sim>_export_report.md` and `<sim>_export.yaml`.

## Checklist for a new simulator
1. **Make your command replay one window.** Write by hand an `IN.csv` with a constant forward force. The vehicle should
   accelerate forward and settle at the speed where drag balances the force.
2. **Check that it's deterministic:** run it twice and compare the two `OUT.csv` files.
3. **Check the frames:** the same window with a sideways force should move the vehicle to starboard (+v). A positive
   yaw torque should turn it clockwise seen from above (+r).
4. **Write `my_sim.yaml` with two or three knobs**, then run `fit-sim` with `--max-iterations 3` as a smoke test.
5. **Add the rest of the knobs, and `export:` if your physics allows it.**
