# Calibrate the simulator you already use

Strathe calibrates your vehicle from its logs (`strathe calibrate`). These commands then put the result into **your own
simulator model**, the same way for every simulator:

```bash
python strathe.py fit-sim    --sim SIMULATOR ... --ingest-spec my_logs.yaml --reference-point X Y Z
python strathe.py export-sim --sim SIMULATOR ... --calibration calibration_output_my_rov --reference-point X Y Z
```

- **`fit-sim`** tunes your model's own settings (drag, added mass, buoyancy, centre of gravity, ...) by replaying your
  logs in your simulator. It scores the original and the calibrated model on recordings it never saw, and gives your
  model file back with only those settings changed.
- **`export-sim`** puts the values `strathe calibrate` found into your model, where your simulator has the same
  physics. It scores the result the same way.

Only `--sim` and the simulator's own options (which model, which part) change between simulators.
`python strathe.py fit-sim --sim SIMULATOR --help` lists them.

**On the local web page** (`python strathe.py web`), use the "Calibrate your simulator model" and "Put your Strathe
calibration into your simulator" cards. Pick the simulator and give your model and logs. Stonefish keeps its own two
cards.

| `--sim` | Status | What `fit-sim` tunes | `export-sim` | Guide |
|---|---|---|---|---|
| `stonefish` | ready | per-axis drag, added mass, buoyancy, CG (one part) | exact with the patch's damping (#80) | [stonefish/README.md](../stonefish/README.md) |
| `external` | ready | whatever your description lists | where your description maps it | [SIMULATOR_PROTOCOL.md](SIMULATOR_PROTOCOL.md) |
| `gazebo` | ready (#82) | Hydrodynamics damping, added mass, buoyancy (collision size), CG, mass | exact with the reference point at the link origin (0.006 m/s over 30 s) | [gazebo/README.md](gazebo/README.md) |
| `mujoco` | ready (#83) | ellipsoid fluid-model drag and lift coefficients, gravcomp buoyancy, CG, mass | none (MuJoCo has no per-axis damping, buoyancy force or full added mass) | [mujoco/README.md](mujoco/README.md) |
| `holoocean` | ready, provisional until checked on Windows (#84) | the scenario's Fossen `dynamics` (time constants, damping ratios, Cd, r44, CG, mass) | Strathe's model as a Fossen class (`strathe_fossen.py`) | [holoocean/README.md](holoocean/README.md) |

**Any other simulator** (in-house, Isaac Sim, ...): describe it in a short YAML and give it a replay command
([SIMULATOR_PROTOCOL.md](SIMULATOR_PROTOCOL.md)). A working example is in [example/](example/).

## Moving to another simulator: `convert-sim`
```bash
python strathe.py convert-sim --from stonefish --scenario my_world.scn --robot my_auv --reference-point -0.76 0 0 \
    --mass 33 --to gazebo --out my_auv_gazebo        # or --to mujoco, --to holoocean
```
It writes your vehicle for the other simulator from what your Stonefish model reports:
- the mass and its distribution;
- the centres of gravity and buoyancy (the model's trim kept);
- the hull's size and displaced volume;
- added mass and drag (quadratic drag matched at 0.5 m/s; pitch and yaw damping by strip theory).

**What each target gets:**
- **Gazebo:** a world with the Hydrodynamics and Buoyancy systems. The hull-lift terms are set to 0, so `fit-sim` can
  find the fins' stabilising lift.
- **MuJoCo:** an MJCF with an ellipsoid hull. A small body at the centre of buoyancy carries the buoyancy, which gives a
  righting moment.
- **HoloOcean:** a torpedo scenario with first-order time constants.

**Two things to know:**
- **Use `--reference-point 0 0 0` with the converted model:** its origin is your logs' reference point.
- **A conversion is a starting point:** your Stonefish model's fins and mesh drag don't carry over one for one. Calibrate
  the converted model with `fit-sim`.

## Two ways to run each simulator
You choose; each guide shows both.

- **On your machine:** install the simulator, then build or point Strathe at its replay command.
  `python strathe.py check` says what's there and what's missing.
- **Nothing to build:** use a pure-Python path (MuJoCo; HoloOcean's Fossen model) or a Docker image
  (`--replay-tool docker:IMAGE`), which mounts your model's folder at the same path.

## What every simulator shares
- **The logs** are the same folder `strathe calibrate` uses (`--data`), or your raw logs with `--ingest-spec`.
- **`--reference-point`** is where your logs' velocity and forces are measured, in the model's frame: x forward, y
  right, z down, metres from the model's origin. For a URDF `base_link`, it's where `base_link` sits in the model.
- **Scoring:**
  - Each held-out recording is cut into 60 s windows every 10 s.
  - Each window starts from the logged state and is driven by the logged thrust, in still water. The model turns and
    tilts on its own.
  - The report gives the drift after 10, 30 and 60 s, plus the velocity and attitude error.
  - It compares against your original model and two baselines: "no motion" and "keeps its starting velocity".
- **`--density-by-date DATE=RHO`:** use it only if the vehicle kept the same ballast in fresh and salt water.
  - **Why:** a vehicle re-trimmed for each water, as is usual, is about neutral in both. Forcing the density makes it
    2–3% heavy or light in the simulator.
  - **On URI's ALPHA RISE:** without it, both the Stonefish and Gazebo exports reproduced Strathe's own model. With it,
    pitch drifted 19° against 11° (#88).
- **Holding values you measured:** `--hold NAME=VALUE` keeps a value fixed (from a tank test or CAD). Surge added mass
  is the usual one: ordinary missions can't pin it down (#73).
- **Outputs:** `<model>_calibrated.<ext>` (or `_strathe` from export), `<sim>_fit_report.md`, `<sim>_fit.yaml`,
  `drift.png`. Gazebo also writes `calibrated_world/`.

## What the fit does for you (#90)

**Reading the report: the bottom line says up front what drift alone would hide:**
- **Worse attitude:** the calibrated model turns or tilts worse than your original. It names the axes, e.g. "pitch
  7.8° -> 17.3°".
- **Blow-ups:** it blew up in some unseen windows. The medians leave those out, so don't simulate with it as it is.
- **Standing still:** "no motion" beats it at some horizon. That's usual in a tank or on station-keeping, where the
  vehicle stays within a few metres; trust the shorter horizons there.

**A second fit when attitude gets worse.** A fit can trade attitude for position by moving the centre of gravity: on
the public bluerov2_gz, pitch error went from 7.8° to 17.3°. When that happens and the centre of gravity was tuned, not
held, `fit-sim` and `fit-stonefish` fit again with it held. Both are scored, and the better one is kept:
- **The one kept:** the refit, if its attitude is no worse than your original's (or fewer axes are worse), its drift is
  within 10% of the first fit's, and it doesn't blow up more often. Otherwise the first fit is kept.
- **The files:** the kept one is `<model>_calibrated.<ext>`. The other is next to it as `<model>_calibrated_cg_tuned`
  or `_cg_held`. Both are rows in the report's table, and the report says which was kept and why.
- **The time:** about as long again. The refit starts from the first fit's other values (Stonefish: from your model as
  it is).
- **To skip it:** `--no-cg-rerun`, or the checkbox on the web cards. Holding the centre of gravity yourself (`--fit`
  without `cg`, or `--hold`) also skips it.

**An interrupted fit can be finished.** A simulator fit can take hours: the public bluerov2_gz took 7–10 h in Gazebo.
- **The checkpoint:** after every evaluation that improves the fit, the best point so far is saved to
  `<sim>_fit_checkpoint.json` in `--out`. The second fit (centre of gravity held) saves to `..._checkpoint_cg_held.json`.
- **If the laptop sleeps, the power goes or you stop it,** run the same command again with `--finish`, or tick
  "Finish an interrupted calibration" on the web card and keep the same name. It doesn't fit again: it scores the best
  point and writes the model, the world copy and the report as usual, in minutes.
- **What the report says:** a result finished this way is marked as not converged. If the second fit hadn't started,
  the report says to run again without `--finish` to get it.
- **Mismatches:** `--finish` refuses a checkpoint from other settings (other values tuned, another model, other logs
  or held-out dates) and says which. A new fit without `--finish` replaces old checkpoints.

**Gazebo: a world that runs the result.**
- **What it is:** `calibrated_world/` is a copy of your world with the calibrated model in it. An included model folder
  is copied, not linked, and referred to as `model://`, so the folder still runs when you move or share it.
  `calibrated_world/RUN_ME.md` has the one-line command (macOS, Linux, Windows). Your own files
  aren't changed.
- **The physics step:** a fit can find damping too stiff for your world's step. Gazebo applies it step by step, so it
  blows up. Strathe checks each axis's damping time constant against the step. When the step is too long, it scores
  the models at a safe step, writes `calibrated_world/` with it, and says which axes and which step in the report.

These are open-loop replays: your controller's commands react to the real vehicle, so any model's error grows over a
minute. The scores rank models against each other on the same windows. They aren't minute-ahead forecasts
([LIMITS.md](../../LIMITS.md)).
