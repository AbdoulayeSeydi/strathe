# Calibrate your HoloOcean vehicle from your logs

For HoloOcean 2.x torpedo vehicles driven by Fossen's equations (`"fossen_model": "torpedo"`, control scheme 1).

```bash
python strathe.py fit-sim    --sim holoocean --scenario my_scenario.json --ingest-spec my_auv.yaml --reference-point 0 0 0
python strathe.py export-sim --sim holoocean --scenario my_scenario.json --calibration calibration_output_my_auv --reference-point 0 0 0
```

- **`fit-sim`** tunes your scenario's `dynamics` settings by replaying your logs through HoloOcean's own Fossen vehicle
  class, then scores the result against your original settings on recordings it never saw.
- **`export-sim`** writes `strathe_fossen.py`: your Strathe calibration as a Fossen vehicle class, with HoloOcean's
  interface, to use with the custom-dynamics control scheme.

## 1. Set it up
- **HoloOcean's Python client:** install it as in the repository's README ("Setup", step 2: link your GitHub account to
  Epic, clone HoloOcean, `pip install` its client). It runs on Windows and Linux, not macOS.
- **Nothing to build:** HoloOcean's Fossen equations are Python (`holoocean.fossen_dynamics`), and `fit-sim` replays
  through them directly, without starting the Unreal engine. `python strathe.py check` says whether they import.
- **Save your scenario dict as JSON:** `json.dump(scenario, open("my_scenario.json", "w"), indent=2)`. The calibrated
  copy has the same shape; load it back with `json.load`.
- **Your own Fossen class instead of HoloOcean's torpedo?** Pass `--fossen-model my_module:MyVehicle` and
  `--configure my_module:configure_from_scenario` (or `--configure ''`). It needs `Minv` and
  `dynamics(eta, nu, u_actual, u_control, sampleTime)`.

## 2. Run it

| Option | Meaning |
|---|---|
| `--scenario`, `--agent` | Your scenario (JSON) and the Fossen agent (default: the one with a `fossen_model`). |
| `--reference-point X Y Z` | Where your logs' velocity and thrust are measured, from the vehicle's origin (Fossen's CO), forward-right-down. |
| `--fit drag lift added_mass cg mass` | What to tune. Default: `drag added_mass cg`. |
| `--hold NAME=VALUE` | Keep a value. Names are the `dynamics` keys: `T_surge`, `T_sway`, `T_yaw`, `zeta_roll`, `zeta_pitch`, `K_nomoto`, `Cd`, `e`, `r44`, `r_bg/0`, `r_bg/2`, `mass`. |
| `--hz` | Integration rate (default 100 Hz). It must divide the logs' rate. |
| `--start-depth`, `--heldout-dates`, `--density-by-date` (writes `rho`), `--compare-params`, `--workers`, `--max-iterations` | As for every simulator ([../README.md](../README.md)). |

**What it tunes, in your scenario's `agents[i]["dynamics"]`:** a key you didn't set starts from the class's default
(REMUS 100 for HoloOcean's torpedo) and is written into the calibrated copy.

| Group | Keys |
|---|---|
| `drag` | the time constants `T_surge`, `T_sway`, `T_yaw`; `zeta_roll`, `zeta_pitch` (damping ratios); `K_nomoto`; `Cd` |
| `lift` | `e` (Oswald efficiency) |
| `added_mass` | `r44` (added roll inertia) |
| `cg` | `r_bg` x and z (centre of gravity) |
| `mass` | `mass` |

**How the replay works:**
- Each window starts from the logged state.
- A hover vehicle's propeller and fins are held at rest. A fin vehicle's logged angles go into the class's own fins, matched by where they
  stand; its propeller's force is in the logged thrust ([FINS.md](../../FINS.md)).
- The logged thrust is added as M⁻¹·τ every step. This is exact, because the equations are linear in τ.
- No frames to convert: Fossen's are Strathe's (north-east-down, forward-right-down).

## 3. Exporting Strathe's model to HoloOcean
HoloOcean's torpedo model is set by time constants and drag coefficients, not per-axis damping. So `export-sim`
doesn't change those settings. It writes **`strathe_fossen.py`**, a self-contained class (numpy only):
- **What it holds:** Strathe's calibrated model (damping, added mass, Coriolis, buoyancy, hull lift), with the centre
  of gravity at your logs' reference point as in Strathe's model.
- **Its interface:** `StratheFossen().dynamics(eta, nu, u_actual, u_control, h)` returns the acceleration without
  thrust; add `Minv @ tau` for your thrust.
- **Using it in HoloOcean:** compute the acceleration with it in your custom-dynamics loop (`env.step(accel)` with
  control scheme 1), in place of `FossenInterface.update`. Convert to HoloOcean's frame there, as `FossenInterface`
  does.

The export report scores it on your held-out recordings. In the tests it reproduces Strathe's own model within 0.02 m/s
over 30 s, including with an offset reference point.

## Status
- **Tested here:** the replay and the export, against a stand-in with HoloOcean's Fossen interface
  (`tests/fake_holoocean`). HoloOcean doesn't install on macOS.
- **Still to check on Windows:**
  - the real `holoocean.fossen_dynamics.torpedo`;
  - an in-engine run of the calibrated scenario against the Python replay;
  - the tick-rate check (`ticks_per_sec × dt = 1`).

  Until then, treat HoloOcean results as provisional ([LIMITS.md](../../../LIMITS.md)).
