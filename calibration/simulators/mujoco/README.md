# Calibrate your MuJoCo model from your logs

```bash
python strathe.py fit-sim --sim mujoco --model my_auv.xml --ingest-spec my_rov.yaml --reference-point 0 0 0
```

`fit-sim` tunes your MJCF's own fluid-model coefficients, buoyancy and centre of mass. It does this by replaying your
logs in MuJoCo, then scores the result against your original model on recordings it never saw. It runs in Python:
MuJoCo comes with `pip install -r requirements.txt`, so there's nothing to build or install.

## What MuJoCo can and can't represent
MuJoCo has no water. Its fluid model gives:
- **drag and lift:** per geom with `fluidshape="ellipsoid"`, five coefficients (blunt drag, slender drag, angular
  drag, Kutta lift, Magnus lift), scaled by `<option density viscosity>`;
- **some added mass:** only the velocity-dependent part. MuJoCo doesn't apply the part that resists acceleration.

Buoyancy is usually `gravcomp` on the body. It acts at the centre of mass, so there's no righting moment.

So **`export-sim` isn't possible for MuJoCo.** Strathe's model has per-axis linear damping, full added mass and
buoyancy at a centre of buoyancy, and MuJoCo has none of them. `fit-sim` calibrates what your MuJoCo model has.
To run Strathe's own calibrated model inside MuJoCo, see `calibration/mujoco_backend.py`.

## What your model needs
- **The vehicle:** a body with a free joint (`--body` when there are several). Axes are x forward and z up, the ROS
  convention.
- **For drag:** name every hull geom that has `fluidshape="ellipsoid"`; their `fluidcoef` is tuned. If `fluidcoef` is
  missing, MuJoCo's default is written in first. With no ellipsoid geom (MuJoCo's inertia-box model), the option's
  `density` and `viscosity` are tuned instead.
- **The timestep must divide the logs' sample time:** `<option timestep="0.002"/>` for 10 Hz logs.
- **For `--density-by-date`:** an `<option density="..."/>`.
- **A worked example:** [`example/auv.xml`](example/auv.xml), a 10 kg neutral vehicle with an ellipsoid hull. The tests
  run on it.

## Run it

| Option | Meaning |
|---|---|
| `--model`, `--body` | Your MJCF, and the vehicle's body. |
| `--reference-point X Y Z` | Where your logs' velocity and thrust are measured, from the body's origin, as forward-right-down. |
| `--fit drag lift buoyancy cg mass` | What to tune. Default: `drag buoyancy cg`. |
| `--hold NAME=VALUE` | Keep a value you measured. Names: `<geom>_blunt_drag`, `_slender_drag`, `_angular_drag`, `_kutta_lift`, `_magnus_lift`, `gravcomp`, `cg_x`, `cg_z`, `mass`. |
| `--start-depth`, `--heldout-dates`, `--density-by-date`, `--compare-params`, `--workers`, `--max-iterations` | As for every simulator ([../README.md](../README.md)). |

**What you get, in `--out`:**
- `<your model>_calibrated.xml`, with only those numbers changed;
- `mujoco_fit_report.md`, `mujoco_fit.yaml` and `drift.png`.

**How the replay works:**
- Each window starts at `--start-depth` with the logged attitude and velocity.
- The logged wrench is applied at the reference point every timestep (`xfrc_applied`).
- The velocity is read back at the same point (a site Strathe adds in memory; your file isn't changed).
- Frames are converted between north-east-down / forward-right-down and z-up / forward-left-up.
- **Speed:** a 10 s window takes about 0.3 s.

## Fin-steered vehicles

Give `--spec` with your vehicle description: each fin is a child body on a hinge `fin_<name>`, with an ellipsoid geom. `--fit fins` tunes each one's Kutta lift. MuJoCo's fluid model isn't a foil: almost no lift below about 3°. See [FINS.md](../../FINS.md).
