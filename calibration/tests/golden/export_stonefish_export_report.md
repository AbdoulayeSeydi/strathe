# Your Strathe calibration in Stonefish: vehicle.scn

## Bottom line

On recordings the calibration never saw (02-01-2030), Stonefish with Strathe's values has **5.79 m** of position error after 60 s, against **14.73 m** for your original model (-61%). Check every horizon below: a change can help over a minute and cost over seconds.

After 10 s, by axis: **better**: forward speed (surge), sideways speed (sway), up/down speed (heave); **worse**: roll, pitch, heading (yaw).

**Strathe's values are not a calibrated Stonefish model: they balance against Strathe's own drag and righting, not Stonefish's. Its mesh drag grows with speed cubed and couples the axes (surge), and its roll and pitch damping come from the same mesh coefficients, so a transplanted centre of gravity no longer balances them (roll, pitch). To calibrate your Stonefish model, run `strathe fit-stonefish`, which fits it through Stonefish itself (on URI's ALPHA RISE: 53-66% less error after 60 s, against 17-28% for this conversion; PROJECT_STATUS.md #73-#75).**

## How the models compare on unseen recordings

Windows of 60 s every 10 s; each starts from the logged state and is driven by the logged forces, the model turning and tilting on its own, in still water. Medians.

| Model | drift 10 s | drift 30 s | drift 60 s | velocity error over 10 s, u v w (m/s) | roll, pitch, yaw after 10 s | diverged |
|---|---|---|---|---|---|---|
| Stonefish, original | 4.68 m | 9.11 m | 14.73 m | 0.551, 0.152, 0.297 | 0.0°, 0.0°, 0.0° | 0/4 |
| Stonefish with Strathe's values | 1.28 m | 2.48 m | 5.79 m | 0.057, 0.103, 0.128 | 13.8°, 5.6°, 12.1° | 0/4 |
| Strathe's own model | 4.03 m | 10.40 m | 18.90 m | 0.422, 0.062, 0.156 | 97.6°, 53.7°, 76.7° | 0/4 |
| no motion | 5.57 m | 8.64 m | 14.12 m | 0.709, 0.145, 0.287 | 64.7°, 18.7°, 85.1° | 0/4 |
| keeps its starting velocity | 8.13 m | 19.26 m | 39.90 m | 0.941, 0.126, 0.127 | 64.7°, 18.7°, 85.1° | 0/4 |

![Drift](drift.png)

"Keeps its starting velocity" is a hard baseline over a closed-loop mission: the logged forces react to the real vehicle, so any open-loop model's error compounds. These numbers rank models; they don't promise minute-ahead prediction.

## Strathe's values, and what Stonefish has before and after

| Quantity | Strathe | Stonefish before | Stonefish after |
|---|---|---|---|
| mass | 32 kg | 30 kg | 32 kg |
| net buoyancy (water 1000 kg/m³) | 4 N | 13.89 N | 3.998 N |
| CG − CB, x | n/a (not fitted) | 0 m | 0 m |
| CG − CB, z | 0.02 m | 0.05 m | 0.02 m |
| roll inertia (rigid + added) | 0.8 kg·m² | 1.1 kg·m² | 0.8 kg·m² |
| pitch inertia (rigid + added) | 6 kg·m² | 5 kg·m² | 6 kg·m² |
| yaw inertia (rigid + added) | 7 kg·m² | 5 kg·m² | 7 kg·m² |
| added mass x | 6 kg | 28.33 kg | 6 kg |
| added mass y | 60 kg | 28.33 kg | 60 kg |
| added mass z | 65 kg | 28.33 kg | 65 kg |

**Drag** (Stonefish's mesh drag is cubic in speed, so it's matched to Strathe's over the speeds your logs cover; water 1000 kg/m³):

| Axis | Cd | Cf | at speeds (median, 95th pct) | Strathe's force | Stonefish's force |
|---|---|---|---|---|---|
| x | 0.9607 | 0.02895 | 0.70, 1.36 m/s | 30.3, 103.0 N | 30.7, 103.3 N |
| y | 2.044 | 0.1036 | 0.13, 0.31 m/s | 9.0, 31.0 N | 9.3, 30.8 N |
| z | 1.96 | 0.1201 | 0.14, 0.40 m/s | 10.7, 53.8 N | 11.2, 54.7 N |

## What changed in your vehicle file

`vehicle_strathe.scn` is `vehicle.scn` with only part `Hull` and the buoyancy part `Foam` changed:

- mass: 32 kg
- quadratic drag 0.960664 2.0442 1.96033, viscous drag 0.0289479 0.103558 0.120126
- added mass 6 60 65, added inertia 0 3 4, `anisotropic_added_mass="true"` (needs Stonefish with `calibration/stonefish/stonefish_added_mass.patch`)
- rigid inertia 0.8 3 3 kg·m²
- centre of gravity moved by +0.000 m (x), -0.030 m (z)
- buoyant volume of `Foam`: x1.032

## Not carried, and why

- Fore-aft CG offset kept as Stonefish has it: the calibration didn't fit cg_cb_offset_x (only with the torpedo-hull terms, vehicle_spec.yaml).
- Drag converted, not carried: Stonefish's mesh drag grows with speed cubed and couples the axes, and its rotational drag comes from the same coefficients over the part's mesh. Stonefish built with calibration/stonefish/stonefish_added_mass.patch takes Strathe's damping as it is.
