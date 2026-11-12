# Unit Economics: compute cost per calibration, on Modal

*Updated 12 Nov 2026 for running the cloud compute on [Modal](https://modal.com/pricing). Prices were read from Modal's
pricing page on 12 Nov 2026. Run times come from this project's own runs, and each one says where it came from. Some
numbers are estimates, and they're labelled as estimates. This document covers cost. It doesn't set the price a lab pays.*

**Nothing in Strathe runs on Modal yet.** No Modal app, image or volume exists in this repository. Every time below was
measured on one Apple M1 laptop (8 cores, 8 GB), usually with 2 to 4 jobs sharing it. The Modal figures are the measured
work multiplied by Modal's prices. They are not Modal bills.

## Modal prices (12 Nov 2026)

| Resource | Modal's price | Per hour |
|---|---|---|
| CPU, per physical core (= 2 vCPU) | $0.0000131 / core / s | **$0.0472 / core-h** |
| Memory | $0.00000222 / GiB / s | **$0.0080 / GiB-h** |
| GPU T4 | $0.000164 / s | $0.590 / h |
| GPU L4 | $0.000222 / s | $0.799 / h |
| GPU A10 | $0.000306 / s | $1.10 / h |
| Volumes (checkpoints, uploaded logs) | $0.09 / GiB / month, first 1 TiB / month free | |
| Network egress | $0.04 / GiB | |

How the bill works:
- **A GPU container is also billed for its CPU and memory**, on top of the GPU price.
- **Preemptible is the default price.** Non-preemptible runs cost 3x. Pinning a region costs 1.15x to 1.75x. Sandboxes
  are billed at the non-preemptible rate.
- **Minimums:** 0.125 core and 128 MiB per container.
- **Plans:**
  - Starter: $0 a month, with $30 of compute included each month. It allows 3 seats, 100 containers and 10 GPUs at once.
  - Team: $250 a month, with $100 included. It allows unlimited seats, 5,000 containers and 50 GPUs at once.
  - Enterprise: custom pricing, with volume discounts.

**Working rate used below:** one core plus 2 GiB of memory costs **$0.063 per core-hour** ($0.0472 + 2 × $0.0080).
2 GiB per core is an assumption, because our fit workers' memory hasn't been measured on Linux. At 1 GiB per core the
rate is $0.055, which is 12% less.

## How the measured times become core-hours

The fit spreads each evaluation across worker processes (`objective.py`). Its cost is the CPU time those workers use.

- **Measured directly (fin fit):** the workers' CPU time, read from the operating system while the fit ran.
- **Estimated (everything else):** the row's wall time × 2 cores. That's roughly what each job got while 2 to 4 jobs shared the
  8 cores. It could be off by up to 2x either way.
- **Excluded:** rows paused with SIGSTOP for the Fin Steer build (over 100 h each).

On Modal, giving a fit more cores should make it finish sooner for about the same core-hours, because the windows split
across workers. That hasn't been measured beyond the laptop's 8 cores, so treat shorter wall times on Modal as a projection.

## Cost per job

| Job | Measured work | Source | Modal cost |
|---|---|---|---|
| Pre-flight check, `strathe check` | Seconds | Any run | ~$0.00 |
| **Hover calibration**, one vehicle's logs | 0.8 to 14.6 h wall (14 runs), median ~3.5 h, so ~1.6 to 29 core-h, median ~7 | `results_94.csv`, the `*-cal` rows across six log formats | **$0.10 to $1.83, typically ~$0.45** |
| **Fin calibration** (fin fit plus free-run validation) | **45 core-s per evaluation, measured**; a full fit is ~3,200 evaluations, so ~40 core-h | T-prestero-long: 18.1 core-h of worker CPU over its first 1,442 second-stage evaluations | **~$2.55** |
| Fin calibration, range across #97 | 6.3 to 23.8 h wall for synthetic truths; 38.8 h and 55 h for the two longest real REMUS fits, so ~13 to 110 core-h | `results_97.csv`, the T-* and O-* rows | **$0.80 to $6.93** |
| Simulator fit, Gazebo (`fit-sim`) | 10.3 h wall, ~21 core-h | `results_93.csv` E19-fit_sim_gazebo | **~$1.30** |
| HoloOcean refinement, per segment | 10.9 h (315 evaluations at ~124 s each, measured Aug 2026) | Earlier HoloOcean segment run, `PROJECT_STATUS.md` | **$9.89 on a T4** container with 4 cores + 16 GiB ($0.907/h); $12.16 on an L4 ($1.116/h) |
| Fleet policy training, 3M steps | ~41 min on one core | `fleet_policy_seed*.json` | ~$0.04 |

HoloOcean has **not run in a Linux container yet**. It still needs the Epic-linked build (the Fin Steer plan's open
HoloOcean step), Unreal running headless, and a GPU driver inside Modal's image. Until it runs there, the HoloOcean line
is a price for a job that might not work on Modal. A T4 being enough is also unconfirmed.

## Cost per lab calibration

| What the lab gets | Modal compute |
|---|---|
| Hover vehicle, stand-in physics only | **~$0.45** (range $0.10 to $1.83) |
| Fin vehicle, stand-in physics only | **~$2.55** (range $0.80 to $6.93) |
| Add a simulator fit (Gazebo; Stonefish has the same shape but isn't timed on its own) | +~$1.30 |
| Add HoloOcean refinement, 1 to 5 segments | +$9.89 to $49.43 (unverified, see above) |
| Storage: logs, checkpoints, report | Inside Modal's free 1 TiB a month for any likely volume |

For scale:
- **Starter plan:** its $30 a month of included compute covers about 65 typical hover calibrations or about 12 typical fin
  calibrations.
- **Team plan:** needed only for more than 3 seats, more than 100 containers at once, or more than 10 GPUs at once.

## Re-running the proof suites on Modal

The stress suites are the largest single compute bill: every PR that changes the physics should rerun them.
- **#93 to #96 (hover):** ~287 wall hours of rows, so ~575 core-h, about **$36**.
- **#97 (Fin Steer):** ~323 wall hours, so ~646 core-h, about **$41**.
- **All five together:** about **$77** per full rerun, against roughly 185 hours on the laptop.

All three figures use the wall-time × 2 core estimate. These rows are independent, so on Modal they can run as separate
containers at once. The suite would then take about as long as its longest row, roughly 1 to 2 days, instead of a week.
Running them this way needs a runner script that doesn't exist yet.

## Things to get right when it's built

- **Use preemptible containers, not the 3x price.** Fits already save a checkpoint after every evaluation. After a
  preemption, `strathe calibrate --finish` (`--resume-from-checkpoint`) warm-starts from the best result so far. That
  only works if the checkpoint is written to a Modal Volume, not the container's own disk.
- **The warm start isn't an exact continuation.** A preempted fit restarts from its best parameters, so a preempted run
  and an uninterrupted one can end slightly differently. Results that need to be reproducible should record any preemption.
- **Don't pin a region** unless a lab's data rules require it: it adds 15% to 75%.
- **Stonefish and Gazebo** run in Docker images today (Colima on the laptop). They would need to be rebuilt as Modal images,
  and their outputs checked against the laptop numbers before any Modal result counts as proof.
- **Lab data leaves the lab's machine.** Uploading logs to Modal is a data-handling change, and labs have to be told.

## The main point is unchanged

Compute is cheap: about $0.45 to $2.55 for a typical calibration, and up to about $58 for a long fin fit with a simulator
fit and five HoloOcean segments. Engineer time is still the main cost, in diagnosing failed fits and checking rows before
believing them. Modal's per-second billing and free tier make compute close to free at pilot volume.

## Still unknown

- Our workers' real memory use on Linux. 2 GiB per core is assumed.
- Whether fits keep the same core-hours on more than 8 cores.
- Whether HoloOcean runs headless on Modal, and on which GPU.
- What a lab is charged: this is cost, not price. Fin calibration also shouldn't be sold at full price yet: its heading
  beats holding course on only one unseen real deployment, and narrowly (`LIMITS.md`).

*Before 12 Nov 2026 this file priced AWS on-demand instances: c5.xlarge at $0.17/h and g4dn.xlarge at $0.526/h,
checked Aug 2026. Git history has that version.*
