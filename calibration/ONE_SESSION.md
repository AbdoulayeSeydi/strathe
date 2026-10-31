# Calibrate from one field day

Strathe checks a calibration on recordings it never trained on. The best check holds out whole days: a new day has its
own trim, battery and water, as your next deployment will. But a lab often has one day of logs, or one long mission.
**Strathe calibrates from that too**, says how it checked itself, and says how much easier that check is.

```bash
python strathe.py calibrate --data my_one_day_logs --name my_rov       # nothing else to give
```

## How it splits your logs

Strathe picks the split from what your logs contain:

| Your logs | What's held out | The report's first line |
|---|---|---|
| Two or more days | The latest 30% of days (as before) | no warning |
| One day, two or more recordings | The latest 30% of the recordings, by run number (at least one) | ⚠️ held out within one session |
| One recording | Its last 30%, after a 10 s gap | ⚠️ held out within one session, with the part's length |

- **The gap** keeps the held-out part from starting where training ended, so the two never share a moment of motion.
- **Each part needs at least 60 s:** a recording shorter than 200 s can't be split at 30% (60 s held out). Strathe says so,
  with the length it needs.
- **The run's first log line** says how the split was made (`Split (by recordings)` or `Split (by time block)`).
  `metrics.json` records it, so `strathe report`, `strathe validate` and `--finish` reuse the same split.

## Choosing what to hold out

`--heldout-recordings` (the same option as `--heldout-dates`) takes recordings, or a recording's last part:

```bash
# hold out two recordings you drove differently
python strathe.py calibrate --data my_logs --name my_rov --heldout-recordings my_rov-05-01-2030-run3 my_rov-05-01-2030-run9
# one long mission: hold out its last 20%
python strathe.py calibrate --data my_logs --name my_rov --heldout-recordings my_rov-05-01-2030-run1::last20%
```

- **Names** are the CSV file names without `.csv`. `strathe ingest` writes them as `<vehicle>-<DD-MM-YYYY>-run<k>`.
- **Hold out recordings like the ones you'll simulate.** If you'll simulate surveys, hold out a survey, not a hover.
- **The same option** works on `fit-sim`, `fit-stonefish`, `export-stonefish`, `export-sim` and `report`. On the web
  page, type names into the "Held out" field. It checks that each one is in your logs folder and lists the ones it
  has.
- **Not allowed:**
  - dates and recordings together;
  - holding out every recording;
  - the last part of more than one recording;
  - the one date there is.

  Each is refused with a sentence saying what to do.

## How good is a one-session check?

The held-out recordings share the day's trim, battery, water and pilot, so a one-session check can't show how another
day changes those. We measured how much it matters: each calibration below was trained on one session, scored on
what it held out of that session, and then scored on sessions it never saw (the whole session, with its water current
estimated when the calibration estimated currents, as calibrate scores any held-out session). Skill: 1 is perfect, 0
is no better than assuming the vehicle doesn't move, averaged over the six axes.

| Calibrated on | Held out | Within the session: skill, drift after 30 s | Scored on | There: skill, drift after 30 s |
|---|---|---|---|---|
| BlueROV2, tank, 12 Jan alone | 16 recordings: the last 5 held out | +0.33, 1.4 m | the other 3 days | +0.37, 1.3 m |
| BlueROV2, tank, 14 Feb alone | 9 recordings: the last 3 | +0.39, 1.4 m | the other 3 days | +0.35, 1.4 m |
| BlueROV2, tank, 16 Feb alone | 8 recordings: the last 2 | +0.38, 1.4 m | the other 3 days | +0.34, 1.5 m |
| BlueROV2, tank, 19 Feb alone | 6 recordings: the last 2 | +0.39, 1.3 m | the other 3 days | +0.34, 1.4 m |
| ALPHA RISE, open water, the bay mission alone | 1 recording: its last 30% | +0.16, 2.4 m | the whole pond mission | +0.22, 2.2 m |
| ALPHA RISE, open water, the pond mission alone | 1 recording: its last 30% | +0.13, 5.7 m | the whole bay mission | +0.19, 3.3 m |

- **In a tank,** the one-day check read 0.04-0.05 higher than other days in three of four days, and 0.04 lower in one.
- **In open water, from one mission,** it read 0.06-0.07 *lower* than the other mission: here the held-out last part
  of each mission was harder to predict than the other mission as a whole. A model from one mission held up on the
  next as well as a two-mission calibration does (0.22 on the pond, against 0.23 for URI's calibration on the whole
  bay mission).

So a one-session score is a fair guide, within 0.07 either way on these vehicles. It still can't tell you about a
day with a different trim, battery or current: **record a second session for the honest number**, even a short one (a
different day, or the same day after re-trimming and changing the battery). Calibrate again with both; Strathe then
holds out a whole session.

## What changes in the report
- **A warning at the top:** "Held out within one session ...", with what was held out.
- **The skill sentence** says "recordings the calibration never saw (from the same session, not another day)",
  instead of "sessions".
- **Everything else is as usual:** per-axis verdicts, the comparison with the real vehicle (`validation/`), the
  identifiability check.

## Proof
PROJECT_STATUS.md #93 and `STRESS_TEST_93_95.md` have every check, on the BlueROV2 (each of its 4 days alone), DFKI's
BlueROV2 (its single session) and URI's ALPHA RISE (each mission alone).
- **Split checks:** the held-out rows never overlap training, re-read from the files; logs with two or more dates
  split exactly as before.
- **Deliberate faults** in the split are caught.
- **Every command that splits the logs** was run on a single session.
