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
- **Each part needs at least 60 s:** a recording shorter than about 3.5 minutes can't be split at 30%. Strathe says so,
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

## How much easier is a one-day check?

The held-out recordings share the day's trim, battery, water and pilot, so the scores look better than they will on
another day. On the real data, a calibration trained on one day scored better within that day than on the days it
never saw:

MEASURED_OPTIMISM_TABLE

So read a one-session report's skill as an upper bound. **To get the honest number, record a second session**, even a
short one: a different day, or the same day after re-trimming and changing the battery. Calibrate again with both;
Strathe then holds out a whole session.

## What changes in the report
- **A warning at the top:** "Held out within one session ...", with what was held out.
- **The skill sentence** says "recordings the calibration never saw (from the same session: an easier check)",
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
