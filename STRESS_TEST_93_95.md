# Stress test: #93-#95

Written by `calibration/stress/report.py` on 2026-10-27 16:37. Every variable each feature has is listed by `stress/inventory.py` from the code, and `tests/test_stress_coverage.py` fails if one has no row here.

## #93. One field day: hold out recordings, or a block of time

**106 of 107 rows pass** (36 variables found in the code, each with a row).

### How good a one-session check is (measured)

Each calibration trained on one session, scored on its own held-out part and on sessions it never saw (mean skill over the 6 axes against predicting no motion; position drift after 30 s, median).

| Calibrated on | Held out | Within the session: mean skill, drift 30 s | Other sessions: mean skill, drift 30 s | Within minus other (skill) |
|---|---|---|---|---|
| bluerov 12-01-2024 | recordings | +0.331, 1.39 m | +0.370, 1.34 m | -0.039 |
| bluerov 14-02-2024 | recordings | +0.393, 1.36 m | +0.353, 1.39 m | +0.040 |
| bluerov 16-02-2024 | recordings | +0.382, 1.43 m | +0.338, 1.48 m | +0.044 |
| bluerov 19-02-2024 | recordings | +0.392, 1.28 m | +0.341, 1.42 m | +0.051 |
| uri bay | time block | +0.158, 2.37 m | +0.218, 2.22 m | -0.060 |
| uri pond | time block | +0.126, 5.73 m | +0.193, 3.32 m | -0.067 |

### Fast checks on the real recordings

| Row | Varied | What | Data | Result | Numbers | Time |
|---|---|---|---|---|---|---|
| F1-bluerov_12-01-2024 | mode = recordings | auto split of bluerov_12-01-2024 (16 recordings, 1 date(s)) | bluerov_12-01-2024 | ✅ | {"overlap_rows": 0, "gap_rows": {}, "train": 11, "held": ["odom-12-01-2024-run12", "odom-12-01-2024-run13", "odom-12-01-2024-run14", "odom-12-01-2024-run15", "o... | 0 min |
| F1-bluerov_14-02-2024 | mode = recordings | auto split of bluerov_14-02-2024 (9 recordings, 1 date(s)) | bluerov_14-02-2024 | ✅ | {"overlap_rows": 0, "gap_rows": {}, "train": 6, "held": ["odom-14-02-2024-run7", "odom-14-02-2024-run8", "odom-14-02-2024-run9"], "mode": "recordings"} | 0 min |
| F1-bluerov_16-02-2024 | mode = recordings | auto split of bluerov_16-02-2024 (8 recordings, 1 date(s)) | bluerov_16-02-2024 | ✅ | {"overlap_rows": 0, "gap_rows": {}, "train": 6, "held": ["odom-16-02-2024-run7", "odom-16-02-2024-run8"], "mode": "recordings"} | 0 min |
| F1-bluerov_19-02-2024 | mode = recordings | auto split of bluerov_19-02-2024 (6 recordings, 1 date(s)) | bluerov_19-02-2024 | ✅ | {"overlap_rows": 0, "gap_rows": {}, "train": 4, "held": ["odom-19-02-2024-run5", "odom-19-02-2024-run6"], "mode": "recordings"} | 0 min |
| F1-dfki_one | mode = recordings | auto split of dfki_one (22 recordings, 1 date(s)) | dfki_one | ✅ | {"overlap_rows": 0, "gap_rows": {}, "train": 15, "held": ["dfki_bluerov2-01-06-2025-run16", "dfki_bluerov2-01-06-2025-run17", "dfki_bluerov2-01-06-2025-run18", ... | 0 min |
| F1-uri_bay | mode = time block | auto split of uri_bay (1 recordings, 1 date(s)) | uri_bay | ✅ | {"overlap_rows": 0, "gap_rows": {"alpha_rise-05-12-2025-run1.csv": 100}, "train": 1, "held": ["alpha_rise-05-12-2025-run1::last30%"], "mode": "time block"} | 0 min |
| F1-uri_pond | mode = time block | auto split of uri_pond (1 recordings, 1 date(s)) | uri_pond | ✅ | {"overlap_rows": 0, "gap_rows": {"alpha_rise-27-03-2026-run1.csv": 100}, "train": 1, "held": ["alpha_rise-27-03-2026-run1::last30%"], "mode": "time block"} | 0 min |
| F1-bluerov_all_days | mode = dates | auto split of bluerov_all_days (39 recordings, 4 date(s)) | bluerov_all_days | ✅ | {"overlap_rows": 0, "gap_rows": {}, "train": 33, "held": ["19-02-2024"], "mode": "dates"} | 0 min |
| F1-bluerov_12-01_2recs | mode = recordings | auto split of bluerov_12-01_2recs (2 recordings, 1 date(s)) | bluerov_12-01_2recs | ✅ | {"overlap_rows": 0, "gap_rows": {}, "train": 1, "held": ["odom-12-01-2024-run10"], "mode": "recordings"} | 0 min |
| F1-bluerov_12-01_3recs | mode = recordings | auto split of bluerov_12-01_3recs (3 recordings, 1 date(s)) | bluerov_12-01_3recs | ✅ | {"overlap_rows": 0, "gap_rows": {}, "train": 2, "held": ["odom-12-01-2024-run11"], "mode": "recordings"} | 0 min |
| F1-bluerov_12-01_5recs | mode = recordings | auto split of bluerov_12-01_5recs (5 recordings, 1 date(s)) | bluerov_12-01_5recs | ✅ | {"overlap_rows": 0, "gap_rows": {}, "train": 3, "held": ["odom-12-01-2024-run12", "odom-12-01-2024-run13"], "mode": "recordings"} | 0 min |
| F2-one | kind = recording | --heldout-recordings, one named | bluerov 12-01 | ✅ | {"overlap_rows": 0, "gap_rows": {}, "train": 15, "held": 1} | 0 min |
| F2-two | kind = recording | --heldout-recordings, two named | bluerov 12-01 | ✅ | {"overlap_rows": 0, "gap_rows": {}, "train": 14, "held": 2} | 0 min |
| F2-all-but-one | kind = recording | --heldout-recordings, all-but-one named | bluerov 12-01 | ✅ | {"overlap_rows": 0, "gap_rows": {}, "train": 1, "held": 15} | 0 min |
| F2-all | kind = recording | every recording named: refused | bluerov 12-01 | ✅ | {"message": "Every recording is held out: leave at least one to calibrate on."} | 0 min |
| F3-uri_bay-0.2 | kind = last part | last 20% of one recording | uri_bay | ✅ | {"rows": 5860, "held_rows": 1172, "train_rows": 4588, "gap_rows": {"alpha_rise-05-12-2025-run1.csv": 100}, "overlap_rows": 0, "train": 1, "held": 1, "rows_read_... | 0 min |
| F3-uri_bay-0.3 | kind = last part | last 30% of one recording | uri_bay | ✅ | {"rows": 5860, "held_rows": 1758, "train_rows": 4002, "gap_rows": {"alpha_rise-05-12-2025-run1.csv": 100}, "overlap_rows": 0, "train": 1, "held": 1, "rows_read_... | 0 min |
| F3-uri_bay-0.5 | kind = last part | last 50% of one recording | uri_bay | ✅ | {"rows": 5860, "held_rows": 2930, "train_rows": 2830, "gap_rows": {"alpha_rise-05-12-2025-run1.csv": 100}, "overlap_rows": 0, "train": 1, "held": 1, "rows_read_... | 0 min |
| F3-uri_pond-0.2 | kind = last part | last 20% of one recording | uri_pond | ✅ | {"rows": 11030, "held_rows": 2206, "train_rows": 8724, "gap_rows": {"alpha_rise-27-03-2026-run1.csv": 100}, "overlap_rows": 0, "train": 1, "held": 1, "rows_read... | 0 min |
| F3-uri_pond-0.3 | kind = last part | last 30% of one recording | uri_pond | ✅ | {"rows": 11030, "held_rows": 3309, "train_rows": 7621, "gap_rows": {"alpha_rise-27-03-2026-run1.csv": 100}, "overlap_rows": 0, "train": 1, "held": 1, "rows_read... | 0 min |
| F3-uri_pond-0.5 | kind = last part | last 50% of one recording | uri_pond | ✅ | {"rows": 11030, "held_rows": 5515, "train_rows": 5415, "gap_rows": {"alpha_rise-27-03-2026-run1.csv": 100}, "overlap_rows": 0, "train": 1, "held": 1, "rows_read... | 0 min |
| F3-bluerov 19-02 run1-0.2 | kind = last part | last 20% of one recording | bluerov 19-02 run1 | ✅ | {"rows": 6000, "held_rows": 1200, "train_rows": 4700, "gap_rows": {"odom-19-02-2024-run1.csv": 100}, "overlap_rows": 0, "train": 1, "held": 1, "rows_read_match"... | 0 min |
| F3-bluerov 19-02 run1-0.3 | kind = last part | last 30% of one recording | bluerov 19-02 run1 | ✅ | {"rows": 6000, "held_rows": 1800, "train_rows": 4100, "gap_rows": {"odom-19-02-2024-run1.csv": 100}, "overlap_rows": 0, "train": 1, "held": 1, "rows_read_match"... | 0 min |
| F3-bluerov 19-02 run1-0.5 | kind = last part | last 50% of one recording | bluerov 19-02 run1 | ✅ | {"rows": 6000, "held_rows": 3000, "train_rows": 2900, "gap_rows": {"odom-19-02-2024-run1.csv": 100}, "overlap_rows": 0, "train": 1, "held": 1, "rows_read_match"... | 0 min |
| F4-gap0 | const = TIME_BLOCK_GAP_S | gap of 0 s between the parts | uri_pond | ✅ | {"gap_rows": 0, "velocity_jump_across_gap_mps": 0.0139} | 0 min |
| F4-gap10 | const = TIME_BLOCK_GAP_S | gap of 10 s between the parts | uri_pond | ✅ | {"gap_rows": 100, "velocity_jump_across_gap_mps": 0.0291} | 0 min |
| F4-gap20 | const = TIME_BLOCK_GAP_S | gap of 20 s between the parts | uri_pond | ✅ | {"gap_rows": 200, "velocity_jump_across_gap_mps": 0.0323} | 0 min |
| F5-too_short | refusal = too short | refused: too short | bluerov | ✅ | {"message": "my_rov-05-01-2030-run1.csv is 100 s long: too short to hold out part of it. Calibration needs at least 60 s to train on and 60 s held out, 10 s apa... | 0 min |
| F5-unknown_recording | refusal = unknown recording | refused: unknown recording | bluerov | ✅ | {"message": "Unknown held-out recording(s): ['odom-12-01-2024-run99']. The recordings are: ['odom-12-01-2024-run1', 'odom-12-01-2024-run2', 'odom-12-01-2024-run... | 0 min |
| F5-date_and_recording | refusal = date and recording | refused: date and recording | bluerov | ✅ | {"message": "Hold out dates or recordings, not both: ['16-02-2024', 'odom-12-01-2024-run1']"} | 0 min |
| F5-the_only_date | refusal = the only date | refused: the only date | bluerov | ✅ | {"message": "All recordings are from 12-01-2024: there's no other day to calibrate on. Leave the held-out dates out and Strathe holds out recordings within the ... | 0 min |
| F5-last_0% | refusal = last 0% | refused: last 0% | bluerov | ✅ | {"message": "the held-out part of a recording must be strictly between 0 and 1, got 0.0"} | 0 min |
| F5-last_100% | refusal = last 100% | refused: last 100% | bluerov | ✅ | {"message": "the held-out part of a recording must be strictly between 0 and 1, got 1.0"} | 0 min |
| F5-last_part_of_two | refusal = last part of two | refused: last part of two | bluerov | ✅ | {"message": "hold out the last part of one recording only (e.g. my_rov-05-01-2030-run1::last30%)"} | 0 min |
| F5-percent_not_a_number | refusal = percent not a number | refused: percent not a number | bluerov | ✅ | {"message": "'odom-12-01-2024-run1::lastx%': give the held-out part as a percentage, e.g. ::last30%"} | 0 min |
| F5-unknown_date | refusal = unknown date | refused: unknown date | bluerov | ✅ | {"message": "Unknown heldout group(s) not present in the data: ['01-01-1999']. Available groups: ['12-01-2024', '14-02-2024', '16-02-2024', '19-02-2024']"} | 0 min |
| F5-fraction_outside_(0,_1) | refusal = fraction outside (0, 1) | refused: fraction outside (0, 1) | bluerov | ✅ | {"message": "heldout_fraction must be strictly between 0 and 1, got 1.0"} | 0 min |
| F6-bluerov_default | mode = dates | regression: bluerov default | bluerov default | ✅ | {"held": ["19-02-2024"]} | 0 min |
| F6-bluerov_16-02_19-02 | mode = dates | regression: bluerov 16-02 19-02 | bluerov 16-02 19-02 | ✅ | {"held": ["16-02-2024", "19-02-2024"]} | 0 min |
| F6-bluerov_12-01-2024 | mode = dates | regression: bluerov 12-01-2024 | bluerov 12-01-2024 | ✅ | {"held": ["12-01-2024"]} | 0 min |
| F6-bluerov_14-02-2024 | mode = dates | regression: bluerov 14-02-2024 | bluerov 14-02-2024 | ✅ | {"held": ["14-02-2024"]} | 0 min |
| F6-bluerov_16-02-2024 | mode = dates | regression: bluerov 16-02-2024 | bluerov 16-02-2024 | ✅ | {"held": ["16-02-2024"]} | 0 min |
| F6-bluerov_19-02-2024 | mode = dates | regression: bluerov 19-02-2024 | bluerov 19-02-2024 | ✅ | {"held": ["19-02-2024"]} | 0 min |
| F6-dfki_placeholder_dates | mode = dates | regression: dfki placeholder dates | dfki placeholder dates | ✅ | {"held": ["02-01-2025"]} | 0 min |
| F7-bluerov_12-01-2024 | mode = recordings | re-split from the stored held-out groups | bluerov_12-01-2024 | ✅ | {"stored": ["odom-12-01-2024-run12", "odom-12-01-2024-run13", "odom-12-01-2024-run14", "odom-12-01-2024-run15", "odom-12-01-2024-run16"]} | 0 min |
| F7-bluerov_14-02-2024 | mode = recordings | re-split from the stored held-out groups | bluerov_14-02-2024 | ✅ | {"stored": ["odom-14-02-2024-run7", "odom-14-02-2024-run8", "odom-14-02-2024-run9"]} | 0 min |
| F7-bluerov_16-02-2024 | mode = recordings | re-split from the stored held-out groups | bluerov_16-02-2024 | ✅ | {"stored": ["odom-16-02-2024-run7", "odom-16-02-2024-run8"]} | 0 min |
| F7-bluerov_19-02-2024 | mode = recordings | re-split from the stored held-out groups | bluerov_19-02-2024 | ✅ | {"stored": ["odom-19-02-2024-run5", "odom-19-02-2024-run6"]} | 0 min |
| F7-dfki_one | mode = recordings | re-split from the stored held-out groups | dfki_one | ✅ | {"stored": ["dfki_bluerov2-01-06-2025-run16", "dfki_bluerov2-01-06-2025-run17", "dfki_bluerov2-01-06-2025-run18", "dfki_bluerov2-01-06-2025-run19", "dfki_bluero... | 0 min |
| F7-uri_bay | mode = time block | re-split from the stored held-out groups | uri_bay | ✅ | {"stored": ["alpha_rise-05-12-2025-run1::last30%"]} | 0 min |
| F7-uri_pond | mode = time block | re-split from the stored held-out groups | uri_pond | ✅ | {"stored": ["alpha_rise-27-03-2026-run1::last30%"]} | 0 min |
| F7-bluerov_all_days | mode = dates | re-split from the stored held-out groups | bluerov_all_days | ✅ | {"stored": ["19-02-2024"]} | 0 min |
| F7-bluerov_12-01_2recs | mode = recordings | re-split from the stored held-out groups | bluerov_12-01_2recs | ✅ | {"stored": ["odom-12-01-2024-run10"]} | 0 min |
| F7-bluerov_12-01_3recs | mode = recordings | re-split from the stored held-out groups | bluerov_12-01_3recs | ✅ | {"stored": ["odom-12-01-2024-run11"]} | 0 min |
| F7-bluerov_12-01_5recs | mode = recordings | re-split from the stored held-out groups | bluerov_12-01_5recs | ✅ | {"stored": ["odom-12-01-2024-run12", "odom-12-01-2024-run13"]} | 0 min |
| F8-format | web = all | web held-out field: valid entries pass, malformed ones are refused | - | ✅ | {"problems": []} | 0 min |
| F8-folder | web = all | a name not in the logs folder is refused, listing the real ones | bluerov 12-01 | ✅ | {"message": "Held-out 'odom-01-01-1999-run1' isn't a date or recording of the logs in /Users/abdoulayeseydi/strathe_session_data/bluerov_12-01-2024. Dates: 12-0... | 0 min |
| F8-cal-heldout | web = cal-heldout | web field cal-heldout passes a recording name to the command | - | ✅ | {"argv_tail": ["--name", "x", "--out", "/Users/abdoulayeseydi/strathe_session_wt/calibration_output_x", "--heldout-dates", "odom-12-01-2024-run3"]} | 0 min |
| F8-sf-heldout | web = sf-heldout | web field sf-heldout passes a recording name to the command | - | ✅ | {"argv_tail": ["0.0", "0.0", "--data", "/Users/abdoulayeseydi/strathe_session_data/bluerov_12-01-2024", "--heldout-dates", "odom-12-01-2024-run3"]} | 0 min |
| F8-simf-heldout | web = simf-heldout | web field simf-heldout passes a recording name to the command | - | ✅ | {"argv_tail": ["0.0", "0.0", "--data", "/Users/abdoulayeseydi/strathe_session_data/bluerov_12-01-2024", "--heldout-dates", "odom-12-01-2024-run3"]} | 0 min |
| F9-calibration_report.py:ap | flag = calibration_report.py:ap | --heldout-recordings is --heldout-dates' alias | calibration_report.py | ✅ | {"calls": 1} | 0 min |
| F9-finalize_checkpoint.py:ap | flag = finalize_checkpoint.py:ap | --heldout-recordings is --heldout-dates' alias | finalize_checkpoint.py | ✅ | {"calls": 1} | 0 min |
| F9-run_calibration.py:parser | flag = run_calibration.py:parser | --heldout-recordings is --heldout-dates' alias | run_calibration.py | ✅ | {"calls": 1} | 0 min |
| F9-simfit/runner.py:ap | flag = simfit/runner.py:ap | --heldout-recordings is --heldout-dates' alias | runner.py | ✅ | {"calls": 1} | 0 min |
| F9-stonefish_export.py:ap | flag = stonefish_export.py:ap | --heldout-recordings is --heldout-dates' alias | stonefish_export.py | ✅ | {"calls": 1} | 0 min |
| F9-stonefish_fit.py:ap | flag = stonefish_fit.py:ap | --heldout-recordings is --heldout-dates' alias | stonefish_fit.py | ✅ | {"calls": 1} | 0 min |
| F9-strathe_cli.py:c | flag = strathe_cli.py:c | --heldout-recordings is --heldout-dates' alias | strathe_cli.py | ✅ | {"calls": 1} | 0 min |
| F9-strathe_cli.py:r | flag = strathe_cli.py:r | --heldout-recordings is --heldout-dates' alias | strathe_cli.py | ✅ | {"calls": 1} | 0 min |
| F9-strathe_cli.py:sf | flag = strathe_cli.py:sf | --heldout-recordings is --heldout-dates' alias | strathe_cli.py | ✅ | {"calls": 1} | 0 min |
| F9-strathe_cli.py:ex | flag = strathe_cli.py:ex | --heldout-recordings is --heldout-dates' alias | strathe_cli.py | ✅ | {"calls": 1} | 0 min |
| F9-parse-calibrate | flag = calibrate | strathe calibrate --heldout-recordings parses | - | ✅ | {"heldout_dates": ["odom-12-01-2024-run3", "x::last20%"]} | 0 min |
| F9-parse-report | flag = report | strathe report --heldout-recordings parses | - | ✅ | {"heldout_dates": ["odom-12-01-2024-run3", "x::last20%"]} | 0 min |
| F9-parse-fit-stonefish | flag = fit-stonefish | strathe fit-stonefish --heldout-recordings parses | - | ✅ | {"heldout_dates": ["odom-12-01-2024-run3", "x::last20%"]} | 0 min |
| F9-parse-export-stonefish | flag = export-stonefish | strathe export-stonefish --heldout-recordings parses | - | ✅ | {"heldout_dates": ["odom-12-01-2024-run3", "x::last20%"]} | 0 min |
| F10-density | consumer = stonefish_export.py | --density-by-date with recording groups | - | ✅ | {"got": [1000, 1000, 1025]} | 0 min |

### Deliberate faults (each must be caught)

| Row | Varied | What | Data | Result | Numbers | Time |
|---|---|---|---|---|---|---|
| M-time_block_off_by_one | mutant = time block off by one | deliberate fault: time block off by one | fast rows | ✅ | {"rows_failed": ["F1-uri_bay", "F1-uri_pond", "F3-uri_bay-0.2", "F3-uri_bay-0.3", "F3-uri_bay-0.5", "F3-uri_pond-0.2", "F3-uri_pond-0.3", "F3-uri_pond-0.5"], "n... | 0 min |
| M-held-out_recording_leaks_into_training | mutant = held-out recording leaks into training | deliberate fault: held-out recording leaks into training | fast rows | ✅ | {"rows_failed": ["F1-bluerov_12-01-2024", "F1-bluerov_14-02-2024", "F1-bluerov_16-02-2024", "F1-bluerov_19-02-2024", "F1-dfki_one", "F1-bluerov_12-01_2recs", "F... | 0 min |
| M-rows_of_a_part_ignored | mutant = rows of a part ignored | deliberate fault: rows of a part ignored | fast rows | ✅ | {"rows_failed": ["crash: ValueError: operands could not be broadcast together with shapes (5860,6) (1172,6) "], "n_failed": 1} | 0 min |
| M-held-out_count_wrong | mutant = held-out count wrong | deliberate fault: held-out count wrong | fast rows | ✅ | {"rows_failed": ["F1-bluerov_12-01-2024", "F1-bluerov_14-02-2024", "F1-bluerov_16-02-2024", "F1-bluerov_19-02-2024", "F1-dfki_one", "F1-bluerov_12-01_3recs", "F... | 0 min |

### End to end on real data

| Row | Varied | What | Data | Result | Numbers | Time |
|---|---|---|---|---|---|---|
| E1-bluerov_12-01-2024 | dataset = bluerov 12-01-2024 | calibrate on one day (12-01-2024), recordings held out | bluerov_12-01-2024 | ✅ | {"exit": 0, "mode": "recordings", "held": ["odom-12-01-2024-run12", "odom-12-01-2024-run13", "odom-12-01-2024-run14", "odom-12-01-2024-run15", "odom-12-01-2024-... | 19 min |
| E2-bluerov_14-02-2024 | dataset = bluerov 14-02-2024 | calibrate on one day (14-02-2024), recordings held out | bluerov_14-02-2024 | ✅ | {"exit": 0, "mode": "recordings", "held": ["odom-14-02-2024-run7", "odom-14-02-2024-run8", "odom-14-02-2024-run9"], "converged": true, "within": {"skill": [0.67... | 66 min |
| E3-bluerov_16-02-2024 | dataset = bluerov 16-02-2024 | calibrate on one day (16-02-2024), recordings held out | bluerov_16-02-2024 | ✅ | {"exit": 0, "mode": "recordings", "held": ["odom-16-02-2024-run7", "odom-16-02-2024-run8"], "converged": true, "within": {"skill": [0.747, 0.456, 0.56, 0.015, -... | 36 min |
| E4-bluerov_19-02-2024 | dataset = bluerov 19-02-2024 | calibrate on one day (19-02-2024), recordings held out | bluerov_19-02-2024 | ✅ | {"exit": 0, "mode": "recordings", "held": ["odom-19-02-2024-run5", "odom-19-02-2024-run6"], "converged": true, "within": {"skill": [0.762, 0.602, 0.622, -0.079,... | 25 min |
| E5-dfki_one | dataset = dfki one session | calibrate DFKI's single session | dfki_one | ✅ | {"exit": 0, "mode": "recordings", "held": ["dfki_bluerov2-01-06-2025-run16", "dfki_bluerov2-01-06-2025-run17", "dfki_bluerov2-01-06-2025-run18", "dfki_bluerov2-... | 29 min |
| E6-uri_bay | dataset = uri bay | calibrate URI's bay mission alone (time block) | uri_bay | ✅ | {"exit": 0, "mode": "time block", "held": ["alpha_rise-05-12-2025-run1::last30%"], "converged": true, "within": {"skill": [0.871, 0.117, 0.032, 0.012, -0.33, 0.... | 37 min |
| E7-uri_pond | dataset = uri pond | calibrate URI's pond mission alone (time block) | uri_pond | ✅ | {"exit": 0, "mode": "time block", "held": ["alpha_rise-27-03-2026-run1::last30%"], "converged": true, "within": {"skill": [0.757, 0.363, -0.399, 0.002, 0.008, 0... | 110 min |
| E8-named | kind = recording | --heldout-recordings, two named | bluerov_12-01-2024 | ❌ | {"exit": 0, "mode": "recordings", "held": ["odom-12-01-2024-run3", "odom-12-01-2024-run9"], "converged": true, "within": {"skill": [0.308, -1.157, -1.05, -9.524... | 76 min |
| E8b-retry | fit = failed start retried | E8 again, same seed: a failed single start is retried | bluerov_12-01-2024 | ✅ | {"exit": 0, "mode": "recordings", "held": ["odom-12-01-2024-run3", "odom-12-01-2024-run9"], "converged": true, "within": {"skill": [0.638, 0.429, 0.477, -0.08, ... | 225 min |
| E8b-retry-check | fit = auto restart | the failed start was noticed and retried | bluerov_12-01-2024 | ✅ | {"auto_restart": {"reason": "the first fit ended no better than the starting description on its training recordings", "why": "no_better", "first_cost": 1598.968... | 0 min |
| E9-last20 | kind = last part | --heldout-recordings ...::last20% | uri_pond | ✅ | {"exit": 0, "mode": "time block", "held": ["alpha_rise-27-03-2026-run1::last20%"], "converged": false, "within": {"skill": [0.765, 0.337, -0.678, 0.017, 0.019, ... | 325 min |
| E10-last50 | kind = last part | --heldout-recordings ...::last50% | uri_pond | ✅ | {"exit": 0, "mode": "time block", "held": ["alpha_rise-27-03-2026-run1::last50%"], "converged": false, "within": {"skill": [0.827, 0.416, -0.201, 0.009, 0.004, ... | 94 min |
| E11-2recs | recordings available = 2 | one day with 2 recordings | bluerov_12-01-2024_2recs | ✅ | {"exit": 0, "mode": "recordings", "held": ["odom-12-01-2024-run10"], "converged": true, "within": {"skill": [0.554, 0.352, 0.445, 0.004, -0.008, 0.648], "mean_s... | 4 min |
| E12-3recs | recordings available = 3 | one day with 3 recordings | bluerov_12-01-2024_3recs | ✅ | {"exit": 0, "mode": "recordings", "held": ["odom-12-01-2024-run11"], "converged": true, "within": {"skill": [0.609, 0.568, 0.517, -0.163, -0.15, 0.479], "mean_s... | 10 min |
| E13-5recs | recordings available = 5 | one day with 5 recordings | bluerov_12-01-2024_5recs | ✅ | {"exit": 0, "mode": "recordings", "held": ["odom-12-01-2024-run12", "odom-12-01-2024-run13"], "converged": true, "within": {"skill": [0.724, 0.462, 0.513, -0.15... | 10 min |
| E14-restarts4 | fit = --restarts 4 | one day, 4 starting points | bluerov_19-02-2024 | ✅ | {"exit": 0, "mode": "recordings", "held": ["odom-19-02-2024-run5", "odom-19-02-2024-run6"], "converged": true, "within": {"skill": [0.762, 0.602, 0.622, -0.079,... | 53 min |
| E15-no_current | fit = no --estimate-current | URI bay alone, current not estimated | uri_bay | ✅ | {"exit": 0, "mode": "time block", "held": ["alpha_rise-05-12-2025-run1::last30%"], "converged": true, "within": {"skill": [0.803, -0.097, -0.171, 0.009, -0.025,... | 7 min |
| E16-finish | consumer = finalize_checkpoint.py | --finish on a one-session fit | bluerov_19-02-2024 | ✅ | {"exit": [0, 0], "dataset": {"train_dates": ["odom-19-02-2024-run1", "odom-19-02-2024-run2", "odom-19-02-2024-run3", "odom-19-02-2024-run4"], "heldout_dates": [... | 3 min |
| E17-report_validate | consumer = calibration_report.py validation.py | strathe report and validate, standalone, on a one-session calibration | bluerov_19-02-2024 | ✅ | {"exit": [0, 0], "held": ["odom-19-02-2024-run5", "odom-19-02-2024-run6"]} | 0 min |
| E18-fit_sim_mujoco | consumer = simfit/runner.py simfit/core.py | fit-sim on one session | bluerov_19-02-2024 | ✅ | {"exit": 0, "report": true, "tail": ["  MuJoCo, calibrated by Strathe        drift 3.50 / 3.93 / 4.23 m", "  no motion                            drift 3.35 / 3... | 10 min |
| E19-fit_sim_gazebo | consumer = simfit (gazebo, Docker) | fit-sim on one session | bluerov_19-02-2024 | ✅ | {"exit": 0, "report": true, "tail": ["  Gazebo, calibrated by Strathe        drift 4.06 / 10.85 / 20.00 m", "  no motion                            drift 3.35 /... | 8 min |
| E20-fit_stonefish | consumer = stonefish_fit.py | fit-stonefish on one session | uri_bay | ✅ | {"exit": 0, "report": true, "tail": ["  Stonefish, calibrated by Strathe     drift 3.43 / 14.30 / 35.38 m", "  no motion                            drift 6.99 /... | 4 min |
| E21-export_stonefish | consumer = stonefish_export.py | export-stonefish on one session | uri_pond | ✅ | {"exit": 0, "report": true, "tail": ["  With the patch's damping, Stonefish runs Strathe's own model, with the centre of gravity at the logs' reference point as... | 4 min |
| E22-web | web = cal-heldout | web calibrate card, one session, a named recording | bluerov_16-02-2024 | ✅ | {"exit": 0, "tail": ["", "It is trustworthy for 2 of 6 motion axes: forward speed (surge), turning rate (yaw). On recordings the calibration never saw (from the... | 5 min |
| E23-sound_untouched | fit = sound fit untouched | E4 again, same seed: probed, not retried | bluerov_19-02-2024 | ✅ | {"exit": 0, "mode": "recordings", "held": ["odom-19-02-2024-run5", "odom-19-02-2024-run6"], "converged": true, "within": {"skill": [0.762, 0.602, 0.622, -0.079,... | 33 min |
| E23-sound_untouched-check | fit = sound fit untouched | no retry, same cost to the last bit | bluerov_19-02-2024 | ✅ | {"cost": 7.102579181661516, "E4_cost": 7.102579181661516, "retried": false} | 0 min |
| E24-wandering | fit = wandering fit retried | #94's CSV route, seed 94: a fit that wandered out of budget | ingested | ✅ | {"exit": 0, "mode": "dates", "held": ["16-02-2024", "19-02-2024"], "converged": true, "within": {"skill": [0.723, 0.508, 0.59, -0.026, -0.103, 0.581], "mean_ski... | 824 min |
| E24-wandering-check | fit = wandering fit retried | caught as rough, retried, a better fit kept | ingested | ✅ | {"auto_restart": {"reason": "the first fit ended where the training cost is rough (a chaotic replay)", "why": "rough", "first_cost": 102.50482563582999, "first_... | 0 min |

## #94. PX4, MOOS-IvP and LCM logs, read directly

**349 of 349 rows pass** (294 variables found in the code, each with a row).


### Deliberate faults (each must be caught)

| Row | Varied | What | Data | Result | Numbers | Time |
|---|---|---|---|---|---|---|
| M-frame_sign_flipped | mutant = frame sign flipped | deliberate fault: frame sign flipped | round trips | ✅ | {"formats_failed": ["px4", "moos", "lcm"], "formats_it_touches": ["lcm", "moos", "px4"]} | 0 min |
| M-PX4_time_unit_wrong_by_1000 | mutant = PX4 time unit wrong by 1000 | deliberate fault: PX4 time unit wrong by 1000 | round trips | ✅ | {"formats_failed": ["px4"], "formats_it_touches": ["px4"]} | 0 min |
| M-array_index_dropped | mutant = array index dropped | deliberate fault: array index dropped | round trips | ✅ | {"formats_failed": ["px4"], "formats_it_touches": ["px4"]} | 0 min |

### End to end on real data

| Row | Varied | What | Data | Result | Numbers | Time |
|---|---|---|---|---|---|---|
| EP-px4-ingest | entry = strathe ingest | px4: strathe ingest --spec | odom-12-01-2024-run1.csv | ✅ | {"exit": 0, "files": 1} | 0 min |
| EP-px4-web | entry = web Convert logs | px4: the web page's Convert logs card | odom-12-01-2024-run1.csv | ✅ | {"exit": 0} | 0 min |
| EP-moos-ingest | entry = strathe ingest | moos: strathe ingest --spec | odom-12-01-2024-run1.csv | ✅ | {"exit": 0, "files": 1} | 0 min |
| EP-moos-web | entry = web Convert logs | moos: the web page's Convert logs card | odom-12-01-2024-run1.csv | ✅ | {"exit": 0} | 0 min |
| EP-lcm-ingest | entry = strathe ingest | lcm: strathe ingest --spec | odom-12-01-2024-run1.csv | ✅ | {"exit": 0, "files": 1} | 0 min |
| EP-lcm-web | entry = web Convert logs | lcm: the web page's Convert logs card | odom-12-01-2024-run1.csv | ✅ | {"exit": 0} | 0 min |
| EP-px4-calibrate | entry = calibrate --ingest-spec | px4: calibrate straight from the logs | bluerov 3 days | ✅ | {"exit": 0} | 4 min |
| EP-moos-calibrate | entry = calibrate --ingest-spec | moos: calibrate straight from the logs | bluerov 3 days | ✅ | {"exit": 0} | 5 min |
| EP-lcm-calibrate | entry = calibrate --ingest-spec | lcm: calibrate straight from the logs | bluerov 3 days | ✅ | {"exit": 0} | 4 min |
| E-bluerov-px4-cal | calibration = px4 | calibrate bluerov from the px4 route | bluerov | ✅ | {"exit": 0} | 111 min |
| E-bluerov-csv-cal | calibration = csv | calibrate bluerov from the csv route | bluerov | ✅ | {"exit": 0} | 189 min |
| E-bluerov-moos-cal | calibration = moos | calibrate bluerov from the moos route | bluerov | ✅ | {"exit": 0} | 236 min |
| E-bluerov-px4 | calibration = px4 | bluerov: the calibration problem from px4 vs from the CSV route | bluerov | ✅ | {"train_recordings": [25, 25], "cost_csv": {"start": 3329.5684284318904, "perturbed start (seed 94)": 3013.271714074699, "calibrated (px4 route)": 77.6467986452... | 2 min |
| E-bluerov-moos | calibration = moos | bluerov: the calibration problem from moos vs from the CSV route | bluerov | ✅ | {"train_recordings": [25, 25], "cost_csv": {"start": 3329.5684284318904, "perturbed start (seed 94)": 3013.271714074699, "calibrated (px4 route)": 77.6467986452... | 4 min |
| E-bluerov-lcm | calibration = lcm | bluerov: the calibration problem from lcm vs from the CSV route | bluerov | ✅ | {"train_recordings": [25, 25], "cost_csv": {"start": 3329.5684284318904, "perturbed start (seed 94)": 3013.271714074699, "calibrated (px4 route)": 77.6467986452... | 6 min |
| E-uri-px4 | calibration = px4 | uri: the calibration problem from px4 vs from the CSV route | uri | ✅ | {"train_recordings": [1, 1], "cost_csv": {"start": 367.6997036506698, "perturbed start (seed 94)": 473.17900130724735}, "cost_px4": {"start": 367.69970365066933... | 1 min |
| E-uri-moos | calibration = moos | uri: the calibration problem from moos vs from the CSV route | uri | ✅ | {"train_recordings": [1, 1], "cost_csv": {"start": 367.6997036506698, "perturbed start (seed 94)": 473.17900130724735}, "cost_moos": {"start": 367.6997036506693... | 1 min |
| E-uri-lcm | calibration = lcm | uri: the calibration problem from lcm vs from the CSV route | uri | ✅ | {"train_recordings": [1, 1], "cost_csv": {"start": 367.6997036506698, "perturbed start (seed 94)": 473.17900130724735}, "cost_lcm": {"start": 367.69970365066933... | 1 min |

