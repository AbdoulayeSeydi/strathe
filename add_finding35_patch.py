with open("PROJECT_STATUS.md", encoding="utf-8") as f:
    content = f.read()

anchor = "## Honest open items"

finding_35 = '''### 35. Garbage-input robustness (item #2 follow-up) - three new pre-flight checks, genuinely wired into both real product paths

Extended preflight_check.py with three new checks, deliberately split by confidence level rather than one blended threshold - a real design correction made along the way: an initial plan to tie FAIL-level rejection thresholds to this session's own observed training-data standard deviations was reconsidered and reworked, since a future, legitimately more energetic real dataset could exceed such a threshold without being wrong. The corrected design separates two genuinely different kinds of check:

check_schema.py (FAIL-level, high confidence) - fast, header-only validation that every CSV has the columns objective.py's own REQUIRED_COLUMNS actually needs. Deliberately imports that list directly rather than maintaining a second, separately-typed copy - the same no-duplicated-schema principle already established elsewhere in this project. Tested against the real, confirmed 70-column dataset header (PASS) and a deliberately broken one missing a required column (correctly FAILs with the right column name named).

check_data_integrity.py (FAIL-level, high confidence) - structural checks that don't depend on units or vehicle context at all: empty/near-empty files, header-only files, malformed CSV rows (wrong column count), non-numeric values in numeric fields, and NaN/Inf poisoning. Tested against seven real fixtures covering every category (clean data, empty file, header-only, too-few-rows, malformed row, non-numeric cell, NaN/Inf) - all seven produced the correct verdict with the correct real exit code.

check_physical_plausibility.py (two-tier: FAIL for absolute physical impossibility, WARNING for statistical anomalies only) - the corrected design: angular rates exceeding an absolute, physically-motivated ceiling (15 rad/s, chosen specifically to sit comfortably above this project's own real confirmed maximum of ~4 rad/s, so it can never false-positive against genuine data) are a real FAIL, directly catching the well-known degrees/s-vs-radians/s unit mistake without depending on any dataset's own statistics. Values beyond 8x this project's own previously-observed per-DOF standard deviation are WARNING-only - an anomaly flag, explicitly not a rejection criterion. Linear-velocity limits tied to the configured vehicle's actual capability were deliberately deferred (no authoritative limit currently exists in vehicle_spec.yaml) rather than invented - flagged as a real, concrete upgrade for when such limits are added there. Verified against four hand-constructed scenarios (genuine real data at its real observed maximum, a synthetic degrees-mislabeled-as-radians case, a moderate real outlier, and a linear-DOF statistical outlier) before trusting it against real data - all four produced the correct tier (FAIL/WARNING/clean).

Run against the full real 39/40-file dataset, all three new checks pass cleanly, with check_physical_plausibility.py correctly surfacing real WARNING-level outliers concentrated specifically on p and q - a pattern that independently lines up with this session's other findings about which rotational axes behave unusually, a real, if informal, cross-check that the tool is measuring something genuine.

A real duplicate-insertion bug, same class as finding #34's - wiring the three new checks into preflight_check.py's main() was accidentally run twice, silently duplicating the three new run_check() calls (harmless in that each check would just run and overwrite its own dict entry, but wasteful and confusing). Caught via grep -c before testing, exactly the same discipline finding #34 already established; fixed with a precise, targeted removal verified against a realistic duplicated fixture before touching the real file.

Confirmed genuinely wired into BOTH real product paths, not just standalone: preflight_check.py is already the automatic first step of both run_full_pipeline.py (the demo) and the documented real workflow (README.md step 1) - since the three new checks live inside that same file rather than needing a second flag threaded through a separate calling script, both paths picked them up automatically. Directly confirmed live via a real run_full_pipeline.py invocation showing all five checks (the original two plus these three) executing in order before calibration begins.

'''

occurrences = content.count(anchor)
if occurrences != 1:
    print("REFUSING TO PATCH - expected exactly 1 occurrence, found", occurrences)
else:
    new_content = content.replace(anchor, finding_35 + anchor, 1)
    with open("PROJECT_STATUS.md", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Finding #35 inserted successfully.")
