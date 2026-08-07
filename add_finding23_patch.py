"""
Safe, self-verifying patch: adds finding #23 to PROJECT_STATUS.md,
inserted right before the "## Honest open items" section. Only touches
the file if the anchor is found EXACTLY ONCE — refuses otherwise, same
discipline as tonight's other patches, given how much confusion stale
files already caused.
"""
with open("PROJECT_STATUS.md", encoding="utf-8") as f:
    content = f.read()

anchor = "## Honest open items"

finding_23 = '''### 23. The Stage 2.3 "mixed" rotational result — isolated to a specific, fixable cause, not a fundamental limitation

Directly tested whether the incomplete HoloOcean refinement itself (finding #19, stopped early due to diminishing AGGREGATE cost returns) was responsible for the mixed rotational picture (finding #20: `p -7.2%`, `r -12.9%`), rather than Coriolis or the calibration approach being fundamentally limited. Built `compare_two_calibrations.py` — directly compares two calibrated parameter sets against real held-out data using the fast stand-in backend (seconds, not hours), specifically isolating the CONVERGED stand-in Coriolis fit (finding #18) against the UNCONVERGED HoloOcean-refined result (finding #19) that Stage 2.3 actually used.

**Result: every single rotational DOF regressed from the incomplete refinement, not just `p`/`r`** — `p: -14.6%`, `q: -9.7%`, `r: -17.7%` (stand-in's converged fit vs. the unconverged HoloOcean-refined one). Translational axes were comparatively unaffected (`u: +2.5%`, `v: -1.8%`, `w: -6.4%`).

**Conclusion, precisely stated:** the mixed Stage 2.3 result is NOT evidence that Coriolis or the calibration methodology has a fundamental rotational limitation. It's an artifact of stopping HoloOcean refinement early based on the AGGREGATE cost curve's diminishing returns — which masked the fact that rotational and translational DOFs were moving in OPPOSITE directions underneath that aggregate number. The earlier "diminishing returns, stop here" decision (finding #19) was reasonable given the information available at the time, but is now understood to have been premature specifically for the rotational axes. This directly motivates properly re-running the HoloOcean refinement to genuine convergence (see "What's actually next"), rather than treating the current placeholder result as final.

'''

occurrences = content.count(anchor)
if occurrences != 1:
    print(f"REFUSING TO PATCH — expected exactly 1 occurrence of the anchor, found {occurrences}. "
          f"Do NOT proceed — paste the current 'Honest open items' section so this can be redone correctly.")
else:
    new_content = content.replace(anchor, finding_23 + anchor, 1)
    with open("PROJECT_STATUS.md", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Finding #23 inserted successfully.")