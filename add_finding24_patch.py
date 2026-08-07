"""
Safe, self-verifying patch: adds finding #24 to PROJECT_STATUS.md,
inserted right before "## Honest open items" — same discipline as every
other patch tonight, refuses if the anchor isn't found exactly once.
"""
with open("PROJECT_STATUS.md", encoding="utf-8") as f:
    content = f.read()

anchor = "## Honest open items"

finding_24 = '''### 24. Finding #23's implied fix tested directly against real HoloOcean physics — and refuted, correcting course before acting on a misleading comparison

Finding #23's comparison (`compare_two_calibrations.py`) suggested the stand-in's own CONVERGED Coriolis fit might already be better than the unconverged HoloOcean-refined checkpoint, which would have meant further HoloOcean refinement wasn't actually necessary. Before committing to a large refinement run (or abandoning one) on that basis, tested it directly: ran the ACTUAL Claim A methodology — `replay_comparison.py` against real held-out data through real HoloOcean physics — using the stand-in's converged fit as the "calibrated" arm, instead of the HoloOcean-refined checkpoint.

**Result: WORSE than finding #20's original result, not better.** `p: -22.6%`, `r: -29.3%` (stand-in's converged fit vs. default, through real HoloOcean physics) — clearly worse than the unconverged HoloOcean-refined checkpoint's own `p: -7.2%`, `r: -12.9%`.

**Why finding #23's comparison was misleading, precisely:** it evaluated BOTH parameter sets using the stand-in's OWN physics — really answering "which parameters does the stand-in's simplified dynamics prefer," not "which parameters actually perform better against real HoloOcean physics." A parameter set optimized to look good under the stand-in's own equations doesn't necessarily transfer to real HoloOcean's actual rigid-body integration, even though the two share the same underlying acceleration equations.

**Corrected conclusion:** the unconverged HoloOcean-refined checkpoint IS genuinely better than the stand-in's own converged fit, when judged the way that actually counts (real HoloOcean physics, the Claim A standard). This REVERSES finding #23's implied direction — rather than being unnecessary, further HoloOcean refinement is confirmed to be the right, well-motivated next step, not just persistence. Proceeding with a properly-sized HoloOcean refinement run on the existing checkpoint (see "What's actually next").

'''

occurrences = content.count(anchor)
if occurrences != 1:
    print(f"REFUSING TO PATCH — expected exactly 1 occurrence of the anchor, found {occurrences}. "
          f"Do NOT proceed — paste the current 'Honest open items' section so this can be redone correctly.")
else:
    new_content = content.replace(anchor, finding_24 + anchor, 1)
    with open("PROJECT_STATUS.md", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Finding #24 inserted successfully.")