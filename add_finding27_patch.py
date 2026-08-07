# Safe, self-verifying patch: adds finding #27 to PROJECT_STATUS.md.
with open("PROJECT_STATUS.md", encoding="utf-8") as f:
    content = f.read()

anchor = "## Honest open items"

finding_27 = '''### 27. Three refinement segments, two different objectives, one consistent result — DOF weighting ruled out as the fix for r; truncated-run overfitting is the leading remaining hypothesis

Following finding #25's regression (p/r worse after an unweighted HoloOcean refinement segment), researched and implemented custom per-DOF weighting in the optimizer (`CalibrationConfig.dof_scales`) — boosting p and r beyond their natural training-data scales, specifically because r already has the LARGEST natural scale of all 6 DOFs (0.5625) yet was still regressing, meaning standard scale-normalization would have REDUCED its weight below today's already-implicit level (confirmed directly: 0.67x vs. today's 1.5x) rather than helping. Verified the weighting mechanism was genuinely active (custom scales printed and matched expectations) before trusting a long run to it.

**Result across all three real segments, same 150-step-truncated training setup:**

| DOF | Segment 3 | Segment 4 (unweighted) | Segment 5 (r boosted 4.46x) |
|---|---|---|---|
| p | -8.0% | -19.9% | -16.9% |
| q | +24.9% | +34.9% | +35.4% |
| r | -8.3% | -24.1% | **-36.0%** |

**r got monotonically worse across all three segments, including the one where its objective weight was deliberately quadrupled relative to today's baseline.** This directly refutes the DOF-under-weighting hypothesis as r's primary cause — a real, useful negative result, not an inconclusive one.

**Leading remaining hypothesis, not yet tested:** every one of these three segments was fit against the SAME 150-step-truncated training runs (15 seconds of real data at 10Hz) — a deliberate speed shortcut from finding #15's original proof-of-concept, never revisited since. The monotonic pattern (each segment's training objective improves while r's REAL held-out performance, on full-length runs, gets worse) is the classic signature of overfitting to a short training window that doesn't capture yaw's true dynamics timescale. This was named as a candidate lever back in finding #25 but not yet tested.

**Decision: stop further weight-tuning on the truncated setup.** Two different objectives (unweighted, then heavily r-boosted) produced the same monotonic regression — a third attempt at reweighting is not expected to help. The real next test (full-length training runs instead of truncated) is a genuinely more expensive commitment (loses the speed-shortcut entirely) and deserves a deliberate decision to invest in, not an incremental extension of tonight's session.

'''

occurrences = content.count(anchor)
if occurrences != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence, found {occurrences}.")
else:
    new_content = content.replace(anchor, finding_27 + anchor, 1)
    with open("PROJECT_STATUS.md", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Finding #27 inserted successfully.")