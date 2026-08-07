"""Safe, self-verifying patch: adds finding #25 to PROJECT_STATUS.md."""
with open("PROJECT_STATUS.md", encoding="utf-8") as f:
    content = f.read()

anchor = "## Honest open items"

finding_25 = '''### 25. HoloOcean refinement segment 3 — genuinely converged, partial fix: `r`/`q` improved, `p` remains a real, isolated gap

Resumed the HoloOcean-backed refinement (finding #24's confirmed-correct direction) for another `315` real evaluations. The cost-reduction pattern this time (`3.55e-03 → ... → 4.67e-09`, shrinking cleanly toward floating-point noise) is the genuine signature of reaching a real local optimum — not another budget-limited cutoff like segments 1-2.

**Real Claim A result, checked precisely against finding #20's original numbers, not just the aggregate:**
- `r`: `-12.9% → -8.3%` — **real improvement**
- `q`: `+21.7% → +24.9%` — **further improvement**
- `p`: `-7.2% → -8.0%` — **essentially unchanged**, within noise

**A related, informative negative result:** this calibration run already included the off-diagonal added-mass terms (`Xq_dot`, `Zq_dot`, from the shelved off-diagonal work — `vehicle_spec.yaml`'s schema change applies globally, so they were live in this run too). Both converged to essentially zero (`Xq_dot: 0.0000 → -0.0019`, `Zq_dot: 0.0000 → 0.0001`) despite the optimizer genuinely converging elsewhere — the model found no benefit to using them in this configuration.

**Conclusion:** this is very likely at or near the true local optimum for this specific setup (150-step truncated runs, this warm-start, these bounds) — 2 of 3 rotational axes are now real, confirmed wins; `p` is a genuine, isolated, unresolved limitation, not an under-converged one. Further `max-nfev` on this exact configuration is unlikely to move `p` further. Fixing `p` specifically likely needs a different lever (full-length runs instead of truncated, or a `p`-specific structural change), not more of the same refinement.

'''

occurrences = content.count(anchor)
if occurrences != 1:
    print(f"REFUSING TO PATCH — expected exactly 1 occurrence, found {occurrences}.")
else:
    new_content = content.replace(anchor, finding_25 + anchor, 1)
    with open("PROJECT_STATUS.md", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Finding #25 inserted successfully.")