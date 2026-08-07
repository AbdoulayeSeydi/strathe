# Safe, self-verifying patch: adds finding #26 (off-diagonal added mass -
# formally shelved) to PROJECT_STATUS.md, right before "## Honest open items".

with open("PROJECT_STATUS.md", encoding="utf-8") as f:
    content = f.read()

anchor = "## Honest open items"

finding_26 = '''### 26. Off-diagonal added mass — formally shelved, not pursued further

Given finding #25's real negative result (both `Xq_dot`/`Zq_dot` converged to essentially zero when actually tested via real HoloOcean physics, despite the optimizer genuinely converging elsewhere) and the still-unresolved performance mystery blocking a clean, unbiased re-test via the stand-in backend (see the open items below), made a deliberate decision to shelve this line of work rather than keep investigating.

**Not pursued further because:** the evidence we have is real, not inconclusive — the model was given room to use this coupling and chose not to. A genuinely fresh (non-warm-started) stand-in test could rule out warm-start bias as an explanation, but doing so requires the performance mystery to be understood first, and isn't worth blocking other progress on.

**Practical consequence:** do not reference off-diagonal added mass in any public-facing material — it's real, tested, verified-correct code sitting unused in the model (harmless, since it defaults to zero coupling), not a capability worth claiming.

'''

occurrences = content.count(anchor)
if occurrences != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence, found {occurrences}.")
else:
    new_content = content.replace(anchor, finding_26 + anchor, 1)
    with open("PROJECT_STATUS.md", "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Finding #26 inserted successfully.")