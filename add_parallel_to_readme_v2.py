with open("README.md", encoding="utf-8") as f:
    content = f.read()

anchor = "3. **HoloOcean refinement**"

insertion = ("   Add --parallel (optionally --max-workers N) to the stand-in step "
             "above to run multiple restarts across real CPU cores instead of "
             "sequentially - confirmed real speedup (roughly 4 restarts in the "
             "time 1 would normally take) on real, smaller data. Honest caveat: "
             "checkpointing is disabled in this mode (a real, deliberate "
             "tradeoff - see PROJECT_STATUS.md finding 34), and the combination "
             "of --parallel with the FULL, untruncated dataset has not yet been "
             "confirmed to finish in a reasonable time - recommend testing on a "
             "smaller subset first, or preferring the default (sequential) mode "
             "for full-dataset runs until that has been verified.\n\n")

count = content.count(anchor)
print("Occurrences:", count)
if count == 1:
    content = content.replace(anchor, insertion + anchor, 1)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("Documented successfully.")
else:
    print("REFUSING - count mismatch, do not proceed blindly.")
