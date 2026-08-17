with open("README.md", encoding="utf-8") as f:
    content = f.read()

old = "   for high-fidelity underwater simulation \u2014 plus a Gym-compatible RL training\n   environment built on top of the calibrated physics."

new = "   for high-fidelity underwater simulation \u2014 plus a Gym-compatible RL training\n   environment built on top of the calibrated physics.\n\n   **Current real scope: BlueROV2 only.** Every real result in this repo -\n   the calibration pipeline, HoloOcean integration, and RL environment -\n   has been built and validated against one vehicle. Expanding to\n   additional vehicle classes (e.g. torpedo-style AUVs) is a real,\n   planned next step, not something built or tested yet - see\n   PRODUCT_VISION.md for the honest scope of that gap."

count = content.count(old)
print("Occurrences:", count)
if count == 1:
    content = content.replace(old, new, 1)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("Scope note added successfully.")
else:
    print("REFUSING - count mismatch, do not proceed blindly.")