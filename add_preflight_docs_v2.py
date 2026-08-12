with open("README.md", encoding="utf-8") as f:
    content = f.read()

old = "      Validates sampling rate and Coriolis-relevant excitation on the REAL,\n      full dataset before committing real compute time to a calibration that\n      might not succeed."

new = "      Validates sampling rate, Coriolis-relevant excitation, schema\n      (required columns), structural data integrity (empty/malformed files,\n      NaN/Inf), and physical/units plausibility (e.g. catches a\n      degrees-vs-radians unit mistake directly) on the REAL, full dataset\n      before committing real compute time to a calibration that might not\n      succeed."

count = content.count(old)
print("Occurrences:", count)
if count == 1:
    content = content.replace(old, new, 1)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("README.md updated successfully.")
else:
    print("REFUSING - count mismatch, do not proceed blindly.")
