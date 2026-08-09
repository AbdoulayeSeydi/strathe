with open("run_calibration_holoocean.py", encoding="utf-8") as f:
    lines = f.readlines()

out = []
i = 0
patched = False
while i < len(lines):
    line = lines[i]
    out.append(line)
    if 'isolating other changes from the separately-tested weighting approach' in line:
        out.append(lines[i + 1])
        out.append(lines[i + 2])
        out.append("    else:\n")
        j = i + 3
        for _ in range(4):
            content_line = lines[j]
            out.append("    " + content_line if content_line.strip() else content_line)
            j += 1
        i = j
        patched = True
        continue
    i += 1

print(f"Patched: {patched}")
if patched:
    with open("run_calibration_holoocean.py", "w", encoding="utf-8") as f:
        f.writelines(out)
    print("File written.")
else:
    print("Anchor substring not found - do NOT proceed.")
