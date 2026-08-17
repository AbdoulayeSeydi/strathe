with open("README.md", encoding="utf-8") as f:
    content = f.read()

old = "   # Calibrated Digital Twin for BlueROV2 (Marine Autonomy)\n\n   A system-identification pipeline that fits a physics model to real BlueROV2\n   mission logs, integrated into [HoloOcean](https://byu-holoocean.github.io/holoocean-docs/v2.0.0/index.html)\n   for high-fidelity underwater simulation \u2014 plus a Gym-compatible RL training"

new = "   # Tryton \u2014 Calibrated Digital Twins for Marine Autonomy\n\n   A system-identification pipeline that fits a physics model to real vehicle\n   mission logs, integrated into [HoloOcean](https://byu-holoocean.github.io/holoocean-docs/v2.0.0/index.html)\n   for high-fidelity underwater simulation \u2014 plus a Gym-compatible RL training"

count = content.count(old)
print("Occurrences:", count)
if count == 1:
    content = content.replace(old, new, 1)
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(content)
    print("Title rebranded successfully.")
else:
    print("REFUSING - count mismatch, do not proceed blindly.")