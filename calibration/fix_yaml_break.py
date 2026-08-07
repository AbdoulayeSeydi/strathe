# Fixes a real bug in the previous patch: a stray doubled-quote that
# broke YAML syntax in Ieff_roll's bounds_rationale. Small, surgical,
# self-verifying - refuses if the exact broken fragment isn't found.

with open("vehicle_spec.yaml", encoding="utf-8") as f:
    content = f.read()

broken_fragment = 'roll inertia ""than previously allowed'
fixed_fragment = 'roll inertia than previously allowed'

count = content.count(broken_fragment)
if count != 1:
    print(f"REFUSING TO FIX - expected exactly 1 occurrence of the broken fragment, found {count}.")
    print("Do NOT proceed. Paste the current content around vehicle_spec.yaml lines 100-102 "
          "so this can be fixed precisely instead.")
else:
    content = content.replace(broken_fragment, fixed_fragment, 1)
    with open("vehicle_spec.yaml", "w", encoding="utf-8") as f:
        f.write(content)
    print("Fixed the broken quote. Verifying YAML now parses correctly...")
    import yaml
    try:
        with open("vehicle_spec.yaml", encoding="utf-8") as f:
            yaml.safe_load(f)
        print("YAML now parses correctly.")
    except yaml.YAMLError as e:
        print(f"STILL BROKEN - a different issue remains: {e}")