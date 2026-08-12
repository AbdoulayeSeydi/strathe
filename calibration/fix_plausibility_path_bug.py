with open("check_physical_plausibility.py", encoding="utf-8") as f:
    content = f.read()

old = "import argparse\nimport sys\n\nimport numpy as np\n\nfrom data_split import discover_csv_files\nfrom objective import DatasetConfig, load_run"
new = "import argparse\nimport sys\nfrom pathlib import Path\n\nimport numpy as np\n\nfrom data_split import discover_csv_files\nfrom objective import DatasetConfig, load_run"
count = content.count(old)
print("Occurrences (imports):", count)
if count == 1:
    content = content.replace(old, new, 1)

old2 = "        try:\n            run = load_run(filepath, dataset_config)\n        except Exception as e:\n            print(f\"  ERROR: {filepath}: could not be loaded at all ({type(e).__name__}: {e})\")"
new2 = "        filename = Path(filepath).name\n        try:\n            run = load_run(filepath, dataset_config)\n        except Exception as e:\n            print(f\"  ERROR: {filename}: could not be loaded at all ({type(e).__name__}: {e})\")"
count2 = content.count(old2)
print("Occurrences (load_run block):", count2)
if count2 == 1:
    content = content.replace(old2, new2, 1)

content = content.replace("filepath.name", "filename")

with open("check_physical_plausibility.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed.")
