# Safe, self-verifying patch: adds the SAME auto-backup protection to
# run_calibration.py that run_calibration_holoocean.py already has -
# backs up any existing calibrated_parameters.yaml/checkpoint.yaml in
# the output dir BEFORE this run touches them, fixing the exact class of
# loss that hit segment 3's real result earlier tonight (which happened
# via the HoloOcean script - this closes the same gap in the stand-in
# script, which never got the fix).

with open("run_calibration.py", encoding="utf-8") as f:
    content = f.read()

old = '''    checkpoint_path = args.checkpoint_path or str(Path(args.output_dir) / "checkpoint.yaml")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Checkpointing every evaluation to: {checkpoint_path}")'''

new = '''    checkpoint_path = args.checkpoint_path or str(Path(args.output_dir) / "checkpoint.yaml")
    Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    # Auto-backup: protect whatever already exists in the output dir from
    # a PRIOR run before this one touches it at all. Same fix already
    # applied to run_calibration_holoocean.py - this closes the identical
    # gap in this (stand-in) script, which never got it.
    import shutil as _shutil
    import time as _time
    _output_dir = Path(checkpoint_path).parent
    _backup_tag = _time.strftime("%Y%m%d_%H%M%S")
    for _fname in ("calibrated_parameters.yaml", "checkpoint.yaml", "metrics.json"):
        _existing = _output_dir / _fname
        if _existing.exists():
            _backup_path = _output_dir / f"{_existing.stem}_backup_{_backup_tag}{_existing.suffix}"
            _shutil.copy2(_existing, _backup_path)
            print(f"Auto-backed up existing {_fname} -> {_backup_path.name} before this run starts")

    print(f"Checkpointing every evaluation to: {checkpoint_path}")'''

count = content.count(old)
if count != 1:
    print(f"REFUSING TO PATCH - expected exactly 1 occurrence, found {count}.")
else:
    content = content.replace(old, new, 1)
    with open("run_calibration.py", "w", encoding="utf-8") as f:
        f.write(content)
    print("Auto-backup added to run_calibration.py successfully.")