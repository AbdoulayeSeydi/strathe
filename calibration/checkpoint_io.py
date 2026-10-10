"""
Checkpoint files that survive an interruption.

A calibration rewrites its checkpoint after every evaluation, every few seconds, for hours. Opening the file for
writing empties it at once, so a run stopped, crashed or cut off by a flat battery at that moment used to leave an
empty checkpoint.yaml, and `strathe calibrate --finish` then had nothing to recover (a URI run lost its best result
this way; PROJECT_STATUS.md #70). write_checkpoint writes a temporary file next to the checkpoint and swaps it in, so
the checkpoint on disk is always the previous complete one or the new complete one. read_checkpoint says plainly
what is wrong with a damaged one.
"""
from __future__ import annotations

import json
import os
import tempfile
import time

import yaml


class CheckpointError(ValueError):
    """A checkpoint that is missing, empty, or not a calibration checkpoint."""


def write_checkpoint(path: str, data: dict, fmt: str = "yaml") -> None:
    """Replace `path` with `data` atomically (a temporary file in the same folder, flushed to disk, then renamed)."""
    folder = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".tmp", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            if fmt == "json":
                json.dump(data, f, indent=2)
            else:
                yaml.dump(data, f, sort_keys=False)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:  # Windows: something (antivirus, a reader) briefly holds the old file open
                if attempt == 19:
                    raise
                time.sleep(0.05)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def read_checkpoint(path: str) -> dict:
    """A calibration checkpoint as a dict, or a CheckpointError saying what is wrong with it."""
    if not os.path.exists(path):
        raise CheckpointError(f"{path}: no checkpoint here (did the fit start?)")
    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise CheckpointError(f"{path} is damaged and can't be read ({e.__class__.__name__}). It was probably cut off "
                              f"while being written, by a checkpoint written before this was fixed; re-run the "
                              f"calibration") from e
    if not data:
        raise CheckpointError(f"{path} is empty: it was cut off while being written (a run stopped or crashed with "
                              f"a version of Strathe before this was fixed), so the best result so far is lost. "
                              f"Re-run the calibration")
    if not isinstance(data, dict) or not isinstance(data.get("best_parameters_so_far"), dict):
        raise CheckpointError(f"{path} isn't a calibration checkpoint (no best_parameters_so_far)")
    return data
