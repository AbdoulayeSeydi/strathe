"""
Splits a set of trajectory files into train/held-out sets by GROUP, not by
random file — the default and current use is grouping by collection DATE,
because runs from the same session share conditions (battery charge,
tank/pool state, vehicle trim). A random split across all runs risks
overfitting to session-level conditions without it showing up in the
numbers: if Run 5/6/7/8 from the same afternoon all end up split across
train and held-out, "95% held-out accuracy" can really mean "memorized one
afternoon's conditions." Holding out entire sessions is the harder, more
honest validation — this is genuinely how it should be done, not just a
convenient default.

split_by_group() is the general primitive (grouping doesn't have to be by
date — vehicle, weather, operator, or pool/ocean are all plausible future
grouping keys); split_by_date() is a thin wrapper over it for the current
use case.

Split is fully deterministic — no randomness anywhere. Same inputs always
produce the same split.

Filenames currently look like: odom-12-01-2024-run1.csv, hover-15-06-2024-run3.csv
(anything containing a DD-MM-YYYY date is matched, regardless of prefix).
"""
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

DATE_PATTERN = re.compile(r"(\d{2}-\d{2}-\d{4})")


def discover_csv_files(data_dir: str) -> tuple:
    """
    Finds all CSVs under data_dir via rglob, DEDUPLICATED BY FILENAME —
    not just full path. This dataset has at least one file present both
    at the top level and inside a dataset/ subfolder with byte-identical
    content (confirmed via diff): odom-12-01-2024-run1.csv. Plain
    Path.rglob() alone counts both copies, silently double-weighting
    that run's residuals in any joint fit across every script that
    discovers files this way — run_calibration.py, diagnose_calibration.py,
    and check_excitation_confound.py all had this exact bug independently
    before being centralized here.

    Keeps the first occurrence (by sorted path) of each filename.
    Returns (deduped_paths, duplicates) where duplicates is a list of
    (skipped_path, kept_path) tuples — callers decide how to surface that
    (log warning, print, etc.), this function only detects it.
    """
    all_paths = sorted(str(p) for p in Path(data_dir).rglob("*.csv"))
    seen: dict = {}
    deduped = []
    duplicates = []
    for p in all_paths:
        name = Path(p).name
        if name in seen:
            duplicates.append((p, seen[name]))
            continue
        seen[name] = p
        deduped.append(p)
    return deduped, duplicates

# 30%: chosen to reserve enough held-out sessions that the held-out RMSE is
# trustworthy, while leaving most sessions for the optimizer to actually
# fit against. Not derived from a formal criterion — just a reasonable
# starting default; override with heldout_fraction= or explicit
# heldout_dates= if a specific split is needed.
DEFAULT_HELDOUT_FRACTION = 0.3


@dataclass
class SplitResult:
    train_files: list
    heldout_files: list
    train_groups: list       # e.g. dates, sorted chronologically (not string sort)
    heldout_groups: list
    n_train: int
    n_heldout: int


def extract_date(filepath: str) -> str:
    """
    Pulls the DD-MM-YYYY date string out of a filename. Matches on the
    basename only (Path(filepath).name) — matching against the full path
    risks an accidental match from a directory component (e.g. an
    "archive-12-01-2024" folder name) rather than the actual filename.
    """
    name = Path(filepath).name
    match = DATE_PATTERN.search(name)
    if not match:
        raise ValueError(f"No DD-MM-YYYY date found in filename: {name!r}")
    return match.group(1)


def _parse_date(date_str: str) -> datetime:
    """
    Explicit strptime with a fixed numeric format — NOT locale-dependent,
    and NOT string sort. Sorting DD-MM-YYYY strings directly is wrong:
    "02-12-2024" sorts before "15-01-2024" as a string, even though
    Jan 15 is chronologically earlier than Dec 2. Always sort parsed
    datetime objects, never the raw strings.
    """
    return datetime.strptime(date_str, "%d-%m-%Y")


def group_by(filepaths: list, group_key_fn: Callable[[str], str]) -> dict:
    """Returns {group_key: [filepaths, ...]}, in first-seen key order."""
    groups: dict = {}
    for fp in filepaths:
        key = group_key_fn(fp)
        groups.setdefault(key, []).append(fp)
    return groups


def split_by_group(
    filepaths: list,
    group_key_fn: Callable[[str], str],
    sort_key_fn: Callable[[str], object] = lambda g: g,
    heldout_groups: Optional[list] = None,
    heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
) -> SplitResult:
    """
    General train/held-out split by group. group_key_fn extracts a group
    label from each filepath (e.g. a date string); sort_key_fn converts
    that label into something correctly sortable (e.g. a parsed datetime,
    since the label itself might not sort correctly as a plain string —
    see _parse_date's docstring for exactly this failure mode).

    If heldout_groups is given, those exact groups are held out (must all
    exist in the data, or this raises). Otherwise, the most recent
    heldout_fraction of groups (by sort_key_fn order) are held out
    automatically.
    """
    if not (0 < heldout_fraction < 1):
        raise ValueError(f"heldout_fraction must be strictly between 0 and 1, got {heldout_fraction}")

    groups = group_by(filepaths, group_key_fn)
    all_groups = sorted(groups.keys(), key=sort_key_fn)

    if len(all_groups) < 2:
        raise ValueError(
            f"Need at least 2 distinct groups to create a train/held-out split; "
            f"found only {len(all_groups)} ({all_groups}). With one group, either "
            f"train or held-out would be empty and the optimizer can't be validated."
        )

    if heldout_groups is not None:
        unknown = sorted(set(heldout_groups) - set(all_groups))
        if unknown:
            raise ValueError(
                f"Unknown heldout group(s) not present in the data: {unknown}. "
                f"Available groups: {all_groups}"
            )
        heldout_groups = list(heldout_groups)
    else:
        n_heldout = max(1, min(len(all_groups) - 1, round(len(all_groups) * heldout_fraction)))
        heldout_groups = all_groups[-n_heldout:]

    heldout_set = set(heldout_groups)
    train_files, heldout_files = [], []
    for g in all_groups:  # iterate in sorted order, not dict insertion order
        target = heldout_files if g in heldout_set else train_files
        target.extend(groups[g])

    return SplitResult(
        train_files=sorted(train_files),
        heldout_files=sorted(heldout_files),
        train_groups=[g for g in all_groups if g not in heldout_set],
        heldout_groups=sorted(heldout_groups, key=sort_key_fn),
        n_train=len(train_files),
        n_heldout=len(heldout_files),
    )


def split_by_date(
    filepaths: list,
    heldout_dates: Optional[list] = None,
    heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
) -> SplitResult:
    """
    split_by_group(), specialized to collection date — the current
    grouping key. heldout_dates, if given, are "DD-MM-YYYY" strings and
    must exactly match dates found in the filenames.
    """
    return split_by_group(
        filepaths,
        group_key_fn=extract_date,
        sort_key_fn=_parse_date,  # chronological, not string, sort
        heldout_groups=heldout_dates,
        heldout_fraction=heldout_fraction,
    )