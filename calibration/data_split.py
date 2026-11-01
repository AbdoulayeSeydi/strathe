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

One session (PROJECT_STATUS.md #93): a lab with one field day can still calibrate. With one date, split_by_date holds
out whole recordings (the latest by run number); with one recording, the last part of it, after a gap. Both are easier
checks than an unseen day (the same trim, battery and water), and SplitResult.warning says so. A part of a recording
is a reference `path::rows=A-B` (rows A to B-1), which objective.read_run_csv reads.
"""
import json
import math
import os
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
    mode: str = "dates"      # "dates" | "recordings" | "time block" (#93)
    warning: Optional[str] = None


SPLIT_MODES = ("dates", "recordings", "time block")          # SplitResult.mode
HELDOUT_KINDS = ("date", "recording", "last part")           # what a held-out entry can name


# --- parts of a recording (#93) ------------------------------------------------------------------------------------
ROWS = "::rows="
LAST = "::last"
TIME_BLOCK_GAP_S = 10.0          # between the training and the held-out part of one recording
MIN_PART_S = 60.0                # each part holds at least one 60 s evaluation window (validation, fit-sim)
_REF = re.compile(r"^(?P<path>.*)::rows=(?P<a>\d+)-(?P<b>\d+)$")


def part_ref(path: str, start: int, stop: int) -> str:
    """Rows start..stop-1 of a recording, as a reference load_run reads."""
    return f"{path}{ROWS}{int(start)}-{int(stop)}"


def parse_ref(ref: str) -> tuple:
    """'path::rows=A-B' -> (path, A, B); a plain path -> (path, None, None)."""
    m = _REF.match(str(ref))
    return (m.group("path"), int(m.group("a")), int(m.group("b"))) if m else (str(ref), None, None)


def recording_path(ref: str) -> str:
    return parse_ref(ref)[0]


def recording_name(path: str) -> str:
    """odom-12-01-2024-run7.csv -> odom-12-01-2024-run7 (what --heldout-recordings takes)."""
    return os.path.splitext(os.path.basename(recording_path(path)))[0]


def _natural(name: str) -> list:
    """run2 before run10."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def _rows(path: str) -> int:
    with open(path, "rb") as f:
        return max(0, sum(1 for _ in f) - 1)


def _rate_hz(path: str, default: float = 10.0) -> float:
    """The recording's sampling rate: ingestion's report next to it, else Strathe's 10 Hz."""
    rp = os.path.join(os.path.dirname(os.path.abspath(path)), "ingest_report.json")
    try:
        return float(json.load(open(rp, encoding="utf-8")).get("sampling_rate_hz") or default)
    except (OSError, ValueError, AttributeError):
        return default


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


WITHIN_SESSION_WARNING = (
    "Held out within one session: the unseen part shares the session's trim, battery and water, so this check can't "
    "show how another day changes them. In our tests a one-session score stayed within 0.07 of the same model's "
    "score on other days (calibration/ONE_SESSION.md), but record a second session for the honest number.")


def _date_or_none(path: str) -> Optional[str]:
    try:
        return extract_date(path)
    except ValueError:
        return None


def _split_recordings(filepaths: list, heldout: Optional[list], heldout_fraction: float) -> SplitResult:
    """Whole recordings held out: the named ones, else the latest heldout_fraction by run number (at least one)."""
    names = {recording_name(p): p for p in filepaths}
    order = sorted(names, key=_natural)
    if heldout is not None:
        unknown = sorted(set(heldout) - set(names))
        if unknown:
            raise ValueError(f"Unknown held-out recording(s): {unknown}. The recordings are: {order}")
        held = [n for n in order if n in set(heldout)]
        if len(held) == len(order):
            raise ValueError("Every recording is held out: leave at least one to calibrate on.")
    else:
        n = max(1, min(len(order) - 1, round(len(order) * heldout_fraction)))
        held = order[-n:]
    train = [n for n in order if n not in set(held)]
    dates = {d for d in (_date_or_none(p) for p in filepaths) if d}
    return SplitResult(train_files=sorted(names[n] for n in train), heldout_files=sorted(names[n] for n in held),
                       train_groups=train, heldout_groups=held, n_train=len(train), n_heldout=len(held),
                       mode="recordings", warning=WITHIN_SESSION_WARNING if len(dates) <= 1 else None)


def _split_time_block(path: str, fraction: float, gap_s: float = TIME_BLOCK_GAP_S) -> SplitResult:
    """One recording: train on the first part, hold out the last `fraction` of it, with a gap between (#93)."""
    if not (0 < fraction < 1):
        raise ValueError(f"the held-out part of a recording must be strictly between 0 and 1, got {fraction}")
    n, rate = _rows(path), _rate_hz(path)
    gap = int(round(gap_s * rate))
    held_n = int(math.floor(n * fraction))
    train_n = n - held_n - gap
    need = int(math.ceil(MIN_PART_S * rate))
    if held_n < need or train_n < need:
        shortest = max(MIN_PART_S / fraction, (MIN_PART_S + gap_s) / (1 - fraction))   # held part, and training + gap
        raise ValueError(
            f"{os.path.basename(path)} is {n / rate:.0f} s long: too short to hold out part of it. Calibration needs "
            f"at least {MIN_PART_S:.0f} s to train on and {MIN_PART_S:.0f} s held out, {gap_s:.0f} s apart: at least "
            f"{shortest:.0f} s with {fraction:.0%} held out. Record longer, or a second recording.")
    name = recording_name(path)
    tag = f"{name}{LAST}{round(fraction * 100):g}%"
    train_ref, held_ref = part_ref(path, 0, train_n), part_ref(path, n - held_n, n)
    return SplitResult(train_files=[train_ref], heldout_files=[held_ref], train_groups=[f"{name}::first part"],
                       heldout_groups=[tag], n_train=1, n_heldout=1, mode="time block",
                       warning=WITHIN_SESSION_WARNING + f" Here the held-out part is the last {fraction:.0%} of one "
                               f"recording ({held_n / rate:.0f} s), after a {gap_s:g} s gap.")


def assert_disjoint(split: SplitResult) -> None:
    """No held-out sample in training: different recordings, or row ranges that don't overlap (#93)."""
    train = {}
    for ref in split.train_files:
        p, a, b = parse_ref(ref)
        train.setdefault(os.path.abspath(p), []).append((a, b))
    for ref in split.heldout_files:
        p, a, b = parse_ref(ref)
        for ta, tb in train.get(os.path.abspath(p), []):
            if a is None or ta is None or not (b <= ta or a >= tb):
                raise RuntimeError(f"held-out data overlaps training: {ref} and {p}::rows={ta}-{tb}")


def split_by_date(
    filepaths: list,
    heldout_dates: Optional[list] = None,
    heldout_fraction: float = DEFAULT_HELDOUT_FRACTION,
) -> SplitResult:
    """
    split_by_group(), specialized to collection date — the current
    grouping key. heldout_dates, if given, are "DD-MM-YYYY" strings and
    must exactly match dates found in the filenames.

    One session (#93): heldout_dates may also name recordings (odom-12-01-2024-run7) or a recording's last part
    (odom-12-01-2024-run7::last30%). With no names and one date, recordings are held out; with one recording, its last
    part. Two or more dates and no recording names: exactly as before.
    """
    if not (0 < heldout_fraction < 1):
        raise ValueError(f"heldout_fraction must be strictly between 0 and 1, got {heldout_fraction}")
    heldout = None if heldout_dates is None else [str(h) for h in heldout_dates]
    if heldout and any(LAST in h for h in heldout) and not all(LAST in h for h in heldout):
        raise ValueError(f"Hold out whole recordings or the last part of one, not both: {heldout}")
    if heldout and all(LAST in h for h in heldout):
        if len(heldout) != 1:
            raise ValueError("hold out the last part of one recording only (e.g. my_rov-05-01-2030-run1::last30%)")
        name, frac = heldout[0].split(LAST, 1)
        match = [p for p in filepaths if recording_name(p) == name]
        if not match:
            raise ValueError(f"Unknown recording {name!r}. The recordings are: "
                             f"{sorted((recording_name(p) for p in filepaths), key=_natural)}")
        try:
            fraction = float(frac.rstrip("%")) / 100
        except ValueError:
            raise ValueError(f"{heldout[0]!r}: give the held-out part as a percentage, e.g. ::last30%")
        split = _split_time_block(match[0], fraction)
        if len(filepaths) > 1:   # the other recordings train too
            others = sorted(p for p in filepaths if p != match[0])
            split.train_files = sorted(others + split.train_files)
            split.train_groups = sorted({recording_name(p) for p in others}, key=_natural) + split.train_groups
            split.n_train = len(split.train_files)
        assert_disjoint(split)
        return split
    dates = {extract_date(p) for p in filepaths}
    is_date = re.compile(r"^\d{2}-\d{2}-\d{4}$")
    if heldout and not all(is_date.match(h) for h in heldout):
        if any(is_date.match(h) for h in heldout):
            raise ValueError(f"Hold out dates or recordings, not both: {heldout}")
        split = _split_recordings(filepaths, heldout, heldout_fraction)
    elif len(dates) == 1:
        if heldout:   # a date given, and it's the only one
            raise ValueError(f"All recordings are from {next(iter(dates))}: there's no other day to calibrate on. "
                             f"Leave the held-out dates out and Strathe holds out recordings within the day, or name "
                             f"them with --heldout-recordings (calibration/ONE_SESSION.md)")
        split = (_split_recordings(filepaths, None, heldout_fraction) if len(filepaths) > 1
                 else _split_time_block(filepaths[0], heldout_fraction))
    else:
        split = split_by_group(
            filepaths,
            group_key_fn=extract_date,
            sort_key_fn=_parse_date,  # chronological, not string, sort
            heldout_groups=heldout,
            heldout_fraction=heldout_fraction,
        )
    assert_disjoint(split)
    return split
