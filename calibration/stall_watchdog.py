"""
stall_watchdog.py -- catch the "one evaluation takes hours" stall in the act.

Background (PROJECT_STATUS.md, open items): a single calibration evaluation
has taken up to ~4 h against a ~100 s baseline. Tested and ruled out so far:
cloud-sync folders, rapid checkpoint writes. The planned next step was to
attach py-spy to a live stall, which needs someone watching at the right
moment (and root on macOS).

This module does the equivalent unattended, from inside the process:

1. Every evaluation records THREE clocks to eval_timing.csv:
       wall_s       time.time()        -- keeps running while the machine sleeps
       monotonic_s  time.monotonic()   -- on macOS, pauses while the machine sleeps
       cpu_s        time.process_time() -- only advances while THIS process computes
   The stand-in evaluation does a fixed amount of arithmetic, so a 100x
   slower evaluation has to come from outside it. The clocks say which kind:
       wall >> monotonic            machine was asleep / suspended (macOS; on Windows the
                                    monotonic clock may keep counting through sleep, so
                                    sleep shows up as the next line instead)
       wall ~ monotonic >> cpu      process was waiting (sleep, swapping, I/O, CPU starved)
       cpu ~ wall, both huge        it really computed that long (pathological input)
2. If an evaluation runs past a threshold, a background thread writes every
   thread's Python stack (faulthandler) plus resource usage and load average
   to stall_log.txt -- and repeats every `dump_every_s` while the stall lasts,
   so a long stall shows whether the stack is moving.

Never raises into the calibration: any failure inside the watchdog is
swallowed after one warning, same "observational, never operational"
rule as mlflow_logging.
"""
from __future__ import annotations

import csv
import faulthandler
import os
import statistics
import sys
import threading
import time
from contextlib import contextmanager
from typing import Optional

try:
    import resource  # POSIX only
except ImportError:  # Windows
    resource = None


def classify(wall_s: float, monotonic_s: float, cpu_s: float) -> str:
    """Name the most likely kind of slowness from the three clocks."""
    if wall_s <= 0:
        return "instant"
    if monotonic_s < 0.5 * wall_s:
        return "machine_asleep_or_suspended"
    if cpu_s < 0.5 * monotonic_s:
        return "process_waiting (swap / I/O / CPU starved)"
    return "computing"


class StallWatchdog:
    def __init__(self, output_dir: str, min_threshold_s: float = 120.0, factor: float = 10.0,
                 dump_every_s: float = 600.0, poll_s: float = 1.0):
        os.makedirs(output_dir, exist_ok=True)
        self.timing_path = os.path.join(output_dir, "eval_timing.csv")
        self.stall_path = os.path.join(output_dir, "stall_log.txt")
        self.min_threshold_s = min_threshold_s
        self.factor = factor
        self.dump_every_s = dump_every_s
        self.poll_s = poll_s
        self._durations: list = []
        self._lock = threading.Lock()
        self._active = None  # [label, t_wall, t_mono, t_cpu, next_dump_at_mono, stalled]
        self._broken = False
        self.n_stalls = 0
        if not os.path.exists(self.timing_path):
            with open(self.timing_path, "w", newline="") as f:
                csv.writer(f).writerow(["eval", "label", "start_unix", "wall_s", "monotonic_s", "cpu_s",
                                        "classification", "stalled"])
        self._count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, name="stall-watchdog", daemon=True)
        self._thread.start()

    def threshold_s(self) -> float:
        with self._lock:
            recent = self._durations[-50:]
        if len(recent) < 3:
            return self.min_threshold_s
        return max(self.min_threshold_s, self.factor * statistics.median(recent))

    @contextmanager
    def measure(self, label: str = "evaluation"):
        t_wall, t_mono, t_cpu = time.time(), time.monotonic(), time.process_time()
        with self._lock:
            self._active = [label, t_wall, t_mono, t_cpu, None, False]
        try:
            yield
        finally:
            wall, mono, cpu = time.time() - t_wall, time.monotonic() - t_mono, time.process_time() - t_cpu
            with self._lock:
                stalled = self._active[5] if self._active else False
                self._active = None
                self._durations.append(mono)
                self._count += 1
                n = self._count
            self._safe(self._append_timing, n, label, t_wall, wall, mono, cpu, stalled)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=5)

    # ------------------------------------------------------------------ internals
    def _append_timing(self, n, label, start, wall, mono, cpu, stalled):
        with open(self.timing_path, "a", newline="") as f:
            csv.writer(f).writerow([n, label, f"{start:.3f}", f"{wall:.3f}", f"{mono:.3f}", f"{cpu:.3f}",
                                    classify(wall, mono, cpu), int(stalled)])
        if stalled:
            with open(self.stall_path, "a") as f:
                f.write(f"\n=== stall ended: eval {n} took wall {wall:.1f}s, monotonic {mono:.1f}s, cpu {cpu:.1f}s "
                        f"-> {classify(wall, mono, cpu)} ===\n")

    def _watch(self):
        while not self._stop.wait(self.poll_s):
            with self._lock:
                active = self._active
                if active is None:
                    continue
                label, t_wall, t_mono, t_cpu, next_dump, _ = active
            elapsed = time.monotonic() - t_mono
            wall_elapsed = time.time() - t_wall
            limit = self.threshold_s()
            # wall clock too: if the machine slept, monotonic may not have moved
            if max(elapsed, wall_elapsed) < limit:
                continue
            now = time.monotonic()
            if next_dump is not None and now < next_dump:
                continue
            with self._lock:
                if self._active is None:
                    continue
                first = not self._active[5]
                self._active[4] = now + self.dump_every_s
                self._active[5] = True
            if first:
                self.n_stalls += 1
            self._safe(self._dump, label, wall_elapsed, elapsed, time.process_time() - t_cpu, limit, first)

    def _dump(self, label, wall, mono, cpu, limit, first):
        with open(self.stall_path, "a") as f:
            head = "STALL DETECTED" if first else "still stalled"
            f.write(f"\n=== {head}: {label} running {wall:.1f}s wall / {mono:.1f}s monotonic / {cpu:.1f}s cpu "
                    f"(threshold {limit:.1f}s) at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(f"likely: {classify(wall, mono, cpu)}\n")
            if hasattr(os, "getloadavg"):
                f.write(f"load average: {os.getloadavg()}\n")
            if resource is not None:
                ru = resource.getrusage(resource.RUSAGE_SELF)
                f.write(f"rusage: user {ru.ru_utime:.1f}s sys {ru.ru_stime:.1f}s maxrss {ru.ru_maxrss} "
                        f"majflt {ru.ru_majflt} nvcsw {ru.ru_nvcsw} nivcsw {ru.ru_nivcsw}\n")
            f.flush()
            faulthandler.dump_traceback(file=f, all_threads=True)

    def _safe(self, fn, *args):
        if self._broken:
            return
        try:
            fn(*args)
        except Exception as e:  # never let diagnostics break a calibration
            self._broken = True
            print(f"WARNING: stall watchdog disabled after an internal error: {type(e).__name__}: {e}",
                  file=sys.stderr)
