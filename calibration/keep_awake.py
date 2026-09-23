"""
keep_awake.py -- stop the machine from sleeping during a long run.

PROJECT_STATUS.md #49: the "one evaluation takes hours" stall was macOS
sleeping through an overnight calibration. The process only advanced during
Power Nap wake-ups, so ~26 s of computing took up to 67 min of wall time.
Every long-running entry point (calibration, training, fleet studies) wraps
itself in this:

    if __name__ == "__main__":
        with keep_awake("stand-in calibration"):
            main()

How, per platform:
    macOS    `caffeinate -i -s -w <pid>`: blocks idle sleep, and system sleep
             while on AC power. `-w` ties it to this process, so it exits
             on its own even if this process is killed. Closing a laptop
             lid still sleeps the machine.
    Windows  SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED),
             cleared on exit (Windows also clears it if the process dies).
    Linux    `systemd-inhibit --what=idle:sleep` holding `tail --pid=<pid>`,
             so the inhibitor ends with this process.
    Anything else, or the tool is missing: one warning, then the run
    carries on normally.

Opt out with the environment variable STRATHE_ALLOW_SLEEP=1.

Never raises into the run: same "observational, never operational" rule as
stall_watchdog and mlflow_logging.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

OPT_OUT_ENV = "STRATHE_ALLOW_SLEEP"

# winbase.h
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@dataclass
class AwakeHandle:
    mechanism: Optional[str]          # "caffeinate" | "SetThreadExecutionState" | "systemd-inhibit" | None
    reason_inactive: str = ""         # why mechanism is None
    proc: Optional[subprocess.Popen] = None

    @property
    def active(self) -> bool:
        return self.mechanism is not None


def _warn(msg: str) -> None:
    print(f"[keep_awake] WARNING: {msg}", file=sys.stderr, flush=True)


def _info(msg: str) -> None:
    print(f"[keep_awake] {msg}", flush=True)


def _start(reason: str, platform: str, popen, which, kernel32) -> AwakeHandle:
    if os.environ.get(OPT_OUT_ENV, "").strip() not in ("", "0"):
        return AwakeHandle(None, f"{OPT_OUT_ENV} is set")
    pid = str(os.getpid())

    if platform == "darwin":
        exe = which("caffeinate")
        if not exe:
            return AwakeHandle(None, "caffeinate not found on PATH")
        proc = popen([exe, "-i", "-s", "-w", pid],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        _info("keeping the Mac awake for this run (caffeinate); closing a laptop lid still sleeps it")
        return AwakeHandle("caffeinate", proc=proc)

    if platform == "win32":
        if kernel32 is None:
            import ctypes
            kernel32 = ctypes.windll.kernel32
        if not kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
            return AwakeHandle(None, "SetThreadExecutionState failed")
        _info("keeping Windows awake for this run (SetThreadExecutionState)")
        return AwakeHandle("SetThreadExecutionState")

    if platform.startswith("linux"):
        inhibit, tail = which("systemd-inhibit"), which("tail")
        if not (inhibit and tail):
            return AwakeHandle(None, "systemd-inhibit not found on PATH")
        proc = popen([inhibit, "--what=idle:sleep", "--who=Strathe", f"--why={reason}", "--mode=block",
                      tail, f"--pid={pid}", "-f", "/dev/null"],
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.2)  # a refused inhibitor (no logind, no permission) exits immediately
        if proc.poll() is not None:
            return AwakeHandle(None, f"systemd-inhibit exited with code {proc.returncode}")
        _info("keeping the machine awake for this run (systemd-inhibit)")
        return AwakeHandle("systemd-inhibit", proc=proc)

    return AwakeHandle(None, f"no sleep-prevention method for platform {platform!r}")


def _stop(handle: AwakeHandle, kernel32) -> None:
    if handle.proc is not None and handle.proc.poll() is None:
        handle.proc.terminate()
        try:
            handle.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            handle.proc.kill()
    if handle.mechanism == "SetThreadExecutionState":
        if kernel32 is None:
            import ctypes
            kernel32 = ctypes.windll.kernel32
        kernel32.SetThreadExecutionState(ES_CONTINUOUS)


@contextmanager
def keep_awake(reason: str = "Strathe long run", *, platform: Optional[str] = None,
               _popen=subprocess.Popen, _which=shutil.which, _kernel32=None):
    """Prevent system sleep for the duration of the block. Yields an AwakeHandle."""
    platform = platform or sys.platform
    try:
        handle = _start(reason, platform, _popen, _which, _kernel32)
    except Exception as e:  # never let sleep prevention break the run itself
        handle = AwakeHandle(None, f"{type(e).__name__}: {e}")
    if not handle.active and not handle.reason_inactive.startswith(OPT_OUT_ENV):
        _warn(f"could not prevent sleep ({handle.reason_inactive}); if the machine sleeps, "
              f"this run pauses with it (PROJECT_STATUS.md #49)")
    try:
        yield handle
    finally:
        try:
            _stop(handle, _kernel32)
        except Exception as e:
            _warn(f"releasing the sleep block failed: {type(e).__name__}: {e}")
