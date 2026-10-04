"""keep_awake.py: sleep prevention for long runs (PROJECT_STATUS.md #49)."""
import os
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

from keep_awake import ES_CONTINUOUS, ES_SYSTEM_REQUIRED, OPT_OUT_ENV, keep_awake


class FakeProc:
    def __init__(self, argv, **kw):
        self.argv, self.returncode, self.terminated = argv, None, False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated, self.returncode = True, -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class FakePopen:
    def __init__(self):
        self.procs = []

    def __call__(self, argv, **kw):
        p = FakeProc(argv, **kw)
        self.procs.append(p)
        return p


class FakeKernel32:
    def __init__(self, ok=True):
        self.calls, self.ok = [], ok

    def SetThreadExecutionState(self, flags):
        self.calls.append(flags)
        return 0x80000000 if self.ok else 0


@pytest.fixture(autouse=True)
def _no_opt_out(monkeypatch):
    monkeypatch.delenv(OPT_OUT_ENV, raising=False)


def _which_all(name):
    return f"/usr/bin/{name}"


def test_macos_runs_caffeinate_tied_to_this_pid_and_stops_it():
    popen = FakePopen()
    with keep_awake(platform="darwin", _popen=popen, _which=_which_all) as h:
        assert h.active and h.mechanism == "caffeinate"
        (p,) = popen.procs
        assert p.argv == ["/usr/bin/caffeinate", "-i", "-s", "-w", str(os.getpid())]
        assert not p.terminated
    assert p.terminated


def test_released_even_when_the_run_raises():
    popen = FakePopen()
    with pytest.raises(RuntimeError):
        with keep_awake(platform="darwin", _popen=popen, _which=_which_all):
            raise RuntimeError("calibration failed")
    assert popen.procs[0].terminated


def test_opt_out_env_spawns_nothing(monkeypatch, capsys):
    monkeypatch.setenv(OPT_OUT_ENV, "1")
    popen = FakePopen()
    with keep_awake(platform="darwin", _popen=popen, _which=_which_all) as h:
        assert not h.active
    assert popen.procs == [] and "WARNING" not in capsys.readouterr().err


def test_opt_out_env_zero_means_keep_awake(monkeypatch):
    monkeypatch.setenv(OPT_OUT_ENV, "0")
    popen = FakePopen()
    with keep_awake(platform="darwin", _popen=popen, _which=_which_all) as h:
        assert h.active


def test_missing_tool_warns_and_the_run_still_happens(capsys):
    ran = False
    with keep_awake(platform="darwin", _popen=FakePopen(), _which=lambda n: None) as h:
        ran = True
    assert ran and not h.active
    assert "caffeinate not found" in capsys.readouterr().err


def test_spawn_failure_never_breaks_the_run(capsys):
    def boom(*a, **k):
        raise OSError("fork failed")
    with keep_awake(platform="darwin", _popen=boom, _which=_which_all) as h:
        assert not h.active
    assert "OSError: fork failed" in capsys.readouterr().err


def test_windows_sets_and_clears_execution_state():
    k = FakeKernel32()
    with keep_awake(platform="win32", _kernel32=k) as h:
        assert h.mechanism == "SetThreadExecutionState"
        assert k.calls == [ES_CONTINUOUS | ES_SYSTEM_REQUIRED]
    assert k.calls[-1] == ES_CONTINUOUS


def test_windows_failure_is_reported_not_raised(capsys):
    with keep_awake(platform="win32", _kernel32=FakeKernel32(ok=False)) as h:
        assert not h.active
    assert "SetThreadExecutionState failed" in capsys.readouterr().err


def test_linux_inhibitor_holds_a_tail_on_this_pid():
    popen = FakePopen()
    with keep_awake("unit test", platform="linux", _popen=popen, _which=_which_all) as h:
        assert h.mechanism == "systemd-inhibit"
        argv = popen.procs[0].argv
        assert argv[0] == "/usr/bin/systemd-inhibit" and "--what=idle:sleep" in argv
        assert "--why=unit test" in argv and f"--pid={os.getpid()}" in argv
    assert popen.procs[0].terminated


def test_linux_refused_inhibitor_is_reported(capsys):
    class Refused(FakePopen):
        def __call__(self, argv, **kw):
            p = super().__call__(argv, **kw)
            p.returncode = 1
            return p
    with keep_awake(platform="linux", _popen=Refused(), _which=_which_all) as h:
        assert not h.active
    assert "exited with code 1" in capsys.readouterr().err


def test_unknown_platform_is_a_warning(capsys):
    with keep_awake(platform="sunos5") as h:
        assert not h.active
    assert "sunos5" in capsys.readouterr().err


# ---- real OS integration -------------------------------------------------

@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("caffeinate"), reason="macOS only")
def test_real_caffeinate_runs_during_the_block_and_holds_an_assertion():
    with keep_awake("pytest") as h:
        assert h.mechanism == "caffeinate" and h.proc.poll() is None
        time.sleep(0.3)
        out = subprocess.run(["pmset", "-g", "assertions"], capture_output=True, text=True).stdout
        assert f"pid {h.proc.pid}(caffeinate)" in out
    assert h.proc.poll() is not None


@pytest.mark.skipif(sys.platform != "darwin" or not shutil.which("caffeinate"), reason="macOS only")
def test_real_caffeinate_dies_with_a_killed_run():
    """-w: a SIGKILLed run (no finally, no cleanup) must not leave the Mac pinned awake."""
    child = textwrap.dedent("""
        import os, sys, time
        sys.path.insert(0, %r)
        from keep_awake import keep_awake
        with keep_awake() as h:
            print("PID", h.proc.pid, flush=True)
            time.sleep(60)
    """ % os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         text=True)
    try:
        caf_pid = next(int(l.split()[1]) for l in p.stdout if l.startswith("PID "))
        os.kill(caf_pid, 0)  # alive
    finally:
        p.kill()  # also on failure, so a broken test can't leave a sleeping child behind
        p.wait()
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.kill(caf_pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    pytest.fail("caffeinate outlived the killed run")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows only")
def test_real_windows_execution_state():
    with keep_awake("pytest") as h:
        assert h.mechanism == "SetThreadExecutionState"


# ----------------------------------------------------------------- battery (PROJECT_STATUS.md #63)

def test_on_battery_warns_at_the_start_and_once_when_low(capsys):
    """The 2.5-hour gap in #62's calibration was the battery running out, not caffeinate failing."""
    readings = iter([(True, 60), (True, 40), (True, 18), (True, 12), (True, 9)])
    with keep_awake("t", platform="darwin", _popen=FakePopen(), _which=_which_all,
                    _battery=lambda: next(readings, (True, 5)), _battery_interval_s=0.01):
        time.sleep(0.3)
    err = capsys.readouterr().err
    assert "running on battery (60%)" in err
    assert err.count("the battery is at") == 1 and "the battery is at 18%" in err and "--finish" in err


def test_on_mains_power_says_nothing_about_the_battery(capsys):
    with keep_awake("t", platform="darwin", _popen=FakePopen(), _which=_which_all,
                    _battery=lambda: (False, 15), _battery_interval_s=0.01):
        time.sleep(0.05)
    assert "battery" not in capsys.readouterr().err


def test_a_failing_battery_check_never_breaks_the_run(capsys):
    def broken():
        raise OSError("no pmset")
    with keep_awake("t", platform="darwin", _popen=FakePopen(), _which=_which_all, _battery=broken) as h:
        assert h.active


def test_battery_status_reads_pmset():
    from keep_awake import battery_status

    class R:
        def __init__(self, out):
            self.stdout = out
    on = "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t23%; discharging; 0:53 remaining present: true"
    ac = "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t80%; charging; present: true"
    assert battery_status("darwin", _run=lambda *a, **k: R(on)) == (True, 23)
    assert battery_status("darwin", _run=lambda *a, **k: R(ac)) == (False, 80)
    assert battery_status("darwin", _run=lambda *a, **k: R("Now drawing from 'AC Power'\n")) == (None, None)
    assert battery_status("plan9") == (None, None)
