"""strathe_cli.py (one-command onboarding) and strathe_web.py (local web page)."""
import http.client
import os
import re
import sys
import threading
from urllib.parse import quote, urlencode

import pytest

import strathe_cli as cli
import strathe_web as web


# ----------------------------------------------------------------- CLI

def test_calibrate_runs_check_fit_report_in_order_with_recommended_settings(tmp_path):
    calls = []
    rc = cli.main(["calibrate", "--data", str(tmp_path), "--name", "rov 1!", "--workers", "3",
                   "--heldout-dates", "16-02-2024"], runner=lambda cmd: calls.append(cmd) or 0)
    assert rc == 0 and [c[1] for c in calls] == ["preflight_check.py", "run_calibration.py", "calibration_report.py"]
    assert all(c[0] == sys.executable for c in calls)
    fit = calls[1]
    out = os.path.join(cli.REPO_DIR, "calibration_output_rov_1")
    assert fit[fit.index("--output-dir") + 1] == out
    assert fit[fit.index("--eval-workers") + 1] == "3" and fit[fit.index("--axis-scaling") + 1] == "zero_rms"
    assert fit[fit.index("--heldout-dates") + 1] == "16-02-2024"
    rep = calls[2]
    assert rep[rep.index("--result") + 1] == out and rep[rep.index("--data-dir") + 1] == str(tmp_path)


def test_calibrate_stops_at_the_first_failing_step(tmp_path, capsys):
    calls = []
    rc = cli.main(["calibrate", "--data", str(tmp_path)], runner=lambda cmd: calls.append(cmd) or (1 if len(calls) == 1 else 0))
    assert rc == 1 and len(calls) == 1
    assert "data check found problems" in capsys.readouterr().out


def test_calibrate_missing_data_folder_is_a_clear_error(tmp_path, capsys):
    assert cli.main(["calibrate", "--data", str(tmp_path / "nope")], runner=lambda c: 0) == 2
    assert "data folder not found" in capsys.readouterr().out


def test_advanced_calibration_options_pass_through(tmp_path):
    calls = []
    cli.main(["calibrate", "--data", str(tmp_path), "--torque-reference-offset-m", "0", "0", "-0.085",
              "--exclude-axes", "p", "q"], runner=lambda cmd: calls.append(cmd) or 0)
    fit = calls[1]
    i = fit.index("--torque-reference-offset-m")
    assert fit[i + 1:i + 4] == ["0.0", "0.0", "-0.085"]
    assert fit[fit.index("--exclude-axes") + 1:][:2] == ["p", "q"]


def test_other_commands_map_to_the_underlying_scripts(tmp_path):
    calls = []
    run = lambda cmd: calls.append(cmd) or 0
    cli.main(["fleet"], runner=run)
    cli.main(["fleet", "--config", "x.yaml"], runner=run)
    cli.main(["ingest", "--spec", "s.yaml", "--out", "o"], runner=run)
    cli.main(["report", "--result", "r"], runner=run)
    assert [c[1] for c in calls] == ["run_fleet_experiment.py"] * 2 + ["data_ingestion.py", "calibration_report.py"]
    assert calls[0][2:] == [] and calls[1][2] == "--config"


def test_safe_name():
    assert cli.safe_name("my vehicle/../x") == "my_vehicle_x"
    with pytest.raises(ValueError):
        cli.safe_name("../")


def test_check_reports_readiness(capsys):
    rc = cli.main(["check"])
    out = capsys.readouterr().out
    assert "numpy" in out and "shipped fleet policies (3/3)" in out and rc == 0


# ----------------------------------------------------------------- web

class FakeProc:
    def __init__(self):
        self.rc = None

    def poll(self):
        return self.rc


@pytest.fixture
def server(tmp_path):
    started = []

    def launcher(cmd, log_file):
        started.append(cmd)
        log_file.write("working...\nReport: results/ci_smoke/summary_report.md\n")
        return FakeProc()

    manager = web.JobManager(jobs_dir=str(tmp_path / "jobs"), launcher=launcher)
    srv = web.make_server(port=0, manager=manager)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    port = srv.server_address[1]
    token = srv.RequestHandlerClass.token

    def request(method, path, body=None, host=None):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        headers = {"Host": host or f"127.0.0.1:{port}"}
        if body is not None:
            body = urlencode(body)
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        conn.request(method, path, body=body, headers=headers)
        r = conn.getresponse()
        return r.status, r.read().decode("utf-8", errors="replace"), dict(r.getheaders())

    yield request, token, started, manager, tmp_path
    srv.shutdown()
    srv.server_close()


def test_home_page_has_the_three_forms(server):
    request, token, *_ = server
    status, body, _ = request("GET", "/")
    assert status == 200 and "Calibrate a vehicle" in body and "Run a fleet experiment" in body
    assert body.count(f"value='{token}'") == 3


def test_calibration_form_starts_exactly_the_cli_command(server):
    request, token, started, manager, tmp_path = server
    status, _, headers = request("POST", "/jobs/calibrate", {"token": token, "data": str(tmp_path),
                                                            "name": "tank rov", "heldout": "16-02-2024"})
    assert status == 303 and headers["Location"].startswith("/jobs/")
    cmd = started[0]
    assert cmd[:3] == [sys.executable, "-u", os.path.join(cli.CAL_DIR, "strathe_cli.py")]
    assert cmd[3:7] == ["calibrate", "--data", str(tmp_path), "--name"] and cmd[7] == "tank_rov"
    assert cmd[-2:] == ["--heldout-dates", "16-02-2024"]
    status, body, _ = request("GET", headers["Location"])
    assert status == 200 and "working..." in body and "running" in body


@pytest.mark.parametrize("form,msg", [
    ({"data": "/definitely/not/here", "name": "x"}, "Folder not found"),
    ({"data": ".", "name": "x", "heldout": "yesterday"}, "Held-out dates"),
])
def test_bad_calibration_input_is_rejected(server, form, msg):
    request, token, started, *_ = server
    status, body, _ = request("POST", "/jobs/calibrate", {"token": token, **form})
    assert status == 400 and msg in body and not started


def test_posts_without_the_session_token_are_rejected(server):
    request, _, started, _, tmp_path = server
    status, body, _ = request("POST", "/jobs/calibrate", {"token": "guess", "data": str(tmp_path), "name": "x"})
    assert status == 403 and not started


def test_requests_for_another_host_are_rejected(server):
    request, *_ = server
    assert request("GET", "/", host="evil.example:80")[0] == 403


def test_fleet_form_only_accepts_shipped_configs_or_existing_yaml(server):
    request, token, started, *_ = server
    assert request("POST", "/jobs/fleet", {"token": token, "shipped": "../../etc/passwd"})[0] == 400
    status, _, headers = request("POST", "/jobs/fleet", {"token": token, "shipped": "ci_smoke_experiment.yaml"})
    assert status == 303 and started[-1][3:5] == ["fleet", "--config"]
    assert started[-1][5] == os.path.join(cli.CAL_DIR, "ci_smoke_experiment.yaml")


def test_files_are_served_only_from_job_outputs(server):
    request, token, started, manager, tmp_path = server
    out = tmp_path / "calib"
    job = manager.start("calibrate", "t", ["calibrate"], [str(out)])
    out.mkdir()
    (out / "calibration_report.md").write_text("# Report\n\n| a | b |\n|---|---|\n| 1 | **2** |\n")
    status, body, _ = request("GET", f"/view?path={quote(str(out / 'calibration_report.md'))}")
    assert status == 200 and "<table>" in body and "<strong>2</strong>" in body
    assert request("GET", f"/file?path={quote(str(out / '..' / '..' / 'secret.txt'))}")[0] == 404
    assert request("GET", f"/file?path={quote('/etc/hosts')}")[0] == 404
    status, body, _ = request("GET", f"/jobs/{job.id}")
    assert "calibration_report.md" in body


def test_markdown_renderer_escapes_html():
    out = web.render_markdown("# T\n<script>alert(1)</script>\n- a `b`\n")
    assert "<script>" not in out and "&lt;script&gt;" in out and "<li>a <code>b</code></li>" in out


def test_outputs_announced_in_the_log_are_viewable(server):
    # fleet experiments name their output folder only in the log ("Report: <path>")
    request, token, started, manager, tmp_path = server
    status, _, headers = request("POST", "/jobs/fleet", {"token": token, "shipped": "ci_smoke_experiment.yaml"})
    job = manager.get(headers["Location"].rsplit("/", 1)[1])
    report = os.path.join(cli.CAL_DIR, "results", "ci_smoke", "summary_report.md")
    assert os.path.dirname(report) in manager.output_dirs(job)
    if os.path.exists(report):
        assert request("GET", f"/view?path={quote(report)}")[0] == 200
    assert manager.allowed_file(os.path.join(cli.CAL_DIR, "fleet_env.py")) is None


# ----------------------------------------------------------------- water current (CLI and web)

def test_fleet_with_a_current_runs_a_generated_experiment(tmp_path, monkeypatch):
    import yaml
    from fleet_experiment_schema import load_fleet_experiment
    monkeypatch.setattr(cli, "JOBS_DIR", str(tmp_path))
    calls = []
    rc = cli.main(["fleet", "--config", os.path.join(cli.CAL_DIR, "ci_smoke_experiment.yaml"),
                   "--current-speed", "0.1", "0.3", "--current-direction", "90", "--current-speed-std", "0.02",
                   "--current-unknown-to-beliefs"], runner=lambda cmd: calls.append(cmd) or 0)
    assert rc == 0 and calls[0][1] == "run_fleet_experiment.py"
    generated = calls[0][calls[0].index("--config") + 1]
    assert generated.startswith(str(tmp_path))
    exp = yaml.safe_load(open(generated, encoding="utf-8"))
    assert exp["current"] == {"speed_mps": [0.1, 0.3], "direction_deg": 90.0, "known_to_beliefs": False,
                              "variability": {"speed_std_mps": 0.02, "direction_std_deg": 0.0, "time_scale_s": 30.0}}
    assert exp["output_dir"].startswith(str(tmp_path))          # never overwrites shipped results
    cwd = os.getcwd()
    try:
        os.chdir(cli.CAL_DIR)
        cfg = load_fleet_experiment(generated)                  # the generated file is a valid experiment
    finally:
        os.chdir(cwd)
    assert cfg.current.to_config().speed_mps == (0.1, 0.3)


def test_fleet_rejects_an_invalid_current(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "JOBS_DIR", str(tmp_path))
    assert cli.main(["fleet", "--current-speed", "-0.2"], runner=lambda c: 0) == 2
    assert "speed_mps" in capsys.readouterr().out
    assert cli.main(["fleet", "--current-speed", "0.1", "0.2", "0.3"], runner=lambda c: 0) == 2


def test_web_form_current_fields_become_cli_flags():
    args = web.current_cli_args({"current_speed": "0.1-0.3", "current_direction": "any", "current_known": "1",
                                 "current_speed_std": "0.05", "current_direction_std": "", "current_time_scale": "20"})
    assert args[:6] == ["--current-speed", "0.1", "0.3", "--current-direction", "0.0", "360.0"]
    assert "--current-speed-std" in args and "--current-time-scale" in args
    assert "--current-direction-std" not in args and "--current-unknown-to-beliefs" not in args
    args = web.current_cli_args({"current_speed": "0.2", "current_direction": "45"})
    assert args == ["--current-speed", "0.2", "--current-direction", "45.0", "--current-unknown-to-beliefs"]
    for bad in ({"current_speed": "fast"}, {"current_speed": "5"}, {"current_speed": "0.1 0.2 0.3"},
                {"current_speed": "0.2", "current_time_scale": "0.1"}, {"current_speed": "0.2", "current_speed_std": "x"}):
        with pytest.raises(ValueError):
            web.current_cli_args({"current_direction": "any", **bad})


def test_web_fleet_form_with_a_current(server):
    request, token, started, *_ = server
    status, _, headers = request("POST", "/jobs/fleet", {"token": token, "shipped": "ci_smoke_experiment.yaml",
                                                         "current_on": "1", "current_speed": "0.2",
                                                         "current_direction": "any", "current_known": "1"})
    assert status == 303
    cmd = started[-1]
    assert cmd[3:5] == ["fleet", "--config"] and cmd[6:9] == ["--current-speed", "0.2", "--current-direction"]
    status, body, _ = request("POST", "/jobs/fleet", {"token": token, "shipped": "ci_smoke_experiment.yaml",
                                                      "current_on": "1", "current_speed": "9"})
    assert status == 400 and "speed" in body
    # the checkbox off = still water, whatever the other fields say
    request("POST", "/jobs/fleet", {"token": token, "shipped": "ci_smoke_experiment.yaml", "current_speed": "0.2"})
    assert "--current-speed" not in started[-1]
    assert "Add a water current" in request("GET", "/")[1]


def test_calibrate_passes_a_known_current_through(tmp_path):
    calls = []
    cli.main(["calibrate", "--data", str(tmp_path), "--current-world-mps", "0.1", "-0.2", "0"],
             runner=lambda cmd: calls.append(cmd) or 0)
    fit = calls[1]
    i = fit.index("--current-world-mps")
    assert fit[i + 1:i + 4] == ["0.1", "-0.2", "0.0"]
