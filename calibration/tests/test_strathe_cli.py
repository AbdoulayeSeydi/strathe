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
    assert rc == 0 and [c[1] for c in calls] == ["preflight_check.py", "run_calibration.py",
                                                 "automated_identifiability_report.py", "validation.py",
                                                 "calibration_report.py"]
    assert all(c[0] == sys.executable for c in calls)
    fit = calls[1]
    out = os.path.join(cli.REPO_DIR, "calibration_output_rov_1")
    assert fit[fit.index("--output-dir") + 1] == out
    assert fit[fit.index("--eval-workers") + 1] == "3" and fit[fit.index("--axis-scaling") + 1] == "zero_rms"
    assert fit[fit.index("--heldout-dates") + 1] == "16-02-2024"
    assert "--sampling-rate-hz" not in fit  # no ingestion report, no flag: the script's 10 Hz default
    ident = calls[2]
    assert ident[ident.index("--result") + 1] == out and ident[ident.index("--workers") + 1] == "3"
    rep = calls[3]
    assert rep[rep.index("--result") + 1] == out and rep[rep.index("--data-dir") + 1] == str(tmp_path)


def test_calibrate_straight_from_raw_logs_uses_the_ingestion_settings(tmp_path, capsys):
    """--ingest-spec: convert first, then calibrate at the rate and torque convention ingestion recorded."""
    import json
    spec = tmp_path / "my_ingest.yaml"
    spec.write_text("vehicle: x\n")
    out = tmp_path / "result"
    calls = []

    def runner(cmd):
        calls.append(cmd)
        if cmd[1] == "data_ingestion.py":  # what a real ingestion writes
            d = cmd[cmd.index("--out") + 1]
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "ingest_report.json"), "w", encoding="utf-8") as f:
                json.dump({"sampling_rate_hz": 20.0, "torque_reference_offset_m": [0.0, 0.0, 0.0]}, f)
        return 0
    rc = cli.main(["calibrate", "--ingest-spec", str(spec), "--out", str(out)], runner=runner)
    assert rc == 0
    assert [c[1] for c in calls] == ["data_ingestion.py", "preflight_check.py", "run_calibration.py",
                                     "automated_identifiability_report.py", "validation.py", "calibration_report.py"]
    data = str(out / "ingested_logs")
    assert calls[0][calls[0].index("--out") + 1] == data
    pre, fit = calls[1], calls[2]
    assert pre[pre.index("--data-dir") + 1] == data and pre[pre.index("--sampling-rate-hz") + 1] == "20"
    assert fit[fit.index("--sampling-rate-hz") + 1] == "20"
    i = fit.index("--torque-reference-offset-m")
    assert fit[i + 1:i + 4] == ["0.0", "0.0", "0.0"]
    assert capsys.readouterr().out.count("Logs are at 20 Hz") == 1
    # an explicit flag wins over the ingestion report
    calls.clear()
    cli.main(["calibrate", "--data", data, "--sampling-rate-hz", "5", "--torque-reference-offset-m", "0", "0",
              "-0.085", "--skip-identifiability"], runner=runner)
    assert [c[1] for c in calls] == ["preflight_check.py", "run_calibration.py", "validation.py",
                                     "calibration_report.py"]
    fit = calls[1]
    assert fit[fit.index("--sampling-rate-hz") + 1] == "5"
    i = fit.index("--torque-reference-offset-m")
    assert fit[i + 1:i + 4] == ["0.0", "0.0", "-0.085"]


def test_calibrate_needs_exactly_one_data_source(tmp_path, capsys):
    assert cli.main(["calibrate"], runner=lambda c: 0) == 2
    assert cli.main(["calibrate", "--data", str(tmp_path), "--ingest-spec", "x.yaml"], runner=lambda c: 0) == 2
    assert cli.main(["calibrate", "--ingest-spec", str(tmp_path / "missing.yaml")], runner=lambda c: 0) == 2
    assert "ingestion spec not found" in capsys.readouterr().out


def test_identifiability_failure_only_warns(tmp_path, capsys):
    calls = []
    rc = cli.main(["calibrate", "--data", str(tmp_path)],
                  runner=lambda c: calls.append(c) or (1 if c[1] == "automated_identifiability_report.py" else 0))
    assert rc == 0 and calls[-1][1] == "calibration_report.py"
    assert "report is written without that section" in capsys.readouterr().out


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


def test_home_page_has_every_form(server):
    request, token, *_ = server
    status, body, _ = request("GET", "/")
    assert status == 200 and "Calibrate a vehicle" in body and "Run a fleet experiment" in body
    assert "One vehicle: train and test a controller" in body and "Train a fleet policy for your vehicle" in body
    assert "Calibrate the communication link" in body and "Roll and pitch from tank tests" in body
    assert "Calibrate your Stonefish model" in body and "Put your Strathe calibration into Stonefish" in body
    assert "Test your fleet" in body and "under acoustic limits" in body
    assert "Calibrate your simulator model" in body and "Put your Strathe calibration into your simulator" in body
    assert "Describe your vehicle from your Stonefish model" in body
    assert body.count(f"value='{token}'") == 13
    ids = re.findall(r" id=([\w-]+)", body)
    assert len(ids) == len(set(ids))  # every label points at exactly one input


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


def test_calibration_form_finishes_an_interrupted_run(server):
    import shutil
    request, token, started, manager, tmp_path = server
    form = {"token": token, "data": str(tmp_path), "name": "pytest_finish_check", "finish": "1"}
    out = os.path.join(cli.REPO_DIR, "calibration_output_pytest_finish_check")
    shutil.rmtree(out, ignore_errors=True)
    status, body, _ = request("POST", "/jobs/calibrate", form)
    assert status == 400 and "Nothing to finish" in body and not started
    try:
        os.makedirs(out)
        open(os.path.join(out, "checkpoint.yaml"), "w").write("best_parameters_so_far: {}\n")
        status, _, _ = request("POST", "/jobs/calibrate", form)
        assert status == 303 and started[-1][-1] == "--finish"
    finally:
        shutil.rmtree(out, ignore_errors=True)


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


# ----------------------------------------------------------------- one vehicle, two vehicles, thrusters

def test_arena_command_builds_the_single_vehicle_experiment(tmp_path):
    calls = []
    rc = cli.main(["arena", "--task", "station_keeping", "--seeds", "3", "--timesteps", "5000", "--compare-generic",
                   "--current-speed", "0", "0.3", "--thrusters", "bluerov2_heavy", "--out", str(tmp_path / "a")],
                  runner=lambda c: calls.append(c) or 0)
    assert rc == 0 and calls[0][1] == "run_arena_experiment.py"
    a = calls[0]
    assert a[a.index("--task") + 1] == "station_keeping" and "--compare-generic" in a
    assert a[a.index("--current-speed") + 1:a.index("--current-speed") + 3] == ["0.0", "0.3"]
    assert a[a.index("--thrusters") + 1] == "bluerov2_heavy" and a[a.index("--out") + 1] == str(tmp_path / "a")
    assert os.path.isabs(a[a.index("--spec") + 1]) and os.path.isabs(a[a.index("--params") + 1])


@pytest.mark.parametrize("argv,msg", [
    (["--params", "nope.yaml"], "calibration file not found"),
    (["--spec", "nope.yaml"], "vehicle description not found"),
    (["--thrusters", "bluerov3"], "only preset is bluerov2_heavy"),
    (["--thrusters", "missing.yaml"], "thrusters file not found"),
    (["--current-speed", "9"], "speed"),
])
def test_arena_rejects_bad_inputs_before_running(argv, msg, capsys):
    started = []
    assert cli.main(["arena", *argv], runner=lambda c: started.append(c) or 0) == 2
    assert msg in capsys.readouterr().out and not started


def test_fleet_with_your_vehicle_calibration_and_thrusters(tmp_path, monkeypatch):
    import yaml
    from fleet_experiment_schema import load_fleet_experiment
    monkeypatch.setattr(cli, "JOBS_DIR", str(tmp_path))
    calls = []
    params = os.path.join(cli.CAL_DIR, "calibrated_parameters_fleet_truth.yaml")
    ingest = os.path.join(cli.CAL_DIR, "templates", "ingest_csv_pwm_custom_thrusters.yaml")
    rc = cli.main(["fleet", "--config", os.path.join(cli.CAL_DIR, "ci_smoke_experiment.yaml"), "--params", params,
                   "--vehicle-spec", os.path.join(cli.CAL_DIR, "vehicle_spec.yaml"), "--thrusters", ingest],
                  runner=lambda c: calls.append(c) or 0)
    assert rc == 0
    generated = calls[0][calls[0].index("--config") + 1]
    exp = yaml.safe_load(open(generated, encoding="utf-8"))
    assert exp["truth_params"] == params and exp["thrusters"] == ingest and exp["output_dir"].startswith(str(tmp_path))
    cwd = os.getcwd()
    try:
        os.chdir(cli.CAL_DIR)
        cfg = load_fleet_experiment(generated)
    finally:
        os.chdir(cwd)
    assert cfg.thruster_config().n_thrusters == 8


def test_train_fleet_command(tmp_path):
    calls = []
    params = os.path.join(cli.CAL_DIR, "calibrated_parameters_fleet_truth.yaml")
    rc = cli.main(["train-fleet", "--params", params, "--thrusters", "bluerov2_heavy", "--current-speed", "0", "0.2",
                   "--out", str(tmp_path / "p")], runner=lambda c: calls.append(c) or 0)
    assert rc == 0 and calls[0][1] == "train_fleet_policy.py"
    t = calls[0]
    assert t[t.index("--truth-params") + 1] == params and t[t.index("--thrusters") + 1] == "bluerov2_heavy"
    assert t[t.index("--out") + 1] == str(tmp_path / "p") and "--current-speed" in t


# ----------------------------------------------------------------- web: raw logs, one vehicle, fleet training

TEMPLATES = os.path.join(cli.CAL_DIR, "templates")


def test_web_calibration_from_raw_logs_with_every_option(server):
    request, token, started, *_ = server
    status, _, _ = request("POST", "/jobs/calibrate", {
        "token": token, "ingest_spec": os.path.join(TEMPLATES, "ingest_rosbag_mavros.yaml"), "name": "lab rov",
        "vehicle_spec": os.path.join(cli.CAL_DIR, "vehicle_spec.yaml"), "rate": "20",
        "current_world": "0.1, -0.05, 0", "max_nfev": "150", "exclude_rp": "1", "heldout": "03-02-2024"})
    assert status == 303
    cmd = started[-1][3:]
    assert cmd[:3] == ["calibrate", "--ingest-spec", os.path.join(TEMPLATES, "ingest_rosbag_mavros.yaml")]
    assert cmd[cmd.index("--sampling-rate-hz") + 1] == "20.0" and cmd[cmd.index("--max-nfev") + 1] == "150"
    assert cmd[cmd.index("--current-world-mps") + 1:cmd.index("--current-world-mps") + 4] == ["0.1", "-0.05", "0.0"]
    assert cmd[cmd.index("--exclude-axes") + 1:][:2] == ["p", "q"] and cmd[-2:] == ["--heldout-dates", "03-02-2024"]
    # the arguments the page builds are ones the CLI accepts
    cli.build_parser().parse_args(cmd)


def test_web_calibration_holds_free_decay_values(server, tmp_path):
    request, token, started, *_ = server
    fd = tmp_path / "free_decay.yaml"
    fd.write_text("hold: {Kp: -0.6}\n")
    status, _, _ = request("POST", "/jobs/calibrate", {"token": token, "data": str(tmp_path), "name": "fd",
                                                       "free_decay": str(fd)})
    assert status == 303
    cmd = started[-1][3:]
    assert cmd[cmd.index("--free-decay") + 1] == str(fd)
    cli.build_parser().parse_args(cmd)
    status, body, _ = request("POST", "/jobs/calibrate", {"token": token, "data": str(tmp_path), "name": "fd",
                                                          "free_decay": str(tmp_path / "nope.yaml")})
    assert status == 400 and "Free-decay result not found" in body


@pytest.mark.parametrize("form,msg", [
    ({"name": "x", "data": ""}, "Give your logs"),
    ({"name": "x", "ingest_spec": "/nope.yaml"}, "Ingestion spec not found"),
    ({"name": "x", "data": ".", "rate": "fast"}, "Sampling rate: expected a number"),
    ({"name": "x", "data": ".", "current_world": "0.1 0.2"}, "three numbers"),
    ({"name": "x", "data": ".", "current_world": "9 0 0"}, "between"),
    ({"name": "x", "data": ".", "vehicle_spec": "/nope.yaml"}, "Vehicle description not found"),
])
def test_web_calibration_rejects_bad_options(server, form, msg):
    request, token, started, *_ = server
    status, body, _ = request("POST", "/jobs/calibrate", {"token": token, **form})
    assert status == 400 and msg in body and not started


def test_web_single_vehicle_experiment(server):
    request, token, started, manager, tmp_path = server
    status, _, headers = request("POST", "/jobs/arena", {
        "token": token, "task": "station_keeping", "seeds": "3", "timesteps": "20000", "episodes": "50",
        "compare_generic": "1", "thrusters": "file",
        "thrusters_file": os.path.join(TEMPLATES, "ingest_csv_pwm_custom_thrusters.yaml"),
        "current_on": "1", "current_speed": "0-0.3", "current_direction": "any"})
    assert status == 303
    cmd = started[-1][3:]
    assert cmd[:3] == ["arena", "--task", "station_keeping"] and "--compare-generic" in cmd
    assert cmd[cmd.index("--thrusters") + 1].endswith("ingest_csv_pwm_custom_thrusters.yaml")
    assert "--current-unknown-to-beliefs" not in cmd and "--current-speed" in cmd
    cli.build_parser().parse_args(cmd)
    out = cmd[cmd.index("--out") + 1]
    assert out in manager.output_dirs(manager.get(headers["Location"].rsplit("/", 1)[1]))
    for bad, msg in (({"task": "flip"}, "Pick one of the tasks"), ({"seeds": "0"}, "Training seeds"),
                     ({"thrusters": "file", "thrusters_file": "/nope.yaml"}, "Thrusters file not found"),
                     ({"thrusters": "file", "thrusters_file": os.path.join(cli.CAL_DIR, "vehicle_spec.yaml")},
                      "has no thrusters: block"),
                     ({"params": "/nope.yaml"}, "Calibration file not found")):
        n = len(started)
        status, body, _ = request("POST", "/jobs/arena", {"token": token, "task": "waypoint", **bad})
        assert status == 400 and msg in body and len(started) == n


def test_web_fleet_with_your_vehicle_and_fleet_training(server):
    request, token, started, manager, tmp_path = server
    params = os.path.join(cli.CAL_DIR, "calibrated_parameters_fleet_truth.yaml")
    spec = os.path.join(cli.CAL_DIR, "vehicle_spec.yaml")
    status, _, _ = request("POST", "/jobs/fleet", {"token": token, "shipped": "ci_smoke_experiment.yaml",
                                                   "params": params, "vehicle_spec": spec,
                                                   "thrusters": "bluerov2_heavy"})
    assert status == 303
    cmd = started[-1][3:]
    assert cmd[cmd.index("--params") + 1] == params and cmd[cmd.index("--vehicle-spec") + 1] == spec
    assert cmd[cmd.index("--thrusters") + 1] == "bluerov2_heavy"
    cli.build_parser().parse_args(cmd)
    status, _, _ = request("POST", "/jobs/train_fleet", {"token": token, "params": params, "thrusters": "bluerov2_heavy",
                                                         "timesteps": "100000", "seed": "2", "current_speed": "0-0.2",
                                                         "out": str(tmp_path / "my_policy")})
    assert status == 303
    cmd = started[-1][3:]
    assert cmd[0] == "train-fleet" and cmd[cmd.index("--out") + 1] == str(tmp_path / "my_policy")
    assert cmd[cmd.index("--current-speed") + 1:cmd.index("--current-speed") + 3] == ["0.0", "0.2"]
    cli.build_parser().parse_args(cmd)
    status, body, _ = request("POST", "/jobs/train_fleet", {"token": token, "params": params,
                                                            "out": "/no/such/folder/p"})
    assert status == 400 and "save the policy in doesn" in body
    status, body, _ = request("POST", "/jobs/train_fleet", {"token": token, "out": str(tmp_path / "p")})
    assert status == 400 and "calibration file" in body


def test_calibrate_keeps_the_best_result_when_the_budget_runs_out(tmp_path, capsys):
    """A fit that stops early leaves checkpoint.yaml; strathe turns it into a marked, not-converged result
    with the same settings, and still checks identifiability and writes the report."""
    out = tmp_path / "res"
    calls = []

    def runner(cmd):
        calls.append(cmd)
        if cmd[1] == "run_calibration.py":
            out.mkdir(exist_ok=True)
            (out / "checkpoint.yaml").write_text("best_parameters_so_far: {}\n")
            return 1
        return 0
    rc = cli.main(["calibrate", "--data", str(tmp_path), "--out", str(out), "--exclude-axes", "p", "q",
                   "--current-world-mps", "0.1", "0", "0", "--max-nfev", "7"], runner=runner)
    assert rc == 0
    assert [c[1] for c in calls] == ["preflight_check.py", "run_calibration.py", "finalize_checkpoint.py",
                                     "automated_identifiability_report.py", "validation.py", "calibration_report.py"]
    fin = calls[2]
    assert fin[fin.index("--checkpoint") + 1] == str(out / "checkpoint.yaml")
    assert fin[fin.index("--axis-scaling") + 1] == "zero_rms" and "--exclude-axes" in fin
    assert fin[fin.index("--current-world-mps") + 1:fin.index("--current-world-mps") + 4] == ["0.1", "0.0", "0.0"]
    assert "--max-nfev 7" in fin[fin.index("--reason") + 1]
    assert "NOT converged" in capsys.readouterr().out
    # every flag handed to finalize_checkpoint is one it accepts
    import finalize_checkpoint
    parsed = finalize_checkpoint.build_parser().parse_args(fin[2:])
    assert parsed.exclude_axes == ["p", "q"] and parsed.axis_scaling == "zero_rms" and parsed.heldout_dates is None


def test_calibrate_finish_turns_an_interrupted_fit_into_a_result(tmp_path, capsys):
    """The machine slept or the run was stopped: the same command plus --finish skips conversion, the data
    check and the fit, and finalizes the checkpoint with the settings the fit was given (from the ingestion
    report here), then checks identifiability and writes the report."""
    out = tmp_path / "res"
    logs = out / "ingested_logs"
    logs.mkdir(parents=True)
    (logs / "ingest_report.json").write_text('{"sampling_rate_hz": 9.4, "torque_reference_offset_m": [0, 0, 0.085]}')
    spec = tmp_path / "spec.yaml"
    spec.write_text("vehicle: x\n")
    calls = []
    run = lambda cmd: calls.append(cmd) or 0
    assert cli.main(["calibrate", "--ingest-spec", str(spec), "--out", str(out), "--finish"], runner=run) == 2
    assert "no checkpoint.yaml" in capsys.readouterr().out and not calls
    (out / "checkpoint.yaml").write_text("best_parameters_so_far: {}\n")
    assert cli.main(["calibrate", "--ingest-spec", str(spec), "--out", str(out), "--finish"], runner=run) == 0
    assert [c[1] for c in calls] == ["finalize_checkpoint.py", "automated_identifiability_report.py",
                                     "validation.py", "calibration_report.py"]
    fin = calls[0]
    assert fin[fin.index("--sampling-rate-hz") + 1] == "9.4" and fin[fin.index("--axis-scaling") + 1] == "zero_rms"
    assert fin[fin.index("--torque-reference-offset-m") + 1:fin.index("--torque-reference-offset-m") + 4] == \
        ["0", "0", "0.085"]
    assert "interrupted" in fin[fin.index("--reason") + 1]
    import finalize_checkpoint
    finalize_checkpoint.build_parser().parse_args(fin[2:])



def test_finish_with_current_estimation_finishes_the_step_that_was_interrupted(tmp_path, capsys):
    """F21: --finish refused --estimate-current runs, the longest calibrations (B8: 5-20 h)."""
    logs, out = tmp_path / "logs", tmp_path / "out"
    logs.mkdir()
    calls = []
    run = lambda cmd: calls.append(cmd) or 0
    args = ["calibrate", "--data", str(logs), "--out", str(out), "--estimate-current", "--current-rounds", "2",
            "--finish"]
    assert cli.main(args, runner=run) == 2 and "nothing to finish" in capsys.readouterr().out
    # interrupted in round 1: finished from round 1's checkpoint, into the output folder, in still water
    (out / "current_round1").mkdir(parents=True)
    (out / "current_round1" / "checkpoint.yaml").write_text("best_parameters_so_far: {}\n")
    assert cli.main(args, runner=run) == 0
    fin = calls[0]
    assert fin[1] == "finalize_checkpoint.py" and fin[fin.index("--checkpoint") + 1] == str(out / "current_round1" /
                                                                                          "checkpoint.yaml")
    assert fin[fin.index("--output-dir") + 1] == str(out) and "--currents-file" not in fin
    assert "current round 1 of 2" in fin[fin.index("--reason") + 1]
    assert not any(c[1] in ("run_calibration.py", "estimate_current.py") for c in calls)   # no new fitting
    # interrupted in round 2, but round 1's currents are missing: a plain message
    calls.clear()
    (out / "current_round2").mkdir()
    (out / "current_round2" / "checkpoint.yaml").write_text("best_parameters_so_far: {}\n")
    assert cli.main(args, runner=run) == 2 and "currents aren't known" in capsys.readouterr().out
    (out / "current_round1" / "currents.csv").write_text("file,vx_mps\n")
    assert cli.main(args, runner=run) == 0
    assert calls[0][calls[0].index("--currents-file") + 1] == str(out / "current_round1" / "currents.csv")
    # interrupted in the final fit: finished from its checkpoint with the last round's currents
    calls.clear()
    (out / "checkpoint.yaml").write_text("best_parameters_so_far: {}\n")
    (out / "current_round2" / "currents.csv").write_text("file,vx_mps\n")
    assert cli.main(args, runner=run) == 0
    fin = calls[0]
    assert fin[fin.index("--checkpoint") + 1] == str(out / "checkpoint.yaml") and "--output-dir" not in fin
    assert fin[fin.index("--currents-file") + 1] == str(out / "current_round2" / "currents.csv")
    assert "final fit" in fin[fin.index("--reason") + 1]


def test_calibrate_without_a_checkpoint_still_fails_clearly(tmp_path, capsys):
    rc = cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "r")],
                  runner=lambda c: 1 if c[1] == "run_calibration.py" else 0)
    assert rc == 1 and "Calibration didn't finish" in capsys.readouterr().out


# ----------------------------------------------------------------- web: engines, disturbance, fleet settings, currents

def test_web_arena_engine_and_disturbance(server):
    request, token, started, *_ = server
    status, _, _ = request("POST", "/jobs/arena", {"token": token, "task": "docking", "backend": "both",
                                                   "disturbance": "fitted", "disturbance_scale": "2"})
    assert status == 303
    cmd = started[-1][3:]
    assert cmd[cmd.index("--backend") + 1] == "both" and cmd[cmd.index("--disturbance") + 1] == "fitted"
    assert cmd[cmd.index("--disturbance-scale") + 1] == "2.0"
    cli.build_parser().parse_args(cmd)
    for bad, msg in (({"backend": "bullet"}, "Pick a physics engine"), ({"disturbance": "fitted", "disturbance_scale": "50"},
                     "between"), ({"disturbance": "file", "disturbance_file": "/nope.yaml"}, "fit file not found")):
        n = len(started)
        status, body, _ = request("POST", "/jobs/arena", {"token": token, "task": "waypoint", **bad})
        assert status == 400 and msg in body and len(started) == n


def test_web_fleet_settings_become_flags(server):
    """Checkboxes sharing a name (comm levels, teammate models) must all arrive, not just the first."""
    request, token, started, *_ = server
    body = [("token", token), ("shipped", "ci_smoke_experiment.yaml"), ("comm", "L0"), ("comm", "L3a"),
            ("beliefs", "calibrated"), ("beliefs", "oracle"), ("beacon_interval", "1 3"), ("seeds", "2"),
            ("target_separation", "3"), ("hold_steps", "25"), ("episode_steps", "250"), ("separations", "3.5, 5"),
            ("backend", "mujoco"), ("disturbance", "fitted"), ("disturbance_scale", "1"), ("modem_delay", "2"),
            ("bandwidth_bytes", "30"), ("workers", "3")]
    import http.client
    from urllib.parse import urlencode
    port = [c.cell_contents for c in request.__closure__ if isinstance(c.cell_contents, int)][0]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/jobs/fleet", body=urlencode(body), headers={"Host": f"127.0.0.1:{port}",
                 "Content-Type": "application/x-www-form-urlencoded"})
    assert conn.getresponse().status == 303
    cmd = started[-1][3:]
    assert cmd[cmd.index("--comm") + 1:cmd.index("--comm") + 3] == ["L0", "L3a"]
    assert cmd[cmd.index("--beliefs") + 1:cmd.index("--beliefs") + 3] == ["calibrated", "oracle"]
    assert cmd[cmd.index("--beacon-interval") + 1:cmd.index("--beacon-interval") + 3] == ["1.0", "3.0"]
    assert cmd[cmd.index("--seeds") + 1] == "2" and cmd[cmd.index("--backend") + 1] == "mujoco"
    assert cmd[cmd.index("--separations") + 1:cmd.index("--separations") + 3] == ["3.5", "5.0"]
    assert cmd[cmd.index("--modem-delay") + 1] == "2.0" and cmd[cmd.index("--bandwidth-bytes") + 1] == "30.0"
    assert cmd[cmd.index("--workers") + 1] == "3"
    cli.build_parser().parse_args(cmd)
    status, body, _ = request("POST", "/jobs/fleet", {"token": token, "shipped": "ci_smoke_experiment.yaml",
                                                      "beliefs": "oracle"})
    assert status == 400 and "at least two teammate models" in body
    # blank settings, default engine, no disturbance: the YAML's own values are kept (no flags added)
    request("POST", "/jobs/fleet", {"token": token, "shipped": "ci_smoke_experiment.yaml"})
    assert started[-1][3:] == ["fleet", "--config", os.path.join(cli.CAL_DIR, "ci_smoke_experiment.yaml")]


def test_web_calibration_can_estimate_the_current(server):
    request, token, started, *_ = server
    status, _, _ = request("POST", "/jobs/calibrate", {"token": token, "data": ".", "name": "sea",
                                                       "estimate_current": "1"})
    assert status == 303 and "--estimate-current" in started[-1]
    status, body, _ = request("POST", "/jobs/calibrate", {"token": token, "data": ".", "name": "sea",
                                                          "estimate_current": "1", "current_world": "0.1 0 0"})
    assert status == 400 and "not both" in body


# ----------------------------------------------------------------- web: clean vs noisy, buoyancy, algorithms

def test_web_noise_buoyancy_and_algorithms_reach_every_command(server):
    request, token, started, manager, tmp_path = server
    params = os.path.join(cli.CAL_DIR, "calibrated_parameters_fleet_truth.yaml")
    noise_file = os.path.join(cli.CAL_DIR, "sensor_noise", "bluerov2_tank.yaml")
    # one vehicle: SAC, noisy, own noise file, buoyancy range, clean-training comparison, explicit no disturbance
    status, _, _ = request("POST", "/jobs/arena", {
        "token": token, "task": "waypoint", "algorithm": "sac", "mode": "noisy", "noise": "file",
        "noise_file": noise_file, "noise_scale": "2", "buoyancy": "-3 to 3", "compare_clean_training": "1",
        "disturbance": "none"})
    assert status == 303
    cmd = started[-1][3:]
    assert cmd[cmd.index("--algorithm") + 1] == "sac" and cmd[cmd.index("--mode") + 1] == "noisy"
    assert cmd[cmd.index("--noise") + 1] == noise_file and cmd[cmd.index("--noise-scale") + 1] == "2.0"
    assert cmd[cmd.index("--buoyancy") + 1:cmd.index("--buoyancy") + 3] == ["-3.0", "3.0"]
    assert "--compare-clean-training" in cmd and cmd[cmd.index("--disturbance") + 1] == "none"
    cli.build_parser().parse_args(cmd)
    # two vehicles: both modes, typical noise, buoyancy known to beliefs
    status, _, _ = request("POST", "/jobs/fleet", {"token": token, "shipped": "ci_smoke_experiment.yaml",
                                                   "mode": "both", "noise": "typical", "buoyancy": "-2",
                                                   "buoyancy_known": "1"})
    assert status == 303
    cmd = started[-1][3:]
    assert cmd[cmd.index("--mode") + 1] == "both" and cmd[cmd.index("--noise") + 1] == "typical"
    assert cmd[cmd.index("--buoyancy") + 1] == "-2.0" and "--buoyancy-known-to-beliefs" in cmd
    cli.build_parser().parse_args(cmd)
    # fleet training: TD3, default steps for the algorithm, noisy, disturbance twice as rough
    status, _, _ = request("POST", "/jobs/train_fleet", {"token": token, "params": params, "algorithm": "td3",
                                                         "mode": "noisy", "disturbance": "fitted",
                                                         "disturbance_scale": "2", "seed": "1",
                                                         "out": str(tmp_path / "td3_pol")})
    assert status == 303
    cmd = started[-1][3:]
    assert cmd[cmd.index("--algorithm") + 1] == "td3" and "--timesteps" not in cmd
    assert cmd[cmd.index("--disturbance-scale") + 1] == "2.0" and cmd[cmd.index("--mode") + 1] == "noisy"
    cli.build_parser().parse_args(cmd)
    # nothing chosen: nothing added (the YAML's or the script's own defaults apply)
    request("POST", "/jobs/fleet", {"token": token, "shipped": "ci_smoke_experiment.yaml", "mode": "", "noise": "",
                                    "buoyancy": "", "disturbance": ""})
    assert started[-1][3:] == ["fleet", "--config", os.path.join(cli.CAL_DIR, "ci_smoke_experiment.yaml")]
    for path, bad, msg in (("/jobs/arena", {"mode": "loud"}, "Pick clean or noisy"),
                           ("/jobs/arena", {"mode": "both"}, "Pick clean or noisy"),   # one vehicle: no 'both'
                           ("/jobs/arena", {"noise": "file", "noise_file": "/nope.yaml"}, "Sensor-noise file not found"),
                           ("/jobs/arena", {"noise": "typical", "noise_scale": "40"}, "between"),
                           ("/jobs/arena", {"buoyancy": "200"}, "units"),
                           ("/jobs/arena", {"buoyancy": "heavy"}, "Buoyancy change"),
                           ("/jobs/arena", {"algorithm": "dqn"}, "Pick a training algorithm"),
                           ("/jobs/fleet", {"shipped": "ci_smoke_experiment.yaml", "noise": "loud"},
                            "Pick a sensor-noise option")):
        n = len(started)
        status, body, _ = request("POST", path, {"token": token, "task": "waypoint", **bad})
        assert status == 400 and msg in body and len(started) == n, (bad, body[:300])


def test_web_report_shows_its_plots(server):
    """The report's validation plots show inside the page (validation.py, #65); only a job's own .png files."""
    request, token, started, manager, tmp_path = server
    out = tmp_path / "jobs" / "cal"
    (out / "validation").mkdir(parents=True)
    png = b"\x89PNG\r\n\x1a\n" + b"0" * 100
    (out / "validation" / "drift.png").write_bytes(png)
    (out / "calibration_report.md").write_text("## Position\n\n![Position drift](validation/drift.png)\n\n"
                                               "See the [typical](validation/drift.png) recording.\n")
    status, body, _ = request("GET", f"/view?path={quote(str(out / 'calibration_report.md'))}")
    src = f"/image?path={quote(str(out / 'validation' / 'drift.png'))}"
    assert status == 200 and f"<img src='{src}'" in body and f"<a href='{src}'>typical</a>" in body
    status, body, headers = request("GET", src)
    assert status == 200 and headers["Content-Type"] == "image/png"
    assert request("GET", f"/image?path={quote(str(out / 'calibration_report.md'))}")[0] == 404
    assert request("GET", f"/image?path={quote('/etc/hosts')}")[0] == 404


def test_web_free_decay_and_fit_link_start_the_cli_commands(server):
    request, token, started, manager, tmp_path = server
    (tmp_path / "tank").mkdir()
    status, _, _ = request("POST", "/jobs/free_decay", {"token": token, "data": str(tmp_path / "tank"), "name": "t 1"})
    assert status == 303
    cmd = started[-1][3:]
    assert cmd[:3] == ["free-decay", "--data", str(tmp_path / "tank")] and cmd[-1].endswith("free_decay_t_1")
    cli.build_parser().parse_args(cmd)
    status, body, _ = request("POST", "/jobs/free_decay", {"token": token, "name": "x"})
    assert status == 400 and "Give the tank-test logs" in body
    log = tmp_path / "msgs.csv"
    log.write_text("send_time_s,receive_time_s,range_m\n0,1,10\n")
    status, _, _ = request("POST", "/jobs/fit_link", {"token": token, "logs": str(log), "name": "lnk",
                                                      "link": "acoustic", "optical_water": "harbor"})
    assert status == 303
    cmd = started[-1][3:]
    assert cmd[:3] == ["fit-link", "--log", str(log)] and cmd[-2:] == ["--optical-water", "harbor"]
    cli.build_parser().parse_args(cmd)


def test_stonefish_form_starts_exactly_the_cli_command(server):
    request, token, started, manager, tmp_path = server
    (tmp_path / "w.scn").write_text("<scenario/>")
    status, _, headers = request("POST", "/jobs/fit_stonefish", {
        "token": token, "scenario": str(tmp_path / "w.scn"), "robot": "alpha_rise", "part": "Hull",
        "reference_point": "-0.76 0 0", "data": str(tmp_path), "buoyancy_part": "Foam", "axisymmetric": "on",
        "name": "alpha rise"})
    assert status == 303
    cmd = started[0]
    assert cmd[3:11] == ["fit-stonefish", "--scenario", str(tmp_path / "w.scn"), "--robot", "alpha_rise", "--part",
                         "Hull", "--reference-point"]
    assert "--axisymmetric" in cmd and cmd[-2:] == ["--out", os.path.join(web.JOBS_DIR, "stonefish_alpha_rise")]
    # and the command line accepts exactly what the form built
    args = cli.build_parser().parse_args(cmd[3:])
    assert args.fn is cli.cmd_fit_stonefish and args.reference_point == [-0.76, 0.0, 0.0]


def test_calibration_form_passes_every_advanced_option(server):
    request, token, started, manager, tmp_path = server
    status, _, _ = request("POST", "/jobs/calibrate", {"token": token, "data": str(tmp_path), "name": "adv",
                                                      "estimate_current": "1", "current_rounds": "3", "workers": "4",
                                                      "seed": "77", "torque_offset": "0.05 0 0", "drag_prior": "1",
                                                      "restarts": "4", "restart_whole_fit": "1"})
    assert status == 303
    cmd = started[0]
    for flag, val in (("--current-rounds", "3"), ("--workers", "4"), ("--seed", "77"), ("--drag", "prior"),
                      ("--restarts", "4")):
        assert cmd[cmd.index(flag) + 1] == val
    assert cmd[cmd.index("--torque-reference-offset-m") + 1:cmd.index("--torque-reference-offset-m") + 4] == ["0.05", "0.0", "0.0"]
    assert "--restart-whole-fit" in cmd and cli.build_parser().parse_args(cmd[3:]).restart_whole_fit
    status, body, _ = request("POST", "/jobs/calibrate", {"token": token, "data": str(tmp_path), "name": "adv2",
                                                         "current_rounds": "3"})
    assert status == 400 and "estimate the current" in body


def test_stonefish_forms_take_the_replay_tool_iterations_and_workers(tmp_path):
    tool = tmp_path / "replay"
    tool.write_text("#!/bin/sh\n"); tool.chmod(0o755)
    (tmp_path / "w.scn").write_text("<scenario/>")
    form = {"scenario": str(tmp_path / "w.scn"), "robot": "auv", "part": "Hull", "reference_point": "0 0 0",
            "data": str(tmp_path), "replay_tool": str(tool), "max_iterations": "30", "workers": "2"}
    args = web.stonefish_cli_args(form)
    assert args[args.index("--replay-tool") + 1] == str(tool)
    assert args[args.index("--max-iterations") + 1] == "30" and args[args.index("--workers") + 1] == "2"
    with pytest.raises(ValueError, match="Replay tool not found"):
        web.stonefish_cli_args({**form, "replay_tool": str(tmp_path / "nope")})


def test_comms_study_form(server):
    """#89: the web card runs `strathe comms-study` with what was filled in."""
    request, token, started, manager, tmp_path = server
    status, _, _ = request("POST", "/jobs/comms_study", {"token": token, "name": "six boats", "fleet_size": "6",
                                                         "modem": "slow", "loss": "0.2", "burst": "4",
                                                         "sweep_latency": "1", "sweep_mac": "1", "episodes": "10",
                                                         "nav_drift": "1", "start_fix_error": "0.5"})
    assert status == 303
    cmd = started[0]
    assert cmd[3] == "comms-study"
    for flag, val in (("--fleet-size", "6"), ("--modem", "slow"), ("--loss", "0.2"), ("--burst", "4.0"),
                      ("--episodes", "10"), ("--nav-drift", "1.0"), ("--start-fix-error", "0.5")):
        assert cmd[cmd.index(flag) + 1] == val
    assert cmd[cmd.index("--sweep") + 1:cmd.index("--sweep") + 3] == ["latency", "mac"]
    assert cmd[-1].endswith("comms_six_boats")
    import comms_study
    comms_study.build_parser().parse_args(cmd[4:])   # the study accepts exactly what the form built
    for bad, msg in (({"fleet_size": "20"}, "Vehicles"), ({"loss": "1.5"}, "Packet loss"),
                     ({"controller": "not a class"}, "module:Class")):
        status, body, _ = request("POST", "/jobs/comms_study", {"token": token, "name": "x", **bad})
        assert status == 400 and msg in body
    status, body, _ = request("GET", "/", None)
    assert "Test your fleet&#x27;s coordination" in body or "Test your fleet's coordination" in body


def test_the_simulator_cards_start_fit_sim_and_export_sim(server):
    """PROJECT_STATUS.md #86: the "your simulator" cards reach `strathe fit-sim` / `export-sim`."""
    request, token, started, manager, tmp_path = server
    world = tmp_path / "w.sdf"
    world.write_text("<sdf/>")
    status, body, _ = request("GET", "/")
    assert "Calibrate your simulator model" in body and 'name=sim' in body
    status, _, _ = request("POST", "/jobs/fit_sim", {"token": token, "sim": "gazebo", "model": str(world),
                                                    "reference_point": "0 0 0", "data": str(tmp_path), "name": "g1"})
    assert status == 303
    cmd = started[-1]
    assert cmd[3:6] == ["fit-sim", "--sim", "gazebo"] and cmd[-2:] == ["--out", os.path.join(web.JOBS_DIR, "sim_fit_g1")]
    status, body, _ = request("POST", "/jobs/fit_sim", {"token": token, "sim": "gazebo", "model": str(world),
                                                       "reference_point": "0 0", "data": str(tmp_path), "name": "g2"})
    assert status == 400 and "three numbers" in body


def test_fleet_modem_delay_and_bandwidth_reach_the_experiment(tmp_path, capsys):
    """The two-vehicle experiment's channel had a modem delay and a byte budget in its YAML, but no flags (#89
    review): a lab had to hand-edit YAML for the two settings the coordination tests are about."""
    import yaml
    out = tmp_path / "exp.yaml"
    assert cli.main(["fleet", "--modem-delay", "2.5", "--bandwidth-bytes", "30", "--save-config", str(out)]) == 2
    assert "add --comm with a single level" in capsys.readouterr().out   # the default experiment has five levels
    assert cli.main(["fleet", "--comm", "L0", "--modem-delay", "2.5", "--bandwidth-bytes", "30",
                     "--save-config", str(out)]) == 0
    comm = yaml.safe_load(out.read_text())["communication"]
    src = yaml.safe_load(open(cli.DEFAULT_FLEET_CONFIG))["communication"]
    assert comm["acoustic_fixed_latency_s"] == 2.5 and comm["overrides"]["bytes_per_tick"] == 30.0
    for k, v in (src.get("overrides") or {}).items():   # the YAML's own loss settings are kept, not replaced
        if k != "bytes_per_tick":
            assert comm["overrides"][k] == v
    lc = tmp_path / "link.yaml"
    lc.write_text("{}")
    assert cli.main(["fleet", "--modem-delay", "1", "--link-calibration", str(lc), "--save-config",
                     str(tmp_path / "x.yaml")]) == 2
    assert "both set the modem delay" in capsys.readouterr().out
    assert cli.main(["fleet", "--modem-delay", "90", "--save-config", str(tmp_path / "y.yaml")]) == 2
    # F18: --workers reaches the experiment, alongside --seeds
    assert cli.main(["fleet", "--workers", "3", "--seeds", "2", "--save-config", str(tmp_path / "w.yaml")]) == 0
    ev = yaml.safe_load((tmp_path / "w.yaml").read_text())["evaluation"]
    assert ev["workers"] == 3 and ev["comm_seeds"] == [1, 2]
    assert cli.main(["fleet", "--workers", "0", "--save-config", str(tmp_path / "w0.yaml")]) == 2
    assert cli.main(["fleet", "--bandwidth-bytes", "0", "--save-config", str(tmp_path / "z.yaml")]) == 2


def test_the_training_and_arena_forms_pass_task_geometry(server):
    """#91: geometry for vehicles bigger than a BlueROV2, from the web page."""
    request, token, started, manager, tmp_path = server
    params = os.path.join(cli.CAL_DIR, "calibrated_parameters_fleet_truth.yaml")
    status, _, _ = request("POST", "/jobs/train_fleet", {"token": token, "params": params, "out": str(tmp_path / "p"),
                                                         "target_separation": "6", "tolerance": "0.6",
                                                         "collision_radius": "1.5", "separations": "5 9"})
    assert status == 303
    cmd = started[-1]
    assert cmd[cmd.index("--target-separation") + 1] == "6.0" and cmd[cmd.index("--separations") + 1:][:2] == ["5.0", "9.0"]
    status, _, _ = request("POST", "/jobs/arena", {"token": token, "task": "docking", "target_radius": "1.5",
                                                   "target_range": "10"})
    assert status == 303
    cmd = started[-1]
    assert cmd[cmd.index("--target-radius") + 1] == "1.5" and cmd[cmd.index("--target-range") + 1] == "10.0"
    cli.build_parser().parse_args(cmd[3:])
    status, body, _ = request("POST", "/jobs/train_fleet", {"token": token, "params": params, "out": str(tmp_path / "q"),
                                                           "separations": "5"})
    assert status == 400 and "two numbers" in body
