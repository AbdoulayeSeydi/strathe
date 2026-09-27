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
                                                 "automated_identifiability_report.py", "calibration_report.py"]
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
                                     "automated_identifiability_report.py", "calibration_report.py"]
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
    assert [c[1] for c in calls] == ["preflight_check.py", "run_calibration.py", "calibration_report.py"]
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
    assert body.count(f"value='{token}'") == 5
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
                                     "automated_identifiability_report.py", "calibration_report.py"]
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


def test_calibrate_without_a_checkpoint_still_fails_clearly(tmp_path, capsys):
    rc = cli.main(["calibrate", "--data", str(tmp_path), "--out", str(tmp_path / "r")],
                  runner=lambda c: 1 if c[1] == "run_calibration.py" else 0)
    assert rc == 1 and "Calibration didn't finish" in capsys.readouterr().out
