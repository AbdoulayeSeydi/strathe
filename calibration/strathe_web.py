"""
strathe_web.py -- a local web page for Strathe: start a calibration or a fleet
experiment, watch it run, read the report. No terminal knowledge needed
after `python strathe.py web`.

Local-only by design:
  - listens on 127.0.0.1 only, and rejects requests whose Host header isn't
    this machine (blocks DNS-rebinding from a web page in the same browser);
  - every form carries a per-session random token (blocks cross-site posts);
  - it can only start the fixed commands in strathe_cli (no shell, no
    arbitrary command lines);
  - it only serves files from the output folders of jobs it started.
It is a single-user convenience for the person at this machine, not a
multi-user server.
"""
from __future__ import annotations

import html
import json
import os
import re
import secrets
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional
from urllib.parse import parse_qs, quote, urlparse

import strathe_cli as cli

JOBS_DIR = os.path.join(cli.REPO_DIR, "strathe_jobs")
SHIPPED_FLEET_CONFIGS = ["default_experiment.yaml", "ci_smoke_experiment.yaml", "fleet_study.yaml",
                         "fleet_study_mujoco.yaml"]


# ----------------------------------------------------------------- jobs

@dataclass
class Job:
    id: str
    kind: str
    title: str
    argv: list
    log_path: str
    output_dirs: list
    started: float = field(default_factory=time.time)
    proc: Optional[subprocess.Popen] = None
    returncode: Optional[int] = None

    @property
    def status(self) -> str:
        if self.proc is not None and self.returncode is None:
            rc = self.proc.poll()
            if rc is not None:
                self.returncode = rc
        if self.returncode is None:
            return "running"
        return "finished" if self.returncode == 0 else f"failed (exit code {self.returncode})"

    def log_tail(self, max_chars: int = 20000) -> str:
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except FileNotFoundError:
            return ""
        return text[-max_chars:]


class JobManager:
    """Starts strathe_cli commands as child processes and remembers them."""

    def __init__(self, jobs_dir: str = JOBS_DIR, launcher: Optional[Callable] = None):
        self.jobs_dir = jobs_dir
        self.jobs: dict = {}
        self._lock = threading.Lock()
        self._launcher = launcher or self._popen

    @staticmethod
    def _popen(cmd, log_file):
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MLFLOW_DISABLE_AGENT_HINT": "1"}
        return subprocess.Popen(cmd, cwd=cli.CAL_DIR, stdout=log_file, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env)

    def start(self, kind: str, title: str, cli_args: list, output_dirs: list) -> Job:
        os.makedirs(self.jobs_dir, exist_ok=True)
        job_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
        log_path = os.path.join(self.jobs_dir, f"{job_id}.log")
        cmd = [sys.executable, "-u", os.path.join(cli.CAL_DIR, "strathe_cli.py"), *cli_args]
        job = Job(job_id, kind, title, cmd, log_path, [os.path.abspath(d) for d in output_dirs])
        with open(log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"$ strathe {' '.join(cli_args)}\n\n")
            log_file.flush()
            job.proc = self._launcher(cmd, log_file)
        with self._lock:
            self.jobs[job_id] = job
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self.jobs.get(job_id)

    def all(self) -> list:
        with self._lock:
            return sorted(self.jobs.values(), key=lambda j: -j.started)

    def output_dirs(self, job: Job) -> list:
        """Folders a job declared when it started, plus those it announced in
        its log (fleet experiments print 'Report: <path>')."""
        return sorted(set(job.output_dirs + discover_outputs(job)))

    def allowed_file(self, path: str) -> Optional[str]:
        """The real path if `path` is inside some job's output folder or is a
        job log, else None."""
        real = os.path.realpath(path)
        roots = [os.path.realpath(self.jobs_dir)] + [os.path.realpath(d) for j in self.all()
                                                     for d in self.output_dirs(j)]
        for root in roots:
            if real == root or real.startswith(root + os.sep):
                return real if os.path.isfile(real) else None
        return None


def discover_outputs(job: Job) -> list:
    """Output folders a job announced in its log (fleet runs print 'Report: <path>')."""
    found = []
    for m in re.finditer(r"Report: (\S+)", job.log_tail(200000)):
        p = m.group(1)
        p = p if os.path.isabs(p) else os.path.join(cli.CAL_DIR, p)
        found.append(os.path.dirname(os.path.abspath(p)))
    return found


# ----------------------------------------------------------------- markdown (just enough for our reports)

def render_markdown(md: str) -> str:
    out, in_table, in_list = [], False, False

    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s

    for line in md.splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(re.fullmatch(r":?-{3,}:?", c) for c in cells):
                continue
            if not in_table:
                out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in cells) + "</tr>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False
        m = re.match(r"^(\s*)(?:[-*]|\d+\.)\s+(.*)", line)
        if m:
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(m.group(2))}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False
        h = re.match(r"^(#{1,4})\s+(.*)", line)
        if h:
            n = len(h.group(1))
            out.append(f"<h{n + 1}>{inline(h.group(2))}</h{n + 1}>")
        elif line.startswith(">"):
            out.append(f"<blockquote>{inline(line.lstrip('> '))}</blockquote>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    if in_table:
        out.append("</table>")
    if in_list:
        out.append("</ul>")
    return "\n".join(out)


# ----------------------------------------------------------------- pages

CSS = """
:root{--bg:#f7f8fa;--fg:#1d2330;--muted:#5d6675;--card:#fff;--line:#dde1e7;--accent:#1f6feb;--ok:#1a7f37;--bad:#cf222e}
@media (prefers-color-scheme:dark){:root{--bg:#0f1319;--fg:#e6e9ef;--muted:#9aa4b2;--card:#171c24;--line:#2a313c;--accent:#58a6ff;--ok:#3fb950;--bad:#f85149}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:920px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:24px 0 8px}
.muted{color:var(--muted)}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:12px 0}
label{display:block;font-weight:600;margin:10px 0 4px}input[type=text],select{width:100%;padding:8px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg)}
button{margin-top:12px;padding:8px 14px;border:0;border-radius:6px;background:var(--accent);color:#fff;font-weight:600;cursor:pointer}
pre{background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;overflow:auto;max-height:520px;font-size:12.5px;white-space:pre-wrap}
table{border-collapse:collapse;width:100%;margin:8px 0;display:block;overflow-x:auto}td,th{border:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
.ok{color:var(--ok)}.bad{color:var(--bad)}a{color:var(--accent)}code{font-size:90%}
fieldset.current{border:1px solid var(--line);border-radius:8px;margin:14px 0 0;padding:8px 12px 12px}
legend{font-weight:600;padding:0 4px}label.inline{display:flex;gap:8px;align-items:center;font-weight:400}
"""


def page(title: str, body: str) -> bytes:
    return (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,"
            f"initial-scale=1'><title>{html.escape(title)}</title><style>{CSS}</style></head><body><main>"
            f"{body}</main></body></html>").encode("utf-8")


def _status_html(job: Job) -> str:
    s = job.status
    cls = "ok" if s == "finished" else ("bad" if s.startswith("failed") else "muted")
    return f"<span class={cls}>{html.escape(s)}</span>"


def home_page(manager: JobManager, token: str) -> bytes:
    fleet_opts = "".join(f"<option value='{c}'>{c}</option>" for c in SHIPPED_FLEET_CONFIGS
                         if os.path.exists(os.path.join(cli.CAL_DIR, c)))
    default_data = cli.REFERENCE_DATA if os.path.isdir(cli.REFERENCE_DATA) else ""
    jobs = "".join(f"<tr><td><a href='/jobs/{j.id}'>{html.escape(j.title)}</a></td><td>{_status_html(j)}</td>"
                   f"<td>{time.strftime('%H:%M', time.localtime(j.started))}</td></tr>" for j in manager.all())
    tok = f"<input type=hidden name=token value='{token}'>"
    body = f"""
<h1>Strathe</h1><p class=muted>Runs on this computer only. Close this tab or press Ctrl+C in the terminal to stop.</p>
<div class=card><h2>Calibrate a vehicle</h2>
<p class=muted>Checks your logs, fits the simulator to them, and writes a plain-language report. About an hour on the
reference dataset; the computer is kept awake while it runs.</p>
<form method=post action=/jobs/calibrate>{tok}
<label for=cal-data>Folder with your logs (Strathe CSV layout; use “Convert logs” first if needed)</label>
<input type=text id=cal-data name=data value="{html.escape(default_data)}" required>
<label for=cal-name>Name for this calibration</label><input type=text id=cal-name name=name value="my_vehicle" required>
<label for=cal-heldout>Recording dates to hold out for validation (optional, e.g. <code>16-02-2024 19-02-2024</code>)</label>
<input type=text id=cal-heldout name=heldout value="">
<button>Start calibration</button></form></div>
<div class=card><h2>Run a fleet experiment</h2>
<p class=muted>Two vehicles holding formation over a lossy acoustic link. Edit a copy of an experiment YAML to change
anything (see CUSTOM_EXPERIMENT.md).</p>
<form method=post action=/jobs/fleet>{tok}
<label for=fleet-shipped>Shipped experiment</label><select id=fleet-shipped name=shipped>{fleet_opts}</select>
<label for=fleet-config>…or your own experiment YAML (full path; leave empty to use the one above)</label>
<input type=text id=fleet-config name=config value="">
<fieldset class=current><legend>Water current (optional; off = still water, like a tank)</legend>
<label class=inline><input type=checkbox id=cur-on name=current_on value=1> Add a water current</label>
<label for=cur-speed>Speed in m/s: one value, or a range drawn per episode (e.g. <code>0.2</code> or
<code>0.1-0.3</code>)</label><input type=text id=cur-speed name=current_speed value="0.2">
<label for=cur-dir>Direction the water flows toward, in degrees (<code>90</code>, a range like <code>0-90</code>,
or <code>any</code>)</label><input type=text id=cur-dir name=current_direction value="any">
<label for=cur-sstd>Slow fluctuation of speed, m/s (optional)</label>
<input type=text id=cur-sstd name=current_speed_std value="0">
<label for=cur-dstd>Slow fluctuation of direction, degrees (optional)</label>
<input type=text id=cur-dstd name=current_direction_std value="0">
<label for=cur-ts>Fluctuation time scale, seconds</label><input type=text id=cur-ts name=current_time_scale value="30">
<label class=inline><input type=checkbox id=cur-known name=current_known value=1 checked> Teammate models know the
average current</label>
</fieldset>
<button>Start experiment</button></form></div>
<div class=card><h2>Convert logs</h2>
<form method=post action=/jobs/ingest>{tok}
<label for=ing-spec>Ingestion spec YAML (full path; see data_ingestion.py)</label><input type=text id=ing-spec name=spec required>
<label for=ing-out>Output folder</label><input type=text id=ing-out name=out required>
<button>Convert</button></form></div>
<div class=card><h2>Jobs</h2>{'<table><tr><th>Job</th><th>Status</th><th>Started</th></tr>' + jobs + '</table>'
                                  if jobs else '<p class=muted>None yet.</p>'}</div>"""
    return page("Strathe", body)


def job_page(job: Job, manager: JobManager) -> bytes:
    outs = manager.output_dirs(job)
    files = []
    for d in outs:
        if os.path.isdir(d):
            for name in sorted(os.listdir(d)):
                p = os.path.join(d, name)
                if os.path.isfile(p):
                    link = (f"/view?path={quote(p)}" if name.endswith(".md") else f"/file?path={quote(p)}")
                    files.append(f"<li><a href='{link}'>{html.escape(name)}</a> <span class=muted>"
                                 f"({html.escape(os.path.basename(d))})</span></li>")
    reports = [f for f in files if "report" in f]
    running = job.status == "running"
    refresh = "<script>setTimeout(()=>location.reload(),4000)</script>" if running else ""
    body = f"""<p><a href='/'>← all jobs</a></p><h1>{html.escape(job.title)}</h1>
<p>Status: {_status_html(job)}{' · refreshing every few seconds' if running else ''}</p>
{('<div class=card><h2>Report</h2><ul>' + ''.join(reports) + '</ul></div>') if reports else ''}
{('<div class=card><h2>Output files</h2><ul>' + ''.join(files) + '</ul></div>') if files else ''}
<div class=card><h2>Log</h2><pre>{html.escape(job.log_tail())}</pre></div>{refresh}"""
    return page(job.title, body)


# ----------------------------------------------------------------- HTTP

class StratheHandler(BaseHTTPRequestHandler):
    manager: JobManager = None
    token: str = ""
    port: int = 0

    def log_message(self, fmt, *args):  # keep the terminal quiet
        pass

    def _send(self, code, body: bytes, ctype="text/html; charset=utf-8", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; "
                                                    "script-src 'unsafe-inline'")
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _host_ok(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        return host in {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}

    def _error(self, code, msg):
        self._send(code, page("Error", f"<p><a href='/'>← back</a></p><h1>{code}</h1><p>{html.escape(msg)}</p>"))

    def do_GET(self):
        if not self._host_ok():
            return self._error(403, "Requests must come from this computer.")
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path == "/":
            return self._send(200, home_page(self.manager, self.token))
        m = re.fullmatch(r"/jobs/([\w-]+)", url.path)
        if m:
            job = self.manager.get(m.group(1))
            return self._send(200, job_page(job, self.manager)) if job else self._error(404, "No such job.")
        m = re.fullmatch(r"/jobs/([\w-]+)/status", url.path)
        if m:
            job = self.manager.get(m.group(1))
            if not job:
                return self._error(404, "No such job.")
            return self._send(200, json.dumps({"status": job.status}).encode(), "application/json")
        if url.path in ("/file", "/view"):
            real = self.manager.allowed_file((q.get("path") or [""])[0])
            if real is None:
                return self._error(404, "That file isn't part of a job's output.")
            with open(real, "rb") as f:
                data = f.read()
            if url.path == "/view":
                text = data.decode("utf-8", errors="replace")
                return self._send(200, page(os.path.basename(real), "<p><a href='/'>← Strathe home</a></p>" + render_markdown(text)))
            return self._send(200, data, "application/octet-stream",
                              {"Content-Disposition": f"attachment; filename=\"{os.path.basename(real)}\""})
        return self._error(404, "Not found.")

    def do_POST(self):
        if not self._host_ok():
            return self._error(403, "Requests must come from this computer.")
        length = int(self.headers.get("Content-Length") or 0)
        if length > 100_000:
            return self._error(413, "Form too large.")
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode("utf-8")).items()}
        if not secrets.compare_digest(form.get("token", ""), self.token):
            return self._error(403, "This form has expired; reload the page and try again.")
        try:
            job = self._start(urlparse(self.path).path, form)
        except ValueError as e:
            return self._error(400, str(e))
        if job is None:
            return self._error(404, "Not found.")
        self._send(303, b"", headers={"Location": f"/jobs/{job.id}"})

    def _start(self, path, form) -> Optional[Job]:
        if path == "/jobs/calibrate":
            data = os.path.abspath(os.path.expanduser(form.get("data", "").strip()))
            if not os.path.isdir(data):
                raise ValueError(f"Folder not found: {data}")
            name = cli.safe_name(form.get("name", "my_vehicle"))
            heldout = form.get("heldout", "").split()
            for d in heldout:
                if not re.fullmatch(r"\d{2}-\d{2}-\d{4}", d):
                    raise ValueError(f"Held-out dates must look like 16-02-2024, got {d!r}")
            out = os.path.join(cli.REPO_DIR, f"calibration_output_{name}")
            args = ["calibrate", "--data", data, "--name", name, "--out", out]
            if heldout:
                args += ["--heldout-dates", *heldout]
            return self.manager.start("calibrate", f"Calibration: {name}", args, [out])
        if path == "/jobs/fleet":
            config = form.get("config", "").strip()
            if config:
                config = os.path.abspath(os.path.expanduser(config))
                if not (os.path.isfile(config) and config.endswith((".yaml", ".yml"))):
                    raise ValueError(f"Experiment YAML not found: {config}")
            else:
                shipped = form.get("shipped", "")
                if shipped not in SHIPPED_FLEET_CONFIGS:
                    raise ValueError("Pick one of the shipped experiments.")
                config = os.path.join(cli.CAL_DIR, shipped)
            args = ["fleet", "--config", config]
            title = f"Fleet experiment: {os.path.basename(config)}"
            if form.get("current_on"):
                args += current_cli_args(form)
                title += " + water current"
            return self.manager.start("fleet", title, args, [])
        if path == "/jobs/ingest":
            spec = os.path.abspath(os.path.expanduser(form.get("spec", "").strip()))
            out = os.path.abspath(os.path.expanduser(form.get("out", "").strip()))
            if not os.path.isfile(spec):
                raise ValueError(f"Ingestion spec not found: {spec}")
            if not form.get("out", "").strip():
                raise ValueError("Give an output folder.")
            return self.manager.start("ingest", f"Convert logs: {os.path.basename(spec)}",
                                      ["ingest", "--spec", spec, "--out", out], [out])
        return None


def _number_or_range(text: str, name: str, allow_any: bool = False) -> list:
    """'0.2' -> ['0.2']; '0.1-0.3' or '0.1 0.3' -> ['0.1', '0.3']; 'any' -> ['0', '360']."""
    t = (text or "").strip().lower()
    if allow_any and t in ("any", "all", "*"):
        return ["0.0", "360.0"]
    parts = [p for p in re.split(r"\s*(?:-|to|,|\s)\s*", t) if p] if not t.startswith("-") else [t]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"{name}: expected a number or a range like 0.1-0.3, got {text!r}")
    if len(vals) not in (1, 2):
        raise ValueError(f"{name}: expected one number or two (a range), got {text!r}")
    return [repr(v) for v in vals]


def current_cli_args(form: dict) -> list:
    """Validated `strathe fleet --current-...` arguments from the web form."""
    speed = _number_or_range(form.get("current_speed", ""), "Current speed")
    direction = _number_or_range(form.get("current_direction", "any"), "Current direction", allow_any=True)
    extra = {}
    for key, flag in (("current_speed_std", "--current-speed-std"), ("current_direction_std", "--current-direction-std"),
                      ("current_time_scale", "--current-time-scale")):
        raw = (form.get(key) or "").strip()
        if raw:
            try:
                extra[flag] = float(raw)
            except ValueError:
                raise ValueError(f"{key.replace('_', ' ')}: expected a number, got {raw!r}")
    rng = lambda xs: float(xs[0]) if len(xs) == 1 else [float(xs[0]), float(xs[1])]
    cli.current_block(rng(speed), rng(direction), extra.get("--current-speed-std", 0.0),
                      extra.get("--current-direction-std", 0.0), extra.get("--current-time-scale", 30.0),
                      bool(form.get("current_known")))   # raises ValueError with a specific message
    args = ["--current-speed", *speed, "--current-direction", *direction]
    for flag, v in extra.items():
        args += [flag, repr(v)]
    if not form.get("current_known"):
        args.append("--current-unknown-to-beliefs")
    return args


def make_server(port: int = 8765, manager: Optional[JobManager] = None) -> ThreadingHTTPServer:
    handler = type("Handler", (StratheHandler,), {})
    handler.manager = manager or JobManager()
    handler.token = secrets.token_urlsafe(24)
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    handler.port = server.server_address[1]
    return server


def serve(port: int = 8765, open_browser: bool = True) -> int:
    server = make_server(port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Strathe is running at {url}  (this computer only; Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0
