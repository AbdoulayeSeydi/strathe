"""
strathe_web.py -- a local web page for Strathe: calibrate a vehicle (from
converted logs or straight from raw CSV/ROS-bag logs), run a single-vehicle
or a fleet experiment, train a fleet policy for your vehicle, watch each job
run and read its report. No terminal knowledge needed after
`python strathe.py web`.

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
import math
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
        env = {**os.environ, "PYTHONUNBUFFERED": "1", "MLFLOW_DISABLE_AGENT_HINT": "1", "PYTHONIOENCODING": "utf-8"}
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

def render_markdown(md: str, base_dir: Optional[str] = None) -> str:
    """base_dir: the report's folder, so its plots show (`![alt](validation/x.png)`) and links to them open."""
    out, in_table, in_list = [], False, False

    def local(rel):
        return f"/image?path={quote(os.path.normpath(os.path.join(base_dir, rel)))}"  # one separator style (Windows)

    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        if base_dir:
            s = re.sub(r"\[([^\]]+)\]\(([^)\s]+\.png)\)", lambda m: f"<a href='{local(m.group(2))}'>{m.group(1)}</a>", s)
        return s

    for line in md.splitlines():
        img = re.fullmatch(r"!\[([^\]]*)\]\(([^)\s]+\.png)\)", line.strip())
        if img:
            if base_dir:
                out.append(f"<p><img src='{local(img.group(2))}' alt='{html.escape(img.group(1))}' "
                           f"style='max-width:100%;height:auto'></p>")
            continue
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


def current_fieldset(prefix: str, teammates: bool) -> str:
    """The water-current inputs (same field names in every form; ids unique per form)."""
    known = (f"<label class=inline><input type=checkbox id={prefix}-known name=current_known value=1 checked "
             f"aria-label=\"Teammate models know the average current\"> Teammate models know the\naverage current</label>"
             if teammates else "")
    return f"""<fieldset class=current><legend>Water current (optional; off = still water, like a tank)</legend>
<label class=inline><input type=checkbox id={prefix}-on name=current_on value=1 aria-label="Add a water current"> Add a water current</label>
<label for={prefix}-speed>Speed in m/s: one value, or a range drawn per episode (e.g. <code>0.2</code> or
<code>0.1-0.3</code>)</label><input type=text id={prefix}-speed name=current_speed value="0.2">
<label for={prefix}-dir>Direction the water flows toward, in degrees (<code>90</code>, a range like <code>0-90</code>,
or <code>any</code>)</label><input type=text id={prefix}-dir name=current_direction value="any">
<label for={prefix}-sstd>Slow fluctuation of speed, m/s (optional)</label>
<input type=text id={prefix}-sstd name=current_speed_std value="0">
<label for={prefix}-dstd>Slow fluctuation of direction, degrees (optional)</label>
<input type=text id={prefix}-dstd name=current_direction_std value="0">
<label for={prefix}-ts>Fluctuation time scale, seconds</label><input type=text id={prefix}-ts name=current_time_scale value="30">
{known}
</fieldset>"""


def vehicle_fields(prefix: str, params_required: bool = False) -> str:
    """Calibration file, vehicle description and thruster limits."""
    req = " required" if params_required else ""
    hint = "" if params_required else " (optional; default: the shipped BlueROV2 calibration)"
    return f"""<fieldset class=current><legend>Your vehicle</legend>
<label for={prefix}-params>Calibration file: <code>calibrated_parameters.yaml</code> from your calibration folder,
full path{hint}</label><input type=text id={prefix}-params name=params value=""{req}>
<label for={prefix}-spec>Vehicle description YAML, full path (default: the BlueROV2 Heavy; any other vehicle: copy calibration/templates/vehicle_spec_template.yaml and fill in its mass and size, LAB_GUIDE.md step 4)</label>
<input type=text id={prefix}-spec name=vehicle_spec value="">
<label for={prefix}-thr>Thruster limits</label><select id={prefix}-thr name=thrusters>
<option value="">None (any command up to ±30 N / N·m per axis)</option>
<option value="bluerov2_heavy">BlueROV2 Heavy (8 × T200)</option>
<option value="file">From a file (below)</option></select>
<label for={prefix}-thrfile>Thrusters file: a YAML with a <code>thrusters:</code> block, e.g. your ingestion spec
(full path; only if “From a file”)</label><input type=text id={prefix}-thrfile name=thrusters_file value="">
</fieldset>"""


def disturbance_select(prefix: str) -> str:
    return f"""<label for={prefix}-dist>Random roll/pitch torques (like the tether and waves in the real logs)</label>
<select id={prefix}-dist name=disturbance><option value="">From the mode above (clean: none; noisy: fitted)</option>
<option value=none>None</option>
<option value=fitted>Fitted to the reference BlueROV2 logs</option><option value=file>From my own fit file (below)</option></select>"""


def noise_fields(prefix: str, both: bool = False, teammates: bool = False) -> str:
    """Clean vs noisy, sensor noise and buoyancy (same field names in every form)."""
    both_opt = ("<option value=both>Both: every condition clean and noisy, compared episode by episode</option>"
                if both else "")
    known = (f"<label class=inline><input type=checkbox id={prefix}-bknown name=buoyancy_known value=1 "
             f"aria-label=\"Teammate models know the buoyancy change\"> Teammate models know the buoyancy change</label>"
             if teammates else "")
    return f"""<fieldset class=current><legend>Clean or noisy simulation</legend>
<label for={prefix}-mode>Mode</label><select id={prefix}-mode name=mode>
<option value="">Clean (exact sensors, no disturbance; the default)</option>
<option value=noisy>Noisy: sensor noise (typical, assumed sizes, unless set below) + the fitted roll/pitch disturbance</option>{both_opt}</select>
<label for={prefix}-noise>Sensor noise (DVL, gyro, attitude, compass, depth, and a position that drifts)</label>
<select id={prefix}-noise name=noise><option value="">From the mode (clean: none; noisy: typical)</option>
<option value=none>None (exact readings)</option>
<option value=tank>Tank: measured from the reference tank logs (small)</option>
<option value=typical>Typical DVL + AHRS vehicle, dead reckoning (assumed sizes, not measured)</option>
<option value=open_water>Open water, BlueROV2 at a fish farm (measured, SOLAQUA; biases assumed)</option>
<option value=file>From my own file (below)</option></select>
<p class=muted>Your own file: measure your sensors from your logs with <code>python strathe.py fit-noise --data
&lt;logs&gt; --out my_noise.yaml</code> in a terminal (there's no form for it yet), then give its path below.
<code>typical</code> is a placeholder of a common size for open water until you have your own.</p>
<label for={prefix}-nscale>Sensor noise size: 1 = as described, 2 = twice as noisy</label>
<input type=text id={prefix}-nscale name=noise_scale value="1">
<label for={prefix}-nfile>Sensor-noise file, full path (only if “From my own file”)</label>
<input type=text id={prefix}-nfile name=noise_file value="">
<label for={prefix}-sfix>Starting position fix error, m per axis (optional; default: the preset's, 0.17 for open
water, 0.2 for typical). In the fleet this is most of what sensor noise costs.</label>
<input type=text id={prefix}-sfix name=start_fix_error value="">
<label for={prefix}-buoy>Buoyancy change from the calibration, N (optional: <code>-3</code> for a heavier vehicle,
e.g. an added payload; a range like <code>-5 to 5</code> is drawn per episode)</label>
<input type=text id={prefix}-buoy name=buoyancy value="">
{known}
</fieldset>"""


def noise_cli_args(form: dict, allow_both: bool = False, teammates: bool = False) -> list:
    """--mode / --noise / --noise-scale / --buoyancy from the form, validated; blanks add nothing."""
    args = []
    mode = form.get("mode", "")
    if mode not in ("", "clean", "noisy") + (("both",) if allow_both else ()):
        raise ValueError("Pick clean or noisy.")
    if mode:
        args += ["--mode", mode]
    choice = form.get("noise", "")
    if choice:
        scale = _number(form.get("noise_scale", "1") or "1", "Sensor noise size", 0, 10)
        if choice == "file":
            path = _existing_file(form.get("noise_file", ""), "Sensor-noise file", (".yaml", ".yml"))
            from sensor_noise import load_sensor_noise
            load_sensor_noise(path)  # raises ValueError naming what's wrong
            args += ["--noise", path, "--noise-scale", repr(scale)]
        elif choice in ("none", "tank", "typical", "open_water"):
            args += ["--noise", choice, "--noise-scale", repr(scale)]
        else:
            raise ValueError("Pick a sensor-noise option.")
    if form.get("start_fix_error", "").strip():
        if not (mode in ("noisy", "both") or choice not in ("", "none")):
            raise ValueError("A starting-fix error needs sensor noise on: pick the noisy mode or a sensor-noise option.")
        args += ["--start-fix-error", repr(_number(form["start_fix_error"], "Starting position fix error", 0, 20))]
    if form.get("buoyancy", "").strip():
        vals = _signed_number_or_range(form["buoyancy"], "Buoyancy change")
        cli.buoyancy_block([float(v) for v in vals])  # raises ValueError with a specific message
        args += ["--buoyancy", *vals]
        if teammates and form.get("buoyancy_known"):
            args.append("--buoyancy-known-to-beliefs")
    return args


LINK_CHOICES = (("acoustic", "Acoustic (today's model)"), ("optical", "Optical only"),
                ("hybrid", "Hybrid: optical when it's up, acoustic otherwise"))
WATER_CHOICES = (("clear", "Clear ocean (c 0.15/m)"), ("coastal", "Coastal (c 0.4/m)"), ("harbor", "Harbour (c 2.2/m)"),
                 ("custom", "Custom (give c below)"))
TERMINAL_CHOICES = (("omni", "Omnidirectional (LED arrays; every direction; the default)"),
                    ("forward_cone", "One cone facing forward"), ("fore_aft_cones", "Two cones, forward and aft"),
                    ("down_cone", "One cone facing down"))


def link_fields(prefix: str, sweep: bool = False) -> str:
    """Acoustic / optical / hybrid link choices (same field names in every form)."""
    if sweep:
        links = "".join(f"<label class=inline><input type=checkbox name=link value={v} aria-label='Link {v}'> {d}"
                        f"</label>" for v, d in LINK_CHOICES)
        waters = "".join(f"<label class=inline><input type=checkbox name=optical_water value={v} "
                         f"aria-label='Water {v}'> {d}</label>" for v, d in WATER_CHOICES)
        link_ui = f"<p class=muted>Links (tick several to compare them on the same episodes; none = the YAML's, " \
                  f"acoustic by default):</p>{links}<p class=muted>Water (optical; tick several to compare):</p>{waters}"
    else:
        link_ui = (f"<label for={prefix}-link>Link</label><select id={prefix}-link name=link>"
                   + "".join(f"<option value={v}>{d}</option>" for v, d in LINK_CHOICES) + "</select>"
                   f"<label for={prefix}-water>Water (optical)</label><select id={prefix}-water name=optical_water>"
                   + "".join(f"<option value={v}{' selected' if v == 'coastal' else ''}>{d}</option>"
                             for v, d in WATER_CHOICES) + "</select>")
    terms = "".join(f"<option value={v}>{d}</option>" for v, d in TERMINAL_CHOICES)
    compare = (f"<label class=inline><input type=checkbox id={prefix}-omni name=optical_compare_omni value=1 "
               f"aria-label='Also run with omnidirectional terminals'> With cone terminals: also run with "
               f"omnidirectional terminals on the same episodes, to see what pointing costs</label>" if sweep else "")
    return f"""<fieldset class=current><legend>Communication link: acoustic, optical or hybrid</legend>
{link_ui}
<label for={prefix}-lcal>Link calibration from your message logs: the <code>link_calibration.yaml</code> that
<code>strathe fit-link</code> (or the card below) wrote, full path (optional). It sets the acoustic losses and latency,
and the optical water, delivery curve and latency, to what your logs measured.</label>
<input type=text id={prefix}-lcal name=link_calibration value="">
<p class=muted>Without a link calibration the optical model is <b>uncalibrated</b>: rough textbook water values and a
placeholder delivery curve; check a vendor's specs, or calibrate it from your link logs (LIMITS.md).</p>
<label for={prefix}-att>Custom water attenuation c, per metre (only for Custom)</label>
<input type=text id={prefix}-att name=optical_attenuation value="">
<label for={prefix}-terms>Optical terminals on each vehicle</label><select id={prefix}-terms name=optical_terminals>{terms}</select>
<label for={prefix}-beam>Beam half-angle, degrees (cone terminals; default 30; 180 = every direction)</label>
<input type=text id={prefix}-beam name=optical_beam value="">
<label for={prefix}-fov>Receiver half-angle, degrees (cone terminals; default 30)</label>
<input type=text id={prefix}-fov name=optical_fov value="">
<label for={prefix}-range>Rated range, m: where a pointed link delivers half its messages in clear water (default 30)</label>
<input type=text id={prefix}-range name=optical_rated_range value="">
<label for={prefix}-hyb>Hybrid policy</label><select id={prefix}-hyb name=hybrid_policy>
<option value="">optical_first (default): skip the acoustic beacon while optical is up</option>
<option value=redundant>redundant: every beacon on both links</option></select>
{compare}
</fieldset>"""


def _stonefish_vehicle_args(form: dict) -> list:
    """The Stonefish vehicle fields both Stonefish forms share: scenario, robot, part, reference point, buoyancy part,
    $(find) folders, data folder, water densities."""
    scn = _existing_file(form.get("scenario", ""), "Stonefish scenario", (".scn", ".xml"))
    robot, part = form.get("robot", "").strip(), form.get("part", "").strip()
    if not robot or not part:
        raise ValueError("Give the robot's name and the part to tune.")
    ref = form.get("reference_point", "").split()
    try:
        ref = [str(float(x)) for x in ref]
    except ValueError:
        ref = []
    if len(ref) != 3:
        raise ValueError("The reference point is three numbers: x y z in metres.")
    args = ["--scenario", scn, "--robot", robot, "--part", part, "--reference-point", *ref]
    if form.get("buoyancy_part", "").strip():
        args += ["--buoyancy-part", form["buoyancy_part"].strip()]
    finds = form.get("find", "").split()
    if finds:
        if not all("=" in f for f in finds):
            raise ValueError("$(find) folders are PKG=/path.")
        args += ["--find", *finds]
    if form.get("stonefish_data", "").strip():
        d = os.path.abspath(os.path.expanduser(form["stonefish_data"].strip()))
        if not os.path.isdir(d):
            raise ValueError(f"Stonefish data folder not found: {d}")
        args += ["--stonefish-data", d]
    dens = form.get("density_by_date", "").split()
    if dens:
        if not all(re.fullmatch(r"[\d-]+=[\d.]+", d) for d in dens):
            raise ValueError("Water densities are DATE=RHO, e.g. 27-03-2026=1000.")
        args += ["--density-by-date", *dens]
    return args


def _calibration_folder(text: str) -> str:
    d = os.path.abspath(os.path.expanduser(text.strip()))
    if not os.path.isfile(os.path.join(d, "calibrated_parameters.yaml")):
        raise ValueError(f"Not a calibration's output folder (no calibrated_parameters.yaml): {d}")
    return d


def _replay_tool_args(form: dict) -> list:
    """--replay-tool from a Stonefish form, if given (the tool must exist and be runnable)."""
    t = form.get("replay_tool", "").strip()
    if not t:
        return []
    t = os.path.abspath(os.path.expanduser(t))
    if not (os.path.isfile(t) and os.access(t, os.X_OK)):
        raise ValueError(f"Replay tool not found or not runnable: {t} (calibration/stonefish/README.md step 2 builds it)")
    return ["--replay-tool", t]


def stonefish_cli_args(form: dict) -> list:
    """`strathe fit-stonefish` arguments from the web form (PROJECT_STATUS.md #72)."""
    args = _stonefish_vehicle_args(form) + _replay_tool_args(form)
    if form.get("max_iterations", "").strip():
        args += ["--max-iterations", str(int(_number(form["max_iterations"], "Fit iterations", 1, 200)))]
    if form.get("workers", "").strip():
        args += ["--workers", str(int(_number(form["workers"], "Stonefish runs at once", 1, 256)))]
    if form.get("ingest_spec", "").strip():
        args += ["--ingest-spec", _existing_file(form["ingest_spec"], "Ingestion spec", (".yaml", ".yml"))]
    elif form.get("data", "").strip():
        data = os.path.abspath(os.path.expanduser(form["data"].strip()))
        if not os.path.isdir(data):
            raise ValueError(f"Folder not found: {data}")
        args += ["--data", data]
    else:
        raise ValueError("Give the logs: an ingestion spec, or a folder of Strathe CSVs.")
    if form.get("heldout", "").strip():
        args += ["--heldout-dates", *form["heldout"].split()]
    if form.get("from_strathe", "").strip():
        args += ["--from-strathe", _calibration_folder(form["from_strathe"])]
    if form.get("compare_params", "").strip():
        args += ["--compare-params", _existing_file(form["compare_params"], "Calibration file", (".yaml", ".yml"))]
    if form.get("axisymmetric"):
        args += ["--axisymmetric"]
    holds = form.get("hold", "").split()
    if holds:
        if not all(re.fullmatch(r"[a-z_]+=-?[\d.]+(e-?\d+)?", h) for h in holds):
            raise ValueError("Held values are NAME=VALUE, e.g. added_mass_x=10.6.")
        args += ["--hold", *holds]
    return args


def describe_stonefish_cli_args(form: dict) -> list:
    """`strathe describe-stonefish` arguments from the web form (PROJECT_STATUS.md #77)."""
    scn = _existing_file(form.get("scenario", ""), "Stonefish scenario", (".scn", ".xml"))
    robot = form.get("robot", "").strip()
    if not robot:
        raise ValueError("Give the robot's name in the scenario.")
    ref = form.get("reference_point", "0 0 0").split()
    try:
        ref = [str(float(x)) for x in ref]
    except ValueError:
        ref = []
    if len(ref) != 3:
        raise ValueError("The reference point is three numbers: x y z in metres.")
    args = ["--scenario", scn, "--robot", robot, "--reference-point", *ref]
    if form.get("mass", "").strip():
        args += ["--mass", str(_number(form["mass"], "Vehicle mass", 0.1, 100000))]
    finds = form.get("find", "").split()
    if finds:
        if not all("=" in f for f in finds):
            raise ValueError("$(find) folders are PKG=/path.")
        args += ["--find", *finds]
    if form.get("stonefish_data", "").strip():
        d = os.path.abspath(os.path.expanduser(form["stonefish_data"].strip()))
        if not os.path.isdir(d):
            raise ValueError(f"Stonefish data folder not found: {d}")
        args += ["--stonefish-data", d]
    if form.get("hull_lift"):
        args += ["--hull-lift"]
    return args + _replay_tool_args(form)


def stonefish_export_cli_args(form: dict) -> list:
    """`strathe export-stonefish` arguments from the web form (PROJECT_STATUS.md #74)."""
    args = ["--calibration", _calibration_folder(form.get("calibration", ""))] + _stonefish_vehicle_args(form) + \
        _replay_tool_args(form)
    if form.get("mass", "").strip():
        args += ["--mass", str(_number(form["mass"], "Vehicle mass", 0.1, 100000))]
    if form.get("heldout", "").strip():
        args += ["--heldout-dates", *form["heldout"].split()]
    return args


def link_cli_args(form: dict, sweep: bool = False) -> list:
    """--link / --optical-* / --hybrid-policy from the form, validated; blanks add nothing."""
    multi = form.get("_multi") or {}

    def many(name):
        return multi.get(name) or ([form[name]] if form.get(name) else [])
    links = many("link") if sweep else ([form["link"]] if form.get("link") else [])
    waters = many("optical_water") if sweep else ([form["optical_water"]] if form.get("optical_water") else [])
    if any(l not in dict(LINK_CHOICES) for l in links):
        raise ValueError("Unknown link.")
    if any(w not in dict(WATER_CHOICES) for w in waters):
        raise ValueError("Unknown water type.")
    optical = any(l in ("optical", "hybrid") for l in links)
    args = []
    if links and links != ["acoustic"]:
        args += ["--link", *links]
    if form.get("link_calibration", "").strip():
        args += ["--link-calibration", _existing_file(form["link_calibration"], "Link calibration", (".yaml", ".yml"))]
    if not optical:
        return args
    if waters:
        args += ["--optical-water", *waters]
    kwargs = {}
    if form.get("optical_attenuation", "").strip():
        kwargs["attenuation_per_m"] = _number(form["optical_attenuation"], "Water attenuation", 0.001, 20)
        args += ["--optical-attenuation", repr(kwargs["attenuation_per_m"])]
    terms = form.get("optical_terminals", "") or "omni"
    if terms not in dict(TERMINAL_CHOICES):
        raise ValueError("Pick optical terminals.")
    if terms != "omni":
        args += ["--optical-terminals", terms]
        kwargs["terminals"] = terms
    for key, flag, name, field in (("optical_beam", "--optical-beam", "Beam half-angle", "beam_half_angle_deg"),
                                   ("optical_fov", "--optical-fov", "Receiver half-angle", "receiver_half_fov_deg"),
                                   ("optical_rated_range", "--optical-rated-range", "Rated range", "rated_range_m")):
        if form.get(key, "").strip():
            hi = 500 if field == "rated_range_m" else 180
            if field != "rated_range_m" and _number(form[key], name, -1e9, 1e9) > 180:
                raise ValueError(f"{name}: it is a HALF-angle; 360 is a full angle, use 180 for every direction")
            kwargs[field] = _number(form[key], name, 0.1, hi)
            args += [flag, repr(kwargs[field])]
    policy = form.get("hybrid_policy", "")
    if policy not in ("", "optical_first", "redundant"):
        raise ValueError("Pick a hybrid policy.")
    if policy and "hybrid" in links:
        args += ["--hybrid-policy", policy]
    if sweep and form.get("optical_compare_omni"):
        if terms == "omni":
            raise ValueError("“Also run with omnidirectional terminals” needs cone terminals.")
        args.append("--optical-compare-omni")
    import sim_mode
    for w in waters or ["coastal"]:   # raises ValueError naming what's wrong
        sim_mode.link_from_flags("hybrid", None, water=w,
                                 **{k: (v if k != "attenuation_per_m" or w == "custom" else None)
                                    for k, v in kwargs.items()})
    if "custom" not in (waters or ["coastal"]) and "attenuation_per_m" in kwargs:
        raise ValueError("Water attenuation is only used with Custom water.")
    return args


def algorithm_select(prefix: str) -> str:
    return f"""<label for={prefix}-alg>Training algorithm</label><select id={prefix}-alg name=algorithm>
<option value=ppo>PPO (the default)</option><option value=sac>SAC</option><option value=td3>TD3</option></select>
<p class=muted>SAC and TD3 were validated on the reference BlueROV2 (one vehicle: waypoint; fleet: formation hold).
For your vehicle, train a few seeds of each and compare them before choosing.</p>"""


def engine_disturbance_fields(prefix: str) -> str:
    """Physics engine and roll/pitch disturbance choices (same field names in every form)."""
    return f"""<fieldset class=current><legend>Physics engine and roll/pitch disturbance</legend>
<label for={prefix}-engine>Physics engine for the tests</label><select id={prefix}-engine name=backend>
<option value=stand_in>Stand-in (fast; the default)</option><option value=mujoco>MuJoCo</option>
<option value=both>Both, side by side</option></select>
{disturbance_select(prefix)}
<label for={prefix}-dscale>Disturbance size: 1 = as fitted, 2 = twice as rough</label>
<input type=text id={prefix}-dscale name=disturbance_scale value="1">
<label for={prefix}-dfile>Disturbance fit file, full path (only if “From my own fit file”)</label>
<input type=text id={prefix}-dfile name=disturbance_file value="">
</fieldset>"""


def fleet_settings_fields() -> str:
    comm = "".join(f"<label class=inline><input type=checkbox name=comm value={c} aria-label='Comm level {c}'> {c} "
                   f"({d})</label>" for c, d in (("L0", "no loss"), ("L1", "10% loss"), ("L2", "30% loss"),
                                                 ("L3a", "60% loss"), ("L3b", "60% loss + too little bandwidth")))
    beliefs = "".join(f"<label class=inline><input type=checkbox name=beliefs value={b} aria-label='Teammate model "
                      f"{b}'> {b}</label>" for b in cli.BELIEF_NAMES)
    return f"""<fieldset class=current><legend>Experiment settings (optional; anything left blank keeps the YAML's value)</legend>
<p class=muted>Packet-loss levels (one result cell each):</p>{comm}
<label for=fs-beacon>Seconds between beacons (one or more, e.g. <code>1 3</code>)</label><input type=text id=fs-beacon name=beacon_interval value="">
<p class=muted>Teammate models to compare:</p>{beliefs}
<label for=fs-sep>Target separation, m</label><input type=text id=fs-sep name=target_separation value="">
<label for=fs-tol>Tolerance, m</label><input type=text id=fs-tol name=tolerance value="">
<label for=fs-hold>Hold steps (ticks of 0.1 s)</label><input type=text id=fs-hold name=hold_steps value="">
<label for=fs-eps>Episode length (ticks of 0.1 s)</label><input type=text id=fs-eps name=episode_steps value="">
<label for=fs-seeds>Sample size: packet-loss seeds per starting condition</label><input type=text id=fs-seeds name=seeds value="">
<label for=fs-starts>Starting separations, m (e.g. <code>2.5 3.5 4.5</code>)</label><input type=text id=fs-starts name=separations value="">
</fieldset>"""


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
<p class=muted>Checks your logs, fits the simulator to them, checks which values the data can pin down, and writes a
plain-language report. About an hour on the reference dataset; the computer is kept awake while it runs.</p>
<form method=post action=/jobs/calibrate>{tok}
<label for=cal-ingest>Your raw logs: the ingestion spec YAML that describes them, full path (CSV or ROS bags, forces or
thruster PWM; templates in <code>calibration/templates/</code>)</label>
<input type=text id=cal-ingest name=ingest_spec value="">
<label for=cal-data>…or a folder of logs already in Strathe's CSV layout (ignored when an ingestion spec is given)</label>
<input type=text id=cal-data name=data value="{html.escape(default_data)}">
<label for=cal-name>Name for this calibration</label><input type=text id=cal-name name=name value="my_vehicle" required>
<label for=cal-heldout>Recording dates to hold out for validation (optional, e.g. <code>16-02-2024 19-02-2024</code>;
default: the latest 30% of dates, at least one)</label>
<input type=text id=cal-heldout name=heldout value="">
<label for=cal-spec>Vehicle description YAML, full path (default: the BlueROV2 Heavy; any other vehicle: copy calibration/templates/vehicle_spec_template.yaml and fill in its mass and size, LAB_GUIDE.md step 4)</label>
<input type=text id=cal-spec name=vehicle_spec value="">
<label for=cal-rate>Sampling rate of the logs in Hz (optional; read from the conversion when there is one, else 10)</label>
<input type=text id=cal-rate name=rate value="">
<label for=cal-cur>Open water: the water current during the recordings, <code>vx vy vz</code> in m/s in the logs'
world frame (optional; empty = still water)</label><input type=text id=cal-cur name=current_world value="">
<label class=inline><input type=checkbox id=cal-estcur name=estimate_current value=1 aria-label="Estimate the water current in each recording"> Open water, current unknown:
estimate each recording's current from the logs (alternates with the fit; about 2-3x the time). Leave the
current field above empty.</label>
<label for=cal-rounds>Current estimation rounds (with the box above; default 2)</label>
<input type=text id=cal-rounds name=current_rounds value="">
<label for=cal-nfev>Optimizer budget, evaluations (default 100; more = slower but may fit better)</label>
<input type=text id=cal-nfev name=max_nfev value="100">
<label for=cal-workers>CPU cores to use (optional; default all but one)</label>
<input type=text id=cal-workers name=workers value="">
<label for=cal-restarts>Starting points (optional; default 1): start the fit from this many points and keep the
lowest training cost. 3-4 for a hard calibration; the first fit takes that many times as long</label>
<input type=text id=cal-restarts name=restarts value="">
<label class=inline><input type=checkbox id=cal-restart-whole name=restart_whole_fit value=1 aria-label="Run the whole fit from that many starting points"> With current estimation:
run the whole fit (every round) from that many starting points and keep the lowest final training cost, instead of
choosing after the first round. For a calibration whose rounds can go two ways; that many times as long</label>
<label for=cal-seed>Starting-point seed (optional): a calibration's seed, from its report, repeats it exactly</label>
<input type=text id=cal-seed name=seed value="">
<label for=cal-torque>Advanced: re-express logged torques about another point, <code>x y z</code> in m (optional)</label>
<input type=text id=cal-torque name=torque_offset value="">
<label class=inline><input type=checkbox id=cal-drag name=drag_prior value=1 aria-label="Pull drag toward published BlueROV2 values"> Advanced, BlueROV2-like frames only:
pull the drag gently toward published BlueROV2 values (when the report says linear and quadratic drag can't be told apart)</label>
<label for=cal-fd>Roll and pitch measured in free-decay tank tests: the <code>free_decay.yaml</code> that
<code>strathe free-decay</code> wrote, full path (optional; see calibration/FREE_DECAY_PROTOCOL.md)</label>
<input type=text id=cal-fd name=free_decay value="">
<label class=inline><input type=checkbox id=cal-excl name=exclude_rp value=1 aria-label="Leave roll and pitch out of the fit"> Leave roll and pitch out of the fit
(advanced: when your logs' roll/pitch motion isn't caused by the thrusters)</label>
<label class=inline><input type=checkbox id=cal-finish name=finish value=1 aria-label="Finish an interrupted calibration"> Finish an
interrupted calibration with this name (the computer slept or the run was stopped): keep its best result so far,
marked not converged, and write the report. Same settings as the interrupted run.</label>
<button>Start calibration</button></form></div>
<div class=card><h2>One vehicle: train and test a controller</h2>
<p class=muted>Trains controllers in the simulator on your calibrated vehicle and tests them on a task, with a
hand-written controller and random commands as references. Tick “compare” to also train on default physics and see
whether the calibration matters. Roughly 5 minutes per policy at the default budget.</p>
<form method=post action=/jobs/arena>{tok}
{vehicle_fields("ar")}
<label for=ar-task>Task</label><select id=ar-task name=task><option value=waypoint>Reach a point</option>
<option value=station_keeping>Reach a point and hold station</option><option value=docking>Dock (slow and level)</option>
<option value=obstacle_avoidance>Reach a point around an obstacle</option></select>
<label for=ar-seeds>Training seeds per condition</label><input type=text id=ar-seeds name=seeds value="2">
<label for=ar-steps>Training steps per policy</label><input type=text id=ar-steps name=timesteps value="450000">
<label for=ar-eps>Evaluation episodes</label><input type=text id=ar-eps name=episodes value="100">
{algorithm_select("ar")}
<label class=inline><input type=checkbox id=ar-cmp name=compare_generic value=1 checked aria-label="Compare with training on default physics"> Compare with training on
default (uncalibrated) physics</label>
{noise_fields("ar")}
<label class=inline><input type=checkbox id=ar-cmpclean name=compare_clean_training value=1 aria-label="Compare with training on a clean simulation"> With noise on: also
train on a clean simulation and compare on the noisy test</label>
{engine_disturbance_fields("ar")}
{current_fieldset("acur", teammates=False)}
<button>Start single-vehicle experiment</button></form></div>
<div class=card><h2>Run a fleet experiment</h2>
<p class=muted>Two vehicles holding formation over a lossy acoustic link (or an optical or hybrid one, below). Edit a copy of an experiment YAML to change
anything (see CUSTOM_EXPERIMENT.md).</p>
<form method=post action=/jobs/fleet>{tok}
<label for=fleet-shipped>Shipped experiment</label><select id=fleet-shipped name=shipped>{fleet_opts}</select>
<label for=fleet-config>…or your own experiment YAML (full path; leave empty to use the one above)</label>
<input type=text id=fleet-config name=config value="">
{vehicle_fields("fl")}
{fleet_settings_fields()}
{link_fields("fl", sweep=True)}
{noise_fields("fl", both=True, teammates=True)}
{engine_disturbance_fields("fl")}
{current_fieldset("cur", teammates=True)}
<button>Start experiment</button></form></div>
<div class=card><h2>Train a fleet policy for your vehicle</h2>
<p class=muted>The shipped fleet policies were trained on the reference BlueROV2. For a different vehicle, train one
on your calibration (about 40 minutes at the default budget), then list it under <code>policies:</code> in your
experiment YAML (CUSTOM_EXPERIMENT.md section 5).</p>
<form method=post action=/jobs/train_fleet>{tok}
{vehicle_fields("tf", params_required=True)}
{algorithm_select("tf")}
<label for=tf-steps>Training steps (optional; default 3,000,000 for PPO, 1,000,000 for SAC and TD3)</label>
<input type=text id=tf-steps name=timesteps value="">
{noise_fields("tf", teammates=True)}
{link_fields("tf")}
{disturbance_select("tf")}
<label for=tf-dscale>Disturbance size: 1 = as fitted, 2 = twice as rough</label>
<input type=text id=tf-dscale name=disturbance_scale value="1">
<label for=tf-dfile>Disturbance fit file, full path (only if “From my own fit file”)</label>
<input type=text id=tf-dfile name=disturbance_file value="">
<label for=tf-seed>Seed</label><input type=text id=tf-seed name=seed value="1">
<label for=tf-cur>Water current to train in, m/s (optional; one value or a range like <code>0-0.3</code>, any
direction)</label><input type=text id=tf-cur name=current_speed value="">
<label for=tf-out>Save the policy as (full path, without .zip)</label><input type=text id=tf-out name=out required>
<button>Start training</button></form></div>
<div class=card><h2>Roll and pitch from tank tests</h2>
<p class=muted>Free-decay tests: tilt the vehicle, let go with the thrusters stopped, record the swing
(<code>calibration/FREE_DECAY_PROTOCOL.md</code>). The result, <code>free_decay.yaml</code>, goes in the calibration
form's free-decay field.</p>
<form method=post action=/jobs/free_decay>{tok}
<label for=fd-spec>Ingestion spec for the tank-test logs (full path; e.g. a copy of
<code>templates/ingest_free_decay_ardupilot.yaml</code>)</label><input type=text id=fd-spec name=ingest_spec value="">
<label for=fd-data>…or a folder of tank-test logs already in Strathe's CSV layout</label>
<input type=text id=fd-data name=data value="">
<label for=fd-name>Name for this result</label><input type=text id=fd-name name=name value="my_tank_tests" required>
<button>Fit roll and pitch</button></form></div>
<div class=card><h2>Calibrate the communication link</h2>
<p class=muted>From your modems' message logs (acoustic, optical or both): delivery against distance and latency,
for fleet experiments and training. Format: <code>calibration/templates/link_log_template.csv</code>.</p>
<form method=post action=/jobs/fit_link>{tok}
<label for=lk-log>Message log CSV(s), full paths (several: one per line or separated by spaces)</label>
<input type=text id=lk-log name=logs required>
<label for=lk-name>Name for this calibration</label><input type=text id=lk-name name=name value="my_link" required>
<label for=lk-link>Link of rows with no <code>link</code> column</label><select id=lk-link name=link>
<option value=acoustic>acoustic</option><option value=optical>optical</option></select>
<label for=lk-water>Optical: the water's clarity (messages can't measure it; it only matters beyond the ranges
you logged)</label>
<select id=lk-water name=optical_water><option value="">coastal (default)</option><option value=clear>clear</option>
<option value=coastal>coastal</option><option value=harbor>harbour</option></select>
<button>Calibrate the link</button></form></div>
<div class=card><h2>Calibrate your Stonefish model</h2>
<p class=muted>Fits your own Stonefish vehicle's drag, added mass, buoyancy and centre of gravity (one part) to your logs,
and scores it against the original on recordings it never saw. Writes your vehicle file back with only that part
changed. Needs the replay tool built against your Stonefish (<code>calibration/stonefish/README.md</code>); an hour
or more.</p>
<form method=post action=/jobs/fit_stonefish>{tok}
<label for=sf-scn>Your Stonefish world scenario (.scn, full path)</label><input type=text id=sf-scn name=scenario required>
<label for=sf-robot>Robot name in the scenario</label><input type=text id=sf-robot name=robot required>
<label for=sf-part>Part to tune (e.g. the hull's external_part name)</label><input type=text id=sf-part name=part required>
<label for=sf-ref>Where the logs' velocity and forces are measured, in the robot's frame: x y z, m (forward-right-down,
e.g. your URDF's base_link)</label><input type=text id=sf-ref name=reference_point value="0 0 0" required>
<label for=sf-spec>Your logs: ingestion spec YAML (full path)</label><input type=text id=sf-spec name=ingest_spec value="">
<label for=sf-data>…or a folder of logs already in Strathe's CSV layout</label><input type=text id=sf-data name=data value="">
<label for=sf-heldout>Recording dates to hold out (optional; default the latest 30%)</label>
<input type=text id=sf-heldout name=heldout value="">
<label for=sf-buoy>Buoyant part to resize for the net buoyancy (a box, cylinder or sphere; optional)</label>
<input type=text id=sf-buoy name=buoyancy_part value="">
<label for=sf-find>ROS-style scenarios: $(find PKG) folders, PKG=/path (one or more, space-separated; optional)</label>
<input type=text id=sf-find name=find value="">
<label for=sf-sfdata>Stonefish data folder for meshes (optional; default the scenario's folder)</label>
<input type=text id=sf-sfdata name=stonefish_data value="">
<label for=sf-density>Water density per recording date, DATE=RHO (optional; e.g. 27-03-2026=1000)</label>
<input type=text id=sf-density name=density_by_date value="">
<label for=sf-from>Start from your Strathe calibration's values instead of your model's: its output folder (optional,
experimental)</label><input type=text id=sf-from name=from_strathe value="">
<label for=sf-cmp>Also score your Strathe calibration: calibrated_parameters.yaml (optional)</label>
<input type=text id=sf-cmp name=compare_params value="">
<label><input type=checkbox id=sf-axi name=axisymmetric> Torpedo-shaped vehicle: one value across the hull</label>
<label for=sf-hold>Hold values you measured, NAME=VALUE (optional; e.g. added_mass_x=10.6)</label>
<input type=text id=sf-hold name=hold value="">
<label for=sf-tool>The built replay tool, full path (optional; default calibration/stonefish/build/strathe_stonefish_replay)</label>
<input type=text id=sf-tool name=replay_tool value="">
<label for=sf-iter>Fit iterations (optional; default 15)</label><input type=text id=sf-iter name=max_iterations value="">
<label for=sf-workers>Stonefish runs at once (optional; default all cores but one)</label><input type=text id=sf-workers name=workers value="">
<label for=sf-name>Name for this calibration</label><input type=text id=sf-name name=name value="my_stonefish" required>
<button>Calibrate the Stonefish model</button></form></div>
<div class=card><h2>Describe your vehicle from your Stonefish model</h2>
<p class=muted>Writes the vehicle description that Calibrate needs (mass, inertia, added mass, drag, buoyancy, centre of
gravity), with starting values from your own Stonefish model, so you don't fill the template by hand. Seconds. Needs the
replay tool (<code>calibration/stonefish/README.md</code>). Then calibrate with it using <b>4 starting points</b>: its
limits are wide, and one start isn't reliable.</p>
<form method=post action=/jobs/describe_stonefish>{tok}
<label for=ds-scn>Your Stonefish world scenario (.scn, full path)</label><input type=text id=ds-scn name=scenario required>
<label for=ds-robot>Robot name in the scenario</label><input type=text id=ds-robot name=robot required>
<label for=ds-ref>Where your logs' velocity and forces are measured, in the robot's frame: x y z, m (forward-right-down,
e.g. your URDF's base_link)</label><input type=text id=ds-ref name=reference_point value="0 0 0" required>
<label for=ds-mass>Weighed vehicle mass, kg (optional; default the model's)</label><input type=text id=ds-mass name=mass value="">
<label for=ds-find>ROS-style scenarios: $(find PKG) folders, PKG=/path (optional)</label>
<input type=text id=ds-find name=find value="">
<label for=ds-sfdata>Stonefish data folder for meshes (optional; default the scenario's folder)</label>
<input type=text id=ds-sfdata name=stonefish_data value="">
<label><input type=checkbox id=ds-lift name=hull_lift> Slender hull: include the hull-lift terms</label>
<label for=ds-tool>The built replay tool, full path (optional; default calibration/stonefish/build/strathe_stonefish_replay)</label>
<input type=text id=ds-tool name=replay_tool value="">
<label for=ds-name>Vehicle name</label><input type=text id=ds-name name=name value="my_vehicle" required>
<button>Write the vehicle description</button></form></div>
<div class=card><h2>Put your Strathe calibration into Stonefish</h2>
<p class=muted>Converts the values your calibration found (mass, buoyancy, centre of gravity, inertia, drag, added mass)
into your own Stonefish vehicle's settings on one part, and scores the original and the converted model on the
recordings the calibration never saw. Minutes, not hours. Needs the replay tool (<code>calibration/stonefish/README.md
</code>). With Stonefish built with <code>calibration/stonefish/stonefish_added_mass.patch</code>, the drag goes in as
Strathe's own damping and the model behaves like Strathe's over seconds; <code>python strathe.py check</code> says
whether yours has it. For the best model over a minute, use "Calibrate your Stonefish model".</p>
<form method=post action=/jobs/export_stonefish>{tok}
<label for=sx-cal>Your calibration's output folder (from Calibrate, full path)</label>
<input type=text id=sx-cal name=calibration required>
<label for=sx-scn>Your Stonefish world scenario (.scn, full path)</label><input type=text id=sx-scn name=scenario required>
<label for=sx-robot>Robot name in the scenario</label><input type=text id=sx-robot name=robot required>
<label for=sx-part>Part that takes the values (e.g. the hull's external_part name)</label>
<input type=text id=sx-part name=part required>
<label for=sx-ref>Where the logs' velocity and forces are measured, in the robot's frame: x y z, m (forward-right-down,
e.g. your URDF's base_link)</label><input type=text id=sx-ref name=reference_point value="0 0 0" required>
<label for=sx-buoy>Buoyant part to resize for the net buoyancy (a box, cylinder or sphere; optional)</label>
<input type=text id=sx-buoy name=buoyancy_part value="">
<label for=sx-mass>Vehicle mass, kg (optional; default the calibration's)</label><input type=text id=sx-mass name=mass value="">
<label for=sx-find>ROS-style scenarios: $(find PKG) folders, PKG=/path (optional)</label>
<input type=text id=sx-find name=find value="">
<label for=sx-sfdata>Stonefish data folder for meshes (optional; default the scenario's folder)</label>
<input type=text id=sx-sfdata name=stonefish_data value="">
<label for=sx-density>Water density per recording date, DATE=RHO (optional; e.g. 27-03-2026=1000)</label>
<input type=text id=sx-density name=density_by_date value="">
<label for=sx-tool>The built replay tool, full path (optional; default calibration/stonefish/build/strathe_stonefish_replay)</label>
<input type=text id=sx-tool name=replay_tool value="">
<label for=sx-name>Name for this export</label><input type=text id=sx-name name=name value="my_stonefish" required>
<button>Put it into Stonefish</button></form></div>
<div class=card><h2>Convert logs</h2>
<form method=post action=/jobs/ingest>{tok}
<label for=ing-spec>Ingestion spec YAML (full path; templates in <code>calibration/templates/</code>)</label><input type=text id=ing-spec name=spec required>
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
        if url.path == "/image":  # a report's plots, shown in the page
            real = self.manager.allowed_file((q.get("path") or [""])[0])
            if real is None or not real.endswith(".png"):
                return self._error(404, "That image isn't part of a job's output.")
            with open(real, "rb") as f:
                return self._send(200, f.read(), "image/png")
        if url.path in ("/file", "/view"):
            real = self.manager.allowed_file((q.get("path") or [""])[0])
            if real is None:
                return self._error(404, "That file isn't part of a job's output.")
            with open(real, "rb") as f:
                data = f.read()
            if url.path == "/view":
                text = data.decode("utf-8", errors="replace")
                return self._send(200, page(os.path.basename(real), "<p><a href='/'>← Strathe home</a></p>"
                                            + render_markdown(text, os.path.dirname(real))))
            return self._send(200, data, "application/octet-stream",
                              {"Content-Disposition": f"attachment; filename=\"{os.path.basename(real)}\""})
        return self._error(404, "Not found.")

    def do_POST(self):
        if not self._host_ok():
            return self._error(403, "Requests must come from this computer.")
        length = int(self.headers.get("Content-Length") or 0)
        if length > 100_000:
            return self._error(413, "Form too large.")
        parsed = parse_qs(self.rfile.read(length).decode("utf-8"))
        form = {k: v[0] for k, v in parsed.items()}
        form["_multi"] = {k: v for k, v in parsed.items() if len(v) > 1}  # checkboxes sharing a name
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
            ingest_spec = form.get("ingest_spec", "").strip()
            data = form.get("data", "").strip()
            name = cli.safe_name(form.get("name", "my_vehicle"))
            if ingest_spec:
                src = ["--ingest-spec", _existing_file(ingest_spec, "Ingestion spec", (".yaml", ".yml"))]
            elif data:
                data = os.path.abspath(os.path.expanduser(data))
                if not os.path.isdir(data):
                    raise ValueError(f"Folder not found: {data}")
                src = ["--data", data]
            else:
                raise ValueError("Give your logs: an ingestion spec, or a folder of Strathe CSVs.")
            heldout = form.get("heldout", "").split()
            for d in heldout:
                if not re.fullmatch(r"\d{2}-\d{2}-\d{4}", d):
                    raise ValueError(f"Held-out dates must look like 16-02-2024, got {d!r}")
            out = os.path.join(cli.REPO_DIR, f"calibration_output_{name}")
            args = ["calibrate", *src, "--name", name, "--out", out]
            if form.get("vehicle_spec", "").strip():
                args += ["--spec", _existing_file(form["vehicle_spec"], "Vehicle description", (".yaml", ".yml"))]
            if form.get("rate", "").strip():
                args += ["--sampling-rate-hz", repr(_number(form["rate"], "Sampling rate", 0.1, 1000))]
            if form.get("current_world", "").strip():
                parts = form["current_world"].replace(",", " ").split()
                if len(parts) != 3:
                    raise ValueError("Water current during the recordings: give three numbers, vx vy vz (m/s).")
                args += ["--current-world-mps", *[repr(_number(p, "Water current", -3, 3)) for p in parts]]
            nfev = form.get("max_nfev", "").strip()
            if nfev:
                args += ["--max-nfev", str(int(_number(nfev, "Optimizer budget", 1, 100000)))]
            if form.get("exclude_rp"):
                args += ["--exclude-axes", "p", "q"]
            if form.get("workers", "").strip():
                args += ["--workers", str(int(_number(form["workers"], "CPU cores", 1, 256)))]
            if form.get("seed", "").strip():
                args += ["--seed", str(int(_number(form["seed"], "Seed", 0, 2 ** 31 - 1)))]
            if form.get("restarts", "").strip():
                args += ["--restarts", str(int(_number(form["restarts"], "Starting points", 1, 20)))]
            if form.get("restart_whole_fit"):
                args += ["--restart-whole-fit"]
            if form.get("torque_offset", "").strip():
                parts = form["torque_offset"].replace(",", " ").split()
                if len(parts) != 3:
                    raise ValueError("Torque reference point: give three numbers, x y z (m).")
                args += ["--torque-reference-offset-m", *[repr(_number(p, "Torque reference point", -10, 10)) for p in parts]]
            if form.get("drag_prior"):
                args += ["--drag", "prior"]
            if form.get("free_decay", "").strip():
                args += ["--free-decay", _existing_file(form["free_decay"], "Free-decay result", (".yaml", ".yml"))]
            if form.get("estimate_current"):
                if form.get("current_world", "").strip():
                    raise ValueError("Give either the known water current or tick 'estimate the current', not both.")
                args.append("--estimate-current")
                if form.get("current_rounds", "").strip():
                    args += ["--current-rounds", str(int(_number(form["current_rounds"], "Current estimation rounds", 1, 10)))]
            elif form.get("current_rounds", "").strip():
                raise ValueError("Current estimation rounds need 'estimate the current' ticked.")
            if heldout:
                args += ["--heldout-dates", *heldout]
            if form.get("finish"):
                if not os.path.exists(os.path.join(out, "checkpoint.yaml")):
                    raise ValueError(f"Nothing to finish: no interrupted calibration named {name!r} "
                                     f"(no checkpoint.yaml in {out}).")
                args.append("--finish")
            return self.manager.start("calibrate", f"Calibration: {name}", args, [out])
        if path == "/jobs/arena":
            task = form.get("task", "waypoint")
            if task not in ("waypoint", "station_keeping", "docking", "obstacle_avoidance"):
                raise ValueError("Pick one of the tasks.")
            out = os.path.join(JOBS_DIR, f"arena_{task}_{time.strftime('%Y%m%d-%H%M%S')}")
            args = ["arena", "--task", task, *vehicle_cli_args(form),
                    "--seeds", str(int(_number(form.get("seeds", "2"), "Training seeds", 1, 20))),
                    "--timesteps", str(int(_number(form.get("timesteps", "450000"), "Training steps", 1000, 1e8))),
                    "--episodes", str(int(_number(form.get("episodes", "100"), "Evaluation episodes", 1, 10000))),
                    "--out", out]
            if form.get("compare_generic"):
                args.append("--compare-generic")
            args += algorithm_cli_args(form)
            args += noise_cli_args(form)
            if form.get("compare_clean_training"):
                args.append("--compare-clean-training")
            args += engine_disturbance_cli_args(form, default_engine="stand_in")
            title = f"One vehicle: {task.replace('_', ' ')}"
            if form.get("current_on"):
                args += [a for a in current_cli_args({**form, "current_known": "1"})]
                title += " + water current"
            return self.manager.start("arena", title, args, [out])
        if path == "/jobs/train_fleet":
            if not form.get("params", "").strip():
                raise ValueError("Give your calibration file (calibrated_parameters.yaml).")
            out = os.path.abspath(os.path.expanduser(form.get("out", "").strip()))
            if not form.get("out", "").strip() or not os.path.isdir(os.path.dirname(out)):
                raise ValueError(f"The folder to save the policy in doesn't exist: {os.path.dirname(out)}")
            args = ["train-fleet", *vehicle_cli_args(form), *algorithm_cli_args(form),
                    "--seed", str(int(_number(form.get("seed", "1"), "Seed", 0, 1e9))), "--out", out]
            if form.get("timesteps", "").strip():
                args += ["--timesteps", str(int(_number(form["timesteps"], "Training steps", 1000, 1e9)))]
            args += noise_cli_args(form, teammates=True)
            args += link_cli_args(form)
            args += disturbance_cli_args(form)
            if form.get("current_speed", "").strip():
                speed = _number_or_range(form["current_speed"], "Current speed")
                cli.current_block(float(speed[0]) if len(speed) == 1 else [float(x) for x in speed], [0.0, 360.0])
                args += ["--current-speed", *speed]
            return self.manager.start("train_fleet", f"Train fleet policy: {os.path.basename(out)}", args,
                                      [os.path.dirname(out)])
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
            args = ["fleet", "--config", config, *vehicle_cli_args(form, "--vehicle-spec"),
                    *fleet_settings_cli_args(form), *link_cli_args(form, sweep=True),
                    *noise_cli_args(form, allow_both=True, teammates=True),
                    *engine_disturbance_cli_args(form)]
            title = f"Fleet experiment: {os.path.basename(config)}"
            if form.get("current_on"):
                args += current_cli_args(form)
                title += " + water current"
            return self.manager.start("fleet", title, args, [])
        if path == "/jobs/free_decay":
            name = re.sub(r"[^\w-]+", "_", form.get("name", "").strip()) or "my_tank_tests"
            out = os.path.join(JOBS_DIR, f"free_decay_{name}")
            if form.get("ingest_spec", "").strip():
                src = ["--ingest-spec", _existing_file(form["ingest_spec"], "Ingestion spec", (".yaml", ".yml"))]
            elif form.get("data", "").strip():
                data = os.path.abspath(os.path.expanduser(form["data"].strip()))
                if not os.path.isdir(data):
                    raise ValueError(f"Folder not found: {data}")
                src = ["--data", data]
            else:
                raise ValueError("Give the tank-test logs: an ingestion spec, or a folder of Strathe CSVs.")
            return self.manager.start("free_decay", f"Free decay: {name}", ["free-decay", *src, "--out", out], [out])
        if path == "/jobs/fit_link":
            logs = [os.path.abspath(os.path.expanduser(x)) for x in form.get("logs", "").split()]
            if not logs:
                raise ValueError("Give at least one message log CSV.")
            for f in logs:
                if not os.path.isfile(f):
                    raise ValueError(f"Message log not found: {f}")
            name = re.sub(r"[^\w-]+", "_", form.get("name", "").strip()) or "my_link"
            link = form.get("link", "acoustic")
            if link not in ("acoustic", "optical"):
                raise ValueError("Pick acoustic or optical.")
            water = form.get("optical_water", "")
            if water not in ("", "clear", "coastal", "harbor"):
                raise ValueError("Pick a water type.")
            out = os.path.join(JOBS_DIR, f"link_{name}")
            args = ["fit-link", "--log", *logs, "--out", out, "--link", link] + (["--optical-water", water] if water else [])
            return self.manager.start("fit_link", f"Link calibration: {name}", args, [out])
        if path == "/jobs/describe_stonefish":
            name = re.sub(r"[^\w-]+", "_", form.get("name", "").strip()) or "my_vehicle"
            folder = os.path.join(JOBS_DIR, f"describe_{name}")
            os.makedirs(folder, exist_ok=True)
            out = os.path.join(folder, f"{name}_vehicle_spec.yaml")
            args = ["describe-stonefish", *describe_stonefish_cli_args(form), "--name", name, "--out", out]
            return self.manager.start("describe_stonefish", f"Vehicle description: {name}", args, [folder])
        if path == "/jobs/export_stonefish":
            name = re.sub(r"[^\w-]+", "_", form.get("name", "").strip()) or "my_stonefish"
            out = os.path.join(JOBS_DIR, f"stonefish_export_{name}")
            args = ["export-stonefish", *stonefish_export_cli_args(form), "--out", out]
            return self.manager.start("export_stonefish", f"Stonefish export: {name}", args, [out])
        if path == "/jobs/fit_stonefish":
            name = re.sub(r"[^\w-]+", "_", form.get("name", "").strip()) or "my_stonefish"
            out = os.path.join(JOBS_DIR, f"stonefish_{name}")
            args = ["fit-stonefish", *stonefish_cli_args(form), "--out", out]
            return self.manager.start("fit_stonefish", f"Stonefish calibration: {name}", args, [out])
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


def _existing_file(text: str, what: str, suffixes: tuple) -> str:
    p = os.path.abspath(os.path.expanduser(text.strip()))
    if not (os.path.isfile(p) and p.endswith(suffixes)):
        raise ValueError(f"{what} not found (need an existing {'/'.join(suffixes)} file): {p}")
    return p


def _number(text: str, name: str, lo: float, hi: float) -> float:
    try:
        v = float(str(text).strip())
    except ValueError:
        raise ValueError(f"{name}: expected a number, got {text!r}")
    if not lo <= v <= hi:
        raise ValueError(f"{name}: must be between {lo:g} and {hi:g}, got {v:g}")
    return v


def engine_disturbance_cli_args(form: dict, default_engine: Optional[str] = None) -> list:
    """--backend / --disturbance / --disturbance-scale from the form, validated. For the fleet, an
    engine left at the default and no disturbance keep the experiment YAML's own values."""
    args = []
    engine = form.get("backend", "stand_in")
    if engine not in ("stand_in", "mujoco", "both"):
        raise ValueError("Pick a physics engine.")
    if default_engine is not None or engine != "stand_in":
        args += ["--backend", engine]
    return args + disturbance_cli_args(form)


def algorithm_cli_args(form: dict) -> list:
    alg = form.get("algorithm", "ppo") or "ppo"
    if alg not in ("ppo", "sac", "td3"):
        raise ValueError("Pick a training algorithm.")
    return ["--algorithm", alg]


def disturbance_cli_args(form: dict) -> list:
    """--disturbance / --disturbance-scale: blank = from the mode (or the YAML); 'none' turns it off explicitly."""
    args = []
    choice = form.get("disturbance", "")
    if choice == "":
        return args
    if choice == "none":
        return ["--disturbance", "none"]
    scale = _number(form.get("disturbance_scale", "1") or "1", "Disturbance size", 0, 10)
    if choice == "fitted":
        return args + ["--disturbance", "fitted", "--disturbance-scale", repr(scale)]
    if choice == "file":
        path = _existing_file(form.get("disturbance_file", ""), "Disturbance fit file", (".yaml", ".yml"))
        from disturbance import load_disturbance
        load_disturbance(path)  # raises ValueError if it isn't a disturbance fit
        return args + ["--disturbance", path, "--disturbance-scale", repr(scale)]
    raise ValueError("Pick a disturbance option.")


def fleet_settings_cli_args(form: dict) -> list:
    """`strathe fleet` experiment-setting flags from the fleet form, validated; blanks are skipped."""
    multi = form.get("_multi") or {}
    args = []
    comm = multi.get("comm") or ([form["comm"]] if form.get("comm") else [])
    if any(c not in cli.SEVERITIES for c in comm):
        raise ValueError("Unknown packet-loss level.")
    if comm:
        args += ["--comm", *comm]
    beliefs = multi.get("beliefs") or ([form["beliefs"]] if form.get("beliefs") else [])
    if any(b not in cli.BELIEF_NAMES for b in beliefs):
        raise ValueError("Unknown teammate model.")
    if beliefs:
        if len(beliefs) < 2:
            raise ValueError("Pick at least two teammate models to compare.")
        args += ["--beliefs", *beliefs]
    if form.get("beacon_interval", "").strip():
        vals = form["beacon_interval"].replace(",", " ").split()
        args += ["--beacon-interval", *[repr(_number(v, "Seconds between beacons", 0.1, 60)) for v in vals]]
    for key, flag, name, lo, hi, as_int in (
            ("target_separation", "--target-separation", "Target separation", 0.1, 100, False),
            ("tolerance", "--tolerance", "Tolerance", 0.01, 10, False),
            ("hold_steps", "--hold-steps", "Hold steps", 1, 10000, True),
            ("episode_steps", "--episode-steps", "Episode length", 1, 100000, True),
            ("seeds", "--seeds", "Sample size", 1, 1000, True)):
        raw = form.get(key, "").strip()
        if raw:
            v = _number(raw, name, lo, hi)
            args += [flag, str(int(v)) if as_int else repr(v)]
    if form.get("separations", "").strip():
        vals = form["separations"].replace(",", " ").split()
        args += ["--separations", *[repr(_number(v, "Starting separation", 0.1, 100)) for v in vals]]
    return args


def vehicle_cli_args(form: dict, spec_flag: str = "--spec") -> list:
    """--params / --spec (fleet: --vehicle-spec) / --thrusters from the 'Your vehicle' fields, validated."""
    args = []
    if form.get("params", "").strip():
        args += ["--params", _existing_file(form["params"], "Calibration file", (".yaml", ".yml"))]
    if form.get("vehicle_spec", "").strip():
        args += [spec_flag, _existing_file(form["vehicle_spec"], "Vehicle description", (".yaml", ".yml"))]
    choice = form.get("thrusters", "")
    if choice == "bluerov2_heavy":
        args += ["--thrusters", "bluerov2_heavy"]
    elif choice == "file":
        path = _existing_file(form.get("thrusters_file", ""), "Thrusters file", (".yaml", ".yml"))
        from fleet_experiment_schema import resolve_thrusters
        resolve_thrusters(path)  # raises ValueError naming what's wrong with the block
        args += ["--thrusters", path]
    elif choice:
        raise ValueError("Pick a thruster option.")
    return args


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


def _signed_number_or_range(text: str, name: str) -> list:
    """'-3' -> ['-3.0']; '-5 to 5', '-5 5' or '-5, 5' -> ['-5.0', '5.0'] (signed, so no '-' as a separator)."""
    parts = [p for p in re.split(r"\s+to\s+|\s*,\s*|\s+", (text or "").strip().lower()) if p]
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"{name}: expected a number or a range like -5 to 5, got {text!r}")
    if len(vals) not in (1, 2) or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"{name}: expected one number or two (a range like -5 to 5), got {text!r}")
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
